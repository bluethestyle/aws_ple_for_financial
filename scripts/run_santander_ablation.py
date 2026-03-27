#!/usr/bin/env python3
"""
4-Dimension Ablation Test Orchestrator — Santander Dataset.

Runs a 6-phase ablation study on the Santander product-recommendation dataset
with 4 ablation dimensions:

    Phase 0  Data Preparation        (Processing Job — santander_adapter)
    Phase 1  Feature Group Ablation  (10 Training Jobs)
    Phase 2  Expert Ablation         (7 Training Jobs)
    Phase 3  Task x Structure Cross  (16 Training Jobs — KEY experiment)
    Phase 4  Best-Config Teacher + Distillation
    Phase 5  Analysis + HTML Report

Usage::

    # All phases
    python scripts/run_santander_ablation.py --phase all

    # Single phase
    python scripts/run_santander_ablation.py --phase 3

    # Dry run (print what would be submitted)
    python scripts/run_santander_ablation.py --phase all --dry-run
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import sys
import tarfile
import tempfile
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import yaml

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger("run_santander_ablation")

# ---------------------------------------------------------------------------
# Project paths
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = os.environ.get(
    "ABLATION_CONFIG_PATH", "configs/santander/pipeline.yaml"
)
FEATURE_GROUPS_PATH = os.environ.get(
    "ABLATION_FEATURE_GROUPS_PATH", "configs/santander/feature_groups.yaml"
)

# Framework versions (for PyTorch Estimator)
PYTORCH_VERSION = "2.1"
PY_VERSION = "py310"


# ---------------------------------------------------------------------------
# Config loading — ALL constants derived from YAML
# ---------------------------------------------------------------------------

def _load_pipeline_config() -> Dict[str, Any]:
    """Load the pipeline YAML config as a raw dict."""
    cfg_path = PROJECT_ROOT / CONFIG_PATH
    with open(cfg_path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def _load_feature_groups_config() -> Dict[str, Any]:
    """Load the feature_groups YAML config as a raw dict."""
    fg_path = PROJECT_ROOT / FEATURE_GROUPS_PATH
    with open(fg_path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def _extract_aws_constants(config: Dict[str, Any]) -> Dict[str, str]:
    """Extract AWS/SageMaker constants from pipeline config.

    Raises ValueError if required aws fields are missing.
    """
    aws = config.get("aws", {})
    required_keys = {"region": "aws.region", "s3_bucket": "aws.s3_bucket"}
    for key, desc in required_keys.items():
        if not aws.get(key):
            raise ValueError(
                f"{desc} not configured in pipeline.yaml — "
                "all AWS constants must be explicit in the config"
            )
    return {
        "region": aws["region"],
        "s3_bucket": aws["s3_bucket"],
        "role_arn": aws.get("role_arn", ""),
        "gpu_instance": aws["instance_type"],
        "cpu_instance": aws.get("cpu_instance_type", aws["instance_type"]),
    }


def _extract_shared_experts(config: Dict[str, Any]) -> List[str]:
    """Extract the shared expert list from model.expert_basket.shared."""
    return config.get("model", {}).get("expert_basket", {}).get("shared", [])


def _build_feature_expert_deps(fg_config: Dict[str, Any]) -> Dict[str, List[str]]:
    """Build feature-group -> expert dependency map from feature_groups.yaml.

    Each feature group may declare ``target_experts`` — when that group is
    removed, any expert that ONLY receives input from that group should be
    deactivated to avoid input-less experts.
    """
    deps: Dict[str, List[str]] = {}
    for g in fg_config.get("feature_groups", []):
        name = g["name"]
        target_experts = g.get("target_experts", [])
        deps[name] = target_experts
    return deps


def _build_feature_scenarios(
    fg_config: Dict[str, Any],
    base_group_names: List[str],
    feature_expert_deps: Dict[str, List[str]],
) -> List[Dict[str, Any]]:
    """Build bottom-up + top-down feature ablation scenarios from config.

    Parameters
    ----------
    fg_config : dict
        Raw feature_groups.yaml content.
    base_group_names : list
        Names of groups considered "base" (not removed in base_only).
    feature_expert_deps : dict
        Feature group -> target expert mapping.

    Returns
    -------
    list of scenario dicts with keys ``name`` and ``remove``.
    """
    all_groups = [
        g["name"]
        for g in fg_config.get("feature_groups", [])
        if g.get("enabled", True)
    ]
    advanced_groups = [g for g in all_groups if g not in base_group_names]

    scenarios: List[Dict[str, Any]] = []

    # Full baseline (reused by Phase 2/3)
    scenarios.append({"name": "full", "remove": []})

    # Bottom-up: base_only, then base + one advanced group
    scenarios.append({"name": "base_only", "remove": list(advanced_groups)})
    for group in advanced_groups:
        scenarios.append({
            "name": f"base+{group}",
            "remove": [g for g in advanced_groups if g != group],
        })

    # Top-down: full minus one group (irreplaceability)
    for group in advanced_groups:
        scenarios.append({
            "name": f"full-{group}",
            "remove": [group],
        })

    return scenarios


def _build_expert_scenarios(all_experts: List[str]) -> List[Dict[str, Any]]:
    """Build bottom-up + top-down expert ablation scenarios from config.

    Uses base_expert and minimal_expert from ablation config
    (defaults: deepfm / mlp).

    Returns
    -------
    list of scenario dicts with keys ``name`` and ``experts``.
    """
    ablation_cfg = _PIPELINE_CONFIG.get("ablation", {})
    base_expert = ablation_cfg.get("base_expert", "deepfm")
    minimal_expert = ablation_cfg.get("minimal_expert", "mlp")

    scenarios: List[Dict[str, Any]] = []

    # Bottom-up: base expert alone, then base expert + one other expert
    scenarios.append({"name": f"{base_expert}_only", "experts": [base_expert]})
    for expert in all_experts:
        if expert == base_expert:
            continue
        # Short name for the scenario
        short = expert.replace("optimal_transport", "ot").replace("temporal_ensemble", "temporal")
        scenarios.append({
            "name": f"{base_expert}+{short}",
            "experts": [base_expert, expert],
        })

    # Top-down: full minus one expert
    for expert in all_experts:
        short = expert.replace("optimal_transport", "ot").replace("temporal_ensemble", "temporal")
        scenarios.append({
            "name": f"full-{short}",
            "experts": [e for e in all_experts if e != expert],
        })

    # Minimal baseline
    scenarios.append({"name": f"{minimal_expert}_only", "experts": [minimal_expert]})

    return scenarios


def _extract_task_tiers(config: Dict[str, Any]) -> Dict[str, List[str]]:
    """Extract task tiers from ablation.task_tiers in pipeline config."""
    return config.get("ablation", {}).get("task_tiers", {})


def _extract_structure_variants(config: Dict[str, Any]) -> Dict[str, Dict[str, bool]]:
    """Extract structure variants from ablation.structure_variants in pipeline config."""
    return config.get("ablation", {}).get("structure_variants", {})


# ---------------------------------------------------------------------------
# Load all config at module level
# ---------------------------------------------------------------------------

_PIPELINE_CONFIG = _load_pipeline_config()
_FG_CONFIG = _load_feature_groups_config()

# AWS constants
_AWS = _extract_aws_constants(_PIPELINE_CONFIG)
REGION = _AWS["region"]
S3_BUCKET = _AWS["s3_bucket"]
ROLE_ARN = _AWS["role_arn"]
GPU_INSTANCE = _AWS["gpu_instance"]
CPU_INSTANCE = _AWS["cpu_instance"]

# Expert list from model config
ALL_SHARED_EXPERTS: List[str] = _extract_shared_experts(_PIPELINE_CONFIG)

# Feature-expert dependency map from feature_groups.yaml
FEATURE_EXPERT_DEPS: Dict[str, List[str]] = _build_feature_expert_deps(_FG_CONFIG)

# Base groups (not removed in base_only scenario)
_base_groups_raw = _PIPELINE_CONFIG.get("ablation", {}).get("base_groups")
if not _base_groups_raw:
    raise ValueError(
        "ablation.base_groups is required in pipeline.yaml but was not found. "
        "Please define it (e.g., base_groups: [demographics, product_holdings])."
    )
_BASE_GROUP_NAMES: List[str] = _base_groups_raw

# Training defaults for all ablation phases (read once from config)
_training_defaults_raw = _PIPELINE_CONFIG.get("ablation", {}).get("training_defaults")
if not _training_defaults_raw:
    raise ValueError(
        "ablation.training_defaults is required in pipeline.yaml but was not found. "
        "Define epochs, batch_size, learning_rate, etc. under ablation.training_defaults."
    )
TRAINING_DEFAULTS: Dict[str, str] = {k: str(v) for k, v in _training_defaults_raw.items()}

# Module-level globals for pre-built source package (set once in main)
_GLOBAL_SOURCE_DIR: Optional[str] = None      # Extracted source dir for training jobs
_GLOBAL_SOURCE_TAR: Optional[str] = None      # Source tar.gz for processing jobs


def _experts_for_scenario(removed_groups: List[str]) -> List[str]:
    """Determine which experts to keep given removed feature groups."""
    experts_to_remove = set()
    for grp in removed_groups:
        for exp in FEATURE_EXPERT_DEPS.get(grp, []):
            experts_to_remove.add(exp)
    return [e for e in ALL_SHARED_EXPERTS if e not in experts_to_remove]


# ---------------------------------------------------------------------------
# Dimension 1: Feature Group Ablation (dynamic from feature_groups.yaml)
# ---------------------------------------------------------------------------
_ablation_cfg = _PIPELINE_CONFIG.get("ablation", {})

if _ablation_cfg.get("feature_scenarios") == "auto":
    FEATURE_SCENARIOS: List[Dict[str, Any]] = _build_feature_scenarios(
        _FG_CONFIG, _BASE_GROUP_NAMES, FEATURE_EXPERT_DEPS,
    )
else:
    FEATURE_SCENARIOS = _ablation_cfg.get("feature_scenarios", [])

# ---------------------------------------------------------------------------
# Dimension 2: Expert Ablation (dynamic from model.expert_basket)
# ---------------------------------------------------------------------------
if _ablation_cfg.get("expert_scenarios") == "auto":
    EXPERT_SCENARIOS: List[Dict[str, Any]] = _build_expert_scenarios(ALL_SHARED_EXPERTS)
else:
    EXPERT_SCENARIOS = _ablation_cfg.get("expert_scenarios", [])

# ---------------------------------------------------------------------------
# Dimension 3: Task x Structure Cross Ablation (from pipeline.yaml)
# ---------------------------------------------------------------------------
TASK_TIERS: Dict[str, List[str]] = _extract_task_tiers(_PIPELINE_CONFIG)
STRUCTURE_VARIANTS: Dict[str, Dict[str, bool]] = _extract_structure_variants(_PIPELINE_CONFIG)


# ===================================================================
# Argument parsing
# ===================================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="4-Dimension Ablation Test Orchestrator for Santander dataset",
    )
    parser.add_argument(
        "--phase",
        type=str,
        default="all",
        choices=["0", "1", "2", "3", "4", "5", "all"],
        help="Which phase to run (0-5 or 'all')",
    )
    parser.add_argument(
        "--instance-type-gpu",
        type=str,
        default=GPU_INSTANCE,
        help="GPU instance type for training jobs",
    )
    parser.add_argument(
        "--instance-type-cpu",
        type=str,
        default=CPU_INSTANCE,
        help="CPU instance type for processing jobs",
    )
    parser.add_argument(
        "--model-uri",
        type=str,
        default="",
        help="S3 URI to existing teacher model (skip teacher training in Phase 4)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print what would be submitted without actually running",
    )
    parser.add_argument(
        "--no-wait",
        action="store_true",
        help="Submit jobs and return immediately (don't wait for completion)",
    )
    parser.add_argument(
        "--skip-fidelity-gate",
        action="store_true",
        default=True,
        help="Skip fidelity gate in Phase 4 distillation (default: True)",
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=3,
        help="Max retry attempts for processing jobs on failure (default: 3)",
    )
    parser.add_argument(
        "--fresh",
        action="store_true",
        help="Force a fresh run, ignoring any saved state from a previous run",
    )
    parser.add_argument(
        "--config-path",
        type=str,
        default=CONFIG_PATH,
        help="Path to pipeline YAML config (default: %(default)s)",
    )
    parser.add_argument(
        "--feature-groups-path",
        type=str,
        default=FEATURE_GROUPS_PATH,
        help="Path to feature_groups YAML config (default: %(default)s)",
    )
    return parser.parse_args()


# ===================================================================
# Source packaging
# ===================================================================

def _prepare_source_package(output_dir: str) -> str:
    """Package the project source into a tar.gz for SageMaker upload.

    Includes ``core/``, ``configs/``, ``containers/``, ``scripts/``, and
    ``adapters/`` directories, excluding ``__pycache__``, ``.git``, and
    large data files.

    Returns the path to the created tar.gz file.
    """
    tar_path = os.path.join(output_dir, "source.tar.gz")
    root = str(PROJECT_ROOT)

    include_dirs = ["core", "configs", "containers", "scripts", "adapters"]
    include_files = ["setup.py", "setup.cfg", "pyproject.toml", "requirements.txt"]

    skip_patterns = {"__pycache__", ".git", ".eggs", "*.egg-info", "node_modules"}
    skip_top_dirs = {"data"}

    def _should_skip(name: str) -> bool:
        for pat in skip_patterns:
            if pat in name:
                return True
        return False

    with tarfile.open(tar_path, "w:gz") as tar:
        for d in include_dirs:
            if d in skip_top_dirs:
                continue
            full_path = os.path.join(root, d)
            if os.path.isdir(full_path):
                for dirpath, dirnames, filenames in os.walk(full_path):
                    dirnames[:] = [dn for dn in dirnames if not _should_skip(dn)]
                    for fn in filenames:
                        if _should_skip(fn) or fn.endswith((".pyc", ".pyo")):
                            continue
                        filepath = os.path.join(dirpath, fn)
                        arcname = os.path.relpath(filepath, root)
                        tar.add(filepath, arcname=arcname)

        for f in include_files:
            full_path = os.path.join(root, f)
            if os.path.isfile(full_path):
                tar.add(full_path, arcname=f)

    logger.info(
        "Source package created: %s (%.1f MB)",
        tar_path,
        os.path.getsize(tar_path) / 1024 / 1024,
    )
    return tar_path


def _prepare_and_extract_source() -> str:
    """Build source package ONCE and extract to a temp directory.

    Returns the path to the extracted source directory, suitable for
    passing as ``source_dir`` to SageMaker Estimator constructions.
    This avoids rebuilding the tar.gz for every job submission.
    """
    tmp = tempfile.mkdtemp(prefix="santander_src_pkg_")
    tar_path = _prepare_source_package(tmp)
    extract_dir = Path(tempfile.mkdtemp(prefix="santander_src_ext_")) / "source"
    extract_dir.mkdir(parents=True, exist_ok=True)
    with tarfile.open(tar_path, "r:gz") as tar:
        tar.extractall(str(extract_dir))
    logger.info("Source extracted once to: %s", extract_dir)
    return str(extract_dir)


# ===================================================================
# Orchestrator state file — crash recovery (Issue 14)
# ===================================================================

_STATE_FILE = PROJECT_ROOT / ".ablation_state.json"


def _save_state(state: Dict[str, Any]) -> None:
    """Persist orchestrator state to a local JSON file for crash recovery.

    Automatically extracts and includes a ``failed_jobs`` section from any
    phase results present in the state dict.
    """
    # Build failed_jobs section from phase results
    failed_jobs = []
    for key, val in state.items():
        if not isinstance(val, list):
            continue
        for r in val:
            if isinstance(r, dict) and r.get("status") in ("Failed", "Stopped"):
                failed_jobs.append({
                    "name": r.get("job_name", r.get("scenario", "unknown")),
                    "status": r["status"],
                    "failure_reason": r.get("failure_reason", ""),
                    "billable_seconds": r.get("billable_seconds", 0),
                })
    if failed_jobs:
        state["failed_jobs"] = failed_jobs

    state["_updated_at"] = datetime.now().isoformat()
    with open(_STATE_FILE, "w") as f:
        json.dump(state, f, indent=2, default=str)
    logger.debug("State saved to %s", _STATE_FILE)


def _load_state() -> Optional[Dict[str, Any]]:
    """Load orchestrator state from a previous run, if available."""
    if _STATE_FILE.exists():
        try:
            with open(_STATE_FILE) as f:
                state = json.load(f)
            logger.info("Loaded previous state from %s (ts=%s)",
                        _STATE_FILE, state.get("timestamp", "?"))
            return state
        except (json.JSONDecodeError, OSError) as e:
            logger.warning("Failed to load state file: %s", e)
    return None


def _recover_running_jobs(state: Dict[str, Any]) -> List[str]:
    """Check SageMaker API for jobs that are still running from a previous run.

    Matches jobs whose name contains the state timestamp.
    """
    import boto3
    sm = boto3.client("sagemaker", region_name=REGION)
    ts = state.get("timestamp", "")
    if not ts:
        return []

    running: List[str] = []
    # Check training jobs
    try:
        resp = sm.list_training_jobs(StatusEquals="InProgress", MaxResults=100)
        for job in resp.get("TrainingJobSummaries", []):
            if ts in job["TrainingJobName"]:
                running.append(job["TrainingJobName"])
    except Exception as e:
        logger.warning("Failed to list running training jobs: %s", e)

    # Check processing jobs
    try:
        resp = sm.list_processing_jobs(StatusEquals="InProgress", MaxResults=100)
        for job in resp.get("ProcessingJobSummaries", []):
            if ts in job["ProcessingJobName"]:
                running.append(job["ProcessingJobName"])
    except Exception as e:
        logger.warning("Failed to list running processing jobs: %s", e)

    if running:
        logger.info("Recovered %d running jobs from previous run: %s", len(running), running)
    return running


# ===================================================================
# Utilities
# ===================================================================

def _is_json_serializable(value: Any) -> bool:
    """Return True if value is safely JSON-serializable."""
    try:
        json.dumps(value, default=str)
        return True
    except (TypeError, ValueError):
        return False


def _sanitize_job_name(name: str) -> str:
    """Sanitize a SageMaker job name to meet API constraints.

    SageMaker job names must match: ^[a-zA-Z0-9](-*[a-zA-Z0-9]){0,62}$
    Max length 63 characters.
    """
    sanitized = re.sub(r"[^a-zA-Z0-9-]", "-", name)
    sanitized = re.sub(r"-{2,}", "-", sanitized)
    sanitized = sanitized.strip("-")
    if len(sanitized) > 63:
        sanitized = sanitized[:63].rstrip("-")
    return sanitized


# ===================================================================
# S3 helpers
# ===================================================================

def _s3_base_path(timestamp: str) -> str:
    return f"s3://{S3_BUCKET}/santander-ablation/{timestamp}"


def _upload_source_to_s3(local_path: str, s3_base: str) -> str:
    """Upload source package to S3 and return the S3 URI."""
    import boto3
    s3 = boto3.client("s3", region_name=REGION)
    key = f"{s3_base.replace(f's3://{S3_BUCKET}/', '')}/source/source.tar.gz"
    s3.upload_file(local_path, S3_BUCKET, key)
    uri = f"s3://{S3_BUCKET}/{key}"
    logger.info("Source uploaded to %s", uri)
    return uri


def _download_json_from_s3(s3_uri: str) -> Optional[dict]:
    """Download a JSON file from S3 and parse it."""
    import boto3
    try:
        s3 = boto3.client("s3", region_name=REGION)
        parts = s3_uri.replace("s3://", "").split("/", 1)
        bucket, key = parts[0], parts[1]
        obj = s3.get_object(Bucket=bucket, Key=key)
        return json.loads(obj["Body"].read().decode("utf-8"))
    except Exception as e:
        logger.warning("Failed to download %s: %s", s3_uri, e)
        return None


def _list_s3_keys(prefix: str) -> List[str]:
    """List all keys under an S3 prefix."""
    import boto3
    s3 = boto3.client("s3", region_name=REGION)
    keys: List[str] = []
    paginator = s3.get_paginator("list_objects_v2")
    s3_prefix = prefix.replace(f"s3://{S3_BUCKET}/", "")
    for page in paginator.paginate(Bucket=S3_BUCKET, Prefix=s3_prefix):
        for obj in page.get("Contents", []):
            keys.append(obj["Key"])
    return keys


# ===================================================================
# SageMaker Job submission helpers
# ===================================================================

def _wait_for_training_job(job_name: str) -> str:
    """Wait for a SageMaker Training Job to complete. Returns final status."""
    import boto3
    sm = boto3.client("sagemaker", region_name=REGION)
    logger.info("Waiting for training job: %s", job_name)

    while True:
        desc = sm.describe_training_job(TrainingJobName=job_name)
        status = desc["TrainingJobStatus"]
        if status in ("Completed", "Failed", "Stopped"):
            billable = desc.get("BillableTimeInSeconds", 0)
            spot_savings = desc.get("TrainingTimeInSeconds", 0) - billable if desc.get("EnableManagedSpotTraining") else 0
            logger.info("Job %s: status=%s, billable=%ds (%.2f hrs), spot savings=%ds",
                        job_name, status, billable, billable / 3600, spot_savings)
            if status == "Failed":
                reason = desc.get("FailureReason", "unknown")
                log_url = (
                    f"https://{REGION}.console.aws.amazon.com/cloudwatch/home?"
                    f"region={REGION}#logsV2:log-groups/log-group/"
                    f"$252Faws$252Fsagemaker$252FTrainingJobs/log-events/{job_name}"
                )
                logger.error("Job %s FAILED: %s", job_name, reason[:200])
                logger.error("CloudWatch logs: %s", log_url)
            if status == "Stopped":
                transitions = desc.get("SecondaryStatusTransitions", [])
                was_spot = any(t.get("Status") == "Interrupted" for t in transitions)
                if was_spot:
                    logger.warning("SPOT INTERRUPTION: %s — will auto-retry", job_name)
                else:
                    logger.info("Job %s stopped (manual or eviction)", job_name)
            return status
        time.sleep(30)


def _wait_for_processing_job(job_name: str) -> str:
    """Wait for a SageMaker Processing Job to complete. Returns final status."""
    import boto3
    sm = boto3.client("sagemaker", region_name=REGION)
    logger.info("Waiting for processing job: %s", job_name)

    while True:
        desc = sm.describe_processing_job(ProcessingJobName=job_name)
        status = desc["ProcessingJobStatus"]
        if status in ("Completed", "Failed", "Stopped"):
            logger.info("Job %s -> %s", job_name, status)
            if status == "Failed":
                reason = desc.get("FailureReason", "unknown")
                log_url = (
                    f"https://{REGION}.console.aws.amazon.com/cloudwatch/home?"
                    f"region={REGION}#logsV2:log-groups/log-group/"
                    f"$252Faws$252Fsagemaker$252FProcessingJobs/log-events/{job_name}"
                )
                logger.error("Job %s FAILED: %s", job_name, reason[:200])
                logger.error("CloudWatch logs: %s", log_url)
            return status
        time.sleep(30)


def _get_model_artifact_uri(job_name: str) -> Optional[str]:
    """Retrieve the S3 model artifact URI from a completed Training Job."""
    import boto3
    sm = boto3.client("sagemaker", region_name=REGION)
    desc = sm.describe_training_job(TrainingJobName=job_name)
    return desc.get("ModelArtifacts", {}).get("S3ModelArtifacts")


def _wait_for_any_job(
    job_names: List[str],
    expected_runtime_s: int = 7200,
) -> Tuple[str, str]:
    """Wait for ANY of the given training jobs to complete, with timeout eviction.

    Parameters
    ----------
    job_names : list
        List of SageMaker training job names to monitor.
    expected_runtime_s : int
        Expected runtime in seconds. Jobs running longer than 2x this value
        will be stopped (evicted) to prevent runaway costs.

    Returns (job_name, status) of the first job to finish or be evicted.
    """
    import boto3
    sm = boto3.client("sagemaker", region_name=REGION)
    logger.info("Waiting for any of %d jobs: %s", len(job_names), job_names)

    deadline = time.time() + expected_runtime_s * 2  # 2x expected as hard timeout
    job_start_times: Dict[str, float] = {}

    while True:
        now = time.time()

        # Global deadline exceeded — evict the longest-running job
        if now > deadline:
            logger.warning(
                "Global deadline exceeded (%.0fs). Evicting longest-running job.",
                expected_runtime_s * 2,
            )
            oldest_job = min(job_start_times, key=job_start_times.get) if job_start_times else job_names[0]
            try:
                sm.stop_training_job(TrainingJobName=oldest_job)
                logger.warning("Evicted (stopped) job: %s", oldest_job)
            except Exception as e:
                logger.error("Failed to stop job %s: %s", oldest_job, e)
            return oldest_job, "Evicted"

        for jn in job_names:
            desc = sm.describe_training_job(TrainingJobName=jn)
            status = desc["TrainingJobStatus"]

            # Track when we first saw each job
            if jn not in job_start_times:
                creation = desc.get("CreationTime")
                if creation:
                    job_start_times[jn] = creation.timestamp()
                else:
                    job_start_times[jn] = now

            if status in ("Completed", "Failed", "Stopped"):
                logger.info("Job %s -> %s", jn, status)
                return jn, status

        time.sleep(30)


def _check_budget(budget_limit: float) -> bool:
    """Check current AWS spend against budget limit. Returns True if under budget."""
    import boto3
    from datetime import timedelta

    try:
        ce = boto3.client("ce", region_name="us-east-1")
        end = (datetime.now() + timedelta(days=1)).strftime("%Y-%m-%d")
        start = (datetime.now() - timedelta(days=30)).strftime("%Y-%m-%d")
        resp = ce.get_cost_and_usage(
            TimePeriod={"Start": start, "End": end},
            Granularity="MONTHLY",
            Metrics=["UnblendedCost"],
        )
        total = sum(float(r["Total"]["UnblendedCost"]["Amount"]) for r in resp["ResultsByTime"])
        if total >= budget_limit:
            logger.error("BUDGET EXCEEDED: $%.2f >= $%.2f limit. Stopping.", total, budget_limit)
            return False
        logger.info("Budget check: $%.2f / $%.2f (%.0f%% used)", total, budget_limit, total / budget_limit * 100)
        return True
    except Exception as e:
        logger.warning("Budget check failed (proceeding): %s", e)
        return True


def _enrich_failure_info(job_name: str, result: Dict[str, Any]) -> None:
    """Fetch failure reason and billable seconds from a finished job and add to result."""
    import boto3
    try:
        sm = boto3.client("sagemaker", region_name=REGION)
        desc = sm.describe_training_job(TrainingJobName=job_name)
        result["failure_reason"] = desc.get("FailureReason", "")[:500]
        result["billable_seconds"] = desc.get("BillableTimeInSeconds", 0)
    except Exception as e:
        logger.warning("Could not fetch failure info for %s: %s", job_name, e)


def _run_scenarios_parallel(
    scenarios: List[Dict[str, Any]],
    make_job_fn,
    max_parallel: int = 4,
    args: Optional[argparse.Namespace] = None,
) -> List[Dict[str, Any]]:
    """Run training scenarios with up to max_parallel concurrent jobs.

    Alternates between spot and on-demand to utilize both quotas.
    Submits up to max_parallel jobs, waits for one to finish,
    then submits the next.

    Args:
        scenarios: list of scenario dicts
        make_job_fn: callable(scenario, use_spot) -> result dict
        max_parallel: max concurrent jobs (default 2: 1 spot + 1 on-demand)
        args: argparse namespace
    """
    results: List[Dict[str, Any]] = []
    running: Dict[str, Dict[str, Any]] = {}  # job_name -> result

    # Check for already-completed scenarios (resume support)
    skip_check_s3_base = os.environ.get("SANTANDER_S3_BASE", "")

    for i, scenario in enumerate(scenarios):
        # Skip if result already exists on S3 (from previous run)
        if skip_check_s3_base and not (args and args.dry_run):
            scenario_name = scenario.get("name", "")
            # Infer phase from scenario structure
            if "remove" in scenario:
                check_path = f"{skip_check_s3_base}/phase1/{scenario_name}/output/eval_metrics.json"
            elif "experts" in scenario:
                check_path = f"{skip_check_s3_base}/phase2/{scenario_name}/output/eval_metrics.json"
            elif "tier" in scenario and "structure" in scenario:
                check_path = f"{skip_check_s3_base}/phase3/{scenario_name}/output/eval_metrics.json"
            else:
                check_path = ""
            if check_path:
                existing = _download_json_from_s3(check_path)
                if existing:
                    logger.info("SKIP: %s (result already on S3)", scenario_name)
                    results.append({"scenario": scenario_name, "status": "Reused", "metrics": existing})
                    continue

        # Budget guard: check cost before each batch of submissions
        if not (args and args.dry_run):
            budget_limit = _PIPELINE_CONFIG.get("ablation", {}).get("budget_limit", 100.0)
            if not _check_budget(budget_limit):
                logger.error("Stopping scenario submissions due to budget limit.")
                break

        # 50/50 spot/on-demand for stability
        use_spot = (i % 2 == 0)  # alternating: spot, on-demand, spot, on-demand

        result = make_job_fn(scenario, use_spot)
        results.append(result)

        if args and args.dry_run:
            continue

        if result.get("status") == "InProgress":
            running[result["job_name"]] = result

        # When we hit max_parallel, wait for one to finish
        if len(running) >= max_parallel:
            finished_name, finished_status = _wait_for_any_job(list(running.keys()))
            running[finished_name]["status"] = finished_status
            if finished_status == "Completed":
                running[finished_name]["model_uri"] = _get_model_artifact_uri(finished_name)
            if finished_status in ("Failed", "Stopped"):
                _enrich_failure_info(finished_name, running[finished_name])
            del running[finished_name]

    # Wait for remaining running jobs
    while running:
        finished_name, finished_status = _wait_for_any_job(list(running.keys()))
        running[finished_name]["status"] = finished_status
        if finished_status == "Completed":
            running[finished_name]["model_uri"] = _get_model_artifact_uri(finished_name)
        if finished_status in ("Failed", "Stopped"):
            _enrich_failure_info(finished_name, running[finished_name])
        del running[finished_name]

    return results


def _submit_training_job(
    job_name: str,
    s3_output: str,
    instance_type: str,
    hyperparameters: Dict[str, str],
    source_uri: str,
    data_uri: str,
    wait: bool = True,
    dry_run: bool = False,
    use_spot: bool = True,
    source_dir: Optional[str] = None,
) -> Dict[str, Any]:
    """Submit a SageMaker PyTorch Training Job.

    Parameters
    ----------
    job_name : str
        Unique training job name.
    s3_output : str
        S3 URI for model output artifacts.
    instance_type : str
        SageMaker instance type.
    hyperparameters : dict
        Hyperparameters to pass to the training script.
    source_uri : str
        S3 URI to the source.tar.gz package.
    data_uri : str
        S3 URI to the training data.
    wait : bool
        Whether to block until job completes.
    dry_run : bool
        Print config without submitting.
    source_dir : str, optional
        Pre-built source directory (from ``_prepare_and_extract_source``).
        If provided, skips per-job source packaging for efficiency.

    Returns
    -------
    dict
        Job metadata including name, status, model_uri.
    """
    result: Dict[str, Any] = {
        "job_name": job_name,
        "instance_type": instance_type,
        "hyperparameters": hyperparameters,
        "s3_output": s3_output,
        "data_uri": data_uri,
    }

    if dry_run:
        logger.info("[DRY RUN] Training Job: %s", job_name)
        logger.info("  Instance: %s", instance_type)
        logger.info("  Output: %s", s3_output)
        logger.info(
            "  Hyperparameters: %s",
            json.dumps(hyperparameters, indent=2, ensure_ascii=False),
        )
        result["status"] = "DRY_RUN"
        return result

    import sagemaker
    from sagemaker.inputs import TrainingInput
    from sagemaker.pytorch import PyTorch

    # SageMaker Debugger/Profiler — DISABLED to avoid costly ProfilerReport
    # Processing Jobs (~$1.50/job). Enable only for debugging specific jobs.
    profiler_config = None
    profiler_rules = None

    # SageMaker Training Metrics — regex patterns for CloudWatch capture
    metric_definitions = [
        {"Name": "train:loss", "Regex": r"train_loss=([\d.]+)"},
        {"Name": "val:loss", "Regex": r"val_loss=([\d.]+)"},
        {"Name": "val:avg_auc", "Regex": r"avg_auc=([\d.]+)"},
        {"Name": "val:avg_mae", "Regex": r"avg_mae=([\d.]+)"},
        {"Name": "epoch", "Regex": r"Epoch (\d+):"},
    ]

    session = sagemaker.Session()

    env = {
        "PYTHONIOENCODING": "utf-8",
        "NCCL_DEBUG": "WARN",
    }

    # Use pre-built source directory if provided, or fall back to module global,
    # or build per-job as last resort (legacy fallback)
    effective_source_dir = source_dir or _GLOBAL_SOURCE_DIR
    if effective_source_dir:
        pkg_dir = effective_source_dir
    else:
        tmp_dir = tempfile.mkdtemp()
        source_tar = _prepare_source_package(tmp_dir)
        pkg_dir_path = Path(tempfile.mkdtemp()) / "source"
        pkg_dir_path.mkdir(parents=True, exist_ok=True)
        with tarfile.open(source_tar, "r:gz") as tar:
            tar.extractall(str(pkg_dir_path))
        pkg_dir = str(pkg_dir_path)

    max_run = 28800  # 8 hours (safety margin; actual ~1h with batch 4096 + AMP)

    estimator_kwargs: Dict[str, Any] = {
        "entry_point": "containers/training/train.py",
        "source_dir": str(pkg_dir),
        "role": ROLE_ARN,
        "instance_type": instance_type,
        "instance_count": 1,
        "framework_version": PYTORCH_VERSION,
        "py_version": PY_VERSION,
        "output_path": s3_output,
        "hyperparameters": hyperparameters,
        "environment": env,
        "use_spot_instances": use_spot,
        "max_run": max_run,
        # max_wait = max_run + 1h buffer for spot provisioning (was 36000 fixed)
        **({"max_wait": max_run + 3600} if use_spot else {}),
        "tags": [
            {"Key": "Project", "Value": "santander-ablation"},
            {"Key": "Phase", "Value": hyperparameters.get("ablation_phase", "unknown")},
        ],
        "sagemaker_session": session,
        "metric_definitions": metric_definitions,
        # SageMaker checkpoint syncing — trainer saves to /opt/ml/checkpoints/
        "checkpoint_s3_uri": f"{s3_output.rstrip('/')}/checkpoints",
        "checkpoint_local_path": "/opt/ml/checkpoints",
        # Explicitly disable profiler to avoid costly ProfilerReport Processing Jobs
        "disable_profiler": True,
    }

    # Warm pool: keep instance alive for 5 min between jobs to avoid cold starts.
    # Requires SageMaker SDK >= 2.100.
    estimator_kwargs["keep_alive_period_in_seconds"] = 300

    # Add profiler config if available
    if profiler_config is not None:
        estimator_kwargs["profiler_config"] = profiler_config
    if profiler_rules is not None:
        estimator_kwargs["rules"] = profiler_rules

    estimator = PyTorch(**estimator_kwargs)

    inputs = {
        "train": TrainingInput(
            data_uri,
            distribution="FullyReplicated",
        ),
    }

    job_name = _sanitize_job_name(job_name)
    logger.info("Submitting Training Job: %s", job_name)
    estimator.fit(inputs, job_name=job_name, wait=False)

    if wait:
        status = _wait_for_training_job(job_name)
        result["status"] = status
        if status == "Completed":
            result["model_uri"] = _get_model_artifact_uri(job_name)
    else:
        result["status"] = "InProgress"

    return result


def _submit_processing_job(
    job_name: str,
    script: str,
    s3_output: str,
    instance_type: str,
    arguments: List[str],
    inputs: Optional[List[Any]] = None,
    wait: bool = True,
    dry_run: bool = False,
    use_gpu: bool = False,
    max_retries: int = 0,
    source_tar_path: Optional[str] = None,
) -> Dict[str, Any]:
    """Submit a SageMaker Processing Job (SKLearn or PyTorch).

    Parameters
    ----------
    job_name : str
        Unique processing job name.
    script : str
        Python script to run (relative to source_dir).
    s3_output : str
        S3 URI for output.
    instance_type : str
        SageMaker instance type.
    arguments : list
        CLI arguments for the script.
    inputs : list, optional
        ProcessingInput objects.
    wait : bool
        Whether to block until job completes.
    dry_run : bool
        Print config without submitting.
    use_gpu : bool
        If True, use PyTorchProcessor with GPU deps instead of SKLearnProcessor.
    max_retries : int
        Number of retry attempts on failure (0 = no retries).
    source_tar_path : str, optional
        Pre-built source tar.gz path. If provided, skips per-job source
        packaging for efficiency.

    Returns
    -------
    dict
        Job metadata.
    """
    result: Dict[str, Any] = {
        "job_name": job_name,
        "instance_type": instance_type,
        "script": script,
        "s3_output": s3_output,
    }

    if dry_run:
        logger.info("[DRY RUN] Processing Job: %s", job_name)
        logger.info("  Script: %s", script)
        logger.info("  Instance: %s", instance_type)
        logger.info("  Output: %s", s3_output)
        logger.info("  Arguments: %s", arguments)
        logger.info("  GPU mode: %s", use_gpu)
        result["status"] = "DRY_RUN"
        return result

    import sagemaker
    from sagemaker.processing import ProcessingInput, ProcessingOutput

    session = sagemaker.Session()

    # Use pre-built source tar.gz if provided, or fall back to module global,
    # or build per-job as last resort (legacy fallback)
    effective_tar = source_tar_path or _GLOBAL_SOURCE_TAR
    if effective_tar:
        source_tar = effective_tar
    else:
        tmp_dir = tempfile.mkdtemp()
        source_tar = _prepare_source_package(tmp_dir)
    s3_source_key = f"santander-ablation/{job_name}/source/source_pkg.tar.gz"
    s3_source = f"s3://{S3_BUCKET}/{s3_source_key}"
    import boto3 as _b3
    _b3.client("s3").upload_file(source_tar, S3_BUCKET, s3_source_key)
    logger.info("Source package uploaded: %s", s3_source)

    # Build pip install list based on whether we need GPU deps
    if use_gpu:
        pip_deps = [
            "pyyaml", "omegaconf", "lightgbm", "pyarrow", "scipy",
            "scikit-learn", "duckdb",
            "hmmlearn", "ripser", "giotto-ph", "persim",
            "cupy-cuda12x", "mamba-ssm",
        ]
    else:
        pip_deps = [
            "pyyaml", "omegaconf", "lightgbm", "pyarrow", "scipy",
            "scikit-learn", "duckdb", "torch",
            "hmmlearn", "ripser", "giotto-ph", "persim",
        ]
    pip_install_str = ", ".join(f'"{d}"' for d in pip_deps)

    # Create wrapper script that unpacks source + installs deps + runs target
    wrapper = Path("_santander_ablation_wrapper.py")
    wrapper.write_text(f"""\
import subprocess, sys, os, tarfile
subprocess.check_call([sys.executable, "-m", "pip", "install", "-q",
                       {pip_install_str}])
src_tar = "/opt/ml/processing/input/source/source_pkg.tar.gz"
src_dir = "/opt/ml/processing/source"
os.makedirs(src_dir, exist_ok=True)
with tarfile.open(src_tar, "r:gz") as tar:
    tar.extractall(src_dir)
import glob
# Unpack any model.tar.gz in teacher input
teacher_dir = "/opt/ml/processing/input/teacher"
if os.path.isdir(teacher_dir):
    for tgz in glob.glob(os.path.join(teacher_dir, "*.tar.gz")):
        with tarfile.open(tgz, "r:gz") as tar:
            tar.extractall(teacher_dir)
        print(f"Unpacked teacher: {{os.listdir(teacher_dir)}}")
sys.path.insert(0, src_dir)
os.chdir(src_dir)
sys.argv = ["{script}"] + {arguments!r}
exec(open("{script}").read())
""")

    all_inputs = [
        ProcessingInput(
            source=s3_source,
            destination="/opt/ml/processing/input/source",
        ),
    ] + (inputs or [])

    outputs = [
        ProcessingOutput(
            source="/opt/ml/processing/output",
            destination=s3_output,
        ),
    ]

    # Retry loop
    attempt = 0
    last_status = "Unknown"
    while attempt <= max_retries:
        attempt_job_name = _sanitize_job_name(
            job_name if attempt == 0 else f"{job_name}-r{attempt}"
        )
        attempt_suffix = (
            f" (attempt {attempt + 1}/{max_retries + 1})" if max_retries > 0 else ""
        )

        if use_gpu:
            from sagemaker.pytorch import PyTorchProcessor
            processor = PyTorchProcessor(
                role=ROLE_ARN,
                framework_version=PYTORCH_VERSION,
                py_version=PY_VERSION,
                instance_type=instance_type,
                instance_count=1,
                sagemaker_session=session,
                env={"PYTHONIOENCODING": "utf-8"},
            )
        else:
            from sagemaker.sklearn import SKLearnProcessor
            processor = SKLearnProcessor(
                role=ROLE_ARN,
                framework_version="1.2-1",
                instance_type=instance_type,
                instance_count=1,
                sagemaker_session=session,
                env={"PYTHONIOENCODING": "utf-8"},
            )

        logger.info("Submitting Processing Job: %s%s", attempt_job_name, attempt_suffix)
        processor.run(
            code=str(wrapper),
            inputs=all_inputs,
            outputs=outputs,
            job_name=attempt_job_name,
            wait=False,
        )

        if wait:
            last_status = _wait_for_processing_job(attempt_job_name)
            if last_status == "Completed" or attempt >= max_retries:
                result["status"] = last_status
                result["job_name"] = attempt_job_name
                if attempt > 0:
                    result["retry_attempts"] = attempt
                break
            logger.warning(
                "Processing Job %s failed -- retrying (%d/%d)",
                attempt_job_name, attempt + 1, max_retries,
            )
            attempt += 1
        else:
            result["status"] = "InProgress"
            result["job_name"] = attempt_job_name
            break

    wrapper.unlink(missing_ok=True)
    return result


# ===================================================================
# Phase implementations
# ===================================================================

def run_phase0(s3_base: str, ts: str, args: argparse.Namespace) -> Dict[str, Any]:
    """Phase 0: Data Preparation -- run santander_adapter via PipelineRunner Stage 1-6.

    Output: features.parquet, labels.parquet, sequences.npy, scaler_params.json
    """
    logger.info("=" * 70)
    logger.info("Phase 0: Data Preparation (santander_adapter)")
    logger.info("=" * 70)

    job_name = f"sant-abl-p0-data-prep-{ts}"
    s3_output = f"{s3_base}/phase0/data/"

    from sagemaker.processing import ProcessingInput

    result = _submit_processing_job(
        job_name=job_name,
        script="adapters/santander_adapter.py",
        s3_output=s3_output,
        instance_type=args.instance_type_cpu,
        inputs=[
            ProcessingInput(
                source=_PIPELINE_CONFIG.get("data", {}).get("s3_path", f"s3://{S3_BUCKET}/data/"),
                destination="/opt/ml/processing/input/raw",
            ),
        ],
        arguments=[
            "--pipeline", CONFIG_PATH,
            "--input-dir", "/opt/ml/processing/input/raw",
            "--output-dir", "/opt/ml/processing/output",
            "--stages", "1-6",
        ],
        wait=not args.no_wait,
        dry_run=args.dry_run,
        use_gpu=False,
        max_retries=args.max_retries,
    )

    result["data_uri"] = s3_output
    return result


def run_phase1(
    s3_base: str,
    ts: str,
    data_uri: str,
    source_uri: str,
    args: argparse.Namespace,
) -> List[Dict[str, Any]]:
    """Phase 1: Feature Group Ablation -- 10 training scenarios.

    Each scenario removes one feature group to measure its contribution
    to the 18-task Santander recommendation model.
    """
    logger.info("=" * 70)
    logger.info(
        "Phase 1: Feature Group Ablation (%d scenarios)", len(FEATURE_SCENARIOS)
    )
    logger.info("=" * 70)

    def make_job(scenario, use_spot):
        name = scenario["name"]
        safe_name = name.replace("_", "-")
        job_name = f"sant-abl-p1-{safe_name}-{ts}"
        s3_output = f"{s3_base}/phase1/{name}/"
        hyperparameters = {
            "config": CONFIG_PATH,
            "ablation_phase": "1",
            "ablation_type": "feature_group",
            "ablation_scenario": name,
            "removed_feature_groups": json.dumps(scenario["remove"]),
            "shared_experts": json.dumps(_experts_for_scenario(scenario["remove"])),
            **TRAINING_DEFAULTS,
            "_s3_output": f"{s3_base}/phase1/{name}",
        }
        active_experts = _experts_for_scenario(scenario["remove"])
        logger.info("--- Scenario: %s (remove=%s, experts=%s, spot=%s) ---",
                     name, scenario["remove"], active_experts, use_spot)
        result = _submit_training_job(
            job_name=job_name, s3_output=s3_output,
            instance_type=args.instance_type_gpu,
            hyperparameters=hyperparameters,
            source_uri=source_uri, data_uri=data_uri,
            wait=False, dry_run=args.dry_run, use_spot=use_spot,
        )
        result["scenario"] = name
        result["phase"] = 1
        return result

    return _run_scenarios_parallel(FEATURE_SCENARIOS, make_job, max_parallel=4, args=args)


def run_phase2(
    s3_base: str,
    ts: str,
    data_uri: str,
    source_uri: str,
    args: argparse.Namespace,
) -> List[Dict[str, Any]]:
    """Phase 2: Expert Ablation -- 7 training scenarios.

    Each scenario removes one shared expert from the PLE basket.
    """
    logger.info("=" * 70)
    logger.info("Phase 2: Expert Ablation (%d scenarios)", len(EXPERT_SCENARIOS))
    logger.info("=" * 70)

    def make_job(scenario, use_spot):
        name = scenario["name"]
        safe_name = name.replace("_", "-")
        job_name = f"sant-abl-p2-{safe_name}-{ts}"
        s3_output = f"{s3_base}/phase2/{name}/"
        hyperparameters = {
            "config": CONFIG_PATH,
            "ablation_phase": "2",
            "ablation_type": "expert",
            "ablation_scenario": name,
            "shared_experts": json.dumps(scenario["experts"]),
            **TRAINING_DEFAULTS,
            "_s3_output": f"{s3_base}/phase2/{name}",
        }
        logger.info("--- Scenario: %s (experts=%s, spot=%s) ---", name, scenario["experts"], use_spot)
        result = _submit_training_job(
            job_name=job_name, s3_output=s3_output,
            instance_type=args.instance_type_gpu,
            hyperparameters=hyperparameters,
            source_uri=source_uri, data_uri=data_uri,
            wait=False, dry_run=args.dry_run, use_spot=use_spot,
        )
        result["scenario"] = name
        result["phase"] = 2
        return result

    return _run_scenarios_parallel(EXPERT_SCENARIOS, make_job, max_parallel=4, args=args)


def run_phase3(
    s3_base: str,
    ts: str,
    data_uri: str,
    source_uri: str,
    args: argparse.Namespace,
) -> List[Dict[str, Any]]:
    """Phase 3: Task x Structure Cross Ablation -- 16 training scenarios.

    This is the KEY experiment: cross 4 task scales (4/8/15/18 tasks) with
    4 structural variants (shared_bottom / ple_only / adatt_only / full).
    """
    n_scenarios = len(TASK_TIERS) * len(STRUCTURE_VARIANTS)
    logger.info("=" * 70)
    logger.info(
        "Phase 3: Task x Structure Cross Ablation (%d scenarios = %d tasks x %d structures)",
        n_scenarios, len(TASK_TIERS), len(STRUCTURE_VARIANTS),
    )
    logger.info("=" * 70)

    # Build flat scenario list for parallel runner
    cross_scenarios = []
    for tier_name, task_list in TASK_TIERS.items():
        for struct_name, struct_flags in STRUCTURE_VARIANTS.items():
            # Skip tasks_18 × full — same as Phase 1 "full" baseline
            if tier_name == "tasks_18" and struct_name == "full":
                continue
            cross_scenarios.append({
                "name": f"{tier_name}-{struct_name}",
                "tier": tier_name,
                "structure": struct_name,
                "task_list": task_list,
                "use_ple": struct_flags["use_ple"],
                "use_adatt": struct_flags["use_adatt"],
            })

    def make_job(scenario, use_spot):
        name = scenario["name"]
        safe_name = name.replace("_", "-")
        job_name = f"sant-abl-p3-{safe_name}-{ts}"
        s3_output = f"{s3_base}/phase3/{name}/"
        hyperparameters = {
            "config": CONFIG_PATH,
            "ablation_phase": "3",
            "ablation_type": "task_structure",
            "ablation_scenario": name,
            "active_tasks": json.dumps(scenario["task_list"]),
            "use_ple": json.dumps(scenario["use_ple"]),
            "use_adatt": json.dumps(scenario["use_adatt"]),
            **TRAINING_DEFAULTS,
            "_s3_output": f"{s3_base}/phase3/{name}",
        }
        logger.info(
            "--- Scenario: %s (%d tasks, PLE=%s, adaTT=%s, spot=%s) ---",
            name, len(scenario["task_list"]),
            scenario["use_ple"], scenario["use_adatt"], use_spot,
        )
        result = _submit_training_job(
            job_name=job_name, s3_output=s3_output,
            instance_type=args.instance_type_gpu,
            hyperparameters=hyperparameters,
            source_uri=source_uri, data_uri=data_uri,
            wait=False, dry_run=args.dry_run, use_spot=use_spot,
        )
        result["scenario"] = name
        result["tier"] = scenario["tier"]
        result["structure"] = scenario["structure"]
        result["phase"] = 3
        return result

    return _run_scenarios_parallel(cross_scenarios, make_job, max_parallel=4, args=args)


def _select_best_config(
    phase1_results: List[Dict[str, Any]],
    phase2_results: List[Dict[str, Any]],
    phase3_results: List[Dict[str, Any]],
    s3_base: str,
) -> Dict[str, Any]:
    """Analyze Phase 1-3 results and select the best configuration.

    Collects eval_metrics.json from each completed scenario, ranks by
    aggregate_score, and returns the best combination of feature group,
    expert basket, task tier, and structure variant.

    Returns
    -------
    dict
        Best configuration with keys: feature_scenario, expert_scenario,
        task_tier, structure_variant, and the aggregate metric score.
    """
    best: Dict[str, Any] = {
        "feature_scenario": "full",
        "expert_scenario": "full_basket",
        "task_tier": "tasks_18",
        "structure_variant": "full",
        "use_ple": True,
        "use_adatt": True,
        "active_tasks": TASK_TIERS["tasks_18"],
        "best_metric": None,
    }

    # -- Phase 1: best feature config --
    best_feature_score = -float("inf")
    for r in phase1_results:
        if r.get("status") != "Completed":
            continue
        s3_path = f"{s3_base}/phase1/{r['scenario']}/output/eval_metrics.json"
        metrics = _download_json_from_s3(s3_path)
        if metrics:
            score = metrics.get("aggregate_score", 0.0)
            if score > best_feature_score:
                best_feature_score = score
                best["feature_scenario"] = r["scenario"]

    # -- Phase 2: best expert config --
    best_expert_score = -float("inf")
    for r in phase2_results:
        if r.get("status") != "Completed":
            continue
        s3_path = f"{s3_base}/phase2/{r['scenario']}/output/eval_metrics.json"
        metrics = _download_json_from_s3(s3_path)
        if metrics:
            score = metrics.get("aggregate_score", 0.0)
            if score > best_expert_score:
                best_expert_score = score
                best["expert_scenario"] = r["scenario"]

    # -- Phase 3: best task x structure config --
    best_cross_score = -float("inf")
    for r in phase3_results:
        if r.get("status") != "Completed":
            continue
        s3_path = f"{s3_base}/phase3/{r['scenario']}/output/eval_metrics.json"
        metrics = _download_json_from_s3(s3_path)
        if metrics:
            score = metrics.get("aggregate_score", 0.0)
            if score > best_cross_score:
                best_cross_score = score
                tier = r.get("tier", "tasks_18")
                struct = r.get("structure", "full")
                best["task_tier"] = tier
                best["structure_variant"] = struct
                best["active_tasks"] = TASK_TIERS.get(tier, TASK_TIERS["tasks_18"])
                best["use_ple"] = STRUCTURE_VARIANTS.get(struct, STRUCTURE_VARIANTS["full"])["use_ple"]
                best["use_adatt"] = STRUCTURE_VARIANTS.get(struct, STRUCTURE_VARIANTS["full"])["use_adatt"]

    scores = [s for s in (best_feature_score, best_expert_score, best_cross_score) if s > -float("inf")]
    best["best_metric"] = max(scores) if scores else None

    logger.info("Best config selected:")
    logger.info("  Feature scenario:  %s", best["feature_scenario"])
    logger.info("  Expert scenario:   %s", best["expert_scenario"])
    logger.info("  Task tier:         %s (%d tasks)", best["task_tier"], len(best["active_tasks"]))
    logger.info("  Structure variant: %s (PLE=%s, adaTT=%s)",
                best["structure_variant"], best["use_ple"], best["use_adatt"])

    return best


def run_phase4(
    s3_base: str,
    ts: str,
    data_uri: str,
    source_uri: str,
    args: argparse.Namespace,
    best_config: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Phase 4: Best Config Teacher Training + Knowledge Distillation.

    Trains a teacher model with the best config from Phase 1-3, then runs
    knowledge distillation.
    """
    logger.info("=" * 70)
    logger.info("Phase 4: Best Config Teacher + Distillation")
    logger.info("=" * 70)

    if best_config is None:
        best_config = {
            "feature_scenario": "full",
            "expert_scenario": "full_basket",
            "task_tier": "tasks_18",
            "structure_variant": "full",
            "use_ple": True,
            "use_adatt": True,
            "active_tasks": TASK_TIERS["tasks_18"],
        }

    # -- Step 4a: Train teacher with best config (or use provided model-uri) --
    model_uri = args.model_uri
    teacher_result: Dict[str, Any] = {}

    if not model_uri:
        job_name = f"sant-abl-p4-teacher-{ts}"
        s3_output = f"{s3_base}/phase4/teacher/"

        # Resolve expert list from best scenario
        expert_scenario = next(
            (s for s in EXPERT_SCENARIOS if s["name"] == best_config["expert_scenario"]),
            EXPERT_SCENARIOS[0],
        )
        # Resolve feature removal from best scenario
        feature_scenario = next(
            (s for s in FEATURE_SCENARIOS if s["name"] == best_config["feature_scenario"]),
            FEATURE_SCENARIOS[0],
        )

        hyperparameters = {
            "config": CONFIG_PATH,
            "ablation_phase": "4",
            "ablation_type": "best_config",
            "removed_feature_groups": json.dumps(feature_scenario["remove"]),
            "shared_experts": json.dumps(expert_scenario["experts"]),
            "active_tasks": json.dumps(best_config.get("active_tasks", TASK_TIERS["tasks_18"])),
            "use_ple": json.dumps(best_config.get("use_ple", True)),
            "use_adatt": json.dumps(best_config.get("use_adatt", True)),
            **TRAINING_DEFAULTS,
            "_s3_output": f"{s3_base}/phase4/teacher",
        }

        teacher_result = _submit_training_job(
            job_name=job_name,
            s3_output=s3_output,
            instance_type=args.instance_type_gpu,
            hyperparameters=hyperparameters,
            source_uri=source_uri,
            data_uri=data_uri,
            wait=not args.no_wait,
            dry_run=args.dry_run,
        )

        model_uri = teacher_result.get("model_uri", "")
        if not model_uri and not args.dry_run:
            logger.error("Teacher training failed -- no model artifact")
            return {"status": "Failed", "teacher": teacher_result}
    else:
        logger.info("Using provided teacher model: %s", model_uri)
        teacher_result = {"model_uri": model_uri, "status": "Provided"}

    # -- Step 4b: Distillation --
    distill_job_name = f"sant-abl-p4-distill-{ts}"
    s3_distill_output = f"{s3_base}/phase4/distillation/"

    from sagemaker.processing import ProcessingInput

    distill_inputs: List[Any] = []
    if model_uri and not args.dry_run:
        distill_inputs.append(
            ProcessingInput(
                source=model_uri,
                destination="/opt/ml/processing/input/teacher",
            )
        )
        distill_inputs.append(
            ProcessingInput(
                source=data_uri,
                destination="/opt/ml/processing/input/data",
            )
        )

    distill_temp = str(_PIPELINE_CONFIG.get("distillation", {}).get("temperature", 5.0))
    distill_arguments = [
        "--teacher-checkpoint", "/opt/ml/processing/input/teacher/model.pth",
        "--data-path", "/opt/ml/processing/input/data/",
        "--output-dir", "/opt/ml/processing/output",
        "--config", CONFIG_PATH,
        "--soft-label-path", "/opt/ml/processing/output/soft_labels.parquet",
        "--temperature", distill_temp,
    ]
    if args.skip_fidelity_gate:
        distill_arguments.append("--skip-fidelity-gate")

    distill_result = _submit_processing_job(
        job_name=distill_job_name,
        script="scripts/run_distillation.py",
        s3_output=s3_distill_output,
        instance_type=args.instance_type_cpu,
        arguments=distill_arguments,
        inputs=distill_inputs if not args.dry_run else None,
        wait=not args.no_wait,
        dry_run=args.dry_run,
        max_retries=args.max_retries,
    )

    return {
        "status": distill_result.get("status", "Unknown"),
        "teacher": teacher_result,
        "distillation": distill_result,
        "best_config": best_config,
        "model_uri": model_uri,
    }


def run_phase5(
    s3_base: str,
    ts: str,
    all_results: Dict[str, Any],
    args: argparse.Namespace,
) -> Dict[str, Any]:
    """Phase 5: Analysis + HTML Report Generation.

    Collects all eval_metrics.json from S3, runs Stage 8.5 analysis
    (IG, CCA, gate weights), and generates a 4-dimensional comparison
    HTML report.
    """
    logger.info("=" * 70)
    logger.info("Phase 5: Analysis + HTML Report")
    logger.info("=" * 70)

    # -- Save orchestration manifest to S3 --
    manifest_path = f"{s3_base}/manifest.json"
    manifest: Dict[str, Any] = {
        "timestamp": ts,
        "s3_base": s3_base,
        "dataset": _PIPELINE_CONFIG.get("task_name", "santander"),
        "dimensions": [
            "feature_group (Phase 1)",
            "expert (Phase 2)",
            "task_x_structure (Phase 3)",
            "distillation (Phase 4)",
        ],
        "phases": {},
    }

    for phase_key, phase_results in all_results.items():
        if isinstance(phase_results, list):
            manifest["phases"][phase_key] = [
                {
                    "scenario": r.get("scenario", ""),
                    "job_name": r.get("job_name", ""),
                    "status": r.get("status", "Unknown"),
                    "model_uri": r.get("model_uri", ""),
                    "tier": r.get("tier", ""),
                    "structure": r.get("structure", ""),
                }
                for r in phase_results
            ]
        elif isinstance(phase_results, dict):
            manifest["phases"][phase_key] = {
                k: v for k, v in phase_results.items()
                if _is_json_serializable(v)
            }

    if not args.dry_run:
        import boto3
        s3 = boto3.client("s3", region_name=REGION)
        manifest_key = manifest_path.replace(f"s3://{S3_BUCKET}/", "")
        s3.put_object(
            Bucket=S3_BUCKET,
            Key=manifest_key,
            Body=json.dumps(manifest, indent=2, ensure_ascii=False, default=str),
            ContentType="application/json",
        )
        logger.info("Manifest saved to %s", manifest_path)

    # -- Collect metrics from all phases --
    collected_metrics: Dict[str, Dict[str, Any]] = {}

    # Phase 1 metrics
    for scenario in FEATURE_SCENARIOS:
        s3_path = f"{s3_base}/phase1/{scenario['name']}/output/eval_metrics.json"
        metrics = _download_json_from_s3(s3_path) if not args.dry_run else None
        if metrics:
            collected_metrics[f"p1_{scenario['name']}"] = metrics

    # Phase 2 metrics (full_basket = Phase 1 "full" baseline)
    p1_full = collected_metrics.get("p1_full")
    if p1_full:
        collected_metrics["p2_full_basket"] = p1_full  # reuse as baseline
    for scenario in EXPERT_SCENARIOS:
        s3_path = f"{s3_base}/phase2/{scenario['name']}/output/eval_metrics.json"
        metrics = _download_json_from_s3(s3_path) if not args.dry_run else None
        if metrics:
            collected_metrics[f"p2_{scenario['name']}"] = metrics

    # Phase 3 metrics (task x structure cross; tasks_18-full = Phase 1 "full")
    if p1_full:
        collected_metrics["p3_tasks_18-full"] = p1_full  # reuse as baseline
    for tier_name in TASK_TIERS:
        for struct_name in STRUCTURE_VARIANTS:
            if tier_name == "tasks_18" and struct_name == "full":
                continue  # already reused from Phase 1
            scenario_name = f"{tier_name}-{struct_name}"
            s3_path = f"{s3_base}/phase3/{scenario_name}/output/eval_metrics.json"
            metrics = _download_json_from_s3(s3_path) if not args.dry_run else None
            if metrics:
                collected_metrics[f"p3_{scenario_name}"] = metrics

    logger.info("Collected metrics from %d scenarios", len(collected_metrics))

    # -- Generate HTML report --
    report_output = f"docs/santander_ablation_report_{ts}.html"
    report_path = str(PROJECT_ROOT / report_output)
    logger.info("Generating HTML report: %s", report_output)

    if not args.dry_run:
        _generate_santander_report(
            collected_metrics=collected_metrics,
            manifest=manifest,
            s3_base=s3_base,
            output_path=report_path,
        )
        logger.info("Report generated: %s", report_output)
    else:
        logger.info("[DRY RUN] Would generate report: %s", report_output)

    return {
        "manifest": manifest_path,
        "report": report_output,
        "collected_metrics_count": len(collected_metrics),
    }


# ===================================================================
# HTML Report Generator (4-dimensional comparison)
# ===================================================================

def _generate_santander_report(
    collected_metrics: Dict[str, Dict[str, Any]],
    manifest: Dict[str, Any],
    s3_base: str,
    output_path: str,
) -> None:
    """Generate a 4-dimensional comparison HTML report.

    Includes:
    - Feature ablation bar chart
    - Expert ablation bar chart
    - Task x Structure heatmap (the key visualization)
    - Per-task metric breakdown
    """
    ts = manifest.get("timestamp", "unknown")

    # -- Extract metric summaries --
    feature_scores: Dict[str, float] = {}
    for scenario in FEATURE_SCENARIOS:
        key = f"p1_{scenario['name']}"
        if key in collected_metrics:
            feature_scores[scenario["name"]] = collected_metrics[key].get("aggregate_score", 0.0)

    expert_scores: Dict[str, float] = {}
    for scenario in EXPERT_SCENARIOS:
        key = f"p2_{scenario['name']}"
        if key in collected_metrics:
            expert_scores[scenario["name"]] = collected_metrics[key].get("aggregate_score", 0.0)

    # Task x Structure heatmap data: rows = task tiers, cols = structure variants
    heatmap_data: Dict[str, Dict[str, float]] = {}
    per_task_data: Dict[str, Dict[str, Any]] = {}
    for tier_name in TASK_TIERS:
        heatmap_data[tier_name] = {}
        for struct_name in STRUCTURE_VARIANTS:
            scenario_name = f"{tier_name}-{struct_name}"
            key = f"p3_{scenario_name}"
            if key in collected_metrics:
                heatmap_data[tier_name][struct_name] = collected_metrics[key].get("aggregate_score", 0.0)
                # Collect per-task breakdown
                task_metrics = collected_metrics[key].get("per_task", {})
                if task_metrics:
                    per_task_data[scenario_name] = task_metrics

    # -- Build HTML --
    def _bar_chart_html(title: str, scores: Dict[str, float], color: str) -> str:
        """Render a horizontal bar chart as HTML/CSS."""
        if not scores:
            return f"<h3>{title}</h3><p>No metrics collected.</p>"

        max_score = max(scores.values()) if scores.values() else 1.0
        rows = ""
        for name, score in sorted(scores.items(), key=lambda x: -x[1]):
            pct = (score / max_score * 100) if max_score > 0 else 0
            rows += f"""
            <tr>
                <td style="width:180px;padding:4px 8px;font-size:13px;">{name}</td>
                <td style="padding:4px 0;">
                    <div style="background:{color};width:{pct:.1f}%;height:22px;border-radius:3px;"></div>
                </td>
                <td style="width:80px;padding:4px 8px;text-align:right;font-size:13px;">{score:.4f}</td>
            </tr>"""
        return f"""
        <h3>{title}</h3>
        <table style="width:100%;border-collapse:collapse;">{rows}</table>
        """

    def _heatmap_html(data: Dict[str, Dict[str, float]]) -> str:
        """Render the Task x Structure heatmap as an HTML table."""
        if not data:
            return "<h3>Task x Structure Cross Ablation</h3><p>No metrics collected.</p>"

        struct_names = list(STRUCTURE_VARIANTS.keys())
        tier_names = list(TASK_TIERS.keys())

        # Collect all values for color scaling
        all_vals = [v for row in data.values() for v in row.values()]
        vmin = min(all_vals) if all_vals else 0.0
        vmax = max(all_vals) if all_vals else 1.0
        vrange = vmax - vmin if vmax > vmin else 1.0

        # Header row
        header = "<th style='padding:8px;border:1px solid #ddd;'>Task Tier \\ Structure</th>"
        for s in struct_names:
            header += f"<th style='padding:8px;border:1px solid #ddd;'>{s}</th>"

        rows = ""
        for tier in tier_names:
            cells = f"<td style='padding:8px;border:1px solid #ddd;font-weight:bold;'>{tier} ({len(TASK_TIERS[tier])} tasks)</td>"
            for struct in struct_names:
                val = data.get(tier, {}).get(struct)
                if val is not None:
                    # Color: green gradient based on score
                    intensity = (val - vmin) / vrange
                    r = int(255 - intensity * 155)
                    g = int(100 + intensity * 155)
                    b = int(100 - intensity * 50)
                    cells += (
                        f"<td style='padding:8px;border:1px solid #ddd;"
                        f"background:rgb({r},{g},{b});color:#fff;"
                        f"text-align:center;font-weight:bold;'>{val:.4f}</td>"
                    )
                else:
                    cells += "<td style='padding:8px;border:1px solid #ddd;text-align:center;color:#999;'>N/A</td>"
            rows += f"<tr>{cells}</tr>"

        return f"""
        <h3>Task x Structure Cross Ablation (KEY Experiment)</h3>
        <p>Rows = task-tier scale, Columns = structural variant.
           Higher scores are greener.</p>
        <table style="width:100%;border-collapse:collapse;margin:10px 0;">
            <tr>{header}</tr>
            {rows}
        </table>
        """

    def _per_task_html(per_task: Dict[str, Dict[str, Any]]) -> str:
        """Render per-task metric breakdown table."""
        if not per_task:
            return "<h3>Per-Task Metric Breakdown</h3><p>No per-task data collected.</p>"

        # Collect all task names across all scenarios
        all_tasks: set = set()
        for metrics in per_task.values():
            all_tasks.update(metrics.keys())
        sorted_tasks = sorted(all_tasks)

        scenarios = sorted(per_task.keys())

        header = "<th style='padding:6px;border:1px solid #ddd;font-size:12px;'>Task</th>"
        for s in scenarios:
            label = s.replace("_", " ")
            header += f"<th style='padding:6px;border:1px solid #ddd;font-size:11px;writing-mode:vertical-lr;'>{label}</th>"

        rows = ""
        for task in sorted_tasks:
            cells = f"<td style='padding:6px;border:1px solid #ddd;font-size:12px;'>{task}</td>"
            for scenario in scenarios:
                val = per_task[scenario].get(task)
                if val is not None:
                    if isinstance(val, dict):
                        # Pick the primary metric if it's a dict
                        val = val.get("auc", val.get("accuracy", val.get("score", 0.0)))
                    cells += f"<td style='padding:6px;border:1px solid #ddd;text-align:center;font-size:12px;'>{float(val):.4f}</td>"
                else:
                    cells += "<td style='padding:6px;border:1px solid #ddd;text-align:center;color:#999;'>-</td>"
            rows += f"<tr>{cells}</tr>"

        return f"""
        <h3>Per-Task Metric Breakdown (Phase 3)</h3>
        <div style="overflow-x:auto;">
        <table style="border-collapse:collapse;margin:10px 0;">
            <tr>{header}</tr>
            {rows}
        </table>
        </div>
        """

    feature_chart = _bar_chart_html(
        "Dimension 1: Feature Group Ablation", feature_scores, "#4a90d9"
    )
    expert_chart = _bar_chart_html(
        "Dimension 2: Expert Ablation", expert_scores, "#e6854a"
    )
    heatmap = _heatmap_html(heatmap_data)
    per_task_table = _per_task_html(per_task_data)

    html = f"""\
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>Santander 4-Dimension Ablation Report ({ts})</title>
    <style>
        body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
               max-width: 1200px; margin: 40px auto; padding: 0 20px; color: #333; }}
        h1 {{ color: #2c3e50; border-bottom: 3px solid #3498db; padding-bottom: 12px; }}
        h2 {{ color: #34495e; margin-top: 40px; }}
        h3 {{ color: #555; margin-top: 24px; }}
        .summary {{ background: #f8f9fa; border-left: 4px solid #3498db;
                    padding: 16px 20px; margin: 20px 0; border-radius: 4px; }}
        .section {{ margin: 30px 0; padding: 20px; background: #fff;
                    border: 1px solid #e0e0e0; border-radius: 6px; }}
        table {{ font-family: monospace; }}
    </style>
</head>
<body>
    <h1>Santander Product Recommendation -- 4-Dimension Ablation Report</h1>

    <div class="summary">
        <strong>Timestamp:</strong> {ts}<br>
        <strong>Dataset:</strong> Santander Product Recommendation (18 tasks)<br>
        <strong>S3 Base:</strong> {s3_base}<br>
        <strong>Dimensions:</strong> Feature Groups (10) | Experts (7) | Task x Structure (4x4=16) | Distillation<br>
        <strong>Total Training Scenarios:</strong> {len(FEATURE_SCENARIOS) + len(EXPERT_SCENARIOS) + len(TASK_TIERS) * len(STRUCTURE_VARIANTS)}
    </div>

    <div class="section">
        <h2>Dimension 1: Feature Group Ablation</h2>
        <p>Remove one feature group at a time from the full 18-task config to measure contribution.</p>
        {feature_chart}
    </div>

    <div class="section">
        <h2>Dimension 2: Expert Ablation</h2>
        <p>Remove one shared expert from the PLE basket to measure expert contribution.</p>
        {expert_chart}
    </div>

    <div class="section">
        <h2>Dimension 3: Task x Structure Cross Ablation</h2>
        <p>The <strong>key experiment</strong>: cross 4 task scales with 4 structural variants to
           understand how PLE and adaTT interact with task complexity.</p>
        {heatmap}
    </div>

    <div class="section">
        <h2>Per-Task Metric Breakdown</h2>
        <p>Detailed per-task performance across Phase 3 scenarios.</p>
        {per_task_table}
    </div>

    <footer style="margin-top:40px;padding:20px 0;border-top:1px solid #ddd;color:#888;font-size:13px;">
        Generated by <code>scripts/run_santander_ablation.py</code> | {ts}
    </footer>
</body>
</html>
"""

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html)

    logger.info("HTML report written to %s", output_path)


# ===================================================================
# Main orchestration
# ===================================================================

def main() -> None:
    global CONFIG_PATH, FEATURE_GROUPS_PATH
    args = parse_args()

    # Allow CLI to override config paths
    if args.config_path != CONFIG_PATH:
        CONFIG_PATH = args.config_path
    if args.feature_groups_path != FEATURE_GROUPS_PATH:
        FEATURE_GROUPS_PATH = args.feature_groups_path

    # Auto-resume from previous run state unless --fresh is specified
    existing_state = _load_state()
    if existing_state and not args.fresh:
        ts = existing_state.get("timestamp", "")
        s3_base = existing_state.get("s3_base", "")
        if ts and s3_base:
            logger.info("Resuming from previous run: %s", s3_base)
        else:
            ts = datetime.now().strftime("%Y%m%d-%H%M%S")
            s3_base = _s3_base_path(ts)
    else:
        ts = datetime.now().strftime("%Y%m%d-%H%M%S")
        s3_base = _s3_base_path(ts)

    # Persist state for future resume
    _save_state({"s3_base": s3_base, "timestamp": ts, "completed": []})

    logger.info("=" * 70)
    logger.info("Santander 4-Dimension Ablation Orchestrator")
    logger.info("  Timestamp:      %s", ts)
    logger.info("  S3 Base:        %s", s3_base)
    logger.info("  Phase:          %s", args.phase)
    logger.info("  GPU Instance:   %s", args.instance_type_gpu)
    logger.info("  CPU Instance:   %s", args.instance_type_cpu)
    logger.info("  Dry Run:        %s", args.dry_run)
    logger.info("  Model URI:      %s", args.model_uri or "(none)")
    logger.info("  Skip Fidelity:  %s", args.skip_fidelity_gate)
    logger.info("  Max Retries:    %d", args.max_retries)
    logger.info("=" * 70)

    run_all = args.phase == "all"
    phase = args.phase

    # Build source package ONCE and reuse across all job submissions
    global _GLOBAL_SOURCE_DIR, _GLOBAL_SOURCE_TAR
    if not args.dry_run:
        _GLOBAL_SOURCE_DIR = _prepare_and_extract_source()
        # Also create a tar.gz for S3 upload (processing jobs need it on S3)
        tmp_dir = tempfile.mkdtemp(prefix="santander_ablation_src_")
        _GLOBAL_SOURCE_TAR = _prepare_source_package(tmp_dir)
        source_uri = _upload_source_to_s3(_GLOBAL_SOURCE_TAR, s3_base)
    else:
        _GLOBAL_SOURCE_DIR = None
        _GLOBAL_SOURCE_TAR = None
        source_uri = f"{s3_base}/source/source.tar.gz"
        logger.info("[DRY RUN] Source package: %s", source_uri)

    # Update state with running job tracking info
    _save_state({
        "s3_base": s3_base,
        "timestamp": ts,
        "phase": args.phase,
        "running_jobs": [],
        "completed_phases": [],
    })

    # Default data URI
    data_uri = os.environ.get(
        "SANTANDER_DATA_URI",
        f"s3://{S3_BUCKET}/data/adapted/santander/",
    )

    # Collect results from each phase
    all_results: Dict[str, Any] = {}

    # Recover running jobs from a previous crashed run
    if existing_state and not args.dry_run:
        try:
            recovered = _recover_running_jobs(existing_state)
            if recovered:
                logger.info("Found %d running jobs from previous run", len(recovered))
        except Exception as e:
            logger.warning("Could not recover running jobs: %s", e)

    # Phase 0: Data Preparation
    if run_all or phase == "0":
        p0_result = run_phase0(s3_base, ts, args)
        all_results["phase0"] = p0_result
        if p0_result.get("data_uri"):
            data_uri = p0_result["data_uri"]
        _save_state({"s3_base": s3_base, "timestamp": ts, "completed_phases": ["0"],
                      "running_jobs": [], "phase_progress": "phase0_done"})

    # Phase 1: Feature Group Ablation (10 scenarios)
    p1_results: List[Dict[str, Any]] = []
    if run_all or phase == "1":
        p1_results = run_phase1(s3_base, ts, data_uri, source_uri, args)
        all_results["phase1"] = p1_results
        _save_state({"s3_base": s3_base, "timestamp": ts, "completed_phases": ["0", "1"],
                      "running_jobs": [], "phase_progress": "phase1_done"})

    # Phase 2: Expert Ablation (7 scenarios)
    p2_results: List[Dict[str, Any]] = []
    if run_all or phase == "2":
        p2_results = run_phase2(s3_base, ts, data_uri, source_uri, args)
        all_results["phase2"] = p2_results
        _save_state({"s3_base": s3_base, "timestamp": ts, "completed_phases": ["0", "1", "2"],
                      "running_jobs": [], "phase_progress": "phase2_done"})

    # Phase 3: Task x Structure Cross Ablation (16 scenarios)
    p3_results: List[Dict[str, Any]] = []
    if run_all or phase == "3":
        p3_results = run_phase3(s3_base, ts, data_uri, source_uri, args)
        all_results["phase3"] = p3_results
        _save_state({"s3_base": s3_base, "timestamp": ts, "completed_phases": ["0", "1", "2", "3"],
                      "running_jobs": [], "phase_progress": "phase3_done"})

    # Phase 4: Best Config Teacher + Distillation
    if run_all or phase == "4":
        best_config = _select_best_config(p1_results, p2_results, p3_results, s3_base)
        p4_result = run_phase4(s3_base, ts, data_uri, source_uri, args, best_config)
        all_results["phase4"] = p4_result
        _save_state({"s3_base": s3_base, "timestamp": ts, "completed_phases": ["0", "1", "2", "3", "4"],
                      "running_jobs": [], "phase_progress": "phase4_done"})

    # Phase 5: Analysis + HTML Report
    if run_all or phase == "5":
        p5_result = run_phase5(s3_base, ts, all_results, args)
        all_results["phase5"] = p5_result

    # Final summary
    logger.info("=" * 70)
    logger.info("Santander 4-Dimension Ablation Complete!")
    logger.info("  S3 Base:  %s", s3_base)
    logger.info("  Phases:   %s", list(all_results.keys()))
    total_jobs = sum(
        len(v) if isinstance(v, list) else 1
        for v in all_results.values()
    )
    logger.info("  Total Jobs Submitted: %d", total_jobs)

    # Accumulate total billable time across all training results
    total_billable_s = 0
    all_job_results = []
    for v in all_results.values():
        if isinstance(v, list):
            all_job_results.extend(v)
        elif isinstance(v, dict):
            all_job_results.append(v)
    for r in all_job_results:
        if isinstance(r, dict) and "billable_seconds" in r:
            total_billable_s += r["billable_seconds"]
    if total_billable_s > 0:
        # Estimate cost: ml.g4dn.xlarge spot ~ $0.1578/hr, on-demand ~ $0.526/hr
        est_cost = total_billable_s / 3600 * 0.526
        logger.info("  Total Billable Time: %ds (%.2f hrs), estimated cost: $%.2f",
                     total_billable_s, total_billable_s / 3600, est_cost)

    # Summary by dimension
    n_p1 = len([r for r in p1_results if r.get("status") in ("Completed", "DRY_RUN", "InProgress")])
    n_p2 = len([r for r in p2_results if r.get("status") in ("Completed", "DRY_RUN", "InProgress")])
    n_p3 = len([r for r in p3_results if r.get("status") in ("Completed", "DRY_RUN", "InProgress")])
    logger.info("  Dim 1 (Feature):       %d/%d", n_p1, len(FEATURE_SCENARIOS))
    logger.info("  Dim 2 (Expert):        %d/%d", n_p2, len(EXPERT_SCENARIOS))
    logger.info("  Dim 3 (Task x Struct): %d/%d", n_p3, len(TASK_TIERS) * len(STRUCTURE_VARIANTS))

    # Detailed failure summary
    failed = [r for r in all_job_results if isinstance(r, dict) and r.get("status") in ("Failed", "Stopped")]
    if failed:
        logger.error("=== FAILED JOBS (%d) ===", len(failed))
        for f in failed:
            logger.error("  %s: %s",
                         f.get("job_name", f.get("scenario", "unknown")),
                         f.get("failure_reason", "unknown")[:100])

    logger.info("=" * 70)


if __name__ == "__main__":
    main()
