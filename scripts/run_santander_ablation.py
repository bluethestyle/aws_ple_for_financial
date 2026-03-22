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
from typing import Any, Dict, List, Optional

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger("run_santander_ablation")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

REGION = "ap-northeast-2"
S3_BUCKET = "aiops-ple-financial"
ROLE_ARN = "arn:aws:iam::795833413857:role/AWSPLEPlatformSageMakerRole"

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = "configs/santander/pipeline.yaml"

# Instance types
GPU_INSTANCE = "ml.g4dn.xlarge"
CPU_INSTANCE = "ml.m5.xlarge"

# Framework versions (for PyTorch Estimator)
PYTORCH_VERSION = "2.1"
PY_VERSION = "py310"

# ---------------------------------------------------------------------------
# Dimension 1: Feature Group Ablation (10 scenarios)
# ---------------------------------------------------------------------------
FEATURE_SCENARIOS: List[Dict[str, Any]] = [
    {"name": "full", "remove": []},
    {"name": "no_demographics", "remove": ["demographics"]},
    {"name": "no_products", "remove": ["product_holdings"]},
    {"name": "no_txn_behavior", "remove": ["txn_behavior"]},
    {"name": "no_sequences", "remove": ["tda_local", "mamba_temporal", "hmm_states"]},
    {"name": "no_derived", "remove": ["derived_temporal"]},
    {"name": "no_tda", "remove": ["tda_global", "tda_local"]},
    {"name": "no_graph", "remove": ["graph_collaborative"]},
    {"name": "no_hierarchy", "remove": ["product_hierarchy"]},
    {
        "name": "base_only",
        "remove": [
            "tda_global", "tda_local", "hmm_states", "mamba_temporal",
            "graph_collaborative", "product_hierarchy", "gmm_clustering",
            "model_derived",
        ],
    },
]

# ---------------------------------------------------------------------------
# Dimension 2: Expert Ablation (7 scenarios)
# ---------------------------------------------------------------------------
ALL_SHARED_EXPERTS = [
    "deepfm", "temporal_ensemble", "hgcn",
    "perslay", "causal", "lightgcn", "optimal_transport",
]

EXPERT_SCENARIOS: List[Dict[str, Any]] = [
    {"name": "full_basket", "experts": list(ALL_SHARED_EXPERTS)},
    {"name": "no_deepfm", "experts": [e for e in ALL_SHARED_EXPERTS if e != "deepfm"]},
    {"name": "no_temporal", "experts": [e for e in ALL_SHARED_EXPERTS if e != "temporal_ensemble"]},
    {"name": "no_hgcn", "experts": [e for e in ALL_SHARED_EXPERTS if e != "hgcn"]},
    {"name": "no_perslay", "experts": [e for e in ALL_SHARED_EXPERTS if e != "perslay"]},
    {"name": "no_causal", "experts": [e for e in ALL_SHARED_EXPERTS if e != "causal"]},
    {"name": "mlp_only", "experts": []},
]

# ---------------------------------------------------------------------------
# Dimension 3: Task x Structure Cross Ablation (4 x 4 = 16 scenarios)
# ---------------------------------------------------------------------------
TASK_TIERS: Dict[str, List[str]] = {
    "tasks_4": [
        "has_nba", "churn_signal", "product_stability", "nba_primary",
    ],
    "tasks_8": [
        "has_nba", "churn_signal", "product_stability", "nba_primary",
        "tenure_stage", "spend_level", "cross_sell_count", "engagement_score",
    ],
    "tasks_15": [
        "has_nba", "churn_signal", "product_stability", "nba_primary",
        "tenure_stage", "spend_level", "cross_sell_count", "engagement_score",
        "will_acquire_deposits", "will_acquire_investments",
        "will_acquire_accounts", "will_acquire_lending",
        "will_acquire_payments", "segment_prediction", "income_tier",
    ],
    "tasks_18": [
        "has_nba", "churn_signal", "product_stability", "nba_primary",
        "tenure_stage", "spend_level", "cross_sell_count", "engagement_score",
        "will_acquire_deposits", "will_acquire_investments",
        "will_acquire_accounts", "will_acquire_lending",
        "will_acquire_payments", "segment_prediction", "income_tier",
        "next_mcc", "mcc_diversity_trend", "top_mcc_shift",
    ],
}

STRUCTURE_VARIANTS: Dict[str, Dict[str, bool]] = {
    "shared_bottom": {"use_ple": False, "use_adatt": False},
    "ple_only": {"use_ple": True, "use_adatt": False},
    "adatt_only": {"use_ple": False, "use_adatt": True},
    "full": {"use_ple": True, "use_adatt": True},
}


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
            logger.info("Job %s -> %s", job_name, status)
            if status == "Failed":
                reason = desc.get("FailureReason", "unknown")
                logger.error("  Failure reason: %s", reason)
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
                logger.error("  Failure reason: %s", reason)
            return status
        time.sleep(30)


def _get_model_artifact_uri(job_name: str) -> Optional[str]:
    """Retrieve the S3 model artifact URI from a completed Training Job."""
    import boto3
    sm = boto3.client("sagemaker", region_name=REGION)
    desc = sm.describe_training_job(TrainingJobName=job_name)
    return desc.get("ModelArtifacts", {}).get("S3ModelArtifacts")


def _submit_training_job(
    job_name: str,
    s3_output: str,
    instance_type: str,
    hyperparameters: Dict[str, str],
    source_uri: str,
    data_uri: str,
    wait: bool = True,
    dry_run: bool = False,
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

    session = sagemaker.Session()

    env = {
        "PYTHONIOENCODING": "utf-8",
        "NCCL_DEBUG": "WARN",
    }

    # Package source code
    tmp_dir = tempfile.mkdtemp()
    source_tar = _prepare_source_package(tmp_dir)

    # Extract tar.gz to a temp dir for SageMaker Estimator source_dir
    import shutil
    pkg_dir = Path(tempfile.mkdtemp()) / "source"
    pkg_dir.mkdir(parents=True, exist_ok=True)
    with tarfile.open(source_tar, "r:gz") as tar:
        tar.extractall(str(pkg_dir))

    estimator = PyTorch(
        entry_point="containers/training/train.py",
        source_dir=str(pkg_dir),
        role=ROLE_ARN,
        instance_type=instance_type,
        instance_count=1,
        framework_version=PYTORCH_VERSION,
        py_version=PY_VERSION,
        output_path=s3_output,
        hyperparameters=hyperparameters,
        environment=env,
        use_spot_instances=True,
        max_run=14400,       # 4 hours
        max_wait=18000,      # 5 hours (spot wait)
        tags=[
            {"Key": "Project", "Value": "santander-ablation"},
            {"Key": "Phase", "Value": hyperparameters.get("ablation_phase", "unknown")},
        ],
        sagemaker_session=session,
    )

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

    # Package source code as tar.gz
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
                source=f"s3://{S3_BUCKET}/data/raw/santander/",
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

    results: List[Dict[str, Any]] = []

    for scenario in FEATURE_SCENARIOS:
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
            "epochs": "30",
            "batch_size": "128",
            "learning_rate": "0.001",
            "seed": "42",
            "_s3_output": f"{s3_base}/phase1/{name}",
        }

        logger.info("--- Scenario: %s (remove=%s) ---", name, scenario["remove"])

        result = _submit_training_job(
            job_name=job_name,
            s3_output=s3_output,
            instance_type=args.instance_type_gpu,
            hyperparameters=hyperparameters,
            source_uri=source_uri,
            data_uri=data_uri,
            wait=not args.no_wait,
            dry_run=args.dry_run,
        )
        result["scenario"] = name
        result["phase"] = 1
        results.append(result)

    return results


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

    results: List[Dict[str, Any]] = []

    for scenario in EXPERT_SCENARIOS:
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
            "epochs": "30",
            "batch_size": "128",
            "learning_rate": "0.001",
            "seed": "42",
            "_s3_output": f"{s3_base}/phase2/{name}",
        }

        logger.info("--- Scenario: %s (experts=%s) ---", name, scenario["experts"])

        result = _submit_training_job(
            job_name=job_name,
            s3_output=s3_output,
            instance_type=args.instance_type_gpu,
            hyperparameters=hyperparameters,
            source_uri=source_uri,
            data_uri=data_uri,
            wait=not args.no_wait,
            dry_run=args.dry_run,
        )
        result["scenario"] = name
        result["phase"] = 2
        results.append(result)

    return results


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

    results: List[Dict[str, Any]] = []

    for tier_name, task_list in TASK_TIERS.items():
        for struct_name, struct_flags in STRUCTURE_VARIANTS.items():
            scenario_name = f"{tier_name}-{struct_name}"
            safe_name = scenario_name.replace("_", "-")
            job_name = f"sant-abl-p3-{safe_name}-{ts}"
            s3_output = f"{s3_base}/phase3/{scenario_name}/"

            hyperparameters = {
                "config": CONFIG_PATH,
                "ablation_phase": "3",
                "ablation_type": "task_structure",
                "ablation_scenario": scenario_name,
                "active_tasks": json.dumps(task_list),
                "use_ple": json.dumps(struct_flags["use_ple"]),
                "use_adatt": json.dumps(struct_flags["use_adatt"]),
                "epochs": "30",
                "batch_size": "128",
                "learning_rate": "0.001",
                "seed": "42",
                "_s3_output": f"{s3_base}/phase3/{scenario_name}",
            }

            logger.info(
                "--- Scenario: %s (%d tasks, PLE=%s, adaTT=%s) ---",
                scenario_name,
                len(task_list),
                struct_flags["use_ple"],
                struct_flags["use_adatt"],
            )

            result = _submit_training_job(
                job_name=job_name,
                s3_output=s3_output,
                instance_type=args.instance_type_gpu,
                hyperparameters=hyperparameters,
                source_uri=source_uri,
                data_uri=data_uri,
                wait=not args.no_wait,
                dry_run=args.dry_run,
            )
            result["scenario"] = scenario_name
            result["tier"] = tier_name
            result["structure"] = struct_name
            result["phase"] = 3
            results.append(result)

    return results


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
            "epochs": "30",
            "batch_size": "128",
            "learning_rate": "0.001",
            "seed": "42",
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

    distill_arguments = [
        "--teacher-checkpoint", "/opt/ml/processing/input/teacher/model.pth",
        "--data-path", "/opt/ml/processing/input/data/",
        "--output-dir", "/opt/ml/processing/output",
        "--config", CONFIG_PATH,
        "--soft-label-path", "/opt/ml/processing/output/soft_labels.parquet",
        "--temperature", "5.0",
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
        "dataset": "santander",
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

    # Phase 2 metrics
    for scenario in EXPERT_SCENARIOS:
        s3_path = f"{s3_base}/phase2/{scenario['name']}/output/eval_metrics.json"
        metrics = _download_json_from_s3(s3_path) if not args.dry_run else None
        if metrics:
            collected_metrics[f"p2_{scenario['name']}"] = metrics

    # Phase 3 metrics (task x structure cross)
    for tier_name in TASK_TIERS:
        for struct_name in STRUCTURE_VARIANTS:
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
    args = parse_args()
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    s3_base = _s3_base_path(ts)

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

    # Prepare source package
    if not args.dry_run:
        tmp_dir = tempfile.mkdtemp(prefix="santander_ablation_src_")
        source_tar = _prepare_source_package(tmp_dir)
        source_uri = _upload_source_to_s3(source_tar, s3_base)
    else:
        source_uri = f"{s3_base}/source/source.tar.gz"
        logger.info("[DRY RUN] Source package: %s", source_uri)

    # Default data URI
    data_uri = os.environ.get(
        "SANTANDER_DATA_URI",
        f"s3://{S3_BUCKET}/data/adapted/santander/",
    )

    # Collect results from each phase
    all_results: Dict[str, Any] = {}

    # Phase 0: Data Preparation
    if run_all or phase == "0":
        p0_result = run_phase0(s3_base, ts, args)
        all_results["phase0"] = p0_result
        if p0_result.get("data_uri"):
            data_uri = p0_result["data_uri"]

    # Phase 1: Feature Group Ablation (10 scenarios)
    p1_results: List[Dict[str, Any]] = []
    if run_all or phase == "1":
        p1_results = run_phase1(s3_base, ts, data_uri, source_uri, args)
        all_results["phase1"] = p1_results

    # Phase 2: Expert Ablation (7 scenarios)
    p2_results: List[Dict[str, Any]] = []
    if run_all or phase == "2":
        p2_results = run_phase2(s3_base, ts, data_uri, source_uri, args)
        all_results["phase2"] = p2_results

    # Phase 3: Task x Structure Cross Ablation (16 scenarios)
    p3_results: List[Dict[str, Any]] = []
    if run_all or phase == "3":
        p3_results = run_phase3(s3_base, ts, data_uri, source_uri, args)
        all_results["phase3"] = p3_results

    # Phase 4: Best Config Teacher + Distillation
    if run_all or phase == "4":
        best_config = _select_best_config(p1_results, p2_results, p3_results, s3_base)
        p4_result = run_phase4(s3_base, ts, data_uri, source_uri, args, best_config)
        all_results["phase4"] = p4_result

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

    # Summary by dimension
    n_p1 = len([r for r in p1_results if r.get("status") in ("Completed", "DRY_RUN", "InProgress")])
    n_p2 = len([r for r in p2_results if r.get("status") in ("Completed", "DRY_RUN", "InProgress")])
    n_p3 = len([r for r in p3_results if r.get("status") in ("Completed", "DRY_RUN", "InProgress")])
    logger.info("  Dim 1 (Feature):       %d/%d", n_p1, len(FEATURE_SCENARIOS))
    logger.info("  Dim 2 (Expert):        %d/%d", n_p2, len(EXPERT_SCENARIOS))
    logger.info("  Dim 3 (Task x Struct): %d/%d", n_p3, len(TASK_TIERS) * len(STRUCTURE_VARIANTS))
    logger.info("=" * 70)


if __name__ == "__main__":
    main()
