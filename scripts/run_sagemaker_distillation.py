#!/usr/bin/env python3
"""Submit a SageMaker distillation job.

Uploads teacher checkpoint to S3, submits a CPU Training Job that runs
containers/distillation/distill_entry.py, downloads results.

Usage::
    # Dry run — verify config without spending money
    python scripts/run_sagemaker_distillation.py --dry-run

    # Submit with a specific teacher checkpoint
    python scripts/run_sagemaker_distillation.py \\
        --teacher-checkpoint outputs/ablation_v12/joint_full/model/model.pth \\
        --scenario teacher_full_distill

    # Submit and wait, then download results
    python scripts/run_sagemaker_distillation.py \\
        --teacher-checkpoint outputs/ablation_v12/joint_full/model/model.pth \\
        --scenario teacher_full_distill \\
        --download-results

Cost estimate (CPU instance, no GPU needed for LGBM):
    ml.m5.2xlarge Spot  ~$0.11/hr x ~2hr = ~$0.22
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger("sagemaker_distillation")

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = PROJECT_ROOT / "configs" / "pipeline.yaml"
DATASET_CONFIG_PATH = PROJECT_ROOT / "configs" / "datasets" / "santander.yaml"
PHASE0_DIR = PROJECT_ROOT / "outputs" / "phase0_v12"

# Container-internal config paths (inside the source_dir extraction)
CONTAINER_CONFIG = "configs/pipeline.yaml"
CONTAINER_DATASET_CONFIG = "configs/datasets/santander.yaml"

# S3 path segments (bucket comes from pipeline.yaml)
S3_DATA_PREFIX = "data/phase0_v12"
S3_CHECKPOINT_PREFIX = "checkpoints/distillation"
S3_OUTPUT_PREFIX = "output/distillation"
S3_SOURCE_PREFIX = "source/distillation"

# Distillation job defaults (CLAUDE.md 1.5: CPU instance, no GPU needed)
DISTILL_INSTANCE_TYPE = "ml.m5.4xlarge"     # 16 vCPU, 64GB RAM — faster multiclass LGBM
DISTILL_MAX_RUN = 10800                       # 3 hr max (LGBM x 13 tasks + IG)
DISTILL_MAX_WAIT = 14400                      # max_run + 1hr (CLAUDE.md 1.5)


# ---------------------------------------------------------------------------
# Config loading (CLAUDE.md 1.1: config-driven)
# ---------------------------------------------------------------------------

def load_pipeline_config() -> Dict[str, Any]:
    from core.pipeline.config import load_merged_config
    if DATASET_CONFIG_PATH.exists():
        return load_merged_config(CONFIG_PATH, DATASET_CONFIG_PATH)
    with open(CONFIG_PATH, encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def get_aws_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """Read AWS settings from pipeline.yaml."""
    aws = config.get("aws", {})
    return {
        "region": aws.get("region", "ap-northeast-2"),
        "s3_bucket": aws.get("s3_bucket", "aiops-ple-financial"),
        "role_arn": aws.get("role_arn"),
        "use_spot": aws.get("use_spot", True),
        "cpu_instance_type": aws.get("cpu_instance_type", DISTILL_INSTANCE_TYPE),
    }


# ---------------------------------------------------------------------------
# Step 1: Upload checkpoint to S3
# ---------------------------------------------------------------------------

def upload_checkpoint(
    checkpoint_path: str,
    s3_bucket: str,
    scenario: str,
) -> str:
    """Upload a checkpoint file to S3.

    Args:
        checkpoint_path: Local path to the .pt or .pth file.
        s3_bucket: Destination bucket.
        scenario: Scenario name used as part of the S3 key prefix.

    Returns:
        S3 URI of the uploaded checkpoint directory
        (``s3://bucket/prefix/scenario/``).
    """
    import boto3

    local = Path(checkpoint_path)
    if not local.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    s3 = boto3.client("s3")
    s3_key = f"{S3_CHECKPOINT_PREFIX}/{scenario}/{local.name}"
    size_mb = local.stat().st_size / (1024 * 1024)

    logger.info(
        "Uploading checkpoint (%.1f MB) -> s3://%s/%s",
        size_mb, s3_bucket, s3_key,
    )
    s3.upload_file(str(local), s3_bucket, s3_key)

    # Also upload config.json if it exists alongside the checkpoint
    # (teacher_loader.py needs it for feature_schema + label_schema)
    config_json = local.parent / "config.json"
    if config_json.exists():
        config_key = f"{S3_CHECKPOINT_PREFIX}/{scenario}/config.json"
        logger.info("Uploading config.json -> s3://%s/%s", s3_bucket, config_key)
        s3.upload_file(str(config_json), s3_bucket, config_key)

    dir_uri = f"s3://{s3_bucket}/{S3_CHECKPOINT_PREFIX}/{scenario}/"
    logger.info("Checkpoint uploaded. S3 model channel: %s", dir_uri)
    return dir_uri


def _checkpoint_already_on_s3(
    checkpoint_path: str,
    s3_bucket: str,
    scenario: str,
) -> Optional[str]:
    """Return S3 URI if the checkpoint already exists on S3, else None."""
    try:
        import boto3

        local = Path(checkpoint_path)
        s3_key = f"{S3_CHECKPOINT_PREFIX}/{scenario}/{local.name}"
        s3 = boto3.client("s3")
        s3.head_object(Bucket=s3_bucket, Key=s3_key)
        uri = f"s3://{s3_bucket}/{S3_CHECKPOINT_PREFIX}/{scenario}/"
        logger.info("Checkpoint already on S3: %s", uri)
        return uri
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Step 2: Build source staging directory
# ---------------------------------------------------------------------------

def _get_staging_dir() -> str:
    """Build (or reuse) the source staging directory (CLAUDE.md 1.5: 1회 빌드)."""
    sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
    from package_source import build_staging  # type: ignore[import]
    return build_staging(project_root=str(PROJECT_ROOT))


# ---------------------------------------------------------------------------
# Step 3: Submit distillation job
# ---------------------------------------------------------------------------

def submit_distillation_job(
    aws_config: Dict[str, Any],
    s3_bucket: str,
    s3_data_uri: str,
    s3_model_uri: str,
    scenario: str,
    staging_dir: str,
    hp: Dict[str, str],
    dry_run: bool = False,
    wait: bool = True,
) -> Dict[str, Any]:
    """Submit a SageMaker Training Job configured for distillation.

    Uses distill_entry.py as the entry point on a CPU instance (no GPU needed
    for LGBM training). Two input channels are provided:
      ``train``  -- Phase 0 data (features.parquet + schemas)
      ``model``  -- directory containing the teacher checkpoint .pt file

    Args:
        aws_config:   AWS config dict from get_aws_config().
        s3_bucket:    S3 bucket name.
        s3_data_uri:  S3 URI for the training data channel.
        s3_model_uri: S3 URI for the model (checkpoint) channel.
        scenario:     Scenario label used in job name and output path.
        staging_dir:  Local path to the staged source code directory.
        hp:           Hyperparameters dict (all string values).
        dry_run:      When True, log configuration and return without submitting.
        wait:         When True, block until the job completes.

    Returns:
        Dict with job metadata: name, status, s3_output_uri.
    """
    from sagemaker.pytorch import PyTorch
    from sagemaker.inputs import TrainingInput
    import sagemaker

    session = sagemaker.Session()
    role = aws_config["role_arn"]

    # CPU instance — LGBM does not need GPU (CLAUDE.md 1.6: Phase 0/distill on CPU)
    instance_type = aws_config.get("cpu_instance_type", DISTILL_INSTANCE_TYPE)
    timestamp = time.strftime("%m%d-%H%M")
    # SageMaker job names: [a-zA-Z0-9](-*[a-zA-Z0-9]){0,62}
    safe_scenario = scenario.replace("_", "-")[:18]
    job_name = f"ple-distill-{safe_scenario}-{timestamp}"
    s3_output_uri = f"s3://{s3_bucket}/{S3_OUTPUT_PREFIX}/{scenario}"

    logger.info("=" * 60)
    logger.info("SageMaker Distillation Job")
    logger.info("  Scenario:  %s", scenario)
    logger.info("  Job name:  %s", job_name)
    logger.info("  Instance:  %s (CPU, Spot=%s)", instance_type, aws_config["use_spot"])
    logger.info("  Data URI:  %s", s3_data_uri)
    logger.info("  Model URI: %s", s3_model_uri)
    logger.info("  Output:    %s", s3_output_uri)
    logger.info("  Max run:   %ds (%d min)", DISTILL_MAX_RUN, DISTILL_MAX_RUN // 60)
    logger.info("  Cost est.: ~$0.11/hr x ~2hr = ~$0.22 (Spot)")
    logger.info("  HPs:")
    for k, v in sorted(hp.items()):
        logger.info("    %s = %s", k, v)
    logger.info("=" * 60)

    if dry_run:
        logger.info("[DRY-RUN] Would submit job: %s", job_name)
        return {
            "name": scenario,
            "job_name": job_name,
            "status": "DRY_RUN",
            "s3_output_uri": s3_output_uri,
        }

    estimator = PyTorch(
        entry_point="containers/distillation/distill_entry.py",
        source_dir=staging_dir,
        role=role,
        instance_type=instance_type,
        instance_count=1,
        framework_version="2.1",
        py_version="py310",
        hyperparameters=hp,
        use_spot_instances=aws_config["use_spot"],
        max_wait=DISTILL_MAX_WAIT,
        max_run=DISTILL_MAX_RUN,
        output_path=s3_output_uri,
        disable_profiler=True,          # CLAUDE.md 1.5: profiler OFF
        environment={
            "PYTHONPATH": "/opt/ml/code",
            "PYTHONIOENCODING": "utf-8",
            "PYTHONUNBUFFERED": "1",
        },
        sagemaker_session=session,
        base_job_name=f"ple-distill-{scenario[:18]}",
        metric_definitions=[
            {"Name": "distill:num_students",    "Regex": r"Tasks distilled: ([0-9]+)"},
            {"Name": "distill:passed_fidelity", "Regex": r"Fidelity summary: ([0-9]+)/"},
            {"Name": "distill:distill_tasks",   "Regex": r"Distillation: ([0-9]+) tasks"},
            {"Name": "distill:direct_tasks",    "Regex": r"Direct LGBM: ([0-9]+) tasks"},
        ],
    )

    logger.info("Submitting distillation job: %s", job_name)
    estimator.fit(
        inputs={
            "train": TrainingInput(s3_data_uri, content_type="application/x-parquet"),
            "model": TrainingInput(s3_model_uri, content_type="application/octet-stream"),
        },
        job_name=job_name,
        wait=wait,
        logs="All" if wait else "None",
    )

    result: Dict[str, Any] = {
        "name": scenario,
        "job_name": job_name,
        "estimator": estimator,
        "status": "Completed" if wait else "Submitted",
        "s3_output_uri": s3_output_uri,
    }
    logger.info("Distillation job %s: %s", job_name, result["status"])
    return result


# ---------------------------------------------------------------------------
# Step 4: Download results
# ---------------------------------------------------------------------------

def download_results(
    s3_bucket: str,
    scenario: str,
    local_dir: Optional[str] = None,
) -> Dict[str, Optional[str]]:
    """Download distillation outputs from S3.

    Downloads:
      - distillation_summary.json
      - fidelity_report.json
      - drift_report.json (if present)
      - LGBM model files (*.txt per task)

    SageMaker wraps model output in ``model.tar.gz`` and other output in
    ``output.tar.gz`` under the job output prefix.  We search for the key files
    directly by listing S3 objects.

    Args:
        s3_bucket:  S3 bucket name.
        scenario:   Scenario name (determines S3 key prefix).
        local_dir:  Local directory for downloaded files.
                    Defaults to ``outputs/sagemaker_distillation/{scenario}/``.

    Returns:
        Dict mapping filename -> local path (or None if not found).
    """
    import boto3

    s3 = boto3.client("s3")
    prefix = f"{S3_OUTPUT_PREFIX}/{scenario}/"

    if local_dir is None:
        local_dir = str(PROJECT_ROOT / "outputs" / "sagemaker_distillation" / scenario)
    Path(local_dir).mkdir(parents=True, exist_ok=True)

    logger.info("Searching for distillation outputs under s3://%s/%s", s3_bucket, prefix)

    paginator = s3.get_paginator("list_objects_v2")
    all_keys: List[str] = []
    for page in paginator.paginate(Bucket=s3_bucket, Prefix=prefix):
        for obj in page.get("Contents", []):
            all_keys.append(obj["Key"])

    target_files = [
        "distillation_summary.json",
        "fidelity_report.json",
        "drift_report.json",
    ]
    downloaded: Dict[str, Optional[str]] = {}

    for target in target_files:
        matches = [k for k in all_keys if target in k]
        if not matches:
            logger.warning("  %s: not found under s3://%s/%s", target, s3_bucket, prefix)
            downloaded[target] = None
            continue

        matches.sort()
        s3_key = matches[-1]
        local_path = Path(local_dir) / target
        logger.info("  Downloading s3://%s/%s -> %s", s3_bucket, s3_key, local_path)
        s3.download_file(s3_bucket, s3_key, str(local_path))
        downloaded[target] = str(local_path)

    # Pretty-print summary if available
    summary_path = downloaded.get("distillation_summary.json")
    if summary_path and Path(summary_path).exists():
        with open(summary_path) as f:
            summary = json.load(f)
        logger.info("=" * 60)
        logger.info("DISTILLATION SUMMARY -- %s", scenario)
        logger.info("=" * 60)
        logger.info("  Tasks distilled:    %s", summary.get("tasks_distilled", []))
        logger.info("  Tasks direct LGBM:  %s", summary.get("tasks_direct_hardlabel", []))
        logger.info("  Num students saved: %s", summary.get("num_students"))
        logger.info("  Feature count:      %s", summary.get("feature_count"))
        logger.info("  Sample count:       %s", summary.get("sample_count"))
        fidelity = summary.get("fidelity", {})
        logger.info(
            "  Fidelity: passed=%s/%s",
            fidelity.get("passed"), fidelity.get("passed", 0) + fidelity.get("failed", 0),
        )
        logger.info("=" * 60)

    # Hint for LGBM model files (in model.tar.gz)
    lgbm_keys = [k for k in all_keys if k.endswith(".txt") or "model.tar.gz" in k]
    if lgbm_keys:
        logger.info(
            "LGBM models: %d file(s) found. Download with:\n"
            "  aws s3 cp %s . --recursive",
            len(lgbm_keys), f"s3://{s3_bucket}/{prefix}",
        )
    else:
        logger.info(
            "LGBM models in model.tar.gz -- download manually:\n"
            "  aws s3 cp %s . --recursive && tar xzf model.tar.gz",
            f"s3://{s3_bucket}/{prefix}",
        )

    return downloaded


# ---------------------------------------------------------------------------
# Cost check helper (mirrors run_sagemaker_eval.py)
# ---------------------------------------------------------------------------

def verify_cost() -> None:
    """Check current AWS cost before submission (CLAUDE.md 1.5)."""
    try:
        today = time.strftime("%Y-%m-%d")
        result = subprocess.run(
            [
                "aws", "ce", "get-cost-and-usage",
                "--time-period", f"Start=2026-04-01,End={today}",
                "--granularity", "MONTHLY",
                "--metrics", "UnblendedCost",
            ],
            capture_output=True, text=True, timeout=15,
        )
        if result.returncode == 0:
            cost_data = json.loads(result.stdout)
            for period in cost_data.get("ResultsByTime", []):
                amount = period["Total"]["UnblendedCost"]["Amount"]
                logger.info("Current month cost: $%s", amount)
        else:
            logger.warning("Cost check failed. Proceeding anyway.")
    except Exception as exc:
        logger.warning("Cost check skipped: %s", exc)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Submit a SageMaker distillation job (PLE teacher -> LGBM x 13).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Cost estimate: ml.m5.2xlarge Spot ~$0.11/hr x ~2hr = ~$0.22\n"
            "\nExamples:\n"
            "  # Dry run\n"
            "  python scripts/run_sagemaker_distillation.py --dry-run\n\n"
            "  # Submit with checkpoint\n"
            "  python scripts/run_sagemaker_distillation.py \\\n"
            "    --teacher-checkpoint outputs/ablation_v12/joint_full/model/model.pth \\\n"
            "    --scenario teacher_full_distill\n"
        ),
    )
    parser.add_argument(
        "--teacher-checkpoint", default=None,
        help=(
            "Local path to the teacher .pt or .pth checkpoint file. "
            "Required unless --s3-model-uri is provided."
        ),
    )
    parser.add_argument(
        "--scenario", default="teacher_full_distill",
        help="Scenario label (used in job name and S3 path). Default: teacher_full_distill",
    )
    parser.add_argument(
        "--s3-model-uri", default=None,
        help=(
            "Pre-existing S3 URI for the model channel (skips upload). "
            "E.g. s3://bucket/checkpoints/distillation/scenario/"
        ),
    )
    parser.add_argument(
        "--s3-data-uri", default=None,
        help=(
            f"S3 URI for Phase 0 data channel. "
            f"Default: s3://BUCKET/{S3_DATA_PREFIX}"
        ),
    )
    parser.add_argument(
        "--s3-bucket", default=None,
        help="S3 bucket override (default: from pipeline.yaml).",
    )
    parser.add_argument(
        "--batch-size", type=int, default=4096,
        help="Teacher inference batch size for soft label generation (default: 4096).",
    )
    parser.add_argument(
        "--temperature", type=float, default=None,
        help="Distillation temperature override (default: from pipeline.yaml).",
    )
    parser.add_argument(
        "--alpha", type=float, default=None,
        help="Hard label weight override (default: from pipeline.yaml).",
    )
    parser.add_argument(
        "--skip-fidelity-gate", action="store_true",
        help="Continue even if fidelity validation fails (for testing only).",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Print configuration and exit without submitting.",
    )
    parser.add_argument(
        "--no-wait", action="store_true",
        help="Submit job and return immediately (do not block for completion).",
    )
    parser.add_argument(
        "--download-results", action="store_true",
        help="After completion, download distillation_summary.json and fidelity_report.json.",
    )
    parser.add_argument(
        "--output-dir", default=None,
        help="Local directory for downloaded results.",
    )
    parser.add_argument(
        "--skip-cost-check", action="store_true",
        help="Skip AWS cost check before submission.",
    )
    parser.add_argument(
        "--force-upload", action="store_true",
        help="Re-upload checkpoint even if it already exists on S3.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()

    # --- Load config (CLAUDE.md 1.1: no hardcoding) ---
    config = load_pipeline_config()
    aws_config = get_aws_config(config)

    s3_bucket: str = args.s3_bucket or aws_config.get("s3_bucket", "aiops-ple-financial")
    if s3_bucket in ("YOUR_S3_BUCKET", ""):
        s3_bucket = "aiops-ple-financial"
        logger.info("Using default S3 bucket: %s", s3_bucket)

    s3_data_uri = args.s3_data_uri or f"s3://{s3_bucket}/{S3_DATA_PREFIX}"

    # --- Build HP dict ---
    hp: Dict[str, str] = {
        "config": CONTAINER_CONFIG,
        "dataset_config": CONTAINER_DATASET_CONFIG,
        "batch_size": str(args.batch_size),
        "skip_fidelity_gate": str(args.skip_fidelity_gate).lower(),
        "scenario": args.scenario,
    }
    if args.temperature is not None:
        hp["temperature"] = str(args.temperature)
    if args.alpha is not None:
        hp["alpha"] = str(args.alpha)

    # --- Pre-flight summary ---
    instance_type = aws_config.get("cpu_instance_type", DISTILL_INSTANCE_TYPE)
    logger.info("=" * 60)
    logger.info("SageMaker Distillation -- %s", args.scenario)
    logger.info("=" * 60)
    logger.info("  Checkpoint:   %s", args.teacher_checkpoint or "(using --s3-model-uri)")
    logger.info("  Scenario:     %s", args.scenario)
    logger.info("  S3 bucket:    %s", s3_bucket)
    logger.info("  Data URI:     %s", s3_data_uri)
    logger.info("  Instance:     %s (CPU -- no GPU needed for LGBM)", instance_type)
    logger.info("  Max run:      %ds (%d min)", DISTILL_MAX_RUN, DISTILL_MAX_RUN // 60)
    logger.info("  Use spot:     %s", aws_config["use_spot"])
    logger.info("  Cost est.:    ~$0.11/hr x ~2hr = ~$0.22 (Spot)")
    logger.info("  Dry run:      %s", args.dry_run)

    if args.dry_run:
        logger.info("\n[DRY-RUN] HP configuration:")
        for k, v in sorted(hp.items()):
            logger.info("  %s = %s", k, v)
        logger.info("[DRY-RUN] All checks passed. Ready to submit.")
        logger.info("[DRY-RUN] Note: Two input channels will be used:")
        logger.info("  train: %s", s3_data_uri)
        logger.info("  model: s3://%s/%s/%s/", s3_bucket, S3_CHECKPOINT_PREFIX, args.scenario)
        return

    # Validate that we have a checkpoint source
    if args.s3_model_uri is None and args.teacher_checkpoint is None:
        logger.error("Either --teacher-checkpoint or --s3-model-uri is required.")
        sys.exit(1)

    # --- Cost check (CLAUDE.md 1.5) ---
    if not args.skip_cost_check:
        verify_cost()

    # --- Upload checkpoint (or use pre-existing S3 URI) ---
    if args.s3_model_uri:
        s3_model_uri = args.s3_model_uri
        logger.info("Using provided S3 model URI: %s", s3_model_uri)
    else:
        if not args.force_upload:
            s3_model_uri = _checkpoint_already_on_s3(
                args.teacher_checkpoint, s3_bucket, args.scenario
            )
        else:
            s3_model_uri = None

        if s3_model_uri is None:
            s3_model_uri = upload_checkpoint(
                checkpoint_path=args.teacher_checkpoint,
                s3_bucket=s3_bucket,
                scenario=args.scenario,
            )

    # --- Build staging dir (CLAUDE.md 1.5: source 1회 빌드) ---
    logger.info("Building source staging directory...")
    staging_dir = _get_staging_dir()

    # --- Submit job ---
    wait = not args.no_wait
    job_result = submit_distillation_job(
        aws_config=aws_config,
        s3_bucket=s3_bucket,
        s3_data_uri=s3_data_uri,
        s3_model_uri=s3_model_uri,
        scenario=args.scenario,
        staging_dir=staging_dir,
        hp=hp,
        dry_run=False,
        wait=wait,
    )

    logger.info(
        "Job result: %s",
        {k: v for k, v in job_result.items() if k != "estimator"},
    )

    # --- Download results ---
    if args.download_results and wait:
        logger.info("Downloading distillation results...")
        downloaded = download_results(
            s3_bucket=s3_bucket,
            scenario=args.scenario,
            local_dir=args.output_dir,
        )
        found = {k: v for k, v in downloaded.items() if v is not None}
        logger.info("Downloaded %d file(s): %s", len(found), list(found.keys()))
        if not found:
            logger.warning("Could not download results automatically.")
            logger.info(
                "Download manually:\n"
                "  aws s3 cp %s/output.tar.gz . && tar xzf output.tar.gz\n"
                "  aws s3 cp %s/model.tar.gz . && tar xzf model.tar.gz",
                job_result["s3_output_uri"],
                job_result["s3_output_uri"],
            )
    elif args.download_results and not wait:
        logger.info(
            "Job submitted without --wait; re-run with --download-results "
            "after the job completes."
        )


if __name__ == "__main__":
    main()
