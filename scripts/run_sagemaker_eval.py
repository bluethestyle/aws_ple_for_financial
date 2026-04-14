#!/usr/bin/env python3
"""Submit a SageMaker evaluation job for a trained checkpoint.

Uploads the checkpoint to S3, submits a SageMaker Processing/Training job
using containers/evaluation/eval_entry.py as the entry point, and
optionally downloads the resulting eval_metrics.json.

Usage::
    # Dry-run: verify config without spending money
    python scripts/run_sagemaker_eval.py \\
        --checkpoint outputs/sagemaker_teacher_30ep/teacher_full/best.pt \\
        --scenario teacher_full \\
        --dry-run

    # Submit and wait for results
    python scripts/run_sagemaker_eval.py \\
        --checkpoint outputs/sagemaker_teacher_30ep/teacher_full/best.pt \\
        --scenario teacher_full \\
        --download-results
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
from typing import Any, Dict, Optional

import yaml

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger("sagemaker_eval")

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = PROJECT_ROOT / "configs" / "santander" / "pipeline.yaml"
PHASE0_DIR = PROJECT_ROOT / "outputs" / "phase0_v12"

# Container-internal config path (inside the source_dir extraction)
CONTAINER_CONFIG = "configs/santander/pipeline.yaml"

# S3 path segments (bucket comes from pipeline.yaml)
S3_DATA_PREFIX = "data/phase0_v12"
S3_CHECKPOINT_PREFIX = "checkpoints/eval"
S3_OUTPUT_PREFIX = "output/eval"
S3_SOURCE_PREFIX = "source/eval"

# Eval job defaults
EVAL_INSTANCE_TYPE = "ml.g4dn.xlarge"   # GPU for fast forward pass
EVAL_MAX_RUN = 1800                       # 30 min — forward pass only
EVAL_MAX_WAIT = 3600                      # 1 hr max wait (CLAUDE.md: max_run + 1hr)


# ---------------------------------------------------------------------------
# Config loading (CLAUDE.md 1.1: config-driven)
# ---------------------------------------------------------------------------

def load_pipeline_config() -> Dict[str, Any]:
    with open(CONFIG_PATH, encoding="utf-8") as f:
        return yaml.safe_load(f)


def get_aws_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """Read AWS settings from pipeline.yaml."""
    aws = config.get("aws", {})
    return {
        "region": aws.get("region", "ap-northeast-2"),
        "s3_bucket": aws.get("s3_bucket", "aiops-ple-financial"),
        "instance_type": aws.get("instance_type", EVAL_INSTANCE_TYPE),
        "role_arn": aws.get("role_arn"),
        "use_spot": aws.get("use_spot", True),
    }


# ---------------------------------------------------------------------------
# Step 1: Upload checkpoint to S3
# ---------------------------------------------------------------------------

def upload_checkpoint(
    checkpoint_path: str,
    s3_bucket: str,
    scenario: str,
) -> str:
    """Upload a checkpoint .pt file to S3.

    Args:
        checkpoint_path: Local path to the .pt file.
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

    # Return the directory URI so SageMaker can use it as a channel
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
# Step 2: Build source staging dir
# ---------------------------------------------------------------------------

def _get_staging_dir() -> str:
    """Build (or reuse) the source staging directory."""
    # Import from package_source sibling script
    sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
    from package_source import build_staging  # type: ignore[import]
    return build_staging(project_root=str(PROJECT_ROOT))


# ---------------------------------------------------------------------------
# Step 3: Submit evaluation job
# ---------------------------------------------------------------------------

def submit_eval_job(
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
    """Submit a SageMaker Training Job configured for eval-only.

    Uses eval_entry.py as the entry point.  Two input channels are provided:
      ``train``  — Phase 0 data (parquet + schemas)
      ``model``  — directory containing the checkpoint .pt file

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

    # Use GPU for fast forward pass but shorter max_run
    instance_type = EVAL_INSTANCE_TYPE
    timestamp = time.strftime("%m%d-%H%M")
    job_name = f"ple-eval-{scenario[:20]}-{timestamp}"
    s3_output_uri = f"s3://{s3_bucket}/{S3_OUTPUT_PREFIX}/{scenario}"

    logger.info("=" * 60)
    logger.info("SageMaker Eval Job")
    logger.info("  Scenario:  %s", scenario)
    logger.info("  Job name:  %s", job_name)
    logger.info("  Instance:  %s (Spot=%s)", instance_type, aws_config["use_spot"])
    logger.info("  Data URI:  %s", s3_data_uri)
    logger.info("  Model URI: %s", s3_model_uri)
    logger.info("  Output:    %s", s3_output_uri)
    logger.info("  HPs:")
    for k, v in sorted(hp.items()):
        logger.info("    %s = %s", k, v)

    if dry_run:
        logger.info("[DRY-RUN] Would submit job: %s", job_name)
        return {"name": scenario, "job_name": job_name, "status": "DRY_RUN",
                "s3_output_uri": s3_output_uri}

    estimator = PyTorch(
        entry_point="containers/evaluation/eval_entry.py",
        source_dir=staging_dir,
        role=role,
        instance_type=instance_type,
        instance_count=1,
        framework_version="2.1",
        py_version="py310",
        hyperparameters=hp,
        use_spot_instances=aws_config["use_spot"],
        max_wait=EVAL_MAX_WAIT,
        max_run=EVAL_MAX_RUN,
        output_path=s3_output_uri,
        disable_profiler=True,      # CLAUDE.md 1.5: profiler OFF
        environment={
            "PYTHONPATH": "/opt/ml/code",
            "PYTHONIOENCODING": "utf-8",
            "PYTHONUNBUFFERED": "1",
        },
        sagemaker_session=session,
        base_job_name=f"ple-eval-{scenario[:20]}",
        metric_definitions=[
            {"Name": "eval:avg_auc",        "Regex": r"avg_auc=([0-9.]+)"},
            {"Name": "eval:avg_f1_macro",   "Regex": r"avg_f1_macro=([0-9.]+)"},
            {"Name": "eval:avg_mae",        "Regex": r"avg_mae=([0-9.]+)"},
            {"Name": "eval:avg_ndcg3",      "Regex": r"avg_ndcg@3=([0-9.]+)"},
            {"Name": "eval:val_samples",    "Regex": r"val_samples=([0-9]+)"},
        ],
    )

    logger.info("Submitting eval job: %s", job_name)
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
    logger.info("Eval job %s: %s", job_name, result["status"])
    return result


# ---------------------------------------------------------------------------
# Step 4: Download results
# ---------------------------------------------------------------------------

def download_results(
    s3_bucket: str,
    scenario: str,
    local_dir: Optional[str] = None,
) -> Optional[str]:
    """Download eval_metrics.json from S3 to a local directory.

    SageMaker writes output into ``output.tar.gz`` under the job output
    prefix.  We download the raw file from the output prefix.

    Args:
        s3_bucket:  S3 bucket name.
        scenario:   Scenario name (determines S3 key prefix).
        local_dir:  Local directory to write eval_metrics.json.
                    Defaults to ``outputs/sagemaker_eval/{scenario}/``.

    Returns:
        Local path to the downloaded file, or None if not found.
    """
    import boto3

    s3 = boto3.client("s3")
    prefix = f"{S3_OUTPUT_PREFIX}/{scenario}/"

    if local_dir is None:
        local_dir = str(PROJECT_ROOT / "outputs" / "sagemaker_eval" / scenario)
    Path(local_dir).mkdir(parents=True, exist_ok=True)

    logger.info("Searching for eval_metrics.json under s3://%s/%s", s3_bucket, prefix)

    # List objects under the output prefix
    paginator = s3.get_paginator("list_objects_v2")
    found_keys = []
    for page in paginator.paginate(Bucket=s3_bucket, Prefix=prefix):
        for obj in page.get("Contents", []):
            key = obj["Key"]
            if "eval_metrics.json" in key:
                found_keys.append(key)

    if not found_keys:
        logger.warning("eval_metrics.json not found under s3://%s/%s", s3_bucket, prefix)
        # Fall back: look in output.tar.gz (SageMaker Training output)
        logger.info("Hint: SageMaker wraps output in output.tar.gz — "
                    "download manually with: aws s3 cp s3://%s/%s . --recursive",
                    s3_bucket, prefix)
        return None

    # Use the most recently modified match
    found_keys.sort()
    s3_key = found_keys[-1]
    local_path = Path(local_dir) / "eval_metrics.json"

    logger.info("Downloading s3://%s/%s -> %s", s3_bucket, s3_key, local_path)
    s3.download_file(s3_bucket, s3_key, str(local_path))
    logger.info("Downloaded: %s", local_path)

    # Pretty-print summary
    with open(local_path) as f:
        metrics = json.load(f)
    logger.info("=" * 60)
    logger.info("EVAL METRICS SUMMARY — %s", scenario)
    logger.info("=" * 60)
    for k in ("avg_auc", "avg_f1_macro", "avg_mae", "avg_ndcg@3",
              "val_samples", "checkpoint_epoch", "checkpoint_path"):
        if k in metrics:
            v = metrics[k]
            if isinstance(v, float):
                logger.info("  %s: %.6f", k, v)
            else:
                logger.info("  %s: %s", k, v)
    logger.info("=" * 60)

    return str(local_path)


# ---------------------------------------------------------------------------
# Cost check helper (mirrors run_sagemaker_teacher.py)
# ---------------------------------------------------------------------------

def verify_cost() -> None:
    """Check current AWS cost before submission."""
    try:
        today = time.strftime("%Y-%m-%d")
        result = subprocess.run(
            ["aws", "ce", "get-cost-and-usage",
             "--time-period", f"Start=2026-04-01,End={today}",
             "--granularity", "MONTHLY",
             "--metrics", "UnblendedCost"],
            capture_output=True, text=True, timeout=15,
        )
        if result.returncode == 0:
            cost_data = json.loads(result.stdout)
            for period in cost_data.get("ResultsByTime", []):
                amount = period["Total"]["UnblendedCost"]["Amount"]
                logger.info("Current month cost: $%s", amount)
        else:
            logger.warning("Cost check failed. Proceeding anyway.")
    except Exception as e:
        logger.warning("Cost check skipped: %s", e)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Submit a SageMaker evaluation job for a trained PLE checkpoint.",
    )
    parser.add_argument(
        "--checkpoint", required=True,
        help="Local path to the .pt checkpoint file (e.g. outputs/.../best.pt).",
    )
    parser.add_argument(
        "--scenario", required=True,
        help="Scenario label (e.g. teacher_full). Used in job name and S3 path.",
    )
    parser.add_argument(
        "--data-dir", default=None,
        help=(
            "Local Phase 0 data directory for S3 path derivation. "
            f"Default: outputs/phase0_v12 -> s3://bucket/{S3_DATA_PREFIX}"
        ),
    )
    parser.add_argument(
        "--s3-bucket", default=None,
        help="S3 bucket override (default: from pipeline.yaml).",
    )
    parser.add_argument(
        "--batch-size", type=int, default=5632,
        help="Inference batch size (default: 5632).",
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
        help="After completion, download eval_metrics.json from S3.",
    )
    parser.add_argument(
        "--output-dir", default=None,
        help="Local directory for downloaded eval_metrics.json.",
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
    # Resolve placeholder values
    if s3_bucket in ("YOUR_S3_BUCKET", ""):
        s3_bucket = "aiops-ple-financial"
        logger.info("Using default S3 bucket: %s", s3_bucket)

    s3_data_uri = f"s3://{s3_bucket}/{S3_DATA_PREFIX}"

    # --- HP dict ---
    hp: Dict[str, str] = {
        "config": CONTAINER_CONFIG,
        "batch_size": str(args.batch_size),
        "num_workers": "2",    # Linux (SageMaker) — 2 is safe
        "device": "auto",
        "ablation_scenario": args.scenario,
        "use_adatt": "false",
        "use_grad_surgery": "false",
    }

    # --- Pre-flight summary ---
    logger.info("=" * 60)
    logger.info("SageMaker Eval — %s", args.scenario)
    logger.info("=" * 60)
    logger.info("  Checkpoint:   %s", args.checkpoint)
    logger.info("  Scenario:     %s", args.scenario)
    logger.info("  S3 bucket:    %s", s3_bucket)
    logger.info("  Data URI:     %s", s3_data_uri)
    logger.info("  Instance:     %s (GPU — forward pass)", EVAL_INSTANCE_TYPE)
    logger.info("  Max run:      %ds (%d min)", EVAL_MAX_RUN, EVAL_MAX_RUN // 60)
    logger.info("  Use spot:     %s", aws_config["use_spot"])
    logger.info("  Dry run:      %s", args.dry_run)

    if args.dry_run:
        logger.info("\n[DRY-RUN] HP configuration:")
        for k, v in sorted(hp.items()):
            logger.info("  %s = %s", k, v)
        logger.info("[DRY-RUN] All checks passed. Ready to submit.")
        return

    # --- Cost check (CLAUDE.md 1.5) ---
    if not args.skip_cost_check:
        verify_cost()

    # --- Upload checkpoint ---
    if not args.force_upload:
        s3_model_uri = _checkpoint_already_on_s3(args.checkpoint, s3_bucket, args.scenario)
    else:
        s3_model_uri = None

    if s3_model_uri is None:
        s3_model_uri = upload_checkpoint(
            checkpoint_path=args.checkpoint,
            s3_bucket=s3_bucket,
            scenario=args.scenario,
        )

    # --- Build staging dir (CLAUDE.md 1.5: source 1회 빌드) ---
    logger.info("Building source staging directory...")
    staging_dir = _get_staging_dir()

    # --- Submit job ---
    wait = not args.no_wait
    job_result = submit_eval_job(
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

    logger.info("Job result: %s", {k: v for k, v in job_result.items() if k != "estimator"})

    # --- Download results ---
    if args.download_results and wait:
        logger.info("Downloading eval_metrics.json...")
        out_path = download_results(
            s3_bucket=s3_bucket,
            scenario=args.scenario,
            local_dir=args.output_dir,
        )
        if out_path:
            logger.info("Results saved to %s", out_path)
        else:
            logger.warning("Could not download eval_metrics.json automatically.")
            logger.info(
                "Download manually: aws s3 cp %s/output.tar.gz . && tar xzf output.tar.gz",
                job_result["s3_output_uri"],
            )
    elif args.download_results and not wait:
        logger.info(
            "Job submitted without --wait; re-run with --download-results "
            "after the job completes."
        )


if __name__ == "__main__":
    main()
