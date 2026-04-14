#!/usr/bin/env python3
"""
SageMaker 30-epoch teacher training: 3 scenarios on Spot instances (parallel).

Scenarios:
  1. teacher_full        — PLE softmax, all 7 experts (main teacher)
  2. teacher_deepfm_base — DeepFM only (comparison)
  3. teacher_shared_bottom — Shared bottom, no PLE (comparison)

Usage:
    # Step 1: Upload Phase 0 data to S3
    python scripts/run_sagemaker_teacher.py --upload-data

    # Step 2: Dry-run — verify HP configuration before spending money
    python scripts/run_sagemaker_teacher.py --dry-run

    # Step 3: Submit all 3 jobs in parallel
    python scripts/run_sagemaker_teacher.py

    # Step 4: Monitor running jobs
    python scripts/run_sagemaker_teacher.py --monitor
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
from typing import Any, Dict, List

import yaml

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger("sagemaker_teacher")

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = PROJECT_ROOT / "configs" / "santander" / "pipeline.yaml"
PHASE0_DIR = PROJECT_ROOT / "outputs" / "phase0_v12"

# Container-internal paths (after source_dir extraction)
CONTAINER_CONFIG = "configs/santander/pipeline.yaml"

# S3 paths
S3_BUCKET = "aiops-ple-financial"
S3_DATA_PREFIX = "data/phase0_v12"
S3_CHECKPOINT_PREFIX = "checkpoints/teacher_30ep"
S3_OUTPUT_PREFIX = "output/teacher_30ep"
S3_SOURCE_PREFIX = "source/teacher_30ep"


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
        "s3_bucket": aws.get("s3_bucket", S3_BUCKET),
        "instance_type": aws.get("instance_type", "ml.g4dn.xlarge"),
        "role_arn": aws.get("role_arn"),
        "use_spot": aws.get("use_spot", True),
    }


# ---------------------------------------------------------------------------
# Base HPs and Scenarios (mirrors run_50ep_teacher.py)
# ---------------------------------------------------------------------------

BASE_HPS: Dict[str, Any] = {
    "config": CONTAINER_CONFIG,
    "epochs": 30,
    "batch_size": 5632,
    "learning_rate": 0.0005,
    "seed": 42,
    "amp": True,
    "early_stopping_patience": 30,
    "warmup_epochs": 5,
    "num_workers": 2,               # Linux (SageMaker) -> 2 OK
    "use_adatt": "false",           # adaTT OFF
    "use_grad_surgery": "false",    # GradSurgery OFF
}

SCENARIOS = [
    {
        "name": "teacher_full",
        "job_name": "teacher-full-30ep",
        "hp": {},
    },
    {
        "name": "teacher_deepfm_base",
        "job_name": "teacher-deepfm-30ep",
        "hp": {"shared_experts": json.dumps(["deepfm"])},
    },
    {
        "name": "teacher_shared_bottom",
        "job_name": "teacher-sb-30ep",
        "hp": {
            "use_ple": "false",
            "use_cgc_gate": "false",
            "use_group_task_expert": "false",
            "use_logit_transfer": "false",
            "use_hmm_projectors": "false",
        },
    },
]


# ---------------------------------------------------------------------------
# Step 1: Upload Phase 0 data to S3
# ---------------------------------------------------------------------------

def upload_phase0_data(s3_bucket: str) -> None:
    """Upload Phase 0 outputs + config files to S3."""
    import boto3

    s3 = boto3.client("s3")

    # Phase 0 files
    phase0_files = [
        "santander_final.parquet",
        "feature_schema.json",
        "label_stats.json",
        "adapter_metadata.json",
        "feature_stats.json",
        "quality_gate_report.json",
    ]

    logger.info("Uploading Phase 0 data to s3://%s/%s/", s3_bucket, S3_DATA_PREFIX)
    for fname in phase0_files:
        local_path = PHASE0_DIR / fname
        if not local_path.exists():
            logger.warning("  SKIP (not found): %s", local_path)
            continue
        s3_key = f"{S3_DATA_PREFIX}/{fname}"
        size_mb = local_path.stat().st_size / (1024 * 1024)
        logger.info("  Uploading %s (%.1f MB) -> s3://%s/%s", fname, size_mb, s3_bucket, s3_key)
        s3.upload_file(str(local_path), s3_bucket, s3_key)

    # Normalizer directory
    normalizer_dir = PHASE0_DIR / "normalizer"
    if normalizer_dir.exists():
        for fpath in normalizer_dir.rglob("*"):
            if fpath.is_file():
                rel = fpath.relative_to(PHASE0_DIR)
                s3_key = f"{S3_DATA_PREFIX}/{rel.as_posix()}"
                logger.info("  Uploading %s -> s3://%s/%s", rel, s3_bucket, s3_key)
                s3.upload_file(str(fpath), s3_bucket, s3_key)

    # Config files (needed inside container)
    config_files = [
        ("configs/santander/pipeline.yaml", "configs/santander/pipeline.yaml"),
        ("configs/santander/feature_groups.yaml", "configs/santander/feature_groups.yaml"),
    ]
    for local_rel, s3_rel in config_files:
        local_path = PROJECT_ROOT / local_rel
        if local_path.exists():
            s3_key = f"{S3_DATA_PREFIX}/{s3_rel}"
            logger.info("  Uploading %s -> s3://%s/%s", local_rel, s3_bucket, s3_key)
            s3.upload_file(str(local_path), s3_bucket, s3_key)

    logger.info("Upload complete.")


# ---------------------------------------------------------------------------
# Step 2: Build source.tar.gz (one build, reuse for all 3 jobs)
# ---------------------------------------------------------------------------

def build_staging_dir() -> str:
    """Create a lightweight staging directory with only the files needed for training.

    Copies:
      - containers/training/train.py (entry point)
      - core/ (model, training, data, pipeline — excludes __pycache__)
      - configs/santander/ (pipeline.yaml, feature_groups.yaml)

    Returns the path to the staging directory.
    """
    import shutil

    staging_dir = PROJECT_ROOT / "outputs" / "_sagemaker_staging"

    # Clean previous staging
    if staging_dir.exists():
        shutil.rmtree(str(staging_dir))

    include_dirs = ["core", "configs/santander", "containers/training"]
    include_files = ["containers/training/train.py"]

    logger.info("Building staging directory: %s", staging_dir)
    total_files = 0

    for rel_dir in include_dirs:
        src = PROJECT_ROOT / rel_dir
        if not src.is_dir():
            continue
        for fpath in src.rglob("*"):
            if not fpath.is_file():
                continue
            if "__pycache__" in str(fpath) or fpath.suffix == ".pyc":
                continue
            rel = fpath.relative_to(PROJECT_ROOT)
            dst = staging_dir / rel
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(str(fpath), str(dst))
            total_files += 1

    # Also copy core/__init__.py if it exists (needed for imports)
    init_py = PROJECT_ROOT / "core" / "__init__.py"
    if init_py.exists():
        dst = staging_dir / "core" / "__init__.py"
        dst.parent.mkdir(parents=True, exist_ok=True)
        if not dst.exists():
            shutil.copy2(str(init_py), str(dst))

    # Calculate size
    total_size = sum(f.stat().st_size for f in staging_dir.rglob("*") if f.is_file())
    logger.info("Staging: %d files, %.1f MB", total_files, total_size / (1024 * 1024))
    return str(staging_dir)


# ---------------------------------------------------------------------------
# Step 3: Create and submit SageMaker Training Jobs
# ---------------------------------------------------------------------------

def _build_hyperparameters(scenario: Dict[str, Any]) -> Dict[str, str]:
    """Merge base HPs with scenario-specific overrides. All values as strings."""
    hp = {}
    for k, v in BASE_HPS.items():
        hp[k] = str(v).lower() if isinstance(v, bool) else str(v)
    for k, v in scenario.get("hp", {}).items():
        hp[k] = str(v)
    # Tag scenario name for logging
    hp["ablation_scenario"] = scenario["name"]
    return hp


def submit_training_jobs(
    aws_config: Dict[str, Any],
    s3_bucket: str,
    dry_run: bool = False,
) -> List[Dict[str, Any]]:
    """Submit 3 training jobs in parallel on Spot instances."""
    from sagemaker.pytorch import PyTorch
    from sagemaker.inputs import TrainingInput
    import sagemaker

    session = sagemaker.Session()
    role = aws_config["role_arn"]
    instance_type = aws_config["instance_type"]

    # Build staging dir once (CLAUDE.md 1.5: source 1회 빌드 재사용)
    staging_dir = build_staging_dir()

    s3_data_uri = f"s3://{s3_bucket}/{S3_DATA_PREFIX}"
    timestamp = time.strftime("%m%d-%H%M")

    submitted_jobs = []

    for scenario in SCENARIOS:
        hp = _build_hyperparameters(scenario)
        job_name = f"{scenario['job_name']}-{timestamp}"

        logger.info("=" * 60)
        logger.info("Scenario: %s", scenario["name"])
        logger.info("Job name: %s", job_name)
        logger.info("Instance: %s (Spot=%s)", instance_type, aws_config["use_spot"])
        logger.info("HPs:")
        for k, v in sorted(hp.items()):
            logger.info("  %s = %s", k, v)

        # Critical HP verification
        if hp.get("use_adatt") != "false":
            logger.error("ABORT: use_adatt is NOT false! Got: %s", hp.get("use_adatt"))
            sys.exit(1)
        if hp.get("use_grad_surgery") != "false":
            logger.error("ABORT: use_grad_surgery is NOT false! Got: %s", hp.get("use_grad_surgery"))
            sys.exit(1)

        if dry_run:
            logger.info("[DRY-RUN] Would submit job: %s", job_name)
            submitted_jobs.append({"name": scenario["name"], "job_name": job_name, "status": "DRY_RUN"})
            continue

        estimator = PyTorch(
            entry_point="containers/training/train.py",
            source_dir=staging_dir,
            role=role,
            instance_type=instance_type,
            instance_count=1,
            framework_version="2.1",
            py_version="py310",
            hyperparameters=hp,
            use_spot_instances=aws_config["use_spot"],
            max_wait=7200,       # 2hr max wait (CLAUDE.md: max_wait = max_run + 1hr)
            max_run=5400,        # 1.5hr max run
            checkpoint_s3_uri=f"s3://{s3_bucket}/{S3_CHECKPOINT_PREFIX}/{scenario['name']}",
            output_path=f"s3://{s3_bucket}/{S3_OUTPUT_PREFIX}",
            disable_profiler=True,  # CLAUDE.md 1.5: profiler OFF
            environment={
                "PYTHONPATH": "/opt/ml/code",
                "PYTHONIOENCODING": "utf-8",
                "PYTHONUNBUFFERED": "1",
            },
            sagemaker_session=session,
            base_job_name=scenario["job_name"],
            # SageMaker Metric Definitions — auto-extract from CloudWatch logs
            metric_definitions=[
                {"Name": "train:loss", "Regex": r"train_loss=([0-9.]+)"},
                {"Name": "val:loss", "Regex": r"val_loss=([0-9.]+)"},
                {"Name": "val:avg_auc", "Regex": r"avg_auc=([0-9.]+)"},
                {"Name": "val:avg_f1_macro", "Regex": r"avg_f1_macro=([0-9.]+)"},
                {"Name": "val:avg_ndcg3", "Regex": r"avg_ndcg@3=([0-9.]+)"},
                {"Name": "val:avg_mae", "Regex": r"avg_mae=([0-9.]+)"},
                {"Name": "epoch", "Regex": r"Epoch (\d+):"},
            ],
        )

        logger.info("Submitting job: %s", job_name)
        estimator.fit(
            inputs={"train": TrainingInput(s3_data_uri, content_type="application/x-parquet")},
            job_name=job_name,
            wait=False,  # non-blocking — all 3 run in parallel
            logs="None",
        )

        submitted_jobs.append({
            "name": scenario["name"],
            "job_name": job_name,
            "estimator": estimator,
            "status": "Submitted",
        })
        logger.info("Submitted: %s", job_name)

    return submitted_jobs


# ---------------------------------------------------------------------------
# Step 4: Monitor jobs
# ---------------------------------------------------------------------------

def monitor_jobs(jobs: List[Dict[str, Any]], poll_interval: int = 60) -> None:
    """Poll job statuses until all complete or fail."""
    import boto3

    sm = boto3.client("sagemaker")
    pending = {j["job_name"]: j for j in jobs if j.get("status") != "DRY_RUN"}

    if not pending:
        logger.info("No jobs to monitor.")
        return

    logger.info("Monitoring %d jobs (poll every %ds)...", len(pending), poll_interval)
    terminal_states = {"Completed", "Failed", "Stopped"}
    results = {}

    while pending:
        for job_name in list(pending.keys()):
            try:
                desc = sm.describe_training_job(TrainingJobName=job_name)
                status = desc["TrainingJobStatus"]
                secondary = desc.get("SecondaryStatus", "")

                if status in terminal_states:
                    elapsed = "N/A"
                    if "TrainingEndTime" in desc and "TrainingStartTime" in desc:
                        dt = desc["TrainingEndTime"] - desc["TrainingStartTime"]
                        elapsed = f"{dt.total_seconds() / 60:.1f}min"

                    billable = desc.get("BillableTimeInSeconds", 0)
                    spot_savings = ""
                    if desc.get("TrainingTimeInSeconds", 0) > 0:
                        savings_pct = (1 - billable / desc["TrainingTimeInSeconds"]) * 100
                        spot_savings = f" (Spot savings: {savings_pct:.0f}%)"

                    if status == "Completed":
                        logger.info("[DONE] %s — %s, billable=%ds%s",
                                    job_name, elapsed, billable, spot_savings)
                    elif status == "Failed":
                        reason = desc.get("FailureReason", "unknown")
                        logger.error("[FAIL] %s — %s: %s", job_name, elapsed, reason)
                        # Distinguish Spot interruption vs algorithm error
                        transitions = desc.get("SecondaryStatusTransitions", [])
                        for t in transitions:
                            if "Interrupted" in t.get("Status", ""):
                                logger.warning("  -> Spot interruption detected")
                    else:
                        logger.warning("[STOP] %s — %s", job_name, elapsed)

                    results[job_name] = {"status": status, "billable_s": billable}
                    del pending[job_name]
                else:
                    logger.info("  [%s] %s — %s", status, job_name, secondary)

            except Exception as e:
                logger.warning("  Error checking %s: %s", job_name, e)

        if pending:
            time.sleep(poll_interval)

    # Summary
    logger.info("=" * 60)
    logger.info("ALL JOBS COMPLETE")
    total_billable = sum(r["billable_s"] for r in results.values())
    cost_estimate = total_billable / 3600 * 0.21  # Spot ~$0.21/hr
    logger.info("Total billable: %ds (%.1f min)", total_billable, total_billable / 60)
    logger.info("Estimated cost: $%.2f (Spot @ $0.21/hr)", cost_estimate)
    for jn, r in results.items():
        logger.info("  %s: %s (%ds)", jn, r["status"], r["billable_s"])


def monitor_existing_jobs() -> None:
    """Monitor jobs by name pattern (for --monitor flag after submission)."""
    import boto3

    sm = boto3.client("sagemaker")

    # Find recent teacher jobs
    response = sm.list_training_jobs(
        NameContains="teacher",
        SortBy="CreationTime",
        SortOrder="Descending",
        MaxResults=10,
    )

    active_jobs = []
    for job in response.get("TrainingJobSummaries", []):
        if job["TrainingJobStatus"] in ("InProgress", "Stopping"):
            active_jobs.append({
                "job_name": job["TrainingJobName"],
                "name": job["TrainingJobName"],
                "status": job["TrainingJobStatus"],
            })

    if not active_jobs:
        logger.info("No active teacher training jobs found.")
        # Show recent completed
        for job in response.get("TrainingJobSummaries", [])[:6]:
            logger.info("  %s: %s", job["TrainingJobName"], job["TrainingJobStatus"])
        return

    logger.info("Found %d active jobs:", len(active_jobs))
    for j in active_jobs:
        logger.info("  %s", j["job_name"])

    monitor_jobs(active_jobs)


# ---------------------------------------------------------------------------
# Verification helpers
# ---------------------------------------------------------------------------

def verify_phase0_data() -> bool:
    """Pre-flight check: verify Phase 0 outputs exist and are valid."""
    required_files = [
        "santander_final.parquet",
        "feature_schema.json",
        "label_stats.json",
    ]
    ok = True
    for fname in required_files:
        fpath = PHASE0_DIR / fname
        if not fpath.exists():
            logger.error("MISSING: %s", fpath)
            ok = False
        else:
            size_mb = fpath.stat().st_size / (1024 * 1024)
            logger.info("  OK: %s (%.1f MB)", fname, size_mb)

    # Check feature_schema
    schema_path = PHASE0_DIR / "feature_schema.json"
    if schema_path.exists():
        with open(schema_path) as f:
            schema = json.load(f)
        n_cols = len(schema.get("feature_columns", []))
        n_groups = len(schema.get("group_ranges", {}))
        logger.info("  feature_schema: %d columns, %d groups", n_cols, n_groups)

    # Check label_stats
    label_path = PHASE0_DIR / "label_stats.json"
    if label_path.exists():
        with open(label_path) as f:
            labels = json.load(f)
        logger.info("  label_stats: %d tasks", len(labels))

    return ok


def verify_cost() -> None:
    """Check current AWS cost before submission."""
    try:
        result = subprocess.run(
            ["aws", "ce", "get-cost-and-usage",
             "--time-period", "Start=2026-04-01,End=2026-04-15",
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
            logger.warning("Cost check failed (non-zero exit). Proceeding anyway.")
    except Exception as e:
        logger.warning("Cost check skipped: %s", e)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="SageMaker 30-epoch teacher training (3 scenarios, Spot parallel)",
    )
    parser.add_argument(
        "--upload-data", action="store_true",
        help="Upload Phase 0 data + configs to S3 (run once before training)",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Print HP configuration and verify without submitting jobs",
    )
    parser.add_argument(
        "--monitor", action="store_true",
        help="Monitor existing teacher training jobs",
    )
    parser.add_argument(
        "--s3-bucket", type=str, default=None,
        help=f"S3 bucket override (default: from pipeline.yaml or {S3_BUCKET})",
    )
    parser.add_argument(
        "--skip-cost-check", action="store_true",
        help="Skip AWS cost check before submission",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # Load config
    config = load_pipeline_config()
    aws_config = get_aws_config(config)
    s3_bucket = args.s3_bucket or aws_config.get("s3_bucket", S3_BUCKET)

    # Resolve placeholder
    if s3_bucket in ("YOUR_S3_BUCKET", ""):
        s3_bucket = S3_BUCKET
        logger.info("Using default S3 bucket: %s", s3_bucket)

    # --- Upload data ---
    if args.upload_data:
        logger.info("=" * 60)
        logger.info("STEP 1: Upload Phase 0 data to S3")
        logger.info("=" * 60)
        if not verify_phase0_data():
            logger.error("Phase 0 data validation failed. Aborting upload.")
            sys.exit(1)
        upload_phase0_data(s3_bucket)
        return

    # --- Monitor ---
    if args.monitor:
        monitor_existing_jobs()
        return

    # --- Submit (or dry-run) ---
    logger.info("=" * 60)
    logger.info("SageMaker Teacher Training — 30 epochs, 3 scenarios")
    logger.info("=" * 60)

    # Pre-flight verification
    logger.info("\n--- Pre-flight checks ---")
    if not verify_phase0_data():
        logger.error("Phase 0 data validation failed.")
        sys.exit(1)

    # Cost check (CLAUDE.md 1.5)
    if not args.dry_run and not args.skip_cost_check:
        verify_cost()

    # Cost estimate
    logger.info("\n--- Cost estimate ---")
    logger.info("  Instance: %s (Spot ~$0.21/hr)", aws_config["instance_type"])
    logger.info("  Max run: 1.5hr x 3 scenarios = 4.5hr max")
    logger.info("  Expected: ~1hr x 3 = ~$0.63 + S3 ~$0.05 = ~$0.68")
    logger.info("  Budget ceiling: $1.00")

    # HP verification
    logger.info("\n--- HP verification ---")
    for scenario in SCENARIOS:
        hp = _build_hyperparameters(scenario)
        logger.info("  %s:", scenario["name"])
        logger.info("    use_adatt=%s, use_grad_surgery=%s, epochs=%s, batch=%s, lr=%s",
                     hp["use_adatt"], hp["use_grad_surgery"],
                     hp["epochs"], hp["batch_size"], hp["learning_rate"])
        extra_keys = [k for k in hp if k not in BASE_HPS and k != "ablation_scenario"]
        if extra_keys:
            logger.info("    overrides: %s", {k: hp[k] for k in extra_keys})

    if args.dry_run:
        logger.info("\n[DRY-RUN] All checks passed. Ready to submit.")
        # Still run through submit with dry_run=True for full validation
        submit_training_jobs(aws_config, s3_bucket, dry_run=True)
        return

    # Submit
    logger.info("\n--- Submitting jobs ---")
    jobs = submit_training_jobs(aws_config, s3_bucket, dry_run=False)

    # Monitor
    logger.info("\n--- Monitoring ---")
    monitor_jobs(jobs, poll_interval=60)


if __name__ == "__main__":
    main()
