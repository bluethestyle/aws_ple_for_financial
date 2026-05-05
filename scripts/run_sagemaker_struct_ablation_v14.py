#!/usr/bin/env python3
"""
SageMaker 15-epoch structure ablation (15 scenarios, Spot parallel).

Resolves the surprising local v14 result where shared_bottom outperformed
PLE+CGC on Avg AUC (0.8139 vs 0.7980). Tests five hypotheses:

  A. Signal-cleaning (mean-aggregation enough): if true, PLE stays behind
     even at 15 epochs.
  B. Epoch budget (PLE under-converged at 10): if true, PLE catches up.
  C. Heterogeneous ensemble (mean-aggregation is the strong baseline).
  D. Gate overfitting (val_loss climbing).
  E. Parameter-count regularization gap.

Scenarios (15):
  Hypothesis A/B core (4):
    - struct_13_shared_bottom
    - struct_13_ple_softmax
    - struct_13_ple_full          (sigmoid + GTE + LT + HMM proj)
    - struct_13_ple_full_adatt    (joint_full with adaTT loss-level)
  Hypothesis C — shared_bottom expert removal (4):
    - struct_13_shared_bottom_minus_causal
    - struct_13_shared_bottom_minus_lightgcn
    - struct_13_shared_bottom_minus_temporal
    - struct_13_shared_bottom_minus_hgcn
  Hypothesis E — regularized PLE (1):
    - struct_13_ple_softmax_reg   (dropout=0.3, weight_decay=1e-4)
  Original 10-scenario plan supplements (6):
    - struct_13_ple_sigmoid
    - struct_13_ple_sigmoid_adatt
    - struct_13_ple_sigmoid_gte
    - struct_13_ple_sigmoid_gte_lt
    - struct_13_ple_softmax_adatt
    - struct_13_shared_bottom_adatt

Phase 0 source: outputs/phase0_v14 (HMM mode-split + GMM K=14 + prob un-scaled).
Batch size 2048 to match local v14 baselines for direct comparison.

Usage:
    # Step 1: upload phase0_v14 + configs
    python scripts/run_sagemaker_struct_ablation_v14.py --upload-data

    # Step 2: dry-run sanity check
    python scripts/run_sagemaker_struct_ablation_v14.py --dry-run

    # Step 3: submit all 15 jobs (limited to 4 concurrent via SageMaker
    # service quota / spot AZ pressure)
    python scripts/run_sagemaker_struct_ablation_v14.py

    # Step 4: monitor running jobs
    python scripts/run_sagemaker_struct_ablation_v14.py --monitor
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
logger = logging.getLogger("struct_ablation_v14")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = PROJECT_ROOT / "configs" / "pipeline.yaml"
DATASET_CONFIG_PATH = PROJECT_ROOT / "configs" / "datasets" / "santander.yaml"
PHASE0_DIR = PROJECT_ROOT / "outputs" / "phase0_v14" / "extracted"

CONTAINER_CONFIG = "configs/pipeline.yaml"
CONTAINER_DATASET_CONFIG = "configs/datasets/santander.yaml"

S3_BUCKET = "aiops-ple-financial"
S3_DATA_PREFIX = "data/phase0_v14"
S3_CHECKPOINT_PREFIX = "checkpoints/struct_v14_15ep"
S3_OUTPUT_PREFIX = "output/struct_v14_15ep"


def load_pipeline_config() -> Dict[str, Any]:
    from core.pipeline.config import load_merged_config
    if DATASET_CONFIG_PATH.exists():
        return load_merged_config(CONFIG_PATH, DATASET_CONFIG_PATH)
    with open(CONFIG_PATH, encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def get_aws_config(config: Dict[str, Any]) -> Dict[str, Any]:
    aws = config.get("aws", {})
    return {
        "region": aws.get("region", "ap-northeast-2"),
        "s3_bucket": aws.get("s3_bucket", S3_BUCKET),
        # ml.g4dn.2xlarge has 32 GB system RAM (vs 16 GB on g4dn.xlarge),
        # which fits the full 1M × 1210-col fp32 Arrow + tensor + model
        # without OOM.  Quota approved 2026-05-04 (8 spot instances).
        "instance_type": "ml.g4dn.2xlarge",
        "role_arn": aws.get("role_arn"),
        "use_spot": aws.get("use_spot", True),
    }


# Base HPs — match local v14 ablation conditions for direct comparability.
BASE_HPS: Dict[str, Any] = {
    "config": CONTAINER_CONFIG,
    "dataset_config": CONTAINER_DATASET_CONFIG,
    "epochs": 15,
    "batch_size": 2048,
    "learning_rate": 0.0005,
    "seed": 42,
    "amp": True,
    "early_stopping_patience": 15,
    "warmup_epochs": 4,             # 30% of 15 (matches 3/10 ratio)
    # num_workers=0 on g4dn.xlarge: each DataLoader worker forks the
    # main process and gets its own copy of the 5 GB Arrow Table once
    # workers touch it.  With num_workers=2 the resident set hits
    # 5 + 5 + 5 = 15 GB, exceeding the 16 GB container budget and
    # triggering OOM.  Single-process DataLoader fits comfortably.
    "num_workers": 0,
    "use_grad_surgery": "false",
    # Full 1M rows: train.py now streams features via DuckDB SELECT with
    # float32 cast (CLAUDE.md §3.3) instead of pq.read_table().
    # This brings the Arrow Table from ~10 GB DOUBLE down to ~5 GB FLOAT,
    # comfortably fitting g4dn.xlarge 16 GB system RAM.  No subsample
    # needed.
}

# Hypothesis-driven 15-scenario structure ablation.
SCENARIOS = [
    # === A/B core (4) — does PLE catch up at 15 epochs? ===
    {"name": "shared_bottom", "job_name": "sb-15ep", "hp": {
        "use_ple": "false", "use_adatt": "false", "use_cgc_gate": "false",
        "use_group_task_expert": "false", "use_logit_transfer": "false",
        "use_hmm_projectors": "false",
    }},
    {"name": "ple_softmax", "job_name": "ple-sm-15ep", "hp": {
        "use_ple": "true", "use_adatt": "false", "gate_type": "softmax",
        "use_cgc_gate": "true",
        "use_group_task_expert": "false", "use_logit_transfer": "false",
        "use_hmm_projectors": "false",
    }},
    {"name": "ple_full", "job_name": "ple-full-15ep", "hp": {
        "use_ple": "true", "use_adatt": "false", "gate_type": "sigmoid",
        "use_cgc_gate": "true",
        "use_group_task_expert": "true", "use_logit_transfer": "true",
        "use_hmm_projectors": "true",
    }},
    {"name": "ple_full_adatt", "job_name": "ple-full-adatt-15ep", "hp": {
        "use_ple": "true", "use_adatt": "true", "gate_type": "sigmoid",
        "use_cgc_gate": "true",
        "use_group_task_expert": "true", "use_logit_transfer": "true",
        "use_hmm_projectors": "true",
    }},
    # === C — shared_bottom expert removal (4) ===
    {"name": "shared_bottom_minus_causal", "job_name": "sb-no-causal-15ep", "hp": {
        "use_ple": "false", "use_adatt": "false", "use_cgc_gate": "false",
        "use_group_task_expert": "false", "use_logit_transfer": "false",
        "use_hmm_projectors": "false",
        "removed_experts": json.dumps(["causal"]),
    }},
    {"name": "shared_bottom_minus_lightgcn", "job_name": "sb-no-lgcn-15ep", "hp": {
        "use_ple": "false", "use_adatt": "false", "use_cgc_gate": "false",
        "use_group_task_expert": "false", "use_logit_transfer": "false",
        "use_hmm_projectors": "false",
        "removed_experts": json.dumps(["lightgcn"]),
    }},
    {"name": "shared_bottom_minus_temporal", "job_name": "sb-no-temp-15ep", "hp": {
        "use_ple": "false", "use_adatt": "false", "use_cgc_gate": "false",
        "use_group_task_expert": "false", "use_logit_transfer": "false",
        "use_hmm_projectors": "false",
        "removed_experts": json.dumps(["temporal_ensemble"]),
    }},
    {"name": "shared_bottom_minus_hgcn", "job_name": "sb-no-hgcn-15ep", "hp": {
        "use_ple": "false", "use_adatt": "false", "use_cgc_gate": "false",
        "use_group_task_expert": "false", "use_logit_transfer": "false",
        "use_hmm_projectors": "false",
        "removed_experts": json.dumps(["hgcn"]),
    }},
    # === E — regularized PLE (1) ===
    {"name": "ple_softmax_reg", "job_name": "ple-sm-reg-15ep", "hp": {
        "use_ple": "true", "use_adatt": "false", "gate_type": "softmax",
        "use_cgc_gate": "true",
        "use_group_task_expert": "false", "use_logit_transfer": "false",
        "use_hmm_projectors": "false",
        "dropout": "0.3",
        "weight_decay": "1e-4",
    }},
    # === Original progressive build (6) — toggle-by-toggle contribution ===
    {"name": "ple_sigmoid", "job_name": "ple-sg-15ep", "hp": {
        "use_ple": "true", "use_adatt": "false", "gate_type": "sigmoid",
        "use_cgc_gate": "true",
        "use_group_task_expert": "false", "use_logit_transfer": "false",
        "use_hmm_projectors": "false",
    }},
    {"name": "ple_sigmoid_adatt", "job_name": "ple-sg-adatt-15ep", "hp": {
        "use_ple": "true", "use_adatt": "true", "gate_type": "sigmoid",
        "use_cgc_gate": "true",
        "use_group_task_expert": "false", "use_logit_transfer": "false",
        "use_hmm_projectors": "false",
    }},
    {"name": "ple_sigmoid_gte", "job_name": "ple-sg-gte-15ep", "hp": {
        "use_ple": "true", "use_adatt": "false", "gate_type": "sigmoid",
        "use_cgc_gate": "true",
        "use_group_task_expert": "true", "use_logit_transfer": "false",
        "use_hmm_projectors": "false",
    }},
    {"name": "ple_sigmoid_gte_lt", "job_name": "ple-sg-gte-lt-15ep", "hp": {
        "use_ple": "true", "use_adatt": "false", "gate_type": "sigmoid",
        "use_cgc_gate": "true",
        "use_group_task_expert": "true", "use_logit_transfer": "true",
        "use_hmm_projectors": "false",
    }},
    {"name": "ple_softmax_adatt", "job_name": "ple-sm-adatt-15ep", "hp": {
        "use_ple": "true", "use_adatt": "true", "gate_type": "softmax",
        "use_cgc_gate": "true",
        "use_group_task_expert": "true", "use_logit_transfer": "true",
        "use_hmm_projectors": "true",
    }},
    {"name": "shared_bottom_adatt", "job_name": "sb-adatt-15ep", "hp": {
        "use_ple": "false", "use_adatt": "true", "use_cgc_gate": "false",
        "use_group_task_expert": "false", "use_logit_transfer": "false",
        "use_hmm_projectors": "false",
    }},
]


def upload_phase0_data(s3_bucket: str) -> None:
    """Upload Phase 0 v14 outputs + config files to S3."""
    import boto3

    s3 = boto3.client("s3")

    phase0_files = [
        "features.parquet",
        "labels.parquet",
        "feature_schema.json",
        "label_schema.json",
        "split_indices.json",
        "scaler_params.json",
        "feature_pipeline",
        "normalizer",
        "txn_sequences.npy",
        "product_sequences.npy",
        "phase0_summary.json",
    ]

    logger.info("Uploading Phase 0 v14 to s3://%s/%s/", s3_bucket, S3_DATA_PREFIX)
    for fname in phase0_files:
        local_path = PHASE0_DIR / fname
        if not local_path.exists():
            logger.warning("  SKIP (not found): %s", local_path)
            continue
        if local_path.is_file():
            s3_key = f"{S3_DATA_PREFIX}/{fname}"
            size_mb = local_path.stat().st_size / (1024 * 1024)
            logger.info("  Uploading %s (%.1f MB) -> s3://%s/%s",
                        fname, size_mb, s3_bucket, s3_key)
            s3.upload_file(str(local_path), s3_bucket, s3_key)
        elif local_path.is_dir():
            for fpath in local_path.rglob("*"):
                if fpath.is_file():
                    rel = fpath.relative_to(PHASE0_DIR)
                    s3_key = f"{S3_DATA_PREFIX}/{rel.as_posix()}"
                    size_mb = fpath.stat().st_size / (1024 * 1024)
                    logger.info("  Uploading %s (%.1f MB) -> s3://%s/%s",
                                rel, size_mb, s3_bucket, s3_key)
                    s3.upload_file(str(fpath), s3_bucket, s3_key)

    config_files = [
        ("configs/pipeline.yaml", "configs/pipeline.yaml"),
        ("configs/datasets/santander.yaml", "configs/datasets/santander.yaml"),
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


def build_staging_dir() -> str:
    """Create lightweight staging dir for SageMaker source upload."""
    from scripts.package_source import build_staging
    return build_staging(project_root=str(PROJECT_ROOT))


def _build_hyperparameters(scenario: Dict[str, Any]) -> Dict[str, str]:
    hp: Dict[str, str] = {}
    for k, v in BASE_HPS.items():
        hp[k] = str(v).lower() if isinstance(v, bool) else str(v)
    for k, v in scenario.get("hp", {}).items():
        hp[k] = str(v)
    hp["ablation_scenario"] = scenario["name"]
    hp["ablation_type"] = "structure_v14"
    return hp


def submit_training_jobs(
    aws_config: Dict[str, Any],
    s3_bucket: str,
    selected: List[str] | None,
    dry_run: bool = False,
) -> List[Dict[str, Any]]:
    from sagemaker.pytorch import PyTorch
    from sagemaker.inputs import TrainingInput
    import sagemaker

    session = sagemaker.Session()
    role = aws_config["role_arn"]
    instance_type = aws_config["instance_type"]
    staging_dir = build_staging_dir()

    s3_data_uri = f"s3://{s3_bucket}/{S3_DATA_PREFIX}"
    timestamp = time.strftime("%m%d-%H%M")

    submitted = []
    scenarios_to_run = SCENARIOS
    if selected:
        names = set(selected)
        scenarios_to_run = [s for s in SCENARIOS if s["name"] in names]
        if not scenarios_to_run:
            logger.error("No scenarios match --scenarios=%s. Available: %s",
                         selected, [s["name"] for s in SCENARIOS])
            sys.exit(1)

    for scenario in scenarios_to_run:
        hp = _build_hyperparameters(scenario)
        job_name = f"{scenario['job_name']}-{timestamp}"

        logger.info("=" * 60)
        logger.info("Scenario: %s", scenario["name"])
        logger.info("Job name: %s", job_name)
        logger.info("Instance: %s (Spot=%s)",
                    instance_type, aws_config["use_spot"])
        logger.info("HPs:")
        for k, v in sorted(hp.items()):
            logger.info("  %s = %s", k, v)

        if dry_run:
            logger.info("[DRY-RUN] Would submit job: %s", job_name)
            submitted.append({"name": scenario["name"], "job_name": job_name,
                              "status": "DRY_RUN"})
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
            max_wait=10800,    # 3hr (CLAUDE.md: max_run + 1hr)
            max_run=7200,      # 2hr (15 epoch ~95 min + buffer)
            checkpoint_s3_uri=(
                f"s3://{s3_bucket}/{S3_CHECKPOINT_PREFIX}/"
                f"{scenario['name']}/{job_name}"
            ),
            output_path=(
                f"s3://{s3_bucket}/{S3_OUTPUT_PREFIX}/"
                f"{scenario['name']}/{job_name}"
            ),
            disable_profiler=True,
            environment={
                "PYTHONPATH": "/opt/ml/code",
                "PYTHONIOENCODING": "utf-8",
                "PYTHONUNBUFFERED": "1",
            },
            sagemaker_session=session,
            base_job_name=scenario["job_name"],
            metric_definitions=[
                {"Name": "train:loss", "Regex": r"train_loss=([0-9.]+)"},
                {"Name": "val:loss", "Regex": r"val_loss=([0-9.]+)"},
                {"Name": "val:avg_auc", "Regex": r"avg_auc=([0-9.]+)"},
                {"Name": "val:avg_f1_macro", "Regex": r"avg_f1_macro=([0-9.]+)"},
                {"Name": "val:avg_mae", "Regex": r"avg_mae=([0-9.]+)"},
                {"Name": "epoch", "Regex": r"Epoch (\d+):"},
            ],
        )

        logger.info("Submitting job: %s", job_name)
        estimator.fit(
            inputs={"train": TrainingInput(
                s3_data_uri, content_type="application/x-parquet"
            )},
            job_name=job_name,
            wait=False,
            logs="None",
        )

        submitted.append({
            "name": scenario["name"],
            "job_name": job_name,
            "estimator": estimator,
            "status": "Submitted",
        })
        logger.info("Submitted: %s", job_name)

    return submitted


def monitor_jobs(jobs: List[Dict[str, Any]], poll: int = 60) -> None:
    import boto3

    sm = boto3.client("sagemaker")
    pending = {j["job_name"]: j for j in jobs if j.get("status") != "DRY_RUN"}
    if not pending:
        return

    logger.info("Monitoring %d jobs (poll every %ds)...", len(pending), poll)
    terminal = {"Completed", "Failed", "Stopped"}
    results = {}

    while pending:
        for job_name in list(pending.keys()):
            try:
                desc = sm.describe_training_job(TrainingJobName=job_name)
                status = desc["TrainingJobStatus"]
                secondary = desc.get("SecondaryStatus", "")
                if status in terminal:
                    elapsed = "N/A"
                    if "TrainingEndTime" in desc and "TrainingStartTime" in desc:
                        dt = desc["TrainingEndTime"] - desc["TrainingStartTime"]
                        elapsed = f"{dt.total_seconds()/60:.1f}min"
                    billable = desc.get("BillableTimeInSeconds", 0)
                    if status == "Completed":
                        logger.info("[DONE] %s — %s, billable=%ds",
                                    job_name, elapsed, billable)
                    elif status == "Failed":
                        reason = desc.get("FailureReason", "unknown")
                        logger.error("[FAIL] %s — %s: %s",
                                     job_name, elapsed, reason)
                    else:
                        logger.warning("[STOP] %s — %s", job_name, elapsed)
                    results[job_name] = {"status": status, "billable_s": billable}
                    del pending[job_name]
                else:
                    logger.info("  [%s] %s — %s", status, job_name, secondary)
            except Exception as e:
                logger.warning("  Error checking %s: %s", job_name, e)
        if pending:
            time.sleep(poll)

    logger.info("=" * 60)
    logger.info("ALL JOBS COMPLETE")
    total = sum(r["billable_s"] for r in results.values())
    cost = total / 3600 * 0.226
    logger.info("Total billable: %ds (%.1f min)", total, total/60)
    logger.info("Estimated cost: $%.2f (Spot @ $0.226/hr g4dn.xlarge)", cost)
    for jn, r in results.items():
        logger.info("  %s: %s (%ds)", jn, r["status"], r["billable_s"])


def monitor_existing_jobs() -> None:
    import boto3
    sm = boto3.client("sagemaker")
    response = sm.list_training_jobs(
        NameContains="-15ep",
        SortBy="CreationTime",
        SortOrder="Descending",
        MaxResults=30,
    )
    active = [
        {"job_name": j["TrainingJobName"], "name": j["TrainingJobName"],
         "status": j["TrainingJobStatus"]}
        for j in response.get("TrainingJobSummaries", [])
        if j["TrainingJobStatus"] in ("InProgress", "Stopping")
    ]
    if not active:
        logger.info("No active 15ep struct ablation jobs.")
        for j in response.get("TrainingJobSummaries", [])[:8]:
            logger.info("  %s: %s", j["TrainingJobName"], j["TrainingJobStatus"])
        return
    logger.info("Found %d active jobs.", len(active))
    monitor_jobs(active)


def verify_phase0_data() -> bool:
    required = ["features.parquet", "labels.parquet",
                "feature_schema.json", "label_schema.json", "split_indices.json"]
    ok = True
    for fname in required:
        fpath = PHASE0_DIR / fname
        if not fpath.exists():
            logger.error("MISSING: %s", fpath)
            ok = False
        else:
            size_mb = fpath.stat().st_size / (1024 * 1024)
            logger.info("  OK: %s (%.1f MB)", fname, size_mb)
    return ok


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="SageMaker 15-epoch structure ablation v14 "
                    "(15 scenarios, Spot parallel)",
    )
    p.add_argument("--upload-data", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--monitor", action="store_true")
    p.add_argument("--scenarios", type=str, default=None,
                   help="Comma-separated scenario names; default = all 15")
    p.add_argument("--s3-bucket", type=str, default=None)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    config = load_pipeline_config()
    aws_config = get_aws_config(config)
    s3_bucket = args.s3_bucket or aws_config["s3_bucket"]

    if args.monitor:
        monitor_existing_jobs()
        return

    if args.upload_data:
        if not verify_phase0_data():
            logger.error("Phase 0 v14 data missing; aborting upload.")
            sys.exit(1)
        upload_phase0_data(s3_bucket)
        return

    if not verify_phase0_data():
        sys.exit(1)

    selected = args.scenarios.split(",") if args.scenarios else None
    submitted = submit_training_jobs(
        aws_config, s3_bucket, selected=selected, dry_run=args.dry_run,
    )

    if not args.dry_run and submitted:
        logger.info("All %d jobs submitted; entering monitor loop.",
                    len(submitted))
        monitor_jobs(submitted)


if __name__ == "__main__":
    main()
