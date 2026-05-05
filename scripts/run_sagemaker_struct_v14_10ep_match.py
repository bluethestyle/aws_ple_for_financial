#!/usr/bin/env python3
"""Submit 4 core scenarios at 10 epochs on g4dn.2xlarge for matched comparison
against the 15-epoch SageMaker batch.  Same code path / instance / precision
as the 15ep run; only ``epochs`` and ``warmup_epochs`` change.  Used to confirm
the hypothesis-B finding that PLE softmax recovers vs shared_bottom between
10 and 15 epochs.
"""
from __future__ import annotations

import sys
from pathlib import Path

# Reuse the main 15ep ablation script's plumbing
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.run_sagemaker_struct_ablation_v14 import (
    BASE_HPS, SCENARIOS, get_aws_config, load_pipeline_config,
    submit_training_jobs, monitor_jobs,
)

# Override epochs in-place — the BASE_HPS dict is mutated for this run only.
BASE_HPS["epochs"] = 10
BASE_HPS["warmup_epochs"] = 3
BASE_HPS["early_stopping_patience"] = 10

# Rename job_name suffix so jobs are distinguishable in CloudWatch.
TARGET_NAMES = {"shared_bottom", "ple_softmax", "ple_full", "ple_full_adatt"}
for s in SCENARIOS:
    if s["name"] in TARGET_NAMES:
        s["job_name"] = s["job_name"].replace("15ep", "10ep")


def main() -> None:
    config = load_pipeline_config()
    aws_config = get_aws_config(config)
    s3_bucket = aws_config["s3_bucket"]

    submitted = submit_training_jobs(
        aws_config, s3_bucket, selected=list(TARGET_NAMES), dry_run=False,
    )

    if submitted:
        print(f"Submitted {len(submitted)} matched 10ep jobs.")
        monitor_jobs(submitted)


if __name__ == "__main__":
    main()
