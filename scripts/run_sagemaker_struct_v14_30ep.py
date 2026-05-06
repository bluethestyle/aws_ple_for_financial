#!/usr/bin/env python3
"""Submit a single PLE softmax 30-epoch run on g4dn.2xlarge for paper3
Finding 6 (long-budget cosine warm restart trajectory).

The v13 Finding 6 reported avg AUC peak at epoch 10 (0.6726), declining
to 0.6687 by epoch 30 with cosine warm restarts triggering at epochs 10
and 20. The v14 phase0 fixes (HMM/GMM/normalizer) shifted AUC magnitudes
into the 0.82+ band, so the trajectory pattern needs re-validation on
v14 data. Single scenario ≈ $1.50 spot, ~3h wallclock.
"""
from __future__ import annotations

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.run_sagemaker_struct_ablation_v14 import (
    BASE_HPS, get_aws_config, load_pipeline_config,
    submit_training_jobs, monitor_jobs,
)
import scripts.run_sagemaker_struct_ablation_v14 as base

BASE_HPS["epochs"] = 30
BASE_HPS["warmup_epochs"] = 10  # T_0 = 10 cosine warm restart base
BASE_HPS["early_stopping_patience"] = 30

SCENARIOS_30EP = [
    {"name": "ple_softmax", "job_name": "ple-sm-30ep", "hp": {
        "use_ple": "true", "use_adatt": "false", "gate_type": "softmax",
        "use_cgc_gate": "true",
        "use_group_task_expert": "false", "use_logit_transfer": "false",
        "use_hmm_projectors": "false",
    }},
]
base.SCENARIOS = SCENARIOS_30EP


def main() -> None:
    config = load_pipeline_config()
    aws_config = get_aws_config(config)
    s3_bucket = aws_config["s3_bucket"]

    submitted = submit_training_jobs(
        aws_config, s3_bucket,
        selected=["ple_softmax"],
        dry_run=False,
    )
    print(f"Submitted {len(submitted)} 30-epoch job.")
    if submitted:
        monitor_jobs(submitted)


if __name__ == "__main__":
    main()
