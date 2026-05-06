#!/usr/bin/env python3
"""Submit 4 paper3 Finding 7 multi-mechanism scenarios on g4dn.2xlarge:
AdaTT-sp (Li 2023), M1 complement, ECEB, BRP.  Companion to the v14 main
batch — these are the residual-recovery / fusion variants whose v13
results sit at @tab:multi-mechanism in paper3.typ.
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

# Scenarios from run_local_ablation, ported here with v14 shapes.
NEW_SCENARIOS = [
    {"name": "adatt_sp", "job_name": "adattsp-15ep", "hp": {
        "use_ple": "true", "use_adatt": "false", "use_adatt_sp": "true",
        "gate_type": "sigmoid", "use_cgc_gate": "true",
        "use_group_task_expert": "false", "use_logit_transfer": "false",
        "use_hmm_projectors": "false",
    }},
    {"name": "residual_complement", "job_name": "m1cmpl-15ep", "hp": {
        "use_ple": "true", "use_adatt": "false", "use_adatt_sp": "false",
        "use_residual_recovery": "true", "residual_method": "complement",
        "gate_type": "sigmoid", "use_cgc_gate": "true",
        "use_group_task_expert": "false", "use_logit_transfer": "false",
        "use_hmm_projectors": "false",
    }},
    {"name": "eceb", "job_name": "eceb-15ep", "hp": {
        "use_ple": "true", "use_adatt": "false", "use_adatt_sp": "false",
        "use_residual_recovery": "false", "use_eceb": "true",
        "gate_type": "sigmoid", "use_cgc_gate": "true",
        "use_group_task_expert": "false", "use_logit_transfer": "false",
        "use_hmm_projectors": "false",
    }},
    {"name": "brp", "job_name": "brp-15ep", "hp": {
        "use_ple": "true", "use_adatt": "false", "use_adatt_sp": "false",
        "use_residual_recovery": "false", "use_eceb": "false",
        "use_brp": "true",
        "gate_type": "sigmoid", "use_cgc_gate": "true",
        "use_group_task_expert": "false", "use_logit_transfer": "false",
        "use_hmm_projectors": "false",
    }},
]

# Inject into the base SCENARIOS list so submit_training_jobs picks them up
base.SCENARIOS = NEW_SCENARIOS


def main() -> None:
    config = load_pipeline_config()
    aws_config = get_aws_config(config)
    s3_bucket = aws_config["s3_bucket"]

    submitted = submit_training_jobs(
        aws_config, s3_bucket,
        selected=[s["name"] for s in NEW_SCENARIOS],
        dry_run=False,
    )
    print(f"Submitted {len(submitted)} multi-mechanism jobs.")
    if submitted:
        monitor_jobs(submitted)


if __name__ == "__main__":
    main()
