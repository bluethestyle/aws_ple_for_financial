#!/usr/bin/env python3
"""OCP (orthogonal-complement residual recovery) 3-way ablation on SageMaker Spot.

Three jobs with identical architecture/HPs, differing only in the residual-
recovery fusion at the CGC gate:

  - ocp_baseline_cgc   : plain CGC gate (no residual recovery)
  - ocp_m1_complement  : M1 — gate-inverse (1-gate) full residual
  - ocp_orthogonal     : OCP — same (1-gate) residual but with the gated
                         primary direction projected out first, so only the
                         component orthogonal to the primary is recovered

The comparison isolates two questions:
  1. Task metrics — does OCP change Avg AUC/F1/MAE vs M1 / baseline?
  2. Expert orthogonality — via the CCA expert_redundancy analysis now wired
     into each training job's eval (analysis.expert_redundancy.enabled), does
     OCP actually lower canonical correlation between shared experts?
     Each job emits expert_redundancy.json to its S3 output path.

Phase 0 source: s3://aiops-ple-financial/data/phase0_v14 (same as struct v14).
Spot, 15 epoch, batch 2048, seed 42 — inherited from BASE_HPS.

Usage:
    # dry-run (DEFAULT — prints merged HPs + Spot config, submits nothing)
    python scripts/run_sagemaker_ocp.py

    # actually launch the 3 Spot jobs
    python scripts/run_sagemaker_ocp.py --submit
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.run_sagemaker_struct_ablation_v14 import (  # noqa: E402
    get_aws_config, load_pipeline_config, submit_training_jobs, monitor_jobs,
)
import scripts.run_sagemaker_struct_ablation_v14 as base  # noqa: E402

# Identical across all three; only the residual-recovery fusion differs.
_COMMON = {
    "use_ple": "true",
    "use_adatt": "false",
    "use_adatt_sp": "false",
    "gate_type": "sigmoid",
    "use_cgc_gate": "true",
    "use_group_task_expert": "false",
    "use_logit_transfer": "false",
    "use_hmm_projectors": "false",
}

NEW_SCENARIOS = [
    {"name": "ocp_baseline_cgc", "job_name": "ocp-base-15ep", "hp": {
        **_COMMON,
        "use_residual_recovery": "false",
        "use_eceb": "false",
    }},
    {"name": "ocp_m1_complement", "job_name": "ocp-m1cmpl-15ep", "hp": {
        **_COMMON,
        "use_residual_recovery": "true",
        "residual_method": "complement",
    }},
    {"name": "ocp_orthogonal", "job_name": "ocp-orth-15ep", "hp": {
        **_COMMON,
        "use_residual_recovery": "true",
        "residual_method": "orthogonal",
    }},
]

# submit_training_jobs reads the module-global SCENARIOS list.
base.SCENARIOS = NEW_SCENARIOS


def main() -> None:
    parser = argparse.ArgumentParser(description="OCP 3-way Spot ablation")
    parser.add_argument(
        "--submit", action="store_true",
        help="Actually launch the Spot jobs (default: dry-run only)",
    )
    parser.add_argument(
        "--monitor", action="store_true",
        help="Poll job status after submitting (blocks until all jobs finish)",
    )
    args = parser.parse_args()

    config = load_pipeline_config()
    aws_config = get_aws_config(config)
    s3_bucket = aws_config["s3_bucket"]

    submitted = submit_training_jobs(
        aws_config, s3_bucket,
        selected=[s["name"] for s in NEW_SCENARIOS],
        dry_run=not args.submit,
    )
    mode = "Submitted" if args.submit else "DRY-RUN (nothing launched)"
    print(f"{mode}: {len(submitted)} OCP jobs "
          f"({', '.join(s['name'] for s in NEW_SCENARIOS)}).")
    if args.submit and submitted:
        for s in submitted:
            print(f"  job: {s['job_name']}  ({s['name']})")
        if args.monitor:
            monitor_jobs(submitted)


if __name__ == "__main__":
    main()
