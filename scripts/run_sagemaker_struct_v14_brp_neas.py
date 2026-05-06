#!/usr/bin/env python3
"""Re-run BRP-detached / NEAS / NEAS+BRP-detached on v14 phase0 (15ep, NDCG).

Closes the v13→v14 gap on paper3 tab:fusion9way rows 8-10. The original
9-way fusion comparison was conducted on v13 phase0 (HMM duplication +
GMM K=20 dead clusters + prob-column scaling); rows 1-7 were re-run on
v14 in the NDCG batch (run_sagemaker_struct_v14_ndcg.py), but the three
follow-up mechanisms (BRP-detached, NEAS, NEAS+BRP-detached) were
deferred. This script runs only those three on v14 with NDCG enabled.

HP toggles (from scripts/run_local_ablation.py):
  - brp_detached:        use_brp=true, brp_detach_input=true
  - neas:                use_neas=true
  - neas_brp_detached:   use_brp=true, brp_detach_input=true, use_neas=true

All three keep use_ple=true, gate_type=sigmoid, use_cgc_gate=true, and
the GTE/LT/HMM-projector stack disabled (matching the local ablation
configs and the v14 NDCG batch's CGC-sigmoid baseline).

Cost: ~$0.40-0.50 per scenario × 3 = ~$1.30-1.60 spot
      (g4dn.2xlarge spot @ $0.226/hr × ~1.5h × 3, parallel).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.run_sagemaker_struct_ablation_v14 import (
    BASE_HPS, get_aws_config, load_pipeline_config,
    submit_training_jobs, monitor_jobs,
)
import scripts.run_sagemaker_struct_ablation_v14 as base

BRP_NEAS_SCENARIOS = [
    {"name": "brp_detached", "job_name": "brp-det-15ep-ndcg", "hp": {
        "use_ple": "true", "use_adatt": "false", "use_adatt_sp": "false",
        "use_residual_recovery": "false", "use_eceb": "false",
        "use_brp": "true", "brp_detach_input": "true",
        "gate_type": "sigmoid", "use_cgc_gate": "true",
        "use_group_task_expert": "false", "use_logit_transfer": "false",
        "use_hmm_projectors": "false",
    }},
    {"name": "neas", "job_name": "neas-15ep-ndcg", "hp": {
        "use_ple": "true", "use_adatt": "false", "use_adatt_sp": "false",
        "use_residual_recovery": "false", "use_eceb": "false",
        "use_brp": "false", "use_neas": "true",
        "gate_type": "sigmoid", "use_cgc_gate": "true",
        "use_group_task_expert": "false", "use_logit_transfer": "false",
        "use_hmm_projectors": "false",
    }},
    {"name": "neas_brp_detached", "job_name": "neas-brp-det-15ep-ndcg", "hp": {
        "use_ple": "true", "use_adatt": "false", "use_adatt_sp": "false",
        "use_residual_recovery": "false", "use_eceb": "false",
        "use_brp": "true", "brp_detach_input": "true",
        "use_neas": "true",
        "gate_type": "sigmoid", "use_cgc_gate": "true",
        "use_group_task_expert": "false", "use_logit_transfer": "false",
        "use_hmm_projectors": "false",
    }},
]

base.SCENARIOS = BRP_NEAS_SCENARIOS


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true",
                        help="Print HP table without submitting.")
    args = parser.parse_args()

    config = load_pipeline_config()
    aws_config = get_aws_config(config)
    s3_bucket = aws_config["s3_bucket"]

    submitted = submit_training_jobs(
        aws_config, s3_bucket,
        selected=[s["name"] for s in BRP_NEAS_SCENARIOS],
        dry_run=args.dry_run,
    )
    print(f"Submitted {len(submitted)} BRP-detached/NEAS jobs.")
    if submitted and not args.dry_run:
        monitor_jobs(submitted)


if __name__ == "__main__":
    main()
