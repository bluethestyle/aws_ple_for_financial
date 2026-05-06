#!/usr/bin/env python3
"""Re-run paper-critical scenarios with NDCG@3 enabled.

The original v14 SageMaker batch did not surface NDCG metrics because
``label_schema.json`` was missing the ``topk_k`` field (Stage-4 schema
builder dropped it). The patched ``label_schema.json`` is now uploaded
to S3 (s3://aiops-ple-financial/data/phase0_v14/), and this script
re-submits the 7 scenarios that paper1 / paper3 narrative depends on
so we have NDCG@3 for the multiclass task analysis (nba_primary,
next_mcc).

Scenarios re-run (15 epochs, ml.g4dn.2xlarge):
  - shared_bottom        (Finding 4 baseline)
  - ple_softmax          (Paper 1 best)
  - ple_full             (full toggles, sigmoid)
  - ple_full_adatt       (full + adaTT)
  - residual_complement  (M1)
  - eceb                 (Error-Conditioned Expert Bank)
  - brp                  (Boosting-Residual Path)

Other scenarios (sb-no-X, ple_sigmoid, ple_softmax_reg, etc.) keep
their non-NDCG numbers in the aggregate; the multiclass NDCG analysis
only needs the seven above.
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
import json

NDCG_SCENARIOS = [
    {"name": "shared_bottom", "job_name": "sb-15ep-ndcg", "hp": {
        "use_ple": "false", "use_adatt": "false", "use_cgc_gate": "false",
        "use_group_task_expert": "false", "use_logit_transfer": "false",
        "use_hmm_projectors": "false",
    }},
    {"name": "ple_softmax", "job_name": "ple-sm-15ep-ndcg", "hp": {
        "use_ple": "true", "use_adatt": "false", "gate_type": "softmax",
        "use_cgc_gate": "true",
        "use_group_task_expert": "false", "use_logit_transfer": "false",
        "use_hmm_projectors": "false",
    }},
    {"name": "ple_full", "job_name": "ple-full-15ep-ndcg", "hp": {
        "use_ple": "true", "use_adatt": "false", "gate_type": "sigmoid",
        "use_cgc_gate": "true",
        "use_group_task_expert": "true", "use_logit_transfer": "true",
        "use_hmm_projectors": "true",
    }},
    {"name": "ple_full_adatt", "job_name": "ple-full-adatt-15ep-ndcg", "hp": {
        "use_ple": "true", "use_adatt": "true", "gate_type": "sigmoid",
        "use_cgc_gate": "true",
        "use_group_task_expert": "true", "use_logit_transfer": "true",
        "use_hmm_projectors": "true",
    }},
    {"name": "residual_complement", "job_name": "m1cmpl-15ep-ndcg", "hp": {
        "use_ple": "true", "use_adatt": "false", "use_adatt_sp": "false",
        "use_residual_recovery": "true", "residual_method": "complement",
        "gate_type": "sigmoid", "use_cgc_gate": "true",
        "use_group_task_expert": "false", "use_logit_transfer": "false",
        "use_hmm_projectors": "false",
    }},
    {"name": "eceb", "job_name": "eceb-15ep-ndcg", "hp": {
        "use_ple": "true", "use_adatt": "false", "use_adatt_sp": "false",
        "use_residual_recovery": "false", "use_eceb": "true",
        "gate_type": "sigmoid", "use_cgc_gate": "true",
        "use_group_task_expert": "false", "use_logit_transfer": "false",
        "use_hmm_projectors": "false",
    }},
    {"name": "brp", "job_name": "brp-15ep-ndcg", "hp": {
        "use_ple": "true", "use_adatt": "false", "use_adatt_sp": "false",
        "use_residual_recovery": "false", "use_eceb": "false",
        "use_brp": "true",
        "gate_type": "sigmoid", "use_cgc_gate": "true",
        "use_group_task_expert": "false", "use_logit_transfer": "false",
        "use_hmm_projectors": "false",
    }},
]

base.SCENARIOS = NDCG_SCENARIOS


def main() -> None:
    config = load_pipeline_config()
    aws_config = get_aws_config(config)
    s3_bucket = aws_config["s3_bucket"]

    submitted = submit_training_jobs(
        aws_config, s3_bucket,
        selected=[s["name"] for s in NDCG_SCENARIOS],
        dry_run=False,
    )
    print(f"Submitted {len(submitted)} NDCG-enabled jobs.")
    if submitted:
        monitor_jobs(submitted)


if __name__ == "__main__":
    main()
