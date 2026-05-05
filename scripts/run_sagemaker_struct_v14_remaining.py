#!/usr/bin/env python3
"""Submit remaining 3 scenarios:
- ple_full_adatt 10ep (matches 1528 10ep batch)
- ple_softmax_adatt 15ep
- shared_bottom_adatt 15ep
"""
from __future__ import annotations

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.run_sagemaker_struct_ablation_v14 import (
    BASE_HPS, SCENARIOS, get_aws_config, load_pipeline_config,
    submit_training_jobs, monitor_jobs,
)

# Submit ple_full_adatt 10ep matched first; rename job_name to "10ep"
import copy
SCENARIOS_LOCAL = copy.deepcopy(SCENARIOS)
for s in SCENARIOS_LOCAL:
    if s["name"] == "ple_full_adatt":
        s["job_name"] = s["job_name"].replace("15ep", "10ep")
        # mark this scenario for 10ep override
        s["_use_10ep"] = True


def main() -> None:
    config = load_pipeline_config()
    aws_config = get_aws_config(config)
    s3_bucket = aws_config["s3_bucket"]

    # Build the 10ep version first (separate submission so BASE_HPS is mutated only for it)
    targets_10ep = ["ple_full_adatt"]
    BASE_HPS["epochs"] = 10
    BASE_HPS["warmup_epochs"] = 3
    BASE_HPS["early_stopping_patience"] = 10
    # Patch SCENARIOS in module to picked up 10ep
    import scripts.run_sagemaker_struct_ablation_v14 as base
    for s in base.SCENARIOS:
        if s["name"] == "ple_full_adatt":
            s["job_name"] = s["job_name"].replace("15ep", "10ep")

    submit_10ep = submit_training_jobs(
        aws_config, s3_bucket, selected=targets_10ep, dry_run=False,
    )

    # Reset BASE_HPS to 15ep for next submissions
    BASE_HPS["epochs"] = 15
    BASE_HPS["warmup_epochs"] = 4
    BASE_HPS["early_stopping_patience"] = 15
    # Restore job_name
    for s in base.SCENARIOS:
        if s["name"] == "ple_full_adatt":
            s["job_name"] = s["job_name"].replace("10ep", "15ep")

    targets_15ep = ["ple_softmax_adatt", "shared_bottom_adatt"]
    submit_15ep = submit_training_jobs(
        aws_config, s3_bucket, selected=targets_15ep, dry_run=False,
    )

    submitted = submit_10ep + submit_15ep
    print(f"Submitted {len(submitted)} jobs total.")
    if submitted:
        monitor_jobs(submitted)


if __name__ == "__main__":
    main()
