#!/usr/bin/env python3
"""Aggregate completed SageMaker v14 struct ablation jobs into a markdown
table.  Saves to ``outputs/sagemaker_struct_v14_results.md`` for direct
paste-and-edit into paper1.typ tables.

Usage:
    PYTHONPATH=. python scripts/aggregate_sagemaker_v14.py
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

import boto3

OUTPUT_PATH = Path("outputs/sagemaker_struct_v14_results.md")
JOB_NAME_PATTERNS = ["15ep-0504", "10ep-0504"]

WANTED_METRICS = {
    "val:avg_auc": "AUC",
    "val:avg_f1_macro": "F1m",
    "val:avg_mae": "MAE",
    "val:loss": "loss",
    "epoch": "ep",
}


def fetch_completed_jobs() -> List[Dict[str, Any]]:
    sm = boto3.client("sagemaker")
    rows: List[Dict[str, Any]] = []
    next_token = None
    while True:
        kwargs = {
            "StatusEquals": "Completed",
            "MaxResults": 100,
            "SortBy": "CreationTime",
            "SortOrder": "Descending",
        }
        if next_token:
            kwargs["NextToken"] = next_token
        resp = sm.list_training_jobs(**kwargs)
        for j in resp.get("TrainingJobSummaries", []):
            name = j["TrainingJobName"]
            if not any(p in name for p in JOB_NAME_PATTERNS):
                continue
            desc = sm.describe_training_job(TrainingJobName=name)
            metrics = {
                m["MetricName"]: m["Value"]
                for m in desc.get("FinalMetricDataList", [])
                if m["MetricName"] in WANTED_METRICS
            }
            hp = desc.get("HyperParameters", {})
            rows.append({
                "job_name": name,
                "scenario": str(hp.get("ablation_scenario", "?")).strip('"'),
                "epochs": int(str(hp.get("epochs", "-1")).strip('"')),
                "instance": desc.get("ResourceConfig", {}).get("InstanceType", "?"),
                "metrics": metrics,
                "billable_s": desc.get("BillableTimeInSeconds", 0),
            })
        next_token = resp.get("NextToken")
        if not next_token:
            break
    return rows


def emit_markdown(rows: List[Dict[str, Any]]) -> str:
    rows.sort(key=lambda r: (r["scenario"], r["epochs"]))
    lines: List[str] = []
    lines.append("# SageMaker v14 Struct Ablation Results")
    lines.append("")
    lines.append(f"Total completed jobs: {len(rows)}")
    lines.append("")
    lines.append("| Scenario | Epochs | Instance | AUC | F1m | MAE | val_loss | Billable (s) |")
    lines.append("|---|---:|---|---:|---:|---:|---:|---:|")
    for r in rows:
        m = r["metrics"]
        auc = f"{m.get('val:avg_auc', float('nan')):.4f}"
        f1m = f"{m.get('val:avg_f1_macro', float('nan')):.4f}"
        mae = f"{m.get('val:avg_mae', float('nan')):.4f}"
        loss = f"{m.get('val:loss', float('nan')):.2f}"
        bs = r["billable_s"]
        lines.append(
            f"| {r['scenario']} | {r['epochs']} | "
            f"{r['instance'].replace('ml.','')} | "
            f"{auc} | {f1m} | {mae} | {loss} | {bs} |"
        )
    lines.append("")
    lines.append("## Source jobs")
    lines.append("")
    for r in rows:
        lines.append(f"- {r['job_name']} ({r['scenario']}, {r['epochs']}ep)")
    return "\n".join(lines) + "\n"


def main() -> None:
    rows = fetch_completed_jobs()
    md = emit_markdown(rows)
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(md, encoding="utf-8")
    print(f"Wrote {len(rows)} jobs to {OUTPUT_PATH}")
    print()
    print(md)


if __name__ == "__main__":
    main()
