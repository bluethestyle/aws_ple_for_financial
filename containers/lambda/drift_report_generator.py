"""
Lambda trigger for drift report generation via SageMaker Processing Job.

Invoked by:
  - EventBridge scheduled rule (daily)
  - Manual invocation via console or API

This Lambda does NOT perform heavy computation itself.  It starts a
SageMaker Processing Job that loads Parquet data and runs DriftDetector.
The Processing Job saves the drift report JSON to S3, where
auto_retrain_trigger reads it on the next daily evaluation.

Event payload::

    {
        "action": "generate",          // or "manual"
        "baseline_s3_uri": "",         // empty = auto-detect from registry
        "current_s3_uri": "",          // empty = auto-detect from serving data
        "instance_type": "ml.m5.large",
        "config_override": {}
    }

Returns::

    {
        "triggered": true,
        "processing_job_name": "drift-report-2026-03-18-...",
        "output_s3": "s3://bucket/drift-reports/2026-03-18/"
    }
"""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timezone
from typing import Any, Dict

logger = logging.getLogger()
logger.setLevel(logging.INFO)

# Environment variables
REGION = os.environ.get("AWS_REGION", "ap-northeast-2")
PROCESSING_IMAGE_URI = os.environ.get("PROCESSING_IMAGE_URI", "")
PROCESSING_ROLE_ARN = os.environ.get("PROCESSING_ROLE_ARN", "")
REGISTRY_BASE = os.environ.get(
    "REGISTRY_BASE", "s3://aiops-ple-financial/models/artifacts"
)
DRIFT_REPORT_BUCKET = os.environ.get("DRIFT_REPORT_BUCKET", "aiops-ple-financial")
DRIFT_REPORT_PREFIX = os.environ.get("DRIFT_REPORT_PREFIX", "drift-reports")
BASELINE_S3_URI = os.environ.get("BASELINE_S3_URI", "")
CURRENT_S3_URI = os.environ.get("CURRENT_S3_URI", "")


def handler(event: Dict[str, Any], context: Any) -> Dict[str, Any]:
    """Lambda entry point — starts SageMaker Processing Job for drift detection."""
    import boto3

    action = event.get("action", "generate")
    instance_type = event.get("instance_type", "ml.m5.large")
    baseline_uri = event.get("baseline_s3_uri", "") or BASELINE_S3_URI
    current_uri = event.get("current_s3_uri", "") or CURRENT_S3_URI

    if not PROCESSING_IMAGE_URI:
        logger.error("PROCESSING_IMAGE_URI not set")
        return {"triggered": False, "reason": "missing_image_uri"}

    if not PROCESSING_ROLE_ARN:
        logger.error("PROCESSING_ROLE_ARN not set")
        return {"triggered": False, "reason": "missing_role_arn"}

    sm = boto3.client("sagemaker", region_name=REGION)
    now = datetime.now(timezone.utc)
    job_name = f"drift-report-{now.strftime('%Y-%m-%d-%H%M%S')}"
    output_s3 = f"s3://{DRIFT_REPORT_BUCKET}/{DRIFT_REPORT_PREFIX}/{now.strftime('%Y-%m-%d')}/"

    # Build entrypoint args
    entrypoint_args = [
        "python", "-c",
        _build_inline_script(baseline_uri, current_uri, output_s3),
    ]

    try:
        sm.create_processing_job(
            ProcessingJobName=job_name,
            ProcessingResources={
                "ClusterConfig": {
                    "InstanceCount": 1,
                    "InstanceType": instance_type,
                    "VolumeSizeInGB": 30,
                }
            },
            AppSpecification={
                "ImageUri": PROCESSING_IMAGE_URI,
                "ContainerEntrypoint": ["python", "-c", _build_inline_script(
                    baseline_uri, current_uri,
                    f"s3://{DRIFT_REPORT_BUCKET}/{DRIFT_REPORT_PREFIX}",
                )],
            },
            ProcessingOutputConfig={
                "Outputs": [
                    {
                        "OutputName": "drift_report",
                        "S3Output": {
                            "S3Uri": output_s3,
                            "LocalPath": "/opt/ml/processing/output",
                            "S3UploadMode": "EndOfJob",
                        },
                    }
                ]
            },
            RoleArn=PROCESSING_ROLE_ARN,
            StoppingCondition={"MaxRuntimeInSeconds": 1800},
            Tags=[
                {"Key": "Project", "Value": "PLE-AIOps"},
                {"Key": "Component", "Value": "DriftDetection"},
            ],
        )

        logger.info("Started drift report job: %s", job_name)
        return {
            "triggered": True,
            "processing_job_name": job_name,
            "output_s3": output_s3,
        }

    except Exception as e:
        logger.error("Failed to start drift report job: %s", e)
        return {"triggered": False, "reason": str(e)}


def _build_inline_script(baseline_uri: str, current_uri: str, output_prefix: str) -> str:
    """Build a small inline Python script for the Processing Job.

    The Processing Job container has the full project code installed,
    so this script just imports and calls the existing DriftDetector.
    """
    return f'''
import json, os, logging
from datetime import datetime, timezone

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("drift_report")

import duckdb
import numpy as np

from core.monitoring.drift_detector import DriftDetector

baseline_uri = "{baseline_uri}"
current_uri = "{current_uri}"
output_prefix = "{output_prefix}"

# Load data via DuckDB (fast Parquet reader, no pandas needed)
con = duckdb.connect()
con.execute("INSTALL httpfs; LOAD httpfs;")
con.execute("SET s3_region='ap-northeast-2';")

if baseline_uri:
    baseline = con.execute(f"SELECT * FROM read_parquet('{{baseline_uri}}')").fetchnumpy()
else:
    logger.warning("No baseline URI; using empty baseline")
    baseline = {{}}

if current_uri:
    current = con.execute(f"SELECT * FROM read_parquet('{{current_uri}}')").fetchnumpy()
else:
    logger.warning("No current URI; using empty current")
    current = {{}}

# Convert to 2D arrays for DriftDetector
if baseline and current:
    common_cols = sorted(set(baseline.keys()) & set(current.keys()))
    baseline_arr = np.column_stack([baseline[c] for c in common_cols]).astype(np.float32)
    current_arr = np.column_stack([current[c] for c in common_cols]).astype(np.float32)

    detector = DriftDetector()
    result = detector.detect_drift(baseline_arr, current_arr)

    # Build report in the format auto_retrain_trigger expects
    psi_scores = {{}}
    if hasattr(result, "psi_scores"):
        psi_scores = result.psi_scores
    elif isinstance(result, dict):
        psi_scores = result.get("psi_scores", {{}})

    report = {{
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "baseline_uri": baseline_uri,
        "current_uri": current_uri,
        "feature_names": common_cols,
        "psi_scores": psi_scores,
        "summary": result.summary if hasattr(result, "summary") else str(result),
    }}
else:
    report = {{
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "error": "missing baseline or current data",
        "psi_scores": {{}},
    }}

# Save to output dir (SageMaker uploads to S3 automatically)
output_dir = "/opt/ml/processing/output"
os.makedirs(output_dir, exist_ok=True)
date_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")
with open(os.path.join(output_dir, f"drift_{{date_str}}.json"), "w") as f:
    json.dump(report, f, indent=2, default=str)

logger.info("Drift report generated: %d features, PSI scores: %s",
    len(report.get("psi_scores", {{}})),
    {{k: round(v, 4) for k, v in list(report.get("psi_scores", {{}}).items())[:5]}})
'''
