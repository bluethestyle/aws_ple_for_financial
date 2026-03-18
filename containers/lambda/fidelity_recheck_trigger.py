"""
Lambda trigger for weekly fidelity re-check SageMaker Processing Job.

Invoked by EventBridge weekly schedule.  Starts a SageMaker Processing Job
that runs ``scripts/run_fidelity_recheck.py`` with the currently promoted
model version.

Why a Lambda trigger instead of direct EventBridge → SageMaker?
  - EventBridge can't dynamically resolve the promoted model version
  - Lambda reads _promoted marker from S3, passes it to the Job
  - Lambda can skip the Job if no new predictions exist

Event payload::

    {
        "action": "schedule",           # or "manual"
        "model_version": "",            # empty = auto-detect promoted
        "instance_type": "ml.g4dn.xlarge",
        "lookback_days": 7,
        "sample_size": 10000
    }

Returns::

    {
        "triggered": true,
        "processing_job_name": "fidelity-recheck-2026-03-18-...",
        "model_version": "v-abc123"
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
REGISTRY_BASE = os.environ.get(
    "REGISTRY_BASE", "s3://aiops-ple-financial/models/artifacts"
)
PROCESSING_IMAGE_URI = os.environ.get("PROCESSING_IMAGE_URI", "")
PROCESSING_ROLE_ARN = os.environ.get("PROCESSING_ROLE_ARN", "")
PREDICTION_LOG_TABLE = os.environ.get("PREDICTION_LOG_TABLE", "ple-prediction-log")
SNS_TOPIC_ARN = os.environ.get("SNS_TOPIC_ARN", "")
REGION = os.environ.get("AWS_REGION", "ap-northeast-2")
OUTPUT_S3_BASE = os.environ.get(
    "OUTPUT_S3_BASE", "s3://aiops-ple-financial/fidelity-reports"
)


def handler(event: Dict[str, Any], context: Any) -> Dict[str, Any]:
    """Lambda entry point."""
    import boto3

    action = event.get("action", "schedule")
    model_version = event.get("model_version", "")
    instance_type = event.get("instance_type", "ml.g4dn.xlarge")
    lookback_days = event.get("lookback_days", 7)
    sample_size = event.get("sample_size", 10000)

    s3 = boto3.client("s3", region_name=REGION)

    # Resolve model version
    if not model_version:
        model_version = _read_promoted_version(s3)
        if not model_version:
            logger.warning("No promoted model version found; skipping")
            return {"triggered": False, "reason": "no_promoted_version"}

    logger.info("Fidelity recheck: version=%s, action=%s", model_version, action)

    # Check if there are recent predictions to validate against
    if action == "schedule":
        has_data = _check_recent_predictions(model_version, lookback_days)
        if not has_data:
            logger.info("No recent predictions for %s; skipping", model_version)
            return {"triggered": False, "reason": "no_recent_predictions"}

    # Start SageMaker Processing Job
    sm = boto3.client("sagemaker", region_name=REGION)
    now = datetime.now(timezone.utc)
    job_name = f"fidelity-recheck-{now.strftime('%Y-%m-%d-%H%M%S')}"

    output_s3 = f"{OUTPUT_S3_BASE.rstrip('/')}/{now.strftime('%Y-%m-%d')}/"

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
                "ContainerEntrypoint": [
                    "python", "scripts/run_fidelity_recheck.py",
                    "--model-version", model_version,
                    "--registry-base", REGISTRY_BASE,
                    "--prediction-log-table", PREDICTION_LOG_TABLE,
                    "--output-dir", "/opt/ml/processing/output",
                    "--sample-size", str(sample_size),
                    "--lookback-days", str(lookback_days),
                    "--region", REGION,
                    "--sns-topic-arn", SNS_TOPIC_ARN,
                    "--device", "cuda" if "g4dn" in instance_type or "p3" in instance_type else "cpu",
                ],
            },
            ProcessingOutputConfig={
                "Outputs": [
                    {
                        "OutputName": "fidelity_report",
                        "S3Output": {
                            "S3Uri": output_s3,
                            "LocalPath": "/opt/ml/processing/output",
                            "S3UploadMode": "EndOfJob",
                        },
                    }
                ]
            },
            RoleArn=PROCESSING_ROLE_ARN,
            StoppingCondition={"MaxRuntimeInSeconds": 3600},
            Tags=[
                {"Key": "Project", "Value": "PLE-AIOps"},
                {"Key": "Component", "Value": "FidelityRecheck"},
                {"Key": "ModelVersion", "Value": model_version},
            ],
        )

        logger.info("Started Processing Job: %s", job_name)

        return {
            "triggered": True,
            "processing_job_name": job_name,
            "model_version": model_version,
            "output_s3": output_s3,
        }

    except Exception as e:
        logger.error("Failed to start Processing Job: %s", e)
        return {"triggered": False, "reason": str(e)}


def _read_promoted_version(s3) -> str:
    """Read the _promoted marker from S3."""
    try:
        bucket_key = REGISTRY_BASE.replace("s3://", "").split("/", 1)
        bucket = bucket_key[0]
        key = (bucket_key[1] if len(bucket_key) > 1 else "") + "/_promoted"
        obj = s3.get_object(Bucket=bucket, Key=key)
        data = json.loads(obj["Body"].read())
        return data.get("active_version", "")
    except Exception:
        return ""


def _check_recent_predictions(version: str, lookback_days: int) -> bool:
    """Quick check if any predictions exist for this version."""
    try:
        import boto3
        from datetime import timedelta
        from boto3.dynamodb.conditions import Key

        dynamodb = boto3.resource("dynamodb", region_name=REGION)
        table = dynamodb.Table(PREDICTION_LOG_TABLE)

        cutoff = (
            datetime.now(timezone.utc) - timedelta(days=lookback_days)
        ).isoformat()

        response = table.query(
            IndexName="version-timestamp-index",
            KeyConditionExpression=(
                Key("version").eq(version)
                & Key("timestamp").gte(cutoff)
            ),
            Limit=1,
            Select="COUNT",
        )
        return response.get("Count", 0) > 0
    except Exception:
        return True  # Assume data exists on error (proceed with job)
