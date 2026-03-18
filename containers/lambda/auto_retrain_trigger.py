"""
Lambda handler for automatic retraining trigger.

Invoked by:
  - EventBridge scheduled rule (daily evaluation)
  - CloudWatch Alarm → SNS → Lambda (performance degradation)
  - Manual invocation via console or API

Evaluates whether model retraining is necessary based on:
  A. PSI critical threshold exceeded (>= 0.25) for any feature
  B. Champion-Challenger: challenger significantly better
  C. Staleness: last training older than N days (default 30)
  D. Manual trigger (event.action == "manual")

On trigger, starts a Step Functions execution of the training pipeline.

Event payload::

    {
        "action": "evaluate",       // "evaluate" or "manual"
        "force": false,
        "config_override": {}       // optional overrides
    }

Returns::

    {
        "triggered": true,
        "reason": "psi_critical_exceeded",
        "execution_arn": "arn:aws:states:...",
        "metrics_summary": {...}
    }
"""

from __future__ import annotations

import json
import logging
import os
import uuid
from datetime import datetime, timezone, timedelta
from typing import Any, Dict, Optional

logger = logging.getLogger()
logger.setLevel(logging.INFO)

# ---------------------------------------------------------------------------
# Default configuration (overridable via env vars or event.config_override)
# ---------------------------------------------------------------------------

DEFAULT_CONFIG = {
    "psi_critical_threshold": 0.25,
    "staleness_days": 30,
    "challenger_lift_threshold": 0.02,
    "region": "ap-northeast-2",
    "state_machine_arn": os.environ.get(
        "STATE_MACHINE_ARN",
        "arn:aws:states:ap-northeast-2:ACCOUNT:stateMachine:ple-training-pipeline",
    ),
    "drift_report_bucket": os.environ.get("DRIFT_REPORT_BUCKET", "ple-monitoring"),
    "drift_report_prefix": os.environ.get("DRIFT_REPORT_PREFIX", "drift-reports/"),
    "monitor_table": os.environ.get("MONITOR_TABLE", "ple-prediction-log"),
    "retrain_log_table": os.environ.get("RETRAIN_LOG_TABLE", "ple-retrain-log"),
    "audit_table_prefix": os.environ.get("AUDIT_TABLE_PREFIX", "ple-audit"),
    "champion_version": os.environ.get("CHAMPION_VERSION", ""),
    "challenger_version": os.environ.get("CHALLENGER_VERSION", ""),
    "task_name": os.environ.get("TASK_NAME", "financial_recommendation_ple"),
    # Pipeline input defaults
    "pipeline_input": {
        "raw_data_uri": os.environ.get("RAW_DATA_URI", ""),
        "features_output_uri": os.environ.get("FEATURES_OUTPUT_URI", ""),
        "model_output_uri": os.environ.get("MODEL_OUTPUT_URI", ""),
        "processing_image_uri": os.environ.get("PROCESSING_IMAGE_URI", ""),
        "training_image_uri": os.environ.get("TRAINING_IMAGE_URI", ""),
        "role_arn": os.environ.get("SAGEMAKER_ROLE_ARN", ""),
        "instance_type": os.environ.get("TRAINING_INSTANCE_TYPE", "ml.m5.xlarge"),
        "use_spot": True,
        "checkpoint_uri": os.environ.get("CHECKPOINT_URI", ""),
        "hyperparameters": {},
        "task_name": os.environ.get("TASK_NAME", "financial_recommendation_ple"),
    },
}


# ---------------------------------------------------------------------------
# Lambda handler
# ---------------------------------------------------------------------------

def handler(event: Dict[str, Any], context: Any) -> Dict[str, Any]:
    """Lambda entry point for auto-retrain evaluation."""

    action = event.get("action", "evaluate")
    force = event.get("force", False)
    config_override = event.get("config_override", {})

    # Merge config
    cfg = {**DEFAULT_CONFIG, **config_override}

    logger.info(
        "AutoRetrainTrigger: action=%s, force=%s", action, force,
    )

    # ------------------------------------------------------------------
    # Condition D: Manual trigger — skip evaluation, start immediately
    # ------------------------------------------------------------------
    if action == "manual" or force:
        reason = "manual_trigger"
        metrics_summary = {"trigger_type": "manual"}
        execution_arn = _start_pipeline(cfg, reason, metrics_summary)

        _record_audit(cfg, reason, metrics_summary, triggered=True, execution_arn=execution_arn)

        return {
            "triggered": True,
            "reason": reason,
            "execution_arn": execution_arn,
            "metrics_summary": metrics_summary,
        }

    # ------------------------------------------------------------------
    # Evaluate conditions A, B, C
    # ------------------------------------------------------------------
    metrics_summary: Dict[str, Any] = {}
    triggered = False
    reason = "no_retrain_needed"

    # --- Condition A: PSI critical threshold ---
    drift_result = _check_drift_report(cfg)
    metrics_summary["drift"] = drift_result

    if drift_result.get("psi_critical_exceeded"):
        triggered = True
        reason = "psi_critical_exceeded"
        logger.warning(
            "Retrain triggered: PSI critical exceeded for features: %s",
            drift_result.get("critical_features"),
        )

    # --- Condition B: Champion-Challenger ---
    if not triggered:
        cc_result = _check_champion_challenger(cfg)
        metrics_summary["champion_challenger"] = cc_result

        if cc_result.get("challenger_significantly_better"):
            triggered = True
            reason = "challenger_significantly_better"
            logger.info(
                "Retrain triggered: challenger is significantly better (lift=%.4f)",
                cc_result.get("lift", 0),
            )

    # --- Condition C: Staleness ---
    if not triggered:
        staleness_result = _check_staleness(cfg)
        metrics_summary["staleness"] = staleness_result

        if staleness_result.get("stale"):
            triggered = True
            reason = "staleness_exceeded"
            logger.info(
                "Retrain triggered: last training %d days ago (threshold: %d)",
                staleness_result.get("days_since_last_train", -1),
                cfg["staleness_days"],
            )

    # ------------------------------------------------------------------
    # Act on evaluation result
    # ------------------------------------------------------------------
    execution_arn = None

    if triggered:
        execution_arn = _start_pipeline(cfg, reason, metrics_summary)
        logger.info("Pipeline started: %s (reason=%s)", execution_arn, reason)
    else:
        logger.info("No retrain needed. Skipping pipeline start.")

    _record_audit(cfg, reason, metrics_summary, triggered=triggered, execution_arn=execution_arn)

    return {
        "triggered": triggered,
        "reason": reason,
        "execution_arn": execution_arn,
        "metrics_summary": metrics_summary,
    }


# ---------------------------------------------------------------------------
# Condition A: Drift report check
# ---------------------------------------------------------------------------

def _check_drift_report(cfg: Dict[str, Any]) -> Dict[str, Any]:
    """Read the latest drift report JSON from S3 and check PSI values."""
    try:
        import boto3

        s3 = boto3.client("s3", region_name=cfg["region"])
        bucket = cfg["drift_report_bucket"]
        prefix = cfg["drift_report_prefix"]

        # List recent drift report files (sorted by key descending)
        response = s3.list_objects_v2(
            Bucket=bucket,
            Prefix=prefix,
            MaxKeys=10,
        )

        contents = response.get("Contents", [])
        if not contents:
            logger.info("No drift reports found in s3://%s/%s", bucket, prefix)
            return {"psi_critical_exceeded": False, "reason": "no_reports_found"}

        # Pick the latest file by LastModified
        latest = max(contents, key=lambda o: o["LastModified"])
        latest_key = latest["Key"]

        obj = s3.get_object(Bucket=bucket, Key=latest_key)
        report = json.loads(obj["Body"].read().decode("utf-8"))

        psi_scores = report.get("psi_scores", {})
        threshold = cfg["psi_critical_threshold"]

        critical_features = [
            feat for feat, psi in psi_scores.items()
            if isinstance(psi, (int, float)) and psi >= threshold
        ]

        max_psi = max(
            (v for v in psi_scores.values() if isinstance(v, (int, float))),
            default=0.0,
        )

        return {
            "psi_critical_exceeded": len(critical_features) > 0,
            "critical_features": critical_features,
            "max_psi": max_psi,
            "threshold": threshold,
            "report_key": latest_key,
        }

    except Exception as exc:
        logger.warning("Drift report check failed: %s", exc, exc_info=True)
        return {"psi_critical_exceeded": False, "error": str(exc)}


# ---------------------------------------------------------------------------
# Condition B: Champion-Challenger check
# ---------------------------------------------------------------------------

def _check_champion_challenger(cfg: Dict[str, Any]) -> Dict[str, Any]:
    """Query DynamoDB prediction log to compare champion vs challenger."""
    champion = cfg.get("champion_version", "")
    challenger = cfg.get("challenger_version", "")

    if not champion or not challenger:
        return {
            "challenger_significantly_better": False,
            "reason": "versions_not_configured",
        }

    try:
        import boto3
        from boto3.dynamodb.conditions import Key

        dynamodb = boto3.resource("dynamodb", region_name=cfg["region"])
        table = dynamodb.Table(cfg["monitor_table"])
        task_name = cfg["task_name"]
        lift_threshold = cfg.get("challenger_lift_threshold", 0.02)

        cutoff = (datetime.now(timezone.utc) - timedelta(days=7)).isoformat()

        champ_scores = _fetch_version_scores(table, champion, task_name, cutoff)
        chall_scores = _fetch_version_scores(table, challenger, task_name, cutoff)

        if len(champ_scores) < 100 or len(chall_scores) < 100:
            return {
                "challenger_significantly_better": False,
                "reason": "insufficient_samples",
                "champion_samples": len(champ_scores),
                "challenger_samples": len(chall_scores),
            }

        champ_mean = sum(champ_scores) / len(champ_scores)
        chall_mean = sum(chall_scores) / len(chall_scores)
        lift = (chall_mean - champ_mean) / champ_mean if champ_mean > 0 else 0.0

        significantly_better = lift > lift_threshold

        return {
            "challenger_significantly_better": significantly_better,
            "champion_mean": round(champ_mean, 6),
            "challenger_mean": round(chall_mean, 6),
            "lift": round(lift, 6),
            "lift_threshold": lift_threshold,
            "champion_samples": len(champ_scores),
            "challenger_samples": len(chall_scores),
        }

    except Exception as exc:
        logger.warning("Champion-Challenger check failed: %s", exc, exc_info=True)
        return {"challenger_significantly_better": False, "error": str(exc)}


def _fetch_version_scores(table, version: str, task_name: str, cutoff: str):
    """Fetch prediction scores for a version from DynamoDB GSI."""
    from boto3.dynamodb.conditions import Key

    scores = []
    response = table.query(
        IndexName="version-timestamp-index",
        KeyConditionExpression=(
            Key("version").eq(version) & Key("timestamp").gte(cutoff)
        ),
        ProjectionExpression="predictions",
    )

    for item in response.get("Items", []):
        preds = item.get("predictions", {})
        if task_name in preds:
            try:
                scores.append(float(preds[task_name]))
            except (ValueError, TypeError):
                pass

    while "LastEvaluatedKey" in response:
        response = table.query(
            IndexName="version-timestamp-index",
            KeyConditionExpression=(
                Key("version").eq(version) & Key("timestamp").gte(cutoff)
            ),
            ProjectionExpression="predictions",
            ExclusiveStartKey=response["LastEvaluatedKey"],
        )
        for item in response.get("Items", []):
            preds = item.get("predictions", {})
            if task_name in preds:
                try:
                    scores.append(float(preds[task_name]))
                except (ValueError, TypeError):
                    pass

    return scores


# ---------------------------------------------------------------------------
# Condition C: Staleness check
# ---------------------------------------------------------------------------

def _check_staleness(cfg: Dict[str, Any]) -> Dict[str, Any]:
    """Check if the last training was more than N days ago."""
    try:
        import boto3

        dynamodb = boto3.resource("dynamodb", region_name=cfg["region"])
        table = dynamodb.Table(cfg["retrain_log_table"])

        # Query for the most recent retrain record
        response = table.query(
            KeyConditionExpression=boto3.dynamodb.conditions.Key("pk").eq("retrain"),
            ScanIndexForward=False,
            Limit=1,
        )

        items = response.get("Items", [])
        if not items:
            return {
                "stale": True,
                "days_since_last_train": -1,
                "reason": "no_training_record_found",
            }

        last_train_ts = items[0].get("timestamp", "")
        if not last_train_ts:
            return {
                "stale": True,
                "days_since_last_train": -1,
                "reason": "missing_timestamp",
            }

        last_dt = datetime.fromisoformat(last_train_ts.replace("Z", "+00:00"))
        days_ago = (datetime.now(timezone.utc) - last_dt).days
        threshold = cfg["staleness_days"]

        return {
            "stale": days_ago >= threshold,
            "days_since_last_train": days_ago,
            "threshold_days": threshold,
            "last_train_timestamp": last_train_ts,
        }

    except Exception as exc:
        logger.warning("Staleness check failed: %s", exc, exc_info=True)
        return {"stale": False, "error": str(exc)}


# ---------------------------------------------------------------------------
# Start Step Functions pipeline
# ---------------------------------------------------------------------------

def _start_pipeline(
    cfg: Dict[str, Any],
    reason: str,
    metrics_summary: Dict[str, Any],
) -> Optional[str]:
    """Start a Step Functions execution for the training pipeline."""
    try:
        import boto3

        sfn = boto3.client("stepfunctions", region_name=cfg["region"])

        execution_name = f"retrain-{reason}-{uuid.uuid4().hex[:8]}"

        # Build pipeline input from config
        pipeline_input = dict(cfg.get("pipeline_input", {}))
        pipeline_input["retrain_reason"] = reason
        pipeline_input["retrain_trigger_time"] = datetime.now(timezone.utc).isoformat()
        pipeline_input["metrics_snapshot"] = _sanitize_for_json(metrics_summary)

        response = sfn.start_execution(
            stateMachineArn=cfg["state_machine_arn"],
            name=execution_name,
            input=json.dumps(pipeline_input, default=str),
        )

        execution_arn = response.get("executionArn", "")
        logger.info("Step Functions execution started: %s", execution_arn)
        return execution_arn

    except Exception as exc:
        logger.error("Failed to start pipeline: %s", exc, exc_info=True)
        return None


# ---------------------------------------------------------------------------
# Audit logging
# ---------------------------------------------------------------------------

def _record_audit(
    cfg: Dict[str, Any],
    reason: str,
    metrics_summary: Dict[str, Any],
    triggered: bool,
    execution_arn: Optional[str],
) -> None:
    """Record the retrain evaluation decision to DynamoDB audit table."""
    try:
        import boto3
        from decimal import Decimal

        dynamodb = boto3.resource("dynamodb", region_name=cfg["region"])
        table_name = f"{cfg['audit_table_prefix']}-retrain"
        table = dynamodb.Table(table_name)

        event_id = str(uuid.uuid4())
        now = datetime.now(timezone.utc).isoformat()

        record = {
            "pk": f"retrain#{event_id}",
            "sk": now,
            "event_id": event_id,
            "timestamp": now,
            "triggered": triggered,
            "reason": reason,
            "execution_arn": execution_arn or "N/A",
            "metrics_summary": _decimal_safe(metrics_summary),
        }

        table.put_item(Item=record)
        logger.info("Audit record written: %s (triggered=%s)", event_id, triggered)

    except Exception as exc:
        logger.warning("Audit recording failed (non-fatal): %s", exc)

    # Also log to retrain-log table for staleness tracking
    if triggered and execution_arn:
        try:
            import boto3

            dynamodb = boto3.resource("dynamodb", region_name=cfg["region"])
            log_table = dynamodb.Table(cfg["retrain_log_table"])
            now = datetime.now(timezone.utc).isoformat()

            log_table.put_item(Item={
                "pk": "retrain",
                "sk": now,
                "timestamp": now,
                "reason": reason,
                "execution_arn": execution_arn,
            })

        except Exception as exc:
            logger.warning("Retrain log write failed (non-fatal): %s", exc)


# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------

def _decimal_safe(value: Any) -> Any:
    """Convert float values to Decimal for DynamoDB compatibility."""
    from decimal import Decimal

    if isinstance(value, float):
        return Decimal(str(value))
    if isinstance(value, dict):
        return {k: _decimal_safe(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_decimal_safe(v) for v in value]
    return value


def _sanitize_for_json(obj: Any) -> Any:
    """Ensure an object is JSON-serializable (strip non-serializable types)."""
    if isinstance(obj, dict):
        return {k: _sanitize_for_json(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_sanitize_for_json(v) for v in obj]
    if isinstance(obj, (str, int, float, bool)):
        return obj
    if obj is None:
        return obj
    return str(obj)
