"""
Lambda handler for scheduled fairness evaluation.

Invoked by:
  - EventBridge scheduled rule (weekly, e.g. rate(7 days))
  - Manual invocation via console or API

Workflow:
  1. Query DynamoDB ``prediction_log`` for the last 7 days of predictions
  2. For each protected attribute (age_group, gender, income_tier, ...),
     run ``FairnessMonitor.evaluate_fairness()``
  3. Violations trigger auto-escalation via ``IncidentReporter``
  4. Results are saved to S3 as JSON + emitted as CloudWatch metrics
  5. Audit trail event logged for AIA-004 compliance verification

Event payload::

    {
        "action": "evaluate",           // "evaluate" or "manual"
        "lookback_days": 7,             // optional, default 7
        "config_override": {}           // optional overrides
    }

Returns::

    {
        "evaluated_at": "2026-03-18T...",
        "attributes_evaluated": 5,
        "violations_found": 2,
        "results": {...},
        "s3_key": "fairness-reports/2026-03-18/fairness_report.json"
    }
"""

from __future__ import annotations

import json
import logging
import os
import uuid
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger()
logger.setLevel(logging.INFO)

# ---------------------------------------------------------------------------
# Default configuration (overridable via env vars or event.config_override)
# ---------------------------------------------------------------------------

DEFAULT_CONFIG = {
    # region: inherited from AWS_DEFAULT_REGION / AWS_REGION (boto3 resolves
    # from env, shared credentials, or instance metadata when None).
    "region": os.environ.get("AWS_DEFAULT_REGION") or os.environ.get("AWS_REGION"),
    "prediction_log_table": os.environ.get("PREDICTION_LOG_TABLE", "ple-prediction-log"),
    "result_bucket": os.environ.get("FAIRNESS_RESULT_BUCKET", "ple-monitoring"),
    "result_prefix": os.environ.get("FAIRNESS_RESULT_PREFIX", "fairness-reports"),
    "audit_table_prefix": os.environ.get("AUDIT_TABLE_PREFIX", "ple-audit"),
    "lookback_days": int(os.environ.get("LOOKBACK_DAYS", "7")),
    "cloudwatch_namespace": os.environ.get("CW_NAMESPACE", "PLE/Fairness"),
    # S7 archive path: every evaluate_fairness result is persisted here
    # for the PromotionGate MetadataAggregator + governance reporter.
    # Falls through to pipeline.yaml::monitoring.fairness.archive_parquet_path
    # when the monitor is constructed with that config attached; the env
    # var here is for Lambda-only deployments where pipeline.yaml is not
    # mounted.
    "archive_parquet_path": os.environ.get("FAIRNESS_ARCHIVE_PARQUET_PATH"),
    # Protected attribute group pairs for evaluation
    # Each attribute maps to a list of (privileged, unprivileged) tuples
    "group_pairs_by_attribute": {
        "age_group": [
            ("30s", "20s"),
            ("30s", "60s"),
            ("40s", "20s"),
        ],
        "gender": [
            ("M", "F"),
        ],
        "income_tier": [
            ("high", "low"),
            ("high", "mid"),
        ],
        "region_type": [
            ("urban", "rural"),
        ],
        "life_stage": [
            ("established", "early_career"),
            ("established", "retired"),
        ],
    },
    # FairnessMonitor thresholds (default: use class defaults)
    "thresholds": {},
}


# ---------------------------------------------------------------------------
# Lambda handler
# ---------------------------------------------------------------------------

def handler(event: Dict[str, Any], context: Any) -> Dict[str, Any]:
    """Lambda entry point for weekly fairness evaluation."""

    config_override = event.get("config_override", {})
    cfg = {**DEFAULT_CONFIG, **config_override}

    lookback_days = event.get("lookback_days", cfg["lookback_days"])
    now = datetime.now(timezone.utc)

    logger.info(
        "FairnessEvaluation: lookback_days=%d, attributes=%d",
        lookback_days,
        len(cfg["group_pairs_by_attribute"]),
    )

    # ------------------------------------------------------------------
    # Step 1: Query prediction log from DynamoDB (last N days)
    # ------------------------------------------------------------------
    recommendations = _fetch_prediction_log(cfg, lookback_days)

    if not recommendations:
        logger.warning("No prediction data found for the last %d days", lookback_days)
        return {
            "evaluated_at": now.isoformat(),
            "attributes_evaluated": 0,
            "violations_found": 0,
            "results": {},
            "message": "no_prediction_data",
        }

    logger.info("Fetched %d prediction records", len(recommendations))

    # ------------------------------------------------------------------
    # Step 2: Run FairnessMonitor.evaluate_fairness() per attribute
    # ------------------------------------------------------------------
    from core.monitoring.fairness_monitor import FairnessMonitor

    monitor_config: Dict[str, Any] = {}
    if cfg.get("archive_parquet_path"):
        monitor_config["monitoring"] = {
            "fairness": {"archive_parquet_path": cfg["archive_parquet_path"]}
        }
    monitor = FairnessMonitor(
        thresholds=cfg.get("thresholds") or None,
        protected_attributes=list(cfg["group_pairs_by_attribute"].keys()),
        auto_incident=False,  # We handle incidents explicitly below
        config=monitor_config or None,
    )

    all_results: Dict[str, Any] = {}
    total_violations = 0

    group_pairs_by_attribute = cfg["group_pairs_by_attribute"]

    for attribute, group_pairs in group_pairs_by_attribute.items():
        metrics = monitor.evaluate_fairness(
            recommendations=recommendations,
            attribute=attribute,
            group_pairs=group_pairs,
        )

        result_dict = monitor.to_dict(metrics)
        all_results[attribute] = result_dict

        # Persist to fairness archive (S7). When archive_parquet_path is
        # configured, every measurement flushes to Parquet for the
        # PromotionGate MetadataAggregator + governance reporter to read.
        monitor.archive_metrics(
            metrics,
            recorded_at=now.isoformat(),
            context={"lookback_days": lookback_days, "source": "lambda"},
        )

        if not metrics.is_fair:
            total_violations += len(metrics.violations)
            logger.warning(
                "Fairness violation [%s]: %s",
                attribute,
                "; ".join(metrics.violations),
            )

            # Step 3: Auto-escalation via IncidentReporter
            _escalate_violation(metrics)

    # ------------------------------------------------------------------
    # Step 4: Save results to S3 + emit CloudWatch metrics
    # ------------------------------------------------------------------
    report = {
        "evaluated_at": now.isoformat(),
        "lookback_days": lookback_days,
        "prediction_count": len(recommendations),
        "attributes_evaluated": len(all_results),
        "violations_found": total_violations,
        "results": all_results,
    }

    s3_key = _save_report_to_s3(cfg, report, now)
    _emit_cloudwatch_metrics(cfg, all_results, now)

    # ------------------------------------------------------------------
    # Step 5: Audit trail (AIA-004 compliance)
    # ------------------------------------------------------------------
    _record_audit(cfg, report)

    return {
        "evaluated_at": now.isoformat(),
        "attributes_evaluated": len(all_results),
        "violations_found": total_violations,
        "results": all_results,
        "s3_key": s3_key,
    }


# ---------------------------------------------------------------------------
# Step 1: Fetch prediction log
# ---------------------------------------------------------------------------

def _fetch_prediction_log(
    cfg: Dict[str, Any],
    lookback_days: int,
) -> List[Dict[str, Any]]:
    """Query DynamoDB prediction_log for recent prediction records."""
    try:
        import boto3
        from boto3.dynamodb.conditions import Key

        dynamodb = boto3.resource("dynamodb", region_name=cfg["region"])
        table = dynamodb.Table(cfg["prediction_log_table"])

        cutoff = (datetime.now(timezone.utc) - timedelta(days=lookback_days)).isoformat()

        # Scan with filter (prediction_log may use different pk schemes)
        # Use a GSI on timestamp if available; fall back to scan
        recommendations: List[Dict[str, Any]] = []

        scan_kwargs = {
            "FilterExpression": boto3.dynamodb.conditions.Attr("timestamp").gte(cutoff),
            "ProjectionExpression": (
                "customer_id, item_id, recommended, actual_positive, "
                "age_group, gender, income_tier, region_type, life_stage, "
                "#ts, predictions"
            ),
            "ExpressionAttributeNames": {"#ts": "timestamp"},
        }

        response = table.scan(**scan_kwargs)
        recommendations.extend(_convert_items(response.get("Items", [])))

        while "LastEvaluatedKey" in response:
            scan_kwargs["ExclusiveStartKey"] = response["LastEvaluatedKey"]
            response = table.scan(**scan_kwargs)
            recommendations.extend(_convert_items(response.get("Items", [])))

        return recommendations

    except Exception as exc:
        logger.error("Failed to fetch prediction log: %s", exc, exc_info=True)
        return []


def _convert_items(items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Convert DynamoDB Decimal values back to native Python types."""
    converted = []
    for item in items:
        record = {}
        for k, v in item.items():
            if isinstance(v, Decimal):
                record[k] = float(v) if v % 1 else int(v)
            else:
                record[k] = v
        # Ensure 'recommended' is boolean
        if "recommended" in record:
            record["recommended"] = bool(record["recommended"])
        if "actual_positive" in record:
            record["actual_positive"] = bool(record["actual_positive"])
        converted.append(record)
    return converted


# ---------------------------------------------------------------------------
# Step 3: Escalate violation via IncidentReporter
# ---------------------------------------------------------------------------

def _escalate_violation(metrics) -> None:
    """Create an incident for a fairness violation via IncidentReporter."""
    try:
        from core.monitoring.incident_reporter import IncidentReporter

        reporter = IncidentReporter()
        reporter.create_incident(
            event_type="fairness_violation",
            metrics={
                "attribute": metrics.attribute,
                "di_value": metrics.disparate_impact,
                "spd_value": metrics.statistical_parity_diff,
                "eod_value": metrics.equal_opportunity_diff,
                "violations": metrics.violations,
            },
            source_module="fairness_evaluation_lambda",
            description=(
                f"Scheduled fairness evaluation violation on '{metrics.attribute}': "
                + "; ".join(metrics.violations)
            ),
        )
        logger.info(
            "Incident created for fairness violation on '%s'",
            metrics.attribute,
        )
    except Exception as exc:
        logger.warning("Incident escalation failed: %s", exc)


# ---------------------------------------------------------------------------
# Step 4a: Save report to S3
# ---------------------------------------------------------------------------

def _save_report_to_s3(
    cfg: Dict[str, Any],
    report: Dict[str, Any],
    now: datetime,
) -> Optional[str]:
    """Persist the fairness report as a JSON file in S3."""
    try:
        import boto3

        s3 = boto3.client("s3", region_name=cfg["region"])
        bucket = cfg["result_bucket"]
        prefix = cfg["result_prefix"].strip("/")
        date_str = now.strftime("%Y-%m-%d")
        s3_key = f"{prefix}/{date_str}/fairness_report.json"

        s3.put_object(
            Bucket=bucket,
            Key=s3_key,
            Body=json.dumps(report, ensure_ascii=False, indent=2, default=str).encode("utf-8"),
            ContentType="application/json",
        )
        logger.info("Fairness report saved to s3://%s/%s", bucket, s3_key)
        return s3_key

    except Exception as exc:
        logger.warning("Failed to save report to S3: %s", exc)
        return None


# ---------------------------------------------------------------------------
# Step 4b: Emit CloudWatch metrics
# ---------------------------------------------------------------------------

def _emit_cloudwatch_metrics(
    cfg: Dict[str, Any],
    all_results: Dict[str, Any],
    now: datetime,
) -> None:
    """Emit per-attribute fairness metrics to CloudWatch."""
    try:
        import boto3

        cw = boto3.client("cloudwatch", region_name=cfg["region"])
        namespace = cfg["cloudwatch_namespace"]

        metric_data = []
        for attribute, result in all_results.items():
            metric_data.extend([
                {
                    "MetricName": "DisparateImpact",
                    "Dimensions": [{"Name": "Attribute", "Value": attribute}],
                    "Timestamp": now,
                    "Value": float(result.get("disparate_impact", 1.0)),
                    "Unit": "None",
                },
                {
                    "MetricName": "StatisticalParityDiff",
                    "Dimensions": [{"Name": "Attribute", "Value": attribute}],
                    "Timestamp": now,
                    "Value": abs(float(result.get("statistical_parity_diff", 0.0))),
                    "Unit": "None",
                },
                {
                    "MetricName": "EqualOpportunityDiff",
                    "Dimensions": [{"Name": "Attribute", "Value": attribute}],
                    "Timestamp": now,
                    "Value": abs(float(result.get("equal_opportunity_diff", 0.0))),
                    "Unit": "None",
                },
                {
                    "MetricName": "ViolationCount",
                    "Dimensions": [{"Name": "Attribute", "Value": attribute}],
                    "Timestamp": now,
                    "Value": float(len(result.get("violations", []))),
                    "Unit": "Count",
                },
            ])

        # CloudWatch accepts up to 1000 metrics per PutMetricData call
        for i in range(0, len(metric_data), 25):
            batch = metric_data[i:i + 25]
            cw.put_metric_data(Namespace=namespace, MetricData=batch)

        logger.info("Emitted %d CloudWatch metrics", len(metric_data))

    except Exception as exc:
        logger.warning("CloudWatch metric emission failed: %s", exc)


# ---------------------------------------------------------------------------
# Step 5: Audit trail (AIA-004)
# ---------------------------------------------------------------------------

def _record_audit(cfg: Dict[str, Any], report: Dict[str, Any]) -> None:
    """Log fairness evaluation event to compliance audit store (AIA-004)."""
    try:
        from core.compliance.audit_store import ComplianceAuditStore

        audit_store = ComplianceAuditStore(
            table_prefix=cfg["audit_table_prefix"],
            region=cfg["region"],
        )

        now = datetime.now(timezone.utc).isoformat()

        audit_store.log_event("embedding", {
            "pk": "fairness_evaluation",
            "sk": now,
            "evaluated_at": report["evaluated_at"],
            "lookback_days": report["lookback_days"],
            "prediction_count": report["prediction_count"],
            "attributes_evaluated": report["attributes_evaluated"],
            "violations_found": report["violations_found"],
            "attribute_results": {
                attr: {
                    "is_fair": r.get("is_fair", True),
                    "disparate_impact": r.get("disparate_impact"),
                    "violations": r.get("violations", []),
                }
                for attr, r in report["results"].items()
            },
        })

        logger.info("Audit trail recorded for fairness evaluation (AIA-004)")

    except Exception as exc:
        logger.warning("Audit trail recording failed: %s", exc)
