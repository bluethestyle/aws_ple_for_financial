"""
Lambda handler for scheduled drift report generation.

Invoked by:
  - EventBridge scheduled rule (daily, e.g. ``rate(1 day)``)
  - Manual invocation via console or API

Workflow:
  1. Load baseline (training) data from S3
  2. Load current (recent serving) data from S3
  3. Run DriftDetector.detect_drift(baseline, current)
  4. Save drift report JSON to S3 (consumed by auto_retrain_trigger)
  5. Emit CloudWatch metrics for dashboarding and alarms

Event payload::

    {
        "baseline_bucket": "ple-data",
        "baseline_key": "training/latest/features.parquet",
        "current_bucket": "ple-data",
        "current_key": "serving/recent/features.parquet",
        "config_override": {}
    }

Returns::

    {
        "status": "success",
        "report_key": "drift-reports/drift_2026-03-18.json",
        "summary": {...}
    }
"""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timezone
from typing import Any, Dict, Optional

logger = logging.getLogger()
logger.setLevel(logging.INFO)

# ---------------------------------------------------------------------------
# Default configuration (overridable via env vars or event.config_override)
# ---------------------------------------------------------------------------

DEFAULT_CONFIG = {
    "region": os.environ.get("AWS_REGION", "ap-northeast-2"),
    "baseline_bucket": os.environ.get("BASELINE_BUCKET", "ple-data"),
    "baseline_key": os.environ.get("BASELINE_KEY", "training/latest/features.parquet"),
    "current_bucket": os.environ.get("CURRENT_BUCKET", "ple-data"),
    "current_key": os.environ.get("CURRENT_KEY", "serving/recent/features.parquet"),
    "report_bucket": os.environ.get("DRIFT_REPORT_BUCKET", "ple-monitoring"),
    "report_prefix": os.environ.get("DRIFT_REPORT_PREFIX", "drift-reports/"),
    "psi_threshold_warning": float(os.environ.get("PSI_THRESHOLD_WARNING", "0.1")),
    "psi_threshold_critical": float(os.environ.get("PSI_THRESHOLD_CRITICAL", "0.25")),
    "cloudwatch_namespace": os.environ.get("CW_NAMESPACE", "PLE/DriftMonitoring"),
    "n_bins": int(os.environ.get("PSI_N_BINS", "10")),
}


# ---------------------------------------------------------------------------
# Lambda handler
# ---------------------------------------------------------------------------

def handler(event: Dict[str, Any], context: Any) -> Dict[str, Any]:
    """Lambda entry point for drift report generation."""

    config_override = event.get("config_override", {})
    cfg = {**DEFAULT_CONFIG, **config_override}

    # Allow event-level overrides for bucket/key
    cfg["baseline_bucket"] = event.get("baseline_bucket", cfg["baseline_bucket"])
    cfg["baseline_key"] = event.get("baseline_key", cfg["baseline_key"])
    cfg["current_bucket"] = event.get("current_bucket", cfg["current_bucket"])
    cfg["current_key"] = event.get("current_key", cfg["current_key"])

    logger.info(
        "DriftReportGenerator: baseline=s3://%s/%s, current=s3://%s/%s",
        cfg["baseline_bucket"], cfg["baseline_key"],
        cfg["current_bucket"], cfg["current_key"],
    )

    try:
        # Step 1: Load baseline data from S3
        logger.info("Loading baseline data from S3...")
        baseline_df = _load_parquet_from_s3(
            cfg["baseline_bucket"], cfg["baseline_key"], cfg["region"],
        )
        logger.info("Baseline data: %d rows, %d columns", len(baseline_df), len(baseline_df.columns))

        # Step 2: Load current data from S3
        logger.info("Loading current data from S3...")
        current_df = _load_parquet_from_s3(
            cfg["current_bucket"], cfg["current_key"], cfg["region"],
        )
        logger.info("Current data: %d rows, %d columns", len(current_df), len(current_df.columns))

        # Step 3: Run drift detection
        logger.info("Running drift detection...")
        from core.monitoring.drift_detector import DriftDetector

        detector = DriftDetector(
            psi_threshold_warning=cfg["psi_threshold_warning"],
            psi_threshold_critical=cfg["psi_threshold_critical"],
            n_bins=cfg["n_bins"],
        )
        drift_result = detector.detect_drift(
            baseline_data=baseline_df,
            current_data=current_df,
        )

        # Step 4: Save drift report to S3
        # Format matches what auto_retrain_trigger._check_drift_report expects:
        #   report["psi_scores"] -> Dict[str, float]
        #   report["summary"]["critical_count"] -> int
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        report = {
            "date": today,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "psi_scores": drift_result["psi_scores"],
            "warning_features": drift_result["warning_features"],
            "critical_features": drift_result["critical_features"],
            "summary": drift_result["summary"],
            "config": {
                "psi_threshold_warning": cfg["psi_threshold_warning"],
                "psi_threshold_critical": cfg["psi_threshold_critical"],
                "n_bins": cfg["n_bins"],
                "baseline_path": f"s3://{cfg['baseline_bucket']}/{cfg['baseline_key']}",
                "current_path": f"s3://{cfg['current_bucket']}/{cfg['current_key']}",
            },
        }

        report_key = f"{cfg['report_prefix']}drift_{today}.json"
        _save_report_to_s3(
            cfg["report_bucket"], report_key, report, cfg["region"],
        )
        logger.info("Drift report saved to s3://%s/%s", cfg["report_bucket"], report_key)

        # Step 5: Emit CloudWatch metrics
        _emit_cloudwatch_metrics(cfg, drift_result)

        return {
            "status": "success",
            "report_key": report_key,
            "summary": drift_result["summary"],
        }

    except Exception as exc:
        logger.error("Drift report generation failed: %s", exc, exc_info=True)
        return {
            "status": "error",
            "error": str(exc),
        }


# ---------------------------------------------------------------------------
# S3 helpers
# ---------------------------------------------------------------------------

def _load_parquet_from_s3(
    bucket: str,
    key: str,
    region: str,
) -> "pandas.DataFrame":
    """Load a parquet file (or prefix of parquet files) from S3."""
    import boto3
    import pandas as pd
    import io

    s3 = boto3.client("s3", region_name=region)

    # Check if key points to a single file or a prefix (directory)
    if key.endswith(".parquet"):
        obj = s3.get_object(Bucket=bucket, Key=key)
        return pd.read_parquet(io.BytesIO(obj["Body"].read()))

    # Prefix mode: list and concatenate all parquet files
    response = s3.list_objects_v2(Bucket=bucket, Prefix=key)
    contents = response.get("Contents", [])
    parquet_keys = [
        o["Key"] for o in contents
        if o["Key"].endswith(".parquet")
    ]

    if not parquet_keys:
        raise FileNotFoundError(
            f"No parquet files found at s3://{bucket}/{key}"
        )

    frames = []
    for pkey in sorted(parquet_keys):
        obj = s3.get_object(Bucket=bucket, Key=pkey)
        frames.append(pd.read_parquet(io.BytesIO(obj["Body"].read())))

    return pd.concat(frames, ignore_index=True)


def _save_report_to_s3(
    bucket: str,
    key: str,
    report: Dict[str, Any],
    region: str,
) -> None:
    """Save a JSON report to S3."""
    import boto3

    s3 = boto3.client("s3", region_name=region)
    body = json.dumps(report, indent=2, default=str)
    s3.put_object(
        Bucket=bucket,
        Key=key,
        Body=body.encode("utf-8"),
        ContentType="application/json",
    )


# ---------------------------------------------------------------------------
# CloudWatch metrics
# ---------------------------------------------------------------------------

def _emit_cloudwatch_metrics(
    cfg: Dict[str, Any],
    drift_result: Dict[str, Any],
) -> None:
    """Emit drift metrics to CloudWatch for dashboarding and alarms."""
    try:
        import boto3

        cw = boto3.client("cloudwatch", region_name=cfg["region"])
        namespace = cfg["cloudwatch_namespace"]
        summary = drift_result["summary"]

        metrics = [
            {
                "MetricName": "CriticalDriftFeatureCount",
                "Value": summary["critical_count"],
                "Unit": "Count",
            },
            {
                "MetricName": "WarningDriftFeatureCount",
                "Value": summary["warning_count"],
                "Unit": "Count",
            },
            {
                "MetricName": "MaxPSI",
                "Value": summary["max_psi"],
                "Unit": "None",
            },
            {
                "MetricName": "AvgPSI",
                "Value": summary["avg_psi"],
                "Unit": "None",
            },
            {
                "MetricName": "TotalFeaturesMonitored",
                "Value": summary["total_features"],
                "Unit": "Count",
            },
        ]

        cw.put_metric_data(
            Namespace=namespace,
            MetricData=[
                {
                    "MetricName": m["MetricName"],
                    "Value": m["Value"],
                    "Unit": m["Unit"],
                    "Dimensions": [
                        {"Name": "Pipeline", "Value": "PLE-Distillation"},
                    ],
                }
                for m in metrics
            ],
        )
        logger.info("CloudWatch metrics emitted: %d metrics to %s", len(metrics), namespace)

    except Exception as exc:
        logger.warning("CloudWatch metric emission failed (non-fatal): %s", exc)
