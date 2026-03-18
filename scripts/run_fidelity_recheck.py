#!/usr/bin/env python3
"""
Post-Deployment Fidelity Re-check.

Runs as a SageMaker Processing Job (weekly or on drift detection).
Compares the deployed LGBM student predictions against the PLE teacher
on recent production data to detect fidelity degradation.

Flow:
    1. Load production prediction_log samples from DynamoDB
    2. Load corresponding features from Feature Store (or S3 snapshot)
    3. Load teacher model from ModelRegistry (PLE checkpoint)
    4. Load student models from ModelRegistry (LGBM per task)
    5. Run teacher inference on sampled features (GPU if available)
    6. Run student inference on same features
    7. Run DistillationValidator (8 metrics per task)
    8. Compare against original fidelity (from manifest)
    9. Report: save results to S3 + CloudWatch + audit_store
   10. If fidelity degraded → trigger alert (SNS) or auto-retrain

Usage (SageMaker Processing Job):
    python scripts/run_fidelity_recheck.py \
        --model-version v-latest \
        --registry-base s3://aiops-ple-financial/models/artifacts \
        --prediction-log-table ple-prediction-log \
        --output-dir /opt/ml/processing/output \
        --sample-size 10000 \
        --lookback-days 7

    # Or with explicit feature data:
    python scripts/run_fidelity_recheck.py \
        --model-version v-latest \
        --feature-data-path s3://bucket/features/latest/ \
        --output-dir /opt/ml/processing/output
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger("fidelity_recheck")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Post-deployment teacher-student fidelity re-check"
    )
    parser.add_argument(
        "--model-version", type=str, default="",
        help="Model version to check (empty = promoted/active version)",
    )
    parser.add_argument(
        "--registry-base", type=str,
        default="s3://aiops-ple-financial/models/artifacts",
    )
    parser.add_argument(
        "--prediction-log-table", type=str, default="ple-prediction-log",
    )
    parser.add_argument(
        "--feature-data-path", type=str, default="",
        help="S3 path to feature parquet (alternative to DynamoDB sampling)",
    )
    parser.add_argument(
        "--output-dir", type=str, default="/opt/ml/processing/output",
    )
    parser.add_argument(
        "--sample-size", type=int, default=10000,
        help="Number of prediction records to sample for re-check",
    )
    parser.add_argument(
        "--lookback-days", type=int, default=7,
    )
    parser.add_argument(
        "--region", type=str, default="ap-northeast-2",
    )
    parser.add_argument(
        "--sns-topic-arn", type=str, default="",
        help="SNS topic for fidelity degradation alerts",
    )
    parser.add_argument(
        "--device", type=str, default="cpu",
        help="Torch device for teacher inference (cpu or cuda)",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    logger.info("=" * 60)
    logger.info("Post-Deployment Fidelity Re-check")
    logger.info("=" * 60)

    import boto3
    import pandas as pd
    import torch

    from core.serving.model_registry import ModelRegistry
    from core.training.distillation_validator import (
        DistillationValidator,
        ValidationCriteria,
    )

    # ------------------------------------------------------------------
    # Step 1: Determine model version
    # ------------------------------------------------------------------
    registry = ModelRegistry(
        s3_base=args.registry_base,
        region=args.region,
    )

    if args.model_version:
        version = args.model_version
    else:
        version = registry.get_promoted()
        if not version:
            version = registry.get_latest()
    if not version:
        logger.error("No model version found")
        sys.exit(1)

    logger.info("Checking fidelity for version: %s", version)
    manifest = registry.load_manifest(version)
    original_fidelity = manifest.fidelity_summary
    task_list = manifest.student_tasks
    logger.info("Tasks: %s", task_list)
    logger.info(
        "Original fidelity: passed=%s, failed=%s",
        original_fidelity.get("passed", "?"),
        original_fidelity.get("failed", "?"),
    )

    # ------------------------------------------------------------------
    # Step 2: Load feature data (S3 parquet or DynamoDB sampling)
    # ------------------------------------------------------------------
    if args.feature_data_path:
        logger.info("Loading features from %s", args.feature_data_path)
        df = pd.read_parquet(args.feature_data_path)
        if len(df) > args.sample_size:
            df = df.sample(n=args.sample_size, random_state=42)
        logger.info("Loaded %d samples", len(df))
    else:
        logger.info(
            "Sampling %d records from DynamoDB prediction log (last %d days)",
            args.sample_size, args.lookback_days,
        )
        df = _sample_from_prediction_log(
            table_name=args.prediction_log_table,
            version=version,
            lookback_days=args.lookback_days,
            sample_size=args.sample_size,
            region=args.region,
        )
        if df is None or len(df) == 0:
            logger.error("No prediction log samples found")
            sys.exit(1)
        logger.info("Sampled %d records", len(df))

    # Separate features and labels
    label_cols = [c for c in df.columns if c.startswith("label_")]
    id_cols = [c for c in df.columns if c in ("user_id", "customer_id")]
    feature_cols = [c for c in df.columns if c not in label_cols + id_cols]
    features = df[feature_cols].values.astype(np.float32)

    hard_labels: Dict[str, np.ndarray] = {}
    for col in label_cols:
        task_name = col.replace("label_", "")
        hard_labels[task_name] = df[col].values

    logger.info("Features: %d dims, Labels: %d tasks", features.shape[1], len(hard_labels))

    # ------------------------------------------------------------------
    # Step 3: Load teacher model
    # ------------------------------------------------------------------
    logger.info("Loading teacher model (PLE)...")
    t0 = time.perf_counter()
    teacher_model, ple_config = registry.load_teacher(version, device=args.device)
    logger.info("Teacher loaded in %.1fs", time.perf_counter() - t0)

    # ------------------------------------------------------------------
    # Step 4: Load student models
    # ------------------------------------------------------------------
    logger.info("Loading student models (LGBM)...")
    students: Dict[str, Any] = {}
    for task_name in task_list:
        try:
            students[task_name] = registry.load_student(version, task_name)
            logger.info("  Loaded student: %s", task_name)
        except FileNotFoundError:
            logger.warning("  Student not found: %s (skipping)", task_name)

    # ------------------------------------------------------------------
    # Step 5: Teacher inference
    # ------------------------------------------------------------------
    logger.info("Running teacher inference...")
    t0 = time.perf_counter()
    teacher_preds: Dict[str, np.ndarray] = {}

    features_tensor = torch.tensor(features, dtype=torch.float32).to(args.device)
    from core.model.ple.model import PLEInput

    batch_size = 512
    all_preds: Dict[str, List[np.ndarray]] = {t: [] for t in task_list}

    teacher_model.eval()
    with torch.no_grad():
        for start in range(0, len(features_tensor), batch_size):
            end = min(start + batch_size, len(features_tensor))
            batch = features_tensor[start:end]
            inputs = PLEInput(features=batch)
            outputs = teacher_model(inputs, compute_loss=False)

            for task_name in task_list:
                if task_name in outputs.predictions:
                    pred = outputs.predictions[task_name].cpu().numpy()
                    all_preds[task_name].append(pred)

    for task_name in task_list:
        if all_preds[task_name]:
            teacher_preds[task_name] = np.concatenate(all_preds[task_name], axis=0)

    teacher_time = time.perf_counter() - t0
    logger.info("Teacher inference: %.1fs (%d samples)", teacher_time, len(features))

    # ------------------------------------------------------------------
    # Step 6: Student inference
    # ------------------------------------------------------------------
    logger.info("Running student inference...")
    t0 = time.perf_counter()
    student_preds: Dict[str, np.ndarray] = {}

    for task_name, student_model in students.items():
        # Load selected features if available
        try:
            sel = registry.get_selected_features(version, task_name)
            indices = sel.get("indices", [])
            if indices:
                task_features = features[:, indices]
            else:
                task_features = features
        except FileNotFoundError:
            task_features = features

        student_preds[task_name] = student_model.predict(task_features)

    student_time = time.perf_counter() - t0
    logger.info("Student inference: %.1fs", student_time)

    # ------------------------------------------------------------------
    # Step 7: Run DistillationValidator
    # ------------------------------------------------------------------
    logger.info("Running fidelity validation (8 metrics per task)...")
    validator = DistillationValidator(criteria=ValidationCriteria())
    results = []

    for task_name in task_list:
        if task_name not in teacher_preds or task_name not in student_preds:
            logger.warning("Skipping %s (missing predictions)", task_name)
            continue

        # Determine task type from manifest metadata
        task_type = "binary"  # default
        details = original_fidelity.get("details", {}).get(task_name, {})
        if details:
            task_type = details.get("task_type", "binary")

        labels = hard_labels.get(task_name)

        result = validator.validate_task(
            task_name=task_name,
            task_type=task_type,
            teacher_preds=teacher_preds[task_name],
            student_preds=student_preds[task_name],
            labels=labels,
        )
        results.append(result)

        status = "PASS" if result.passed else "FAIL"
        logger.info(
            "  [%s] %s — %s",
            status, task_name,
            {k: round(v, 4) for k, v in result.metrics.items()},
        )

    # ------------------------------------------------------------------
    # Step 8: Compare against original fidelity
    # ------------------------------------------------------------------
    logger.info("Comparing against training-time fidelity...")
    degraded_tasks: List[Dict[str, Any]] = []

    for result in results:
        task_name = result.task_name
        original_detail = original_fidelity.get("details", {}).get(task_name, {})
        original_metrics = original_detail.get("metrics", {})

        if not original_metrics:
            continue

        for metric_name, current_value in result.metrics.items():
            original_value = original_metrics.get(metric_name)
            if original_value is None:
                continue

            # Check for significant degradation (>50% worse)
            if metric_name in ("auc_gap", "jsd", "calibration_gap"):
                # Lower is better — degradation = current > original * 1.5
                if current_value > original_value * 1.5 + 0.01:
                    degraded_tasks.append({
                        "task": task_name,
                        "metric": metric_name,
                        "original": round(original_value, 4),
                        "current": round(current_value, 4),
                        "degradation_pct": round(
                            (current_value - original_value) / max(original_value, 0.001) * 100, 1
                        ),
                    })
            else:
                # Higher is better — degradation = current < original * 0.9
                if current_value < original_value * 0.9 - 0.01:
                    degraded_tasks.append({
                        "task": task_name,
                        "metric": metric_name,
                        "original": round(original_value, 4),
                        "current": round(current_value, 4),
                        "degradation_pct": round(
                            (original_value - current_value) / max(original_value, 0.001) * 100, 1
                        ),
                    })

    # ------------------------------------------------------------------
    # Step 9: Report
    # ------------------------------------------------------------------
    passed_count = sum(1 for r in results if r.passed)
    failed_count = len(results) - passed_count
    overall_passed = failed_count == 0 and len(degraded_tasks) == 0

    report = {
        "version": version,
        "checked_at": datetime.now(timezone.utc).isoformat(),
        "sample_size": len(features),
        "lookback_days": args.lookback_days,
        "teacher_inference_time_s": round(teacher_time, 2),
        "student_inference_time_s": round(student_time, 2),
        "speed_ratio": round(student_time / max(teacher_time, 0.001), 4),
        "overall_passed": overall_passed,
        "fidelity_summary": {
            "passed": passed_count,
            "failed": failed_count,
        },
        "degraded_metrics": degraded_tasks,
        "per_task": {
            r.task_name: {
                "passed": r.passed,
                "metrics": {k: round(v, 4) for k, v in r.metrics.items()},
                "failures": r.failures,
            }
            for r in results
        },
    }

    # Save to output dir
    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    report_path = output_path / "fidelity_recheck_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2, default=str)
    logger.info("Report saved to %s", report_path)

    # CloudWatch metrics
    _emit_cloudwatch_metrics(args.region, version, results, degraded_tasks)

    # Audit store
    _log_to_audit_store(args.region, version, report)

    # ------------------------------------------------------------------
    # Step 10: Alert if degraded
    # ------------------------------------------------------------------
    if degraded_tasks:
        logger.warning(
            "FIDELITY DEGRADATION DETECTED: %d metrics degraded across %d tasks",
            len(degraded_tasks),
            len(set(d["task"] for d in degraded_tasks)),
        )
        for d in degraded_tasks:
            logger.warning(
                "  %s.%s: %.4f → %.4f (%.1f%% worse)",
                d["task"], d["metric"], d["original"], d["current"], d["degradation_pct"],
            )

        if args.sns_topic_arn:
            _send_alert(args.region, args.sns_topic_arn, version, degraded_tasks)

    if not overall_passed:
        logger.error("Fidelity re-check FAILED — consider retraining")
        # Don't sys.exit(1) — this is a monitoring job, not a gate
    else:
        logger.info("Fidelity re-check PASSED — all metrics within bounds")

    logger.info("=" * 60)
    logger.info("Done.")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _sample_from_prediction_log(
    table_name: str,
    version: str,
    lookback_days: int,
    sample_size: int,
    region: str,
) -> Optional["pd.DataFrame"]:
    """Sample recent prediction records from DynamoDB."""
    try:
        import boto3
        import pandas as pd
        from boto3.dynamodb.conditions import Key

        dynamodb = boto3.resource("dynamodb", region_name=region)
        table = dynamodb.Table(table_name)

        cutoff = (
            datetime.now(timezone.utc) - timedelta(days=lookback_days)
        ).isoformat()

        response = table.query(
            IndexName="version-timestamp-index",
            KeyConditionExpression=(
                Key("version").eq(version)
                & Key("timestamp").gte(cutoff)
            ),
            Limit=sample_size,
        )

        items = response.get("Items", [])
        while "LastEvaluatedKey" in response and len(items) < sample_size:
            response = table.query(
                IndexName="version-timestamp-index",
                KeyConditionExpression=(
                    Key("version").eq(version)
                    & Key("timestamp").gte(cutoff)
                ),
                Limit=sample_size - len(items),
                ExclusiveStartKey=response["LastEvaluatedKey"],
            )
            items.extend(response.get("Items", []))

        if not items:
            return None

        return pd.DataFrame(items)

    except Exception as e:
        logger.warning("Failed to sample from prediction log: %s", e)
        return None


def _emit_cloudwatch_metrics(
    region: str,
    version: str,
    results: list,
    degraded: list,
) -> None:
    """Emit fidelity re-check metrics to CloudWatch."""
    try:
        import boto3

        cw = boto3.client("cloudwatch", region_name=region)
        metric_data = [
            {
                "MetricName": "FidelityRecheckPassedTasks",
                "Dimensions": [{"Name": "Version", "Value": version}],
                "Value": sum(1 for r in results if r.passed),
                "Unit": "Count",
            },
            {
                "MetricName": "FidelityRecheckFailedTasks",
                "Dimensions": [{"Name": "Version", "Value": version}],
                "Value": sum(1 for r in results if not r.passed),
                "Unit": "Count",
            },
            {
                "MetricName": "FidelityDegradedMetrics",
                "Dimensions": [{"Name": "Version", "Value": version}],
                "Value": len(degraded),
                "Unit": "Count",
            },
        ]
        cw.put_metric_data(Namespace="PLE/Fidelity", MetricData=metric_data)
    except Exception:
        logger.debug("CloudWatch emit failed (non-fatal)", exc_info=True)


def _log_to_audit_store(region: str, version: str, report: dict) -> None:
    """Log fidelity re-check to audit store."""
    try:
        from core.compliance.audit_store import get_audit_store

        store = get_audit_store()
        store.log_distillation(
            task_name="fidelity_recheck",
            teacher_version=version,
            student_version=version,
            metrics={
                "overall_passed": report["overall_passed"],
                "passed": report["fidelity_summary"]["passed"],
                "failed": report["fidelity_summary"]["failed"],
                "degraded_count": len(report["degraded_metrics"]),
                "sample_size": report["sample_size"],
            },
        )
    except Exception:
        logger.debug("Audit store log failed (non-fatal)", exc_info=True)


def _send_alert(
    region: str, topic_arn: str, version: str, degraded: list,
) -> None:
    """Send SNS alert for fidelity degradation."""
    try:
        import boto3

        sns = boto3.client("sns", region_name=region)
        subject = f"[PLE Fidelity Alert] {version}: {len(degraded)} metrics degraded"
        message_lines = [
            f"Model version: {version}",
            f"Degraded metrics: {len(degraded)}",
            "",
        ]
        for d in degraded:
            message_lines.append(
                f"  {d['task']}.{d['metric']}: "
                f"{d['original']} → {d['current']} ({d['degradation_pct']}% worse)"
            )
        message_lines.append("")
        message_lines.append("Action: Consider triggering model retraining.")

        sns.publish(
            TopicArn=topic_arn,
            Subject=subject[:100],
            Message="\n".join(message_lines),
        )
        logger.info("Alert sent to %s", topic_arn)
    except Exception:
        logger.warning("SNS alert failed (non-fatal)", exc_info=True)


if __name__ == "__main__":
    main()
