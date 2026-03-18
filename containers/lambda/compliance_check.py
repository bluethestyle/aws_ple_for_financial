"""
Lambda handler for scheduled regulatory compliance checks.

Invoked by:
  - EventBridge scheduled rule (quarterly, e.g. rate(90 days))
  - Manual invocation via console or API

Workflow:
  1. Build ``RegulatoryComplianceChecker(audit_store, config)``
  2. Run ``run_all_checks()``
  3. Critical failures trigger SNS notification
  4. Full results saved to S3 as JSON (governance report attachment)
  5. Audit trail event logged

Event payload::

    {
        "action": "evaluate",           // "evaluate" or "manual"
        "config_override": {},          // optional system config overrides
        "regulation": null              // optional: filter to specific regulation
    }

Returns::

    {
        "checked_at": "2026-03-18T...",
        "total": 20,
        "passed": 18,
        "failed": 2,
        "pass_rate": 0.9,
        "critical_failures": [...],
        "s3_key": "compliance-reports/2026-Q1/compliance_report.json"
    }
"""

from __future__ import annotations

import json
import logging
import math
import os
import uuid
from datetime import datetime, timezone
from decimal import Decimal
from typing import Any, Dict, List, Optional

logger = logging.getLogger()
logger.setLevel(logging.INFO)

# ---------------------------------------------------------------------------
# Default configuration (overridable via env vars or event.config_override)
# ---------------------------------------------------------------------------

DEFAULT_CONFIG = {
    "region": os.environ.get("AWS_DEFAULT_REGION", "ap-northeast-2"),
    "audit_table_prefix": os.environ.get("AUDIT_TABLE_PREFIX", "ple-audit"),
    "result_bucket": os.environ.get("COMPLIANCE_RESULT_BUCKET", "ple-monitoring"),
    "result_prefix": os.environ.get("COMPLIANCE_RESULT_PREFIX", "compliance-reports"),
    "sns_topic_arn": os.environ.get("COMPLIANCE_SNS_TOPIC_ARN", ""),
}

# System configuration for RegulatoryComplianceChecker
# These reflect the actual state of the system features
DEFAULT_SYSTEM_CONFIG = {
    "ai_disclosure_text": os.environ.get(
        "AI_DISCLOSURE_TEXT",
        "본 추천은 AI 시스템에 의해 생성되었습니다. 최종 판단은 고객님께서 직접 하시기 바랍니다.",
    ),
    "suitability_assessment_enabled": os.environ.get("SUITABILITY_ENABLED", "true").lower() == "true",
    "cooling_off_days": int(os.environ.get("COOLING_OFF_DAYS", "14")),
    "complaint_handler_module": os.environ.get("COMPLAINT_HANDLER_MODULE", ""),
    "consent_required": os.environ.get("CONSENT_REQUIRED", "true").lower() == "true",
    "data_retention_days": int(os.environ.get("DATA_RETENTION_DAYS", "365")),
    "deletion_handler_enabled": os.environ.get("DELETION_HANDLER_ENABLED", "true").lower() == "true",
    "purpose_limitation_enforced": os.environ.get("PURPOSE_LIMITATION", "true").lower() == "true",
    "data_minimization_enforced": os.environ.get("DATA_MINIMIZATION", "true").lower() == "true",
    "xai_enabled": os.environ.get("XAI_ENABLED", "true").lower() == "true",
    "kill_switch_enabled": os.environ.get("KILL_SWITCH_ENABLED", "true").lower() == "true",
    "optout_mechanism_enabled": os.environ.get("OPTOUT_ENABLED", "true").lower() == "true",
    "bias_monitoring_enabled": os.environ.get("BIAS_MONITORING_ENABLED", "true").lower() == "true",
    "performance_monitoring_enabled": os.environ.get("PERF_MONITORING_ENABLED", "true").lower() == "true",
    "model_registry_enabled": os.environ.get("MODEL_REGISTRY_ENABLED", "true").lower() == "true",
    "audit_trail_enabled": os.environ.get("AUDIT_TRAIL_ENABLED", "true").lower() == "true",
    "incident_response_enabled": os.environ.get("INCIDENT_RESPONSE_ENABLED", "true").lower() == "true",
    "data_lineage_enabled": os.environ.get("DATA_LINEAGE_ENABLED", "true").lower() == "true",
    "encryption_at_rest": os.environ.get("ENCRYPTION_AT_REST", "true").lower() == "true",
    "encryption_in_transit": os.environ.get("ENCRYPTION_IN_TRANSIT", "true").lower() == "true",
}


# ---------------------------------------------------------------------------
# Lambda handler
# ---------------------------------------------------------------------------

def handler(event: Dict[str, Any], context: Any) -> Dict[str, Any]:
    """Lambda entry point for quarterly compliance checks."""

    config_override = event.get("config_override", {})
    regulation_filter = event.get("regulation", None)

    cfg = {**DEFAULT_CONFIG, **config_override}
    system_config = {**DEFAULT_SYSTEM_CONFIG, **config_override.get("system", {})}

    now = datetime.now(timezone.utc)

    logger.info(
        "ComplianceCheck: regulation_filter=%s",
        regulation_filter or "(all)",
    )

    # ------------------------------------------------------------------
    # Step 1: Build RegulatoryComplianceChecker
    # ------------------------------------------------------------------
    from core.compliance.audit_store import ComplianceAuditStore
    from core.compliance.regulatory_checker import RegulatoryComplianceChecker

    audit_store = ComplianceAuditStore(
        table_prefix=cfg["audit_table_prefix"],
        region=cfg["region"],
    )

    checker = RegulatoryComplianceChecker(
        audit_store=audit_store,
        config=system_config,
    )

    # ------------------------------------------------------------------
    # Step 2: Run checks
    # ------------------------------------------------------------------
    if regulation_filter:
        results = checker.run_regulation(regulation_filter)
    else:
        results = checker.run_all_checks()

    summary = checker.get_summary(results)

    logger.info(
        "Compliance results: %d/%d passed (%.1f%%)",
        summary["passed"],
        summary["total"],
        summary["pass_rate"] * 100,
    )

    # ------------------------------------------------------------------
    # Step 3: Critical failure alerting via SNS
    # ------------------------------------------------------------------
    if summary["critical_failures"]:
        _notify_critical_failures(cfg, summary)

    # ------------------------------------------------------------------
    # Step 4: Save results to S3 (governance report)
    # ------------------------------------------------------------------
    report = _build_governance_report(now, summary, results, system_config)
    s3_key = _save_report_to_s3(cfg, report, now)

    # ------------------------------------------------------------------
    # Step 5: Audit trail
    # ------------------------------------------------------------------
    _record_audit(cfg, audit_store, summary, now)

    return {
        "checked_at": now.isoformat(),
        "total": summary["total"],
        "passed": summary["passed"],
        "failed": summary["failed"],
        "pass_rate": summary["pass_rate"],
        "critical_failures": summary["critical_failures"],
        "s3_key": s3_key,
    }


# ---------------------------------------------------------------------------
# Step 3: SNS notification for critical failures
# ---------------------------------------------------------------------------

def _notify_critical_failures(
    cfg: Dict[str, Any],
    summary: Dict[str, Any],
) -> None:
    """Publish critical compliance failure alert to SNS."""
    sns_topic_arn = cfg.get("sns_topic_arn", "")
    if not sns_topic_arn:
        logger.info("No SNS topic configured; skipping critical failure notification")
        return

    try:
        import boto3

        sns = boto3.client("sns", region_name=cfg["region"])

        failure_details = "\n".join(
            f"  - [{f['id']}] {f['description']}: {f['message']}"
            for f in summary["critical_failures"]
        )

        subject = (
            f"[CRITICAL] Regulatory Compliance: "
            f"{len(summary['critical_failures'])} critical failure(s)"
        )

        message = json.dumps(
            {
                "alert_type": "regulatory_compliance_critical",
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "total_checks": summary["total"],
                "passed": summary["passed"],
                "failed": summary["failed"],
                "pass_rate": summary["pass_rate"],
                "critical_failures": summary["critical_failures"],
                "detail_text": failure_details,
            },
            ensure_ascii=False,
            indent=2,
        )

        sns.publish(
            TopicArn=sns_topic_arn,
            Subject=subject[:100],  # SNS subject length limit
            Message=message,
        )
        logger.info(
            "SNS critical compliance alert sent (%d failures)",
            len(summary["critical_failures"]),
        )

    except Exception as exc:
        logger.warning("SNS notification failed: %s", exc)


# ---------------------------------------------------------------------------
# Step 4: Build governance report and save to S3
# ---------------------------------------------------------------------------

def _build_governance_report(
    now: datetime,
    summary: Dict[str, Any],
    results: list,
    system_config: Dict[str, Any],
) -> Dict[str, Any]:
    """Build a structured governance report from compliance results."""
    quarter = f"Q{math.ceil(now.month / 3)}"
    period = f"{now.year}-{quarter}"

    check_details = []
    for r in results:
        check_details.append({
            "id": r.item.id,
            "regulation": r.item.regulation,
            "category": r.item.category,
            "description": r.item.description,
            "severity": r.item.severity,
            "passed": r.passed,
            "message": r.message,
            "details": r.details,
            "checked_at": r.checked_at,
        })

    return {
        "report_type": "regulatory_compliance",
        "generated_at": now.isoformat(),
        "period": period,
        "summary": summary,
        "check_details": check_details,
        "system_config_snapshot": system_config,
        "governance_note": (
            "This report is auto-generated by the PLE compliance checking system. "
            "Critical failures require immediate remediation and regulatory notification."
        ),
    }


def _save_report_to_s3(
    cfg: Dict[str, Any],
    report: Dict[str, Any],
    now: datetime,
) -> Optional[str]:
    """Persist the compliance report as a JSON file in S3."""
    try:
        import boto3

        s3 = boto3.client("s3", region_name=cfg["region"])
        bucket = cfg["result_bucket"]
        prefix = cfg["result_prefix"].strip("/")
        quarter = f"Q{math.ceil(now.month / 3)}"
        s3_key = f"{prefix}/{now.year}-{quarter}/compliance_report.json"

        s3.put_object(
            Bucket=bucket,
            Key=s3_key,
            Body=json.dumps(report, ensure_ascii=False, indent=2, default=str).encode("utf-8"),
            ContentType="application/json",
        )
        logger.info("Compliance report saved to s3://%s/%s", bucket, s3_key)
        return s3_key

    except Exception as exc:
        logger.warning("Failed to save report to S3: %s", exc)
        return None


# ---------------------------------------------------------------------------
# Step 5: Audit trail
# ---------------------------------------------------------------------------

def _record_audit(
    cfg: Dict[str, Any],
    audit_store: Any,
    summary: Dict[str, Any],
    now: datetime,
) -> None:
    """Log compliance check event to the compliance audit store."""
    try:
        audit_store.log_event("embedding", {
            "pk": "compliance_check",
            "sk": now.isoformat(),
            "total_checks": summary["total"],
            "passed": summary["passed"],
            "failed": summary["failed"],
            "pass_rate": summary["pass_rate"],
            "critical_failures": [f["id"] for f in summary["critical_failures"]],
            "by_regulation": summary.get("by_regulation", {}),
        })
        logger.info("Audit trail recorded for compliance check")

    except Exception as exc:
        logger.warning("Audit trail recording failed: %s", exc)
