"""
Incident management with auto-classification, SNS notification, and status tracking.

Severity levels and response windows:

- **CRITICAL** -- 1 hour (kill-switch, DI < 0.6, security breach)
- **MAJOR**    -- 4 hours (DI < 0.8, herding critical, model rollback)
- **MINOR**    -- 24 hours (drift warning, quality drop, herding high)

Incident records are persisted to S3 as JSON and optionally to the
:class:`~core.monitoring.compliance_store.ComplianceAuditStore` DynamoDB table.

Auto-escalation publishes to an SNS topic for CRITICAL and MAJOR incidents.
"""

from __future__ import annotations

import json
import logging
import os
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from enum import Enum
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Enums and data classes
# ---------------------------------------------------------------------------

class IncidentSeverity(Enum):
    """Incident severity classification."""

    CRITICAL = "critical"
    MAJOR = "major"
    MINOR = "minor"


@dataclass
class IncidentRecord:
    """Immutable record for a single incident."""

    incident_id: str
    timestamp: str
    event_type: str
    severity: str
    source_module: str
    description: str
    metrics: Dict[str, Any]
    affected_scope: Dict[str, Any]
    immediate_action: str
    status: str  # "open" | "investigating" | "resolved" | "closed"
    resolution: Optional[str] = None
    resolved_at: Optional[str] = None


# ---------------------------------------------------------------------------
# Severity criteria (configurable defaults)
# ---------------------------------------------------------------------------

DEFAULT_SEVERITY_CRITERIA: Dict[str, Dict[str, Any]] = {
    "critical": {
        "conditions": [
            "kill_switch_activation",
            "di_below_0.6",
            "security_breach",
        ],
        "response_time": "1 hour",
        "escalate": True,
    },
    "major": {
        "conditions": [
            "di_below_0.8",
            "herding_critical",
            "model_rollback",
        ],
        "response_time": "4 hours",
        "escalate": True,
    },
    "minor": {
        "conditions": [
            "drift_warning",
            "quality_drop",
            "herding_high",
        ],
        "response_time": "24 hours",
        "escalate": False,
    },
}


# ---------------------------------------------------------------------------
# IncidentReporter
# ---------------------------------------------------------------------------

class IncidentReporter:
    """Create, classify, persist, and escalate operational incidents.

    Parameters
    ----------
    sns_topic_arn : str, optional
        ARN of the SNS topic for auto-escalation notifications.
        Falls back to ``INCIDENT_SNS_TOPIC_ARN`` environment variable.
    s3_bucket : str, optional
        S3 bucket for incident archive storage.
        Falls back to ``INCIDENT_S3_BUCKET`` environment variable.
    s3_prefix : str
        Key prefix for incident objects.
    severity_criteria : dict, optional
        Override the default severity classification rules.
    region : str
        AWS region.
    """

    def __init__(
        self,
        sns_topic_arn: Optional[str] = None,
        s3_bucket: Optional[str] = None,
        s3_prefix: str = "incidents",
        severity_criteria: Optional[Dict[str, Dict[str, Any]]] = None,
        region: Optional[str] = None,
    ) -> None:
        self.sns_topic_arn = sns_topic_arn or os.environ.get("INCIDENT_SNS_TOPIC_ARN", "")
        self.s3_bucket = s3_bucket or os.environ.get("INCIDENT_S3_BUCKET", "")
        self.s3_prefix = s3_prefix.strip("/")
        self.severity_criteria = severity_criteria or dict(DEFAULT_SEVERITY_CRITERIA)
        self.region = region or os.environ.get("AWS_DEFAULT_REGION")

        self._sns_client = None
        self._s3_client = None
        try:
            import boto3

            if self.sns_topic_arn:
                self._sns_client = boto3.client("sns", region_name=self.region)
            if self.s3_bucket:
                self._s3_client = boto3.client("s3", region_name=self.region)
        except Exception as exc:
            logger.warning("AWS client init failed: %s", exc)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def create_incident(
        self,
        event_type: str,
        metrics: Dict[str, Any],
        source_module: str = "unknown",
        description: Optional[str] = None,
    ) -> IncidentRecord:
        """Create and classify a new incident.

        Parameters
        ----------
        event_type : str
            Type of triggering event (e.g. ``"kill_switch_activation"``).
        metrics : dict
            Metric snapshot associated with the event.
        source_module : str
            Originating module name.
        description : str, optional
            Human-readable description (auto-generated if omitted).

        Returns
        -------
        IncidentRecord
            The persisted incident record.
        """
        now = datetime.now(timezone.utc)
        short_id = uuid.uuid4().hex[:8]
        incident_id = f"INC-{now.strftime('%Y%m%d-%H%M%S')}-{short_id}"

        severity = self._determine_severity(event_type, metrics)
        affected_scope = self._determine_affected_scope(event_type, metrics)
        immediate_action = self._determine_immediate_action(severity, event_type)
        auto_description = description or self._auto_description(event_type, metrics)

        record = IncidentRecord(
            incident_id=incident_id,
            timestamp=now.isoformat(),
            event_type=event_type,
            severity=severity,
            source_module=source_module,
            description=auto_description,
            metrics=metrics,
            affected_scope=affected_scope,
            immediate_action=immediate_action,
            status="open",
        )

        logger.warning(
            "Incident created: %s  severity=%s  event_type=%s",
            incident_id,
            severity,
            event_type,
        )

        # Persist
        self._archive_incident(record)

        # Escalate via SNS for critical / major
        criteria = self.severity_criteria.get(severity, {})
        if criteria.get("escalate", False):
            self._notify_sns(record)

        return record

    def generate_report(self, incident: IncidentRecord) -> Dict[str, Any]:
        """Generate a structured incident report.

        Returns
        -------
        dict
            Internal report structure suitable for governance review.
        """
        severity_info = self.severity_criteria.get(incident.severity, {})

        return {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "internal_report": {
                "incident_id": incident.incident_id,
                "timestamp": incident.timestamp,
                "event_type": incident.event_type,
                "severity": incident.severity,
                "source_module": incident.source_module,
                "description": incident.description,
                "metrics_snapshot": incident.metrics,
                "affected_scope": incident.affected_scope,
                "immediate_action": incident.immediate_action,
                "response_time_requirement": severity_info.get("response_time", "N/A"),
                "status": incident.status,
            },
        }

    def generate_post_incident_review(self, incident_id: str) -> Dict[str, Any]:
        """Generate a post-incident review (PIR) template.

        Parameters
        ----------
        incident_id : str
            ID of the incident to review.

        Returns
        -------
        dict
            PIR template with timeline, root-cause analysis, and
            corrective-action placeholders.
        """
        return {
            "report_type": "post_incident_review",
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "incident_id": incident_id,
            "timeline": [
                {"time": "(to be filled)", "event": "Incident detected", "actor": "Monitoring system"},
                {"time": "(to be filled)", "event": "Incident confirmed", "actor": "On-call engineer"},
                {"time": "(to be filled)", "event": "Incident resolved", "actor": "On-call engineer"},
            ],
            "root_cause_analysis": {
                "what_happened": "(to be filled)",
                "why_happened": "(root cause pending)",
                "contributing_factors": ["(factor 1)", "(factor 2)"],
            },
            "impact_assessment": {
                "affected_customers": 0,
                "duration_hours": 0.0,
            },
            "corrective_actions": [
                {
                    "action": "(to be filled)",
                    "owner": "(to be assigned)",
                    "deadline": (datetime.now(timezone.utc) + timedelta(days=7)).strftime("%Y-%m-%d"),
                    "status": "pending",
                },
            ],
            "prevention_measures": ["(to be filled)"],
            "lessons_learned": ["(to be filled)"],
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _determine_severity(self, event_type: str, metrics: Dict[str, Any]) -> str:
        """Map event type and metrics to a severity level."""
        if event_type in ("kill_switch_activation", "security_alert", "security_breach"):
            return IncidentSeverity.CRITICAL.value

        if event_type == "fairness_violation":
            di_value = metrics.get("di_value", 1.0)
            if di_value < 0.6:
                return IncidentSeverity.CRITICAL.value
            if di_value < 0.8:
                return IncidentSeverity.MAJOR.value
            return IncidentSeverity.MINOR.value

        if event_type in ("herding_critical", "model_rollback"):
            return IncidentSeverity.MAJOR.value

        if event_type in ("drift_warning", "quality_drop", "herding_high"):
            return IncidentSeverity.MINOR.value

        # Fallback: use metric-supplied severity
        raw = metrics.get("severity", "minor")
        try:
            return IncidentSeverity(raw).value
        except ValueError:
            return IncidentSeverity.MINOR.value

    @staticmethod
    def _determine_affected_scope(event_type: str, metrics: Dict[str, Any]) -> Dict[str, Any]:
        """Estimate the scope of impact."""
        if event_type == "kill_switch_activation":
            scope = metrics.get("scope", "global")
            return {
                "scope_type": scope,
                "estimated_customers": metrics.get("estimated_customers", 0),
                "description": f"Recommendation service halted (scope={scope}).",
            }
        if event_type == "fairness_violation":
            attribute = metrics.get("attribute", "unknown")
            return {
                "scope_type": "demographic_group",
                "attribute": attribute,
                "estimated_customers": metrics.get("affected_count", 0),
                "description": f"Fairness violation on protected attribute '{attribute}'.",
            }
        if event_type in ("drift_warning", "quality_drop"):
            return {
                "scope_type": "system_wide",
                "estimated_customers": metrics.get("estimated_customers", 0),
                "description": "Potential degradation of recommendation quality.",
            }
        return {
            "scope_type": "unknown",
            "estimated_customers": 0,
            "description": "Impact scope undetermined; manual investigation required.",
        }

    def _determine_immediate_action(self, severity: str, event_type: str) -> str:
        """Suggest an immediate action based on severity."""
        if severity == IncidentSeverity.CRITICAL.value:
            if event_type == "kill_switch_activation":
                return (
                    "Maintain kill-switch state; activate human fallback path; "
                    "notify regulatory authority within 1 hour."
                )
            return (
                "Halt affected service; activate human fallback; "
                "notify regulatory authority within 1 hour."
            )
        if severity == IncidentSeverity.MAJOR.value:
            return "Increase monitoring; begin root-cause investigation; internal report within 4 hours."
        return "Collect logs; monitor for escalation; internal report within 24 hours."

    @staticmethod
    def _auto_description(event_type: str, metrics: Dict[str, Any]) -> str:
        """Generate a human-readable description from event type and metrics."""
        templates = {
            "kill_switch_activation": "Kill-switch activated (scope={scope}). Recommendation service halted.",
            "fairness_violation": "Fairness violation: attribute={attribute}, DI={di_value}",
            "herding_critical": "Critical herding detected (risk_score={herding_risk_score})",
            "herding_high": "High herding detected (risk_score={herding_risk_score})",
            "drift_warning": "Feature drift warning (drift_score={drift_score})",
            "quality_drop": "Recommendation quality drop (metric={quality_metric})",
            "model_rollback": "Model rollback from={from_version} to={to_version}",
            "security_alert": "Security alert: type={alert_type}",
            "security_breach": "Security breach: type={breach_type}",
        }
        template = templates.get(event_type, f"Event: {event_type}")
        try:
            return template.format_map({**{"scope": "unknown", "attribute": "unknown"}, **metrics})
        except (KeyError, ValueError):
            return template

    def _archive_incident(self, record: IncidentRecord) -> None:
        """Persist the incident record to S3 and the compliance audit store."""
        doc = {
            "incident_id": record.incident_id,
            "timestamp": record.timestamp,
            "event_type": record.event_type,
            "severity": record.severity,
            "source_module": record.source_module,
            "description": record.description,
            "metrics": record.metrics,
            "affected_scope": record.affected_scope,
            "immediate_action": record.immediate_action,
            "status": record.status,
            "resolution": record.resolution,
            "resolved_at": record.resolved_at,
        }

        # S3 archive
        if self._s3_client and self.s3_bucket:
            try:
                date_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")
                s3_key = f"{self.s3_prefix}/{date_str}/{record.incident_id}.json"
                self._s3_client.put_object(
                    Bucket=self.s3_bucket,
                    Key=s3_key,
                    Body=json.dumps(doc, ensure_ascii=False, indent=2).encode("utf-8"),
                    ContentType="application/json",
                )
                logger.info("Incident archived to S3: s3://%s/%s", self.s3_bucket, s3_key)
            except Exception as exc:
                logger.warning("S3 incident archive failed: %s", exc)

        # Compliance audit store
        try:
            from core.monitoring.compliance_store import get_compliance_audit_store

            store = get_compliance_audit_store()
            store.log_incident(
                incident_id=record.incident_id,
                event_type=record.event_type,
                severity=record.severity,
                source_module=record.source_module,
                status=record.status,
                description=record.description,
                resolution=record.resolution,
                resolved_at=record.resolved_at,
            )
        except Exception as exc:
            logger.warning("Compliance audit store write failed (incident): %s", exc)

    def _notify_sns(self, record: IncidentRecord) -> None:
        """Publish an incident notification to SNS."""
        if not self._sns_client or not self.sns_topic_arn:
            return
        try:
            subject = f"[{record.severity.upper()}] Incident {record.incident_id}"
            message = json.dumps(
                {
                    "incident_id": record.incident_id,
                    "severity": record.severity,
                    "event_type": record.event_type,
                    "description": record.description,
                    "immediate_action": record.immediate_action,
                    "timestamp": record.timestamp,
                },
                ensure_ascii=False,
                indent=2,
            )
            self._sns_client.publish(
                TopicArn=self.sns_topic_arn,
                Subject=subject[:100],  # SNS subject length limit
                Message=message,
            )
            logger.info("SNS notification sent for incident %s", record.incident_id)
        except Exception as exc:
            logger.warning("SNS notification failed: %s", exc)


__all__ = ["IncidentReporter", "IncidentRecord", "IncidentSeverity"]
