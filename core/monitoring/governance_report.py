"""
Governance report generator for AI oversight committees.

Produces monthly or quarterly governance reports covering 9 sections:

1. Fairness summary
2. Drift summary
3. Incident summary
4. Model change history
5. Kill-switch history
6. Reason quality
7. Herding analysis
8. Audit summary
9. Executive summary

Reports are output as JSON-serializable dicts and can optionally be
persisted to S3 for archival.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class GovernanceReport:
    """Container for a governance committee report."""

    report_id: str
    period: str                                 # "monthly" | "quarterly"
    period_start: str                           # ISO date
    period_end: str                             # ISO date
    generated_at: str                           # ISO datetime

    # Section data
    fairness_summary: Dict[str, Any]
    drift_summary: Dict[str, Any]
    incident_summary: Dict[str, Any]
    model_change_history: List[Dict[str, Any]]
    kill_switch_history: List[Dict[str, Any]]
    recommendation_quality: Dict[str, Any]
    herding_summary: Dict[str, Any]
    audit_summary: Dict[str, Any]
    executive_summary: str
    action_items: List[str]


# ---------------------------------------------------------------------------
# GovernanceReportGenerator
# ---------------------------------------------------------------------------

class GovernanceReportGenerator:
    """Generate periodic governance reports.

    The generator collects data from the various monitoring subsystems
    (fairness, drift, incidents, etc.) and assembles a unified report
    for governance committee review.

    Parameters
    ----------
    s3_bucket : str, optional
        S3 bucket for report archival.
    s3_prefix : str
        Key prefix for archived reports.
    system_name : str
        Name of the ML system (appears in reports).
    """

    PERIOD_DAYS = {"monthly": 30, "quarterly": 90}

    def __init__(
        self,
        s3_bucket: Optional[str] = None,
        s3_prefix: str = "governance_reports",
        system_name: str = "PLE-Cluster-adaTT",
    ) -> None:
        self.s3_bucket = s3_bucket or os.environ.get("GOVERNANCE_S3_BUCKET", "")
        self.s3_prefix = s3_prefix.strip("/")
        self.system_name = system_name

        self._s3_client = None
        if self.s3_bucket:
            try:
                import boto3

                region = os.environ.get("AWS_DEFAULT_REGION")
                self._s3_client = boto3.client("s3", region_name=region)
            except Exception as exc:
                logger.warning("S3 client init failed (governance): %s", exc)

    # ------------------------------------------------------------------
    # Report generation
    # ------------------------------------------------------------------

    def generate_report(
        self,
        period: str = "monthly",
        fairness_data: Optional[Dict[str, Any]] = None,
        drift_data: Optional[Dict[str, Any]] = None,
        incident_data: Optional[List[Dict[str, Any]]] = None,
        model_changes: Optional[List[Dict[str, Any]]] = None,
        kill_switch_events: Optional[List[Dict[str, Any]]] = None,
        recommendation_quality: Optional[Dict[str, Any]] = None,
        herding_data: Optional[Dict[str, Any]] = None,
        audit_data: Optional[Dict[str, Any]] = None,
    ) -> GovernanceReport:
        """Assemble a governance report from component data.

        Parameters
        ----------
        period : str
            ``"monthly"`` or ``"quarterly"``.
        fairness_data : dict, optional
            Output from :meth:`FairnessMonitor.evaluate_all_attributes`.
        drift_data : dict, optional
            Output from :meth:`DriftDetector.detect_drift`.
        incident_data : list of dict, optional
            Incident records for the period.
        model_changes : list of dict, optional
            Model version change log entries.
        kill_switch_events : list of dict, optional
            Kill-switch activation/deactivation events.
        recommendation_quality : dict, optional
            Recommendation quality metrics.
        herding_data : dict, optional
            Output from :meth:`HerdingDetector.detect_herding`.
        audit_data : dict, optional
            Audit log summary statistics.

        Returns
        -------
        GovernanceReport
        """
        now = datetime.now(timezone.utc)
        days = self.PERIOD_DAYS.get(period, 30)
        period_start = (now - timedelta(days=days)).strftime("%Y-%m-%d")
        period_end = now.strftime("%Y-%m-%d")
        report_id = f"GOV-{now.strftime('%Y%m%d')}"

        fairness_summary = self._build_fairness_summary(fairness_data or {})
        drift_summary = self._build_drift_summary(drift_data or {})
        incident_summary = self._build_incident_summary(incident_data or [])
        herding_summary = self._build_herding_summary(herding_data or {})
        audit_summary = audit_data or {"total_entries": 0, "note": "No audit data provided."}
        quality = recommendation_quality or {}

        action_items = self._derive_action_items(
            fairness_summary, drift_summary, incident_summary, herding_summary,
        )
        executive_summary = self._build_executive_summary(
            period, fairness_summary, drift_summary, incident_summary, herding_summary,
        )

        report = GovernanceReport(
            report_id=report_id,
            period=period,
            period_start=period_start,
            period_end=period_end,
            generated_at=now.isoformat(),
            fairness_summary=fairness_summary,
            drift_summary=drift_summary,
            incident_summary=incident_summary,
            model_change_history=model_changes or [],
            kill_switch_history=kill_switch_events or [],
            recommendation_quality=quality,
            herding_summary=herding_summary,
            audit_summary=audit_summary,
            executive_summary=executive_summary,
            action_items=action_items,
        )

        logger.info("Governance report generated: %s (%s)", report_id, period)
        return report

    def to_dict(self, report: GovernanceReport) -> Dict[str, Any]:
        """Serialize a ``GovernanceReport`` to a JSON-compatible dict."""
        return {
            "report_id": report.report_id,
            "system_name": self.system_name,
            "period": report.period,
            "period_start": report.period_start,
            "period_end": report.period_end,
            "generated_at": report.generated_at,
            "sections": {
                "fairness": report.fairness_summary,
                "drift": report.drift_summary,
                "incidents": report.incident_summary,
                "model_changes": report.model_change_history,
                "kill_switch": report.kill_switch_history,
                "recommendation_quality": report.recommendation_quality,
                "herding": report.herding_summary,
                "audit": report.audit_summary,
            },
            "executive_summary": report.executive_summary,
            "action_items": report.action_items,
        }

    def archive_report(self, report: GovernanceReport) -> Optional[str]:
        """Persist the report to S3 as JSON.

        Returns
        -------
        str or None
            The S3 URI if upload succeeded, else ``None``.
        """
        if not self._s3_client or not self.s3_bucket:
            logger.warning("S3 not configured; report not archived.")
            return None

        try:
            report_dict = self.to_dict(report)
            body = json.dumps(report_dict, ensure_ascii=False, indent=2).encode("utf-8")
            s3_key = f"{self.s3_prefix}/{report.period}/{report.report_id}.json"
            self._s3_client.put_object(
                Bucket=self.s3_bucket,
                Key=s3_key,
                Body=body,
                ContentType="application/json",
            )
            uri = f"s3://{self.s3_bucket}/{s3_key}"
            logger.info("Governance report archived: %s", uri)
            return uri
        except Exception as exc:
            logger.warning("Failed to archive governance report: %s", exc)
            return None

    # ------------------------------------------------------------------
    # Section builders
    # ------------------------------------------------------------------

    @staticmethod
    def _build_fairness_summary(fairness_data: Dict[str, Any]) -> Dict[str, Any]:
        """Build the fairness section from evaluation results."""
        if not fairness_data:
            return {"status": "no_data", "attributes_checked": 0, "violations": 0}

        total_violations = 0
        attribute_details: Dict[str, Any] = {}
        for attr, metrics in fairness_data.items():
            if isinstance(metrics, dict):
                violations = metrics.get("violations", [])
            elif hasattr(metrics, "violations"):
                violations = metrics.violations
            else:
                violations = []
            total_violations += len(violations)
            attribute_details[attr] = {
                "is_fair": len(violations) == 0,
                "violation_count": len(violations),
            }

        return {
            "status": "pass" if total_violations == 0 else "fail",
            "attributes_checked": len(fairness_data),
            "total_violations": total_violations,
            "details": attribute_details,
        }

    @staticmethod
    def _build_drift_summary(drift_data: Dict[str, Any]) -> Dict[str, Any]:
        """Build the drift section from detection results."""
        summary = drift_data.get("summary", {})
        return {
            "drift_detected": summary.get("drift_detected", False),
            "total_features": summary.get("total_features", 0),
            "warning_count": summary.get("warning_count", 0),
            "critical_count": summary.get("critical_count", 0),
            "max_psi": summary.get("max_psi", 0.0),
            "avg_psi": summary.get("avg_psi", 0.0),
        }

    @staticmethod
    def _build_incident_summary(incidents: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Build the incident section."""
        if not incidents:
            return {"total": 0, "critical": 0, "major": 0, "minor": 0, "open": 0}

        severity_counts = {"critical": 0, "major": 0, "minor": 0}
        open_count = 0
        for inc in incidents:
            sev = inc.get("severity", "minor")
            severity_counts[sev] = severity_counts.get(sev, 0) + 1
            if inc.get("status", "open") in ("open", "investigating"):
                open_count += 1

        return {
            "total": len(incidents),
            "critical": severity_counts.get("critical", 0),
            "major": severity_counts.get("major", 0),
            "minor": severity_counts.get("minor", 0),
            "open": open_count,
        }

    @staticmethod
    def _build_herding_summary(herding_data: Dict[str, Any]) -> Dict[str, Any]:
        """Build the herding section."""
        if not herding_data:
            return {"status": "no_data", "is_herding": False}

        return {
            "status": "herding_detected" if herding_data.get("is_herding") else "normal",
            "is_herding": herding_data.get("is_herding", False),
            "severity": herding_data.get("severity", "none"),
            "metrics": herding_data.get("metrics", {}),
        }

    # ------------------------------------------------------------------
    # Executive summary and action items
    # ------------------------------------------------------------------

    @staticmethod
    def _derive_action_items(
        fairness: Dict[str, Any],
        drift: Dict[str, Any],
        incidents: Dict[str, Any],
        herding: Dict[str, Any],
    ) -> List[str]:
        """Derive action items from section summaries."""
        items: List[str] = []

        if fairness.get("total_violations", 0) > 0:
            items.append(
                f"Investigate {fairness['total_violations']} fairness violation(s) "
                "and apply bias mitigation."
            )
        if drift.get("critical_count", 0) > 0:
            items.append(
                f"{drift['critical_count']} feature(s) show critical drift; "
                "evaluate model retraining."
            )
        if incidents.get("critical", 0) > 0:
            items.append(
                f"{incidents['critical']} critical incident(s) occurred; "
                "complete post-incident reviews."
            )
        if incidents.get("open", 0) > 0:
            items.append(f"{incidents['open']} incident(s) remain open; accelerate resolution.")
        if herding.get("is_herding"):
            items.append(
                f"Herding detected (severity={herding.get('severity', 'unknown')}); "
                "review recommendation diversity."
            )
        if not items:
            items.append("No immediate action items. Continue routine monitoring.")
        return items

    @staticmethod
    def _build_executive_summary(
        period: str,
        fairness: Dict[str, Any],
        drift: Dict[str, Any],
        incidents: Dict[str, Any],
        herding: Dict[str, Any],
    ) -> str:
        """Auto-generate an executive summary paragraph."""
        parts: List[str] = [f"Governance review for the {period} period."]

        # Fairness
        fv = fairness.get("total_violations", 0)
        if fv == 0:
            parts.append("All fairness checks passed.")
        else:
            parts.append(f"Fairness: {fv} violation(s) detected -- requires attention.")

        # Drift
        if drift.get("drift_detected"):
            parts.append(
                f"Data drift: {drift.get('critical_count', 0)} critical, "
                f"{drift.get('warning_count', 0)} warning feature(s)."
            )
        else:
            parts.append("No significant data drift observed.")

        # Incidents
        total_inc = incidents.get("total", 0)
        if total_inc > 0:
            parts.append(
                f"Incidents: {total_inc} total "
                f"({incidents.get('critical', 0)} critical, "
                f"{incidents.get('major', 0)} major, "
                f"{incidents.get('minor', 0)} minor)."
            )
        else:
            parts.append("No incidents reported.")

        # Herding
        if herding.get("is_herding"):
            parts.append(f"Herding detected at {herding.get('severity', 'unknown')} severity.")
        else:
            parts.append("Recommendation diversity within normal range.")

        return " ".join(parts)


__all__ = ["GovernanceReportGenerator", "GovernanceReport"]
