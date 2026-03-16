"""
Financial Risk Impact Assessment (FRIA) evaluator aligned with EU AI Act Article 9.

Performs structured risk assessment across five dimensions for each ML task:

- **Fairness risk**      Bias and discrimination potential
- **Transparency risk**  Explainability and interpretability gaps
- **Accuracy risk**      Model performance degradation potential
- **Security risk**      Adversarial and data-integrity threats
- **Oversight risk**     Human-in-the-loop adequacy

Each dimension yields a score in [0.0, 1.0] (higher = more risk).  The
aggregate score determines the EU AI Act risk classification:

- ``UNACCEPTABLE`` (>= 0.9)
- ``HIGH``         (>= 0.7)
- ``LIMITED``      (>= 0.4)
- ``MINIMAL``      (< 0.4)

Results are captured in a :class:`FRIAReport` dataclass and can optionally be
persisted via :class:`~core.monitoring.compliance_store.ComplianceAuditStore`.
"""

from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Risk classification thresholds
# ---------------------------------------------------------------------------

RISK_THRESHOLDS: Dict[str, float] = {
    "UNACCEPTABLE": 0.9,
    "HIGH": 0.7,
    "LIMITED": 0.4,
    "MINIMAL": 0.0,
}

ASSESSMENT_DIMENSIONS: List[str] = [
    "fairness_risk",
    "transparency_risk",
    "accuracy_risk",
    "security_risk",
    "oversight_risk",
]


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class TaskRiskProfile:
    """Risk profile for a single ML task."""

    task_name: str
    dimension_scores: Dict[str, float]
    aggregate_score: float
    risk_level: str  # UNACCEPTABLE | HIGH | LIMITED | MINIMAL
    remediations: List[str]
    details: Dict[str, Any] = field(default_factory=dict)


@dataclass
class FRIAReport:
    """Container for a full FRIA evaluation report."""

    report_id: str
    assessed_at: str  # ISO datetime
    system_name: str
    assessor: str
    task_profiles: List[TaskRiskProfile]
    overall_risk_level: str
    overall_score: float
    summary: str
    regulatory_references: List[str]
    metadata: Dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Remediation catalogue
# ---------------------------------------------------------------------------

_REMEDIATION_MAP: Dict[str, Dict[str, str]] = {
    "fairness_risk": {
        "HIGH": "Implement bias mitigation (re-sampling, adversarial debiasing) and increase fairness monitoring frequency.",
        "LIMITED": "Review fairness metrics quarterly and document disparate impact analysis.",
        "UNACCEPTABLE": "Halt deployment until fairness violations are resolved; conduct independent bias audit.",
    },
    "transparency_risk": {
        "HIGH": "Add SHAP/LIME explanations for every prediction and publish model cards.",
        "LIMITED": "Provide feature-importance summaries and improve documentation.",
        "UNACCEPTABLE": "System must not operate without full explainability; implement XAI pipeline.",
    },
    "accuracy_risk": {
        "HIGH": "Establish tighter drift monitoring, automated retraining triggers, and fallback models.",
        "LIMITED": "Increase validation frequency and add shadow-mode evaluation.",
        "UNACCEPTABLE": "Remove model from production; accuracy below acceptable safety threshold.",
    },
    "security_risk": {
        "HIGH": "Deploy adversarial input detection, enable model encryption, and run penetration tests.",
        "LIMITED": "Review access controls and add input validation for anomalous patterns.",
        "UNACCEPTABLE": "Isolate system immediately; critical security vulnerabilities detected.",
    },
    "oversight_risk": {
        "HIGH": "Implement mandatory human review for high-impact decisions and real-time alerting.",
        "LIMITED": "Establish periodic human audit of automated decisions.",
        "UNACCEPTABLE": "Automated decisions must not proceed without human approval at this risk level.",
    },
}


# ---------------------------------------------------------------------------
# FRIAEvaluator
# ---------------------------------------------------------------------------

class FRIAEvaluator:
    """Financial Risk Impact Assessment evaluator.

    Assesses ML system risk aligned with EU AI Act Article 9 requirements
    for high-risk AI systems in financial services.

    Parameters
    ----------
    system_name : str
        Name of the ML system under assessment.
    assessor : str
        Identity of the assessor (person or automated pipeline).
    risk_thresholds : dict, optional
        Override default risk-level thresholds.
    dimension_weights : dict, optional
        Custom weights for each assessment dimension (must sum to 1.0).
        Defaults to equal weighting.
    audit_store : ComplianceAuditStore, optional
        If provided, FRIA results are persisted to the compliance audit store.
    """

    def __init__(
        self,
        system_name: str = "PLE-Cluster-adaTT",
        assessor: str = "automated",
        risk_thresholds: Optional[Dict[str, float]] = None,
        dimension_weights: Optional[Dict[str, float]] = None,
        audit_store: Optional[Any] = None,
    ) -> None:
        self.system_name = system_name
        self.assessor = assessor
        self.risk_thresholds = risk_thresholds or dict(RISK_THRESHOLDS)
        self.audit_store = audit_store

        if dimension_weights:
            self.dimension_weights = dict(dimension_weights)
        else:
            n = len(ASSESSMENT_DIMENSIONS)
            self.dimension_weights = {d: 1.0 / n for d in ASSESSMENT_DIMENSIONS}

    # ------------------------------------------------------------------
    # Per-task risk scoring
    # ------------------------------------------------------------------

    def assess_task(
        self,
        task_name: str,
        fairness_metrics: Optional[Dict[str, Any]] = None,
        transparency_metrics: Optional[Dict[str, Any]] = None,
        accuracy_metrics: Optional[Dict[str, Any]] = None,
        security_metrics: Optional[Dict[str, Any]] = None,
        oversight_metrics: Optional[Dict[str, Any]] = None,
    ) -> TaskRiskProfile:
        """Assess risk for a single ML task.

        Each ``*_metrics`` dict is expected to contain domain-specific
        indicators.  When ``None``, a conservative default score of 0.5
        (``LIMITED``) is assigned for that dimension.

        Parameters
        ----------
        task_name : str
            Identifier for the task (e.g. ``"cvr_prediction"``).
        fairness_metrics : dict, optional
            Keys: ``violation_count``, ``di_value``, ``spd_value``.
        transparency_metrics : dict, optional
            Keys: ``has_explanations``, ``explanation_coverage``, ``model_card_exists``.
        accuracy_metrics : dict, optional
            Keys: ``auc``, ``drift_detected``, ``performance_degradation``.
        security_metrics : dict, optional
            Keys: ``input_validation``, ``encryption_enabled``, ``access_control_level``.
        oversight_metrics : dict, optional
            Keys: ``human_review_rate``, ``alert_configured``, ``kill_switch_enabled``.

        Returns
        -------
        TaskRiskProfile
        """
        dimension_scores: Dict[str, float] = {}
        details: Dict[str, Any] = {}

        dimension_scores["fairness_risk"], details["fairness_risk"] = (
            self._score_fairness(fairness_metrics or {})
        )
        dimension_scores["transparency_risk"], details["transparency_risk"] = (
            self._score_transparency(transparency_metrics or {})
        )
        dimension_scores["accuracy_risk"], details["accuracy_risk"] = (
            self._score_accuracy(accuracy_metrics or {})
        )
        dimension_scores["security_risk"], details["security_risk"] = (
            self._score_security(security_metrics or {})
        )
        dimension_scores["oversight_risk"], details["oversight_risk"] = (
            self._score_oversight(oversight_metrics or {})
        )

        # Weighted aggregate
        aggregate = sum(
            dimension_scores[d] * self.dimension_weights.get(d, 0.0)
            for d in ASSESSMENT_DIMENSIONS
        )
        aggregate = round(min(max(aggregate, 0.0), 1.0), 4)

        risk_level = self._classify_risk(aggregate)
        remediations = self._generate_remediations(dimension_scores)

        profile = TaskRiskProfile(
            task_name=task_name,
            dimension_scores={k: round(v, 4) for k, v in dimension_scores.items()},
            aggregate_score=aggregate,
            risk_level=risk_level,
            remediations=remediations,
            details=details,
        )

        logger.info(
            "FRIA task assessment: %s -> %s (score=%.4f)",
            task_name, risk_level, aggregate,
        )
        return profile

    # ------------------------------------------------------------------
    # Full report generation
    # ------------------------------------------------------------------

    def generate_report(
        self,
        task_assessments: Optional[List[Dict[str, Any]]] = None,
        task_profiles: Optional[List[TaskRiskProfile]] = None,
    ) -> FRIAReport:
        """Generate a comprehensive FRIA report.

        Accepts either pre-computed ``task_profiles`` or raw
        ``task_assessments`` (list of dicts with keys matching
        :meth:`assess_task` parameters).

        Parameters
        ----------
        task_assessments : list of dict, optional
            Raw assessment inputs; each dict must contain ``task_name``
            and optionally the ``*_metrics`` dicts.
        task_profiles : list of TaskRiskProfile, optional
            Pre-computed profiles (takes precedence).

        Returns
        -------
        FRIAReport
        """
        if task_profiles is None:
            task_profiles = []
            for assessment in (task_assessments or []):
                profile = self.assess_task(
                    task_name=assessment.get("task_name", "unknown"),
                    fairness_metrics=assessment.get("fairness_metrics"),
                    transparency_metrics=assessment.get("transparency_metrics"),
                    accuracy_metrics=assessment.get("accuracy_metrics"),
                    security_metrics=assessment.get("security_metrics"),
                    oversight_metrics=assessment.get("oversight_metrics"),
                )
                task_profiles.append(profile)

        # Overall risk is the worst-case across tasks
        if task_profiles:
            overall_score = round(max(p.aggregate_score for p in task_profiles), 4)
        else:
            overall_score = 0.0

        overall_risk_level = self._classify_risk(overall_score)
        summary = self._build_summary(task_profiles, overall_risk_level)

        report = FRIAReport(
            report_id=f"FRIA-{datetime.now(timezone.utc).strftime('%Y%m%d')}-{uuid.uuid4().hex[:8]}",
            assessed_at=datetime.now(timezone.utc).isoformat(),
            system_name=self.system_name,
            assessor=self.assessor,
            task_profiles=task_profiles,
            overall_risk_level=overall_risk_level,
            overall_score=overall_score,
            summary=summary,
            regulatory_references=[
                "EU AI Act Article 9 - Risk Management System",
                "EU AI Act Article 6 - Classification Rules for High-Risk AI",
                "EU AI Act Annex III - High-Risk AI Systems (Financial Services)",
            ],
        )

        # Persist to audit store if available
        if self.audit_store is not None:
            self._persist_to_audit_store(report)

        logger.info(
            "FRIA report generated: %s (overall=%s, score=%.4f, tasks=%d)",
            report.report_id, overall_risk_level, overall_score, len(task_profiles),
        )
        return report

    def to_dict(self, report: FRIAReport) -> Dict[str, Any]:
        """Serialize a ``FRIAReport`` to a JSON-compatible dict."""
        return {
            "report_id": report.report_id,
            "assessed_at": report.assessed_at,
            "system_name": report.system_name,
            "assessor": report.assessor,
            "overall_risk_level": report.overall_risk_level,
            "overall_score": report.overall_score,
            "summary": report.summary,
            "regulatory_references": report.regulatory_references,
            "task_profiles": [
                {
                    "task_name": p.task_name,
                    "dimension_scores": p.dimension_scores,
                    "aggregate_score": p.aggregate_score,
                    "risk_level": p.risk_level,
                    "remediations": p.remediations,
                    "details": p.details,
                }
                for p in report.task_profiles
            ],
            "metadata": report.metadata,
        }

    # ------------------------------------------------------------------
    # Dimension scoring helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _score_fairness(metrics: Dict[str, Any]) -> tuple[float, Dict[str, Any]]:
        """Score fairness risk from violation counts and metric values."""
        if not metrics:
            return 0.5, {"reason": "No fairness data provided; conservative default."}

        violations = metrics.get("violation_count", 0)
        di = metrics.get("di_value", 1.0)

        score = 0.0
        if violations > 0:
            score += min(violations * 0.15, 0.6)
        if di < 0.8 or di > 1.25:
            score += 0.3
        if metrics.get("spd_value") is not None and abs(metrics["spd_value"]) > 0.1:
            score += 0.1

        score = min(score, 1.0)
        return score, {"violation_count": violations, "di_value": di, "computed_score": round(score, 4)}

    @staticmethod
    def _score_transparency(metrics: Dict[str, Any]) -> tuple[float, Dict[str, Any]]:
        """Score transparency risk from explanation availability."""
        if not metrics:
            return 0.5, {"reason": "No transparency data provided; conservative default."}

        score = 0.5  # Start at moderate risk
        has_explanations = metrics.get("has_explanations", False)
        coverage = metrics.get("explanation_coverage", 0.0)
        model_card = metrics.get("model_card_exists", False)

        if has_explanations:
            score -= 0.2
        if coverage > 0.8:
            score -= 0.15
        elif coverage > 0.5:
            score -= 0.1
        if model_card:
            score -= 0.1

        score = min(max(score, 0.0), 1.0)
        return score, {
            "has_explanations": has_explanations,
            "explanation_coverage": coverage,
            "model_card_exists": model_card,
            "computed_score": round(score, 4),
        }

    @staticmethod
    def _score_accuracy(metrics: Dict[str, Any]) -> tuple[float, Dict[str, Any]]:
        """Score accuracy risk from performance metrics and drift status."""
        if not metrics:
            return 0.5, {"reason": "No accuracy data provided; conservative default."}

        score = 0.0
        auc = metrics.get("auc", 0.8)
        drift = metrics.get("drift_detected", False)
        degradation = metrics.get("performance_degradation", 0.0)

        # Lower AUC = higher risk
        if auc < 0.6:
            score += 0.5
        elif auc < 0.7:
            score += 0.3
        elif auc < 0.8:
            score += 0.15

        if drift:
            score += 0.25
        score += min(degradation, 0.3)

        score = min(score, 1.0)
        return score, {
            "auc": auc,
            "drift_detected": drift,
            "degradation": degradation,
            "computed_score": round(score, 4),
        }

    @staticmethod
    def _score_security(metrics: Dict[str, Any]) -> tuple[float, Dict[str, Any]]:
        """Score security risk from infrastructure controls."""
        if not metrics:
            return 0.5, {"reason": "No security data provided; conservative default."}

        score = 0.6  # Start at moderate-high if not all controls present
        if metrics.get("input_validation", False):
            score -= 0.2
        if metrics.get("encryption_enabled", False):
            score -= 0.2
        acl = metrics.get("access_control_level", "none")
        if acl == "strict":
            score -= 0.15
        elif acl == "moderate":
            score -= 0.1

        score = min(max(score, 0.0), 1.0)
        return score, {
            "input_validation": metrics.get("input_validation", False),
            "encryption_enabled": metrics.get("encryption_enabled", False),
            "access_control_level": acl,
            "computed_score": round(score, 4),
        }

    @staticmethod
    def _score_oversight(metrics: Dict[str, Any]) -> tuple[float, Dict[str, Any]]:
        """Score oversight risk from human-in-the-loop controls."""
        if not metrics:
            return 0.5, {"reason": "No oversight data provided; conservative default."}

        score = 0.6
        review_rate = metrics.get("human_review_rate", 0.0)
        alert = metrics.get("alert_configured", False)
        kill_switch = metrics.get("kill_switch_enabled", False)

        # Higher review rate = lower risk
        score -= min(review_rate * 0.3, 0.3)
        if alert:
            score -= 0.15
        if kill_switch:
            score -= 0.15

        score = min(max(score, 0.0), 1.0)
        return score, {
            "human_review_rate": review_rate,
            "alert_configured": alert,
            "kill_switch_enabled": kill_switch,
            "computed_score": round(score, 4),
        }

    # ------------------------------------------------------------------
    # Classification and remediation
    # ------------------------------------------------------------------

    def _classify_risk(self, score: float) -> str:
        """Map an aggregate score to an EU AI Act risk level."""
        for level in ("UNACCEPTABLE", "HIGH", "LIMITED"):
            if score >= self.risk_thresholds[level]:
                return level
        return "MINIMAL"

    @staticmethod
    def _generate_remediations(dimension_scores: Dict[str, float]) -> List[str]:
        """Generate remediation recommendations based on dimension scores."""
        remediations: List[str] = []

        for dimension, score in dimension_scores.items():
            if score >= 0.9:
                level = "UNACCEPTABLE"
            elif score >= 0.7:
                level = "HIGH"
            elif score >= 0.4:
                level = "LIMITED"
            else:
                continue  # No remediation needed for MINIMAL

            suggestion = _REMEDIATION_MAP.get(dimension, {}).get(level)
            if suggestion:
                remediations.append(f"[{dimension}] {suggestion}")

        return remediations

    # ------------------------------------------------------------------
    # Summary and persistence
    # ------------------------------------------------------------------

    @staticmethod
    def _build_summary(
        profiles: List[TaskRiskProfile],
        overall_level: str,
    ) -> str:
        """Build an executive summary for the FRIA report."""
        if not profiles:
            return "No tasks assessed. FRIA evaluation requires at least one task profile."

        n_tasks = len(profiles)
        level_counts: Dict[str, int] = {}
        for p in profiles:
            level_counts[p.risk_level] = level_counts.get(p.risk_level, 0) + 1

        parts = [
            f"FRIA assessment completed for {n_tasks} task(s).",
            f"Overall system risk classification: {overall_level}.",
        ]

        for level in ("UNACCEPTABLE", "HIGH", "LIMITED", "MINIMAL"):
            count = level_counts.get(level, 0)
            if count > 0:
                parts.append(f"{count} task(s) classified as {level}.")

        total_remediations = sum(len(p.remediations) for p in profiles)
        if total_remediations > 0:
            parts.append(f"{total_remediations} remediation action(s) recommended.")

        return " ".join(parts)

    def _persist_to_audit_store(self, report: FRIAReport) -> None:
        """Write FRIA results to the compliance audit store."""
        try:
            self.audit_store.log_incident(
                incident_id=report.report_id,
                event_type="fria_assessment",
                severity="minor" if report.overall_risk_level == "MINIMAL" else "major",
                source_module="fria_evaluator",
                status="completed",
                description=report.summary,
            )
            logger.info("FRIA report persisted to audit store: %s", report.report_id)
        except Exception as exc:
            logger.warning("Failed to persist FRIA report: %s", exc)


__all__ = ["FRIAEvaluator", "FRIAReport", "TaskRiskProfile"]
