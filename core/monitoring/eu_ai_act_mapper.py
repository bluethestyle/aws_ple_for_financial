"""
EU AI Act compliance mapper.

Maps ML system characteristics to EU AI Act risk levels and provides
article-by-article compliance checklists with automated remediation
suggestions.

Integrates with the existing compliance checker
(:class:`~core.monitoring.compliance_checker.ComplianceChecker`) to extend
coverage with EU AI Act-specific requirements.
"""

from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# EU AI Act risk classification
# ---------------------------------------------------------------------------

EU_AI_ACT_RISK_LEVELS: List[str] = [
    "UNACCEPTABLE",  # Article 5 - Prohibited practices
    "HIGH",          # Article 6 / Annex III - High-risk AI
    "LIMITED",       # Article 50 - Transparency obligations
    "MINIMAL",       # No specific obligations
]

# Financial services high-risk indicators (Annex III, Section 5)
FINANCIAL_HIGH_RISK_INDICATORS: Dict[str, str] = {
    "creditworthiness": "Credit scoring and creditworthiness assessment",
    "risk_pricing": "Risk assessment and pricing for insurance/financial products",
    "fraud_detection": "Fraud detection in financial services",
    "credit_scoring": "Evaluation of credit score or establishment of credit rating",
    "insurance_pricing": "Risk assessment and pricing for life and health insurance",
}


# ---------------------------------------------------------------------------
# Article compliance checklist
# ---------------------------------------------------------------------------

@dataclass
class ArticleComplianceItem:
    """A single EU AI Act article compliance check."""

    article_id: str
    article_title: str
    description: str
    requirement: str
    is_compliant: bool = False
    evidence: str = ""
    remediation: str = ""
    priority: str = "medium"  # "critical" | "high" | "medium" | "low"


@dataclass
class EUAIActComplianceReport:
    """Container for an EU AI Act compliance assessment."""

    report_id: str
    assessed_at: str
    system_name: str
    risk_classification: str
    classification_rationale: str
    article_checklist: List[ArticleComplianceItem]
    compliance_rate: float
    compliant_count: int
    non_compliant_count: int
    remediations: List[Dict[str, Any]]
    summary: str
    metadata: Dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Full article checklist template (for high-risk financial AI)
# ---------------------------------------------------------------------------

_HIGH_RISK_CHECKLIST: List[Dict[str, Any]] = [
    {
        "article_id": "Art.9(1)",
        "article_title": "Risk Management System",
        "description": "Establish and maintain a risk management system throughout the AI lifecycle.",
        "requirement": "Documented risk management process with identification, analysis, and mitigation.",
        "priority": "critical",
    },
    {
        "article_id": "Art.9(2)",
        "article_title": "Risk Identification",
        "description": "Identify and analyse known and reasonably foreseeable risks.",
        "requirement": "Risk register with identified risks, likelihood, and impact assessments.",
        "priority": "critical",
    },
    {
        "article_id": "Art.9(4)",
        "article_title": "Risk Mitigation",
        "description": "Adopt suitable risk management measures for identified risks.",
        "requirement": "Implemented controls for each identified risk with evidence of effectiveness.",
        "priority": "critical",
    },
    {
        "article_id": "Art.10(1)",
        "article_title": "Data Governance",
        "description": "Training, validation, and testing data shall be subject to data governance.",
        "requirement": "Data governance framework covering quality, representativeness, and bias.",
        "priority": "critical",
    },
    {
        "article_id": "Art.10(2)",
        "article_title": "Data Quality",
        "description": "Training data shall be relevant, representative, and free of errors.",
        "requirement": "Data quality metrics, validation pipeline, and bias assessment.",
        "priority": "high",
    },
    {
        "article_id": "Art.10(5)",
        "article_title": "Bias Examination",
        "description": "Examine data for possible biases relevant to the intended purpose.",
        "requirement": "Documented bias analysis across protected attributes.",
        "priority": "high",
    },
    {
        "article_id": "Art.11(1)",
        "article_title": "Technical Documentation",
        "description": "Technical documentation shall be drawn up before placing on the market.",
        "requirement": "Model cards, system architecture documentation, and performance reports.",
        "priority": "high",
    },
    {
        "article_id": "Art.12(1)",
        "article_title": "Record-Keeping",
        "description": "High-risk AI shall allow automatic recording of events (logs).",
        "requirement": "Audit logging with tamper-evident records and configurable retention.",
        "priority": "critical",
    },
    {
        "article_id": "Art.13(1)",
        "article_title": "Transparency",
        "description": "Designed and developed to ensure sufficiently transparent operation.",
        "requirement": "User-facing documentation and explanation of AI system behaviour.",
        "priority": "high",
    },
    {
        "article_id": "Art.14(1)",
        "article_title": "Human Oversight",
        "description": "Designed to allow effective oversight by natural persons.",
        "requirement": "Human-in-the-loop controls, kill-switch, and override mechanisms.",
        "priority": "critical",
    },
    {
        "article_id": "Art.14(4)",
        "article_title": "Override Capability",
        "description": "Human overseer can intervene or interrupt the AI system.",
        "requirement": "Kill-switch and manual override functionality with audit trail.",
        "priority": "critical",
    },
    {
        "article_id": "Art.15(1)",
        "article_title": "Accuracy",
        "description": "Achieve appropriate level of accuracy for the intended purpose.",
        "requirement": "Performance benchmarks, validation metrics, and monitoring.",
        "priority": "high",
    },
    {
        "article_id": "Art.15(3)",
        "article_title": "Robustness",
        "description": "Resilient against errors, faults, and adversarial manipulation.",
        "requirement": "Adversarial testing, drift monitoring, and fallback mechanisms.",
        "priority": "high",
    },
    {
        "article_id": "Art.15(4)",
        "article_title": "Cybersecurity",
        "description": "Appropriate level of cybersecurity throughout the lifecycle.",
        "requirement": "Security controls, access management, and vulnerability assessments.",
        "priority": "high",
    },
    {
        "article_id": "Art.17(1)",
        "article_title": "Quality Management",
        "description": "Providers shall put a quality management system in place.",
        "requirement": "QMS covering development, testing, deployment, and monitoring processes.",
        "priority": "high",
    },
    {
        "article_id": "Art.61(1)",
        "article_title": "Post-Market Monitoring",
        "description": "Establish and document a post-market monitoring system.",
        "requirement": "Continuous monitoring with incident detection and reporting.",
        "priority": "high",
    },
    {
        "article_id": "Art.62(1)",
        "article_title": "Incident Reporting",
        "description": "Report serious incidents to the relevant market surveillance authority.",
        "requirement": "Incident reporting process with severity classification.",
        "priority": "critical",
    },
]


# ---------------------------------------------------------------------------
# EUAIActMapper
# ---------------------------------------------------------------------------

class EUAIActMapper:
    """Map ML system characteristics to EU AI Act compliance requirements.

    Provides risk classification, article-by-article compliance checklists,
    and automated remediation suggestions for financial AI systems.

    Parameters
    ----------
    system_name : str
        Name of the ML system.
    system_purpose : str
        Description of the system's intended purpose.
    domain : str
        Application domain (default ``"financial_services"``).
    existing_compliance : dict, optional
        Results from the existing 36-item compliance checker to integrate.
    """

    def __init__(
        self,
        system_name: str = "PLE-Cluster-adaTT",
        system_purpose: str = "Multi-task recommendation system for financial products",
        domain: str = "financial_services",
        existing_compliance: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.system_name = system_name
        self.system_purpose = system_purpose
        self.domain = domain
        self.existing_compliance = existing_compliance or {}

    # ------------------------------------------------------------------
    # Risk classification
    # ------------------------------------------------------------------

    def classify_risk(
        self,
        system_characteristics: Optional[Dict[str, Any]] = None,
    ) -> tuple[str, str]:
        """Classify the system's EU AI Act risk level.

        Parameters
        ----------
        system_characteristics : dict, optional
            Keys: ``use_cases`` (list of str matching FINANCIAL_HIGH_RISK_INDICATORS),
            ``affects_individuals``, ``autonomous_decisions``,
            ``manipulative_techniques``, ``social_scoring``.

        Returns
        -------
        tuple of (str, str)
            ``(risk_level, rationale)``
        """
        chars = system_characteristics or {}
        use_cases = chars.get("use_cases", [])

        # Check for prohibited practices (Article 5)
        if chars.get("manipulative_techniques", False):
            return "UNACCEPTABLE", "System employs manipulative techniques (Art. 5(1)(a))."
        if chars.get("social_scoring", False):
            return "UNACCEPTABLE", "System performs social scoring (Art. 5(1)(c))."

        # Check for high-risk indicators in financial services
        high_risk_matches = [
            uc for uc in use_cases if uc in FINANCIAL_HIGH_RISK_INDICATORS
        ]
        if high_risk_matches:
            reasons = [FINANCIAL_HIGH_RISK_INDICATORS[uc] for uc in high_risk_matches]
            return "HIGH", (
                f"High-risk AI per Annex III Section 5: {'; '.join(reasons)}."
            )

        # Financial services with individual impact defaults to HIGH
        if self.domain == "financial_services" and chars.get("affects_individuals", True):
            return "HIGH", (
                "Financial services AI affecting individuals defaults to HIGH risk "
                "per Annex III Section 5 interpretation."
            )

        # Autonomous decisions = at least LIMITED
        if chars.get("autonomous_decisions", False):
            return "LIMITED", "System makes autonomous decisions requiring transparency (Art. 52)."

        return "MINIMAL", "System does not fall into higher risk categories."

    # ------------------------------------------------------------------
    # Article checklist evaluation
    # ------------------------------------------------------------------

    def evaluate_compliance(
        self,
        evidence: Optional[Dict[str, Dict[str, Any]]] = None,
    ) -> List[ArticleComplianceItem]:
        """Evaluate compliance against the EU AI Act article checklist.

        Parameters
        ----------
        evidence : dict, optional
            Mapping of ``article_id`` -> ``{"is_compliant": bool, "evidence": str}``.

        Returns
        -------
        list of ArticleComplianceItem
        """
        evidence = evidence or {}
        checklist: List[ArticleComplianceItem] = []

        for template in _HIGH_RISK_CHECKLIST:
            article_id = template["article_id"]
            ev = evidence.get(article_id, {})

            is_compliant = ev.get("is_compliant", False)
            evidence_text = ev.get("evidence", "")
            remediation = ""

            if not is_compliant:
                remediation = self._suggest_remediation(article_id, template)

            item = ArticleComplianceItem(
                article_id=article_id,
                article_title=template["article_title"],
                description=template["description"],
                requirement=template["requirement"],
                is_compliant=is_compliant,
                evidence=evidence_text,
                remediation=remediation,
                priority=template.get("priority", "medium"),
            )
            checklist.append(item)

        return checklist

    # ------------------------------------------------------------------
    # Full report generation
    # ------------------------------------------------------------------

    def generate_report(
        self,
        system_characteristics: Optional[Dict[str, Any]] = None,
        evidence: Optional[Dict[str, Dict[str, Any]]] = None,
    ) -> EUAIActComplianceReport:
        """Generate a comprehensive EU AI Act compliance report.

        Parameters
        ----------
        system_characteristics : dict, optional
            System characteristics for risk classification.
        evidence : dict, optional
            Compliance evidence keyed by article ID.

        Returns
        -------
        EUAIActComplianceReport
        """
        risk_level, rationale = self.classify_risk(system_characteristics)
        checklist = self.evaluate_compliance(evidence)

        compliant_count = sum(1 for item in checklist if item.is_compliant)
        non_compliant_count = len(checklist) - compliant_count
        total = len(checklist)
        compliance_rate = round(compliant_count / total, 4) if total > 0 else 0.0

        remediations = [
            {
                "article_id": item.article_id,
                "article_title": item.article_title,
                "priority": item.priority,
                "remediation": item.remediation,
            }
            for item in checklist
            if not item.is_compliant and item.remediation
        ]

        # Sort remediations by priority
        priority_order = {"critical": 0, "high": 1, "medium": 2, "low": 3}
        remediations.sort(key=lambda r: priority_order.get(r["priority"], 99))

        summary = self._build_summary(
            risk_level, compliance_rate, compliant_count, non_compliant_count, remediations,
        )

        report = EUAIActComplianceReport(
            report_id=f"EUAIA-{datetime.now(timezone.utc).strftime('%Y%m%d')}-{uuid.uuid4().hex[:8]}",
            assessed_at=datetime.now(timezone.utc).isoformat(),
            system_name=self.system_name,
            risk_classification=risk_level,
            classification_rationale=rationale,
            article_checklist=checklist,
            compliance_rate=compliance_rate,
            compliant_count=compliant_count,
            non_compliant_count=non_compliant_count,
            remediations=remediations,
            summary=summary,
        )

        logger.info(
            "EU AI Act report generated: %s (risk=%s, compliance=%.1f%%)",
            report.report_id, risk_level, compliance_rate * 100,
        )
        return report

    def to_dict(self, report: EUAIActComplianceReport) -> Dict[str, Any]:
        """Serialize an ``EUAIActComplianceReport`` to a JSON-compatible dict."""
        return {
            "report_id": report.report_id,
            "assessed_at": report.assessed_at,
            "system_name": report.system_name,
            "risk_classification": report.risk_classification,
            "classification_rationale": report.classification_rationale,
            "compliance_rate": report.compliance_rate,
            "compliant_count": report.compliant_count,
            "non_compliant_count": report.non_compliant_count,
            "summary": report.summary,
            "article_checklist": [
                {
                    "article_id": item.article_id,
                    "article_title": item.article_title,
                    "description": item.description,
                    "requirement": item.requirement,
                    "is_compliant": item.is_compliant,
                    "evidence": item.evidence,
                    "remediation": item.remediation,
                    "priority": item.priority,
                }
                for item in report.article_checklist
            ],
            "remediations": report.remediations,
            "metadata": report.metadata,
        }

    # ------------------------------------------------------------------
    # Integration with existing compliance checker
    # ------------------------------------------------------------------

    def integrate_existing_compliance(
        self,
        checker_results: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Map results from the 36-item compliance checker to EU AI Act articles.

        Parameters
        ----------
        checker_results : dict
            Output from :meth:`ComplianceChecker.run_full_check`.

        Returns
        -------
        dict
            Mapping of EU AI Act article IDs to existing compliance status.
        """
        items = checker_results.get("items", {})
        mapping: Dict[str, Any] = {}

        # Map existing compliance items to EU AI Act articles
        _ITEM_TO_ARTICLE: Dict[str, str] = {
            "MON-01": "Art.12(1)",   # Audit logger -> Record-Keeping
            "MON-02": "Art.61(1)",   # Compliance store -> Post-Market Monitoring
            "MON-03": "Art.10(5)",   # Fairness monitoring -> Bias Examination
            "MON-05": "Art.15(3)",   # Herding detection -> Robustness
            "MON-06": "Art.15(1)",   # Drift detection -> Accuracy
            "MON-07": "Art.62(1)",   # Incident management -> Incident Reporting
            "MON-08": "Art.17(1)",   # Governance report -> Quality Management
            "MON-09": "Art.10(1)",   # Data lineage -> Data Governance
        }

        for item_id, article_id in _ITEM_TO_ARTICLE.items():
            item_data = items.get(item_id, {})
            status = item_data.get("status", "unknown")
            mapping[article_id] = {
                "is_compliant": status == "compliant",
                "evidence": f"Existing compliance item {item_id}: {item_data.get('description', '')} -> {status}",
                "source_item": item_id,
            }

        return mapping

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _suggest_remediation(article_id: str, template: Dict[str, Any]) -> str:
        """Generate a remediation suggestion for a non-compliant article."""
        _REMEDIATION_SUGGESTIONS: Dict[str, str] = {
            "Art.9(1)": "Establish a documented risk management system covering the full AI lifecycle. Include risk identification, assessment, and mitigation procedures.",
            "Art.9(2)": "Create a risk register documenting all known and foreseeable risks with likelihood and impact assessments.",
            "Art.9(4)": "Implement and evidence risk mitigation controls for each identified risk in the risk register.",
            "Art.10(1)": "Implement data governance framework covering training data quality, representativeness, and lineage tracking.",
            "Art.10(2)": "Establish data quality pipeline with automated validation, deduplication, and error correction.",
            "Art.10(5)": "Conduct bias analysis across protected attributes using DI/SPD/EOD metrics; document results.",
            "Art.11(1)": "Create technical documentation package: model cards, architecture docs, performance benchmarks.",
            "Art.12(1)": "Deploy audit logging with HMAC signing, hash-chain integrity, and WORM storage.",
            "Art.13(1)": "Publish user-facing transparency notices explaining system behaviour and limitations.",
            "Art.14(1)": "Implement human-in-the-loop review workflow for high-impact decisions.",
            "Art.14(4)": "Deploy kill-switch and manual override mechanisms with audit trail.",
            "Art.15(1)": "Establish performance monitoring with automated drift detection and retraining triggers.",
            "Art.15(3)": "Implement adversarial testing, robustness evaluation, and fallback mechanisms.",
            "Art.15(4)": "Conduct security assessment; implement WAF, encryption, and access controls.",
            "Art.17(1)": "Establish quality management system covering development, testing, and deployment.",
            "Art.61(1)": "Deploy continuous post-market monitoring with automated alerting and reporting.",
            "Art.62(1)": "Implement incident management with severity classification and regulatory notification workflow.",
        }
        return _REMEDIATION_SUGGESTIONS.get(
            article_id,
            f"Address requirement: {template.get('requirement', 'See article text.')}",
        )

    @staticmethod
    def _build_summary(
        risk_level: str,
        compliance_rate: float,
        compliant: int,
        non_compliant: int,
        remediations: List[Dict[str, Any]],
    ) -> str:
        """Build an executive summary for the EU AI Act compliance report."""
        parts = [f"EU AI Act compliance assessment completed."]
        parts.append(f"System classified as {risk_level} risk.")
        parts.append(
            f"Compliance rate: {compliance_rate * 100:.1f}% "
            f"({compliant} compliant, {non_compliant} non-compliant)."
        )

        critical_rems = [r for r in remediations if r["priority"] == "critical"]
        if critical_rems:
            parts.append(
                f"{len(critical_rems)} critical remediation(s) required before deployment."
            )

        if compliance_rate >= 0.9:
            parts.append("System is substantially compliant.")
        elif compliance_rate >= 0.7:
            parts.append("Significant compliance gaps remain; targeted remediation needed.")
        else:
            parts.append("Major compliance gaps identified; comprehensive remediation required.")

        return " ".join(parts)


__all__ = [
    "EUAIActMapper",
    "EUAIActComplianceReport",
    "ArticleComplianceItem",
    "FINANCIAL_HIGH_RISK_INDICATORS",
]
