"""
Privacy Impact Assessment (PIA) evaluator aligned with GDPR Article 35 and
Korean Personal Information Protection Act (개인정보보호법).

Performs structured privacy risk assessment across six domains:

- **Data collection**       Scope and lawfulness of data gathering
- **Data processing**       Purpose limitation and processing safeguards
- **Data storage**          Retention policies and encryption
- **Data sharing**          Third-party transfers and access controls
- **Data minimization**     Necessity and proportionality analysis
- **Cross-border transfer** Multi-region data flow risks (AWS-specific)

Results are captured in a :class:`PIAReport` dataclass and can optionally be
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
# Risk level definitions
# ---------------------------------------------------------------------------

PIA_RISK_LEVELS: List[str] = ["CRITICAL", "HIGH", "MODERATE", "LOW"]

PIA_ASSESSMENT_DOMAINS: List[str] = [
    "data_collection",
    "data_processing",
    "data_storage",
    "data_sharing",
    "data_minimization",
    "cross_border_transfer",
]


# ---------------------------------------------------------------------------
# PII categories (aligned with existing DataValidator PII detection)
# ---------------------------------------------------------------------------

PII_CATEGORIES: Dict[str, str] = {
    "direct_identifier": "Name, SSN, passport number, etc.",
    "quasi_identifier": "Age, gender, zip code, etc.",
    "sensitive_data": "Health, biometric, financial, racial/ethnic data",
    "online_identifier": "IP address, cookie ID, device fingerprint",
    "financial_data": "Account numbers, transaction history, credit scores",
}


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class PIIInventoryItem:
    """A single PII field in the data inventory."""

    field_name: str
    pii_category: str  # Key from PII_CATEGORIES
    sensitivity_level: str  # "high" | "medium" | "low"
    purpose: str
    retention_days: int
    encrypted: bool
    anonymized: bool


@dataclass
class DomainRiskAssessment:
    """Risk assessment for a single PIA domain."""

    domain: str
    score: float  # 0.0 (no risk) to 1.0 (critical risk)
    risk_level: str  # CRITICAL | HIGH | MODERATE | LOW
    findings: List[str]
    recommendations: List[str]


@dataclass
class PIAReport:
    """Container for a full PIA evaluation report."""

    report_id: str
    assessed_at: str  # ISO datetime
    system_name: str
    assessor: str
    pii_inventory: List[PIIInventoryItem]
    domain_assessments: List[DomainRiskAssessment]
    overall_risk_level: str
    overall_score: float
    consent_adequacy: Dict[str, Any]
    data_minimization_score: float
    summary: str
    regulatory_references: List[str]
    metadata: Dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# PIAEvaluator
# ---------------------------------------------------------------------------

class PIAEvaluator:
    """Privacy Impact Assessment evaluator.

    Assesses privacy risks for ML systems handling personal data, aligned
    with GDPR Article 35 (DPIA) and Korean PIPA requirements.

    Parameters
    ----------
    system_name : str
        Name of the ML system under assessment.
    assessor : str
        Identity of the assessor.
    data_regions : list of str, optional
        AWS regions where data is stored/processed (for cross-border analysis).
    audit_store : ComplianceAuditStore, optional
        If provided, PIA results are persisted to the compliance audit store.
    """

    # Regions within the same jurisdiction for cross-border analysis
    _JURISDICTION_MAP: Dict[str, str] = {
        "ap-northeast-2": "KR",
        "ap-northeast-1": "JP",
        "ap-northeast-3": "JP",
        "eu-west-1": "EU",
        "eu-west-2": "EU",
        "eu-central-1": "EU",
        "us-east-1": "US",
        "us-west-2": "US",
    }

    def __init__(
        self,
        system_name: str = "PLE-Cluster-adaTT",
        assessor: str = "automated",
        data_regions: Optional[List[str]] = None,
        audit_store: Optional[Any] = None,
    ) -> None:
        self.system_name = system_name
        self.assessor = assessor
        # data_regions has no hardcoded AWS default per CLAUDE.md §1.1 —
        # callers must pass the deployment region list explicitly (derived
        # from pipeline.yaml::aws.region).
        self.data_regions = data_regions or []
        self.audit_store = audit_store

    # ------------------------------------------------------------------
    # PII inventory analysis
    # ------------------------------------------------------------------

    def analyze_pii_inventory(
        self,
        fields: List[Dict[str, Any]],
    ) -> List[PIIInventoryItem]:
        """Build a PII inventory from field definitions.

        Parameters
        ----------
        fields : list of dict
            Each dict should contain: ``field_name``, ``pii_category``,
            ``sensitivity_level``, ``purpose``, ``retention_days``,
            ``encrypted``, ``anonymized``.

        Returns
        -------
        list of PIIInventoryItem
        """
        inventory: List[PIIInventoryItem] = []
        for f in fields:
            item = PIIInventoryItem(
                field_name=f.get("field_name", "unknown"),
                pii_category=f.get("pii_category", "quasi_identifier"),
                sensitivity_level=f.get("sensitivity_level", "medium"),
                purpose=f.get("purpose", "unspecified"),
                retention_days=f.get("retention_days", 365),
                encrypted=f.get("encrypted", False),
                anonymized=f.get("anonymized", False),
            )
            inventory.append(item)

        logger.info("PII inventory analyzed: %d field(s)", len(inventory))
        return inventory

    # ------------------------------------------------------------------
    # Domain assessments
    # ------------------------------------------------------------------

    def assess_data_collection(
        self,
        collection_config: Dict[str, Any],
    ) -> DomainRiskAssessment:
        """Assess risks related to data collection practices.

        Parameters
        ----------
        collection_config : dict
            Keys: ``lawful_basis``, ``consent_obtained``, ``purpose_specified``,
            ``data_sources_documented``, ``collection_scope`` (narrow/moderate/broad).
        """
        findings: List[str] = []
        recommendations: List[str] = []
        score = 0.0

        if not collection_config.get("lawful_basis"):
            findings.append("No lawful basis documented for data collection.")
            recommendations.append("Document lawful basis per GDPR Article 6 / PIPA Article 15.")
            score += 0.35

        if not collection_config.get("consent_obtained", False):
            findings.append("Explicit consent not obtained from data subjects.")
            recommendations.append("Implement consent collection mechanism.")
            score += 0.25

        if not collection_config.get("purpose_specified", False):
            findings.append("Collection purpose not clearly specified.")
            recommendations.append("Document specific, explicit purposes for each data type.")
            score += 0.15

        if not collection_config.get("data_sources_documented", False):
            findings.append("Data sources not fully documented.")
            recommendations.append("Create comprehensive data source registry.")
            score += 0.1

        scope = collection_config.get("collection_scope", "broad")
        if scope == "broad":
            findings.append("Broad data collection scope increases privacy risk.")
            recommendations.append("Narrow collection scope to strictly necessary data.")
            score += 0.15

        score = min(score, 1.0)
        return DomainRiskAssessment(
            domain="data_collection",
            score=round(score, 4),
            risk_level=self._classify_domain_risk(score),
            findings=findings,
            recommendations=recommendations,
        )

    def assess_data_processing(
        self,
        processing_config: Dict[str, Any],
    ) -> DomainRiskAssessment:
        """Assess risks related to data processing practices.

        Parameters
        ----------
        processing_config : dict
            Keys: ``purpose_limitation``, ``automated_decision_making``,
            ``profiling_enabled``, ``opt_out_available``, ``processing_logged``.
        """
        findings: List[str] = []
        recommendations: List[str] = []
        score = 0.0

        if not processing_config.get("purpose_limitation", False):
            findings.append("Processing may exceed original collection purpose.")
            recommendations.append("Enforce purpose limitation controls.")
            score += 0.25

        if processing_config.get("automated_decision_making", False):
            score += 0.2
            findings.append("Automated decision-making in use.")
            if not processing_config.get("opt_out_available", False):
                findings.append("No opt-out mechanism for automated decisions.")
                recommendations.append(
                    "Implement opt-out per GDPR Article 22 / PIPA Article 37-2."
                )
                score += 0.2

        if processing_config.get("profiling_enabled", False):
            findings.append("Data subject profiling is enabled.")
            recommendations.append("Document profiling logic and provide transparency notices.")
            score += 0.15

        if not processing_config.get("processing_logged", False):
            findings.append("Data processing activities not logged.")
            recommendations.append("Enable audit logging for all processing operations.")
            score += 0.1

        score = min(score, 1.0)
        return DomainRiskAssessment(
            domain="data_processing",
            score=round(score, 4),
            risk_level=self._classify_domain_risk(score),
            findings=findings,
            recommendations=recommendations,
        )

    def assess_data_storage(
        self,
        storage_config: Dict[str, Any],
    ) -> DomainRiskAssessment:
        """Assess risks related to data storage practices.

        Parameters
        ----------
        storage_config : dict
            Keys: ``encryption_at_rest``, ``encryption_in_transit``,
            ``retention_policy_defined``, ``retention_days``,
            ``backup_encrypted``, ``access_logging``.
        """
        findings: List[str] = []
        recommendations: List[str] = []
        score = 0.0

        if not storage_config.get("encryption_at_rest", False):
            findings.append("Data not encrypted at rest.")
            recommendations.append("Enable S3 SSE-KMS or DynamoDB encryption at rest.")
            score += 0.3

        if not storage_config.get("encryption_in_transit", False):
            findings.append("Data not encrypted in transit.")
            recommendations.append("Enforce TLS for all data transfers.")
            score += 0.2

        if not storage_config.get("retention_policy_defined", False):
            findings.append("No data retention policy defined.")
            recommendations.append("Define retention policy aligned with regulatory requirements.")
            score += 0.2

        retention_days = storage_config.get("retention_days", 0)
        if retention_days > 730:
            findings.append(f"Retention period ({retention_days} days) exceeds recommended maximum.")
            recommendations.append("Review data retention necessity; consider lifecycle policies.")
            score += 0.15

        if not storage_config.get("access_logging", False):
            findings.append("Storage access logging not enabled.")
            recommendations.append("Enable AWS CloudTrail and S3 access logging.")
            score += 0.1

        score = min(score, 1.0)
        return DomainRiskAssessment(
            domain="data_storage",
            score=round(score, 4),
            risk_level=self._classify_domain_risk(score),
            findings=findings,
            recommendations=recommendations,
        )

    def assess_data_sharing(
        self,
        sharing_config: Dict[str, Any],
    ) -> DomainRiskAssessment:
        """Assess risks related to data sharing with third parties.

        Parameters
        ----------
        sharing_config : dict
            Keys: ``third_party_recipients``, ``data_sharing_agreements``,
            ``anonymization_before_sharing``, ``sharing_logged``.
        """
        findings: List[str] = []
        recommendations: List[str] = []
        score = 0.0

        recipients = sharing_config.get("third_party_recipients", [])
        agreements = sharing_config.get("data_sharing_agreements", False)

        if recipients:
            findings.append(f"Data shared with {len(recipients)} third-party recipient(s).")
            score += 0.15
            if not agreements:
                findings.append("No formal data sharing agreements in place.")
                recommendations.append("Establish DPA/DSA with all third-party recipients.")
                score += 0.3

        if not sharing_config.get("anonymization_before_sharing", False) and recipients:
            findings.append("Data not anonymized before sharing with third parties.")
            recommendations.append("Apply anonymization or pseudonymization before sharing.")
            score += 0.25

        if not sharing_config.get("sharing_logged", False) and recipients:
            findings.append("Data sharing activities not logged.")
            recommendations.append("Implement sharing audit trail.")
            score += 0.1

        score = min(score, 1.0)
        return DomainRiskAssessment(
            domain="data_sharing",
            score=round(score, 4),
            risk_level=self._classify_domain_risk(score),
            findings=findings,
            recommendations=recommendations,
        )

    def assess_cross_border_transfer(
        self,
        transfer_config: Optional[Dict[str, Any]] = None,
    ) -> DomainRiskAssessment:
        """Assess risks related to cross-border data transfers.

        Uses the configured ``data_regions`` to automatically detect
        cross-jurisdictional data flows.

        Parameters
        ----------
        transfer_config : dict, optional
            Keys: ``adequacy_decision``, ``standard_contractual_clauses``,
            ``binding_corporate_rules``, ``transfer_impact_assessed``.
        """
        transfer_config = transfer_config or {}
        findings: List[str] = []
        recommendations: List[str] = []
        score = 0.0

        # Determine jurisdictions involved
        jurisdictions = set()
        for region in self.data_regions:
            jurisdiction = self._JURISDICTION_MAP.get(region, "UNKNOWN")
            jurisdictions.add(jurisdiction)

        is_cross_border = len(jurisdictions) > 1

        if is_cross_border:
            findings.append(
                f"Cross-border data transfer detected across jurisdictions: "
                f"{', '.join(sorted(jurisdictions))}."
            )
            score += 0.3

            if not transfer_config.get("adequacy_decision", False):
                findings.append("No adequacy decision covers all target jurisdictions.")
                recommendations.append(
                    "Verify adequacy decisions or implement SCCs per GDPR Article 46."
                )
                score += 0.2

            if not transfer_config.get("standard_contractual_clauses", False):
                findings.append("Standard Contractual Clauses not in place.")
                recommendations.append("Execute SCCs with data importers.")
                score += 0.15

            if not transfer_config.get("transfer_impact_assessed", False):
                findings.append("Transfer Impact Assessment not conducted.")
                recommendations.append(
                    "Conduct Transfer Impact Assessment per EDPB recommendations."
                )
                score += 0.15
        else:
            findings.append(
                f"Data processing within single jurisdiction: "
                f"{', '.join(sorted(jurisdictions))}."
            )

        score = min(score, 1.0)
        return DomainRiskAssessment(
            domain="cross_border_transfer",
            score=round(score, 4),
            risk_level=self._classify_domain_risk(score),
            findings=findings,
            recommendations=recommendations,
        )

    # ------------------------------------------------------------------
    # Data minimization scoring
    # ------------------------------------------------------------------

    def compute_data_minimization_score(
        self,
        inventory: List[PIIInventoryItem],
    ) -> float:
        """Compute a data minimization score based on the PII inventory.

        Higher score (closer to 1.0) indicates better minimization practices.

        Parameters
        ----------
        inventory : list of PIIInventoryItem

        Returns
        -------
        float
            Score in [0.0, 1.0].
        """
        if not inventory:
            return 1.0  # No PII = perfectly minimized

        total_score = 0.0
        for item in inventory:
            item_score = 0.0
            # Anonymized data is best
            if item.anonymized:
                item_score += 0.5
            # Encrypted data is good
            if item.encrypted:
                item_score += 0.3
            # Shorter retention is better
            if item.retention_days <= 90:
                item_score += 0.2
            elif item.retention_days <= 365:
                item_score += 0.1
            total_score += item_score

        return round(total_score / len(inventory), 4)

    # ------------------------------------------------------------------
    # Consent adequacy check
    # ------------------------------------------------------------------

    @staticmethod
    def check_consent_adequacy(
        consent_config: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Evaluate whether consent mechanisms meet regulatory requirements.

        Parameters
        ----------
        consent_config : dict
            Keys: ``explicit_consent``, ``granular_consent``, ``withdrawable``,
            ``records_maintained``, ``age_verification``, ``plain_language``.

        Returns
        -------
        dict
            Assessment with ``is_adequate``, ``score``, ``gaps``.
        """
        checks = {
            "explicit_consent": consent_config.get("explicit_consent", False),
            "granular_consent": consent_config.get("granular_consent", False),
            "withdrawable": consent_config.get("withdrawable", False),
            "records_maintained": consent_config.get("records_maintained", False),
            "age_verification": consent_config.get("age_verification", False),
            "plain_language": consent_config.get("plain_language", False),
        }

        passed = sum(1 for v in checks.values() if v)
        total = len(checks)
        score = round(passed / total, 4) if total > 0 else 0.0

        gaps: List[str] = []
        if not checks["explicit_consent"]:
            gaps.append("Explicit consent mechanism required (GDPR Art. 7 / PIPA Art. 15).")
        if not checks["granular_consent"]:
            gaps.append("Granular per-purpose consent required (GDPR Art. 6(1)(a)).")
        if not checks["withdrawable"]:
            gaps.append("Consent withdrawal mechanism required (GDPR Art. 7(3)).")
        if not checks["records_maintained"]:
            gaps.append("Consent records must be maintained (GDPR Art. 7(1)).")
        if not checks["age_verification"]:
            gaps.append("Age verification for minors may be required (GDPR Art. 8).")
        if not checks["plain_language"]:
            gaps.append("Consent notices must use plain, clear language (GDPR Art. 7(2)).")

        return {
            "is_adequate": score >= 0.8,
            "score": score,
            "checks": checks,
            "gaps": gaps,
        }

    # ------------------------------------------------------------------
    # Full report generation
    # ------------------------------------------------------------------

    def generate_report(
        self,
        pii_fields: Optional[List[Dict[str, Any]]] = None,
        collection_config: Optional[Dict[str, Any]] = None,
        processing_config: Optional[Dict[str, Any]] = None,
        storage_config: Optional[Dict[str, Any]] = None,
        sharing_config: Optional[Dict[str, Any]] = None,
        transfer_config: Optional[Dict[str, Any]] = None,
        consent_config: Optional[Dict[str, Any]] = None,
    ) -> PIAReport:
        """Generate a comprehensive PIA report.

        Parameters
        ----------
        pii_fields : list of dict, optional
            PII field definitions for inventory analysis.
        collection_config : dict, optional
            Data collection assessment inputs.
        processing_config : dict, optional
            Data processing assessment inputs.
        storage_config : dict, optional
            Data storage assessment inputs.
        sharing_config : dict, optional
            Data sharing assessment inputs.
        transfer_config : dict, optional
            Cross-border transfer assessment inputs.
        consent_config : dict, optional
            Consent adequacy check inputs.

        Returns
        -------
        PIAReport
        """
        # PII inventory
        inventory = self.analyze_pii_inventory(pii_fields or [])

        # Domain assessments
        domain_assessments: List[DomainRiskAssessment] = [
            self.assess_data_collection(collection_config or {}),
            self.assess_data_processing(processing_config or {}),
            self.assess_data_storage(storage_config or {}),
            self.assess_data_sharing(sharing_config or {}),
            self.assess_cross_border_transfer(transfer_config),
        ]

        # Data minimization
        minimization_score = self.compute_data_minimization_score(inventory)

        # Add minimization as a domain assessment
        minimization_risk = round(1.0 - minimization_score, 4)
        min_findings = []
        min_recs = []
        if minimization_score < 0.5:
            min_findings.append("Data minimization practices below acceptable threshold.")
            min_recs.append("Review data collection for unnecessary PII fields.")
        domain_assessments.append(DomainRiskAssessment(
            domain="data_minimization",
            score=minimization_risk,
            risk_level=self._classify_domain_risk(minimization_risk),
            findings=min_findings,
            recommendations=min_recs,
        ))

        # Consent
        consent_result = self.check_consent_adequacy(consent_config or {})

        # Overall risk (worst-case across domains)
        if domain_assessments:
            overall_score = round(max(da.score for da in domain_assessments), 4)
        else:
            overall_score = 0.0
        overall_risk_level = self._classify_domain_risk(overall_score)

        summary = self._build_summary(
            domain_assessments, inventory, consent_result, overall_risk_level,
        )

        report = PIAReport(
            report_id=f"PIA-{datetime.now(timezone.utc).strftime('%Y%m%d')}-{uuid.uuid4().hex[:8]}",
            assessed_at=datetime.now(timezone.utc).isoformat(),
            system_name=self.system_name,
            assessor=self.assessor,
            pii_inventory=inventory,
            domain_assessments=domain_assessments,
            overall_risk_level=overall_risk_level,
            overall_score=overall_score,
            consent_adequacy=consent_result,
            data_minimization_score=minimization_score,
            summary=summary,
            regulatory_references=[
                "GDPR Article 35 - Data Protection Impact Assessment",
                "GDPR Article 6 - Lawfulness of Processing",
                "GDPR Article 22 - Automated Individual Decision-Making",
                "GDPR Article 46 - Transfers Subject to Appropriate Safeguards",
                "Korean PIPA Article 15 - Collection and Use of Personal Information",
                "Korean PIPA Article 37-2 - Right to Refuse Automated Decisions",
            ],
        )

        # Persist to audit store if available
        if self.audit_store is not None:
            self._persist_to_audit_store(report)

        logger.info(
            "PIA report generated: %s (overall=%s, score=%.4f, pii_fields=%d)",
            report.report_id, overall_risk_level, overall_score, len(inventory),
        )
        return report

    def to_dict(self, report: PIAReport) -> Dict[str, Any]:
        """Serialize a ``PIAReport`` to a JSON-compatible dict."""
        return {
            "report_id": report.report_id,
            "assessed_at": report.assessed_at,
            "system_name": report.system_name,
            "assessor": report.assessor,
            "overall_risk_level": report.overall_risk_level,
            "overall_score": report.overall_score,
            "data_minimization_score": report.data_minimization_score,
            "consent_adequacy": report.consent_adequacy,
            "summary": report.summary,
            "regulatory_references": report.regulatory_references,
            "pii_inventory": [
                {
                    "field_name": item.field_name,
                    "pii_category": item.pii_category,
                    "sensitivity_level": item.sensitivity_level,
                    "purpose": item.purpose,
                    "retention_days": item.retention_days,
                    "encrypted": item.encrypted,
                    "anonymized": item.anonymized,
                }
                for item in report.pii_inventory
            ],
            "domain_assessments": [
                {
                    "domain": da.domain,
                    "score": da.score,
                    "risk_level": da.risk_level,
                    "findings": da.findings,
                    "recommendations": da.recommendations,
                }
                for da in report.domain_assessments
            ],
            "metadata": report.metadata,
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _classify_domain_risk(score: float) -> str:
        """Map a domain score to a PIA risk level."""
        if score >= 0.8:
            return "CRITICAL"
        if score >= 0.6:
            return "HIGH"
        if score >= 0.3:
            return "MODERATE"
        return "LOW"

    @staticmethod
    def _build_summary(
        assessments: List[DomainRiskAssessment],
        inventory: List[PIIInventoryItem],
        consent: Dict[str, Any],
        overall_level: str,
    ) -> str:
        """Build an executive summary for the PIA report."""
        parts = [f"Privacy Impact Assessment completed."]
        parts.append(f"Overall privacy risk classification: {overall_level}.")
        parts.append(f"{len(inventory)} PII field(s) identified in data inventory.")

        high_risk_domains = [a.domain for a in assessments if a.risk_level in ("CRITICAL", "HIGH")]
        if high_risk_domains:
            parts.append(
                f"High-risk domains requiring attention: {', '.join(high_risk_domains)}."
            )

        total_findings = sum(len(a.findings) for a in assessments)
        total_recs = sum(len(a.recommendations) for a in assessments)
        parts.append(f"{total_findings} finding(s) and {total_recs} recommendation(s) identified.")

        if not consent.get("is_adequate", False):
            parts.append("Consent mechanisms do not meet regulatory requirements.")

        return " ".join(parts)

    def _persist_to_audit_store(self, report: PIAReport) -> None:
        """Write PIA results to the compliance audit store."""
        try:
            self.audit_store.log_incident(
                incident_id=report.report_id,
                event_type="pia_assessment",
                severity="minor" if report.overall_risk_level == "LOW" else "major",
                source_module="pia_evaluator",
                status="completed",
                description=report.summary,
            )
            logger.info("PIA report persisted to audit store: %s", report.report_id)
        except Exception as exc:
            logger.warning("Failed to persist PIA report: %s", exc)


__all__ = [
    "PIAEvaluator",
    "PIAReport",
    "PIIInventoryItem",
    "DomainRiskAssessment",
    "PII_CATEGORIES",
]
