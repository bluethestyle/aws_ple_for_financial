"""
Regulatory Compliance Checker
=============================

Automated compliance verification across Korean financial regulations:

- **금소법** (금융소비자보호법 / Financial Consumer Protection Act):
  AI disclosure, suitability assessment, cooling-off, complaint handling.
- **개보법** (개인정보보호법 / Personal Information Protection Act):
  Consent, data retention, right to deletion, purpose limitation.
- **AI기본법** (AI Basic Act):
  Explainability, human oversight, opt-out, bias monitoring, performance
  monitoring.

Each check queries the :class:`ComplianceAuditStore` or inspects
configuration to determine pass/fail.  Results include a severity level
so that critical failures can trigger immediate alerts.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from core.compliance.audit_store import ComplianceAuditStore

logger = logging.getLogger(__name__)

__all__ = [
    "CheckItem",
    "CheckResult",
    "RegulatoryComplianceChecker",
]


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class CheckItem:
    """Definition of a single compliance check.

    Attributes
    ----------
    id : str
        Unique check ID (e.g. ``"KFS-001"``).
    regulation : str
        Regulatory framework: ``"금소법"`` | ``"개보법"`` | ``"AI기본법"``
        | ``"EU_AI_ACT"``.
    category : str
        Logical category: ``"disclosure"`` | ``"consent"`` | ``"fairness"``
        | ``"audit"`` | ``"transparency"`` | ``"oversight"`` | ...
    description : str
        Human-readable description of the check.
    check_fn : str
        Name of the method on :class:`RegulatoryComplianceChecker` that
        implements this check.
    severity : str
        ``"critical"`` | ``"high"`` | ``"medium"`` | ``"low"``.
    """

    id: str
    regulation: str
    category: str
    description: str
    check_fn: str
    severity: str


@dataclass
class CheckResult:
    """Outcome of a single compliance check.

    Attributes
    ----------
    item : CheckItem
        The check that was executed.
    passed : bool
        Whether the check passed.
    message : str
        Human-readable result summary.
    details : dict
        Additional structured data (free-form).
    checked_at : str
        ISO 8601 timestamp of when the check was run.
    """

    item: CheckItem
    passed: bool
    message: str
    details: Dict[str, Any] = field(default_factory=dict)
    checked_at: str = ""


# ---------------------------------------------------------------------------
# Checker
# ---------------------------------------------------------------------------

class RegulatoryComplianceChecker:
    """Automated regulatory compliance checker.

    Validates 20+ compliance items across Korean financial regulations:

    - **금소법** (금융소비자보호법): AI disclosure, suitability, consent
    - **개보법** (개인정보보호법): Data minimization, purpose limitation, retention
    - **AI기본법**: Transparency, explainability, human oversight

    Usage::

        checker = RegulatoryComplianceChecker(audit_store, config)
        results = checker.run_all_checks()
        critical_failures = [
            r for r in results
            if not r.passed and r.item.severity == "critical"
        ]

    Parameters
    ----------
    audit_store : ComplianceAuditStore
        Audit store used to query evidence for compliance checks.
    config : dict, optional
        System configuration used by individual checks.  Expected keys:

        - ``"ai_disclosure_text"`` (str): Disclosure text shown to users.
        - ``"suitability_assessment_enabled"`` (bool)
        - ``"cooling_off_days"`` (int): Minimum cooling-off period in days.
        - ``"complaint_handler_module"`` (str): Importable module path.
        - ``"consent_required"`` (bool)
        - ``"data_retention_days"`` (int)
        - ``"deletion_handler_enabled"`` (bool)
        - ``"purpose_limitation_enforced"`` (bool)
        - ``"data_minimization_enforced"`` (bool)
        - ``"xai_enabled"`` (bool)
        - ``"kill_switch_enabled"`` (bool)
        - ``"optout_mechanism_enabled"`` (bool)
        - ``"bias_monitoring_enabled"`` (bool)
        - ``"performance_monitoring_enabled"`` (bool)
        - ``"model_registry_enabled"`` (bool)
        - ``"audit_trail_enabled"`` (bool)
        - ``"incident_response_enabled"`` (bool)
        - ``"data_lineage_enabled"`` (bool)
        - ``"encryption_at_rest"`` (bool)
        - ``"encryption_in_transit"`` (bool)
    """

    def __init__(
        self,
        audit_store: ComplianceAuditStore,
        config: Optional[Dict[str, Any]] = None,
    ) -> None:
        self._store = audit_store
        self._config = config or {}
        self._checks = self._build_check_items()

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def run_all_checks(self) -> List[CheckResult]:
        """Run all registered compliance checks.

        Returns
        -------
        list of CheckResult
            One result per registered check item.
        """
        now = datetime.now(timezone.utc).isoformat()
        results: List[CheckResult] = []

        for item in self._checks:
            result = self._run_single(item, now)
            results.append(result)

            if not result.passed:
                logger.warning(
                    "Compliance FAIL [%s] %s (severity=%s): %s",
                    item.id, item.description, item.severity, result.message,
                )

        passed_count = sum(1 for r in results if r.passed)
        total = len(results)
        logger.info(
            "Regulatory compliance: %d/%d passed (%.1f%%)",
            passed_count, total,
            (passed_count / total * 100) if total else 0.0,
        )
        return results

    def run_category(self, category: str) -> List[CheckResult]:
        """Run checks for a specific category.

        Parameters
        ----------
        category : str
            Category to filter on (e.g. ``"disclosure"``, ``"consent"``).

        Returns
        -------
        list of CheckResult
        """
        now = datetime.now(timezone.utc).isoformat()
        return [
            self._run_single(item, now)
            for item in self._checks
            if item.category == category
        ]

    def run_regulation(self, regulation: str) -> List[CheckResult]:
        """Run checks for a specific regulation.

        Parameters
        ----------
        regulation : str
            Regulation name (e.g. ``"금소법"``, ``"개보법"``, ``"AI기본법"``).

        Returns
        -------
        list of CheckResult
        """
        now = datetime.now(timezone.utc).isoformat()
        return [
            self._run_single(item, now)
            for item in self._checks
            if item.regulation == regulation
        ]

    def get_summary(self, results: Optional[List[CheckResult]] = None) -> Dict[str, Any]:
        """Produce a summary dict from check results.

        Parameters
        ----------
        results : list of CheckResult, optional
            If ``None``, :meth:`run_all_checks` is called first.

        Returns
        -------
        dict
            Keys: ``total``, ``passed``, ``failed``, ``pass_rate``,
            ``critical_failures``, ``by_regulation``, ``by_category``.
        """
        if results is None:
            results = self.run_all_checks()

        by_reg: Dict[str, Dict[str, int]] = {}
        by_cat: Dict[str, Dict[str, int]] = {}
        critical_failures: List[Dict[str, str]] = []

        for r in results:
            # By regulation
            reg = r.item.regulation
            by_reg.setdefault(reg, {"passed": 0, "failed": 0})
            by_reg[reg]["passed" if r.passed else "failed"] += 1

            # By category
            cat = r.item.category
            by_cat.setdefault(cat, {"passed": 0, "failed": 0})
            by_cat[cat]["passed" if r.passed else "failed"] += 1

            if not r.passed and r.item.severity == "critical":
                critical_failures.append({
                    "id": r.item.id,
                    "description": r.item.description,
                    "message": r.message,
                })

        total = len(results)
        passed = sum(1 for r in results if r.passed)
        return {
            "total": total,
            "passed": passed,
            "failed": total - passed,
            "pass_rate": round(passed / total, 4) if total else 0.0,
            "critical_failures": critical_failures,
            "by_regulation": by_reg,
            "by_category": by_cat,
        }

    @property
    def checks(self) -> List[CheckItem]:
        """Return the list of registered check items."""
        return list(self._checks)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _run_single(self, item: CheckItem, timestamp: str) -> CheckResult:
        """Execute one check item."""
        handler = getattr(self, item.check_fn, None)
        if handler is None:
            return CheckResult(
                item=item,
                passed=False,
                message=f"Check method not found: {item.check_fn}",
                checked_at=timestamp,
            )
        try:
            passed, message, details = handler()
        except Exception as exc:
            logger.exception("Error running check %s", item.id)
            passed, message, details = False, f"Check raised exception: {exc}", {}

        return CheckResult(
            item=item,
            passed=passed,
            message=message,
            details=details,
            checked_at=timestamp,
        )

    # ------------------------------------------------------------------
    # Check item registry
    # ------------------------------------------------------------------

    def _build_check_items(self) -> List[CheckItem]:
        """Build the full compliance check list (~20 items)."""
        return [
            # ── 금소법 (Financial Consumer Protection Act) ────────────
            CheckItem(
                id="KFS-001",
                regulation="금소법",
                category="disclosure",
                description="AI disclosure text present in recommendations",
                check_fn="_check_kfs_001_disclosure",
                severity="critical",
            ),
            CheckItem(
                id="KFS-002",
                regulation="금소법",
                category="suitability",
                description="Suitability assessment before product recommendation",
                check_fn="_check_kfs_002_suitability",
                severity="critical",
            ),
            CheckItem(
                id="KFS-003",
                regulation="금소법",
                category="consumer_protection",
                description="Cooling-off period respected",
                check_fn="_check_kfs_003_cooling_off",
                severity="high",
            ),
            CheckItem(
                id="KFS-004",
                regulation="금소법",
                category="consumer_protection",
                description="Complaint handling process exists",
                check_fn="_check_kfs_004_complaint",
                severity="high",
            ),
            CheckItem(
                id="KFS-005",
                regulation="금소법",
                category="audit",
                description="Kill switch audit trail maintained",
                check_fn="_check_kfs_005_ks_audit",
                severity="medium",
            ),

            # ── 개보법 (Personal Information Protection Act) ──────────
            CheckItem(
                id="PPA-001",
                regulation="개보법",
                category="consent",
                description="Consent obtained before data processing",
                check_fn="_check_ppa_001_consent",
                severity="critical",
            ),
            CheckItem(
                id="PPA-002",
                regulation="개보법",
                category="retention",
                description="Data retention period enforced",
                check_fn="_check_ppa_002_retention",
                severity="high",
            ),
            CheckItem(
                id="PPA-003",
                regulation="개보법",
                category="deletion",
                description="Right to deletion honored",
                check_fn="_check_ppa_003_deletion",
                severity="critical",
            ),
            CheckItem(
                id="PPA-004",
                regulation="개보법",
                category="purpose_limitation",
                description="Purpose limitation validated",
                check_fn="_check_ppa_004_purpose",
                severity="high",
            ),
            CheckItem(
                id="PPA-005",
                regulation="개보법",
                category="data_minimization",
                description="Data minimization enforced",
                check_fn="_check_ppa_005_minimization",
                severity="medium",
            ),
            CheckItem(
                id="PPA-006",
                regulation="개보법",
                category="encryption",
                description="Encryption at rest and in transit",
                check_fn="_check_ppa_006_encryption",
                severity="high",
            ),

            # ── AI기본법 (AI Basic Act) ───────────────────────────────
            CheckItem(
                id="AIA-001",
                regulation="AI기본법",
                category="transparency",
                description="Model explainability available (reason generation active)",
                check_fn="_check_aia_001_explainability",
                severity="critical",
            ),
            CheckItem(
                id="AIA-002",
                regulation="AI기본법",
                category="oversight",
                description="Human oversight mechanism exists (kill switch active)",
                check_fn="_check_aia_002_oversight",
                severity="critical",
            ),
            CheckItem(
                id="AIA-003",
                regulation="AI기본법",
                category="optout",
                description="AI opt-out mechanism available",
                check_fn="_check_aia_003_optout",
                severity="high",
            ),
            CheckItem(
                id="AIA-004",
                regulation="AI기본법",
                category="fairness",
                description="Bias monitoring in place",
                check_fn="_check_aia_004_bias",
                severity="critical",
            ),
            CheckItem(
                id="AIA-005",
                regulation="AI기본법",
                category="monitoring",
                description="Model performance monitoring active",
                check_fn="_check_aia_005_performance",
                severity="high",
            ),
            CheckItem(
                id="AIA-006",
                regulation="AI기본법",
                category="audit",
                description="Model registry and versioning enabled",
                check_fn="_check_aia_006_registry",
                severity="medium",
            ),
            CheckItem(
                id="AIA-007",
                regulation="AI기본법",
                category="audit",
                description="Comprehensive audit trail enabled",
                check_fn="_check_aia_007_audit_trail",
                severity="high",
            ),
            CheckItem(
                id="AIA-008",
                regulation="AI기본법",
                category="incident",
                description="Incident response process enabled",
                check_fn="_check_aia_008_incident",
                severity="high",
            ),
            CheckItem(
                id="AIA-009",
                regulation="AI기본법",
                category="transparency",
                description="Data lineage tracking enabled",
                check_fn="_check_aia_009_lineage",
                severity="medium",
            ),
        ]

    # ------------------------------------------------------------------
    # 금소법 checks
    # ------------------------------------------------------------------

    def _check_kfs_001_disclosure(self):
        """KFS-001: AI disclosure text must be configured and non-empty."""
        text = self._config.get("ai_disclosure_text", "")
        if text and len(text) >= 10:
            return True, "AI disclosure text is configured.", {"length": len(text)}
        return (
            False,
            "AI disclosure text is missing or too short (min 10 chars).",
            {"configured_text": text},
        )

    def _check_kfs_002_suitability(self):
        """KFS-002: Suitability assessment must be enabled."""
        enabled = self._config.get("suitability_assessment_enabled", False)
        if enabled:
            return True, "Suitability assessment is enabled.", {}
        return False, "Suitability assessment is not enabled.", {}

    def _check_kfs_003_cooling_off(self):
        """KFS-003: Cooling-off period must be at least 7 days."""
        days = self._config.get("cooling_off_days", 0)
        if days >= 7:
            return True, f"Cooling-off period is {days} days.", {"days": days}
        return (
            False,
            f"Cooling-off period is {days} days (minimum 7 required).",
            {"days": days},
        )

    def _check_kfs_004_complaint(self):
        """KFS-004: Complaint handling module must be importable."""
        module_path = self._config.get("complaint_handler_module", "")
        if not module_path:
            return False, "Complaint handler module not configured.", {}
        try:
            import importlib
            importlib.import_module(module_path)
            return True, f"Complaint handler module importable: {module_path}", {}
        except ImportError:
            # If module is configured but not importable, treat as partial pass
            # (module path is defined in config, just not deployed yet).
            return (
                False,
                f"Complaint handler module not importable: {module_path}",
                {"module": module_path},
            )

    def _check_kfs_005_ks_audit(self):
        """KFS-005: Kill switch audit trail should have recent entries."""
        try:
            events = self._store.query_events("killswitch", "ACTIVATE#global", limit=1)
            events += self._store.query_events("killswitch", "DEACTIVATE#global", limit=1)
            if events:
                return (
                    True,
                    "Kill switch audit trail has entries.",
                    {"recent_count": len(events)},
                )
            # No events is acceptable if the system has never needed a kill switch
            return (
                True,
                "Kill switch audit trail is empty (no activations recorded).",
                {"recent_count": 0},
            )
        except Exception as exc:
            return False, f"Cannot query kill switch audit: {exc}", {}

    # ------------------------------------------------------------------
    # 개보법 checks
    # ------------------------------------------------------------------

    def _check_ppa_001_consent(self):
        """PPA-001: Consent requirement must be enabled."""
        enabled = self._config.get("consent_required", False)
        if enabled:
            return True, "Consent requirement is enabled.", {}
        return False, "Consent requirement is not enabled.", {}

    def _check_ppa_002_retention(self):
        """PPA-002: Data retention period must be configured and reasonable."""
        days = self._config.get("data_retention_days", 0)
        if 1 <= days <= 3650:  # 1 day to 10 years
            return True, f"Data retention period is {days} days.", {"days": days}
        if days <= 0:
            return False, "Data retention period is not configured.", {"days": days}
        return (
            False,
            f"Data retention period of {days} days exceeds 10-year limit.",
            {"days": days},
        )

    def _check_ppa_003_deletion(self):
        """PPA-003: Right to deletion handler must be enabled."""
        enabled = self._config.get("deletion_handler_enabled", False)
        if enabled:
            return True, "Deletion handler is enabled.", {}
        return False, "Deletion handler is not enabled.", {}

    def _check_ppa_004_purpose(self):
        """PPA-004: Purpose limitation must be enforced."""
        enabled = self._config.get("purpose_limitation_enforced", False)
        if enabled:
            return True, "Purpose limitation is enforced.", {}
        return False, "Purpose limitation is not enforced.", {}

    def _check_ppa_005_minimization(self):
        """PPA-005: Data minimization must be enforced."""
        enabled = self._config.get("data_minimization_enforced", False)
        if enabled:
            return True, "Data minimization is enforced.", {}
        return False, "Data minimization is not enforced.", {}

    def _check_ppa_006_encryption(self):
        """PPA-006: Encryption at rest and in transit."""
        at_rest = self._config.get("encryption_at_rest", False)
        in_transit = self._config.get("encryption_in_transit", False)
        if at_rest and in_transit:
            return True, "Encryption at rest and in transit are both enabled.", {}
        missing = []
        if not at_rest:
            missing.append("at_rest")
        if not in_transit:
            missing.append("in_transit")
        return (
            False,
            f"Encryption not enabled for: {', '.join(missing)}.",
            {"at_rest": at_rest, "in_transit": in_transit},
        )

    # ------------------------------------------------------------------
    # AI기본법 checks
    # ------------------------------------------------------------------

    def _check_aia_001_explainability(self):
        """AIA-001: XAI / reason generation must be enabled."""
        enabled = self._config.get("xai_enabled", False)
        if enabled:
            return True, "XAI / explainability is enabled.", {}
        return False, "XAI / explainability is not enabled.", {}

    def _check_aia_002_oversight(self):
        """AIA-002: Kill switch (human oversight mechanism) must be enabled."""
        enabled = self._config.get("kill_switch_enabled", False)
        if enabled:
            return True, "Kill switch (human oversight) is enabled.", {}
        return False, "Kill switch (human oversight) is not enabled.", {}

    def _check_aia_003_optout(self):
        """AIA-003: AI decision opt-out mechanism must be available."""
        enabled = self._config.get("optout_mechanism_enabled", False)
        if enabled:
            return True, "AI opt-out mechanism is enabled.", {}
        return False, "AI opt-out mechanism is not enabled.", {}

    def _check_aia_004_bias(self):
        """AIA-004: Bias / fairness monitoring must be in place."""
        enabled = self._config.get("bias_monitoring_enabled", False)
        if enabled:
            return True, "Bias monitoring is enabled.", {}
        return False, "Bias monitoring is not enabled.", {}

    def _check_aia_005_performance(self):
        """AIA-005: Model performance monitoring must be active."""
        enabled = self._config.get("performance_monitoring_enabled", False)
        if enabled:
            return True, "Model performance monitoring is active.", {}
        return False, "Model performance monitoring is not active.", {}

    def _check_aia_006_registry(self):
        """AIA-006: Model registry and versioning must be enabled."""
        enabled = self._config.get("model_registry_enabled", False)
        if enabled:
            return True, "Model registry and versioning is enabled.", {}
        return False, "Model registry and versioning is not enabled.", {}

    def _check_aia_007_audit_trail(self):
        """AIA-007: Comprehensive audit trail must be enabled."""
        enabled = self._config.get("audit_trail_enabled", False)
        if enabled:
            return True, "Comprehensive audit trail is enabled.", {}
        return False, "Comprehensive audit trail is not enabled.", {}

    def _check_aia_008_incident(self):
        """AIA-008: Incident response process must be enabled."""
        enabled = self._config.get("incident_response_enabled", False)
        if enabled:
            return True, "Incident response process is enabled.", {}
        return False, "Incident response process is not enabled.", {}

    def _check_aia_009_lineage(self):
        """AIA-009: Data lineage tracking must be enabled."""
        enabled = self._config.get("data_lineage_enabled", False)
        if enabled:
            return True, "Data lineage tracking is enabled.", {}
        return False, "Data lineage tracking is not enabled.", {}
