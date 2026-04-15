"""
Quality Gate -- pipeline data-quality gate replacing Great Expectations.

Wraps :class:`~core.data.validation.DataValidator` with gate-level
verdict logic (PASS / WARN / FAIL) and optional pipeline blocking.

Example::

    from core.data.quality_gate import QualityGate

    gate = QualityGate(config=config, schema_registry=registry)
    result = gate.evaluate(df, "user_events")
    if result.verdict == Verdict.FAIL:
        print("Pipeline blocked:", result)

    # Or block automatically:
    result = gate.evaluate_and_block(df, "user_events")
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────
# Enums & exceptions
# ──────────────────────────────────────────────────────────────────────


class Verdict(Enum):
    """Gate verdict."""

    PASS = "PASS"
    WARN = "WARN"
    FAIL = "FAIL"


class QualityGateError(Exception):
    """Raised when :meth:`QualityGate.evaluate_and_block` encounters a FAIL."""

    def __init__(self, result: "QualityGateResult") -> None:
        self.result = result
        super().__init__(
            f"Quality gate FAILED for source '{result.source_name}': "
            f"{len(result.failed_checks)} critical check(s) failed."
        )


# ──────────────────────────────────────────────────────────────────────
# Result data classes
# ──────────────────────────────────────────────────────────────────────


@dataclass
class QualityGateResult:
    """Outcome of a quality-gate evaluation."""

    source_name: str
    verdict: Verdict
    checks: List[Any] = field(default_factory=list)  # List[CheckResult]
    schema_valid: bool = True
    null_check_passed: bool = True
    range_check_passed: bool = True
    drift_detected: bool = False
    psi_scores: Dict[str, float] = field(default_factory=dict)
    timestamp: str = ""
    blocking: bool = False

    def __post_init__(self) -> None:
        if not self.timestamp:
            self.timestamp = datetime.now(timezone.utc).isoformat()
        if self.verdict == Verdict.FAIL:
            self.blocking = True

    @property
    def failed_checks(self) -> List[Any]:
        from core.data.validation import CheckStatus

        return [c for c in self.checks if c.status == CheckStatus.FAIL]

    @property
    def warning_checks(self) -> List[Any]:
        from core.data.validation import CheckStatus

        return [c for c in self.checks if c.status == CheckStatus.WARN]


# ──────────────────────────────────────────────────────────────────────
# Default configuration
# ──────────────────────────────────────────────────────────────────────

_DEFAULT_CONFIG: Dict[str, Any] = {
    # PSI thresholds
    "psi_warning_threshold": 0.1,
    "psi_critical_threshold": 0.2,
    # Checks considered critical (FAIL stops pipeline)
    "critical_checks": {
        "schema_columns",
        "null_ratio",
        "value_range",
    },
    # Non-critical checks (failures produce WARN)
    "non_critical_checks": {
        "schema_extra_columns",
        "schema_type",
        "drift",
        "pii",
    },
}


# ──────────────────────────────────────────────────────────────────────
# QualityGate
# ──────────────────────────────────────────────────────────────────────


class QualityGate:
    """Pipeline data-quality gate.

    Combines schema validation, null checks, range checks, and drift
    detection from :class:`~core.data.validation.DataValidator` with
    gate-level verdict logic:

    * **FAIL** -- at least one critical check failed **or** any column
      PSI exceeds ``psi_critical_threshold``.
    * **WARN** -- a non-critical check failed **or** any column PSI
      exceeds ``psi_warning_threshold``.
    * **PASS** -- all checks passed within thresholds.

    Parameters
    ----------
    config : dict, optional
        Override default thresholds and critical/non-critical sets.
    schema_registry : SchemaRegistry, optional
        Schema registry passed to the underlying ``DataValidator``.
    validation_engine : DataValidator, optional
        Pre-configured validator.  When ``None``, one is created
        internally using *schema_registry* and *config*.
    """

    def __init__(
        self,
        config: Optional[Dict[str, Any]] = None,
        schema_registry: Optional[Any] = None,
        validation_engine: Optional[Any] = None,
    ) -> None:
        self._config: Dict[str, Any] = {**_DEFAULT_CONFIG, **(config or {})}
        self._registry = schema_registry

        if validation_engine is not None:
            self._validator = validation_engine
        else:
            from core.data.validation import DataValidator

            self._validator = DataValidator(
                schema_registry=schema_registry,
                null_ratio_threshold=self._config.get("null_ratio_threshold", 0.5),
                psi_threshold=self._config.get("psi_critical_threshold", 0.2),
            )

    # ── Public API ────────────────────────────────────────────────────

    def evaluate(
        self,
        df: Any,
        source_name: str,
        *,
        reference_df: Optional[Any] = None,
    ) -> QualityGateResult:
        """Run all quality checks and return a gate result.

        Parameters
        ----------
        df : pandas.DataFrame or pyarrow.Table
            Data to validate.  PyArrow Tables are validated natively
            without ``to_pandas()`` conversion (CLAUDE.md 3.3).
        source_name : str
            Name of the data source (must match schema registry key).
        reference_df : pandas.DataFrame or pyarrow.Table, optional
            Reference dataset for drift detection.

        Returns
        -------
        QualityGateResult
        """
        from core.data.validation import CheckStatus, ValidationResult

        validation: ValidationResult = self._validator.validate(
            source_name, df, reference_df=reference_df,
        )

        # Aggregate per-category flags
        schema_valid = self._all_checks_ok(validation, "schema")
        null_check_passed = self._all_checks_ok(validation, "null_ratio")
        range_check_passed = self._all_checks_ok(validation, "value_range")
        drift_detected = self._any_check_status(
            validation, "drift", {CheckStatus.WARN, CheckStatus.FAIL},
        )

        # Extract PSI scores from drift check details
        psi_scores: Dict[str, float] = {}
        for check in validation.checks:
            if check.name == "drift" and "psi_scores" in check.details:
                psi_scores = check.details["psi_scores"]
                break

        # Determine verdict
        verdict = self._compute_verdict(validation, psi_scores)

        result = QualityGateResult(
            source_name=source_name,
            verdict=verdict,
            checks=validation.checks,
            schema_valid=schema_valid,
            null_check_passed=null_check_passed,
            range_check_passed=range_check_passed,
            drift_detected=drift_detected,
            psi_scores=psi_scores,
        )

        logger.info(
            "QualityGate [%s] verdict=%s  schema=%s null=%s range=%s drift=%s",
            source_name, verdict.value,
            schema_valid, null_check_passed, range_check_passed, drift_detected,
        )
        return result

    def evaluate_and_block(
        self,
        df: Any,
        source_name: str,
        *,
        reference_df: Optional[Any] = None,
    ) -> QualityGateResult:
        """Evaluate and raise :class:`QualityGateError` on FAIL.

        Identical to :meth:`evaluate` except that a ``FAIL`` verdict
        raises an exception, blocking the calling pipeline.

        Accepts both :class:`pandas.DataFrame` and :class:`pyarrow.Table`.

        Raises
        ------
        QualityGateError
            When the verdict is ``FAIL``.
        """
        result = self.evaluate(df, source_name, reference_df=reference_df)
        if result.verdict == Verdict.FAIL:
            raise QualityGateError(result)
        return result

    def get_report(self, result: QualityGateResult) -> Dict[str, Any]:
        """Serialise a gate result to a JSON-friendly dict.

        Parameters
        ----------
        result : QualityGateResult
            The result returned by :meth:`evaluate`.

        Returns
        -------
        dict
            JSON-serialisable report.
        """
        return {
            "source_name": result.source_name,
            "verdict": result.verdict.value,
            "timestamp": result.timestamp,
            "blocking": result.blocking,
            "summary": {
                "schema_valid": result.schema_valid,
                "null_check_passed": result.null_check_passed,
                "range_check_passed": result.range_check_passed,
                "drift_detected": result.drift_detected,
            },
            "psi_scores": result.psi_scores,
            "checks": [
                {
                    "name": c.name,
                    "status": c.status.value,
                    "message": c.message,
                    "details": c.details,
                }
                for c in result.checks
            ],
            "total_checks": len(result.checks),
            "failed_count": len(result.failed_checks),
            "warning_count": len(result.warning_checks),
        }

    # ── Internal helpers ──────────────────────────────────────────────

    def _compute_verdict(
        self,
        validation: Any,
        psi_scores: Dict[str, float],
    ) -> Verdict:
        """Determine PASS / WARN / FAIL from check results and PSI."""
        from core.data.validation import CheckStatus

        critical_names: set = set(self._config.get("critical_checks", set()))
        non_critical_names: set = set(self._config.get("non_critical_checks", set()))
        psi_critical: float = self._config.get("psi_critical_threshold", 0.2)
        psi_warning: float = self._config.get("psi_warning_threshold", 0.1)

        has_critical_fail = False
        has_non_critical_fail = False

        for check in validation.checks:
            if check.status != CheckStatus.FAIL:
                continue
            # Match check against critical set (prefix match)
            if self._is_critical(check.name, critical_names):
                has_critical_fail = True
            else:
                has_non_critical_fail = True

        # PSI-based verdict
        psi_critical_breach = any(v > psi_critical for v in psi_scores.values())
        psi_warning_breach = any(v > psi_warning for v in psi_scores.values())

        if has_critical_fail or psi_critical_breach:
            return Verdict.FAIL

        # Non-critical failures or warnings
        has_warnings = any(
            c.status == CheckStatus.WARN for c in validation.checks
        )
        if has_non_critical_fail or psi_warning_breach or has_warnings:
            return Verdict.WARN

        return Verdict.PASS

    @staticmethod
    def _is_critical(check_name: str, critical_names: set) -> bool:
        """Check whether *check_name* matches any critical category.

        Uses prefix matching so that e.g. ``null_ratio_col_a`` matches
        the critical category ``null_ratio``.
        """
        for crit in critical_names:
            if check_name == crit or check_name.startswith(crit + "_"):
                return True
        return False

    @staticmethod
    def _all_checks_ok(validation: Any, prefix: str) -> bool:
        """Return ``True`` if no check with *prefix* has status FAIL."""
        from core.data.validation import CheckStatus

        for check in validation.checks:
            if check.name == prefix or check.name.startswith(prefix + "_"):
                if check.status == CheckStatus.FAIL:
                    return False
        return True

    @staticmethod
    def _any_check_status(
        validation: Any,
        prefix: str,
        statuses: set,
    ) -> bool:
        """Return ``True`` if any check with *prefix* has one of *statuses*."""
        for check in validation.checks:
            if check.name == prefix or check.name.startswith(prefix + "_"):
                if check.status in statuses:
                    return True
        return False

    # ── Repr ──────────────────────────────────────────────────────────

    def __repr__(self) -> str:  # pragma: no cover
        return (
            f"QualityGate(psi_warn={self._config.get('psi_warning_threshold')}, "
            f"psi_crit={self._config.get('psi_critical_threshold')})"
        )
