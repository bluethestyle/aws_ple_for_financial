"""
Data Validation — schema checks, null ratios, value ranges, drift, PII.

Produces a structured :class:`ValidationResult` with per-check outcomes
so callers can decide whether to halt or proceed.

Example::

    from core.data.validation import DataValidator

    validator = DataValidator(schema_registry=registry)
    result = validator.validate("user_events", df)
    if not result.passed:
        for check in result.failed_checks:
            print(check)
"""

from __future__ import annotations

import logging
import math
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Sequence, Set, Union

import numpy as np

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────────────
# Result data classes
# ──────────────────────────────────────────────────────────────────────


class CheckStatus(Enum):
    PASS = "pass"
    FAIL = "fail"
    WARN = "warn"
    SKIP = "skip"


@dataclass
class CheckResult:
    """Outcome of a single validation check."""

    name: str
    status: CheckStatus
    message: str = ""
    details: Dict[str, Any] = field(default_factory=dict)

    @property
    def passed(self) -> bool:
        return self.status in (CheckStatus.PASS, CheckStatus.WARN, CheckStatus.SKIP)


@dataclass
class ValidationResult:
    """Aggregate outcome of all checks executed by :class:`DataValidator`."""

    source: str
    checks: List[CheckResult] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        """``True`` when no check has status ``FAIL``."""
        return all(c.status != CheckStatus.FAIL for c in self.checks)

    @property
    def failed_checks(self) -> List[CheckResult]:
        return [c for c in self.checks if c.status == CheckStatus.FAIL]

    @property
    def warnings(self) -> List[CheckResult]:
        return [c for c in self.checks if c.status == CheckStatus.WARN]

    def summary(self) -> str:
        """Return a human-readable summary string."""
        total = len(self.checks)
        passed = sum(1 for c in self.checks if c.status == CheckStatus.PASS)
        failed = len(self.failed_checks)
        warned = len(self.warnings)
        skipped = sum(1 for c in self.checks if c.status == CheckStatus.SKIP)
        lines = [
            f"ValidationResult for '{self.source}': "
            f"{passed}/{total} passed, {failed} failed, {warned} warnings, {skipped} skipped",
        ]
        for c in self.checks:
            marker = {
                CheckStatus.PASS: "[PASS]",
                CheckStatus.FAIL: "[FAIL]",
                CheckStatus.WARN: "[WARN]",
                CheckStatus.SKIP: "[SKIP]",
            }[c.status]
            lines.append(f"  {marker} {c.name}: {c.message}")
        return "\n".join(lines)

    def __repr__(self) -> str:  # pragma: no cover
        return self.summary()


# ──────────────────────────────────────────────────────────────────────
# PII detection patterns
# ──────────────────────────────────────────────────────────────────────

_PII_PATTERNS: List[Dict[str, Any]] = [
    {
        "name": "email",
        "pattern": re.compile(
            r"[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+", re.ASCII
        ),
        "min_match_ratio": 0.3,
    },
    {
        "name": "phone_kr",
        "pattern": re.compile(r"01[016789]-?\d{3,4}-?\d{4}"),
        "min_match_ratio": 0.3,
    },
    {
        "name": "phone_intl",
        "pattern": re.compile(r"\+?\d{1,3}[-.\s]?\(?\d{1,4}\)?[-.\s]?\d{3,4}[-.\s]?\d{4}"),
        "min_match_ratio": 0.2,
    },
    {
        "name": "ssn_kr",
        "pattern": re.compile(r"\d{6}-?[1-4]\d{6}"),
        "min_match_ratio": 0.1,
    },
    {
        "name": "credit_card",
        "pattern": re.compile(r"\d{4}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}"),
        "min_match_ratio": 0.1,
    },
    {
        "name": "ipv4",
        "pattern": re.compile(
            r"\b(?:\d{1,3}\.){3}\d{1,3}\b"
        ),
        "min_match_ratio": 0.3,
    },
]

# Column-name heuristics for PII
_PII_NAME_PATTERNS = re.compile(
    r"(email|phone|ssn|social_security|passport|driver_license|"
    r"credit_card|card_no|card_number|account_no|account_number|"
    r"resident_number|birth_date|dob|address|zip_code|postal|ip_addr|"
    r"name|first_name|last_name|full_name|customer_name)",
    re.IGNORECASE,
)


# ──────────────────────────────────────────────────────────────────────
# DataValidator
# ──────────────────────────────────────────────────────────────────────


class DataValidator:
    """
    Validates DataFrames against schema definitions and data-quality rules.

    Parameters
    ----------
    schema_registry : SchemaRegistry, optional
        An instance of :class:`~core.data.schema_registry.SchemaRegistry`.
        When provided, schema validation (column existence, types) is
        performed automatically.
    null_ratio_threshold : float
        Default maximum allowed null ratio per column (0.0 -- 1.0).
        Individual columns can override this in the schema.
    psi_threshold : float
        Population Stability Index threshold.  Columns with PSI above
        this value are flagged.
    sample_size : int
        Maximum number of rows to scan for PII detection.
    """

    def __init__(
        self,
        schema_registry: Optional[Any] = None,
        null_ratio_threshold: float = 0.5,
        psi_threshold: float = 0.2,
        sample_size: int = 10_000,
    ) -> None:
        self._registry = schema_registry
        self._null_threshold = null_ratio_threshold
        self._psi_threshold = psi_threshold
        self._sample_size = sample_size

    # ── Full validation ────────────────────────────────────────────────

    def validate(
        self,
        source: str,
        df: "pandas.DataFrame",  # noqa: F821
        *,
        reference_df: Optional["pandas.DataFrame"] = None,  # noqa: F821
        checks: Optional[Sequence[str]] = None,
    ) -> ValidationResult:
        """Run all (or selected) validation checks on *df*.

        Parameters
        ----------
        source : str
            Name of the data source (must be registered in the schema
            registry if schema validation is desired).
        df : pandas.DataFrame
            The DataFrame to validate.
        reference_df : pandas.DataFrame, optional
            A reference DataFrame for distribution drift checks.  When
            omitted, the drift check is skipped.
        checks : sequence of str, optional
            Subset of check names to execute.  ``None`` runs all.
            Available: ``schema``, ``null_ratio``, ``value_range``,
            ``drift``, ``pii``.

        Returns
        -------
        ValidationResult
        """
        all_checks = {"schema", "null_ratio", "value_range", "drift", "pii"}
        run_checks: Set[str] = set(checks) if checks else all_checks

        result = ValidationResult(source=source)

        if "schema" in run_checks:
            result.checks.extend(self._check_schema(source, df))

        if "null_ratio" in run_checks:
            result.checks.extend(self._check_null_ratios(source, df))

        if "value_range" in run_checks:
            result.checks.extend(self._check_value_ranges(source, df))

        if "drift" in run_checks:
            if reference_df is not None:
                result.checks.extend(
                    self._check_drift(source, df, reference_df)
                )
            else:
                result.checks.append(
                    CheckResult(
                        name="drift",
                        status=CheckStatus.SKIP,
                        message="No reference DataFrame provided; drift check skipped.",
                    )
                )

        if "pii" in run_checks:
            result.checks.extend(self._check_pii(source, df))

        logger.info(result.summary())
        return result

    # ── Individual checks ──────────────────────────────────────────────

    def _check_schema(
        self, source: str, df: "pandas.DataFrame"
    ) -> List[CheckResult]:
        """Validate columns existence and types against the registry."""
        results: List[CheckResult] = []

        if self._registry is None or not self._registry.has(source):
            results.append(
                CheckResult(
                    name="schema_columns",
                    status=CheckStatus.SKIP,
                    message=f"No schema registered for '{source}'; skipping.",
                )
            )
            return results

        schema = self._registry.get(source)
        df_cols = set(df.columns)

        # missing required columns
        missing: List[str] = []
        for col_name, col_spec in schema.columns.items():
            if col_name not in df_cols and not col_spec.nullable:
                missing.append(col_name)

        if missing:
            results.append(
                CheckResult(
                    name="schema_columns",
                    status=CheckStatus.FAIL,
                    message=f"Missing non-nullable columns: {missing}",
                    details={"missing": missing},
                )
            )
        else:
            results.append(
                CheckResult(
                    name="schema_columns",
                    status=CheckStatus.PASS,
                    message="All required columns present.",
                )
            )

        # extra columns (warn only)
        expected = set(schema.columns.keys())
        extra = df_cols - expected
        if extra:
            results.append(
                CheckResult(
                    name="schema_extra_columns",
                    status=CheckStatus.WARN,
                    message=f"Extra columns not in schema: {sorted(extra)}",
                    details={"extra": sorted(extra)},
                )
            )

        # type checks — lightweight (numeric vs non-numeric)
        _NUMERIC_DTYPES = {"int64", "int32", "float64", "float32", "double", "int"}
        for col_name, col_spec in schema.columns.items():
            if col_name not in df_cols:
                continue
            actual_kind = df[col_name].dtype.kind  # i=int, f=float, O=object, M=datetime
            expected_numeric = col_spec.dtype in _NUMERIC_DTYPES
            actual_numeric = actual_kind in ("i", "f", "u")
            if expected_numeric and not actual_numeric:
                results.append(
                    CheckResult(
                        name=f"schema_type_{col_name}",
                        status=CheckStatus.WARN,
                        message=(
                            f"Column '{col_name}' expected numeric "
                            f"({col_spec.dtype}), got dtype {df[col_name].dtype}"
                        ),
                    )
                )

        return results

    def _check_null_ratios(
        self, source: str, df: "pandas.DataFrame"
    ) -> List[CheckResult]:
        """Check null ratios for each column."""
        results: List[CheckResult] = []
        n = len(df)
        if n == 0:
            results.append(
                CheckResult(
                    name="null_ratio",
                    status=CheckStatus.WARN,
                    message="DataFrame is empty; null check skipped.",
                )
            )
            return results

        # per-column thresholds from schema, if available
        col_thresholds: Dict[str, float] = {}
        if self._registry is not None and self._registry.has(source):
            schema = self._registry.get(source)
            for col_name, col_spec in schema.columns.items():
                if not col_spec.nullable:
                    col_thresholds[col_name] = 0.0
                # else use default threshold

        for col in df.columns:
            null_ratio = float(df[col].isna().sum()) / n
            threshold = col_thresholds.get(col, self._null_threshold)

            if null_ratio > threshold:
                status = CheckStatus.FAIL if threshold == 0.0 else CheckStatus.WARN
                results.append(
                    CheckResult(
                        name=f"null_ratio_{col}",
                        status=status,
                        message=(
                            f"Column '{col}' null ratio {null_ratio:.2%} "
                            f"exceeds threshold {threshold:.2%}"
                        ),
                        details={"column": col, "null_ratio": null_ratio, "threshold": threshold},
                    )
                )

        if not any(c.status == CheckStatus.FAIL for c in results):
            results.append(
                CheckResult(
                    name="null_ratio",
                    status=CheckStatus.PASS,
                    message="All columns within null-ratio thresholds.",
                )
            )

        return results

    def _check_value_ranges(
        self, source: str, df: "pandas.DataFrame"
    ) -> List[CheckResult]:
        """Check numeric min/max and categorical allowed values."""
        results: List[CheckResult] = []

        if self._registry is None or not self._registry.has(source):
            results.append(
                CheckResult(
                    name="value_range",
                    status=CheckStatus.SKIP,
                    message="No schema registered; value range check skipped.",
                )
            )
            return results

        schema = self._registry.get(source)
        n = len(df)

        for col_name, col_spec in schema.columns.items():
            if col_name not in df.columns:
                continue

            # numeric range
            if col_spec.min is not None or col_spec.max is not None:
                series = df[col_name].dropna()
                if len(series) == 0:
                    continue

                if col_spec.min is not None:
                    actual_min = float(series.min())
                    if actual_min < col_spec.min:
                        results.append(
                            CheckResult(
                                name=f"value_range_{col_name}_min",
                                status=CheckStatus.FAIL,
                                message=(
                                    f"Column '{col_name}' min={actual_min:.4f} "
                                    f"< expected min={col_spec.min}"
                                ),
                                details={
                                    "column": col_name,
                                    "actual_min": actual_min,
                                    "expected_min": col_spec.min,
                                },
                            )
                        )

                if col_spec.max is not None:
                    actual_max = float(series.max())
                    if actual_max > col_spec.max:
                        results.append(
                            CheckResult(
                                name=f"value_range_{col_name}_max",
                                status=CheckStatus.FAIL,
                                message=(
                                    f"Column '{col_name}' max={actual_max:.4f} "
                                    f"> expected max={col_spec.max}"
                                ),
                                details={
                                    "column": col_name,
                                    "actual_max": actual_max,
                                    "expected_max": col_spec.max,
                                },
                            )
                        )

            # categorical allowed values
            if col_spec.allowed_values is not None:
                unique_vals = set(df[col_name].dropna().unique())
                allowed = set(col_spec.allowed_values)
                unexpected = unique_vals - allowed
                if unexpected:
                    results.append(
                        CheckResult(
                            name=f"value_range_{col_name}_allowed",
                            status=CheckStatus.FAIL,
                            message=(
                                f"Column '{col_name}' has unexpected values: "
                                f"{sorted(unexpected)[:10]}"
                            ),
                            details={
                                "column": col_name,
                                "unexpected": sorted(str(v) for v in unexpected)[:10],
                            },
                        )
                    )

        if not any(c.status == CheckStatus.FAIL for c in results):
            results.append(
                CheckResult(
                    name="value_range",
                    status=CheckStatus.PASS,
                    message="All columns within value-range constraints.",
                )
            )

        return results

    def _check_drift(
        self,
        source: str,
        df: "pandas.DataFrame",
        reference_df: "pandas.DataFrame",
    ) -> List[CheckResult]:
        """Compute PSI for numeric columns to detect distribution drift."""
        results: List[CheckResult] = []

        # determine numeric columns
        numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
        ref_numeric = reference_df.select_dtypes(include=["number"]).columns.tolist()
        shared_cols = [c for c in numeric_cols if c in ref_numeric]

        if not shared_cols:
            results.append(
                CheckResult(
                    name="drift",
                    status=CheckStatus.SKIP,
                    message="No shared numeric columns for drift check.",
                )
            )
            return results

        drifted: List[str] = []
        psi_scores: Dict[str, float] = {}

        for col in shared_cols:
            ref_vals = reference_df[col].dropna().values
            cur_vals = df[col].dropna().values
            if len(ref_vals) < 10 or len(cur_vals) < 10:
                continue

            psi = self._compute_psi(ref_vals, cur_vals)
            psi_scores[col] = psi
            if psi > self._psi_threshold:
                drifted.append(col)

        if drifted:
            results.append(
                CheckResult(
                    name="drift",
                    status=CheckStatus.WARN,
                    message=(
                        f"{len(drifted)}/{len(shared_cols)} columns show "
                        f"distribution drift (PSI > {self._psi_threshold}): "
                        f"{drifted[:10]}"
                    ),
                    details={"drifted_columns": drifted, "psi_scores": psi_scores},
                )
            )
        else:
            results.append(
                CheckResult(
                    name="drift",
                    status=CheckStatus.PASS,
                    message="No significant distribution drift detected.",
                    details={"psi_scores": psi_scores},
                )
            )

        return results

    @staticmethod
    def _compute_psi(
        reference: np.ndarray,
        current: np.ndarray,
        n_bins: int = 10,
        eps: float = 1e-6,
    ) -> float:
        """Compute Population Stability Index between two arrays.

        Parameters
        ----------
        reference : np.ndarray
            Baseline distribution values.
        current : np.ndarray
            New distribution values.
        n_bins : int
            Number of histogram bins.
        eps : float
            Small constant to avoid log(0).

        Returns
        -------
        float
            PSI value.  < 0.1 = stable, 0.1--0.2 = slight shift,
            > 0.2 = significant shift.
        """
        # use reference quantiles as bin edges for consistency
        breakpoints = np.quantile(reference, np.linspace(0, 1, n_bins + 1))
        breakpoints[0] = -np.inf
        breakpoints[-1] = np.inf
        # ensure unique breakpoints
        breakpoints = np.unique(breakpoints)

        ref_counts = np.histogram(reference, bins=breakpoints)[0].astype(float)
        cur_counts = np.histogram(current, bins=breakpoints)[0].astype(float)

        ref_pct = ref_counts / ref_counts.sum() + eps
        cur_pct = cur_counts / cur_counts.sum() + eps

        psi = float(np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct)))
        return psi

    def _check_pii(
        self, source: str, df: "pandas.DataFrame"
    ) -> List[CheckResult]:
        """Scan string columns for PII patterns."""
        results: List[CheckResult] = []
        detected: List[Dict[str, Any]] = []

        # column-name heuristic
        for col in df.columns:
            if _PII_NAME_PATTERNS.search(col):
                detected.append(
                    {"column": col, "method": "column_name_heuristic", "pattern": col}
                )

        # content scanning (sample)
        string_cols = df.select_dtypes(include=["object", "string"]).columns.tolist()
        sample = df.head(self._sample_size)

        for col in string_cols:
            col_values = sample[col].dropna().astype(str)
            if len(col_values) == 0:
                continue
            for pii_def in _PII_PATTERNS:
                matches = col_values.apply(lambda v: bool(pii_def["pattern"].search(v)))
                match_ratio = float(matches.sum()) / len(col_values)
                if match_ratio >= pii_def["min_match_ratio"]:
                    detected.append(
                        {
                            "column": col,
                            "method": "content_scan",
                            "pattern": pii_def["name"],
                            "match_ratio": round(match_ratio, 4),
                        }
                    )

        if detected:
            cols = sorted(set(d["column"] for d in detected))
            results.append(
                CheckResult(
                    name="pii",
                    status=CheckStatus.WARN,
                    message=(
                        f"Potential PII detected in {len(cols)} column(s): "
                        f"{cols}"
                    ),
                    details={"detections": detected},
                )
            )
        else:
            results.append(
                CheckResult(
                    name="pii",
                    status=CheckStatus.PASS,
                    message="No PII patterns detected.",
                )
            )

        return results

    # ── Repr ───────────────────────────────────────────────────────────

    def __repr__(self) -> str:  # pragma: no cover
        return (
            f"DataValidator(null_threshold={self._null_threshold}, "
            f"psi_threshold={self._psi_threshold})"
        )
