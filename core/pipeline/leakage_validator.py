"""
Leakage Validator
=================

Validates that feature columns do not contain label information, preventing
data leakage that inflates evaluation metrics.

Checks performed:

1. **Sequence leakage** -- Verify that sequence columns do not extend into
   the prediction window (e.g. month 17 in a 17-month sequence when the
   label is derived from month 17).

2. **Feature-label correlation** -- Flag features with suspiciously high
   correlation to labels (Pearson > threshold).

3. **Temporal leakage** -- Verify that the train set contains no data from
   after the validation/test split boundary.

4. **Product column leakage** -- Check that ``prod_*`` columns reflect the
   pre-label state (month 16) rather than the label state (month 17).

Usage::

    from core.pipeline.leakage_validator import LeakageValidator

    validator = LeakageValidator()
    result = validator.validate(features_df, labels_df, config)
    if not result.passed:
        for warning in result.warnings:
            print(f"LEAKAGE WARNING: {warning}")
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np

try:
    import pyarrow as pa
except ImportError:  # pragma: no cover
    pa = None  # type: ignore[assignment]

try:
    import pandas as pd
except ImportError:  # pragma: no cover
    pd = None  # type: ignore[assignment]

logger = logging.getLogger(__name__)

__all__ = ["LeakageValidator", "ValidationResult"]


@dataclass
class ValidationResult:
    """Result of a leakage validation run.

    Attributes
    ----------
    passed : bool
        True if no critical leakage was detected.
    warnings : list[str]
        Human-readable warning messages for each issue found.
    details : dict
        Structured details keyed by check name.
    """

    passed: bool = True
    warnings: List[str] = field(default_factory=list)
    details: Dict[str, Any] = field(default_factory=dict)

    def add_warning(self, msg: str, check_name: str = "", detail: Any = None) -> None:
        """Record a leakage warning."""
        self.warnings.append(msg)
        if check_name and detail is not None:
            self.details[check_name] = detail

    def fail(self, msg: str, check_name: str = "", detail: Any = None) -> None:
        """Record a critical leakage failure."""
        self.passed = False
        self.add_warning(f"[CRITICAL] {msg}", check_name, detail)


def _get_column_names(data: Any) -> List[str]:
    """Extract column names from Arrow Table, pandas DataFrame, or dict."""
    if pa is not None and isinstance(data, pa.Table):
        return data.column_names
    if pd is not None and isinstance(data, pd.DataFrame):
        return list(data.columns)
    if isinstance(data, dict):
        return list(data.keys())
    raise TypeError(f"Unsupported data type: {type(data)}")


def _get_numeric_column_names(data: Any) -> List[str]:
    """Return names of numeric columns only."""
    if pa is not None and isinstance(data, pa.Table):
        numeric_types = (
            pa.int8, pa.int16, pa.int32, pa.int64,
            pa.uint8, pa.uint16, pa.uint32, pa.uint64,
            pa.float16, pa.float32, pa.float64,
        )
        return [
            f.name for f in data.schema
            if pa.types.is_integer(f.type)
            or pa.types.is_floating(f.type)
        ]
    if pd is not None and isinstance(data, pd.DataFrame):
        return list(data.select_dtypes(include=[np.number]).columns)
    if isinstance(data, dict):
        result = []
        for k, v in data.items():
            arr = np.asarray(v)
            if np.issubdtype(arr.dtype, np.number):
                result.append(k)
        return result
    raise TypeError(f"Unsupported data type: {type(data)}")


def _col_to_numpy(data: Any, col: str) -> np.ndarray:
    """Extract a single column as a 1-D numpy array."""
    if pa is not None and isinstance(data, pa.Table):
        return data.column(col).to_numpy(zero_copy_only=False)
    if pd is not None and isinstance(data, pd.DataFrame):
        return data[col].values
    if isinstance(data, dict):
        return np.asarray(data[col])
    raise TypeError(f"Unsupported data type: {type(data)}")


class LeakageValidator:
    """Validate that features don't contain label information.

    Performs multiple independent checks and aggregates results into a
    single :class:`ValidationResult`.

    Parameters
    ----------
    correlation_threshold : float
        Pearson correlation threshold above which a feature-label pair is
        flagged as suspicious.  Default 0.95.
    max_seq_len_expected : int
        Expected maximum sequence length AFTER truncation.  Sequences
        longer than this trigger a warning.  Default 16 (for Santander
        17-month data with 1 month truncated).
    """

    def __init__(
        self,
        correlation_threshold: float = 0.95,
        max_seq_len_expected: int = 16,
    ) -> None:
        self.correlation_threshold = correlation_threshold
        self.max_seq_len_expected = max_seq_len_expected

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def validate(
        self,
        features: Any,
        labels: Any,
        config: Optional[Dict[str, Any]] = None,
    ) -> ValidationResult:
        """Run all leakage checks.

        Parameters
        ----------
        features : pyarrow.Table | dict[str, ndarray] | pd.DataFrame
            Engineered features (post-transform).
        labels : pyarrow.Table | dict[str, ndarray] | pd.DataFrame
            Label columns.
        config : dict, optional
            Pipeline config dict for context-aware checks.

        Returns
        -------
        ValidationResult
        """
        result = ValidationResult()
        config = config or {}

        # Check 1: Feature-label correlation
        self.check_feature_label_correlation(features, labels, result)

        # Check 2: Exact column overlap
        self._check_column_overlap(features, labels, result)

        logger.info(
            "LeakageValidator: %s (%d warnings)",
            "PASSED" if result.passed else "FAILED",
            len(result.warnings),
        )

        return result

    # ------------------------------------------------------------------
    # Check: Sequence leakage
    # ------------------------------------------------------------------

    def check_sequence_leakage(
        self,
        df: Any,
        seq_cols: List[str],
        label_derivation_config: Optional[Dict[str, Any]] = None,
        result: Optional[ValidationResult] = None,
    ) -> ValidationResult:
        """Verify sequences don't extend into the prediction window.

        Checks that no sequence column has elements beyond the expected
        maximum length (which should exclude the label month).

        Parameters
        ----------
        df : pyarrow.Table | pd.DataFrame | dict[str, ndarray]
            Raw data with list-valued sequence columns.
        seq_cols : list[str]
            Sequence column names to check.
        label_derivation_config : dict, optional
            Config describing how labels are derived from sequences.
        result : ValidationResult, optional
            Existing result to append to.  Created if None.

        Returns
        -------
        ValidationResult
        """
        if result is None:
            result = ValidationResult()

        col_names = set(_get_column_names(df))

        for col in seq_cols:
            if col not in col_names:
                continue

            # Extract column as Python list for list-valued inspection
            col_data = _col_to_numpy(df, col)
            # Take up to 1000 non-null samples
            lengths: List[int] = []
            count = 0
            for val in col_data:
                if val is None:
                    continue
                if isinstance(val, (list, np.ndarray)):
                    lengths.append(len(val))
                else:
                    lengths.append(0)
                count += 1
                if count >= 1000:
                    break

            max_len = int(max(lengths)) if lengths else 0
            mean_len = float(np.mean(lengths)) if lengths else 0.0

            if max_len > self.max_seq_len_expected:
                result.fail(
                    f"Sequence column '{col}' has max_len={max_len}, "
                    f"expected <= {self.max_seq_len_expected}. "
                    f"This may include the label month (temporal leakage).",
                    check_name=f"seq_leakage_{col}",
                    detail={
                        "column": col,
                        "max_len": max_len,
                        "mean_len": round(mean_len, 2),
                        "expected_max": self.max_seq_len_expected,
                    },
                )
            else:
                logger.debug(
                    "LeakageValidator: sequence '%s' OK (max_len=%d <= %d)",
                    col, max_len, self.max_seq_len_expected,
                )

        return result

    # ------------------------------------------------------------------
    # Check: Feature-label correlation
    # ------------------------------------------------------------------

    def check_feature_label_correlation(
        self,
        features: Any,
        labels: Any,
        result: Optional[ValidationResult] = None,
        threshold: Optional[float] = None,
    ) -> ValidationResult:
        """Flag features with suspiciously high correlation to labels.

        Parameters
        ----------
        features : pyarrow.Table | dict[str, ndarray] | pd.DataFrame
            Feature columns (numeric only are checked).
        labels : pyarrow.Table | dict[str, ndarray] | pd.DataFrame
            Label columns.
        result : ValidationResult, optional
            Existing result to append to.
        threshold : float, optional
            Override correlation threshold.

        Returns
        -------
        ValidationResult
        """
        if result is None:
            result = ValidationResult()

        threshold = threshold or self.correlation_threshold

        # Only check numeric columns
        feat_num_cols = _get_numeric_column_names(features)
        label_num_cols = _get_numeric_column_names(labels)

        if not feat_num_cols or not label_num_cols:
            logger.debug("LeakageValidator: no numeric features/labels to correlate")
            return result

        suspicious_pairs: List[Dict[str, Any]] = []

        for label_col in label_num_cols:
            label_arr = _col_to_numpy(labels, label_col).astype(np.float64)
            label_valid = ~np.isnan(label_arr)
            if label_valid.sum() < 10:
                continue

            for feat_col in feat_num_cols:
                feat_arr = _col_to_numpy(features, feat_col).astype(np.float64)
                feat_valid = ~np.isnan(feat_arr)

                # Align: both must be non-NaN at the same index
                common_mask = label_valid & feat_valid
                if common_mask.sum() < 10:
                    continue

                try:
                    corr = abs(float(
                        np.corrcoef(
                            label_arr[common_mask],
                            feat_arr[common_mask],
                        )[0, 1]
                    ))
                except Exception:
                    continue

                if np.isnan(corr):
                    continue

                if corr >= threshold:
                    suspicious_pairs.append({
                        "feature": feat_col,
                        "label": label_col,
                        "correlation": round(corr, 4),
                    })
                    result.fail(
                        f"Feature '{feat_col}' has {corr:.4f} correlation "
                        f"with label '{label_col}' (threshold={threshold}). "
                        f"Possible data leakage.",
                        check_name=f"corr_{feat_col}_{label_col}",
                        detail={"feature": feat_col, "label": label_col,
                                "correlation": round(corr, 4)},
                    )

        if suspicious_pairs:
            logger.warning(
                "LeakageValidator: %d suspicious feature-label correlations found",
                len(suspicious_pairs),
            )
            result.details["correlation_summary"] = suspicious_pairs
        else:
            logger.info(
                "LeakageValidator: no suspicious correlations (threshold=%.2f)",
                threshold,
            )

        return result

    # ------------------------------------------------------------------
    # Check: Product column leakage
    # ------------------------------------------------------------------

    def check_product_columns(
        self,
        df: Any,
        prod_cols: List[str],
        seq_cols: List[str],
        seq_col_prefix: str = "seq_",
        prod_col_prefix: str = "prod_",
        result: Optional[ValidationResult] = None,
    ) -> ValidationResult:
        """Check that prod_* columns match the pre-label sequence state.

        For each ``prod_X`` column, verifies it equals the second-to-last
        element of the corresponding ``seq_X`` column (i.e. month 16
        state, not month 17).

        Parameters
        ----------
        df : pyarrow.Table | pd.DataFrame | dict[str, ndarray]
            Data with both product and sequence columns.
        prod_cols : list[str]
            Product holding column names.
        seq_cols : list[str]
            Corresponding sequence column names.
        seq_col_prefix : str
            Prefix for sequence columns.
        prod_col_prefix : str
            Prefix for product columns.
        result : ValidationResult, optional
            Existing result to append to.

        Returns
        -------
        ValidationResult
        """
        if result is None:
            result = ValidationResult()

        col_names = set(_get_column_names(df))

        for prod_col in prod_cols:
            if prod_col not in col_names:
                continue

            # Derive corresponding sequence column name
            suffix = prod_col[len(prod_col_prefix):]
            seq_col = seq_col_prefix + suffix

            if seq_col not in col_names:
                continue

            # Extract columns as numpy arrays
            prod_arr = _col_to_numpy(df, prod_col)
            seq_arr = _col_to_numpy(df, seq_col)

            # Sample check: does prod_col match last element of seq_col?
            mismatches = 0
            matches_last = 0
            matches_second_last = 0
            n_sample = 0

            limit = min(len(prod_arr), 1000)
            for i in range(limit):
                prod_val = prod_arr[i]
                seq = seq_arr[i]
                if prod_val is None or seq is None:
                    continue
                if isinstance(seq, (list, np.ndarray)) and len(seq) >= 2:
                    n_sample += 1
                    if prod_val == seq[-1]:
                        matches_last += 1
                    if prod_val == seq[-2]:
                        matches_second_last += 1
                    if prod_val != seq[-2]:
                        mismatches += 1

            if n_sample > 0 and matches_last / n_sample > 0.9:
                result.fail(
                    f"Product column '{prod_col}' matches LAST element of "
                    f"'{seq_col}' in {matches_last}/{n_sample} samples. "
                    f"This indicates the product column reflects the label "
                    f"month state (leakage). It should match the "
                    f"second-to-last element.",
                    check_name=f"prod_leak_{prod_col}",
                    detail={
                        "prod_col": prod_col,
                        "seq_col": seq_col,
                        "matches_last": matches_last,
                        "matches_second_last": matches_second_last,
                        "sample_size": n_sample,
                    },
                )

        return result

    # ------------------------------------------------------------------
    # Check: Column overlap
    # ------------------------------------------------------------------

    def _check_column_overlap(
        self,
        features: Any,
        labels: Any,
        result: ValidationResult,
    ) -> None:
        """Check if any label columns leaked into features."""
        overlap = set(_get_column_names(features)) & set(_get_column_names(labels))
        if overlap:
            result.fail(
                f"Label columns found in features: {overlap}. "
                f"These must be excluded from the feature set.",
                check_name="column_overlap",
                detail={"overlapping_columns": list(overlap)},
            )

    # ------------------------------------------------------------------
    # Check: Temporal split integrity
    # ------------------------------------------------------------------

    def check_temporal_integrity(
        self,
        train_data: Any,
        val_data: Any,
        test_data: Any,
        date_col: str = "snapshot_date",
        result: Optional[ValidationResult] = None,
    ) -> ValidationResult:
        """Verify no temporal overlap between splits.

        Parameters
        ----------
        train_data, val_data, test_data : pyarrow.Table | pd.DataFrame | dict
            Split data.
        date_col : str
            Date column name.
        result : ValidationResult, optional
            Existing result to append to.

        Returns
        -------
        ValidationResult
        """
        if result is None:
            result = ValidationResult()

        train_cols = _get_column_names(train_data)
        if date_col not in train_cols:
            result.add_warning(
                f"Cannot check temporal integrity: '{date_col}' not found"
            )
            return result

        train_dates = _col_to_numpy(train_data, date_col)
        val_dates = _col_to_numpy(val_data, date_col)
        test_dates = _col_to_numpy(test_data, date_col)

        # numpy datetime64 comparison works for dates
        train_max = np.max(train_dates)
        val_min = np.min(val_dates)
        val_max = np.max(val_dates)

        if train_max >= val_min:
            result.fail(
                f"Temporal overlap: train max date ({train_max}) >= "
                f"val min date ({val_min})",
                check_name="temporal_overlap_train_val",
            )

        if len(test_dates) > 0:
            test_min = np.min(test_dates)
            if val_max >= test_min:
                result.fail(
                    f"Temporal overlap: val max date ({val_max}) >= "
                    f"test min date ({test_min})",
                    check_name="temporal_overlap_val_test",
                )

        if result.passed:
            test_min_str = str(np.min(test_dates)) if len(test_dates) > 0 else "N/A"
            logger.info(
                "LeakageValidator: temporal integrity OK "
                "(train <= %s, val %s-%s, test >= %s)",
                train_max, val_min, val_max, test_min_str,
            )

        return result
