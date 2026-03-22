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
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

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
        feature_df: pd.DataFrame,
        label_df: pd.DataFrame,
        config: Optional[Dict[str, Any]] = None,
    ) -> ValidationResult:
        """Run all leakage checks.

        Parameters
        ----------
        feature_df : pd.DataFrame
            Engineered features (post-transform).
        label_df : pd.DataFrame
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
        self.check_feature_label_correlation(feature_df, label_df, result)

        # Check 2: Exact column overlap
        self._check_column_overlap(feature_df, label_df, result)

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
        df: pd.DataFrame,
        seq_cols: List[str],
        label_derivation_config: Optional[Dict[str, Any]] = None,
        result: Optional[ValidationResult] = None,
    ) -> ValidationResult:
        """Verify sequences don't extend into the prediction window.

        Checks that no sequence column has elements beyond the expected
        maximum length (which should exclude the label month).

        Parameters
        ----------
        df : pd.DataFrame
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

        for col in seq_cols:
            if col not in df.columns:
                continue

            # Sample sequence lengths
            sample = df[col].dropna().head(1000)
            lengths = sample.apply(
                lambda x: len(x) if isinstance(x, (list, np.ndarray)) else 0
            )

            max_len = int(lengths.max()) if len(lengths) > 0 else 0
            mean_len = float(lengths.mean()) if len(lengths) > 0 else 0

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
        features: pd.DataFrame,
        labels: pd.DataFrame,
        result: Optional[ValidationResult] = None,
        threshold: Optional[float] = None,
    ) -> ValidationResult:
        """Flag features with suspiciously high correlation to labels.

        Parameters
        ----------
        features : pd.DataFrame
            Feature columns (numeric only are checked).
        labels : pd.DataFrame
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
        numeric_features = features.select_dtypes(include=[np.number])
        numeric_labels = labels.select_dtypes(include=[np.number])

        if numeric_features.empty or numeric_labels.empty:
            logger.debug("LeakageValidator: no numeric features/labels to correlate")
            return result

        suspicious_pairs: List[Dict[str, Any]] = []

        for label_col in numeric_labels.columns:
            label_series = numeric_labels[label_col].dropna()
            if len(label_series) < 10:
                continue

            for feat_col in numeric_features.columns:
                feat_series = numeric_features[feat_col].dropna()

                # Align indices
                common_idx = label_series.index.intersection(feat_series.index)
                if len(common_idx) < 10:
                    continue

                try:
                    corr = abs(
                        label_series.loc[common_idx].corr(
                            feat_series.loc[common_idx]
                        )
                    )
                except Exception:
                    continue

                if corr >= threshold:
                    suspicious_pairs.append({
                        "feature": feat_col,
                        "label": label_col,
                        "correlation": round(float(corr), 4),
                    })
                    result.fail(
                        f"Feature '{feat_col}' has {corr:.4f} correlation "
                        f"with label '{label_col}' (threshold={threshold}). "
                        f"Possible data leakage.",
                        check_name=f"corr_{feat_col}_{label_col}",
                        detail={"feature": feat_col, "label": label_col,
                                "correlation": round(float(corr), 4)},
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
        df: pd.DataFrame,
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
        df : pd.DataFrame
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

        for prod_col in prod_cols:
            if prod_col not in df.columns:
                continue

            # Derive corresponding sequence column name
            suffix = prod_col[len(prod_col_prefix):]
            seq_col = seq_col_prefix + suffix

            if seq_col not in df.columns:
                continue

            # Sample check: does prod_col match last element of seq_col?
            sample = df[[prod_col, seq_col]].dropna().head(1000)
            mismatches = 0
            matches_last = 0
            matches_second_last = 0

            for _, row in sample.iterrows():
                seq = row[seq_col]
                prod_val = row[prod_col]
                if isinstance(seq, (list, np.ndarray)) and len(seq) >= 2:
                    if prod_val == seq[-1]:
                        matches_last += 1
                    if prod_val == seq[-2]:
                        matches_second_last += 1
                    if prod_val != seq[-2]:
                        mismatches += 1

            n_sample = len(sample)
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
        features: pd.DataFrame,
        labels: pd.DataFrame,
        result: ValidationResult,
    ) -> None:
        """Check if any label columns leaked into features."""
        overlap = set(features.columns) & set(labels.columns)
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
        train_df: pd.DataFrame,
        val_df: pd.DataFrame,
        test_df: pd.DataFrame,
        date_col: str = "snapshot_date",
        result: Optional[ValidationResult] = None,
    ) -> ValidationResult:
        """Verify no temporal overlap between splits.

        Parameters
        ----------
        train_df, val_df, test_df : pd.DataFrame
            Split DataFrames.
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

        if date_col not in train_df.columns:
            result.add_warning(
                f"Cannot check temporal integrity: '{date_col}' not found"
            )
            return result

        train_max = pd.to_datetime(train_df[date_col]).max()
        val_min = pd.to_datetime(val_df[date_col]).min()
        val_max = pd.to_datetime(val_df[date_col]).max()
        test_min = pd.to_datetime(test_df[date_col]).min()

        if train_max >= val_min:
            result.fail(
                f"Temporal overlap: train max date ({train_max}) >= "
                f"val min date ({val_min})",
                check_name="temporal_overlap_train_val",
            )

        if len(test_df) > 0 and val_max >= test_min:
            result.fail(
                f"Temporal overlap: val max date ({val_max}) >= "
                f"test min date ({test_min})",
                check_name="temporal_overlap_val_test",
            )

        if result.passed:
            logger.info(
                "LeakageValidator: temporal integrity OK "
                "(train <= %s, val %s-%s, test >= %s)",
                train_max.date(), val_min.date(), val_max.date(),
                test_min.date() if len(test_df) > 0 else "N/A",
            )

        return result
