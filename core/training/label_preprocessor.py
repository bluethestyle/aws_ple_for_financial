"""
Label Preprocessor — task-aware target variable preparation.

Each task type requires different preprocessing:
  - binary:      threshold validation, class balance check
  - regression:  log transform, outlier clipping, standardization
  - multiclass:  label encoding validation, rare class merging

Also handles train/val/test split with stratification for binary tasks.

Usage::

    preprocessor = LabelPreprocessor(task_specs)

    # Split first
    train_df, val_df, test_df = preprocessor.split(
        df, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15, seed=42
    )

    # Then preprocess labels
    train_labels = preprocessor.preprocess(train_df)
    val_labels = preprocessor.preprocess(val_df)
    test_labels = preprocessor.preprocess(test_df)

    # After training, inverse-transform for evaluation
    predictions = preprocessor.inverse_transform("ltv", raw_predictions)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

__all__ = ["LabelPreprocessor", "SplitResult"]


@dataclass
class SplitResult:
    """Result of train/val/test split."""

    train_idx: np.ndarray
    val_idx: np.ndarray
    test_idx: np.ndarray
    train_size: int
    val_size: int
    test_size: int


class LabelPreprocessor:
    """Task-aware label preprocessing and data splitting.

    Args:
        task_specs: List of task spec objects with ``name``, ``type``,
            ``label_col``, and optionally ``num_classes``.
    """

    def __init__(self, task_specs: List[Any]) -> None:
        self._tasks = task_specs
        self._task_map = {t.name: t for t in task_specs}

        # Fitted statistics (populated by fit())
        self._fitted: bool = False
        self._regression_stats: Dict[str, Dict[str, float]] = {}
        # {task_name: {"mean": ..., "std": ..., "clip_lo": ..., "clip_hi": ..., "log": bool}}

    # ------------------------------------------------------------------
    # Split
    # ------------------------------------------------------------------

    def split(
        self,
        df: Any,
        train_ratio: float = 0.70,
        val_ratio: float = 0.15,
        test_ratio: float = 0.15,
        seed: int = 42,
        stratify_col: Optional[str] = None,
    ) -> Tuple[Any, Any, Any]:
        """Split DataFrame into train / val / test.

        Uses stratified split when a binary label column is specified
        (maintains class ratio across splits).

        Args:
            df: Full DataFrame.
            train_ratio: Fraction for training (default 0.70).
            val_ratio: Fraction for validation (default 0.15).
            test_ratio: Fraction for test (default 0.15).
            seed: Random seed for reproducibility.
            stratify_col: Column name to stratify on.  If None,
                auto-selects the first binary task's label_col.

        Returns:
            ``(train_df, val_df, test_df)`` tuple.
        """
        import pandas as pd

        assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, \
            f"Ratios must sum to 1.0, got {train_ratio + val_ratio + test_ratio}"

        n = len(df)
        rng = np.random.RandomState(seed)

        # Auto-detect stratification column (first binary task)
        if stratify_col is None:
            for t in self._tasks:
                if t.type == "binary" and t.label_col in df.columns:
                    stratify_col = t.label_col
                    break

        if stratify_col and stratify_col in df.columns:
            # Stratified split: maintain class ratio across splits
            train_idx, val_idx, test_idx = self._stratified_split(
                df[stratify_col].values, train_ratio, val_ratio, seed,
            )
        else:
            # Random shuffle split
            indices = rng.permutation(n)
            n_train = int(n * train_ratio)
            n_val = int(n * val_ratio)
            train_idx = indices[:n_train]
            val_idx = indices[n_train : n_train + n_val]
            test_idx = indices[n_train + n_val :]

        train_df = df.iloc[train_idx].reset_index(drop=True)
        val_df = df.iloc[val_idx].reset_index(drop=True)
        test_df = df.iloc[test_idx].reset_index(drop=True)

        logger.info(
            "Split: train=%d (%.0f%%), val=%d (%.0f%%), test=%d (%.0f%%)",
            len(train_df), len(train_df) / n * 100,
            len(val_df), len(val_df) / n * 100,
            len(test_df), len(test_df) / n * 100,
        )

        if stratify_col and stratify_col in df.columns:
            for split_name, split_df in [("train", train_df), ("val", val_df), ("test", test_df)]:
                pos_rate = split_df[stratify_col].mean()
                logger.info(
                    "  %s: %s positive rate = %.2f%%",
                    split_name, stratify_col, pos_rate * 100,
                )

        return train_df, val_df, test_df

    @staticmethod
    def _stratified_split(
        labels: np.ndarray,
        train_ratio: float,
        val_ratio: float,
        seed: int,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Stratified split maintaining class distribution."""
        rng = np.random.RandomState(seed)
        classes = np.unique(labels)

        train_idx, val_idx, test_idx = [], [], []

        for cls in classes:
            cls_indices = np.where(labels == cls)[0]
            rng.shuffle(cls_indices)

            n_cls = len(cls_indices)
            n_train = int(n_cls * train_ratio)
            n_val = int(n_cls * val_ratio)

            train_idx.extend(cls_indices[:n_train])
            val_idx.extend(cls_indices[n_train : n_train + n_val])
            test_idx.extend(cls_indices[n_train + n_val :])

        return np.array(train_idx), np.array(val_idx), np.array(test_idx)

    # ------------------------------------------------------------------
    # Fit (learn statistics from training data only)
    # ------------------------------------------------------------------

    def fit(self, df: Any) -> "LabelPreprocessor":
        """Learn preprocessing statistics from training data.

        Must be called on **training set only** (not val/test) to avoid
        data leakage.

        For regression tasks, computes:
          - mean, std (for standardization)
          - clip_lo, clip_hi (1st and 99th percentile, for outlier clipping)
          - log transform flag (if skewness > 2)
        """
        for t in self._tasks:
            if t.label_col not in df.columns:
                continue

            values = df[t.label_col].dropna().values.astype(np.float64)

            if t.type == "regression":
                skewness = float(np.abs(self._skewness(values)))
                use_log = skewness > 2.0 and np.all(values >= 0)

                if use_log:
                    values_for_stats = np.log1p(values)
                else:
                    values_for_stats = values

                self._regression_stats[t.name] = {
                    "mean": float(np.mean(values_for_stats)),
                    "std": float(np.std(values_for_stats).clip(min=1e-8)),
                    "clip_lo": float(np.percentile(values, 1)),
                    "clip_hi": float(np.percentile(values, 99)),
                    "log": use_log,
                    "skewness": skewness,
                    "raw_mean": float(np.mean(values)),
                    "raw_std": float(np.std(values)),
                }

                logger.info(
                    "  [%s] regression: mean=%.2f, std=%.2f, skew=%.2f, log=%s, clip=[%.2f, %.2f]",
                    t.name,
                    self._regression_stats[t.name]["mean"],
                    self._regression_stats[t.name]["std"],
                    skewness, use_log,
                    self._regression_stats[t.name]["clip_lo"],
                    self._regression_stats[t.name]["clip_hi"],
                )

            elif t.type == "binary":
                pos_rate = float(np.mean(values))
                n_pos = int(np.sum(values))
                n_neg = len(values) - n_pos
                logger.info(
                    "  [%s] binary: pos=%d (%.1f%%), neg=%d (%.1f%%)",
                    t.name, n_pos, pos_rate * 100, n_neg, (1 - pos_rate) * 100,
                )

                if pos_rate < 0.01 or pos_rate > 0.99:
                    logger.warning(
                        "  [%s] EXTREME IMBALANCE: %.2f%% positive — "
                        "consider resampling or adjusting loss weight",
                        t.name, pos_rate * 100,
                    )

            elif t.type == "multiclass":
                n_classes = len(np.unique(values[~np.isnan(values)]))
                expected = getattr(t, "num_classes", n_classes)
                if n_classes != expected:
                    logger.warning(
                        "  [%s] multiclass: found %d classes, expected %d",
                        t.name, n_classes, expected,
                    )
                else:
                    logger.info(
                        "  [%s] multiclass: %d classes", t.name, n_classes,
                    )

                # Check for rare classes
                unique, counts = np.unique(
                    values[~np.isnan(values)].astype(int), return_counts=True,
                )
                min_count = counts.min()
                if min_count < 10:
                    rare = unique[counts < 10].tolist()
                    logger.warning(
                        "  [%s] RARE CLASSES (<%d samples): %s",
                        t.name, 10, rare[:5],
                    )

        self._fitted = True
        return self

    # ------------------------------------------------------------------
    # Preprocess
    # ------------------------------------------------------------------

    def preprocess(self, df: Any) -> Dict[str, np.ndarray]:
        """Preprocess labels for all tasks.

        Must call :meth:`fit` first (on training data).

        Args:
            df: DataFrame with label columns.

        Returns:
            ``{task_name: preprocessed_array}`` dict.
        """
        if not self._fitted:
            raise RuntimeError("Call fit() on training data before preprocess()")

        result: Dict[str, np.ndarray] = {}

        for t in self._tasks:
            if t.label_col not in df.columns:
                continue

            values = df[t.label_col].values.copy().astype(np.float64)

            if t.type == "regression" and t.name in self._regression_stats:
                stats = self._regression_stats[t.name]

                # 1. Outlier clipping
                values = np.clip(values, stats["clip_lo"], stats["clip_hi"])

                # 2. Log transform (if skewed)
                if stats["log"]:
                    values = np.log1p(np.maximum(values, 0))

                # 3. Standardize
                values = (values - stats["mean"]) / stats["std"]

                result[t.name] = values.astype(np.float32)

            elif t.type == "binary":
                # Ensure 0/1 encoding
                values = np.where(values > 0.5, 1, 0).astype(np.float32)
                result[t.name] = values

            elif t.type == "multiclass":
                # Ensure integer encoding, handle NaN
                values = np.nan_to_num(values, nan=0).astype(np.int64)
                n_classes = getattr(t, "num_classes", values.max() + 1)
                values = np.clip(values, 0, n_classes - 1)
                result[t.name] = values.astype(np.float32)

            else:
                result[t.name] = values.astype(np.float32)

        return result

    # ------------------------------------------------------------------
    # Inverse transform (for evaluation)
    # ------------------------------------------------------------------

    def inverse_transform(
        self, task_name: str, values: np.ndarray,
    ) -> np.ndarray:
        """Reverse regression preprocessing for human-readable evaluation.

        Args:
            task_name: Task name.
            values: Preprocessed (standardized, log-transformed) values.

        Returns:
            Original-scale values.
        """
        if task_name not in self._regression_stats:
            return values

        stats = self._regression_stats[task_name]
        result = values * stats["std"] + stats["mean"]

        if stats["log"]:
            result = np.expm1(result)

        return result

    # ------------------------------------------------------------------
    # Report
    # ------------------------------------------------------------------

    def get_stats_report(self) -> Dict[str, Any]:
        """Return preprocessing statistics for all tasks."""
        return {
            "regression_stats": self._regression_stats,
            "fitted": self._fitted,
        }

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _skewness(x: np.ndarray) -> float:
        """Compute skewness (Fisher's definition)."""
        n = len(x)
        if n < 3:
            return 0.0
        mean = np.mean(x)
        std = np.std(x)
        if std < 1e-10:
            return 0.0
        return float(np.mean(((x - mean) / std) ** 3))
