"""3-stage feature normalization pipeline.

Matches the on-prem design with proper train-only fitting and
power-law raw copies that are NOT scaled.

Stages
------
1. **Amount-column log1p pre-transform** — applied in-place to
   power-law columns before scaling so the scaler sees the log-space
   values (this is handled implicitly: we detect power-law columns,
   and their raw values are used to create unscaled log copies).
2. **Z-score normalization** — mean/std computed on *training split
   only* using CuPy (GPU) when available, numpy otherwise.  Applied
   to continuous columns.  Binary columns are passed through as-is.
3. **Power-law raw copies** — ``log1p`` of the original (pre-scaled)
   values, appended as ``{col}_log`` columns.  These are **never**
   scaled so the model can see raw magnitude.

Output column order: ``[scaled_continuous | binary | power_law_log_copies]``
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

try:
    import cupy as cp  # GPU-accelerated array ops

    _HAS_CUPY = True
except ImportError:  # pragma: no cover
    cp = None
    _HAS_CUPY = False

logger = logging.getLogger(__name__)


def _xp():
    """Return CuPy if available, else numpy."""
    return cp if _HAS_CUPY else np


def _to_numpy(arr) -> np.ndarray:
    """Ensure *arr* is a plain numpy array (move off GPU if needed)."""
    if _HAS_CUPY and isinstance(arr, cp.ndarray):
        return cp.asnumpy(arr)
    return np.asarray(arr)

__all__ = ["FeatureNormalizer"]


class FeatureNormalizer:
    """3-stage normalization pipeline.

    Usage::

        normalizer = FeatureNormalizer()
        normalizer.fit(train_df, feature_cols)

        train_normed = normalizer.transform(train_df, feature_cols)
        val_normed   = normalizer.transform(val_df, feature_cols)
        test_normed  = normalizer.transform(test_df, feature_cols)

    The returned DataFrames have columns in a deterministic order:
    ``[scaled_continuous | binary | power_law_log_copies]``.
    """

    # Power-law detection thresholds (2-stage: fast filter + log-log R²)
    # These are class-level defaults; instance-level values come from config.
    SKEW_THRESH: float = 2.0
    KURT_THRESH: float = 6.0
    R2_THRESH: float = 0.9

    def __init__(self, config: Optional[Dict] = None):
        cfg = config or {}
        self.config = cfg
        # Allow config overrides for detection thresholds
        self.SKEW_THRESH = cfg.get("skew_threshold", self.__class__.SKEW_THRESH)
        self.KURT_THRESH = cfg.get("kurtosis_threshold", self.__class__.KURT_THRESH)
        self.R2_THRESH = cfg.get("loglog_r2_threshold", self.__class__.R2_THRESH)
        self._min_nunique: int = cfg.get("min_nunique", 20)
        self._min_samples: int = cfg.get("min_samples_loglog", 50)
        self._mean: Optional[np.ndarray] = None  # per-column means
        self._std: Optional[np.ndarray] = None   # per-column stds
        self.power_law_cols: List[str] = []
        self.continuous_cols: List[str] = []
        self.binary_cols: List[str] = []
        self.categorical_int_cols: List[str] = []  # integer IDs excluded from scaler
        self.probability_cols: List[str] = []       # already 0~1, excluded from scaler
        self.power_law_details: Dict[str, Dict] = {}
        self._fitted: bool = False

        # Patterns for columns that should NOT be StandardScaled
        self._categorical_id_suffixes: List[str] = cfg.get(
            "categorical_id_suffixes",
            ["_cluster_id", "_state_id", "_segment_id", "_group_id"],
        )
        self._probability_prefixes: List[str] = cfg.get(
            "probability_prefixes",
            ["gmm_clustering_cluster_prob_", "model_derived_gmm_prob_"],
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit(self, df: pd.DataFrame, feature_cols: List[str]) -> "FeatureNormalizer":
        """Fit on **training data only**.

        Parameters
        ----------
        df : pd.DataFrame
            Training split DataFrame.
        feature_cols : list[str]
            Columns to normalise (numeric feature columns).

        Returns
        -------
        self
        """
        # Only consider columns actually present in df
        feature_cols = [c for c in feature_cols if c in df.columns]

        # --- Classify columns ---
        self.binary_cols = [
            c for c in feature_cols
            if set(df[c].dropna().unique()).issubset({0, 0.0, 1, 1.0})
        ]

        # Categorical integer columns (cluster_id, state_id, etc.)
        # These are nominal — scaler would impose ordinal distance semantics.
        self.categorical_int_cols = [
            c for c in feature_cols
            if c not in self.binary_cols
            and any(c.endswith(sfx) for sfx in self._categorical_id_suffixes)
        ]

        # Probability columns already in [0, 1] — scaler destroys interpretability.
        self.probability_cols = [
            c for c in feature_cols
            if c not in self.binary_cols
            and c not in self.categorical_int_cols
            and any(c.startswith(pfx) for pfx in self._probability_prefixes)
        ]

        _exclude = set(self.binary_cols) | set(self.categorical_int_cols) | set(self.probability_cols)
        self.continuous_cols = [
            c for c in feature_cols if c not in _exclude
        ]

        if self.categorical_int_cols:
            logger.info(
                "Scaler-excluded categorical IDs: %d cols %s",
                len(self.categorical_int_cols), self.categorical_int_cols,
            )
        if self.probability_cols:
            logger.info(
                "Scaler-excluded probabilities: %d cols",
                len(self.probability_cols),
            )

        # --- Stage 1: Detect power-law columns ---
        self.power_law_cols, self.power_law_details = self._detect_power_law(df)
        if self.power_law_cols:
            logger.info(
                "Power-law detected: %d columns confirmed (R²>%.1f): %s",
                len(self.power_law_cols),
                self.R2_THRESH,
                self.power_law_cols,
            )

        # --- Stage 2: Compute mean / std (CuPy when available) ---
        self._mean = None
        self._std = None
        if self.continuous_cols:
            xp = _xp()
            raw = df[self.continuous_cols].fillna(0).values.astype(np.float64)
            arr = xp.asarray(raw)
            self._mean = _to_numpy(xp.mean(arr, axis=0))
            std = _to_numpy(xp.std(arr, axis=0))
            # Guard against zero-variance columns (mirror sklearn behaviour)
            std[std < 1e-10] = 1.0
            self._std = std

        self._fitted = True
        logger.info(
            "FeatureNormalizer fitted: %d continuous, %d binary, %d power-law",
            len(self.continuous_cols),
            len(self.binary_cols),
            len(self.power_law_cols),
        )
        return self

    def transform(
        self,
        df: pd.DataFrame,
        feature_cols: List[str],
    ) -> pd.DataFrame:
        """Transform a split using fitted parameters.

        Call this on train, val, and test splits separately.  The scaler
        parameters come from the training fit — val/test are *not* re-fit.

        Returns a new DataFrame with column order:
        ``[scaled_continuous | binary | power_law_log_copies]``
        """
        if not self._fitted:
            raise RuntimeError("FeatureNormalizer.transform() called before fit()")

        parts: List[pd.DataFrame] = []

        # --- Scaled continuous columns ---
        if self.continuous_cols and self._mean is not None:
            xp = _xp()
            raw = df[self.continuous_cols].fillna(0).values.astype(np.float64)
            arr = xp.asarray(raw)
            mean = xp.asarray(self._mean)
            std = xp.asarray(self._std)
            scaled = _to_numpy((arr - mean) / std)
            cont = pd.DataFrame(
                scaled,
                columns=self.continuous_cols,
                index=df.index,
            )
            parts.append(cont)

        # --- Binary columns (pass-through) ---
        if self.binary_cols:
            parts.append(df[self.binary_cols].copy())

        # --- Categorical integer columns (pass-through, no scaling) ---
        if self.categorical_int_cols:
            cat_present = [c for c in self.categorical_int_cols if c in df.columns]
            if cat_present:
                parts.append(df[cat_present].copy())

        # --- Probability columns (pass-through, already 0~1) ---
        if self.probability_cols:
            prob_present = [c for c in self.probability_cols if c in df.columns]
            if prob_present:
                parts.append(df[prob_present].copy())

        # --- Power-law log copies (log1p, NOT scaled) ---
        if self.power_law_cols:
            log_df = pd.DataFrame(index=df.index)
            for col in self.power_law_cols:
                log_df[f"{col}_log"] = np.log1p(
                    df[col].fillna(0).clip(lower=0)
                )
            parts.append(log_df)

        if parts:
            result = pd.concat(parts, axis=1)
        else:
            result = pd.DataFrame(index=df.index)

        return result

    def fit_transform(
        self,
        df: pd.DataFrame,
        feature_cols: List[str],
    ) -> pd.DataFrame:
        """Convenience: fit on *df* then transform it."""
        return self.fit(df, feature_cols).transform(df, feature_cols)

    # ------------------------------------------------------------------
    # Column introspection
    # ------------------------------------------------------------------

    @property
    def output_columns(self) -> List[str]:
        """Return the ordered list of output column names."""
        cols = list(self.continuous_cols)
        cols.extend(self.binary_cols)
        cols.extend(f"{c}_log" for c in self.power_law_cols)
        return cols

    # ------------------------------------------------------------------
    # Power-law detection (private)
    # ------------------------------------------------------------------

    def _detect_power_law(self, df: pd.DataFrame):
        """2-stage detection: skew+kurt filter then log-log R² confirmation.

        Returns
        -------
        confirmed : list[str]
        details : dict[str, dict]
        """
        candidates = []
        for col in self.continuous_cols:
            try:
                skew = float(df[col].skew())
                kurt = float(df[col].kurtosis())
                nunique = int(df[col].nunique())
                if (
                    abs(skew) > self.SKEW_THRESH
                    and kurt > self.KURT_THRESH
                    and df[col].min() >= 0
                    and nunique > self._min_nunique
                ):
                    candidates.append((col, skew, kurt))
            except (TypeError, ValueError):
                pass

        confirmed = []
        details = {}
        for col, skew, kurt in candidates:
            r2 = self._loglog_r2(df[col])
            if r2 >= self.R2_THRESH:
                confirmed.append(col)
                details[col] = {
                    "skew": round(skew, 2),
                    "kurt": round(kurt, 2),
                    "loglog_r2": round(r2, 4),
                }
            else:
                logger.debug(
                    "Power-law rejected '%s': skew=%.1f, kurt=%.1f, "
                    "loglog_R²=%.3f < %.1f",
                    col, skew, kurt, r2, self.R2_THRESH,
                )

        return confirmed, details

    def _loglog_r2(self, series: pd.Series) -> float:
        """Log-log rank-frequency R²."""
        min_samples = self._min_samples
        vals = series.dropna()
        vals = vals[vals > 0].sort_values(ascending=False).values
        if len(vals) < min_samples:
            return 0.0
        n = max(min_samples, len(vals) // 2)
        vals = vals[:n]
        log_rank = np.log(1 + np.arange(n))
        log_val = np.log(vals)
        if log_val.std() < 1e-10:
            return 0.0
        corr = np.corrcoef(log_rank, log_val)[0, 1]
        return corr ** 2

    def _detect_power_law_from_numpy(
        self, data: dict, continuous_cols: list
    ) -> None:
        """Power-law detection from numpy dict (DuckDB fetchnumpy output).

        Sets self.power_law_cols and self.power_law_details.
        No pandas dependency.
        """
        candidates = []
        for col in continuous_cols:
            arr = data.get(col)
            if arr is None:
                continue
            arr = arr[~np.isnan(arr)] if np.issubdtype(arr.dtype, np.floating) else arr
            if len(arr) < self._min_samples:
                continue
            skew = float(np.mean(((arr - arr.mean()) / max(arr.std(), 1e-10)) ** 3))
            kurt = float(np.mean(((arr - arr.mean()) / max(arr.std(), 1e-10)) ** 4) - 3)
            nunique = len(np.unique(arr))
            if abs(skew) > self.SKEW_THRESH and kurt > self.KURT_THRESH and arr.min() >= 0 and nunique > self._min_nunique:
                candidates.append((col, skew, kurt))

        self.power_law_cols = []
        self.power_law_details = {}
        for col, skew, kurt in candidates:
            arr = data[col]
            arr = arr[~np.isnan(arr)] if np.issubdtype(arr.dtype, np.floating) else arr
            vals = arr[arr > 0]
            vals = np.sort(vals)[::-1]
            n = max(self._min_samples, len(vals) // 2)
            vals = vals[:n]
            if len(vals) < self._min_samples:
                continue
            log_rank = np.log(1 + np.arange(len(vals)))
            log_val = np.log(vals.astype(np.float64))
            if log_val.std() < 1e-10:
                continue
            corr = np.corrcoef(log_rank, log_val)[0, 1]
            r2 = corr ** 2
            if r2 >= self.R2_THRESH:
                self.power_law_cols.append(col)
                self.power_law_details[col] = {"skew": round(skew, 2), "kurt": round(kurt, 2), "loglog_r2": round(r2, 4)}

        if self.power_law_cols:
            logger.info("Power-law detected: %d columns confirmed (R²>%.1f): %s",
                        len(self.power_law_cols), self.R2_THRESH, self.power_law_cols)

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str | Path) -> None:
        """Save fitted normalizer to *path* directory."""
        if not self._fitted:
            raise RuntimeError("Cannot save unfitted FeatureNormalizer")
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        if self._mean is not None:
            np.savez(
                path / "scaler_params.npz",
                mean=np.asarray(self._mean),
                std=np.asarray(self._std),
            )

        meta = {
            "continuous_cols": self.continuous_cols,
            "binary_cols": self.binary_cols,
            "categorical_int_cols": self.categorical_int_cols,
            "probability_cols": self.probability_cols,
            "power_law_cols": self.power_law_cols,
            "power_law_details": self.power_law_details,
        }
        with open(path / "normalizer_meta.json", "w") as f:
            json.dump(meta, f, indent=2)

        logger.info("FeatureNormalizer saved to %s", path)

    @classmethod
    def load(cls, path: str | Path) -> "FeatureNormalizer":
        """Load a previously saved normalizer."""
        path = Path(path)

        obj = cls()
        params_path = path / "scaler_params.npz"
        if params_path.exists():
            data = np.load(params_path)
            obj._mean = data["mean"]
            obj._std = data["std"]

        with open(path / "normalizer_meta.json", "r") as f:
            meta = json.load(f)

        obj.continuous_cols = meta["continuous_cols"]
        obj.binary_cols = meta["binary_cols"]
        obj.categorical_int_cols = meta.get("categorical_int_cols", [])
        obj.probability_cols = meta.get("probability_cols", [])
        obj.power_law_cols = meta["power_law_cols"]
        obj.power_law_details = meta.get("power_law_details", {})
        obj._fitted = True

        logger.info("FeatureNormalizer loaded from %s", path)
        return obj
