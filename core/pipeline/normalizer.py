"""3-stage feature normalization pipeline.

Matches the on-prem design with proper train-only fitting and
power-law raw copies that are NOT scaled.

Stages
------
1. **Amount-column log1p pre-transform** — applied in-place to
   power-law columns before scaling so the scaler sees the log-space
   values (this is handled implicitly: we detect power-law columns,
   and their raw values are used to create unscaled log copies).
2. **StandardScaler** — fit on *training split only*, applied to
   continuous columns.  Binary columns are passed through as-is.
3. **Power-law raw copies** — ``log1p`` of the original (pre-scaled)
   values, appended as ``{col}_log`` columns.  These are **never**
   scaled so the model can see raw magnitude.

Output column order: ``[scaled_continuous | binary | power_law_log_copies]``
"""

from __future__ import annotations

import json
import logging
import pickle
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

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
    SKEW_THRESH: float = 2.0
    KURT_THRESH: float = 6.0
    R2_THRESH: float = 0.9

    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {}
        self.scaler = None  # sklearn StandardScaler
        self.power_law_cols: List[str] = []
        self.continuous_cols: List[str] = []
        self.binary_cols: List[str] = []
        self.power_law_details: Dict[str, Dict] = {}
        self._fitted: bool = False

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
        self.continuous_cols = [
            c for c in feature_cols if c not in self.binary_cols
        ]

        # --- Stage 1: Detect power-law columns ---
        self.power_law_cols, self.power_law_details = self._detect_power_law(df)
        if self.power_law_cols:
            logger.info(
                "Power-law detected: %d columns confirmed (R²>%.1f): %s",
                len(self.power_law_cols),
                self.R2_THRESH,
                self.power_law_cols,
            )

        # --- Stage 2: Fit StandardScaler on continuous columns only ---
        from sklearn.preprocessing import StandardScaler

        self.scaler = None
        if self.continuous_cols:
            self.scaler = StandardScaler()
            self.scaler.fit(df[self.continuous_cols].fillna(0).values)

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
        if self.continuous_cols and self.scaler is not None:
            cont = pd.DataFrame(
                self.scaler.transform(
                    df[self.continuous_cols].fillna(0).values,
                ),
                columns=self.continuous_cols,
                index=df.index,
            )
            parts.append(cont)

        # --- Binary columns (pass-through) ---
        if self.binary_cols:
            parts.append(df[self.binary_cols].copy())

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
                    and nunique > 20
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

    @staticmethod
    def _loglog_r2(series: pd.Series) -> float:
        """Log-log rank-frequency R²."""
        vals = series.dropna()
        vals = vals[vals > 0].sort_values(ascending=False).values
        if len(vals) < 50:
            return 0.0
        n = max(50, len(vals) // 2)
        vals = vals[:n]
        log_rank = np.log(1 + np.arange(n))
        log_val = np.log(vals)
        if log_val.std() < 1e-10:
            return 0.0
        corr = np.corrcoef(log_rank, log_val)[0, 1]
        return corr ** 2

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str | Path) -> None:
        """Save fitted normalizer to *path* directory."""
        if not self._fitted:
            raise RuntimeError("Cannot save unfitted FeatureNormalizer")
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        if self.scaler is not None:
            with open(path / "scaler.pkl", "wb") as f:
                pickle.dump(self.scaler, f)

        meta = {
            "continuous_cols": self.continuous_cols,
            "binary_cols": self.binary_cols,
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
        scaler_path = path / "scaler.pkl"
        if scaler_path.exists():
            with open(scaler_path, "rb") as f:
                obj.scaler = pickle.load(f)

        with open(path / "normalizer_meta.json", "r") as f:
            meta = json.load(f)

        obj.continuous_cols = meta["continuous_cols"]
        obj.binary_cols = meta["binary_cols"]
        obj.power_law_cols = meta["power_law_cols"]
        obj.power_law_details = meta.get("power_law_details", {})
        obj._fitted = True

        logger.info("FeatureNormalizer loaded from %s", path)
        return obj
