"""
Temporal Pattern Feature Generator.

Extracts rich temporal features from sequential data, going beyond simple
aggregations to capture cyclical patterns, velocity, acceleration, and
distributional shifts over time.

Feature categories
------------------
1. **Sequence aggregation**: rolling statistics (mean, std, trend) over
   configurable windows.
2. **Cyclical encoding**: time-of-day, day-of-week, month-of-year
   encoded as sine/cosine pairs to preserve circular continuity.
3. **Velocity and acceleration**: first and second derivatives of
   activity signals over time.
4. **Recency features**: exponentially weighted recency scores.

This is a **placeholder implementation** with proper interfaces.
Production use would incorporate actual timestamp parsing and
window-based computations.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from ..generator import AbstractFeatureGenerator, FeatureGeneratorRegistry

logger = logging.getLogger(__name__)


@FeatureGeneratorRegistry.register(
    "temporal_pattern",
    description="Temporal sequence aggregation, cyclical encoding, and velocity features.",
    tags=["temporal", "sequential", "cyclical", "time"],
)
class TemporalPatternGenerator(AbstractFeatureGenerator):
    """Generate temporal pattern features from time-indexed data.

    Parameters
    ----------
    time_column : str, optional
        Column containing datetime or ordinal time values.
    value_columns : list[str]
        Columns containing the numeric signals to aggregate over time.
    windows : list[int]
        Rolling window sizes for sequence aggregation (e.g. ``[7, 30, 90]``
        for weekly, monthly, quarterly).
    cyclical_periods : list[str]
        Which cyclical encodings to include.  Subset of
        ``["hour", "dayofweek", "month", "quarter"]``.
    include_velocity : bool
        Whether to compute first-derivative (velocity) features.
    include_acceleration : bool
        Whether to compute second-derivative (acceleration) features.
    include_recency : bool
        Whether to compute exponentially-weighted recency scores.
    recency_halflife : float
        Half-life (in the same units as the time column) for the
        recency exponential decay.
    prefix : str
        Column name prefix for generated features.
    """

    def __init__(
        self,
        time_column: Optional[str] = None,
        value_columns: Optional[List[str]] = None,
        windows: Optional[List[int]] = None,
        cyclical_periods: Optional[List[str]] = None,
        include_velocity: bool = True,
        include_acceleration: bool = True,
        include_recency: bool = True,
        recency_halflife: float = 30.0,
        prefix: str = "temp",
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.time_column = time_column
        self.value_columns = value_columns or []
        self.windows = windows or [7, 30, 90]
        self.cyclical_periods = cyclical_periods or ["hour", "dayofweek", "month"]
        self.include_velocity = include_velocity
        self.include_acceleration = include_acceleration
        self.include_recency = include_recency
        self.recency_halflife = recency_halflife
        self.prefix = prefix

        # Fitted state
        self._value_means: Optional[Dict[str, float]] = None
        self._value_stds: Optional[Dict[str, float]] = None
        self._has_time_column: bool = False

    # -- Output description --------------------------------------------

    @property
    def output_dim(self) -> int:
        """Total number of generated temporal features."""
        dim = 0

        # Cyclical encodings: 2 per period (sin + cos)
        dim += len(self.cyclical_periods) * 2

        # Per value column:
        n_val = max(len(self.value_columns), 1)  # at least 1

        # Rolling window aggregations: 3 stats (mean, std, trend) per window
        dim += n_val * len(self.windows) * 3

        # Velocity (1 per value column)
        if self.include_velocity:
            dim += n_val

        # Acceleration (1 per value column)
        if self.include_acceleration:
            dim += n_val

        # Recency (1 per value column)
        if self.include_recency:
            dim += n_val

        return dim

    @property
    def output_columns(self) -> List[str]:
        """Generated column names."""
        cols: List[str] = []

        # Cyclical
        for period in self.cyclical_periods:
            cols.append(f"{self.prefix}_{period}_sin")
            cols.append(f"{self.prefix}_{period}_cos")

        # Determine value column names
        val_names = self.value_columns if self.value_columns else ["signal"]

        for v in val_names:
            # Rolling windows
            for w in self.windows:
                cols.append(f"{self.prefix}_{v}_roll{w}_mean")
                cols.append(f"{self.prefix}_{v}_roll{w}_std")
                cols.append(f"{self.prefix}_{v}_roll{w}_trend")

            # Velocity
            if self.include_velocity:
                cols.append(f"{self.prefix}_{v}_velocity")

            # Acceleration
            if self.include_acceleration:
                cols.append(f"{self.prefix}_{v}_acceleration")

            # Recency
            if self.include_recency:
                cols.append(f"{self.prefix}_{v}_recency")

        return cols

    # -- Core API ------------------------------------------------------

    def fit(self, df: pd.DataFrame, **context: Any) -> "TemporalPatternGenerator":
        """Learn value distribution parameters for normalisation.

        Also detects whether the time column exists and contains
        valid datetime data.
        """
        # Check for time column
        self._has_time_column = (
            self.time_column is not None
            and self.time_column in df.columns
        )

        # Resolve value columns
        val_cols = self._resolve_value_columns(df)

        # Learn normalisation stats
        self._value_means = {}
        self._value_stds = {}
        for col in val_cols:
            self._value_means[col] = float(df[col].mean())
            std = float(df[col].std())
            self._value_stds[col] = std if std > 0 else 1.0

        self._fitted = True
        logger.info(
            "TemporalPatternGenerator fitted: %d value columns, "
            "%d windows, %d cyclical periods, output_dim=%d",
            len(val_cols), len(self.windows),
            len(self.cyclical_periods), self.output_dim,
        )
        return self

    def generate(self, df: pd.DataFrame, **context: Any) -> pd.DataFrame:
        """Generate temporal pattern features.

        .. note::
           Placeholder: derives temporal features from available data
           statistics.  Production would use actual time-series
           computations with proper windowing.
        """
        if not self._fitted:
            raise RuntimeError(
                "TemporalPatternGenerator must be fitted before generate()."
            )

        n_rows = len(df)
        results: Dict[str, np.ndarray] = {}

        # -- Cyclical encodings ----------------------------------------
        self._generate_cyclical(df, results)

        # -- Per-value-column features ---------------------------------
        val_cols = self._resolve_value_columns(df)

        for v in val_cols:
            if v in df.columns:
                values = df[v].values.astype(np.float64)
            else:
                values = np.zeros(n_rows, dtype=np.float64)

            mean = self._value_means.get(v, 0.0)
            std = self._value_stds.get(v, 1.0)
            normed = (values - mean) / std

            # Rolling window aggregations (placeholder: use global stats
            # since we don't have actual sequential ordering per row)
            for w in self.windows:
                # In production: compute actual rolling statistics
                results[f"{self.prefix}_{v}_roll{w}_mean"] = (
                    normed * (1.0 - 1.0 / w)
                ).astype(np.float32)
                results[f"{self.prefix}_{v}_roll{w}_std"] = (
                    np.abs(normed) * (1.0 / np.sqrt(w))
                ).astype(np.float32)
                # Trend: sign of the difference from mean, scaled by window
                results[f"{self.prefix}_{v}_roll{w}_trend"] = (
                    np.sign(normed) * np.log1p(np.abs(normed)) / np.log1p(w)
                ).astype(np.float32)

            # Velocity (first derivative proxy)
            if self.include_velocity:
                results[f"{self.prefix}_{v}_velocity"] = (
                    np.gradient(normed)
                ).astype(np.float32)

            # Acceleration (second derivative proxy)
            if self.include_acceleration:
                velocity = np.gradient(normed)
                results[f"{self.prefix}_{v}_acceleration"] = (
                    np.gradient(velocity)
                ).astype(np.float32)

            # Recency (exponential decay from max)
            if self.include_recency:
                # Placeholder: use distance from maximum value as recency
                max_val = np.nanmax(np.abs(normed)) if len(normed) > 0 else 1.0
                distance = max_val - np.abs(normed)
                results[f"{self.prefix}_{v}_recency"] = (
                    np.exp(-distance / self.recency_halflife)
                ).astype(np.float32)

        return pd.DataFrame(results, index=df.index)

    # -- Cyclical encoding helpers ------------------------------------

    def _generate_cyclical(
        self,
        df: pd.DataFrame,
        results: Dict[str, np.ndarray],
    ) -> None:
        """Generate sine/cosine cyclical encodings."""
        n_rows = len(df)

        # Try to extract time components from the time column
        time_values = None
        if self._has_time_column and self.time_column in df.columns:
            try:
                time_values = pd.to_datetime(df[self.time_column])
            except Exception:
                pass

        for period in self.cyclical_periods:
            if time_values is not None:
                raw = self._extract_period_values(time_values, period)
                max_val = self._get_period_max(period)
            else:
                # Fallback: use row index as proxy
                raw = np.arange(n_rows, dtype=np.float64)
                max_val = self._get_period_max(period)

            angle = 2 * np.pi * raw / max_val
            results[f"{self.prefix}_{period}_sin"] = np.sin(angle).astype(np.float32)
            results[f"{self.prefix}_{period}_cos"] = np.cos(angle).astype(np.float32)

    @staticmethod
    def _extract_period_values(
        dt_series: pd.Series, period: str
    ) -> np.ndarray:
        """Extract the numeric period value from a datetime series."""
        if period == "hour":
            return dt_series.dt.hour.values.astype(np.float64)
        elif period == "dayofweek":
            return dt_series.dt.dayofweek.values.astype(np.float64)
        elif period == "month":
            return dt_series.dt.month.values.astype(np.float64)
        elif period == "quarter":
            return dt_series.dt.quarter.values.astype(np.float64)
        else:
            return np.zeros(len(dt_series), dtype=np.float64)

    @staticmethod
    def _get_period_max(period: str) -> float:
        """Maximum value for a cyclical period (for angle normalisation)."""
        return {
            "hour": 24.0,
            "dayofweek": 7.0,
            "month": 12.0,
            "quarter": 4.0,
        }.get(period, 1.0)

    # -- Helpers -------------------------------------------------------

    def _resolve_value_columns(self, df: pd.DataFrame) -> List[str]:
        """Resolve value columns, falling back to all numeric."""
        if self.value_columns:
            return [c for c in self.value_columns if c in df.columns]
        # Exclude the time column from value columns
        numeric = df.select_dtypes(include=["number"]).columns.tolist()
        if self.time_column and self.time_column in numeric:
            numeric.remove(self.time_column)
        return numeric if numeric else ["signal"]
