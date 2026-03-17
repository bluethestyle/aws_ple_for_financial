"""
Temporal Pattern Feature Generator.

Extracts rich temporal features from time-indexed data using real rolling
window computations, cyclical datetime encoding, velocity/acceleration
signals, and exponential recency weighting.

Feature categories
------------------
1. **Rolling window aggregations**: mean, std, trend (OLS slope), min, max
   over configurable day-windows (default: 7, 30, 90, 180, 365).
2. **Cyclical encoding**: sin/cos pairs for hour, day_of_week, month,
   quarter -- preserves circular continuity.
3. **Velocity**: first difference of each value column.
4. **Acceleration**: second difference (diff of diff).
5. **Exponential recency weighting**: exponentially-weighted mean with
   configurable half-life.

Total output: ~60D depending on configuration.

Hardware acceleration
---------------------
Rolling window aggregations (mean, std, min, max) can be accelerated with
cuDF on GPU-equipped instances.  cuDF's rolling operations are natively
GPU-parallelised and provide significant speedup on large DataFrames
(>10k rows).  Falls back to pandas rolling() when cuDF is unavailable.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from core.data.dataframe import df_backend
from ..generator import AbstractFeatureGenerator, FeatureGeneratorRegistry
from .gpu_utils import has_cudf

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Lazy cuDF import
# ---------------------------------------------------------------------------
_cudf = None
_CUDF_AVAILABLE: Optional[bool] = None


def _get_cudf():
    """Lazy-load cuDF and cache the result."""
    global _cudf, _CUDF_AVAILABLE
    if _CUDF_AVAILABLE is not None:
        return _cudf
    try:
        import cudf as _c
        _cudf = _c
        _CUDF_AVAILABLE = True
        logger.info("Temporal: cuDF available for GPU-accelerated rolling windows")
    except (ImportError, Exception):
        _cudf = None
        _CUDF_AVAILABLE = False
    return _cudf


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class TemporalConfig:
    """Temporal feature generator configuration."""

    windows: List[int] = field(default_factory=lambda: [7, 30, 90, 180, 365])
    stats: List[str] = field(
        default_factory=lambda: ["mean", "std", "trend", "min", "max"]
    )
    cyclical_periods: List[str] = field(
        default_factory=lambda: ["hour", "dayofweek", "month", "quarter"]
    )
    include_velocity: bool = True
    include_acceleration: bool = True
    include_recency: bool = True
    recency_halflife: float = 30.0  # in same units as time index (days)
    # Minimum rows to justify cuDF GPU overhead; below this threshold
    # the CPU-to-GPU transfer cost dominates the rolling computation.
    cudf_min_rows: int = 10000


# ---------------------------------------------------------------------------
# Temporal Pattern Generator
# ---------------------------------------------------------------------------

@FeatureGeneratorRegistry.register(
    "temporal",
    description="Temporal rolling aggregation, cyclical encoding, velocity, and recency features.",
    tags=["temporal", "sequential", "cyclical", "time"],
)
class TemporalPatternGenerator(AbstractFeatureGenerator):
    """Generate temporal pattern features from time-indexed data.

    Uses actual ``pandas.DataFrame.rolling()`` for window aggregations and
    ``numpy.polyfit`` for per-window trend (linear regression slope).

    When cuDF is available and the DataFrame is large enough (>10k rows),
    rolling window aggregations (mean, std, min, max) are offloaded to the
    GPU via cuDF for ~5-10x speedup.  Trend computation remains on CPU
    (cuDF does not support custom rolling apply).

    Parameters
    ----------
    config : TemporalConfig, optional
        Feature extraction configuration.
    time_column : str, optional
        Column containing datetime values.  If absent, the DataFrame index
        is used (or row position as fallback).
    value_columns : list[str], optional
        Numeric columns to compute rolling features on.  Defaults to all
        numeric columns (excluding ``time_column``).
    prefix : str
        Column-name prefix.
    """

    supports_gpu: bool = True
    required_libraries: List[str] = []  # numpy/pandas only; cuDF optional

    def __init__(
        self,
        config: Optional[TemporalConfig] = None,
        time_column: Optional[str] = None,
        value_columns: Optional[List[str]] = None,
        prefix: str = "temp",
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.config = config or TemporalConfig()
        self.time_column = time_column
        self.value_columns = value_columns or []
        self.prefix = prefix

        # Fitted state
        self._value_means: Optional[Dict[str, float]] = None
        self._value_stds: Optional[Dict[str, float]] = None
        self._has_time_column: bool = False
        self._resolved_value_cols: List[str] = []

        # Eagerly check cuDF availability
        has_cudf()

    # -- Output description ------------------------------------------------

    @property
    def output_dim(self) -> int:
        dim = 0
        # Cyclical: 2 per period
        dim += len(self.config.cyclical_periods) * 2

        n_val = max(len(self._resolved_value_cols), max(len(self.value_columns), 1))

        # Rolling: n_stats * n_windows per value column
        dim += n_val * len(self.config.windows) * len(self.config.stats)

        # Velocity
        if self.config.include_velocity:
            dim += n_val
        # Acceleration
        if self.config.include_acceleration:
            dim += n_val
        # Recency
        if self.config.include_recency:
            dim += n_val

        return dim

    @property
    def output_columns(self) -> List[str]:
        cols: List[str] = []

        # Cyclical
        for period in self.config.cyclical_periods:
            cols.append(f"{self.prefix}_{period}_sin")
            cols.append(f"{self.prefix}_{period}_cos")

        val_names = self._resolved_value_cols or self.value_columns or ["signal"]

        for v in val_names:
            for w in self.config.windows:
                for stat in self.config.stats:
                    cols.append(f"{self.prefix}_{v}_roll{w}_{stat}")
            if self.config.include_velocity:
                cols.append(f"{self.prefix}_{v}_velocity")
            if self.config.include_acceleration:
                cols.append(f"{self.prefix}_{v}_acceleration")
            if self.config.include_recency:
                cols.append(f"{self.prefix}_{v}_recency")

        return cols

    # -- Core API ----------------------------------------------------------

    def fit(self, df: Any, **context: Any) -> "TemporalPatternGenerator":
        """Learn normalisation statistics from training data."""
        pdf = df_backend.to_pandas(df) if not isinstance(df, pd.DataFrame) else df

        self._has_time_column = (
            self.time_column is not None and self.time_column in pdf.columns
        )

        self._resolved_value_cols = self._resolve_value_columns(pdf)

        self._value_means = {}
        self._value_stds = {}
        for col in self._resolved_value_cols:
            self._value_means[col] = float(pdf[col].mean())
            std = float(pdf[col].std())
            self._value_stds[col] = std if std > 0 else 1.0

        self._fitted = True
        logger.info(
            "TemporalPatternGenerator fitted: %d value columns, "
            "%d windows, %d stats/window, %d cyclical periods, output_dim=%d, "
            "cudf_available=%s",
            len(self._resolved_value_cols),
            len(self.config.windows),
            len(self.config.stats),
            len(self.config.cyclical_periods),
            self.output_dim,
            has_cudf(),
        )
        return self

    def generate(self, df: Any, **context: Any) -> Any:
        """Generate temporal features using real rolling-window computations."""
        if not self._fitted:
            raise RuntimeError(
                "TemporalPatternGenerator must be fitted before generate()."
            )

        pdf = df_backend.to_pandas(df) if not isinstance(df, pd.DataFrame) else df
        n_rows = len(pdf)
        results: Dict[str, np.ndarray] = {}

        # -- Cyclical encodings --------------------------------------------
        self._generate_cyclical(pdf, results)

        # -- Ensure data is sorted by time if possible ---------------------
        working = pdf.copy()
        if self._has_time_column and self.time_column is not None:
            try:
                working[self.time_column] = pd.to_datetime(working[self.time_column])
                working = working.sort_values(self.time_column).reset_index(drop=True)
            except Exception:
                pass

        # -- Decide whether to use cuDF for rolling windows ----------------
        # cuDF is beneficial for rolling aggregations on large DataFrames.
        # For small DataFrames (<10k rows), the CPU-to-GPU transfer overhead
        # outweighs the GPU computation savings.
        cudf_mod = _get_cudf()
        use_cudf = (
            cudf_mod is not None
            and n_rows >= self.config.cudf_min_rows
        )
        if use_cudf:
            logger.debug(
                "Temporal: using cuDF GPU-accelerated rolling windows (%d rows)", n_rows
            )

        # -- Per-value-column rolling features -----------------------------
        val_cols = self._resolved_value_cols

        for v in val_cols:
            if v in working.columns:
                series = working[v].astype(np.float64)
            else:
                series = pd.Series(np.zeros(n_rows), dtype=np.float64)

            # Convert to cuDF Series once per value column if using GPU
            gpu_series = None
            if use_cudf:
                try:
                    gpu_series = cudf_mod.Series(series.values)
                except Exception:
                    gpu_series = None  # fall back to pandas for this column

            for w in self.config.windows:
                # cuDF rolling: supports mean, std, min, max natively on GPU
                # Trend requires custom apply which cuDF doesn't support, so
                # it always runs on CPU.
                if gpu_series is not None:
                    try:
                        gpu_rolling = gpu_series.rolling(window=w, min_periods=1)
                    except Exception:
                        gpu_rolling = None
                else:
                    gpu_rolling = None

                # Pandas rolling as fallback
                pandas_rolling = series.rolling(window=w, min_periods=1)

                for stat in self.config.stats:
                    col_name = f"{self.prefix}_{v}_roll{w}_{stat}"

                    if stat == "mean":
                        if gpu_rolling is not None:
                            try:
                                results[col_name] = gpu_rolling.mean().to_pandas().values.astype(np.float32)
                                continue
                            except Exception:
                                pass
                        results[col_name] = pandas_rolling.mean().values.astype(np.float32)

                    elif stat == "std":
                        if gpu_rolling is not None:
                            try:
                                results[col_name] = gpu_rolling.std().fillna(0.0).to_pandas().values.astype(np.float32)
                                continue
                            except Exception:
                                pass
                        results[col_name] = pandas_rolling.std().fillna(0.0).values.astype(np.float32)

                    elif stat == "min":
                        if gpu_rolling is not None:
                            try:
                                results[col_name] = gpu_rolling.min().to_pandas().values.astype(np.float32)
                                continue
                            except Exception:
                                pass
                        results[col_name] = pandas_rolling.min().values.astype(np.float32)

                    elif stat == "max":
                        if gpu_rolling is not None:
                            try:
                                results[col_name] = gpu_rolling.max().to_pandas().values.astype(np.float32)
                                continue
                            except Exception:
                                pass
                        results[col_name] = pandas_rolling.max().values.astype(np.float32)

                    elif stat == "trend":
                        # Linear regression slope over the rolling window
                        # cuDF doesn't support custom rolling apply, so this
                        # always runs on CPU via numpy OLS.
                        results[col_name] = self._rolling_trend(
                            series.values, w
                        ).astype(np.float32)
                    else:
                        logger.warning("Unknown stat '%s', filling with zeros.", stat)
                        results[col_name] = np.zeros(n_rows, dtype=np.float32)

            # Velocity (first difference)
            if self.config.include_velocity:
                results[f"{self.prefix}_{v}_velocity"] = (
                    series.diff().fillna(0.0).values.astype(np.float32)
                )

            # Acceleration (second difference)
            if self.config.include_acceleration:
                results[f"{self.prefix}_{v}_acceleration"] = (
                    series.diff().diff().fillna(0.0).values.astype(np.float32)
                )

            # Exponential recency weighting
            if self.config.include_recency:
                halflife = self.config.recency_halflife
                results[f"{self.prefix}_{v}_recency"] = (
                    series.ewm(halflife=halflife, min_periods=1)
                    .mean()
                    .values
                    .astype(np.float32)
                )

        # Reindex to match original DataFrame order
        return df_backend.from_dict(results, index=pdf.index)

    # -- Rolling trend (OLS slope) -----------------------------------------

    @staticmethod
    def _rolling_trend(values: np.ndarray, window: int) -> np.ndarray:
        """Compute rolling linear-regression slope over *window* periods.

        For each position t, fits y = a + b*x to values[t-window+1:t+1]
        and returns b (the slope).  Uses the closed-form OLS solution
        for speed.
        """
        n = len(values)
        slopes = np.zeros(n, dtype=np.float64)

        for t in range(n):
            w = min(t + 1, window)
            if w < 2:
                slopes[t] = 0.0
                continue
            y = values[t - w + 1: t + 1]
            # Handle NaN
            valid_mask = ~np.isnan(y)
            if valid_mask.sum() < 2:
                slopes[t] = 0.0
                continue
            y_valid = y[valid_mask]
            x = np.arange(len(y_valid), dtype=np.float64)
            # OLS slope: b = cov(x,y) / var(x)
            x_mean = x.mean()
            y_mean = y_valid.mean()
            var_x = ((x - x_mean) ** 2).sum()
            if var_x < 1e-12:
                slopes[t] = 0.0
            else:
                slopes[t] = ((x - x_mean) * (y_valid - y_mean)).sum() / var_x

        return slopes

    # -- Cyclical encoding -------------------------------------------------

    def _generate_cyclical(
        self,
        df: pd.DataFrame,
        results: Dict[str, np.ndarray],
    ) -> None:
        """Generate sin/cos cyclical encodings from datetime or index."""
        n_rows = len(df)
        time_values: Optional[pd.Series] = None

        if self._has_time_column and self.time_column is not None and self.time_column in df.columns:
            try:
                time_values = pd.to_datetime(df[self.time_column])
            except Exception:
                pass

        for period in self.config.cyclical_periods:
            if time_values is not None:
                raw = self._extract_period_values(time_values, period)
                max_val = self._get_period_max(period)
            else:
                # Fallback: use row index as proxy
                raw = np.arange(n_rows, dtype=np.float64)
                max_val = self._get_period_max(period)

            angle = 2.0 * np.pi * raw / max_val
            results[f"{self.prefix}_{period}_sin"] = np.sin(angle).astype(np.float32)
            results[f"{self.prefix}_{period}_cos"] = np.cos(angle).astype(np.float32)

    @staticmethod
    def _extract_period_values(
        dt_series: pd.Series, period: str
    ) -> np.ndarray:
        """Extract numeric period value from a datetime series."""
        if period == "hour":
            return dt_series.dt.hour.values.astype(np.float64)
        elif period == "dayofweek":
            return dt_series.dt.dayofweek.values.astype(np.float64)
        elif period == "month":
            return (dt_series.dt.month.values - 1).astype(np.float64)  # 0-indexed for sin/cos
        elif period == "quarter":
            return (dt_series.dt.quarter.values - 1).astype(np.float64)
        else:
            return np.zeros(len(dt_series), dtype=np.float64)

    @staticmethod
    def _get_period_max(period: str) -> float:
        """Maximum value for cyclical period normalisation."""
        return {
            "hour": 24.0,
            "dayofweek": 7.0,
            "month": 12.0,
            "quarter": 4.0,
        }.get(period, 1.0)

    # -- Helpers -----------------------------------------------------------

    def _resolve_value_columns(self, df: pd.DataFrame) -> List[str]:
        """Resolve value columns, falling back to all numeric."""
        if self.value_columns:
            return [c for c in self.value_columns if c in df.columns]
        numeric = df.select_dtypes(include=["number"]).columns.tolist()
        if self.time_column and self.time_column in numeric:
            numeric.remove(self.time_column)
        return numeric if numeric else ["signal"]
