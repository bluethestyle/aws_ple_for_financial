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
from .gpu_utils import has_cudf, has_cupy

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Lazy cuDF / CuPy imports
# ---------------------------------------------------------------------------
_cudf = None
_CUDF_AVAILABLE: Optional[bool] = None

_cupy = None
_CUPY_AVAILABLE: Optional[bool] = None


def _get_cudf():
    """Lazy-load cuDF and cache the result."""
    global _cudf, _CUDF_AVAILABLE
    if _CUDF_AVAILABLE is not None:
        return _cudf
    try:
        import cudf as _c
        _cudf = _c
        _CUDF_AVAILABLE = True
        logger.info("Temporal: cuDF available for GPU-accelerated operations")
    except (ImportError, Exception):
        _cudf = None
        _CUDF_AVAILABLE = False
    return _cudf


def _get_cupy():
    """Lazy-load CuPy and cache the result."""
    global _cupy, _CUPY_AVAILABLE
    if _CUPY_AVAILABLE is not None:
        return _cupy
    try:
        import cupy as _cp
        _cp.cuda.Device(0).compute_capability
        _cupy = _cp
        _CUPY_AVAILABLE = True
        logger.info("Temporal: CuPy available for GPU-accelerated EWM")
    except (ImportError, Exception):
        _cupy = None
        _CUPY_AVAILABLE = False
    return _cupy


def _ewm_mean_cupy(arr_np: np.ndarray, halflife: float) -> np.ndarray:
    """Compute exponentially-weighted mean using CuPy on GPU.

    cuDF does not support ``ewm()``, so we implement it manually via CuPy.
    The recurrence is: result[i] = alpha * arr[i] + (1 - alpha) * result[i-1]
    where alpha = 1 - exp(-ln(2) / halflife).

    Parameters
    ----------
    arr_np : np.ndarray
        1-D float64 input array (on CPU).
    halflife : float
        Half-life in the same units as the time index.

    Returns
    -------
    np.ndarray
        EWM result on CPU as float32.
    """
    cp = _get_cupy()
    if cp is None:
        raise ImportError("CuPy is not available")
    alpha = 1.0 - np.exp(-np.log(2.0) / halflife)
    one_minus_alpha = 1.0 - alpha
    arr_gpu = cp.asarray(arr_np, dtype=cp.float64)
    result = cp.empty_like(arr_gpu)
    n = len(arr_gpu)
    if n == 0:
        return np.empty(0, dtype=np.float32)
    # Sequential scan -- runs on GPU but is inherently serial.
    # For large arrays (>100k) the GPU memory bandwidth still helps.
    result[0] = arr_gpu[0]
    for i in range(1, n):
        result[i] = alpha * arr_gpu[i] + one_minus_alpha * result[i - 1]
    return cp.asnumpy(result).astype(np.float32)


def _is_cudf_frame(obj: Any) -> bool:
    """Check if *obj* is a cuDF DataFrame or Series without importing cudf."""
    type_name = type(obj).__module__
    return "cudf" in type_name


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

    # -- Input columns declaration -----------------------------------------

    @property
    def input_cols(self) -> List[str]:
        """Source columns consumed by fit() and generate().

        Returns the union of ``time_column`` (if set) and ``value_columns``
        (if explicitly declared), plus any additional resolved value columns
        discovered during fit().  Derivable from existing config attrs and
        fitted state with no new __init__ variables.
        """
        cols: List[str] = []
        if self.time_column:
            cols.append(self.time_column)
        for c in self.value_columns:
            if c not in cols:
                cols.append(c)
        # After fit, include any auto-resolved value cols not in the explicit list
        for c in self._resolved_value_cols:
            if c not in cols:
                cols.append(c)
        return cols

    # -- Core API ----------------------------------------------------------

    def fit(self, df: Any, **context: Any) -> "TemporalPatternGenerator":
        """Learn normalisation statistics from training data."""
        # Extract declared input columns once to bound memory access.
        # _resolved_value_cols is empty before fit so input_cols = time_col + value_columns.
        fit_input_cols = (
            ([self.time_column] if self.time_column else []) + list(self.value_columns)
        ) or None  # None -> extract all for the auto-resolve path
        col_arrays = self._input_to_numpy(df, columns=fit_input_cols)

        gdf = self._to_working_frame(df)

        col_list = list(gdf.columns) if hasattr(gdf, "columns") else []
        self._has_time_column = (
            self.time_column is not None and self.time_column in col_list
        )

        self._resolved_value_cols = self._resolve_value_columns(gdf)

        self._value_means = {}
        self._value_stds = {}
        for col in self._resolved_value_cols:
            if col in col_arrays:
                arr = col_arrays[col].astype(np.float64)
                mean_val = float(np.nanmean(arr))
                std_val = float(np.nanstd(arr))
            else:
                # Fallback to working frame for cols discovered by auto-resolve
                mean_val = float(gdf[col].mean())
                std_val = float(gdf[col].std())
            self._value_means[col] = mean_val
            self._value_stds[col] = std_val if std_val > 0 else 1.0

        self._fitted = True
        logger.info(
            "TemporalPatternGenerator fitted: %d value columns, "
            "%d windows, %d stats/window, %d cyclical periods, output_dim=%d, "
            "cudf_available=%s, cupy_available=%s",
            len(self._resolved_value_cols),
            len(self.config.windows),
            len(self.config.stats),
            len(self.config.cyclical_periods),
            self.output_dim,
            has_cudf(),
            has_cupy(),
        )
        return self

    def generate(self, df: Any, **context: Any) -> Any:
        """Generate temporal features using real rolling-window computations.

        When cuDF is available and the DataFrame exceeds ``cudf_min_rows``,
        all rolling, diff, and sort operations run on the GPU.  EWM is
        offloaded to CuPy (cuDF lacks ``ewm()`` support).  Falls back to
        pandas transparently when RAPIDS is unavailable.
        """
        if not self._fitted:
            raise RuntimeError(
                "TemporalPatternGenerator must be fitted before generate()."
            )

        # Extract declared input columns once to bound memory access.
        gen_input_cols = self.input_cols or None  # None -> extract all when empty
        col_arrays = self._input_to_numpy(df, columns=gen_input_cols) if gen_input_cols else {}

        gdf = self._to_working_frame(df)
        n_rows = (
            len(next(iter(col_arrays.values())))
            if col_arrays
            else len(gdf)
        )
        results: Dict[str, np.ndarray] = {}

        # -- Decide GPU vs CPU path ----------------------------------------
        cudf_mod = _get_cudf()
        use_gpu = (
            cudf_mod is not None
            and n_rows >= self.config.cudf_min_rows
        )

        # -- Cyclical encodings (lightweight, always numpy) ----------------
        self._generate_cyclical(gdf, results)

        # -- Ensure data is sorted by time if possible ---------------------
        working = gdf.copy()
        if self._has_time_column and self.time_column is not None:
            try:
                if use_gpu and _is_cudf_frame(working):
                    working[self.time_column] = cudf_mod.to_datetime(
                        working[self.time_column]
                    )
                else:
                    working[self.time_column] = pd.to_datetime(
                        working[self.time_column]
                    )
                working = working.sort_values(self.time_column).reset_index(drop=True)
            except Exception:
                pass

        if use_gpu:
            logger.debug(
                "Temporal: using cuDF GPU-accelerated pipeline (%d rows)", n_rows
            )
            # Ensure working is a cuDF DataFrame for the GPU path
            if not _is_cudf_frame(working):
                try:
                    working = cudf_mod.DataFrame(working)
                except Exception:
                    use_gpu = False

        # -- Per-value-column rolling features -----------------------------
        val_cols = self._resolved_value_cols

        for v in val_cols:
            if v in working.columns:
                series = working[v].astype(np.float64)
            else:
                if use_gpu and cudf_mod is not None:
                    series = cudf_mod.Series(np.zeros(n_rows, dtype=np.float64))
                else:
                    series = pd.Series(np.zeros(n_rows), dtype=np.float64)

            for w in self.config.windows:
                rolling_obj = series.rolling(window=w, min_periods=1)

                for stat in self.config.stats:
                    col_name = f"{self.prefix}_{v}_roll{w}_{stat}"

                    if stat == "trend":
                        # OLS slope -- cuDF has no custom rolling apply,
                        # always use CPU numpy.
                        vals_np = self._series_to_numpy(series)
                        results[col_name] = self._rolling_trend(
                            vals_np, w
                        ).astype(np.float32)

                    elif stat in ("mean", "std", "min", "max"):
                        try:
                            agg = getattr(rolling_obj, stat)()
                            if stat == "std":
                                agg = agg.fillna(0.0)
                            results[col_name] = self._series_to_numpy(
                                agg
                            ).astype(np.float32)
                        except Exception:
                            # Fallback: convert to pandas and retry
                            pd_series = self._to_pandas_series(series)
                            pd_rolling = pd_series.rolling(window=w, min_periods=1)
                            agg = getattr(pd_rolling, stat)()
                            if stat == "std":
                                agg = agg.fillna(0.0)
                            results[col_name] = agg.values.astype(np.float32)
                    else:
                        logger.warning("Unknown stat '%s', filling with zeros.", stat)
                        results[col_name] = np.zeros(n_rows, dtype=np.float32)

            # Velocity (first difference) -- cuDF supports .diff()
            if self.config.include_velocity:
                try:
                    results[f"{self.prefix}_{v}_velocity"] = (
                        self._series_to_numpy(
                            series.diff().fillna(0.0)
                        ).astype(np.float32)
                    )
                except Exception:
                    pd_s = self._to_pandas_series(series)
                    results[f"{self.prefix}_{v}_velocity"] = (
                        pd_s.diff().fillna(0.0).values.astype(np.float32)
                    )

            # Acceleration (second difference) -- cuDF supports chained .diff()
            if self.config.include_acceleration:
                try:
                    results[f"{self.prefix}_{v}_acceleration"] = (
                        self._series_to_numpy(
                            series.diff().diff().fillna(0.0)
                        ).astype(np.float32)
                    )
                except Exception:
                    pd_s = self._to_pandas_series(series)
                    results[f"{self.prefix}_{v}_acceleration"] = (
                        pd_s.diff().diff().fillna(0.0).values.astype(np.float32)
                    )

            # Exponential recency weighting
            # cuDF does NOT support .ewm(), so use CuPy when available.
            if self.config.include_recency:
                halflife = self.config.recency_halflife
                recency_key = f"{self.prefix}_{v}_recency"

                vals_np = self._series_to_numpy(series)
                cupy_mod = _get_cupy()
                if cupy_mod is not None:
                    try:
                        results[recency_key] = _ewm_mean_cupy(vals_np, halflife)
                    except Exception:
                        # CuPy failed -- fall back to pandas ewm
                        pd_s = pd.Series(vals_np)
                        results[recency_key] = (
                            pd_s.ewm(halflife=halflife, min_periods=1)
                            .mean()
                            .values
                            .astype(np.float32)
                        )
                else:
                    pd_s = pd.Series(vals_np)
                    results[recency_key] = (
                        pd_s.ewm(halflife=halflife, min_periods=1)
                        .mean()
                        .values
                        .astype(np.float32)
                    )

        # Reindex to match original DataFrame order.
        # df_backend.from_dict expects numpy arrays; results already are.
        original_index = (
            gdf.index.to_pandas() if _is_cudf_frame(gdf)
            else gdf.index
        )
        return df_backend.from_dict(results, index=original_index)

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
        df: Any,
        results: Dict[str, np.ndarray],
    ) -> None:
        """Generate sin/cos cyclical encodings from datetime or index.

        Accepts either a pandas or cuDF DataFrame.  Datetime extraction
        always happens via pandas (lightweight operation) to avoid cuDF
        datetime accessor edge cases.
        """
        n_rows = len(df)
        time_values: Optional[pd.Series] = None

        col_list = list(df.columns) if hasattr(df, "columns") else []
        if self._has_time_column and self.time_column is not None and self.time_column in col_list:
            try:
                raw_col = self._series_to_numpy(df[self.time_column])
                time_values = pd.to_datetime(pd.Series(raw_col))
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

    def _to_working_frame(self, df: Any) -> Any:
        """Convert input to cuDF DataFrame (GPU) or pandas DataFrame (CPU).

        Attempts cuDF first when available.  Falls back to pandas via
        ``df_backend.to_pandas()`` for non-DataFrame inputs (Arrow tables,
        DuckDB relations, etc.).
        """
        cudf_mod = _get_cudf()

        # Already cuDF
        if cudf_mod is not None and _is_cudf_frame(df):
            return df

        # Already pandas
        if isinstance(df, pd.DataFrame):
            if cudf_mod is not None:
                try:
                    return cudf_mod.DataFrame(df)
                except Exception:
                    return df
            return df

        # Other backend (Arrow, DuckDB, etc.) -- convert to pandas first
        pdf = df_backend.to_pandas(df) if not isinstance(df, pd.DataFrame) else df
        if cudf_mod is not None:
            try:
                return cudf_mod.DataFrame(pdf)
            except Exception:
                return pdf
        return pdf

    @staticmethod
    def _series_to_numpy(series: Any) -> np.ndarray:
        """Extract a numpy array from a cuDF or pandas Series."""
        if _is_cudf_frame(series):
            # cuDF Series: .values_host gives a numpy array on CPU
            try:
                return series.values_host
            except AttributeError:
                return series.to_pandas().values
        if hasattr(series, "values"):
            return series.values
        return np.asarray(series)

    @staticmethod
    def _to_pandas_series(series: Any) -> pd.Series:
        """Convert a cuDF or other Series to pandas."""
        if _is_cudf_frame(series):
            return series.to_pandas()
        if isinstance(series, pd.Series):
            return series
        return pd.Series(np.asarray(series))

    def _resolve_value_columns(self, df: Any) -> List[str]:
        """Resolve value columns, falling back to all numeric.

        Works with both cuDF and pandas DataFrames.  ``select_dtypes()``
        is supported by both backends with the same API.
        """
        if self.value_columns:
            col_list = list(df.columns) if hasattr(df, "columns") else []
            return [c for c in self.value_columns if c in col_list]
        try:
            numeric = df.select_dtypes(include=["number"]).columns.tolist()
        except Exception:
            # Fallback for unusual backends
            numeric = [
                c for c in df.columns
                if str(df[c].dtype).startswith(("int", "float", "uint"))
            ]
        if self.time_column and self.time_column in numeric:
            numeric.remove(self.time_column)
        return numeric if numeric else ["signal"]
