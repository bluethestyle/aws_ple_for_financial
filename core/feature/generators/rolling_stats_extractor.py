"""
Rolling-window statistics generator -- on-prem 2026-04-25 redesign axis-2.

Complements the lag tensor (axis-1) by preserving total-volume signal that
gets truncated when len(seq) > K.  Per (window_days, metric) emits one
column.  Window is anchored at the most-recent observation:
    elapsed = max(day_offset_seq) - day_offset_seq[i]
    window includes events with elapsed <= window_days

Default windows: [7, 30, 90, 180] days.
Default metrics: [sum, mean, std, count, days_active]
Output dim default = 4 windows * 5 metrics = 20D per amount-stream.

For Santander where day_offset spans ~360 days, 180d covers ~half-year;
for higher-velocity domains the same generator applies with shorter windows.
"""
from __future__ import annotations

import logging
import math
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from ..generator import AbstractFeatureGenerator, FeatureGeneratorRegistry
from .gpu_utils import _to_pandas_safe
from .lag_extractor import _parse_seq

logger = logging.getLogger(__name__)


_VALID_METRICS = ("sum", "mean", "std", "count", "days_active")


@FeatureGeneratorRegistry.register(
    "rolling_stats_extractor",
    description="Rolling-window stats over LIST sequences (sum/mean/std/count/days_active).",
    tags=["rolling", "temporal", "axis2"],
)
class RollingStatsExtractor(AbstractFeatureGenerator):
    """Compute per-customer rolling-window stats over a value+day-offset pair.

    Config keys (all config-driven):
        amount_column      : str    LIST<float>, defaults to "txn_amount_seq"
        day_offset_column  : str    LIST<int>,   defaults to "txn_day_offset_seq"
        windows_days       : List[int]   default [7, 30, 90, 180]
        metrics            : List[str]   subset of {sum, mean, std, count, days_active}
        truncate_seq_last  : int    drop last N elements (label-leakage guard).
                                    Default 0.
        prefix             : str    default "txn_roll".

    Output column naming: f"{prefix}_w{window}d_{metric}"  (e.g. txn_roll_w30d_sum)
    """

    supports_gpu: bool = False
    required_libraries: List[str] = []

    def __init__(
        self,
        config: Optional[Dict[str, Any]] = None,
        prefix: str = "txn_roll",
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        cfg: Dict[str, Any] = config or {}
        for k in ("amount_column", "day_offset_column",
                  "windows_days", "metrics", "truncate_seq_last"):
            if k in kwargs and k not in cfg:
                cfg[k] = kwargs[k]
        self._cfg = cfg
        self.prefix = prefix

        self._amount_col: str = str(cfg.get("amount_column", "txn_amount_seq"))
        self._day_col:    str = str(cfg.get("day_offset_column", "txn_day_offset_seq"))
        self._windows: List[int] = [int(w) for w in cfg.get("windows_days", [7, 30, 90, 180])]
        if not self._windows:
            raise ValueError("rolling_stats_extractor requires non-empty windows_days")

        metrics = cfg.get("metrics") or list(_VALID_METRICS)
        self._metrics: List[str] = [str(m) for m in metrics]
        bad = [m for m in self._metrics if m not in _VALID_METRICS]
        if bad:
            raise ValueError(f"unknown metrics in rolling_stats_extractor: {bad}; "
                             f"valid={_VALID_METRICS}")

        self._truncate_last: int = int(cfg.get("truncate_seq_last", 0))
        if self._truncate_last < 0:
            raise ValueError("truncate_seq_last must be >= 0")

    # ------------------------------------------------------------------
    # Output contract
    # ------------------------------------------------------------------

    @property
    def output_dim(self) -> int:
        return len(self._windows) * len(self._metrics)

    @property
    def output_columns(self) -> List[str]:
        return [f"{self.prefix}_w{w}d_{m}"
                for w in self._windows
                for m in self._metrics]

    @classmethod
    def estimated_output_dim(cls, config: Dict[str, Any]) -> int:
        windows = config.get("windows_days") or [7, 30, 90, 180]
        metrics = config.get("metrics") or list(_VALID_METRICS)
        return len(windows) * len(metrics)

    # ------------------------------------------------------------------
    # Fit / Generate
    # ------------------------------------------------------------------

    def fit(self, df: Any, **context: Any) -> "RollingStatsExtractor":
        """No learned parameters; just validates required source columns."""
        pdf = _to_pandas_safe(df)
        present = set(pdf.columns)
        for col in (self._amount_col, self._day_col):
            if col not in present:
                logger.warning(
                    "RollingStatsExtractor: source column missing on fit "
                    "(will emit zero output): %s", col,
                )
        self._fitted = True
        return self

    def _row_stats(self, amounts: List[float], days: List[float]) -> np.ndarray:
        """Compute rolling stats for one customer.

        Returns shape (n_windows * n_metrics,) flat in window-major order.
        """
        n_w = len(self._windows)
        n_m = len(self._metrics)
        out = np.zeros(n_w * n_m, dtype=np.float32)

        n = min(len(amounts), len(days))
        if n == 0:
            return out

        a = np.asarray(amounts[:n], dtype=np.float64)
        d = np.asarray(days[:n],    dtype=np.float64)
        anchor = float(d.max())  # most-recent observation = window anchor
        elapsed = anchor - d     # 0 means "today", larger = older

        for wi, w in enumerate(self._windows):
            mask = elapsed <= float(w)
            if not mask.any():
                continue
            vals = a[mask]
            dvals = d[mask]
            base = wi * n_m
            for mi, metric in enumerate(self._metrics):
                if metric == "sum":
                    out[base + mi] = float(vals.sum())
                elif metric == "mean":
                    out[base + mi] = float(vals.mean())
                elif metric == "std":
                    out[base + mi] = float(vals.std()) if len(vals) > 1 else 0.0
                elif metric == "count":
                    out[base + mi] = float(len(vals))
                elif metric == "days_active":
                    out[base + mi] = float(len(np.unique(dvals.astype(np.int64))))

        # Guard against any nan/inf from degenerate stats
        if not np.all(np.isfinite(out)):
            out = np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)
        return out

    def generate(self, df: Any, **context: Any) -> pd.DataFrame:
        if not self._fitted:
            raise RuntimeError("RollingStatsExtractor must be fitted before generate().")

        pdf = _to_pandas_safe(df)
        n_rows = len(pdf)
        present = set(pdf.columns)
        a_present = self._amount_col in present
        d_present = self._day_col    in present
        trunc = self._truncate_last
        out_dim = self.output_dim
        out = np.zeros((n_rows, out_dim), dtype=np.float32)

        if a_present and d_present:
            a_series = pdf[self._amount_col]
            d_series = pdf[self._day_col]
            for ri in range(n_rows):
                amts = _parse_seq(a_series.iat[ri])
                days = _parse_seq(d_series.iat[ri])
                if trunc > 0:
                    amts = amts[:-trunc] if len(amts) > trunc else []
                    days = days[:-trunc] if len(days) > trunc else []
                if not amts or not days:
                    continue
                out[ri] = self._row_stats(amts, days)
        else:
            logger.warning(
                "RollingStatsExtractor: amount_col=%s present=%s, day_col=%s present=%s "
                "-- emitting zero output", self._amount_col, a_present,
                self._day_col, d_present,
            )

        if out.shape[0] != n_rows:
            raise RuntimeError(
                f"RollingStatsExtractor row-count mismatch: "
                f"input={n_rows}, output={out.shape[0]}"
            )

        return pd.DataFrame(out, columns=self.output_columns, index=pdf.index)
