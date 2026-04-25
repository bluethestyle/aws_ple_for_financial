"""
Lag tensor feature generator -- explicit time-lag flattening of LIST sequences.

Implements the on-prem 2026-04-25 redesign axis-1 ("Sequence Lag") on AWS:
takes LIST<int|float> columns (e.g. txn_amount_seq, txn_mcc_seq,
txn_day_offset_seq, txn_hour_seq) and emits K explicit lag columns per
input feature with right-aligned ordering (lag_001 = most recent).

Output schema (K=200, features=[amount, mcc, day_offset, hour]):
  txn_lag_amount_001 .. txn_lag_amount_200
  txn_lag_mcc_001    .. txn_lag_mcc_200
  txn_lag_day_001    .. txn_lag_day_200
  txn_lag_hour_001   .. txn_lag_hour_200
  -> 800 columns total

Right-alignment rationale:
  * len(seq) <  K  -> left-pad with 0   (lag positions 1..n filled)
  * len(seq) >= K  -> keep most-recent K (oldest dropped, "right-side wins")

Output is one row per input row (1 customer = 1 row invariant preserved).
"""
from __future__ import annotations

import ast
import logging
import math
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from ..generator import AbstractFeatureGenerator, FeatureGeneratorRegistry
from .gpu_utils import _to_pandas_safe

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Sequence parsing helpers (mirror merchant_hierarchy._parse_*)
# ---------------------------------------------------------------------------

def _parse_seq(raw: Any) -> List[float]:
    """Parse a LIST cell (list / tuple / string-encoded list / scalar / None)
    into a list[float]. Returns [] for nan / empty / unparseable input."""
    if raw is None:
        return []
    if isinstance(raw, (list, tuple)):
        return [float(v) for v in raw if v is not None]
    if hasattr(raw, "tolist"):
        try:
            return [float(v) for v in raw.tolist() if v is not None]
        except Exception:
            pass
    if isinstance(raw, str):
        s = raw.strip()
        if not s or s in ("nan", "None", "[]"):
            return []
        try:
            p = ast.literal_eval(s)
            return [float(v) for v in (p if isinstance(p, (list, tuple)) else [p])
                    if v is not None]
        except Exception:
            return []
    try:
        v = float(raw)
        return [] if math.isnan(v) else [v]
    except Exception:
        return []


# ---------------------------------------------------------------------------
# Generator
# ---------------------------------------------------------------------------

@FeatureGeneratorRegistry.register(
    "lag_extractor",
    description="K-step right-aligned lag flattening of LIST sequence columns.",
    tags=["lag", "temporal", "sequence", "axis1"],
)
class LagFeatureExtractor(AbstractFeatureGenerator):
    """Flatten LIST sequences into K explicit lag columns per feature.

    Config keys (all config-driven, no hardcoding):
        sequence_columns : Dict[str, str]
            short_name -> source column name. Example:
                {"amount": "txn_amount_seq", "mcc": "txn_mcc_seq",
                 "day": "txn_day_offset_seq", "hour": "txn_hour_seq"}
        k : int
            Number of lag positions to emit (default 200, matches Santander cap).
        prefix : str
            Column-name prefix. Default "txn_lag".
        truncate_seq_last : int
            Drop last N elements before flattening (label-leakage guard).
            Default 0.  Set to 1 when next_mcc / top_mcc_shift labels are
            derived from the last sequence position.
        pad_value : float
            Fill value for short sequences. Default 0.0.
        cast : Dict[str, str]
            Optional per-feature dtype hint ("int" or "float").  MCC and hour
            are typically int; amount and day_offset are float.
            Only affects internal np dtype, output is always float32.
    """

    supports_gpu: bool = False
    required_libraries: List[str] = []

    def __init__(
        self,
        config: Optional[Dict[str, Any]] = None,
        prefix: str = "txn_lag",
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        cfg: Dict[str, Any] = config or {}
        # Allow params via either `config={...}` or top-level kwargs (for
        # registry.create() convenience).
        for k in ("sequence_columns", "k", "truncate_seq_last", "pad_value", "cast"):
            if k in kwargs and k not in cfg:
                cfg[k] = kwargs[k]
        self._cfg: Dict[str, Any] = cfg
        self.prefix = prefix

        seq_cols = cfg.get("sequence_columns")
        if not seq_cols or not isinstance(seq_cols, dict):
            raise ValueError(
                "LagFeatureExtractor requires `sequence_columns` mapping "
                "(short_name -> source column), e.g. "
                "{'amount': 'txn_amount_seq', 'mcc': 'txn_mcc_seq'}"
            )
        # Preserve insertion order for column emission ordering
        self._seq_cols: Dict[str, str] = {str(k): str(v) for k, v in seq_cols.items()}
        self._feat_names: List[str] = list(self._seq_cols.keys())

        self._k: int = int(cfg.get("k", 200))
        if self._k <= 0:
            raise ValueError(f"k must be positive, got {self._k}")

        self._truncate_last: int = int(cfg.get("truncate_seq_last", 0))
        if self._truncate_last < 0:
            raise ValueError(f"truncate_seq_last must be >= 0, got {self._truncate_last}")

        self._pad_value: float = float(cfg.get("pad_value", 0.0))
        cast_cfg = cfg.get("cast", {}) or {}
        self._cast: Dict[str, str] = {str(k): str(v) for k, v in cast_cfg.items()}

    # ------------------------------------------------------------------
    # Output contract
    # ------------------------------------------------------------------

    @property
    def output_dim(self) -> int:
        return self._k * len(self._feat_names)

    @property
    def output_columns(self) -> List[str]:
        cols: List[str] = []
        for feat in self._feat_names:
            cols.extend(f"{self.prefix}_{feat}_{i:03d}" for i in range(1, self._k + 1))
        return cols

    @classmethod
    def estimated_output_dim(cls, config: Dict[str, Any]) -> int:
        seq_cols = config.get("sequence_columns") or {}
        k = int(config.get("k", 200))
        return k * max(len(seq_cols), 1)

    # ------------------------------------------------------------------
    # Fit / Generate
    # ------------------------------------------------------------------

    def fit(self, df: Any, **context: Any) -> "LagFeatureExtractor":
        """No-op: this generator has no learned parameters.

        We still validate that required columns exist on the fit dataframe
        and log a length-distribution sanity summary so downstream readers
        can confirm K is appropriate.
        """
        pdf = _to_pandas_safe(df)
        present = set(pdf.columns)
        missing = [c for c in self._seq_cols.values() if c not in present]
        if missing:
            logger.warning(
                "LagFeatureExtractor: source columns missing on fit (will emit "
                "all-pad output for these): %s", missing,
            )

        # Length sanity for the *first* present sequence column
        for short, src in self._seq_cols.items():
            if src not in present:
                continue
            try:
                lens = pdf[src].apply(lambda v: len(_parse_seq(v))).to_numpy()
                if len(lens) > 0:
                    logger.info(
                        "LagFeatureExtractor[%s <- %s] fit-sample lengths: "
                        "n=%d, min=%d, p50=%d, p90=%d, max=%d, K=%d, "
                        "%%cap_hit=%.1f",
                        short, src, len(lens),
                        int(lens.min()), int(np.percentile(lens, 50)),
                        int(np.percentile(lens, 90)), int(lens.max()),
                        self._k, 100.0 * float((lens >= self._k).mean()),
                    )
                break  # one sample is enough
            except Exception as exc:
                logger.debug("length sanity skipped for %s: %s", src, exc)

        self._fitted = True
        return self

    def generate(self, df: Any, **context: Any) -> pd.DataFrame:
        """Return a (n_rows, K * F) pandas DataFrame of lag columns.

        Right-aligned: most-recent observation goes to lag_001 (sequences
        in source columns are stored chronologically, oldest first; we
        reverse so that the most-recent ends up at the leftmost lag slot).
        """
        if not self._fitted:
            raise RuntimeError("LagFeatureExtractor must be fitted before generate().")

        pdf = _to_pandas_safe(df)
        n_rows = len(pdf)
        n_feat = len(self._feat_names)
        K = self._k
        present = set(pdf.columns)
        trunc = self._truncate_last
        pad = self._pad_value

        # (n_rows, K * F) preallocated; column-major iteration per feature
        out = np.full((n_rows, K * n_feat), pad, dtype=np.float32)

        for fi, short in enumerate(self._feat_names):
            src = self._seq_cols[short]
            col_offset = fi * K
            if src not in present:
                continue
            series = pdf[src]
            for ri in range(n_rows):
                seq = _parse_seq(series.iat[ri])
                if trunc > 0:
                    seq = seq[:-trunc] if len(seq) > trunc else []
                if not seq:
                    continue
                # Reverse so most-recent comes first (lag_001).
                rev = seq[::-1]
                take = rev[:K]
                # Fill from offset (lag_001 == col_offset)
                ln = len(take)
                out[ri, col_offset:col_offset + ln] = take

        # Sanity: row count invariant
        if out.shape[0] != n_rows:
            raise RuntimeError(
                f"LagFeatureExtractor row-count mismatch: "
                f"input={n_rows}, output={out.shape[0]}"
            )

        cols = self.output_columns
        result = pd.DataFrame(out, columns=cols, index=pdf.index)
        return result
