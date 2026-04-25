"""
Top-N multi-hot generator -- on-prem 2026-04-25 redesign axis-3 helper.

Flattens product-list / category-list LIST columns into multi-hot vectors:
  * One named-vocabulary mode: emit one column per item in a known fixed
    vocabulary (e.g. nba_label with 24 product slots).
  * Top-N + others mode: learn the Top-N most frequent values during fit()
    and emit N + 1 columns at generate() time (e.g. MCC top-30 + others).

For Santander where nba_label has only 18 unique values and txn_mcc_seq
has 50 cap'd values, this generator gives the LightGCN expert an explicit
membership signal that complements the collaborative-filtering embedding.
"""
from __future__ import annotations

import logging
from collections import Counter
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from ..generator import AbstractFeatureGenerator, FeatureGeneratorRegistry
from .gpu_utils import _to_pandas_safe
from .lag_extractor import _parse_seq

logger = logging.getLogger(__name__)


@FeatureGeneratorRegistry.register(
    "topn_multihot_extractor",
    description="Top-N (or fixed-vocab) multi-hot encoding of LIST<int> columns.",
    tags=["multihot", "categorical", "axis3"],
)
class TopNMultiHotExtractor(AbstractFeatureGenerator):
    """Multi-hot encode a LIST<int> column with either a fixed vocabulary or
    a Top-N + others scheme.

    Config keys (all config-driven):
        source_column     : str   required; LIST<int> column to encode
        mode              : str   "fixed_vocab" | "top_n"        (default: top_n)
        vocab             : List[int]
                            required when mode="fixed_vocab"; one column per
                            vocab entry, in the given order.
        top_n             : int   used when mode="top_n" (default 30)
        include_others    : bool  add a single trailing "others" count column
                            (default True for top_n, ignored for fixed_vocab)
        truncate_seq_last : int   drop last N elements (label-leakage guard).
                            Default 0.
        prefix            : str   default "multihot"
        binary            : bool  if True (default) emit 0/1; if False emit
                            occurrence counts.

    Output columns: f"{prefix}_v{value}" for fixed_vocab, or
                    f"{prefix}_top{rank:02d}" for top_n,
                    plus optional f"{prefix}_others".
    """

    supports_gpu: bool = False
    required_libraries: List[str] = []

    def __init__(
        self,
        config: Optional[Dict[str, Any]] = None,
        prefix: str = "multihot",
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        cfg: Dict[str, Any] = config or {}
        for k in ("source_column", "mode", "vocab", "top_n",
                  "include_others", "truncate_seq_last", "binary"):
            if k in kwargs and k not in cfg:
                cfg[k] = kwargs[k]
        self._cfg = cfg
        self.prefix = prefix

        src = cfg.get("source_column")
        if not src:
            raise ValueError("topn_multihot_extractor requires `source_column`")
        self._source_col: str = str(src)

        self._mode: str = str(cfg.get("mode", "top_n"))
        if self._mode not in ("fixed_vocab", "top_n"):
            raise ValueError(f"unknown mode: {self._mode}")

        self._top_n: int = int(cfg.get("top_n", 30))
        self._include_others: bool = bool(cfg.get("include_others", True))
        self._truncate_last: int = int(cfg.get("truncate_seq_last", 0))
        if self._truncate_last < 0:
            raise ValueError("truncate_seq_last must be >= 0")
        self._binary: bool = bool(cfg.get("binary", True))

        # Vocab: explicit (fixed_vocab) or learned (top_n)
        if self._mode == "fixed_vocab":
            vocab = cfg.get("vocab")
            if not vocab:
                raise ValueError("mode=fixed_vocab requires non-empty `vocab` list")
            self._vocab: List[int] = [int(v) for v in vocab]
            # No "others" slot in fixed_vocab mode
            self._has_others: bool = False
        else:
            self._vocab = []  # learned during fit()
            self._has_others = self._include_others

        # value -> column index
        self._index_map: Dict[int, int] = {}

    # ------------------------------------------------------------------
    # Output contract
    # ------------------------------------------------------------------

    @property
    def output_dim(self) -> int:
        base = len(self._vocab) if self._mode == "fixed_vocab" else self._top_n
        return base + (1 if self._has_others else 0)

    @property
    def output_columns(self) -> List[str]:
        cols: List[str] = []
        if self._mode == "fixed_vocab":
            cols = [f"{self.prefix}_v{v}" for v in self._vocab]
        else:
            cols = [f"{self.prefix}_top{i+1:02d}" for i in range(self._top_n)]
        if self._has_others:
            cols.append(f"{self.prefix}_others")
        return cols

    @classmethod
    def estimated_output_dim(cls, config: Dict[str, Any]) -> int:
        mode = config.get("mode", "top_n")
        include_others = bool(config.get("include_others", True))
        if mode == "fixed_vocab":
            return len(config.get("vocab") or [])
        return int(config.get("top_n", 30)) + (1 if include_others else 0)

    # ------------------------------------------------------------------
    # Fit / Generate
    # ------------------------------------------------------------------

    def fit(self, df: Any, **context: Any) -> "TopNMultiHotExtractor":
        if self._mode == "fixed_vocab":
            self._index_map = {v: i for i, v in enumerate(self._vocab)}
            self._fitted = True
            logger.info("TopNMultiHotExtractor[%s]: fixed_vocab mode, %d entries",
                        self._source_col, len(self._vocab))
            return self

        # Learn Top-N from training fit-sample
        pdf = _to_pandas_safe(df)
        if self._source_col not in pdf.columns:
            logger.warning("TopNMultiHotExtractor[%s]: source column missing on fit; "
                           "using empty vocab", self._source_col)
            self._vocab = []
            self._index_map = {}
            self._fitted = True
            return self

        counter: Counter = Counter()
        for raw in pdf[self._source_col]:
            for v in _parse_seq(raw):
                counter[int(v)] += 1
        most_common = counter.most_common(self._top_n)
        self._vocab = [v for v, _ in most_common]
        # If observed vocab is smaller than top_n, pad index map only;
        # output_columns still emits top_n named slots (missing ones inert).
        self._index_map = {v: i for i, v in enumerate(self._vocab)}
        total = sum(c for _, c in most_common)
        grand = sum(counter.values())
        cov = (100.0 * total / grand) if grand > 0 else 0.0
        logger.info(
            "TopNMultiHotExtractor[%s]: learned top-%d vocab (%d unique), "
            "coverage=%.1f%%, others=%s",
            self._source_col, self._top_n, len(counter), cov, self._has_others,
        )
        self._fitted = True
        return self

    def generate(self, df: Any, **context: Any) -> pd.DataFrame:
        if not self._fitted:
            raise RuntimeError("TopNMultiHotExtractor must be fitted before generate().")

        pdf = _to_pandas_safe(df)
        n_rows = len(pdf)
        out_dim = self.output_dim
        out = np.zeros((n_rows, out_dim), dtype=np.float32)

        if self._source_col not in pdf.columns:
            logger.warning("TopNMultiHotExtractor[%s]: source column absent; "
                           "emitting zero output", self._source_col)
            return pd.DataFrame(out, columns=self.output_columns, index=pdf.index)

        series = pdf[self._source_col]
        idx_map = self._index_map
        n_named = (len(self._vocab) if self._mode == "fixed_vocab" else self._top_n)
        others_idx = n_named if self._has_others else -1
        trunc = self._truncate_last
        binary = self._binary

        for ri in range(n_rows):
            seq = _parse_seq(series.iat[ri])
            if trunc > 0:
                seq = seq[:-trunc] if len(seq) > trunc else []
            if not seq:
                continue
            for v in seq:
                vi = int(v)
                slot = idx_map.get(vi, -1)
                if slot >= 0:
                    if binary:
                        out[ri, slot] = 1.0
                    else:
                        out[ri, slot] += 1.0
                elif others_idx >= 0:
                    out[ri, others_idx] += 1.0  # others is always a count

        if out.shape[0] != n_rows:
            raise RuntimeError(
                f"TopNMultiHotExtractor row-count mismatch: "
                f"input={n_rows}, output={out.shape[0]}"
            )

        return pd.DataFrame(out, columns=self.output_columns, index=pdf.index)
