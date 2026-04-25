"""
Top-N multi-hot generator -- DuckDB-native SQL implementation.

On-prem 2026-04-25 redesign axis-3 helper. Two modes:
  * ``fixed_vocab``: emit one column per item in a known vocabulary
    (e.g. nba_label with 24 product slots). No fit-time learning.
  * ``top_n``: learn the Top-N most frequent values during fit() via
    a DuckDB UNNEST + GROUP BY query, then emit N + 1 columns at
    generate() (e.g. MCC top-30 + others).

CLAUDE.md §3.3: SQL-native — fit() and generate() both run as DuckDB
queries; no per-row Python loop, no pandas materialisation. The adapter
sees ``supports_sql_native=True`` and dispatches accordingly.
"""
from __future__ import annotations

import logging
import re
from typing import Any, Dict, List, Optional

import duckdb
import pyarrow as pa

from ..generator import AbstractFeatureGenerator, FeatureGeneratorRegistry

logger = logging.getLogger(__name__)


_PREFIX_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


@FeatureGeneratorRegistry.register(
    "topn_multihot_extractor",
    description="Top-N or fixed-vocab multi-hot encoding of LIST<int> columns (DuckDB SQL).",
    tags=["multihot", "categorical", "axis3", "sql_native"],
)
class TopNMultiHotExtractor(AbstractFeatureGenerator):
    """Multi-hot encode a LIST<int> column.

    Config keys (config-driven):
        source_column     : str   required; LIST<int> column to encode.
        mode              : str   "fixed_vocab" | "top_n" (default top_n).
        vocab             : List[int]  required when mode="fixed_vocab".
        top_n             : int   for mode="top_n" (default 30).
        include_others    : bool  trailing "others" count column
                                  (default True for top_n, ignored for fixed_vocab).
        truncate_seq_last : int   drop last N elements (label-leak guard).
        prefix            : str   default "multihot".
        binary            : bool  if True (default) emit 0/1; if False
                                  emit occurrence counts.
    """

    supports_gpu: bool = False
    required_libraries: List[str] = ["duckdb"]

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
        if not _PREFIX_RE.match(self.prefix):
            raise ValueError(f"prefix must be a SQL identifier, got {self.prefix!r}")

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

        if self._mode == "fixed_vocab":
            vocab = cfg.get("vocab")
            if not vocab:
                raise ValueError("mode=fixed_vocab requires non-empty `vocab` list")
            self._vocab: List[int] = [int(v) for v in vocab]
            self._has_others: bool = False
        else:
            self._vocab = []  # learned during fit()
            self._has_others = self._include_others

    # ------------------------------------------------------------------
    # Capability + output contract
    # ------------------------------------------------------------------

    @property
    def supports_sql_native(self) -> bool:  # CLAUDE.md §3.3
        return True

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
    # Fit / Generate (SQL-native)
    # ------------------------------------------------------------------

    def _truncated_seq(self) -> str:
        """SQL fragment for the (label-leak-safe) source list."""
        col = self._source_col
        if self._truncate_last > 0:
            return (
                f"list_slice({col}, 1, "
                f"GREATEST(len({col}) - {self._truncate_last}, 0))"
            )
        return col

    def fit(self, df: Any, **context: Any) -> "TopNMultiHotExtractor":
        if self._mode == "fixed_vocab":
            self._fitted = True
            logger.info(
                "TopNMultiHotExtractor[%s]: fixed_vocab mode, %d entries",
                self._source_col, len(self._vocab),
            )
            return self

        # Learn Top-N from the source table via DuckDB UNNEST.
        con = context.get("duckdb_con")
        source_table = context.get("source_table")

        owned_con = False
        if con is None or source_table is None:
            owned_con = True
            con = duckdb.connect()
            con.register("_topn_fit_input", df)
            source_table = "_topn_fit_input"

        try:
            seq_expr = self._truncated_seq()
            # UNNEST + GROUP BY + ORDER BY count DESC LIMIT N
            sql = (
                f"SELECT v, COUNT(*) AS cnt FROM ("
                f"  SELECT UNNEST({seq_expr}) AS v FROM {source_table}"
                f") WHERE v IS NOT NULL GROUP BY v "
                f"ORDER BY cnt DESC LIMIT {self._top_n}"
            )
            rows = con.execute(sql).fetchall()
            self._vocab = [int(r[0]) for r in rows]
            top_count = sum(int(r[1]) for r in rows)
            grand = con.execute(
                f"SELECT COUNT(*) FROM ("
                f"  SELECT UNNEST({seq_expr}) AS v FROM {source_table}"
                f") WHERE v IS NOT NULL"
            ).fetchone()[0]
            cov = (100.0 * top_count / grand) if grand else 0.0
            logger.info(
                "TopNMultiHotExtractor[%s]: learned top-%d vocab "
                "(%d distinct kept), coverage=%.1f%%, others=%s",
                self._source_col, self._top_n, len(self._vocab), cov,
                self._has_others,
            )
        finally:
            if owned_con:
                try:
                    con.unregister("_topn_fit_input")
                except Exception:
                    pass
                con.close()

        self._fitted = True
        return self

    def _build_sql(self, source_table: str) -> str:
        """Build the SQL query producing all multi-hot columns."""
        seq_expr = self._truncated_seq()
        proj_terms: List[str] = []

        n_named = (len(self._vocab) if self._mode == "fixed_vocab" else self._top_n)
        # If learned vocab is shorter than top_n (unlikely), pad missing
        # slots with a value that can never appear in valid input.
        vocab = list(self._vocab)
        while len(vocab) < n_named:
            vocab.append(-1_000_000)  # sentinel; list_contains() will be false

        for i, v in enumerate(vocab):
            if self._mode == "fixed_vocab":
                col = f"{self.prefix}_v{v}"
            else:
                col = f"{self.prefix}_top{i+1:02d}"
            if self._binary:
                proj_terms.append(
                    f"CASE WHEN list_contains({seq_expr}, {v}) "
                    f"THEN 1.0 ELSE 0.0 END AS \"{col}\""
                )
            else:
                # occurrence count = sum over filtered list
                proj_terms.append(
                    f"CAST(len(list_filter({seq_expr}, x -> x = {v})) AS DOUBLE) "
                    f"AS \"{col}\""
                )

        if self._has_others:
            # others = total length minus elements that match the vocab
            vocab_literal = "[" + ", ".join(str(v) for v in vocab) + "]"
            proj_terms.append(
                f"CAST(len(list_filter({seq_expr}, x -> NOT list_contains({vocab_literal}, x))) "
                f"AS DOUBLE) AS \"{self.prefix}_others\""
            )

        return f"SELECT {', '.join(proj_terms)} FROM {source_table}"

    def generate(self, df: Any, **context: Any) -> pa.Table:
        if not self._fitted:
            raise RuntimeError("TopNMultiHotExtractor must be fitted before generate().")

        con = context.get("duckdb_con")
        source_table = context.get("source_table")

        owned_con = False
        if con is None or source_table is None:
            owned_con = True
            con = duckdb.connect()
            con.register("_topn_input", df)
            source_table = "_topn_input"

        try:
            sql = self._build_sql(source_table)
            result = con.execute(sql).fetch_arrow_table()
        finally:
            if owned_con:
                try:
                    con.unregister("_topn_input")
                except Exception:
                    pass
                con.close()
        return result
