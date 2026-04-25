"""
Lag tensor feature generator -- DuckDB-native flattening of LIST sequences.

Implements the on-prem 2026-04-25 redesign axis-1 ("Sequence Lag") on AWS.
Materialises K explicit lag columns per input feature with right-aligned
ordering (lag_001 = most recent observation).

CLAUDE.md §3.3 policy: this generator is SQL-native. Given a DuckDB
connection and source table name in ``**context``, ``generate()``
runs entirely as DuckDB SQL using ``list_slice``, ``list_reverse`` and
``list_resize`` — no per-row Python loop, no full-row pandas
materialisation. The adapter's run_generators_duckdb dispatcher detects
``supports_sql_native=True`` and skips the legacy
``con.execute(...).df()`` step for this generator.

Output schema (K=200, features=[amount, mcc, day, hour]):
  txn_lag_amount_001 .. txn_lag_amount_200
  txn_lag_mcc_001    .. txn_lag_mcc_200
  txn_lag_day_001    .. txn_lag_day_200
  txn_lag_hour_001   .. txn_lag_hour_200

Right-alignment rationale (matches list_reverse + list_resize trim):
  * len(seq) <  K  -> trailing zero-pad (lag positions 1..n filled,
                       n+1..K = 0).
  * len(seq) >= K  -> oldest dropped, K most-recent kept.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

import duckdb
import pyarrow as pa

from ..generator import AbstractFeatureGenerator, FeatureGeneratorRegistry

logger = logging.getLogger(__name__)


# Feature short-name -> default integer flag. MCC ids and hour-of-day
# are nominal/cyclical integer columns (zero-padded with int 0); amounts
# and day offsets are continuous (zero-padded with double 0.0).
_INT_FEATURES = {"mcc", "hour"}


@FeatureGeneratorRegistry.register(
    "lag_extractor",
    description="K-step right-aligned lag flattening of LIST sequence columns (DuckDB SQL).",
    tags=["lag", "temporal", "sequence", "axis1", "sql_native"],
)
class LagFeatureExtractor(AbstractFeatureGenerator):
    """Flatten LIST sequences into K explicit lag columns per feature.

    Config keys (config-driven, no hardcoding):
        sequence_columns : Dict[str, str]
            short_name -> source column. Example:
                {"amount": "txn_amount_seq", "mcc": "txn_mcc_seq",
                 "day": "txn_day_offset_seq", "hour": "txn_hour_seq"}
        k : int
            Number of lag positions to emit (default 200).
        prefix : str
            Column-name prefix. Default "txn_lag".
        truncate_seq_last : int
            Drop last N elements before flattening (label-leakage guard).
            Default 0. Use 1 when next_mcc / top_mcc_shift labels are
            derived from the last sequence position.
        pad_value : float
            Fill value for short sequences (continuous features). Default 0.0.
            Integer features ("mcc", "hour") always pad with 0.
    """

    supports_gpu: bool = False
    required_libraries: List[str] = ["duckdb"]

    def __init__(
        self,
        config: Optional[Dict[str, Any]] = None,
        prefix: str = "txn_lag",
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        cfg: Dict[str, Any] = config or {}
        for k in ("sequence_columns", "k", "truncate_seq_last", "pad_value"):
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
        self._seq_cols: Dict[str, str] = {str(k): str(v) for k, v in seq_cols.items()}
        self._feat_names: List[str] = list(self._seq_cols.keys())

        self._k: int = int(cfg.get("k", 200))
        if self._k <= 0:
            raise ValueError(f"k must be positive, got {self._k}")

        self._truncate_last: int = int(cfg.get("truncate_seq_last", 0))
        if self._truncate_last < 0:
            raise ValueError(f"truncate_seq_last must be >= 0, got {self._truncate_last}")

        self._pad_value: float = float(cfg.get("pad_value", 0.0))

    # ------------------------------------------------------------------
    # Capability + output contract
    # ------------------------------------------------------------------

    @property
    def supports_sql_native(self) -> bool:  # CLAUDE.md §3.3
        return True

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
    # Fit / Generate (SQL-native)
    # ------------------------------------------------------------------

    def fit(self, df: Any, **context: Any) -> "LagFeatureExtractor":
        """No learned parameters; validates inputs lazily on the SQL side."""
        # Optional: log a lightweight length sanity using the connection
        # if present. Cheap because it doesn't materialise rows.
        con = context.get("duckdb_con")
        source_table = context.get("source_table")
        if con is not None and source_table is not None:
            for short, src in self._seq_cols.items():
                try:
                    row = con.execute(
                        f"SELECT MIN(len({src})), QUANTILE_CONT(len({src}), 0.5), "
                        f"MAX(len({src})), COUNT(*) FROM {source_table}"
                    ).fetchone()
                    if row and row[3]:
                        cap_hit = con.execute(
                            f"SELECT SUM(CASE WHEN len({src}) >= {self._k} THEN 1 ELSE 0 END) * 100.0 / "
                            f"COUNT(*) FROM {source_table}"
                        ).fetchone()[0] or 0.0
                        logger.info(
                            "LagFeatureExtractor[%s <- %s] length: min=%s, p50=%s, max=%s, "
                            "K=%d, %%cap_hit=%.1f",
                            short, src, row[0], row[1], row[2], self._k, cap_hit,
                        )
                    break
                except Exception as exc:
                    logger.debug("length sanity skipped for %s: %s", src, exc)
        self._fitted = True
        return self

    def _build_sql(self, source_table: str) -> str:
        """Compose the DuckDB SQL that emits all output columns.

        Strategy: a single CTE materialises the trimmed+reversed+resized
        list per feature; the outer SELECT projects K element accesses
        per feature. ``list_resize(list, K, pad)`` zero-pads when shorter
        and truncates oldest when longer, matching right-alignment.
        """
        prep_terms: List[str] = []
        for short in self._feat_names:
            src = self._seq_cols[short]
            pad = "0" if short in _INT_FEATURES else f"{self._pad_value}"
            if self._truncate_last > 0:
                seq_expr = (
                    f"list_slice({src}, 1, "
                    f"GREATEST(len({src}) - {self._truncate_last}, 0))"
                )
            else:
                seq_expr = src
            # `list_reverse` : oldest -> last, newest -> first
            # `list_resize(.., K, pad)` : pad short ones with `pad`,
            #                             truncate oldest from long ones
            prep_terms.append(
                f"list_resize(list_reverse({seq_expr}), {self._k}, {pad}) AS _{short}_lag"
            )
        prep_sql = ",\n    ".join(prep_terms)

        proj_terms: List[str] = []
        for short in self._feat_names:
            for i in range(1, self._k + 1):
                col = f"{self.prefix}_{short}_{i:03d}"
                proj_terms.append(f'_{short}_lag[{i}] AS "{col}"')
        proj_sql = ",\n    ".join(proj_terms)

        sql = (
            f"WITH _prep AS (\n"
            f"  SELECT\n    {prep_sql}\n"
            f"  FROM {source_table}\n"
            f")\n"
            f"SELECT\n    {proj_sql}\n"
            f"FROM _prep"
        )
        return sql

    def generate(self, df: Any, **context: Any) -> pa.Table:
        """Emit the lag tensor as a pyarrow Table, computed in DuckDB.

        Two entry modes:
          1. SQL-native (preferred): ``context`` has ``duckdb_con`` and
             ``source_table`` -- runs against the live source table.
          2. Fallback: pandas/Arrow dataframe ``df`` is registered in a
             scratch DuckDB connection and the same SQL is run there.
             This still avoids the per-row Python loop and the
             ``np.full((n, K*F))`` numpy spike.
        """
        if not self._fitted:
            raise RuntimeError("LagFeatureExtractor must be fitted before generate().")

        con = context.get("duckdb_con")
        source_table = context.get("source_table")

        # Fallback: build a scratch connection if none given
        owned_con = False
        if con is None or source_table is None:
            owned_con = True
            con = duckdb.connect()
            con.register("_lag_input", df)
            source_table = "_lag_input"

        try:
            sql = self._build_sql(source_table)
            result = con.execute(sql).fetch_arrow_table()
        finally:
            if owned_con:
                try:
                    con.unregister("_lag_input")
                except Exception:
                    pass
                con.close()
        return result
