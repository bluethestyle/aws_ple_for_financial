"""
Rolling-window stats generator -- DuckDB-native SQL implementation.

On-prem 2026-04-25 redesign axis-2: complements the lag tensor by
preserving total-volume signal that gets truncated when len(seq) > K.
Each window is anchored at the most-recent observation:
    anchor = max(day_offset)
    elapsed[i] = anchor - day_offset[i]
    window includes events with elapsed <= window_days

CLAUDE.md §3.3: SQL-native — runs as a single DuckDB query against the
source table, with ``list_zip`` + ``list_filter`` + ``list_aggregate``
for per-row windowed aggregates. No per-row Python loop, no full pandas
materialisation.

Default windows: [7, 30, 90, 180] days.
Default metrics: [sum, mean, std, count, days_active].
Default output dim = 4 × 5 = 20D.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

import duckdb
import pyarrow as pa

from ..generator import AbstractFeatureGenerator, FeatureGeneratorRegistry

logger = logging.getLogger(__name__)


_VALID_METRICS = ("sum", "mean", "std", "count", "days_active")


@FeatureGeneratorRegistry.register(
    "rolling_stats_extractor",
    description="Rolling-window stats over LIST sequences (DuckDB SQL).",
    tags=["rolling", "temporal", "axis2", "sql_native"],
)
class RollingStatsExtractor(AbstractFeatureGenerator):
    """Per-customer rolling stats over a paired (amount, day_offset) LIST.

    Config keys (all config-driven):
        amount_column      : str   LIST<float>, default "txn_amount_seq"
        day_offset_column  : str   LIST<int>,   default "txn_day_offset_seq"
        windows_days       : List[int]  default [7, 30, 90, 180]
        metrics            : List[str]  subset of {sum, mean, std, count, days_active}
        truncate_seq_last  : int        drop last N elements (label-leak guard).
        prefix             : str        default "txn_roll".
    """

    supports_gpu: bool = False
    required_libraries: List[str] = ["duckdb"]

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
            raise ValueError(
                f"unknown metrics in rolling_stats_extractor: {bad}; "
                f"valid={_VALID_METRICS}"
            )

        self._truncate_last: int = int(cfg.get("truncate_seq_last", 0))
        if self._truncate_last < 0:
            raise ValueError("truncate_seq_last must be >= 0")

    # ------------------------------------------------------------------
    # Capability + output contract
    # ------------------------------------------------------------------

    @property
    def supports_sql_native(self) -> bool:  # CLAUDE.md §3.3
        return True

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
    # Fit / Generate (SQL-native)
    # ------------------------------------------------------------------

    def fit(self, df: Any, **context: Any) -> "RollingStatsExtractor":
        """No learned parameters. Validates source columns when a connection
        is supplied; otherwise defers validation to generate()."""
        con = context.get("duckdb_con")
        source_table = context.get("source_table")
        if con is not None and source_table is not None:
            try:
                cols = {r[0] for r in con.execute(
                    f"SELECT column_name FROM (DESCRIBE SELECT * FROM {source_table})"
                ).fetchall()}
                missing = [c for c in (self._amount_col, self._day_col) if c not in cols]
                if missing:
                    logger.warning(
                        "RollingStatsExtractor: source columns missing on fit "
                        "(rows will emit zeros): %s", missing,
                    )
            except Exception as exc:
                logger.debug("RollingStatsExtractor fit-time check skipped: %s", exc)
        self._fitted = True
        return self

    def _metric_sql(self, amts_expr: str, days_expr: str, metric: str) -> str:
        """SQL fragment computing one metric on a windowed amount list.

        ``amts_expr`` and ``days_expr`` must be DuckDB list expressions
        already filtered to the desired window. Uses ``COALESCE`` to map
        empty-window NULLs (e.g. stddev of <2 elements) to 0.0.
        """
        if metric == "sum":
            return f"COALESCE(list_aggregate({amts_expr}, 'sum'), 0.0)"
        if metric == "mean":
            return f"COALESCE(list_aggregate({amts_expr}, 'avg'), 0.0)"
        if metric == "std":
            # stddev_samp is NULL when len<2; coalesce to 0.0 for stable
            # downstream tensor consumption.
            return (
                f"COALESCE("
                f"CASE WHEN list_count({amts_expr}) > 1 "
                f"THEN list_aggregate({amts_expr}, 'stddev_samp') ELSE 0.0 END, "
                f"0.0)"
            )
        if metric == "count":
            return f"CAST(list_count({amts_expr}) AS DOUBLE)"
        if metric == "days_active":
            return f"CAST(len(list_distinct({days_expr})) AS DOUBLE)"
        raise ValueError(f"unknown metric: {metric}")

    def _build_sql(self, source_table: str) -> str:
        """Build the single SQL query that emits all windowed metrics.

        Pipeline: prep (truncate) -> zipped (anchor + pairs) -> windowed
        lists per window (CTE) -> projection of metric columns.
        """
        trunc = self._truncate_last
        amts_src = (
            f"list_slice({self._amount_col}, 1, "
            f"GREATEST(len({self._amount_col}) - {trunc}, 0))"
            if trunc > 0 else self._amount_col
        )
        days_src = (
            f"list_slice({self._day_col}, 1, "
            f"GREATEST(len({self._day_col}) - {trunc}, 0))"
            if trunc > 0 else self._day_col
        )

        # CTE 1: prep — truncated lists + anchor
        prep_sql = (
            f"prep AS (\n"
            f"  SELECT\n"
            f"    {amts_src} AS amts,\n"
            f"    {days_src} AS days\n"
            f"  FROM {source_table}\n"
            f")"
        )

        # CTE 2: zipped — paired amounts/days + anchor
        zipped_sql = (
            "zipped AS (\n"
            "  SELECT amts, days,\n"
            "    list_aggregate(days, 'max') AS anchor,\n"
            "    list_zip(amts, days) AS pairs\n"
            "  FROM prep\n"
            ")"
        )

        # CTE 3: windowed — one filtered (amts, days) pair per window
        win_terms = []
        for w in self._windows:
            # list_filter keeps pairs where anchor - day <= window
            # list_transform extracts amounts / days from struct
            win_terms.append(
                f"list_transform(list_filter(pairs, p -> "
                f"(anchor - p[2])::DOUBLE <= {w}), p -> p[1]::DOUBLE) "
                f"AS _w{w}d_amts"
            )
            win_terms.append(
                f"list_transform(list_filter(pairs, p -> "
                f"(anchor - p[2])::DOUBLE <= {w}), p -> p[2]::DOUBLE) "
                f"AS _w{w}d_days"
            )
        windowed_sql = (
            "windowed AS (\n"
            "  SELECT\n    "
            + ",\n    ".join(win_terms)
            + "\n  FROM zipped\n"
            ")"
        )

        # Projection
        proj_terms: List[str] = []
        for w in self._windows:
            amts = f"_w{w}d_amts"
            days = f"_w{w}d_days"
            for m in self._metrics:
                col = f"{self.prefix}_w{w}d_{m}"
                proj_terms.append(
                    f'{self._metric_sql(amts, days, m)} AS "{col}"'
                )

        sql = (
            f"WITH {prep_sql},\n{zipped_sql},\n{windowed_sql}\n"
            f"SELECT\n    "
            + ",\n    ".join(proj_terms)
            + "\nFROM windowed"
        )
        return sql

    def generate(self, df: Any, **context: Any) -> pa.Table:
        if not self._fitted:
            raise RuntimeError("RollingStatsExtractor must be fitted before generate().")

        con = context.get("duckdb_con")
        source_table = context.get("source_table")

        owned_con = False
        if con is None or source_table is None:
            owned_con = True
            con = duckdb.connect()
            con.register("_roll_input", df)
            source_table = "_roll_input"

        try:
            sql = self._build_sql(source_table)
            result = con.execute(sql).fetch_arrow_table()
        finally:
            if owned_con:
                try:
                    con.unregister("_roll_input")
                except Exception:
                    pass
                con.close()
        return result
