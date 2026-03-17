"""
Sequence Densifier -- DuckDB-based LIST column expansion to flat features.

Converts DuckDB LIST columns (produced by aggregation queries) into flat
columnar features suitable for model input.  Uses DuckDB's vectorized
``list_element()`` function for efficient extraction.

Workflow::

    DuckDB LIST aggregation (e.g., card_sequence_amounts LIST)
      |
    densify() -- DuckDB SQL vectorized LIST_ELEMENT extraction
      |
    Flat columns: txn_card_amount_001, txn_card_amount_002, ..., txn_card_amount_180

Right-alignment convention:
    Recent events are placed at the END of the fixed-length vector.
    If a sequence has fewer elements than ``max_length``, left positions
    are filled with ``fill_value`` (zero-padding from the left).

Usage::

    from core.data.sequence_densifier import SequenceDensifier, SequenceDensifyConfig, SequenceSpec

    config = SequenceDensifyConfig(
        sequences=[
            SequenceSpec(
                list_column="card_sequence_amounts",
                output_prefix="txn_card_amount",
                max_length=180,
            ),
            SequenceSpec(
                list_column="card_sequence_timestamps",
                output_prefix="txn_card_ts",
                max_length=180,
                dtype="BIGINT",
            ),
        ],
        chunk_size=500_000,
    )
    densifier = SequenceDensifier(config)

    # conn is a DuckDB connection with a table already loaded
    result_table = densifier.densify(conn, "raw_sequences")
    df = conn.execute(f"SELECT * FROM {result_table}").fetchdf()
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


# ============================================================================
# Configuration dataclasses
# ============================================================================

@dataclass
class SequenceSpec:
    """Specification for a single LIST column to densify.

    Args:
        list_column: Source column name containing a DuckDB LIST
            (e.g., ``"card_sequence_amounts"``).
        output_prefix: Prefix for the generated flat columns
            (e.g., ``"txn_card_amount"`` produces ``txn_card_amount_001``).
        max_length: Fixed output sequence length.  Sequences shorter than
            this are left-padded; sequences longer are right-truncated
            (keeping the most recent events).
        dtype: DuckDB type for the output columns (``"FLOAT"``,
            ``"DOUBLE"``, ``"BIGINT"``, etc.).
        fill_value: Padding value for positions that have no data
            (left side when sequence is shorter than ``max_length``).
    """
    list_column: str
    output_prefix: str
    max_length: int = 180
    dtype: str = "FLOAT"
    fill_value: float = 0.0


@dataclass
class SequenceDensifyConfig:
    """Configuration for the SequenceDensifier.

    Args:
        sequences: List of ``SequenceSpec`` objects, one per LIST column
            to densify.
        chunk_size: Number of rows to process per chunk.  Controls memory
            usage -- larger values are faster but use more RAM.
        derive_time_delta: Generate inter-event time deltas from timestamp
            sequences.  Requires a companion timestamp LIST column named
            ``{output_prefix}_ts`` or a matching SequenceSpec with
            ``dtype="BIGINT"`` and the same ``max_length``.
        derive_hour: Extract hour-of-day from timestamp columns.
        derive_day_of_week: Extract day-of-week from timestamp columns.
        derive_is_weekend: Generate weekend flag (Sat/Sun = 1) from
            timestamp columns.
        derive_cumulative_count: Generate cumulative event position
            (1-indexed count from the first event in the sequence).
    """
    sequences: List[SequenceSpec] = field(default_factory=list)
    chunk_size: int = 500_000

    # Derived temporal features to generate per sequence
    derive_time_delta: bool = True
    derive_hour: bool = True
    derive_day_of_week: bool = True
    derive_is_weekend: bool = True
    derive_cumulative_count: bool = True


# ============================================================================
# SequenceDensifier
# ============================================================================

class SequenceDensifier:
    """Convert DuckDB LIST columns into flat feature columns for model input.

    Uses DuckDB's ``list_element()`` for vectorized extraction and SQL-level
    chunked processing to prevent OOM on large datasets.

    The densification is right-aligned: recent events sit at the end of the
    output vector, with left-padding for sequences shorter than
    ``max_length``.

    Args:
        config: ``SequenceDensifyConfig`` with sequence specifications.

    Example::

        densifier = SequenceDensifier(config)
        result_table = densifier.densify(conn, "my_table")
        # result_table is a DuckDB table name with all flat columns
    """

    def __init__(self, config: SequenceDensifyConfig) -> None:
        self._config = config
        self._output_columns: List[str] = []
        self._column_specs: Dict[str, List[str]] = {}

        # Pre-compute output column names
        for spec in config.sequences:
            cols = self._get_output_columns(spec)
            self._column_specs[spec.list_column] = cols
            self._output_columns.extend(cols)

        logger.info(
            "SequenceDensifier: %d sequences, %d total output columns, "
            "chunk_size=%d",
            len(config.sequences),
            len(self._output_columns),
            config.chunk_size,
        )

    @property
    def output_columns(self) -> List[str]:
        """All output column names that will be generated."""
        return list(self._output_columns)

    @property
    def output_dim(self) -> int:
        """Total number of flat columns produced."""
        return len(self._output_columns)

    def densify(self, conn: Any, table_name: str) -> str:
        """Run densification SQL and return the name of the result table.

        Uses DuckDB's ``list_element()`` for vectorized extraction.
        Right-aligned: recent events at the end, padding at the start.

        Processes rows in chunks of ``config.chunk_size`` to prevent OOM.
        All chunks are combined via ``UNION ALL`` into a single result
        table.

        Args:
            conn: DuckDB connection (``duckdb.DuckDBPyConnection``).
            table_name: Name of the source table containing LIST columns.

        Returns:
            Name of the result table (created in-memory) containing all
            flat columns plus any non-LIST columns from the source.
        """
        result_table = f"_densified_{table_name}"

        # Get total row count
        row_count = conn.execute(
            f"SELECT COUNT(*) FROM {table_name}"
        ).fetchone()[0]

        chunk_size = self._config.chunk_size

        if row_count == 0:
            logger.warning("Source table '%s' is empty", table_name)
            # Create empty result table with correct schema
            create_sql = self._build_empty_table_sql(result_table, conn, table_name)
            conn.execute(create_sql)
            return result_table

        # Identify non-LIST columns to pass through
        passthrough_cols = self._get_passthrough_columns(conn, table_name)

        # Build the densification SQL for all sequences
        densify_exprs = []
        for spec in self._config.sequences:
            densify_exprs.extend(self._build_densify_expressions(spec))

        # Add derived temporal features
        derived_exprs = self._build_derived_expressions()
        densify_exprs.extend(derived_exprs)

        passthrough_expr = ", ".join(f'"{c}"' for c in passthrough_cols)
        all_exprs = densify_exprs
        select_clause = passthrough_expr
        if all_exprs:
            select_clause += ",\n    " + ",\n    ".join(all_exprs)

        # Process in chunks using ROW_NUMBER()
        num_chunks = max(1, (row_count + chunk_size - 1) // chunk_size)

        if num_chunks == 1:
            # Single chunk -- no need for ROW_NUMBER overhead
            full_sql = f"""
                CREATE OR REPLACE TABLE {result_table} AS
                SELECT
                    {select_clause}
                FROM {table_name}
            """
            conn.execute(full_sql)
            logger.info(
                "Densified %d rows from '%s' -> '%s' (%d columns, single chunk)",
                row_count, table_name, result_table, len(self._output_columns),
            )
        else:
            # Multi-chunk processing with ROW_NUMBER
            # First, add a row number column
            numbered_table = f"_numbered_{table_name}"
            conn.execute(f"""
                CREATE OR REPLACE TABLE {numbered_table} AS
                SELECT *, ROW_NUMBER() OVER () AS _densify_rownum
                FROM {table_name}
            """)

            for chunk_idx in range(num_chunks):
                offset = chunk_idx * chunk_size
                limit_val = min(chunk_size, row_count - offset)

                chunk_sql = f"""
                    SELECT
                        {select_clause}
                    FROM {numbered_table}
                    WHERE _densify_rownum > {offset}
                      AND _densify_rownum <= {offset + limit_val}
                """

                if chunk_idx == 0:
                    conn.execute(
                        f"CREATE OR REPLACE TABLE {result_table} AS {chunk_sql}"
                    )
                else:
                    conn.execute(
                        f"INSERT INTO {result_table} {chunk_sql}"
                    )

                logger.debug(
                    "Densified chunk %d/%d (%d rows)",
                    chunk_idx + 1, num_chunks, limit_val,
                )

            # Clean up numbered table
            conn.execute(f"DROP TABLE IF EXISTS {numbered_table}")

            logger.info(
                "Densified %d rows from '%s' -> '%s' "
                "(%d columns, %d chunks of %d)",
                row_count, table_name, result_table,
                len(self._output_columns), num_chunks, chunk_size,
            )

        return result_table

    # ------------------------------------------------------------------
    # SQL generation
    # ------------------------------------------------------------------

    def _build_densify_expressions(self, spec: SequenceSpec) -> List[str]:
        """Generate SQL expressions for one sequence type.

        Produces ``max_length`` expressions that extract elements from the
        LIST column using ``list_element()``.  Uses right-alignment: the
        last LIST element maps to position ``max_length``, and earlier
        positions are padded with ``fill_value`` when the list is shorter.

        Args:
            spec: Sequence specification.

        Returns:
            List of SQL expression strings, one per output column.
        """
        exprs: List[str] = []
        col = spec.list_column
        prefix = spec.output_prefix
        max_len = spec.max_length
        fill = spec.fill_value
        dtype = spec.dtype

        for i in range(1, max_len + 1):
            # Right-aligned index: position i maps to
            # list_element(col, len(col) - max_length + i)
            # list_element is 1-indexed in DuckDB
            # When the computed index is <= 0, the element doesn't exist
            # and we use COALESCE to fill with the padding value.
            col_name = f"{prefix}_{i:03d}"
            expr = (
                f"COALESCE("
                f"CAST(list_element(\"{col}\", "
                f"len(\"{col}\") - {max_len} + {i}) AS {dtype}), "
                f"{fill}"
                f") AS \"{col_name}\""
            )
            exprs.append(expr)

        return exprs

    def _build_derived_expressions(self) -> List[str]:
        """Generate SQL expressions for derived temporal features.

        Derived features are computed from the densified columns using
        window-style SQL operations.  These include:

        - **time_delta**: difference between consecutive timestamps
        - **hour**: hour of day extracted from epoch timestamps
        - **day_of_week**: day of week (0=Monday, 6=Sunday)
        - **is_weekend**: 1 if Saturday or Sunday, 0 otherwise
        - **cumulative_count**: 1-indexed position from the start

        Returns:
            List of SQL expression strings for all derived features.
        """
        exprs: List[str] = []
        cfg = self._config

        for spec in cfg.sequences:
            prefix = spec.output_prefix
            max_len = spec.max_length
            col = spec.list_column

            # Cumulative count is always derivable (position index)
            if cfg.derive_cumulative_count:
                for i in range(1, max_len + 1):
                    col_name = f"{prefix}_cumcount_{i:03d}"
                    # Position is valid only if the base column is non-fill
                    base_col = f"{prefix}_{i:03d}"
                    expr = (
                        f"CASE WHEN \"{base_col}\" != {spec.fill_value} "
                        f"THEN {i} ELSE 0 END AS \"{col_name}\""
                    )
                    exprs.append(expr)
                self._output_columns.extend(
                    f"{prefix}_cumcount_{i:03d}" for i in range(1, max_len + 1)
                )

            # Timestamp-based derivations require the dtype to be a
            # timestamp-like type (BIGINT epoch or TIMESTAMP).
            # We check by naming convention: specs with "timestamp" or "ts"
            # in the list_column or output_prefix are treated as temporal.
            is_temporal = any(
                tok in col.lower() or tok in prefix.lower()
                for tok in ("timestamp", "_ts", "time", "epoch")
            )

            if not is_temporal:
                continue

            if cfg.derive_time_delta:
                for i in range(1, max_len + 1):
                    col_name = f"{prefix}_tdelta_{i:03d}"
                    cur_col = f"{prefix}_{i:03d}"
                    if i == 1:
                        expr = f"CAST(0 AS {spec.dtype}) AS \"{col_name}\""
                    else:
                        prev_col = f"{prefix}_{i - 1:03d}"
                        expr = (
                            f"CASE WHEN \"{cur_col}\" != {spec.fill_value} "
                            f"AND \"{prev_col}\" != {spec.fill_value} "
                            f"THEN CAST(\"{cur_col}\" - \"{prev_col}\" AS {spec.dtype}) "
                            f"ELSE CAST(0 AS {spec.dtype}) END AS \"{col_name}\""
                        )
                    exprs.append(expr)
                self._output_columns.extend(
                    f"{prefix}_tdelta_{i:03d}" for i in range(1, max_len + 1)
                )

            if cfg.derive_hour:
                for i in range(1, max_len + 1):
                    col_name = f"{prefix}_hour_{i:03d}"
                    src_col = f"{prefix}_{i:03d}"
                    # Assume epoch seconds; extract hour
                    expr = (
                        f"CASE WHEN \"{src_col}\" != {spec.fill_value} "
                        f"THEN CAST(EXTRACT(HOUR FROM "
                        f"EPOCH_MS(CAST(\"{src_col}\" AS BIGINT) * 1000)) AS INTEGER) "
                        f"ELSE 0 END AS \"{col_name}\""
                    )
                    exprs.append(expr)
                self._output_columns.extend(
                    f"{prefix}_hour_{i:03d}" for i in range(1, max_len + 1)
                )

            if cfg.derive_day_of_week:
                for i in range(1, max_len + 1):
                    col_name = f"{prefix}_dow_{i:03d}"
                    src_col = f"{prefix}_{i:03d}"
                    expr = (
                        f"CASE WHEN \"{src_col}\" != {spec.fill_value} "
                        f"THEN CAST(EXTRACT(DOW FROM "
                        f"EPOCH_MS(CAST(\"{src_col}\" AS BIGINT) * 1000)) AS INTEGER) "
                        f"ELSE 0 END AS \"{col_name}\""
                    )
                    exprs.append(expr)
                self._output_columns.extend(
                    f"{prefix}_dow_{i:03d}" for i in range(1, max_len + 1)
                )

            if cfg.derive_is_weekend:
                for i in range(1, max_len + 1):
                    col_name = f"{prefix}_weekend_{i:03d}"
                    src_col = f"{prefix}_{i:03d}"
                    expr = (
                        f"CASE WHEN \"{src_col}\" != {spec.fill_value} "
                        f"AND EXTRACT(DOW FROM "
                        f"EPOCH_MS(CAST(\"{src_col}\" AS BIGINT) * 1000)) IN (0, 6) "
                        f"THEN 1 ELSE 0 END AS \"{col_name}\""
                    )
                    exprs.append(expr)
                self._output_columns.extend(
                    f"{prefix}_weekend_{i:03d}" for i in range(1, max_len + 1)
                )

        return exprs

    def _build_densify_sql(self, sequence_spec: SequenceSpec) -> str:
        """Generate a complete SQL SELECT for one sequence type.

        This is a convenience method that wraps ``_build_densify_expressions``
        into a full SQL statement.

        Args:
            sequence_spec: Sequence specification.

        Returns:
            SQL string with SELECT expressions for all positions.

        Example output::

            SELECT
              COALESCE(CAST(list_element("card_sequence_amounts",
                len("card_sequence_amounts") - 180 + 1) AS FLOAT), 0.0)
                AS "txn_card_amount_001",
              ...
              COALESCE(CAST(list_element("card_sequence_amounts",
                len("card_sequence_amounts") - 180 + 180) AS FLOAT), 0.0)
                AS "txn_card_amount_180"
            FROM source_table
        """
        exprs = self._build_densify_expressions(sequence_spec)
        return "SELECT\n    " + ",\n    ".join(exprs)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _get_output_columns(spec: SequenceSpec) -> List[str]:
        """Return the list of output column names for a sequence spec."""
        return [f"{spec.output_prefix}_{i:03d}" for i in range(1, spec.max_length + 1)]

    def _get_passthrough_columns(
        self, conn: Any, table_name: str,
    ) -> List[str]:
        """Identify non-LIST columns from the source table.

        These columns are passed through unchanged to the result table.

        Args:
            conn: DuckDB connection.
            table_name: Source table name.

        Returns:
            List of column names that are not LIST type.
        """
        list_columns = {spec.list_column for spec in self._config.sequences}
        schema = conn.execute(f"DESCRIBE {table_name}").fetchall()

        passthrough = []
        for col_name, col_type, *_ in schema:
            if col_name in list_columns:
                continue
            # Skip any column with LIST type
            if "[]" in col_type or col_type.upper().startswith("LIST"):
                continue
            passthrough.append(col_name)

        return passthrough

    def _build_empty_table_sql(
        self, result_table: str, conn: Any, source_table: str,
    ) -> str:
        """Build a CREATE TABLE statement for an empty result.

        Args:
            result_table: Name for the new table.
            conn: DuckDB connection.
            source_table: Source table (for passthrough column schema).

        Returns:
            SQL CREATE TABLE statement.
        """
        passthrough_cols = self._get_passthrough_columns(conn, source_table)
        passthrough_expr = ", ".join(f'"{c}"' for c in passthrough_cols)

        densify_exprs = []
        for spec in self._config.sequences:
            densify_exprs.extend(self._build_densify_expressions(spec))

        select_clause = passthrough_expr
        if densify_exprs:
            select_clause += ",\n    " + ",\n    ".join(densify_exprs)

        return f"""
            CREATE OR REPLACE TABLE {result_table} AS
            SELECT {select_clause}
            FROM {source_table}
            WHERE 1=0
        """

    def get_sequence_column_ranges(self) -> Dict[str, Tuple[int, int]]:
        """Return per-sequence output column index ranges.

        Useful for wiring to ``FeatureGroupPipeline`` group ranges.

        Returns:
            ``{output_prefix: (start_idx, end_idx)}`` mapping.
        """
        ranges: Dict[str, Tuple[int, int]] = {}
        offset = 0
        for spec in self._config.sequences:
            n_cols = spec.max_length
            ranges[spec.output_prefix] = (offset, offset + n_cols)
            offset += n_cols
        return ranges

    def __repr__(self) -> str:
        lines = [f"SequenceDensifier({len(self._config.sequences)} sequences):"]
        for spec in self._config.sequences:
            lines.append(
                f"  {spec.list_column} -> {spec.output_prefix}_XXX "
                f"(max_length={spec.max_length}, dtype={spec.dtype}, "
                f"fill={spec.fill_value})"
            )
        lines.append(f"  Total output columns: {self.output_dim}")
        lines.append(f"  Chunk size: {self._config.chunk_size}")
        return "\n".join(lines)
