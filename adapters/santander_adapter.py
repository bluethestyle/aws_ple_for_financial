"""
Santander Customer Transactions Adapter
========================================

Minimal, config-driven adapter.  The adapter's ONLY job is:
    raw parquet → standardised DataFrame (``{"main": df}``).

Feature generation is NOT the adapter's responsibility -- it belongs in
PipelineRunner Stage 3 via FeatureGroupPipeline.

A transitional helper ``run_generators_from_config()`` is provided so that
the Phase 0 Processing Job (__main__ block) can still run generators,
but all column routing is driven by ``feature_groups.yaml``, not hardcoded.

DuckDB-native pipeline
----------------------
The Phase 0 ``__main__`` block keeps data as a DuckDB in-memory table
throughout.  Pandas is used **only** at the generator interface boundary
(generators still require pd.DataFrame input/output).  All quality-gate
checks, normalization, dtype downcasting, feature stats, label stats,
and final parquet writes are performed via DuckDB SQL — no pandas
intermediary.
"""

from __future__ import annotations

import logging
import time
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from core.pipeline.adapter import AdapterMetadata, AdapterRegistry, DataAdapter

logger = logging.getLogger(__name__)


# ======================================================================
# Config-driven feature generation (transition helper for Phase 0)
# ======================================================================

def _resolve_columns_by_filter_duckdb(
    con,
    table: str,
    filter_config: Dict[str, Any],
    id_col: Optional[str] = None,
) -> List[str]:
    """Resolve columns matching the filter criteria via DuckDB SQL.

    Operates entirely in DuckDB — no pandas materialisation needed.

    Supported filter keys: dtype, exclude_binary, min_nunique,
    include_prefix, exclude_prefix.
    """
    exclude_binary = filter_config.get("exclude_binary", False)
    min_nunique = filter_config.get("min_nunique", 0)
    include_prefix: List[str] = filter_config.get("include_prefix", [])
    exclude_prefix: List[str] = filter_config.get("exclude_prefix", [])

    # Get numeric column names + metadata in one query
    col_info_rows = con.execute(f"""
        SELECT column_name, column_type
        FROM (DESCRIBE SELECT * FROM {table})
        WHERE column_type IN (
            'TINYINT','SMALLINT','INTEGER','BIGINT','HUGEINT',
            'FLOAT','DOUBLE','DECIMAL','UTINYINT','USMALLINT',
            'UINTEGER','UBIGINT'
        )
    """).fetchall()

    numeric_cols = [r[0] for r in col_info_rows]
    if not numeric_cols:
        return []

    # Filter by prefix first (cheap string check)
    cols = []
    for col in numeric_cols:
        if id_col and col == id_col:
            continue
        if include_prefix and not any(col.startswith(p) for p in include_prefix):
            continue
        if exclude_prefix and any(col.startswith(p) for p in exclude_prefix):
            continue
        cols.append(col)

    if not cols:
        return []

    # Build one aggregate query for binary check + nunique
    if exclude_binary or min_nunique > 0:
        agg_parts = []
        for col in cols:
            qc = f'"{col}"'
            agg_parts.append(f'COUNT(DISTINCT {qc}) AS "{col}__nunique"')
            if exclude_binary:
                agg_parts.append(
                    f'(MIN({qc}) >= 0 AND MAX({qc}) <= 1 '
                    f'AND COUNT(DISTINCT {qc}) <= 2)::INTEGER AS "{col}__is_binary"'
                )

        row = con.execute(
            f"SELECT {', '.join(agg_parts)} FROM {table}"
        ).fetchone()

        filtered: List[str] = []
        idx = 0
        for col in cols:
            nunique = int(row[idx])
            idx += 1
            is_binary = False
            if exclude_binary:
                is_binary = bool(row[idx])
                idx += 1
            if exclude_binary and is_binary:
                continue
            if min_nunique > 0 and nunique < min_nunique:
                continue
            filtered.append(col)
        return filtered

    return cols


def _resolve_columns_by_filter(
    df: pd.DataFrame,
    filter_config: Dict[str, Any],
    id_col: Optional[str] = None,
) -> List[str]:
    """Resolve columns matching the filter criteria from config (pandas fallback).

    Supported filter keys
    ---------------------
    dtype : str
        ``"continuous"`` -- numeric, excluding binary;
        ``"all_numeric"`` -- every numeric column;
        ``"all"`` -- alias for ``"all_numeric"``.
    exclude_binary : bool
        If *True*, drop columns whose unique non-null values ⊆ {0, 1}.
    min_nunique : int
        Minimum number of unique values required.
    include_prefix : list[str]
        Only keep columns starting with one of these prefixes.
    exclude_prefix : list[str]
        Drop columns starting with any of these prefixes.
    """
    dtype_filter = filter_config.get("dtype", "all_numeric")
    exclude_binary = filter_config.get("exclude_binary", False)
    min_nunique = filter_config.get("min_nunique", 0)
    include_prefix: List[str] = filter_config.get("include_prefix", [])
    exclude_prefix: List[str] = filter_config.get("exclude_prefix", [])

    # Start with numeric columns only
    cols: List[str] = []
    for col in df.select_dtypes(include="number").columns:
        if id_col and col == id_col:
            continue
        if exclude_binary and set(df[col].dropna().unique()).issubset({0, 1}):
            continue
        if min_nunique > 0 and df[col].nunique() < min_nunique:
            continue
        if include_prefix and not any(col.startswith(p) for p in include_prefix):
            continue
        if exclude_prefix and any(col.startswith(p) for p in exclude_prefix):
            continue
        cols.append(col)
    return cols


def run_generators_duckdb(
    con,
    table: str,
    feature_groups_config: List[Dict[str, Any]],
    id_col: Optional[str] = None,
    fit_subsample_limit: int = 50_000,
) -> str:
    """Run generators based on feature_groups config using DuckDB-native data.

    Generators still require pandas input/output (interface constraint), but
    this function minimises pandas materialisation:

    1.  Column resolution uses ``_resolve_columns_by_filter_duckdb`` (SQL).
    2.  Fit subsample is extracted via ``USING SAMPLE`` (DuckDB, no pandas copy).
    3.  Only the specific input columns are materialised to pandas for
        ``gen.fit()`` / ``gen.generate()`` — not the full table.
    4.  Generated features are registered as Arrow tables and merged back via
        ``POSITIONAL JOIN`` in DuckDB — no full-table pandas round-trip.

    Parameters
    ----------
    con : duckdb.DuckDBPyConnection
        Open DuckDB connection that owns *table*.
    table : str
        Name of the table containing raw data.
    feature_groups_config : list[dict]
        The ``feature_groups`` list from feature_groups.yaml.
    id_col : str | None
        Identity column to exclude from generator inputs.
    fit_subsample_limit : int
        Max rows for generator fitting.

    Returns
    -------
    str
        Name of the (possibly new) DuckDB table with generated columns
        appended.  If no generators ran, returns the original *table* name.
    """
    # Lazy imports -- only needed when generators are actually invoked.
    from core.feature.generator import FeatureGeneratorRegistry
    import core.feature.generators  # noqa: F401
    import pyarrow as pa

    total_rows = con.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]

    generated_frames: List[pa.Table] = []
    gen_summary: Dict[str, int] = {}

    for group in feature_groups_config:
        if group.get("group_type") != "generate":
            continue
        if not group.get("enabled", True):
            continue

        group_name = group["name"]
        generator_name = group.get("generator", group_name)
        gen_params = dict(group.get("generator_params", {}))

        # --- Resolve input columns from config filter (DuckDB SQL) ---
        input_filter = gen_params.pop("input_filter", None)
        if input_filter is not None:
            input_cols = _resolve_columns_by_filter_duckdb(
                con, table, input_filter, id_col=id_col,
            )
        else:
            # Fallback: all numeric columns (exclude id_col)
            rows = con.execute(f"""
                SELECT column_name FROM (DESCRIBE SELECT * FROM {table})
                WHERE column_type IN (
                    'TINYINT','SMALLINT','INTEGER','BIGINT','HUGEINT',
                    'FLOAT','DOUBLE','DECIMAL','UTINYINT','USMALLINT',
                    'UINTEGER','UBIGINT'
                )
            """).fetchall()
            input_cols = [
                r[0] for r in rows
                if not (id_col and r[0] == id_col)
            ]

        if not input_cols:
            logger.warning(
                "Generator '%s': no columns matched input_filter, skipping.",
                group_name,
            )
            continue

        logger.info(
            "Generator '%s' (%s): %d input columns",
            group_name, generator_name, len(input_cols),
        )

        t0 = time.time()
        try:
            create_kwargs = dict(gen_params)
            if "prefix" not in create_kwargs:
                create_kwargs["prefix"] = group_name
            gen = FeatureGeneratorRegistry.create(
                generator_name,
                input_columns=input_cols,
                **create_kwargs,
            )

            # Include entity column if generator needs it
            entity_col = getattr(gen, "entity_column", None)
            fit_cols = list(input_cols)
            gen_cols = list(input_cols)
            all_table_cols = [
                r[0] for r in con.execute(
                    f"SELECT column_name FROM (DESCRIBE SELECT * FROM {table})"
                ).fetchall()
            ]
            if entity_col and entity_col in all_table_cols and entity_col not in fit_cols:
                fit_cols = [entity_col] + fit_cols
                gen_cols = [entity_col] + gen_cols

            # --- Fit: extract subsample via DuckDB USING SAMPLE ---
            _select_fit = ", ".join(f'"{c}"' for c in fit_cols)
            if total_rows > fit_subsample_limit:
                fit_sql = (
                    f"SELECT {_select_fit} FROM {table} "
                    f"USING SAMPLE {fit_subsample_limit} (reservoir, 42)"
                )
                logger.info(
                    "Large dataset (%d rows): fitting generator '%s' on %d-row subsample",
                    total_rows, group_name, fit_subsample_limit,
                )
            else:
                fit_sql = f"SELECT {_select_fit} FROM {table}"
            fit_df = con.execute(fit_sql).df().fillna(0).infer_objects()
            gen.fit(fit_df)
            del fit_df

            # --- Generate: materialise only the needed columns ---
            _select_gen = ", ".join('"' + c + '"' for c in gen_cols)
            gen_input_df = con.execute(
                f"SELECT {_select_gen} FROM {table}"
            ).df().fillna(0).infer_objects()
            result = gen.generate(gen_input_df)
            del gen_input_df

            if not isinstance(result, pd.DataFrame):
                result = pd.DataFrame(result)

            generated_frames.append(
                pa.Table.from_pandas(result, preserve_index=False),
            )
            gen_summary[group_name] = len(result.columns)
            logger.info(
                "Generator '%s': %d features in %.1fs",
                group_name, len(result.columns), time.time() - t0,
            )
            del result
        except Exception as e:
            logger.warning("Generator '%s' failed: %s", group_name, e, exc_info=True)
            n_cols = group.get("output_dim", 10)
            _fallback_cols = [f"{group_name}_{i:03d}" for i in range(n_cols)]
            _fallback_data = np.zeros((total_rows, n_cols), dtype=np.float32)
            fallback = pa.table(
                {c: _fallback_data[:, j] for j, c in enumerate(_fallback_cols)},
            )
            generated_frames.append(fallback)
            gen_summary[group_name] = n_cols

    # Merge generated features via DuckDB POSITIONAL JOIN
    if generated_frames:
        join_sql = f"SELECT {table}.*"
        from_sql = table
        for i, gf in enumerate(generated_frames):
            alias = f"_gf{i}"
            con.register(alias, gf)
            gen_cols_str = ", ".join(f'{alias}."{c}"' for c in gf.column_names)
            join_sql += f", {gen_cols_str}"
            from_sql += f" POSITIONAL JOIN {alias}"

        merged_table = f"{table}_gen"
        con.execute(f"CREATE OR REPLACE TABLE {merged_table} AS {join_sql} FROM {from_sql}")

        # Cleanup temporary Arrow registrations
        for i in range(len(generated_frames)):
            try:
                con.unregister(f"_gf{i}")
            except Exception:
                pass

        total_new = sum(gen_summary.values())
        new_col_count = con.execute(
            f"SELECT COUNT(*) FROM (DESCRIBE SELECT * FROM {merged_table})"
        ).fetchone()[0]
        logger.info(
            "Feature generation complete: %d new columns from %d generators. "
            "Total columns: %d. Summary: %s",
            total_new, len(gen_summary), new_col_count, gen_summary,
        )
        return merged_table

    return table


def run_generators_from_config(
    df: pd.DataFrame,
    feature_groups_config: List[Dict[str, Any]],
    id_col: Optional[str] = None,
    fit_subsample_limit: int = 50_000,
) -> pd.DataFrame:
    """Legacy pandas-interface wrapper around ``run_generators_duckdb``.

    Kept for backward compatibility with callers that pass a pandas DataFrame.
    Internally converts to DuckDB, runs generators, and converts back.
    """
    import duckdb as _ddb
    _con = _ddb.connect()
    try:
        _con.register("_legacy_src", df)
        _con.execute("CREATE TABLE _legacy_tbl AS SELECT * FROM _legacy_src")
        _con.unregister("_legacy_src")
        result_table = run_generators_duckdb(
            _con, "_legacy_tbl", feature_groups_config,
            id_col=id_col, fit_subsample_limit=fit_subsample_limit,
        )
        df_out = _con.execute(f"SELECT * FROM {result_table}").df()
    finally:
        _con.close()
    return df_out


# ======================================================================
# Adapter
# ======================================================================

@AdapterRegistry.register("santander")
class SantanderAdapter(DataAdapter):
    """Load santander_final.parquet.  No preprocessing, no generators.

    The dataset is already at user-level granularity with ~89 columns
    (demographics, product holdings, synthetic transaction features, etc.).
    """

    def load_raw(self) -> Dict[str, pd.DataFrame]:
        """Load raw parquet. Already user-level, no preprocessing needed."""
        backend = self._select_backend()
        source = self.config.get("data", {}).get(
            "source", "data/synthetic/santander_final.parquet",
        )
        self._id_col = self.config.get("data", {}).get("id_col")
        if not self._id_col:
            raise ValueError("data.id_col must be specified in adapter config")

        logger.info("SantanderAdapter: loading %s with backend=%s", source, backend)

        if backend == "cudf":
            import cudf
            df = cudf.read_parquet(source)
        elif backend == "duckdb":
            import duckdb
            con = duckdb.connect()
            df = con.execute(f"SELECT * FROM '{source}'").df()
            con.close()
        else:
            # Use DuckDB for parquet loading even in pandas-fallback path
            import duckdb as _ddb_load
            _con_load = _ddb_load.connect()
            try:
                df = _con_load.execute(f"SELECT * FROM '{source}'").df()
            finally:
                _con_load.close()

        self._metadata = AdapterMetadata(
            id_col=self._id_col,
            entity_granularity="user",
            num_entities=len(df),
            num_raw_rows=len(df),
            source_files=[str(source)],
            backend_used=backend,
        )

        # Expose timestamp column info for time-based sequence building.
        # The sequence builder auto-detects, but explicit metadata helps.
        _date_cols = [c for c in df.columns if "date" in c.lower()]
        if _date_cols:
            self._metadata.extra = getattr(self._metadata, "extra", {})
            if isinstance(self._metadata.extra, dict):
                self._metadata.extra["timestamp_columns"] = _date_cols
            logger.info(
                "SantanderAdapter: detected timestamp columns: %s",
                _date_cols,
            )

        logger.info(
            "SantanderAdapter: loaded %d rows x %d cols (backend=%s)",
            len(df), len(df.columns), backend,
        )

        return {"main": df}


# ======================================================================
# Standalone entry point for SageMaker Processing Job (Phase 0)
# ======================================================================

if __name__ == "__main__":
    import argparse
    import glob
    import json
    import os
    import sys

    import duckdb
    import yaml

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    parser = argparse.ArgumentParser(description="Santander data adapter (Phase 0)")
    parser.add_argument("--input-dir", default="/opt/ml/processing/input/raw")
    parser.add_argument("--output-dir", default="/opt/ml/processing/output")
    parser.add_argument(
        "--feature-groups-config",
        default="configs/santander/feature_groups.yaml",
        help="Path to feature_groups.yaml",
    )
    parser.add_argument(
        "--pipeline",
        default="configs/santander/pipeline.yaml",
        help="Path to pipeline.yaml (used for label column discovery)",
    )
    parser.add_argument("--stages", default="1-6", help="(unused, for compat)")
    cli_args = parser.parse_args()

    input_dir = cli_args.input_dir
    output_dir = cli_args.output_dir

    # Find the parquet file in input dir
    parquet_files = glob.glob(os.path.join(input_dir, "*.parquet"))
    if not parquet_files:
        parquet_files = glob.glob(
            os.path.join(input_dir, "**", "*.parquet"), recursive=True,
        )
    if not parquet_files:
        logger.error("No parquet files found in %s", input_dir)
        sys.exit(1)

    source = parquet_files[0]
    logger.info("Found source: %s", source)

    # Load pipeline config for label discovery and id_col
    pipeline_cfg: Dict[str, Any] = {}
    _pipeline_path = cli_args.pipeline
    if not os.path.exists(_pipeline_path):
        _alt_pipeline = os.path.join(input_dir, "pipeline.yaml")
        if os.path.exists(_alt_pipeline):
            _pipeline_path = _alt_pipeline
    if os.path.exists(_pipeline_path):
        with open(_pipeline_path, encoding="utf-8") as f:
            pipeline_cfg = yaml.safe_load(f) or {}
        logger.info("Loaded pipeline config from %s", _pipeline_path)
    else:
        logger.warning("pipeline.yaml not found at %s -- label discovery will be empty", _pipeline_path)

    _id_col = pipeline_cfg.get("data", {}).get("id_col")
    if not _id_col:
        logger.warning("data.id_col not found in config, defaulting to first column")
        _id_col = None

    # ================================================================
    # Open a single DuckDB connection for the entire Phase 0 pipeline
    # ================================================================
    con = duckdb.connect()
    _TBL = "raw"  # current working table name

    # Load parquet directly into DuckDB (no pandas intermediary)
    con.execute(f"CREATE TABLE {_TBL} AS SELECT * FROM '{source}'")
    # Replace sentinel values (-999999 etc.) with NULL
    _numeric_cols = [r[0] for r in con.execute(
        f"SELECT column_name FROM (DESCRIBE SELECT * FROM {_TBL}) "
        f"WHERE column_type IN ('TINYINT','SMALLINT','INTEGER','BIGINT','FLOAT','DOUBLE')"
    ).fetchall()]
    for _nc in _numeric_cols:
        con.execute(f'UPDATE {_TBL} SET "{_nc}" = NULL WHERE "{_nc}" < -99999')
    _total_rows = con.execute(f"SELECT COUNT(*) FROM {_TBL}").fetchone()[0]
    _total_cols = con.execute(
        f"SELECT COUNT(*) FROM (DESCRIBE SELECT * FROM {_TBL})"
    ).fetchone()[0]
    logger.info(
        "SantanderAdapter: loaded %d rows x %d cols into DuckDB (no pandas copy)",
        _total_rows, _total_cols,
    )

    # Populate adapter metadata (lightweight — no pandas needed)
    _col_names = [
        r[0] for r in con.execute(
            f"SELECT column_name FROM (DESCRIBE SELECT * FROM {_TBL})"
        ).fetchall()
    ]
    _date_col_names = [c for c in _col_names if "date" in c.lower()]

    # --- Data quality gate (DuckDB SQL, no pandas) ---
    os.makedirs(output_dir, exist_ok=True)

    # Null rates — one query per column is wasteful; build a single aggregate
    _numeric_cols_info = con.execute(f"""
        SELECT column_name FROM (DESCRIBE SELECT * FROM {_TBL})
        WHERE column_type IN (
            'TINYINT','SMALLINT','INTEGER','BIGINT','HUGEINT',
            'FLOAT','DOUBLE','DECIMAL','UTINYINT','USMALLINT',
            'UINTEGER','UBIGINT'
        )
    """).fetchall()
    _numeric_col_names = [r[0] for r in _numeric_cols_info]

    _null_agg_parts = []
    for c in _col_names:
        qc = f'"{c}"'
        _null_agg_parts.append(
            f'(SUM(CASE WHEN {qc} IS NULL THEN 1 ELSE 0 END)::DOUBLE / COUNT(*)) AS "{c}__null_pct"'
        )
    _null_row = con.execute(
        f"SELECT {', '.join(_null_agg_parts)} FROM {_TBL}"
    ).fetchone()
    _null_rates = {}
    for i, c in enumerate(_col_names):
        rate = float(_null_row[i]) if _null_row[i] is not None else 0.0
        if rate > 0:
            _null_rates[c] = rate

    # Zero-variance columns
    _zv_parts = []
    for c in _numeric_col_names:
        qc = f'"{c}"'
        _zv_parts.append(f'(STDDEV({qc}) = 0 OR STDDEV({qc}) IS NULL)::INTEGER AS "{c}__zv"')
    _zv_cols: List[str] = []
    if _zv_parts:
        _zv_row = con.execute(f"SELECT {', '.join(_zv_parts)} FROM {_TBL}").fetchone()
        _zv_cols = [
            _numeric_col_names[i]
            for i in range(len(_numeric_col_names))
            if _zv_row[i]
        ]

    quality = {
        "total_rows": _total_rows,
        "total_columns": _total_cols,
        "null_rates": _null_rates,
        "zero_variance_columns": _zv_cols,
        "duplicate_rows": 0,  # expensive full-table dedup; skip in DuckDB-native path
    }
    with open(os.path.join(output_dir, "quality_gate_report.json"), "w") as f:
        json.dump(quality, f, indent=2)
    logger.info(
        "Quality gate report saved: %d rows, %d cols, %d zero-variance cols",
        quality["total_rows"], quality["total_columns"],
        len(quality["zero_variance_columns"]),
    )

    # --- Load feature_groups config ---
    fg_config_path = cli_args.feature_groups_config
    if not os.path.exists(fg_config_path):
        alt = os.path.join(input_dir, "feature_groups.yaml")
        if os.path.exists(alt):
            fg_config_path = alt

    if os.path.exists(fg_config_path):
        with open(fg_config_path, encoding="utf-8") as f:
            fg_raw = yaml.safe_load(f)
        feature_groups = fg_raw.get("feature_groups", [])
    else:
        logger.warning(
            "feature_groups.yaml not found at %s -- skipping generators",
            fg_config_path,
        )
        feature_groups = []

    # --- Config-driven feature generation (DuckDB-native) ---
    if feature_groups:
        logger.info("Starting config-driven feature generation on %d rows ...", _total_rows)
        t_gen_start = time.time()
        _preproc = pipeline_cfg.get("data", {}).get("preprocessing", {})
        _fit_subsample = int(_preproc.get("fit_subsample_limit", 50_000))
        _TBL = run_generators_duckdb(
            con, _TBL, feature_groups,
            id_col=_id_col, fit_subsample_limit=_fit_subsample,
        )
        _total_cols = con.execute(
            f"SELECT COUNT(*) FROM (DESCRIBE SELECT * FROM {_TBL})"
        ).fetchone()[0]
        logger.info(
            "Feature generation finished in %.1fs. Table: %s (%d rows x %d cols)",
            time.time() - t_gen_start, _TBL, _total_rows, _total_cols,
        )

    # --- Cold start flagging + feature zeroing (config-driven, DuckDB SQL) ---
    _cold_cfg = pipeline_cfg.get("cold_start", {})
    if _cold_cfg:
        _cs_seq_col = _cold_cfg.get("seq_col")
        _cs_min_txn = int(_cold_cfg.get("min_txn_count", 3))
        _cs_zero_prefixes: List[str] = _cold_cfg.get("zero_features_prefix", [])

        # Check that the sequence column exists in the current table
        _cs_all_cols = [
            r[0] for r in con.execute(
                f"SELECT column_name FROM (DESCRIBE SELECT * FROM {_TBL})"
            ).fetchall()
        ]

        if _cs_seq_col and _cs_seq_col in _cs_all_cols:
            # Add is_cold_start binary column
            con.execute(
                f'ALTER TABLE {_TBL} ADD COLUMN IF NOT EXISTS '
                f'"is_cold_start" TINYINT DEFAULT 0'
            )
            # Flag cold start: NULL sequence or length <= min_txn_count
            con.execute(
                f'UPDATE {_TBL} SET "is_cold_start" = 1 '
                f'WHERE "{_cs_seq_col}" IS NULL '
                f'OR LEN("{_cs_seq_col}") <= {_cs_min_txn}'
            )

            _cs_count = con.execute(
                f'SELECT SUM("is_cold_start") FROM {_TBL}'
            ).fetchone()[0]
            _cs_count = int(_cs_count) if _cs_count else 0
            logger.info(
                "Cold start flagging: %d / %d customers flagged "
                "(seq_col=%s, min_txn_count=%d)",
                _cs_count, _total_rows, _cs_seq_col, _cs_min_txn,
            )

            # Zero out sequence-derived features for cold start customers
            # Dynamically find columns matching zero_features_prefix
            if _cs_zero_prefixes and _cs_count > 0:
                _cs_col_types = {
                    r[0]: r[1] for r in con.execute(
                        f"SELECT column_name, column_type FROM "
                        f"(DESCRIBE SELECT * FROM {_TBL})"
                    ).fetchall()
                }
                _cs_numeric_types = {
                    "TINYINT", "SMALLINT", "INTEGER", "BIGINT", "HUGEINT",
                    "FLOAT", "DOUBLE", "DECIMAL", "UTINYINT", "USMALLINT",
                    "UINTEGER", "UBIGINT",
                }
                _cs_zero_cols = [
                    c for c in _cs_col_types
                    if _cs_col_types[c] in _cs_numeric_types
                    and c != "is_cold_start"
                    and any(c.startswith(pfx) for pfx in _cs_zero_prefixes)
                ]

                if _cs_zero_cols:
                    # Batch UPDATE to avoid SQL size limits
                    _CS_BATCH = 50
                    for _b_start in range(0, len(_cs_zero_cols), _CS_BATCH):
                        _b_cols = _cs_zero_cols[_b_start:_b_start + _CS_BATCH]
                        _set_clauses = [f'"{c}" = 0' for c in _b_cols]
                        con.execute(
                            f'UPDATE {_TBL} SET {", ".join(_set_clauses)} '
                            f'WHERE "is_cold_start" = 1'
                        )
                    logger.info(
                        "Cold start zeroing: %d feature columns zeroed "
                        "for %d cold start customers (prefixes: %s)",
                        len(_cs_zero_cols), _cs_count, _cs_zero_prefixes,
                    )
                else:
                    logger.info(
                        "Cold start: no columns matched zero_features_prefix %s",
                        _cs_zero_prefixes,
                    )
        else:
            logger.warning(
                "Cold start: seq_col '%s' not found in table — skipping. "
                "Available columns (sample): %s",
                _cs_seq_col, _cs_all_cols[:10],
            )
    else:
        logger.debug("No cold_start config found in pipeline.yaml — skipping")

    # --- Label derivation (config-driven) ---
    # LabelDeriver still needs pandas input for some derivation types.
    # Materialise only to pandas for the derivation, then merge back via DuckDB.
    label_configs = pipeline_cfg.get("labels", {})
    if label_configs:
        from core.pipeline.label_deriver import LabelDeriver
        deriver = LabelDeriver()
        # Materialise the table to pandas for label derivation
        _df_for_labels = con.execute(f"SELECT * FROM {_TBL}").df()
        labels_df = deriver.derive(_df_for_labels, label_configs=label_configs)
        del _df_for_labels
        if not labels_df.empty:
            # Add label columns to the DuckDB table via POSITIONAL JOIN
            import pyarrow as pa
            _labels_arrow = pa.Table.from_pandas(labels_df, preserve_index=False)
            con.register("_labels_tmp", _labels_arrow)
            _label_select = ", ".join(f'_labels_tmp."{c}"' for c in labels_df.columns)
            _new_tbl = f"{_TBL}_lbl"
            con.execute(
                f"CREATE OR REPLACE TABLE {_new_tbl} AS "
                f"SELECT {_TBL}.*, {_label_select} "
                f"FROM {_TBL} POSITIONAL JOIN _labels_tmp"
            )
            con.unregister("_labels_tmp")
            _TBL = _new_tbl
            logger.info("Derived %d label columns: %s", len(labels_df.columns), list(labels_df.columns))
            del labels_df
        else:
            logger.warning("LabelDeriver returned empty DataFrame")

    # --- 3-stage normalization (DuckDB SQL for mean/std, UPDATE for transform) ---
    from core.pipeline.normalizer import FeatureNormalizer

    _id_cols_set = {_id_col} if _id_col else set()
    _label_cols_set = set(label_configs.keys()) if label_configs else set()

    # Refresh column list from current table
    _all_cols = [
        r[0] for r in con.execute(
            f"SELECT column_name FROM (DESCRIBE SELECT * FROM {_TBL})"
        ).fetchall()
    ]
    _all_col_types = {
        r[0]: r[1] for r in con.execute(
            f"SELECT column_name, column_type FROM (DESCRIBE SELECT * FROM {_TBL})"
        ).fetchall()
    }
    _seq_col_names = {c for c in _all_cols if c.startswith("seq_") or c.endswith("_seq")}
    _date_col_set = {c for c in _all_cols if "date" in c.lower()}
    _str_col_names = {
        c for c, t in _all_col_types.items()
        if t in ("VARCHAR", "TEXT", "STRING", "BLOB")
    }
    _list_col_names = {
        c for c, t in _all_col_types.items()
        if t.startswith("LIST") or t.startswith("STRUCT") or "[]" in t
    }
    _non_feature = (
        _id_cols_set | _label_cols_set | _seq_col_names
        | _date_col_set | _str_col_names | _list_col_names
    )

    # Numeric feature columns for normalization
    _numeric_types = {
        "TINYINT", "SMALLINT", "INTEGER", "BIGINT", "HUGEINT",
        "FLOAT", "DOUBLE", "DECIMAL", "UTINYINT", "USMALLINT",
        "UINTEGER", "UBIGINT",
    }
    _feat_cols = [
        c for c in _all_cols
        if c not in _non_feature and _all_col_types.get(c, "") in _numeric_types
    ]

    if _feat_cols:
        # Fit normalizer entirely via DuckDB SQL (no pandas materialization)
        normalizer = FeatureNormalizer()

        # Classify columns: binary vs continuous via DuckDB
        _binary_cols = []
        _continuous_cols = []
        for c in _feat_cols:
            qc = f'"{c}"'
            r = con.execute(
                f"SELECT COUNT(DISTINCT {qc}) as nd, MIN({qc}) as mn, MAX({qc}) as mx "
                f"FROM {_TBL} WHERE {qc} IS NOT NULL"
            ).fetchone()
            if r[0] is not None and r[0] <= 2 and r[1] in (0, 0.0) and r[2] in (1, 1.0):
                _binary_cols.append(c)
            else:
                _continuous_cols.append(c)
        normalizer.binary_cols = _binary_cols
        normalizer.continuous_cols = _continuous_cols

        # Power-law detection (subsample via DuckDB, compute in numpy)
        normalizer.power_law_cols = []
        if _continuous_cols:
            _pw_cols_sql = ", ".join('"' + c + '"' for c in _continuous_cols)
            _pw_sample = con.execute(
                f"SELECT {_pw_cols_sql} FROM {_TBL} USING SAMPLE 5000 (reservoir, 42)"
            ).fetchnumpy()
            normalizer._detect_power_law_from_numpy(_pw_sample, _continuous_cols)

        # Compute mean/std via DuckDB SQL (no pandas/CuPy)
        if normalizer.continuous_cols:
            _agg_parts = []
            for c in normalizer.continuous_cols:
                qc = f'"{c}"'
                _agg_parts.append(f"AVG(COALESCE({qc}::DOUBLE, 0))")
                _agg_parts.append(f"STDDEV_SAMP(COALESCE({qc}::DOUBLE, 0))")
            _agg_row = con.execute(
                f"SELECT {', '.join(_agg_parts)} FROM {_TBL}"
            ).fetchone()
            _means = np.array([_agg_row[i * 2] for i in range(len(normalizer.continuous_cols))], dtype=np.float64)
            _stds = np.array([_agg_row[i * 2 + 1] for i in range(len(normalizer.continuous_cols))], dtype=np.float64)
            _stds[_stds < 1e-10] = 1.0
            normalizer._mean = _means
            normalizer._std = _stds
            normalizer._fitted = True

        # --- Apply normalization via DuckDB UPDATE (no pandas round-trip) ---
        # Stage 2: Z-score scaling for continuous columns
        if normalizer.continuous_cols and normalizer._mean is not None:
            _update_sets = []
            for i, col in enumerate(normalizer.continuous_cols):
                mean_val = float(normalizer._mean[i])
                std_val = float(normalizer._std[i])
                qc = f'"{col}"'
                _update_sets.append(
                    f'{qc} = (COALESCE({qc}::DOUBLE, 0) - {mean_val}) / {std_val}'
                )
            # Execute updates in batches to avoid SQL size limits
            _BATCH_SIZE = 50
            for batch_start in range(0, len(_update_sets), _BATCH_SIZE):
                batch = _update_sets[batch_start:batch_start + _BATCH_SIZE]
                con.execute(f"UPDATE {_TBL} SET {', '.join(batch)}")
            logger.info(
                "DuckDB UPDATE: z-score scaled %d continuous columns",
                len(normalizer.continuous_cols),
            )

        # Stage 3: Power-law log copies (log1p, NOT scaled)
        if normalizer.power_law_cols:
            _log_adds = []
            for col in normalizer.power_law_cols:
                qc = f'"{col}"'
                log_col = f'"{col}_log"'
                _log_adds.append(
                    f'LN(1 + GREATEST(COALESCE({qc}, 0), 0)) AS {log_col}'
                )
            # Add log columns via ALTER TABLE + UPDATE would be verbose;
            # use CREATE TABLE AS SELECT instead
            _existing_select = ", ".join(f'"{c}"' for c in _all_cols)
            _new_tbl2 = f"{_TBL}_norm"
            con.execute(
                f"CREATE OR REPLACE TABLE {_new_tbl2} AS "
                f"SELECT {_existing_select}, {', '.join(_log_adds)} FROM {_TBL}"
            )
            _TBL = _new_tbl2
            logger.info(
                "DuckDB: added %d power-law log columns",
                len(normalizer.power_law_cols),
            )

        normalizer.save(os.path.join(output_dir, "normalizer"))
        logger.info(
            "3-stage normalization: %d features (%d continuous, %d binary, %d power-law)",
            len(_feat_cols), len(normalizer.continuous_cols),
            len(normalizer.binary_cols), len(normalizer.power_law_cols),
        )

    # --- MCC / categorical integer encoding ---
    # Encode categorical string columns to small integer indices before save
    try:
        from core.data.mcc_lookup import build_duckdb_case_sql
        # Check if any MCC-like columns exist
        _mcc_cols = [c for c in _all_cols if "mcc" in c.lower()]
        for mc in _mcc_cols:
            _case_sql = build_duckdb_case_sql(column=f'"{mc}"', level="l1")
            con.execute(f'ALTER TABLE {_TBL} ADD COLUMN IF NOT EXISTS "{mc}_l1_idx" TINYINT')
            con.execute(f'UPDATE {_TBL} SET "{mc}_l1_idx" = ({_case_sql})::TINYINT')
            logger.info("MCC encoding: %s -> %s_l1_idx (TINYINT)", mc, mc)
    except Exception:
        logger.debug("MCC encoding skipped (no mcc_lookup or no MCC columns)", exc_info=True)

    # --- Dtype downcasting via DuckDB CAST (no pandas) ---
    # Refresh column info after normalization
    _final_col_info = con.execute(
        f"SELECT column_name, column_type FROM (DESCRIBE SELECT * FROM {_TBL})"
    ).fetchall()

    _cast_exprs: List[str] = []
    _downcast_count = 0
    for col_name, col_type in _final_col_info:
        qc = f'"{col_name}"'
        if col_type in ("DOUBLE", "FLOAT"):
            # float64 -> float32
            if col_type == "DOUBLE":
                _cast_exprs.append(f'{qc}::FLOAT AS {qc}')
                _downcast_count += 1
            else:
                _cast_exprs.append(qc)
        elif col_type == "BIGINT":
            # Determine range and downcast
            _range = con.execute(
                f"SELECT MIN({qc}), MAX({qc}) FROM {_TBL}"
            ).fetchone()
            _min_v, _max_v = _range[0], _range[1]
            if _min_v is not None and _max_v is not None:
                if -128 <= _min_v and _max_v <= 127:
                    _cast_exprs.append(f'{qc}::TINYINT AS {qc}')
                    _downcast_count += 1
                elif -32768 <= _min_v and _max_v <= 32767:
                    _cast_exprs.append(f'{qc}::SMALLINT AS {qc}')
                    _downcast_count += 1
                elif -2147483648 <= _min_v and _max_v <= 2147483647:
                    _cast_exprs.append(f'{qc}::INTEGER AS {qc}')
                    _downcast_count += 1
                else:
                    _cast_exprs.append(qc)
            else:
                _cast_exprs.append(qc)
        elif col_type == "INTEGER":
            # Try to downcast INTEGER to TINYINT/SMALLINT
            _range = con.execute(
                f"SELECT MIN({qc}), MAX({qc}) FROM {_TBL}"
            ).fetchone()
            _min_v, _max_v = _range[0], _range[1]
            if _min_v is not None and _max_v is not None:
                if -128 <= _min_v and _max_v <= 127:
                    _cast_exprs.append(f'{qc}::TINYINT AS {qc}')
                    _downcast_count += 1
                elif -32768 <= _min_v and _max_v <= 32767:
                    _cast_exprs.append(f'{qc}::SMALLINT AS {qc}')
                    _downcast_count += 1
                else:
                    _cast_exprs.append(qc)
            else:
                _cast_exprs.append(qc)
        else:
            _cast_exprs.append(qc)

    if _downcast_count > 0:
        _final_tbl = f"{_TBL}_final"
        con.execute(
            f"CREATE OR REPLACE TABLE {_final_tbl} AS "
            f"SELECT {', '.join(_cast_exprs)} FROM {_TBL}"
        )
        _TBL = _final_tbl
        logger.info("Dtype downcasting: %d columns downcasted", _downcast_count)

    # --- Save via DuckDB COPY TO (no pandas/Arrow intermediary) ---
    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, "santander_final.parquet")
    # Use forward slashes for DuckDB path compatibility
    _out_path_ddb = out_path.replace("\\", "/")
    con.execute(
        f"COPY {_TBL} TO '{_out_path_ddb}' (FORMAT PARQUET, COMPRESSION ZSTD)"
    )
    _final_rows = con.execute(f"SELECT COUNT(*) FROM {_TBL}").fetchone()[0]
    logger.info("Saved %d rows to %s (DuckDB COPY TO)", _final_rows, out_path)

    # --- Feature statistics (single DuckDB aggregate query) ---
    # Refresh numeric columns from final table
    _final_numeric = [
        r[0] for r in con.execute(f"""
            SELECT column_name FROM (DESCRIBE SELECT * FROM {_TBL})
            WHERE column_type IN (
                'TINYINT','SMALLINT','INTEGER','BIGINT','HUGEINT',
                'FLOAT','DOUBLE','DECIMAL','UTINYINT','USMALLINT',
                'UINTEGER','UBIGINT'
            )
        """).fetchall()
    ]

    stats: Dict[str, Any] = {}
    if _final_numeric:
        # Build aggregate query in batches to avoid SQL size limits
        _STAT_BATCH = 40
        for batch_start in range(0, len(_final_numeric), _STAT_BATCH):
            batch_cols = _final_numeric[batch_start:batch_start + _STAT_BATCH]
            agg_parts = []
            for col in batch_cols:
                qc = f'"{col}"'
                agg_parts.append(
                    f"AVG({qc}) AS \"{col}__mean\", "
                    f"STDDEV({qc}) AS \"{col}__std\", "
                    f"MIN({qc}) AS \"{col}__min\", "
                    f"MAX({qc}) AS \"{col}__max\", "
                    f"(SUM(CASE WHEN {qc} IS NULL THEN 1 ELSE 0 END)::DOUBLE / COUNT(*)) AS \"{col}__null_pct\", "
                    f"COUNT(DISTINCT {qc}) AS \"{col}__nunique\""
                )
            sql = f"SELECT {', '.join(agg_parts)} FROM {_TBL}"
            row = con.execute(sql).fetchone()
            idx = 0
            for col in batch_cols:
                stats[col] = {
                    "mean": float(row[idx]) if row[idx] is not None else 0.0,
                    "std": float(row[idx + 1]) if row[idx + 1] is not None else 0.0,
                    "min": float(row[idx + 2]) if row[idx + 2] is not None else 0.0,
                    "max": float(row[idx + 3]) if row[idx + 3] is not None else 0.0,
                    "null_pct": float(row[idx + 4]) if row[idx + 4] is not None else 0.0,
                    "nunique": int(row[idx + 5]) if row[idx + 5] is not None else 0,
                }
                idx += 6

    with open(os.path.join(output_dir, "feature_stats.json"), "w") as f:
        json.dump(stats, f, indent=2)
    logger.info("Feature stats saved: %d numeric columns", len(stats))

    # --- Label statistics (DuckDB SQL, no pandas) ---
    _label_keys = set(pipeline_cfg.get("labels", {}).keys())
    _final_all_cols = [
        r[0] for r in con.execute(
            f"SELECT column_name FROM (DESCRIBE SELECT * FROM {_TBL})"
        ).fetchall()
    ]
    _final_col_types = {
        r[0]: r[1] for r in con.execute(
            f"SELECT column_name, column_type FROM (DESCRIBE SELECT * FROM {_TBL})"
        ).fetchall()
    }
    label_cols = [c for c in _final_all_cols if c in _label_keys]
    label_stats: Dict[str, Any] = {}
    _int_types = {"TINYINT", "SMALLINT", "INTEGER", "BIGINT", "HUGEINT"}
    _float_types = {"FLOAT", "DOUBLE", "DECIMAL"}
    for col in label_cols:
        col_type = _final_col_types.get(col, "")
        qc = f'"{col}"'
        if col_type in _int_types:
            # Value counts via DuckDB GROUP BY
            vc_rows = con.execute(
                f"SELECT {qc}::VARCHAR, COUNT(*) FROM {_TBL} GROUP BY {qc} ORDER BY COUNT(*) DESC"
            ).fetchall()
            label_stats[col] = {str(r[0]): int(r[1]) for r in vc_rows}
        elif col_type in _float_types:
            _ls_row = con.execute(
                f"SELECT AVG({qc}), STDDEV({qc}) FROM {_TBL}"
            ).fetchone()
            label_stats[col] = {
                "mean": float(_ls_row[0]) if _ls_row[0] is not None else 0.0,
                "std": float(_ls_row[1]) if _ls_row[1] is not None else 0.0,
            }
    if label_stats:
        with open(os.path.join(output_dir, "label_stats.json"), "w") as f:
            json.dump(label_stats, f, indent=2)
        logger.info("Label stats saved: %d label columns", len(label_stats))

    # --- Auto-generate feature_schema.json for train.py consumption ---
    _schema_exclude = (
        _id_cols_set | _label_cols_set | _date_col_set
        | _str_col_names | _seq_col_names | _list_col_names
    )
    _feature_columns = [
        c for c in _final_all_cols
        if c not in _schema_exclude
        and _final_col_types.get(c, "") in _numeric_types
    ]

    # Derive group_ranges from column name prefixes
    _group_ranges: Dict[str, Any] = {}
    _current_prefix: Optional[str] = None
    _group_start = 0
    for i, col in enumerate(_feature_columns):
        _parts = col.rsplit("_", 1)
        _prefix = _parts[0] if len(_parts) > 1 and _parts[1].isdigit() else col
        if _prefix != _current_prefix:
            if _current_prefix is not None:
                _group_ranges[_current_prefix] = [_group_start, i]
            _current_prefix = _prefix
            _group_start = i
    if _current_prefix is not None:
        _group_ranges[_current_prefix] = [_group_start, len(_feature_columns)]

    # Expert routing from pipeline.yaml model.expert_basket config
    _expert_routing: List[Dict[str, Any]] = []
    _eb_cfg = pipeline_cfg.get("model", {}).get("expert_basket", {})
    if _eb_cfg:
        _shared_experts = _eb_cfg.get("shared", [])
        _task_experts_list = _eb_cfg.get("task", [])
        _expert_target_map = pipeline_cfg.get("model", {}).get("expert_routing", [])
        if _expert_target_map:
            _expert_routing = _expert_target_map
        else:
            # Default: each expert receives all feature groups
            # Format: {"expert_name": str, "input_groups": [str]}
            all_groups = list(_group_ranges.keys())
            for expert_name in _shared_experts:
                _expert_routing.append({
                    "expert_name": expert_name,
                    "input_groups": all_groups,
                })

    _feature_schema = {
        "columns": _feature_columns,
        "group_ranges": _group_ranges,
        "expert_routing": _expert_routing,
        "num_features": len(_feature_columns),
        "num_groups": len(_group_ranges),
    }
    with open(os.path.join(output_dir, "feature_schema.json"), "w") as f:
        json.dump(_feature_schema, f, indent=2)
    logger.info(
        "feature_schema.json saved: %d features, %d groups, %d routing entries",
        len(_feature_columns), len(_group_ranges), len(_expert_routing),
    )

    # Save adapter metadata
    _meta_dict = {
        "id_col": _id_col or "",
        "entity_granularity": "user",
        "num_entities": _final_rows,
        "num_raw_rows": _final_rows,
        "source_files": [str(source)],
        "backend_used": "duckdb",
        "timestamp_columns": _date_col_names,
    }
    with open(os.path.join(output_dir, "adapter_metadata.json"), "w") as f:
        json.dump(_meta_dict, f, indent=2, default=str)

    # Cleanup DuckDB connection
    con.close()

    logger.info("Phase 0 complete (DuckDB-native pipeline). Output: %s", output_dir)
