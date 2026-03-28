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

def _resolve_columns_by_filter(
    df: pd.DataFrame,
    filter_config: Dict[str, Any],
    id_col: Optional[str] = None,
) -> List[str]:
    """Resolve columns matching the filter criteria from config.

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


def run_generators_from_config(
    df: pd.DataFrame,
    feature_groups_config: List[Dict[str, Any]],
    id_col: Optional[str] = None,
    fit_subsample_limit: int = 50_000,
) -> pd.DataFrame:
    """Run generators based on feature_groups config, NOT hardcoded columns.

    Only groups with ``group_type == "generate"`` and ``enabled != False``
    are executed.  Each generator is wrapped in try/except so a single
    failure does not block the rest.

    Parameters
    ----------
    df : pd.DataFrame
        The raw DataFrame (e.g. santander_final.parquet).
    feature_groups_config : list[dict]
        The ``feature_groups`` list from feature_groups.yaml.

    Returns
    -------
    pd.DataFrame
        Original *df* with generated feature columns appended.
    """
    # Lazy imports -- only needed when generators are actually invoked.
    from core.feature.generator import FeatureGeneratorRegistry
    # Trigger side-effect registration of all built-in generators.
    import core.feature.generators  # noqa: F401

    generated_frames: List[pd.DataFrame] = []
    gen_summary: Dict[str, int] = {}

    # Subsample for fitting to avoid OOM on large datasets
    _FIT_SUBSAMPLE_LIMIT = fit_subsample_limit
    if len(df) > _FIT_SUBSAMPLE_LIMIT:
        fit_df = df.sample(_FIT_SUBSAMPLE_LIMIT, random_state=42)
        logger.info(
            "Large dataset (%d rows): fitting generators on %d-row subsample",
            len(df), _FIT_SUBSAMPLE_LIMIT,
        )
    else:
        fit_df = df

    for group in feature_groups_config:
        if group.get("group_type") != "generate":
            continue
        if not group.get("enabled", True):
            continue

        group_name = group["name"]
        generator_name = group.get("generator", group_name)
        gen_params = dict(group.get("generator_params", {}))

        # --- Resolve input columns from config filter ---
        input_filter = gen_params.pop("input_filter", None)
        if input_filter is not None:
            input_cols = _resolve_columns_by_filter(df, input_filter, id_col=id_col)
        else:
            # Fallback: no filter → all numeric columns (exclude id_col)
            input_cols = [
                c for c in df.select_dtypes(include="number").columns
                if not (id_col and c == id_col)
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
            gen = FeatureGeneratorRegistry.create(
                generator_name,
                input_columns=input_cols,
                prefix=group_name,
                **gen_params,
            )
            gen.fit(fit_df[input_cols])
            result = gen.generate(df[input_cols])

            if not isinstance(result, pd.DataFrame):
                result = pd.DataFrame(result)
            result.index = df.index

            generated_frames.append(result)
            gen_summary[group_name] = len(result.columns)
            logger.info(
                "Generator '%s': %d features in %.1fs",
                group_name, len(result.columns), time.time() - t0,
            )
        except Exception as e:
            logger.warning("Generator '%s' failed: %s", group_name, e, exc_info=True)
            # Produce fallback zeros using output_dim from config
            n_cols = group.get("output_dim", 10)
            fallback = pd.DataFrame(
                np.zeros((len(df), n_cols), dtype=np.float32),
                columns=[f"{group_name}_{i:03d}" for i in range(n_cols)],
                index=df.index,
            )
            generated_frames.append(fallback)
            gen_summary[group_name] = n_cols

    # Merge all generated features via DuckDB horizontal join
    if generated_frames:
        import duckdb as _ddb_merge
        _con_m = _ddb_merge.connect()
        try:
            # Register all generated frames and build a chained POSITIONAL JOIN
            _con_m.register("_base", df)
            join_sql = "SELECT _base.*"
            from_sql = "_base"
            for i, gf in enumerate(generated_frames):
                alias = f"_gf{i}"
                _con_m.register(alias, gf)
                gen_cols = ", ".join(f'{alias}."{c}"' for c in gf.columns)
                join_sql += f", {gen_cols}"
                from_sql += f" POSITIONAL JOIN {alias}"
            df = _con_m.execute(f"{join_sql} FROM {from_sql}").df()
        finally:
            _con_m.close()
        total_new = sum(gen_summary.values())
        logger.info(
            "Feature generation complete: %d new columns from %d generators. "
            "Total columns: %d. Summary: %s",
            total_new, len(gen_summary), len(df.columns), gen_summary,
        )

    return df


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
            df = pd.read_parquet(source)

        self._metadata = AdapterMetadata(
            id_col=self._id_col,
            entity_granularity="user",
            num_entities=len(df),
            num_raw_rows=len(df),
            source_files=[str(source)],
            backend_used=backend,
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
        with open(_pipeline_path) as f:
            pipeline_cfg = yaml.safe_load(f) or {}
        logger.info("Loaded pipeline config from %s", _pipeline_path)
    else:
        logger.warning("pipeline.yaml not found at %s -- label discovery will be empty", _pipeline_path)

    _id_col = pipeline_cfg.get("data", {}).get("id_col")
    if not _id_col:
        logger.warning("data.id_col not found in config, defaulting to first column")
        _id_col = None

    # Load via adapter
    config: Dict[str, Any] = {"data": {"source": source, "backend": ["duckdb", "pandas"], "id_col": _id_col}}
    adapter = SantanderAdapter(config)
    raw_data = adapter.load_raw()
    df = raw_data["main"]

    # --- Data quality gate ---
    quality = {
        "total_rows": len(df),
        "total_columns": len(df.columns),
        "null_rates": {
            col: float(df[col].isna().mean())
            for col in df.columns if df[col].isna().any()
        },
        "zero_variance_columns": [
            col for col in df.select_dtypes("number").columns
            if df[col].std() == 0
        ],
        "duplicate_rows": int(df.select_dtypes(include="number").duplicated().sum()),
    }
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, "quality_gate_report.json"), "w") as f:
        json.dump(quality, f, indent=2)
    logger.info(
        "Quality gate report saved: %d rows, %d cols, %d zero-variance cols",
        quality["total_rows"], quality["total_columns"],
        len(quality["zero_variance_columns"]),
    )

    # --- Load feature_groups config ---
    fg_config_path = cli_args.feature_groups_config
    # Also check inside /opt/ml/processing/input/ if the default isn't found
    if not os.path.exists(fg_config_path):
        alt = os.path.join(input_dir, "feature_groups.yaml")
        if os.path.exists(alt):
            fg_config_path = alt

    if os.path.exists(fg_config_path):
        with open(fg_config_path) as f:
            fg_raw = yaml.safe_load(f)
        feature_groups = fg_raw.get("feature_groups", [])
    else:
        logger.warning(
            "feature_groups.yaml not found at %s -- skipping generators",
            fg_config_path,
        )
        feature_groups = []

    # --- Config-driven feature generation ---
    if feature_groups:
        logger.info("Starting config-driven feature generation on %d rows ...", len(df))
        t_gen_start = time.time()
        _preproc = pipeline_cfg.get("data", {}).get("preprocessing", {})
        _fit_subsample = int(_preproc.get("fit_subsample_limit", 50_000))
        df = run_generators_from_config(df, feature_groups, id_col=_id_col,
                                         fit_subsample_limit=_fit_subsample)
        logger.info(
            "Feature generation finished in %.1fs. Shape: %s",
            time.time() - t_gen_start, df.shape,
        )

    # Save to output dir
    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, "santander_final.parquet")
    import duckdb as _ddb_save
    _con_s = _ddb_save.connect()
    try:
        _con_s.register("_df_save", df)
        _con_s.execute(f"COPY _df_save TO '{out_path}' (FORMAT PARQUET)")
    finally:
        _con_s.close()
    logger.info("Saved %d rows to %s", len(df), out_path)

    # --- Feature statistics (single DuckDB aggregate query) ---
    numeric = df.select_dtypes(include="number")
    stats = {}
    if len(numeric.columns) > 0:
        import duckdb as _ddb_stats
        _con_st = _ddb_stats.connect()
        try:
            _con_st.register("_num_df", numeric)
            # Build one aggregate query for all numeric columns
            agg_parts = []
            for col in numeric.columns:
                qc = f'"{col}"'
                agg_parts.append(
                    f"AVG({qc}) AS \"{col}__mean\", "
                    f"STDDEV({qc}) AS \"{col}__std\", "
                    f"MIN({qc}) AS \"{col}__min\", "
                    f"MAX({qc}) AS \"{col}__max\", "
                    f"(SUM(CASE WHEN {qc} IS NULL THEN 1 ELSE 0 END)::DOUBLE / COUNT(*)) AS \"{col}__null_pct\", "
                    f"COUNT(DISTINCT {qc}) AS \"{col}__nunique\""
                )
            sql = f"SELECT {', '.join(agg_parts)} FROM _num_df"
            row = _con_st.execute(sql).fetchone()
            idx = 0
            for col in numeric.columns:
                stats[col] = {
                    "mean": float(row[idx]) if row[idx] is not None else 0.0,
                    "std": float(row[idx + 1]) if row[idx + 1] is not None else 0.0,
                    "min": float(row[idx + 2]) if row[idx + 2] is not None else 0.0,
                    "max": float(row[idx + 3]) if row[idx + 3] is not None else 0.0,
                    "null_pct": float(row[idx + 4]) if row[idx + 4] is not None else 0.0,
                    "nunique": int(row[idx + 5]) if row[idx + 5] is not None else 0,
                }
                idx += 6
        finally:
            _con_st.close()
    with open(os.path.join(output_dir, "feature_stats.json"), "w") as f:
        json.dump(stats, f, indent=2)
    logger.info("Feature stats saved: %d numeric columns", len(stats))

    # --- Label statistics (config-driven from pipeline.yaml labels section) ---
    _label_keys = set(pipeline_cfg.get("labels", {}).keys())
    label_cols = [c for c in df.columns if c in _label_keys]
    label_stats = {}
    for col in label_cols:
        if df[col].dtype in [np.int32, np.int64]:
            label_stats[col] = {
                str(k): int(v) for k, v in df[col].value_counts().to_dict().items()
            }
        elif df[col].dtype == float:
            label_stats[col] = {
                "mean": float(df[col].mean()),
                "std": float(df[col].std()),
            }
    if label_stats:
        with open(os.path.join(output_dir, "label_stats.json"), "w") as f:
            json.dump(label_stats, f, indent=2)
        logger.info("Label stats saved: %d label columns", len(label_stats))

    # Save metadata
    meta = adapter.metadata
    with open(os.path.join(output_dir, "adapter_metadata.json"), "w") as f:
        json.dump(meta.__dict__, f, indent=2, default=str)

    logger.info("Phase 0 complete. Output: %s", output_dir)
