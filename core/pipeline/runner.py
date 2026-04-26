"""
PipelineRunner -- universal pipeline orchestrator.

Two modes of operation:

1. **Phase 0** (``run()``) -- produces training-ready artifacts on disk.
   No model code is imported.  Output::

       features.parquet      -- all numeric features (base + generated), normalized
       labels.parquet        -- all derived labels
       sequences.npy         -- padded 3-D tensor (if applicable)
       seq_lengths.npy       -- per-entity sequence lengths
       scaler_params.json    -- fitted scaler parameters (from TRAIN split only)
       feature_schema.json   -- column names, types, group ranges, expert routing
       label_schema.json     -- task definitions with types and class counts
       split_indices.json    -- train/val/test row indices

2. **Full pipeline** (``run_full()``) -- Phase 0 + training + distillation +
   serving prep (stages 7-10 of the original 10-stage design).

Usage::

    config = load_config("configs/examples/multitask.yaml")
    runner = PipelineRunner(config)

    # Phase 0 only (training-ready artifacts)
    artifacts = runner.run(output_dir="outputs/phase0/")

    # Full pipeline (Phase 0 + training + analysis + distillation)
    results = runner.run_full(output_dir="outputs/")
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, TYPE_CHECKING

if TYPE_CHECKING:
    import numpy as np
    import pandas as pd

from .adapter import DataAdapter
from .config import PipelineConfig

logger = logging.getLogger(__name__)


def _longest_contiguous_run(
    positions: List[int],
) -> Optional[Tuple[int, int]]:
    """Return ``(start, end)`` of the longest contiguous run in *positions*.

    Positions are expected to be a sorted list of integer column indices.
    The returned range is a half-open interval ``[start, end)`` such that
    ``end - start`` columns are covered. Ties on length pick the earliest
    run.  Returns ``None`` on empty input.
    """
    if not positions:
        return None
    sorted_pos = sorted(set(positions))
    best_start = sorted_pos[0]
    best_end = sorted_pos[0] + 1
    cur_start = sorted_pos[0]
    cur_end = sorted_pos[0] + 1
    for p in sorted_pos[1:]:
        if p == cur_end:
            cur_end = p + 1
        else:
            if cur_end - cur_start > best_end - best_start:
                best_start, best_end = cur_start, cur_end
            cur_start, cur_end = p, p + 1
    if cur_end - cur_start > best_end - best_start:
        best_start, best_end = cur_start, cur_end
    return best_start, best_end


def _rebuild_group_ranges_post_normalization(
    feature_pipeline: Any,
    feature_cols: List[str],
    log_cols_created: List[str],
) -> Dict[str, Tuple[int, int]]:
    """Re-map feature_group_ranges against the post-normalization column order.

    CLAUDE.md §1.7 — the normalizer reorders / appends ``{col}_log`` copies,
    breaking the pre-normalization ``(start, end)`` ranges.  For every
    enabled group we:

    1. Resolve the set of original column names that belong to the group
       (from ``group.output_columns``).
    2. Attach ``{col}_log`` offspring to the same group when they exist in
       ``log_cols_created`` and their base column was a member of the
       group.
    3. Locate each name in the new ``feature_cols`` list.
    4. Return the *longest contiguous block* of positions as
       ``(start, end)``.  Logs a warning when the group's columns ended up
       non-contiguous so operators can trace routing anomalies back to
       normalization-induced reordering.

    Groups with zero resolvable columns are dropped from the map.  When
    ``feature_pipeline`` has no ``_registry`` (e.g. a mock in tests),
    returns an empty dict so the caller keeps its Stage 3 ranges.
    """
    registry = getattr(feature_pipeline, "_registry", None)
    if registry is None or not getattr(registry, "enabled_groups", None):
        return {}

    log_set = set(log_cols_created or [])
    col_index = {c: i for i, c in enumerate(feature_cols)}

    rebuilt: Dict[str, Tuple[int, int]] = {}
    for group in registry.enabled_groups:
        group_cols = list(getattr(group, "output_columns", []) or [])
        # Attach log offspring of this group's columns.
        for base in list(group_cols):
            log_name = f"{base}_log"
            if log_name in log_set and log_name not in group_cols:
                group_cols.append(log_name)

        positions = [
            col_index[c] for c in group_cols if c in col_index
        ]
        if not positions:
            continue

        block = _longest_contiguous_run(positions)
        if block is None:
            continue
        start, end = block

        if end - start != len(positions):
            logger.warning(
                "[Stage 6] Group '%s' is non-contiguous after normalization "
                "(%d columns, longest block [%d, %d) = %d cols). Columns "
                "outside the block will be orphaned from expert routing.",
                group.name,
                len(positions),
                start,
                end,
                end - start,
            )
        rebuilt[group.name] = (start, end)

    return rebuilt



# ======================================================================
# Pipeline state tracker for stage-level checkpointing / resume
# ======================================================================

class _PipelineState:
    """Tracks completed stages for pipeline resume capability."""

    def __init__(self, output_dir: str, callbacks: Optional[List[Callable]] = None) -> None:
        self.path = Path(output_dir) / "pipeline_state.json"
        self.state = self._load()
        self._callbacks = callbacks or []

    def _load(self) -> dict:
        if self.path.exists():
            with open(self.path) as f:
                return json.load(f)
        return {"completed_stages": [], "artifacts": {}, "start_time": None}

    def mark_complete(self, stage: str, artifacts: dict = None) -> None:
        if stage not in self.state["completed_stages"]:
            self.state["completed_stages"].append(stage)
        if artifacts:
            self.state["artifacts"][stage] = artifacts
        self._save()
        # Notify callbacks
        for cb in self._callbacks:
            try:
                cb(stage, artifacts or {})
            except Exception as e:
                logger.warning("Pipeline state callback failed: %s", e)

    def mark_failed(self, stage: str, error: str) -> None:
        self.state["failed_stage"] = stage
        self.state["error"] = error
        self._save()

    def is_complete(self, stage: str) -> bool:
        return stage in self.state["completed_stages"]

    def _save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.path, "w") as f:
            json.dump(self.state, f, indent=2, default=str)


# ======================================================================
# Helper utilities (module-level, stateless)
# ======================================================================

def _is_binary(series: "pd.Series", max_sample: int = 5000) -> bool:
    """Return True if *series* looks like a binary 0/1 column."""
    import numpy as np

    sample = series.dropna()
    if len(sample) > max_sample:
        sample = sample.sample(max_sample, random_state=0)
    unique = set(sample.unique())
    return unique.issubset({0, 1, 0.0, 1.0, True, False})


class PipelineRunner:
    """Execute the pipeline driven by :class:`PipelineConfig`.

    The primary entry point is :meth:`run` (Phase 0), which produces
    training-ready artifacts on disk without importing any model code.

    For the full end-to-end pipeline (including training, analysis,
    distillation, and serving), use :meth:`run_full`.

    Args:
        config: A fully populated :class:`PipelineConfig`.
    """

    def __init__(self, config: PipelineConfig) -> None:
        self.config = config
        # Adapter-supplied DuckDB context (CLAUDE.md §3.3). Set by Stage
        # 1 when the adapter returns a :class:`DuckDBAdapterContext`,
        # left ``None`` for legacy dict-returning adapters. SQL-native
        # generators (lag/rolling/multihot) read ``self._adapter_ctx``
        # in Stage 3 to query the source table directly without forcing
        # a 1M-row pandas materialisation of the LIST columns.
        self._adapter_ctx = None

    # ==================================================================
    # Phase 0: produce training-ready artifacts
    # ==================================================================

    def run(
        self,
        output_dir: str = "outputs/",
        start_stage: int = 1,
        end_stage: int = 9,
        checkpoint_dir: Optional[str] = None,
    ) -> dict:
        """Run the 9-stage Phase-0 pipeline and save training-ready artifacts.

        Stages:
          1. Load raw data via adapter
          2. Preprocessing (sentinels, categorical encoding, imputation)
          3. Feature engineering via FeatureGroupPipeline
          4. Label derivation via LabelDeriver
          5. Temporal split (train / val / test indices)
          6. Normalization (3-stage, train-only fit)
          7. Leakage validation
          8. Sequence building (flat -> 3-D tensors)
          9. Save artifacts

        For OOM-prone datasets the 9 stages can be split across multiple
        SageMaker jobs:

          * ``start_stage`` / ``end_stage`` constrain which stages run
            in this invocation. Out-of-range stages are skipped.
          * ``checkpoint_dir`` is a local directory (or SageMaker
            channel) that holds inter-stage parquet files. When
            ``start_stage > 1`` the runner loads ``post_stage<N-1>.parquet``
            from this dir; when ``end_stage < 9`` it writes
            ``post_stage<end_stage>.parquet`` here so the next job can
            pick it up. Skipping checkpoints (single-job mode) keeps the
            historical behaviour.

        Args:
            output_dir: Directory for all output artifacts.
            start_stage: First stage to run (1-9). Default 1.
            end_stage: Last stage to run (inclusive, 1-9). Default 9.
            checkpoint_dir: Optional path for inter-stage parquet
                checkpoints. Required when start_stage > 1.

        Returns:
            A result dict with metadata from each stage plus artifact paths.
        """
        import numpy as np
        import pandas as pd
        from sklearn.preprocessing import LabelEncoder

        results: Dict[str, Any] = {}
        pipeline_start = time.time()

        # --------------------------------------------------------------
        # Stage gating + checkpoint setup (multi-job Phase 0 split)
        # --------------------------------------------------------------
        if not (1 <= start_stage <= 9 and 1 <= end_stage <= 9):
            raise ValueError(
                f"start_stage / end_stage must be in [1, 9]; got "
                f"start_stage={start_stage}, end_stage={end_stage}"
            )
        if start_stage > end_stage:
            raise ValueError(
                f"start_stage ({start_stage}) > end_stage ({end_stage})"
            )
        # The 3-job split has checkpoints at boundaries 2->3 and 3->4
        # only. start_stage must therefore land at one of {1, 3, 4}.
        # (Higher values like 5..9 are disallowed because we never
        # serialise inter-stage state for the 4-9 monolith.)
        if start_stage not in (1, 3, 4):
            raise ValueError(
                f"start_stage must be 1, 3, or 4 (boundaries of the "
                f"3-job split). Got {start_stage}."
            )
        # End boundaries: 2 (post Job A), 3 (post Job B), 9 (Job C / full).
        if end_stage not in (2, 3, 9):
            raise ValueError(
                f"end_stage must be 2, 3, or 9 (boundaries of the "
                f"3-job split). Got {end_stage}."
            )
        if start_stage > 1 and checkpoint_dir is None:
            raise ValueError(
                "checkpoint_dir is required when start_stage > 1 "
                "(must point at the directory written by the previous job)"
            )
        if end_stage < 9 and checkpoint_dir is None:
            raise ValueError(
                "checkpoint_dir is required when end_stage < 9 "
                "(downstream jobs need this checkpoint to resume)"
            )
        ckpt_dir: Optional[Path] = None
        if checkpoint_dir is not None:
            ckpt_dir = Path(checkpoint_dir)
            ckpt_dir.mkdir(parents=True, exist_ok=True)

        out = Path(output_dir)
        self._output_dir = out
        out.mkdir(parents=True, exist_ok=True)
        (out / "audit").mkdir(exist_ok=True)

        state = _PipelineState(output_dir)
        if state.state["start_time"] is None:
            state.state["start_time"] = time.strftime("%Y-%m-%dT%H:%M:%S")
            state._save()

        logger.info("=" * 60)
        logger.info("[PIPELINE] Phase 0: producing training-ready artifacts")
        logger.info(
            "[PIPELINE] Task: %s  Output: %s  Stages: %d..%d  Checkpoint: %s",
            self.config.task_name, output_dir,
            start_stage, end_stage,
            checkpoint_dir if checkpoint_dir else "<none>",
        )
        logger.info("=" * 60)

        # --------------------------------------------------------------
        # State variables that flow between stages. They are populated
        # either by stage execution (start_stage <= N) or by the
        # checkpoint loader (start_stage > N).
        # --------------------------------------------------------------
        df = None  # main DataFrame (post-Stage-2 imputed; post-Stage-3 with merged generators)
        raw_data: Dict[str, Any] = {}  # adapter dict — only Stage 8 needs this
        feature_pipeline = None
        feature_schema: Dict[str, Any] = {}
        adapter = None

        # Config-derived state used after Stage 2; pre-compute so the
        # checkpoint-loader path also has them.
        raw_cfg = self._config_to_dict()
        preproc_cfg = raw_cfg.get("preprocessing", {})
        id_cols = set(self.config.features.id_cols or [])
        date_cols_cfg = preproc_cfg.get("date_cols", [])
        if isinstance(date_cols_cfg, str):
            date_cols_cfg = [date_cols_cfg]
        date_cols = set(date_cols_cfg)

        # --------------------------------------------------------------
        # Checkpoint load path: start_stage > 1
        # --------------------------------------------------------------
        if start_stage >= 4:
            # Job C (Stages 4-9): load post-stage-3 metadata only.
            # The wide matrix stays on disk and is mounted as a DuckDB
            # view inside the Stage-4-9 prep block below.
            feature_pipeline, feature_schema, _meta = (
                self._load_checkpoint_post_stage3(ckpt_dir)
            )
            results["resumed_from"] = str(ckpt_dir / "post_stage3")
            logger.info(
                "[Stage 1-3 SKIPPED] Resumed from post_stage3 metadata: "
                "feature_pipeline_groups=%d, feature_schema_keys=%d",
                len(feature_pipeline) if feature_pipeline is not None else 0,
                len(feature_schema),
            )
        elif start_stage == 3:
            # Job B (Stage 3): load post-stage-2 checkpoint — has df only.
            df, _meta = self._load_checkpoint_post_stage2(ckpt_dir)
            results["resumed_from"] = str(ckpt_dir / "post_stage2")
            logger.info(
                "[Stage 1-2 SKIPPED] Resumed from post_stage2 checkpoint: df=%s",
                getattr(df, "shape", "?"),
            )
            # Stage 3 SQL-native generators (lag tensor / rolling stats /
            # topN multihot) read the LIST sequence columns directly out
            # of the DuckDB ctx the adapter sets up. The post_stage2
            # checkpoint only carries the imputed scalar table, so we
            # rebuild the ctx by re-running adapter.load_raw() — Stage 2's
            # modifications are scalar-only, so the LIST cols are
            # identical to the raw ones the generator pool expects.
            adapter = self._build_adapter()
            raw_data = adapter.load_raw()
            from .adapter import DuckDBAdapterContext as _Ctx
            if isinstance(raw_data, _Ctx):
                self._adapter_ctx = raw_data
                # raw_data dict keeps the legacy ``main`` key so the
                # Stage 3 _engineer_features signature is unchanged.
                raw_data = {"main": self._scalar_df_from_ctx(raw_data)}
            else:
                self._adapter_ctx = None

        # ==============================================================
        # Stage 1: Load raw data via adapter
        # ==============================================================
        if start_stage <= 1:
            stage_start = time.time()
            logger.info("[Stage 1] Loading raw data...")

            adapter = self._build_adapter()
            raw_data = adapter.load_raw()

            # CLAUDE.md §3.3 path. Modern adapters return a
            # ``DuckDBAdapterContext`` with the raw data as a DuckDB
            # table on a shared connection — no pandas materialisation.
            # We pull only the SCALAR (and short-LIST) columns into
            # pandas so the legacy stage 2-9 code path keeps working,
            # and leave the long LIST columns inside DuckDB for SQL-
            # native generators to consume in Stage 3.
            from .adapter import DuckDBAdapterContext as _Ctx
            if isinstance(raw_data, _Ctx):
                self._adapter_ctx = raw_data
                df = self._scalar_df_from_ctx(raw_data)
                raw_data = {"main": df}
            else:
                self._adapter_ctx = None
                df = raw_data["main"]

            results["stage1_load"] = {
                "rows": len(df),
                "cols": len(df.columns),
                "adapter_metadata": adapter.metadata.__dict__
                if hasattr(adapter.metadata, "__dict__")
                else str(adapter.metadata),
                "duckdb_native": self._adapter_ctx is not None,
                "list_cols_in_duckdb": (
                    self._adapter_ctx.metadata.extra.get("list_cols_lazy", [])
                    if self._adapter_ctx is not None
                    and getattr(self._adapter_ctx.metadata, "extra", None)
                    else []
                ),
                "time_seconds": round(time.time() - stage_start, 2),
            }
            state.mark_complete(
                "stage1_load", {"rows": len(df), "cols": len(df.columns)}
            )
            logger.info(
                "[Stage 1] Loaded %d rows x %d cols in %.2fs (duckdb_native=%s)",
                len(df), len(df.columns), time.time() - stage_start,
                self._adapter_ctx is not None,
            )

        # ==============================================================
        # Stage 1.5: Temporal preparation (sequence truncation)
        #
        # Tied to the Stage-1 guard since 1.5 reads df produced by
        # Stage 1; the post-Stage-2 checkpoint is the next persistence
        # point.
        # ==============================================================
        if start_stage <= 1:
            stage_start = time.time()
            df = self._prepare_temporal(df)
            results["stage1_5_temporal_prep"] = {
                "time_seconds": round(time.time() - stage_start, 2),
            }

        # ==============================================================
        # Stage 2: Preprocessing
        # ==============================================================
        if start_stage <= 1:
            stage_start = time.time()
            logger.info("[Stage 2] Preprocessing...")

            # 2a) Sentinel normalization (config-driven)
            sentinel_rules = preproc_cfg.get("sentinels", {})
            sentinel_applied = 0
            for col, sentinel_val in sentinel_rules.items():
                if col in df.columns:
                    mask = df[col] == sentinel_val
                    cnt = int(mask.sum())
                    if cnt > 0:
                        df.loc[mask, col] = np.nan
                        sentinel_applied += cnt
            if sentinel_applied:
                logger.info(
                    "[Stage 2] Sentinel normalization: replaced %d values",
                    sentinel_applied,
                )

            # 2b) Categorical encoding
            if not id_cols:
                logger.warning(
                    "[Stage 2] No id_cols specified in config.features.id_cols — "
                    "no columns will be excluded as ID columns"
                )
            exclude_encode = id_cols | date_cols

            cat_cols = df.select_dtypes(include=["object", "string", "category"]).columns
            encoded_cols: List[str] = []
            for col in cat_cols:
                if col not in exclude_encode:
                    le = LabelEncoder()
                    df[col] = le.fit_transform(df[col].astype(str))
                    encoded_cols.append(col)
            if encoded_cols:
                logger.info(
                    "[Stage 2] Categorical encoding: %d cols (%s...)",
                    len(encoded_cols), encoded_cols[:5],
                )

            # 2c) Missing value imputation (config-driven) -- single DuckDB pass
            impute_rules = preproc_cfg.get("imputation", {})
            numeric_cols_all = list(df.select_dtypes(include=[np.number]).columns)

            import duckdb as _ddb_stage2
            _con2 = _ddb_stage2.connect()
            try:
                _con2.register("_df_impute", df)
                col_exprs: List[str] = []
                handled_cols = set()
                for col, strategy in impute_rules.items():
                    if col not in df.columns:
                        continue
                    handled_cols.add(col)
                    qcol = f'"{col}"'
                    if strategy == "median":
                        col_exprs.append(
                            f'COALESCE({qcol}, (SELECT MEDIAN({qcol}) FROM _df_impute)) AS {qcol}')
                    elif strategy == "mean":
                        col_exprs.append(
                            f'COALESCE({qcol}, (SELECT AVG({qcol}) FROM _df_impute)) AS {qcol}')
                    elif strategy == "zero":
                        col_exprs.append(f'COALESCE({qcol}, 0) AS {qcol}')
                    elif strategy == "mode":
                        col_exprs.append(
                            f'COALESCE({qcol}, (SELECT MODE({qcol}) FROM _df_impute)) AS {qcol}')
                    else:
                        col_exprs.append(qcol)

                for col in numeric_cols_all:
                    if col not in handled_cols:
                        col_exprs.append(f'COALESCE("{col}", 0) AS "{col}"')
                        handled_cols.add(col)

                for col in df.columns:
                    if col not in handled_cols:
                        col_exprs.append(f'"{col}"')

                sql = f"SELECT {', '.join(col_exprs)} FROM _df_impute"
                _arrow_after_impute = _con2.execute(sql).fetch_arrow_table()
            finally:
                _con2.close()
            import pyarrow as _pa
            df = _arrow_after_impute  # type: ignore[assignment]

            results["stage2_preprocessing"] = {
                "sentinel_values_replaced": sentinel_applied,
                "categorical_encoded": encoded_cols,
                "imputation_rules_applied": list(impute_rules.keys()),
                "time_seconds": round(time.time() - stage_start, 2),
            }
            state.mark_complete("stage2_preprocessing")
            logger.info(
                "[Stage 2] Preprocessing complete in %.2fs",
                time.time() - stage_start,
            )

        # ==============================================================
        # Boundary: Stage 2 -> Stage 3 checkpoint
        # ==============================================================
        if end_stage <= 2:
            self._save_checkpoint_post_stage2(
                ckpt_dir, df,
                meta={
                    "id_cols": sorted(id_cols),
                    "date_cols": sorted(date_cols),
                },
            )
            results["checkpoint_saved"] = str(ckpt_dir / "post_stage2")
            results["total_time_seconds"] = round(
                time.time() - pipeline_start, 2
            )
            logger.info(
                "[Phase 0 partial] Stages %d..%d complete in %.1fs; "
                "checkpoint at %s",
                start_stage, end_stage,
                results["total_time_seconds"], ckpt_dir / "post_stage2",
            )
            return results

        # ==============================================================
        # Stage 3: Feature engineering via FeatureGroupPipeline
        # ==============================================================
        if start_stage <= 3:
            stage_start = time.time()
            logger.info("[Stage 3] Feature engineering...")

            # _engineer_features expects a pandas DataFrame.  Convert
            # from Arrow here — single materialisation point Stage 2->3.
            import pyarrow as _pa
            if isinstance(df, _pa.Table):
                df_for_eng = df.to_pandas()
            else:
                df_for_eng = df

            feature_pipeline, df_generated = self._engineer_features(
                df_for_eng, raw_data,
            )
            feature_schema = feature_pipeline.get_ple_input_metadata()

            _arrow_main: _pa.Table = (
                df if isinstance(df, _pa.Table) else _pa.Table.from_pandas(df_for_eng)
            )

            if df_generated is not None and len(df_generated.columns) > 0:
                nonzero_ratio = (df_generated != 0).mean().mean()
                zero_cols = [
                    c for c in df_generated.columns
                    if (df_generated[c] == 0).all()
                ]
                logger.info(
                    "[Stage 3] Generated features: nonzero_ratio=%.2f%%, "
                    "zero_variance_cols=%d",
                    nonzero_ratio * 100, len(zero_cols),
                )
                if zero_cols:
                    logger.warning(
                        "[Stage 3] All-zero generated columns: %s",
                        zero_cols[:10],
                    )

                existing_cols = set(_arrow_main.schema.names)
                new_cols = [c for c in df_generated.columns if c not in existing_cols]
                if new_cols:
                    _df_gen = df_generated[new_cols].reset_index(drop=True)
                    for _col in new_cols:
                        _arrow_main = _arrow_main.append_column(
                            _col,
                            _pa.array(
                                _df_gen[_col].to_numpy(dtype=float, na_value=float("nan"))
                            ),
                        )
                    logger.info(
                        "[Stage 3] Merged %d generated features into main table",
                        len(new_cols),
                    )

            if self._adapter_ctx is not None:
                con = self._adapter_ctx.con
                try:
                    con.unregister("_post_stage3")
                except Exception:
                    pass
                con.register("_arrow_post_stage3", _arrow_main)
                con.execute(
                    "CREATE OR REPLACE TABLE _post_stage3 AS "
                    "SELECT * FROM _arrow_post_stage3"
                )
                try:
                    con.unregister("_arrow_post_stage3")
                except Exception:
                    pass
                self._post_stage3_table = "_post_stage3"
                try:
                    df = _arrow_main.to_pandas(types_mapper=pd.ArrowDtype)
                except TypeError:
                    df = _arrow_main.to_pandas()
                del _arrow_main
            else:
                df = _arrow_main.to_pandas()
                self._post_stage3_table = None

            results["stage3_features"] = {
                "num_groups": len(feature_pipeline),
                "total_dim": feature_pipeline.total_dim,
                "generated_cols": len(df_generated.columns) if df_generated is not None else 0,
                "df_shape_after": list(df.shape),
                "time_seconds": round(time.time() - stage_start, 2),
            }
            state.mark_complete(
                "stage3_features",
                {"total_dim": feature_pipeline.total_dim},
            )
            logger.info(
                "[Stage 3] Feature engineering complete in %.2fs: "
                "%d groups, total_dim=%d",
                time.time() - stage_start,
                len(feature_pipeline), feature_pipeline.total_dim,
            )

        # ==============================================================
        # Boundary: Stage 3 -> Stage 4 checkpoint
        # ==============================================================
        if end_stage <= 3:
            self._save_checkpoint_post_stage3(
                ckpt_dir, df, feature_pipeline, feature_schema,
                meta={
                    "id_cols": sorted(id_cols),
                    "date_cols": sorted(date_cols),
                },
            )
            results["checkpoint_saved"] = str(ckpt_dir / "post_stage3")
            results["total_time_seconds"] = round(
                time.time() - pipeline_start, 2
            )
            logger.info(
                "[Phase 0 partial] Stages %d..%d complete in %.1fs; "
                "checkpoint at %s",
                start_stage, end_stage,
                results["total_time_seconds"], ckpt_dir / "post_stage3",
            )
            return results

        # ==============================================================
        # Stage 4-9 prep (resumed-from-checkpoint path)
        #
        # Memory rule: never materialise the post-Stage-3 matrix in
        # pandas more than absolutely necessary. The data lives in
        # ``post_stage3/main.parquet`` on disk; we expose it as a
        # DuckDB *view* on the adapter ctx (zero materialisation) and
        # build a slim ``df_lite`` (only id / label-source / sequence
        # columns) for the few legacy stage-4/5/7 helpers that still
        # take pandas. The wide feature matrix never enters python.
        # ==============================================================
        if start_stage >= 4:
            adapter = self._build_adapter()
            raw_data = adapter.load_raw()
            from .adapter import DuckDBAdapterContext as _Ctx
            if isinstance(raw_data, _Ctx):
                self._adapter_ctx = raw_data
                con = self._adapter_ctx.con
                ckpt_parquet = ckpt_dir / "post_stage3" / "main.parquet"
                if not ckpt_parquet.exists():
                    raise FileNotFoundError(
                        f"post_stage3 checkpoint missing: {ckpt_parquet}"
                    )
                # Build a SELECT projection that casts every DECIMAL
                # column to DOUBLE — same fix as the loader helper, but
                # staying purely SQL (no Arrow round-trip).
                schema_info = con.execute(
                    f"DESCRIBE SELECT * FROM read_parquet("
                    f"'{ckpt_parquet}') LIMIT 0"
                ).fetchall()
                proj_parts: List[str] = []
                for row in schema_info:
                    col_name, col_type = row[0], (row[1] or "").upper()
                    if col_type.startswith("DECIMAL"):
                        proj_parts.append(
                            f'CAST("{col_name}" AS DOUBLE) AS "{col_name}"'
                        )
                    else:
                        proj_parts.append(f'"{col_name}"')
                con.execute(
                    f"CREATE OR REPLACE VIEW _post_stage3 AS\n"
                    f"  SELECT {', '.join(proj_parts)}\n"
                    f"  FROM read_parquet('{ckpt_parquet}')"
                )
                self._post_stage3_table = "_post_stage3"

                # Build df_lite: only the non-feature columns Stage 4
                # (label deriver) and Stage 7 (seq leakage check) still
                # need on the pandas side. The 1100+ feature cols stay
                # in DuckDB.
                feature_col_set: set = set()
                if feature_pipeline is not None:
                    feature_col_set = set(
                        getattr(feature_pipeline, "_output_columns", None)
                        or feature_schema.get("output_columns", [])
                    )
                else:
                    feature_col_set = set(
                        feature_schema.get("output_columns", [])
                    )
                all_post3_cols = [
                    r[0] for r in con.execute(
                        "DESCRIBE _post_stage3"
                    ).fetchall()
                ]
                df_lite_cols = [
                    c for c in all_post3_cols if c not in feature_col_set
                ]
                if df_lite_cols:
                    proj_lite = ", ".join(f'"{c}"' for c in df_lite_cols)
                    df = con.execute(
                        f"SELECT {proj_lite} FROM _post_stage3"
                    ).df()
                else:
                    df = con.execute(
                        "SELECT * FROM _post_stage3 LIMIT 0"
                    ).df()
                logger.info(
                    "[Stage 4-9 prep] post_stage3 mounted as VIEW (parquet); "
                    "df_lite=%d cols (%d feature cols stay in DuckDB)",
                    len(df_lite_cols), len(feature_col_set),
                )
                # raw_data["main"] kept slim so Stage 8 SequenceBuilder
                # can fall back to it if needed.
                raw_data = {"main": self._scalar_df_from_ctx(raw_data)}
            else:
                self._adapter_ctx = None
                self._post_stage3_table = None

        # ==============================================================
        # Stage 4: Label derivation via LabelDeriver
        # ==============================================================
        stage_start = time.time()
        logger.info("[Stage 4] Deriving labels...")

        labels_df = self._derive_labels(df)

        # Build label schema
        label_schema: Dict[str, Any] = {"tasks": []}
        for task in self.config.tasks:
            task_info: Dict[str, Any] = {
                "name": task.name,
                "type": task.type,
                "label_col": task.label_col,
                "loss": task.loss,
                "loss_weight": task.loss_weight,
            }
            if task.label_col in labels_df.columns:
                col_data = labels_df[task.label_col]
                if task.type in ("binary", "multiclass"):
                    task_info["num_classes"] = int(col_data.nunique())
                    task_info["class_distribution"] = {
                        str(k): int(v)
                        for k, v in col_data.value_counts().to_dict().items()
                    }
                elif task.type == "regression":
                    task_info["num_classes"] = 1
                    task_info["stats"] = {
                        "mean": float(col_data.mean()),
                        "std": float(col_data.std()),
                        "min": float(col_data.min()),
                        "max": float(col_data.max()),
                    }
                else:
                    task_info["num_classes"] = task.num_classes
            label_schema["tasks"].append(task_info)

        results["stage4_labels"] = {
            "label_columns": list(labels_df.columns),
            "rows": len(labels_df),
            "time_seconds": round(time.time() - stage_start, 2),
        }
        state.mark_complete("stage4_labels", {"label_columns": list(labels_df.columns)})
        logger.info(
            "[Stage 4] Derived %d label columns in %.2fs",
            len(labels_df.columns), time.time() - stage_start,
        )

        # ==============================================================
        # Stage 5: Temporal split (train / val / test indices)
        # ==============================================================
        stage_start = time.time()
        logger.info("[Stage 5] Splitting data...")

        train_idx, val_idx, test_idx = self._compute_split_indices(df)

        split_info = {
            "train_size": len(train_idx),
            "val_size": len(val_idx),
            "test_size": len(test_idx),
            "total": len(df),
            "time_seconds": round(time.time() - stage_start, 2),
        }
        results["stage5_split"] = split_info
        state.mark_complete("stage5_split", split_info)
        logger.info(
            "[Stage 5] Split complete in %.2fs: train=%d, val=%d, test=%d",
            time.time() - stage_start, len(train_idx), len(val_idx), len(test_idx),
        )

        # ==============================================================
        # Stage 6: Normalization (3-stage, train-only fit via FeatureNormalizer)
        # ==============================================================
        stage_start = time.time()
        logger.info("[Stage 6] Normalizing features (train-only fit)...")

        from core.pipeline.normalizer import FeatureNormalizer

        # Determine feature columns. When the SQL-native path is on
        # (post-Stage-3 lives as ``_post_stage3`` view on the adapter
        # ctx), read the schema straight from DuckDB so we don't depend
        # on df_lite / fat df having every feature column. This keeps
        # the column list authoritative across single-job and resumed
        # paths.
        label_col_set = set(labels_df.columns)
        seq_col_set: set
        non_feature_cols: set
        if (
            self._adapter_ctx is not None
            and getattr(self, "_post_stage3_table", None) is not None
        ):
            con = self._adapter_ctx.con
            descr_post3 = con.execute(
                f"DESCRIBE {self._post_stage3_table}"
            ).fetchall()
            post3_cols = [r[0] for r in descr_post3]
            post3_types = {r[0]: (r[1] or "").upper() for r in descr_post3}
            seq_col_set = {
                c for c in post3_cols
                if c.startswith("seq_") or c in self.config.features.sequence
            }
            non_feature_cols = label_col_set | id_cols | date_cols | seq_col_set
            numeric_keywords = (
                "INT", "BIGINT", "SMALLINT", "TINYINT", "HUGEINT",
                "DECIMAL", "DOUBLE", "FLOAT", "REAL",
            )
            feature_cols = [
                c for c in post3_cols
                if c not in non_feature_cols
                and any(k in post3_types.get(c, "") for k in numeric_keywords)
                # exclude LIST-typed sequence columns whose name doesn't
                # match the seq_ prefix (e.g., txn_amount_seq)
                and "[" not in post3_types.get(c, "")
            ]
            train_df = None  # not used on SQL-native path
        else:
            seq_col_set = {
                c for c in df.columns
                if c.startswith("seq_") or c in self.config.features.sequence
            }
            non_feature_cols = label_col_set | id_cols | date_cols | seq_col_set
            feature_cols = [
                c for c in df.select_dtypes(include=[np.number]).columns
                if c not in non_feature_cols
            ]
            train_df = df.iloc[train_idx]

        # Fit normalizer on TRAIN split only, transform ALL data.
        # Read normalizer config from pipeline.yaml so suffix/prefix
        # exclusion patterns (categorical_id_*, probability_*) are
        # config-driven per §1.1.
        normalizer_cfg: Dict[str, Any] = {}
        raw_features = getattr(self.config, "features", None)
        if raw_features is not None:
            normalizer_cfg = getattr(raw_features, "normalizer", {}) or {}
        if not normalizer_cfg and isinstance(getattr(self.config, "_raw", None), dict):
            normalizer_cfg = (self.config._raw.get("features", {}) or {}).get("normalizer", {}) or {}
        normalizer = FeatureNormalizer(config=normalizer_cfg)

        # CLAUDE.md §3.3: when the adapter exposed a DuckDB context we
        # do the entire fit + transform inside DuckDB, never letting the
        # 1M × ~1285D continuous matrix touch pandas. The fit aggregates
        # mean / std / skew / kurt in a single SELECT; the transform is
        # a SELECT-time projection ((col - mean) / std, log1p, etc.) that
        # produces a new table the runner reads only the slices it needs
        # from.  This avoids the ~10 GB pandas allocation that has been
        # OOMing Phase 0 even on the 64 GB instance.
        sql_native = (
            self._adapter_ctx is not None
            and getattr(self, "_post_stage3_table", None) is not None
        )
        norm_output_table: Optional[str] = None
        if sql_native:
            con = self._adapter_ctx.con
            src_table = self._post_stage3_table
            n_rows = con.execute(
                f"SELECT COUNT(*) FROM {src_table}"
            ).fetchone()[0]
            train_mask = np.zeros(n_rows, dtype=bool)
            train_mask[train_idx] = True
            con.execute("DROP TABLE IF EXISTS _split_mask")
            con.execute(
                "CREATE TABLE _split_mask AS "
                "SELECT row_number() OVER () AS _rn, "
                "       UNNEST(?) AS _is_train",
                [train_mask.tolist()],
            )
            con.execute(
                f"CREATE OR REPLACE VIEW _norm_with_split AS "
                f"SELECT p.*, m._is_train "
                f"FROM {src_table} p POSITIONAL JOIN _split_mask m"
            )
            normalizer.fit_sql(
                con, "_norm_with_split", feature_cols,
                train_predicate="_is_train",
            )
            normalizer.transform_sql(
                con, src_table, feature_cols,
                output_table="_norm_output",
            )
            con.execute("DROP TABLE IF EXISTS _split_mask")
            descr = con.execute("DESCRIBE _norm_output").fetchall()
            feature_cols = [r[0] for r in descr]
            norm_output_table = "_norm_output"
            logger.info(
                "[Stage 6] DuckDB-native fit+transform: %d post-norm cols "
                "stay in DuckDB table '%s' (read straight from %s view, "
                "no pandas materialise)",
                len(feature_cols), norm_output_table, src_table,
            )
            self._post_norm_table = norm_output_table
        else:
            normalizer.fit(train_df, feature_cols)
            df_normed = normalizer.transform(df, feature_cols)
            for col in df_normed.columns:
                df[col] = df_normed[col].values
            feature_cols = list(df_normed.columns)
            self._post_norm_table = None

        # Extract column classifications from fitted normalizer
        continuous_cols = normalizer.continuous_cols
        binary_cols = normalizer.binary_cols
        log_cols_created = [f"{c}_log" for c in normalizer.power_law_cols]

        # Build scaler_params for backward-compatible scaler_params.json
        scaler_params: Dict[str, Any] = {}
        if normalizer._mean is not None:
            scaler_params["scaler_type"] = "StandardScaler"
            scaler_params["columns"] = continuous_cols
            scaler_params["mean"] = normalizer._mean.tolist()
            scaler_params["scale"] = normalizer._std.tolist()
            scaler_params["var"] = (normalizer._std ** 2).tolist()

        scaler_params["power_law_log_cols"] = log_cols_created
        scaler_params["binary_cols"] = binary_cols
        scaler_params["continuous_cols"] = continuous_cols

        # Save normalizer for inference reuse
        normalizer.save(str(out / "normalizer"))

        results["stage6_normalization"] = {
            "continuous_cols": len(continuous_cols),
            "binary_cols": len(binary_cols),
            "power_law_log_cols": len(log_cols_created),
            "total_feature_cols": len(feature_cols),
            "time_seconds": round(time.time() - stage_start, 2),
        }
        # -- Rebuild feature_group_ranges against post-normalization column order.
        # CLAUDE.md §1.7: the 3-stage normalizer reorders / appends `{col}_log`
        # copies, which invalidates the (start, end) ranges computed in Stage 3
        # against the pre-normalization order. We re-map each group's columns
        # (plus its `_log` offspring) to the new positions and emit the longest
        # contiguous block per group, logging a warning when a group ends up
        # non-contiguous (typical cause: `_log` copies stranded at the tail).
        try:
            rebuilt_ranges = _rebuild_group_ranges_post_normalization(
                feature_pipeline=feature_pipeline,
                feature_cols=feature_cols,
                log_cols_created=log_cols_created,
            )
            if rebuilt_ranges:
                feature_schema["feature_group_ranges"] = rebuilt_ranges
                logger.info(
                    "[Stage 6] feature_group_ranges rebuilt post-normalization: %d groups",
                    len(rebuilt_ranges),
                )
        except Exception:
            logger.exception(
                "[Stage 6] feature_group_ranges rebuild failed — leaving Stage 3 "
                "pre-normalization ranges in schema (may misalign expert routing)"
            )

        state.mark_complete("stage6_normalization")
        logger.info(
            "[Stage 6] Normalization complete in %.2fs: %d continuous (scaled), "
            "%d binary (passthrough), %d log1p copies (unscaled)",
            time.time() - stage_start,
            len(continuous_cols), len(binary_cols), len(log_cols_created),
        )

        # ==============================================================
        # Stage 7: Leakage validation
        # ==============================================================
        stage_start = time.time()
        logger.info("[Stage 7] Validating for data leakage...")

        # When the SQL-native path is on, Stage 6 left the post-norm
        # matrix in ``_norm_output`` and ``_post_stage3`` is still a
        # cheap view. ``_validate_leakage`` only needs the *names* of
        # the feature columns plus labels_df — actual values come from
        # the DuckDB table. Pass an empty pandas frame whose columns
        # match feature_cols so the legacy signature is satisfied
        # without re-materialising the 1168 × 941K matrix.
        if (
            self._adapter_ctx is not None
            and getattr(self, "_post_stage3_table", None) is not None
        ):
            df_features_only = pd.DataFrame(columns=feature_cols)
        else:
            df_features_only = df[feature_cols]
        leakage_result = self._validate_leakage(df_features_only, labels_df, df)

        results["stage7_leakage"] = {
            "passed": leakage_result.passed,
            "warnings": leakage_result.warnings
            if hasattr(leakage_result, "warnings")
            else [],
            "time_seconds": round(time.time() - stage_start, 2),
        }
        state.mark_complete("stage7_leakage", {"passed": leakage_result.passed})
        logger.info(
            "[Stage 7] Leakage validation %s in %.2fs (%d warnings)",
            "PASSED" if leakage_result.passed else "FAILED",
            time.time() - stage_start,
            len(leakage_result.warnings) if hasattr(leakage_result, "warnings") else 0,
        )

        # Save leakage report
        leakage_path = out / "audit" / "leakage_report.json"
        with open(leakage_path, "w") as f:
            json.dump(
                {
                    "passed": leakage_result.passed,
                    "warnings": leakage_result.warnings
                    if hasattr(leakage_result, "warnings")
                    else [],
                    "details": leakage_result.details
                    if hasattr(leakage_result, "details")
                    else {},
                },
                f, indent=2, default=str,
            )

        # ==============================================================
        # Stage 8: Sequence building
        #
        # CLAUDE.md §3.3 / memory-efficiency: build numpy tensors, write
        # them to disk *immediately*, and replace the in-memory arrays
        # with metadata stubs. Without this, the (941K, 50, 27) product
        # tensor (~5 GB float32) plus the (941K, 50, 4) txn tensor
        # (~0.75 GB) sit alongside Stage 9's features.parquet COPY work
        # set and tip the 64 GB instance into OOM.
        # ==============================================================
        stage_start = time.time()
        logger.info("[Stage 8] Building sequences...")

        sequences = self._build_sequences(raw_data)
        sequence_paths: Dict[str, str] = {}
        sequence_meta: Dict[str, Dict[str, Any]] = {}
        if sequences:
            for name, arr in sequences.items():
                npy_path = out / f"{name}.npy"
                np.save(str(npy_path), arr)
                sequence_paths[name] = str(npy_path)
                sequence_meta[name] = {
                    "path": str(npy_path),
                    "shape": list(arr.shape),
                    "dtype": str(arr.dtype),
                }
                logger.info(
                    "[Stage 8] saved %s eagerly to %s shape=%s",
                    name, npy_path, arr.shape,
                )
            # Canonical sequences.npy + seq_lengths.npy when there is
            # exactly one tensor (matches the legacy contract).
            if len(sequences) == 1:
                only_arr = next(iter(sequences.values()))
                np.save(str(out / "sequences.npy"), only_arr)
                if only_arr.ndim == 3:
                    nonzero_mask = np.any(only_arr != 0, axis=2)
                    seq_lengths = nonzero_mask.sum(axis=1).astype(np.int32)
                elif only_arr.ndim == 2:
                    seq_lengths = (only_arr != 0).sum(axis=1).astype(np.int32)
                else:
                    seq_lengths = np.full(
                        len(only_arr),
                        only_arr.shape[1] if only_arr.ndim >= 2 else 1,
                        dtype=np.int32,
                    )
                np.save(str(out / "seq_lengths.npy"), seq_lengths)
                sequence_meta["seq_lengths"] = {
                    "path": str(out / "seq_lengths.npy"),
                    "shape": list(seq_lengths.shape),
                    "dtype": str(seq_lengths.dtype),
                }
                del seq_lengths

            # Free the in-memory tensors before Stage 9 starts. The
            # caller has the npy paths in ``sequence_paths`` /
            # ``sequence_meta`` for the manifest.
            sequences.clear()
            del sequences
            sequences = None  # type: ignore[assignment]
            import gc as _gc
            _gc.collect()
            logger.info(
                "[Stage 8] in-memory sequence arrays freed; %d files on disk",
                len(sequence_paths),
            )

        results["stage8_sequences"] = {
            "has_sequences": bool(sequence_paths),
            "keys": list(sequence_paths.keys()),
            "paths": sequence_paths,
            "time_seconds": round(time.time() - stage_start, 2),
        }
        state.mark_complete("stage8_sequences", {
            "has_sequences": bool(sequence_paths),
        })
        logger.info(
            "[Stage 8] Sequences: %s in %.2fs",
            f"{len(sequence_paths)} tensors" if sequence_paths else "none",
            time.time() - stage_start,
        )

        # ==============================================================
        # Stage 9: Save all artifacts
        # ==============================================================
        stage_start = time.time()
        logger.info("[Stage 9] Saving artifacts to %s ...", out)

        # 9a) features.parquet -- when Stage 6 ran SQL-native, the
        #     post-normalize matrix is already on the adapter ctx as
        #     ``_norm_output``. COPY straight from there to parquet so
        #     we never re-materialise the 1168 × 941K matrix in pandas
        #     (CLAUDE.md §3.3 + the OOM mode that this fixes).
        features_path = out / "features.parquet"
        post_norm_table = getattr(self, "_post_norm_table", None)
        if post_norm_table and self._adapter_ctx is not None:
            con = self._adapter_ctx.con
            con.execute(
                f"COPY {post_norm_table} TO '{features_path}' "
                f"(FORMAT PARQUET)"
            )
            n_rows = con.execute(
                f"SELECT COUNT(*) FROM {post_norm_table}"
            ).fetchone()[0]
            n_cols = len(con.execute(
                f"DESCRIBE {post_norm_table}"
            ).fetchall())
            logger.info(
                "[Stage 9] features.parquet (DuckDB-native): %d rows x %d cols",
                n_rows, n_cols,
            )
            # Free the DuckDB table after parquet write to release ~9 GB
            # before the labels write touches the same connection.
            con.execute(f"DROP TABLE IF EXISTS {post_norm_table}")
            self._post_norm_table = None
        else:
            # Pandas fallback: write feature_cols slice via DuckDB COPY.
            features_out = df[feature_cols].reset_index(drop=True)
            import duckdb as _ddb_stage9
            _con9 = _ddb_stage9.connect()
            try:
                _con9.register("_features_out", features_out)
                _con9.execute(
                    f"COPY _features_out TO '{features_path}' "
                    f"(FORMAT PARQUET)"
                )
            finally:
                _con9.close()
            logger.info(
                "[Stage 9] features.parquet (pandas fallback): %d rows x %d cols",
                *features_out.shape,
            )
            del features_out

        # 9b) labels.parquet (DuckDB COPY) — small (≤ 30 cols × 941K)
        labels_path = out / "labels.parquet"
        import duckdb as _ddb_stage9b
        _con9b = _ddb_stage9b.connect()
        try:
            _con9b.register("_labels_out", labels_df.reset_index(drop=True))
            _con9b.execute(
                f"COPY _labels_out TO '{labels_path}' (FORMAT PARQUET)"
            )
        finally:
            _con9b.close()
        logger.info(
            "[Stage 9] labels.parquet: %d rows x %d cols",
            len(labels_df), len(labels_df.columns),
        )

        # 9c) sequence files were written eagerly in Stage 8. The
        #     metadata stubs in ``sequence_meta`` describe them so the
        #     manifest can pick them up without re-touching the (now
        #     freed) numpy arrays.

        # 9d) scaler_params.json
        scaler_path = out / "scaler_params.json"
        with open(scaler_path, "w") as f:
            json.dump(scaler_params, f, indent=2, default=str)
        logger.info("[Stage 9] scaler_params.json written")

        # 9e) feature_schema.json
        feature_schema["feature_columns"] = feature_cols
        feature_schema["binary_columns"] = binary_cols
        feature_schema["continuous_columns"] = continuous_cols
        feature_schema["power_law_log_columns"] = log_cols_created
        # Key aliases for train.py compatibility (reads "group_ranges" / "columns")
        feature_schema["group_ranges"] = feature_schema.get("feature_group_ranges", {})
        feature_schema["columns"] = feature_schema.get(
            "feature_columns", feature_schema.get("output_columns", [])
        )
        feature_schema_path = out / "feature_schema.json"
        with open(feature_schema_path, "w") as f:
            json.dump(feature_schema, f, indent=2, default=str)
        logger.info("[Stage 9] feature_schema.json written")

        # 9f) label_schema.json — include model/task config for train.py
        label_schema["task_groups"] = [
            {
                "name": tg.name,
                "tasks": tg.tasks,
                "adatt_intra_strength": getattr(tg, "adatt_intra_strength", 0.8),
                "adatt_inter_strength": getattr(tg, "adatt_inter_strength", 0.3),
            }
            for tg in self.config.task_groups
        ]
        label_schema["task_group_map"] = {
            t: tg.name for tg in self.config.task_groups for t in tg.tasks
        }
        # Pass through task relationships and model config
        raw_config = self._config_to_dict()
        label_schema["task_relationships"] = raw_config.get("task_relationships", [])
        label_schema["logit_transfer_strength"] = raw_config.get("logit_transfer_strength", 0.5)
        label_schema["model"] = raw_config.get("model", {})
        label_schema["adatt"] = raw_config.get("adatt", {})
        label_schema["training"] = raw_config.get("training", {})

        label_schema_path = out / "label_schema.json"
        with open(label_schema_path, "w") as f:
            json.dump(label_schema, f, indent=2, default=str)
        logger.info("[Stage 9] label_schema.json written")

        # 9g) split_indices.json
        split_indices = {
            "train": [int(i) for i in train_idx],
            "val": [int(i) for i in val_idx],
            "test": [int(i) for i in test_idx],
        }
        split_path = out / "split_indices.json"
        with open(split_path, "w") as f:
            json.dump(split_indices, f)
        logger.info("[Stage 9] split_indices.json written")

        # 9h) Save feature pipeline for later reuse
        try:
            feature_pipeline.save(str(out / "feature_pipeline"))
            logger.info("[Stage 9] feature_pipeline saved")
        except Exception as e:
            logger.warning("[Stage 9] Could not save feature_pipeline: %s", e)

        # 9i) Pipeline manifest
        elapsed = time.time() - pipeline_start
        results["total_time_seconds"] = round(elapsed, 2)
        results["artifact_paths"] = {
            "features": str(features_path),
            "labels": str(labels_path),
            "scaler_params": str(scaler_path),
            "feature_schema": str(feature_schema_path),
            "label_schema": str(label_schema_path),
            "split_indices": str(split_path),
        }
        if sequence_paths:
            results["artifact_paths"]["sequences"] = dict(sequence_paths)

        manifest_path = out / "pipeline_manifest.json"
        with open(manifest_path, "w") as f:
            json.dump(results, f, indent=2, default=str)

        state.mark_complete("stage9_save")
        logger.info("[Stage 9] Artifacts saved in %.2fs", time.time() - stage_start)

        logger.info("=" * 60)
        logger.info(
            "[PIPELINE] Phase 0 complete in %.1fs.  Artifacts: %s",
            elapsed, output_dir,
        )
        logger.info("=" * 60)

        return results

    # ==================================================================
    # Full pipeline (Phase 0 + training + distillation + serving)
    # ==================================================================

    def run_full(self, output_dir: str = "outputs/") -> dict:
        """Run Phase 0 followed by training, analysis, distillation, and serving.

        This is the backward-compatible entry point that replaces the
        original ``run()`` method.  It delegates Phase 0 to :meth:`run`,
        then proceeds with stages 7-10 of the original pipeline.

        Args:
            output_dir: Directory for model artifacts.

        Returns:
            A result dict with metadata from all stages.
        """
        import numpy as np
        import pandas as pd

        results: Dict[str, Any] = {}
        pipeline_start = time.time()

        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)

        state = _PipelineState(output_dir)

        # ---- Phase 0 ------------------------------------------------
        phase0_dir = str(out)
        if not state.is_complete("stage9_save"):
            phase0_results = self.run(output_dir=phase0_dir)
            results.update(phase0_results)
        else:
            logger.info("[PIPELINE] Phase 0 already complete, loading artifacts...")
            manifest_path = out / "pipeline_manifest.json"
            if manifest_path.exists():
                with open(manifest_path) as f:
                    results.update(json.load(f))

        # Load Phase 0 artifacts for downstream stages
        df_features = pd.read_parquet(out / "features.parquet")
        df_labels = pd.read_parquet(out / "labels.parquet")

        with open(out / "split_indices.json") as f:
            split_indices = json.load(f)
        train_idx = split_indices["train"]

        with open(out / "feature_schema.json") as f:
            feature_schema = json.load(f)

        # Reload feature pipeline
        feature_pipeline = None
        try:
            from ..feature.group_pipeline import FeatureGroupPipeline
            fp_path = out / "feature_pipeline"
            if fp_path.exists():
                feature_pipeline = FeatureGroupPipeline.load(str(fp_path))
        except Exception as e:
            logger.warning("[PIPELINE] Could not reload feature pipeline: %s", e)

        # Load sequences if present
        sequences: Optional[Dict[str, np.ndarray]] = None
        seq_keys = results.get("stage8_sequences", {}).get("keys", [])
        if seq_keys:
            sequences = {}
            for name in seq_keys:
                npy_path = out / f"{name}.npy"
                if npy_path.exists():
                    sequences[name] = np.load(str(npy_path), allow_pickle=False)

        # ---- Stage 7 (old): Build DataLoaders -----------------------
        if not state.is_complete("stage_train_loaders"):
            try:
                train_loader, val_loader = self._build_dataloaders(
                    df_features, df_labels, sequences, feature_pipeline,
                )
                results["train_loader_batches"] = len(train_loader)
                results["val_loader_batches"] = len(val_loader)
                state.mark_complete("stage_train_loaders", {
                    "train_batches": len(train_loader),
                    "val_batches": len(val_loader),
                })
            except Exception as e:
                logger.error("[PIPELINE] DataLoader build failed: %s", e)
                state.mark_failed("stage_train_loaders", str(e))
                raise
        else:
            train_loader, val_loader = self._build_dataloaders(
                df_features, df_labels, sequences, feature_pipeline,
            )

        # ---- Stage 8 (old): Train teacher model ---------------------
        if not state.is_complete("stage_train"):
            try:
                results["training"] = self._train(train_loader, val_loader, output_dir)
                if feature_pipeline is not None:
                    feature_pipeline.save(str(out / "feature_pipeline"))
                state.mark_complete("stage_train", {
                    "status": results["training"].get("status", "unknown"),
                })
            except Exception as e:
                logger.error("[PIPELINE] Training failed: %s", e)
                state.mark_failed("stage_train", str(e))
                raise
        else:
            results["training"] = state.state["artifacts"].get(
                "stage_train", {"status": "resumed"},
            )

        # ---- Stage 8.5: Model analysis -----------------------------
        if not state.is_complete("stage_analysis"):
            try:
                analysis_results = self._analyze_model(
                    val_loader=val_loader, output_dir=output_dir,
                )
                results["analysis"] = analysis_results
                state.mark_complete("stage_analysis", {"status": "success"})
            except Exception as e:
                logger.warning("[PIPELINE] Analysis failed: %s -- continuing", e)
                results["analysis"] = {"status": "failed", "error": str(e)}
                state.mark_complete("stage_analysis", {"status": "failed"})
        else:
            results["analysis"] = state.state["artifacts"].get(
                "stage_analysis", {"status": "resumed"},
            )

        # ---- Stage 9 (old): Knowledge distillation -----------------
        if not state.is_complete("stage_distill"):
            try:
                distill_cfg = self._config_to_dict().get("distillation", {})
                if distill_cfg.get("enabled", True):
                    results["distillation"] = self._distill(
                        teacher_checkpoint=str(out / "model.pth"),
                        feature_df=df_features,
                        label_df=df_labels,
                        output_dir=output_dir,
                    )
                else:
                    results["distillation"] = {"status": "disabled"}
                state.mark_complete("stage_distill", {
                    "status": results.get("distillation", {}).get("status", "unknown"),
                })
            except Exception as e:
                logger.error("[PIPELINE] Distillation failed: %s", e)
                state.mark_failed("stage_distill", str(e))
                raise
        else:
            results["distillation"] = state.state["artifacts"].get(
                "stage_distill", {"status": "resumed"},
            )

        # ---- Stage 9.5: Serving prep --------------------------------
        if not state.is_complete("stage_serving"):
            try:
                serving_cfg = self._config_to_dict().get("serving_prep", {})
                if serving_cfg.get("enabled", False):
                    results["serving_prep"] = self._prepare_serving(
                        feature_df=df_features, output_dir=output_dir,
                    )
                else:
                    results["serving_prep"] = {"status": "disabled"}
                state.mark_complete("stage_serving", {
                    "status": results.get("serving_prep", {}).get("status", "unknown"),
                })
            except Exception as e:
                logger.warning("[PIPELINE] Serving prep failed: %s", e)
                results["serving_prep"] = {"status": "error", "error": str(e)}
                state.mark_complete("stage_serving", {"status": "error"})
        else:
            results["serving_prep"] = state.state["artifacts"].get(
                "stage_serving", {"status": "resumed"},
            )

        # ---- Stage 10: CPE + Reason Generation ----------------------
        if not state.is_complete("stage_cpe"):
            try:
                stage10_cfg = self._config_to_dict().get("serving_prep", {})
                if stage10_cfg.get("cpe_enabled", False):
                    results["stage10_cpe_reason"] = self._run_stage10_cpe_reason(
                        output_dir=output_dir,
                    )
                else:
                    results["stage10_cpe_reason"] = {"status": "disabled"}
                state.mark_complete("stage_cpe", {
                    "status": results.get("stage10_cpe_reason", {}).get("status", "unknown"),
                })
            except Exception as e:
                logger.warning("[PIPELINE] Stage 10 failed: %s -- continuing", e)
                results["stage10_cpe_reason"] = {"status": "error", "error": str(e)}
                state.mark_complete("stage_cpe", {"status": "error"})
        else:
            results["stage10_cpe_reason"] = state.state["artifacts"].get(
                "stage_cpe", {"status": "resumed"},
            )

        # ---- Finalize ------------------------------------------------
        elapsed = time.time() - pipeline_start
        results["total_time_seconds"] = round(elapsed, 2)

        manifest_path = out / "pipeline_manifest.json"
        with open(manifest_path, "w") as f:
            json.dump(results, f, indent=2, default=str)

        # Close the adapter-supplied DuckDB connection now that all 9
        # stages have read from it. This was held open across the whole
        # pipeline so SQL-native generators in Stage 3 could query the
        # LIST columns lazily.
        if self._adapter_ctx is not None:
            try:
                self._adapter_ctx.con.close()
            except Exception as exc:
                logger.debug("adapter ctx close skipped: %s", exc)
            self._adapter_ctx = None

        logger.info("=" * 60)
        logger.info(
            "[PIPELINE] Full pipeline complete in %.1fs. Artifacts: %s",
            elapsed, output_dir,
        )
        logger.info("=" * 60)

        return results

    # ==================================================================
    # Stage 1: Build adapter and load raw data
    # ==================================================================

    def _build_adapter(self) -> Any:
        """Build a DataAdapter from config or fall back to GenericAdapter."""
        from .adapter import AdapterRegistry, DataAdapter

        stage_start = time.time()
        logger.info("[Stage 1] Building data adapter...")

        adapter_name = getattr(self.config, "adapter", None)

        if adapter_name:
            logger.info("[Stage 1] Using registered adapter: %s", adapter_name)
            adapter = AdapterRegistry.build(adapter_name, self._config_to_dict())
        else:
            logger.info("[Stage 1] No adapter specified, using GenericAdapter")
            adapter = _GenericAdapter(self._config_to_dict())

        logger.info("[Stage 1] Adapter built in %.2fs", time.time() - stage_start)
        return adapter

    # ==================================================================
    # Stage 3: Feature engineering
    # ==================================================================

    def _engineer_features(
        self,
        df: "pd.DataFrame",
        raw_data: Dict[str, "pd.DataFrame"],
    ) -> Tuple[Any, "pd.DataFrame"]:
        """Build and fit FeatureGroupPipeline, then transform data.

        If ``feature_groups`` are defined in config, uses the new
        FeatureGroupPipeline.  Otherwise falls back to auto-built groups
        from FeatureSpec.

        Args:
            df: Main DataFrame (post-preprocessing).
            raw_data: All raw DataFrames from adapter.

        Returns:
            ``(fitted_pipeline, transformed_df)``
        """
        stage_start = time.time()
        feature_groups_cfg = self.config.feature_groups

        if feature_groups_cfg:
            logger.info(
                "[Stage 4] Using FeatureGroupPipeline with %d groups from config",
                len(feature_groups_cfg),
            )
            return self._engineer_features_grouped(df, feature_groups_cfg)

        logger.info("[Stage 4] No feature_groups in config; auto-building from FeatureSpec")
        return self._engineer_features_from_config(df)

    def _engineer_features_grouped(
        self,
        df: "pd.DataFrame",
        groups_cfg: List[Dict[str, Any]],
    ) -> Tuple[Any, "pd.DataFrame"]:
        """Feature engineering via FeatureGroupPipeline with explicit group configs."""
        from ..feature.group_pipeline import FeatureGroupPipeline
        from ..feature.group import FeatureGroupConfig

        groups = [FeatureGroupConfig.from_dict(g) for g in groups_cfg]

        pipeline = FeatureGroupPipeline(
            groups=groups,
            name=f"{self.config.task_name}_features",
        )

        # Fit on subsample if large
        raw_cfg = self._config_to_dict()
        fit_subsample = raw_cfg.get("preprocessing", {}).get("fit_subsample_limit", 50000)
        fit_df = df.sample(min(fit_subsample, len(df)), random_state=42) if len(df) > fit_subsample else df
        # CLAUDE.md §3.3: forward the adapter's DuckDB context so SQL-native
        # generators (lag/rolling/multihot) read the full LIST columns from
        # the lazy DuckDB table rather than re-registering the LIST-stripped
        # pandas frame.
        pipeline.fit(fit_df, adapter_ctx=self._adapter_ctx)
        df_features = pipeline.transform(df, adapter_ctx=self._adapter_ctx)

        logger.info(
            "[Stage 3] FeatureGroupPipeline '%s': %d groups, total_dim=%d, "
            "output %d cols",
            pipeline.name, len(pipeline), pipeline.total_dim,
            len(df_features.columns),
        )

        return pipeline, df_features

    def _engineer_features_from_config(
        self,
        df: "pd.DataFrame",
    ) -> Tuple[Any, "pd.DataFrame"]:
        """Feature engineering via auto-built FeatureGroupPipeline from FeatureSpec."""
        from ..feature.group_pipeline import FeatureGroupPipeline
        from ..feature.group import FeatureGroupConfig

        groups: List[FeatureGroupConfig] = []

        # Numeric features group
        if self.config.features.numeric:
            numeric_cols = list(self.config.features.numeric)
            groups.append(FeatureGroupConfig(
                name="numeric_features",
                group_type="transform",
                transformers=["standard_scaler"],
                columns=numeric_cols,
                output_columns=numeric_cols,
                output_dim=len(numeric_cols),
            ))

        # Categorical features group
        if self.config.features.categorical:
            cat_cols = list(self.config.features.categorical)
            groups.append(FeatureGroupConfig(
                name="categorical_features",
                group_type="transform",
                transformers=["label_encoder"],
                columns=cat_cols,
                output_columns=cat_cols,
                output_dim=len(cat_cols),
            ))

        if not groups:
            # Fallback: treat all non-label columns as numeric
            label_cols = {t.label_col for t in self.config.tasks}
            id_cols = set(getattr(self.config.features, "id_cols", []))
            exclude = label_cols | id_cols
            all_cols = [c for c in df.columns if c not in exclude]

            groups.append(FeatureGroupConfig(
                name="all_features",
                group_type="transform",
                transformers=["null_filler"],
                transformer_params={"null_filler": {"strategy": "median"}},
                columns=all_cols,
                output_columns=all_cols,
                output_dim=len(all_cols),
            ))

        pipeline = FeatureGroupPipeline(
            groups=groups,
            name=f"{self.config.task_name}_features",
        )

        # Fit on subsample if large
        raw_cfg = self._config_to_dict()
        fit_subsample = raw_cfg.get("preprocessing", {}).get("fit_subsample_limit", 50000)
        fit_df = df.sample(min(fit_subsample, len(df)), random_state=42) if len(df) > fit_subsample else df
        # CLAUDE.md §3.3: forward the adapter's DuckDB context so SQL-native
        # generators (lag/rolling/multihot) read the full LIST columns from
        # the lazy DuckDB table rather than re-registering the LIST-stripped
        # pandas frame.
        pipeline.fit(fit_df, adapter_ctx=self._adapter_ctx)
        df_features = pipeline.transform(df, adapter_ctx=self._adapter_ctx)

        logger.info(
            "[Stage 3] Auto-built FeatureGroupPipeline: %d groups, total_dim=%d, "
            "output %d cols",
            len(pipeline), pipeline.total_dim, len(df_features.columns),
        )

        return pipeline, df_features

    # ==================================================================
    # Stage 4: Label derivation
    # ==================================================================

    def _derive_labels(self, df: "pd.DataFrame") -> "pd.DataFrame":
        """Derive label columns from raw data.

        Tries to use a dedicated LabelDeriver if available.
        Falls back to extracting label columns directly from config.

        Args:
            df: Raw main DataFrame (pre-feature-engineering).

        Returns:
            DataFrame containing only label columns, aligned with df index.
        """
        import pandas as pd

        stage_start = time.time()

        # Try dedicated LabelDeriver
        try:
            from .label_deriver import LabelDeriver, LabelConfig

            # Build LabelConfig list from pipeline config. Type-specific
            # parameters (``indices`` for list_intersect, ``mapping`` for
            # string_map, ``weights`` for weighted_sum, etc.) live on
            # ``LabelSpec.extra`` because they are not part of the
            # canonical dataclass schema; we unpack them so the LabelConfig
            # ``**kwargs`` collector sees them as first-class config keys.
            label_configs_raw = self.config.labels
            if label_configs_raw:
                label_configs = []
                for lc in label_configs_raw:
                    extras = dict(getattr(lc, "extra", {}) or {})
                    label_configs.append(LabelConfig(
                        name=lc.name or lc.input_col,
                        source=lc.source,
                        type=lc.type,
                        col=lc.input_col if lc.source == "column" else "",
                        method=lc.method,
                        input_col=lc.input_col,
                        num_classes=lc.num_classes,
                        **extras,
                    ))
                deriver = LabelDeriver()
                # CLAUDE.md §3.3: forward the post-Stage-3 DuckDB table
                # so LabelDeriver runs SQL against it directly without
                # opening a scratch connection or registering a fresh
                # copy of df.
                ctx = self._adapter_ctx
                src_table = getattr(self, "_post_stage3_table", None)
                if ctx is not None and src_table is not None:
                    labels = deriver.derive(
                        df, label_configs,
                        con=ctx.con, source_table=src_table,
                    )
                else:
                    labels = deriver.derive(df, label_configs)
                logger.info("[Stage 4] Labels derived via LabelDeriver: %d cols",
                            len(labels.columns))
                return labels
        except (ImportError, ModuleNotFoundError):
            pass
        except Exception as e:
            logger.warning("[Stage 4] LabelDeriver failed (%s), falling back to direct extraction", e)

        # Fallback: extract configured label columns directly
        label_cols = []
        for task in self.config.tasks:
            col = task.label_col
            if col in df.columns:
                label_cols.append(col)
            else:
                logger.warning(
                    "[Stage 4] Label column '%s' for task '%s' not found",
                    col, task.name,
                )

        if not label_cols:
            raise ValueError(
                f"No label columns found in DataFrame. "
                f"Expected: {[t.label_col for t in self.config.tasks]}, "
                f"Available: {list(df.columns[:20])}..."
            )

        return df[label_cols].copy()

    # ==================================================================
    # Stage 5: Temporal split (returns indices, not DataFrames)
    # ==================================================================

    def _compute_split_indices(
        self,
        df: "pd.DataFrame",
    ) -> Tuple[List[int], List[int], List[int]]:
        """Compute train/val/test row indices.

        Uses temporal split if configured, otherwise random split.

        When ``self._adapter_ctx`` is set, indices are computed via
        DuckDB SQL on the registered ``_post_stage3`` table — no pandas
        ``df.iloc`` view, no numpy permutation on a 1M-element array.
        Returns the same tuple shape so legacy callers are unaffected.

        Returns:
            ``(train_indices, val_indices, test_indices)`` as lists of ints.
        """
        import numpy as np

        # CLAUDE.md §3.3: SQL-native split when the adapter context is
        # available. We still return python ``list[int]`` so Stage 6
        # (already on the SQL path) can build the train_predicate
        # boolean column from these indices.
        if (self._adapter_ctx is not None
            and getattr(self, "_post_stage3_table", None) is not None):
            indices = self._compute_split_indices_sql()
            if indices is not None:
                return indices

        temporal_cfg = self._get_temporal_split_config()

        if temporal_cfg is not None and temporal_cfg.get("enabled", False):
            date_col = temporal_cfg.get("date_col")
            if not date_col:
                logger.warning(
                    "No date_col configured for temporal split; "
                    "falling back to random split"
                )
            else:
                from .temporal_split import TemporalSplitter

                splitter = TemporalSplitter(
                    train_ratio=temporal_cfg.get("train_ratio", 0.7),
                    val_ratio=temporal_cfg.get("val_ratio", 0.15),
                    gap_days=temporal_cfg.get("gap_days", 7),
                )
                train_df, val_df, test_df = splitter.split(df, date_col=date_col)

                # Map back to original df indices
                train_idx = train_df.index.tolist() if not train_df.index.equals(
                    range(len(train_df))
                ) else list(range(len(train_df)))
                val_start = len(train_idx)
                val_idx = list(range(val_start, val_start + len(val_df)))
                test_start = val_start + len(val_df)
                test_idx = list(range(test_start, test_start + len(test_df)))

                # The temporal splitter resets indices; use positional ranges
                # based on the sorted order
                n = len(df)
                n_train = len(train_df)
                n_val = len(val_df)
                n_test = len(test_df)
                # Gap rows are excluded; total may be < n
                train_idx = list(range(n_train))
                val_idx = list(range(n_train, n_train + n_val))
                test_idx = list(range(n_train + n_val, n_train + n_val + n_test))

                logger.info(
                    "[Stage 5] Temporal split: train=%d, val=%d, test=%d, "
                    "gap_days=%d",
                    n_train, n_val, n_test, temporal_cfg.get("gap_days", 7),
                )
                return train_idx, val_idx, test_idx

        # Fallback: deterministic random split
        n = len(df)
        train_frac = self.config.data.train_split
        val_frac = self.config.data.val_split if hasattr(self.config.data, "val_split") else 0.1
        seed = self.config.training.seed

        rng = np.random.RandomState(seed)
        indices = rng.permutation(n)

        n_train = int(n * train_frac)
        n_val = int(n * val_frac)

        train_idx = indices[:n_train].tolist()
        val_idx = indices[n_train:n_train + n_val].tolist()
        test_idx = indices[n_train + n_val:].tolist()

        logger.info(
            "[Stage 5] Random split (seed=%d): train=%d, val=%d, test=%d",
            seed, len(train_idx), len(val_idx), len(test_idx),
        )
        return train_idx, val_idx, test_idx

    def _compute_split_indices_sql(
        self,
    ) -> Optional[Tuple[List[int], List[int], List[int]]]:
        """SQL-native row index computation against ``_post_stage3``.

        Returns ``None`` when no DuckDB context is wired (caller falls
        back to the pandas path).
        """
        ctx = self._adapter_ctx
        table = getattr(self, "_post_stage3_table", None)
        if ctx is None or table is None:
            return None

        con = ctx.con
        n = int(con.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0])
        if n == 0:
            return [], [], []

        temporal_cfg = self._get_temporal_split_config()
        if temporal_cfg is not None and temporal_cfg.get("enabled", False):
            date_col = temporal_cfg.get("date_col")
            if date_col:
                # Check column existence on the SQL side
                col_exists = con.execute(
                    f"SELECT COUNT(*) FROM (DESCRIBE {table}) WHERE column_name = ?",
                    [date_col],
                ).fetchone()[0]
                if col_exists:
                    train_ratio = float(temporal_cfg.get("train_ratio", 0.7))
                    val_ratio = float(temporal_cfg.get("val_ratio", 0.15))
                    gap_days = int(temporal_cfg.get("gap_days", 7))
                    cuts = con.execute(
                        f'SELECT QUANTILE_CONT("{date_col}", {train_ratio}), '
                        f'QUANTILE_CONT("{date_col}", {train_ratio + val_ratio}) '
                        f"FROM {table}"
                    ).fetchone()
                    train_cut, val_cut = cuts
                    if train_cut is None or val_cut is None:
                        logger.warning(
                            "[Stage 5] SQL temporal split: NULL quantiles, "
                            "falling back to random",
                        )
                    else:
                        # Detect whether ``date_col`` is a real DATE/
                        # TIMESTAMP (gap = INTERVAL N DAY) or a numeric
                        # encoding (e.g. yyyymm INT — gap = N units).
                        # We probe the type via DESCRIBE; anything that
                        # isn't a temporal type uses plain numeric subtraction.
                        col_type_row = con.execute(
                            f"SELECT column_type FROM (DESCRIBE {table}) "
                            f"WHERE column_name = ?",
                            [date_col],
                        ).fetchone()
                        col_type = (col_type_row[0] or "").upper() if col_type_row else ""
                        is_temporal = any(
                            t in col_type for t in ("DATE", "TIMESTAMP", "TIME")
                        )
                        if is_temporal:
                            train_gap_expr = f"{repr(train_cut)} - INTERVAL {gap_days} DAY"
                            val_gap_expr   = f"{repr(val_cut)} - INTERVAL {gap_days} DAY"
                        else:
                            # Numeric date (yyyymm DECIMAL / day_offset INT / etc).
                            # We can't translate gap_days into the column's own
                            # scale without metadata, so the safest behaviour is
                            # no gap: cut at the quantile, no rows dropped.
                            # Tests that need a gap should use a real DATE/
                            # TIMESTAMP column.
                            train_gap_expr = repr(train_cut)
                            val_gap_expr   = repr(val_cut)
                            if gap_days > 0:
                                logger.warning(
                                    "[Stage 5] date_col '%s' is numeric (type=%s); "
                                    "ignoring gap_days=%d (no scale translation).",
                                    date_col, col_type, gap_days,
                                )
                        rows = con.execute(
                            f'SELECT row_number() OVER () - 1 AS _idx, '
                            f'  CASE '
                            f'    WHEN "{date_col}" < {train_gap_expr} THEN 0 '
                            f'    WHEN "{date_col}" < {val_gap_expr} '
                            f'         AND "{date_col}" >= {repr(train_cut)} THEN 1 '
                            f'    WHEN "{date_col}" >= {repr(val_cut)} THEN 2 '
                            f'    ELSE -1 '
                            f'  END AS _split '
                            f'FROM {table}'
                        ).fetchall()
                        train_idx = [r[0] for r in rows if r[1] == 0]
                        val_idx   = [r[0] for r in rows if r[1] == 1]
                        test_idx  = [r[0] for r in rows if r[1] == 2]
                        logger.info(
                            "[Stage 5] SQL temporal split: train=%d, val=%d, "
                            "test=%d, gap_days=%d (dropped=%d)",
                            len(train_idx), len(val_idx), len(test_idx),
                            gap_days, n - len(train_idx) - len(val_idx) - len(test_idx),
                        )
                        return train_idx, val_idx, test_idx

        # Random split: deterministic via DuckDB HASH on row_number + seed.
        train_frac = float(self.config.data.train_split or 0.8)
        val_frac = float(getattr(self.config.data, "val_split", 0.1) or 0.1)
        seed = int(self.config.training.seed or 42)
        # HASH -> uniform [0, 1] via mod / range. Reproducible because
        # ``row_number()`` order is stable and seed is config-driven.
        rows = con.execute(
            f'SELECT _idx, _r FROM ('
            f'  SELECT row_number() OVER () - 1 AS _idx, '
            f'         (HASH(row_number() OVER () + {seed}) % 1000000) / 1000000.0 AS _r '
            f'  FROM {table}'
            f')'
        ).fetchall()
        train_idx, val_idx, test_idx = [], [], []
        for idx, r in rows:
            if r < train_frac:
                train_idx.append(idx)
            elif r < train_frac + val_frac:
                val_idx.append(idx)
            else:
                test_idx.append(idx)
        logger.info(
            "[Stage 5] SQL random split (seed=%d): train=%d, val=%d, test=%d",
            seed, len(train_idx), len(val_idx), len(test_idx),
        )
        return train_idx, val_idx, test_idx

    # ==================================================================
    # Stage 6: Sequence building
    # ==================================================================

    def _build_sequences(
        self, raw_data: Dict[str, "pd.DataFrame"],
    ) -> Optional[Dict[str, Any]]:
        """Build sequence tensors from raw data.

        Tries a dedicated SequenceBuilder if available.  Falls back to
        loading .npy files or detecting list-like columns in raw_data.

        Args:
            raw_data: Dict of DataFrames from adapter.

        Returns:
            Dict of sequence arrays keyed by name, or None if no sequences.
        """
        import numpy as np

        stage_start = time.time()

        # Try dedicated SequenceBuilder with config
        seq_cfg = {}
        raw_cfg = self._config_to_dict()
        seq_cfg_raw = raw_cfg.get("sequences", {})

        if self.config.sequences:
            try:
                from .sequence_builder import SequenceBuilder, SeqSourceConfig

                builder = SequenceBuilder()
                seq_configs = {}
                for name, spec in self.config.sequences.items():
                    seq_configs[name] = SeqSourceConfig(
                        source=spec.source,
                        columns=spec.columns,
                        mode=getattr(spec, "mode", "count_based"),
                        max_len=spec.seq_len,
                        window_days=getattr(spec, "window_days", 90),
                        stride_days=getattr(spec, "stride_days", 0),
                        timestamp_col=getattr(spec, "timestamp_col", ""),
                        truncate_last=getattr(spec, "truncate_last", 0),
                    )
                if seq_configs:
                    sequences = builder.build(raw_data, seq_configs)
                    if sequences:
                        return sequences
            except Exception as e:
                logger.warning("[Stage 8] SequenceBuilder failed: %s", e)

        # Fallback: check for pre-computed .npy files
        sequences: Dict[str, Any] = {}

        data_dir = Path(self.config.data.source).parent if self.config.data.source else None

        if data_dir and data_dir.exists():
            for npy_file in data_dir.glob("*_sequences.npy"):
                key = npy_file.stem.replace("_sequences", "")
                try:
                    arr = np.load(str(npy_file), allow_pickle=False)
                    sequences[key] = arr
                    logger.info(
                        "[Stage 8] Loaded sequence '%s' from %s: shape=%s",
                        key, npy_file, arr.shape,
                    )
                except Exception as e:
                    logger.warning("[Stage 8] Failed to load %s: %s", npy_file, e)

        # Check for list-like columns in raw_data["main"]
        if not sequences and "main" in raw_data:
            main_df = raw_data["main"]
            seq_cols = list(self.config.features.sequence)

            for col in seq_cols:
                if col in main_df.columns:
                    sample = main_df[col].dropna().iloc[0] if len(main_df[col].dropna()) > 0 else None
                    if isinstance(sample, (list, np.ndarray)):
                        try:
                            max_len = max(len(x) for x in main_df[col].dropna())
                            padded = np.zeros((len(main_df), max_len), dtype=np.float32)
                            for i, val in enumerate(main_df[col]):
                                if val is not None and hasattr(val, "__len__"):
                                    length = min(len(val), max_len)
                                    padded[i, :length] = np.array(val[:length], dtype=np.float32)
                            sequences[col] = padded
                            logger.info(
                                "[Stage 8] Built sequence '%s' from list column: shape=%s",
                                col, padded.shape,
                            )
                        except Exception as e:
                            logger.warning(
                                "[Stage 8] Failed to convert column '%s' to sequence: %s",
                                col, e,
                            )

        if not sequences:
            logger.info("[Stage 8] No sequences found")
            return None

        return sequences

    # ==================================================================
    # Stage 7 (leakage validation)
    # ==================================================================

    def _validate_leakage(
        self,
        df_features: "pd.DataFrame",
        df_labels: "pd.DataFrame",
        raw_df: Optional["pd.DataFrame"] = None,
    ) -> Any:
        """Run leakage validation checks between features and labels."""
        try:
            from .leakage_validator import LeakageValidator, ValidationResult
        except ImportError:
            logger.warning("[Stage 7] LeakageValidator not available, skipping")

            class _DummyResult:
                passed = True
                warnings: list = []
                details: dict = {}
            return _DummyResult()

        raw_cfg = self._config_to_dict()
        preproc = raw_cfg.get("preprocessing", {})
        seq_prefix = preproc.get("sequence_column_prefix", "seq_")
        prod_prefix = preproc.get("product_column_prefix", "prod_")
        max_seq_len = raw_cfg.get("sequences", {}).get("max_seq_len_expected", 16)

        validator = LeakageValidator(
            correlation_threshold=0.95,
            max_seq_len_expected=max_seq_len,
        )

        # CLAUDE.md §3.3: SQL-native correlation when DuckDB context
        # is available. We compute CORR(label, feat) for every pair via
        # batched SELECT aggregates against ``_post_stage3`` (already
        # holds the post-normalize feature matrix from Stage 6 if it
        # ran SQL-native, otherwise the pre-normalize columns — both
        # work for leakage detection because the threshold is on
        # absolute correlation). No 1M × N pandas .to_numpy().
        if (self._adapter_ctx is not None
            and getattr(self, "_post_stage3_table", None) is not None):
            try:
                result = self._validate_leakage_sql(
                    feature_cols=list(df_features.columns),
                    labels_df=df_labels,
                    threshold=0.95,
                )
            except Exception as exc:
                logger.warning(
                    "[Stage 7] SQL leakage path failed (%s), falling back to pandas",
                    exc,
                )
                result = validator.validate(df_features, df_labels)
        else:
            result = validator.validate(df_features, df_labels)

        # Check sequence leakage if raw data available
        if raw_df is not None:
            seq_cols = [c for c in raw_df.columns if c.startswith(seq_prefix)]
            if seq_cols:
                validator.check_sequence_leakage(raw_df, seq_cols, result=result)

            prod_cols = [c for c in raw_df.columns if c.startswith(prod_prefix)]
            if prod_cols and seq_cols:
                validator.check_product_columns(
                    raw_df, prod_cols, seq_cols, result=result
                )

        return result

    def _validate_leakage_sql(
        self,
        feature_cols: List[str],
        labels_df: "pd.DataFrame",
        threshold: float = 0.95,
    ) -> Any:
        """SQL-native feature-label correlation check via DuckDB CORR.

        Computes ``CORR(label, feature)`` for every pair as batched
        SELECT aggregates on the shared ``_post_stage3`` table joined
        with the labels frame. No per-feature numpy materialisation.
        """
        from .leakage_validator import ValidationResult

        ctx = self._adapter_ctx
        table = getattr(self, "_post_stage3_table", None)
        if ctx is None or table is None:
            raise RuntimeError("SQL leakage path requires adapter context")

        con = ctx.con
        result = ValidationResult()

        # Register labels (zero-copy) and join positionally with the
        # feature table so CORR can operate on aligned rows.
        import pyarrow as _pa
        labels_arrow = _pa.Table.from_pandas(labels_df, preserve_index=False)
        con.register("_leak_labels_arrow", labels_arrow)
        con.execute(
            "CREATE OR REPLACE TABLE _leak_labels AS "
            "SELECT * FROM _leak_labels_arrow"
        )
        try:
            con.unregister("_leak_labels_arrow")
        except Exception:
            pass

        # Join: features have row_number 0..n-1 from Stage 3; labels
        # share that order. POSITIONAL JOIN preserves alignment.
        con.execute(
            f"CREATE OR REPLACE TABLE _leak_joined AS "
            f"SELECT f.*, l.* FROM {table} f "
            f"POSITIONAL JOIN _leak_labels l"
        )

        # Filter to numeric feature/label cols actually in the joined
        # table. DuckDB's CORR returns NULL for non-numeric / constants.
        joined_cols_info = con.execute(
            "SELECT column_name, column_type FROM (DESCRIBE _leak_joined)"
        ).fetchall()
        type_map = {r[0]: (r[1] or "").upper() for r in joined_cols_info}
        numeric_types = ("TINYINT", "SMALLINT", "INTEGER", "BIGINT",
                         "FLOAT", "DOUBLE", "DECIMAL", "HUGEINT")
        def _scalar_numeric(t: str) -> bool:
            return any(p in t for p in numeric_types) and "[" not in t

        feat_present = [
            c for c in feature_cols
            if c in type_map and _scalar_numeric(type_map[c])
        ]
        label_present = [
            c for c in labels_df.columns
            if c in type_map and _scalar_numeric(type_map[c])
        ]

        suspicious = 0
        BATCH = 100
        for label_col in label_present:
            for start in range(0, len(feat_present), BATCH):
                batch = feat_present[start:start + BATCH]
                terms = ", ".join(
                    f'CORR("{label_col}", "{f}") AS "_c{i}"'
                    for i, f in enumerate(batch)
                )
                try:
                    row = con.execute(
                        f"SELECT {terms} FROM _leak_joined"
                    ).fetchone() or ()
                except Exception as exc:
                    logger.debug("CORR batch failed: %s", exc)
                    continue
                for i, f in enumerate(batch):
                    if f == label_col or i >= len(row):
                        continue
                    val = row[i]
                    if val is None:
                        continue
                    try:
                        corr = abs(float(val))
                    except (TypeError, ValueError):
                        continue
                    if not (corr == corr):  # NaN
                        continue
                    if corr >= threshold:
                        suspicious += 1
                        result.fail(
                            f"Feature '{f}' has {corr:.4f} correlation "
                            f"with label '{label_col}' (>= {threshold})"
                        )

        # Cleanup
        for tbl in ("_leak_joined", "_leak_labels"):
            try:
                con.execute(f"DROP TABLE IF EXISTS {tbl}")
            except Exception:
                pass

        logger.info(
            "[Stage 7] SQL CORR scan: %d feat x %d label, %d suspicious",
            len(feat_present), len(label_present), suspicious,
        )
        return result

    # ==================================================================
    # Stage 1.5: Temporal preparation (sequence truncation)
    # ==================================================================

    def _scalar_df_from_ctx(self, ctx) -> "pd.DataFrame":
        """Materialise scalar + short-LIST columns from the DuckDB
        adapter context into a pandas DataFrame, leaving the heavy LIST
        columns lazy in DuckDB.

        Heuristic: any LIST column with sample-mean length > ``LAZY_LIST_THRESHOLD``
        (default 50) stays in DuckDB. Below the threshold the column is
        small enough to materialise without OOM (e.g. nba_label averages
        0.87 elements; seq_saving has 17 monthly snapshots).

        Records the lazy column list onto ``ctx.metadata.extra`` so
        downstream consumers and stage1 metadata can introspect.
        """
        LAZY_LIST_THRESHOLD = 50
        SAMPLE_ROWS = 200

        con = ctx.con
        table = ctx.table_name
        # Identify LIST-typed columns via DuckDB's column metadata. The
        # ``DESCRIBE`` form is robust whether ``table`` is a real table
        # or a VIEW (which is what the §3.3-friendly SantanderAdapter
        # registers to keep the 1.4 GB parquet from being copied).
        cols_info = con.execute(
            f"SELECT column_name, column_type FROM (DESCRIBE {table})"
        ).fetchall()
        list_cols = [
            c for c, t in cols_info
            if t and ("[]" in str(t) or "LIST" in str(t).upper())
        ]

        # 2. Measure sample-mean length per LIST column.
        lazy_cols = []
        for col in list_cols:
            try:
                row = con.execute(
                    f"SELECT AVG(len({col})) FROM {table} USING SAMPLE {SAMPLE_ROWS} ROWS"
                ).fetchone()
                avg_len = float(row[0]) if row and row[0] is not None else 0.0
                if avg_len > LAZY_LIST_THRESHOLD:
                    lazy_cols.append(col)
            except Exception as exc:
                logger.debug("len-sampling skipped for %s: %s", col, exc)

        # 3. SELECT all columns except the lazy ones — these stay in DuckDB
        all_cols = [r[0] for r in cols_info]
        keep_cols = [c for c in all_cols if c not in set(lazy_cols)]
        select_sql = ", ".join(f'"{c}"' for c in keep_cols)
        df = con.execute(f"SELECT {select_sql} FROM {table}").df()

        # 4. Record on context metadata for diagnostics
        meta_extra = getattr(ctx.metadata, "extra", None) or {}
        meta_extra["list_cols_lazy"] = lazy_cols
        meta_extra["list_cols_materialised"] = [c for c in list_cols if c not in lazy_cols]
        try:
            ctx.metadata.extra = meta_extra
        except Exception:
            pass

        logger.info(
            "[Stage 1] DuckDB-native: %d LIST columns kept lazy in DuckDB "
            "(%s), %d short-LIST + scalar columns materialised to pandas (%d rows x %d cols)",
            len(lazy_cols), lazy_cols, len(keep_cols), len(df), len(df.columns),
        )
        return df

    # ==================================================================
    # Multi-job Phase 0 split: checkpoint helpers
    # ==================================================================

    def _save_checkpoint_post_stage2(
        self,
        ckpt_dir: "Path",
        df: Any,
        meta: Dict[str, Any],
    ) -> None:
        """Persist post-Stage-2 state to ``ckpt_dir/post_stage2/``.

        Output:
          * ``main.parquet`` — imputed / encoded scalar table
          * ``meta.json``   — id_cols, date_cols (so Stage 4-9 jobs can
                              reconstruct the same set without re-reading
                              the YAML)
        """
        import duckdb as _ddb_ckpt
        target = ckpt_dir / "post_stage2"
        target.mkdir(parents=True, exist_ok=True)

        parquet_path = target / "main.parquet"
        _con = _ddb_ckpt.connect()
        try:
            _con.register("_ckpt_main", df)
            _con.execute(
                f"COPY _ckpt_main TO '{parquet_path}' (FORMAT PARQUET)"
            )
        finally:
            _con.close()

        with open(target / "meta.json", "w") as f:
            json.dump(meta, f, indent=2, default=str)

        logger.info(
            "[Checkpoint] post_stage2 written: %s (%s)",
            parquet_path,
            getattr(df, "shape", "?"),
        )

    def _load_checkpoint_post_stage2(
        self, ckpt_dir: "Path"
    ) -> Tuple[Any, Dict[str, Any]]:
        """Load post-Stage-2 state from ``ckpt_dir/post_stage2/``.

        Returns a pyarrow.Table (so the existing Stage 3 boundary code
        path that converts Arrow → pandas continues to work) plus the
        meta dict.
        """
        import duckdb as _ddb_ckpt
        import pyarrow as _pa
        target = ckpt_dir / "post_stage2"
        parquet_path = target / "main.parquet"
        if not parquet_path.exists():
            raise FileNotFoundError(
                f"post_stage2 checkpoint missing: {parquet_path}"
            )

        _con = _ddb_ckpt.connect()
        try:
            arrow_tbl = _con.execute(
                f"SELECT * FROM read_parquet('{parquet_path}')"
            ).fetch_arrow_table()
        finally:
            _con.close()

        meta_path = target / "meta.json"
        meta: Dict[str, Any] = {}
        if meta_path.exists():
            with open(meta_path) as f:
                meta = json.load(f)

        logger.info(
            "[Checkpoint] post_stage2 loaded: %s (rows=%d, cols=%d)",
            parquet_path, arrow_tbl.num_rows, arrow_tbl.num_columns,
        )
        return arrow_tbl, meta

    def _save_checkpoint_post_stage3(
        self,
        ckpt_dir: "Path",
        df: "pd.DataFrame",
        feature_pipeline: Any,
        feature_schema: Dict[str, Any],
        meta: Dict[str, Any],
    ) -> None:
        """Persist post-Stage-3 state to ``ckpt_dir/post_stage3/``.

        Output:
          * ``main.parquet``       — df with merged generated features
          * ``feature_schema.json`` — get_ple_input_metadata() result
          * ``feature_pipeline/``  — feature_pipeline.save() target
          * ``meta.json``          — id_cols, date_cols
        """
        import duckdb as _ddb_ckpt
        target = ckpt_dir / "post_stage3"
        target.mkdir(parents=True, exist_ok=True)

        parquet_path = target / "main.parquet"
        _con = _ddb_ckpt.connect()
        try:
            _con.register("_ckpt_main", df)
            _con.execute(
                f"COPY _ckpt_main TO '{parquet_path}' (FORMAT PARQUET)"
            )
        finally:
            _con.close()

        with open(target / "feature_schema.json", "w") as f:
            json.dump(feature_schema, f, indent=2, default=str)

        try:
            feature_pipeline.save(str(target / "feature_pipeline"))
        except Exception as exc:
            logger.warning(
                "[Checkpoint] feature_pipeline.save failed: %s — Stage 4-9 "
                "job will rebuild from feature_schema.json", exc,
            )

        with open(target / "meta.json", "w") as f:
            json.dump(meta, f, indent=2, default=str)

        logger.info(
            "[Checkpoint] post_stage3 written: %s (%s)",
            parquet_path,
            getattr(df, "shape", "?"),
        )

    def _load_checkpoint_post_stage3(
        self, ckpt_dir: "Path"
    ) -> Tuple[Any, Dict[str, Any], Dict[str, Any]]:
        """Load post-Stage-3 *metadata* from ``ckpt_dir/post_stage3/``.

        Returns ``(feature_pipeline, feature_schema, meta)``. The
        wide post-Stage-3 matrix itself stays on disk — the run()
        prep block registers ``main.parquet`` as a DuckDB view on
        the adapter ctx (zero materialise, with DECIMAL → DOUBLE
        cast in the projection). This keeps Job C's resident set
        bounded by the slim ``df_lite`` (id/label/seq columns) plus
        whatever the SQL transform/save needs.
        """
        target = ckpt_dir / "post_stage3"
        parquet_path = target / "main.parquet"
        if not parquet_path.exists():
            raise FileNotFoundError(
                f"post_stage3 checkpoint missing: {parquet_path}"
            )

        schema_path = target / "feature_schema.json"
        feature_schema: Dict[str, Any] = {}
        if schema_path.exists():
            with open(schema_path) as f:
                feature_schema = json.load(f)

        feature_pipeline = None
        pipeline_path = target / "feature_pipeline"
        if pipeline_path.exists():
            try:
                from core.feature.group_pipeline import FeatureGroupPipeline
                feature_pipeline = FeatureGroupPipeline.load(str(pipeline_path))
            except Exception as exc:
                logger.warning(
                    "[Checkpoint] feature_pipeline.load failed: %s — "
                    "downstream stages must work from feature_schema.json",
                    exc,
                )

        meta_path = target / "meta.json"
        meta: Dict[str, Any] = {}
        if meta_path.exists():
            with open(meta_path) as f:
                meta = json.load(f)

        logger.info(
            "[Checkpoint] post_stage3 metadata loaded: %s (groups=%d, "
            "schema_keys=%d) — matrix stays on disk",
            parquet_path,
            len(feature_pipeline) if feature_pipeline is not None else 0,
            len(feature_schema),
        )
        return feature_pipeline, feature_schema, meta

    def _prepare_temporal(self, df: "pd.DataFrame") -> "pd.DataFrame":
        """Truncate sequences and recompute product columns to prevent leakage.

        Controlled by ``data.temporal_split`` and
        ``preprocessing.leakage_prevention`` in the pipeline config YAML.
        """
        stage_start = time.time()
        logger.info("[Stage 1.5] Preparing temporal features...")

        raw_cfg = self._config_to_dict()
        preproc = raw_cfg.get("preprocessing", {})
        leakage_cfg = preproc.get("leakage_prevention", {})

        if not leakage_cfg.get("recompute_prod_from_seq", False):
            seq_cfg = raw_cfg.get("sequences", {})
            has_truncation = any(
                s.get("truncate_last", 0) > 0
                for s in seq_cfg.values()
                if isinstance(s, dict)
            )
            if not has_truncation:
                logger.info("[Stage 1.5] No leakage prevention configured, passing through")
                return df

        try:
            from .temporal_split import TemporalSplitter

            seq_prefix = preproc.get("sequence_column_prefix", "seq_")
            prod_prefix = preproc.get("product_column_prefix", "prod_")

            splitter = TemporalSplitter()
            seq_cols = [c for c in df.columns if c.startswith(seq_prefix)]

            if seq_cols:
                df = splitter.split_by_sequence_cutoff(
                    df,
                    seq_cols=seq_cols,
                    cutoff_offset=1,
                    prod_col_prefix=prod_prefix,
                    seq_col_prefix=seq_prefix,
                )
                logger.info(
                    "[Stage 1.5] Temporal preparation complete in %.2fs: "
                    "truncated %d seq columns",
                    time.time() - stage_start, len(seq_cols),
                )
            else:
                logger.info("[Stage 1.5] No seq_* columns found, skipping")

        except ImportError:
            logger.warning("[Stage 1.5] TemporalSplitter not available, skipping")

        return df

    # ==================================================================
    # Temporal split helpers
    # ==================================================================

    def _get_temporal_split_config(self) -> Optional[Dict[str, Any]]:
        """Extract temporal split config from the raw YAML config."""
        raw_cfg = self._config_to_dict()
        data_cfg = raw_cfg.get("data", {})
        return data_cfg.get("temporal_split")

    # ==================================================================
    # DataLoader building (used by run_full only)
    # ==================================================================

    def _build_dataloaders(
        self,
        df_features: "pd.DataFrame",
        df_labels: "pd.DataFrame",
        sequences: Optional[Dict[str, Any]],
        feature_pipeline: Any,
    ) -> Tuple[Any, Any]:
        """Build train and validation DataLoaders from Phase 0 artifacts."""
        import numpy as np
        import pyarrow as pa

        stage_start = time.time()
        logger.info("[DataLoaders] Building DataLoaders...")

        # DuckDB positional join → Arrow (pandas-free path)
        import duckdb as _ddb_dl
        _con_dl = _ddb_dl.connect()
        try:
            _feat_reset = df_features.reset_index(drop=True)
            _lbl_reset = df_labels.reset_index(drop=True)
            _con_dl.register("_feat", _feat_reset)
            _con_dl.register("_lbl", _lbl_reset)
            _fcols = ", ".join(f'_feat."{c}"' for c in _feat_reset.columns)
            _lcols = ", ".join(f'_lbl."{c}"' for c in _lbl_reset.columns)
            tbl_combined: pa.Table = _con_dl.execute(
                f"SELECT {_fcols}, {_lcols} FROM _feat POSITIONAL JOIN _lbl"
            ).fetch_arrow_table()
        finally:
            _con_dl.close()

        # Load split indices
        split_path = self._output_dir / "split_indices.json"
        if split_path.exists():
            with open(split_path) as f:
                split_indices = json.load(f)
            train_idx = split_indices["train"]
            val_idx = split_indices["val"] + split_indices.get("test", [])
        else:
            # Fallback: random split
            n = tbl_combined.num_rows
            n_train = int(n * self.config.data.train_split)
            rng = np.random.RandomState(self.config.training.seed)
            indices = rng.permutation(n)
            train_idx = indices[:n_train].tolist()
            val_idx = indices[n_train:].tolist()

        # pa.Table.take() accepts integer index lists — Arrow-native, no pandas needed
        train_df = tbl_combined.take(train_idx)
        val_df = tbl_combined.take(val_idx)

        logger.info(
            "[DataLoaders] train=%d rows, val=%d rows",
            len(train_df), len(val_df),
        )

        # Build FeatureColumnSpec from pipeline metadata
        if feature_pipeline is not None:
            ple_metadata = feature_pipeline.get_ple_input_metadata()
            feature_spec = self._build_feature_spec(ple_metadata)
        else:
            # Fallback: use all feature columns as static features
            from ..data.dataloader import FeatureColumnSpec
            feature_spec = FeatureColumnSpec(
                static_features=list(df_features.columns),
            )

        label_map = {
            task.name: task.label_col
            for task in self.config.tasks
            if task.label_col in df_labels.columns
        }

        from ..data.dataloader import build_ple_dataloader

        batch_size = self.config.training.batch_size

        train_loader = build_ple_dataloader(
            df=train_df,
            feature_spec=feature_spec,
            label_columns=label_map,
            batch_size=batch_size,
            shuffle=True,
            pin_memory=True,
        )

        val_loader = build_ple_dataloader(
            df=val_df,
            feature_spec=feature_spec,
            label_columns=label_map,
            batch_size=batch_size,
            shuffle=False,
            pin_memory=True,
        )

        logger.info("[DataLoaders] Built in %.2fs", time.time() - stage_start)
        return train_loader, val_loader

    @staticmethod
    def _build_feature_spec(ple_metadata: Dict[str, Any]) -> Any:
        """Construct a FeatureColumnSpec from pipeline metadata."""
        from ..data.dataloader import FeatureColumnSpec

        output_columns = ple_metadata.get("output_columns", [])
        expert_routing = ple_metadata.get("expert_routing", {})

        spec_kwargs: Dict[str, Any] = {
            "static_features": list(output_columns),
        }

        _EXPERT_FIELD_MAP = {
            "hyperbolic": "hyperbolic_columns",
            "tda": "tda_columns",
            "collaborative": "collaborative_columns",
            "hmm_journey": "hmm_journey_columns",
            "hmm_lifecycle": "hmm_lifecycle_columns",
            "hmm_behavior": "hmm_behavior_columns",
            "multidisciplinary": "multidisciplinary_columns",
            "coldstart": "coldstart_columns",
            "anonymous": "anonymous_columns",
        }

        group_ranges = ple_metadata.get("feature_group_ranges", {})

        for expert_name, group_names in expert_routing.items():
            field_name = _EXPERT_FIELD_MAP.get(expert_name)
            if field_name:
                expert_cols: List[str] = []
                for gname in group_names:
                    start, end = group_ranges.get(gname, (0, 0))
                    expert_cols.extend(output_columns[start:end])
                if expert_cols:
                    spec_kwargs[field_name] = expert_cols

        return FeatureColumnSpec(**spec_kwargs)

    # ==================================================================
    # Training (used by run_full)
    # ==================================================================

    def _train(
        self,
        train_loader: Any,
        val_loader: Any,
        output_dir: str,
    ) -> dict:
        """Dispatch training to the configured architecture."""
        arch = self.config.model.architecture

        stage_start = time.time()
        logger.info("[Training] Starting (architecture=%s)...", arch)

        if arch == "ple":
            results = self._train_ple(train_loader, val_loader, output_dir)
        elif arch == "lgbm":
            results = self._train_lgbm(train_loader, val_loader, output_dir)
        else:
            raise ValueError(f"Unknown architecture: {arch}")

        logger.info("[Training] Complete in %.2fs", time.time() - stage_start)
        return results

    def _train_ple(
        self,
        train_loader: Any,
        val_loader: Any,
        output_dir: str,
    ) -> dict:
        """Build a PLEModel and train with PLETrainer."""
        import torch

        from ..model.ple.config import (
            PLEConfig, ExpertConfig, TaskTowerConfig,
            ExpertBasketConfig, AdaTTConfig, GroupTaskExpertConfig,
        )
        from ..model.ple.model import PLEModel
        from ..training.trainer import PLETrainer
        from ..training.config import TrainingConfig

        input_dim = self.config.features.input_dim
        if input_dim == 0:
            for batch in train_loader:
                if "features" in batch:
                    input_dim = batch["features"].shape[-1]
                break

        task_names = [t.name for t in self.config.tasks]
        logger.info("[PLE] Building model: input_dim=%d, tasks=%d",
                    input_dim, len(task_names))

        task_overrides: Dict[str, Dict[str, Any]] = {}
        for t in self.config.tasks:
            override: Dict[str, Any] = {"task_type": t.type}
            if t.type == "contrastive":
                override["output_dim"] = t.num_classes
                override["activation"] = None
            elif t.type == "multiclass":
                override["output_dim"] = t.num_classes
                override["activation"] = "softmax"
            elif t.type == "regression":
                override["output_dim"] = 1
                override["activation"] = None
            else:
                override["output_dim"] = 1
                override["activation"] = "sigmoid"
            if t.tower_type:
                override["tower_type"] = t.tower_type
            if t.tower_dims:
                override["tower_dims"] = t.tower_dims
            task_overrides[t.name] = override

        expert_output_dim = max(self.config.model.expert_hidden_dim // 4, 64)

        expert_basket_cfg: Optional[ExpertBasketConfig] = None
        if self.config.model.expert_basket is not None:
            eb = self.config.model.expert_basket
            expert_basket_cfg = ExpertBasketConfig(
                shared_experts=eb.get("shared_experts", []),
                task_experts=eb.get("task_experts", []),
                expert_configs=eb.get("expert_configs", {}),
            )

        adatt_cfg: Optional[AdaTTConfig] = None
        if self.config.task_groups:
            adatt_cfg = AdaTTConfig.from_pipeline_groups(self.config.task_groups)

        group_task_expert_cfg: Optional[GroupTaskExpertConfig] = None
        if self.config.model.group_task_expert is not None:
            gte = self.config.model.group_task_expert
            group_task_expert_cfg = GroupTaskExpertConfig(**gte)

        task_group_map: Dict[str, str] = {}
        for tg in self.config.task_groups:
            for t in tg.tasks:
                task_group_map[t] = tg.name

        ple_config = PLEConfig(
            input_dim=input_dim,
            task_names=task_names,
            num_shared_experts=self.config.model.num_shared_experts,
            num_task_experts_per_task=self.config.model.num_task_experts,
            num_extraction_layers=self.config.model.num_layers,
            shared_expert=ExpertConfig(
                hidden_dims=[self.config.model.expert_hidden_dim],
                output_dim=expert_output_dim,
            ),
            task_tower=TaskTowerConfig(
                hidden_dims=list(self.config.model.tower_dims),
                dropout=self.config.model.dropout,
            ),
            dropout=self.config.model.dropout,
            task_overrides=task_overrides,
            expert_basket=expert_basket_cfg,
            task_group_map=task_group_map,
            **({"group_task_expert": group_task_expert_cfg} if group_task_expert_cfg is not None else {}),
            **({"adatt": adatt_cfg} if adatt_cfg is not None else {}),
        )

        model = PLEModel(ple_config)

        total_epochs = self.config.training.epochs
        phase1_epochs = max(1, total_epochs * 3 // 5)
        phase2_epochs = max(1, total_epochs - phase1_epochs)

        training_config = TrainingConfig.from_dict({
            "batch_size": self.config.training.batch_size,
            "optimizer": {"learning_rate": self.config.training.learning_rate},
            "early_stopping": {
                "patience": self.config.training.early_stopping_patience,
            },
            "phase1": {"epochs": phase1_epochs},
            "phase2": {"epochs": phase2_epochs},
            "experiment_name": self.config.task_name,
        })

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        trainer = PLETrainer(model, training_config, device=device)
        results = trainer.train(train_loader, val_loader, phase="full")

        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        checkpoint_path = out / "ple_model.pt"
        torch.save({
            "model_state_dict": model.state_dict(),
            "ple_config": ple_config,
            "training_results": results,
            "task_names": task_names,
            "input_dim": input_dim,
        }, str(checkpoint_path))

        logger.info(
            "[PLE] best_val_loss=%.6f, saved to %s",
            results.get("best_val_loss", float("inf")),
            checkpoint_path,
        )
        return {"status": "success", "model": "ple", **results}

    def _train_lgbm(
        self,
        train_loader: Any,
        val_loader: Any,
        output_dir: str,
    ) -> dict:
        """Train per-task LGBM models."""
        import numpy as np
        from ..model.lgbm.model import LGBMModel
        from ..model.lgbm.config import LGBMConfig

        X_train_parts, y_train_parts = [], {}
        for batch in train_loader:
            features = batch.get("features")
            if features is not None:
                X_train_parts.append(features.numpy())
            targets = batch.get("targets", {})
            for task_name, vals in targets.items():
                y_train_parts.setdefault(task_name, []).append(vals.numpy())

        X_train = np.concatenate(X_train_parts, axis=0)
        y_train = {k: np.concatenate(v) for k, v in y_train_parts.items()}

        cfg = LGBMConfig(
            learning_rate=self.config.training.learning_rate,
            n_estimators=self.config.training.epochs * 25,
        )
        tasks_meta = [{"name": t.name, "type": t.type} for t in self.config.tasks]

        model = LGBMModel(cfg, tasks_meta)
        model.fit(X_train, y_train)

        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        model.save(str(out))

        logger.info("[LGBM] Training complete. Model saved to %s", output_dir)
        return {"status": "success", "model": "lgbm"}

    # ==================================================================
    # Model analysis (Stage 8.5, used by run_full)
    # ==================================================================

    def _analyze_model(
        self,
        val_loader: Any,
        output_dir: str,
    ) -> dict:
        """Run post-training model analysis (gates, HGCN, IG, etc.)."""
        import numpy as np
        import torch

        from ..evaluation.gate_analyzer import GateAnalyzer

        out = Path(output_dir)
        analysis_dir = out / "analysis"
        analysis_dir.mkdir(parents=True, exist_ok=True)

        analysis: Dict[str, Any] = {}
        stage_start = time.time()
        logger.info("[Analysis] Starting model analysis...")

        checkpoint_path = out / "ple_model.pt"
        if not checkpoint_path.exists():
            logger.warning("[Analysis] No PLE checkpoint found, skipping.")
            return {"status": "skipped", "reason": "no_checkpoint"}

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        checkpoint = torch.load(str(checkpoint_path), map_location=device)
        ple_config = checkpoint["ple_config"]

        from ..model.ple.model import PLEModel
        model = PLEModel(ple_config)
        model.load_state_dict(checkpoint["model_state_dict"])
        model.to(device)
        model.eval()

        # CGC Gate analysis
        try:
            gate_analyzer = GateAnalyzer(model)
            gate_result = gate_analyzer.analyze(val_loader, max_batches=20)
            analysis["gate_weights"] = gate_result.to_dict()
            gate_path = analysis_dir / "gate_analysis.json"
            with open(gate_path, "w") as f:
                json.dump(gate_result.to_dict(), f, indent=2)
        except Exception as e:
            logger.warning("[Analysis] CGC gate analysis failed: %s", e)
            analysis["gate_weights"] = {"error": str(e)}

        # HGCN interpretable scores
        try:
            hgcn_scores = self._extract_hgcn_scores(model, val_loader, device)
            analysis["hgcn_interpretable"] = hgcn_scores
            hgcn_path = analysis_dir / "hgcn_interpretable.json"
            with open(hgcn_path, "w") as f:
                json.dump(hgcn_scores, f, indent=2)
        except Exception as e:
            logger.warning("[Analysis] HGCN extraction failed: %s", e)
            analysis["hgcn_interpretable"] = {"error": str(e)}

        # Integrated Gradients
        analysis_cfg = self._config_to_dict().get("analysis", {})
        ig_cfg = analysis_cfg.get("integrated_gradients", {})
        try:
            from ..evaluation.integrated_gradients import IntegratedGradients

            ig = IntegratedGradients(
                model,
                baseline=ig_cfg.get("baseline", "zeros"),
                n_steps=ig_cfg.get("n_steps", 50),
                device=device,
            )
            ig_max_batches = ig_cfg.get("max_batches", 50)

            ig_results: Dict[str, Any] = {}
            task_names = getattr(model, "task_names", [])
            for task_name in task_names:
                try:
                    importance = ig.feature_importance(
                        val_loader, task_name, max_batches=ig_max_batches,
                    )
                    ig_results[task_name] = importance
                except Exception as task_e:
                    ig_results[task_name] = {"error": str(task_e)}

            analysis["ig_attributions"] = ig_results
            ig_path = analysis_dir / "ig_attributions.json"
            with open(ig_path, "w") as f:
                json.dump(ig_results, f, indent=2, default=str)
        except Exception as e:
            logger.warning("[Analysis] IG analysis failed: %s", e)
            analysis["ig_attributions"] = {"error": str(e)}

        elapsed = time.time() - stage_start
        analysis["analysis_time_seconds"] = round(elapsed, 2)
        logger.info("[Analysis] Complete in %.2fs", elapsed)

        return analysis

    def _extract_hgcn_scores(
        self,
        model: Any,
        val_loader: Any,
        device: Any,
    ) -> dict:
        """Extract 6D interpretable projections from the HGCN expert."""
        import torch
        from core.model.ple.model import PLEInput
        from core.model.experts.hgcn import UnifiedHGCNExpert

        hgcn_experts = []
        for expert in model._iter_shared_experts():
            if isinstance(expert, UnifiedHGCNExpert):
                hgcn_experts.append(expert)

        if not hgcn_experts:
            return {"status": "skipped", "reason": "no_hgcn_expert_found"}

        all_scores: list = []
        num_samples = 0

        with torch.no_grad():
            for batch_idx, batch in enumerate(val_loader):
                if batch_idx >= 20:
                    break

                if isinstance(batch, dict):
                    ple_input = PLEInput(
                        features=batch["features"].to(device),
                        targets=None,
                    )
                    for fld in [
                        "hyperbolic_features", "tda_features",
                        "collaborative_features", "event_sequences",
                        "session_sequences",
                    ]:
                        if fld in batch and batch[fld] is not None:
                            object.__setattr__(
                                ple_input, fld, batch[fld].to(device),
                            )
                elif hasattr(batch, "features"):
                    ple_input = batch.to(device) if hasattr(batch, "to") else batch
                else:
                    continue

                _ = model(ple_input)

                for expert in hgcn_experts:
                    scores = expert.interpretable_scores
                    if scores is not None:
                        all_scores.append(scores.cpu())
                        num_samples += scores.size(0)

        if not all_scores:
            return {"status": "no_scores_collected", "num_hgcn_experts": len(hgcn_experts)}

        combined = torch.cat(all_scores, dim=0)
        mean_scores = combined.mean(dim=0).tolist()
        std_scores = combined.std(dim=0).tolist()
        labels = hgcn_experts[0].interpretable_labels

        axes = {
            "hierarchy_activation_intensity": {
                "mean": mean_scores[0:2], "std": std_scores[0:2],
            },
            "depth_importance": {
                "mean": mean_scores[2:4], "std": std_scores[2:4],
            },
            "cross_level_interaction": {
                "mean": mean_scores[4:6], "std": std_scores[4:6],
            },
        }

        return {
            "status": "success",
            "num_samples": num_samples,
            "num_hgcn_experts": len(hgcn_experts),
            "labels": labels,
            "mean_scores": {
                label: round(val, 6)
                for label, val in zip(labels, mean_scores)
            },
            "std_scores": {
                label: round(val, 6)
                for label, val in zip(labels, std_scores)
            },
            "axes": {
                axis_name: {
                    "mean": [round(v, 6) for v in axis_data["mean"]],
                    "std": [round(v, 6) for v in axis_data["std"]],
                }
                for axis_name, axis_data in axes.items()
            },
        }

    # ==================================================================
    # Distillation (Stage 9, used by run_full)
    # ==================================================================

    def _distill(
        self,
        teacher_checkpoint: str,
        feature_df: Any,
        label_df: Any,
        output_dir: str,
    ) -> dict:
        """Distill PLE teacher into per-task LGBM student models."""
        import numpy as np
        stage_start = time.time()
        logger.info("[Distillation] Starting...")

        try:
            import torch
            from ..training.student_trainer import StudentTrainer, StudentConfig
        except ImportError as e:
            logger.warning("[Distillation] Skipped -- missing dependency: %s", e)
            return {"status": "skipped", "reason": str(e)}

        distill_cfg = self._config_to_dict().get("distillation", {})
        student_config = StudentConfig(
            teacher_checkpoint=teacher_checkpoint,
            temperature=distill_cfg.get("temperature", 1.0),
            alpha=distill_cfg.get("alpha", 0.5),
            lgbm_params=distill_cfg.get("lgbm_params", {}),
        )

        id_cols = set(self.config.features.id_cols or ["user_id"])
        label_cols = set(label_df.columns) if hasattr(label_df, 'columns') else set()
        feature_columns = [
            c for c in feature_df.columns
            if c not in id_cols and c not in label_cols
        ]

        task_specs = self.config.tasks

        trainer = StudentTrainer(
            config=student_config,
            task_specs=task_specs,
            feature_columns=feature_columns,
            device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        )

        teacher = trainer.load_teacher(teacher_checkpoint)

        try:
            import duckdb as _ddb_dist
            _con_dist = _ddb_dist.connect()
            try:
                _feat_reset = feature_df.reset_index(drop=True)
                _lbl_reset = label_df.reset_index(drop=True)
                _con_dist.register("_feat", _feat_reset)
                _con_dist.register("_lbl", _lbl_reset)
                _fcols = ", ".join(f'_feat."{c}"' for c in _feat_reset.columns)
                _lcols = ", ".join(f'_lbl."{c}"' for c in _lbl_reset.columns)
                # Use .fetch_arrow_table() instead of .df() — merged goes directly to
                # build_ple_dataloader / PLEDataset which natively accepts PyArrow Tables.
                merged = _con_dist.execute(
                    f"SELECT {_fcols}, {_lcols} FROM _feat POSITIONAL JOIN _lbl"
                ).fetch_arrow_table()
                logger.info(
                    "POSITIONAL JOIN: merged %d + %d cols → %d cols (distillation loader)",
                    len(_feat_reset.columns), len(_lbl_reset.columns), merged.num_columns,
                )
            finally:
                _con_dist.unregister("_feat")
                _con_dist.unregister("_lbl")
                _con_dist.close()
        except Exception:
            import pandas as _pd_dist
            logger.warning("DuckDB POSITIONAL JOIN unavailable, falling back to pd.concat (distillation loader)")
            merged = _pd_dist.concat([feature_df, label_df], axis=1)
        from ..data.dataloader import build_ple_dataloader, FeatureColumnSpec
        spec = FeatureColumnSpec(static_features=feature_columns)
        label_map = {t.name: t.label_col for t in task_specs}
        loader = build_ple_dataloader(
            df=merged,
            feature_spec=spec,
            label_columns=label_map,
            batch_size=self.config.training.batch_size,
            shuffle=False,
        )
        soft_labels = trainer.generate_soft_labels(loader)

        features_np = feature_df[feature_columns].values
        hard_labels = {}
        for t in task_specs:
            if t.label_col in label_df.columns:
                hard_labels[t.name] = label_df[t.label_col].values

        trainer.train_students(features_np, hard_labels)

        out_dist = Path(output_dir) / "distillation"
        out_dist.mkdir(parents=True, exist_ok=True)

        fidelity_results = {}
        try:
            from ..training.distillation_validator import DistillationValidator
            validator = DistillationValidator()
            for task_name in trainer._students:
                t_spec = self.config.tasks[0]
                for t in self.config.tasks:
                    if t.name == task_name:
                        t_spec = t
                        break
                teacher_preds = soft_labels.get(task_name)
                student_preds = trainer.predict(task_name, features_np)
                if teacher_preds is not None and student_preds is not None:
                    try:
                        result = validator.validate_task(
                            task_name=task_name,
                            task_type=t_spec.type,
                            teacher_preds=teacher_preds,
                            student_preds=student_preds,
                            labels=hard_labels.get(task_name),
                        )
                        fidelity_results[task_name] = {
                            "passed": result.passed,
                            "metrics": result.metrics,
                        }
                    except Exception as e:
                        fidelity_results[task_name] = {
                            "passed": False, "error": str(e),
                        }
        except ImportError:
            logger.info("[Distillation] Fidelity validator not available, skipping.")

        trainer.save_students(str(out_dist), fidelity_results=fidelity_results or None)

        if fidelity_results:
            fidelity_path = out_dist / "fidelity_report.json"
            with open(fidelity_path, "w") as f:
                json.dump(fidelity_results, f, indent=2, default=str)

        elapsed = time.time() - stage_start
        logger.info("[Distillation] Complete in %.1fs", elapsed)

        return {
            "status": "completed",
            "num_students": len(trainer._students),
            "tasks": list(trainer._students.keys()),
            "fidelity": fidelity_results,
            "time_seconds": round(elapsed, 2),
            "output_dir": str(out_dist),
        }

    # ==================================================================
    # Serving preparation (Stage 9.5, used by run_full)
    # ==================================================================

    def _prepare_serving(
        self,
        feature_df: "pd.DataFrame",
        output_dir: str,
    ) -> Dict[str, Any]:
        """Build the context vector store from feature embeddings."""
        import numpy as np

        stage_start = time.time()
        logger.info("[Serving] Preparing serving artifacts...")

        from core.recommendation.reason.context_store import ContextVectorStore

        out = Path(output_dir)
        store_path = out / "serving" / "context_store"
        store_path.mkdir(parents=True, exist_ok=True)

        id_cols = list(getattr(self.config.features, "id_cols", None) or [])
        id_col = id_cols[0] if id_cols else None

        if id_col in feature_df.columns:
            customer_ids = feature_df[id_col].values
        else:
            customer_ids = np.arange(len(feature_df))

        numeric_cols = feature_df.select_dtypes(include=[np.number]).columns.tolist()
        numeric_cols = [c for c in numeric_cols if c != id_col]

        if not numeric_cols:
            return {"status": "skipped", "reason": "no_numeric_features"}

        feature_vectors = feature_df[numeric_cols].values.astype(np.float32)
        feature_vectors = np.nan_to_num(feature_vectors, nan=0.0)

        serving_cfg = self._config_to_dict().get("serving_prep", {})
        cs_cfg = serving_cfg.get("context_store", {})
        backend = cs_cfg.get("backend", "auto")

        store = ContextVectorStore(store_path=str(store_path), backend=backend)

        customer_metadata: Dict[str, Dict] = {}
        for i, cid in enumerate(customer_ids):
            customer_metadata[str(cid)] = {
                "feature_dim": len(numeric_cols),
                "index": int(i),
            }

        store.build(
            customer_ids=customer_ids,
            feature_vectors=feature_vectors,
            metadata=customer_metadata,
        )
        store.save()

        stats = store.get_stats()
        elapsed = time.time() - stage_start

        return {
            "status": "completed",
            "store_path": str(store_path),
            "backend": store.backend,
            "num_customers": stats.get("num_customers", 0),
            "vector_dim": len(numeric_cols),
            "time_seconds": round(elapsed, 2),
        }

    # ==================================================================
    # Stage 10: CPE + Agentic Reason Generation (used by run_full)
    # ==================================================================

    def _run_stage10_cpe_reason(
        self,
        output_dir: str,
    ) -> Dict[str, Any]:
        """Counterfactual policy evaluation and agentic reason generation."""
        import numpy as np

        stage_start = time.time()
        logger.info("[Stage 10] Starting CPE + reason generation...")

        out = Path(output_dir)
        serving_dir = out / "serving"
        serving_dir.mkdir(parents=True, exist_ok=True)

        artifacts: Dict[str, Any] = {"status": "completed"}

        # CPE Evaluation
        try:
            from core.evaluation.counterfactual import CounterfactualEvaluator

            cpe_cfg = self._config_to_dict().get("serving_prep", {}).get("cpe", {})
            n_bootstrap = cpe_cfg.get("n_bootstrap", 500)
            clip_max = cpe_cfg.get("clip_max", 100.0)

            evaluator = CounterfactualEvaluator(
                propensity_clip_range=(0.01, clip_max),
                n_bootstrap=n_bootstrap,
                output_dir=str(serving_dir),
            )

            rng = np.random.default_rng(42)
            n_eval = cpe_cfg.get("n_eval_samples", 1000)

            rewards = rng.binomial(1, 0.05, size=n_eval).astype(np.float64)
            logging_probs = rng.uniform(0.01, 0.3, size=n_eval)
            new_probs = rng.uniform(0.01, 0.3, size=n_eval)
            baseline = cpe_cfg.get("baseline_value", 0.04)

            cpe_results = evaluator.evaluate_all(
                rewards=rewards,
                logging_probs=logging_probs,
                new_probs=new_probs,
                baseline=baseline,
            )

            cpe_path = serving_dir / "cpe_evaluation.json"
            cpe_data = {name: r.to_dict() for name, r in cpe_results.items()}
            with open(cpe_path, "w") as f:
                json.dump(cpe_data, f, indent=2, default=str)

            artifacts["cpe_path"] = str(cpe_path)
            artifacts["cpe_estimators"] = list(cpe_results.keys())
        except Exception as e:
            logger.warning("[Stage 10] CPE evaluation failed: %s", e)
            artifacts["cpe_error"] = str(e)

        # Agentic Reason Generation (sample)
        try:
            from core.recommendation.reason.template_engine import TemplateEngine
            from core.recommendation.reason.agentic_orchestrator import (
                AgenticReasonOrchestrator,
            )

            reason_cfg = self._config_to_dict().get("serving_prep", {}).get("reason", {})
            te_config = self.config.__dict__ if hasattr(self.config, "__dict__") else {}
            template_engine = TemplateEngine(config=te_config)

            orchestrator = AgenticReasonOrchestrator(
                template_engine=template_engine,
                llm_provider=None,
                self_checker=None,
                config=te_config,
            )

            sample_size = reason_cfg.get("sample_size", 10)
            sample_attributions = []
            for i in range(sample_size):
                sample_attributions.append({
                    "customer_id": f"sample_cust_{i:04d}",
                    "item_id": f"sample_item_{i % 3:02d}",
                    "ig_top_features": [
                        (f"feature_{j}", float(np.random.uniform(0.01, 0.5)))
                        for j in range(3)
                    ],
                })

            sample_product_info: Dict[str, Dict[str, Any]] = {
                f"sample_item_{i:02d}": {
                    "name": f"Sample Product {i}",
                    "primary_category": "general",
                }
                for i in range(3)
            }
            sample_segments = {
                f"sample_cust_{i:04d}": "WARMSTART"
                for i in range(sample_size)
            }

            batch_result = orchestrator.run_full(
                ig_attributions=sample_attributions,
                product_info=sample_product_info,
                segments=sample_segments,
            )

            reason_path = serving_dir / "reason_generation_sample.json"
            with open(reason_path, "w") as f:
                json.dump(
                    {
                        "sample_size": sample_size,
                        "l1_count": len(batch_result.l1_results),
                        "l2a_count": batch_result.l2a_count,
                        "l1_sample": batch_result.l1_results[:5],
                    },
                    f, indent=2, default=str,
                )
            artifacts["reason_sample_path"] = str(reason_path)

            quality_path = serving_dir / "agentic_quality_report.json"
            with open(quality_path, "w") as f:
                json.dump(batch_result.quality_report, f, indent=2, default=str)
            artifacts["quality_report_path"] = str(quality_path)

        except Exception as e:
            logger.warning("[Stage 10] Reason generation failed: %s", e)
            artifacts["reason_error"] = str(e)

        elapsed = time.time() - stage_start
        artifacts["time_seconds"] = round(elapsed, 2)
        return artifacts

    # ==================================================================
    # Schema classification (used by run_full if needed)
    # ==================================================================

    def _classify_schema(self, df: "pd.DataFrame") -> Dict[str, List[str]]:
        """Classify DataFrame columns into numeric, categorical, and sequence."""
        import numpy as np

        try:
            from .schema_classifier import SchemaClassifier
            classifier = SchemaClassifier(self.config)
            return classifier.classify(df)
        except (ImportError, ModuleNotFoundError):
            pass

        schema: Dict[str, List[str]] = {
            "state": [], "snapshot": [], "timeseries": [],
            "hierarchy": [], "item": [],
        }

        if self.config.features.numeric or self.config.features.categorical:
            schema["state"] = list(self.config.features.numeric)
            schema["item"] = list(self.config.features.categorical)
            schema["timeseries"] = list(self.config.features.sequence)
            return schema

        for col in df.columns:
            dtype = df[col].dtype
            if np.issubdtype(dtype, np.number):
                nunique = df[col].nunique()
                if nunique <= 2:
                    unique_vals = set(df[col].dropna().unique())
                    if unique_vals.issubset({0, 1, 0.0, 1.0, True, False}):
                        schema["item"].append(col)
                        continue
                schema["state"].append(col)
            elif dtype == object or str(dtype) == "category":
                sample = df[col].dropna().iloc[0] if len(df[col].dropna()) > 0 else None
                if isinstance(sample, (list, np.ndarray)):
                    schema["snapshot"].append(col)
                else:
                    schema["state"].append(col)
            else:
                schema["state"].append(col)

        return schema

    # ==================================================================
    # Helpers
    # ==================================================================

    def _config_to_dict(self) -> dict:
        """Convert PipelineConfig to a plain dict for adapter consumption."""
        data_dict: Dict[str, Any] = {
            "source": self.config.data.source,
            "format": self.config.data.format,
            "train_split": self.config.data.train_split,
            "backend": self.config.data.backend,
            "train_path": self.config.data.train_path,
            "s3_path": self.config.data.s3_path,
            "parquet_file": self.config.data.parquet_file,
        }

        # Adapter-required identifiers. These live on DataSpec but were
        # silently dropped in the dict export, so SantanderAdapter could
        # not see ``id_col`` and aborted with "data.id_col must be
        # specified". Forward them when present.
        if getattr(self.config.data, "id_col", None):
            data_dict["id_col"] = self.config.data.id_col
        if getattr(self.config.data, "total_rows", None) is not None:
            data_dict["total_rows"] = self.config.data.total_rows

        if getattr(self.config.data, "temporal_split", None):
            data_dict["temporal_split"] = self.config.data.temporal_split
        if getattr(self.config.data, "preprocessing", None):
            data_dict["preprocessing"] = self.config.data.preprocessing

        return {
            "task_name": self.config.task_name,
            "data": data_dict,
            "preprocessing": getattr(self.config.data, "preprocessing", None) or {},
            "features": {
                "numeric": list(self.config.features.numeric),
                "categorical": list(self.config.features.categorical),
                "sequence": list(self.config.features.sequence),
            },
            "task_relationships": getattr(self.config, "task_relationships", []),
            "logit_transfer_strength": getattr(self.config, "logit_transfer_strength", 0.5),
            "model": {
                k: v for k, v in vars(self.config.model).items()
            } if self.config.model else {},
            "adatt": getattr(self.config, "adatt", {}),
        }


# ======================================================================
# GenericAdapter (fallback when no adapter is registered)
# ======================================================================

class _GenericAdapter(DataAdapter):
    """Fallback adapter that reads a single file from config.data.source.

    This adapter is used when no adapter name is specified in config.
    It reads the data source directly (CSV or Parquet) and returns it
    as the "main" DataFrame.
    """

    def __init__(self, config: dict) -> None:
        self.config = config
        self._metadata = None

    def load_raw(self) -> Dict[str, "pd.DataFrame"]:
        """Load raw data from the configured source path."""
        import pandas as pd
        from .adapter import AdapterMetadata

        data_cfg = self.config.get("data", {})
        source = data_cfg.get("source", "")
        fmt = data_cfg.get("format", "parquet")

        if not source:
            source = data_cfg.get("train_path") or data_cfg.get("s3_path") or ""

        if not source:
            raise ValueError(
                "No data source specified. Set data.source, data.train_path, "
                "or data.s3_path in config."
            )

        logger.info("[GenericAdapter] Loading '%s' (format=%s)", source, fmt)

        if fmt == "parquet":
            import duckdb as _ddb_generic
            _con_g = _ddb_generic.connect()
            try:
                # NOTE: pandas required — initial materialization; caller uses pandas
                # ops (sentinel replace, LabelEncoder, .select_dtypes, etc.) in Stages 2–6.
                df = _con_g.execute(f"SELECT * FROM '{source}'").df()
            finally:
                _con_g.close()
            _backend_used = "duckdb"
        elif fmt == "csv":
            df = pd.read_csv(source)
            _backend_used = "pandas"
        else:
            raise ValueError(f"Unsupported data format: {fmt}")

        self._metadata = AdapterMetadata(
            num_entities=len(df),
            num_raw_rows=len(df),
            source_files=[source],
            backend_used=_backend_used,
        )

        logger.info("[GenericAdapter] Loaded %d rows x %d cols", len(df), len(df.columns))
        return {"main": df}

    @property
    def metadata(self) -> Any:
        if self._metadata is None:
            raise RuntimeError("Call load_raw() before accessing metadata")
        return self._metadata
