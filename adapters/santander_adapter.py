"""
Santander Customer Transactions Adapter
========================================

Loads the santander_final.parquet dataset (941K users, 89 columns).
Already user-level — no aggregation needed.

Input:
    data/synthetic/santander_final.parquet  (or overridden by config)

Output (from load_raw):
    {"main": DataFrame}  -- 941,132 rows × 89 columns

Phase 0 Processing Job additionally runs feature generators (TDA, Graph,
HMM, Mamba, GMM, Model-derived) to enrich the parquet before training.
"""

from __future__ import annotations

import logging
import time
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from core.pipeline.adapter import AdapterMetadata, AdapterRegistry, DataAdapter

logger = logging.getLogger(__name__)


# ======================================================================
# Feature generation for Phase 0
# ======================================================================

def _resolve_santander_columns(df: pd.DataFrame):
    """Identify column groups in the Santander dataset."""
    product_cols = sorted([c for c in df.columns if c.startswith("prod_")])
    synth_cols = sorted([c for c in df.columns if c.startswith("synth_")])
    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    # Exclude id-like and label-like columns from numeric features
    _exclude = {"customer_id"}
    numeric_cols = [c for c in numeric_cols if c not in _exclude]
    return product_cols, synth_cols, numeric_cols


def _run_santander_generators(df: pd.DataFrame) -> pd.DataFrame:
    """Run TDA, Graph, HMM, Mamba, GMM, and Model-derived generators.

    Each generator is wrapped in a try/except so a single failure does not
    block the others.  Fallback zeros are produced when a generator fails.

    Returns the original DataFrame with generated feature columns appended.
    """
    product_cols, synth_cols, numeric_cols = _resolve_santander_columns(df)
    # Use product + synth columns as the base feature set for most generators
    base_cols = product_cols + synth_cols
    if not base_cols:
        base_cols = numeric_cols[:40]  # fallback to first 40 numeric

    generated_frames: List[pd.DataFrame] = []
    gen_summary: Dict[str, int] = {}

    # ------------------------------------------------------------------
    # 1. TDA Global (population topology from product holdings)
    # ------------------------------------------------------------------
    t0 = time.time()
    try:
        from core.feature.generators.tda import TDAGlobalGenerator
        tda_global = TDAGlobalGenerator(
            entity_column="customer_id",
            input_columns=product_cols or base_cols[:24],
            prefix="tda_global",
            max_homology_dim=1,
        )
        tda_global.fit(df)
        tda_global_feat = tda_global.generate(df)
        # Convert to pandas if needed
        if not isinstance(tda_global_feat, pd.DataFrame):
            tda_global_feat = pd.DataFrame(tda_global_feat)
        tda_global_feat.index = df.index
        generated_frames.append(tda_global_feat)
        gen_summary["tda_global"] = len(tda_global_feat.columns)
        logger.info("TDA Global: %d features in %.1fs",
                     len(tda_global_feat.columns), time.time() - t0)
    except Exception as e:
        logger.warning("TDA Global generation failed: %s", e)
        n_cols = 14  # (max_homology_dim+1) * 7 stats = 14
        fallback = pd.DataFrame(
            np.zeros((len(df), n_cols), dtype=np.float32),
            columns=[f"tda_global_{i:03d}" for i in range(n_cols)],
            index=df.index,
        )
        generated_frames.append(fallback)
        gen_summary["tda_global"] = n_cols

    # ------------------------------------------------------------------
    # 2. TDA Local (per-entity topology)
    # ------------------------------------------------------------------
    t0 = time.time()
    try:
        from core.feature.generators.tda import TDALocalGenerator
        tda_local = TDALocalGenerator(
            entity_column="customer_id",
            input_columns=product_cols or base_cols[:24],
            prefix="tda_local",
            max_homology_dim=1,
            n_jobs=1,  # conservative for Processing Job memory
        )
        tda_local.fit(df)
        tda_local_feat = tda_local.generate(df)
        if not isinstance(tda_local_feat, pd.DataFrame):
            tda_local_feat = pd.DataFrame(tda_local_feat)
        tda_local_feat.index = df.index
        generated_frames.append(tda_local_feat)
        gen_summary["tda_local"] = len(tda_local_feat.columns)
        logger.info("TDA Local: %d features in %.1fs",
                     len(tda_local_feat.columns), time.time() - t0)
    except Exception as e:
        logger.warning("TDA Local generation failed: %s", e)
        n_cols = 14
        fallback = pd.DataFrame(
            np.zeros((len(df), n_cols), dtype=np.float32),
            columns=[f"tda_local_{i:03d}" for i in range(n_cols)],
            index=df.index,
        )
        generated_frames.append(fallback)
        gen_summary["tda_local"] = n_cols

    # ------------------------------------------------------------------
    # 3. Graph Collaborative (customer-product bipartite graph)
    # ------------------------------------------------------------------
    t0 = time.time()
    try:
        from core.feature.generators.graph import GraphEmbeddingGenerator
        graph_gen = GraphEmbeddingGenerator(
            embedding_dim=18,
            entity_column="customer_id",
            feature_columns=product_cols or base_cols[:24],
            prefix="graph_collaborative",
            prefer_gpu=False,
            num_epochs=15,  # fewer epochs for Processing Job speed
            k_neighbors=8,
        )
        graph_gen.fit(df)
        graph_feat = graph_gen.generate(df)
        if not isinstance(graph_feat, pd.DataFrame):
            graph_feat = pd.DataFrame(graph_feat)
        graph_feat.index = df.index
        generated_frames.append(graph_feat)
        gen_summary["graph_collaborative"] = len(graph_feat.columns)
        logger.info("Graph Collaborative: %d features in %.1fs",
                     len(graph_feat.columns), time.time() - t0)
    except Exception as e:
        logger.warning("Graph generation failed: %s", e)
        n_cols = 20  # embedding_dim(18) + norm + depth
        fallback = pd.DataFrame(
            np.zeros((len(df), n_cols), dtype=np.float32),
            columns=[f"graph_collaborative_{i:03d}" for i in range(n_cols)],
            index=df.index,
        )
        generated_frames.append(fallback)
        gen_summary["graph_collaborative"] = n_cols

    # ------------------------------------------------------------------
    # 4. HMM States (triple-mode: journey, lifecycle, behavior)
    # ------------------------------------------------------------------
    t0 = time.time()
    try:
        from core.feature.generators.hmm import HMMFeatureGenerator
        hmm_gen = HMMFeatureGenerator(
            modes=["journey", "lifecycle", "behavior"],
            sequence_columns=synth_cols or base_cols[:20],
            prefix="hmm",
        )
        hmm_gen.fit(df)
        hmm_feat = hmm_gen.generate(df)
        if not isinstance(hmm_feat, pd.DataFrame):
            hmm_feat = pd.DataFrame(hmm_feat)
        hmm_feat.index = df.index
        generated_frames.append(hmm_feat)
        gen_summary["hmm_states"] = len(hmm_feat.columns)
        logger.info("HMM States: %d features in %.1fs",
                     len(hmm_feat.columns), time.time() - t0)
    except Exception as e:
        logger.warning("HMM generation failed: %s", e)
        # 3 modes * (1 state_id + n_states probs + 1 dwell + 1 entropy)
        # journey(5): 8, lifecycle(5): 8, behavior(6): 9 = 25
        n_cols = 25
        fallback = pd.DataFrame(
            np.zeros((len(df), n_cols), dtype=np.float32),
            columns=[f"hmm_{i:03d}" for i in range(n_cols)],
            index=df.index,
        )
        generated_frames.append(fallback)
        gen_summary["hmm_states"] = n_cols

    # ------------------------------------------------------------------
    # 5. Mamba Temporal (SSM embeddings from synth features as pseudo-sequences)
    # ------------------------------------------------------------------
    t0 = time.time()
    try:
        from core.feature.generators.mamba import MambaFeatureGenerator
        mamba_gen = MambaFeatureGenerator(
            output_dim=50,
            d_model=128,  # smaller for speed
            seq_len=30,
            entity_column="customer_id",
            time_column="snapshot_date",
            feature_columns=synth_cols or base_cols[:20],
            prefix="mamba",
            prefer_gpu=False,
            num_epochs=10,
        )
        mamba_gen.fit(df)
        mamba_feat = mamba_gen.generate(df)
        if not isinstance(mamba_feat, pd.DataFrame):
            mamba_feat = pd.DataFrame(mamba_feat)
        mamba_feat.index = df.index
        generated_frames.append(mamba_feat)
        gen_summary["mamba_temporal"] = len(mamba_feat.columns)
        logger.info("Mamba Temporal: %d features in %.1fs",
                     len(mamba_feat.columns), time.time() - t0)
    except Exception as e:
        logger.warning("Mamba generation failed: %s", e)
        n_cols = 50
        fallback = pd.DataFrame(
            np.zeros((len(df), n_cols), dtype=np.float32),
            columns=[f"mamba_{i:03d}" for i in range(n_cols)],
            index=df.index,
        )
        generated_frames.append(fallback)
        gen_summary["mamba_temporal"] = n_cols

    # ------------------------------------------------------------------
    # 6. GMM Clustering (dynamic k via BIC)
    # ------------------------------------------------------------------
    t0 = time.time()
    try:
        from core.feature.generators.gmm import GMMClusteringGenerator
        gmm_gen = GMMClusteringGenerator(
            feature_columns=base_cols,
            prefix="gmm",
        )
        gmm_gen.fit(df)
        gmm_feat = gmm_gen.generate(df)
        if not isinstance(gmm_feat, pd.DataFrame):
            gmm_feat = pd.DataFrame(gmm_feat)
        gmm_feat.index = df.index
        generated_frames.append(gmm_feat)
        gen_summary["gmm_clustering"] = len(gmm_feat.columns)
        logger.info("GMM Clustering: %d features in %.1fs",
                     len(gmm_feat.columns), time.time() - t0)
    except Exception as e:
        logger.warning("GMM generation failed: %s", e)
        n_cols = 22  # default K=20 -> cluster_id + 20 probs + entropy
        fallback = pd.DataFrame(
            np.zeros((len(df), n_cols), dtype=np.float32),
            columns=[f"gmm_{i:03d}" for i in range(n_cols)],
            index=df.index,
        )
        generated_frames.append(fallback)
        gen_summary["gmm_clustering"] = n_cols

    # ------------------------------------------------------------------
    # 7. Model-Derived Features (HMM summary + Bandit + LNN = 27D)
    # ------------------------------------------------------------------
    t0 = time.time()
    try:
        from core.feature.generators.model_features import ModelFeaturesGenerator
        model_gen = ModelFeaturesGenerator(
            feature_columns=base_cols,
            engagement_columns=product_cols,
            temporal_columns=synth_cols[:2] if len(synth_cols) >= 2 else base_cols[:2],
            prefix="model_derived",
        )
        model_gen.fit(df)
        model_feat = model_gen.generate(df)
        if not isinstance(model_feat, pd.DataFrame):
            model_feat = pd.DataFrame(model_feat)
        model_feat.index = df.index
        generated_frames.append(model_feat)
        gen_summary["model_derived"] = len(model_feat.columns)
        logger.info("Model Derived: %d features in %.1fs",
                     len(model_feat.columns), time.time() - t0)
    except Exception as e:
        logger.warning("Model-derived generation failed: %s", e)
        n_cols = 27
        fallback = pd.DataFrame(
            np.zeros((len(df), n_cols), dtype=np.float32),
            columns=[f"model_derived_{i:03d}" for i in range(n_cols)],
            index=df.index,
        )
        generated_frames.append(fallback)
        gen_summary["model_derived"] = n_cols

    # ------------------------------------------------------------------
    # Merge all generated features
    # ------------------------------------------------------------------
    if generated_frames:
        all_gen = pd.concat(generated_frames, axis=1)
        df = pd.concat([df, all_gen], axis=1)
        total_new = sum(gen_summary.values())
        logger.info(
            "Feature generation complete: %d new columns from %d generators. "
            "Total columns: %d. Summary: %s",
            total_new, len(gen_summary), len(df.columns), gen_summary,
        )

    return df


@AdapterRegistry.register("santander")
class SantanderAdapter(DataAdapter):
    """DataAdapter for the Santander synthetic dataset.

    The dataset is already at user-level granularity with:
    - Scalar numeric columns (age, income, tenure, etc.)
    - String categorical columns (gender, segment, country, channel,
      age_group, income_group)
    - List-type sequence columns (txn_amount_seq, txn_count_seq,
      txn_mcc_seq, seq_saving, seq_current, seq_num_products,
      seq_acquisitions, seq_churns, etc.)
    - Binary product holding flags (prod_*)
    - Synthetic derived features (synth_*)
    """

    def load_raw(self) -> Dict[str, pd.DataFrame]:
        """Load santander_final.parquet. Already user-level, minimal preprocessing."""
        backend = self._select_backend()
        source = self.config.get("data", {}).get(
            "source", "data/synthetic/santander_final.parquet"
        )

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
            id_col="customer_id",
            entity_granularity="user",
            num_entities=len(df),
            num_raw_rows=len(df),
            source_files=[str(source)],
            backend_used=backend,
        )

        logger.info(
            "SantanderAdapter: loaded %d rows × %d cols (backend=%s)",
            len(df),
            len(df.columns),
            backend,
        )

        return {"main": df}


# ======================================================================
# Standalone entry point for SageMaker Processing Job
# ======================================================================

if __name__ == "__main__":
    import argparse
    import json
    import os
    import sys
    import shutil

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    parser = argparse.ArgumentParser(description="Santander data adapter")
    parser.add_argument("--input-dir", default="/opt/ml/processing/input/raw")
    parser.add_argument("--output-dir", default="/opt/ml/processing/output")
    parser.add_argument("--pipeline", default="", help="Pipeline config path (unused, for compat)")
    parser.add_argument("--stages", default="1-6", help="Stages to run (unused, for compat)")
    cli_args = parser.parse_args()

    input_dir = cli_args.input_dir
    output_dir = cli_args.output_dir

    # Find the parquet file in input dir
    import glob
    parquet_files = glob.glob(os.path.join(input_dir, "*.parquet"))
    if not parquet_files:
        # Try subdirectories
        parquet_files = glob.glob(os.path.join(input_dir, "**", "*.parquet"), recursive=True)

    if not parquet_files:
        logger.error("No parquet files found in %s", input_dir)
        sys.exit(1)

    source = parquet_files[0]
    logger.info("Found source: %s", source)

    # Load via adapter
    config = {"data": {"source": source, "backend": ["duckdb", "pandas"]}}
    adapter = SantanderAdapter(config)
    raw_data = adapter.load_raw()
    df = raw_data["main"]

    # --- HIGH-2: Data quality gate ---
    quality = {
        "total_rows": len(df),
        "total_columns": len(df.columns),
        "null_rates": {col: float(df[col].isna().mean()) for col in df.columns if df[col].isna().any()},
        "zero_variance_columns": [col for col in df.select_dtypes("number").columns if df[col].std() == 0],
        "duplicate_rows": int(df.duplicated().sum()),
    }
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, "quality_gate_report.json"), "w") as f:
        json.dump(quality, f, indent=2)
    logger.info("Quality gate report saved: %d rows, %d cols, %d zero-variance cols",
                quality["total_rows"], quality["total_columns"],
                len(quality["zero_variance_columns"]))

    # --- Feature generation (TDA, Graph, HMM, Mamba, GMM, Model-derived) ---
    logger.info("Starting feature generation on %d rows ...", len(df))
    t_gen_start = time.time()
    df = _run_santander_generators(df)
    logger.info("Feature generation finished in %.1fs. Shape: %s",
                time.time() - t_gen_start, df.shape)

    # Save to output dir
    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, "santander_final.parquet")
    df.to_parquet(out_path, index=False)
    logger.info("Saved %d rows to %s", len(df), out_path)

    # --- HIGH-1: Feature statistics ---
    numeric = df.select_dtypes(include="number")
    stats = {}
    for col in numeric.columns:
        stats[col] = {
            "mean": float(numeric[col].mean()),
            "std": float(numeric[col].std()),
            "min": float(numeric[col].min()),
            "max": float(numeric[col].max()),
            "null_pct": float(numeric[col].isna().mean()),
            "nunique": int(numeric[col].nunique()),
        }
    with open(os.path.join(output_dir, "feature_stats.json"), "w") as f:
        json.dump(stats, f, indent=2)
    logger.info("Feature stats saved: %d numeric columns", len(stats))

    # --- HIGH-1: Label statistics ---
    label_cols = [c for c in df.columns if c in ["has_nba", "churn_signal", "product_stability", "nba_label"]]
    label_stats = {}
    for col in label_cols:
        if df[col].dtype in [np.int32, np.int64]:
            label_stats[col] = {str(k): int(v) for k, v in df[col].value_counts().to_dict().items()}
        elif df[col].dtype == float:
            label_stats[col] = {"mean": float(df[col].mean()), "std": float(df[col].std())}
    if label_stats:
        with open(os.path.join(output_dir, "label_stats.json"), "w") as f:
            json.dump(label_stats, f, indent=2)
        logger.info("Label stats saved: %d label columns", len(label_stats))

    # Save metadata
    meta = adapter.metadata
    with open(os.path.join(output_dir, "adapter_metadata.json"), "w") as f:
        json.dump(meta.__dict__, f, indent=2, default=str)

    logger.info("Phase 0 complete. Output: %s", output_dir)
