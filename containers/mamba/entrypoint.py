"""SageMaker Training Job entry point — Mamba GPU pre-compute.

Produces the per-customer Mamba temporal embedding parquet that
``MambaFeatureGenerator(cached_embedding_uri=...)`` consumes during
the CPU Phase 0 run. The CPU Phase 0 cannot fit Mamba in-process
because mamba_ssm requires CUDA wheels that fail to build on
ml.m5.* (CPU-only). So Mamba moves to a separate GPU job.

Pipeline
--------
1. Load the same santander parquet Phase 0 reads (mounted at
   ``/opt/ml/input/data/raw``).
2. Build per-customer sequences from ``synth_*`` columns (the same
   filter ``feature_groups.yaml::mamba_temporal::input_filter``
   declares).
3. Fit + generate via :class:`MambaFeatureGenerator` (CUDA mamba_ssm
   when available, pure-torch fallback otherwise).
4. Write ``embedding.parquet`` to ``/opt/ml/model`` so SageMaker
   packages it into ``model.tar.gz``. The parquet has one row per
   customer with columns ``[customer_id, mamba_d0, ..., mamba_d49]``.

Hyperparameters (all string per SageMaker convention)
* ``config`` — pipeline yaml path inside the container
* ``dataset_config`` — dataset yaml path
* ``output_dim`` (default 50) — final embedding dim
* ``entity_column`` (default ``customer_id``)
* ``feature_filter_prefix`` (default ``synth_``) — column prefix to
  pick as Mamba input features
"""
from __future__ import annotations

import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
)
logger = logging.getLogger("mamba.precompute")


def _hyperparameters() -> Dict[str, Any]:
    sm_hps = os.environ.get("SM_HPS")
    if sm_hps:
        try:
            return json.loads(sm_hps)
        except Exception:
            logger.warning("Could not parse SM_HPS; falling back to HP file")
    hp_file = "/opt/ml/input/config/hyperparameters.json"
    if os.path.exists(hp_file):
        with open(hp_file, encoding="utf-8") as f:
            return json.load(f)
    return {}


def _resolve_raw_input(explicit: str = "") -> str:
    if explicit:
        return explicit
    for key in ("SM_CHANNEL_RAW", "SM_CHANNEL_TRAIN"):
        val = os.environ.get(key)
        if val:
            return val
    return "/opt/ml/input/data/raw"


def _pick_parquet(raw_dir: str) -> str:
    p = Path(raw_dir)
    if p.is_file():
        return str(p)
    if not p.is_dir():
        raise FileNotFoundError(f"Raw input path not found: {raw_dir}")
    parquets = sorted(p.rglob("*.parquet"))
    if not parquets:
        raise FileNotFoundError(f"No *.parquet under {raw_dir}")
    return str(parquets[0])


def _select_input_columns(
    parquet_path: str, entity_col: str, feature_prefix: str,
) -> List[str]:
    """Return the column list Mamba should read.

    Mamba operates on numeric ``feature_prefix*`` columns (the
    Santander synthetic features). We additionally pull
    ``entity_column`` for the per-customer grouping.
    """
    import duckdb
    con = duckdb.connect()
    try:
        rows = con.execute(
            f"DESCRIBE SELECT * FROM read_parquet('{parquet_path}') LIMIT 0"
        ).fetchall()
    finally:
        con.close()
    cols = [r[0] for r in rows]
    if entity_col not in cols:
        raise RuntimeError(
            f"entity_column '{entity_col}' not in parquet schema "
            f"({len(cols)} cols)"
        )
    feat = [c for c in cols if c.startswith(feature_prefix)]
    if not feat:
        raise RuntimeError(
            f"No columns matching prefix '{feature_prefix}'"
        )
    logger.info(
        "Mamba input: entity=%s, %d feature cols matching prefix '%s'",
        entity_col, len(feat), feature_prefix,
    )
    return [entity_col] + feat


def _install_gpu_wheels() -> None:
    """Install mamba_ssm + causal-conv1d on first start.

    Top-level ``requirements.txt`` deliberately omits these because
    the CPU Phase 0 instance can't build their CUDA wheels. Here we
    are on a GPU host (g4dn / g5), nvcc is in the DLC, so the build
    succeeds.
    """
    import subprocess
    pkgs = ["causal-conv1d>=1.1,<2.0", "mamba-ssm>=1.2,<3.0"]
    logger.info("Installing GPU-only wheels: %s", pkgs)
    try:
        subprocess.run(
            ["/opt/conda/bin/pip", "install", "--no-cache-dir", *pkgs],
            check=True,
        )
        logger.info("GPU wheels installed")
    except subprocess.CalledProcessError as exc:
        logger.warning(
            "mamba_ssm install failed (%s) — Mamba will fall back to "
            "pure-torch SSM scan. This is expected on CPU instances; "
            "on GPU it usually means nvcc is missing.",
            exc,
        )


def main() -> None:
    start = time.time()
    hp = _hyperparameters()
    logger.info("Mamba precompute hyperparameters: %s", hp)
    _install_gpu_wheels()

    raw_dir = _resolve_raw_input(hp.get("raw_dir", ""))
    raw_parquet = _pick_parquet(raw_dir)
    output_dir = os.environ.get("SM_MODEL_DIR", "/opt/ml/model")
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    output_dim = int(hp.get("output_dim", 50))
    entity_col = str(hp.get("entity_column", "customer_id"))
    feature_prefix = str(hp.get("feature_filter_prefix", "synth_"))
    d_model = int(hp.get("d_model", 128))
    num_epochs = int(hp.get("num_epochs", 3))
    base_batch_size = int(hp.get("base_batch_size", 256))
    seq_len = int(hp.get("seq_len", 90))

    needed_cols = _select_input_columns(
        raw_parquet, entity_col, feature_prefix,
    )

    # Pull only the needed columns into pandas via DuckDB.
    import duckdb
    proj = ", ".join(f'"{c}"' for c in needed_cols)
    logger.info("Loading %d columns from %s …", len(needed_cols), raw_parquet)
    _ld = duckdb.connect()
    try:
        df = _ld.execute(
            f"SELECT {proj} FROM read_parquet('{raw_parquet}')"
        ).df()
    finally:
        _ld.close()
    logger.info("Loaded df: shape=%s", df.shape)

    # Run Mamba fit + generate.
    from core.feature.generators.mamba import MambaFeatureGenerator
    feature_columns = [c for c in needed_cols if c != entity_col]
    gen = MambaFeatureGenerator(
        d_model=d_model,
        output_dim=output_dim,
        num_epochs=num_epochs,
        base_batch_size=base_batch_size,
        seq_len=seq_len,
        entity_column=entity_col,
        feature_columns=feature_columns,
        prefer_gpu=True,
    )
    logger.info("Fitting Mamba …")
    gen.fit(df)
    logger.info("Generating embeddings …")
    emb_df = gen.generate(df)
    # emb_df has the 50 mamba_d<i> cols; we want one row per entity.
    # The fit() loop produced a per-entity dict, but generate() returns
    # row-aligned features. For the cached lookup we need one row per
    # *entity*: take the first occurrence.
    emb_df = emb_df.copy()
    emb_df[entity_col] = df[entity_col].values
    per_entity = (
        emb_df.drop_duplicates(subset=[entity_col], keep="first")
              .reset_index(drop=True)
    )
    logger.info(
        "Mamba per-entity embedding shape: %s (deduped from %d rows)",
        per_entity.shape, len(emb_df),
    )

    out_path = os.path.join(output_dir, "embedding.parquet")
    _w = duckdb.connect()
    try:
        _w.register("_emb", per_entity)
        _w.execute(
            f"COPY _emb TO '{out_path}' (FORMAT PARQUET)"
        )
    finally:
        _w.close()
    logger.info(
        "Wrote %s (%.1f MB)",
        out_path, os.path.getsize(out_path) / 1024 / 1024,
    )

    summary = {
        "elapsed_seconds": round(time.time() - start, 2),
        "raw_parquet": raw_parquet,
        "entity_column": entity_col,
        "feature_columns": feature_columns,
        "output_dim": output_dim,
        "rows": len(per_entity),
        "embedding_parquet": out_path,
    }
    with open(os.path.join(output_dir, "mamba_precompute_summary.json"), "w") as f:
        json.dump(summary, f, indent=2, default=str)
    logger.info("Mamba precompute complete in %.1fs", time.time() - start)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        logger.exception("mamba/entrypoint.py failed")
        sys.exit(1)
