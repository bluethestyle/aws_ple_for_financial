"""
SageMaker Training Job entry point for Phase 0 (feature engineering).

Delegates the 9-stage Phase-0 pipeline to
:class:`core.pipeline.runner.PipelineRunner.run`, which is the same
implementation that runs locally for ``outputs/phase0_v12/``.  Wraps
the SageMaker I/O contract around it:

* Input: ingestion parquet mounted at ``SM_CHANNEL_RAW`` (e.g.
  ``/opt/ml/input/data/raw/santander_final.parquet``).
* Output: every artifact (feature matrix, feature_schema.json,
  feature_stats.json, normalizer/, label_stats.json, ...) written
  under ``SM_MODEL_DIR`` (``/opt/ml/model``). SageMaker's training
  toolkit tars ``/opt/ml/model`` into ``output/model.tar.gz`` at end
  of job.  ``submit_pipeline`` (and the Phase 1 trainer) unpacks that
  tarball or points its channel at the pre-extracted prefix, matching
  the behaviour of the legacy ``data/phase0_v12/`` prefix.

Hyperparameters (all values strings per SageMaker convention):

* ``config`` — container-relative path to the common pipeline YAML,
  typically ``configs/pipeline.yaml``.
* ``dataset_config`` — optional dataset-specific YAML, typically
  ``configs/datasets/santander.yaml``.

Pandas is avoided where possible (CLAUDE.md §3.3); the runner itself
uses the project's ``df_backend`` chain (cuDF → DuckDB → pandas) and
not pandas directly.
"""
from __future__ import annotations

import json
import logging
import os
import shutil
import sys
import time
from pathlib import Path
from typing import Any, Dict

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
)
logger = logging.getLogger("phase0.entrypoint")


def _hyperparameters() -> Dict[str, Any]:
    """Load hyperparameters from SM_HPS env var or the local HP file."""
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
    """Resolve the raw ingestion parquet mount point.

    Order of precedence:

    1. ``explicit`` argument (local CLI mode).
    2. ``SM_CHANNEL_RAW`` → the SageMaker input channel.
    3. ``SM_CHANNEL_TRAIN`` → the 'train' channel (legacy fallback).
    4. ``/opt/ml/input/data/raw`` as a last resort.
    """
    if explicit:
        return explicit
    for key in ("SM_CHANNEL_RAW", "SM_CHANNEL_TRAIN"):
        val = os.environ.get(key)
        if val:
            return val
    return "/opt/ml/input/data/raw"


def _pick_parquet_file(raw_dir: str) -> str:
    """Return the single parquet file SageMaker mounted.

    SageMaker's ``S3DataType=S3Prefix`` mode downloads every object
    under a prefix; we accept either one file (parquet) or a directory
    of parquets and return the first we find. The caller merges the
    path into ``config.data.source`` so the runner loads it as its raw
    input.
    """
    p = Path(raw_dir)
    if p.is_file():
        return str(p)
    if not p.is_dir():
        raise FileNotFoundError(f"Raw input path not found: {raw_dir}")
    parquets = sorted(p.rglob("*.parquet"))
    if not parquets:
        raise FileNotFoundError(f"No *.parquet under {raw_dir}")
    return str(parquets[0])


def main() -> None:
    start = time.time()
    hp = _hyperparameters()
    logger.info("Phase 0 hyperparameters: %s", hp)

    # ---- 1. Load pipeline config ------------------------------------
    from containers.path_resolver import resolve_config_path
    from core.pipeline.config import load_config, load_merged_config

    cfg_path = resolve_config_path(
        hp.get("config", "configs/pipeline.yaml"),
    )
    dataset_cfg_path = hp.get("dataset_config", "")
    if dataset_cfg_path:
        dataset_cfg_path = resolve_config_path(dataset_cfg_path)
        config = load_config(cfg_path, dataset_path=dataset_cfg_path)
    else:
        config = load_config(cfg_path)

    # ---- 2. Point data.source at the SageMaker mount ----------------
    raw_dir = _resolve_raw_input()
    raw_parquet = _pick_parquet_file(raw_dir)
    logger.info("Raw parquet: %s", raw_parquet)
    config.data.source = raw_parquet

    # ---- 3. Resolve output dir --------------------------------------
    output_dir = os.environ.get("SM_MODEL_DIR", "/opt/ml/model")
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # ---- 4. Run the 9-stage Phase 0 runner --------------------------
    from core.pipeline.runner import PipelineRunner

    logger.info("Starting Phase 0 (9-stage runner) → %s", output_dir)
    runner = PipelineRunner(config)
    results = runner.run(output_dir=output_dir)

    # ---- 5. Write a small summary next to the artifacts -------------
    summary_path = os.path.join(output_dir, "phase0_summary.json")
    summary: Dict[str, Any] = {
        "elapsed_seconds": round(time.time() - start, 2),
        "raw_parquet": raw_parquet,
        "output_dir": output_dir,
        "stages": list(results.keys()),
    }
    try:
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, default=str)
    except Exception:
        logger.exception("Failed to write phase0_summary.json (non-fatal)")

    # Mirror the parquet into /opt/ml/output/data too if that channel
    # is mounted — SageMaker treats it as the "output" tarball vs
    # /opt/ml/model's "model" tarball. Keeping both lets either
    # consumption path work without post-hoc S3 copies.
    extra_dir = os.environ.get("SM_OUTPUT_DATA_DIR")
    if extra_dir and os.path.isdir(output_dir) and extra_dir != output_dir:
        try:
            Path(extra_dir).mkdir(parents=True, exist_ok=True)
            for child in Path(output_dir).iterdir():
                tgt = Path(extra_dir) / child.name
                if child.is_file():
                    shutil.copy2(child, tgt)
                elif child.is_dir() and not tgt.exists():
                    shutil.copytree(child, tgt)
            logger.info("Also mirrored artifacts to %s", extra_dir)
        except Exception:
            logger.exception("Mirror to SM_OUTPUT_DATA_DIR failed (non-fatal)")

    logger.info("Phase 0 complete in %.1fs", time.time() - start)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        logger.exception("phase0/entrypoint.py failed with unhandled exception")
        sys.exit(1)
