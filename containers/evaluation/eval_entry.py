#!/usr/bin/env python3
"""SageMaker entry point for evaluating a trained PLE checkpoint.

This is the SageMaker-compatible version of ``scripts/eval_checkpoint.py``.
It replicates the same predictor + evaluator flow but reads paths from
SageMaker environment variables instead of CLI arguments.

SageMaker channel layout
------------------------
  SM_CHANNEL_TRAIN   — data directory (*.parquet, feature_schema.json,
                        label_schema.json, split_indices.json)
  SM_CHANNEL_MODEL   — checkpoint directory (best.pt or *.pt)
  SM_HPS             — JSON string with hyperparameters (config, batch_size, …)
  SM_OUTPUT_DATA_DIR — write eval_metrics.json here
  SM_MODEL_DIR       — model output directory (unused for eval)

Hyperparameters (SM_HPS)
------------------------
  config        — container-internal path to common pipeline.yaml
                  (default: configs/pipeline.yaml)
  dataset_config — optional dataset-specific YAML deep-merged on top of config
                  (e.g. configs/datasets/santander.yaml)
  batch_size    — inference batch size (default: 5632)
  num_workers   — DataLoader workers (default: 2 on Linux, 0 on Windows)
  device        — "auto", "cuda", or "cpu" (default: "auto")

Local execution
---------------
When SM_CHANNEL_TRAIN is absent the script falls back to CLI argument
parsing identical to eval_checkpoint.py so it can be run locally for
debugging without a SageMaker job.
"""

from __future__ import annotations

import json
import logging
import os
import platform
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

# Ensure duckdb is installed (mirrors train.py pattern)
try:
    import duckdb  # noqa: F401
except ImportError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "duckdb>=1.0.0"])

os.environ.setdefault("PYTHONIOENCODING", "utf-8")

# When running inside SageMaker the working directory is /opt/ml/code
# where the source_dir was extracted.  Ensure it is importable.
_code_dir = Path("/opt/ml/code")
if _code_dir.exists() and str(_code_dir) not in sys.path:
    sys.path.insert(0, str(_code_dir))

# Also support running from project root locally
_project_root = Path(__file__).resolve().parent.parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

import torch
import yaml

# Force UTF-8 stdout (avoids cp949 errors on Windows dev boxes)
_utf8_stdout = open(sys.stdout.fileno(), mode="w", encoding="utf-8", closefd=False)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(_utf8_stdout)],
)
logger = logging.getLogger("eval-entry")


# ---------------------------------------------------------------------------
# SageMaker environment helpers (mirrors train.py)
# ---------------------------------------------------------------------------

def _parse_hp_value(v: Any) -> Any:
    """Best-effort parse a stringified hyperparameter value."""
    if isinstance(v, str):
        lower = v.lower()
        if lower in ("true", "false"):
            return lower == "true"
        if lower.startswith("[") or lower.startswith("{"):
            try:
                return json.loads(v)
            except (json.JSONDecodeError, ValueError):
                pass
        try:
            return int(v)
        except ValueError:
            pass
        try:
            return float(v)
        except ValueError:
            pass
    return v


def get_hyperparameters() -> Dict[str, Any]:
    """Read hyperparameters from SM_HPS or /opt/ml/input/config/."""
    sm_hps = os.environ.get("SM_HPS")
    if sm_hps:
        return json.loads(sm_hps)

    hp_path = Path("/opt/ml/input/config/hyperparameters.json")
    if hp_path.exists():
        with open(hp_path) as f:
            raw = json.load(f)
        return {k: _parse_hp_value(v) for k, v in raw.items()}

    return {}


def _is_sagemaker() -> bool:
    """Return True when running inside a SageMaker container."""
    return "SM_CHANNEL_TRAIN" in os.environ


def _default_num_workers() -> int:
    """Return a safe default num_workers for the current platform."""
    if platform.system() == "Windows":
        return 0       # Windows multiprocessing in DataLoader is unreliable
    return 2           # Linux (SageMaker) — 2 is safe


# ---------------------------------------------------------------------------
# Checkpoint discovery
# ---------------------------------------------------------------------------

def _find_checkpoint(model_dir: str) -> str:
    """Locate the best checkpoint in *model_dir*.

    Preference order:
      1. ``best.pt``
      2. Any ``*.pt`` file (most recently modified wins)

    Args:
        model_dir: Directory containing the .pt file(s).

    Returns:
        Absolute path to the checkpoint.

    Raises:
        FileNotFoundError: When no .pt file is found.
    """
    model_path = Path(model_dir)

    best = model_path / "best.pt"
    if best.exists():
        logger.info("Using checkpoint: %s", best)
        return str(best)

    candidates = sorted(
        list(model_path.glob("*.pt")) + list(model_path.glob("*.pth")),
        key=lambda p: p.stat().st_mtime, reverse=True,
    )
    if not candidates:
        raise FileNotFoundError(f"No .pt/.pth checkpoint found in {model_dir}")

    logger.info("Using checkpoint (most recent): %s", candidates[0])
    return str(candidates[0])


# ---------------------------------------------------------------------------
# Data loading (mirrors eval_checkpoint.py)
# ---------------------------------------------------------------------------

def _build_val_loader(
    channel_path: Path,
    config: Dict[str, Any],
    feature_schema: Dict[str, Any],
    batch_size: int,
    num_workers: int,
):
    """Build a validation DataLoader from Phase 0 data.

    Reads split_indices.json for the val split (falls back to last 15% of rows).
    Uses PyArrow for zero-copy parquet loading — no pandas in the hot path.

    Returns:
        Tuple of (dataloader, n_val_samples)
    """
    import pyarrow.parquet as pq
    from core.data.dataloader import build_ple_dataloader, FeatureColumnSpec

    # Locate parquet
    features_parquet = channel_path / "features.parquet"
    if features_parquet.exists():
        table = pq.read_table(str(features_parquet))
        logger.info("Loaded features.parquet: %d rows, %d columns",
                    table.num_rows, table.num_columns)
    else:
        parquet_files = list(channel_path.glob("*.parquet"))
        if not parquet_files:
            raise FileNotFoundError(f"No .parquet files found in {channel_path}")
        # Prefer the largest file (usually the main data file)
        parquet_files.sort(key=lambda p: p.stat().st_size, reverse=True)
        table = pq.read_table(str(parquet_files[0]))
        logger.info("Loaded %s: %d rows, %d columns",
                    parquet_files[0].name, table.num_rows, table.num_columns)

    tasks = config.get("tasks", [])
    label_map = {t["name"]: t.get("label_col", t["name"]) for t in tasks}

    # Feature columns — prefer schema, fall back to table columns minus labels/ids
    feature_columns: List[str] = feature_schema.get("columns", [])
    if not feature_columns:
        feature_columns = feature_schema.get("feature_columns", [])
    merged_cols = set(table.column_names)
    static_features = [c for c in feature_columns if c in merged_cols]

    if not static_features:
        label_col_set = set(label_map.values())
        id_cols_cfg: List[str] = config.get("dataset", {}).get("id_columns", [])
        date_cols_cfg: List[str] = config.get("dataset", {}).get("date_columns", [])
        exclude_cols = set(id_cols_cfg + date_cols_cfg)
        static_features = [
            c for c in table.column_names
            if c not in label_col_set and c not in exclude_cols
        ]
        logger.info("Feature columns from table fallback: %d", len(static_features))

    feature_spec = FeatureColumnSpec(static_features=static_features)
    logger.info("Feature spec: %d columns", len(static_features))

    # Split indices
    split_path = channel_path / "split_indices.json"
    if split_path.exists():
        with open(split_path) as f:
            split_indices = json.load(f)
        val_idx = split_indices.get("val", split_indices.get("validation", []))
        if not val_idx:
            val_idx = split_indices.get("test", [])
        if val_idx:
            tbl_val = table.take(val_idx)
            logger.info("Validation set from split_indices.json: %d samples", tbl_val.num_rows)
        else:
            logger.warning("No val/test indices in split_indices.json — using last 15%%")
            n = table.num_rows
            tbl_val = table.slice(int(n * 0.85))
    else:
        logger.warning("split_indices.json not found — using last 15%%")
        n = table.num_rows
        tbl_val = table.slice(int(n * 0.85))

    n_val = tbl_val.num_rows
    logger.info("Validation samples: %d", n_val)

    val_loader = build_ple_dataloader(
        df=tbl_val,
        feature_spec=feature_spec,
        label_columns=label_map,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )
    return val_loader, n_val


# ---------------------------------------------------------------------------
# Inference loop (mirrors eval_checkpoint.py)
# ---------------------------------------------------------------------------

def _run_inference(predictor, val_loader) -> tuple[dict, dict]:
    """Collect predictions and labels from a full DataLoader pass."""
    all_predictions: Dict[str, List[torch.Tensor]] = {}
    all_labels: Dict[str, List[torch.Tensor]] = {}

    with torch.no_grad():
        for batch in val_loader:
            output = predictor.predict_batch(batch)

            for task_name, pred in output.predictions.items():
                if pred is not None:
                    all_predictions.setdefault(task_name, []).append(pred.cpu())

            # Extract labels
            if isinstance(batch, dict) and "targets" in batch:
                targets = batch["targets"]
            elif hasattr(batch, "targets") and batch.targets is not None:
                targets = batch.targets
            else:
                targets = {}

            if isinstance(targets, dict):
                for task_name, label in targets.items():
                    if label is not None:
                        all_labels.setdefault(task_name, []).append(
                            label.cpu() if isinstance(label, torch.Tensor)
                            else torch.tensor(label)
                        )

    predictions = {k: torch.cat(v, dim=0) for k, v in all_predictions.items() if v}
    labels = {k: torch.cat(v, dim=0) for k, v in all_labels.items() if v}

    logger.info(
        "Inference complete — predictions: %d tasks, labels: %d tasks",
        len(predictions), len(labels),
    )
    for tn in predictions:
        logger.info(
            "  %s: pred=%s, label=%s",
            tn,
            predictions[tn].shape,
            labels[tn].shape if tn in labels else "MISSING",
        )
    return predictions, labels


# ---------------------------------------------------------------------------
# Main evaluation flow
# ---------------------------------------------------------------------------

def run_eval(
    data_dir: str,
    model_dir: str,
    config_path: str,
    output_dir: str,
    batch_size: int = 5632,
    num_workers: int = 2,
    device: str = "auto",
    hp_overrides: Optional[Dict[str, Any]] = None,
    dataset_config_path: Optional[str] = None,
) -> str:
    """Core evaluation routine shared by SageMaker and local paths.

    Args:
        data_dir:            Directory with parquet + feature_schema.json + split_indices.json.
        model_dir:           Directory containing the .pt checkpoint.
        config_path:         Path to pipeline.yaml (common or legacy single-file).
        output_dir:          Destination directory for eval_metrics.json.
        batch_size:          Inference batch size.
        num_workers:         DataLoader workers.
        device:              "auto", "cuda", or "cpu".
        hp_overrides:        Optional dict of HP overrides passed to PLEPredictor.
        dataset_config_path: Optional path to dataset-specific YAML to deep-merge
                             on top of *config_path* (split-config pattern).

    Returns:
        Path to the written eval_metrics.json.
    """
    from core.pipeline.config import load_merged_config

    channel_path = Path(data_dir)
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "eval_metrics.json"

    # --- Config ---
    if not Path(config_path).exists():
        raise FileNotFoundError(f"Config not found: {config_path}")
    if dataset_config_path and Path(dataset_config_path).exists():
        config = load_merged_config(config_path, dataset_config_path)
        logger.info("Config loaded (merged): %s + %s", config_path, dataset_config_path)
    else:
        with open(config_path, encoding="utf-8") as f:
            config = yaml.safe_load(f)
        logger.info("Config loaded: %s", config_path)

    # --- Feature schema ---
    feature_schema_path = channel_path / "feature_schema.json"
    if not feature_schema_path.exists():
        raise FileNotFoundError(f"feature_schema.json not found in {channel_path}")
    with open(feature_schema_path) as f:
        feature_schema = json.load(f)

    # --- Find checkpoint ---
    checkpoint_path = _find_checkpoint(model_dir)

    # --- Build predictor ---
    from core.inference.predictor import PLEPredictor

    effective_hp = {
        "use_adatt": "false",
        "use_grad_surgery": "false",
    }
    if hp_overrides:
        effective_hp.update(hp_overrides)

    predictor = PLEPredictor(
        checkpoint_path=checkpoint_path,
        config_path=config_path,
        feature_schema_path=str(feature_schema_path),
        device=device,
        hp_overrides=effective_hp,
    )

    # --- Val dataloader ---
    val_loader, n_val = _build_val_loader(
        channel_path=channel_path,
        config=config,
        feature_schema=feature_schema,
        batch_size=batch_size,
        num_workers=num_workers,
    )

    # --- Inference ---
    logger.info("Running inference (batch_size=%d, num_workers=%d)...", batch_size, num_workers)
    predictions, labels = _run_inference(predictor, val_loader)

    # --- Evaluate ---
    from core.evaluation.evaluator import PLEEvaluator

    tasks = config.get("tasks", [])
    task_configs = []
    for t in tasks:
        tc: Dict[str, Any] = {
            "name": t["name"],
            "task_type": t.get("type", "binary"),
        }
        if "num_classes" in t:
            tc["num_classes"] = t["num_classes"]
        if "topk_k" in t:
            tc["topk_k"] = t["topk_k"]
        task_configs.append(tc)

    evaluator = PLEEvaluator(task_configs=task_configs)
    metrics = evaluator.evaluate(predictions, labels)

    # Checkpoint metadata
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    metrics["checkpoint_epoch"] = ckpt.get("epoch", "unknown")
    metrics["checkpoint_path"] = Path(checkpoint_path).name
    metrics["val_samples"] = n_val
    metrics["data_dir"] = str(channel_path)

    # --- Summary log ---
    logger.info("=" * 60)
    logger.info("EVALUATION RESULTS")
    logger.info("=" * 60)
    for k, v in sorted(metrics.items()):
        if isinstance(v, float):
            logger.info("  %s: %.6f", k, v)
        elif isinstance(v, int):
            logger.info("  %s: %d", k, v)
    logger.info("=" * 60)

    # --- Save ---
    evaluator.save(metrics, str(out_path))
    logger.info("eval_metrics.json written to %s", out_path)
    return str(out_path)


# ---------------------------------------------------------------------------
# Entry points: SageMaker vs local
# ---------------------------------------------------------------------------

def _sagemaker_main() -> None:
    """Entry point when running inside SageMaker."""
    hp = get_hyperparameters()
    logger.info("SageMaker HPs: %s", hp)

    data_dir = os.environ["SM_CHANNEL_TRAIN"]
    model_dir = os.environ["SM_CHANNEL_MODEL"]
    output_dir = os.environ["SM_OUTPUT_DATA_DIR"]

    # Config path: relative to /opt/ml/code (source_dir extraction root).
    # Supports two patterns:
    #   (a) Split-config (new):  config="configs/pipeline.yaml"
    #                            dataset_config="configs/datasets/santander.yaml"
    #   (b) Legacy single-file:  config="configs/santander/pipeline.yaml"
    default_config = "configs/pipeline.yaml"
    config_path_raw: str = hp.get("config", default_config)
    config_path = Path(config_path_raw)
    if not config_path.is_absolute():
        config_path = Path("/opt/ml/code") / config_path_raw
    config_path_str = str(config_path)

    dataset_config_path_str: Optional[str] = None
    dataset_config_raw: str = hp.get("dataset_config", "")
    if dataset_config_raw:
        dataset_config_path = Path(dataset_config_raw)
        if not dataset_config_path.is_absolute():
            dataset_config_path = Path("/opt/ml/code") / dataset_config_raw
        dataset_config_path_str = str(dataset_config_path)

    batch_size: int = int(hp.get("batch_size", 5632))
    num_workers: int = int(hp.get("num_workers", _default_num_workers()))
    device: str = str(hp.get("device", "auto"))

    # Pass through any ablation-related HP overrides
    hp_overrides: Dict[str, Any] = {}
    for k in ("use_adatt", "use_grad_surgery", "use_ple", "use_cgc_gate",
               "use_group_task_expert", "use_logit_transfer", "use_hmm_projectors",
               "shared_experts", "gate_type", "num_layers"):
        if k in hp:
            hp_overrides[k] = hp[k]

    run_eval(
        data_dir=data_dir,
        model_dir=model_dir,
        config_path=config_path_str,
        output_dir=output_dir,
        batch_size=batch_size,
        num_workers=num_workers,
        device=device,
        hp_overrides=hp_overrides,
        dataset_config_path=dataset_config_path_str,
    )


def _local_main() -> None:
    """Entry point for local CLI execution (mirrors eval_checkpoint.py)."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Evaluate a PLE checkpoint locally (SageMaker entry point).",
    )
    parser.add_argument("--checkpoint", required=True, help="Path to .pt checkpoint")
    parser.add_argument("--data-dir", required=True, help="Phase 0 data directory")
    parser.add_argument("--config", default="configs/pipeline.yaml",
                        help="Common pipeline YAML (or legacy single-file path)")
    parser.add_argument("--dataset", default="",
                        help="Dataset-specific YAML to deep-merge on top of --config "
                             "(e.g. configs/datasets/santander.yaml)")
    parser.add_argument("--output-dir", default=None,
                        help="Directory for eval_metrics.json (default: beside checkpoint)")
    parser.add_argument("--batch-size", type=int, default=5632)
    parser.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    parser.add_argument("--num-workers", type=int, default=0)
    args = parser.parse_args()

    if args.output_dir is None:
        args.output_dir = str(Path(args.checkpoint).parent)

    # For local mode the model_dir is the directory containing the checkpoint
    model_dir = str(Path(args.checkpoint).parent)

    run_eval(
        data_dir=args.data_dir,
        model_dir=model_dir,
        config_path=args.config,
        output_dir=args.output_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        device=args.device,
        dataset_config_path=args.dataset or None,
    )


def main() -> None:
    try:
        if _is_sagemaker():
            logger.info("Detected SageMaker environment")
            _sagemaker_main()
        else:
            logger.info("Running in local mode")
            _local_main()
    except Exception:
        logger.exception("eval_entry.py failed with unhandled exception")
        sys.exit(1)


if __name__ == "__main__":
    main()
