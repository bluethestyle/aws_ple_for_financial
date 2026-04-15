"""
Evaluate a trained PLE checkpoint — inference only, no training.

Uses the modular predictor + evaluator pipeline:
  PLEPredictor (loads model, runs forward pass)
  → PLEEvaluator (computes per-task metrics, saves JSON)

Usage::

    python scripts/eval_checkpoint.py \
        --checkpoint outputs/sagemaker_teacher_30ep/teacher_full/best.pt \
        --data-dir outputs/phase0_v12 \
        --output outputs/sagemaker_teacher_30ep/teacher_full/eval_metrics_pertask.json
"""
import json
import logging
import os
import sys
from pathlib import Path

os.environ["PYTHONIOENCODING"] = "utf-8"
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch
import yaml

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger("eval_checkpoint")


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Evaluate a checkpoint (inference only)")
    parser.add_argument("--checkpoint", required=True, help="Path to .pt checkpoint file")
    parser.add_argument("--data-dir", default="outputs/phase0_v12", help="Phase 0 output dir")
    parser.add_argument("--config", default="configs/pipeline.yaml",
                        help="Common pipeline YAML (or legacy single-file path)")
    parser.add_argument("--dataset", default="",
                        help="Dataset-specific YAML to deep-merge on top of --config "
                             "(e.g. configs/datasets/santander.yaml)")
    parser.add_argument("--output", default=None, help="Output eval_metrics.json path (default: beside checkpoint)")
    parser.add_argument("--batch-size", type=int, default=5632)
    parser.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    parser.add_argument("--num-workers", type=int, default=0)
    args = parser.parse_args()

    # Default output path: beside checkpoint
    if args.output is None:
        args.output = str(Path(args.checkpoint).parent / "eval_metrics_pertask.json")

    channel_path = Path(args.data_dir)

    # --- Load config (supports split-config pattern) ---
    from core.pipeline.config import load_merged_config
    if args.dataset and Path(args.dataset).exists():
        config = load_merged_config(args.config, args.dataset)
        logger.info("Config loaded (merged): %s + %s", args.config, args.dataset)
    else:
        with open(args.config, encoding="utf-8") as f:
            config = yaml.safe_load(f)

    # --- Load feature schema ---
    feature_schema_path = channel_path / "feature_schema.json"
    with open(feature_schema_path) as f:
        feature_schema = json.load(f)

    # --- Build predictor ---
    from core.inference.predictor import PLEPredictor

    hp_overrides = {
        "use_adatt": "false",
        "use_grad_surgery": "false",
    }

    predictor = PLEPredictor(
        checkpoint_path=args.checkpoint,
        config_path=args.config,
        feature_schema_path=str(feature_schema_path),
        device=args.device,
        hp_overrides=hp_overrides,
        dataset_config_path=args.dataset or None,
    )

    # --- Load data and build val dataloader ---
    import pyarrow.parquet as pq
    from core.data.dataloader import build_ple_dataloader, FeatureColumnSpec

    parquet_files = list(channel_path.glob("*.parquet"))
    if not parquet_files:
        logger.error("No parquet files found in %s", channel_path)
        sys.exit(1)

    table = pq.read_table(parquet_files[0])
    logger.info("Loaded %s: %d rows, %d columns", parquet_files[0].name, table.num_rows, table.num_columns)

    tasks = config.get("tasks", [])
    task_names = [t["name"] for t in tasks]
    label_map = {t["name"]: t.get("label_col", t["name"]) for t in tasks}

    # Feature columns from schema
    feature_columns = feature_schema.get("columns", [])
    if not feature_columns:
        feature_columns = feature_schema.get("feature_columns", [])
    merged_cols_set = set(table.column_names)
    static_features = [c for c in feature_columns if c in merged_cols_set]

    if not static_features:
        # Fallback: all non-label, non-id columns
        label_col_set = set(label_map.values())
        id_cols_cfg = config.get("dataset", {}).get("id_columns", [])
        date_cols_cfg = config.get("dataset", {}).get("date_columns", [])
        exclude_cols = set(id_cols_cfg + date_cols_cfg)
        static_features = [c for c in table.column_names if c not in label_col_set and c not in exclude_cols]

    feature_spec = FeatureColumnSpec(static_features=static_features)
    logger.info("Features: %d columns", len(static_features))

    # Split indices
    split_path = channel_path / "split_indices.json"
    if split_path.exists():
        with open(split_path) as f:
            split_indices = json.load(f)
        val_idx = split_indices.get("val", split_indices.get("validation", []))
        if not val_idx:
            # Use test split as fallback
            val_idx = split_indices.get("test", [])
        if val_idx:
            tbl_val = table.take(val_idx)
            logger.info("Validation set from split_indices: %d samples", tbl_val.num_rows)
        else:
            logger.warning("No val/test indices in split_indices.json, using last 15%%")
            n = table.num_rows
            val_start = int(n * 0.85)
            tbl_val = table.slice(val_start)
    else:
        logger.warning("No split_indices.json found, using last 15%%")
        n = table.num_rows
        val_start = int(n * 0.85)
        tbl_val = table.slice(val_start)

    logger.info("Validation: %d samples", tbl_val.num_rows)

    val_loader = build_ple_dataloader(
        df=tbl_val,
        feature_spec=feature_spec,
        label_columns=label_map,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )

    # --- Run inference ---
    logger.info("Running inference...")
    all_predictions = {}
    all_labels = {}

    with torch.no_grad():
        for batch in val_loader:
            output = predictor.predict_batch(batch)

            for task_name, pred in output.predictions.items():
                if pred is not None:
                    all_predictions.setdefault(task_name, []).append(pred.cpu())

            # Extract labels from batch
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
                            label.cpu() if isinstance(label, torch.Tensor) else torch.tensor(label)
                        )

    # Concatenate
    predictions = {k: torch.cat(v, dim=0) for k, v in all_predictions.items() if v}
    labels = {k: torch.cat(v, dim=0) for k, v in all_labels.items() if v}

    logger.info("Predictions: %d tasks, Labels: %d tasks",
                len(predictions), len(labels))
    for tn in predictions:
        logger.info("  %s: pred=%s, label=%s",
                     tn, predictions[tn].shape,
                     labels[tn].shape if tn in labels else "MISSING")

    # --- Evaluate ---
    from core.evaluation.evaluator import PLEEvaluator

    task_configs = []
    for t in tasks:
        tc = {
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

    # Add checkpoint info
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    metrics["checkpoint_epoch"] = ckpt.get("epoch", "unknown")
    metrics["checkpoint_path"] = str(Path(args.checkpoint).name)
    metrics["val_samples"] = tbl_val.num_rows

    # --- Print summary ---
    logger.info("=" * 60)
    logger.info("EVALUATION RESULTS")
    logger.info("=" * 60)
    for k, v in sorted(metrics.items()):
        if isinstance(v, float):
            logger.info("  %s: %.6f", k, v)
        elif isinstance(v, (int,)):
            logger.info("  %s: %d", k, v)
    logger.info("=" * 60)

    # --- Save ---
    evaluator.save(metrics, args.output)
    logger.info("Results saved to %s", args.output)


if __name__ == "__main__":
    main()
