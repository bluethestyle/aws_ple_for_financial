"""
Run Expert Redundancy (CCA) analysis on a trained PLE checkpoint.

Wires the previously-unused ``core.evaluation.expert_redundancy``
(``ExpertRedundancyAnalyzer``) into a runnable entry point: it loads a
checkpoint, runs a forward pass over the validation split capturing each
shared expert's output, and computes pairwise canonical correlations to
quantify how much the experts' representation *directions* overlap.

This is the measurement half of the "measure -> intervene -> re-measure"
loop around OCP (orthogonal-complement residual recovery): run it on the
baseline checkpoint to see whether experts are actually redundant, then
re-run on the OCP-trained checkpoint to confirm the projection lowered
canonical correlation.

Inference only (forward pass + numpy SVD) — no training, CPU is fine.

Usage::

    python scripts/run_expert_redundancy.py \
        --checkpoint outputs/<job>/best.pt \
        --data-dir   outputs/phase0_v12 \
        --output     outputs/<job>/expert_redundancy.json
"""
import json
import logging
import os
import sys
from pathlib import Path

os.environ["PYTHONIOENCODING"] = "utf-8"
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import yaml  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger("run_expert_redundancy")


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser(
        description="Expert Redundancy (CCA) analysis on a checkpoint"
    )
    parser.add_argument("--checkpoint", required=True, help="Path to .pt checkpoint")
    parser.add_argument("--data-dir", default="outputs/phase0_v12", help="Phase 0 output dir")
    parser.add_argument("--config", default="configs/pipeline.yaml",
                        help="Common pipeline YAML")
    parser.add_argument("--dataset", default="",
                        help="Dataset-specific YAML to deep-merge on top of --config")
    parser.add_argument("--output", default=None,
                        help="Output JSON path (default: expert_redundancy.json beside checkpoint)")
    parser.add_argument("--batch-size", type=int, default=5632)
    parser.add_argument("--max-batches", type=int, default=20)
    parser.add_argument("--n-components", type=int, default=10)
    parser.add_argument("--min-samples", type=int, default=256)
    parser.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    parser.add_argument("--num-workers", type=int, default=0)
    args = parser.parse_args()

    if args.output is None:
        args.output = str(Path(args.checkpoint).parent / "expert_redundancy.json")

    channel_path = Path(args.data_dir)

    # --- Load config (supports split-config pattern) ---
    from core.pipeline.config import load_merged_config
    if args.dataset and Path(args.dataset).exists():
        config = load_merged_config(args.config, args.dataset)
        logger.info("Config loaded (merged): %s + %s", args.config, args.dataset)
    else:
        with open(args.config, encoding="utf-8") as f:
            config = yaml.safe_load(f)

    # --- Feature schema ---
    feature_schema_path = channel_path / "feature_schema.json"
    with open(feature_schema_path) as f:
        feature_schema = json.load(f)

    # --- Build predictor (loads model + weights) ---
    from core.inference.predictor import PLEPredictor

    predictor = PLEPredictor(
        checkpoint_path=args.checkpoint,
        config_path=args.config,
        feature_schema_path=str(feature_schema_path),
        device=args.device,
        # Keep analysis structure identical to inference: no train-only fusions
        # toggled on here; the checkpoint's own config governs the experts.
        hp_overrides={"use_adatt": "false", "use_grad_surgery": "false"},
        dataset_config_path=args.dataset or None,
    )

    # --- Build validation dataloader (same scaffold as eval_checkpoint.py) ---
    import pyarrow.parquet as pq
    from core.data.dataloader import build_ple_dataloader, FeatureColumnSpec

    parquet_files = list(channel_path.glob("*.parquet"))
    if not parquet_files:
        logger.error("No parquet files found in %s", channel_path)
        return 1

    table = pq.read_table(parquet_files[0])
    logger.info("Loaded %s: %d rows, %d columns",
                parquet_files[0].name, table.num_rows, table.num_columns)

    tasks = config.get("tasks", [])
    label_map = {t["name"]: t.get("label_col", t["name"]) for t in tasks}

    feature_columns = feature_schema.get("columns", []) or feature_schema.get("feature_columns", [])
    merged_cols_set = set(table.column_names)
    static_features = [c for c in feature_columns if c in merged_cols_set]
    if not static_features:
        label_col_set = set(label_map.values())
        id_cols_cfg = config.get("dataset", {}).get("id_columns", [])
        date_cols_cfg = config.get("dataset", {}).get("date_columns", [])
        exclude_cols = set(id_cols_cfg + date_cols_cfg)
        static_features = [
            c for c in table.column_names
            if c not in label_col_set and c not in exclude_cols
        ]
    feature_spec = FeatureColumnSpec(static_features=static_features)
    logger.info("Features: %d columns", len(static_features))

    # Validation split (fallbacks mirror eval_checkpoint.py)
    split_path = channel_path / "split_indices.json"
    if split_path.exists():
        with open(split_path) as f:
            split_indices = json.load(f)
        val_idx = (split_indices.get("val")
                   or split_indices.get("validation")
                   or split_indices.get("test")
                   or [])
        if val_idx:
            tbl_val = table.take(val_idx)
        else:
            tbl_val = table.slice(int(table.num_rows * 0.85))
    else:
        tbl_val = table.slice(int(table.num_rows * 0.85))
    logger.info("Validation: %d samples", tbl_val.num_rows)

    val_loader = build_ple_dataloader(
        df=tbl_val,
        feature_spec=feature_spec,
        label_columns=label_map,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )

    # --- Run CCA redundancy analysis ---
    from core.evaluation.expert_redundancy import ExpertRedundancyAnalyzer

    analyzer = ExpertRedundancyAnalyzer(
        predictor.model,
        n_components=args.n_components,
        min_samples=args.min_samples,
    )
    result = analyzer.analyze(val_loader, max_batches=args.max_batches)

    if result is None:
        logger.error(
            "CCA analysis returned no result (need >=2 shared experts and "
            ">=%d samples). Nothing written.", args.min_samples,
        )
        return 2

    logger.info("\n%s", result.summary())

    payload = result.to_dict()
    payload["checkpoint"] = str(Path(args.checkpoint).name)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    logger.info("Wrote redundancy result -> %s", args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
