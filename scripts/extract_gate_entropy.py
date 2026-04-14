"""Extract CGC gate weight entropy from PLE teacher model checkpoints.

Gate weights in PLE's CGC layers determine how much each expert contributes
to each task's representation.  Gate Entropy measures expert utilization
diversity:
  - High entropy (close to log(n_experts)) → all experts used equally → no collapse
  - Low entropy (close to 0) → one expert dominates → routing collapse

Two gate sources are analysed per checkpoint:
  1. CGC layer gate weights  (_last_gate_weights stored on CGCLayer in eval mode)
     - Shape: (batch, num_task_experts + num_shared_experts) per task per layer
  2. CGC attention weights   (cgc_attention_weights returned in PLEOutput)
     - Shape: (batch, num_shared_experts) per task

Results are averaged over validation batches, then the table and entropy
values are written to `outputs/gate_entropy_analysis.json`.

Usage::

    PYTHONPATH=. python scripts/extract_gate_entropy.py \\
        --data-dir outputs/phase0_v12 \\
        --config configs/santander/pipeline.yaml \\
        --output outputs/gate_entropy_analysis.json

DO NOT run on SageMaker -- local-only analysis script.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional

os.environ["PYTHONIOENCODING"] = "utf-8"
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch
import yaml

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger("extract_gate_entropy")


# ---------------------------------------------------------------------------
# Entropy helpers
# ---------------------------------------------------------------------------

def gate_entropy(weights: np.ndarray, eps: float = 1e-8) -> float:
    """Shannon entropy of gate weight distribution, averaged over batch.

    Args:
        weights: (N, n_experts) array of gate weights (should sum to 1 per row).
        eps:     Small constant for numerical stability.

    Returns:
        Scalar mean entropy (nats).
    """
    log_w = np.log(np.clip(weights, eps, None))
    return float(-(weights * log_w).sum(axis=-1).mean())


def max_weight_ratio(weights: np.ndarray) -> float:
    """Mean of the maximum gate weight per sample (1.0 = full collapse)."""
    return float(weights.max(axis=-1).mean())


def active_expert_count(weights: np.ndarray, threshold: float = 0.05) -> float:
    """Mean number of experts with weight > threshold per sample."""
    return float((weights > threshold).sum(axis=-1).mean())


# ---------------------------------------------------------------------------
# Core analysis
# ---------------------------------------------------------------------------

def collect_gate_weights_from_batches(
    model,
    val_loader,
    device: torch.device,
    max_batches: int,
) -> Dict:
    """Run forward passes and accumulate gate weights.

    Returns a dict with keys:
      "cgc_layer_gates":  {layer_idx: {task_idx: list of (batch, n_experts) arrays}}
      "cgc_attn_weights": {task_name: list of (batch, n_experts) arrays}
    """
    cgc_layer_gates: Dict[int, Dict[int, List[np.ndarray]]] = {}
    cgc_attn_weights: Dict[str, List[np.ndarray]] = {}

    model.eval()

    for batch_idx, batch in enumerate(val_loader):
        if batch_idx >= max_batches:
            break

        if batch_idx % 20 == 0:
            logger.info("Processing batch %d / %d", batch_idx, max_batches)

        # Build PLEInput from the dataloader dict
        from core.model.ple.model import PLEInput
        if isinstance(batch, dict):
            features = batch["features"].to(device)
            cluster_ids = batch.get("cluster_ids")
            cluster_probs = batch.get("cluster_probs")
            ple_input = PLEInput(
                features=features,
                cluster_ids=cluster_ids.to(device) if cluster_ids is not None else None,
                cluster_probs=cluster_probs.to(device) if cluster_probs is not None else None,
            )
        elif isinstance(batch, PLEInput):
            ple_input = batch.to(device)
        else:
            raise TypeError(f"Unsupported batch type: {type(batch)}")

        with torch.no_grad():
            output = model(ple_input, compute_loss=False)

        # --- CGC layer gate weights ---
        # Each CGCLayer stores _last_gate_weights in eval mode:
        #   {task_idx: (batch, n_task_experts + n_shared_experts)}
        for layer_idx, layer in enumerate(model.extraction_layers):
            if not hasattr(layer, "_last_gate_weights"):
                continue
            if layer_idx not in cgc_layer_gates:
                cgc_layer_gates[layer_idx] = {}
            for task_idx, w in layer._last_gate_weights.items():
                w_np = w.cpu().numpy()  # (batch, n_experts)
                if task_idx not in cgc_layer_gates[layer_idx]:
                    cgc_layer_gates[layer_idx][task_idx] = []
                cgc_layer_gates[layer_idx][task_idx].append(w_np)

        # --- CGC attention weights (PLEOutput.cgc_attention_weights) ---
        if output.cgc_attention_weights is not None:
            for task_name, w in output.cgc_attention_weights.items():
                w_np = w.cpu().numpy()  # (batch, n_shared_experts)
                if task_name not in cgc_attn_weights:
                    cgc_attn_weights[task_name] = []
                cgc_attn_weights[task_name].append(w_np)

    return {
        "cgc_layer_gates": cgc_layer_gates,
        "cgc_attn_weights": cgc_attn_weights,
    }


def summarize_gate_weights(
    gate_data: Dict,
    task_names: List[str],
) -> Dict:
    """Compute per-task entropy / max-weight / active-experts from accumulated batches.

    Returns a nested dict suitable for JSON serialization:
    {
      "cgc_layer_gates": {
        "layer_0": {
          "task_a": {"entropy": ..., "max_weight": ..., "active_experts": ...,
                     "mean_weights": [w0, w1, ...], "n_experts": ...},
          ...
        }
      },
      "cgc_attn_weights": {
        "task_a": {"entropy": ..., "max_weight": ..., "active_experts": ...,
                   "mean_weights": [w0, w1, ...], "n_experts": ...},
        ...
      }
    }
    """
    results: Dict = {}

    # --- CGC layer gates ---
    cgc_layer_gates = gate_data.get("cgc_layer_gates", {})
    if cgc_layer_gates:
        results["cgc_layer_gates"] = {}
        for layer_idx, task_dict in sorted(cgc_layer_gates.items()):
            layer_key = f"layer_{layer_idx}"
            results["cgc_layer_gates"][layer_key] = {}
            for task_idx, w_list in sorted(task_dict.items()):
                combined = np.concatenate(w_list, axis=0)  # (N, n_experts)
                mean_w = combined.mean(axis=0).tolist()
                n_experts = len(mean_w)
                task_name = task_names[task_idx] if task_idx < len(task_names) else f"task_{task_idx}"
                results["cgc_layer_gates"][layer_key][task_name] = {
                    "entropy": gate_entropy(combined),
                    "max_entropy": float(np.log(n_experts)),
                    "entropy_ratio": gate_entropy(combined) / max(np.log(n_experts), 1e-8),
                    "max_weight": max_weight_ratio(combined),
                    "active_experts": active_expert_count(combined),
                    "n_experts": n_experts,
                    "mean_weights": mean_w,
                }

    # --- CGC attention weights ---
    cgc_attn = gate_data.get("cgc_attn_weights", {})
    if cgc_attn:
        results["cgc_attn_weights"] = {}
        for task_name, w_list in sorted(cgc_attn.items()):
            combined = np.concatenate(w_list, axis=0)
            mean_w = combined.mean(axis=0).tolist()
            n_experts = len(mean_w)
            results["cgc_attn_weights"][task_name] = {
                "entropy": gate_entropy(combined),
                "max_entropy": float(np.log(n_experts)),
                "entropy_ratio": gate_entropy(combined) / max(np.log(n_experts), 1e-8),
                "max_weight": max_weight_ratio(combined),
                "active_experts": active_expert_count(combined),
                "n_experts": n_experts,
                "mean_weights": mean_w,
            }

    return results


def print_entropy_table(summary: Dict, checkpoint_label: str) -> None:
    """Pretty-print a gate entropy table for a single checkpoint."""
    print(f"\n{'='*70}")
    print(f"  Gate Entropy Table - {checkpoint_label}")
    print(f"{'='*70}")

    # CGC layer gates
    layer_gates = summary.get("cgc_layer_gates", {})
    for layer_key, task_dict in sorted(layer_gates.items()):
        print(f"\n  [{layer_key}] CGC Gate Weights (task × expert)")
        print(f"  {'Task':<30} {'Entropy':>8} {'Ratio':>7} {'MaxW':>7} {'Active':>7} {'n_exp':>6}")
        print(f"  {'-'*30} {'-'*8} {'-'*7} {'-'*7} {'-'*7} {'-'*6}")
        for task_name, stats in sorted(task_dict.items()):
            print(
                f"  {task_name:<30} "
                f"{stats['entropy']:>8.4f} "
                f"{stats['entropy_ratio']:>7.3f} "
                f"{stats['max_weight']:>7.4f} "
                f"{stats['active_experts']:>7.2f} "
                f"{stats['n_experts']:>6d}"
            )

    # CGC attention weights
    attn_weights = summary.get("cgc_attn_weights", {})
    if attn_weights:
        print(f"\n  [CGC Attention] Shared Expert Attention Weights (task × shared_expert)")
        print(f"  {'Task':<30} {'Entropy':>8} {'Ratio':>7} {'MaxW':>7} {'Active':>7} {'n_exp':>6}")
        print(f"  {'-'*30} {'-'*8} {'-'*7} {'-'*7} {'-'*7} {'-'*6}")
        for task_name, stats in sorted(attn_weights.items()):
            print(
                f"  {task_name:<30} "
                f"{stats['entropy']:>8.4f} "
                f"{stats['entropy_ratio']:>7.3f} "
                f"{stats['max_weight']:>7.4f} "
                f"{stats['active_experts']:>7.2f} "
                f"{stats['n_experts']:>6d}"
            )

    print()


# ---------------------------------------------------------------------------
# Checkpoint loader
# ---------------------------------------------------------------------------

def load_predictor_and_model(
    checkpoint_path: str,
    config_path: str,
    feature_schema_path: str,
    device: torch.device,
) -> "PLEModel":
    """Load checkpoint into a PLEModel via PLEPredictor."""
    from core.inference.predictor import PLEPredictor

    hp_overrides = {
        "use_adatt": "false",
        "use_grad_surgery": "false",
    }

    predictor = PLEPredictor(
        checkpoint_path=checkpoint_path,
        config_path=config_path,
        feature_schema_path=feature_schema_path,
        device=str(device),
        hp_overrides=hp_overrides,
    )

    # Force eval mode
    predictor.model.eval()
    return predictor.model, predictor.task_names


# ---------------------------------------------------------------------------
# Data loading (mirroring eval_checkpoint.py)
# ---------------------------------------------------------------------------

def build_val_loader(
    data_dir: Path,
    config: dict,
    feature_schema: dict,
    batch_size: int,
    num_workers: int,
):
    """Build validation DataLoader from phase0 parquet + split_indices."""
    import pyarrow.parquet as pq
    from core.data.dataloader import build_ple_dataloader, FeatureColumnSpec

    parquet_files = list(data_dir.glob("*.parquet"))
    if not parquet_files:
        raise FileNotFoundError(f"No parquet files in {data_dir}")

    table = pq.read_table(parquet_files[0])
    logger.info("Loaded %s: %d rows, %d columns",
                parquet_files[0].name, table.num_rows, table.num_columns)

    tasks = config.get("tasks", [])
    label_map = {t["name"]: t.get("label_col", t["name"]) for t in tasks}

    # Feature columns
    feature_columns = feature_schema.get("columns", []) or feature_schema.get("feature_columns", [])
    merged_cols_set = set(table.column_names)
    static_features = [c for c in feature_columns if c in merged_cols_set]

    if not static_features:
        label_col_set = set(label_map.values())
        id_cols_cfg = config.get("dataset", {}).get("id_columns", [])
        date_cols_cfg = config.get("dataset", {}).get("date_columns", [])
        exclude_cols = set(id_cols_cfg + date_cols_cfg)
        static_features = [c for c in table.column_names
                           if c not in label_col_set and c not in exclude_cols]

    feature_spec = FeatureColumnSpec(static_features=static_features)
    logger.info("Feature columns: %d", len(static_features))

    # Split
    split_path = data_dir / "split_indices.json"
    if split_path.exists():
        with open(split_path) as f:
            split_indices = json.load(f)
        val_idx = split_indices.get("val", split_indices.get("validation", []))
        if not val_idx:
            val_idx = split_indices.get("test", [])
        if val_idx:
            tbl_val = table.take(val_idx)
            logger.info("Validation from split_indices: %d samples", tbl_val.num_rows)
        else:
            logger.warning("No val/test in split_indices.json; using last 15%%")
            tbl_val = table.slice(int(table.num_rows * 0.85))
    else:
        logger.warning("No split_indices.json; using last 15%%")
        tbl_val = table.slice(int(table.num_rows * 0.85))

    logger.info("Validation set: %d samples", tbl_val.num_rows)

    loader = build_ple_dataloader(
        df=tbl_val,
        feature_spec=feature_spec,
        label_columns=label_map,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )
    return loader


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

CHECKPOINTS = {
    "10ep_local": "outputs/ablation_v12/joint_full/model/model.pth",
    "30ep_sagemaker": "outputs/sagemaker_teacher_30ep/teacher_full/best.pt",
}


def main():
    parser = argparse.ArgumentParser(
        description="Extract CGC gate entropy from PLE teacher checkpoints."
    )
    parser.add_argument(
        "--data-dir",
        default="outputs/phase0_v12",
        help="Phase 0 output directory (contains .parquet + split_indices.json)",
    )
    parser.add_argument(
        "--config",
        default="configs/santander/pipeline.yaml",
    )
    parser.add_argument(
        "--output",
        default="outputs/gate_entropy_analysis.json",
        help="Path for JSON output.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=5632,
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=0,
        help="DataLoader workers. Use 0 on Windows.",
    )
    parser.add_argument(
        "--max-batches",
        type=int,
        default=50,
        help="Maximum validation batches to process per checkpoint (for speed).",
    )
    parser.add_argument(
        "--device",
        default="cpu",
        choices=["cpu", "cuda", "auto"],
        help="Inference device. Use cpu when GPU may be busy.",
    )
    parser.add_argument(
        "--checkpoints",
        nargs="+",
        default=None,
        help=(
            "Optional list of label=path pairs to override the default two "
            "checkpoints, e.g. --checkpoints 10ep=outputs/.../model.pth"
        ),
    )
    args = parser.parse_args()

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    logger.info("Device: %s", device)

    data_dir = Path(args.data_dir)
    feature_schema_path = data_dir / "feature_schema.json"

    with open(args.config, encoding="utf-8") as f:
        config = yaml.safe_load(f)

    with open(feature_schema_path, encoding="utf-8") as f:
        feature_schema = json.load(f)

    # Resolve checkpoint list
    if args.checkpoints:
        ckpt_map = {}
        for item in args.checkpoints:
            label, path = item.split("=", 1)
            ckpt_map[label.strip()] = path.strip()
    else:
        ckpt_map = CHECKPOINTS

    # Build shared validation loader (same data for all checkpoints)
    val_loader = build_val_loader(
        data_dir=data_dir,
        config=config,
        feature_schema=feature_schema,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    all_results: Dict = {}

    for label, ckpt_path in ckpt_map.items():
        ckpt_path_resolved = Path(ckpt_path)
        if not ckpt_path_resolved.exists():
            logger.warning("Checkpoint not found, skipping: %s", ckpt_path)
            all_results[label] = {"error": f"checkpoint not found: {ckpt_path}"}
            continue

        logger.info("=" * 60)
        logger.info("Loading checkpoint: %s  (%s)", label, ckpt_path)

        try:
            model, task_names = load_predictor_and_model(
                checkpoint_path=str(ckpt_path_resolved),
                config_path=args.config,
                feature_schema_path=str(feature_schema_path),
                device=device,
            )
        except Exception:
            logger.exception("Failed to load checkpoint %s", ckpt_path)
            all_results[label] = {"error": f"load failed: {ckpt_path}"}
            continue

        logger.info("Tasks (%d): %s", len(task_names), task_names)
        logger.info("Collecting gate weights from up to %d batches...", args.max_batches)

        gate_data = collect_gate_weights_from_batches(
            model=model,
            val_loader=val_loader,
            device=device,
            max_batches=args.max_batches,
        )

        summary = summarize_gate_weights(gate_data, task_names)
        all_results[label] = {
            "checkpoint_path": str(ckpt_path_resolved),
            "task_names": task_names,
            "gate_entropy": summary,
        }

        print_entropy_table(summary, checkpoint_label=label)

    # Save JSON
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2)

    logger.info("Results saved to %s", output_path)

    # Print interpretation guide
    print("Interpretation:")
    print("  entropy_ratio = entropy / log(n_experts)")
    print("  1.0 = perfectly uniform (no collapse)")
    print("  0.0 = single expert dominates (full collapse)")
    print("  active_experts = mean experts with weight > 0.05")
    print()


if __name__ == "__main__":
    main()
