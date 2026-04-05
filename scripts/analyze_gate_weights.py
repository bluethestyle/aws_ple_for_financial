"""Analyze CGC gate weight distribution for routing collapse detection.

Usage:
    PYTHONPATH=. python scripts/analyze_gate_weights.py \
        --checkpoint outputs/ablation_results/struct_18_ple_softmax/model/model.pth \
        --data outputs/phase0 \
        --config configs/santander/pipeline.yaml
"""
import argparse
import json
import os
import sys

import numpy as np
import torch
import yaml


def compute_gate_entropy(gate_weights: np.ndarray) -> float:
    """Compute average entropy of gate weight distribution."""
    # gate_weights: (batch, n_experts)
    # Entropy: -sum(w * log(w))
    eps = 1e-8
    log_w = np.log(gate_weights + eps)
    entropy = -(gate_weights * log_w).sum(axis=-1)  # (batch,)
    return float(entropy.mean())


def compute_max_weight_ratio(gate_weights: np.ndarray) -> float:
    """Average ratio of max expert weight (1.0 = full collapse)."""
    return float(gate_weights.max(axis=-1).mean())


def compute_expert_utilization(gate_weights: np.ndarray, threshold: float = 0.05) -> float:
    """Average number of experts with weight > threshold."""
    active = (gate_weights > threshold).sum(axis=-1)  # (batch,)
    return float(active.mean())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data", default="outputs/phase0")
    parser.add_argument("--config", default="configs/santander/pipeline.yaml")
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--max-samples", type=int, default=10000)
    parser.add_argument("--output", default=None, help="Save JSON report")
    args = parser.parse_args()

    print(f"Loading checkpoint: {args.checkpoint}")
    print(f"This script extracts gate weights from a trained model")
    print(f"and computes entropy/utilization metrics for routing collapse analysis.")
    print()
    print("Gate entropy interpretation:")
    print("  High entropy (close to log(n_experts)) = experts used equally = no collapse")
    print("  Low entropy (close to 0) = one expert dominates = routing collapse")
    print()

    # TODO: Load model, run inference on validation data,
    # collect _last_gate_weights from CGCLayer,
    # compute per-task entropy/utilization/max_weight
    #
    # This requires the full model loading pipeline from train.py
    # For now, this is a template that will be filled after
    # the structure ablation completes and checkpoints are available.

    print("Note: Full implementation requires model checkpoint.")
    print("Run after structure ablation completes.")


if __name__ == "__main__":
    main()
