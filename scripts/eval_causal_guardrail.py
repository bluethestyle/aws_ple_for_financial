"""MV evaluation of the Causal Guardrail (Paper 3 Axis-3 B).

Loads the teacher_ceh_demeaned checkpoint, computes per-sample DAG
coherence scores ``||z - z W^2||^2 / ||z||^2`` on validation samples
and on synthetic out-of-distribution probes, and writes a summary
showing whether the score separates in-distribution from OOD inputs.

Three OOD probes:
  1. *Uniform random*: each feature drawn uniformly in [-3, 3] (the
     normalized feature range). Breaks inter-feature correlations.
  2. *Permutation*: per-column shuffle across samples. Preserves
     marginals, destroys joint structure.
  3. *Extreme tail*: every feature set to its validation-set 99th
     percentile (or 1st for negative values). A degenerate concentration
     of all-extreme values no real customer would have.

All on CPU. Runtime <1 min for 5k + 3*5k samples. Zero SageMaker cost.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from collections import OrderedDict
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.model.experts.causal import CausalExpert  # noqa: E402
# Reuse load helpers + causal-slice logic from the attribution eval.
from scripts.eval_ceh_attribution import (  # noqa: E402
    load_causal_expert,
    get_causal_indices,
    load_val_features,
)

logger = logging.getLogger("eval_cg")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")


# ---------------------------------------------------------------------------
# OOD probes
# ---------------------------------------------------------------------------

def make_uniform_random(x: np.ndarray, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    # After standard scaling most normalized features live in ~[-3, 3]
    return rng.uniform(low=-3.0, high=3.0, size=x.shape).astype(x.dtype)


def make_permuted(x: np.ndarray, seed: int = 1) -> np.ndarray:
    rng = np.random.default_rng(seed)
    out = x.copy()
    n = x.shape[0]
    for j in range(x.shape[1]):
        perm = rng.permutation(n)
        out[:, j] = x[perm, j]
    return out


def make_extreme_tail(x: np.ndarray, tail: float = 99.0) -> np.ndarray:
    """Every feature set to its (tail)-th percentile; sign-aware."""
    high = np.percentile(x, tail, axis=0)
    low = np.percentile(x, 100 - tail, axis=0)
    # Keep the sign of mean so we don't blindly pick the high side
    means = x.mean(axis=0)
    extreme = np.where(means >= 0, high, low).astype(x.dtype)
    out = np.tile(extreme, (x.shape[0], 1))
    return out


# ---------------------------------------------------------------------------
# Summary helpers
# ---------------------------------------------------------------------------

def summarise(scores: np.ndarray) -> Dict[str, float]:
    return {
        "n": int(scores.size),
        "mean": float(scores.mean()),
        "median": float(np.median(scores)),
        "p25": float(np.percentile(scores, 25)),
        "p75": float(np.percentile(scores, 75)),
        "p95": float(np.percentile(scores, 95)),
        "p99": float(np.percentile(scores, 99)),
        "max": float(scores.max()),
    }


def separation(
    in_dist: np.ndarray, ood: np.ndarray, thresh: float
) -> Dict[str, float]:
    """False-positive + true-positive rates at a given threshold."""
    tpr = float((ood > thresh).mean())
    fpr = float((in_dist > thresh).mean())
    return {"threshold": thresh, "tpr_ood": tpr, "fpr_id": fpr}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--ckpt",
        default="outputs/sagemaker_teacher_30ep/teacher_ceh_demeaned/extracted/model.pth",
    )
    p.add_argument(
        "--config",
        default="outputs/sagemaker_teacher_30ep/teacher_ceh_demeaned/extracted/config.json",
    )
    p.add_argument(
        "--parquet", default="outputs/phase0_v12/santander_final.parquet"
    )
    p.add_argument(
        "--schema", default="outputs/phase0_v12/feature_schema.json"
    )
    p.add_argument("--max-samples", type=int, default=5000)
    p.add_argument(
        "--out", default="outputs/cg_guardrail_eval.json",
    )
    args = p.parse_args()

    expert, full_config = load_causal_expert(Path(args.ckpt), Path(args.config))
    logger.info("Loaded CausalExpert: input=%d output=%d n_vars=%d",
                expert.ceh_input_dim, expert.output_dim, expert.n_causal_vars)

    causal_idx, spans = get_causal_indices(full_config)
    features = load_val_features(
        Path(args.parquet), Path(args.schema),
        val_frac=0.2, max_samples=args.max_samples,
    )
    x_id = features[:, causal_idx]
    logger.info("In-distribution slice: %s", x_id.shape)

    probes: "OrderedDict[str, np.ndarray]" = OrderedDict()
    probes["in_distribution"] = x_id
    probes["ood_uniform_random"] = make_uniform_random(x_id, seed=0)
    probes["ood_permuted"] = make_permuted(x_id, seed=1)
    probes["ood_extreme_tail"] = make_extreme_tail(x_id, tail=99.0)

    all_scores: Dict[str, np.ndarray] = {}
    # v2 baseline: Mahalanobis-like z-score in the causal latent space.
    # Computed from in-distribution z statistics and reused across probes
    # so OOD probes are scored against the same reference distribution.
    z_id_tensor = expert.get_causal_latent(torch.from_numpy(x_id))
    z_id = z_id_tensor.cpu().numpy()
    z_mu = z_id.mean(axis=0)
    z_sigma = z_id.std(axis=0) + 1e-6

    def z_mahal(arr: np.ndarray) -> np.ndarray:
        z = expert.get_causal_latent(torch.from_numpy(arr)).cpu().numpy()
        standardized = (z - z_mu) / z_sigma
        return (standardized * standardized).sum(axis=-1)

    all_scores_zmahal: Dict[str, np.ndarray] = {}

    for name, arr in probes.items():
        tensor = torch.from_numpy(arr)
        # v1: W-reconstruction coherence
        scores = expert.get_causal_coherence_score(tensor).cpu().numpy()
        all_scores[name] = scores
        # v2: z-space Mahalanobis
        z_scores = z_mahal(arr)
        all_scores_zmahal[name] = z_scores
        logger.info(
            "%-25s  v1-recon median=%.4f p99=%.4f  |  v2-zmahal median=%.2f p99=%.2f",
            name, np.median(scores), np.percentile(scores, 99),
            np.median(z_scores), np.percentile(z_scores, 99),
        )

    # Threshold recommendations using in-dist p95 and p99.
    id_scores = all_scores["in_distribution"]
    p95 = float(np.percentile(id_scores, 95))
    p99 = float(np.percentile(id_scores, 99))

    seps: Dict[str, Dict[str, Dict[str, float]]] = {}
    for ood_name, ood_scores in all_scores.items():
        if ood_name == "in_distribution":
            continue
        seps[ood_name] = {
            "at_p95": separation(id_scores, ood_scores, p95),
            "at_p99": separation(id_scores, ood_scores, p99),
        }

    id_z = all_scores_zmahal["in_distribution"]
    z_p95 = float(np.percentile(id_z, 95))
    z_p99 = float(np.percentile(id_z, 99))
    seps_z: Dict[str, Dict[str, Dict[str, float]]] = {}
    for ood_name, ood_scores in all_scores_zmahal.items():
        if ood_name == "in_distribution":
            continue
        seps_z[ood_name] = {
            "at_p95": separation(id_z, ood_scores, z_p95),
            "at_p99": separation(id_z, ood_scores, z_p99),
        }

    results = {
        "checkpoint": str(args.ckpt),
        "causal_input_dim": int(x_id.shape[1]),
        "n_causal_vars": int(expert.n_causal_vars),
        "w_frobenius": float(torch.linalg.norm(expert.W.detach()).item()),
        "v1_w_reconstruction": {
            "score_distributions": {k: summarise(v)
                                     for k, v in all_scores.items()},
            "recommended_thresholds": {"p95": p95, "p99": p99},
            "ood_detection_rates": seps,
        },
        "v2_z_mahalanobis": {
            "score_distributions": {k: summarise(v)
                                     for k, v in all_scores_zmahal.items()},
            "recommended_thresholds": {"p95": z_p95, "p99": z_p99},
            "ood_detection_rates": seps_z,
        },
    }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info("Wrote results to %s", out_path)

    # Pretty console summary
    print()
    print("=" * 72)
    print("CAUSAL GUARDRAIL (CG) - MV EVALUATION")
    print("=" * 72)
    print(f"Checkpoint: {args.ckpt}")
    print(f"W Frobenius norm: {results['w_frobenius']:.3f}   "
          f"Causal vars: {expert.n_causal_vars}")
    print()
    print("[v1] W-reconstruction residual  ||z - z W^2||^2 / ||z||^2")
    print(f"  {'probe':25s} {'median':>9s} {'p95':>9s} {'p99':>9s} {'max':>9s}")
    for name, s in results["v1_w_reconstruction"]["score_distributions"].items():
        print(f"  {name:25s} {s['median']:>9.4f} {s['p95']:>9.4f} "
              f"{s['p99']:>9.4f} {s['max']:>9.4f}")
    print(f"  thr(p95)={p95:.4f}  thr(p99)={p99:.4f}")
    for ood, modes in seps.items():
        v = modes["at_p95"]
        print(f"  {ood:25s} @p95: TPR={v['tpr_ood']*100:5.2f}%  FPR={v['fpr_id']*100:5.2f}%")

    print()
    print("[v2] z-space Mahalanobis  sum((z - mu) / sigma)^2")
    print(f"  {'probe':25s} {'median':>9s} {'p95':>9s} {'p99':>9s} {'max':>9s}")
    for name, s in results["v2_z_mahalanobis"]["score_distributions"].items():
        print(f"  {name:25s} {s['median']:>9.2f} {s['p95']:>9.2f} "
              f"{s['p99']:>9.2f} {s['max']:>9.2f}")
    print(f"  thr(p95)={z_p95:.2f}  thr(p99)={z_p99:.2f}")
    for ood, modes in seps_z.items():
        v = modes["at_p95"]
        print(f"  {ood:25s} @p95: TPR={v['tpr_ood']*100:5.2f}%  FPR={v['fpr_id']*100:5.2f}%")
    print("=" * 72)


if __name__ == "__main__":
    main()
