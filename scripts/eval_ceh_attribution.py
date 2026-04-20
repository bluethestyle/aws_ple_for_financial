"""
Evaluate CEH attribution quality on a trained CEH checkpoint.

Standalone analysis — extracts the causal expert's attribution_head from a
PLE checkpoint and measures:

  1. Training fit: Spearman rank corr(CEH attribution, gradient × input).
     High = head has learned its training target.
  2. Discriminative power: between-sample variance vs within-sample variance.
     Low ratio = collapses to a global importance pattern.
  3. Stability under input noise: Spearman corr(attr(x), attr(x + eps)).
     High = attribution is locally smooth.
  4. Per-feature-group mass: mean |attribution| aggregated by the causal
     expert's input feature groups (sanity check on where signal lives).

Outputs a JSON summary + prints a compact table. Runs on CPU with the
full phase0_v12 val split taking <1 min for ~5k samples.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from collections import OrderedDict
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from scipy.stats import spearmanr

# Make core imports work when run as a script.
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from core.model.experts.causal import CausalExpert  # noqa: E402

logger = logging.getLogger("eval_ceh")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")


# ---------------------------------------------------------------------------
# Checkpoint loading
# ---------------------------------------------------------------------------

CAUSAL_PREFIX = "extraction_layers.0.shared_experts.4."


def load_causal_expert(ckpt_path: Path, config_path: Path) -> Tuple[CausalExpert, dict]:
    """Load the causal expert (layer 0) out of a full PLE checkpoint.

    Returns the reconstructed CausalExpert + the full model config dict
    (so callers can read routing info).
    """
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model_state = ckpt["model_state_dict"]

    with open(config_path, "r") as f:
        full_config = json.load(f)

    # Resolve the causal expert config from the saved config.
    # Checkpoint stores it under label_schema/model/expert_config/causal.
    causal_cfg = (
        full_config.get("label_schema", {})
        .get("model", {})
        .get("expert_config", {})
        .get("causal", {})
    )
    if not causal_cfg:
        raise RuntimeError("Causal expert config not found in checkpoint config")

    # Extract causal expert weights.
    causal_state: Dict[str, torch.Tensor] = OrderedDict()
    for k, v in model_state.items():
        if k.startswith(CAUSAL_PREFIX):
            causal_state[k[len(CAUSAL_PREFIX):]] = v

    if not causal_state:
        raise RuntimeError(f"No keys found with prefix {CAUSAL_PREFIX}")

    # Infer input_dim from attribution_head output shape (Linear .3 bias).
    attr_out_shape = causal_state["attribution_head.3.bias"].shape
    input_dim = int(attr_out_shape[0])
    output_dim = int(causal_state["causal_encoder.4.bias"].shape[0])
    n_causal_vars = int(causal_state["W"].shape[0])

    config_for_expert = dict(causal_cfg)
    config_for_expert["output_dim"] = output_dim
    config_for_expert["n_causal_vars"] = n_causal_vars

    expert = CausalExpert(input_dim=input_dim, config=config_for_expert)
    missing, unexpected = expert.load_state_dict(causal_state, strict=False)
    if missing:
        logger.warning("Missing keys when loading causal state: %s", missing)
    if unexpected:
        logger.warning("Unexpected keys: %s", unexpected)
    expert.eval()

    return expert, full_config


# ---------------------------------------------------------------------------
# Data loading & routing
# ---------------------------------------------------------------------------

def get_causal_indices(full_config: dict) -> Tuple[np.ndarray, List[Tuple[str, int, int]]]:
    """Compute the flat column indices the causal expert sees.

    Returns (flat_indices, group_spans) where group_spans is a list of
    (group_name, start_in_flat, end_in_flat) so downstream code can
    aggregate attribution mass per group.
    """
    schema = full_config.get("feature_schema", {})
    group_ranges = schema.get("feature_group_ranges", {})
    routing = schema.get("expert_routing", [])

    causal_groups = None
    for entry in routing:
        if entry.get("expert_name") == "causal":
            causal_groups = entry.get("input_groups", [])
            break
    if not causal_groups:
        raise RuntimeError("No causal routing entry found")

    # Sort by start index (FeatureRouter behavior).
    ordered = sorted(
        [(g, *group_ranges[g]) for g in causal_groups],
        key=lambda t: t[1],
    )
    indices: List[int] = []
    spans: List[Tuple[str, int, int]] = []
    pos = 0
    for name, s, e in ordered:
        n = e - s
        indices.extend(range(s, e))
        spans.append((name, pos, pos + n))
        pos += n
    return np.asarray(indices, dtype=np.int64), spans


def load_val_features(
    parquet_path: Path,
    schema_path: Path,
    val_frac: float = 0.2,
    max_samples: int = 5000,
    seed: int = 0,
) -> np.ndarray:
    """Load the last val_frac rows (temporal tail) as a numpy feature matrix.

    Santander final parquet contains all splits concatenated. We take
    the temporal tail as the val proxy; fine-grained split boundaries
    are not needed for attribution analysis.
    """
    import duckdb

    with open(schema_path, "r") as f:
        schema = json.load(f)
    cols: List[str] = schema["columns"]

    total_q = duckdb.execute(f"SELECT COUNT(*) FROM '{parquet_path.as_posix()}'")
    total = int(total_q.fetchone()[0])
    val_start = int(total * (1.0 - val_frac))

    col_list = ", ".join(f'"{c}"' for c in cols)
    sql = (
        f"SELECT {col_list} FROM '{parquet_path.as_posix()}' "
        f"LIMIT {total - val_start} OFFSET {val_start}"
    )
    df = duckdb.execute(sql).df()
    if len(df) > max_samples:
        rng = np.random.default_rng(seed)
        idx = rng.choice(len(df), size=max_samples, replace=False)
        df = df.iloc[idx].reset_index(drop=True)

    arr = df.to_numpy(dtype=np.float32)
    # Fill any NaN with 0 (scaler output shouldn't have them but be safe).
    arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
    return arr


# ---------------------------------------------------------------------------
# Attribution computations
# ---------------------------------------------------------------------------

@torch.no_grad()
def get_ceh_attribution(expert: CausalExpert, x: torch.Tensor) -> torch.Tensor:
    """Run a forward pass and return the CEH head output (no grad needed)."""
    _ = expert(x)
    return expert._last_attribution.detach()  # (batch, input_dim)


def get_grad_times_input(expert: CausalExpert, x: torch.Tensor) -> torch.Tensor:
    """Compute gradient × input of causal_encoder output sum w.r.t. x.

    This replicates the head's training target without the training-time
    wrapper, giving a clean per-sample per-feature saliency baseline.
    """
    x_clone = x.detach().clone().requires_grad_(True)
    z = expert.feature_compressor(x_clone)
    z_hat = expert._apply_causal_mechanism(z)
    out = expert.causal_encoder(z_hat)
    scalar = out.sum()
    grad_x = torch.autograd.grad(scalar, x_clone, retain_graph=False)[0]
    return (grad_x * x_clone).detach()


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def per_sample_spearman(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Spearman correlation computed per row (ignoring rows with zero variance)."""
    out = np.full(a.shape[0], np.nan, dtype=np.float64)
    for i in range(a.shape[0]):
        ai, bi = a[i], b[i]
        if ai.std() < 1e-12 or bi.std() < 1e-12:
            continue
        rho, _ = spearmanr(ai, bi)
        out[i] = rho
    return out


def discriminative_power(attr: np.ndarray) -> dict:
    """Ratio of between-sample to within-sample variance.

    If the CEH head outputs nearly identical scores for every sample
    (collapsed to a global importance vector), the ratio approaches 0.
    If scores are very sample-specific, the ratio is large.
    """
    col_var = attr.var(axis=0).mean()  # per-feature variance across samples
    per_sample_var = attr.var(axis=1).mean()
    ratio = float(col_var / (per_sample_var + 1e-12))
    # Also compute top-K overlap: do different samples attribute to the
    # same features or different ones?
    k = max(5, attr.shape[1] // 10)
    top_sets = [set(np.argsort(-np.abs(row))[:k].tolist()) for row in attr]
    n = len(top_sets)
    if n > 1:
        overlaps = []
        for i in range(min(n, 500)):
            for j in range(i + 1, min(n, 500)):
                inter = len(top_sets[i] & top_sets[j])
                overlaps.append(inter / k)
        mean_topk_overlap = float(np.mean(overlaps))
    else:
        mean_topk_overlap = float("nan")
    return {
        "between_sample_var": float(col_var),
        "within_sample_var": float(per_sample_var),
        "between_over_within": ratio,
        "mean_topk_overlap": mean_topk_overlap,
        "topk": k,
    }


def stability_under_noise(
    expert: CausalExpert,
    x: torch.Tensor,
    noise_std: float = 0.05,
    seed: int = 0,
) -> float:
    """Spearman correlation between attr(x) and attr(x + eps)."""
    a0 = get_ceh_attribution(expert, x).cpu().numpy()
    g = torch.Generator().manual_seed(seed)
    noise = torch.randn(x.shape, generator=g) * noise_std
    a1 = get_ceh_attribution(expert, x + noise).cpu().numpy()
    return float(np.nanmean(per_sample_spearman(a0, a1)))


def per_group_mass(
    attr: np.ndarray, spans: List[Tuple[str, int, int]]
) -> Dict[str, float]:
    """Return mean |attribution| per feature group (normalized to 1)."""
    totals = {}
    abs_attr = np.abs(attr).mean(axis=0)  # per-feature avg over samples
    for name, s, e in spans:
        totals[name] = float(abs_attr[s:e].sum())
    total_mass = sum(totals.values()) + 1e-12
    return {k: v / total_mass for k, v in totals.items()}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--ckpt",
        default="outputs/sagemaker_teacher_30ep/teacher_ceh/extracted/model.pth",
    )
    p.add_argument(
        "--config",
        default="outputs/sagemaker_teacher_30ep/teacher_ceh/extracted/config.json",
    )
    p.add_argument(
        "--parquet", default="outputs/phase0_v12/santander_final.parquet"
    )
    p.add_argument(
        "--schema", default="outputs/phase0_v12/feature_schema.json"
    )
    p.add_argument("--max-samples", type=int, default=5000)
    p.add_argument("--noise-std", type=float, default=0.05)
    p.add_argument(
        "--out",
        default="outputs/ceh_attribution_eval.json",
    )
    args = p.parse_args()

    ckpt_path = Path(args.ckpt)
    cfg_path = Path(args.config)
    expert, full_config = load_causal_expert(ckpt_path, cfg_path)
    logger.info("Loaded CausalExpert: input=%d output=%d n_vars=%d ceh=%s",
                expert.ceh_input_dim, expert.output_dim,
                expert.n_causal_vars, expert.ceh_enabled)

    causal_idx, spans = get_causal_indices(full_config)
    logger.info("Causal expert sees %d features across %d groups",
                len(causal_idx), len(spans))

    # Load val features, slice to causal-expert input columns.
    features = load_val_features(
        Path(args.parquet), Path(args.schema),
        val_frac=0.2, max_samples=args.max_samples,
    )
    x_np = features[:, causal_idx]
    logger.info("Val sample matrix: %s (causal slice %s)",
                features.shape, x_np.shape)
    x = torch.from_numpy(x_np)

    # 1. CEH attribution + grad × input
    attr_ceh = get_ceh_attribution(expert, x).cpu().numpy()
    attr_grad = get_grad_times_input(expert, x).cpu().numpy()
    rho_per_sample = per_sample_spearman(attr_ceh, attr_grad)
    training_fit = {
        "spearman_mean": float(np.nanmean(rho_per_sample)),
        "spearman_median": float(np.nanmedian(rho_per_sample)),
        "spearman_p25": float(np.nanpercentile(rho_per_sample, 25)),
        "spearman_p75": float(np.nanpercentile(rho_per_sample, 75)),
        "n_samples_scored": int(np.sum(~np.isnan(rho_per_sample))),
    }

    # 2. Discriminative power
    disc = discriminative_power(attr_ceh)

    # 3. Stability under noise
    stab = stability_under_noise(expert, x, noise_std=args.noise_std)

    # 4. Per-group mass distribution
    group_mass = per_group_mass(attr_ceh, spans)

    # 5. Compare to grad × input's own group mass for sanity
    group_mass_grad = per_group_mass(attr_grad, spans)

    results = {
        "checkpoint": str(ckpt_path),
        "n_samples": int(x.shape[0]),
        "n_features": int(x.shape[1]),
        "training_fit_spearman_ceh_vs_gradxinput": training_fit,
        "discriminative_power": disc,
        "stability_under_noise": {
            "noise_std": args.noise_std,
            "spearman_self_vs_noised": stab,
        },
        "per_group_mass_ceh": group_mass,
        "per_group_mass_gradxinput": group_mass_grad,
        "feature_groups": [{"name": n, "start": s, "end": e} for n, s, e in spans],
    }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info("Wrote results to %s", out_path)

    # Pretty print a compact summary.
    print()
    print("=" * 72)
    print("CEH ATTRIBUTION QUALITY - MV EVALUATION")
    print("=" * 72)
    print(f"Samples: {x.shape[0]}   Causal input dim: {x.shape[1]}")
    print()
    print("1. Training fit (CEH vs gradient x input):")
    print(f"   Spearman  mean={training_fit['spearman_mean']:.3f}"
          f"  median={training_fit['spearman_median']:.3f}"
          f"  [{training_fit['spearman_p25']:.3f}, "
          f"{training_fit['spearman_p75']:.3f}]")
    print()
    print("2. Discriminative power:")
    print(f"   between/within variance ratio: {disc['between_over_within']:.4f}")
    print(f"   mean top-{disc['topk']} overlap across samples: "
          f"{disc['mean_topk_overlap']:.3f}"
          "  (1.0 = same set, 0.0 = disjoint)")
    print()
    print(f"3. Stability under noise (sigma={args.noise_std}):")
    print(f"   Spearman attr(x) vs attr(x+noise) = {stab:.3f}")
    print()
    print("4. Attribution mass per feature group:")
    print(f"   {'group':24s} {'CEH':>10s} {'grad_x_input':>13s}")
    for name in [s[0] for s in spans]:
        print(f"   {name:24s} {group_mass[name]*100:>9.2f}%"
              f" {group_mass_grad[name]*100:>11.2f}%")
    print("=" * 72)


if __name__ == "__main__":
    main()
