"""MV evaluation of the Causal Counterfactual Probe (Paper 3 Axis-3 E).

Pearl Rung 3 (counterfactuals) demo on the learned DAG.

For each of N validation samples and each of d causal-latent
dimensions, intervene do(z_j = v) and compare:

  direct_effect  = || encoder(z' + z W^2) - encoder(z + z W^2) ||
  total_effect   = || encoder(z' + z' W^2) - encoder(z + z W^2) ||
  mediated       = || encoder(z' + z' W^2) - encoder(z' + z W^2) ||

A DAG that actually mediates counterfactuals produces a non-trivial
``mediated / total`` ratio. A decorative $W$ (Finding 10 baseline)
collapses ``mediated`` toward zero --- direct and full-CF converge
because $W^2$ is too small to propagate the intervention.

Compares both the baseline (teacher_ceh_demeaned, $||W||_F = 0.36$)
and the W-amplified (teacher_ceh_w_amp, $||W||_F = 5.03$) checkpoints
so the CCP signal can be read against the W-magnitude gradient
established in Findings 10 and 11.

All on CPU. Runtime <30s for 1,000 samples $times$ 32 dims.
Zero SageMaker cost.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.eval_ceh_attribution import (  # noqa: E402
    load_causal_expert,
    get_causal_indices,
    load_val_features,
)

logger = logging.getLogger("eval_ccp")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")


# ---------------------------------------------------------------------------
# Core CCP computation
# ---------------------------------------------------------------------------

def ccp_per_checkpoint(
    ckpt_path: Path,
    config_path: Path,
    x: torch.Tensor,
    intervention_values: List[float],
) -> Dict[str, object]:
    """Run CCP over all causal-latent dims + intervention values."""
    expert, _ = load_causal_expert(ckpt_path, config_path)
    expert.eval()
    d = expert.n_causal_vars
    W_norm = float(torch.linalg.norm(expert.W.detach()).item())

    # Per (feature_idx, intervention) collect aggregate norms across samples
    direct_norms: List[float] = []
    total_norms: List[float] = []
    mediated_norms: List[float] = []
    mediation_ratios: List[float] = []  # mediated / total, per (f, v)

    for j in range(d):
        for v in intervention_values:
            out = expert.get_counterfactual(x, feature_idx=j,
                                             intervention_value=v)
            factual = out["factual"]
            direct_only = out["direct_only"]
            full_cf = out["full_cf"]

            # Per-sample L2 norms, averaged across samples
            direct = (direct_only - factual).norm(dim=-1).mean().item()
            total = (full_cf - factual).norm(dim=-1).mean().item()
            mediated = (full_cf - direct_only).norm(dim=-1).mean().item()
            direct_norms.append(direct)
            total_norms.append(total)
            mediated_norms.append(mediated)
            if total > 1e-12:
                mediation_ratios.append(mediated / total)

    def summarise(arr: List[float]) -> Dict[str, float]:
        a = np.asarray(arr, dtype=np.float64)
        return {
            "mean": float(a.mean()),
            "median": float(np.median(a)),
            "p95": float(np.percentile(a, 95)),
            "max": float(a.max()),
        }

    return {
        "w_frobenius": W_norm,
        "n_causal_vars": d,
        "n_interventions": len(intervention_values),
        "n_samples": int(x.shape[0]),
        "direct_effect_norm": summarise(direct_norms),
        "total_effect_norm": summarise(total_norms),
        "mediated_effect_norm": summarise(mediated_norms),
        "mediation_ratio": summarise(mediation_ratios),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--baseline-ckpt",
        default="outputs/sagemaker_teacher_30ep/teacher_ceh_demeaned/extracted/model.pth",
    )
    p.add_argument(
        "--baseline-config",
        default="outputs/sagemaker_teacher_30ep/teacher_ceh_demeaned/extracted/config.json",
    )
    p.add_argument(
        "--wamp-ckpt",
        default="outputs/sagemaker_teacher_30ep/teacher_ceh_w_amp/extracted/model.pth",
    )
    p.add_argument(
        "--wamp-config",
        default="outputs/sagemaker_teacher_30ep/teacher_ceh_w_amp/extracted/config.json",
    )
    p.add_argument(
        "--parquet", default="outputs/phase0_v12/santander_final.parquet"
    )
    p.add_argument(
        "--schema", default="outputs/phase0_v12/feature_schema.json"
    )
    p.add_argument("--max-samples", type=int, default=1000)
    p.add_argument(
        "--out",
        default="outputs/ccp_counterfactual_eval.json",
    )
    args = p.parse_args()

    # Load the causal expert's routed input slice once.
    import json as _json
    with open(args.baseline_config, "r") as f:
        full_config = _json.load(f)
    causal_idx, _ = get_causal_indices(full_config)
    features = load_val_features(
        Path(args.parquet), Path(args.schema),
        val_frac=0.2, max_samples=args.max_samples,
    )
    x = torch.from_numpy(features[:, causal_idx])
    logger.info("CCP sample matrix: %s", tuple(x.shape))

    # Use 3 intervention values spanning a reasonable range of z.
    # (z is a compressor output; post-LayerNorm values live around
    # [-3, 3] but are not strictly bounded.)
    interventions = [-2.0, 0.0, 2.0]

    logger.info("Evaluating baseline (teacher_ceh_demeaned) ...")
    baseline = ccp_per_checkpoint(
        Path(args.baseline_ckpt), Path(args.baseline_config),
        x, interventions,
    )
    logger.info("  ||W||_F = %.3f", baseline["w_frobenius"])

    logger.info("Evaluating W-amp (teacher_ceh_w_amp) ...")
    wamp = ccp_per_checkpoint(
        Path(args.wamp_ckpt), Path(args.wamp_config),
        x, interventions,
    )
    logger.info("  ||W||_F = %.3f", wamp["w_frobenius"])

    results = {
        "baseline": baseline,
        "w_amplified": wamp,
        "interventions_tested": interventions,
    }
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info("Wrote %s", out_path)

    # Compact printout
    print()
    print("=" * 72)
    print("CAUSAL COUNTERFACTUAL PROBE (CCP) - MV EVALUATION")
    print("=" * 72)

    def row(label: str, summary: Dict[str, float]) -> str:
        return (f"  {label:22s} mean={summary['mean']:.4f}"
                f" med={summary['median']:.4f}"
                f" p95={summary['p95']:.4f}"
                f" max={summary['max']:.4f}")

    for name, r in (("Baseline (demeaned)", baseline),
                    ("W-amp", wamp)):
        print()
        print(f"[{name}]  ||W||_F = {r['w_frobenius']:.3f}   "
              f"dims={r['n_causal_vars']}  "
              f"interventions={r['n_interventions']}  "
              f"samples={r['n_samples']}")
        print(row("direct_effect", r["direct_effect_norm"]))
        print(row("total_effect", r["total_effect_norm"]))
        print(row("mediated_effect", r["mediated_effect_norm"]))
        print(row("mediation_ratio", r["mediation_ratio"]))
    print()
    print("Key signal: mediation_ratio median.")
    print("  If baseline << W-amp -> DAG is the mediator (Pearl Rung 3 viable).")
    print("  If baseline ~ W-amp  -> W still decorative (Rung 3 blocked).")
    print("=" * 72)


if __name__ == "__main__":
    main()
