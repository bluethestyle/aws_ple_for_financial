"""Smoke test: CausalExpert with ceh.target_mode='demeaned'.

Constructs a tiny CausalExpert with CEH enabled in demeaned mode,
runs a training-mode forward, verifies the attribution target is
actually demeaned (row mean ~ 0), and checks attribution loss is
finite. No GPU, <1s runtime.
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.model.experts.causal import CausalExpert


def main() -> None:
    torch.manual_seed(0)
    input_dim = 32
    expert = CausalExpert(
        input_dim=input_dim,
        config={
            "output_dim": 16,
            "hidden_dim": 32,
            "n_causal_vars": 8,
            "ceh": {
                "enabled": True,
                "hidden_dim": 16,
                "loss_weight": 0.1,
                "dropout": 0.0,
                "target_mode": "demeaned",
            },
        },
    )
    expert.train()
    x = torch.randn(64, input_dim)
    out = expert(x)
    assert out.shape == (64, 16), f"bad output shape {out.shape}"

    tgt = expert._last_attr_target
    assert tgt is not None, "attribution target not computed"
    # Demeaned => each feature column has ~zero mean across the batch
    col_mean = tgt.mean(dim=0)
    max_abs = col_mean.abs().max().item()
    assert max_abs < 1e-5, f"demeaned target not zero-mean (max {max_abs:.2e})"
    print(f"demeaned target per-column mean max abs: {max_abs:.2e} (pass)")

    attr_loss = expert.get_attribution_loss()
    assert torch.isfinite(attr_loss), "attribution loss is not finite"
    print(f"attribution loss: {float(attr_loss):.6f} (finite)")

    # Sanity: switching to 'raw' should NOT demean (columns mean != 0).
    expert_raw = CausalExpert(
        input_dim=input_dim,
        config={
            "output_dim": 16,
            "hidden_dim": 32,
            "n_causal_vars": 8,
            "ceh": {
                "enabled": True,
                "hidden_dim": 16,
                "loss_weight": 0.1,
                "dropout": 0.0,
                "target_mode": "raw",
            },
        },
    )
    expert_raw.train()
    _ = expert_raw(x)
    tgt_raw = expert_raw._last_attr_target
    assert tgt_raw is not None
    raw_col_mean_abs = tgt_raw.mean(dim=0).abs().max().item()
    print(f"raw target per-column mean max abs: {raw_col_mean_abs:.2e} (should be > demeaned)")
    assert raw_col_mean_abs > max_abs, "raw target should not be demeaned"

    print()
    print("ALL CHECKS PASSED")


if __name__ == "__main__":
    main()
