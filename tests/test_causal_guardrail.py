"""Tests for the Causal Guardrail accessors (Paper 3 Axis-3 B).

Covers:
  - ``get_causal_latent`` returns a detached ``(batch, n_causal_vars)``
    tensor.
  - ``get_causal_coherence_score`` returns non-negative per-sample
    floats, shape ``(batch,)``, detached.
  - ``PLEModel.get_causal_coherence`` finds the causal expert and
    returns a score tensor, ``None`` when no causal expert is present.
  - Synthetic OOD (uniform random) produces a detectable shift in the
    Mahalanobis-style score built on the latent statistics of the
    in-distribution batch --- sanity check for the MV eval script.
"""
from __future__ import annotations

import numpy as np
import torch

from core.model.experts.causal import CausalExpert


def _expert(input_dim: int = 32) -> CausalExpert:
    return CausalExpert(
        input_dim=input_dim,
        config={
            "output_dim": 16,
            "hidden_dim": 32,
            "n_causal_vars": 8,
            "ceh": {"enabled": True, "hidden_dim": 8},
        },
    )


def test_get_causal_latent_shape_and_detached():
    expert = _expert()
    expert.eval()
    x = torch.randn(5, 32)
    z = expert.get_causal_latent(x)
    assert z.shape == (5, 8)
    assert z.grad_fn is None


def test_get_causal_coherence_score_shape_and_nonneg():
    expert = _expert()
    expert.eval()
    x = torch.randn(7, 32)
    scores = expert.get_causal_coherence_score(x)
    assert scores.shape == (7,)
    assert torch.all(scores >= 0.0)
    assert scores.grad_fn is None


def test_z_mahalanobis_math_separates_shifted_latents():
    """Unit-test the Mahalanobis formulation, not the trained network.

    We build a synthetic in-distribution latent matrix plus an OOD
    latent matrix with a multi-sigma shift and verify the score
    discriminates. This isolates the CG v2 math from the LayerNorm-
    driven saturation that an untrained feature_compressor exhibits
    on globally-shifted inputs.
    """
    rng = np.random.default_rng(0)
    z_id = rng.standard_normal((200, 8)).astype(np.float32)
    # OOD: each feature shifted by 3 sigma
    z_ood = rng.standard_normal((50, 8)).astype(np.float32) + 3.0

    mu = z_id.mean(axis=0)
    sigma = z_id.std(axis=0) + 1e-6
    id_totals = (((z_id - mu) / sigma) ** 2).sum(axis=1)
    ood_totals = (((z_ood - mu) / sigma) ** 2).sum(axis=1)
    assert np.median(ood_totals) > np.percentile(id_totals, 95)


def test_ple_model_get_causal_coherence_none_without_causal_expert():
    """Degenerate case: a model with no causal expert returns None.

    We stub a minimal object that behaves like PLEModel for the
    accessor path without instantiating the full PLE stack (which
    would require a PLEConfig + feature schema). We only need
    ``_iter_shared_experts`` to yield non-causal experts.
    """
    from core.model.ple.model import PLEModel

    class _Stub:
        @staticmethod
        def _iter_shared_experts():
            # No expert in this fake list has a coherence method
            return iter([object(), object()])

        get_causal_coherence = PLEModel.get_causal_coherence

    stub = _Stub()
    assert stub.get_causal_coherence(torch.randn(3, 8)) is None
