"""
Causal Guardrail (Paper 3 Findings 10 / 11).

Per-prediction reliability flag derived from the causal expert's
latent space. Pairs with the Causal Explainability Head attribution
(Paper 3 Finding 9 / Paper 2 v2) to provide a complete
*regulator-usable audit pair* at inference time:

  * CEH attribution  -> "why did the model recommend this?"
  * Causal guardrail -> "can we trust this recommendation?"

Both run on the same causal expert's forward pass; both feed the
same HMAC-signed, hash-chained ``AuditLogger`` store.

This module provides the operational wrapper around the core
primitives already in place:

  - ``CausalExpert.get_causal_latent`` / ``get_causal_coherence_score``
  - ``PLEModel.get_causal_coherence``
  - ``AuditLogger.log_guardrail``

Usage::

    guard = CausalGuardrail.calibrate(model, causal_input_batch,
                                      reference_percentile=95.0)
    result = guard.check(model, x_sample, sample_id="cust-123",
                         audit_logger=audit_logger)
    if result.triggered:
        # route to human review / layer-3 fallback
        ...
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch

logger = logging.getLogger(__name__)


@dataclass
class GuardrailResult:
    """Outcome of a single guardrail check."""
    sample_id: str
    coherence_score: float
    threshold: float
    triggered: bool


class CausalGuardrail:
    """Calibrated causal guardrail with z-space Mahalanobis scoring.

    Finding 10 showed the W-reconstruction formulation fails at
    chance-level discrimination; Finding 11's W-amplification shifts
    it to weak-but-nonzero but never approaches the latent-space
    baseline. This class therefore implements *only* the v2
    (z-space Mahalanobis) formulation, which hit 100% TPR at 5% FPR
    on all three synthetic OOD probes in both the baseline and the
    amplified checkpoint.

    Calibration caches per-dimension mean + standard deviation of
    the causal latent ``z`` computed on a reference in-distribution
    batch, plus a threshold set by a percentile of the in-dist
    scores (default p95).
    """

    def __init__(
        self,
        mu: np.ndarray,
        sigma: np.ndarray,
        threshold: float,
        reference_percentile: float,
    ) -> None:
        self.mu = mu.astype(np.float32)
        self.sigma = sigma.astype(np.float32) + 1e-6
        self.threshold = float(threshold)
        self.reference_percentile = float(reference_percentile)

    # --------------------------------------------------------------
    # Construction
    # --------------------------------------------------------------
    @classmethod
    def calibrate(
        cls,
        model,
        causal_input_batch: torch.Tensor,
        reference_percentile: float = 95.0,
    ) -> "CausalGuardrail":
        """Fit ``mu, sigma, threshold`` from a reference batch.

        ``causal_input_batch`` must already be sliced to the causal
        expert's input subset (``FeatureRouter.route(x, "causal")``).
        We call ``model.get_causal_coherence`` to walk the z-space
        distribution on the reference batch; the resulting
        Mahalanobis-like score's ``reference_percentile``-th
        percentile becomes the guardrail threshold.
        """
        if causal_input_batch.ndim != 2:
            raise ValueError(
                "causal_input_batch must be [batch, features]"
            )

        # Pull the causal latent for mu / sigma estimation.
        # PLE models that have get_ceh_attribution also have a causal
        # expert; the latent lives on that expert.
        z = _extract_latent(model, causal_input_batch)
        z_np = z.cpu().numpy()
        mu = z_np.mean(axis=0)
        sigma = z_np.std(axis=0)
        standardized = (z_np - mu) / (sigma + 1e-6)
        ref_scores = (standardized * standardized).sum(axis=-1)
        threshold = float(np.percentile(ref_scores, reference_percentile))
        logger.info(
            "CausalGuardrail calibrated: n_ref=%d threshold=%.4f "
            "(p%.1f of in-dist score distribution)",
            z_np.shape[0], threshold, reference_percentile,
        )
        return cls(
            mu=mu,
            sigma=sigma,
            threshold=threshold,
            reference_percentile=reference_percentile,
        )

    # --------------------------------------------------------------
    # Check
    # --------------------------------------------------------------
    def score(self, model, causal_input: torch.Tensor) -> np.ndarray:
        """Return the per-sample z-space Mahalanobis scores."""
        z = _extract_latent(model, causal_input).cpu().numpy()
        standardized = (z - self.mu) / self.sigma
        return (standardized * standardized).sum(axis=-1)

    def check(
        self,
        model,
        causal_input: torch.Tensor,
        sample_id: str,
        audit_logger=None,
        model_id: str = "ple-teacher",
        user: str = "system",
    ) -> GuardrailResult:
        """Score a single input and optionally emit an audit record.

        ``causal_input`` must be ``[1, input_dim]`` or ``[input_dim]``.
        For batch use, iterate or call ``score`` directly.
        """
        if causal_input.ndim == 1:
            causal_input = causal_input.unsqueeze(0)
        if causal_input.shape[0] != 1:
            raise ValueError(
                "check() expects a single sample; use score() for batch"
            )
        s = float(self.score(model, causal_input)[0])
        triggered = s > self.threshold
        result = GuardrailResult(
            sample_id=sample_id,
            coherence_score=s,
            threshold=self.threshold,
            triggered=triggered,
        )
        if audit_logger is not None:
            try:
                audit_logger.log_guardrail(
                    model_id=model_id,
                    sample_id=sample_id,
                    coherence_score=s,
                    threshold=self.threshold,
                    triggered=triggered,
                    user=user,
                )
            except Exception:  # pragma: no cover - non-blocking
                logger.warning(
                    "log_guardrail failed for sample_id=%s (non-fatal)",
                    sample_id, exc_info=True,
                )
        return result


def _extract_latent(model, causal_input: torch.Tensor) -> torch.Tensor:
    """Walk the model to the causal expert and pull its latent z.

    Accepts either a ``PLEModel`` (uses ``get_ceh_attribution`` path
    as the discovery mechanism) or a raw ``CausalExpert`` with the
    ``get_causal_latent`` accessor.
    """
    if hasattr(model, "get_causal_latent"):
        return model.get_causal_latent(causal_input)
    # PLEModel path: find the causal expert ourselves
    if hasattr(model, "_iter_shared_experts"):
        for expert in model._iter_shared_experts():
            if hasattr(expert, "get_causal_latent"):
                return expert.get_causal_latent(causal_input)
    raise AttributeError(
        "CausalGuardrail requires a model with get_causal_latent "
        "(CausalExpert) or a PLEModel containing a causal expert"
    )
