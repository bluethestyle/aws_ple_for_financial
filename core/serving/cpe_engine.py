"""
Counterfactual Policy Evaluation Engine (Stage C Production)
=============================================================

Real-time CPE scoring for serving-layer deployment decisions.
Wraps :class:`core.evaluation.counterfactual.CounterfactualEvaluator`
with streaming / per-request evaluation capability.

This is a STUB -- production implementation is not needed for ablation.
Use :class:`core.evaluation.counterfactual.CounterfactualEvaluator` for
offline evaluation instead.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import numpy as np


__all__ = ["CPEEngine", "CPEDecision"]


@dataclass
class CPEDecision:
    """Per-request CPE decision result."""

    go: bool
    estimator: str          # "ips" | "snips" | "dr"
    estimated_value: float
    confidence_interval: tuple
    sensitivity_pass: bool


class CPEEngine:
    """Real-time CPE scoring engine for production serving.

    This is a Stage C production module. Not required for ablation.

    Args:
        propensity_model_path: Path to serialised propensity model.
        config: Engine configuration dict.
    """

    def __init__(
        self,
        propensity_model_path: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None,
    ) -> None:
        raise NotImplementedError(
            "CPEEngine is a Stage C production module. "
            "Use core.evaluation.counterfactual.CounterfactualEvaluator "
            "for offline ablation evaluation."
        )

    def evaluate_request(
        self,
        user_features: np.ndarray,
        recommended_items: List[str],
        context: Optional[Dict[str, Any]] = None,
    ) -> CPEDecision:
        """Evaluate a single recommendation request via CPE."""
        raise NotImplementedError

    def evaluate_batch(
        self,
        user_features: np.ndarray,
        recommended_items: List[List[str]],
    ) -> List[CPEDecision]:
        """Evaluate a batch of recommendation requests."""
        raise NotImplementedError

    def update_propensity(self, model_path: str) -> None:
        """Hot-reload propensity model without downtime."""
        raise NotImplementedError
