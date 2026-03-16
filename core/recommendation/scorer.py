"""
Scoring System
==============

Pluggable scoring layer that transforms raw model predictions into a
single priority score per customer-item pair.

Architecture:
    AbstractScorer          -- Base class for all scorers.
    ScorerRegistry          -- Decorator-based plugin registry.
    WeightedSumScorer       -- Default: configurable task weight sum.
    FDTVSScorer (plugin)    -- 4-stage composite scoring (Financial DNA example).

All weights, thresholds, and coefficients are read from a config dict --
nothing is hardcoded.
"""

from __future__ import annotations

import logging
import math
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, ClassVar, Dict, List, Optional, Type

import numpy as np

logger = logging.getLogger(__name__)

__all__ = [
    "AbstractScorer",
    "ScorerRegistry",
    "ScoringResult",
    "WeightedSumScorer",
    "FDTVSScorer",
]


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass
class ScoringResult:
    """Output of a scorer for a single customer-item pair.

    Attributes:
        customer_id: Unique customer identifier.
        item_id: Unique item / product identifier.
        score: Final composite score (higher is better).
        components: Named intermediate values for auditability.
        metadata: Arbitrary extra data attached by the scorer.
    """

    customer_id: str
    item_id: str
    score: float
    components: Dict[str, float] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Registry (plugin pattern)
# ---------------------------------------------------------------------------

class ScorerRegistry:
    """Global registry of scorer implementations.

    Usage::

        @ScorerRegistry.register("weighted_sum")
        class WeightedSumScorer(AbstractScorer):
            ...

        scorer = ScorerRegistry.create("weighted_sum", config)
    """

    _registry: ClassVar[Dict[str, Type["AbstractScorer"]]] = {}

    @classmethod
    def register(cls, name: str):
        """Decorator that registers a scorer class under *name*."""

        def decorator(scorer_cls: Type[AbstractScorer]):
            if name in cls._registry:
                logger.warning(
                    "ScorerRegistry: overwriting existing scorer '%s'", name,
                )
            cls._registry[name] = scorer_cls
            return scorer_cls

        return decorator

    @classmethod
    def create(cls, name: str, config: Dict[str, Any]) -> "AbstractScorer":
        """Instantiate a registered scorer by name.

        Args:
            name: Registry key (e.g. ``"weighted_sum"``).
            config: Full config dict -- the scorer's ``__init__`` receives
                    ``config.get("scorer", {}).get(name, {})``.

        Raises:
            KeyError: If *name* is not registered.
        """
        if name not in cls._registry:
            available = ", ".join(sorted(cls._registry)) or "(none)"
            raise KeyError(
                f"Unknown scorer '{name}'. Available: {available}"
            )
        scorer_config = config.get("scorer", {}).get(name, {})
        return cls._registry[name](scorer_config)

    @classmethod
    def list_registered(cls) -> List[str]:
        """Return sorted list of registered scorer names."""
        return sorted(cls._registry)


# ---------------------------------------------------------------------------
# Abstract base
# ---------------------------------------------------------------------------

class AbstractScorer(ABC):
    """Base class for all scoring implementations.

    Subclasses must implement :meth:`score`.
    """

    def __init__(self, config: Dict[str, Any]) -> None:
        self.config = config

    @abstractmethod
    def score(
        self,
        customer_id: str,
        item_id: str,
        predictions: Dict[str, float],
        context: Optional[Dict[str, Any]] = None,
    ) -> ScoringResult:
        """Compute a priority score for one customer-item pair.

        Args:
            customer_id: Customer identifier.
            item_id: Item / product identifier.
            predictions: Task-keyed model predictions
                         (e.g. ``{"ctr": 0.12, "cvr": 0.08}``).
            context: Optional extra signals (fatigue, engagement, etc.).

        Returns:
            A :class:`ScoringResult` with the final composite score.
        """

    def score_batch(
        self,
        records: List[Dict[str, Any]],
    ) -> List[ScoringResult]:
        """Score a batch of customer-item pairs.

        Each element of *records* must contain at least:
        ``customer_id``, ``item_id``, ``predictions``, and optionally
        ``context``.

        The default implementation loops over :meth:`score`; subclasses
        may override for vectorised execution.
        """
        results: List[ScoringResult] = []
        for rec in records:
            result = self.score(
                customer_id=rec["customer_id"],
                item_id=rec["item_id"],
                predictions=rec["predictions"],
                context=rec.get("context"),
            )
            results.append(result)
        return results


# ---------------------------------------------------------------------------
# Default: Weighted-sum scorer
# ---------------------------------------------------------------------------

@ScorerRegistry.register("weighted_sum")
class WeightedSumScorer(AbstractScorer):
    """Simple task-weighted sum scorer (default).

    Config example (YAML)::

        scorer:
          weighted_sum:
            weights:
              ctr: 0.30
              cvr: 0.40
              nba: 0.20
              ltv: 0.10
            min_score: 0.0
            max_score: 1.0

    Missing prediction keys are silently skipped (weight redistributed).
    """

    def __init__(self, config: Dict[str, Any]) -> None:
        super().__init__(config)
        self.weights: Dict[str, float] = config.get("weights", {})
        if not self.weights:
            raise ValueError(
                "WeightedSumScorer: 'weights' must be provided and non-empty. "
                "Example: {\"ctr\": 0.30, \"cvr\": 0.40, \"nba\": 0.20, \"ltv\": 0.10}"
            )
        self.min_score: float = config.get("min_score", 0.0)
        self.max_score: float = config.get("max_score", 1.0)

        total = sum(self.weights.values())
        if total > 0 and abs(total - 1.0) > 1e-6:
            logger.warning(
                "WeightedSumScorer: weights sum to %.4f, not 1.0", total,
            )

    def score(
        self,
        customer_id: str,
        item_id: str,
        predictions: Dict[str, float],
        context: Optional[Dict[str, Any]] = None,
    ) -> ScoringResult:
        weighted_sum = 0.0
        active_weight = 0.0
        components: Dict[str, float] = {}

        for task, weight in self.weights.items():
            if task in predictions:
                contribution = weight * predictions[task]
                weighted_sum += contribution
                active_weight += weight
                components[f"w_{task}"] = weight
                components[f"p_{task}"] = predictions[task]

        # Normalise if some tasks are missing
        if active_weight > 0 and abs(active_weight - 1.0) > 1e-6:
            weighted_sum /= active_weight

        score = float(np.clip(weighted_sum, self.min_score, self.max_score))

        return ScoringResult(
            customer_id=customer_id,
            item_id=item_id,
            score=score,
            components=components,
        )


# ---------------------------------------------------------------------------
# Plugin example: FD-TVS (4-stage composite)
# ---------------------------------------------------------------------------

@ScorerRegistry.register("fd_tvs")
class FDTVSScorer(AbstractScorer):
    """4-Stage composite scoring -- a plugin example.

    FD-TVS formula::

        FD-TVS = S_task * W_modifier * V_velocity
                 * max(0, 1 - R_penalty)
                 * fatigue_factor * engagement_boost

    Stage 1 -- Task weighted sum  (S_task)
    Stage 2 -- Modifier           (W_modifier, from context signal)
    Stage 3 -- Behavioral velocity (V_velocity = 1 + gamma * flare)
    Stage 4 -- Risk penalty + fatigue decay + engagement boost

    All coefficients come from config::

        scorer:
          fd_tvs:
            task_weights:
              ctr: 0.30
              cvr: 0.40
              nba: 0.20
              ltv: 0.10
            modifier_map:
              high: 1.2
              medium: 1.0
              low: 0.8
            modifier_default: 1.0
            gamma_velocity: 0.15
            risk_weight_limit_util: 0.3
            risk_weight_churn: 0.5
            risk_weight_message_freq: 0.2
            risk_threshold_limit_util: 0.8
            risk_threshold_churn: 0.3
            risk_threshold_message_count: 3
            fatigue_base_decay: 0.85
            fatigue_channel_multiplier:
              app_push: 1.0
              email: 0.7
              sms: 0.9
            engagement_boost_scale: 0.15
    """

    def __init__(self, config: Dict[str, Any]) -> None:
        super().__init__(config)

        # Stage 1
        self.task_weights: Dict[str, float] = config.get("task_weights", {})
        if not self.task_weights:
            raise ValueError(
                "FDTVSScorer: 'task_weights' must be provided and non-empty. "
                "Example: {\"ctr\": 0.30, \"cvr\": 0.40, \"nba\": 0.20, \"ltv\": 0.10}"
            )

        # Stage 2
        self.modifier_map: Dict[str, float] = config.get("modifier_map", {
            "high": 1.2, "medium": 1.0, "low": 0.8,
        })
        self.modifier_default: float = config.get("modifier_default", 1.0)

        # Stage 3
        self.gamma_velocity: float = config.get("gamma_velocity", 0.15)

        # Stage 4 -- risk
        self.risk_w_limit = config.get("risk_weight_limit_util", 0.3)
        self.risk_w_churn = config.get("risk_weight_churn", 0.5)
        self.risk_w_msg = config.get("risk_weight_message_freq", 0.2)
        self.risk_t_limit = config.get("risk_threshold_limit_util", 0.8)
        self.risk_t_churn = config.get("risk_threshold_churn", 0.3)
        self.risk_t_msg = config.get("risk_threshold_message_count", 3)

        # Fatigue & engagement
        self.fatigue_base_decay: float = config.get("fatigue_base_decay", 0.85)
        self.fatigue_channel_mult: Dict[str, float] = config.get(
            "fatigue_channel_multiplier", {"default": 1.0},
        )
        self.engagement_boost_scale: float = config.get("engagement_boost_scale", 0.15)

    # -- stages -------------------------------------------------------

    def _stage1_task_score(self, predictions: Dict[str, float]) -> float:
        """Weighted sum of task predictions."""
        s = 0.0
        w_total = 0.0
        for task, weight in self.task_weights.items():
            if task in predictions:
                s += weight * predictions[task]
                w_total += weight
        return s / w_total if w_total > 0 else 0.0

    def _stage2_modifier(self, context: Dict[str, Any]) -> float:
        """Look up modifier from context signal (e.g. customer segment)."""
        segment = context.get("modifier_segment", "")
        return self.modifier_map.get(segment, self.modifier_default)

    def _stage3_velocity(self, context: Dict[str, Any]) -> float:
        """Behavioral velocity: 1 + gamma * flare."""
        flare = 1.0 if context.get("flare_detected", False) else 0.0
        return 1.0 + self.gamma_velocity * flare

    def _stage4_risk_penalty(self, context: Dict[str, Any]) -> float:
        """Weighted risk penalty in [0, 1]."""
        penalty = 0.0
        limit_util = context.get("limit_util", 0.0)
        churn_prob = context.get("churn_prob", 0.0)
        n_messages = context.get("n_messages_7d", 0)

        if limit_util > self.risk_t_limit:
            penalty += self.risk_w_limit
        if churn_prob > self.risk_t_churn:
            penalty += self.risk_w_churn
        if n_messages > self.risk_t_msg:
            penalty += self.risk_w_msg

        return float(np.clip(penalty, 0.0, 1.0))

    def _fatigue_factor(self, context: Dict[str, Any]) -> float:
        """Exponential fatigue decay based on recent message count."""
        n_messages = context.get("n_messages_7d", 0)
        channel = context.get("channel", "default")
        ch_mult = self.fatigue_channel_mult.get(
            channel, self.fatigue_channel_mult.get("default", 1.0),
        )
        return float(self.fatigue_base_decay ** (n_messages * ch_mult))

    def _engagement_boost(self, context: Dict[str, Any]) -> float:
        """Engagement multiplier: 1 + scale * engagement_score."""
        eng = context.get("engagement_score", 0.0)
        return 1.0 + self.engagement_boost_scale * float(eng)

    # -- public API ---------------------------------------------------

    def score(
        self,
        customer_id: str,
        item_id: str,
        predictions: Dict[str, float],
        context: Optional[Dict[str, Any]] = None,
    ) -> ScoringResult:
        ctx = context or {}

        s_task = self._stage1_task_score(predictions)
        w_mod = self._stage2_modifier(ctx)
        v_vel = self._stage3_velocity(ctx)
        r_pen = self._stage4_risk_penalty(ctx)
        f_fat = self._fatigue_factor(ctx)
        e_boost = self._engagement_boost(ctx)

        raw = s_task * w_mod * v_vel * max(0.0, 1.0 - r_pen) * f_fat * e_boost

        # NaN guard
        if math.isnan(raw):
            logger.warning(
                "FDTVSScorer: NaN for customer=%s item=%s, defaulting to 0.0",
                customer_id, item_id,
            )
            raw = 0.0

        return ScoringResult(
            customer_id=customer_id,
            item_id=item_id,
            score=raw,
            components={
                "s_task": s_task,
                "w_modifier": w_mod,
                "v_velocity": v_vel,
                "risk_penalty": r_pen,
                "fatigue_factor": f_fat,
                "engagement_boost": e_boost,
            },
        )
