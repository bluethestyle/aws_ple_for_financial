"""
Constraint Filtering Engine
============================

Chains configurable filters that eliminate ineligible customer-item pairs
before top-K selection.

Architecture:
    AbstractFilter      -- Base class for all filters.
    FilterRegistry      -- Decorator-based plugin registry.
    FatigueFilter       -- Message frequency + channel decay.
    EligibilityFilter   -- Min score, churn threshold.
    OwnedProductFilter  -- Exclude already-owned items.
    ConstraintEngine    -- Orchestrator that chains filters from config.

All thresholds and parameters come from the config dict.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, ClassVar, Dict, List, Optional, Type

logger = logging.getLogger(__name__)

__all__ = [
    "AbstractFilter",
    "FilterRegistry",
    "FilterResult",
    "FatigueFilter",
    "EligibilityFilter",
    "OwnedProductFilter",
    "ConstraintEngine",
]


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass
class FilterResult:
    """Outcome of a single filter evaluation.

    Attributes:
        passed: Whether the candidate survived this filter.
        filter_name: Name of the filter that produced this result.
        reason: Human-readable explanation when ``passed`` is False.
        details: Arbitrary metadata for audit logging.
    """

    passed: bool
    filter_name: str
    reason: str = ""
    details: Dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

class FilterRegistry:
    """Global registry of filter implementations.

    Usage::

        @FilterRegistry.register("fatigue")
        class FatigueFilter(AbstractFilter):
            ...

        f = FilterRegistry.create("fatigue", config)
    """

    _registry: ClassVar[Dict[str, Type["AbstractFilter"]]] = {}

    @classmethod
    def register(cls, name: str):
        """Decorator that registers a filter class under *name*."""

        def decorator(filter_cls: Type[AbstractFilter]):
            if name in cls._registry:
                logger.warning(
                    "FilterRegistry: overwriting existing filter '%s'", name,
                )
            cls._registry[name] = filter_cls
            return filter_cls

        return decorator

    @classmethod
    def create(cls, name: str, config: Dict[str, Any]) -> "AbstractFilter":
        """Instantiate a registered filter by name.

        Args:
            name: Registry key.
            config: Full pipeline config; the filter receives
                    ``config.get("filters", {}).get(name, {})``.

        Raises:
            KeyError: If *name* is not registered.
        """
        if name not in cls._registry:
            available = ", ".join(sorted(cls._registry)) or "(none)"
            raise KeyError(
                f"Unknown filter '{name}'. Available: {available}"
            )
        filter_config = config.get("filters", {}).get(name, {})
        return cls._registry[name](filter_config)

    @classmethod
    def list_registered(cls) -> List[str]:
        """Return sorted list of registered filter names."""
        return sorted(cls._registry)


# ---------------------------------------------------------------------------
# Abstract base
# ---------------------------------------------------------------------------

class AbstractFilter(ABC):
    """Base class for all constraint filters.

    Subclasses must implement :meth:`evaluate`.
    """

    def __init__(self, config: Dict[str, Any]) -> None:
        self.config = config

    @abstractmethod
    def evaluate(
        self,
        customer_id: str,
        item_id: str,
        context: Dict[str, Any],
    ) -> FilterResult:
        """Decide whether (*customer_id*, *item_id*) passes this filter.

        Args:
            customer_id: Customer identifier.
            item_id: Item / product identifier.
            context: Shared context dict containing scores, customer state,
                     product metadata, etc.

        Returns:
            A :class:`FilterResult`.
        """


# ---------------------------------------------------------------------------
# Built-in filters
# ---------------------------------------------------------------------------

@FilterRegistry.register("fatigue")
class FatigueFilter(AbstractFilter):
    """Reject candidates when the customer is over-contacted.

    Config example::

        filters:
          fatigue:
            max_messages_7d: 3
            channel_decay:
              app_push: 0.85
              email: 0.70
              sms: 0.80
              default: 0.90
            min_fatigue_score: 0.2
    """

    def __init__(self, config: Dict[str, Any]) -> None:
        super().__init__(config)
        self.max_messages_7d: int = config.get("max_messages_7d", 3)
        self.channel_decay: Dict[str, float] = config.get("channel_decay", {
            "default": 0.90,
        })
        self.min_fatigue_score: float = config.get("min_fatigue_score", 0.2)

    def evaluate(
        self,
        customer_id: str,
        item_id: str,
        context: Dict[str, Any],
    ) -> FilterResult:
        n_messages = context.get("n_messages_7d", 0)
        channel = context.get("channel", "default")

        # Message count gate
        if n_messages >= self.max_messages_7d:
            return FilterResult(
                passed=False,
                filter_name="fatigue",
                reason=(
                    f"Message count {n_messages} >= max {self.max_messages_7d}"
                ),
                details={"n_messages_7d": n_messages, "channel": channel},
            )

        # Channel-weighted fatigue score
        decay = self.channel_decay.get(
            channel, self.channel_decay.get("default", 0.90),
        )
        fatigue_score = decay ** n_messages
        if fatigue_score < self.min_fatigue_score:
            return FilterResult(
                passed=False,
                filter_name="fatigue",
                reason=(
                    f"Fatigue score {fatigue_score:.3f} < "
                    f"min {self.min_fatigue_score}"
                ),
                details={
                    "fatigue_score": fatigue_score,
                    "channel": channel,
                    "n_messages_7d": n_messages,
                },
            )

        return FilterResult(
            passed=True,
            filter_name="fatigue",
            details={"fatigue_score": fatigue_score},
        )


@FilterRegistry.register("eligibility")
class EligibilityFilter(AbstractFilter):
    """Reject candidates below minimum score or above churn threshold.

    Config example::

        filters:
          eligibility:
            min_score: 0.10
            max_churn_prob: 0.70
    """

    def __init__(self, config: Dict[str, Any]) -> None:
        super().__init__(config)
        self.min_score: float = config.get("min_score", 0.10)
        self.max_churn_prob: float = config.get("max_churn_prob", 0.70)

    def evaluate(
        self,
        customer_id: str,
        item_id: str,
        context: Dict[str, Any],
    ) -> FilterResult:
        score = context.get("score", 0.0)
        if score < self.min_score:
            return FilterResult(
                passed=False,
                filter_name="eligibility",
                reason=f"Score {score:.4f} < min {self.min_score}",
                details={"score": score},
            )

        churn_prob = context.get("churn_prob", 0.0)
        if churn_prob > self.max_churn_prob:
            return FilterResult(
                passed=False,
                filter_name="eligibility",
                reason=(
                    f"Churn probability {churn_prob:.4f} > "
                    f"max {self.max_churn_prob}"
                ),
                details={"churn_prob": churn_prob},
            )

        return FilterResult(
            passed=True,
            filter_name="eligibility",
            details={"score": score, "churn_prob": churn_prob},
        )


@FilterRegistry.register("suitability")
class SuitabilityFilter(AbstractFilter):
    """금소법 §17 적합성 원칙 필터 (Sprint 2 S13).

    Korean Financial Consumer Protection Act (금소법) Article 17 requires
    that a product's risk level not exceed the customer's assessed risk
    tolerance. Unsuitable combinations must be blocked before recommendation.

    Context keys (all optional but at least risk_tolerance + risk_level
    pair must be present):
        - ``customer_risk_tolerance`` (1-5, higher = more tolerant)
        - ``item_risk_level`` (1-5, higher = more risky)
        - ``customer_age`` (int)
        - ``customer_income`` (float, KRW annualized)

    Config example::

        filters:
          suitability:
            require_assessment: true  # reject if risk_tolerance missing
            senior_age_threshold: 65  # force conservative for seniors
            senior_max_risk_level: 2  # cap risk for senior customers
            low_income_threshold: 30000000  # 3천만원
            low_income_max_risk_level: 3
    """

    def __init__(self, config: Dict[str, Any]) -> None:
        super().__init__(config)
        self.require_assessment: bool = bool(
            config.get("require_assessment", True)
        )
        self.senior_age_threshold: int = int(
            config.get("senior_age_threshold", 65)
        )
        self.senior_max_risk_level: int = int(
            config.get("senior_max_risk_level", 2)
        )
        self.low_income_threshold: float = float(
            config.get("low_income_threshold", 30_000_000)
        )
        self.low_income_max_risk_level: int = int(
            config.get("low_income_max_risk_level", 3)
        )

    def evaluate(
        self,
        customer_id: str,
        item_id: str,
        context: Dict[str, Any],
    ) -> FilterResult:
        tolerance = context.get("customer_risk_tolerance")
        risk_level = context.get("item_risk_level")

        if risk_level is None:
            # No risk profile on the item: pass-through (other filters apply).
            return FilterResult(
                passed=True, filter_name="suitability",
                details={"reason": "no_item_risk_level"},
            )

        if tolerance is None:
            if self.require_assessment:
                return FilterResult(
                    passed=False,
                    filter_name="suitability",
                    reason="금소법 §17: missing customer_risk_tolerance",
                    details={"item_risk_level": risk_level},
                )
            return FilterResult(
                passed=True, filter_name="suitability",
                details={"reason": "tolerance_missing_but_not_required"},
            )

        if int(risk_level) > int(tolerance):
            return FilterResult(
                passed=False,
                filter_name="suitability",
                reason=(
                    f"금소법 §17: item risk {risk_level} exceeds "
                    f"customer tolerance {tolerance}"
                ),
                details={
                    "item_risk_level": risk_level,
                    "customer_risk_tolerance": tolerance,
                },
            )

        # Senior cap
        age = context.get("customer_age")
        if age is not None and int(age) >= self.senior_age_threshold:
            if int(risk_level) > self.senior_max_risk_level:
                return FilterResult(
                    passed=False,
                    filter_name="suitability",
                    reason=(
                        f"금소법 §17 senior cap: age={age} "
                        f"max_risk={self.senior_max_risk_level} "
                        f"< item_risk={risk_level}"
                    ),
                    details={"age": age, "risk_level": risk_level},
                )

        # Low-income cap
        income = context.get("customer_income")
        if income is not None and float(income) < self.low_income_threshold:
            if int(risk_level) > self.low_income_max_risk_level:
                return FilterResult(
                    passed=False,
                    filter_name="suitability",
                    reason=(
                        f"금소법 §17 low-income cap: income={income} "
                        f"max_risk={self.low_income_max_risk_level} "
                        f"< item_risk={risk_level}"
                    ),
                    details={"income": income, "risk_level": risk_level},
                )

        return FilterResult(
            passed=True, filter_name="suitability",
            details={
                "item_risk_level": risk_level,
                "customer_risk_tolerance": tolerance,
            },
        )


@FilterRegistry.register("owned_product")
class OwnedProductFilter(AbstractFilter):
    """Exclude items the customer already owns.

    Expects ``context["owned_products"]`` to be a set (or list) of item IDs.

    Config example::

        filters:
          owned_product:
            enabled: true
    """

    def evaluate(
        self,
        customer_id: str,
        item_id: str,
        context: Dict[str, Any],
    ) -> FilterResult:
        owned = context.get("owned_products", set())
        if isinstance(owned, list):
            owned = set(owned)

        if item_id in owned:
            return FilterResult(
                passed=False,
                filter_name="owned_product",
                reason=f"Customer already owns item '{item_id}'",
                details={"item_id": item_id},
            )

        return FilterResult(passed=True, filter_name="owned_product")


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

class ConstraintEngine:
    """Chain multiple filters from config and apply them sequentially.

    Config example::

        constraint_engine:
          filter_chain:
            - fatigue
            - eligibility
            - owned_product
          fail_fast: true          # stop on first rejection (default true)

    Usage::

        engine = ConstraintEngine(config)
        passed, results = engine.apply(customer_id, item_id, context)
    """

    def __init__(self, config: Dict[str, Any]) -> None:
        self.config = config
        engine_cfg = config.get("constraint_engine", {})
        self.fail_fast: bool = engine_cfg.get("fail_fast", True)

        chain_names: List[str] = engine_cfg.get("filter_chain", [
            "fatigue", "eligibility", "owned_product",
        ])

        self.filters: List[AbstractFilter] = []
        for name in chain_names:
            try:
                f = FilterRegistry.create(name, config)
                self.filters.append(f)
            except KeyError:
                logger.warning(
                    "ConstraintEngine: skipping unknown filter '%s'", name,
                )

        logger.info(
            "ConstraintEngine initialised with %d filters: %s",
            len(self.filters),
            [type(f).__name__ for f in self.filters],
        )

    def apply(
        self,
        customer_id: str,
        item_id: str,
        context: Dict[str, Any],
    ) -> tuple[bool, List[FilterResult]]:
        """Run all filters on a single candidate.

        Args:
            customer_id: Customer identifier.
            item_id: Item / product identifier.
            context: Shared context dict.

        Returns:
            Tuple of (all_passed, list_of_FilterResults).
        """
        results: List[FilterResult] = []
        all_passed = True

        for f in self.filters:
            result = f.evaluate(customer_id, item_id, context)
            results.append(result)

            if not result.passed:
                all_passed = False
                if self.fail_fast:
                    break

        return all_passed, results

    def apply_batch(
        self,
        candidates: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """Filter a batch of candidates, returning only those that pass.

        Each candidate dict must contain ``customer_id``, ``item_id``,
        and ``context``.  The returned dicts are the passing subset,
        each augmented with a ``"filter_results"`` key.
        """
        passed_candidates: List[Dict[str, Any]] = []
        rejected = 0

        for cand in candidates:
            ok, results = self.apply(
                cand["customer_id"],
                cand["item_id"],
                cand.get("context", {}),
            )
            if ok:
                cand["filter_results"] = results
                passed_candidates.append(cand)
            else:
                rejected += 1

        logger.info(
            "ConstraintEngine batch: %d passed, %d rejected out of %d",
            len(passed_candidates), rejected, len(candidates),
        )
        return passed_candidates
