"""
Per-Task Cold-Start Strategies
==============================

Each of the 16 financial recommendation tasks has a different semantic
relationship to cold-start users.  This module defines a strategy per task
that controls:

1. **default_prediction** — what value to return as the model prediction when
   the user has no history (e.g. low churn probability, global mean LTV).
2. **catalog_key** — which popularity sub-catalog to draw candidates from.
3. **use_lgbm** — whether to attempt LGBM inference with the default feature
   vector (makes sense for engagement tasks, not for regression).
4. **candidates()** — how to build the cold-start candidate list.

Strategy taxonomy (4 groups × 16 tasks):

  Engagement (ctr, cvr)
    → PopularityStrategy: top items by global engagement score.

  Lifecycle (churn, retention, life_stage, ltv)
    → churn/retention: RetentionCatalogStrategy — loyalty/benefit items.
    → life_stage:      UniformDistributionStrategy — 6 classes, flat prior.
    → ltv:             GlobalMeanStrategy — configurable global mean value.

  Value (balance_util, engagement, channel, timing)
    → balance_util/engagement: GlobalMeanStrategy.
    → channel:         ContextualChannelStrategy — infer from request context.
    → timing:          ContextualTimingStrategy  — infer from current hour.

  Consumption (nba, spending_category, consumption_cycle,
               spending_bucket, merchant_affinity, brand_prediction)
    → nba:             ActionCatalogStrategy — popular action distribution.
    → spending_category/consumption_cycle: UniformDistributionStrategy.
    → spending_bucket/merchant_affinity: GlobalMeanStrategy.
    → brand_prediction: BrandPopularityStrategy — top brands by popularity.

Usage::

    registry = TaskColdStartRegistry.from_config(config)
    strategy = registry.get("churn")

    # Get default prediction value
    pred = strategy.default_prediction(context={})

    # Get popularity candidates for the pipeline
    candidates = strategy.candidates(catalog_map, k=20, context={})
"""

from __future__ import annotations

import logging
import math
from abc import ABC, abstractmethod
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

__all__ = [
    "AbstractTaskStrategy",
    "PopularityStrategy",
    "RetentionCatalogStrategy",
    "UniformDistributionStrategy",
    "GlobalMeanStrategy",
    "ContextualChannelStrategy",
    "ContextualTimingStrategy",
    "ActionCatalogStrategy",
    "BrandPopularityStrategy",
    "TaskColdStartRegistry",
]


# ---------------------------------------------------------------------------
# Abstract base
# ---------------------------------------------------------------------------

class AbstractTaskStrategy(ABC):
    """Per-task cold-start strategy.

    Attributes:
        task_name: Name of the task this strategy handles.
        task_type: One of ``binary``, ``multiclass``, ``regression``,
            ``contrastive``.
        catalog_key: Key into the ``catalogs`` dict in the config.  ``None``
            means this task does not use a product catalog.
        use_lgbm: Whether to run LGBM inference with the default feature
            vector.  Disabled for regression/contrastive tasks where the
            default-feature output is uninformative.
    """

    task_name: str = ""
    task_type: str = ""
    catalog_key: Optional[str] = None
    use_lgbm: bool = True

    @abstractmethod
    def default_prediction(
        self, context: Optional[Dict[str, Any]] = None,
    ) -> Any:
        """Return a sensible cold-start prediction value.

        For binary tasks: a low-confidence probability.
        For multiclass: a probability vector (uniform or context-based).
        For regression: the global mean or a safe default.
        For contrastive: None (skip).
        """

    def candidates(
        self,
        catalog_map: Dict[str, List[Dict[str, Any]]],
        k: int = 20,
        context: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """Return cold-start candidate items.

        Default implementation looks up ``self.catalog_key`` in
        ``catalog_map``, sorts by score, and returns the top-k items
        formatted for the recommendation pipeline.

        Override for strategies that build candidates differently.
        """
        if self.catalog_key is None:
            return []

        raw = catalog_map.get(self.catalog_key, [])
        sorted_items = sorted(raw, key=lambda x: x.get("score", 0.0), reverse=True)

        return [
            {
                "item_id": item.get("item_id", ""),
                "predictions": {},
                "item_info": {
                    "name": item.get("name", item.get("item_id", "")),
                    "benefit_type": item.get("benefit_type", ""),
                    "category": item.get("category", ""),
                },
                "ig_top_features": [],
                "popularity_score": float(item.get("score", 0.0)),
                "is_coldstart_candidate": True,
            }
            for item in sorted_items[:k]
        ]


# ---------------------------------------------------------------------------
# Concrete strategies
# ---------------------------------------------------------------------------

class PopularityStrategy(AbstractTaskStrategy):
    """Engagement tasks (ctr, cvr): popularity catalog + LGBM with default vec.

    Popularity ranking is a valid proxy for engagement signal for new users:
    popular items have the highest prior probability of being clicked/converted.
    """

    use_lgbm = True

    def __init__(self, task_name: str, catalog_key: str = "engagement"):
        self.task_name = task_name
        self.task_type = "binary"
        self.catalog_key = catalog_key

    def default_prediction(self, context=None) -> float:
        # Prior probability for new users: low but non-zero
        return 0.05


class RetentionCatalogStrategy(AbstractTaskStrategy):
    """Lifecycle binary tasks (churn, retention).

    For churn: new users have no churn signal → return a low churn probability.
    For retention: return a moderate retention probability.
    Candidates come from a retention-focused catalog (loyalty/rewards products).
    """

    use_lgbm = False  # Default vector produces noisy churn/retention output

    def __init__(
        self,
        task_name: str,
        catalog_key: str = "retention",
        default_churn: float = 0.08,
        default_retention: float = 0.60,
    ):
        self.task_name = task_name
        self.task_type = "binary"
        self.catalog_key = catalog_key
        self._default_churn = default_churn
        self._default_retention = default_retention

    def default_prediction(self, context=None) -> float:
        if self.task_name == "churn":
            return self._default_churn
        return self._default_retention


class UniformDistributionStrategy(AbstractTaskStrategy):
    """Multiclass tasks where cold-start provides no useful signal.

    Returns a uniform probability vector of length ``num_classes``.
    Tasks: life_stage (6), spending_category (12), consumption_cycle (7).
    """

    use_lgbm = False
    catalog_key = None  # No product catalog relevant

    def __init__(self, task_name: str, num_classes: int):
        self.task_name = task_name
        self.task_type = "multiclass"
        self._num_classes = num_classes

    def default_prediction(self, context=None) -> List[float]:
        p = round(1.0 / self._num_classes, 6)
        return [p] * self._num_classes

    def candidates(self, catalog_map, k=20, context=None):
        return []  # These tasks don't drive product recommendations directly


class GlobalMeanStrategy(AbstractTaskStrategy):
    """Regression tasks: return a configurable global mean.

    LGBM inference with default features produces high-variance output for
    regression tasks, so we skip it and return a pre-computed global average.

    Tasks: ltv, balance_util, engagement, spending_bucket, merchant_affinity.
    """

    use_lgbm = False
    catalog_key = None

    def __init__(
        self,
        task_name: str,
        global_mean: float = 0.0,
        catalog_key: Optional[str] = None,
    ):
        self.task_name = task_name
        self.task_type = "regression"
        self._global_mean = global_mean
        self.catalog_key = catalog_key

    def default_prediction(self, context=None) -> float:
        return self._global_mean

    def candidates(self, catalog_map, k=20, context=None):
        if self.catalog_key is None:
            return []
        return super().candidates(catalog_map, k, context)


class ContextualChannelStrategy(AbstractTaskStrategy):
    """Channel task (multiclass, 3 classes): infer from request context.

    Channel classes: 0=app, 1=web, 2=branch.
    If the request context contains a ``channel`` key, we bias the prediction
    towards that channel; otherwise return uniform.
    """

    use_lgbm = False
    catalog_key = None

    CHANNEL_MAP = {"app": 0, "mobile": 0, "web": 1, "branch": 2, "offline": 2}
    NUM_CLASSES = 3

    def __init__(self, task_name: str = "channel"):
        self.task_name = task_name
        self.task_type = "multiclass"

    def default_prediction(self, context=None) -> List[float]:
        ctx = context or {}
        channel = str(ctx.get("channel", "")).lower()
        idx = self.CHANNEL_MAP.get(channel)

        if idx is not None:
            # Strong prior for known channel
            probs = [0.05] * self.NUM_CLASSES
            probs[idx] = 0.90
            return probs

        # Uniform prior when channel is unknown
        p = round(1.0 / self.NUM_CLASSES, 6)
        return [p] * self.NUM_CLASSES

    def candidates(self, catalog_map, k=20, context=None):
        return []


class ContextualTimingStrategy(AbstractTaskStrategy):
    """Timing task (multiclass, 28 classes): infer from current hour.

    28 timing slots = 4 slots per day × 7 days-of-week.
    Slot index: day_of_week * 4 + hour_bucket (0=0-5, 1=6-11, 2=12-17, 3=18-23).
    Cold-start prediction peaks at the current time slot and decays nearby.
    """

    use_lgbm = False
    catalog_key = None
    NUM_CLASSES = 28

    def __init__(self, task_name: str = "timing"):
        self.task_name = task_name
        self.task_type = "multiclass"

    def default_prediction(self, context=None) -> List[float]:
        ctx = context or {}

        # Use provided timestamp or current UTC time
        ts_str = ctx.get("timestamp")
        if ts_str:
            try:
                from datetime import datetime, timezone
                dt = datetime.fromisoformat(ts_str)
            except Exception:
                dt = datetime.now(timezone.utc)
        else:
            dt = datetime.now(timezone.utc)

        day_of_week = dt.weekday()  # 0=Mon … 6=Sun
        hour_bucket = min(dt.hour // 6, 3)  # 0-3
        peak_slot = day_of_week * 4 + hour_bucket

        # Soft distribution: peak slot gets highest probability, decay by distance
        probs = []
        for i in range(self.NUM_CLASSES):
            # Circular distance
            dist = min(abs(i - peak_slot), self.NUM_CLASSES - abs(i - peak_slot))
            # Gaussian-like decay; sigma=3 slots
            probs.append(math.exp(-0.5 * (dist / 3.0) ** 2))

        total = sum(probs)
        return [round(p / total, 6) for p in probs]

    def candidates(self, catalog_map, k=20, context=None):
        return []


class ActionCatalogStrategy(AbstractTaskStrategy):
    """NBA task (multiclass, 12 classes): popular action distribution.

    For cold-start users, we return a distribution over popular actions
    (apply_card, open_deposit, etc.) based on a global popularity catalog,
    plus a candidate list of the corresponding financial products.
    """

    use_lgbm = False
    NUM_CLASSES = 12

    # Default global action distribution (index → action name)
    DEFAULT_ACTION_PROBS = {
        0: 0.22,  # apply_card
        1: 0.18,  # open_deposit
        2: 0.14,  # apply_loan
        3: 0.12,  # upgrade_plan
        4: 0.10,  # subscribe_insurance
        5: 0.08,  # invest_fund
        6: 0.06,  # enroll_loyalty
        7: 0.04,  # activate_benefit
        8: 0.02,  # redeem_points
        9: 0.02,  # referral
        10: 0.01, # close_account
        11: 0.01, # other
    }

    def __init__(
        self,
        task_name: str = "nba",
        catalog_key: str = "actions",
    ):
        self.task_name = task_name
        self.task_type = "multiclass"
        self.catalog_key = catalog_key

    def default_prediction(self, context=None) -> List[float]:
        probs = [self.DEFAULT_ACTION_PROBS.get(i, 0.0) for i in range(self.NUM_CLASSES)]
        total = sum(probs)
        return [round(p / total, 6) for p in probs]


class BrandPopularityStrategy(AbstractTaskStrategy):
    """Brand prediction task (contrastive, 128 classes): top brands by popularity.

    Contrastive inference with a default vector is meaningless, so we skip
    LGBM and return None as the prediction value.  Candidates come from a
    brand popularity catalog.
    """

    use_lgbm = False

    def __init__(
        self,
        task_name: str = "brand_prediction",
        catalog_key: str = "brands",
    ):
        self.task_name = task_name
        self.task_type = "contrastive"
        self.catalog_key = catalog_key

    def default_prediction(self, context=None):
        return None  # Contrastive output is an embedding — not meaningful as default


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

class TaskColdStartRegistry:
    """Maps task names to their cold-start strategies.

    Built from the ``cold_start.task_strategies`` section of the config.
    Falls back to type-based defaults for tasks not explicitly configured.

    Default mapping (auto-applied if not overridden in config):

        ctr, cvr             → PopularityStrategy
        churn, retention     → RetentionCatalogStrategy
        life_stage           → UniformDistributionStrategy(6)
        ltv                  → GlobalMeanStrategy(default=150.0)
        balance_util         → GlobalMeanStrategy(default=0.5)
        engagement           → GlobalMeanStrategy(default=0.5)
        channel              → ContextualChannelStrategy
        timing               → ContextualTimingStrategy
        nba                  → ActionCatalogStrategy
        spending_category    → UniformDistributionStrategy(12)
        consumption_cycle    → UniformDistributionStrategy(7)
        spending_bucket      → GlobalMeanStrategy(default=500_000)
        merchant_affinity    → GlobalMeanStrategy(default=0.3)
        brand_prediction     → BrandPopularityStrategy

    Usage::

        registry = TaskColdStartRegistry.from_config(config)
        strategy = registry.get("churn")
        default_pred = strategy.default_prediction(context)
        candidates = strategy.candidates(catalog_map, k=20)
    """

    # Hardcoded defaults for all 16 tasks
    _BUILT_IN_DEFAULTS: Dict[str, AbstractTaskStrategy] = {
        "ctr":               PopularityStrategy("ctr", catalog_key="engagement"),
        "cvr":               PopularityStrategy("cvr", catalog_key="engagement"),
        "churn":             RetentionCatalogStrategy("churn"),
        "retention":         RetentionCatalogStrategy("retention"),
        "life_stage":        UniformDistributionStrategy("life_stage", 6),
        "ltv":               GlobalMeanStrategy("ltv", global_mean=150.0),
        "balance_util":      GlobalMeanStrategy("balance_util", global_mean=0.5),
        "engagement":        GlobalMeanStrategy("engagement", global_mean=0.5),
        "channel":           ContextualChannelStrategy("channel"),
        "timing":            ContextualTimingStrategy("timing"),
        "nba":               ActionCatalogStrategy("nba"),
        "spending_category": UniformDistributionStrategy("spending_category", 12),
        "consumption_cycle": UniformDistributionStrategy("consumption_cycle", 7),
        "spending_bucket":   GlobalMeanStrategy("spending_bucket", global_mean=500_000.0),
        "merchant_affinity": GlobalMeanStrategy("merchant_affinity", global_mean=0.3),
        "brand_prediction":  BrandPopularityStrategy("brand_prediction"),
    }

    def __init__(
        self,
        strategies: Optional[Dict[str, AbstractTaskStrategy]] = None,
    ) -> None:
        self._strategies: Dict[str, AbstractTaskStrategy] = dict(self._BUILT_IN_DEFAULTS)
        if strategies:
            self._strategies.update(strategies)

    # ------------------------------------------------------------------
    # Factory
    # ------------------------------------------------------------------

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "TaskColdStartRegistry":
        """Build registry from the ``cold_start.task_strategies`` config block.

        Config example::

            cold_start:
              task_strategies:
                ltv:
                  type: global_mean
                  global_mean: 200.0
                churn:
                  type: retention_catalog
                  catalog_key: retention
                  default_churn: 0.05
        """
        cs_cfg = config.get("cold_start", {})
        task_overrides = cs_cfg.get("task_strategies", {})

        custom: Dict[str, AbstractTaskStrategy] = {}

        for task_name, spec in task_overrides.items():
            strategy = cls._build_strategy(task_name, spec)
            if strategy is not None:
                custom[task_name] = strategy

        registry = cls(strategies=custom)

        # Apply global_mean overrides from cold_start.global_means
        global_means = cs_cfg.get("global_means", {})
        for task_name, mean_val in global_means.items():
            existing = registry._strategies.get(task_name)
            if isinstance(existing, GlobalMeanStrategy):
                existing._global_mean = float(mean_val)

        logger.info(
            "TaskColdStartRegistry built: %d tasks (%d from config overrides)",
            len(registry._strategies),
            len(custom),
        )
        return registry

    @staticmethod
    def _build_strategy(
        task_name: str, spec: Dict[str, Any],
    ) -> Optional[AbstractTaskStrategy]:
        """Build a strategy instance from a config spec dict."""
        strategy_type = spec.get("type", "")

        if strategy_type == "popularity":
            return PopularityStrategy(
                task_name,
                catalog_key=spec.get("catalog_key", "engagement"),
            )
        elif strategy_type == "retention_catalog":
            return RetentionCatalogStrategy(
                task_name,
                catalog_key=spec.get("catalog_key", "retention"),
                default_churn=spec.get("default_churn", 0.08),
                default_retention=spec.get("default_retention", 0.60),
            )
        elif strategy_type == "uniform":
            return UniformDistributionStrategy(
                task_name,
                num_classes=spec.get("num_classes", 2),
            )
        elif strategy_type == "global_mean":
            return GlobalMeanStrategy(
                task_name,
                global_mean=spec.get("global_mean", 0.0),
                catalog_key=spec.get("catalog_key"),
            )
        elif strategy_type == "contextual_channel":
            return ContextualChannelStrategy(task_name)
        elif strategy_type == "contextual_timing":
            return ContextualTimingStrategy(task_name)
        elif strategy_type == "action_catalog":
            return ActionCatalogStrategy(
                task_name,
                catalog_key=spec.get("catalog_key", "actions"),
            )
        elif strategy_type == "brand_popularity":
            return BrandPopularityStrategy(
                task_name,
                catalog_key=spec.get("catalog_key", "brands"),
            )
        else:
            logger.warning(
                "TaskColdStartRegistry: unknown strategy type '%s' for task '%s'",
                strategy_type, task_name,
            )
            return None

    # ------------------------------------------------------------------
    # Lookup
    # ------------------------------------------------------------------

    def get(self, task_name: str) -> AbstractTaskStrategy:
        """Return the strategy for *task_name*.

        Returns a :class:`GlobalMeanStrategy` with 0.0 default if the task
        is unknown (safe fallback — new tasks added later won't break serving).
        """
        strategy = self._strategies.get(task_name)
        if strategy is None:
            logger.warning(
                "TaskColdStartRegistry: unknown task '%s', "
                "using GlobalMeanStrategy(0.0) fallback. "
                "Add it to cold_start.task_strategies in recommendation.yaml "
                "or call registry.register() at startup.",
                task_name,
            )
            return GlobalMeanStrategy(task_name, global_mean=0.0)
        return strategy

    def register(
        self,
        task_name: str,
        strategy: AbstractTaskStrategy,
        overwrite: bool = False,
    ) -> None:
        """Programmatically register a strategy for a task.

        Use this at application startup for tasks that are added after the
        initial 16 (e.g. risk, compliance, ESG scoring tasks):

        .. code-block:: python

            registry.register(
                "credit_risk",
                GlobalMeanStrategy("credit_risk", global_mean=0.15),
            )

        Args:
            task_name: Task identifier.
            strategy: Strategy instance to register.
            overwrite: If False (default) and the task is already registered,
                raises a ``ValueError`` to prevent accidental overwrites.
        """
        if task_name in self._strategies and not overwrite:
            raise ValueError(
                f"Task '{task_name}' is already registered. "
                "Pass overwrite=True to replace it."
            )
        self._strategies[task_name] = strategy
        logger.info(
            "TaskColdStartRegistry: registered '%s' → %s",
            task_name, type(strategy).__name__,
        )

    def register_from_task_spec(
        self,
        task_name: str,
        task_type: str,
        num_classes: int = 1,
        global_mean: float = 0.0,
        catalog_key: Optional[str] = None,
    ) -> None:
        """Register a strategy by task type, without writing custom code.

        Convenience method for new tasks:  just supply the ``task_type``
        and the registry will pick the most appropriate base strategy.

        Strategy selection by type:
            ``binary``      → :class:`PopularityStrategy` (with ``engagement`` catalog)
            ``multiclass``  → :class:`UniformDistributionStrategy`
            ``regression``  → :class:`GlobalMeanStrategy`
            ``contrastive`` → :class:`BrandPopularityStrategy`
            ``ranking``     → :class:`PopularityStrategy`

        Args:
            task_name: New task identifier.
            task_type: One of ``binary``, ``multiclass``, ``regression``,
                ``contrastive``, ``ranking``.
            num_classes: Required for ``multiclass`` tasks.
            global_mean: Default prediction value for ``regression`` tasks.
            catalog_key: Override the default catalog key.
        """
        type_map = {
            "binary":      lambda: PopularityStrategy(
                task_name, catalog_key=catalog_key or "engagement"
            ),
            "multiclass":  lambda: UniformDistributionStrategy(
                task_name, num_classes=num_classes
            ),
            "regression":  lambda: GlobalMeanStrategy(
                task_name, global_mean=global_mean, catalog_key=catalog_key
            ),
            "contrastive": lambda: BrandPopularityStrategy(
                task_name, catalog_key=catalog_key or "brands"
            ),
            "ranking":     lambda: PopularityStrategy(
                task_name, catalog_key=catalog_key or "engagement"
            ),
        }
        factory = type_map.get(task_type)
        if factory is None:
            logger.warning(
                "register_from_task_spec: unknown type '%s' for task '%s', "
                "falling back to GlobalMeanStrategy",
                task_type, task_name,
            )
            strategy = GlobalMeanStrategy(task_name, global_mean=global_mean)
        else:
            strategy = factory()

        self.register(task_name, strategy, overwrite=True)

    def all_tasks(self) -> List[str]:
        """Return all registered task names."""
        return list(self._strategies.keys())

    def default_predictions(
        self,
        task_names: Optional[List[str]] = None,
        context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Return default predictions for all (or a subset of) tasks.

        Convenience method for building the ``predictions`` field of a
        cold-start response in one call.

        Args:
            task_names: Subset of task names.  ``None`` → all registered.
            context: Request context passed to each strategy.

        Returns:
            Dict mapping task_name → default prediction value.
        """
        names = task_names or self.all_tasks()
        return {
            name: self.get(name).default_prediction(context)
            for name in names
        }
