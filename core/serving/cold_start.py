"""
Cold-Start Handler
==================

Manages the full cold-start recommendation path for users who have no
entry in the Feature Store.  Delegates per-task logic to
:class:`~core.serving.cold_start_strategy.TaskColdStartRegistry`.

Three user states are handled:

* **ANONYMOUS** (segment=0): No user_id context.
  → Generic popularity-only recommendations, no task predictions.

* **COLDSTART** (segment=1): Known user_id but no historical features.
  → Task-aware default predictions + task-specific candidate catalog
    + COLDSTART reasons from the template engine.

* **WARMSTART** (segment=2): Normal user with features.
  → Standard path (not handled here).

Task-awareness is provided by :class:`TaskColdStartRegistry`:
  - Each task has a strategy that controls default_prediction, catalog_key,
    and whether to attempt LGBM inference.
  - Unknown / future tasks fall back to GlobalMeanStrategy automatically.

Usage::

    handler = ColdStartHandler.from_config(config)

    # Check segment
    segment = handler.classify(user_id, context)

    # Get task-aware predictions (no LGBM needed)
    predictions = handler.task_predictions(task_names=["ctr", "churn"], context=ctx)

    # Get task-appropriate candidates for the pipeline
    candidates = handler.popularity_candidates(k=20, context=ctx, task_name="ctr")

    # Register a new future task without touching strategy code
    handler.registry.register_from_task_spec("credit_risk", "binary")
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from .cold_start_strategy import TaskColdStartRegistry

logger = logging.getLogger(__name__)

__all__ = ["ColdStartHandler", "UserSegment"]


class UserSegment:
    """User engagement segment. Keep string values to stay JSON-friendly
    across Lambda / ECS boundaries."""

    ANONYMOUS = "ANONYMOUS"    # segment=0: no user_id context
    COLDSTART = "COLDSTART"    # segment=1: user_id known, no history
    WARMSTART = "WARMSTART"    # segment=2: full feature history available


class ColdStartHandler:
    """Handles recommendations for users with no feature store entry.

    Args:
        catalog_map: Dict mapping catalog key → list of item dicts.
            Keys include ``"engagement"``, ``"retention"``, ``"actions"``,
            ``"brands"``.  Built from ``cold_start.catalogs`` config block.
        coldstart_features: Feature names the model expects.  Used by
            :meth:`default_features` to build a zero-filled vector.
        anonymous_threshold_days: Users with ``account_age_days`` below this
            are routed to ANONYMOUS instead of COLDSTART.  0 = disabled.
        registry: :class:`TaskColdStartRegistry` for per-task strategies.
            Auto-built from config when not provided.
        region: AWS region for S3 access.
    """

    def __init__(
        self,
        catalog_map: Optional[Dict[str, List[Dict[str, Any]]]] = None,
        coldstart_features: Optional[List[str]] = None,
        anonymous_threshold_days: int = 0,
        registry: Optional[TaskColdStartRegistry] = None,
        region: str = "ap-northeast-2",
    ) -> None:
        self._catalog_map: Dict[str, List[Dict[str, Any]]] = catalog_map or {}
        self._coldstart_features = coldstart_features or []
        self._anonymous_threshold_days = anonymous_threshold_days
        self._region = region
        self.registry: TaskColdStartRegistry = registry or TaskColdStartRegistry()

        total_items = sum(len(v) for v in self._catalog_map.values())
        logger.info(
            "ColdStartHandler: catalogs=%s (%d total items), "
            "anonymous_threshold_days=%d",
            list(self._catalog_map.keys()), total_items,
            anonymous_threshold_days,
        )

    # ------------------------------------------------------------------
    # Factory
    # ------------------------------------------------------------------

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "ColdStartHandler":
        """Build from ``cold_start:`` section of recommendation config.

        Expected config structure::

            cold_start:
              anonymous_threshold_days: 0
              coldstart_features: []          # optional; read from model at serve time
              catalogs:
                engagement:
                  - item_id: CARD_CASHBACK_PREMIUM
                    score: 0.95
                    name: "Cashback Premium Card"
                    benefit_type: cashback
                    category: credit_card
                retention:
                  - item_id: CARD_LOYALTY_BOOST
                    score: 0.90
                    ...
                actions:
                  - item_id: apply_card
                    score: 0.80
                    name: "신용카드 신청"
                brands:
                  - item_id: BRAND_STARBUCKS
                    score: 0.85
                    name: "Starbucks"
              # Optional: override default strategies per task
              task_strategies:
                ltv:
                  type: global_mean
                  global_mean: 200.0
              # Optional: override per-task global means
              global_means:
                ltv: 200.0
                spending_bucket: 600000
        """
        cs_cfg = config.get("cold_start", {})

        # Build catalog_map from sub-catalogs, with S3 override support
        catalog_map: Dict[str, List[Dict[str, Any]]] = {}
        for key, items in cs_cfg.get("catalogs", {}).items():
            catalog_map[key] = items or []

        # Flat popularity_catalog → put into "engagement" (backward compat)
        flat_catalog = cs_cfg.get("popularity_catalog", [])
        if flat_catalog and "engagement" not in catalog_map:
            catalog_map["engagement"] = flat_catalog

        # S3 catalog override
        s3_uri = cs_cfg.get("popularity_catalog_s3_uri", "")
        if s3_uri:
            s3_items = cls._load_catalog_from_s3_static(
                s3_uri, region=config.get("region", "ap-northeast-2"),
            )
            if s3_items:
                catalog_map["engagement"] = s3_items

        registry = TaskColdStartRegistry.from_config(config)

        return cls(
            catalog_map=catalog_map,
            coldstart_features=cs_cfg.get("coldstart_features", []),
            anonymous_threshold_days=cs_cfg.get("anonymous_threshold_days", 0),
            registry=registry,
            region=config.get("region", "ap-northeast-2"),
        )

    # ------------------------------------------------------------------
    # Segment classification
    # ------------------------------------------------------------------

    def classify(
        self,
        user_id: str,
        context: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Classify a user as ANONYMOUS or COLDSTART.

        Args:
            user_id: Empty string → ANONYMOUS.
            context: Checked fields: ``segment`` (0/1/2 or string),
                ``account_age_days``, ``is_anonymous``.

        Returns:
            ``"ANONYMOUS"`` or ``"COLDSTART"``.
        """
        ctx = context or {}

        if ctx.get("is_anonymous") or not user_id:
            return UserSegment.ANONYMOUS

        seg = ctx.get("segment")
        if seg is not None:
            seg_str = str(seg).upper()
            if seg_str in ("0", "ANONYMOUS"):
                return UserSegment.ANONYMOUS
            if seg_str in ("1", "COLDSTART"):
                return UserSegment.COLDSTART

        account_age = ctx.get("account_age_days")
        if (
            account_age is not None
            and self._anonymous_threshold_days > 0
            and int(account_age) < self._anonymous_threshold_days
        ):
            return UserSegment.ANONYMOUS

        return UserSegment.COLDSTART

    # ------------------------------------------------------------------
    # Task-aware predictions
    # ------------------------------------------------------------------

    def task_predictions(
        self,
        task_names: List[str],
        context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Return per-task default cold-start predictions.

        Each task's strategy determines the appropriate default:
        - binary (ctr/cvr) → low probability float
        - binary (churn) → very low probability
        - multiclass (life_stage) → uniform distribution
        - multiclass (channel) → context-inferred distribution
        - multiclass (timing) → time-of-day distribution
        - regression (ltv) → global mean
        - contrastive (brand) → None (skipped)

        Args:
            task_names: List of task names to get predictions for.
            context: Request context (used by channel/timing strategies).

        Returns:
            Dict of task_name → default prediction value.
            Tasks with None predictions are excluded.
        """
        result: Dict[str, Any] = {}
        for task_name in task_names:
            strategy = self.registry.get(task_name)
            pred = strategy.default_prediction(context)
            if pred is not None:
                result[task_name] = pred
        return result

    def should_use_lgbm(self, task_name: str) -> bool:
        """Check if LGBM inference should be attempted for a task."""
        return self.registry.get(task_name).use_lgbm

    # ------------------------------------------------------------------
    # Default feature vector (for LGBM-enabled tasks)
    # ------------------------------------------------------------------

    def default_features(
        self,
        feature_names: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """Zero-filled feature vector with cold-start signal overrides.

        Args:
            feature_names: Feature names.  Falls back to ``coldstart_features``
                from config if not provided.

        Returns:
            Dict of feature_name → 0.0, with cold-start signals overridden.
        """
        names = feature_names or self._coldstart_features
        features: Dict[str, Any] = {name: 0.0 for name in names}

        coldstart_overrides = {
            "is_coldstart": 1,
            "customer_segment": 1,
            "coldstart_confidence": 1.0,
            "coldstart_seq_confidence": 0.0,
            "coldstart_feature_confidence": 0.0,
            "coldstart_confidence_decay": 1.0,
        }
        for k, v in coldstart_overrides.items():
            if k in features or not names:
                features[k] = v

        return features

    # ------------------------------------------------------------------
    # Task-aware popularity candidates
    # ------------------------------------------------------------------

    def popularity_candidates(
        self,
        k: int = 20,
        context: Optional[Dict[str, Any]] = None,
        task_name: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Return cold-start candidates appropriate for the given task.

        When ``task_name`` is provided, the strategy for that task determines
        which catalog to use.  Unknown tasks or tasks with no catalog key
        fall back to the ``engagement`` catalog.

        Args:
            k: Maximum candidates to return.
            context: Request context.
            task_name: Task driving the recommendation (e.g. ``"ctr"``).
                ``None`` → use ``engagement`` catalog.

        Returns:
            List of candidate dicts compatible with the pipeline.
        """
        if task_name is not None:
            strategy = self.registry.get(task_name)
            return strategy.candidates(self._catalog_map, k=k, context=context)

        # Fallback: use engagement catalog directly
        items = self._catalog_map.get("engagement", [])
        sorted_items = sorted(items, key=lambda x: x.get("score", 0.0), reverse=True)
        if not sorted_items:
            logger.warning(
                "ColdStartHandler: no engagement catalog items. "
                "Check cold_start.catalogs.engagement in recommendation.yaml",
            )
        return [
            {
                "item_id": it.get("item_id", ""),
                "predictions": {},
                "item_info": {
                    "name": it.get("name", it.get("item_id", "")),
                    "benefit_type": it.get("benefit_type", ""),
                    "category": it.get("category", ""),
                },
                "ig_top_features": [],
                "popularity_score": float(it.get("score", 0.0)),
                "is_coldstart_candidate": True,
            }
            for it in sorted_items[:k]
        ]

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _load_catalog_from_s3_static(
        s3_uri: str,
        region: str = "ap-northeast-2",
    ) -> List[Dict[str, Any]]:
        """Download a flat popularity catalog JSON from S3."""
        try:
            import boto3, json

            s3 = boto3.client("s3", region_name=region)
            path = s3_uri.replace("s3://", "")
            parts = path.split("/", 1)
            bucket, key = parts[0], (parts[1] if len(parts) > 1 else "")
            obj = s3.get_object(Bucket=bucket, Key=key)
            data = json.loads(obj["Body"].read())
            catalog = data if isinstance(data, list) else data.get("items", [])
            logger.info("Loaded %d items from %s", len(catalog), s3_uri)
            return catalog
        except Exception as e:
            logger.warning("Failed to load catalog from %s: %s", s3_uri, e)
            return []
