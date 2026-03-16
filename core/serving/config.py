"""
Serving Configuration
=====================

Dataclass-based configuration for the serving layer.  Supports three compute
modes (``auto``, ``lambda``, ``ecs``) and two feature store backends
(``memory``, ``dynamodb``).  ``auto`` mode switches based on measured or
projected traffic volume.

Example YAML::

    serving:
      mode: auto
      auto_threshold: 150_000_000
      feature_store: auto
      auto_feature_threshold: 5_000_000

      lambda:
        memory_mb: 1024
        timeout_seconds: 30
        reserved_concurrency: 100

      ecs:
        cpu: 1024
        memory: 2048
        min_tasks: 2
        max_tasks: 10
        target_cpu_pct: 70

      model:
        s3_uri: s3://my-bucket/models/lgbm/latest/
        tasks_meta:
          - name: ctr
            type: binary
          - name: cvr
            type: binary
          - name: ltv
            type: regression

      ab_test:
        enabled: false
        variants:
          - name: control
            model_path: s3://my-bucket/models/v1/
            weight: 0.8
          - name: challenger
            model_path: s3://my-bucket/models/v2/
            weight: 0.2

      kill_switch:
        table_name: ple-kill-switch
        fallback_strategy: rule_based

      pipeline:
        enabled: false
        scorer_name: weighted_sum
        enable_reasons: false
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

__all__ = [
    "ServingMode",
    "FeatureStoreMode",
    "LambdaConfig",
    "ECSConfig",
    "ABVariant",
    "ServingConfig",
]


class ServingMode(str, Enum):
    """Compute backend selection."""
    AUTO = "auto"
    LAMBDA = "lambda"
    ECS = "ecs"


class FeatureStoreMode(str, Enum):
    """Feature store backend selection."""
    AUTO = "auto"
    MEMORY = "memory"
    DYNAMODB = "dynamodb"


@dataclass
class LambdaConfig:
    """AWS Lambda function configuration."""

    memory_mb: int = 1024
    timeout_seconds: int = 30
    reserved_concurrency: int = 100

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> LambdaConfig:
        return cls(
            memory_mb=d.get("memory_mb", 1024),
            timeout_seconds=d.get("timeout_seconds", 30),
            reserved_concurrency=d.get("reserved_concurrency", 100),
        )


@dataclass
class ECSConfig:
    """AWS ECS Fargate task configuration."""

    cpu: int = 1024
    memory: int = 2048
    min_tasks: int = 2
    max_tasks: int = 10
    target_cpu_pct: int = 70

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> ECSConfig:
        return cls(
            cpu=d.get("cpu", 1024),
            memory=d.get("memory", 2048),
            min_tasks=d.get("min_tasks", 2),
            max_tasks=d.get("max_tasks", 10),
            target_cpu_pct=d.get("target_cpu_pct", 70),
        )


@dataclass
class ABVariant:
    """A single A/B test variant definition.

    Attributes:
        name: Human-readable variant name (e.g. ``"control"``, ``"challenger"``).
        model_path: S3 URI to the model artefact for this variant.
        weight: Traffic fraction in ``[0, 1]``.  All variant weights should
                sum to 1.0.
    """

    name: str
    model_path: str
    weight: float = 0.5

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> ABVariant:
        return cls(
            name=d["name"],
            model_path=d["model_path"],
            weight=d.get("weight", 0.5),
        )


@dataclass
class ServingConfig:
    """Top-level serving configuration.

    Use :meth:`from_dict` to hydrate from a parsed YAML config dictionary.

    Attributes:
        mode: Compute backend (``auto`` | ``lambda`` | ``ecs``).
        auto_threshold: Monthly request count above which ``auto`` mode
            switches from Lambda to ECS.
        feature_store: Feature store backend (``auto`` | ``memory`` | ``dynamodb``).
        auto_feature_threshold: User count above which ``auto`` mode switches
            from in-memory to DynamoDB.
        lambda_config: Lambda-specific settings.
        ecs_config: ECS-specific settings.
        model_s3_uri: S3 URI for the model artefact directory.
        tasks_meta: List of task metadata dicts (``name``, ``type``).
        ab_test_enabled: Whether A/B testing is active.
        ab_variants: List of :class:`ABVariant` definitions.
        kill_switch_table: DynamoDB table name for the kill switch.
        fallback_strategy: What to do when the kill switch fires.
        pipeline_config: Optional config dict passed to
            :class:`~core.recommendation.pipeline.RecommendationPipeline`.
        feature_store_config: Extra config for the feature store backend
            (e.g. ``s3_uri``, ``dynamodb_table``).
    """

    # -- Compute mode --
    mode: ServingMode = ServingMode.AUTO
    auto_threshold: int = 150_000_000

    # -- Feature store --
    feature_store: FeatureStoreMode = FeatureStoreMode.AUTO
    auto_feature_threshold: int = 5_000_000
    feature_store_config: Dict[str, Any] = field(default_factory=dict)

    # -- Backend configs --
    lambda_config: LambdaConfig = field(default_factory=LambdaConfig)
    ecs_config: ECSConfig = field(default_factory=ECSConfig)

    # -- Model --
    model_s3_uri: str = ""
    tasks_meta: List[Dict[str, str]] = field(default_factory=list)

    # -- A/B test --
    ab_test_enabled: bool = False
    ab_variants: List[ABVariant] = field(default_factory=list)

    # -- Kill switch --
    kill_switch_table: str = "ple-kill-switch"
    fallback_strategy: str = "rule_based"

    # -- Recommendation pipeline --
    pipeline_config: Dict[str, Any] = field(default_factory=dict)

    # ------------------------------------------------------------------
    # Factory
    # ------------------------------------------------------------------

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> ServingConfig:
        """Hydrate from a parsed YAML ``serving:`` block."""
        serving = d.get("serving", d)

        variants: List[ABVariant] = []
        ab_cfg = serving.get("ab_test", {})
        for v in ab_cfg.get("variants", []):
            variants.append(ABVariant.from_dict(v))

        ks_cfg = serving.get("kill_switch", {})
        model_cfg = serving.get("model", {})

        return cls(
            mode=ServingMode(serving.get("mode", "auto")),
            auto_threshold=serving.get("auto_threshold", 150_000_000),
            feature_store=FeatureStoreMode(
                serving.get("feature_store", "auto"),
            ),
            auto_feature_threshold=serving.get("auto_feature_threshold", 5_000_000),
            feature_store_config=serving.get("feature_store_config", {}),
            lambda_config=LambdaConfig.from_dict(serving.get("lambda", {})),
            ecs_config=ECSConfig.from_dict(serving.get("ecs", {})),
            model_s3_uri=model_cfg.get("s3_uri", ""),
            tasks_meta=model_cfg.get("tasks_meta", []),
            ab_test_enabled=ab_cfg.get("enabled", False),
            ab_variants=variants,
            kill_switch_table=ks_cfg.get("table_name", "ple-kill-switch"),
            fallback_strategy=ks_cfg.get("fallback_strategy", "rule_based"),
            pipeline_config=serving.get("pipeline", {}),
        )

    # ------------------------------------------------------------------
    # Resolved mode helpers
    # ------------------------------------------------------------------

    def resolve_compute_mode(
        self, monthly_requests: Optional[int] = None,
    ) -> ServingMode:
        """Return the effective compute mode.

        When ``mode`` is ``auto``, the decision is based on
        *monthly_requests* compared to :attr:`auto_threshold`.
        """
        if self.mode != ServingMode.AUTO:
            return self.mode
        if monthly_requests is None:
            logger.info(
                "resolve_compute_mode: no traffic data, defaulting to Lambda",
            )
            return ServingMode.LAMBDA
        if monthly_requests >= self.auto_threshold:
            logger.info(
                "resolve_compute_mode: %s monthly requests >= threshold %s, "
                "selecting ECS",
                monthly_requests, self.auto_threshold,
            )
            return ServingMode.ECS
        logger.info(
            "resolve_compute_mode: %s monthly requests < threshold %s, "
            "selecting Lambda",
            monthly_requests, self.auto_threshold,
        )
        return ServingMode.LAMBDA

    def resolve_feature_store_mode(
        self, user_count: Optional[int] = None,
    ) -> FeatureStoreMode:
        """Return the effective feature store backend.

        When ``feature_store`` is ``auto``, the decision is based on
        *user_count* compared to :attr:`auto_feature_threshold`.
        """
        if self.feature_store != FeatureStoreMode.AUTO:
            return self.feature_store
        if user_count is None:
            logger.info(
                "resolve_feature_store_mode: no user count, "
                "defaulting to memory",
            )
            return FeatureStoreMode.MEMORY
        if user_count >= self.auto_feature_threshold:
            logger.info(
                "resolve_feature_store_mode: %s users >= threshold %s, "
                "selecting DynamoDB",
                user_count, self.auto_feature_threshold,
            )
            return FeatureStoreMode.DYNAMODB
        logger.info(
            "resolve_feature_store_mode: %s users < threshold %s, "
            "selecting memory",
            user_count, self.auto_feature_threshold,
        )
        return FeatureStoreMode.MEMORY
