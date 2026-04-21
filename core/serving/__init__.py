"""
Serving Layer
=============

Unified inference serving that works identically in AWS Lambda and ECS.

Modules:
    config          -- ServingConfig with auto Lambda/ECS switching
    feature_store   -- Abstracted feature lookup (Memory / DynamoDB)
    predict         -- RecommendationService (shared between Lambda & ECS)
    kill_switch     -- Per-request circuit breaker (DynamoDB-backed)
    ab_test         -- Deterministic A/B variant assignment + auto-promote
    model_monitor   -- Prediction logging + CloudWatch metrics + Champion-Challenger

Stage C stubs (production, not needed for ablation):
    cpe_engine              -- Real-time CPE scoring
    agentic_orchestrator    -- Serving-layer agentic reason generation
    vector_store            -- Production ANN vector store
"""

from .config import ServingConfig, LambdaConfig, ECSConfig, ABVariant
from .feature_store import (
    AbstractFeatureStore,
    MemoryFeatureStore,
    DynamoDBFeatureStore,
    FeatureStoreFactory,
)
from .predict import RecommendationService, PredictionResponse, OutputNormalizer
from .kill_switch import KillSwitch, KillSwitchState, FallbackStrategy
from .ab_test import ABTestManager, VariantAssignment
from .model_registry import ModelRegistry, ModelVersion
from .model_monitor import ModelMonitor, ChampionChallengerResult
from .cold_start import ColdStartHandler, UserSegment
from .cold_start_strategy import (
    TaskColdStartRegistry,
    AbstractTaskStrategy,
    PopularityStrategy,
    RetentionCatalogStrategy,
    UniformDistributionStrategy,
    GlobalMeanStrategy,
    ContextualChannelStrategy,
    ContextualTimingStrategy,
    ActionCatalogStrategy,
    BrandPopularityStrategy,
)
# Stage C stubs (production modules -- NotImplementedError on init)
from .cpe_engine import CPEEngine, CPEDecision
from .agentic_orchestrator import ServingAgenticOrchestrator, ReasonResponse
from .vector_store import ServingVectorStore, SearchResult
# Sprint 3 additions
from .review import (
    HumanReviewQueue,
    ReviewConfig,
    ReviewItem,
    ReviewState,
    build_human_review_queue,
)

__all__ = [
    # config
    "ServingConfig",
    "LambdaConfig",
    "ECSConfig",
    "ABVariant",
    # feature_store
    "AbstractFeatureStore",
    "MemoryFeatureStore",
    "DynamoDBFeatureStore",
    "FeatureStoreFactory",
    # predict
    "RecommendationService",
    "PredictionResponse",
    "OutputNormalizer",
    # kill_switch
    "KillSwitch",
    "KillSwitchState",
    "FallbackStrategy",
    # ab_test
    "ABTestManager",
    "VariantAssignment",
    # model_registry
    "ModelRegistry",
    "ModelVersion",
    # model_monitor
    "ModelMonitor",
    "ChampionChallengerResult",
    # cold_start
    "ColdStartHandler",
    "UserSegment",
    # cold_start_strategy
    "TaskColdStartRegistry",
    "AbstractTaskStrategy",
    "PopularityStrategy",
    "RetentionCatalogStrategy",
    "UniformDistributionStrategy",
    "GlobalMeanStrategy",
    "ContextualChannelStrategy",
    "ContextualTimingStrategy",
    "ActionCatalogStrategy",
    "BrandPopularityStrategy",
    # Stage C stubs
    "CPEEngine",
    "CPEDecision",
    "ServingAgenticOrchestrator",
    "ReasonResponse",
    "ServingVectorStore",
    "SearchResult",
    # Sprint 3 M1: Human Review Queue
    "HumanReviewQueue",
    "ReviewConfig",
    "ReviewItem",
    "ReviewState",
    "build_human_review_queue",
]
