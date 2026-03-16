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
"""

from .config import ServingConfig, LambdaConfig, ECSConfig, ABVariant
from .feature_store import (
    AbstractFeatureStore,
    MemoryFeatureStore,
    DynamoDBFeatureStore,
    FeatureStoreFactory,
)
from .predict import RecommendationService, PredictionResponse
from .kill_switch import KillSwitch, KillSwitchState
from .ab_test import ABTestManager

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
    # kill_switch
    "KillSwitch",
    "KillSwitchState",
    # ab_test
    "ABTestManager",
]
