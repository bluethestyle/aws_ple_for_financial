"""
ECS FastAPI Application
=======================

Serves the same :class:`~core.serving.predict.RecommendationService` as
the Lambda handler, but wrapped in a FastAPI app with proper HTTP
endpoints, health checks, and OpenAPI documentation.

Endpoints::

    POST /v1/recommend        -- single-user recommendation
    POST /v1/recommend/batch  -- batch recommendation
    GET  /v1/health           -- health check (also POST for ALB probes)
    GET  /                    -- root / readiness probe

Startup::

    uvicorn containers.inference.app:app --host 0.0.0.0 --port 8080
"""

from __future__ import annotations

import json
import logging
import os
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)
logger.setLevel(os.environ.get("LOG_LEVEL", "INFO"))

# ---------------------------------------------------------------------------
# Pydantic request / response models
# ---------------------------------------------------------------------------


class RecommendRequest(BaseModel):
    """Single-user recommendation request."""

    user_id: str = Field(..., description="Unique user identifier")
    context: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Optional context (channel, segment, task, cluster, etc.)",
    )


class BatchRecommendRequest(BaseModel):
    """Batch recommendation request."""

    user_ids: List[str] = Field(..., description="List of user identifiers")
    context: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Shared context applied to all users",
    )


class PredictionResponseModel(BaseModel):
    """Serialised prediction response."""

    user_id: str
    predictions: Dict[str, Any] = Field(default_factory=dict)
    recommendations: List[Dict[str, Any]] = Field(default_factory=list)
    variant: str = ""
    kill_switch_active: bool = False
    fallback_used: bool = False
    elapsed_ms: float = 0.0
    metadata: Dict[str, Any] = Field(default_factory=dict)


class HealthResponse(BaseModel):
    """Health check response."""

    healthy: bool
    feature_store: Dict[str, Any] = Field(default_factory=dict)
    model_loaded: bool = False
    tasks: List[str] = Field(default_factory=list)
    kill_switch_enabled: bool = False
    ab_test_enabled: bool = False
    pipeline_enabled: bool = False


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------

app = FastAPI(
    title="PLE Inference Service",
    description="Multi-task recommendation inference (LGBM)",
    version="1.0.0",
)

# Module-level service reference (initialised at startup)
_service = None


@app.on_event("startup")
async def startup_event() -> None:
    """Load model, feature store, and all serving components at startup.

    This mirrors the Lambda cold-start init but runs once when the ECS
    container starts.
    """
    global _service

    logger.info("ECS startup: initialising RecommendationService")

    from core.serving.config import ServingConfig
    from core.serving.feature_store import FeatureStoreFactory
    from core.serving.predict import RecommendationService
    from core.serving.kill_switch import KillSwitch
    from core.serving.ab_test import ABTestManager
    from core.model.lgbm import LGBMModel, LGBMConfig

    # ---- Config ----
    config_json = os.environ.get("SERVING_CONFIG", "{}")
    config_dict = json.loads(config_json)
    config = ServingConfig.from_dict(config_dict)

    # ---- Model ----
    model_dir = os.environ.get("MODEL_DIR", "/opt/ml/model")
    model = LGBMModel.load(
        dir_path=model_dir,
        config=LGBMConfig(),
        tasks_meta=config.tasks_meta,
    )
    logger.info("Model loaded from %s", model_dir)

    # ---- Feature store ----
    user_count_str = os.environ.get("ESTIMATED_USER_COUNT")
    user_count = int(user_count_str) if user_count_str else None
    feature_store = FeatureStoreFactory.create(config, user_count=user_count)

    # ---- Kill switch ----
    kill_switch = None
    if config.kill_switch_table:
        try:
            kill_switch = KillSwitch(
                table_name=config.kill_switch_table,
                fallback_strategy=config.fallback_strategy,
            )
        except Exception:
            logger.warning(
                "KillSwitch init failed, proceeding without it",
                exc_info=True,
            )

    # ---- A/B test ----
    ab_manager = None
    if config.ab_test_enabled and config.ab_variants:
        ab_manager = ABTestManager(variants=config.ab_variants)

    # ---- Pipeline (optional) ----
    pipeline = None
    if config.pipeline_config.get("enabled", False):
        try:
            from core.recommendation.pipeline import RecommendationPipeline
            pipeline = RecommendationPipeline(config.pipeline_config)
        except Exception:
            logger.warning(
                "RecommendationPipeline init failed, proceeding without it",
                exc_info=True,
            )

    _service = RecommendationService(
        model=model,
        feature_store=feature_store,
        tasks_meta=config.tasks_meta,
        kill_switch=kill_switch,
        ab_manager=ab_manager,
        pipeline=pipeline,
        pipeline_config=config.pipeline_config,
    )

    logger.info("RecommendationService initialised successfully")


def _get_service():
    """Return the initialised service or raise if not ready."""
    if _service is None:
        raise HTTPException(
            status_code=503,
            detail="Service not initialised yet",
        )
    return _service


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/", summary="Readiness probe")
async def root() -> Dict[str, str]:
    """Simple readiness probe for load balancer."""
    return {"status": "ok", "service": "ple-inference"}


@app.post(
    "/v1/recommend",
    response_model=PredictionResponseModel,
    summary="Single-user recommendation",
)
async def recommend(request: RecommendRequest) -> Dict[str, Any]:
    """Run inference for a single user.

    Returns normalised predictions across all tasks, with optional
    recommendation scoring, filtering, and reason generation.
    """
    service = _get_service()
    result = service.predict(
        user_id=request.user_id,
        context=request.context,
    )
    return result.to_dict()


@app.post(
    "/v1/recommend/batch",
    response_model=List[PredictionResponseModel],
    summary="Batch recommendation",
)
async def recommend_batch(request: BatchRecommendRequest) -> List[Dict[str, Any]]:
    """Run inference for multiple users in a single request.

    Shares the same context across all users.  For truly independent
    per-user contexts, make individual ``/v1/recommend`` calls.
    """
    service = _get_service()
    results = service.predict_batch(
        user_ids=request.user_ids,
        context=request.context,
    )
    return [r.to_dict() for r in results]


@app.get(
    "/v1/health",
    response_model=HealthResponse,
    summary="Health check",
)
@app.post("/v1/health", response_model=HealthResponse, include_in_schema=False)
async def health_check() -> Dict[str, Any]:
    """Return health status of all sub-components.

    Supports both GET and POST to accommodate different ALB health
    check configurations.
    """
    service = _get_service()
    return service.health_check()
