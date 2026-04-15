"""
AWS Lambda Entry Point
======================

Cold-start initialisation loads the model and feature store once;
subsequent invocations reuse the cached :class:`RecommendationService`
instance.

Event format::

    {
        "user_id": "U12345",
        "context": {            // optional
            "channel": "app_push",
            "segment": "VIP",
            "task": "ctr",
            "cluster": "cluster_A"
        }
    }

    // Batch:
    {
        "batch": true,
        "user_ids": ["U1", "U2", "U3"],
        "context": {}
    }

Response format::

    {
        "statusCode": 200,
        "body": {
            "user_id": "U12345",
            "predictions": {"ctr": 0.83, "cvr": 0.12, "ltv": 42.5},
            "recommendations": [...],
            "variant": "control",
            "elapsed_ms": 14.3
        }
    }
"""

from __future__ import annotations

import json
import logging
import os
from typing import Any, Dict

logger = logging.getLogger(__name__)
logger.setLevel(os.environ.get("LOG_LEVEL", "INFO"))

# ---------------------------------------------------------------------------
# Global singleton -- survives across Lambda invocations (warm start)
# ---------------------------------------------------------------------------
_service = None


def _get_service():
    """Lazy-initialise the RecommendationService on first invocation.

    This runs once per Lambda cold start.  All heavy resources (model,
    feature store, kill switch, A/B manager) are loaded here and cached
    in the module-level ``_service`` variable.
    """
    global _service
    if _service is not None:
        return _service

    logger.info("Lambda cold start: initialising RecommendationService")

    from core.serving.config import ServingConfig
    from core.serving.feature_store import FeatureStoreFactory
    from core.serving.predict import RecommendationService
    from core.serving.kill_switch import KillSwitch
    from core.serving.ab_test import ABTestManager
    from core.model.lgbm import LGBMModel, LGBMConfig

    # ---- Load config from environment / S3 ----
    config_json = os.environ.get("SERVING_CONFIG", "{}")
    config_dict = json.loads(config_json)
    config = ServingConfig.from_dict(config_dict)

    # ---- Load model ----
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

    # ---- Compliance module (config-driven, all components non-blocking) ----
    compliance_cfg = config_dict.get("compliance", {})
    compliance_enabled = compliance_cfg.get("enabled", False)

    consent_manager = None
    ai_opt_out = None
    profiling_rights_manager = None
    regulatory_checker = None
    compliance_audit_store = None

    if compliance_enabled:
        use_dynamo = compliance_cfg.get("use_dynamo", True)
        region = compliance_cfg.get("region", os.environ.get("AWS_DEFAULT_REGION", "ap-northeast-2"))
        audit_prefix = compliance_cfg.get("audit_table_prefix", "ple-audit")

        try:
            from core.compliance import (
                ComplianceAuditStore,
                InMemoryAuditStore,
                ConsentManager,
                AIDecisionOptOut,
                ProfilingRightsManager,
                RegulatoryComplianceChecker,
                get_audit_store,
            )

            # Audit store — shared across all compliance components
            if use_dynamo:
                compliance_audit_store = ComplianceAuditStore(
                    table_prefix=audit_prefix,
                    region=region,
                )
            else:
                compliance_audit_store = InMemoryAuditStore()
            logger.info("ComplianceAuditStore initialised (dynamo=%s)", use_dynamo)

            # Consent manager (channel-level, 금소법)
            consent_manager = ConsentManager(
                table_name=compliance_cfg.get("consent_table", "ple-consent"),
                region=region,
                use_dynamo=use_dynamo,
            )
            logger.info("ConsentManager initialised")

            # AI opt-out (AI기본법 제31조 / GDPR Art. 22)
            ai_opt_out = AIDecisionOptOut(
                table_name=compliance_cfg.get("optout_table", "ple-ai-optout"),
                region=region,
                use_dynamo=use_dynamo,
            )
            logger.info("AIDecisionOptOut initialised")

            # Profiling rights manager (개보법 + GDPR)
            profiling_rights_manager = ProfilingRightsManager(
                table_name=compliance_cfg.get(
                    "profiling_rights_table", "ple-profiling-rights"
                ),
                region=region,
                use_dynamo=use_dynamo,
            )
            logger.info("ProfilingRightsManager initialised")

            # Regulatory checker — uses audit store + serving config as context
            regulatory_checker = RegulatoryComplianceChecker(
                audit_store=compliance_audit_store,
                config=compliance_cfg.get("regulatory_config", {}),
            )
            logger.info("RegulatoryComplianceChecker initialised")

        except Exception:
            logger.warning(
                "Compliance module init failed; proceeding without compliance checks "
                "(non-blocking)",
                exc_info=True,
            )
            # Reset to None so the service runs unchecked rather than crashing
            consent_manager = None
            ai_opt_out = None
            profiling_rights_manager = None
            regulatory_checker = None
            compliance_audit_store = None

    _service = RecommendationService(
        model=model,
        feature_store=feature_store,
        tasks_meta=config.tasks_meta,
        kill_switch=kill_switch,
        ab_manager=ab_manager,
        pipeline=pipeline,
        pipeline_config=config.pipeline_config,
        consent_manager=consent_manager,
        regulatory_checker=regulatory_checker,
        ai_opt_out=ai_opt_out,
        compliance_audit_store=compliance_audit_store,
    )

    logger.info(
        "RecommendationService initialised successfully "
        "(compliance_enabled=%s)",
        compliance_enabled,
    )
    return _service


# ---------------------------------------------------------------------------
# Lambda handler
# ---------------------------------------------------------------------------

def handler(event: Dict[str, Any], context: Any) -> Dict[str, Any]:
    """AWS Lambda entry point.

    Args:
        event: API Gateway proxy event or direct invocation payload.
        context: Lambda context object (runtime metadata).

    Returns:
        Dict with ``statusCode`` and ``body``.
    """
    try:
        service = _get_service()

        # Parse body from API Gateway proxy integration if present
        body = event
        if "body" in event and isinstance(event["body"], str):
            body = json.loads(event["body"])

        # Batch mode
        if body.get("batch", False):
            user_ids = body.get("user_ids", [])
            ctx = body.get("context")
            results = service.predict_batch(user_ids, context=ctx)
            return _success([r.to_dict() for r in results])

        # Single user
        user_id = body.get("user_id")
        if not user_id:
            return _error(400, "Missing required field: user_id")

        ctx = body.get("context")
        result = service.predict(user_id, context=ctx)
        return _success(result.to_dict())

    except Exception as exc:
        logger.exception("Lambda handler error")
        return _error(500, str(exc))


# ---------------------------------------------------------------------------
# Response helpers
# ---------------------------------------------------------------------------

def _success(body: Any) -> Dict[str, Any]:
    return {
        "statusCode": 200,
        "headers": {"Content-Type": "application/json"},
        "body": json.dumps(body, default=str),
    }


def _error(status: int, message: str) -> Dict[str, Any]:
    return {
        "statusCode": status,
        "headers": {"Content-Type": "application/json"},
        "body": json.dumps({"error": message}),
    }
