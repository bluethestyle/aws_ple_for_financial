"""
Lambda handler for LGBM multi-task inference (serverless serving).

Cold-start flow:
    1. Read s3://bucket/models/artifacts/_promoted → active version
    2. Download {version}/students/{task}/model.lgbm (all tasks)
    3. Download {version}/students/{task}/selected_features.json

Warm invocation: models are cached in the Lambda execution environment.

Event payload:
    {
        "user_id": "hashed_user_123",
        "features": {"feat_0": 1.2, "feat_1": 0.5, ...},   # full feature vector
        "context": {"channel": "app", "segment": "vip"},    # optional
        "tasks": ["ctr", "churn"]                            # optional subset
    }

Returns:
    {
        "user_id": "...",
        "version": "v-...",
        "predictions": {"ctr": 0.82, "churn": 0.05, ...},
        "variant": "control",
        "elapsed_ms": 12.4
    }
"""

from __future__ import annotations

import json
import logging
import math
import os
import time
import tempfile
from typing import Any, Dict, List, Optional

logger = logging.getLogger()
logger.setLevel(logging.INFO)

# ---------------------------------------------------------------------------
# Security: lazy-import guards so the handler works even if core.security
# is not packaged (e.g. lightweight Lambda layer without full source tree).
# ---------------------------------------------------------------------------

def _get_prompt_sanitizer():
    """Return a cached PromptSanitizer instance, or None if unavailable."""
    if _CACHE.get("_prompt_sanitizer") is None:
        try:
            from core.security import PromptSanitizer
            _CACHE["_prompt_sanitizer"] = PromptSanitizer(
                internal_provider="bedrock",
                external_provider="gemini",
                scrub_for_external=True,
            )
            logger.info("PromptSanitizer initialised")
        except Exception as e:
            logger.warning("PromptSanitizer unavailable: %s", e)
            _CACHE["_prompt_sanitizer"] = False  # sentinel: skip future attempts
    obj = _CACHE.get("_prompt_sanitizer")
    return obj if obj is not False else None


def _get_pii_encryptor():
    """Return a cached PIIEncryptor for inbound feature scrubbing, or None."""
    if _CACHE.get("_pii_encryptor") is None:
        try:
            from core.security import LocalSaltManager, PIIEncryptor
            salt_mgr = LocalSaltManager()
            _CACHE["_pii_encryptor"] = PIIEncryptor(salt_mgr)
            logger.info("PIIEncryptor initialised (LocalSaltManager)")
        except Exception as e:
            logger.warning("PIIEncryptor unavailable: %s", e)
            _CACHE["_pii_encryptor"] = False
    obj = _CACHE.get("_pii_encryptor")
    return obj if obj is not False else None


# Env-var flags (set in Lambda config / CDK)
_SECURITY_PII_SCRUB = os.environ.get("SECURITY_PII_SCRUB", "true").lower() == "true"
_SECURITY_FEATURE_SCAN = os.environ.get("SECURITY_FEATURE_SCAN", "true").lower() == "true"

# ---------------------------------------------------------------------------
# Module-level cache (persists across warm invocations)
# ---------------------------------------------------------------------------

_CACHE: Dict[str, Any] = {
    "version": None,
    "models": {},           # task_name -> lgb.Booster (champion)
    "features": {},         # task_name -> {"indices": [...], "names": [...]}
    "tasks_meta": [],       # list of {"name": str, "type": str}
    # 3-layer fallback components (loaded once at cold start)
    "fallback_router": None,   # FallbackRouter instance | None
    "rule_engine": None,       # RuleBasedRecommender instance | None
    "calibrators": {},         # task_name -> calibrator object
    # Security components (lazy-initialised on first use)
    "_prompt_sanitizer": None,   # PromptSanitizer | False
    "_pii_encryptor": None,      # PIIEncryptor | False
}

# Variant model cache: variant_name -> {"version": str, "models": {task->Booster}, "features": {}}
_VARIANT_CACHE: Dict[str, Dict[str, Any]] = {}

# ---------------------------------------------------------------------------
# Agent infrastructure — lazy singleton (optional, non-blocking)
# Activated when AGENT_ENABLED=true is set in the Lambda environment.
# ChangeDetector emits model-version events; HeartbeatScheduler tracks
# agent health across warm invocations.
#
# DESIGN NOTE — ChangeDetector is NOT called in the predict hot path.
# It fires exclusively on pipeline stage completions (cold-start model load
# and version promotions) via _ensure_loaded().  Calling it on every
# predict invocation would add latency and produce meaningless "change"
# events — model versions change infrequently, not per request.
# Do not add ChangeDetector calls inside handler() warm-path code.
# ---------------------------------------------------------------------------

AGENT_ENABLED = os.environ.get("AGENT_ENABLED", "false").lower() == "true"

_CHANGE_DETECTOR: Optional[Any] = None   # core.agent.change_detector.ChangeDetector


def _get_change_detector() -> Optional[Any]:
    """Lazy-initialise the ChangeDetector singleton (once per execution environment)."""
    global _CHANGE_DETECTOR
    if not AGENT_ENABLED:
        return None
    if _CHANGE_DETECTOR is None:
        try:
            from core.agent.change_detector import ChangeDetector
            _CHANGE_DETECTOR = ChangeDetector()
            logger.info("ChangeDetector initialised (agent infrastructure)")
        except Exception as e:
            logger.warning("ChangeDetector unavailable: %s", e)
            _CHANGE_DETECTOR = False  # sentinel: skip future attempts
    return _CHANGE_DETECTOR if _CHANGE_DETECTOR is not False else None


# A/B test configuration (from env or event)
AB_ENABLED = os.environ.get("AB_ENABLED", "false").lower() == "true"
AB_CHALLENGER_VERSION = os.environ.get("AB_CHALLENGER_VERSION", "")
AB_CHALLENGER_WEIGHT = float(os.environ.get("AB_CHALLENGER_WEIGHT", "0.2"))

# ---------------------------------------------------------------------------
# Environment variables (set in Lambda config / CDK)
# ---------------------------------------------------------------------------

REGISTRY_BASE = os.environ.get(
    "REGISTRY_BASE", "s3://aiops-ple-financial/models/artifacts"
)
MONITOR_TABLE = os.environ.get("MONITOR_TABLE", "ple-prediction-log")
CW_NAMESPACE = os.environ.get("CW_NAMESPACE", "PLE/Serving")
REGION = os.environ.get("AWS_REGION", "ap-northeast-2")


# ---------------------------------------------------------------------------
# Handler
# ---------------------------------------------------------------------------

def handler(event: Dict[str, Any], context: Any) -> Dict[str, Any]:
    """Lambda entry point for LGBM multi-task inference."""
    t0 = time.perf_counter()

    # ---- Heartbeat / health check (agent infrastructure) ----
    # Invoked with {"action": "heartbeat"} by the OpsAgent HeartbeatScheduler.
    # Returns current cache state without running inference.
    if event.get("action") == "heartbeat":
        return {
            "status": "ok",
            "version": _CACHE.get("version"),
            "models_loaded": list(_CACHE.get("models", {}).keys()),
            "agent_enabled": AGENT_ENABLED,
            "elapsed_ms": round((time.perf_counter() - t0) * 1000.0, 2),
        }

    user_id: str = event.get("user_id", "")
    features: Dict[str, float] = event.get("features", {})
    ctx: Dict[str, Any] = event.get("context", {})
    requested_tasks: Optional[List[str]] = event.get("tasks")

    if not user_id:
        return {"error": "user_id required", "status": 400}

    # ---- Security: scan inbound features for raw PII columns (non-blocking) ----
    if _SECURITY_FEATURE_SCAN and features:
        encryptor = _get_pii_encryptor()
        if encryptor is not None:
            try:
                import pandas as _pd
                from core.security.domains import PIIDomain, resolve_domain
                pii_cols = [
                    col for col in features
                    if resolve_domain(col) != PIIDomain.DEFAULT
                ]
                if pii_cols:
                    logger.warning(
                        "Inbound features contain %d raw PII column(s): %s — "
                        "these should be pre-hashed by the caller",
                        len(pii_cols), pii_cols,
                    )
                    # Hash them in place so they don't propagate raw
                    feat_df = _pd.DataFrame([features])
                    col_domain_map = {c: resolve_domain(c) for c in pii_cols}
                    hashed_df = encryptor.hash_dataframe(feat_df, col_domain_map)
                    features = hashed_df.iloc[0].to_dict()
            except Exception:
                logger.warning(
                    "Feature PII scan failed (non-fatal)", exc_info=True
                )

    import boto3
    s3 = boto3.client("s3", region_name=REGION)

    # --- Ensure models are loaded (cold start or version change) ---
    _ensure_loaded(s3)

    # --- A/B variant selection ---
    variant_name = "control"
    version = _CACHE["version"]
    models = _CACHE["models"]
    feat_meta = _CACHE["features"]
    tasks_meta = _CACHE["tasks_meta"]

    if AB_ENABLED and AB_CHALLENGER_VERSION:
        # Deterministic assignment: hash(user_id) → [0,1)
        import hashlib
        h = int(hashlib.sha256(f"ple_ab:{user_id}".encode()).hexdigest()[:8], 16) / 0x1_0000_0000
        if h < AB_CHALLENGER_WEIGHT:
            variant_name = "challenger"
            # Load challenger models if not cached
            _ensure_variant_loaded(s3, AB_CHALLENGER_VERSION)
            vc = _VARIANT_CACHE.get(AB_CHALLENGER_VERSION, {})
            if vc.get("models"):
                version = AB_CHALLENGER_VERSION
                models = vc["models"]
                feat_meta = vc.get("features", feat_meta)
                tasks_meta = vc.get("tasks_meta", tasks_meta)

    if not models:
        return {"error": "no models available", "status": 503}

    # --- Cold-start path: no features provided or empty ---
    if not features:
        elapsed = round((time.perf_counter() - t0) * 1000.0, 2)
        result = _handle_coldstart(user_id, ctx, version, tasks_meta, models, feat_meta, elapsed)
        try:
            _log_prediction(boto3, user_id, version, {}, elapsed, ctx)
        except Exception:
            logger.warning("Monitoring write failed (non-fatal)", exc_info=True)
        return result

    # --- Determine which tasks to score ---
    all_tasks = list(models.keys())
    tasks_to_score = requested_tasks if requested_tasks else all_tasks
    tasks_to_score = [t for t in tasks_to_score if t in models]

    # --- Per-task inference ---
    predictions: Dict[str, Any] = {}
    import numpy as np

    for task_name in tasks_to_score:
        booster = models[task_name]

        # Build feature vector in the exact column order the model was trained on
        task_meta = feat_meta.get(task_name, {})
        feature_columns = task_meta.get("feature_columns", [])
        if not feature_columns:
            # Fallback: use booster's feature names if available
            feature_columns = booster.feature_name()

        if feature_columns:
            selected = [float(features.get(c, 0.0) or 0.0) for c in feature_columns]
        else:
            selected = list(features.values())

        X = np.array(selected, dtype=np.float32).reshape(1, -1)
        raw = booster.predict(X)  # shape: (1,) binary or (1, n_classes)

        # Normalise
        task_type = _get_task_type(task_name, tasks_meta)
        predictions[task_name] = _normalise(raw[0], task_type)

    # --- 3-layer fallback routing ---
    # When fallback_router is None (not configured), all tasks use LGBM as-is
    # (backward-compatible path).
    layer_used_per_task: Dict[str, int] = {}
    fallback_router = _CACHE.get("fallback_router")
    rule_engine = _CACHE.get("rule_engine")
    calibrators = _CACHE.get("calibrators") or {}

    if fallback_router is not None:
        routing = fallback_router.route_all(
            task_names=tasks_to_score,
            lgbm_models={t: models[t] for t in tasks_to_score},
            rule_engine=rule_engine,
        )
        for task_name, layer in routing.items():
            layer_used_per_task[task_name] = layer
            if layer == 3 and rule_engine is not None:
                try:
                    rule_result = rule_engine.predict(
                        features=features,
                        task_name=task_name,
                    )
                    predictions[task_name] = rule_result["prediction"]
                    logger.debug(
                        "FallbackRouter: task=%s → Layer 3 (rule=%s)",
                        task_name, rule_result.get("rule_name", ""),
                    )
                except Exception:
                    logger.warning(
                        "Rule engine failed for task=%s, keeping LGBM output",
                        task_name,
                        exc_info=True,
                    )

    # --- Calibration ---
    # Applied silently: caller sees the same response shape.
    calibrated_tasks: List[str] = []
    if calibrators:
        for task_name, raw_val in list(predictions.items()):
            cal = calibrators.get(task_name)
            if cal is None:
                continue
            try:
                task_type = _get_task_type(task_name, tasks_meta)
                if task_type == "binary":
                    x = np.array([[float(raw_val)]])
                    prob = cal.predict_proba(x)[0, 1]
                    predictions[task_name] = round(float(prob), 6)
                elif task_type == "multiclass":
                    x = np.array([raw_val])
                    prob = cal.predict_proba(x)[0].tolist()
                    predictions[task_name] = [round(float(p), 6) for p in prob]
                calibrated_tasks.append(task_name)
            except Exception:
                logger.warning(
                    "Calibration failed for task=%s, keeping raw value",
                    task_name,
                    exc_info=True,
                )

    # --- Recommendation Reason Generation (3-agent pipeline) ---
    reasons: Dict[str, Any] = {}
    try:
        reason_cfg = _CACHE.get("_reason_config")
        if reason_cfg is None:
            # Read pre-built JSON config (converted from YAML at package time)
            _local_cfg = os.path.join(os.path.dirname(__file__), "configs", "merged_config.json")
            try:
                with open(_local_cfg, encoding="utf-8") as _cf:
                    reason_cfg = json.load(_cf)
            except Exception:
                reason_cfg = {}
            _CACHE["_reason_config"] = reason_cfg

        # Agent 1: FactExtractor
        fact_extractor = _CACHE.get("_fact_extractor")
        if fact_extractor is None:
            try:
                from core.recommendation.reason.fact_extractor import FactExtractor
                fe_path = reason_cfg.get("reason", {}).get("fact_extractor", {}).get(
                    "config_path", "configs/financial/fact_extraction.yaml",
                )
                # Resolve relative to Lambda package root
                if not os.path.isabs(fe_path):
                    fe_path = os.path.join(os.path.dirname(__file__), fe_path)
                fact_extractor = FactExtractor(fe_path)
                _CACHE["_fact_extractor"] = fact_extractor
            except Exception:
                _CACHE["_fact_extractor"] = False

        # Agent 2: TemplateEngine
        template_engine = _CACHE.get("_template_engine")
        if template_engine is None:
            try:
                from core.recommendation.reason.template_engine import TemplateEngine
                template_engine = TemplateEngine(reason_cfg)
                _CACHE["_template_engine"] = template_engine
            except Exception:
                _CACHE["_template_engine"] = False

        # Generate reasons for top-3 tasks
        if template_engine and template_engine is not False:
            import numpy as _np
            sorted_tasks = sorted(
                [(t, v) for t, v in predictions.items() if isinstance(v, (int, float))],
                key=lambda x: -x[1],
            )[:3]

            facts = []
            if fact_extractor and fact_extractor is not False:
                facts = fact_extractor.extract(features)

            for task_name, _ in sorted_tasks:
                task_type = _get_task_type(task_name, tasks_meta)
                # Get LGBM gain top features for this task
                booster = models.get(task_name)
                ig_top = []
                if booster is not None:
                    gains = booster.feature_importance(importance_type="gain")
                    feat_names = booster.feature_name()
                    top_idx = _np.argsort(gains)[::-1][:5]
                    ig_top = [(feat_names[i], float(gains[i])) for i in top_idx]

                reason_result = template_engine.generate_reason(
                    customer_id=user_id,
                    item_id=task_name,
                    ig_top_features=ig_top,
                    segment="WARMSTART",
                    task_type=task_type,
                    task_name=task_name,
                )
                reason_texts = [r.get("text", "") for r in reason_result.get("reasons", [])]
                reasons[task_name] = {
                    "l1_reasons": reason_texts,
                    "facts": facts,
                    "generation_method": "template_l1",
                }

            # --- Bedrock L2a Rewrite: top-1 task only ---
            # Rewrites the top-scoring task's first L1 reason into natural Korean.
            # Falls back to L1 text on any error (non-blocking).
            if sorted_tasks:
                top_task_name = sorted_tasks[0][0]
                top_task_reasons = reasons.get(top_task_name, {})
                l1_texts = top_task_reasons.get("l1_reasons", [])
                l1_text = l1_texts[0] if l1_texts else ""

                if l1_text:
                    l2a_text = l1_text  # default fallback
                    try:
                        bedrock_cfg = (
                            reason_cfg
                            .get("reason", {})
                            .get("llm_provider", {})
                            .get("bedrock", {})
                        )
                        # model_id from config; fall back to inference profile
                        model_cfg = (
                            bedrock_cfg
                            .get("models", {})
                            .get("reason_generation", {})
                        )
                        model_id = model_cfg.get(
                            "model_id",
                            "global.anthropic.claude-sonnet-4-6",
                        )
                        bedrock_region = bedrock_cfg.get("region", REGION)
                        max_tokens = model_cfg.get("max_tokens", 256)
                        temperature = model_cfg.get("temperature", 0.3)

                        from core.recommendation.reason.llm_provider import BedrockProvider
                        _bedrock = BedrockProvider({
                            "model_id": model_id,
                            "region": bedrock_region,
                            "max_tokens": max_tokens,
                            "temperature": temperature,
                        })
                        _system_prompt = (
                            "당신은 금융 상품 추천사유 작성자입니다. "
                            "입력된 추천사유를 고객에게 전달할 자연스러운 한국어 1~2문장으로 "
                            "다듬어 출력하세요. 분석, 검토, 마크다운, 제목, 목록 없이 "
                            "오직 다듬어진 문장만 출력하세요."
                        )
                        l2a_text = _bedrock.generate(
                            l1_text,
                            system=_system_prompt,
                        ).strip() or l1_text
                        logger.info(
                            "L2a Bedrock rewrite OK: task=%s model=%s",
                            top_task_name, model_id,
                        )
                    except Exception:
                        logger.warning(
                            "L2a Bedrock rewrite failed (non-fatal), using L1 text",
                            exc_info=True,
                        )

                    reasons[top_task_name]["l2a_reason"] = l2a_text

    except Exception as _reason_exc:
        logger.warning("Reason generation failed (non-fatal)", exc_info=True)
        reasons["_error"] = str(_reason_exc)

    elapsed = round((time.perf_counter() - t0) * 1000.0, 2)

    # Extract top-level l2a_reason for convenience (from top-scoring task)
    sorted_scalar = sorted(
        [(t, v) for t, v in predictions.items() if isinstance(v, (int, float))],
        key=lambda x: -x[1],
    )
    top_l2a_reason = ""
    if sorted_scalar:
        _top_task = sorted_scalar[0][0]
        top_l2a_reason = reasons.get(_top_task, {}).get("l2a_reason", "")

    result: Dict[str, Any] = {
        "user_id": user_id,
        "version": version,
        "variant": variant_name,
        "predictions": predictions,
        "reasons": reasons,
        "elapsed_ms": elapsed,
    }
    if top_l2a_reason:
        result["l2a_reason"] = top_l2a_reason
    if layer_used_per_task:
        result["layer_used_per_task"] = layer_used_per_task
    if calibrated_tasks:
        result["calibrated_tasks"] = calibrated_tasks

    # ---- Security: scrub PII from outbound response (fail-closed for PII) ----
    if _SECURITY_PII_SCRUB:
        sanitizer = _get_prompt_sanitizer()
        if sanitizer is not None:
            try:
                # Scrub the echoed user_id — mask any raw PII patterns
                raw_uid = result.get("user_id", "")
                if raw_uid:
                    scrubbed_uid, scrub_count = sanitizer.scrub(raw_uid)
                    if scrub_count > 0:
                        logger.warning(
                            "Scrubbed %d PII item(s) from outbound user_id "
                            "(original hash=%s)",
                            scrub_count,
                            __import__("hashlib").sha256(
                                raw_uid.encode()
                            ).hexdigest()[:8],
                        )
                        result["user_id"] = scrubbed_uid
            except Exception:
                logger.warning(
                    "Outbound PII scrub failed (non-fatal)", exc_info=True
                )

    # --- Async performance logging (best-effort) ---
    try:
        ctx_with_variant = {**ctx, "variant": variant_name}
        _log_prediction(
            boto3, user_id, version, predictions, elapsed, ctx_with_variant,
            reasons=reasons,
            l2a_reason=top_l2a_reason,
        )
        _emit_metrics(
            boto3, version, predictions, elapsed, tasks_meta,
            layer_used_per_task=layer_used_per_task if layer_used_per_task else None,
        )
    except Exception:
        logger.warning("Monitoring write failed (non-fatal)", exc_info=True)

    return result


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def _ensure_loaded(s3) -> None:
    """Load champion models into module-level cache if not already loaded."""
    try:
        active_version = _read_promoted_version(s3)
    except Exception as e:
        logger.error("Cannot determine active model version: %s", e)
        return

    if _CACHE["version"] == active_version and _CACHE["models"]:
        return  # Already loaded and up-to-date

    logger.info("Loading model version: %s", active_version)

    tmp = tempfile.mkdtemp(prefix="lgbm_")
    try:
        import lightgbm as lgb

        # Discover task directories
        bucket, prefix = _parse_s3_uri(
            f"{REGISTRY_BASE.rstrip('/')}/{active_version}/students/"
        )
        paginator = s3.get_paginator("list_objects_v2")
        task_dirs: Dict[str, bool] = {}
        for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
            for obj in page.get("Contents", []):
                key = obj["Key"]
                rel = key[len(prefix):]
                parts = rel.strip("/").split("/")
                if len(parts) >= 1 and parts[0]:
                    task_dirs[parts[0]] = True

        models: Dict[str, Any] = {}
        feat_meta: Dict[str, Any] = {}
        tasks_meta: List[Dict[str, str]] = []

        for task_name in task_dirs:
            model_key = f"{prefix}{task_name}/model.lgbm"
            feat_key = f"{prefix}{task_name}/selected_features.json"
            meta_key = f"{prefix}{task_name}/metadata.json"

            local_model = f"{tmp}/{task_name}.lgbm"
            try:
                s3.download_file(bucket, model_key, local_model)
                booster = lgb.Booster(model_file=local_model)
                models[task_name] = booster
            except Exception as e:
                logger.warning("Skipping task %s: %s", task_name, e)
                continue

            # Feature selection metadata
            try:
                obj = s3.get_object(Bucket=bucket, Key=feat_key)
                feat_meta[task_name] = json.loads(obj["Body"].read())
            except Exception:
                feat_meta[task_name] = {}

            # Task type from metadata.json
            task_type = "binary"
            try:
                obj = s3.get_object(Bucket=bucket, Key=meta_key)
                meta = json.loads(obj["Body"].read())
                task_type = meta.get("task_type", "binary")
            except Exception:
                pass

            tasks_meta.append({"name": task_name, "type": task_type})

        _CACHE["version"] = active_version
        _CACHE["models"] = models
        _CACHE["features"] = feat_meta
        _CACHE["tasks_meta"] = tasks_meta

        # Load 3-layer fallback components (best-effort; failures are non-fatal)
        _CACHE["fallback_router"] = _load_fallback_router(s3, bucket, active_version, tasks_meta)
        _CACHE["rule_engine"] = _load_rule_engine(s3, bucket, active_version)
        _CACHE["calibrators"] = _load_calibrators(s3, bucket, active_version, list(models.keys()))

        logger.info(
            "Loaded %d student models for version %s",
            len(models), active_version,
        )

        # ---- Agent: model version change event (non-blocking) ----
        # Fires on every cold start or version promotion so ChangeDetector
        # can emit a "model" event to the OpsAgent / AuditAgent pipeline.
        _cd = _get_change_detector()
        if _cd is not None:
            try:
                _cd.on_pipeline_stage_complete(
                    stage="stage_serving",
                    artifacts={
                        "version": active_version,
                        "tasks_loaded": list(models.keys()),
                        "fallback_router": _CACHE.get("fallback_router") is not None,
                        "rule_engine": _CACHE.get("rule_engine") is not None,
                        "calibrators": list(_CACHE.get("calibrators", {}).keys()),
                        "source": "lambda_cold_start",
                    },
                )
            except Exception:
                logger.debug("ChangeDetector cold-start event failed (non-fatal)", exc_info=True)

    finally:
        import shutil
        shutil.rmtree(tmp, ignore_errors=True)


def _ensure_variant_loaded(s3, variant_version: str) -> None:
    """Load a challenger model version into the variant cache."""
    if variant_version in _VARIANT_CACHE and _VARIANT_CACHE[variant_version].get("models"):
        return  # Already loaded

    logger.info("Loading challenger model version: %s", variant_version)

    tmp = __import__("tempfile").mkdtemp(prefix="lgbm_variant_")
    try:
        import lightgbm as lgb

        bucket, prefix = _parse_s3_uri(
            f"{REGISTRY_BASE.rstrip('/')}/{variant_version}/students/"
        )
        paginator = s3.get_paginator("list_objects_v2")
        task_dirs: Dict[str, bool] = {}
        for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
            for obj in page.get("Contents", []):
                key = obj["Key"]
                rel = key[len(prefix):]
                parts = rel.strip("/").split("/")
                if len(parts) >= 1 and parts[0]:
                    task_dirs[parts[0]] = True

        models: Dict[str, Any] = {}
        feat_meta: Dict[str, Any] = {}
        tasks_meta: List[Dict[str, str]] = []

        for task_name in task_dirs:
            model_key = f"{prefix}{task_name}/model.lgbm"
            feat_key = f"{prefix}{task_name}/selected_features.json"
            meta_key = f"{prefix}{task_name}/metadata.json"

            local_model = f"{tmp}/{task_name}.lgbm"
            try:
                s3.download_file(bucket, model_key, local_model)
                booster = lgb.Booster(model_file=local_model)
                models[task_name] = booster
            except Exception as e:
                logger.warning("Variant %s: skip task %s: %s", variant_version, task_name, e)
                continue

            try:
                obj = s3.get_object(Bucket=bucket, Key=feat_key)
                feat_meta[task_name] = json.loads(obj["Body"].read())
            except Exception:
                feat_meta[task_name] = {}

            task_type = "binary"
            try:
                obj = s3.get_object(Bucket=bucket, Key=meta_key)
                meta = json.loads(obj["Body"].read())
                task_type = meta.get("task_type", "binary")
            except Exception:
                pass
            tasks_meta.append({"name": task_name, "type": task_type})

        _VARIANT_CACHE[variant_version] = {
            "version": variant_version,
            "models": models,
            "features": feat_meta,
            "tasks_meta": tasks_meta,
        }

        logger.info(
            "Loaded %d challenger models for version %s",
            len(models), variant_version,
        )
    except Exception as e:
        logger.error("Failed to load variant %s: %s", variant_version, e)
        _VARIANT_CACHE[variant_version] = {"models": {}}
    finally:
        import shutil
        shutil.rmtree(tmp, ignore_errors=True)


def _read_promoted_version(s3) -> str:
    """Read the _promoted marker to get the active version."""
    bucket, key = _parse_s3_uri(f"{REGISTRY_BASE.rstrip('/')}/_promoted")
    obj = s3.get_object(Bucket=bucket, Key=key)
    data = json.loads(obj["Body"].read())
    return data["active_version"]


# ---------------------------------------------------------------------------
# 3-layer fallback component loaders
# ---------------------------------------------------------------------------

def _load_fallback_router(
    s3, bucket: str, version: str, tasks_meta: List[Dict[str, str]]
) -> Optional[Any]:
    """Load FallbackRouter from pipeline config stored in S3.

    Expects ``{version}/pipeline_config.json`` to exist alongside the model
    artifacts.  Returns ``None`` if the config cannot be loaded or the
    FallbackRouter import fails.
    """
    try:
        config_key = f"models/artifacts/{version}/pipeline_config.json"
        # Strip leading bucket path from REGISTRY_BASE prefix
        _, base_prefix = _parse_s3_uri(REGISTRY_BASE.rstrip("/"))
        config_key = f"{base_prefix.rstrip('/')}/{version}/pipeline_config.json"

        obj = s3.get_object(Bucket=bucket, Key=config_key)
        pipeline_cfg = json.loads(obj["Body"].read())

        # Inject tasks_meta so the router can look up task types
        pipeline_cfg.setdefault("tasks", tasks_meta)

        from core.recommendation.fallback_router import FallbackRouter
        router = FallbackRouter(pipeline_cfg)
        logger.info("FallbackRouter loaded for version %s", version)
        return router
    except s3.exceptions.NoSuchKey:
        logger.info(
            "pipeline_config.json not found for version %s — "
            "FallbackRouter disabled (LGBM-only mode)",
            version,
        )
        return None
    except Exception:
        logger.warning(
            "Failed to load FallbackRouter for version %s",
            version,
            exc_info=True,
        )
        return None


def _load_rule_engine(s3, bucket: str, version: str) -> Optional[Any]:
    """Load RuleBasedRecommender using the pipeline config from S3.

    Returns ``None`` if the config is absent or import fails.
    """
    try:
        _, base_prefix = _parse_s3_uri(REGISTRY_BASE.rstrip("/"))
        config_key = f"{base_prefix.rstrip('/')}/{version}/pipeline_config.json"

        obj = s3.get_object(Bucket=bucket, Key=config_key)
        pipeline_cfg = json.loads(obj["Body"].read())

        from core.recommendation.rule_engine import RuleBasedRecommender
        engine = RuleBasedRecommender(pipeline_cfg)
        logger.info("RuleBasedRecommender loaded for version %s", version)
        return engine
    except s3.exceptions.NoSuchKey:
        logger.info(
            "pipeline_config.json not found for version %s — "
            "RuleBasedRecommender disabled",
            version,
        )
        return None
    except Exception:
        logger.warning(
            "Failed to load RuleBasedRecommender for version %s",
            version,
            exc_info=True,
        )
        return None


def _load_calibrators(
    s3, bucket: str, version: str, task_names: List[str]
) -> Dict[str, Any]:
    """Download and deserialise ``calibrator.joblib`` for each task.

    Expects the layout::

        {REGISTRY_BASE}/{version}/students/{task_name}/calibrator.joblib

    Tasks without a calibrator file are silently skipped.
    """
    try:
        import joblib
    except ImportError:
        logger.warning("joblib not installed — calibration disabled")
        return {}

    _, base_prefix = _parse_s3_uri(REGISTRY_BASE.rstrip("/"))
    students_prefix = f"{base_prefix.rstrip('/')}/{version}/students"

    calibrators: Dict[str, Any] = {}
    for task_name in task_names:
        cal_key = f"{students_prefix}/{task_name}/calibrator.joblib"
        try:
            with tempfile.NamedTemporaryFile(suffix=".joblib", delete=False) as f:
                tmp_path = f.name
            s3.download_file(bucket, cal_key, tmp_path)
            calibrators[task_name] = joblib.load(tmp_path)
            logger.info("Calibrator loaded for task=%s", task_name)
        except s3.exceptions.NoSuchKey:
            pass  # No calibrator for this task — normal
        except Exception:
            logger.warning(
                "Failed to load calibrator for task=%s",
                task_name,
                exc_info=True,
            )
        finally:
            try:
                import os as _os
                _os.unlink(tmp_path)
            except Exception:
                pass

    logger.info(
        "Calibrators loaded: %d/%d tasks", len(calibrators), len(task_names)
    )
    return calibrators


# ---------------------------------------------------------------------------
# Output normalisation
# ---------------------------------------------------------------------------

def _normalise(raw: Any, task_type: str) -> Any:
    """Normalise raw LGBM output based on task type."""
    import numpy as np

    if task_type == "binary":
        val = float(np.asarray(raw).ravel()[0]) if hasattr(raw, "__len__") else float(raw)
        # Custom objective LGBM outputs raw logits; apply sigmoid
        val = 1.0 / (1.0 + np.exp(-val))
        return round(val, 6)
    elif task_type == "multiclass":
        arr = np.asarray(raw).ravel()
        # Already probabilities from LGBM
        return [round(float(x), 6) for x in arr]
    else:
        # regression / ranking
        val = float(np.asarray(raw).ravel()[0]) if hasattr(raw, "__len__") else float(raw)
        return round(val, 6)


def _get_task_type(task_name: str, tasks_meta: List[Dict[str, str]]) -> str:
    for t in tasks_meta:
        if t["name"] == task_name:
            return t.get("type", "binary")
    return "binary"


# ---------------------------------------------------------------------------
# Monitoring
# ---------------------------------------------------------------------------

def _log_prediction(
    boto3,
    user_id: str,
    version: str,
    predictions: Dict[str, Any],
    elapsed_ms: float,
    ctx: Dict[str, Any],
    reasons: Optional[Dict[str, Any]] = None,
    l2a_reason: str = "",
) -> None:
    """Write prediction record to DynamoDB for Champion-Challenger analysis.

    Also writes to ple-audit-distillation when distillation metadata is present
    in the version string (e.g. version contains 'distill').
    """
    import uuid
    from datetime import datetime, timezone

    dynamodb = boto3.resource("dynamodb", region_name=REGION)
    table = dynamodb.Table(MONITOR_TABLE)

    prediction_id = str(uuid.uuid4())
    now_iso = datetime.now(timezone.utc).isoformat()
    ttl_val = int(time.time()) + 90 * 86400  # 90-day TTL

    # Build top-task reason summary for the audit record (non-PII text only)
    top_reason_text = ""
    if reasons:
        sorted_scalar = sorted(
            [(t, v) for t, v in predictions.items() if isinstance(v, (int, float))],
            key=lambda x: -x[1],
        )
        if sorted_scalar:
            top_task = sorted_scalar[0][0]
            task_reasons = reasons.get(top_task, {})
            l1_texts = task_reasons.get("l1_reasons", [])
            top_reason_text = l2a_reason or (l1_texts[0] if l1_texts else "")

    record = {
        "prediction_id": prediction_id,
        "user_id": user_id,
        "version": version,
        "variant": ctx.get("variant", "control"),
        "predictions": {k: str(v) for k, v in predictions.items()},
        "elapsed_ms": str(round(elapsed_ms, 2)),
        "channel": ctx.get("channel", "unknown"),
        "segment": ctx.get("segment", "unknown"),
        "timestamp": now_iso,
        "ttl": ttl_val,
    }
    if top_reason_text:
        record["top_reason"] = top_reason_text
    if l2a_reason:
        record["l2a_reason"] = l2a_reason

    table.put_item(Item=record)

    # --- Distillation audit log (best-effort) ---
    # Written when version metadata indicates distilled students are serving.
    try:
        distill_table_name = "ple-audit-distillation"
        distill_table = dynamodb.Table(distill_table_name)
        # Determine tasks served by distilled student (all tasks by default)
        tasks_served = list(predictions.keys())
        distill_record = {
            "pk": f"predict#{prediction_id}",
            "prediction_id": prediction_id,
            "user_id": user_id,
            "version": version,
            "tasks_served": tasks_served,
            "channel": ctx.get("channel", "unknown"),
            "timestamp": now_iso,
            "ttl": ttl_val,
        }
        if top_reason_text:
            distill_record["top_reason"] = top_reason_text
        distill_table.put_item(Item=distill_record)
        logger.debug("Distillation audit written: prediction_id=%s", prediction_id)
    except Exception:
        logger.debug("Distillation audit write failed (non-fatal)", exc_info=True)


def _emit_metrics(
    boto3,
    version: str,
    predictions: Dict[str, Any],
    elapsed_ms: float,
    tasks_meta: List[Dict[str, str]],
    layer_used_per_task: Optional[Dict[str, int]] = None,
) -> None:
    """Emit per-task prediction scores and latency to CloudWatch."""
    cw = boto3.client("cloudwatch", region_name=REGION)

    metric_data = [
        {
            "MetricName": "InferenceLatency",
            "Dimensions": [{"Name": "Version", "Value": version}],
            "Value": elapsed_ms,
            "Unit": "Milliseconds",
        }
    ]

    for task_name, value in predictions.items():
        # Only emit scalar scores
        if isinstance(value, (int, float)):
            metric_data.append({
                "MetricName": "PredictionScore",
                "Dimensions": [
                    {"Name": "Version", "Value": version},
                    {"Name": "Task", "Value": task_name},
                ],
                "Value": float(value),
                "Unit": "None",
            })

    # Layer usage metrics (3-layer fallback observability)
    layer_used_map = layer_used_per_task or {}
    for task_name, layer in layer_used_map.items():
        metric_data.append({
            "MetricName": "LayerUsed",
            "Dimensions": [
                {"Name": "Version", "Value": version},
                {"Name": "Task", "Value": str(task_name)},
            ],
            "Value": float(layer),
            "Unit": "None",
        })

    # Layer distribution summary (count per layer)
    from collections import Counter
    layer_counts = Counter(layer_used_map.values())
    for layer_num, count in layer_counts.items():
        metric_data.append({
            "MetricName": "LayerDistribution",
            "Dimensions": [
                {"Name": "Version", "Value": version},
                {"Name": "Layer", "Value": str(layer_num)},
            ],
            "Value": float(count),
            "Unit": "Count",
        })

    # CloudWatch put_metric_data accepts up to 20 items per call
    for i in range(0, len(metric_data), 20):
        cw.put_metric_data(
            Namespace=CW_NAMESPACE,
            MetricData=metric_data[i : i + 20],
        )


# ---------------------------------------------------------------------------
# Cold-start path
# ---------------------------------------------------------------------------

def _handle_coldstart(
    user_id: str,
    ctx: Dict[str, Any],
    version: str,
    tasks_meta: List[Dict[str, str]],
    models: Dict[str, Any],
    feat_meta: Dict[str, Any],
    elapsed_ms: float,
) -> Dict[str, Any]:
    """Build a cold-start response using popularity candidates.

    COLDSTART: user_id is known but no features are available.
    ANONYMOUS: empty user_id (caller should set user_id="" for anonymous).

    For COLDSTART users, we run LGBM inference with a default feature vector
    (zeros + cold-start signal overrides) so the model can still rank items.
    For ANONYMOUS users, we return the static popularity catalog directly.
    """
    import numpy as np

    # Classify segment
    is_anonymous = not user_id or ctx.get("is_anonymous", False)
    explicit_seg = str(ctx.get("segment", "")).upper()
    if explicit_seg in ("0", "ANONYMOUS") or is_anonymous:
        segment = "ANONYMOUS"
    else:
        segment = "COLDSTART"

    # Build default feature vector for LGBM (COLDSTART only)
    coldstart_predictions: Dict[str, Any] = {}
    if segment == "COLDSTART" and models:
        # Determine number of expected features from first model
        first_task = next(iter(models))
        booster = models[first_task]
        n_features = booster.num_feature()

        # All-zero vector with cold-start overrides
        default_vec = [0.0] * n_features

        # Map well-known feature names to indices if feat_meta is available
        for task_name, model in models.items():
            sel = feat_meta.get(task_name, {})
            indices = sel.get("indices", [])
            names = sel.get("names", [])

            if indices:
                # Build per-task default vector
                task_vec = [0.0] * len(indices)
                # Apply cold-start overrides by name
                cs_overrides = {
                    "is_coldstart": 1.0,
                    "customer_segment": 1.0,
                    "coldstart_confidence": 1.0,
                }
                for feat_idx, feat_name in enumerate(names):
                    if feat_name in cs_overrides:
                        task_vec[feat_idx] = cs_overrides[feat_name]
                X = np.array(task_vec, dtype=np.float32).reshape(1, -1)
            else:
                X = np.array(default_vec, dtype=np.float32).reshape(1, -1)

            try:
                raw = model.predict(X)
                task_type = _get_task_type(task_name, tasks_meta)
                coldstart_predictions[task_name] = _normalise(raw[0], task_type)
            except Exception as e:
                logger.debug("Cold-start inference failed for task=%s: %s", task_name, e)

    # Load popularity catalog from Lambda environment variable (pre-loaded or S3)
    catalog = _CACHE.get("popularity_catalog")
    if catalog is None:
        catalog = _load_popularity_catalog()
        _CACHE["popularity_catalog"] = catalog

    recommendations = []
    for i, item in enumerate(catalog[:20], start=1):
        recommendations.append({
            "item_id": item.get("item_id", ""),
            "rank": i,
            "score": float(item.get("score", 0.0)),
            "score_components": {"popularity": float(item.get("score", 0.0))},
            "reasons": [],
            "metadata": {
                "is_coldstart_candidate": True,
                "benefit_type": item.get("benefit_type", ""),
            },
        })

    logger.info(
        "Cold-start response: user_id=%s, segment=%s, candidates=%d",
        user_id, segment, len(recommendations),
    )

    return {
        "user_id": user_id,
        "version": version,
        "segment": segment,
        "is_coldstart": True,
        "predictions": coldstart_predictions,
        "recommendations": recommendations,
        "elapsed_ms": elapsed_ms,
        "metadata": {"coldstart_path": True},
    }


def _load_popularity_catalog() -> List[Dict[str, Any]]:
    """Load popularity catalog from S3 or return empty list."""
    catalog_uri = os.environ.get("POPULARITY_CATALOG_URI", "")
    if not catalog_uri:
        return []
    try:
        import boto3, json
        s3 = boto3.client("s3", region_name=REGION)
        bucket, key = _parse_s3_uri(catalog_uri)
        obj = s3.get_object(Bucket=bucket, Key=key)
        data = json.loads(obj["Body"].read())
        return data if isinstance(data, list) else data.get("items", [])
    except Exception as e:
        logger.warning("Failed to load popularity catalog: %s", e)
        return []


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _parse_s3_uri(uri: str):
    """Parse 's3://bucket/key' into (bucket, key)."""
    path = uri.replace("s3://", "")
    parts = path.split("/", 1)
    return parts[0], (parts[1] if len(parts) > 1 else "")
