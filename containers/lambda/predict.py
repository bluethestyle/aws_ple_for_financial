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
# Module-level cache (persists across warm invocations)
# ---------------------------------------------------------------------------

_CACHE: Dict[str, Any] = {
    "version": None,
    "models": {},           # task_name -> lgb.Booster
    "features": {},         # task_name -> {"indices": [...], "names": [...]}
    "tasks_meta": [],       # list of {"name": str, "type": str}
}

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

    user_id: str = event.get("user_id", "")
    features: Dict[str, float] = event.get("features", {})
    ctx: Dict[str, Any] = event.get("context", {})
    requested_tasks: Optional[List[str]] = event.get("tasks")

    if not user_id:
        return {"error": "user_id required", "status": 400}

    import boto3
    s3 = boto3.client("s3", region_name=REGION)

    # --- Ensure models are loaded (cold start or version change) ---
    _ensure_loaded(s3)

    version = _CACHE["version"]
    models = _CACHE["models"]
    feat_meta = _CACHE["features"]
    tasks_meta = _CACHE["tasks_meta"]

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
    raw_feature_list = list(features.values())

    for task_name in tasks_to_score:
        booster = models[task_name]
        sel = feat_meta.get(task_name, {})
        indices = sel.get("indices", [])

        if indices:
            # Slice the full feature vector to selected indices
            try:
                selected = [raw_feature_list[i] for i in indices]
            except IndexError:
                logger.warning(
                    "Feature index out of range for task=%s, "
                    "feature_count=%d, max_index=%d",
                    task_name, len(raw_feature_list),
                    max(indices) if indices else -1,
                )
                selected = raw_feature_list
        else:
            selected = raw_feature_list

        import numpy as np
        X = np.array(selected, dtype=np.float32).reshape(1, -1)
        raw = booster.predict(X)  # shape: (1,) binary or (1, n_classes)

        # Normalise
        task_type = _get_task_type(task_name, tasks_meta)
        predictions[task_name] = _normalise(raw[0], task_type)

    elapsed = round((time.perf_counter() - t0) * 1000.0, 2)

    result = {
        "user_id": user_id,
        "version": version,
        "predictions": predictions,
        "elapsed_ms": elapsed,
    }

    # --- Async performance logging (best-effort) ---
    try:
        _log_prediction(boto3, user_id, version, predictions, elapsed, ctx)
        _emit_metrics(boto3, version, predictions, elapsed, tasks_meta)
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

        logger.info(
            "Loaded %d student models for version %s",
            len(models), active_version,
        )

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
# Output normalisation
# ---------------------------------------------------------------------------

def _normalise(raw: Any, task_type: str) -> Any:
    """Normalise raw LGBM output based on task type."""
    import numpy as np

    if task_type == "binary":
        # LGBM binary returns P(positive class)
        val = float(np.asarray(raw).ravel()[0]) if hasattr(raw, "__len__") else float(raw)
        # LGBM predict returns probability directly (not logit) by default
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
) -> None:
    """Write prediction record to DynamoDB for Champion-Challenger analysis."""
    import uuid
    from datetime import datetime, timezone

    dynamodb = boto3.resource("dynamodb", region_name=REGION)
    table = dynamodb.Table(MONITOR_TABLE)

    record = {
        "prediction_id": str(uuid.uuid4()),
        "user_id": user_id,
        "version": version,
        "predictions": {k: str(v) for k, v in predictions.items()},
        "elapsed_ms": str(round(elapsed_ms, 2)),
        "channel": ctx.get("channel", "unknown"),
        "segment": ctx.get("segment", "unknown"),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "ttl": int(time.time()) + 90 * 86400,  # 90-day TTL
    }

    table.put_item(Item=record)


def _emit_metrics(
    boto3,
    version: str,
    predictions: Dict[str, Any],
    elapsed_ms: float,
    tasks_meta: List[Dict[str, str]],
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
