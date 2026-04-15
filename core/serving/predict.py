"""
Recommendation Service
======================

Unified inference logic shared identically between the Lambda handler and
the ECS FastAPI application.  No framework-specific code lives here --
only pure business logic.

Flow::

    request
      --> kill switch check
      --> A/B variant selection (if enabled)
      --> feature store lookup
      --> context enrichment
      --> LGBM multi-task inference
      --> output normalisation (per task type)
      --> optional RecommendationPipeline (scoring + filtering + reasons)
      --> structured response

All external dependencies (feature store, model, kill switch, A/B manager)
are injected at construction time so the service is trivially testable.
"""

from __future__ import annotations

import logging
import math
import time
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)

__all__ = ["RecommendationService", "PredictionResponse", "OutputNormalizer"]


# ---------------------------------------------------------------------------
# Response data classes
# ---------------------------------------------------------------------------

@dataclass
class TaskPrediction:
    """Prediction for a single task.

    Attributes:
        task_name: Name of the prediction task (e.g. ``"ctr"``).
        task_type: One of ``binary``, ``multiclass``, ``regression``.
        raw_value: Raw model output (logit or raw score).
        normalised_value: After sigmoid / softmax / identity.
        layer_used: Which fallback layer produced the prediction
            (1=distilled LGBM, 2=direct LGBM, 3=rule-based).
            ``None`` when no FallbackRouter is configured.
        calibrated: Whether Platt / isotonic calibration was applied.
    """

    task_name: str
    task_type: str
    raw_value: Any
    normalised_value: Any
    layer_used: Optional[int] = None
    calibrated: bool = False


@dataclass
class PredictionResponse:
    """Structured response from :meth:`RecommendationService.predict`.

    Attributes:
        user_id: The user for whom predictions were made.
        predictions: Per-task normalised predictions dict.
        recommendations: Ordered recommendation items (if pipeline enabled).
        variant: A/B variant name (if A/B testing enabled).
        kill_switch_active: Whether the kill switch blocked inference.
        fallback_used: Whether a fallback response was returned.
        elapsed_ms: Total wall-clock time for the request (ms).
        metadata: Extra metadata.
    """

    user_id: str
    predictions: Dict[str, Any] = field(default_factory=dict)
    recommendations: List[Dict[str, Any]] = field(default_factory=list)
    variant: str = ""
    kill_switch_active: bool = False
    fallback_used: bool = False
    elapsed_ms: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Serialise to a JSON-compatible dict."""
        return asdict(self)


# ---------------------------------------------------------------------------
# Output normalisation
# ---------------------------------------------------------------------------

class OutputNormalizer:
    """Normalise raw model outputs based on task type.

    * ``binary``      -- sigmoid activation.
    * ``multiclass``  -- softmax activation.
    * ``regression``  -- identity (raw passthrough).
    * ``ranking``     -- identity.
    """

    @staticmethod
    def normalise(
        raw: Any,
        task_type: str,
    ) -> Any:
        """Apply the correct activation for *task_type*.

        Args:
            raw: Raw model output.  For binary this is a scalar or 1-D
                array.  For multiclass this is a 1-D array of logits /
                probabilities.
            task_type: One of ``binary``, ``multiclass``, ``regression``,
                ``ranking``.

        Returns:
            Normalised value (float, list of floats, or passthrough).
        """
        if task_type == "binary":
            return OutputNormalizer._sigmoid(raw)
        elif task_type == "multiclass":
            return OutputNormalizer._softmax(raw)
        else:
            # regression / ranking -- pass through
            return OutputNormalizer._to_python(raw)

    # -- activations -------------------------------------------------------

    @staticmethod
    def _sigmoid(x: Any) -> float:
        """Numerically stable sigmoid."""
        if isinstance(x, (np.ndarray, list)):
            # For binary classifiers that return [p_neg, p_pos],
            # take the positive-class probability.
            arr = np.asarray(x).ravel()
            if arr.shape[0] == 2:
                return float(arr[1])
            x = float(arr[0])
        x = float(x)
        # Clamp to avoid overflow
        x = max(-500.0, min(500.0, x))
        if x >= 0:
            return 1.0 / (1.0 + math.exp(-x))
        else:
            exp_x = math.exp(x)
            return exp_x / (1.0 + exp_x)

    @staticmethod
    def _softmax(x: Any) -> List[float]:
        """Stable softmax over a 1-D array."""
        arr = np.asarray(x, dtype=np.float64).ravel()
        # If already probabilities (sums to ~1), return as-is
        if arr.sum() > 0 and abs(arr.sum() - 1.0) < 1e-3 and np.all(arr >= 0):
            return arr.tolist()
        shifted = arr - arr.max()
        exps = np.exp(shifted)
        return (exps / exps.sum()).tolist()

    @staticmethod
    def _to_python(x: Any) -> Any:
        """Convert numpy types to native Python for JSON serialisation."""
        if isinstance(x, np.ndarray):
            return x.tolist()
        if isinstance(x, (np.floating, np.integer)):
            return x.item()
        return x


# ---------------------------------------------------------------------------
# Main service
# ---------------------------------------------------------------------------

class RecommendationService:
    """Unified inference entry point for Lambda and ECS.

    This class is instantiated once (at cold start / startup) and
    reused across requests.  It holds references to the model,
    feature store, kill switch, and optional A/B test manager.

    Args:
        model: A loaded :class:`~core.model.lgbm.LGBMModel` instance.
        feature_store: An :class:`~core.serving.feature_store.AbstractFeatureStore`.
        tasks_meta: List of task metadata dicts (``name``, ``type``).
        kill_switch: Optional :class:`~core.serving.kill_switch.KillSwitch`.
        ab_manager: Optional :class:`~core.serving.ab_test.ABTestManager`.
        pipeline: Optional :class:`~core.recommendation.pipeline.RecommendationPipeline`.
        pipeline_config: Config dict for the pipeline (used when
            ``pipeline`` is ``None`` but pipeline is enabled).
        cold_start_handler: Optional :class:`~core.serving.cold_start.ColdStartHandler`.
            When provided, users with no feature store entry are routed to the
            cold-start path (popularity candidates + COLDSTART reasons) instead
            of receiving an empty error response.
        consent_manager: Optional :class:`~core.compliance.consent_manager.ConsentManager`.
            When provided, the ``predict()`` method checks channel-level
            consent before running inference.  If the user has not granted
            consent for the requested channel, a blocked response is returned
            without generating recommendations.
    """

    def __init__(
        self,
        model: Any,
        feature_store: "AbstractFeatureStore",
        tasks_meta: List[Dict[str, str]],
        kill_switch: Optional["KillSwitch"] = None,
        ab_manager: Optional["ABTestManager"] = None,
        pipeline: Optional[Any] = None,
        pipeline_config: Optional[Dict[str, Any]] = None,
        cold_start_handler: Optional[Any] = None,
        variant_models: Optional[Dict[str, Any]] = None,
        consent_manager: Optional[Any] = None,
        fallback_router: Optional[Any] = None,
        rule_engine: Optional[Any] = None,
        calibrators: Optional[Dict[str, Any]] = None,
    ) -> None:
        from .feature_store import AbstractFeatureStore

        self._model = model
        self._feature_store: AbstractFeatureStore = feature_store
        self._tasks_meta = tasks_meta
        self._kill_switch = kill_switch
        self._ab_manager = ab_manager
        self._pipeline = pipeline
        self._pipeline_config = pipeline_config or {}
        self._cold_start_handler = cold_start_handler
        self._consent_manager = consent_manager

        # --- 3-layer fallback components (all optional) ---
        # When fallback_router is None the service behaves exactly as before
        # (pure LGBM inference, no routing, no calibration).
        self._fallback_router = fallback_router
        self._rule_engine = rule_engine
        # {task_name: calibrator} — sklearn-compatible objects with .predict_proba()
        self._calibrators: Dict[str, Any] = calibrators or {}

        # Variant-specific models for A/B testing.
        # Key: variant name (e.g. "control", "challenger").
        # Value: model instance with a .predict() interface.
        # When a variant has no entry here, falls back to self._model.
        self._variant_models: Dict[str, Any] = variant_models or {}

        # Build task name -> type mapping for normalisation
        self._task_type_map: Dict[str, str] = {
            t["name"]: t["type"] for t in tasks_meta
        }

        # Feature names for cold-start default vector synthesis
        self._coldstart_features: Optional[List[str]] = None

        logger.info(
            "RecommendationService initialised: tasks=%s, "
            "kill_switch=%s, ab_test=%s (%d variant models), "
            "pipeline=%s, cold_start=%s, consent_manager=%s, "
            "fallback_router=%s, rule_engine=%s, calibrators=%d",
            [t["name"] for t in tasks_meta],
            kill_switch is not None,
            ab_manager is not None,
            len(self._variant_models),
            pipeline is not None,
            cold_start_handler is not None,
            consent_manager is not None,
            fallback_router is not None,
            rule_engine is not None,
            len(self._calibrators),
        )

    def register_variant_model(
        self, variant_name: str, model: Any,
    ) -> None:
        """Register a model for a specific A/B variant at runtime.

        Allows dynamic model loading when a new challenger is deployed
        without restarting the service.

        Args:
            variant_name: Variant name matching :class:`ABVariant.name`.
            model: Model instance with a ``.predict()`` method.
        """
        self._variant_models[variant_name] = model
        logger.info(
            "Registered variant model: %s (total variants: %d)",
            variant_name, len(self._variant_models),
        )

    # ------------------------------------------------------------------
    # Single-user inference
    # ------------------------------------------------------------------

    def predict(
        self,
        user_id: str,
        context: Optional[Dict[str, Any]] = None,
    ) -> PredictionResponse:
        """Run inference for a single user.

        Args:
            user_id: Unique user identifier.
            context: Optional extra context (channel, segment, etc.).

        Returns:
            :class:`PredictionResponse` with normalised predictions and
            optional recommendations.
        """
        t0 = time.perf_counter()
        ctx = context or {}
        metadata: Dict[str, Any] = {}

        # ---- 1. Kill switch ----
        if self._kill_switch is not None:
            ks_state = self._kill_switch.check(
                task=ctx.get("task"),
                cluster=ctx.get("cluster"),
            )
            if ks_state.active:
                return self._fallback_response(
                    user_id, ks_state, t0,
                )

        # ---- 1b. Consent check ----
        if self._consent_manager is not None:
            channel = ctx.get("channel")
            if channel:
                contactable, reason = self._consent_manager.is_contactable(
                    user_id, channel,
                )
                if not contactable:
                    logger.info(
                        "Consent blocked: user_id=%s, channel=%s, reason=%s",
                        user_id, channel, reason,
                    )
                    return PredictionResponse(
                        user_id=user_id,
                        elapsed_ms=self._elapsed(t0),
                        metadata={"blocked": "consent_not_granted", "reason": reason},
                    )

        # ---- 2. A/B variant selection ----
        variant_name = ""
        active_model = self._model  # default: champion
        if self._ab_manager is not None:
            assignment = self._ab_manager.assign(user_id)
            variant_name = assignment.variant_name
            metadata["ab_variant"] = variant_name
            metadata["ab_hash"] = assignment.hash_value

            # Select the variant-specific model if registered
            if variant_name in self._variant_models:
                active_model = self._variant_models[variant_name]
                metadata["ab_model_source"] = "variant"
            else:
                metadata["ab_model_source"] = "default"

        # ---- 3. Feature lookup (or cold-start synthesis) ----
        is_coldstart = False
        features = self._feature_store.get(user_id)

        if features is None and self._cold_start_handler is not None:
            # Synthesise a default feature vector — the rest of the
            # pipeline runs identically to the normal path.
            handler = self._cold_start_handler
            segment = handler.classify(user_id=user_id, context=ctx)
            ctx["segment"] = segment
            is_coldstart = True

            features = handler.default_features(self._coldstart_features)
            metadata["coldstart_path"] = True
            metadata["segment"] = segment

            logger.info(
                "Cold-start: user_id=%s, segment=%s (synthesised features)",
                user_id, segment,
            )

        if features is None:
            logger.warning(
                "RecommendationService: no features for user_id=%s "
                "(cold_start_handler not configured)",
                user_id,
            )
            return PredictionResponse(
                user_id=user_id,
                variant=variant_name,
                elapsed_ms=self._elapsed(t0),
                metadata={"error": "user_not_found", **metadata},
            )

        # ---- 4. Context enrichment ----
        enriched_features = {**features, **ctx}

        # ---- 5. LGBM inference (using variant-specific model) ----
        import pandas as pd

        feature_df = pd.DataFrame([enriched_features])
        raw_predictions = active_model.predict(feature_df)

        # ---- 6. Output normalisation ----
        normalised: Dict[str, Any] = {}
        layer_used_map: Dict[str, Optional[int]] = {}

        for task_name, raw in raw_predictions.items():
            task_type = self._task_type_map.get(task_name, "regression")
            raw_val = raw[0] if isinstance(raw, np.ndarray) and raw.ndim >= 1 else raw
            normalised[task_name] = OutputNormalizer.normalise(raw_val, task_type)
            layer_used_map[task_name] = None  # unknown until router decides

        # ---- 6b. 3-layer fallback routing ----
        # When no FallbackRouter is configured, all tasks stay on LGBM
        # (backward-compatible path).
        if self._fallback_router is not None:
            routing = self._fallback_router.route_all(
                task_names=list(raw_predictions.keys()),
                lgbm_models={t: active_model for t in raw_predictions},
                rule_engine=self._rule_engine,
            )

            for task_name, layer in routing.items():
                layer_used_map[task_name] = layer
                if layer == 3 and self._rule_engine is not None:
                    # Replace LGBM output with rule-based prediction
                    try:
                        rule_result = self._rule_engine.predict(
                            features=enriched_features,
                            task_name=task_name,
                        )
                        normalised[task_name] = rule_result["prediction"]
                        # Carry rule metadata into the per-task entry
                        metadata.setdefault("rule_reasons", {})[task_name] = (
                            rule_result.get("reason", "")
                        )
                        logger.debug(
                            "FallbackRouter: task=%s → Layer 3 (rule), "
                            "rule=%s, confidence=%.3f",
                            task_name,
                            rule_result.get("rule_name", ""),
                            rule_result.get("confidence", 0.0),
                        )
                    except Exception:
                        logger.warning(
                            "Rule engine failed for task=%s, keeping LGBM output",
                            task_name,
                            exc_info=True,
                        )

            # Summarise layer distribution for observability
            from collections import Counter
            layer_counts = Counter(routing.values())
            metadata["fallback_layers"] = dict(layer_counts)
            metadata["layer_used_per_task"] = layer_used_map

        # ---- 6c. Calibration ----
        # Apply per-task Platt / isotonic calibrator when available.
        # Calibration is applied silently — caller sees the same response shape.
        calibrated_tasks: List[str] = []
        if self._calibrators:
            for task_name, cal_value in list(normalised.items()):
                calibrator = self._calibrators.get(task_name)
                if calibrator is None:
                    continue
                try:
                    task_type = self._task_type_map.get(task_name, "regression")
                    if task_type == "binary":
                        x = np.array([[float(cal_value)]])
                        prob = calibrator.predict_proba(x)[0, 1]
                        normalised[task_name] = round(float(prob), 6)
                    elif task_type == "multiclass":
                        x = np.array([cal_value])
                        prob = calibrator.predict_proba(x)[0].tolist()
                        normalised[task_name] = [round(float(p), 6) for p in prob]
                    # regression calibration not applied (no standard form)
                    calibrated_tasks.append(task_name)
                    logger.debug("Calibrated task=%s", task_name)
                except Exception:
                    logger.warning(
                        "Calibration failed for task=%s, keeping raw value",
                        task_name,
                        exc_info=True,
                    )
            if calibrated_tasks:
                metadata["calibrated_tasks"] = calibrated_tasks

        # For cold-start, overlay strategy-based defaults for tasks where
        # LGBM output with a zero-filled vector is unreliable.
        if is_coldstart and self._cold_start_handler is not None:
            handler = self._cold_start_handler
            for task_name in list(normalised.keys()):
                if not handler.should_use_lgbm(task_name):
                    strategy_default = handler.registry.get(task_name).default_prediction(ctx)
                    if strategy_default is not None:
                        normalised[task_name] = strategy_default

        # ---- 7. Optional pipeline (scoring + reasons) ----
        recommendations: List[Dict[str, Any]] = []
        if self._pipeline is not None:
            recommendations = self._run_pipeline(
                user_id, normalised, ctx,
            )

        # ---- 8. A/B metric recording ----
        elapsed = self._elapsed(t0)
        if self._ab_manager is not None and variant_name:
            self._ab_manager.record_latency(variant_name, elapsed)

        return PredictionResponse(
            user_id=user_id,
            predictions=normalised,
            recommendations=recommendations,
            variant=variant_name,
            kill_switch_active=False,
            fallback_used=is_coldstart,
            elapsed_ms=elapsed,
            metadata=metadata,
        )

    # ------------------------------------------------------------------
    # Batch inference
    # ------------------------------------------------------------------

    def predict_batch(
        self,
        user_ids: List[str],
        context: Optional[Dict[str, Any]] = None,
    ) -> List[PredictionResponse]:
        """Run inference for multiple users.

        Currently iterates over :meth:`predict` per user.  For very
        large batches consider a dedicated batch-optimised path that
        calls ``model.predict`` once with a stacked DataFrame.

        Args:
            user_ids: List of user identifiers.
            context: Shared context applied to all users.

        Returns:
            List of :class:`PredictionResponse`, one per user.
        """
        results: List[PredictionResponse] = []
        for uid in user_ids:
            results.append(self.predict(uid, context))

        logger.info(
            "predict_batch: processed %d users", len(results),
        )
        return results

    # ------------------------------------------------------------------
    # Pipeline integration
    # ------------------------------------------------------------------

    def _run_pipeline(
        self,
        user_id: str,
        predictions: Dict[str, Any],
        context: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        """Feed normalised predictions into the RecommendationPipeline.

        The pipeline expects ``candidate_items`` with ``predictions``.
        Since we are running per-user single-item inference, we wrap the
        predictions as a single candidate.  Downstream the pipeline handles
        scoring, filtering, and reason generation.
        """
        try:
            candidate = {
                "item_id": context.get("item_id", "default"),
                "predictions": predictions,
                "item_info": context.get("item_info", {}),
                "ig_top_features": context.get("ig_top_features", []),
            }

            result = self._pipeline.recommend(
                customer_id=user_id,
                candidate_items=[candidate],
                customer_context=context,
            )

            return [
                {
                    "item_id": item.item_id,
                    "rank": item.rank,
                    "score": item.score,
                    "score_components": item.score_components,
                    "reasons": item.reasons,
                    "metadata": item.metadata,
                }
                for item in result.items
            ]
        except Exception:
            logger.exception(
                "RecommendationService: pipeline failed for user_id=%s",
                user_id,
            )
            return []

    # ------------------------------------------------------------------
    # Fallback
    # ------------------------------------------------------------------

    def _fallback_response(
        self,
        user_id: str,
        ks_state: "KillSwitchState",
        t0: float,
    ) -> PredictionResponse:
        """Build a response when the kill switch is active."""
        from .kill_switch import FallbackStrategy

        metadata: Dict[str, Any] = {
            "kill_switch_scope": ks_state.scope,
            "kill_switch_reason": ks_state.reason,
            "fallback_strategy": ks_state.fallback_strategy.value,
        }

        logger.warning(
            "RecommendationService: kill switch active for user_id=%s, "
            "scope=%s, fallback=%s",
            user_id, ks_state.scope, ks_state.fallback_strategy.value,
        )

        return PredictionResponse(
            user_id=user_id,
            kill_switch_active=True,
            fallback_used=True,
            elapsed_ms=self._elapsed(t0),
            metadata=metadata,
        )

    # ------------------------------------------------------------------
    # Calibration
    # ------------------------------------------------------------------

    @staticmethod
    def _load_calibrators(model_dir: str) -> Dict[str, Any]:
        """Load ``calibrator.joblib`` files from per-task subdirectories.

        Expects the layout::

            {model_dir}/
              {task_name}/
                calibrator.joblib

        Tasks without a ``calibrator.joblib`` are silently skipped.

        Args:
            model_dir: Local directory containing per-task subdirectories.

        Returns:
            ``{task_name: calibrator_object}`` — only tasks that have a
            calibrator file are included.
        """
        import os
        try:
            import joblib
        except ImportError:
            logger.warning(
                "_load_calibrators: joblib not installed, skipping calibration"
            )
            return {}

        calibrators: Dict[str, Any] = {}
        if not os.path.isdir(model_dir):
            logger.warning(
                "_load_calibrators: model_dir not found: %s", model_dir
            )
            return calibrators

        for task_name in os.listdir(model_dir):
            task_dir = os.path.join(model_dir, task_name)
            cal_path = os.path.join(task_dir, "calibrator.joblib")
            if not os.path.isfile(cal_path):
                continue
            try:
                calibrators[task_name] = joblib.load(cal_path)
                logger.info(
                    "_load_calibrators: loaded calibrator for task=%s", task_name
                )
            except Exception:
                logger.warning(
                    "_load_calibrators: failed to load calibrator for task=%s",
                    task_name,
                    exc_info=True,
                )

        logger.info(
            "_load_calibrators: loaded %d calibrators from %s",
            len(calibrators), model_dir,
        )
        return calibrators

    # ------------------------------------------------------------------
    # Health
    # ------------------------------------------------------------------

    def health_check(self) -> Dict[str, Any]:
        """Return health status of all sub-components."""
        fs_health = self._feature_store.health_check()
        return {
            "healthy": fs_health.get("healthy", False),
            "feature_store": fs_health,
            "model_loaded": self._model is not None,
            "tasks": [t["name"] for t in self._tasks_meta],
            "kill_switch_enabled": self._kill_switch is not None,
            "ab_test_enabled": self._ab_manager is not None,
            "pipeline_enabled": self._pipeline is not None,
            "fallback_router_enabled": self._fallback_router is not None,
            "rule_engine_enabled": self._rule_engine is not None,
            "calibrators_loaded": list(self._calibrators.keys()),
        }

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _elapsed(t0: float) -> float:
        return round((time.perf_counter() - t0) * 1000.0, 2)
