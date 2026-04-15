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
        regulatory_checker: Optional
            :class:`~core.compliance.regulatory_checker.RegulatoryComplianceChecker`.
            When provided, a suitability check (금소법 §17 / KFS-002) is run
            before each prediction.  On failure the response is returned with
            ``compliance_blocked=True`` in metadata; the check is non-blocking
            when the compliance service is unavailable.
        ai_opt_out: Optional :class:`~core.compliance.ai_opt_out.AIDecisionOptOut`.
            When provided, customers who have opted out of AI decisions receive
            a blocked response without AI inference (AI기본법 제31조 / GDPR Art. 22).
        compliance_audit_store: Optional
            :class:`~core.compliance.audit_store.ComplianceAuditStore`.
            When provided, every recommendation decision is logged to the
            compliance audit trail (post-prediction).
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
        regulatory_checker: Optional[Any] = None,
        ai_opt_out: Optional[Any] = None,
        compliance_audit_store: Optional[Any] = None,
        change_detector: Optional[Any] = None,
        audit_logger: Optional[Any] = None,
        fairness_monitor: Optional[Any] = None,
        prompt_sanitizer: Optional[Any] = None,
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

        # --- Compliance components (all optional, non-blocking) ---
        self._regulatory_checker = regulatory_checker
        self._ai_opt_out = ai_opt_out
        self._compliance_audit_store = compliance_audit_store

        # --- Agent infrastructure (optional, backward-compatible) ---
        # ChangeDetector receives serving-stage events (prediction audit trail).
        self._change_detector = change_detector

        # --- Monitoring components (all optional) ---
        # AuditLogger: HMAC-signed immutable audit trail for every inference.
        self._audit_logger = audit_logger
        # FairnessMonitor: post-batch bias detection across protected attributes.
        self._fairness_monitor = fairness_monitor

        # --- Security: PromptSanitizer (core.security) ---
        # Classifies and scrubs prompts before they reach any LLM provider.
        # When None, no sanitization is performed (backward-compatible).
        self._prompt_sanitizer = prompt_sanitizer

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
            "fallback_router=%s, rule_engine=%s, calibrators=%d, "
            "regulatory_checker=%s, ai_opt_out=%s, compliance_audit_store=%s, "
            "change_detector=%s, audit_logger=%s, fairness_monitor=%s",
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
            regulatory_checker is not None,
            ai_opt_out is not None,
            compliance_audit_store is not None,
            change_detector is not None,
            audit_logger is not None,
            fairness_monitor is not None,
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

        # ---- 1c. AI opt-out check (AI기본법 제31조 / GDPR Art. 22) ----
        # Non-blocking: if the opt-out service is down we log a warning and
        # continue so that customers are never blocked by a service failure.
        if self._ai_opt_out is not None:
            try:
                if self._ai_opt_out.is_opted_out(user_id):
                    fallback_type = self._ai_opt_out.get_fallback_type(user_id)
                    logger.info(
                        "AI opt-out active: user_id=%s, fallback=%s",
                        user_id, fallback_type,
                    )
                    return PredictionResponse(
                        user_id=user_id,
                        elapsed_ms=self._elapsed(t0),
                        fallback_used=True,
                        metadata={
                            "blocked": "ai_opt_out",
                            "fallback_type": fallback_type,
                        },
                    )
            except Exception:
                logger.warning(
                    "AI opt-out check failed for user_id=%s; continuing (non-blocking)",
                    user_id,
                    exc_info=True,
                )

        # ---- 1d. Regulatory suitability check (금소법 §17 / KFS-002) ----
        # Runs the suitability category check before inference.  Failures are
        # non-blocking — the recommendation proceeds but metadata is annotated.
        if self._regulatory_checker is not None:
            try:
                suitability_results = self._regulatory_checker.run_category("suitability")
                critical_failures = [
                    r for r in suitability_results
                    if not r.passed and r.item.severity == "critical"
                ]
                if critical_failures:
                    logger.warning(
                        "Regulatory suitability check FAILED for user_id=%s: %s",
                        user_id,
                        [r.item.id for r in critical_failures],
                    )
                    metadata["compliance_suitability_failed"] = [
                        r.item.id for r in critical_failures
                    ]
                else:
                    metadata["compliance_suitability_passed"] = True
            except Exception:
                logger.warning(
                    "Regulatory suitability check raised exception for user_id=%s; "
                    "continuing (non-blocking)",
                    user_id,
                    exc_info=True,
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
                        # contributing_features for reason pipeline
                        # Convert [{name, value, role}] → [(name, value)]
                        # for compatibility with generate_l1 / InterpretationRegistry
                        cf = rule_result.get("contributing_features", [])
                        if cf:
                            metadata.setdefault("contributing_features", {})[task_name] = [
                                (f["name"], f.get("value", 0.0)) for f in cf
                            ]
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
        # Inject contributing_features from Layer 3 rules into context
        # so RecommendationPipeline can use them as ig_top_features
        # for reason generation (bypasses Agent 1 / IG computation).
        layer3_features = metadata.get("contributing_features", {})
        if layer3_features:
            # Merge all Layer 3 task features into a single list for the pipeline
            all_cf = []
            for task_cf in layer3_features.values():
                all_cf.extend(task_cf)
            # Deduplicate by feature name, keep highest value
            seen = {}
            for name, value in all_cf:
                if name not in seen or value > seen[name]:
                    seen[name] = value
            ctx["ig_top_features"] = sorted(seen.items(), key=lambda x: -x[1])

        recommendations: List[Dict[str, Any]] = []
        if self._pipeline is not None:
            recommendations = self._run_pipeline(
                user_id, normalised, ctx,
            )

        # ---- 8. A/B metric recording ----
        elapsed = self._elapsed(t0)
        if self._ab_manager is not None and variant_name:
            self._ab_manager.record_latency(variant_name, elapsed)

        # ---- 9. Agent audit event (optional, non-blocking) ----
        # Fire a serving-stage change event so OpsAgent CP6 can track
        # prediction volume, latency, and layer distribution.
        if self._change_detector is not None:
            try:
                self._change_detector.on_pipeline_stage_complete(
                    stage="stage_serving",
                    artifacts={
                        "user_id": user_id,
                        "tasks": list(normalised.keys()),
                        "layer_used_per_task": layer_used_map,
                        "elapsed_ms": elapsed,
                        "variant": variant_name,
                        "is_coldstart": is_coldstart,
                        "recommendations_count": len(recommendations),
                    },
                )
            except Exception:
                logger.debug(
                    "ChangeDetector audit event failed (non-fatal)",
                    exc_info=True,
                )

        response = PredictionResponse(
            user_id=user_id,
            predictions=normalised,
            recommendations=recommendations,
            variant=variant_name,
            kill_switch_active=False,
            fallback_used=is_coldstart,
            elapsed_ms=elapsed,
            metadata=metadata,
        )

        # ---- 10. Compliance audit trail (post-prediction) ----
        # Every recommendation decision is logged for regulatory audit.
        # Non-blocking: a DynamoDB write failure must not affect the user response.
        if self._compliance_audit_store is not None:
            try:
                self._compliance_audit_store.log_event("embedding", {
                    "pk": f"recommendation#{user_id}",
                    "user_id": user_id,
                    "tasks_predicted": list(normalised.keys()),
                    "items_returned": len(recommendations),
                    "variant": variant_name,
                    "fallback_used": is_coldstart,
                    "elapsed_ms": elapsed,
                    "compliance_suitability_passed": metadata.get(
                        "compliance_suitability_passed", False
                    ),
                    "compliance_suitability_failed": metadata.get(
                        "compliance_suitability_failed", []
                    ),
                })
            except Exception:
                logger.warning(
                    "Compliance audit log failed for user_id=%s; "
                    "continuing (non-blocking)",
                    user_id,
                    exc_info=True,
                )

        # ---- 11. AuditLogger — model inference record (non-blocking) ----
        # Records model_id, input/output dims, and latency for HMAC-signed
        # immutable audit trail.  Gated by self._audit_logger being set.
        if self._audit_logger is not None:
            try:
                self._audit_logger.log_model_inference(
                    model_id=variant_name or "champion",
                    input_dim=len(enriched_features),
                    output_dim=len(normalised),
                    latency_ms=elapsed,
                    status="SUCCESS",
                    user="system",
                )
            except Exception:
                logger.debug(
                    "AuditLogger.log_model_inference failed (non-fatal)",
                    exc_info=True,
                )

        return response

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

        # ---- Fairness monitoring (non-blocking, batch-level) ----
        # Requires each response's metadata to carry protected-attribute fields
        # (e.g. "age_group", "gender") so FairnessMonitor can compute DI/SPD/EOD.
        # If the metadata fields are absent the monitor is a no-op.
        if self._fairness_monitor is not None and results:
            try:
                # Build recommendation list expected by FairnessMonitor:
                # [{attribute_key: value, "recommended": bool, ...}, ...]
                _fairness_recs: List[Dict[str, Any]] = []
                for _res in results:
                    _rec_entry: Dict[str, Any] = {
                        "recommended": len(_res.recommendations) > 0,
                    }
                    # Carry protected attributes from context if present
                    _ctx = context or {}
                    for _attr in self._fairness_monitor.protected_attributes:
                        if _attr in _ctx:
                            _rec_entry[_attr] = _ctx[_attr]
                        elif _attr in _res.metadata:
                            _rec_entry[_attr] = _res.metadata[_attr]
                    _fairness_recs.append(_rec_entry)

                # Only run if at least one protected attribute column is populated
                _populated_attrs = [
                    attr for attr in self._fairness_monitor.protected_attributes
                    if any(attr in r for r in _fairness_recs)
                ]
                if _populated_attrs:
                    # Evaluate fairness for each populated attribute using
                    # automatic group-pair detection (distinct values in batch).
                    _group_pairs_by_attr: Dict[str, List] = {}
                    for _attr in _populated_attrs:
                        _vals = list({r[_attr] for r in _fairness_recs if _attr in r})
                        if len(_vals) >= 2:
                            _group_pairs_by_attr[_attr] = [
                                (_vals[0], _vals[i]) for i in range(1, len(_vals))
                            ]
                    if _group_pairs_by_attr:
                        _fairness_results = self._fairness_monitor.evaluate_all_attributes(
                            recommendations=_fairness_recs,
                            group_pairs_by_attribute=_group_pairs_by_attr,
                        )
                        _violations = sum(
                            len(m.violations)
                            for m in _fairness_results.values()
                        )
                        if _violations:
                            logger.warning(
                                "FairnessMonitor: %d violation(s) detected "
                                "across %d attribute(s) in batch of %d",
                                _violations, len(_fairness_results), len(results),
                            )
                        else:
                            logger.info(
                                "FairnessMonitor: all checks passed "
                                "(batch=%d, attributes=%s)",
                                len(results), list(_fairness_results.keys()),
                            )
                else:
                    logger.debug(
                        "FairnessMonitor: no protected-attribute data in batch "
                        "(pass attributes via context or response metadata)"
                    )
            except Exception:
                logger.debug(
                    "FairnessMonitor batch check failed (non-fatal)",
                    exc_info=True,
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

            # ---- Security: sanitize reason_prompt before LLM call ----
            # If the pipeline will call an LLM for reason generation,
            # classify and scrub the prompt to enforce VPC boundary rules.
            # HIGH/MEDIUM prompts are routed to Bedrock; LOW can use external.
            if self._prompt_sanitizer is not None:
                reason_prompt = context.get("reason_prompt", "")
                if reason_prompt:
                    try:
                        import hashlib as _hl
                        final_prompt, provider, san_result = (
                            self._prompt_sanitizer.sanitize_and_route(reason_prompt)
                        )
                        context = {
                            **context,
                            "reason_prompt": final_prompt,
                            "llm_provider": provider,
                        }
                        if san_result.scrubbed:
                            logger.info(
                                "PromptSanitizer: scrubbed %d item(s) from "
                                "reason_prompt (sensitivity=%s, provider=%s, "
                                "user_hash=%s)",
                                san_result.scrub_count,
                                san_result.sensitivity,
                                provider,
                                _hl.sha256(user_id.encode()).hexdigest()[:8],
                            )
                    except Exception:
                        logger.warning(
                            "PromptSanitizer failed for reason_prompt (non-fatal)",
                            exc_info=True,
                        )

            result = self._pipeline.recommend(
                customer_id=user_id,
                candidate_items=[candidate],
                customer_context=context,
            )

            # ---- Security: scrub PII from returned reason strings ----
            # Reason strings may contain interpolated customer values;
            # scrub before they leave the service boundary.
            items_out = []
            for item in result.items:
                reasons = item.reasons
                if self._prompt_sanitizer is not None and reasons:
                    scrubbed_reasons = []
                    for r in reasons:
                        if isinstance(r, str):
                            try:
                                r_scrubbed, _ = self._prompt_sanitizer.scrub(r)
                                scrubbed_reasons.append(r_scrubbed)
                            except Exception:
                                scrubbed_reasons.append(r)
                        else:
                            scrubbed_reasons.append(r)
                    reasons = scrubbed_reasons
                items_out.append({
                    "item_id": item.item_id,
                    "rank": item.rank,
                    "score": item.score,
                    "score_components": item.score_components,
                    "reasons": reasons,
                    "metadata": item.metadata,
                })
            return items_out
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
