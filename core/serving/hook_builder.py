"""Regulatory serving-hook builder (single entrypoint wiring).

Background
----------
The Sprint 1~4 regulatory components (M1 Human Review Queue, M4 Opt-out,
M5 Profiling Workflow, M6 Explanation SLA, M9 AI Risk Classifier, M10 Item
Universe Loader, M12 LLM Marker) are accepted by
:class:`core.serving.predict.RecommendationService` as optional keyword
hooks. Historically the Lambda / container entrypoints constructed only the
legacy modules (ConsentManager / AIDecisionOptOut / ProfilingRightsManager /
RegulatoryComplianceChecker) and never passed the new hooks, so the Sprint
3/4 behaviour only ever ran inside unit tests.

This module closes that gap: a single :func:`build_compliance_hooks` reads the
``compliance`` and ``serving`` blocks of ``pipeline.yaml`` and instantiates the
hooks through their per-module ``build_*`` factories, sharing one
``ComplianceStore`` across the store-backed rights modules. Both entrypoints
call it so the wiring lives in exactly one place (CLAUDE.md §0.2 단일 진입점).

Failure posture
---------------
``strict`` controls the failure direction:

* ``strict=True``  — fail-closed. A hook that fails to build raises, so a
  misconfigured deployment surfaces loudly instead of silently serving
  without that regulatory control. Recommended for production.
* ``strict=False`` — degraded. The failing hook is logged and omitted; the
  rest are still wired. Backward-compatible default.

``strict`` defaults to ``compliance.hooks.strict`` (then ``False``).
"""

from __future__ import annotations

import logging
from typing import Any, Callable, Dict, Optional

logger = logging.getLogger(__name__)

# RecommendationService kwarg names produced by this builder.
HOOK_KEYS = (
    "review_queue",            # M1
    "opt_out_manager",         # M4
    "profiling_workflow",      # M5
    "explanation_sla_tracker", # M6
    "ai_risk_classifier",      # M9
    "item_universe_loader",    # M10
    "marker_applier",          # M12
)


def _hooks_block(pipeline_config: Dict[str, Any]) -> Dict[str, Any]:
    compliance = pipeline_config.get("compliance") or {}
    return compliance.get("hooks") or {}


def hooks_enabled(pipeline_config: Optional[Dict[str, Any]]) -> bool:
    """Whether the regulatory serving hooks should be wired.

    Defaults to ``True`` — the regulatory hooks are on unless a deployment
    explicitly sets ``compliance.hooks.enabled: false``.
    """
    return bool(_hooks_block(pipeline_config or {}).get("enabled", True))


def build_compliance_hooks(
    pipeline_config: Optional[Dict[str, Any]],
    *,
    strict: Optional[bool] = None,
    audit_callback: Optional[Callable[..., Any]] = None,
    recompute_provider: Optional[Callable[..., Any]] = None,
    profile_provider: Optional[Callable[..., Any]] = None,
    duckdb_conn: Any = None,
) -> Dict[str, Any]:
    """Build the regulatory serving hooks from ``pipeline_config``.

    Parameters
    ----------
    pipeline_config:
        The pipeline.yaml-shaped config dict (must contain the ``compliance``
        and ``serving`` blocks). ``None`` / empty → returns ``{}``.
    strict:
        Fail-closed when True. Defaults to ``compliance.hooks.strict`` → False.
    audit_callback:
        Optional callable passed to the review queue so queue decisions land
        in the HMAC audit trail.
    recompute_provider / profile_provider:
        Optional injected providers for the profiling workflow (M5 recompute /
        access). Decoupled from the serving re-run path; ``None`` leaves those
        request types pending until a provider is supplied.
    duckdb_conn:
        Optional DuckDB connection for the item-universe loader (M10).

    Returns
    -------
    dict
        Mapping of RecommendationService kwarg → hook instance. Splat directly:
        ``RecommendationService(..., **build_compliance_hooks(cfg))``.
    """
    pipeline_config = pipeline_config or {}
    if not pipeline_config:
        # No config to build from — caller should pass a pipeline.yaml-shaped
        # dict. Returning {} keeps non-regulatory callers a no-op.
        return {}
    if not hooks_enabled(pipeline_config):
        logger.info("compliance.hooks.enabled=false — regulatory hooks not wired")
        return {}

    if strict is None:
        strict = bool(_hooks_block(pipeline_config).get("strict", False))

    compliance_cfg = pipeline_config.get("compliance") or {}
    hooks: Dict[str, Any] = {}

    # Shared ComplianceStore for the store-backed rights modules (M4/M5/M6/M9).
    store = None
    try:
        from core.compliance.store import build_compliance_store
        store = build_compliance_store(pipeline_config)
    except Exception:
        if strict:
            raise
        logger.warning(
            "ComplianceStore build failed; store-backed hooks (M4/M5/M6/M9) "
            "will be skipped",
            exc_info=True,
        )

    def _try(key: str, factory: Callable[[], Any]) -> None:
        try:
            hooks[key] = factory()
        except Exception:
            if strict:
                raise
            logger.warning(
                "Regulatory hook %r failed to build; omitting (serving "
                "continues without it)", key, exc_info=True,
            )

    # M1 — Human Review Queue (audit_callback wires queue decisions to the trail)
    def _review_queue() -> Any:
        from core.serving.review.human_review_queue import build_human_review_queue
        return build_human_review_queue(
            pipeline_config=pipeline_config, audit_callback=audit_callback,
        )
    _try("review_queue", _review_queue)

    # M10 — Dynamic Item Universe Loader
    def _item_universe() -> Any:
        from core.recommendation.universe.dynamic_loader import build_item_universe_loader
        return build_item_universe_loader(
            pipeline_config=pipeline_config, duckdb_conn=duckdb_conn,
        )
    _try("item_universe_loader", _item_universe)

    # M12 — LLM generation marker (compliance.llm_marker)
    def _marker() -> Any:
        from core.recommendation.reason.marker_applier import MarkerApplier
        return MarkerApplier.from_config(pipeline_config)
    _try("marker_applier", _marker)

    # M4/M5/M6/M9 require the shared store.
    if store is not None:
        def _opt_out() -> Any:
            from core.compliance.rights import OptOutConfig, OptOutManager
            return OptOutManager(
                store=store,
                config=OptOutConfig.from_dict(compliance_cfg.get("opt_out")),
            )
        _try("opt_out_manager", _opt_out)

        def _profiling() -> Any:
            from core.compliance.rights import ProfilingConfig, ProfilingWorkflow
            return ProfilingWorkflow(
                store=store,
                config=ProfilingConfig.from_dict(compliance_cfg.get("profiling")),
                profile_provider=profile_provider,
                recompute_provider=recompute_provider,
            )
        _try("profiling_workflow", _profiling)

        def _explanation_sla() -> Any:
            from core.compliance.rights.explanation_sla import build_explanation_sla_tracker
            return build_explanation_sla_tracker(
                store=store, config=compliance_cfg.get("sla"),
            )
        _try("explanation_sla_tracker", _explanation_sla)

        def _ai_risk() -> Any:
            from core.compliance.ai_risk_classifier import build_ai_risk_classifier
            return build_ai_risk_classifier(
                store=store, config=compliance_cfg.get("ai_risk"),
            )
        _try("ai_risk_classifier", _ai_risk)
    elif strict:
        # store is None only reachable here when strict=False (else we raised)
        pass

    logger.info(
        "Regulatory hooks wired: %s (strict=%s)",
        sorted(hooks.keys()), strict,
    )
    return hooks
