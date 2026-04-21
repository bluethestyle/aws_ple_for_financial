"""
PromotionGate - Sprint 2 addition to the Champion-Challenger pipeline.

Orchestrates the compliance checks that must pass **after** the core
ModelCompetition has already approved a challenger. Driven by the
``compliance.promotion_gate`` config block in pipeline.yaml.

This module does NOT replace ``scripts/submit_pipeline.py::_decide_promotion``;
it is called from inside that function as an optional post-check. Per
CLAUDE.md 1.10, ``_decide_promotion`` remains the single entry point.

Gate steps:
1. FRIA evaluation       (M7) - UNACCEPTABLE -> reject (safety floor)
2. AI Risk Classifier    (M9) - escalation to 'high' -> require_approval
3. Return aggregated verdict

Dimension scores default to a conservative 0.5 for every dimension unless
the caller supplies a scores provider function. This keeps the gate
testable without requiring every caller to wire real metrics.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence

from core.compliance.ai_risk_classifier import (
    AIRiskAssessment,
    AIRiskClassifier,
    AIRiskConfig,
)
from core.compliance.fria_assessment import (
    FRIAConfig,
    FRIAResult,
    KoreanFRIAAssessor,
)
from core.compliance.store import (
    ComplianceStore,
    InMemoryComplianceStore,
    build_compliance_store,
)

logger = logging.getLogger(__name__)

__all__ = [
    "GateVerdict",
    "PromotionGate",
    "build_promotion_gate",
]


DimensionScoresFn = Callable[[str], Dict[str, float]]
# Signature: model_version -> {dim_name: score_in_[0,1]}


# ---------------------------------------------------------------------------
# Verdict
# ---------------------------------------------------------------------------

@dataclass
class GateVerdict:
    decision: str            # "pass" | "reject" | "require_approval" | "skip"
    reason: str = ""
    fria: Optional[FRIAResult] = None
    ai_risk: Optional[AIRiskAssessment] = None
    details: Dict[str, Any] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return self.decision == "pass"

    @property
    def blocks_promotion(self) -> bool:
        return self.decision in ("reject", "require_approval")


# ---------------------------------------------------------------------------
# Gate
# ---------------------------------------------------------------------------

class PromotionGate:
    """Runs FRIA + AI Risk evaluation after ModelCompetition.evaluate()."""

    def __init__(
        self,
        fria_assessor: KoreanFRIAAssessor,
        ai_risk_classifier: AIRiskClassifier,
        enabled: bool = True,
        require_approval_on_escalation: bool = True,
        fria_scores_provider: Optional[DimensionScoresFn] = None,
        ai_risk_scores_provider: Optional[DimensionScoresFn] = None,
        default_score: float = 0.5,
    ) -> None:
        self._fria = fria_assessor
        self._ai_risk = ai_risk_classifier
        self._enabled = enabled
        self._require_approval_on_escalation = require_approval_on_escalation
        self._fria_provider = fria_scores_provider
        self._ai_risk_provider = ai_risk_scores_provider
        self._default_score = default_score

    def evaluate(
        self,
        model_version: str,
        context: Optional[Dict[str, Any]] = None,
    ) -> GateVerdict:
        if not self._enabled:
            return GateVerdict(
                decision="skip", reason="promotion_gate disabled",
            )

        ctx = dict(context or {})

        fria_scores = self._scores_for(
            self._fria_provider, model_version,
            dimensions=self._fria._cfg.dimensions,  # type: ignore[attr-defined]
        )
        fria_result = self._fria.evaluate(
            model_version=model_version,
            dimension_scores=fria_scores,
            context={"source": "promotion_gate", **ctx},
        )
        if fria_result.blocks_promotion():
            return GateVerdict(
                decision="reject",
                reason=(
                    f"FRIA {fria_result.risk_category} "
                    f"(score={fria_result.total_score:.4f})"
                ),
                fria=fria_result,
                details=self._build_details(
                    model_version, fria_scores, None,
                ),
            )

        ai_scores = self._scores_for(
            self._ai_risk_provider, model_version,
            dimensions=list(self._ai_risk._cfg.dimensions.keys()),  # type: ignore[attr-defined]
        )
        ai_result = self._ai_risk.classify(
            model_version=model_version,
            dimension_scores=ai_scores,
            context={"source": "promotion_gate", **ctx},
        )

        if (self._require_approval_on_escalation
                and ai_result.requires_additional_approval()):
            return GateVerdict(
                decision="require_approval",
                reason=(
                    f"AI Risk escalated: {ai_result.prev_grade} -> "
                    f"{ai_result.grade}"
                ),
                fria=fria_result,
                ai_risk=ai_result,
                details=self._build_details(
                    model_version, fria_scores, ai_scores,
                ),
            )

        return GateVerdict(
            decision="pass",
            reason=(
                f"FRIA={fria_result.risk_category}, "
                f"AI Risk={ai_result.grade}"
            ),
            fria=fria_result,
            ai_risk=ai_result,
            details=self._build_details(
                model_version, fria_scores, ai_scores,
            ),
        )

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _scores_for(
        self,
        provider: Optional[DimensionScoresFn],
        model_version: str,
        dimensions,
    ) -> Dict[str, float]:
        if provider is not None:
            scores = provider(model_version) or {}
        else:
            scores = {}
        # Fill in defaults for any missing dimension.
        return {
            d: float(scores.get(d, self._default_score))
            for d in dimensions
        }

    def _build_details(
        self,
        model_version: str,
        fria_scores: Dict[str, float],
        ai_scores: Optional[Dict[str, float]],
    ) -> Dict[str, Any]:
        """Assemble audit payload: metadata snapshot + per-dim derivation.

        Always-safe: any introspection failure returns a details dict
        with an ``_error`` field rather than raising. Audit must not
        block promotion (CLAUDE.md §1.10).
        """
        details: Dict[str, Any] = {
            "model_version": model_version,
            "fria_dimension_scores": dict(fria_scores),
        }
        if ai_scores is not None:
            details["ai_risk_dimension_scores"] = dict(ai_scores)

        details["fria_derivation"] = self._explain_if_possible(
            self._fria_provider, model_version,
        )
        details["ai_risk_derivation"] = self._explain_if_possible(
            self._ai_risk_provider, model_version,
        )
        details["metadata_snapshot"] = self._metadata_snapshot(
            self._fria_provider, model_version,
        )
        return details

    @staticmethod
    def _explain_if_possible(
        provider: Optional[DimensionScoresFn], model_version: str,
    ) -> Optional[Dict[str, Dict[str, Any]]]:
        if provider is None:
            return None
        candidates = [provider]
        inner = getattr(provider, "_providers", None)
        if inner is not None:
            candidates.extend(inner)
        for c in candidates:
            explain = getattr(c, "explain", None)
            if callable(explain):
                try:
                    return explain(model_version)
                except Exception:
                    logger.exception(
                        "provider.explain failed for %s", model_version,
                    )
                    return {"_error": "explain_failed"}
        return None

    @staticmethod
    def _metadata_snapshot(
        provider: Optional[DimensionScoresFn], model_version: str,
    ) -> Optional[Dict[str, Any]]:
        """Extract the raw metadata dict the provider saw, if exposed.

        MetricsDerivedScoreProvider holds a metadata_lookup callable; we
        dip into it to freeze the exact inputs for audit. CompositeProvider
        is walked as well.
        """
        if provider is None:
            return None
        candidates = [provider]
        inner = getattr(provider, "_providers", None)
        if inner is not None:
            candidates.extend(inner)
        for c in candidates:
            lookup = getattr(c, "_lookup", None)
            if callable(lookup):
                try:
                    snap = lookup(model_version) or {}
                    return dict(snap) if isinstance(snap, dict) else {
                        "value": snap,
                    }
                except Exception:
                    logger.exception(
                        "metadata snapshot failed for %s", model_version,
                    )
                    return {"_error": "snapshot_failed"}
        return None


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def build_promotion_gate(
    config: Dict[str, Any],
    store: Optional[ComplianceStore] = None,
    fria_scores_provider: Optional[DimensionScoresFn] = None,
    ai_risk_scores_provider: Optional[DimensionScoresFn] = None,
    metadata_aggregator: Optional[Callable[[str], Dict[str, Any]]] = None,
) -> PromotionGate:
    """Build a PromotionGate from the top-level pipeline config.

    Dimension-score provider wiring (in precedence order, earliest wins):

    1. Explicit ``fria_scores_provider`` / ``ai_risk_scores_provider``
       kwargs — callers can fully override the auto-wired chain.
    2. ``compliance.promotion_gate.providers.manual_overrides`` — operator
       override block in pipeline.yaml, keyed by model_version.
    3. ``metadata_aggregator`` — real metadata pulled from lineage /
       fairness / registry / LLM config. When omitted the heuristics
       fall back to per-rule defaults (0.5) and the gate behaves as
       conservative LIMITED.

    See ``core.compliance.metadata_aggregator`` for aggregator wiring.
    """
    compliance_cfg = (config.get("compliance") or {})
    gate_cfg = compliance_cfg.get("promotion_gate") or {}
    enabled = bool(gate_cfg.get("enabled", False))
    require_approval = bool(gate_cfg.get(
        "require_approval_on_escalation", True
    ))
    default_score = float(gate_cfg.get("default_score", 0.5))

    # Build or reuse a ComplianceStore
    if store is None:
        store_cfg = compliance_cfg.get("store", {"backend": "in_memory"})
        try:
            store = build_compliance_store({"store": store_cfg})
        except Exception:
            logger.warning(
                "PromotionGate: failed to build configured ComplianceStore, "
                "falling back to InMemoryComplianceStore",
                exc_info=True,
            )
            store = InMemoryComplianceStore()

    fria_cfg = FRIAConfig.from_dict(compliance_cfg.get("fria"))
    fria = KoreanFRIAAssessor(store=store, config=fria_cfg)

    ai_cfg = AIRiskConfig.from_dict(compliance_cfg.get("ai_risk"))
    ai_risk = AIRiskClassifier(store=store, config=ai_cfg)

    fria_provider, ai_provider = _auto_compose_providers(
        compliance_cfg,
        metadata_aggregator=metadata_aggregator,
        explicit_fria=fria_scores_provider,
        explicit_ai=ai_risk_scores_provider,
        fria_dimensions=fria_cfg.dimensions,
        ai_dimensions=list(ai_cfg.dimensions.keys()),
    )

    return PromotionGate(
        fria_assessor=fria,
        ai_risk_classifier=ai_risk,
        enabled=enabled,
        require_approval_on_escalation=require_approval,
        fria_scores_provider=fria_provider,
        ai_risk_scores_provider=ai_provider,
        default_score=default_score,
    )


def _auto_compose_providers(
    compliance_cfg: Dict[str, Any],
    *,
    metadata_aggregator: Optional[Callable[[str], Dict[str, Any]]],
    explicit_fria: Optional[DimensionScoresFn],
    explicit_ai: Optional[DimensionScoresFn],
    fria_dimensions: Sequence[str],
    ai_dimensions: Sequence[str],
) -> tuple[Optional[DimensionScoresFn], Optional[DimensionScoresFn]]:
    """Compose Manual + MetricsDerived providers from config + aggregator.

    Precedence: explicit kwarg → (manual_overrides + aggregator-backed
    heuristic). Returns ``(fria_provider, ai_risk_provider)``.
    """
    from core.compliance.dimension_scores import (
        CompositeProvider,
        ManualScoreProvider,
        MetricsDerivedScoreProvider,
    )

    gate_cfg = compliance_cfg.get("promotion_gate") or {}
    providers_cfg = gate_cfg.get("providers") or {}
    manual_overrides = providers_cfg.get("manual_overrides") or {}

    def _compose(
        explicit: Optional[DimensionScoresFn],
        dimensions: Sequence[str],
    ) -> Optional[DimensionScoresFn]:
        if explicit is not None:
            return explicit
        layers: List[DimensionScoresFn] = []
        if manual_overrides:
            layers.append(
                ManualScoreProvider(
                    overrides=manual_overrides,
                    dimensions=dimensions,
                )
            )
        if metadata_aggregator is not None:
            layers.append(
                MetricsDerivedScoreProvider(
                    metadata_lookup=metadata_aggregator,
                    dimensions=dimensions,
                )
            )
        if not layers:
            return None
        if len(layers) == 1:
            return layers[0]
        return CompositeProvider(layers)

    return (
        _compose(explicit_fria, fria_dimensions),
        _compose(explicit_ai, ai_dimensions),
    )
