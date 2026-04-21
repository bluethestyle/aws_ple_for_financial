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
from typing import Any, Callable, Dict, Optional

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
            )

        return GateVerdict(
            decision="pass",
            reason=(
                f"FRIA={fria_result.risk_category}, "
                f"AI Risk={ai_result.grade}"
            ),
            fria=fria_result,
            ai_risk=ai_result,
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


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def build_promotion_gate(
    config: Dict[str, Any],
    store: Optional[ComplianceStore] = None,
    fria_scores_provider: Optional[DimensionScoresFn] = None,
    ai_risk_scores_provider: Optional[DimensionScoresFn] = None,
) -> PromotionGate:
    """Build a PromotionGate from the top-level pipeline config."""
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

    return PromotionGate(
        fria_assessor=fria,
        ai_risk_classifier=ai_risk,
        enabled=enabled,
        require_approval_on_escalation=require_approval,
        fria_scores_provider=fria_scores_provider,
        ai_risk_scores_provider=ai_risk_scores_provider,
        default_score=default_score,
    )
