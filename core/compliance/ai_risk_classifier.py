"""
AIRiskClassifier - 금감원 AI RMF 6-dimensional risk classifier (M9).

Legal / supervisory basis:
- 금감원 「금융부문 AI 활용 모범규준」 (AI RMF 2024)
- 6-차원 리스크 평가:
  1. data_sensitivity       — 데이터 민감도
  2. automation_level       — 자동화 수준
  3. scope_of_impact        — 영향 범위
  4. model_complexity       — 모델 복잡도
  5. external_dependency    — 외부 의존도
  6. fairness_risk          — 공정성 리스크

Aggregate score = Σ(dim_score × weight). Default weights are the ones in
``compliance.ai_risk.dimensions`` in pipeline.yaml (sum = 1.0).

Grade mapping (default): total_score → high / medium / low.

History: every assessment is persisted as an ``AI_RISK_ASSESSMENT`` event.
The classifier can detect grade **changes** against the previous assessment
for the same model and emit a marker in the payload so downstream (model
promotion gates) can require additional approvals when a model drifts into
a higher risk bucket.
"""

from __future__ import annotations

import logging
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Sequence

from core.compliance.store import ComplianceStore
from core.compliance.types import (
    ComplianceEvent,
    EventType,
    new_event_id,
    utcnow,
)

logger = logging.getLogger(__name__)

__all__ = [
    "AI_RISK_DIMENSIONS",
    "AIRiskConfig",
    "AIRiskAssessment",
    "AIRiskClassifier",
    "build_ai_risk_classifier",
]


AI_RISK_DIMENSIONS: Sequence[str] = (
    "data_sensitivity",
    "automation_level",
    "scope_of_impact",
    "model_complexity",
    "external_dependency",
    "fairness_risk",
)

DEFAULT_WEIGHTS: Dict[str, float] = {
    "data_sensitivity": 0.25,
    "automation_level": 0.20,
    "scope_of_impact": 0.20,
    "model_complexity": 0.15,
    "external_dependency": 0.10,
    "fairness_risk": 0.10,
}

DEFAULT_GRADE_THRESHOLDS: Dict[str, float] = {
    "high": 0.70,
    "medium": 0.40,
    # anything below "medium" is "low"
}

VALID_GRADES = ("high", "medium", "low")


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class AIRiskConfig:
    """Config driven from ``compliance.ai_risk`` block of pipeline.yaml."""

    dimensions: Dict[str, float] = field(
        default_factory=lambda: dict(DEFAULT_WEIGHTS)
    )
    grade_thresholds: Dict[str, float] = field(
        default_factory=lambda: dict(DEFAULT_GRADE_THRESHOLDS)
    )

    def __post_init__(self) -> None:
        if not self.dimensions:
            raise ValueError("dimensions weights must be non-empty")
        total = sum(self.dimensions.values())
        if not (0.99 <= total <= 1.01):
            raise ValueError(
                f"dimension weights must sum to 1.0; got {total}"
            )
        for grade in ("high", "medium"):
            if grade not in self.grade_thresholds:
                raise ValueError(
                    f"grade_thresholds missing {grade!r}"
                )
        if self.grade_thresholds["high"] <= self.grade_thresholds["medium"]:
            raise ValueError(
                "grade_thresholds['high'] must be > grade_thresholds['medium']"
            )

    @classmethod
    def from_dict(cls, data: Optional[Dict[str, Any]]) -> "AIRiskConfig":
        if not data:
            return cls()
        dimensions = data.get("dimensions", DEFAULT_WEIGHTS)
        thresholds = data.get("grade_thresholds", DEFAULT_GRADE_THRESHOLDS)
        return cls(
            dimensions={k: float(v) for k, v in dimensions.items()},
            grade_thresholds={k: float(v) for k, v in thresholds.items()},
        )


# ---------------------------------------------------------------------------
# Result
# ---------------------------------------------------------------------------

@dataclass
class AIRiskAssessment:
    assessment_id: str
    model_version: str
    assessed_at: datetime
    dimensions: Dict[str, float]    # 차원별 점수 [0.0, 1.0]
    weights: Dict[str, float]
    total_score: float
    grade: str                       # "high" | "medium" | "low"
    prev_grade: Optional[str] = None
    grade_change: bool = False
    context: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["assessed_at"] = self.assessed_at.isoformat()
        return d

    def escalated(self) -> bool:
        """True if grade moved toward higher risk vs previous assessment."""
        rank = {"low": 0, "medium": 1, "high": 2}
        if self.prev_grade is None:
            return False
        return rank[self.grade] > rank[self.prev_grade]

    def requires_additional_approval(self) -> bool:
        """Promotion gate: a grade bump to 'high' needs override."""
        return self.grade == "high" and self.escalated()


# ---------------------------------------------------------------------------
# Classifier
# ---------------------------------------------------------------------------

class AIRiskClassifier:
    def __init__(
        self,
        store: ComplianceStore,
        config: Optional[AIRiskConfig] = None,
    ) -> None:
        self._store = store
        self._cfg = config or AIRiskConfig()

    # ------------------------------------------------------------------
    # Classification
    # ------------------------------------------------------------------

    def classify(
        self,
        model_version: str,
        dimension_scores: Dict[str, float],
        context: Optional[Dict[str, Any]] = None,
        assessed_at: Optional[datetime] = None,
    ) -> AIRiskAssessment:
        weights = self._cfg.dimensions
        missing = [d for d in weights if d not in dimension_scores]
        if missing:
            raise ValueError(
                f"dimension_scores missing keys: {missing}; "
                f"required {tuple(weights.keys())}"
            )
        for d, v in dimension_scores.items():
            if d not in weights:
                # Unknown dim: allowed as noise but not used in total.
                continue
            if not (0.0 <= v <= 1.0):
                raise ValueError(
                    f"dimension {d!r} score={v} outside [0.0, 1.0]"
                )

        total = sum(
            dimension_scores[d] * weights[d] for d in weights
        )
        total = round(min(max(total, 0.0), 1.0), 4)
        grade = self._grade_for(total)

        prev = self.latest_for_model(model_version)
        prev_grade = prev.grade if prev is not None else None

        assessment = AIRiskAssessment(
            assessment_id=f"airisk_{uuid.uuid4().hex}",
            model_version=model_version,
            assessed_at=assessed_at or utcnow(),
            dimensions={d: round(float(v), 4)
                        for d, v in dimension_scores.items() if d in weights},
            weights=dict(weights),
            total_score=total,
            grade=grade,
            prev_grade=prev_grade,
            grade_change=(prev_grade is not None and prev_grade != grade),
            context=dict(context or {}),
        )

        self._persist(assessment)
        if assessment.grade_change:
            logger.warning(
                "AI risk grade CHANGE: model=%s %s -> %s (score=%.4f)",
                model_version, prev_grade, grade, total,
            )
        else:
            logger.info(
                "AI risk classified: model=%s grade=%s (score=%.4f)",
                model_version, grade, total,
            )
        return assessment

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    def list_assessments(
        self, model_version: Optional[str] = None,
    ) -> List[AIRiskAssessment]:
        events = self._store.query_events(
            event_type=EventType.AI_RISK_ASSESSMENT,
        )
        out = [self._event_to_assessment(e) for e in events]
        if model_version is not None:
            out = [a for a in out if a.model_version == model_version]
        out.sort(key=lambda a: a.assessed_at)
        return out

    def latest_for_model(
        self, model_version: str,
    ) -> Optional[AIRiskAssessment]:
        lst = self.list_assessments(model_version=model_version)
        return lst[-1] if lst else None

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _grade_for(self, total: float) -> str:
        th = self._cfg.grade_thresholds
        if total >= th["high"]:
            return "high"
        if total >= th["medium"]:
            return "medium"
        return "low"

    def _persist(self, assessment: AIRiskAssessment) -> None:
        evt = ComplianceEvent(
            event_id=new_event_id(),
            user_id=f"model:{assessment.model_version}",
            event_type=EventType.AI_RISK_ASSESSMENT,
            timestamp=assessment.assessed_at,
            payload=assessment.to_dict(),
        )
        self._store.put_event(evt)

    @staticmethod
    def _event_to_assessment(evt: ComplianceEvent) -> AIRiskAssessment:
        p = evt.payload
        return AIRiskAssessment(
            assessment_id=p["assessment_id"],
            model_version=p["model_version"],
            assessed_at=datetime.fromisoformat(p["assessed_at"]),
            dimensions=dict(p["dimensions"]),
            weights=dict(p["weights"]),
            total_score=float(p["total_score"]),
            grade=p["grade"],
            prev_grade=p.get("prev_grade"),
            grade_change=bool(p.get("grade_change", False)),
            context=dict(p.get("context", {})),
        )


def build_ai_risk_classifier(
    store: ComplianceStore,
    config: Optional[Dict[str, Any]] = None,
) -> AIRiskClassifier:
    """Factory consuming the ``compliance.ai_risk`` block."""
    return AIRiskClassifier(
        store=store, config=AIRiskConfig.from_dict(config),
    )
