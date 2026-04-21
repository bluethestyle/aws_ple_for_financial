"""
KoreanFRIAAssessor - Korean AI Basic Act FRIA (M7).

Distinct from ``core.monitoring.fria_evaluator.FRIAEvaluator`` which targets
the **EU AI Act Article 9**. This module targets:

- AI기본법 §35 ②③ (국가기관등 FRIA 강화조항)
- AI기본법 시행령 §27 (FRIA 의무 평가 7개 차원)
- 5년 보존 (AI기본법 §35③)

The 7 assessment dimensions map directly to the enumerated 평가 항목
in 시행령 §27:

1. data_sensitivity       — 데이터 민감도
2. automation_level       — 자동화 수준
3. scope_of_impact        — 영향 범위
4. model_complexity       — 모델 복잡도
5. external_dependency    — 외부 의존도
6. fairness_risk          — 공정성 리스크
7. explainability_gap     — 설명가능성 갭

Each dimension ∈ [0.0, 1.0] (higher = more risk). Aggregate uses configured
weights (default: equal). Risk category maps to:

- UNACCEPTABLE (>= 0.90) — 배포 금지, 추가 심사 필요
- HIGH         (>= 0.70) — 강화된 감독 + 분기 재평가
- LIMITED      (>= 0.40) — 표준 감독
- MINIMAL      (< 0.40)  — 연 1회 재평가

Results are persisted to ComplianceStore as ``FRIA_ASSESSMENT`` events
with an automatic ``retention_expiry`` timestamp.
"""

from __future__ import annotations

import logging
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta, timezone
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
    "FRIA_DIMENSIONS",
    "FRIA_RISK_THRESHOLDS",
    "FRIAConfig",
    "FRIAResult",
    "KoreanFRIAAssessor",
    "build_fria_assessor",
]


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

FRIA_DIMENSIONS: Sequence[str] = (
    "data_sensitivity",
    "automation_level",
    "scope_of_impact",
    "model_complexity",
    "external_dependency",
    "fairness_risk",
    "explainability_gap",
)

FRIA_RISK_THRESHOLDS: Dict[str, float] = {
    "UNACCEPTABLE": 0.90,
    "HIGH": 0.70,
    "LIMITED": 0.40,
    "MINIMAL": 0.0,
}

VALID_OPERATOR_TYPES = ("국가기관등", "민간")


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class FRIAConfig:
    """Config driven from ``compliance.fria`` block of pipeline.yaml."""

    operator_type: str = "국가기관등"
    retention_days: int = 1825                 # 5년 (AI기본법 §35③)
    warning_before_expiry_days: int = 180      # 만료 180일 전 경고
    dimensions: Sequence[str] = FRIA_DIMENSIONS
    dimension_weights: Optional[Dict[str, float]] = None   # None → equal
    risk_thresholds: Dict[str, float] = field(
        default_factory=lambda: dict(FRIA_RISK_THRESHOLDS)
    )

    def __post_init__(self) -> None:
        if self.operator_type not in VALID_OPERATOR_TYPES:
            raise ValueError(
                f"operator_type={self.operator_type!r} must be in "
                f"{VALID_OPERATOR_TYPES}"
            )
        if self.retention_days <= 0:
            raise ValueError("retention_days must be > 0")
        if not self.dimensions:
            raise ValueError("dimensions must be non-empty")
        if self.dimension_weights is not None:
            total = sum(self.dimension_weights.values())
            if not (0.99 <= total <= 1.01):
                raise ValueError(
                    f"dimension_weights must sum to 1.0; got {total}"
                )

    @classmethod
    def from_dict(cls, data: Optional[Dict[str, Any]]) -> "FRIAConfig":
        if not data:
            return cls()
        dims = data.get("dimensions", FRIA_DIMENSIONS)
        weights = data.get("dimension_weights")
        return cls(
            operator_type=data.get("operator_type", "국가기관등"),
            retention_days=int(data.get("retention_days", 1825)),
            warning_before_expiry_days=int(
                data.get("warning_before_expiry_days", 180)
            ),
            dimensions=tuple(dims),
            dimension_weights=(
                {k: float(v) for k, v in weights.items()}
                if weights else None
            ),
            risk_thresholds=dict(
                data.get("risk_thresholds", FRIA_RISK_THRESHOLDS)
            ),
        )

    def resolved_weights(self) -> Dict[str, float]:
        if self.dimension_weights is None:
            n = len(self.dimensions)
            return {d: 1.0 / n for d in self.dimensions}
        return dict(self.dimension_weights)


# ---------------------------------------------------------------------------
# Result
# ---------------------------------------------------------------------------

@dataclass
class FRIAResult:
    assessment_id: str
    model_version: str
    operator_type: str
    assessed_at: datetime
    retention_expiry: datetime
    dimensions: Dict[str, float]
    total_score: float
    risk_category: str
    mitigation_plan: Optional[str] = None
    context: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["assessed_at"] = self.assessed_at.isoformat()
        d["retention_expiry"] = self.retention_expiry.isoformat()
        return d

    def is_expiring(self, within_days: int, now: Optional[datetime] = None) -> bool:
        n = now or utcnow()
        return self.retention_expiry <= n + timedelta(days=within_days)

    def blocks_promotion(self) -> bool:
        return self.risk_category == "UNACCEPTABLE"


# ---------------------------------------------------------------------------
# Assessor
# ---------------------------------------------------------------------------

class KoreanFRIAAssessor:
    """Korean AI Basic Act FRIA evaluator (M7).

    Persists each assessment to a :class:`ComplianceStore` as a
    ``FRIA_ASSESSMENT`` event, computes a ``retention_expiry`` = +5 years
    by default, and exposes list queries for dashboards / compliance
    sweeps.
    """

    def __init__(
        self,
        store: ComplianceStore,
        config: Optional[FRIAConfig] = None,
    ) -> None:
        self._store = store
        self._cfg = config or FRIAConfig()

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------

    def evaluate(
        self,
        model_version: str,
        dimension_scores: Dict[str, float],
        mitigation_plan: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None,
        assessed_at: Optional[datetime] = None,
    ) -> FRIAResult:
        """Score a single model version across the 7 FRIA dimensions."""
        missing = [d for d in self._cfg.dimensions if d not in dimension_scores]
        if missing:
            raise ValueError(
                f"dimension_scores missing keys: {missing}; "
                f"required {tuple(self._cfg.dimensions)}"
            )
        for d, v in dimension_scores.items():
            if not (0.0 <= v <= 1.0):
                raise ValueError(
                    f"dimension {d!r} score={v} outside [0.0, 1.0]"
                )

        weights = self._cfg.resolved_weights()
        total = sum(
            dimension_scores[d] * weights.get(d, 0.0)
            for d in self._cfg.dimensions
        )
        total = round(min(max(total, 0.0), 1.0), 4)
        risk_category = self._classify(total)

        now = assessed_at or utcnow()
        retention_expiry = now + timedelta(days=self._cfg.retention_days)
        result = FRIAResult(
            assessment_id=f"fria_{uuid.uuid4().hex}",
            model_version=model_version,
            operator_type=self._cfg.operator_type,
            assessed_at=now,
            retention_expiry=retention_expiry,
            dimensions={d: round(dimension_scores[d], 4)
                        for d in self._cfg.dimensions},
            total_score=total,
            risk_category=risk_category,
            mitigation_plan=mitigation_plan,
            context=dict(context or {}),
        )

        self._persist(result)
        logger.info(
            "FRIA evaluated: model=%s total=%.4f risk=%s retention_expiry=%s",
            model_version, total, risk_category,
            retention_expiry.isoformat(),
        )
        return result

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    def list_assessments(
        self,
        model_version: Optional[str] = None,
        since: Optional[datetime] = None,
    ) -> List[FRIAResult]:
        events = self._store.query_events(
            event_type=EventType.FRIA_ASSESSMENT,
            since=since,
        )
        results = [self._event_to_result(e) for e in events]
        if model_version is not None:
            results = [r for r in results if r.model_version == model_version]
        results.sort(key=lambda r: r.assessed_at)
        return results

    def latest_for_model(self, model_version: str) -> Optional[FRIAResult]:
        lst = self.list_assessments(model_version=model_version)
        return lst[-1] if lst else None

    def list_expiring(
        self, within_days: Optional[int] = None,
        now: Optional[datetime] = None,
    ) -> List[FRIAResult]:
        n = now or utcnow()
        w = within_days if within_days is not None else (
            self._cfg.warning_before_expiry_days
        )
        return [
            r for r in self.list_assessments()
            if r.is_expiring(w, now=n)
        ]

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _classify(self, total: float) -> str:
        # Iterate thresholds in descending order for the first match.
        ordered = sorted(
            self._cfg.risk_thresholds.items(),
            key=lambda kv: kv[1], reverse=True,
        )
        for name, th in ordered:
            if total >= th:
                return name
        return "MINIMAL"

    def _persist(self, result: FRIAResult) -> None:
        evt = ComplianceEvent(
            event_id=new_event_id(),
            user_id=f"model:{result.model_version}",
            event_type=EventType.FRIA_ASSESSMENT,
            timestamp=result.assessed_at,
            payload=result.to_dict(),
        )
        self._store.put_event(evt)

    @staticmethod
    def _event_to_result(evt: ComplianceEvent) -> FRIAResult:
        p = evt.payload
        return FRIAResult(
            assessment_id=p["assessment_id"],
            model_version=p["model_version"],
            operator_type=p["operator_type"],
            assessed_at=datetime.fromisoformat(p["assessed_at"]),
            retention_expiry=datetime.fromisoformat(p["retention_expiry"]),
            dimensions=dict(p["dimensions"]),
            total_score=float(p["total_score"]),
            risk_category=p["risk_category"],
            mitigation_plan=p.get("mitigation_plan"),
            context=dict(p.get("context", {})),
        )


def build_fria_assessor(
    store: ComplianceStore,
    config: Optional[Dict[str, Any]] = None,
) -> KoreanFRIAAssessor:
    """Factory: build a KoreanFRIAAssessor from ``compliance.fria`` block."""
    return KoreanFRIAAssessor(
        store=store, config=FRIAConfig.from_dict(config),
    )
