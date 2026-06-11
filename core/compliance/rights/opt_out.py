"""
OptOutManager - AI decision opt-out + explanation request (M4).

Legal basis:
- 개보법 §37의2 (자동화된 결정의 거부권 / 설명요구권)
- AI기본법 §31 (AI 거부권)

Flow:
1. opt_out(user_id, fallback_type, reason)
   -> ComplianceRequest (type=OPT_OUT, status=PENDING)
   -> store records opt_out state
2. Subsequent predict.py calls check is_opted_out(user_id)
3. request_explanation(user_id, recommendation_id)
   -> opens an explanation request (internal SLA default 10d; legal max 30d
      per 시행령 §44의3⑤)
4. mark_explanation_provided(request_id, explanation)
   -> closes the request, emits REQUEST_PROCESSED event
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

from core.compliance.store import ComplianceStore
from core.compliance.types import (
    ComplianceEvent,
    ComplianceRequest,
    EventType,
    RequestStatus,
    RequestType,
    new_event_id,
    new_request_id,
    utcnow,
)

logger = logging.getLogger(__name__)


VALID_FALLBACKS = ("rule_based", "human_review", "disable")


@dataclass
class OptOutConfig:
    """Config for OptOutManager. Consumes compliance.opt_out block."""

    default_fallback: str = "rule_based"
    explanation_sla_days: int = 10        # 내부 SLA (법정 30일, 시행령 §44의3⑤)
    opt_out_response_days: int = 30       # 개보법 §37의2 / 시행령 §44의3⑤

    def __post_init__(self) -> None:
        if self.default_fallback not in VALID_FALLBACKS:
            raise ValueError(
                f"default_fallback={self.default_fallback!r} must be in "
                f"{VALID_FALLBACKS}"
            )

    @classmethod
    def from_dict(cls, data: Optional[Dict[str, Any]]) -> "OptOutConfig":
        if not data:
            return cls()
        return cls(
            default_fallback=data.get("default_fallback", "rule_based"),
            explanation_sla_days=int(data.get("explanation_sla_days", 10)),
            opt_out_response_days=int(data.get("opt_out_response_days", 30)),
        )


@dataclass
class OptOutDecision:
    """Runtime decision returned to predict.py."""

    is_opted_out: bool
    fallback_type: Optional[str] = None
    request_id: Optional[str] = None
    reason: Optional[str] = None


@dataclass
class CreditExplanationElements:
    """신용정보법 §36의2 자동화평가 설명 disclosure 요소 (구조화).

    PIPA §37의2 일반 설명과 달리 §36의2 는 구체 항목을 요구하므로 별도 구조로
    분리한다: 자동화평가 실시 여부, 평가 결과, 주요 기준, 사용된 기초정보
    (= 피처 → 원천 데이터 lineage). 기준·lineage 는 호출자가 주입한다
    (DataLineageTracker / 추천 근거 산출물과 결합).
    """

    user_id: str
    assessment_performed: bool                              # 자동화평가 실시 여부
    assessment_result: Optional[str] = None                 # 평가 결과(요약)
    main_criteria: List[str] = field(default_factory=list)  # 주요 기준
    # 사용된 기초정보: [{"feature": ..., "source": ...}, ...] (피처→원천 lineage)
    base_information: List[Dict[str, str]] = field(default_factory=list)
    recommendation_id: Optional[str] = None
    legal_basis: str = "신용정보법 §36의2"
    generated_at: Optional[datetime] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "user_id": self.user_id,
            "assessment_performed": self.assessment_performed,
            "assessment_result": self.assessment_result,
            "main_criteria": list(self.main_criteria),
            "base_information": [dict(b) for b in self.base_information],
            "recommendation_id": self.recommendation_id,
            "legal_basis": self.legal_basis,
            "generated_at": (
                self.generated_at.isoformat() if self.generated_at else None
            ),
        }


class OptOutManager:
    """AI decision opt-out + explanation request manager (M4)."""

    def __init__(
        self,
        store: ComplianceStore,
        config: Optional[OptOutConfig] = None,
    ) -> None:
        self._store = store
        self._cfg = config or OptOutConfig()

    # ------------------------------------------------------------------
    # Opt-out lifecycle
    # ------------------------------------------------------------------

    def opt_out(
        self,
        user_id: str,
        reason: str,
        fallback_type: Optional[str] = None,
    ) -> ComplianceRequest:
        """Record an opt-out + open a tracked ComplianceRequest."""
        fb = fallback_type or self._cfg.default_fallback
        if fb not in VALID_FALLBACKS:
            raise ValueError(
                f"fallback_type={fb!r} must be in {VALID_FALLBACKS}"
            )

        now = utcnow()
        req = ComplianceRequest(
            request_id=new_request_id(),
            user_id=user_id,
            request_type=RequestType.OPT_OUT,
            submitted_at=now,
            sla_deadline=now + timedelta(days=self._cfg.opt_out_response_days),
            status=RequestStatus.PROCESSED,  # opt-out itself is immediate
            processed_at=now,
            metadata={
                "fallback_type": fb,
                "reason": reason,
                "legal_basis": "개보법 §37의2 / AI기본법 §31",
            },
        )
        self._store.put_request(req)
        self._store.put_event(
            ComplianceEvent(
                event_id=new_event_id(),
                user_id=user_id,
                event_type=EventType.REQUEST_PROCESSED,
                timestamp=now,
                payload={
                    "sla_name": "opt_out",
                    "on_time": True,
                    "fallback_type": fb,
                    "reason": reason,
                },
                request_id=req.request_id,
            )
        )
        logger.info(
            "AI opt-out recorded: user_id=%s fallback=%s reason=%s",
            user_id, fb, reason,
        )
        return req

    def opt_in(self, user_id: str, reason: str = "") -> ComplianceRequest:
        """Revoke a prior opt-out."""
        now = utcnow()
        req = ComplianceRequest(
            request_id=new_request_id(),
            user_id=user_id,
            request_type=RequestType.OPT_OUT_REVOKE,
            submitted_at=now,
            sla_deadline=now + timedelta(days=self._cfg.opt_out_response_days),
            status=RequestStatus.PROCESSED,
            processed_at=now,
            metadata={"reason": reason},
        )
        self._store.put_request(req)
        self._store.put_event(
            ComplianceEvent(
                event_id=new_event_id(),
                user_id=user_id,
                event_type=EventType.REQUEST_PROCESSED,
                timestamp=now,
                payload={"sla_name": "opt_out_revoke", "reason": reason},
                request_id=req.request_id,
            )
        )
        logger.info("AI opt-in recorded: user_id=%s", user_id)
        return req

    # ------------------------------------------------------------------
    # Runtime query (predict.py integration)
    # ------------------------------------------------------------------

    def is_opted_out(self, user_id: str) -> bool:
        """Return True if the user currently has an active opt-out."""
        decision = self.get_decision(user_id)
        return decision.is_opted_out

    def get_decision(self, user_id: str) -> OptOutDecision:
        """Return the current opt-out state for the user."""
        events = self._store.query_events(user_id=user_id)
        latest_opt_out: Optional[Dict[str, Any]] = None
        for evt in events:
            if evt.payload.get("sla_name") == "opt_out":
                latest_opt_out = {"at": evt.timestamp, "payload": evt.payload,
                                  "request_id": evt.request_id}
            elif evt.payload.get("sla_name") == "opt_out_revoke":
                latest_opt_out = None

        if latest_opt_out is None:
            return OptOutDecision(is_opted_out=False)

        return OptOutDecision(
            is_opted_out=True,
            fallback_type=latest_opt_out["payload"].get(
                "fallback_type", self._cfg.default_fallback
            ),
            request_id=latest_opt_out.get("request_id"),
            reason=latest_opt_out["payload"].get("reason"),
        )

    # ------------------------------------------------------------------
    # Explanation request (개보법 §37의2 / 시행령 §44의2; 응답기한 §44의3⑤)
    # ------------------------------------------------------------------

    def request_explanation(
        self,
        user_id: str,
        recommendation_id: str,
        reason: str = "",
    ) -> ComplianceRequest:
        """Open an explanation request (internal SLA default 10d; legal max 30d)."""
        now = utcnow()
        deadline = now + timedelta(days=self._cfg.explanation_sla_days)
        req = ComplianceRequest(
            request_id=new_request_id(),
            user_id=user_id,
            request_type=RequestType.EXPLANATION,
            submitted_at=now,
            sla_deadline=deadline,
            status=RequestStatus.PENDING,
            metadata={
                "recommendation_id": recommendation_id,
                "reason": reason,
                "legal_basis": "개보법 §37의2 / 시행령 §44의2 (응답기한 §44의3⑤, 30일)",
            },
        )
        self._store.put_request(req)
        self._store.put_event(
            ComplianceEvent(
                event_id=new_event_id(),
                user_id=user_id,
                event_type=EventType.REQUEST_CREATED,
                timestamp=now,
                payload={
                    "sla_name": "explanation",
                    "recommendation_id": recommendation_id,
                    "sla_deadline": deadline.isoformat(),
                },
                request_id=req.request_id,
            )
        )
        logger.info(
            "Explanation request opened: user_id=%s recommendation_id=%s "
            "deadline=%s",
            user_id, recommendation_id, deadline.isoformat(),
        )
        return req

    def request_credit_explanation(
        self,
        user_id: str,
        recommendation_id: str,
        elements: Optional[CreditExplanationElements] = None,
        reason: str = "",
    ) -> ComplianceRequest:
        """Open a 신용정보법 §36의2 credit-explanation request.

        Distinct from ``request_explanation`` (PIPA §37의2): when ``elements``
        is supplied it carries structured §36의2 disclosure (assessment flag,
        main criteria, base-information lineage) in the request metadata.
        """
        now = utcnow()
        deadline = now + timedelta(days=self._cfg.explanation_sla_days)
        meta: Dict[str, Any] = {
            "recommendation_id": recommendation_id,
            "reason": reason,
            "legal_basis": "신용정보법 §36의2 (자동화평가 설명)",
        }
        if elements is not None:
            meta["credit_explanation_elements"] = elements.to_dict()
        req = ComplianceRequest(
            request_id=new_request_id(),
            user_id=user_id,
            request_type=RequestType.CREDIT_EXPLANATION,
            submitted_at=now,
            sla_deadline=deadline,
            status=RequestStatus.PENDING,
            metadata=meta,
        )
        self._store.put_request(req)
        self._store.put_event(
            ComplianceEvent(
                event_id=new_event_id(),
                user_id=user_id,
                event_type=EventType.REQUEST_CREATED,
                timestamp=now,
                payload={
                    "sla_name": "credit_explanation",
                    "recommendation_id": recommendation_id,
                    "sla_deadline": deadline.isoformat(),
                },
                request_id=req.request_id,
            )
        )
        logger.info(
            "Credit explanation request opened: user_id=%s recommendation_id=%s",
            user_id, recommendation_id,
        )
        return req

    @staticmethod
    def build_credit_explanation_elements(
        user_id: str,
        assessment_performed: bool,
        main_criteria: Optional[List[str]] = None,
        base_information: Optional[List[Dict[str, str]]] = None,
        assessment_result: Optional[str] = None,
        recommendation_id: Optional[str] = None,
    ) -> CreditExplanationElements:
        """Assemble §36의2 disclosure elements from caller-supplied data.

        ``base_information`` is the feature→source lineage (e.g. from a
        DataLineageTracker); ``main_criteria`` the assessment's main factors.
        """
        return CreditExplanationElements(
            user_id=user_id,
            assessment_performed=assessment_performed,
            assessment_result=assessment_result,
            main_criteria=list(main_criteria or []),
            base_information=[dict(b) for b in (base_information or [])],
            recommendation_id=recommendation_id,
            generated_at=utcnow(),
        )

    def mark_explanation_provided(
        self,
        request_id: str,
        explanation: str,
        provided_at: Optional[datetime] = None,
    ) -> None:
        """Close an explanation request with the actual explanation text.

        Accepts both PIPA §37의2 (EXPLANATION) and 신정법 §36의2
        (CREDIT_EXPLANATION) requests.
        """
        provided_at = provided_at or utcnow()
        req = self._store.get_request(request_id)
        if req is None:
            raise KeyError(f"Unknown request_id={request_id!r}")
        if req.request_type not in (
            RequestType.EXPLANATION, RequestType.CREDIT_EXPLANATION
        ):
            raise ValueError(
                f"request_id={request_id!r} is type={req.request_type!r}, "
                f"not an explanation type"
            )
        sla_name = (
            "credit_explanation"
            if req.request_type == RequestType.CREDIT_EXPLANATION
            else "explanation"
        )

        on_time = provided_at <= req.sla_deadline
        self._store.update_request_status(
            request_id, RequestStatus.PROCESSED, processed_at=provided_at
        )
        self._store.put_event(
            ComplianceEvent(
                event_id=new_event_id(),
                user_id=req.user_id,
                event_type=EventType.REQUEST_PROCESSED,
                timestamp=provided_at,
                payload={
                    "sla_name": sla_name,
                    "on_time": on_time,
                    "explanation_length": len(explanation),
                },
                request_id=request_id,
            )
        )
        if not on_time:
            self._store.put_event(
                ComplianceEvent(
                    event_id=new_event_id(),
                    user_id=req.user_id,
                    event_type=EventType.SLA_BREACH,
                    timestamp=provided_at,
                    payload={
                        "sla_name": sla_name,
                        "breach_type": "late_processing",
                        "sla_deadline": req.sla_deadline.isoformat(),
                    },
                    request_id=request_id,
                )
            )
            logger.warning(
                "Explanation SLA breach: request_id=%s user_id=%s "
                "deadline=%s provided_at=%s",
                request_id, req.user_id,
                req.sla_deadline.isoformat(),
                provided_at.isoformat(),
            )

    def list_pending_explanations(
        self, user_id: Optional[str] = None
    ) -> List[ComplianceRequest]:
        return [
            r for r in self._store.list_requests(
                user_id=user_id, status=RequestStatus.PENDING,
                request_type=RequestType.EXPLANATION,
            )
        ]
