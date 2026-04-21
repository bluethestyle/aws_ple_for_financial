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
   -> opens a 10-day-SLA explanation request
4. mark_explanation_provided(request_id, explanation)
   -> closes the request, emits REQUEST_PROCESSED event
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
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
    explanation_sla_days: int = 10        # 개보법 시행령 §44의2~4
    opt_out_response_days: int = 30       # 개보법 §37의2

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
    # Explanation request (개보법 §44의2~4)
    # ------------------------------------------------------------------

    def request_explanation(
        self,
        user_id: str,
        recommendation_id: str,
        reason: str = "",
    ) -> ComplianceRequest:
        """Open a 10-day-SLA explanation request."""
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
                "legal_basis": "개보법 시행령 §44의2~4",
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

    def mark_explanation_provided(
        self,
        request_id: str,
        explanation: str,
        provided_at: Optional[datetime] = None,
    ) -> None:
        """Close an explanation request with the actual explanation text."""
        provided_at = provided_at or utcnow()
        req = self._store.get_request(request_id)
        if req is None:
            raise KeyError(f"Unknown request_id={request_id!r}")
        if req.request_type != RequestType.EXPLANATION:
            raise ValueError(
                f"request_id={request_id!r} is type={req.request_type!r}, "
                f"not {RequestType.EXPLANATION}"
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
                    "sla_name": "explanation",
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
                        "sla_name": "explanation",
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
