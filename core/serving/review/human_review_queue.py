"""
HumanReviewQueue - Sprint 3 M1.

3-tier review queue. The caller (predict.py) tags each recommendation with
its tier, the queue decides whether to enqueue, and reviewers dequeue +
dispose (approve / reject / modify) asynchronously.

Backends:
- In-memory (tests / local dev)
- DynamoDB (production; Sprint 4 wire-in)

All state transitions emit an audit event so an auditor can reconstruct
the disposition of every tier-2/3 recommendation that entered production.
"""

from __future__ import annotations

import json
import logging
import random
import threading
import uuid
from collections import deque
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any, Deque, Dict, List, Optional

logger = logging.getLogger(__name__)

__all__ = [
    "ReviewConfig",
    "ReviewItem",
    "ReviewState",
    "HumanReviewQueue",
    "build_human_review_queue",
]


class ReviewState:
    PENDING = "pending"
    IN_REVIEW = "in_review"
    APPROVED = "approved"
    REJECTED = "rejected"
    MODIFIED = "modified"
    EXPIRED = "expired"

    VALID = frozenset({PENDING, IN_REVIEW, APPROVED, REJECTED,
                       MODIFIED, EXPIRED})


@dataclass
class ReviewConfig:
    """Config-driven tiering policy. Read from ``serving.review`` block."""

    tier_1_sample_rate: float = 0.05       # 5% post-hoc sample
    tier_2_review_required: bool = True    # 100% agent review
    tier_3_human_fallback: bool = True     # must go to HumanFallback
    queue_backend: str = "in_memory"       # in_memory | dynamodb
    dynamodb_table: str = "ple-review-queue"
    region: str = "ap-northeast-2"
    sla_hours: int = 24                    # how long a reviewer has

    def __post_init__(self) -> None:
        if not (0.0 <= self.tier_1_sample_rate <= 1.0):
            raise ValueError(
                f"tier_1_sample_rate={self.tier_1_sample_rate} "
                f"must be in [0.0, 1.0]"
            )
        if self.sla_hours <= 0:
            raise ValueError("sla_hours must be > 0")

    @classmethod
    def from_dict(cls, data: Optional[Dict[str, Any]]) -> "ReviewConfig":
        if not data:
            return cls()
        return cls(
            tier_1_sample_rate=float(data.get("tier_1_sample_rate", 0.05)),
            tier_2_review_required=bool(
                data.get("tier_2_review_required", True)
            ),
            tier_3_human_fallback=bool(data.get("tier_3_human_fallback", True)),
            queue_backend=str(data.get("queue_backend", "in_memory")),
            dynamodb_table=str(data.get("dynamodb_table", "ple-review-queue")),
            region=str(data.get("region", "ap-northeast-2")),
            sla_hours=int(data.get("sla_hours", 24)),
        )


@dataclass
class ReviewItem:
    review_id: str
    user_id: str
    recommendation_id: str
    tier: int                               # 1 / 2 / 3
    state: str = ReviewState.PENDING
    created_at: str = ""
    payload: Dict[str, Any] = field(default_factory=dict)
    reviewer_id: Optional[str] = None
    disposition_at: Optional[str] = None
    disposition_reason: Optional[str] = None
    modifications: Optional[Dict[str, Any]] = None

    def __post_init__(self) -> None:
        if self.tier not in (1, 2, 3):
            raise ValueError(f"tier={self.tier} must be 1 / 2 / 3")
        if self.state not in ReviewState.VALID:
            raise ValueError(f"state={self.state!r} invalid")
        if not self.created_at:
            self.created_at = datetime.now(timezone.utc).isoformat()

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        # payload / modifications may contain nested JSON-incompatible types;
        # keep as-is but guarantee serialisability at flush time
        return d

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, default=str)


# ---------------------------------------------------------------------------
# Queue
# ---------------------------------------------------------------------------

class HumanReviewQueue:
    """Tier-aware human review queue with pluggable backends."""

    def __init__(
        self,
        config: Optional[ReviewConfig] = None,
        audit_callback: Optional[Any] = None,
        seed: Optional[int] = None,
    ) -> None:
        self._cfg = config or ReviewConfig()
        self._audit = audit_callback     # callable(dict) -> None
        self._rng = random.Random(seed)

        self._items: Dict[str, ReviewItem] = {}
        self._pending_by_tier: Dict[int, Deque[str]] = {
            1: deque(), 2: deque(), 3: deque(),
        }
        self._lock = threading.RLock()

    # ------------------------------------------------------------------
    # Decision: should we enqueue?
    # ------------------------------------------------------------------

    def should_enqueue(self, tier: int) -> bool:
        """Determine whether a tier-N recommendation must be queued."""
        if tier == 3 and self._cfg.tier_3_human_fallback:
            return True
        if tier == 2 and self._cfg.tier_2_review_required:
            return True
        if tier == 1:
            return self._rng.random() < self._cfg.tier_1_sample_rate
        return False

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def enqueue(
        self,
        user_id: str,
        recommendation_id: str,
        tier: int,
        payload: Optional[Dict[str, Any]] = None,
    ) -> Optional[ReviewItem]:
        """Enqueue a recommendation for review if policy requires it."""
        if not self.should_enqueue(tier):
            return None
        item = ReviewItem(
            review_id=f"rev_{uuid.uuid4().hex}",
            user_id=user_id,
            recommendation_id=recommendation_id,
            tier=int(tier),
            payload=dict(payload or {}),
        )
        with self._lock:
            self._items[item.review_id] = item
            self._pending_by_tier[item.tier].append(item.review_id)
        self._emit_audit("human_review:enqueue", item)
        logger.info(
            "Review enqueued: review_id=%s tier=%d user=%s rec=%s",
            item.review_id, item.tier, user_id, recommendation_id,
        )
        return item

    def dequeue(
        self, reviewer_id: str, tier: int,
    ) -> Optional[ReviewItem]:
        """Dequeue the next PENDING item for a reviewer (if any)."""
        if tier not in (1, 2, 3):
            raise ValueError(f"tier={tier} must be 1 / 2 / 3")
        with self._lock:
            queue = self._pending_by_tier[tier]
            while queue:
                rid = queue.popleft()
                item = self._items.get(rid)
                if item is None:
                    continue
                if item.state != ReviewState.PENDING:
                    continue
                item.state = ReviewState.IN_REVIEW
                item.reviewer_id = reviewer_id
                self._emit_audit("human_review:dequeue", item)
                logger.info(
                    "Review dequeued: review_id=%s tier=%d reviewer=%s",
                    item.review_id, tier, reviewer_id,
                )
                return item
        return None

    def approve(
        self, review_id: str, reviewer_id: str, reason: str = "",
    ) -> ReviewItem:
        return self._dispose(
            review_id, reviewer_id, ReviewState.APPROVED, reason,
        )

    def reject(
        self, review_id: str, reviewer_id: str, reason: str,
    ) -> ReviewItem:
        if not reason:
            raise ValueError("reject reason is required")
        return self._dispose(
            review_id, reviewer_id, ReviewState.REJECTED, reason,
        )

    def modify(
        self,
        review_id: str,
        reviewer_id: str,
        modifications: Dict[str, Any],
        reason: str = "",
    ) -> ReviewItem:
        if not modifications:
            raise ValueError("modify requires a non-empty modifications dict")
        with self._lock:
            item = self._items.get(review_id)
            if item is None:
                raise KeyError(f"Unknown review_id={review_id!r}")
            if item.state not in (ReviewState.PENDING, ReviewState.IN_REVIEW):
                raise ValueError(
                    f"Cannot modify review in state={item.state!r}"
                )
            item.modifications = dict(modifications)
        return self._dispose(
            review_id, reviewer_id, ReviewState.MODIFIED, reason,
        )

    def get(self, review_id: str) -> Optional[ReviewItem]:
        with self._lock:
            return self._items.get(review_id)

    def list_pending(
        self, tier: Optional[int] = None,
    ) -> List[ReviewItem]:
        with self._lock:
            items = list(self._items.values())
        items = [i for i in items if i.state == ReviewState.PENDING]
        if tier is not None:
            items = [i for i in items if i.tier == tier]
        items.sort(key=lambda i: i.created_at)
        return items

    def summary(self) -> Dict[str, Any]:
        with self._lock:
            items = list(self._items.values())
        by_state: Dict[str, int] = {}
        by_tier: Dict[int, int] = {}
        for it in items:
            by_state[it.state] = by_state.get(it.state, 0) + 1
            by_tier[it.tier] = by_tier.get(it.tier, 0) + 1
        return {
            "total": len(items),
            "by_state": by_state,
            "by_tier": by_tier,
        }

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _dispose(
        self, review_id: str, reviewer_id: str,
        state: str, reason: str,
    ) -> ReviewItem:
        if state not in (ReviewState.APPROVED, ReviewState.REJECTED,
                          ReviewState.MODIFIED):
            raise ValueError(f"Invalid disposition state={state!r}")
        now = datetime.now(timezone.utc).isoformat()
        with self._lock:
            item = self._items.get(review_id)
            if item is None:
                raise KeyError(f"Unknown review_id={review_id!r}")
            if item.state in (ReviewState.APPROVED, ReviewState.REJECTED,
                               ReviewState.MODIFIED, ReviewState.EXPIRED):
                raise ValueError(
                    f"Review already disposed: state={item.state!r}"
                )
            item.state = state
            item.reviewer_id = reviewer_id
            item.disposition_at = now
            item.disposition_reason = reason
        self._emit_audit(f"human_review:{state}", item)
        logger.info(
            "Review %s: review_id=%s reviewer=%s reason=%s",
            state, review_id, reviewer_id, reason,
        )
        return item

    def _emit_audit(self, action: str, item: ReviewItem) -> None:
        if self._audit is None:
            return
        try:
            self._audit({
                "action": action,
                "review_id": item.review_id,
                "user_id": item.user_id,
                "recommendation_id": item.recommendation_id,
                "tier": item.tier,
                "state": item.state,
                "reviewer_id": item.reviewer_id,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            })
        except Exception:
            logger.exception(
                "HumanReviewQueue audit callback failed for action=%s", action,
            )


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def build_human_review_queue(
    pipeline_config: Optional[Dict[str, Any]] = None,
    audit_callback: Optional[Any] = None,
) -> HumanReviewQueue:
    """Instantiate from the ``serving.review`` block of pipeline.yaml."""
    serving = (pipeline_config or {}).get("serving") or {}
    review_cfg = ReviewConfig.from_dict(serving.get("review"))
    return HumanReviewQueue(config=review_cfg, audit_callback=audit_callback)
