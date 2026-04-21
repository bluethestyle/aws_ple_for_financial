"""
AI Decision Opt-Out
====================

Tracks customer opt-out preferences for AI-driven automatic decisions,
in compliance with AI기본법 제31조 (AI Basic Act, Article 31).

When a customer opts out:

* AI recommendations are suppressed for that customer.
* Fallback to rule-based, human-curated, or disabled recommendations.
* Full audit trail is maintained.

Storage: DynamoDB (production) or in-memory dict (testing/local).

DynamoDB Table Schema
---------------------

Partition key: ``customer_id`` (String)

Example items::

    {"customer_id": "C001", "opted_out": true,
     "fallback_type": "rule_based", "reason": "customer_request", ...}
"""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

__all__ = ["OptOutRecord", "AIDecisionOptOut"]

_VALID_FALLBACKS = {"rule_based", "human_review", "disable"}


@dataclass
class OptOutRecord:
    """AI decision opt-out state for a single customer."""

    customer_id: str
    opted_out: bool
    opted_out_at: Optional[str] = None   # ISO 8601
    reason: Optional[str] = None
    fallback_type: str = "rule_based"    # "rule_based" | "human_review" | "disable"


class AIDecisionOptOut:
    """AI automatic decision opt-out management (AI기본법 제31조).

    Customers have the right to opt out of AI-driven recommendations.
    When opted out, the system routes to a fallback recommendation
    method and records a full audit trail.

    Args:
        table_name: DynamoDB table name.
        region: AWS region. ``None`` lets boto3 resolve from env /
            credentials; callers should pass ``pipeline.yaml::aws.region``.
        audit_store: Optional callable ``(event_dict) -> None`` for audit
            logging.
        use_dynamo: If ``False``, use in-memory dict.  Defaults to ``True``;
            falls back to in-memory if boto3 is unavailable.
    """

    def __init__(
        self,
        table_name: str = "ple-ai-optout",
        region: Optional[str] = None,
        audit_store: Any = None,
        use_dynamo: bool = True,
    ) -> None:
        self._table_name = table_name
        self._region = region
        self._audit_store = audit_store
        self._table = None
        self._memory: Dict[str, Dict[str, Any]] = {}  # fallback store

        if use_dynamo:
            try:
                import boto3

                dynamodb = boto3.resource("dynamodb", region_name=region)
                self._table = dynamodb.Table(table_name)
                logger.info(
                    "AIDecisionOptOut: DynamoDB table=%s, region=%s",
                    table_name, region,
                )
            except Exception:
                logger.warning(
                    "AIDecisionOptOut: boto3 unavailable, using in-memory store",
                )
        else:
            logger.info("AIDecisionOptOut: using in-memory store")

    # ------------------------------------------------------------------
    # Read operations
    # ------------------------------------------------------------------

    def is_opted_out(self, customer_id: str) -> bool:
        """Check if customer has opted out of AI decisions."""
        record = self._get_record(customer_id)
        if record is None:
            return False
        return record.opted_out

    def get_fallback_type(self, customer_id: str) -> str:
        """Get the fallback recommendation method for an opted-out customer.

        Returns ``"rule_based"`` as the default if the customer has not
        opted out or has no explicit fallback preference.
        """
        record = self._get_record(customer_id)
        if record is None or not record.opted_out:
            return "rule_based"
        return record.fallback_type

    def get_opt_out_stats(self) -> Dict[str, int]:
        """Return aggregate opt-out statistics.

        Returns a dict with keys: ``total_opted_out``, ``rule_based``,
        ``human_review``, ``disable``.
        """
        stats = {"total_opted_out": 0, "rule_based": 0,
                 "human_review": 0, "disable": 0}

        if self._table is not None:
            return self._dynamo_get_stats()

        for item in self._memory.values():
            if item.get("opted_out", False):
                stats["total_opted_out"] += 1
                fb = item.get("fallback_type", "rule_based")
                if fb in stats:
                    stats[fb] += 1

        return stats

    # ------------------------------------------------------------------
    # Write operations
    # ------------------------------------------------------------------

    def opt_out(
        self,
        customer_id: str,
        reason: str = "",
        fallback_type: str = "rule_based",
    ) -> None:
        """Record customer AI opt-out.

        Args:
            customer_id: Customer identifier.
            reason: Free-text reason for opting out.
            fallback_type: One of ``rule_based``, ``human_review``, ``disable``.
        """
        if fallback_type not in _VALID_FALLBACKS:
            raise ValueError(
                f"Invalid fallback_type '{fallback_type}'. "
                f"Must be one of {_VALID_FALLBACKS}"
            )

        now = datetime.now(timezone.utc).isoformat()
        record = OptOutRecord(
            customer_id=customer_id,
            opted_out=True,
            opted_out_at=now,
            reason=reason,
            fallback_type=fallback_type,
        )

        self._put_record(record)
        self._audit("AI_OPT_OUT", customer_id, reason=reason,
                     fallback_type=fallback_type)
        logger.info(
            "AI opt-out recorded: customer=%s, fallback=%s, reason=%s",
            customer_id, fallback_type, reason,
        )

    def opt_in(self, customer_id: str) -> None:
        """Re-enable AI decisions for customer."""
        now = datetime.now(timezone.utc).isoformat()
        record = OptOutRecord(
            customer_id=customer_id,
            opted_out=False,
            opted_out_at=None,
            reason=None,
            fallback_type="rule_based",
        )

        self._put_record(record)
        self._audit("AI_OPT_IN", customer_id)
        logger.info("AI opt-in recorded: customer=%s", customer_id)

    # ------------------------------------------------------------------
    # Storage helpers
    # ------------------------------------------------------------------

    def _get_record(self, customer_id: str) -> Optional[OptOutRecord]:
        if self._table is not None:
            return self._dynamo_get(customer_id)

        item = self._memory.get(customer_id)
        if item is None:
            return None
        return OptOutRecord(
            customer_id=item["customer_id"],
            opted_out=item.get("opted_out", False),
            opted_out_at=item.get("opted_out_at"),
            reason=item.get("reason"),
            fallback_type=item.get("fallback_type", "rule_based"),
        )

    def _put_record(self, record: OptOutRecord) -> None:
        if self._table is not None:
            self._dynamo_put(record)
            return

        self._memory[record.customer_id] = asdict(record)

    def _dynamo_get(self, customer_id: str) -> Optional[OptOutRecord]:
        try:
            response = self._table.get_item(
                Key={"customer_id": customer_id},
                ConsistentRead=False,
            )
        except Exception:
            logger.exception(
                "AIDecisionOptOut: DynamoDB get failed customer=%s",
                customer_id,
            )
            return None

        item = response.get("Item")
        if item is None:
            return None

        return OptOutRecord(
            customer_id=str(item["customer_id"]),
            opted_out=bool(item.get("opted_out", False)),
            opted_out_at=item.get("opted_out_at"),
            reason=item.get("reason"),
            fallback_type=str(item.get("fallback_type", "rule_based")),
        )

    def _dynamo_put(self, record: OptOutRecord) -> None:
        item: Dict[str, Any] = {
            "customer_id": record.customer_id,
            "opted_out": record.opted_out,
            "fallback_type": record.fallback_type,
        }
        if record.opted_out_at:
            item["opted_out_at"] = record.opted_out_at
        if record.reason:
            item["reason"] = record.reason

        try:
            self._table.put_item(Item=item)
        except Exception:
            logger.exception("AIDecisionOptOut: DynamoDB put failed")
            raise

    def _dynamo_get_stats(self) -> Dict[str, int]:
        """Scan DynamoDB for aggregate opt-out statistics.

        Note: scan is expensive; in production, prefer a CloudWatch metric
        or a pre-aggregated counter.
        """
        stats = {"total_opted_out": 0, "rule_based": 0,
                 "human_review": 0, "disable": 0}
        try:
            response = self._table.scan(
                FilterExpression="opted_out = :t",
                ExpressionAttributeValues={":t": True},
                ProjectionExpression="fallback_type",
            )
            for item in response.get("Items", []):
                stats["total_opted_out"] += 1
                fb = str(item.get("fallback_type", "rule_based"))
                if fb in stats:
                    stats[fb] += 1

            # Handle pagination
            while "LastEvaluatedKey" in response:
                response = self._table.scan(
                    FilterExpression="opted_out = :t",
                    ExpressionAttributeValues={":t": True},
                    ProjectionExpression="fallback_type",
                    ExclusiveStartKey=response["LastEvaluatedKey"],
                )
                for item in response.get("Items", []):
                    stats["total_opted_out"] += 1
                    fb = str(item.get("fallback_type", "rule_based"))
                    if fb in stats:
                        stats[fb] += 1

        except Exception:
            logger.exception("AIDecisionOptOut: DynamoDB scan for stats failed")

        return stats

    # ------------------------------------------------------------------
    # Audit
    # ------------------------------------------------------------------

    def _audit(self, action: str, customer_id: str, **kwargs: Any) -> None:
        event = {
            "action": action,
            "customer_id": customer_id,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            **kwargs,
        }
        logger.info("AI_OPTOUT_AUDIT | %s", event)
        if self._audit_store is not None:
            try:
                self._audit_store(event)
            except Exception:
                logger.exception("AIDecisionOptOut: audit_store callback failed")
