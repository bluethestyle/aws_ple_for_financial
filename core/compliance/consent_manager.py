"""
Consent Manager
===============

Channel-level marketing consent management with night-time restriction
enforcement and Do-Not-Contact (DNC) registry.

Enforces:

* **Per-channel consent** -- SMS, email, app_push, third_party.
* **Night-time restriction** -- 21:00-08:00 KST per 금소법 (Financial
  Consumer Protection Act).
* **Do-Not-Contact registry** -- hard block regardless of consent.
* **Consent expiry** -- auto-revoke after configurable period.

Storage: DynamoDB (production) or in-memory dict (testing/local).

DynamoDB Table Schema
---------------------

Partition key: ``customer_id`` (String)
Sort key:      ``record_key`` (String)

Example items::

    {"customer_id": "C001", "record_key": "consent:sms",
     "consented": true, "source": "customer_portal", ...}
    {"customer_id": "C001", "record_key": "dnc",
     "blocked": true, "reason": "customer_request", ...}
"""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, Optional, Tuple

logger = logging.getLogger(__name__)

__all__ = ["ConsentRecord", "ConsentManager"]

# KST = UTC+9
_KST = timezone(timedelta(hours=9))

_VALID_CHANNELS = {"sms", "email", "app_push", "third_party"}
_VALID_SOURCES = {"customer_portal", "call_center", "branch", "auto"}

# Night-time restriction window (KST)
_NIGHT_START_HOUR = 21  # 21:00
_NIGHT_END_HOUR = 8     # 08:00


@dataclass
class ConsentRecord:
    """A single channel-level consent entry."""

    customer_id: str
    channel: str            # "sms" | "email" | "app_push" | "third_party"
    consented: bool
    updated_at: str         # ISO 8601
    source: str             # "customer_portal" | "call_center" | "branch" | "auto"
    expires_at: Optional[str] = None


class ConsentManager:
    """Channel-level marketing consent management.

    Args:
        table_name: DynamoDB table name.
        region: AWS region.
        audit_store: Optional callable ``(event_dict) -> None`` for audit
            logging.  When ``None``, audit events are only written to the
            Python logger.
        use_dynamo: If ``False``, use an in-memory dict instead of DynamoDB.
            Defaults to ``True``; automatically falls back to in-memory if
            boto3 is unavailable.
    """

    def __init__(
        self,
        table_name: str = "ple-consent",
        region: str = "ap-northeast-2",
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
                    "ConsentManager: DynamoDB table=%s, region=%s",
                    table_name, region,
                )
            except Exception:
                logger.warning(
                    "ConsentManager: boto3 unavailable, using in-memory store",
                )
        else:
            logger.info("ConsentManager: using in-memory store")

    # ------------------------------------------------------------------
    # Read operations
    # ------------------------------------------------------------------

    def check_consent(self, customer_id: str, channel: str) -> bool:
        """Check if customer has active (non-expired) consent for *channel*.

        Returns ``True`` if consent is granted and not expired.
        """
        self._validate_channel(channel)
        record = self._get_consent_record(customer_id, channel)
        if record is None:
            return False
        if not record.consented:
            return False

        # Check expiry
        if record.expires_at:
            try:
                expires = datetime.fromisoformat(record.expires_at)
                if datetime.now(timezone.utc) > expires:
                    logger.info(
                        "Consent expired: customer=%s, channel=%s",
                        customer_id, channel,
                    )
                    return False
            except ValueError:
                logger.warning(
                    "Invalid expires_at for customer=%s: %s",
                    customer_id, record.expires_at,
                )
        return True

    def check_nighttime(self, channel: str) -> bool:
        """Check if current KST time is within night restriction window.

        The restriction applies to outbound marketing channels (sms, email,
        app_push).  Returns ``True`` if sending is **BLOCKED**.
        """
        # Third-party is not subject to night-time restriction in this impl
        if channel == "third_party":
            return False

        now_kst = datetime.now(_KST)
        hour = now_kst.hour
        return hour >= _NIGHT_START_HOUR or hour < _NIGHT_END_HOUR

    def check_dnc(self, customer_id: str) -> bool:
        """Check if customer is on the Do-Not-Contact list.

        Returns ``True`` if the customer is **BLOCKED**.
        """
        if self._table is not None:
            return self._dynamo_check_dnc(customer_id)

        key = self._mem_key(customer_id, "dnc")
        item = self._memory.get(key)
        return bool(item and item.get("blocked", False))

    def is_contactable(
        self, customer_id: str, channel: str,
    ) -> Tuple[bool, str]:
        """Full contactability check: consent + nighttime + DNC.

        Returns:
            A tuple ``(contactable, reason_if_blocked)``.  When contactable
            is ``True``, the reason string is empty.
        """
        self._validate_channel(channel)

        # 1. DNC check (hard block)
        if self.check_dnc(customer_id):
            return False, "customer_on_dnc_list"

        # 2. Night-time restriction
        if self.check_nighttime(channel):
            return False, "nighttime_restriction_21_08_KST"

        # 3. Channel consent
        if not self.check_consent(customer_id, channel):
            return False, f"no_active_consent_for_{channel}"

        return True, ""

    def get_consent_status(
        self, customer_id: str,
    ) -> Dict[str, ConsentRecord]:
        """Get all channel consent records for a customer.

        Returns a dict keyed by channel name.
        """
        result: Dict[str, ConsentRecord] = {}
        for channel in _VALID_CHANNELS:
            record = self._get_consent_record(customer_id, channel)
            if record is not None:
                result[channel] = record
        return result

    # ------------------------------------------------------------------
    # Write operations
    # ------------------------------------------------------------------

    def grant_consent(
        self,
        customer_id: str,
        channel: str,
        source: str,
        expires_days: int = 365,
    ) -> None:
        """Grant marketing consent for *channel*.

        Args:
            customer_id: Customer identifier.
            channel: One of ``sms``, ``email``, ``app_push``, ``third_party``.
            source: Origin of the consent action.
            expires_days: Days until consent auto-expires.
        """
        self._validate_channel(channel)
        now = datetime.now(timezone.utc)
        expires_at = (now + timedelta(days=expires_days)).isoformat()

        record = ConsentRecord(
            customer_id=customer_id,
            channel=channel,
            consented=True,
            updated_at=now.isoformat(),
            source=source,
            expires_at=expires_at,
        )

        self._put_consent_record(record)
        self._audit("GRANT_CONSENT", customer_id, channel=channel, source=source)
        logger.info(
            "Consent granted: customer=%s, channel=%s, source=%s, expires=%s",
            customer_id, channel, source, expires_at,
        )

    def revoke_consent(
        self,
        customer_id: str,
        channel: str,
        source: str,
    ) -> None:
        """Revoke marketing consent for *channel*."""
        self._validate_channel(channel)
        now = datetime.now(timezone.utc).isoformat()

        record = ConsentRecord(
            customer_id=customer_id,
            channel=channel,
            consented=False,
            updated_at=now,
            source=source,
            expires_at=None,
        )

        self._put_consent_record(record)
        self._audit("REVOKE_CONSENT", customer_id, channel=channel, source=source)
        logger.info(
            "Consent revoked: customer=%s, channel=%s, source=%s",
            customer_id, channel, source,
        )

    def add_to_dnc(self, customer_id: str, reason: str) -> None:
        """Add customer to the Do-Not-Contact registry."""
        now = datetime.now(timezone.utc).isoformat()

        if self._table is not None:
            self._dynamo_put_dnc(customer_id, reason, now)
        else:
            key = self._mem_key(customer_id, "dnc")
            self._memory[key] = {
                "customer_id": customer_id,
                "record_key": "dnc",
                "blocked": True,
                "reason": reason,
                "updated_at": now,
            }

        self._audit("ADD_DNC", customer_id, reason=reason)
        logger.warning(
            "Customer added to DNC: customer=%s, reason=%s",
            customer_id, reason,
        )

    # ------------------------------------------------------------------
    # DynamoDB helpers
    # ------------------------------------------------------------------

    def _get_consent_record(
        self, customer_id: str, channel: str,
    ) -> Optional[ConsentRecord]:
        """Fetch a single consent record from the backing store."""
        if self._table is not None:
            return self._dynamo_get_consent(customer_id, channel)

        key = self._mem_key(customer_id, f"consent:{channel}")
        item = self._memory.get(key)
        if item is None:
            return None
        return ConsentRecord(
            customer_id=item["customer_id"],
            channel=item["channel"],
            consented=item["consented"],
            updated_at=item["updated_at"],
            source=item["source"],
            expires_at=item.get("expires_at"),
        )

    def _put_consent_record(self, record: ConsentRecord) -> None:
        """Write a consent record to the backing store."""
        if self._table is not None:
            self._dynamo_put_consent(record)
            return

        key = self._mem_key(record.customer_id, f"consent:{record.channel}")
        self._memory[key] = asdict(record)

    def _dynamo_get_consent(
        self, customer_id: str, channel: str,
    ) -> Optional[ConsentRecord]:
        try:
            response = self._table.get_item(
                Key={
                    "customer_id": customer_id,
                    "record_key": f"consent:{channel}",
                },
                ConsistentRead=False,
            )
        except Exception:
            logger.exception(
                "ConsentManager: DynamoDB get failed customer=%s channel=%s",
                customer_id, channel,
            )
            return None

        item = response.get("Item")
        if item is None:
            return None

        return ConsentRecord(
            customer_id=str(item["customer_id"]),
            channel=channel,
            consented=bool(item.get("consented", False)),
            updated_at=str(item.get("updated_at", "")),
            source=str(item.get("source", "")),
            expires_at=item.get("expires_at"),
        )

    def _dynamo_put_consent(self, record: ConsentRecord) -> None:
        item: Dict[str, Any] = {
            "customer_id": record.customer_id,
            "record_key": f"consent:{record.channel}",
            "channel": record.channel,
            "consented": record.consented,
            "updated_at": record.updated_at,
            "source": record.source,
        }
        if record.expires_at:
            item["expires_at"] = record.expires_at

        try:
            self._table.put_item(Item=item)
        except Exception:
            logger.exception("ConsentManager: DynamoDB put_consent failed")
            raise

    def _dynamo_check_dnc(self, customer_id: str) -> bool:
        try:
            response = self._table.get_item(
                Key={"customer_id": customer_id, "record_key": "dnc"},
                ConsistentRead=False,
            )
        except Exception:
            logger.exception(
                "ConsentManager: DynamoDB DNC check failed customer=%s",
                customer_id,
            )
            return False

        item = response.get("Item")
        return bool(item and item.get("blocked", False))

    def _dynamo_put_dnc(
        self, customer_id: str, reason: str, updated_at: str,
    ) -> None:
        try:
            self._table.put_item(Item={
                "customer_id": customer_id,
                "record_key": "dnc",
                "blocked": True,
                "reason": reason,
                "updated_at": updated_at,
            })
        except Exception:
            logger.exception("ConsentManager: DynamoDB put_dnc failed")
            raise

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    @staticmethod
    def _mem_key(customer_id: str, record_key: str) -> str:
        return f"{customer_id}##{record_key}"

    @staticmethod
    def _validate_channel(channel: str) -> None:
        if channel not in _VALID_CHANNELS:
            raise ValueError(
                f"Invalid channel '{channel}'. Must be one of {_VALID_CHANNELS}"
            )

    def _audit(self, action: str, customer_id: str, **kwargs: Any) -> None:
        """Emit an audit event."""
        event = {
            "action": action,
            "customer_id": customer_id,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            **kwargs,
        }
        logger.info("CONSENT_AUDIT | %s", event)
        if self._audit_store is not None:
            try:
                self._audit_store(event)
            except Exception:
                logger.exception("ConsentManager: audit_store callback failed")
