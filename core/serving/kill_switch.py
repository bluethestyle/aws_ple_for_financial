"""
Kill Switch
===========

DynamoDB-backed circuit breaker that can disable inference at three
granularity levels:

* **global** -- disable all inference.
* **per-task** -- disable a specific task (e.g. ``ctr``).
* **per-cluster** -- disable inference for a specific customer cluster.

The kill switch is checked on every request.  When tripped, the service
returns a fallback response according to the configured strategy:

* ``rule_based`` -- return a static / rule-based recommendation.
* ``previous_model`` -- fall back to the prior model version.
* ``disable`` -- return an empty result with an explanatory message.

All state transitions (activate / deactivate) are audit-logged to
CloudWatch.

DynamoDB Table Schema
---------------------

Partition key: ``switch_key`` (String)

Example items::

    {"switch_key": "global",         "active": true, "reason": "hotfix", ...}
    {"switch_key": "task:ctr",       "active": false, ...}
    {"switch_key": "cluster:VIP_A",  "active": true, "reason": "drift", ...}
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

__all__ = ["KillSwitch", "KillSwitchState", "FallbackStrategy"]


class FallbackStrategy(str, Enum):
    """What to do when the kill switch is active."""
    RULE_BASED = "rule_based"
    PREVIOUS_MODEL = "previous_model"
    DISABLE = "disable"


@dataclass
class KillSwitchState:
    """Snapshot of the kill switch for a particular scope.

    Attributes:
        active: ``True`` if the switch is tripped (inference blocked).
        scope: The scope that was checked (``global``, ``task:xxx``,
            ``cluster:xxx``).
        reason: Human-readable reason for activation.
        activated_at: ISO-8601 timestamp of activation (if active).
        activated_by: Identity that activated the switch.
        fallback_strategy: Configured fallback.
    """

    active: bool = False
    scope: str = "global"
    reason: str = ""
    activated_at: str = ""
    activated_by: str = ""
    fallback_strategy: FallbackStrategy = FallbackStrategy.DISABLE
    metadata: Dict[str, Any] = field(default_factory=dict)


class KillSwitch:
    """DynamoDB-backed kill switch checked on every inference request.

    Args:
        table_name: DynamoDB table name.
        fallback_strategy: Default fallback when the switch fires.
        region: AWS region.
    """

    def __init__(
        self,
        table_name: str = "ple-kill-switch",
        fallback_strategy: str = "rule_based",
        region: str = "ap-northeast-2",
        audit_store=None,
    ) -> None:
        import boto3

        self._table_name = table_name
        self._fallback = FallbackStrategy(fallback_strategy)
        self._region = region
        self._audit_store = audit_store

        dynamodb = boto3.resource("dynamodb", region_name=region)
        self._table = dynamodb.Table(table_name)

        logger.info(
            "KillSwitch: table=%s, fallback=%s", table_name, fallback_strategy,
        )

    # ------------------------------------------------------------------
    # Read (per-request hot path)
    # ------------------------------------------------------------------

    def check(
        self,
        task: Optional[str] = None,
        cluster: Optional[str] = None,
    ) -> KillSwitchState:
        """Check whether inference is disabled.

        The check order is: **global** -> **task** -> **cluster**.
        The first active switch wins.

        Args:
            task: Optional task name (e.g. ``"ctr"``).
            cluster: Optional customer cluster (e.g. ``"VIP_A"``).

        Returns:
            :class:`KillSwitchState` -- ``active=True`` means the caller
            should NOT run inference.
        """
        # Check in priority order
        keys_to_check = ["global"]
        if task:
            keys_to_check.append(f"task:{task}")
        if cluster:
            keys_to_check.append(f"cluster:{cluster}")

        for switch_key in keys_to_check:
            state = self._get_switch(switch_key)
            if state.active:
                logger.warning(
                    "KillSwitch ACTIVE: scope=%s, reason=%s",
                    state.scope, state.reason,
                )
                return state

        return KillSwitchState(active=False, scope="none")

    def _get_switch(self, switch_key: str) -> KillSwitchState:
        """Fetch a single switch entry from DynamoDB."""
        try:
            response = self._table.get_item(
                Key={"switch_key": switch_key},
                ConsistentRead=False,
            )
        except Exception:
            logger.exception(
                "KillSwitch: failed to read key=%s, defaulting to inactive",
                switch_key,
            )
            return KillSwitchState(active=False, scope=switch_key)

        item = response.get("Item")
        if item is None:
            return KillSwitchState(active=False, scope=switch_key)

        return KillSwitchState(
            active=bool(item.get("active", False)),
            scope=switch_key,
            reason=str(item.get("reason", "")),
            activated_at=str(item.get("activated_at", "")),
            activated_by=str(item.get("activated_by", "")),
            fallback_strategy=self._fallback,
            metadata={
                k: v for k, v in item.items()
                if k not in {"switch_key", "active", "reason",
                             "activated_at", "activated_by"}
            },
        )

    # ------------------------------------------------------------------
    # Write (admin operations)
    # ------------------------------------------------------------------

    def activate(
        self,
        scope: str = "global",
        reason: str = "",
        activated_by: str = "system",
    ) -> KillSwitchState:
        """Activate (trip) the kill switch for a given scope.

        Args:
            scope: One of ``"global"``, ``"task:<name>"``, or
                ``"cluster:<name>"``.
            reason: Human-readable explanation.
            activated_by: Identity of the actor (for audit).

        Returns:
            The new :class:`KillSwitchState`.
        """
        from datetime import datetime, timezone

        now = datetime.now(timezone.utc).isoformat()

        item = {
            "switch_key": scope,
            "active": True,
            "reason": reason,
            "activated_at": now,
            "activated_by": activated_by,
        }

        try:
            self._table.put_item(Item=item)
        except Exception:
            logger.exception("KillSwitch.activate failed: scope=%s", scope)
            raise

        logger.warning(
            "KillSwitch ACTIVATED: scope=%s, reason=%s, by=%s",
            scope, reason, activated_by,
        )
        self._emit_audit_log("ACTIVATE", scope, reason, activated_by, now)

        if self._audit_store:
            level = scope.split(":")[0] if ":" in scope else "global"
            self._audit_store.log_killswitch(
                action="ACTIVATE", level=level, target=scope,
                reason=reason, actor=activated_by,
            )

        return KillSwitchState(
            active=True,
            scope=scope,
            reason=reason,
            activated_at=now,
            activated_by=activated_by,
            fallback_strategy=self._fallback,
        )

    def deactivate(
        self,
        scope: str = "global",
        deactivated_by: str = "system",
    ) -> KillSwitchState:
        """Deactivate the kill switch for a given scope.

        The DynamoDB item is updated (not deleted) so the audit trail
        is preserved.

        Args:
            scope: Kill switch scope to deactivate.
            deactivated_by: Identity of the actor (for audit).

        Returns:
            The new :class:`KillSwitchState` (``active=False``).
        """
        from datetime import datetime, timezone

        now = datetime.now(timezone.utc).isoformat()

        try:
            self._table.update_item(
                Key={"switch_key": scope},
                UpdateExpression=(
                    "SET active = :a, deactivated_at = :t, "
                    "deactivated_by = :b"
                ),
                ExpressionAttributeValues={
                    ":a": False,
                    ":t": now,
                    ":b": deactivated_by,
                },
            )
        except Exception:
            logger.exception("KillSwitch.deactivate failed: scope=%s", scope)
            raise

        logger.info(
            "KillSwitch DEACTIVATED: scope=%s, by=%s", scope, deactivated_by,
        )
        self._emit_audit_log("DEACTIVATE", scope, "", deactivated_by, now)

        if self._audit_store:
            level = scope.split(":")[0] if ":" in scope else "global"
            self._audit_store.log_killswitch(
                action="DEACTIVATE", level=level, target=scope,
                reason="", actor=deactivated_by,
            )

        return KillSwitchState(active=False, scope=scope)

    # ------------------------------------------------------------------
    # Audit logging
    # ------------------------------------------------------------------

    @staticmethod
    def _emit_audit_log(
        action: str,
        scope: str,
        reason: str,
        actor: str,
        timestamp: str,
    ) -> None:
        """Emit a structured audit log entry.

        In production this would go to CloudWatch Logs via a dedicated
        log group with a metric filter for alerting.
        """
        logger.info(
            "KILL_SWITCH_AUDIT | action=%s | scope=%s | reason=%s | "
            "actor=%s | timestamp=%s",
            action, scope, reason, actor, timestamp,
        )
