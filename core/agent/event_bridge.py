"""
Agent Event Bridge — Inter-Agent Trigger System
==================================================

Enables Ops and Audit agents to trigger each other based on findings.

Trigger rules:
    - Ops → Audit: drift critical 3 days → reason quality check on degraded segment
    - Audit → Ops: segment bias found → enhanced CP6 monitoring
    - Ops → Audit: latency SLA exceeded → check reason generation skip
    - Audit → Ops: regulatory critical failure → immediate pipeline stage check
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)

__all__ = ["AgentEventBridge", "AgentTrigger"]


@dataclass
class AgentTrigger:
    """A trigger event from one agent to another."""
    source_agent: str      # "ops" or "audit"
    target_agent: str      # "ops" or "audit"
    trigger_type: str      # e.g., "drift_critical", "bias_found"
    message: str           # human-readable description
    affected_parts: List[str] = field(default_factory=list)
    data: Dict[str, Any] = field(default_factory=dict)
    timestamp: str = ""

    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = datetime.now(timezone.utc).isoformat()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "source_agent": self.source_agent,
            "target_agent": self.target_agent,
            "trigger_type": self.trigger_type,
            "message": self.message,
            "affected_parts": self.affected_parts,
            "data": self.data,
            "timestamp": self.timestamp,
        }


class AgentEventBridge:
    """Routes trigger events between Ops and Audit agents.

    Usage:
        bridge = AgentEventBridge()
        bridge.subscribe("audit", on_audit_trigger)
        bridge.emit(AgentTrigger(source_agent="ops", target_agent="audit", ...))
    """

    def __init__(self) -> None:
        self._subscribers: Dict[str, List[Callable[[AgentTrigger], None]]] = {
            "ops": [],
            "audit": [],
        }
        self._trigger_history: List[AgentTrigger] = []

    def subscribe(self, agent: str, callback: Callable[[AgentTrigger], None]) -> None:
        """Register a callback for an agent to receive triggers."""
        if agent not in self._subscribers:
            self._subscribers[agent] = []
        self._subscribers[agent].append(callback)

    def emit(self, trigger: AgentTrigger) -> None:
        """Emit a trigger to the target agent."""
        self._trigger_history.append(trigger)

        target = trigger.target_agent
        callbacks = self._subscribers.get(target, [])

        logger.info(
            "Trigger: %s → %s [%s] %s",
            trigger.source_agent, target, trigger.trigger_type, trigger.message,
        )

        for cb in callbacks:
            try:
                cb(trigger)
            except Exception as e:
                logger.error("Trigger callback failed for %s: %s", target, e)

    def emit_ops_to_audit(self, trigger_type: str, message: str, **kwargs) -> AgentTrigger:
        """Convenience: Ops → Audit trigger."""
        trigger = AgentTrigger(
            source_agent="ops",
            target_agent="audit",
            trigger_type=trigger_type,
            message=message,
            **kwargs,
        )
        self.emit(trigger)
        return trigger

    def emit_audit_to_ops(self, trigger_type: str, message: str, **kwargs) -> AgentTrigger:
        """Convenience: Audit → Ops trigger."""
        trigger = AgentTrigger(
            source_agent="audit",
            target_agent="ops",
            trigger_type=trigger_type,
            message=message,
            **kwargs,
        )
        self.emit(trigger)
        return trigger

    @property
    def trigger_count(self) -> int:
        return len(self._trigger_history)

    def get_history(
        self,
        source: Optional[str] = None,
        target: Optional[str] = None,
    ) -> List[AgentTrigger]:
        """Get trigger history, optionally filtered."""
        history = self._trigger_history
        if source:
            history = [t for t in history if t.source_agent == source]
        if target:
            history = [t for t in history if t.target_agent == target]
        return history
