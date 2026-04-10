"""
Heartbeat Scheduler — Self-Scheduling Agent Loop
===================================================

PaperClip-inspired heartbeat pattern: agents wake periodically,
check if action is needed, and skip if nothing changed.

Decision logic per checkpoint:
    1. Time since last run > configured interval? → RUN
    2. Pending change events for this checkpoint's parts? → RUN
    3. Neither? → HEARTBEAT_OK (skip, no tool calls)

Integrates with BudgetTracker for graceful degradation:
    Budget HARD_STOP → SKIPPED_BUDGET (rule engine only)
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from core.agent.base import BaseAgent, AgentReport
    from core.agent.budget import BudgetTracker
    from core.agent.change_detector import ChangeDetector, ChangeEvent

logger = logging.getLogger(__name__)

__all__ = ["HeartbeatScheduler", "HeartbeatResult"]


# Interval parsing
_INTERVAL_MAP = {
    "1min": 60,
    "5min": 300,
    "10min": 600,
    "30min": 1800,
    "1h": 3600,
    "6h": 21600,
    "daily": 86400,
    "weekly": 604800,
}


@dataclass
class HeartbeatResult:
    """Result of a single heartbeat tick."""
    agent_id: str
    checkpoint: str
    outcome: str            # "RAN", "HEARTBEAT_OK", "SKIPPED_BUDGET"
    report: Optional[Any] = None  # AgentReport if RAN
    timestamp: str = ""
    next_check_at: str = ""

    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = datetime.now(timezone.utc).isoformat()

    def to_dict(self) -> Dict[str, Any]:
        d = {
            "agent_id": self.agent_id,
            "checkpoint": self.checkpoint,
            "outcome": self.outcome,
            "timestamp": self.timestamp,
        }
        if self.report:
            d["report_status"] = self.report.status if hasattr(self.report, "status") else "unknown"
        return d


class HeartbeatScheduler:
    """Self-scheduling loop for ops/audit agents.

    Args:
        agent: The BaseAgent instance (ops or audit).
        change_detector: For checking pending change events.
        config: Heartbeat config from agent.yaml.
        budget_tracker: For budget-aware degraded mode.
        state_path: JSON file for last-run timestamps.
    """

    def __init__(
        self,
        agent: "BaseAgent",
        change_detector: Optional["ChangeDetector"] = None,
        config: Optional[Dict[str, Any]] = None,
        budget_tracker: Optional["BudgetTracker"] = None,
        state_path: str = "outputs/heartbeat_state.json",
    ) -> None:
        self._agent = agent
        self._change_detector = change_detector
        self._config = config or {}
        self._budget_tracker = budget_tracker
        self._state_path = Path(state_path)
        self._pending_events_cache: List[Any] = []

        # Parse intervals from config
        self._intervals = {}
        for cp, interval_str in self._config.get("intervals", {}).items():
            self._intervals[cp] = self._parse_interval(interval_str)

        # Checkpoint → pipeline parts mapping
        self._checkpoint_parts = self._config.get("checkpoint_parts", {})

    def tick(self, checkpoint: str) -> HeartbeatResult:
        """Single heartbeat tick for one checkpoint.

        Returns HEARTBEAT_OK if no action needed, RAN if checklist executed.
        """
        agent_id = self._agent.agent_type
        state = self._load_state()
        interval = self._intervals.get(checkpoint, float("inf"))
        last_run = state.get(checkpoint)

        # Check if action needed
        time_elapsed = self._needs_run(interval, last_run)
        pending = self._get_pending_for_checkpoint(checkpoint)

        if not time_elapsed and not pending:
            return HeartbeatResult(
                agent_id=agent_id,
                checkpoint=checkpoint,
                outcome="HEARTBEAT_OK",
            )

        # Budget check
        if self._budget_tracker:
            budget_status = self._budget_tracker.check_budget(agent_id)
            if budget_status.status == "HARD_STOP":
                logger.warning("Heartbeat %s: budget hard stop", checkpoint)
                return HeartbeatResult(
                    agent_id=agent_id,
                    checkpoint=checkpoint,
                    outcome="SKIPPED_BUDGET",
                )

        # Run agent for this checkpoint's parts
        parts = self._checkpoint_parts.get(checkpoint, [])
        report = self._agent.heartbeat_run(
            parts=parts if parts else None,
            budget_tracker=self._budget_tracker,
        )

        # Update state
        state[checkpoint] = datetime.now(timezone.utc).isoformat()
        self._save_state(state)

        logger.info(
            "Heartbeat %s: RAN (status=%s, parts=%s)",
            checkpoint, report.status, parts,
        )

        return HeartbeatResult(
            agent_id=agent_id,
            checkpoint=checkpoint,
            outcome="RAN",
            report=report,
        )

    def tick_all(self) -> List[HeartbeatResult]:
        """Tick all configured checkpoints.

        Drains pending change events once, then distributes to checkpoints.
        """
        # Drain pending events once
        if self._change_detector:
            self._pending_events_cache = self._change_detector.get_pending_events()
        else:
            self._pending_events_cache = []

        results = []
        for checkpoint in self._intervals:
            result = self.tick(checkpoint)
            results.append(result)

        # Clear cache after full cycle
        self._pending_events_cache = []

        # Summary log
        ran = sum(1 for r in results if r.outcome == "RAN")
        skipped = sum(1 for r in results if r.outcome == "HEARTBEAT_OK")
        budget_skip = sum(1 for r in results if r.outcome == "SKIPPED_BUDGET")
        logger.info(
            "Heartbeat cycle: %d RAN, %d HEARTBEAT_OK, %d SKIPPED_BUDGET",
            ran, skipped, budget_skip,
        )

        return results

    def _needs_run(self, interval_seconds: float, last_run: Optional[str]) -> bool:
        """Check if enough time has elapsed since last run."""
        if interval_seconds == float("inf"):
            return False  # event-only, never by timer
        if last_run is None:
            return True  # never run before
        try:
            last_dt = datetime.fromisoformat(last_run)
            elapsed = (datetime.now(timezone.utc) - last_dt).total_seconds()
            return elapsed >= interval_seconds
        except (ValueError, TypeError):
            return True

    def _get_pending_for_checkpoint(self, checkpoint: str) -> List[Any]:
        """Get pending change events relevant to this checkpoint."""
        parts = set(self._checkpoint_parts.get(checkpoint, []))
        if not parts:
            return []
        return [
            ev for ev in self._pending_events_cache
            if any(p in parts for p in getattr(ev, "affected_parts", []))
        ]

    def _load_state(self) -> Dict[str, str]:
        if self._state_path.exists():
            try:
                with open(self._state_path, encoding="utf-8") as f:
                    return json.load(f)
            except (json.JSONDecodeError, KeyError):
                pass
        return {}

    def _save_state(self, state: Dict[str, str]) -> None:
        self._state_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self._state_path, "w", encoding="utf-8") as f:
            json.dump(state, f, ensure_ascii=False, indent=2)

    @staticmethod
    def _parse_interval(s: str) -> float:
        """Parse interval string to seconds."""
        s = s.strip().lower()
        if s in _INTERVAL_MAP:
            return _INTERVAL_MAP[s]
        if s == "event":
            return float("inf")
        # Try numeric seconds
        try:
            return float(s)
        except ValueError:
            logger.warning("Unknown interval '%s', defaulting to daily", s)
            return 86400
