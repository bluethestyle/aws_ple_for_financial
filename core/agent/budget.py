"""
Budget Tracker — Per-Agent Token Budget Management
=====================================================

PaperClip-inspired "prepaid debit card" model:
    - Per-agent monthly token limit
    - 80% soft warning → notification
    - 100% hard stop → LLM calls blocked, rule engine continues
    - Manual reset by operator
    - JSON persistence for restart survival
    - Automatic monthly rollover

When budget is exceeded, the system degrades gracefully to
"on-prem mode" — rule engine continues, only LLM calls are blocked.
"""

from __future__ import annotations

import json
import logging
import os
import threading
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

__all__ = ["BudgetTracker", "BudgetStatus"]


@dataclass
class BudgetStatus:
    """Current budget status for an agent."""
    agent_id: str
    tokens_used: int
    token_limit: int         # 0 = unlimited
    pct_used: float          # 0.0 - 1.0+
    status: str              # "OK", "WARN", "HARD_STOP"
    degraded_mode: bool      # True when HARD_STOP

    def to_dict(self) -> Dict[str, Any]:
        return {
            "agent_id": self.agent_id,
            "tokens_used": self.tokens_used,
            "token_limit": self.token_limit,
            "pct_used": round(self.pct_used, 4),
            "status": self.status,
            "degraded_mode": self.degraded_mode,
        }


class BudgetTracker:
    """Tracks per-agent token usage against monthly budgets.

    Args:
        config: Budget config from agent.yaml. Structure:
            ops: {monthly_token_limit: int, soft_warning_pct: float, hard_stop_pct: float}
            audit: {monthly_token_limit: int, ...}
            consensus: {per_session_limit: int, daily_limit: int}
        storage_path: JSON file for persistence.
    """

    def __init__(
        self,
        config: Optional[Dict[str, Any]] = None,
        storage_path: str = "outputs/budget_state.json",
    ) -> None:
        self._config = config or {}
        self._storage_path = Path(storage_path)
        self._lock = threading.RLock()
        self._state = self._load_state()

    def _current_period(self) -> str:
        return datetime.now(timezone.utc).strftime("%Y-%m")

    def _current_date(self) -> str:
        return datetime.now(timezone.utc).strftime("%Y-%m-%d")

    def _load_state(self) -> Dict[str, Any]:
        if self._storage_path.exists():
            try:
                with open(self._storage_path, encoding="utf-8") as f:
                    state = json.load(f)
                # Auto-rollover if month changed
                if state.get("period") != self._current_period():
                    logger.info(
                        "Budget period rollover: %s → %s",
                        state.get("period"),
                        self._current_period(),
                    )
                    return self._fresh_state()
                return state
            except (json.JSONDecodeError, KeyError):
                pass
        return self._fresh_state()

    def _fresh_state(self) -> Dict[str, Any]:
        return {
            "period": self._current_period(),
            "agents": {},
            "consensus": {"daily": {"tokens_used": 0, "date": self._current_date()}},
        }

    def _save_state(self) -> None:
        self._storage_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self._storage_path.with_suffix(".tmp")
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(self._state, f, ensure_ascii=False, indent=2)
        os.replace(str(tmp), str(self._storage_path))

    def record_tokens(self, agent_id: str, input_tokens: int, output_tokens: int) -> BudgetStatus:
        """Record token usage and return updated status.

        Thread-safe. Persists to disk after each update.
        """
        total = input_tokens + output_tokens
        with self._lock:
            agents = self._state.setdefault("agents", {})
            if agent_id not in agents:
                agents[agent_id] = {
                    "tokens_used": 0,
                    "last_reset": datetime.now(timezone.utc).isoformat(),
                }
            agents[agent_id]["tokens_used"] += total
            self._save_state()
            return self.check_budget(agent_id)

    def check_budget(self, agent_id: str) -> BudgetStatus:
        """Check current budget status for an agent. Read-only (no side effects)."""
        with self._lock:
            agent_cfg = self._config.get(agent_id, {})
            limit = agent_cfg.get("monthly_token_limit", 0)
            soft_pct = agent_cfg.get("soft_warning_pct", 0.80)
            hard_pct = agent_cfg.get("hard_stop_pct", 1.00)

            agents = self._state.get("agents", {})
            used = agents.get(agent_id, {}).get("tokens_used", 0)

            if limit == 0:
                return BudgetStatus(agent_id, used, 0, 0.0, "OK", False)

            pct = used / limit

            if pct >= hard_pct:
                status = "HARD_STOP"
            elif pct >= soft_pct:
                status = "WARN"
            else:
                status = "OK"

            return BudgetStatus(
                agent_id=agent_id,
                tokens_used=used,
                token_limit=limit,
                pct_used=pct,
                status=status,
                degraded_mode=(status == "HARD_STOP"),
            )

    def reset_agent(self, agent_id: str) -> None:
        """Manual reset by operator."""
        with self._lock:
            agents = self._state.setdefault("agents", {})
            agents[agent_id] = {
                "tokens_used": 0,
                "last_reset": datetime.now(timezone.utc).isoformat(),
            }
            self._save_state()
            logger.info("Budget reset for agent '%s'", agent_id)

    def get_all_status(self) -> Dict[str, BudgetStatus]:
        """Get status for all configured agents."""
        result = {}
        for agent_id in self._config:
            if agent_id in ("consensus", "storage_path", "enabled"):
                continue
            result[agent_id] = self.check_budget(agent_id)
        return result
