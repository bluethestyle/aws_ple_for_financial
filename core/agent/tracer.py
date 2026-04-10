"""
Tool Tracer — Full Tool Call Trace Capture
=============================================

PaperClip-inspired full trace: every tool call is logged with
timing, parameters, results, and token cost.

Trace data flows into:
    - In-memory ring buffer (recent N traces)
    - JSONL files (periodic flush, per-agent per-day)
    - BudgetTracker (token cost feeds into budget)
"""

from __future__ import annotations

import json
import logging
import time
import uuid
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from core.agent.budget import BudgetTracker

logger = logging.getLogger(__name__)

__all__ = ["ToolTracer", "ToolTrace"]


@dataclass
class ToolTrace:
    """Single tool call trace record."""
    trace_id: str
    agent_id: str
    tool_name: str
    params_summary: str     # truncated params
    result_summary: str     # truncated result
    token_cost: int         # 0 for non-LLM tools
    duration_ms: float
    timestamp: str
    success: bool
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        d = {
            "trace_id": self.trace_id,
            "agent_id": self.agent_id,
            "tool_name": self.tool_name,
            "params_summary": self.params_summary,
            "result_summary": self.result_summary,
            "token_cost": self.token_cost,
            "duration_ms": round(self.duration_ms, 2),
            "timestamp": self.timestamp,
            "success": self.success,
        }
        if self.error:
            d["error"] = self.error
        return d


class ToolTracer:
    """Captures traces for all tool calls.

    Args:
        agent_id: Agent identifier ("ops", "audit", "dialog").
        budget_tracker: Optional BudgetTracker for token cost recording.
        trace_dir: Directory for JSONL trace files.
        ring_buffer_size: Max traces in memory.
        flush_interval_calls: Flush to disk every N calls.
    """

    def __init__(
        self,
        agent_id: str = "unknown",
        budget_tracker: Optional["BudgetTracker"] = None,
        trace_dir: str = "outputs/traces",
        ring_buffer_size: int = 1000,
        flush_interval_calls: int = 50,
    ) -> None:
        self._agent_id = agent_id
        self._budget_tracker = budget_tracker
        self._trace_dir = Path(trace_dir)
        self._buffer: deque[ToolTrace] = deque(maxlen=ring_buffer_size)
        self._flush_interval = flush_interval_calls
        self._call_count = 0

    def trace(
        self,
        tool_name: str,
        params: Optional[Dict[str, Any]],
        func: Callable,
        token_cost: int = 0,
    ) -> Any:
        """Execute a tool function with full tracing.

        Args:
            tool_name: Name of the tool.
            params: Parameters to pass to the tool.
            func: The actual callable.
            token_cost: Estimated token cost (0 for non-LLM tools).

        Returns:
            The tool function's return value.

        Raises:
            Whatever the tool function raises (re-raised after tracing).
        """
        start = time.monotonic()
        error_msg = None
        result = None
        success = True

        try:
            result = func(**(params or {}))
            return result
        except Exception as e:
            error_msg = str(e)
            success = False
            raise
        finally:
            duration_ms = (time.monotonic() - start) * 1000

            trace = ToolTrace(
                trace_id=uuid.uuid4().hex[:12],
                agent_id=self._agent_id,
                tool_name=tool_name,
                params_summary=self._summarize(params, max_len=200),
                result_summary=self._summarize(result, max_len=300) if success else "",
                token_cost=token_cost,
                duration_ms=duration_ms,
                timestamp=datetime.now(timezone.utc).isoformat(),
                success=success,
                error=error_msg,
            )

            self._buffer.append(trace)
            self._call_count += 1

            # Feed token cost into budget tracker
            if self._budget_tracker and token_cost > 0:
                self._budget_tracker.record_tokens(self._agent_id, token_cost, 0)

            # Periodic flush
            if self._call_count % self._flush_interval == 0:
                self.flush()

    def flush(self) -> int:
        """Flush buffered traces to JSONL file.

        Returns:
            Number of traces written.
        """
        if not self._buffer:
            return 0

        traces = list(self._buffer)
        date_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        out_dir = self._trace_dir / self._agent_id
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"{date_str}.jsonl"

        try:
            with open(out_path, "a", encoding="utf-8") as f:
                for t in traces:
                    f.write(json.dumps(t.to_dict(), ensure_ascii=False) + "\n")
            logger.debug("Flushed %d traces to %s", len(traces), out_path)
            return len(traces)
        except Exception as e:
            logger.error("Trace flush failed: %s", e)
            return 0

    def get_recent(self, n: int = 20) -> List[Dict[str, Any]]:
        """Get the most recent N traces."""
        recent = list(self._buffer)[-n:]
        return [t.to_dict() for t in recent]

    @property
    def total_calls(self) -> int:
        return self._call_count

    @property
    def buffer_size(self) -> int:
        return len(self._buffer)

    @staticmethod
    def _summarize(obj: Any, max_len: int = 200) -> str:
        """Summarize an object to a truncated string."""
        if obj is None:
            return ""
        try:
            s = json.dumps(obj, ensure_ascii=False, default=str)
        except (TypeError, ValueError):
            s = str(obj)
        if len(s) > max_len:
            return s[:max_len] + "..."
        return s
