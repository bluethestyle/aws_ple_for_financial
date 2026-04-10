"""
Base Agent for Pipeline Monitoring
====================================

Abstract base class for Ops and Audit agents.
Implements the Collect → Diagnose → Report loop with checklist-driven evaluation.

Both agents share this base:
    - Load checklist items from YAML config
    - Call tools via ToolRegistry for each item
    - Apply threshold-based judgment (PASS/WARN/FAIL)
    - Generate structured report
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, TYPE_CHECKING

import yaml

if TYPE_CHECKING:
    from core.agent.tool_registry import ToolRegistry

logger = logging.getLogger(__name__)

__all__ = ["BaseAgent", "ChecklistItem", "CheckResult", "AgentReport"]


@dataclass
class ChecklistItem:
    """Single checklist item definition from YAML config."""
    item_id: str           # e.g., "1.1"
    pipeline_part: str     # e.g., "P1"
    agent: str             # "ops" or "audit"
    description: str       # e.g., "도메인별 row count 변동률 ±20% 이내"
    tool_name: str         # Tool to call from ToolRegistry
    tool_params: Dict[str, Any] = field(default_factory=dict)
    threshold: Dict[str, Any] = field(default_factory=dict)  # judgment thresholds
    verdict_logic: str = ""  # e.g., "abs(delta) / prev < 0.20"
    enabled: bool = True


@dataclass
class CheckResult:
    """Result of a single checklist item evaluation."""
    item: ChecklistItem
    verdict: str            # "PASS", "WARN", "FAIL"
    value: Any = None       # measured value
    detail: str = ""        # human-readable detail
    error: Optional[str] = None  # if tool call failed
    timestamp: str = ""

    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = datetime.now(timezone.utc).isoformat()


@dataclass
class AgentReport:
    """Structured agent report."""
    agent_type: str          # "ops" or "audit"
    generated_at: str = ""
    status: str = "GREEN"    # GREEN / YELLOW / RED
    results: List[CheckResult] = field(default_factory=list)
    attention_required: List[Dict[str, Any]] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if not self.generated_at:
            self.generated_at = datetime.now(timezone.utc).isoformat()

    @property
    def pass_count(self) -> int:
        return sum(1 for r in self.results if r.verdict == "PASS")

    @property
    def warn_count(self) -> int:
        return sum(1 for r in self.results if r.verdict == "WARN")

    @property
    def fail_count(self) -> int:
        return sum(1 for r in self.results if r.verdict == "FAIL")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "agent_type": self.agent_type,
            "generated_at": self.generated_at,
            "status": self.status,
            "summary": {
                "total": len(self.results),
                "pass": self.pass_count,
                "warn": self.warn_count,
                "fail": self.fail_count,
            },
            "attention_required": self.attention_required,
            "results": [
                {
                    "item_id": r.item.item_id,
                    "pipeline_part": r.item.pipeline_part,
                    "description": r.item.description,
                    "verdict": r.verdict,
                    "value": r.value,
                    "detail": r.detail,
                    "error": r.error,
                }
                for r in self.results
            ],
            "metadata": self.metadata,
        }


class BaseAgent(ABC):
    """Abstract base agent implementing Collect → Diagnose → Report.

    Args:
        agent_type: "ops" or "audit".
        registry: ToolRegistry instance for tool invocation.
        checklist_path: Path to checklist YAML config.
        config: Additional agent-specific config dict.
    """

    def __init__(
        self,
        agent_type: str,
        registry: "ToolRegistry",
        checklist_path: str = "",
        config: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.agent_type = agent_type
        self.registry = registry
        self.config = config or {}
        self._checklist: List[ChecklistItem] = []

        if checklist_path and Path(checklist_path).exists():
            self._load_checklist(checklist_path)

    def _load_checklist(self, path: str) -> None:
        """Load checklist items from YAML, filtered by agent_type."""
        with open(path, encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}

        for item_cfg in data.get("checklist", []):
            if item_cfg.get("agent") != self.agent_type:
                continue
            if not item_cfg.get("enabled", True):
                continue

            item = ChecklistItem(
                item_id=item_cfg["item_id"],
                pipeline_part=item_cfg["pipeline_part"],
                agent=item_cfg["agent"],
                description=item_cfg["description"],
                tool_name=item_cfg.get("tool_name", ""),
                tool_params=item_cfg.get("tool_params", {}),
                threshold=item_cfg.get("threshold", {}),
                verdict_logic=item_cfg.get("verdict_logic", ""),
                enabled=item_cfg.get("enabled", True),
            )
            self._checklist.append(item)

        logger.info(
            "Loaded %d checklist items for %s agent from %s",
            len(self._checklist), self.agent_type, path,
        )

    # ------------------------------------------------------------------
    # Collect → Diagnose → Report loop
    # ------------------------------------------------------------------

    def run(self, parts: Optional[List[str]] = None) -> AgentReport:
        """Execute full Collect → Diagnose → Report loop.

        Args:
            parts: Optional filter — only run checklist items for these parts.
                   e.g., ["P1", "P2"] to only check ingestion + features.
        """
        items = self._checklist
        if parts:
            items = [it for it in items if it.pipeline_part in parts]

        # Collect + Diagnose (per item)
        results = []
        for item in items:
            result = self._evaluate_item(item)
            results.append(result)

        # Build report
        report = self._build_report(results)

        # Post-processing hook for subclasses
        report = self.post_process(report)

        return report

    def _evaluate_item(self, item: ChecklistItem) -> CheckResult:
        """Evaluate a single checklist item via tool call + threshold judgment."""
        if not item.tool_name:
            return CheckResult(
                item=item,
                verdict="PASS",
                detail="No tool configured — auto-pass",
            )

        try:
            raw_result = self.registry.call(item.tool_name, item.tool_params)
            verdict, value, detail = self._judge(item, raw_result)
            return CheckResult(
                item=item,
                verdict=verdict,
                value=value,
                detail=detail,
            )
        except Exception as e:
            logger.error("Checklist item %s failed: %s", item.item_id, e, exc_info=True)
            return CheckResult(
                item=item,
                verdict="FAIL",
                error=str(e),
                detail=f"Tool call failed: {e}",
            )

    def _judge(self, item: ChecklistItem, raw_result: Any) -> tuple:
        """Apply threshold-based judgment to tool result.

        Returns:
            (verdict, measured_value, detail_string)
        """
        # Subclasses can override for complex judgment
        # Default: delegate to _apply_threshold
        return self._apply_threshold(item, raw_result)

    def _apply_threshold(self, item: ChecklistItem, raw_result: Any) -> tuple:
        """Generic threshold application.

        Supports threshold configs like:
            threshold: {field: "psi_max", warn: 0.1, fail: 0.25}
            threshold: {field: "pass_rate", warn_below: 0.95}
            threshold: {field: "passed", expect: true}
        """
        threshold = item.threshold
        if not threshold:
            return ("PASS", raw_result, "No threshold defined")

        field_name = threshold.get("field", "")
        if field_name and isinstance(raw_result, dict):
            value = raw_result.get(field_name, raw_result)
        else:
            value = raw_result

        # Boolean check
        if "expect" in threshold:
            expected = threshold["expect"]
            if value == expected:
                return ("PASS", value, f"{field_name}={value}")
            return ("FAIL", value, f"{field_name}={value}, expected {expected}")

        # Upper threshold (warn/fail when value exceeds)
        if "fail" in threshold:
            if isinstance(value, (int, float)) and value >= threshold["fail"]:
                return ("FAIL", value, f"{field_name}={value} >= {threshold['fail']}")
        if "warn" in threshold:
            if isinstance(value, (int, float)) and value >= threshold["warn"]:
                return ("WARN", value, f"{field_name}={value} >= {threshold['warn']}")

        # Lower threshold (warn/fail when value is below)
        if "fail_below" in threshold:
            if isinstance(value, (int, float)) and value < threshold["fail_below"]:
                return ("FAIL", value, f"{field_name}={value} < {threshold['fail_below']}")
        if "warn_below" in threshold:
            if isinstance(value, (int, float)) and value < threshold["warn_below"]:
                return ("WARN", value, f"{field_name}={value} < {threshold['warn_below']}")

        return ("PASS", value, f"{field_name}={value}")

    def _build_report(self, results: List[CheckResult]) -> AgentReport:
        """Build AgentReport from check results."""
        # Determine overall status
        has_fail = any(r.verdict == "FAIL" for r in results)
        has_warn = any(r.verdict == "WARN" for r in results)
        status = "RED" if has_fail else ("YELLOW" if has_warn else "GREEN")

        # Build attention_required from WARN/FAIL items
        attention = []
        for r in results:
            if r.verdict in ("WARN", "FAIL"):
                attention.append({
                    "item_id": r.item.item_id,
                    "pipeline_part": r.item.pipeline_part,
                    "severity": r.verdict,
                    "finding": r.detail,
                })

        return AgentReport(
            agent_type=self.agent_type,
            status=status,
            results=results,
            attention_required=attention,
        )

    @abstractmethod
    def post_process(self, report: AgentReport) -> AgentReport:
        """Subclass hook for additional processing (e.g., cross-checkpoint correlation)."""
        ...
