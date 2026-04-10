"""
Tool Registry for Ops/Audit Agents
====================================

Manages tool definitions and wraps existing monitoring components.
Supports two invocation modes:
    - Direct Python call: ``registry.call("tool_name", params)``
    - Bedrock Tool Use export: ``registry.get_bedrock_tools(agent="ops")``

All tools are defined in YAML config (agent_tools.yaml) and registered
with Python callables at runtime.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, TYPE_CHECKING

import yaml

logger = logging.getLogger(__name__)

__all__ = ["ToolRegistry", "ToolDefinition"]


class ToolDefinition:
    """Single tool definition with schema and callable."""

    def __init__(
        self,
        name: str,
        description: str,
        category: str,  # "query" or "action"
        agents: List[str],  # ["ops"], ["audit"], ["ops", "audit"]
        parameters: Dict[str, Any],
        returns: str = "",
        func: Optional[Callable] = None,
    ) -> None:
        self.name = name
        self.description = description
        self.category = category
        self.agents = agents
        self.parameters = parameters
        self.returns = returns
        self.func = func

    def to_bedrock_tool(self) -> Dict[str, Any]:
        """Export as Bedrock Tool Use format."""
        return {
            "toolSpec": {
                "name": self.name,
                "description": self.description,
                "inputSchema": {
                    "json": self.parameters,
                },
            }
        }


class ToolRegistry:
    """Central registry for agent tools.

    Args:
        config_path: Path to agent_tools.yaml defining tool schemas.
        approval_callback: Optional callback for Action tool approval.
            Signature: ``(tool_name, params) -> bool``.
            If None, Action tools raise without approval.
    """

    def __init__(
        self,
        config_path: Optional[str] = None,
        approval_callback: Optional[Callable[[str, Dict], bool]] = None,
    ) -> None:
        self._tools: Dict[str, ToolDefinition] = {}
        self._approval_callback = approval_callback

        if config_path and Path(config_path).exists():
            self._load_from_yaml(config_path)

    def _load_from_yaml(self, path: str) -> None:
        """Load tool definitions from YAML config."""
        with open(path, encoding="utf-8") as f:
            config = yaml.safe_load(f) or {}

        for tool_cfg in config.get("tools", []):
            defn = ToolDefinition(
                name=tool_cfg["name"],
                description=tool_cfg.get("description", ""),
                category=tool_cfg.get("category", "query"),
                agents=tool_cfg.get("agents", ["ops", "audit"]),
                parameters=tool_cfg.get("parameters", {"type": "object", "properties": {}}),
                returns=tool_cfg.get("returns", ""),
            )
            self._tools[defn.name] = defn

        logger.info("ToolRegistry loaded %d tool definitions from %s", len(self._tools), path)

    def register(
        self,
        name: str,
        func: Callable,
        description: str = "",
        category: str = "query",
        agents: Optional[List[str]] = None,
        parameters: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Register a Python callable as a tool.

        If the tool was already loaded from YAML, the callable is attached.
        Otherwise a new definition is created.
        """
        if name in self._tools:
            self._tools[name].func = func
        else:
            self._tools[name] = ToolDefinition(
                name=name,
                description=description,
                category=category,
                agents=agents or ["ops", "audit"],
                parameters=parameters or {"type": "object", "properties": {}},
                func=func,
            )

    def call(self, name: str, params: Optional[Dict[str, Any]] = None) -> Any:
        """Call a registered tool by name.

        Action tools require approval via the approval_callback.

        Raises:
            KeyError: Tool not found.
            PermissionError: Action tool without approval.
            RuntimeError: Tool has no callable registered.
        """
        if name not in self._tools:
            raise KeyError(f"Tool '{name}' not found in registry")

        tool = self._tools[name]

        if tool.func is None:
            raise RuntimeError(f"Tool '{name}' has no callable registered")

        if tool.category == "action":
            if self._approval_callback is None:
                raise PermissionError(
                    f"Action tool '{name}' requires approval but no callback set"
                )
            if not self._approval_callback(name, params or {}):
                logger.warning("Action tool '%s' denied by approval callback", name)
                return {"status": "denied", "tool": name}

        try:
            result = tool.func(**(params or {}))
            return result
        except Exception as e:
            logger.error("Tool '%s' execution failed: %s", name, e, exc_info=True)
            raise

    def get_tools(self, agent: Optional[str] = None, category: Optional[str] = None) -> List[ToolDefinition]:
        """Get tool definitions filtered by agent and/or category."""
        tools = list(self._tools.values())
        if agent:
            tools = [t for t in tools if agent in t.agents]
        if category:
            tools = [t for t in tools if t.category == category]
        return tools

    def get_bedrock_tools(self, agent: Optional[str] = None) -> List[Dict[str, Any]]:
        """Export tool definitions in Bedrock converse API format.

        Args:
            agent: Filter to tools accessible by this agent ("ops" or "audit").

        Returns:
            List of tool specs for Bedrock ``converse`` API ``toolConfig.tools``.
        """
        tools = self.get_tools(agent=agent)
        return [t.to_bedrock_tool() for t in tools]

    def get_tool_names(self, agent: Optional[str] = None) -> List[str]:
        """Get list of tool names, optionally filtered by agent."""
        return [t.name for t in self.get_tools(agent=agent)]

    @property
    def tool_count(self) -> Dict[str, int]:
        """Count tools by category."""
        counts = {"query": 0, "action": 0, "total": 0}
        for tool in self._tools.values():
            counts[tool.category] = counts.get(tool.category, 0) + 1
            counts["total"] += 1
        return counts

    def summary(self) -> str:
        """Human-readable summary of registered tools."""
        counts = self.tool_count
        lines = [f"ToolRegistry: {counts['total']} tools ({counts['query']} query, {counts['action']} action)"]
        for tool in self._tools.values():
            registered = "+" if tool.func else "-"
            lines.append(f"  [{registered}] {tool.name} ({tool.category}) -> {tool.agents}")
        return "\n".join(lines)
