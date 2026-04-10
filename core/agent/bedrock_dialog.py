"""
Bedrock Dialog Session — Tool-Enabled Conversation Interface
==============================================================

Enables operators to discuss diagnostic results with the agent.
The agent can call tools from ToolRegistry during conversation
to look up data, run checks, or search similar cases.

Uses Bedrock ``converse`` API with tool definitions exported
from ToolRegistry in Bedrock Tool Use format.

Example dialog:
    Operator: "elderly ∩ low_income DI가 0.68인데, 필터 문제인지 모수 문제인지?"
    Agent: [calls evaluate_fairness tool]
           "age_group=elderly 단독은 DI 0.85로 통과하지만..."
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from core.agent.tool_registry import ToolRegistry

logger = logging.getLogger(__name__)

__all__ = ["BedrockDialogSession", "DialogTurn"]


@dataclass
class DialogTurn:
    """A single turn in the dialog."""
    role: str          # "user" or "assistant"
    content: str       # text content
    tool_calls: List[Dict[str, Any]] = field(default_factory=list)
    tool_results: List[Dict[str, Any]] = field(default_factory=list)


class BedrockDialogSession:
    """Conversational interface with Bedrock Sonnet + ToolRegistry.

    Args:
        registry: ToolRegistry for tool definitions and execution.
        agent_type: "ops" or "audit" — filters available tools.
        model_id: Bedrock model ID (default Claude Sonnet).
        region: AWS region.
        system_prompt: Optional system prompt override.
        config: Additional config dict.
    """

    def __init__(
        self,
        registry: "ToolRegistry",
        agent_type: str = "ops",
        model_id: str = "anthropic.claude-sonnet-4-6-20250514-v1:0",
        region: str = "ap-northeast-2",
        system_prompt: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None,
    ) -> None:
        self._registry = registry
        self._agent_type = agent_type
        self._model_id = model_id
        self._region = region
        self._config = config or {}
        self._history: List[DialogTurn] = []

        self._system_prompt = system_prompt or self._default_system_prompt()
        self._client = None  # lazy init

    def _default_system_prompt(self) -> str:
        role = "운영(Ops)" if self._agent_type == "ops" else "감사(Audit)"
        return (
            f"당신은 금융 AI 추천 시스템의 {role} 에이전트입니다.\n"
            "파이프라인 진단 결과를 해석하고 담당자와 논의하는 역할입니다.\n"
            "수치를 인용할 때는 정확한 출처를 밝히세요.\n"
            "확신이 없으면 도구를 호출하여 데이터를 직접 확인하세요.\n"
            "판단은 권고만 하고 최종 결정은 담당자에게 맡기세요.\n"
            "한국어로 응답하세요."
        )

    def _get_client(self):
        """Lazy init Bedrock client."""
        if self._client is None:
            try:
                import boto3
                self._client = boto3.client(
                    "bedrock-runtime",
                    region_name=self._region,
                )
            except Exception as e:
                logger.error("Failed to create Bedrock client: %s", e)
                raise
        return self._client

    def chat(self, user_message: str) -> str:
        """Send a message and get a response, with automatic tool execution.

        Args:
            user_message: The operator's message.

        Returns:
            The agent's text response.
        """
        # Add user turn
        self._history.append(DialogTurn(role="user", content=user_message))

        # Build messages for Bedrock
        messages = self._build_messages()
        tool_config = self._build_tool_config()

        # Conversation loop (may involve multiple tool calls)
        max_iterations = 5  # prevent infinite tool loops
        for _ in range(max_iterations):
            response = self._call_bedrock(messages, tool_config)

            # Check if response contains tool use
            tool_uses = self._extract_tool_uses(response)
            if not tool_uses:
                # Pure text response
                text = self._extract_text(response)
                self._history.append(DialogTurn(role="assistant", content=text))
                return text

            # Execute tools and add results
            text_parts = self._extract_text(response)
            tool_results = []
            for tool_use in tool_uses:
                result = self._execute_tool(tool_use)
                tool_results.append(result)

            # Add assistant turn with tool calls
            self._history.append(DialogTurn(
                role="assistant",
                content=text_parts,
                tool_calls=tool_uses,
                tool_results=tool_results,
            ))

            # Add tool results to messages and continue
            messages = self._build_messages()

        # If we hit max iterations, return what we have
        return "도구 호출 반복 한도에 도달했습니다. 결과를 종합해주세요."

    def _build_messages(self) -> List[Dict[str, Any]]:
        """Build Bedrock converse API messages from history."""
        messages = []
        for turn in self._history:
            if turn.role == "user":
                messages.append({
                    "role": "user",
                    "content": [{"text": turn.content}],
                })
            elif turn.role == "assistant":
                content = []
                if turn.content:
                    content.append({"text": turn.content})
                for tc in turn.tool_calls:
                    content.append({
                        "toolUse": {
                            "toolUseId": tc.get("toolUseId", ""),
                            "name": tc.get("name", ""),
                            "input": tc.get("input", {}),
                        }
                    })
                if content:
                    messages.append({"role": "assistant", "content": content})

                # Add tool results as user messages
                for tr in turn.tool_results:
                    messages.append({
                        "role": "user",
                        "content": [{
                            "toolResult": {
                                "toolUseId": tr.get("toolUseId", ""),
                                "content": [{"text": json.dumps(tr.get("result", {}), ensure_ascii=False, default=str)}],
                            }
                        }],
                    })

        return messages

    def _build_tool_config(self) -> Dict[str, Any]:
        """Build Bedrock tool configuration from ToolRegistry."""
        tools = self._registry.get_bedrock_tools(agent=self._agent_type)
        if not tools:
            return {}
        return {"tools": tools}

    def _call_bedrock(self, messages: List[Dict], tool_config: Dict) -> Dict:
        """Call Bedrock converse API."""
        client = self._get_client()

        kwargs: Dict[str, Any] = {
            "modelId": self._model_id,
            "messages": messages,
            "system": [{"text": self._system_prompt}],
        }
        if tool_config:
            kwargs["toolConfig"] = tool_config

        try:
            response = client.converse(**kwargs)
            return response
        except Exception as e:
            logger.error("Bedrock converse failed: %s", e)
            return {"output": {"message": {"content": [{"text": f"Bedrock 호출 실패: {e}"}]}}}

    def _extract_text(self, response: Dict) -> str:
        """Extract text content from Bedrock response."""
        try:
            content = response.get("output", {}).get("message", {}).get("content", [])
            texts = [block.get("text", "") for block in content if "text" in block]
            return " ".join(texts)
        except Exception:
            return ""

    def _extract_tool_uses(self, response: Dict) -> List[Dict]:
        """Extract tool use blocks from Bedrock response."""
        try:
            content = response.get("output", {}).get("message", {}).get("content", [])
            return [block["toolUse"] for block in content if "toolUse" in block]
        except Exception:
            return []

    def _execute_tool(self, tool_use: Dict) -> Dict:
        """Execute a tool call via ToolRegistry."""
        name = tool_use.get("name", "")
        params = tool_use.get("input", {})
        tool_use_id = tool_use.get("toolUseId", "")

        try:
            result = self._registry.call(name, params)
            return {"toolUseId": tool_use_id, "result": result}
        except Exception as e:
            logger.error("Tool execution failed for %s: %s", name, e)
            return {"toolUseId": tool_use_id, "result": {"error": str(e)}}

    @property
    def turn_count(self) -> int:
        return len(self._history)

    def clear_history(self) -> None:
        """Clear conversation history."""
        self._history.clear()

    def get_history(self) -> List[Dict[str, Any]]:
        """Get conversation history as dicts."""
        return [
            {
                "role": t.role,
                "content": t.content,
                "tool_calls": t.tool_calls,
                "tool_results": t.tool_results,
            }
            for t in self._history
        ]
