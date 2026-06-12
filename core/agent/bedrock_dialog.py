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
    from core.agent.dialog_recall import DialogRecallMemory  # NEW

logger = logging.getLogger(__name__)

__all__ = ["BedrockDialogSession", "DialogTurn"]


# ── 도구 동적 선택 라우팅 (on-prem sync 2026-06-09) ──
# focus(체크포인트/관점) → 관련 도구만 LLM 에 노출 → 도구 과부하 방지(소형/대형 모두).
# agent_tool_routing.yaml 로 관리, 미존재시 내장 폴백.
_ROUTING_DEFAULT = {
    "curated": {"ops": ["read_feature_stats", "read_pipeline_state", "detect_drift"],
                "audit": ["run_regulatory_checks", "trace_feature_lineage", "read_feature_stats"]},
    "tool_map": {
        "CP1": ["read_ingestion_manifest"], "CP2": ["read_feature_stats", "detect_drift"],
        "CP3": ["read_experiment_metrics"], "CP4": ["read_distillation_fidelity"],
        "CP5": ["check_feature_store_health"],
        "CP6": ["read_audit_archive", "check_feature_store_health"],
        "CP7": ["query_cloudwatch_metrics"],
        "herding": ["detect_herding"], "fairness": ["evaluate_fairness"],
        "regulatory": ["run_regulatory_checks"], "lineage": ["trace_feature_lineage"],
    },
    "common": {"ops": ["read_feature_stats", "read_pipeline_state"], "audit": ["read_feature_stats"]},
    "availability": {},
}


def _load_routing() -> Dict[str, Any]:
    import os
    from pathlib import Path
    path = os.getenv("AGENT_TOOL_ROUTING") or str(
        Path(__file__).resolve().parents[2] / "configs" / "financial" / "agent_tool_routing.yaml")
    try:
        if Path(path).exists():
            import yaml
            cfg = yaml.safe_load(open(path, encoding="utf-8")) or {}
            return {k: cfg.get(k, _ROUTING_DEFAULT[k]) for k in _ROUTING_DEFAULT}
    except Exception as e:
        logger.warning("tool routing config 로드 실패(%s) — 내장 기본값", e)
    return _ROUTING_DEFAULT


_ROUTING = _load_routing()


def _tool_available(name: str) -> bool:
    import os
    from pathlib import Path
    env_var = _ROUTING["availability"].get(name)
    if not env_var:
        return True
    p = os.getenv(env_var, "")
    return bool(p) and Path(p).exists()


def _select_tools(agent_type: str, focus_keys: Optional[List[str]]) -> Optional[List[str]]:
    """focus_keys → 동적 도구명 목록. None 이면 전체(필터 안함)."""
    if not focus_keys:
        cur = _ROUTING["curated"].get(agent_type)
        return [t for t in cur if _tool_available(t)] if cur else None
    tools, seen, out = list(_ROUTING["common"].get(agent_type, [])), set(), []
    for k in focus_keys:
        tools += _ROUTING["tool_map"].get(k, [])
    for t in tools:
        if t not in seen and _tool_available(t):
            seen.add(t); out.append(t)
    return out


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
        region: AWS region. ``None`` lets boto3 resolve from env /
            credentials; callers should pass ``pipeline.yaml::aws.region``.
        system_prompt: Optional system prompt override.
        config: Additional config dict.
    """

    def __init__(
        self,
        registry: "ToolRegistry",
        agent_type: str = "ops",
        model_id: str = "anthropic.claude-sonnet-4-6-20250514-v1:0",
        region: Optional[str] = None,
        system_prompt: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None,
        recall_memory: Optional[Any] = None,  # DialogRecallMemory instance
        focus_keys: Optional[List[str]] = None,  # 동적 도구 선택(체크포인트/관점)
    ) -> None:
        self._registry = registry
        self._agent_type = agent_type
        self._model_id = model_id
        self._region = region
        self._config = config or {}
        self._history: List[DialogTurn] = []
        self._recall_memory = recall_memory
        # focus_keys → 관련 도구만 노출(None 이면 전체). on-prem sync.
        self._allowed_tools = _select_tools(agent_type, focus_keys)

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

        # Retrieve related past dialog turns (Letta-inspired)
        recall_context = ""
        if self._recall_memory is not None:
            try:
                operator_id = self._config.get("operator_id", "default") if hasattr(self, "_config") else "default"
                related = self._recall_memory.search_related(
                    operator_id=operator_id,
                    query_text=user_message,
                    limit=5,
                )
                if related:
                    lines = []
                    for r in related:
                        ts = r.get("timestamp", "")[:10]  # date only
                        user_msg = r.get("user_message", "")[:100]
                        agent_resp = r.get("agent_response", "")[:100]
                        lines.append(f"[{ts}] 담당자: {user_msg}\n에이전트: {agent_resp}")
                    recall_context = "\n\n---\n[과거 대화 참고]\n" + "\n\n".join(lines)
            except Exception as e:
                logger.debug("Dialog recall failed: %s", e)

        # Build augmented system prompt with recall context
        augmented_system = self._system_prompt + recall_context if recall_context else None

        # Build messages for Bedrock
        messages = self._build_messages()
        tool_config = self._build_tool_config()

        # Conversation loop (may involve multiple tool calls)
        max_iterations = 5  # prevent infinite tool loops
        for _ in range(max_iterations):
            response = self._call_bedrock(messages, tool_config, system_override=augmented_system)

            # Check if response contains tool use
            tool_uses = self._extract_tool_uses(response)
            if not tool_uses:
                # Pure text response
                text = self._extract_text(response)
                if not text.strip():
                    # 도구 호출 후 결론이 빈 응답으로 누락 → 강제 합성 (on-prem sync)
                    text = self._synthesize()
                self._history.append(DialogTurn(role="assistant", content=text))

                # Save this turn for future recall
                if self._recall_memory is not None:
                    try:
                        operator_id = self._config.get("operator_id", "default") if hasattr(self, "_config") else "default"
                        turn_id = str(len(self._history))
                        self._recall_memory.save_turn(
                            operator_id=operator_id,
                            session_id=str(id(self)),  # session scoped to this object instance
                            turn_id=turn_id,
                            user_msg=user_message,
                            agent_response=text,
                        )
                    except Exception as e:
                        logger.debug("Dialog recall save_turn failed: %s", e)

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

        # 반복 한도 도달 → 지금까지의 도구 결과로 강제 결론 (on-prem sync)
        text = self._synthesize()
        self._history.append(DialogTurn(role="assistant", content=text))
        return text

    def _synthesize(self) -> str:
        """도구 호출 후 결론이 비거나 반복 한도에 닿으면 강제 종합.

        도구 없이(빈 toolConfig) Bedrock 1회 호출로 [원인][근거][권고]
        텍스트를 생성한다. 실패 시 고정 문구 폴백 (on-prem _synthesize 등가).
        """
        instruction = (
            "지금까지 도구로 확인한 결과를 종합해 [원인], [근거], [권고] "
            "세 항목으로 한국어 결론만 작성하세요. 도구는 더 호출하지 마세요."
        )
        try:
            messages = self._build_messages()
            # converse 는 역할 교대를 요구 — 마지막이 user(toolResult 포함)면
            # 새 user 메시지 대신 기존 content 에 지시 블록을 병합한다.
            if messages and messages[-1]["role"] == "user":
                messages[-1]["content"].append({"text": instruction})
            else:
                messages.append({"role": "user", "content": [{"text": instruction}]})
            response = self._call_bedrock(messages, {})  # tools 없이 → 반드시 텍스트
            text = self._extract_text(response)
            if text.strip():
                return text
        except Exception as e:
            logger.warning("강제 합성 실패: %s", e)
        return "조사 완료(결론 생성 실패) — 도구 결과를 직접 확인하세요."

    def investigate(self, finding: str) -> Dict[str, Any]:
        """RED/HIGH 자동 조사·추론 (on-prem reasoning_agent sync).

        finding(문제 설명) → 도구사용 추론 루프(chat, 최대 5회)로 근본원인을
        조사하고 {reasoning, tool_calls, n_tool_calls} 를 반환한다.
        """
        prompt = (
            f"다음 문제가 감지되었습니다. 도구로 직접 데이터를 확인하며 "
            f"근본원인을 조사·추론하세요.\n\n[문제]\n{finding}"
        )
        text = self.chat(prompt)
        tool_calls = [tc for t in self._history for tc in t.tool_calls]
        return {
            "reasoning": text,
            "tool_calls": [{"name": tc.get("name")} for tc in tool_calls],
            "n_tool_calls": len(tool_calls),
            "_model_reasoned": True,
        }

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
        """Build Bedrock tool configuration from ToolRegistry.
        focus_keys 지정시 _allowed_tools 로 동적 필터(과부하 방지). on-prem sync."""
        tools = self._registry.get_bedrock_tools(agent=self._agent_type)
        if self._allowed_tools is not None:
            allowed = set(self._allowed_tools)
            tools = [t for t in tools if t.get("toolSpec", {}).get("name") in allowed]
        if not tools:
            return {}
        return {"tools": tools}

    def triage(self, finding: str) -> Dict[str, Any]:
        """WARNING(경고) 트리아지 — 도구로 관련성·미래영향 확인 후 분류(on-prem sync).
        IGNORE(무관→로그정리) / MONITOR(미미·관찰) / FIX_NOW(증가·누적→즉시수정)."""
        import re
        prompt = (
            f"다음은 경고(WARNING)입니다. 도구로 현재 파이프라인과의 관련성과 미래 영향 가능성을 "
            f"직접 확인한 뒤 분류하세요.\n\n[경고]\n{finding}\n\n"
            "- IGNORE: 현재 파이프라인과 무관 → 추후 워닝 로그 정리 권장\n"
            "- MONITOR: 현재 미미하고 악화 징후도 없음 → 관찰만\n"
            "- FIX_NOW: 증가/누적 추세이거나 방치시 추후 큰 영향 가능 → 즉시 수정 신호\n"
            "  (증가추세/누적 신호가 있으면 임계값 미만이어도 FIX_NOW 적극 고려)\n"
            "조사 후 마지막 줄에 '[분류] IGNORE|MONITOR|FIX_NOW' 형식으로 결론을 쓰고 [근거][권고]도 작성하세요."
        )
        text = self.chat(prompt)
        m = re.search(r"\[분류\]\s*(IGNORE|MONITOR|FIX_NOW)", text)
        return {"reasoning": text, "triage": m.group(1) if m else "MONITOR"}

    def _call_bedrock(self, messages: List[Dict], tool_config: Dict, system_override: Optional[str] = None) -> Dict:
        """Call Bedrock converse API."""
        client = self._get_client()

        kwargs: Dict[str, Any] = {
            "modelId": self._model_id,
            "messages": messages,
            "system": [{"text": system_override if system_override is not None else self._system_prompt}],
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
