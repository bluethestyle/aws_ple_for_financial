"""PORT-02 regression tests — agent_tool_routing.yaml must only reference
tools that actually exist in agent_tools.yaml (38-tool schema).

Background: CP5 (check_served_artifact) and CP6 (read_recommendation_output)
were on-prem tool names with 0 matches in the AWS schema, so focus_keys
["CP5"]/["CP6"] silently exposed only the common tools. The AWS equivalents
are check_feature_store_health (CP5 serving health) and read_audit_archive
(CP6 recommendation audit archive) per pipeline_reports.py registrations.

Run: pytest tests/test_agent_tool_routing.py -v
"""

from __future__ import annotations

from pathlib import Path

import yaml

from core.agent.bedrock_dialog import _ROUTING, _ROUTING_DEFAULT, _select_tools

_REPO = Path(__file__).resolve().parents[1]


def _schema_tool_names() -> set:
    cfg = yaml.safe_load(
        (_REPO / "configs" / "financial" / "agent_tools.yaml").read_text(encoding="utf-8")
    )
    return {t["name"] for t in cfg.get("tools", [])}


def _routing_tool_names(routing: dict) -> set:
    names = set()
    for agent_tools in routing.get("curated", {}).values():
        names.update(agent_tools)
    for focus_tools in routing.get("tool_map", {}).values():
        names.update(focus_tools)
    for agent_tools in routing.get("common", {}).values():
        names.update(agent_tools)
    return names


class TestRoutingToolNamesExist:
    def test_yaml_routing_names_all_exist_in_schema(self):
        routing = yaml.safe_load(
            (_REPO / "configs" / "financial" / "agent_tool_routing.yaml").read_text(
                encoding="utf-8"
            )
        )
        unknown = _routing_tool_names(routing) - _schema_tool_names()
        assert not unknown, f"agent_tools.yaml 에 없는 도구명: {sorted(unknown)}"

    def test_builtin_fallback_names_all_exist_in_schema(self):
        unknown = _routing_tool_names(_ROUTING_DEFAULT) - _schema_tool_names()
        assert not unknown, f"내장 폴백에 schema 미존재 도구명: {sorted(unknown)}"

    def test_loaded_routing_names_all_exist_in_schema(self):
        # _ROUTING is whatever bedrock_dialog actually loaded at import time.
        unknown = _routing_tool_names(_ROUTING) - _schema_tool_names()
        assert not unknown, f"로드된 라우팅에 schema 미존재 도구명: {sorted(unknown)}"


class TestFocusExposure:
    """focus_keys 별 노출 도구가 의도한 점검 도구를 실제로 포함하는지."""

    def test_cp5_exposes_serving_health(self):
        out = _select_tools("ops", ["CP5"])
        assert "check_feature_store_health" in out
        # common ops (2) + CP5 (1)
        assert len(out) == 3

    def test_cp6_exposes_audit_archive(self):
        out = _select_tools("ops", ["CP6"])
        assert "read_audit_archive" in out
        assert "check_feature_store_health" in out
        # common ops (2) + CP6 (2)
        assert len(out) == 4

    def test_cp7_exposes_ab_metrics(self):
        out = _select_tools("ops", ["CP7"])
        assert "query_cloudwatch_metrics" in out

    def test_every_focus_key_adds_at_least_one_tool(self):
        # 각 focus 가 common 대비 최소 1개 도구를 추가로 노출해야
        # "이름 불일치 → silent no-op" 회귀를 막는다.
        for agent_type in ("ops", "audit"):
            common = set(_ROUTING["common"].get(agent_type, []))
            for focus in _ROUTING["tool_map"]:
                out = set(_select_tools(agent_type, [focus]))
                assert out - common, (
                    f"focus={focus} ({agent_type}) 가 common 외 도구를 노출하지 않음"
                )
