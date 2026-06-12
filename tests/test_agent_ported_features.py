"""Unit tests for the on-prem agent features ported in this branch
(sync/ops-audit-agent-from-onprem, commit ed37117).

core/agent had 0% pytest coverage; these tests pin the three ported
behaviours the commit introduced:

* ToolRegistry._filter_params  — drop params a tool's func doesn't accept
* bedrock_dialog._select_tools — focus_keys → dynamic tool list
* BedrockDialogSession.triage  — WARNING triage classification parsing

Run: pytest tests/test_agent_ported_features.py -v
"""

from __future__ import annotations

import os

import pytest

from core.agent.tool_registry import ToolRegistry
from core.agent.bedrock_dialog import BedrockDialogSession, _select_tools, _ROUTING


# ---------------------------------------------------------------------------
# _filter_params
# ---------------------------------------------------------------------------

class TestFilterParams:
    def test_drops_unknown_params(self):
        def func(a, b):
            return a + b
        out = ToolRegistry._filter_params(func, {"a": 1, "b": 2, "c": 3})
        assert out == {"a": 1, "b": 2}

    def test_passthrough_when_var_keyword(self):
        def func(a, **kwargs):
            return a
        params = {"a": 1, "x": 2, "y": 3}
        assert ToolRegistry._filter_params(func, params) == params

    def test_empty_params(self):
        def func(a):
            return a
        assert ToolRegistry._filter_params(func, {}) == {}
        assert ToolRegistry._filter_params(func, None) == {}

    def test_preserves_all_matching_required(self):
        def func(x, y, z=0):
            return x
        out = ToolRegistry._filter_params(func, {"x": 1, "y": 2, "z": 3})
        assert out == {"x": 1, "y": 2, "z": 3}

    def test_signature_error_returns_params(self, monkeypatch):
        # When the signature cannot be introspected, the helper must not
        # raise — it returns params unchanged (robust fallback).
        import inspect

        def _boom(_):
            raise ValueError("no signature")
        monkeypatch.setattr(inspect, "signature", _boom)
        params = {"a": 1, "x": 2}
        assert ToolRegistry._filter_params(lambda a: a, params) == params


# ---------------------------------------------------------------------------
# _select_tools
# ---------------------------------------------------------------------------

class TestSelectTools:
    def test_no_focus_returns_curated(self):
        out = _select_tools("ops", None)
        assert out == _ROUTING["curated"]["ops"]

    def test_no_focus_unknown_agent_returns_none(self):
        assert _select_tools("nonexistent_agent", None) is None

    def test_focus_keys_map_and_dedup(self):
        # CP2 → [read_feature_stats, detect_drift]; common ops includes
        # read_feature_stats → must be de-duplicated.
        out = _select_tools("ops", ["CP2"])
        assert out is not None
        assert out.count("read_feature_stats") == 1
        assert "detect_drift" in out

    def test_focus_keys_combine_multiple(self):
        out = _select_tools("audit", ["regulatory", "lineage"])
        assert "run_regulatory_checks" in out
        assert "trace_feature_lineage" in out

    def test_availability_filter_excludes_unavailable(self, monkeypatch):
        # Point detect_drift's availability env var at a nonexistent path.
        monkeypatch.setitem(
            _ROUTING["availability"], "detect_drift", "AGENT_TEST_MISSING_PATH"
        )
        monkeypatch.setenv("AGENT_TEST_MISSING_PATH", "/no/such/file/xyz")
        try:
            out = _select_tools("ops", ["CP2"])
            assert "detect_drift" not in out
            assert "read_feature_stats" in out
        finally:
            _ROUTING["availability"].pop("detect_drift", None)


# ---------------------------------------------------------------------------
# triage
# ---------------------------------------------------------------------------

class TestTriage:
    def _session(self, monkeypatch, reply: str) -> BedrockDialogSession:
        sess = BedrockDialogSession(registry=ToolRegistry(), agent_type="ops")
        monkeypatch.setattr(sess, "chat", lambda prompt: reply)
        return sess

    def test_parses_fix_now(self, monkeypatch):
        sess = self._session(
            monkeypatch,
            "조사 결과 누적 추세입니다.\n[분류] FIX_NOW\n[근거] ...\n[권고] ...",
        )
        out = sess.triage("PSI 증가 경고")
        assert out["triage"] == "FIX_NOW"
        assert "reasoning" in out

    def test_parses_ignore(self, monkeypatch):
        sess = self._session(monkeypatch, "무관합니다.\n[분류] IGNORE")
        assert sess.triage("무관 경고")["triage"] == "IGNORE"

    def test_defaults_to_monitor_when_unparseable(self, monkeypatch):
        sess = self._session(monkeypatch, "결론 형식이 없는 응답")
        assert sess.triage("애매한 경고")["triage"] == "MONITOR"

    def test_prompt_contains_financial_severity_factors(self, monkeypatch):
        """PORT-05: 금융 AIOps 심각도 4요소 (on-prem c3df4317, AWS 규제 어휘)."""
        sess = BedrockDialogSession(registry=ToolRegistry(), agent_type="ops")
        captured = {}

        def fake_chat(prompt):
            captured["prompt"] = prompt
            return "[분류] MONITOR"

        monkeypatch.setattr(sess, "chat", fake_chat)
        sess.triage("PSI 경고")
        prompt = captured["prompt"]
        # 4요소: 규제(AWS 어휘), 고객 노출 산출물, 데이터 무결성/PII/누수, 성능 저하
        for needle in (
            "EU AI Act", "SR 11-7", "금소법 §17",
            "고객에게 직접 노출", "PII", "누수", "성능 저하", "드리프트",
        ):
            assert needle in prompt, f"심각도 요소 누락: {needle}"

    def test_returns_tool_call_trace(self, monkeypatch):
        """PORT-05: triage 도 investigate 처럼 도구 호출 추적을 반환."""
        from core.agent.bedrock_dialog import DialogTurn

        sess = self._session(monkeypatch, "[분류] FIX_NOW")
        sess._history.append(DialogTurn(
            role="assistant", content="",
            tool_calls=[{"toolUseId": "t1", "name": "detect_drift", "input": {}}],
            tool_results=[{"toolUseId": "t1", "result": {"psi": 0.3}}],
        ))
        out = sess.triage("드리프트 경고")
        assert out["tool_calls"] == [{"name": "detect_drift"}]
        assert out["n_tool_calls"] == 1
        assert out["_model_reasoned"] is True
