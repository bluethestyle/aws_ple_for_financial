"""PORT-04 tests — BedrockDialogSession.investigate/_synthesize and the
run_pipeline_reports verdict-branched LLM follow-up wiring.

All Bedrock calls are mocked; no AWS access required.

Run: pytest tests/test_agent_llm_followup.py -v
"""

from __future__ import annotations

import json

from core.agent.bedrock_dialog import BedrockDialogSession, DialogTurn
from core.agent.pipeline_reports import (
    _attach_llm_followup,
    _audit_finding_text,
    _audit_focus_keys,
    _followup_mode,
    _llm_followup_enabled,
    _llm_verify_enabled,
    _ops_finding_text,
    _ops_focus_keys,
)
from core.agent.tool_registry import ToolRegistry


def _text_response(text: str) -> dict:
    return {"output": {"message": {"content": [{"text": text}]}}}


def _tooluse_response(name: str = "read_feature_stats") -> dict:
    return {"output": {"message": {"content": [
        {"toolUse": {"toolUseId": "t1", "name": name, "input": {}}},
    ]}}}


# ---------------------------------------------------------------------------
# BedrockDialogSession.investigate / _synthesize
# ---------------------------------------------------------------------------

class TestInvestigate:
    def test_returns_reasoning_and_tool_call_trace(self, monkeypatch):
        sess = BedrockDialogSession(registry=ToolRegistry(), agent_type="ops")
        monkeypatch.setattr(sess, "chat", lambda prompt: "[원인] PSI 누적 [근거] ... [권고] ...")
        sess._history.append(DialogTurn(
            role="assistant", content="",
            tool_calls=[{"name": "detect_drift", "input": {}}],
            tool_results=[{"name": "detect_drift", "result": {"psi": 0.3}}],
        ))
        out = sess.investigate("CP2 드리프트 RED")
        assert "[원인]" in out["reasoning"]
        assert out["tool_calls"] == [{"name": "detect_drift"}]
        assert out["n_tool_calls"] == 1
        assert out["_model_reasoned"] is True


class TestSynthesizeFallback:
    def test_max_iteration_loop_forces_synthesis(self, monkeypatch):
        """반복 한도 도달 시 고정 문구가 아니라 도구-없는 합성 결론을 반환."""
        sess = BedrockDialogSession(registry=ToolRegistry(), agent_type="ops")

        def fake_call(messages, tool_config, system_override=None):
            if tool_config:  # 도구 노출 → 모델이 계속 도구만 호출
                return _tooluse_response()
            return _text_response("[원인] 합성 결론 [근거] ... [권고] ...")

        monkeypatch.setattr(sess, "_call_bedrock", fake_call)
        monkeypatch.setattr(
            sess, "_build_tool_config",
            lambda: {"tools": [{"toolSpec": {"name": "read_feature_stats"}}]},
        )
        out = sess.chat("조사하세요")
        assert "합성 결론" in out
        # 합성 결과가 history 에도 남는다
        assert sess._history[-1].content == out

    def test_empty_text_response_forces_synthesis(self, monkeypatch):
        sess = BedrockDialogSession(registry=ToolRegistry(), agent_type="ops")

        calls = {"n": 0}

        def fake_call(messages, tool_config, system_override=None):
            calls["n"] += 1
            if calls["n"] == 1:
                return _text_response("")  # 소형/이상 응답: 빈 텍스트
            return _text_response("합성된 결론")

        monkeypatch.setattr(sess, "_call_bedrock", fake_call)
        assert sess.chat("질문") == "합성된 결론"

    def test_synthesis_failure_returns_fallback_text(self, monkeypatch):
        sess = BedrockDialogSession(registry=ToolRegistry(), agent_type="ops")
        monkeypatch.setattr(
            sess, "_call_bedrock",
            lambda *a, **k: _text_response(""),
        )
        out = sess._synthesize()
        assert "결론 생성 실패" in out


# ---------------------------------------------------------------------------
# verdict 분기 + finding/focus 빌더
# ---------------------------------------------------------------------------

class TestFollowupMode:
    def test_mapping(self):
        assert _followup_mode("RED") == "investigate"
        assert _followup_mode("HIGH") == "investigate"
        assert _followup_mode("CRITICAL") == "investigate"
        assert _followup_mode("YELLOW") == "triage"
        assert _followup_mode("MEDIUM") == "triage"
        assert _followup_mode("GREEN") is None
        assert _followup_mode("LOW") is None
        assert _followup_mode("") is None
        assert _followup_mode("UNKNOWN") is None


class TestFindingBuilders:
    def test_ops_focus_keys_parse_cp_tokens(self):
        attention = [
            {"checkpoint": "CP2", "severity": "FAIL", "finding": "drift"},
            {"checkpoint": "CP1, CP3", "severity": "WARNING", "finding": "x"},
            {"checkpoint": "CP2", "severity": "WARNING", "finding": "dup"},
        ]
        assert _ops_focus_keys(attention) == ["CP2", "CP1", "CP3"]

    def test_ops_finding_text_includes_cause(self):
        text = _ops_finding_text([{
            "checkpoint": "CP4", "severity": "FAIL",
            "finding": "fidelity gap 0.4", "likely_cause": "teacher 미달",
        }])
        assert "CP4" in text and "fidelity gap" in text and "teacher 미달" in text

    def test_audit_focus_keys_map_korean_areas(self):
        focus = [
            {"area": "교차속성 공정성", "priority": "HIGH", "finding": "..."},
            {"area": "추천 집중도", "priority": "MEDIUM", "finding": "..."},
            {"area": "규제 적합성", "priority": "HIGH", "finding": "..."},
            {"area": "데이터 계보", "priority": "LOW", "finding": "..."},
        ]
        assert _audit_focus_keys(focus) == [
            "fairness", "herding", "regulatory", "lineage",
        ]

    def test_audit_finding_text(self):
        text = _audit_finding_text(
            [{"area": "공정성", "priority": "MEDIUM", "finding": "위반 2건"}]
        )
        assert "공정성" in text and "위반 2건" in text


# ---------------------------------------------------------------------------
# _attach_llm_followup
# ---------------------------------------------------------------------------

class _FakeSession:
    def __init__(self, registry=None, agent_type="ops", region=None,
                 focus_keys=None, **kwargs):
        self.focus_keys = focus_keys

    def investigate(self, finding, verify=False):
        out = {"reasoning": "근본원인: teacher 미달", "n_tool_calls": 2,
               "tool_calls": [{"name": "read_distillation_fidelity"}],
               "_model_reasoned": True}
        if verify:
            out["grounding_check"] = {"checked": True, "grounded": True}
        return out

    def triage(self, finding, verify=False):
        out = {"reasoning": "관찰 권고", "triage": "MONITOR"}
        if verify:
            out["grounding_check"] = {"checked": True, "grounded": True}
        return out


class TestAttachLlmFollowup:
    def _report(self, tmp_path, payload=None):
        p = tmp_path / "ops_report.json"
        p.write_text(json.dumps(payload or {"status": "RED"}), encoding="utf-8")
        return p

    def test_red_runs_investigate_and_writes_json(self, tmp_path):
        p = self._report(tmp_path)
        mode = _attach_llm_followup(
            report_path=p, verdict="RED", finding="- CP4 [FAIL] fidelity",
            focus_keys=["CP4"], registry=object(), agent_type="ops",
            session_factory=_FakeSession,
        )
        assert mode == "investigate"
        data = json.loads(p.read_text(encoding="utf-8"))
        assert data["llm_followup"]["mode"] == "investigate"
        assert data["llm_followup"]["verdict"] == "RED"
        assert "근본원인" in data["llm_followup"]["reasoning"]

    def test_yellow_runs_triage(self, tmp_path):
        p = self._report(tmp_path, {"status": "YELLOW"})
        mode = _attach_llm_followup(
            report_path=p, verdict="YELLOW", finding="- CP2 [WARNING] drift",
            focus_keys=["CP2"], registry=object(), agent_type="ops",
            session_factory=_FakeSession,
        )
        assert mode == "triage"
        data = json.loads(p.read_text(encoding="utf-8"))
        assert data["llm_followup"]["triage"] == "MONITOR"

    def test_green_skips(self, tmp_path):
        p = self._report(tmp_path, {"status": "GREEN"})
        mode = _attach_llm_followup(
            report_path=p, verdict="GREEN", finding="x", focus_keys=[],
            registry=object(), agent_type="ops", session_factory=_FakeSession,
        )
        assert mode is None
        assert "llm_followup" not in json.loads(p.read_text(encoding="utf-8"))

    def test_no_registry_skips(self, tmp_path):
        p = self._report(tmp_path)
        mode = _attach_llm_followup(
            report_path=p, verdict="RED", finding="x", focus_keys=[],
            registry=None, agent_type="ops", session_factory=_FakeSession,
        )
        assert mode is None

    def test_session_failure_is_swallowed(self, tmp_path):
        p = self._report(tmp_path)

        class _Boom:
            def __init__(self, **kwargs):
                raise RuntimeError("bedrock down")

        mode = _attach_llm_followup(
            report_path=p, verdict="RED", finding="x", focus_keys=[],
            registry=object(), agent_type="ops", session_factory=_Boom,
        )
        assert mode is None  # 실패 swallow — 보고서 흐름 차단 금지
        assert "llm_followup" not in json.loads(p.read_text(encoding="utf-8"))


class TestEnvGate:
    def test_default_off(self, monkeypatch):
        monkeypatch.delenv("REPORTS_LLM_TRIAGE_ENABLED", raising=False)
        assert _llm_followup_enabled() is False

    def test_enabled_values(self, monkeypatch):
        for v in ("1", "true", "YES", "on"):
            monkeypatch.setenv("REPORTS_LLM_TRIAGE_ENABLED", v)
            assert _llm_followup_enabled() is True
        monkeypatch.setenv("REPORTS_LLM_TRIAGE_ENABLED", "0")
        assert _llm_followup_enabled() is False

    def test_verify_default_on_within_followup(self, monkeypatch):
        # 후속 조사 자체가 옵트인이므로 그 안에서 grounding 검증은 기본 on.
        monkeypatch.delenv("REPORTS_LLM_VERIFY_GROUNDING", raising=False)
        assert _llm_verify_enabled() is True
        for v in ("0", "false", "NO", "off"):
            monkeypatch.setenv("REPORTS_LLM_VERIFY_GROUNDING", v)
            assert _llm_verify_enabled() is False


# ---------------------------------------------------------------------------
# verify_grounding (PORT-03 — 할루시네이션 더블체크)
# ---------------------------------------------------------------------------

class TestVerifyGrounding:
    def _session_with_evidence(self) -> BedrockDialogSession:
        sess = BedrockDialogSession(registry=ToolRegistry(), agent_type="ops")
        sess._history.append(DialogTurn(
            role="assistant", content="",
            tool_calls=[{"toolUseId": "t1", "name": "detect_drift", "input": {}}],
            tool_results=[{"toolUseId": "t1", "result": {"psi": 0.31}}],
        ))
        return sess

    def test_unverifiable_without_tool_use(self):
        sess = BedrockDialogSession(registry=ToolRegistry(), agent_type="ops")
        out = sess.verify_grounding("PSI가 0.5로 급증했습니다")
        assert out["checked"] is False
        assert "도구 미사용" in out["note"]

    def test_collect_evidence_resolves_tool_name(self):
        sess = self._session_with_evidence()
        evidence = sess._collect_evidence()
        assert "detect_drift" in evidence
        assert "0.31" in evidence

    def test_parses_grounded_false_with_code_fence(self, monkeypatch):
        sess = self._session_with_evidence()
        reply = (
            "```json\n"
            '{"grounded": false, "issues": ["PSI 0.5는 도구 결과(0.31)와 모순"], '
            '"note": "수치 불일치"}\n'
            "```"
        )
        captured = {}

        def fake_call(messages, tool_config, system_override=None):
            captured["tool_config"] = tool_config
            captured["system"] = system_override
            return _text_response(reply)

        monkeypatch.setattr(sess, "_call_bedrock", fake_call)
        out = sess.verify_grounding("PSI가 0.5로 급증했습니다")
        assert out == {
            "checked": True,
            "grounded": False,
            "issues": ["PSI 0.5는 도구 결과(0.31)와 모순"],
            "note": "수치 불일치",
        }
        assert captured["tool_config"] == {}  # 검증 호출은 도구 없이
        assert "검증" in captured["system"]

    def test_unparseable_reply_returns_unchecked(self, monkeypatch):
        sess = self._session_with_evidence()
        monkeypatch.setattr(
            sess, "_call_bedrock", lambda *a, **k: _text_response("JSON 아님"),
        )
        out = sess.verify_grounding("결론")
        assert out["checked"] is False
        assert "검증 호출 실패" in out["note"]

    def test_investigate_verify_attaches_grounding_check(self, monkeypatch):
        sess = self._session_with_evidence()
        monkeypatch.setattr(sess, "chat", lambda prompt: "[원인] ...")
        monkeypatch.setattr(
            sess, "verify_grounding",
            lambda text: {"checked": True, "grounded": True, "issues": [], "note": ""},
        )
        out = sess.investigate("CP2 RED", verify=True)
        assert out["grounding_check"]["grounded"] is True
        # verify=False 면 키 자체가 없어야 한다
        assert "grounding_check" not in sess.investigate("CP2 RED")

    def test_triage_verify_attaches_grounding_check(self, monkeypatch):
        sess = self._session_with_evidence()
        monkeypatch.setattr(sess, "chat", lambda prompt: "[분류] MONITOR")
        monkeypatch.setattr(
            sess, "verify_grounding",
            lambda text: {"checked": True, "grounded": False, "issues": ["x"], "note": ""},
        )
        out = sess.triage("경고", verify=True)
        assert out["triage"] == "MONITOR"
        assert out["grounding_check"]["grounded"] is False

    def test_followup_records_grounding_check_in_report(self, tmp_path):
        p = tmp_path / "audit_report.json"
        p.write_text(json.dumps({"risk_level": "HIGH"}), encoding="utf-8")
        mode = _attach_llm_followup(
            report_path=p, verdict="HIGH", finding="- 공정성 [HIGH] 위반",
            focus_keys=["fairness"], registry=object(), agent_type="audit",
            session_factory=_FakeSession, verify=True,
        )
        assert mode == "investigate"
        data = json.loads(p.read_text(encoding="utf-8"))
        assert data["llm_followup"]["grounding_check"]["checked"] is True

    def test_followup_verify_off_omits_grounding_check(self, tmp_path):
        p = tmp_path / "ops_report.json"
        p.write_text(json.dumps({"status": "RED"}), encoding="utf-8")
        _attach_llm_followup(
            report_path=p, verdict="RED", finding="- CP2 [FAIL] x",
            focus_keys=["CP2"], registry=object(), agent_type="ops",
            session_factory=_FakeSession, verify=False,
        )
        data = json.loads(p.read_text(encoding="utf-8"))
        assert "grounding_check" not in data["llm_followup"]
