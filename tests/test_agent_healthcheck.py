"""PORT-07 tests — scripts/agent_healthcheck.py offline checks.

AWS-credential checks (c_credentials, c_bedrock_models, c_bedrock_live) are
NOT exercised here; only the offline contract checks that must always pass
in CI: tool schema↔implementation parity, routing-name existence,
case_store backend, and agent module imports.

Run: pytest tests/test_agent_healthcheck.py -v
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

_SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "agent_healthcheck.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("agent_healthcheck", _SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class TestOfflineChecks:
    def setup_method(self):
        self.hc = _load_module()

    def test_tool_contract_parity(self):
        ok, detail = self.hc.c_tool_contract()
        assert ok, detail

    def test_tool_routing_names_exist(self):
        ok, detail = self.hc.c_tool_routing()
        assert ok, detail

    def test_case_store_backend(self):
        ok, detail = self.hc.c_case_store()
        assert ok, detail
        assert "backend=" in detail

    def test_agent_imports(self):
        ok, detail = self.hc.c_imports()
        assert ok, detail

    def test_configured_model_ids_nonempty(self, monkeypatch):
        # pipeline.yaml llm_provider.bedrock 에서 model_id 를 읽어와야 한다
        # (healthcheck 가 config-driven 으로 동작하는지 — 하드코딩 금지).
        ids = self.hc._configured_model_ids()
        assert ids, "pipeline.yaml llm_provider.bedrock 에서 model_id 를 찾지 못함"
        assert all(isinstance(v, str) and v for v in ids.values())

        # consensus provider 도 같은 config 경로에서 model_id/region 을
        # 읽어야 한다 (_BedrockProvider 하드코딩 정리, 2026-06-10 감사 P3 —
        # _agent_dialog_model_id 와 동일 키 경로 패턴).
        import yaml
        from core.agent import pipeline_reports as pr

        assert "agent_consensus" in ids, (
            "pipeline.yaml llm_provider.bedrock.models 에 agent_consensus 슬롯 없음"
        )
        region, model_id = pr._consensus_bedrock_settings()
        assert model_id == ids["agent_consensus"]
        cfg = self.hc._pipeline_cfg()
        bedrock_cfg = (cfg.get("llm_provider") or {}).get("bedrock") or {}
        assert region == (
            bedrock_cfg.get("region") or (cfg.get("aws") or {}).get("region")
        )

        # config 읽기 실패 시 env → 종전 하드코딩 값과 동일한 기본값 폴백
        # (consensus 는 best-effort 경로 — 폴백으로 현행 동작 유지).
        def _boom(*_a, **_k):
            raise RuntimeError("yaml unreadable")

        monkeypatch.setattr(yaml, "safe_load", _boom)
        monkeypatch.setenv("AWS_REGION", "us-west-2")
        assert pr._consensus_bedrock_settings() == (
            "us-west-2", pr._CONSENSUS_FALLBACK_MODEL_ID,
        )
        monkeypatch.delenv("AWS_REGION", raising=False)
        monkeypatch.delenv("AWS_DEFAULT_REGION", raising=False)
        assert pr._consensus_bedrock_settings() == (
            pr._CONSENSUS_FALLBACK_REGION, pr._CONSENSUS_FALLBACK_MODEL_ID,
        )

    def test_region_resolves(self):
        assert self.hc._region()
