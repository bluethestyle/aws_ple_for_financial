"""PORT-10 tests — _embed_finding 의 Titan Embed v2 옵트인 provider + 폴백.

Run: pytest tests/test_agent_embed_provider.py -v
"""

from __future__ import annotations

import io
import json

import numpy as np

from core.agent.tool_wrappers import _embed_finding


class _FakeBedrockRuntime:
    def __init__(self, fail=False):
        self.fail = fail
        self.calls = []

    def invoke_model(self, modelId, body):
        self.calls.append({"modelId": modelId, "body": json.loads(body)})
        if self.fail:
            raise RuntimeError("bedrock down")
        emb = [0.1] * 1024
        return {"body": io.BytesIO(json.dumps({"embedding": emb}).encode())}


class TestEmbedFinding:
    def test_default_backend_is_hashing_384(self, monkeypatch):
        monkeypatch.delenv("AGENT_EMBED_BACKEND", raising=False)
        vec = _embed_finding("CP2 드리프트 경고")
        assert vec.shape == (384,)
        assert abs(float(np.linalg.norm(vec)) - 1.0) < 1e-5

    def test_titan_opt_in_returns_1024(self, monkeypatch):
        fake = _FakeBedrockRuntime()
        import boto3
        monkeypatch.setattr(boto3, "client", lambda *a, **k: fake)
        monkeypatch.setenv("AGENT_EMBED_BACKEND", "titan")
        vec = _embed_finding("CP2 드리프트 경고")
        assert vec.shape == (1024,)
        call = fake.calls[0]
        assert call["modelId"] == "amazon.titan-embed-text-v2:0"
        assert call["body"]["dimensions"] == 1024
        assert call["body"]["normalize"] is True

    def test_titan_model_id_env_override(self, monkeypatch):
        fake = _FakeBedrockRuntime()
        import boto3
        monkeypatch.setattr(boto3, "client", lambda *a, **k: fake)
        monkeypatch.setenv("AGENT_EMBED_BACKEND", "titan")
        monkeypatch.setenv("AGENT_EMBED_MODEL_ID", "amazon.titan-embed-text-v2:1")
        _embed_finding("x")
        assert fake.calls[0]["modelId"] == "amazon.titan-embed-text-v2:1"

    def test_titan_failure_falls_back_to_hashing(self, monkeypatch):
        import boto3
        monkeypatch.setattr(
            boto3, "client", lambda *a, **k: _FakeBedrockRuntime(fail=True),
        )
        monkeypatch.setenv("AGENT_EMBED_BACKEND", "titan")
        vec = _embed_finding("CP2 드리프트 경고")  # 예외 없이 폴백
        assert vec.shape == (384,)
