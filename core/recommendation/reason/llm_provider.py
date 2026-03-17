"""
LLM Provider Abstraction
=========================

Unified interface to call various LLM backends.

Implementations:
    BedrockProvider  -- AWS Bedrock (Claude / Titan / etc.)
    OpenAIProvider   -- OpenAI API (GPT-4, etc.)
    GeminiProvider   -- Google Gemini API (gemini-2.0-flash, gemini-2.5-pro)
    DummyProvider    -- Deterministic stub for unit tests.

Usage::

    provider = LLMProviderFactory.create(config)
    response = provider.generate("Summarise this document.")

Config example::

    llm_provider:
      backend: bedrock       # bedrock | openai | dummy
      bedrock:
        model_id: anthropic.claude-3-haiku-20240307-v1:0
        region: us-east-1
        max_tokens: 1024
        temperature: 0.2
      openai:
        model: gpt-4o-mini
        max_tokens: 1024
        temperature: 0.2
        api_key_env: OPENAI_API_KEY
      dummy:
        response: "This is a dummy LLM response."
"""

from __future__ import annotations

import json
import logging
import os
from abc import ABC, abstractmethod
from typing import Any, ClassVar, Dict, List, Optional, Type

logger = logging.getLogger(__name__)

__all__ = [
    "AbstractLLMProvider",
    "BedrockProvider",
    "OpenAIProvider",
    "GeminiProvider",
    "DummyProvider",
    "LLMProviderFactory",
]


# ---------------------------------------------------------------------------
# Abstract base
# ---------------------------------------------------------------------------

class AbstractLLMProvider(ABC):
    """Base interface for LLM backends."""

    def __init__(self, config: Dict[str, Any]) -> None:
        self.config = config

    @abstractmethod
    def generate(self, prompt: str, **kwargs) -> str:
        """Generate a completion for *prompt*.

        Args:
            prompt: User prompt text.
            **kwargs: Backend-specific overrides (temperature, max_tokens, ...).

        Returns:
            Generated text string.
        """

    @abstractmethod
    def is_available(self) -> bool:
        """Check whether this provider can service requests right now."""


# ---------------------------------------------------------------------------
# AWS Bedrock
# ---------------------------------------------------------------------------

class BedrockProvider(AbstractLLMProvider):
    """AWS Bedrock LLM provider.

    Requires ``boto3`` and valid AWS credentials (via env, profile, or
    instance role).
    """

    def __init__(self, config: Dict[str, Any]) -> None:
        super().__init__(config)
        self.model_id: str = config.get(
            "model_id", "anthropic.claude-3-haiku-20240307-v1:0",
        )
        self.region: str = config.get("region", "us-east-1")
        self.max_tokens: int = config.get("max_tokens", 1024)
        self.temperature: float = config.get("temperature", 0.2)
        self._client = None

    def _get_client(self):
        if self._client is None:
            try:
                import boto3
                self._client = boto3.client(
                    "bedrock-runtime", region_name=self.region,
                )
            except Exception as exc:
                logger.error("Failed to create Bedrock client: %s", exc)
                raise
        return self._client

    def generate(self, prompt: str, **kwargs) -> str:
        client = self._get_client()
        max_tokens = kwargs.get("max_tokens", self.max_tokens)
        temperature = kwargs.get("temperature", self.temperature)

        # Anthropic Messages API format for Bedrock
        body = json.dumps({
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": max_tokens,
            "temperature": temperature,
            "messages": [{"role": "user", "content": prompt}],
        })

        response = client.invoke_model(
            modelId=self.model_id,
            body=body,
            contentType="application/json",
        )
        response_body = json.loads(response["body"].read())

        # Extract text from Claude Messages API response
        content = response_body.get("content", [])
        if content and isinstance(content, list):
            return content[0].get("text", "")
        return str(response_body)

    def is_available(self) -> bool:
        try:
            self._get_client()
            return True
        except Exception:
            return False


# ---------------------------------------------------------------------------
# OpenAI
# ---------------------------------------------------------------------------

class OpenAIProvider(AbstractLLMProvider):
    """OpenAI API provider (GPT-4, GPT-4o-mini, etc.).

    Requires the ``openai`` package and an API key in an env variable.
    """

    def __init__(self, config: Dict[str, Any]) -> None:
        super().__init__(config)
        self.model: str = config.get("model", "gpt-4o-mini")
        self.max_tokens: int = config.get("max_tokens", 1024)
        self.temperature: float = config.get("temperature", 0.2)
        api_key_env: str = config.get("api_key_env", "OPENAI_API_KEY")
        self.api_key: str = os.environ.get(api_key_env, "")
        self._client = None

    def _get_client(self):
        if self._client is None:
            try:
                import openai
                self._client = openai.OpenAI(api_key=self.api_key)
            except Exception as exc:
                logger.error("Failed to create OpenAI client: %s", exc)
                raise
        return self._client

    def generate(self, prompt: str, **kwargs) -> str:
        client = self._get_client()
        max_tokens = kwargs.get("max_tokens", self.max_tokens)
        temperature = kwargs.get("temperature", self.temperature)

        response = client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_tokens,
            temperature=temperature,
        )
        return response.choices[0].message.content or ""

    def is_available(self) -> bool:
        return bool(self.api_key)


# ---------------------------------------------------------------------------
# Dummy (testing)
# ---------------------------------------------------------------------------

class DummyProvider(AbstractLLMProvider):
    """Deterministic dummy provider for testing.

    Always returns a fixed response string.

    Config::

        dummy:
          response: "Dummy response for testing."
    """

    def __init__(self, config: Dict[str, Any]) -> None:
        super().__init__(config)
        self.response: str = config.get(
            "response", "This is a dummy LLM response for testing.",
        )

    def generate(self, prompt: str, **kwargs) -> str:
        logger.debug("DummyProvider.generate called with prompt length=%d", len(prompt))
        return self.response

    def is_available(self) -> bool:
        return True


# ---------------------------------------------------------------------------
# Google Gemini
# ---------------------------------------------------------------------------

class GeminiProvider(AbstractLLMProvider):
    """Google Gemini API provider.

    Supports gemini-2.0-flash (fast, cheap, L2a bulk) and
    gemini-2.5-pro (quality, L2b validation).

    Requires: GEMINI_API_KEY environment variable or config.

    Config::

        gemini:
          api_key: ""              # or set GEMINI_API_KEY env var
          model_id: gemini-2.0-flash
          temperature: 0.3
          max_tokens: 500
    """

    def __init__(self, config: Dict[str, Any] = None) -> None:
        super().__init__(config or {})
        self._api_key: str = self.config.get("api_key") or os.environ.get("GEMINI_API_KEY", "")
        self._model: str = self.config.get("model_id", "gemini-2.0-flash")
        self._temperature: float = self.config.get("temperature", 0.3)
        self._max_tokens: int = self.config.get("max_tokens", 500)
        self._base_url: str = "https://generativelanguage.googleapis.com/v1beta"

    def generate(self, prompt: str, **kwargs) -> str:
        """Call Gemini API via REST (no SDK dependency)."""
        import urllib.request

        temperature = kwargs.get("temperature", self._temperature)
        max_tokens = kwargs.get("max_tokens", self._max_tokens)

        url = f"{self._base_url}/models/{self._model}:generateContent?key={self._api_key}"
        payload = {
            "contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {
                "temperature": temperature,
                "maxOutputTokens": max_tokens,
            },
        }

        req = urllib.request.Request(
            url,
            data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                result = json.loads(resp.read().decode())
                return result["candidates"][0]["content"]["parts"][0]["text"]
        except Exception as e:
            logger.warning("Gemini API call failed: %s", e)
            return ""

    def is_available(self) -> bool:
        return bool(self._api_key)


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

class LLMProviderFactory:
    """Create an LLM provider from config.

    Usage::

        provider = LLMProviderFactory.create(config)
    """

    _backends: ClassVar[Dict[str, Type[AbstractLLMProvider]]] = {
        "bedrock": BedrockProvider,
        "openai": OpenAIProvider,
        "gemini": GeminiProvider,
        "dummy": DummyProvider,
    }

    @classmethod
    def register(cls, name: str, provider_cls: Type[AbstractLLMProvider]) -> None:
        """Register a custom backend."""
        cls._backends[name] = provider_cls

    @classmethod
    def create(cls, config: Dict[str, Any]) -> AbstractLLMProvider:
        """Instantiate the configured backend.

        Args:
            config: Full pipeline config.  Reads ``config["llm_provider"]``.

        Returns:
            Concrete :class:`AbstractLLMProvider` instance.

        Raises:
            KeyError: If the requested backend is unknown.
        """
        llm_cfg = config.get("llm_provider", {})
        backend = llm_cfg.get("backend", "dummy")
        backend_config = llm_cfg.get(backend, {})

        if backend not in cls._backends:
            available = ", ".join(sorted(cls._backends))
            raise KeyError(
                f"Unknown LLM backend '{backend}'. Available: {available}"
            )

        logger.info("Creating LLM provider: backend=%s", backend)
        return cls._backends[backend](backend_config)
