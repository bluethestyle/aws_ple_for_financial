"""
MarkerApplier - Sprint 3 M12.

Auto-inserts the AI-generation marker required by AI기본법 §31 (AI 생성
표시 의무) + §34 (고지 의무) on any LLM-produced recommendation text.

Driven by ``compliance.llm_marker`` block of pipeline.yaml. Idempotent:
if the marker substring is already present, the text is returned as-is.

Usage
-----
- ``apply(text)`` -> annotated text (or original if disabled / already present)
- ``MarkerApplier.from_config(pipeline_config)`` -> instance driven by YAML
- ``wrap_provider(llm_provider)`` -> wraps an ``AbstractLLMProvider`` so
  every ``generate()`` output is post-processed with ``apply``.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

__all__ = [
    "MarkerConfig",
    "MarkerApplier",
    "DEFAULT_MARKER_TEXT",
    "wrap_provider",
]


DEFAULT_MARKER_TEXT = "※ 본 추천 사유는 AI가 생성하였습니다. (AI기본법 §31)"

VALID_POSITIONS = ("append", "prepend")


@dataclass
class MarkerConfig:
    enabled: bool = True
    marker_text: str = DEFAULT_MARKER_TEXT
    position: str = "append"            # append | prepend
    separator: str = "\n\n"
    idempotent_key: str = "(AI기본법"   # substring used to detect existing marker

    def __post_init__(self) -> None:
        if self.position not in VALID_POSITIONS:
            raise ValueError(
                f"position={self.position!r} must be in {VALID_POSITIONS}"
            )
        if not self.marker_text.strip():
            raise ValueError("marker_text must be non-empty")
        if not self.idempotent_key.strip():
            raise ValueError("idempotent_key must be non-empty")

    @classmethod
    def from_dict(cls, data: Optional[Dict[str, Any]]) -> "MarkerConfig":
        if not data:
            return cls()
        return cls(
            enabled=bool(data.get("enabled", True)),
            marker_text=str(data.get("marker_text", DEFAULT_MARKER_TEXT)),
            position=str(data.get("position", "append")),
            separator=str(data.get("separator", "\n\n")),
            idempotent_key=str(
                data.get("idempotent_key", "(AI기본법")
            ),
        )


class MarkerApplier:
    """Post-processor that appends / prepends the AI-generation marker."""

    def __init__(self, config: Optional[MarkerConfig] = None) -> None:
        self._cfg = config or MarkerConfig()

    @classmethod
    def from_config(
        cls, pipeline_config: Optional[Dict[str, Any]] = None,
    ) -> "MarkerApplier":
        compliance = (pipeline_config or {}).get("compliance") or {}
        return cls(
            config=MarkerConfig.from_dict(compliance.get("llm_marker"))
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def apply(self, text: Optional[str]) -> str:
        """Return ``text`` with the marker applied per config."""
        if text is None:
            return ""
        base = str(text)
        if not self._cfg.enabled:
            return base
        if self._already_marked(base):
            return base
        marker = self._cfg.marker_text
        if self._cfg.position == "append":
            if not base:
                return marker
            return f"{base}{self._cfg.separator}{marker}"
        if not base:
            return marker
        return f"{marker}{self._cfg.separator}{base}"

    def has_marker(self, text: Optional[str]) -> bool:
        return self._already_marked(str(text or ""))

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _already_marked(self, text: str) -> bool:
        key = self._cfg.idempotent_key
        if key and key in text:
            return True
        # Also treat a full marker_text match as idempotent.
        return self._cfg.marker_text in text


# ---------------------------------------------------------------------------
# Provider wrapper
# ---------------------------------------------------------------------------

def wrap_provider(provider: Any, applier: MarkerApplier) -> Any:
    """Return a wrapper whose ``generate()`` output is marker-annotated.

    The wrapper delegates every other attribute to the original provider,
    so it can stand in wherever the original is used.
    """

    class _MarkerWrappedProvider:
        def __init__(self, inner: Any, marker: MarkerApplier) -> None:
            self._inner = inner
            self._marker = marker

        def generate(self, prompt: str, **kwargs: Any) -> str:
            out = self._inner.generate(prompt, **kwargs)
            return self._marker.apply(out)

        def is_available(self) -> bool:
            if hasattr(self._inner, "is_available"):
                return bool(self._inner.is_available())
            return True

        def __getattr__(self, name: str) -> Any:
            return getattr(self._inner, name)

    return _MarkerWrappedProvider(provider, applier)
