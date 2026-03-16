"""
Reason Generation Sub-package
===============================

Generates human-readable recommendation reasons, verifies them for
compliance, and manages the LLM abstraction layer.

Modules:
    template_engine  -- L1 template-based reason generation (no LLM).
    reverse_mapper   -- Feature importance to natural-language interpretation.
    self_checker     -- Rule-based compliance + optional LLM factuality check.
    llm_provider     -- Abstract LLM provider (Bedrock / OpenAI / Dummy).
"""

from .template_engine import TemplateEngine
from .reverse_mapper import ReverseMapper
from .self_checker import SelfChecker, CheckResult
from .llm_provider import AbstractLLMProvider, LLMProviderFactory

__all__ = [
    "TemplateEngine",
    "ReverseMapper",
    "SelfChecker",
    "CheckResult",
    "AbstractLLMProvider",
    "LLMProviderFactory",
]
