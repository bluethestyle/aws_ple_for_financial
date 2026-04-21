"""
AISecurityChecker (C4) — LLM prompt & output hardening.

Complements the existing `PromptSanitizer` (sensitivity classification +
PII scrubbing + VPC routing) with two hardening layers that the on-prem
reference system carries but AWS lacked:

1. **Prompt injection detection** — catches known jailbreak / instruction-
   override patterns before the prompt reaches any LLM provider.
2. **Output scanning** — inspects the model's reply for system-prompt
   leaks, PII echoes, and tool-/role-breakout attempts.

Findings are returned as structured verdicts so the caller (typically
L2a LLM rewrite) can decide whether to escalate, fall back, or block.
Every verdict carries a severity so upstream observability can prioritise.

Design
------
- Rule-based (regex + keyword list). Stable + auditable + no model
  dependency. Rule catalogue is config-driven so security ops can add
  patterns without code changes.
- Stateless. Callers instantiate once (rule compilation) and call
  :meth:`check_prompt` / :meth:`check_output` on every request.
- Works with any `AbstractLLMProvider` — see :func:`wrap_provider`.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence

logger = logging.getLogger(__name__)

__all__ = [
    "SecurityFinding",
    "SecurityVerdict",
    "AISecurityConfig",
    "AISecurityChecker",
    "DEFAULT_PROMPT_INJECTION_PATTERNS",
    "DEFAULT_OUTPUT_LEAK_PATTERNS",
    "wrap_provider",
]


# ---------------------------------------------------------------------------
# Default rule catalogues
# ---------------------------------------------------------------------------

# Known prompt-injection / jailbreak patterns. Public catalogue; operators
# should extend via pipeline.yaml::security.ai_checker.
DEFAULT_PROMPT_INJECTION_PATTERNS: List[str] = [
    r"(?i)ignore\s+(all\s+)?previous\s+instructions",
    r"(?i)disregard\s+(all\s+)?above",
    r"(?i)forget\s+(your|all|everything)\s+(rules|instructions|prompt)",
    r"(?i)\byou\s+are\s+now\s+(?:a|an)\s+\w+",                # role swap
    r"(?i)pretend\s+(you\s+are|to\s+be)\s+",
    r"(?i)\bdeveloper\s+mode\b",
    r"(?i)\bjailbreak\b",
    r"(?i)\bDAN\s+(mode|prompt)\b",
    r"(?i)do\s+anything\s+now",
    r"(?i)respond\s+as\s+(?:if\s+you\s+)?(?:have\s+no|without)\s+restrictions",
    r"(?i)reveal\s+(your|the)\s+(system|initial)\s+prompt",
    r"(?i)show\s+me\s+(your|the)\s+(instructions|system\s+prompt)",
    r"(?i)output\s+(your|the)\s+(instructions|prompt|rules)",
    r"(?i)base64\s*:\s*[A-Za-z0-9+/=]{20,}",                 # encoded payload
    # Tool / function-calling breakouts
    r"(?i)\bcall\s+(?:the|any)\s+(?:tool|function)\s+(?:named|called)\b",
    r"(?i)<\s*tool[_\s]*call\s*>",
]


# Patterns that should *not* appear in an LLM's reply. Matching means the
# model leaked something it should not have or is being manipulated.
DEFAULT_OUTPUT_LEAK_PATTERNS: List[str] = [
    # System-prompt / meta-prompt echo
    r"(?i)system\s+prompt\s*[:=]",
    r"(?i)my\s+(?:original\s+)?instructions?\s+(?:were|are)\s*[:=]",
    r"(?i)the\s+rules?\s+i\s+(?:was\s+)?given\s*[:=]",
    r"(?i)i\s+am\s+(?:actually|really)\s+(?:a|an)\s+\w+\s+(?:AI|model|assistant)",
    # Role breakout in response
    r"(?i)as\s+(?:a|an)\s+\w+\s+(?:without\s+restrictions|in\s+developer\s+mode)",
    # PII echoes (Korean resident reg number, card numbers)
    r"\b\d{6}[-\s]?\d{7}\b",
    r"\b\d{4}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}\b",
]


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

SEVERITY_LEVELS = ("low", "medium", "high", "critical")


@dataclass
class SecurityFinding:
    layer: str              # "prompt_injection" | "output_leak"
    rule: str               # pattern that matched
    match_preview: str      # first 80 chars of match, for audit
    severity: str = "high"
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.severity not in SEVERITY_LEVELS:
            raise ValueError(
                f"severity={self.severity!r} must be in {SEVERITY_LEVELS}"
            )


@dataclass
class SecurityVerdict:
    passed: bool
    findings: List[SecurityFinding] = field(default_factory=list)

    @property
    def should_block(self) -> bool:
        return not self.passed and any(
            f.severity in ("high", "critical") for f in self.findings
        )

    @property
    def highest_severity(self) -> Optional[str]:
        if not self.findings:
            return None
        order = {s: i for i, s in enumerate(SEVERITY_LEVELS)}
        return max(self.findings, key=lambda f: order[f.severity]).severity


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class AISecurityConfig:
    enabled: bool = True
    prompt_injection_patterns: Sequence[str] = field(
        default_factory=lambda: list(DEFAULT_PROMPT_INJECTION_PATTERNS)
    )
    output_leak_patterns: Sequence[str] = field(
        default_factory=lambda: list(DEFAULT_OUTPUT_LEAK_PATTERNS)
    )
    prompt_injection_severity: str = "high"
    output_leak_severity: str = "high"
    max_matches_reported: int = 10

    @classmethod
    def from_dict(cls, data: Optional[Dict[str, Any]]) -> "AISecurityConfig":
        if not data:
            return cls()
        return cls(
            enabled=bool(data.get("enabled", True)),
            prompt_injection_patterns=list(
                data.get("prompt_injection_patterns",
                         DEFAULT_PROMPT_INJECTION_PATTERNS)
            ),
            output_leak_patterns=list(
                data.get("output_leak_patterns",
                         DEFAULT_OUTPUT_LEAK_PATTERNS)
            ),
            prompt_injection_severity=str(
                data.get("prompt_injection_severity", "high")
            ),
            output_leak_severity=str(
                data.get("output_leak_severity", "high")
            ),
            max_matches_reported=int(data.get("max_matches_reported", 10)),
        )


# ---------------------------------------------------------------------------
# Checker
# ---------------------------------------------------------------------------

class AISecurityChecker:
    """Prompt-injection + output-leak scanner for LLM pipelines."""

    def __init__(self, config: Optional[AISecurityConfig] = None) -> None:
        self._cfg = config or AISecurityConfig()
        self._compiled_prompt = [
            re.compile(p) for p in self._cfg.prompt_injection_patterns
        ]
        self._compiled_output = [
            re.compile(p) for p in self._cfg.output_leak_patterns
        ]

    # ------------------------------------------------------------------
    # Prompt side
    # ------------------------------------------------------------------

    def check_prompt(self, prompt: Optional[str]) -> SecurityVerdict:
        """Scan a caller-supplied prompt for injection patterns."""
        if not self._cfg.enabled or not prompt:
            return SecurityVerdict(passed=True)
        findings = self._scan(
            prompt,
            self._compiled_prompt,
            layer="prompt_injection",
            severity=self._cfg.prompt_injection_severity,
        )
        return SecurityVerdict(passed=not findings, findings=findings)

    # ------------------------------------------------------------------
    # Output side
    # ------------------------------------------------------------------

    def check_output(self, output: Optional[str]) -> SecurityVerdict:
        """Scan an LLM reply for leaked system prompts / PII echoes."""
        if not self._cfg.enabled or not output:
            return SecurityVerdict(passed=True)
        findings = self._scan(
            output,
            self._compiled_output,
            layer="output_leak",
            severity=self._cfg.output_leak_severity,
        )
        return SecurityVerdict(passed=not findings, findings=findings)

    # ------------------------------------------------------------------
    # Combined pass
    # ------------------------------------------------------------------

    def check_exchange(
        self, prompt: str, output: str,
    ) -> Dict[str, SecurityVerdict]:
        return {
            "prompt": self.check_prompt(prompt),
            "output": self.check_output(output),
        }

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _scan(
        self,
        text: str,
        patterns: Iterable[re.Pattern],
        layer: str,
        severity: str,
    ) -> List[SecurityFinding]:
        findings: List[SecurityFinding] = []
        cap = self._cfg.max_matches_reported
        for pattern in patterns:
            if len(findings) >= cap:
                break
            m = pattern.search(text)
            if m is None:
                continue
            match_text = m.group(0)
            findings.append(SecurityFinding(
                layer=layer,
                rule=pattern.pattern,
                match_preview=match_text[:80],
                severity=severity,
            ))
        return findings


# ---------------------------------------------------------------------------
# Provider wrapper
# ---------------------------------------------------------------------------

def wrap_provider(
    provider: Any,
    checker: AISecurityChecker,
    on_prompt_block: Optional[Callable[[SecurityVerdict], str]] = None,
    on_output_block: Optional[Callable[[SecurityVerdict], str]] = None,
) -> Any:
    """Wrap any LLM provider (`.generate(prompt)`) with AISecurityChecker.

    - ``on_prompt_block``: called when the prompt is blocked. Should
      return a safe canned reply string. Defaults to a static refusal.
    - ``on_output_block``: same but for output-side blocks.

    All other provider attributes are delegated through `__getattr__`.
    """

    default_refusal = (
        "죄송합니다. 해당 요청은 보안 정책에 따라 처리할 수 없습니다."
    )

    class _SecurityWrappedProvider:
        def __init__(self, inner: Any) -> None:
            self._inner = inner
            self._checker = checker
            self._on_prompt = on_prompt_block or (lambda _v: default_refusal)
            self._on_output = on_output_block or (lambda _v: default_refusal)

        def generate(self, prompt: str, **kwargs: Any) -> str:
            prompt_verdict = self._checker.check_prompt(prompt)
            if prompt_verdict.should_block:
                logger.warning(
                    "AISecurityChecker blocked prompt: %s",
                    [f.rule for f in prompt_verdict.findings],
                )
                return self._on_prompt(prompt_verdict)
            reply = self._inner.generate(prompt, **kwargs)
            output_verdict = self._checker.check_output(reply)
            if output_verdict.should_block:
                logger.warning(
                    "AISecurityChecker blocked output: %s",
                    [f.rule for f in output_verdict.findings],
                )
                return self._on_output(output_verdict)
            return reply

        def is_available(self) -> bool:
            return getattr(self._inner, "is_available", lambda: True)()

        def __getattr__(self, name: str) -> Any:
            return getattr(self._inner, name)

    return _SecurityWrappedProvider(provider)
