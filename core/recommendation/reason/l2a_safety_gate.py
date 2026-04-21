"""
L2a Safety Gate - 3-layer post-LLM validation (Sprint 2 S11).

Stack:
  Layer 1 - Parsing       (structural / schema validation)
  Layer 2 - Rule-based    (PII regex, banned phrases, length limits)
  Layer 3 - Quality check (optional; delegates to SelfChecker / LLM judge)

Any layer can veto. Failures are returned via `SafetyVerdict` so the caller
can fall back to L1 template reasons instead of serving the L2a output.

Config (``reason.safety_gate`` block in pipeline.yaml)::

    reason:
      safety_gate:
        parse:
          min_length: 8
          max_length: 600
          require_sentences: 1
        rules:
          banned_phrases: []            # free-form substrings
          banned_regex: []              # Python regex patterns
          pii_patterns:                  # Korean resident registration number,
            - "\\\\b\\\\d{6}-\\\\d{7}\\\\b"  # card numbers, phone, etc.
            - "\\\\b\\\\d{4}-\\\\d{4}-\\\\d{4}-\\\\d{4}\\\\b"
        quality:
          enabled: false
          min_score: 0.5
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

logger = logging.getLogger(__name__)

__all__ = [
    "SafetyGateConfig",
    "ParseConfig",
    "RuleConfig",
    "QualityConfig",
    "SafetyVerdict",
    "L2aSafetyGate",
    "build_l2a_safety_gate",
]


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class ParseConfig:
    min_length: int = 1
    max_length: int = 1000
    require_sentences: int = 0           # 0 disables the check

    @classmethod
    def from_dict(cls, data: Optional[Dict[str, Any]]) -> "ParseConfig":
        if not data:
            return cls()
        return cls(
            min_length=int(data.get("min_length", 1)),
            max_length=int(data.get("max_length", 1000)),
            require_sentences=int(data.get("require_sentences", 0)),
        )


@dataclass
class RuleConfig:
    banned_phrases: Sequence[str] = field(default_factory=tuple)
    banned_regex: Sequence[str] = field(default_factory=tuple)
    pii_patterns: Sequence[str] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        self._compiled_banned = [
            re.compile(p, re.IGNORECASE) for p in self.banned_regex
        ]
        self._compiled_pii = [
            re.compile(p) for p in self.pii_patterns
        ]

    @classmethod
    def from_dict(cls, data: Optional[Dict[str, Any]]) -> "RuleConfig":
        if not data:
            return cls()
        return cls(
            banned_phrases=tuple(data.get("banned_phrases", []) or ()),
            banned_regex=tuple(data.get("banned_regex", []) or ()),
            pii_patterns=tuple(data.get("pii_patterns", []) or ()),
        )


@dataclass
class QualityConfig:
    enabled: bool = False
    min_score: float = 0.5

    @classmethod
    def from_dict(cls, data: Optional[Dict[str, Any]]) -> "QualityConfig":
        if not data:
            return cls()
        return cls(
            enabled=bool(data.get("enabled", False)),
            min_score=float(data.get("min_score", 0.5)),
        )


@dataclass
class SafetyGateConfig:
    parse: ParseConfig = field(default_factory=ParseConfig)
    rules: RuleConfig = field(default_factory=RuleConfig)
    quality: QualityConfig = field(default_factory=QualityConfig)

    @classmethod
    def from_dict(cls, data: Optional[Dict[str, Any]]) -> "SafetyGateConfig":
        if not data:
            return cls()
        return cls(
            parse=ParseConfig.from_dict(data.get("parse")),
            rules=RuleConfig.from_dict(data.get("rules")),
            quality=QualityConfig.from_dict(data.get("quality")),
        )


# ---------------------------------------------------------------------------
# Verdict
# ---------------------------------------------------------------------------

@dataclass
class SafetyVerdict:
    passed: bool
    layer: str                    # "parse" | "rules" | "quality" | "all"
    findings: List[Dict[str, Any]] = field(default_factory=list)
    sanitized_text: Optional[str] = None     # rule-applied text if needed

    @property
    def should_fallback_to_l1(self) -> bool:
        return not self.passed


# ---------------------------------------------------------------------------
# Gate
# ---------------------------------------------------------------------------

class L2aSafetyGate:
    """3-layer safety gate for L2a LLM output."""

    def __init__(
        self,
        config: Optional[SafetyGateConfig] = None,
        quality_checker: Optional[Any] = None,
    ) -> None:
        self._cfg = config or SafetyGateConfig()
        self._quality_checker = quality_checker   # SelfChecker-compatible

    def validate(self, text: Optional[str]) -> SafetyVerdict:
        findings: List[Dict[str, Any]] = []
        if text is None:
            return SafetyVerdict(
                passed=False, layer="parse",
                findings=[{"layer": "parse", "issue": "text is None"}],
            )

        # ---- Layer 1: parse ----
        parse_ok, parse_findings = self._check_parse(text)
        if not parse_ok:
            return SafetyVerdict(
                passed=False, layer="parse", findings=parse_findings,
            )

        # ---- Layer 2: rules ----
        rule_ok, rule_findings = self._check_rules(text)
        if not rule_ok:
            return SafetyVerdict(
                passed=False, layer="rules", findings=rule_findings,
            )

        # ---- Layer 3: quality (optional) ----
        if self._cfg.quality.enabled and self._quality_checker is not None:
            q_ok, q_findings = self._check_quality(text)
            if not q_ok:
                return SafetyVerdict(
                    passed=False, layer="quality", findings=q_findings,
                )

        return SafetyVerdict(passed=True, layer="all")

    # ------------------------------------------------------------------
    # Layer 1 - Parse
    # ------------------------------------------------------------------

    def _check_parse(self, text: str):
        findings: List[Dict[str, Any]] = []
        n = len(text.strip())
        if n < self._cfg.parse.min_length:
            findings.append({
                "layer": "parse", "issue": "too_short",
                "length": n, "min": self._cfg.parse.min_length,
            })
        if n > self._cfg.parse.max_length:
            findings.append({
                "layer": "parse", "issue": "too_long",
                "length": n, "max": self._cfg.parse.max_length,
            })
        if self._cfg.parse.require_sentences > 0:
            sentence_count = sum(
                1 for ch in text if ch in ".!?。！？"
            )
            if sentence_count < self._cfg.parse.require_sentences:
                findings.append({
                    "layer": "parse", "issue": "insufficient_sentences",
                    "count": sentence_count,
                    "required": self._cfg.parse.require_sentences,
                })
        return not findings, findings

    # ------------------------------------------------------------------
    # Layer 2 - Rules
    # ------------------------------------------------------------------

    def _check_rules(self, text: str):
        findings: List[Dict[str, Any]] = []
        lowered = text.lower()
        for phrase in self._cfg.rules.banned_phrases:
            if phrase.lower() in lowered:
                findings.append({
                    "layer": "rules", "issue": "banned_phrase",
                    "match": phrase,
                })
        for pattern in self._cfg.rules._compiled_banned:
            m = pattern.search(text)
            if m:
                findings.append({
                    "layer": "rules", "issue": "banned_regex",
                    "pattern": pattern.pattern,
                    "match": m.group(0),
                })
        for pattern in self._cfg.rules._compiled_pii:
            m = pattern.search(text)
            if m:
                findings.append({
                    "layer": "rules", "issue": "pii_leak",
                    "pattern": pattern.pattern,
                    "match_preview": m.group(0)[:4] + "***",
                })
        return not findings, findings

    # ------------------------------------------------------------------
    # Layer 3 - Quality
    # ------------------------------------------------------------------

    def _check_quality(self, text: str):
        findings: List[Dict[str, Any]] = []
        try:
            result = self._quality_checker.check(text)
        except Exception as exc:
            findings.append({
                "layer": "quality", "issue": "checker_error",
                "error": str(exc),
            })
            return False, findings

        score = getattr(result, "score", None)
        if score is None and isinstance(result, dict):
            score = result.get("score")
        if score is None:
            return True, findings

        if score < self._cfg.quality.min_score:
            findings.append({
                "layer": "quality", "issue": "low_quality",
                "score": float(score),
                "min_score": self._cfg.quality.min_score,
            })
            return False, findings
        return True, findings


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def build_l2a_safety_gate(
    pipeline_config: Optional[Dict[str, Any]] = None,
    quality_checker: Optional[Any] = None,
) -> L2aSafetyGate:
    reason_cfg = (pipeline_config or {}).get("reason") or {}
    gate_cfg = SafetyGateConfig.from_dict(reason_cfg.get("safety_gate"))
    return L2aSafetyGate(config=gate_cfg, quality_checker=quality_checker)
