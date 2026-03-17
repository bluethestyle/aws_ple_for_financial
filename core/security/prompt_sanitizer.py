"""
Prompt Sanitizer -- sensitivity classification, PII scrubbing, and LLM routing.

Ensures customer data never leaves the AWS VPC boundary:
- HIGH sensitivity (customer behavioral data) → Bedrock only (VPC internal)
- MEDIUM sensitivity (aggregated/anonymized) → Bedrock preferred, Gemini with scrubbing
- LOW sensitivity (no customer data) → Gemini OK (cost optimization)

3-step process before any LLM call:
  1. classify() — detect sensitivity level from prompt content
  2. scrub()    — remove/anonymize PII for external API transmission
  3. route()    — select LLM provider based on sensitivity

Usage::

    sanitizer = PromptSanitizer()

    # Before LLM call:
    sensitivity = sanitizer.classify(prompt)
    provider_name = sanitizer.route(sensitivity)

    if provider_name != "bedrock" and sensitivity != "LOW":
        prompt = sanitizer.scrub(prompt)

    provider = LLMProviderFactory.create({"backend": provider_name, ...})
    response = provider.generate(prompt)

    # Audit log
    sanitizer.log_audit(prompt_hash, provider_name, sensitivity)

금감원 기준 충족:
  - 금융클라우드 이용 가이드라인: 고객 데이터는 허가된 클라우드에서만 처리
  - 개인정보보호법 제17조: 제3자 제공 시 사전 동의
  - 금감원 AI 7대 원칙 ⑦: 보안성 확보
"""

from __future__ import annotations

import hashlib
import logging
import re
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


# ============================================================================
# Sensitivity levels
# ============================================================================

class Sensitivity:
    HIGH = "HIGH"      # Customer-specific behavioral data → Bedrock only
    MEDIUM = "MEDIUM"  # Aggregated/anonymized data → Bedrock preferred
    LOW = "LOW"        # No customer data → any provider OK


# ============================================================================
# Detection patterns
# ============================================================================

# Patterns indicating customer-specific data in the prompt
_HIGH_SENSITIVITY_PATTERNS = [
    # Customer identifiers (even if integer-indexed)
    r"고객\s*(ID|번호|식별)",
    r"customer[_\s]*(id|no|idx)",
    r"CSNO|CUST_NO|cust_id",

    # Behavioral specifics with concrete values
    r"월\s*평균\s*\d+\s*(건|회|원|만원)",
    r"거래\s*(건수|금액|횟수)\s*[:=]?\s*\d+",
    r"소득\s*(안정성|추세|등급)\s*[:=]?\s*[0-9.]+",
    r"한도\s*(소진율|사용률)\s*[:=]?\s*[0-9.]+",
    r"이탈\s*(위험|확률|점수)\s*[:=]?\s*[0-9.]+",
    r"churn[_\s]*(risk|prob|score)\s*[:=]?\s*[0-9.]+",

    # Segment with numeric detail
    r"세그먼트\s*[:=]?\s*\d+",
    r"cluster[_\s]*(id|prob)\s*[:=]?\s*\d+",

    # Feature values with specific numbers
    r"rfm[_\s]*\d+\s*[:=]?\s*[0-9.]+",
    r"feature[_\s]*\d+\s*[:=]?\s*[0-9.]+",
]

# Patterns indicating aggregated but still somewhat sensitive data
_MEDIUM_SENSITIVITY_PATTERNS = [
    r"(높|중|낮)(음|은|은\s*수준)",
    r"(상위|하위)\s*\d+%",
    r"평균\s*(대비|이상|이하)",
    r"등급\s*[:=]?\s*[A-F가-힣]",
]

# Korean PII patterns (for scrubbing)
_PII_PATTERNS = {
    "korean_name": re.compile(r"[가-힣]{2,4}(님|씨|고객)"),
    "phone": re.compile(r"01[016789]-?\d{3,4}-?\d{4}"),
    "email": re.compile(r"[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+"),
    "ssn": re.compile(r"\d{6}-?[1-4]\d{6}"),
    "card_number": re.compile(r"\d{4}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}"),
    "account_number": re.compile(r"\d{3}-?\d{2,6}-?\d{2,6}"),
    "business_reg": re.compile(r"\d{3}-?\d{2}-?\d{5}"),
}

# Numeric value pattern for range-ification
_NUMERIC_VALUE = re.compile(
    r"(\d+(?:\.\d+)?)\s*(건|회|원|만원|%|점|개월|일)"
)

# Compiled high/medium patterns
_HIGH_COMPILED = [re.compile(p, re.IGNORECASE) for p in _HIGH_SENSITIVITY_PATTERNS]
_MEDIUM_COMPILED = [re.compile(p, re.IGNORECASE) for p in _MEDIUM_SENSITIVITY_PATTERNS]


# ============================================================================
# Scrubbing helpers
# ============================================================================

def _rangeify_number(value: float, unit: str) -> str:
    """Convert a specific number to a range for anonymization.

    47건 → "40~50건대", 350만원 → "300~400만원대", 0.85 → "높은 수준"
    """
    if unit == "%":
        if value < 20:
            return "낮은 수준"
        elif value < 50:
            return "중간 수준"
        elif value < 80:
            return "높은 수준"
        else:
            return "매우 높은 수준"

    if unit in ("점",) or 0 < value < 1.5:
        # Scores (0~1 range)
        if value < 0.3:
            return "낮은 수준"
        elif value < 0.7:
            return "중간 수준"
        else:
            return "높은 수준"

    # Integer-like values: round to nearest bucket
    if value < 10:
        bucket = 5
    elif value < 100:
        bucket = 10
    elif value < 1000:
        bucket = 100
    else:
        bucket = 1000

    low = int(value // bucket) * bucket
    high = low + bucket
    return f"{low}~{high}{unit}대"


# ============================================================================
# PromptSanitizer
# ============================================================================

@dataclass
class SanitizeResult:
    """Result of prompt sanitization."""
    original_hash: str          # SHA256 hash of original prompt (for audit)
    sensitivity: str            # HIGH / MEDIUM / LOW
    provider: str               # routed LLM provider name
    scrubbed: bool              # whether scrubbing was applied
    scrub_count: int = 0        # number of items scrubbed
    pii_detected: List[str] = field(default_factory=list)  # PII types found


class PromptSanitizer:
    """Prompt sensitivity classification, PII scrubbing, and LLM routing.

    Ensures compliance with 금감원 AI security requirements by routing
    customer-specific prompts to VPC-internal LLM providers only.

    Args:
        internal_provider: Provider name for HIGH/MEDIUM sensitivity
            (default ``"bedrock"``).
        external_provider: Provider name for LOW sensitivity
            (default ``"gemini"``).
        scrub_for_external: Whether to scrub prompts before external
            API calls (default ``True``).
        audit_store: Optional callable for audit logging.
    """

    def __init__(
        self,
        internal_provider: str = "bedrock",
        external_provider: str = "gemini",
        scrub_for_external: bool = True,
        audit_store=None,
    ):
        self._internal = internal_provider
        self._external = external_provider
        self._scrub_for_external = scrub_for_external
        self._audit_store = audit_store
        self._audit_log: List[Dict[str, Any]] = []

    # ------------------------------------------------------------------
    # Step 1: Classify
    # ------------------------------------------------------------------

    def classify(self, prompt: str) -> str:
        """Classify prompt sensitivity level.

        Returns:
            ``"HIGH"``, ``"MEDIUM"``, or ``"LOW"``.
        """
        # Check for HIGH sensitivity patterns
        for pattern in _HIGH_COMPILED:
            if pattern.search(prompt):
                return Sensitivity.HIGH

        # Check for MEDIUM sensitivity patterns
        for pattern in _MEDIUM_COMPILED:
            if pattern.search(prompt):
                return Sensitivity.MEDIUM

        return Sensitivity.LOW

    # ------------------------------------------------------------------
    # Step 2: Scrub
    # ------------------------------------------------------------------

    def scrub(self, prompt: str) -> Tuple[str, int]:
        """Remove/anonymize PII and specific values from prompt.

        Returns:
            ``(scrubbed_prompt, scrub_count)``
        """
        scrubbed = prompt
        count = 0

        # Remove explicit PII patterns
        for pii_type, pattern in _PII_PATTERNS.items():
            matches = pattern.findall(scrubbed)
            if matches:
                scrubbed = pattern.sub(f"[{pii_type}]", scrubbed)
                count += len(matches)

        # Range-ify numeric values: "47건" → "40~50건대"
        def _replace_numeric(match):
            nonlocal count
            value = float(match.group(1))
            unit = match.group(2)
            count += 1
            return _rangeify_number(value, unit)

        scrubbed = _NUMERIC_VALUE.sub(_replace_numeric, scrubbed)

        # Remove integer indices that look like customer IDs
        scrubbed = re.sub(
            r"(customer_id|cust_id|CSNO|고객\s*ID)\s*[:=]?\s*\d+",
            r"\1: [익명]",
            scrubbed,
            flags=re.IGNORECASE,
        )
        if "[익명]" in scrubbed:
            count += 1

        # Remove segment/cluster specific numbers
        scrubbed = re.sub(
            r"(세그먼트|segment|cluster)\s*[:=]?\s*\d+",
            r"\1: [그룹]",
            scrubbed,
            flags=re.IGNORECASE,
        )

        return scrubbed, count

    # ------------------------------------------------------------------
    # Step 3: Route
    # ------------------------------------------------------------------

    def route(self, sensitivity: str) -> str:
        """Select LLM provider based on sensitivity level.

        HIGH   → internal provider (Bedrock, VPC 내부)
        MEDIUM → internal provider (Bedrock, 안전 우선)
        LOW    → external provider (Gemini, 비용 최적화)
        """
        if sensitivity in (Sensitivity.HIGH, Sensitivity.MEDIUM):
            return self._internal
        return self._external

    # ------------------------------------------------------------------
    # Combined: sanitize + route
    # ------------------------------------------------------------------

    def sanitize_and_route(
        self, prompt: str
    ) -> Tuple[str, str, SanitizeResult]:
        """Full sanitization pipeline: classify → scrub (if needed) → route.

        Returns:
            ``(final_prompt, provider_name, sanitize_result)``
        """
        prompt_hash = hashlib.sha256(prompt.encode("utf-8")).hexdigest()[:16]
        sensitivity = self.classify(prompt)
        provider = self.route(sensitivity)

        # Detect PII types present
        pii_detected = [
            pii_type for pii_type, pattern in _PII_PATTERNS.items()
            if pattern.search(prompt)
        ]

        scrubbed = False
        scrub_count = 0
        final_prompt = prompt

        # Scrub if routing to external provider and sensitivity is not LOW
        if provider == self._external and sensitivity != Sensitivity.LOW:
            # MEDIUM going to external (fallback case) → scrub
            if self._scrub_for_external:
                final_prompt, scrub_count = self.scrub(prompt)
                scrubbed = True
        elif provider != self._internal and pii_detected:
            # Any external call with PII detected → force scrub
            final_prompt, scrub_count = self.scrub(prompt)
            scrubbed = True

        result = SanitizeResult(
            original_hash=prompt_hash,
            sensitivity=sensitivity,
            provider=provider,
            scrubbed=scrubbed,
            scrub_count=scrub_count,
            pii_detected=pii_detected,
        )

        # Audit log
        self._log_audit(result)

        logger.info(
            "PromptSanitizer: sensitivity=%s, provider=%s, scrubbed=%s "
            "(scrub_count=%d, pii=%s, hash=%s)",
            sensitivity, provider, scrubbed,
            scrub_count, pii_detected, prompt_hash,
        )

        return final_prompt, provider, result

    # ------------------------------------------------------------------
    # Audit
    # ------------------------------------------------------------------

    def _log_audit(self, result: SanitizeResult) -> None:
        """Record sanitization event for regulatory compliance."""
        import time as _time

        record = {
            "prompt_hash": result.original_hash,
            "sensitivity": result.sensitivity,
            "provider": result.provider,
            "scrubbed": result.scrubbed,
            "scrub_count": result.scrub_count,
            "pii_detected": result.pii_detected,
            "timestamp": _time.strftime("%Y-%m-%dT%H:%M:%S"),
        }
        self._audit_log.append(record)

        if self._audit_store and callable(self._audit_store):
            try:
                self._audit_store(record)
            except Exception as e:
                logger.debug("Audit store write failed: %s", e)

    def get_audit_log(self) -> List[Dict[str, Any]]:
        """Return all sanitization audit records."""
        return list(self._audit_log)

    def get_stats(self) -> Dict[str, int]:
        """Return aggregate statistics."""
        stats = {"total": len(self._audit_log), "HIGH": 0, "MEDIUM": 0, "LOW": 0,
                 "scrubbed": 0, "pii_found": 0}
        for record in self._audit_log:
            stats[record["sensitivity"]] += 1
            if record["scrubbed"]:
                stats["scrubbed"] += 1
            if record["pii_detected"]:
                stats["pii_found"] += 1
        return stats
