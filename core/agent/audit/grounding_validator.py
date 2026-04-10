"""
Grounding Validator for Audit Agent AV3
=========================================

Validates that recommendation reason text is grounded in actual
IG (Integrated Gradients) feature attributions.

Quality formula:
    Q = 0.30 × Faithfulness + 0.25 × Grounding + 0.25 × Compliance + 0.20 × Readability

This module computes the Grounding and Readability components.
Faithfulness comes from XAIQualityEvaluator, Compliance from SelfChecker.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

__all__ = ["GroundingValidator", "GroundingResult", "ReasonQualityScore"]


@dataclass
class GroundingResult:
    """Result of grounding validation for a single reason."""
    reason_text: str
    ig_top_k: List[str]  # feature names from IG
    mentioned_features: List[str]  # features referenced in text
    grounding_score: float  # mentioned / total top_k
    ungrounded_claims: List[str] = field(default_factory=list)  # text segments not backed by features


@dataclass
class ReasonQualityScore:
    """Composite quality score for a recommendation reason."""
    faithfulness: float = 0.0
    grounding: float = 0.0
    compliance: float = 0.0
    readability: float = 0.0

    @property
    def overall(self) -> float:
        """Weighted quality score: 0.30F + 0.25G + 0.25C + 0.20R"""
        return (
            0.30 * self.faithfulness
            + 0.25 * self.grounding
            + 0.25 * self.compliance
            + 0.20 * self.readability
        )


class GroundingValidator:
    """Validates reason text grounding against IG feature attributions.

    Args:
        feature_glossary: Dict mapping feature names to Korean display names.
            e.g., {"spend_monthly": "월 평균 지출", "txn_count_3m": "3개월 거래 횟수"}
        config: Optional config dict.
    """

    def __init__(
        self,
        feature_glossary: Optional[Dict[str, str]] = None,
        config: Optional[Dict[str, Any]] = None,
    ) -> None:
        self._glossary = feature_glossary or {}
        cfg = config or {}
        self._min_grounding = cfg.get("min_grounding_score", 0.5)

        # Korean readability thresholds
        self._max_sentence_length = cfg.get("max_sentence_length", 80)
        self._max_jargon_ratio = cfg.get("max_jargon_ratio", 0.15)

        # Financial jargon patterns (Korean)
        self._jargon_patterns = [
            re.compile(p) for p in cfg.get("jargon_patterns", [
                r"IG\s*기여도",
                r"PSI",
                r"AUC",
                r"logit",
                r"sigmoid",
                r"softmax",
                r"attribution",
                r"perturbation",
            ])
        ]

    def validate(
        self,
        reason_text: str,
        ig_top_features: List[Dict[str, Any]],
    ) -> GroundingResult:
        """Validate grounding of reason text against IG features.

        Args:
            reason_text: The generated recommendation reason text.
            ig_top_features: List of dicts with at least "name" key (feature name).
                May also have "text" (Korean interpretation), "value", "ig_score".

        Returns:
            GroundingResult with grounding score and details.
        """
        if not reason_text or not ig_top_features:
            return GroundingResult(
                reason_text=reason_text,
                ig_top_k=[],
                mentioned_features=[],
                grounding_score=0.0,
            )

        # Extract feature names
        ig_names = [f.get("name", "") for f in ig_top_features if f.get("name")]

        # Build search terms: feature name + Korean glossary name + interpretation text
        search_terms: Dict[str, List[str]] = {}
        for feat in ig_top_features:
            name = feat.get("name", "")
            if not name:
                continue
            terms = [name]
            # Add Korean glossary name
            if name in self._glossary:
                terms.append(self._glossary[name])
            # Add interpretation text keywords
            interp = feat.get("text", "")
            if interp:
                # Extract key noun phrases (simple: words > 2 chars)
                terms.extend([w for w in interp.split() if len(w) > 2])
            search_terms[name] = terms

        # Check which features are mentioned in reason text
        mentioned = []
        for name, terms in search_terms.items():
            for term in terms:
                if term.lower() in reason_text.lower():
                    mentioned.append(name)
                    break

        grounding_score = len(mentioned) / len(ig_names) if ig_names else 0.0

        return GroundingResult(
            reason_text=reason_text,
            ig_top_k=ig_names,
            mentioned_features=mentioned,
            grounding_score=round(grounding_score, 4),
        )

    def compute_readability(self, text: str) -> float:
        """Compute readability score (0.0 to 1.0).

        Factors:
        - Sentence length: shorter is more readable
        - Jargon ratio: lower is more readable
        - Vague expressions: fewer is more readable
        """
        if not text:
            return 0.0

        # Sentence length score
        sentences = re.split(r'[.!?。]\s*', text)
        sentences = [s for s in sentences if s.strip()]
        if sentences:
            avg_len = sum(len(s) for s in sentences) / len(sentences)
            length_score = max(0.0, 1.0 - max(0.0, avg_len - 30) / self._max_sentence_length)
        else:
            length_score = 0.5

        # Jargon ratio score
        total_words = len(text.split())
        if total_words > 0:
            jargon_count = sum(1 for p in self._jargon_patterns if p.search(text))
            jargon_ratio = jargon_count / total_words
            jargon_score = max(0.0, 1.0 - jargon_ratio / self._max_jargon_ratio)
        else:
            jargon_score = 0.5

        # Vague expression score
        vague_patterns = [r"어느\s*정도", r"다소", r"약간", r"상당히", r"매우\s*다양"]
        vague_count = sum(1 for p in vague_patterns if re.search(p, text))
        vague_score = max(0.0, 1.0 - vague_count * 0.2)

        return round(
            0.4 * length_score + 0.35 * jargon_score + 0.25 * vague_score,
            4,
        )

    def compute_quality_score(
        self,
        reason_text: str,
        ig_top_features: List[Dict[str, Any]],
        faithfulness: float = 0.0,
        compliance: float = 1.0,
    ) -> ReasonQualityScore:
        """Compute composite quality score.

        Args:
            reason_text: The recommendation reason text.
            ig_top_features: IG feature list.
            faithfulness: From XAIQualityEvaluator (external).
            compliance: From SelfChecker (1.0 if pass, 0.0 if reject).
        """
        grounding_result = self.validate(reason_text, ig_top_features)
        readability = self.compute_readability(reason_text)

        return ReasonQualityScore(
            faithfulness=faithfulness,
            grounding=grounding_result.grounding_score,
            compliance=compliance,
            readability=readability,
        )

    def batch_validate(
        self,
        cases: List[Dict[str, Any]],
    ) -> List[GroundingResult]:
        """Validate grounding for a batch of cases.

        Args:
            cases: List of dicts with "reason_text" and "ig_top_features" keys.
        """
        return [
            self.validate(c.get("reason_text", ""), c.get("ig_top_features", []))
            for c in cases
        ]
