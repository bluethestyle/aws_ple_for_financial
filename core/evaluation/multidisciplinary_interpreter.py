"""
Multidisciplinary Feature Interpreter
=======================================

Interprets 24D multidisciplinary features into business language for
LLM grounding and recommendation explanation.

Four domains:
  - chemical_kinetics (6D): Consumption growth dynamics
  - epidemic_diffusion (5D): Product/category adoption patterns
  - interference (8D): Cross-category interference effects
  - crime_pattern (5D): Spatio-temporal routine analysis

Reference:
    gotothemoon/workspace/code/src/grounding/multidisciplinary_interpreter.py
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)

__all__ = [
    "MultidisciplinaryInterpreter",
    "FeatureInterpretation",
    "InterpretationRange",
]


# ======================================================================
# Data classes
# ======================================================================

@dataclass
class InterpretationRange:
    """Score range -> label + business impact mapping."""
    min_val: float
    max_val: float
    label: str
    business_impact: str
    recommendation: str


@dataclass
class FeatureInterpretation:
    """Interpretation result for a single domain."""
    domain: str                          # "chemical_kinetics" | "epidemic_diffusion" | etc.
    score: float                         # aggregated score (mean of sub-features)
    label: str                           # "급성장" / "안정" / etc.
    business_impact: str                 # Korean financial language
    recommendation: str                  # actionable recommendation
    confidence: float                    # 1 - std(sub-features), clamped [0,1]
    detail_breakdown: Dict[str, float] = field(default_factory=dict)


# ======================================================================
# Interpreter
# ======================================================================

class MultidisciplinaryInterpreter:
    """Interpret 24D multidisciplinary features into business language.

    All interpretation ranges and component names are defined as class-level
    constants so that no external config file is required (though the
    constructor accepts an optional config dict for overrides).

    Usage::

        interpreter = MultidisciplinaryInterpreter()
        result = interpreter.interpret(features_24d)
        text = interpreter.generate_text_explanation(features_24d)
    """

    # -- Feature slices within the 24D vector --------------------------

    DOMAINS: Dict[str, Dict[str, Any]] = {
        "chemical_kinetics": {
            "indices": list(range(0, 6)),        # 6D
            "description": "소비 성장 동역학",
        },
        "epidemic_diffusion": {
            "indices": list(range(6, 11)),       # 5D
            "description": "상품 확산 패턴",
        },
        "interference": {
            "indices": list(range(11, 19)),       # 8D
            "description": "채널 간 간섭 효과",
        },
        "crime_pattern": {
            "indices": list(range(19, 24)),       # 5D
            "description": "이상 행동 패턴",
        },
    }

    # -- Sub-feature component names (1:1 mapping with extractor) ------

    FEATURE_COMPONENTS: Dict[str, List[str]] = {
        "chemical_kinetics": [
            "new_category_activation_rate",
            "spending_half_life",
            "spending_acceleration",
            "dormancy_reactivation_rate",
            "catalyst_sensitivity",
            "saturation_proximity",
        ],
        "epidemic_diffusion": [
            "sir_susceptible",
            "sir_infected",
            "sir_recovered",
            "max_weekly_new_mcc",
            "category_lifecycle_mean",
        ],
        "interference": [
            "spectral_entropy",
            "weekly_harmonic_power",
            "cross_spectral_coherence",
            "dominant_period",
            "spectral_centroid_shift",
            "phase_locking_value",
            "mean_phase_difference",
            "constructive_interference_ratio",
        ],
        "crime_pattern": [
            "burstiness",
            "recurrence_period",
            "routine_breakpoint_count",
            "circular_variance",
            "max_amount_zscore",
        ],
    }

    # -- Display names for Korean text output --------------------------

    FEATURE_DISPLAY_NAMES: Dict[str, str] = {
        "new_category_activation_rate": "새 업종 시도율",
        "spending_half_life": "소비 주기 (일)",
        "spending_acceleration": "소비 속도 변화",
        "dormancy_reactivation_rate": "중단 업종 재이용율",
        "catalyst_sensitivity": "급여일 전후 반응도",
        "saturation_proximity": "최대 소비 근접도",
        "sir_susceptible": "미이용 업종 비율",
        "sir_infected": "최근 늘어난 업종 비율",
        "sir_recovered": "안 쓰게 된 업종 비율",
        "max_weekly_new_mcc": "주간 최대 새 업종 수",
        "category_lifecycle_mean": "업종 평균 이용 기간 (일)",
        "spectral_entropy": "소비 리듬 복잡도",
        "weekly_harmonic_power": "주간 주기 강도",
        "cross_spectral_coherence": "업종 간 리듬 동조도",
        "dominant_period": "소비 지배 주기 (일)",
        "spectral_centroid_shift": "소비 리듬 변화 방향",
        "phase_locking_value": "업종 간 위상 동기화",
        "mean_phase_difference": "업종 간 평균 위상차",
        "constructive_interference_ratio": "보강 간섭 업종 비율",
        "burstiness": "소비 몰림 정도",
        "recurrence_period": "반복 주기 (일)",
        "routine_breakpoint_count": "소비 패턴 변곡점 수",
        "circular_variance": "이용 시간대 분산",
        "max_amount_zscore": "이상 거래 가능성",
    }

    # -- Interpretation ranges per domain ------------------------------

    INTERPRETATION_RANGES: Dict[str, List[InterpretationRange]] = {
        "chemical_kinetics": [
            InterpretationRange(
                0.7, float("inf"), "급성장",
                "최근 새로운 업종을 빠르게 시도하고 소비 속도가 가속되고 있습니다",
                "지금 관심 가질 만한 새 업종 혜택을 적극 추천해 보세요",
            ),
            InterpretationRange(
                0.4, 0.7, "안정",
                "소비 범위가 조금씩 넓어지고 있으며 새 업종에 관심이 생기고 있습니다",
                "관련 업종을 단계적으로 제안하면 효과적입니다",
            ),
            InterpretationRange(
                0.2, 0.4, "둔화",
                "주로 기존에 이용하던 업종에서 안정적으로 소비합니다",
                "자주 이용하는 업종의 추가 혜택을 안내해 보세요",
            ),
            InterpretationRange(
                0.0, 0.2, "정체",
                "소비 습관이 매우 고정적이며 새로운 시도가 적습니다",
                "핵심 이용 업종의 맞춤 혜택에 집중하세요",
            ),
        ],
        "epidemic_diffusion": [
            InterpretationRange(
                0.6, float("inf"), "확산 중",
                "여러 새로운 업종을 적극 시도하며 오래 유지합니다",
                "요즘 인기 있는 업종이나 신규 서비스를 먼저 추천해 보세요",
            ),
            InterpretationRange(
                0.3, 0.6, "포화",
                "몇몇 새 업종을 시도하고 있으나 전체적으로는 신중합니다",
                "이미 성장 중인 관심 업종과 연결되는 혜택을 제안하세요",
            ),
            InterpretationRange(
                0.0, 0.3, "미확산",
                "아직 시도하지 않은 업종이 많으며 새 업종 채택이 느립니다",
                "인기 업종의 입문 혜택으로 관심을 유도해 보세요",
            ),
        ],
        "interference": [
            InterpretationRange(
                0.6, float("inf"), "시너지",
                "여러 업종을 같은 날 함께 이용하며 소비 트렌드가 비슷하게 움직입니다",
                "세트 혜택이나 묶음 할인을 적극 제안해 보세요",
            ),
            InterpretationRange(
                0.3, 0.6, "중립",
                "특정 업종들을 연결해서 소비하는 경향이 있습니다",
                "자주 함께 이용하는 업종끼리 순차적으로 혜택을 안내하세요",
            ),
            InterpretationRange(
                0.0, 0.3, "충돌",
                "각 업종을 따로따로 이용하는 편입니다",
                "업종별 개별 맞춤 혜택이 효과적입니다",
            ),
        ],
        "crime_pattern": [
            InterpretationRange(
                0.6, float("inf"), "정상",
                "정해진 시간대와 주기에 맞춰 규칙적으로 소비합니다",
                "소비 주기에 맞춘 정기 추천이 잘 맞습니다",
            ),
            InterpretationRange(
                0.3, 0.6, "주의",
                "대체로 일정한 소비 패턴이 있지만 가끔 변동이 있습니다",
                "주기적인 혜택 알림이 효과적입니다",
            ),
            InterpretationRange(
                0.0, 0.3, "이상",
                "소비 시점과 패턴이 일정하지 않습니다",
                "특정 이벤트나 시즌에 맞춘 추천이 좋습니다",
            ),
        ],
    }

    # -- Business context for LLM prompt injection ---------------------

    BUSINESS_CONTEXT: Dict[str, Dict[str, str]] = {
        "chemical_kinetics": {
            "domain": "소비 변화 속도",
            "source_model": "카테고리 전환·활성화 분석",
            "business_meaning": "새로운 업종 시도 속도, 소비 빈도 변화, 급여일 전후 반응",
            "key_indicator": "새 업종 시도율, 소비 주기",
        },
        "epidemic_diffusion": {
            "domain": "업종 탐색 패턴",
            "source_model": "업종 채택·이탈 패턴 분석",
            "business_meaning": "아직 안 써본 업종, 최근 늘어난 업종, 안 쓰게 된 업종 비율",
            "key_indicator": "미이용 업종 비율, 신규 업종 채택 속도",
        },
        "interference": {
            "domain": "업종 간 소비 리듬 분석",
            "source_model": "파동 물리학 (FFT·Hilbert·교차 스펙트럼)",
            "business_meaning": "소비 리듬 복잡도, 주간 주기 강도, 업종 간 위상 동기화",
            "key_indicator": "스펙트럼 엔트로피, 위상 고정값, 보강 간섭 비율",
        },
        "crime_pattern": {
            "domain": "소비 규칙성",
            "source_model": "소비 시간·주기 규칙성 분석",
            "business_meaning": "소비 간격의 규칙성, 반복 주기, 이용 시간대 집중도",
            "key_indicator": "소비 규칙성 점수, 반복 주기",
        },
    }

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        """Initialize interpreter.

        Args:
            config: Optional config dict. Reads ``config["interpretability"]``
                    for threshold overrides and output settings.
        """
        self._cfg = (config or {}).get("interpretability", {})
        logger.info("MultidisciplinaryInterpreter initialised")

    # ------------------------------------------------------------------
    # Public API -- single customer
    # ------------------------------------------------------------------

    def interpret(
        self,
        features_24d: np.ndarray,
    ) -> List[FeatureInterpretation]:
        """Interpret a single customer's 24D features.

        Args:
            features_24d: 1-D array of shape ``(24,)`` or ``(N,)`` where
                          ``N >= 24`` (extra dimensions are ignored).

        Returns:
            List of :class:`FeatureInterpretation`, one per domain.
        """
        vec = self._normalise_vector(features_24d)
        results: List[FeatureInterpretation] = []

        for domain_name, domain_cfg in self.DOMAINS.items():
            indices = domain_cfg["indices"]
            values = vec[indices]
            values = np.nan_to_num(values, nan=0.0, posinf=1.0, neginf=0.0)

            agg_score = float(np.mean(values))
            confidence = max(0.0, min(1.0, 1.0 - float(np.std(values))))

            # Range lookup
            label, impact, recommendation = self._classify_score(
                domain_name, agg_score,
            )

            # Detail breakdown
            components = self.FEATURE_COMPONENTS.get(domain_name, [])
            detail: Dict[str, float] = {}
            for i, comp_name in enumerate(components):
                if i < len(values):
                    detail[comp_name] = float(values[i])

            results.append(FeatureInterpretation(
                domain=domain_name,
                score=round(agg_score, 4),
                label=label,
                business_impact=impact,
                recommendation=recommendation,
                confidence=round(confidence, 4),
                detail_breakdown=detail,
            ))

        return results

    # ------------------------------------------------------------------
    # Public API -- batch
    # ------------------------------------------------------------------

    def interpret_batch(
        self,
        features: np.ndarray,
    ) -> List[Dict[str, Any]]:
        """Interpret a batch of customers.

        Args:
            features: 2-D array of shape ``(N, 24)``.

        Returns:
            List of dicts, each containing domain-level scores and labels.
            Suitable for conversion to ``pd.DataFrame``.
        """
        if features.ndim == 1:
            features = features.reshape(1, -1)

        rows: List[Dict[str, Any]] = []
        for i in range(features.shape[0]):
            interps = self.interpret(features[i])
            row: Dict[str, Any] = {"row_index": i}
            for interp in interps:
                row[f"{interp.domain}_score"] = interp.score
                row[f"{interp.domain}_label"] = interp.label
                row[f"{interp.domain}_impact"] = interp.business_impact
                row[f"{interp.domain}_confidence"] = interp.confidence
            row["summary"] = self._generate_summary(interps)
            rows.append(row)

        logger.info(
            "Batch interpreted %d customers across %d domains",
            len(rows), len(self.DOMAINS),
        )
        return rows

    # ------------------------------------------------------------------
    # Public API -- text generation
    # ------------------------------------------------------------------

    def generate_text_explanation(
        self,
        features_24d: np.ndarray,
        include_details: bool = True,
    ) -> str:
        """Generate Korean text explanation for LLM grounding.

        Args:
            features_24d: 1-D array of shape ``(24,)``.
            include_details: Whether to include per-component breakdowns.

        Returns:
            Multi-line Korean markdown string.
        """
        interps = self.interpret(features_24d)
        lines: List[str] = ["## 고객 소비 행동 분석 결과\n"]

        for interp in interps:
            context = self.BUSINESS_CONTEXT.get(interp.domain, {})
            domain_label = context.get("domain", interp.domain)

            lines.append(f"### {domain_label}")
            lines.append(
                f"- **분석 방법**: {context.get('source_model', '알 수 없음')}",
            )
            lines.append(f"- **종합 점수**: {interp.score:.2f}")
            lines.append(f"- **유형**: {interp.label}")
            lines.append(f"- **의미**: {interp.business_impact}")
            lines.append(f"- **추천 전략**: {interp.recommendation}")
            lines.append(f"- **신뢰도**: {interp.confidence:.0%}")

            if include_details and interp.detail_breakdown:
                lines.append("- **세부 지표**:")
                for comp, val in interp.detail_breakdown.items():
                    display = self.FEATURE_DISPLAY_NAMES.get(comp, comp)
                    lines.append(f"  - {display}: {val:.3f}")

            lines.append("")

        return "\n".join(lines)

    def to_json_serializable(
        self,
        features_24d: np.ndarray,
    ) -> Dict[str, Any]:
        """Return a fully JSON-serializable interpretation dict.

        Suitable for saving as ``analysis/multidisciplinary_interpretation.json``.
        """
        interps = self.interpret(features_24d)
        return {
            "summary": self._generate_summary(interps),
            "domains": {
                interp.domain: {
                    "score": interp.score,
                    "label": interp.label,
                    "business_impact": interp.business_impact,
                    "recommendation": interp.recommendation,
                    "confidence": interp.confidence,
                    "detail_breakdown": interp.detail_breakdown,
                }
                for interp in interps
            },
            "business_context": self.BUSINESS_CONTEXT,
        }

    def to_json_serializable_batch(
        self,
        features: np.ndarray,
        max_samples: int = 100,
    ) -> Dict[str, Any]:
        """Batch interpretation as JSON-serializable dict.

        Args:
            features: 2-D array ``(N, 24)``.
            max_samples: Maximum samples to include (for storage efficiency).

        Returns:
            Dict with sample interpretations and aggregate statistics.
        """
        if features.ndim == 1:
            features = features.reshape(1, -1)

        n = min(features.shape[0], max_samples)
        samples = []
        domain_scores: Dict[str, List[float]] = {d: [] for d in self.DOMAINS}

        for i in range(n):
            interps = self.interpret(features[i])
            sample: Dict[str, Any] = {"index": i}
            for interp in interps:
                sample[interp.domain] = {
                    "score": interp.score,
                    "label": interp.label,
                    "confidence": interp.confidence,
                }
                domain_scores[interp.domain].append(interp.score)
            samples.append(sample)

        # Aggregate statistics
        aggregate: Dict[str, Any] = {}
        for domain, scores in domain_scores.items():
            if scores:
                arr = np.array(scores)
                aggregate[domain] = {
                    "mean_score": round(float(np.mean(arr)), 4),
                    "std_score": round(float(np.std(arr)), 4),
                    "min_score": round(float(np.min(arr)), 4),
                    "max_score": round(float(np.max(arr)), 4),
                }

        return {
            "total_samples": features.shape[0],
            "interpreted_samples": n,
            "aggregate_statistics": aggregate,
            "samples": samples,
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _normalise_vector(vec: np.ndarray) -> np.ndarray:
        """Ensure vector is 1-D and at least 24 elements."""
        if vec.ndim == 2:
            vec = vec[0]
        if len(vec) < 24:
            vec = np.pad(vec, (0, 24 - len(vec)))
        return vec[:24]

    def _classify_score(
        self,
        domain: str,
        score: float,
    ) -> tuple:
        """Classify an aggregated score into label + impact."""
        ranges = self.INTERPRETATION_RANGES.get(domain, [])
        for r in ranges:
            if r.min_val <= score <= r.max_val:
                return r.label, r.business_impact, r.recommendation

        # Fallback
        return "알 수 없음", "정보 없음", "추가 분석 필요"

    @staticmethod
    def _generate_summary(interps: List[FeatureInterpretation]) -> str:
        """One-line summary for metadata / vector store."""
        parts = []
        for interp in interps:
            parts.append(f"{interp.domain}:{interp.label}({interp.score:.2f})")
        return " | ".join(parts)
