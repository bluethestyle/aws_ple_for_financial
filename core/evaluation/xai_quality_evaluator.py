"""XAI quality assessment: fidelity, completeness, consistency.

Evaluates explanation quality along three axes aligned with Korean
financial regulation (AI Basic Act Article 34, Financial Consumer
Protection Act Article 19):

- **Fidelity**: proportion of IG Top-5 features reflected in explanation text
- **Completeness**: coverage of mandatory explanation items (5 categories)
- **Consistency**: Jaccard similarity for same product-segment explanations

Reference: gotothemoon/workspace/code/src/monitoring/xai_quality_evaluator.py
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# ======================================================================
# Mandatory explanation items (Financial Consumer Protection Act s.19)
# ======================================================================

REQUIRED_EXPLANATION_ITEMS: List[Dict[str, Any]] = [
    {
        "item_id": "product_type",
        "name_ko": "상품 유형",
        "detection_keywords": [
            "체크카드", "적금", "보험", "펀드", "카드", "상품", "금융",
            "계좌", "예금", "대출", "연금",
        ],
        "regulation": "금소법 §19①(1)",
        "required_for": ["all"],
    },
    {
        "item_id": "fee_structure",
        "name_ko": "수수료 구조",
        "detection_keywords": [
            "수수료", "연회비", "비용", "이용료", "무료", "할인율", "면제",
            "이자", "금리", "요금",
        ],
        "regulation": "금소법 §19①(2)",
        "required_for": ["all"],
    },
    {
        "item_id": "benefit_conditions",
        "name_ko": "혜택 조건",
        "detection_keywords": [
            "혜택", "할인", "캐시백", "포인트", "적립", "실적", "조건",
            "기준", "받으실", "우대", "보상", "이벤트",
        ],
        "regulation": "금소법 §19①(3)",
        "required_for": ["all"],
    },
    {
        "item_id": "risk_factors",
        "name_ko": "위험 요인",
        "detection_keywords": [
            "위험", "리스크", "손실", "변동", "주의", "유의", "안전",
        ],
        "regulation": "금소법 §19①(4)",
        "required_for": ["investment", "insurance"],
    },
    {
        "item_id": "cancellation_conditions",
        "name_ko": "해지 조건",
        "detection_keywords": [
            "해지", "해약", "취소", "위약", "철회", "중도", "해제",
            "환불", "만기",
        ],
        "regulation": "금소법 §19①(5)",
        "required_for": ["investment", "insurance"],
    },
]


@dataclass
class XAIReport:
    """Aggregate XAI quality evaluation result."""

    fidelity_score: float           # 0-1, proportion of IG Top-5 reflected
    completeness_score: float       # 0-1, coverage of mandatory items
    consistency_score: float        # 0-1, Jaccard for same product-segment
    per_explanation_fidelity: List[float] = field(default_factory=list)
    missing_items: List[str] = field(default_factory=list)
    thresholds: Dict[str, float] = field(default_factory=lambda: {
        "fidelity": 0.6,
        "completeness": 0.5,
        "consistency": 0.7,
    })

    @property
    def passed(self) -> bool:
        return (
            self.fidelity_score >= self.thresholds["fidelity"]
            and self.completeness_score >= self.thresholds["completeness"]
            and self.consistency_score >= self.thresholds["consistency"]
        )

    def to_dict(self) -> dict:
        return {
            "fidelity": round(self.fidelity_score, 4),
            "completeness": round(self.completeness_score, 4),
            "consistency": round(self.consistency_score, 4),
            "passed": self.passed,
            "missing_items": self.missing_items,
            "per_explanation_fidelity": [round(f, 4) for f in self.per_explanation_fidelity],
            "thresholds": self.thresholds,
        }


class XAIQualityEvaluator:
    """Quantitative XAI explanation quality evaluator.

    Mandatory explanation items per Korean Financial Consumer Protection
    Act Article 19 (5 categories).
    """

    # Mandatory explanation items
    MANDATORY_ITEMS = [
        "product_type",         # 상품 유형
        "fee_structure",        # 수수료 구조
        "benefit_conditions",   # 혜택 조건
        "risk_factors",         # 위험 요소
        "cancellation_terms",   # 해지 조건
    ]

    # Korean keywords for each mandatory item
    ITEM_KEYWORDS: Dict[str, List[str]] = {
        "product_type": [
            "상품", "계좌", "예금", "적금", "펀드", "대출", "카드", "보험", "연금",
        ],
        "fee_structure": [
            "수수료", "비용", "이자", "금리", "요금", "부담", "연회비", "이용료", "면제",
        ],
        "benefit_conditions": [
            "혜택", "우대", "할인", "적립", "보상", "이벤트", "캐시백", "포인트",
        ],
        "risk_factors": [
            "위험", "손실", "변동", "리스크", "주의", "유의", "안전",
        ],
        "cancellation_terms": [
            "해지", "중도", "취소", "환불", "철회", "만기", "해약", "위약",
        ],
    }

    # IG feature-name prefix -> Korean keyword mapping
    FEATURE_KEYWORD_MAP: Dict[str, List[str]] = {
        "spend_": ["지출", "사용", "소비", "결제"],
        "txn_count_": ["이용", "빈도", "횟수", "건수"],
        "merchant_": ["가맹점", "업종", "상호"],
        "mcc_": ["이용", "빈도", "가맹점"],
        "life_stage_": ["생애", "연령", "고객"],
        "age_": ["연령", "나이", "세대"],
        "benefit_": ["혜택", "상품", "금융"],
        "product_": ["상품", "카드", "금융"],
        "income_": ["소득", "급여", "수입"],
        "region_": ["지역", "거주"],
        "churn_": ["이탈", "유지"],
        "fatigue_": ["피로도", "빈도"],
        "card_": ["카드", "체크"],
        "ctx_": ["컨텍스트", "특성"],
    }

    def evaluate(
        self,
        explanations: List[str],
        ig_top_features: Optional[List[List[str]]] = None,
        product_segments: Optional[List[str]] = None,
    ) -> XAIReport:
        """Full XAI quality evaluation.

        Args:
            explanations: List of explanation texts.
            ig_top_features: Per-explanation list of IG top feature names.
            product_segments: Per-explanation product-segment identifier.

        Returns:
            XAIReport with fidelity, completeness, consistency scores.
        """
        fidelity, per_fidelity = (
            self._compute_fidelity(explanations, ig_top_features)
            if ig_top_features
            else (0.0, [])
        )
        completeness, missing = self._compute_completeness(explanations)
        consistency = (
            self._compute_consistency(explanations, product_segments)
            if product_segments
            else 1.0
        )

        return XAIReport(
            fidelity_score=fidelity,
            completeness_score=completeness,
            consistency_score=consistency,
            per_explanation_fidelity=per_fidelity,
            missing_items=missing,
        )

    # ------------------------------------------------------------------
    # Internal metrics
    # ------------------------------------------------------------------

    def _compute_fidelity(
        self,
        explanations: List[str],
        ig_top_features: List[List[str]],
    ) -> tuple:
        """Proportion of IG Top-5 features mentioned in explanation text."""
        scores: List[float] = []
        for text, top_feats in zip(explanations, ig_top_features):
            if not top_feats:
                scores.append(0.0)
                continue
            top5 = top_feats[:5]
            mentioned = 0
            for feat in top5:
                # Try direct match first
                if feat.lower() in text.lower():
                    mentioned += 1
                    continue
                # Try keyword mapping
                keywords = self._feature_to_keywords(feat)
                if any(kw in text for kw in keywords):
                    mentioned += 1
            scores.append(mentioned / min(len(top5), 5))

        mean_score = float(np.mean(scores)) if scores else 0.0
        return mean_score, scores

    def _feature_to_keywords(self, feature_name: str) -> List[str]:
        """Map IG feature name to Korean explanation keywords."""
        for prefix, keywords in self.FEATURE_KEYWORD_MAP.items():
            if feature_name.startswith(prefix):
                return keywords
        return [feature_name]

    def _compute_completeness(
        self, explanations: List[str],
    ) -> tuple:
        """Coverage of mandatory items across all explanations."""
        item_present: Dict[str, int] = {item: 0 for item in self.MANDATORY_ITEMS}

        for text in explanations:
            for item, keywords in self.ITEM_KEYWORDS.items():
                if any(kw in text for kw in keywords):
                    item_present[item] += 1

        n = len(explanations) if explanations else 1
        coverage = {item: count / n for item, count in item_present.items()}
        missing = [item for item, cov in coverage.items() if cov < 0.5]
        score = sum(coverage.values()) / len(coverage) if coverage else 0.0
        return score, missing

    def _compute_consistency(
        self,
        explanations: List[str],
        product_segments: List[str],
    ) -> float:
        """Jaccard similarity for same product-segment combinations."""
        from collections import defaultdict

        groups: Dict[str, List[set]] = defaultdict(list)
        for text, ps in zip(explanations, product_segments):
            groups[ps].append(set(text.lower().split()))

        similarities: List[float] = []
        for _ps, word_sets in groups.items():
            if len(word_sets) < 2:
                continue
            for i in range(len(word_sets)):
                for j in range(i + 1, min(i + 5, len(word_sets))):
                    inter = len(word_sets[i] & word_sets[j])
                    union = len(word_sets[i] | word_sets[j])
                    if union > 0:
                        similarities.append(inter / union)

        return float(np.mean(similarities)) if similarities else 1.0

    # ------------------------------------------------------------------
    # Rich batch evaluation (Stage B interpretability)
    # ------------------------------------------------------------------

    def evaluate_batch(
        self,
        batch_explanations: List[Dict[str, Any]],
        ig_results: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        """Batch evaluation producing a comprehensive report dict.

        This method accepts structured dicts (as produced by the
        TemplateEngine) rather than plain strings, making it suitable for
        pipeline-level integration (Stage 8.6).

        Args:
            batch_explanations: Each dict has ``customer_id``,
                ``product_id``, ``segment``, ``reasons`` (list of reason
                dicts with ``text`` key).
            ig_results: Each dict has ``customer_id``, ``product_id``,
                ``ig_top_features`` (list of ``(feat, score)`` tuples).

        Returns:
            Comprehensive evaluation dict with metrics, alerts, and
            segment breakdown.
        """
        # Index IG results by (customer_id, product_id)
        ig_index: Dict[Tuple[str, str], List[Tuple[str, float]]] = {}
        if ig_results:
            for ig in ig_results:
                key = (ig.get("customer_id", ""), ig.get("product_id", ""))
                ig_index[key] = ig.get("ig_top_features", [])

        total = len(batch_explanations)
        fidelity_scores: List[float] = []
        completeness_scores: List[float] = []
        consistency_inputs: List[Dict[str, str]] = []
        segment_data: Dict[str, Dict[str, Any]] = {}

        for expl in batch_explanations:
            cust_id = expl.get("customer_id", "")
            prod_id = expl.get("product_id", "")
            segment = expl.get("segment", "WARMSTART")
            reasons = expl.get("reasons", [])
            reasons_text = " ".join(r.get("text", "") for r in reasons)

            # Fidelity
            ig_feats = ig_index.get((cust_id, prod_id), [])
            if ig_feats:
                feat_names = [f[0] if isinstance(f, (list, tuple)) else f
                              for f in ig_feats[:5]]
                mentioned = 0
                for feat in feat_names:
                    if feat.lower() in reasons_text.lower():
                        mentioned += 1
                        continue
                    keywords = self._feature_to_keywords(feat)
                    if any(kw in reasons_text for kw in keywords):
                        mentioned += 1
                fid = mentioned / min(len(feat_names), 5)
            else:
                fid = 0.0
            fidelity_scores.append(fid)

            # Completeness
            comp = self._completeness_single(reasons_text)
            completeness_scores.append(comp)

            # Consistency input
            consistency_inputs.append({
                "product_id": prod_id,
                "segment": segment,
                "reasons_text": reasons_text,
            })

            # Per-segment
            if segment not in segment_data:
                segment_data[segment] = {
                    "fidelity": [], "completeness": [], "count": 0,
                }
            segment_data[segment]["fidelity"].append(fid)
            segment_data[segment]["completeness"].append(comp)
            segment_data[segment]["count"] += 1

        # Consistency (bigram Jaccard)
        consistency = self._consistency_bigram(consistency_inputs)

        avg_fid = sum(fidelity_scores) / len(fidelity_scores) if fidelity_scores else 0.0
        avg_comp = sum(completeness_scores) / len(completeness_scores) if completeness_scores else 0.0

        # Alerts
        alerts: List[Dict[str, Any]] = []
        if total > 0:
            low_fid_count = sum(1 for s in fidelity_scores if s < 0.6)
            low_fid_ratio = low_fid_count / total
            if low_fid_ratio >= 0.10:
                alerts.append({
                    "level": "warning",
                    "metric": "fidelity",
                    "value": round(low_fid_ratio, 4),
                    "message": (
                        f"Fidelity 기준 미달 비율 {low_fid_ratio:.1%} — "
                        "IG 피처와 설명 텍스트 불일치 다수"
                    ),
                })
        if avg_comp < 0.5:
            alerts.append({
                "level": "warning",
                "metric": "completeness",
                "value": round(avg_comp, 4),
                "message": f"평균 Completeness {avg_comp:.2f} — 금소법 §19 필수항목 충족률 미달",
            })
        if consistency < 0.7:
            alerts.append({
                "level": "warning",
                "metric": "consistency",
                "value": round(consistency, 4),
                "message": f"설명 일관성 {consistency:.2f} — 동일 상품-세그먼트 간 설명 편차 과다",
            })

        # Segment breakdown
        seg_breakdown: Dict[str, Any] = {}
        for seg, data in segment_data.items():
            f_scores = data["fidelity"]
            c_scores = data["completeness"]
            seg_breakdown[seg] = {
                "avg_fidelity": round(sum(f_scores) / len(f_scores), 4) if f_scores else 0.0,
                "avg_completeness": round(sum(c_scores) / len(c_scores), 4) if c_scores else 0.0,
                "count": data["count"],
            }

        # Distribution helper
        def _dist(scores: List[float]) -> Dict[str, int]:
            bins = {"0.0-0.2": 0, "0.2-0.4": 0, "0.4-0.6": 0, "0.6-0.8": 0, "0.8-1.0": 0}
            for s in scores:
                if s < 0.2:
                    bins["0.0-0.2"] += 1
                elif s < 0.4:
                    bins["0.2-0.4"] += 1
                elif s < 0.6:
                    bins["0.4-0.6"] += 1
                elif s < 0.8:
                    bins["0.6-0.8"] += 1
                else:
                    bins["0.8-1.0"] += 1
            return bins

        logger.info(
            "XAI batch evaluation: %d items, fidelity=%.2f, "
            "completeness=%.2f, consistency=%.2f",
            total, avg_fid, avg_comp, consistency,
        )

        return {
            "total_evaluated": total,
            "generated_at": datetime.utcnow().isoformat() + "Z",
            "regulatory_basis": "AI 기본법 §34①②, 금소법 §19",
            "metrics": {
                "avg_fidelity": round(avg_fid, 4),
                "avg_completeness": round(avg_comp, 4),
                "consistency": round(consistency, 4),
                "fidelity_distribution": _dist(fidelity_scores),
                "completeness_distribution": _dist(completeness_scores),
            },
            "segment_breakdown": seg_breakdown,
            "alerts": alerts,
        }

    def _completeness_single(
        self,
        text: str,
        product_type: str = "checkcard",
    ) -> float:
        """Completeness score for a single explanation text."""
        if not text:
            return 0.0
        satisfied = 0.0
        total_weight = 0.0
        for item in REQUIRED_EXPLANATION_ITEMS:
            if "all" in item["required_for"]:
                weight = 1.0
            elif product_type in item["required_for"]:
                weight = 1.0
            else:
                weight = 0.5
            total_weight += weight
            if any(kw in text for kw in item["detection_keywords"]):
                satisfied += weight
        return satisfied / total_weight if total_weight > 0 else 0.0

    def _consistency_bigram(
        self,
        inputs: List[Dict[str, str]],
    ) -> float:
        """Bigram Jaccard consistency across product-segment groups."""
        groups: Dict[Tuple[str, str], List[str]] = {}
        for inp in inputs:
            key = (inp.get("product_id", ""), inp.get("segment", ""))
            groups.setdefault(key, []).append(inp.get("reasons_text", ""))

        def _bigrams(t: str) -> Set[str]:
            return {t[i:i + 2] for i in range(len(t) - 1)}

        total_sim = 0.0
        total_weight = 0
        for _key, texts in groups.items():
            if len(texts) < 2:
                continue
            sims = []
            for i in range(len(texts)):
                for j in range(i + 1, len(texts)):
                    bg_a, bg_b = _bigrams(texts[i]), _bigrams(texts[j])
                    if not bg_a and not bg_b:
                        sims.append(1.0)
                    else:
                        union = len(bg_a | bg_b)
                        sims.append(len(bg_a & bg_b) / union if union else 0.0)
            if sims:
                total_sim += (sum(sims) / len(sims)) * len(texts)
                total_weight += len(texts)

        return round(total_sim / total_weight, 4) if total_weight > 0 else 1.0


__all__ = ["XAIQualityEvaluator", "XAIReport", "REQUIRED_EXPLANATION_ITEMS"]
