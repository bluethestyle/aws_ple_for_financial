"""
Simplified Template Reason Engine for Ablation
================================================

Takes IG top-3 feature indices per sample, maps feature indices to
6 template categories, and generates templated reason text.

This is a lightweight version for ablation reports -- not production.
For production, use :class:`core.recommendation.reason.template_engine.TemplateEngine`.

Reference:
    gotothemoon/workspace/code/src/grounding/template_reason_engine.py
"""

from __future__ import annotations

import hashlib
import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

__all__ = ["TemplateReasonEngine"]


# ---------------------------------------------------------------------------
# Category templates (6 categories, simplified from reference 30-variant pool)
# ---------------------------------------------------------------------------

CATEGORIES: Dict[str, str] = {
    "spending_pattern": "소비 패턴 분석에 기반하여",
    "frequency_pattern": "거래 빈도 분석에 기반하여",
    "life_stage": "고객 생애주기 분석에 기반하여",
    "benefit_match": "혜택 적합도 분석에 기반하여",
    "portfolio": "포트폴리오 다양성 분석에 기반하여",
    "risk": "리스크 프로파일 분석에 기반하여",
}

CATEGORY_TEMPLATES: Dict[str, str] = {
    "spending_pattern": (
        "{category_reason} 고객님의 주요 지출 영역에서 높은 혜택을 "
        "제공하는 상품으로 분석되었습니다."
    ),
    "frequency_pattern": (
        "{category_reason} 고객님의 이용 빈도에 최적화된 "
        "상품으로 선정되었습니다."
    ),
    "life_stage": (
        "{category_reason} 고객님의 현재 생애 주기에 적합한 "
        "금융 설계를 지원합니다."
    ),
    "benefit_match": (
        "{category_reason} 고객님의 금융 이용 행태에 가장 "
        "높은 가치를 제공할 것으로 예측되었습니다."
    ),
    "portfolio": (
        "{category_reason} 고객님의 금융 포트폴리오 다양성을 "
        "높이는 상품으로 추천합니다."
    ),
    "risk": (
        "{category_reason} 고객님의 리스크 프로파일에 적합한 "
        "안전한 금융 상품으로 분석되었습니다."
    ),
}

# ---------------------------------------------------------------------------
# Feature-name prefix -> category mapping
# ---------------------------------------------------------------------------

_PREFIX_TO_CATEGORY: Dict[str, str] = {
    "spend_": "spending_pattern",
    "amount_": "spending_pattern",
    "avg_txn_": "spending_pattern",
    "total_": "spending_pattern",
    "txn_count_": "frequency_pattern",
    "freq_": "frequency_pattern",
    "count_": "frequency_pattern",
    "mcc_": "frequency_pattern",
    "merchant_": "frequency_pattern",
    "age_": "life_stage",
    "life_stage_": "life_stage",
    "tenure_": "life_stage",
    "income_": "life_stage",
    "benefit_": "benefit_match",
    "reward_": "benefit_match",
    "cashback_": "benefit_match",
    "product_": "portfolio",
    "portfolio_": "portfolio",
    "card_": "portfolio",
    "churn_": "risk",
    "risk_": "risk",
    "fatigue_": "risk",
    "ctx_": "spending_pattern",
}


class TemplateReasonEngine:
    """Generate recommendation reasons from IG feature attributions.

    Simplified ablation version that maps feature names to 6 template
    categories and produces deterministic Korean-language reasons.
    """

    CATEGORIES = CATEGORIES

    def __init__(self, feature_names: Optional[List[str]] = None) -> None:
        """
        Args:
            feature_names: Ordered list of feature names matching model input
                           dimension.  If ``None``, feature indices are used
                           as-is (``feature_0``, ``feature_1``, ...).
        """
        self.feature_names: List[str] = feature_names or []
        self._feature_category_map: Dict[str, str] = {}
        if self.feature_names:
            self._build_feature_category_map()

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _build_feature_category_map(self) -> None:
        """Map each feature name to a template category via prefix matching."""
        for feat in self.feature_names:
            cat = self._classify_feature(feat)
            self._feature_category_map[feat] = cat
        logger.debug(
            "Feature-category map built: %d features across %d categories",
            len(self._feature_category_map),
            len(set(self._feature_category_map.values())),
        )

    @staticmethod
    def _classify_feature(feature_name: str) -> str:
        """Classify a single feature name into a template category."""
        lower = feature_name.lower()
        for prefix, category in _PREFIX_TO_CATEGORY.items():
            if lower.startswith(prefix):
                return category
        # Fallback: hash-based deterministic assignment
        h = int(hashlib.md5(feature_name.encode()).hexdigest(), 16)
        cats = list(CATEGORIES.keys())
        return cats[h % len(cats)]

    def _get_feature_name(self, idx: int) -> str:
        """Get feature name by index, falling back to generic name."""
        if idx < len(self.feature_names):
            return self.feature_names[idx]
        return f"feature_{idx}"

    def _get_category(self, feature_name: str) -> str:
        """Get category for a feature, building map entry on-the-fly if needed."""
        if feature_name not in self._feature_category_map:
            self._feature_category_map[feature_name] = self._classify_feature(
                feature_name,
            )
        return self._feature_category_map[feature_name]

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def generate(
        self,
        ig_top_features: List[int],
        task_name: str = "",
    ) -> str:
        """Generate a reason string from top IG feature indices.

        Args:
            ig_top_features: Top-K feature indices (e.g. from IG attribution).
            task_name: Task identifier for context (optional).

        Returns:
            Korean-language reason text string.
        """
        if not ig_top_features:
            return "고객님의 금융 데이터를 종합적으로 분석하여 추천드립니다."

        top3 = ig_top_features[:3]
        feat_names = [self._get_feature_name(idx) for idx in top3]
        categories = [self._get_category(fn) for fn in feat_names]

        # Primary reason from most important feature's category
        primary_cat = categories[0]
        reason_prefix = CATEGORIES.get(primary_cat, "종합 분석에 기반하여")
        reason_body = CATEGORY_TEMPLATES.get(primary_cat, "{category_reason} 추천드립니다.")
        primary_reason = reason_body.format(category_reason=reason_prefix)

        # Auxiliary categories (deduplicated)
        aux_cats = []
        seen = {primary_cat}
        for cat in categories[1:]:
            if cat not in seen:
                aux_cats.append(cat)
                seen.add(cat)

        if aux_cats:
            aux_parts = [CATEGORIES[c] for c in aux_cats if c in CATEGORIES]
            if aux_parts:
                aux_text = " 또한 " + ", ".join(aux_parts) + " 추천의 신뢰도가 높습니다."
                primary_reason += aux_text

        return primary_reason

    def generate_from_names(
        self,
        ig_top_feature_names: List[str],
        task_name: str = "",
    ) -> str:
        """Generate a reason string from top IG feature names (not indices).

        Args:
            ig_top_feature_names: Top-K feature names.
            task_name: Task identifier for context.

        Returns:
            Korean-language reason text string.
        """
        if not ig_top_feature_names:
            return "고객님의 금융 데이터를 종합적으로 분석하여 추천드립니다."

        top3 = ig_top_feature_names[:3]
        categories = [self._get_category(fn) for fn in top3]

        primary_cat = categories[0]
        reason_prefix = CATEGORIES.get(primary_cat, "종합 분석에 기반하여")
        reason_body = CATEGORY_TEMPLATES.get(primary_cat, "{category_reason} 추천드립니다.")
        primary_reason = reason_body.format(category_reason=reason_prefix)

        aux_cats = []
        seen = {primary_cat}
        for cat in categories[1:]:
            if cat not in seen:
                aux_cats.append(cat)
                seen.add(cat)

        if aux_cats:
            aux_parts = [CATEGORIES[c] for c in aux_cats if c in CATEGORIES]
            if aux_parts:
                aux_text = " 또한 " + ", ".join(aux_parts) + " 추천의 신뢰도가 높습니다."
                primary_reason += aux_text

        return primary_reason

    def batch_generate(
        self,
        ig_attributions: np.ndarray,
        task_name: str = "",
        top_k: int = 3,
    ) -> List[str]:
        """Generate reasons for a batch of samples.

        Args:
            ig_attributions: 2-D array of shape ``(N, D)`` with attribution
                             scores per feature.  Top-K indices are extracted
                             automatically.
            task_name: Task identifier for context.
            top_k: Number of top features to use per sample.

        Returns:
            List of reason strings, one per sample.
        """
        if ig_attributions.ndim == 1:
            ig_attributions = ig_attributions.reshape(1, -1)

        reasons: List[str] = []
        for i in range(ig_attributions.shape[0]):
            row = ig_attributions[i]
            top_indices = np.argsort(np.abs(row))[::-1][:top_k].tolist()
            reasons.append(self.generate(top_indices, task_name=task_name))

        logger.info(
            "Batch generated %d reasons for task '%s'",
            len(reasons), task_name,
        )
        return reasons

    def batch_generate_from_names(
        self,
        ig_top_features_list: List[List[str]],
        task_name: str = "",
    ) -> List[str]:
        """Generate reasons for a batch given pre-extracted feature names.

        Args:
            ig_top_features_list: List of per-sample top feature name lists.
            task_name: Task identifier.

        Returns:
            List of reason strings.
        """
        return [
            self.generate_from_names(feats, task_name=task_name)
            for feats in ig_top_features_list
        ]

    def get_category_distribution(
        self,
        ig_attributions: np.ndarray,
        top_k: int = 3,
    ) -> Dict[str, int]:
        """Count how often each category appears as primary reason.

        Useful for ablation reports to show which explanation categories
        dominate.

        Args:
            ig_attributions: 2-D array ``(N, D)``.
            top_k: Number of top features per sample.

        Returns:
            Dict mapping category name to count.
        """
        if ig_attributions.ndim == 1:
            ig_attributions = ig_attributions.reshape(1, -1)

        counts: Dict[str, int] = {cat: 0 for cat in CATEGORIES}
        for i in range(ig_attributions.shape[0]):
            row = ig_attributions[i]
            top_idx = int(np.argmax(np.abs(row)))
            feat_name = self._get_feature_name(top_idx)
            cat = self._get_category(feat_name)
            counts[cat] = counts.get(cat, 0) + 1

        return counts
