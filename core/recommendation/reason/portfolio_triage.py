"""
PortfolioTriageAgent — Context Richness-Based L2a Priority (v3.0.0).

Regulatory basis
----------------
- **금소법 §19 (금융소비자 보호에 관한 법률 제19조)**: 동등 설명의무.
  금융상품 추천 사유 생성 시 고객 세그먼트(VIP 여부 등)에 따라 처리 순서나
  품질을 차별하는 것은 금소법 §19의 동등 설명의무를 위반한다.
- **개보법 §37의2(2) (개인정보 보호법 제37조의2 제2항)**: 자동화 의사결정에
  대한 설명 요구권. 모든 고객은 AI 기반 추천에 대해 동등하게 설명을 요청할
  수 있어야 한다.

Design philosophy
-----------------
**"모든 세그먼트는 동일한 L1 파이프라인을 받는다. 풍부도(richness)는 L2a
처리 순서에만 영향을 미치며, 고객이 추천 사유를 받는지 여부를 결정하지 않는다."**

Prior (deprecated) approach
----------------------------
The previous AWS implementation used ``_Priority.HIGH if segment == "VIP"
else _Priority.MODERATE``.  This was a direct violation of the equal-explanation
obligation and has been replaced by this module.

Context richness
----------------
Richness is determined entirely by the *quantity and quality of available data*,
not by customer value tier.  A sparse-context customer is not penalised; they
receive the same L1 template as everyone else and may still receive L2a
processing if enough context becomes available in the future.

Three levels
~~~~~~~~~~~~
- **rich**: ≥2 products held *and* ≥1 recent consultation on record.
  Sufficient context for a high-quality LLM rewrite.
- **moderate**: Transaction within the past 180 days.  Some behavioural
  signal is available.
- **sparse**: Default fallback (cold-start / anonymous).  Excluded from L2a
  rewrite (no context for the LLM to work with), but oversampled in L2b for
  quality monitoring.

Configuration
-------------
All thresholds are config-driven.  Pass a ``rules`` dict that mirrors
``CONTEXT_RICHNESS_RULES`` to override defaults — no code changes required
for new datasets.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

__all__ = ["PortfolioTriageAgent", "CONTEXT_RICHNESS_RULES"]


# ---------------------------------------------------------------------------
# Default rules — can be overridden via constructor ``rules`` parameter
# ---------------------------------------------------------------------------

CONTEXT_RICHNESS_RULES: Dict[str, Any] = {
    "rich": {
        "description": "Feature-rich with recent consultation history",
        "conditions": {
            "total_products_min": 2,
            "recent_consultation_count_min": 1,
        },
        "l2a_rewrite_priority": 1,   # L2a highest-priority rewrite target
        "l2b_sample_rate": 0.002,    # 0.2% sample after L2a applied
    },
    "moderate": {
        "description": "Moderate feature availability",
        "conditions": {
            "last_transaction_days_max": 180,
        },
        "l2a_rewrite_priority": 2,   # L2a second-priority rewrite target
        "l2b_sample_rate": 0.003,    # 0.3% sample
    },
    "sparse": {
        "description": "Sparse features (cold-start or anonymous)",
        "conditions": {},            # default fallback — always matches
        "l2a_rewrite_priority": None,  # excluded from L2a (insufficient context)
        "l2b_sample_rate": 0.008,    # 0.8% oversample for quality monitoring
    },
}


# ---------------------------------------------------------------------------
# Agent
# ---------------------------------------------------------------------------

class PortfolioTriageAgent:
    """Classify customers by *context richness* to determine L2a priority.

    Rule-based, zero LLM calls.  Replaces the deprecated VIP segment-based
    priority (``_Priority.HIGH if segment == "VIP"``).

    Regulatory compliance
    ~~~~~~~~~~~~~~~~~~~~~
    Korean 금소법 §19 (equal-explanation obligation) and 개보법 §37의2(2)
    (right to automated-decision explanation) require that *all* customers
    receive the same L1 template pipeline.  This class controls only the
    *order* in which L2a LLM rewrites are processed — not whether a customer
    gets a reason at all.

    Args:
        rules: Optional dict overriding ``CONTEXT_RICHNESS_RULES``.  Must
               contain ``"rich"``, ``"moderate"``, and ``"sparse"`` keys with
               ``conditions``, ``l2a_rewrite_priority``, and
               ``l2b_sample_rate`` sub-keys.  If ``None``, the module-level
               defaults are used.

    Example::

        agent = PortfolioTriageAgent()
        richness = agent.classify_richness({
            "total_products": 3,
            "recent_consultation_count": 2,
            "last_transaction_days": 30,
        })
        # richness == "rich"

        priority = agent.get_l2a_rewrite_priority(richness)
        # priority == 1
    """

    def __init__(self, rules: Optional[Dict[str, Any]] = None) -> None:
        self._rules: Dict[str, Any] = rules if rules is not None else CONTEXT_RICHNESS_RULES
        logger.info(
            "PortfolioTriageAgent initialised: %d richness levels defined "
            "(regulatory: 금소법 §19 equal-explanation, 개보법 §37의2(2))",
            len(self._rules),
        )

    # ------------------------------------------------------------------
    # Per-customer classification
    # ------------------------------------------------------------------

    def classify_richness(self, features: Dict[str, Any]) -> str:
        """Classify a single customer's context richness.

        Evaluation order: ``rich`` → ``moderate`` → ``sparse``.
        The first level whose conditions are *all* satisfied is returned.
        ``sparse`` has no conditions and always matches (safe fallback).

        Args:
            features: Dict of customer feature values.  Missing keys are
                      treated as 0 / 999 (safe defaults for numeric checks).

        Returns:
            One of ``"rich"``, ``"moderate"``, or ``"sparse"``.
        """
        rich_cond = self._rules.get("rich", {}).get("conditions", {})
        moderate_cond = self._rules.get("moderate", {}).get("conditions", {})

        # --- rich ---
        tp_min = rich_cond.get("total_products_min", 2)
        cc_min = rich_cond.get("recent_consultation_count_min", 1)
        total_products = features.get("total_products", 0) or 0
        consultation_count = features.get("recent_consultation_count", 0) or 0
        if total_products >= tp_min and consultation_count >= cc_min:
            return "rich"

        # --- moderate ---
        txn_max = moderate_cond.get("last_transaction_days_max", 180)
        last_txn_days = features.get("last_transaction_days", 999)
        if last_txn_days is None:
            last_txn_days = 999
        if last_txn_days <= txn_max:
            return "moderate"

        # --- sparse (default fallback) ---
        return "sparse"

    # ------------------------------------------------------------------
    # Batch classification
    # ------------------------------------------------------------------

    def classify_batch(self, features_df) -> Dict[str, str]:
        """Classify a DataFrame of customers by context richness.

        Iterates row-by-row using :meth:`classify_richness`.  For large
        batches prefer the DuckDB SQL path in the on-prem version; this
        implementation keeps AWS dependencies minimal.

        Args:
            features_df: A pandas-compatible DataFrame with a
                         ``customer_id`` column and feature columns
                         expected by :meth:`classify_richness`.

        Returns:
            Dict mapping ``customer_id`` → richness string.
        """
        result: Dict[str, str] = {}
        for _, row in features_df.iterrows():
            cid = row.get("customer_id") or row.name
            richness = self.classify_richness(row.to_dict())
            result[cid] = richness

        from collections import Counter
        dist = Counter(result.values())
        total = len(result)
        logger.info(
            "classify_batch complete: %d customers — "
            "rich=%d, moderate=%d, sparse=%d",
            total,
            dist.get("rich", 0),
            dist.get("moderate", 0),
            dist.get("sparse", 0),
        )
        return result

    # ------------------------------------------------------------------
    # Priority / rate helpers
    # ------------------------------------------------------------------

    def get_l2a_rewrite_priority(self, richness: str) -> Optional[int]:
        """Return L2a rewrite queue priority for a richness level.

        Lower integer = higher priority (SQS MessageAttribute convention).

        Args:
            richness: One of ``"rich"``, ``"moderate"``, ``"sparse"``.

        Returns:
            Integer priority, or ``None`` if this richness level should
            be *excluded* from L2a processing entirely (sparse context —
            insufficient data for meaningful LLM rewrite).
        """
        level = self._rules.get(richness, self._rules.get("sparse", {}))
        return level.get("l2a_rewrite_priority")

    def get_l2b_sample_rate(self, richness: str) -> float:
        """Return L2b quality-validation sample rate for a richness level.

        Args:
            richness: One of ``"rich"``, ``"moderate"``, ``"sparse"``.

        Returns:
            Float in [0, 1] representing the fraction of L1-only customers
            to include in L2b quality monitoring.  Sparse customers are
            oversampled (0.8 %) because their L1 template quality is less
            predictable.
        """
        level = self._rules.get(richness, self._rules.get("sparse", {}))
        return float(level.get("l2b_sample_rate", 0.008))
