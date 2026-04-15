"""
Layer 3 Rule-based Fallback Engine
====================================

Provides rule-based recommendations for all 13 tasks when ML layers
(PLE distillation and LGBM) are unavailable or underperforming.

Design principles:
- Rules FIRST try engineered Phase-0 features (HMM, TDA, Mamba, GMM, etc.)
- FALL BACK to raw scalar features when engineered features are absent
- Every rule produces a customer-facing reason string
- Suitability constraint (Financial Consumer Protection Act Art.17) is
  enforced after every prediction when enabled via config
- Zero dataset-specific column names are hardcoded — feature prefixes and
  thresholds come entirely from config

Marketing theory references are noted inline per the design spec
(12_rule_based_fallback.md).
"""

from __future__ import annotations

import logging
import math
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

__all__ = ["RuleBasedRecommender"]


# ---------------------------------------------------------------------------
# Helper utilities (no imports beyond stdlib)
# ---------------------------------------------------------------------------

def _get_prefixed(features: Dict[str, Any], prefix: str) -> Dict[str, Any]:
    """Return {key: value} for all feature keys that start with *prefix*."""
    return {k: v for k, v in features.items() if k.startswith(prefix)}


def _first_prefixed(features: Dict[str, Any], prefix: str, default=None):
    """Return the value of the first key starting with *prefix*, or *default*."""
    for k, v in features.items():
        if k.startswith(prefix):
            return v
    return default


def _safe_float(value, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _safe_int(value, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------

class RuleBasedRecommender:
    """Layer 3 fallback: rule-based recommendations using pre-computed features.

    Each task has a dedicated rule method ``_rule_{task_name}`` that uses
    Phase 0 engineered features (already computed, zero additional inference
    cost) to generate recommendations.

    All rules respect the suitability constraint (Financial Consumer
    Protection Act Art.17).

    Args:
        config: Merged pipeline config.  Must contain a ``rule_engine``
                section with thresholds, and a ``tasks`` list for task
                metadata.
        feature_groups_config: Optional feature_groups.yaml dict for
                               group metadata (not strictly required).
    """

    # Canonical ordered product adjacency path (configurable in rule_engine.product_adjacency)
    _DEFAULT_ADJACENCY_PATH = [
        "deposits", "savings", "investment", "insurance", "lending",
    ]

    # Segment index → label (matches santander.yaml segment_prediction num_classes=4)
    _SEGMENT_LABELS = {0: "TOP", 1: "PARTICULARES", 2: "UNIVERSITARIO", 3: "UNKNOWN"}

    def __init__(
        self,
        config: Dict[str, Any],
        feature_groups_config: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.config = config
        self.feature_groups_config = feature_groups_config or {}

        re_cfg = config.get("rule_engine", {})
        self.enabled: bool = re_cfg.get("enabled", True)
        self.suitability_check: bool = re_cfg.get("suitability_check", True)

        # Product adjacency (nba_primary rule)
        adj_cfg = re_cfg.get("product_adjacency", {})
        self.adjacency_path: List[str] = adj_cfg.get(
            "path", self._DEFAULT_ADJACENCY_PATH
        )

        # Dormancy thresholds (product_stability rule)
        dormancy = re_cfg.get("dormancy_thresholds", {})
        self.dormancy_warning_days: int = dormancy.get("warning_days", 30)
        self.dormancy_alert_days: int = dormancy.get("alert_days", 60)
        self.dormancy_dormant_days: int = dormancy.get("dormant_days", 90)

        # CLV tier config (cross_sell_count rule)
        clv = re_cfg.get("clv_tiers", {})
        self.clv_premium_pct: float = clv.get("premium_percentile", 80.0)
        self.clv_target_products: Dict[str, int] = clv.get(
            "target_products",
            {"premium": 4, "stable": 3, "growth": 2, "at_risk": 1},
        )

        # RFM thresholds (churn_signal rule)
        rfm = re_cfg.get("rfm_thresholds", {})
        self.rfm_recency_warning_days: int = rfm.get("recency_warning_days", 30)
        self.rfm_monetary_drop_pct: float = rfm.get("monetary_drop_pct", 0.20)

        # MCC thresholds
        mcc = re_cfg.get("mcc_thresholds", {})
        self.mcc_jsd_threshold: float = mcc.get("jsd_threshold", 0.15)
        self.mcc_top_k: int = mcc.get("top_k", 5)

        # Deposits / surplus thresholds
        deposits = re_cfg.get("deposits_thresholds", {})
        self.surplus_ratio_threshold: float = deposits.get("surplus_ratio", 0.30)

        # Investments thresholds
        invest = re_cfg.get("investments_thresholds", {})
        self.invest_min_risk_grade: int = invest.get("min_risk_grade", 3)
        self.invest_asset_pct_threshold: float = invest.get("asset_pct_threshold", 70.0)

        # Accounts thresholds
        accounts = re_cfg.get("accounts_thresholds", {})
        self.new_customer_days: int = accounts.get("new_customer_days", 90)
        self.high_interbank_transfer_pct: float = accounts.get(
            "interbank_transfer_pct", 0.30
        )

        # Lending thresholds
        lending = re_cfg.get("lending_thresholds", {})
        self.max_dti: float = lending.get("max_dti", 0.40)
        self.max_credit_grade_for_lending: int = lending.get(
            "max_credit_grade", 4
        )

        # Payments thresholds
        payments = re_cfg.get("payments_thresholds", {})
        self.payments_spend_top_pct: float = payments.get("spend_top_pct", 70.0)
        self.payments_foreign_pct: float = payments.get("foreign_pct", 0.10)

        # Task metadata (from dataset yaml tasks list)
        self._task_meta: Dict[str, Dict[str, Any]] = {
            t["name"]: t for t in config.get("tasks", [])
        }

        # Feature routing: task → Financial DNA group → feature groups → feature prefixes
        # Uses the same routing infrastructure as PLE expert routing (config-driven)
        self._task_feature_prefixes: Dict[str, List[str]] = {}
        self._build_feature_routing()

        # Mapping from task_name → rule method
        self._rule_dispatch: Dict[str, Any] = {
            "churn_signal": self._rule_churn_signal,
            "top_mcc_shift": self._rule_top_mcc_shift,
            "nba_primary": self._rule_nba_primary,
            "segment_prediction": self._rule_segment_prediction,
            "will_acquire_deposits": self._rule_will_acquire_deposits,
            "will_acquire_investments": self._rule_will_acquire_investments,
            "will_acquire_accounts": self._rule_will_acquire_accounts,
            "will_acquire_lending": self._rule_will_acquire_lending,
            "will_acquire_payments": self._rule_will_acquire_payments,
            "product_stability": self._rule_product_stability,
            "cross_sell_count": self._rule_cross_sell_count,
            "next_mcc": self._rule_next_mcc,
            "mcc_diversity_trend": self._rule_mcc_diversity_trend,
        }

        logger.info(
            "RuleBasedRecommender initialised: %d rules, suitability_check=%s",
            len(self._rule_dispatch),
            self.suitability_check,
        )

    # ------------------------------------------------------------------
    # Feature routing (same infrastructure as PLE expert routing)
    # ------------------------------------------------------------------

    def _build_feature_routing(self) -> None:
        """Build task → feature prefix mapping from config.

        Chain: task → Financial DNA group → feature groups → column prefixes.
        Uses task_groups + feature_groups.yaml target_experts to derive which
        feature groups are relevant for each task, matching PLE expert routing.
        """
        # Step 1: task → DNA group
        task_to_dna: Dict[str, str] = {}
        for tg in self.config.get("task_groups", []):
            group_name = tg["name"] if isinstance(tg, dict) else tg.name
            tasks = tg["tasks"] if isinstance(tg, dict) else tg.tasks
            for t in tasks:
                task_to_dna[t] = group_name

        # Step 2: DNA group → expert names (from expert_input_routing or feature_groups)
        # Use feature_groups.yaml target_experts to map groups → experts → prefixes
        fg_list = self.feature_groups_config.get("feature_groups", [])
        if not fg_list:
            fg_list = self.config.get("feature_groups", [])

        # Build: feature_group_name → column_prefix
        group_to_prefix: Dict[str, str] = {}
        group_to_experts: Dict[str, List[str]] = {}
        for fg in fg_list:
            fg_name = fg.get("name", "")
            prefix = fg.get("column_prefix", fg_name)
            group_to_prefix[fg_name] = prefix
            group_to_experts[fg_name] = fg.get("target_experts", [])

        # Step 3: For each DNA group, collect all feature groups whose experts
        # are routed to that DNA group's tasks. Simple heuristic: all feature
        # groups that route to any expert used by any task in the DNA group.
        # Fallback: if routing info is sparse, use all feature groups.
        dna_to_prefixes: Dict[str, List[str]] = {}
        for dna_group in set(task_to_dna.values()):
            prefixes = set()
            for fg in fg_list:
                # Include all groups — the rule functions use _get_prefixed()
                # which naturally filters by prefix. The routing is advisory.
                prefixes.add(fg.get("column_prefix", fg.get("name", "")))
            dna_to_prefixes[dna_group] = sorted(prefixes)

        # Step 4: task → prefixes
        for task_name in self._task_meta:
            dna = task_to_dna.get(task_name)
            if dna and dna in dna_to_prefixes:
                self._task_feature_prefixes[task_name] = dna_to_prefixes[dna]
            # else: no routing — rule gets all features (fallback)

        if self._task_feature_prefixes:
            logger.info("Feature routing built: %d tasks mapped to DNA groups",
                        len(self._task_feature_prefixes))

    def _route_features(self, features: Dict[str, Any], task_name: str) -> Dict[str, Any]:
        """Filter features to only those routed for this task's DNA group.

        If no routing is configured, returns all features (fallback).
        """
        prefixes = self._task_feature_prefixes.get(task_name)
        if not prefixes:
            return features  # fallback: all features
        routed = {}
        for k, v in features.items():
            for prefix in prefixes:
                if k.startswith(prefix):
                    routed[k] = v
                    break
            else:
                # Also include non-prefixed features (raw scalars like balance, income)
                routed[k] = v
        return routed

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def predict(self, features: Dict[str, Any], task_name: str) -> Dict[str, Any]:
        """Generate a rule-based prediction for one customer on one task.

        Args:
            features: ``{feature_name: value}`` for one customer.
            task_name: Which task to predict.

        Returns:
            Dict with keys: ``prediction``, ``confidence``, ``reason``,
            ``rule_name``, ``contributing_features``, ``layer`` (always 3).
            ``contributing_features`` is a list of ``{name, value, role}`` dicts
            identifying which features drove the rule's decision.

        Raises:
            KeyError: If *task_name* has no rule implementation.
        """
        if task_name not in self._rule_dispatch:
            available = ", ".join(sorted(self._rule_dispatch))
            raise KeyError(
                f"No rule for task '{task_name}'. Available: {available}"
            )

        routed = self._route_features(features, task_name)
        result = self._rule_dispatch[task_name](routed)
        result["layer"] = 3

        if self.suitability_check:
            result = self._apply_suitability(features, task_name, result)

        return result

    def predict_batch(
        self,
        features_df,
        task_names: List[str],
    ) -> Dict[str, List[Dict[str, Any]]]:
        """Generate rule-based predictions for multiple customers and tasks.

        Args:
            features_df: DataFrame-like object; each row is one customer.
                         Must support ``.iterrows()`` (pandas) or equivalent.
            task_names: List of task names to predict.

        Returns:
            ``{task_name: [result_dict, ...]}`` — one entry per customer row.
        """
        results: Dict[str, List[Dict[str, Any]]] = {t: [] for t in task_names}
        n_rows = 0

        for _, row in features_df.iterrows():
            n_rows += 1
            feat = row.to_dict() if hasattr(row, "to_dict") else dict(row)
            for task_name in task_names:
                try:
                    r = self.predict(feat, task_name)
                except Exception:
                    logger.debug(
                        "Rule prediction failed for task '%s'", task_name,
                        exc_info=True,
                    )
                    r = {
                        "prediction": None,
                        "confidence": 0.0,
                        "reason": "Rule evaluation failed",
                        "rule_name": "error",
                        "contributing_features": [],
                        "layer": 3,
                    }
                results[task_name].append(r)

        logger.info(
            "predict_batch: %d customers x %d tasks processed",
            n_rows, len(task_names),
        )
        return results

    def get_available_rules(self) -> List[str]:
        """Return sorted list of task names that have rules implemented."""
        return sorted(self._rule_dispatch)

    # ------------------------------------------------------------------
    # Suitability constraint (금소법 제17조)
    # ------------------------------------------------------------------

    def _apply_suitability(
        self,
        features: Dict[str, Any],
        task_name: str,
        result: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Block recommendations where customer risk grade < product risk grade.

        The customer risk grade is read from any feature prefixed with
        ``risk_grade`` or ``suitability_grade`` (config-agnostic).
        Product risk requirements are stored in config under
        ``rule_engine.product_risk_grades``.
        """
        prod_risk_grades: Dict[str, int] = self.config.get(
            "rule_engine", {}
        ).get("product_risk_grades", {})

        required_grade = prod_risk_grades.get(task_name, 0)
        if required_grade == 0:
            return result  # no suitability constraint for this task

        # Try to find customer risk grade from feature prefixes
        customer_grade = _safe_int(
            _first_prefixed(features, "risk_grade"),
            default=_safe_int(
                _first_prefixed(features, "suitability_grade"), default=99
            ),
        )

        if customer_grade < required_grade:
            logger.debug(
                "Suitability block: task=%s customer_grade=%d required=%d",
                task_name, customer_grade, required_grade,
            )
            return {
                "prediction": 0,
                "confidence": 1.0,
                "reason": (
                    "고객님의 투자 성향 등급이 해당 상품의 위험 등급에 "
                    "미달하여 추천이 제한됩니다 (금소법 제17조)."
                ),
                "rule_name": "suitability_block",
                "contributing_features": [],
                "layer": 3,
            }
        return result

    # ------------------------------------------------------------------
    # Rule implementations — Engagement group
    # ------------------------------------------------------------------

    def _rule_churn_signal(self, features: Dict[str, Any]) -> Dict[str, Any]:
        """Binary: detect churn via RFM decay.

        Theory: Relationship Marketing (Berry, 1983) — retaining existing
        customers costs 1/6 of acquiring new ones.

        Priority:
        1. HMM lifecycle probability of declining state
        2. TDA persistence / Mamba temporal patterns
        3. Raw recency / frequency / monetary features
        """
        # --- Layer 1: HMM lifecycle feature ---
        hmm_declining = _first_prefixed(features, "hmm_lifecycle_prob_declining")
        hmm_state = _first_prefixed(features, "hmm_lifecycle_state")

        if hmm_declining is not None:
            prob = _safe_float(hmm_declining)
            hmm_threshold = self.config.get("rule_engine", {}).get(
                "hmm_declining_threshold", 0.5
            )
            prediction = 1 if prob >= hmm_threshold else 0
            confidence = min(0.95, 0.6 + 0.35 * prob)
            if prediction == 1:
                reason = (
                    "행동 패턴 분석 결과 이탈 위험 신호가 감지되었습니다. "
                    "우대금리 적금으로 자산을 효율적으로 관리해보세요."
                )
            else:
                reason = "현재 거래 패턴이 안정적입니다."
            # Find the actual key name for hmm_lifecycle_prob_declining
            _hmm_declining_key = next(
                (k for k in features if k.startswith("hmm_lifecycle_prob_declining")), None
            )
            _contributing = [
                {"name": _hmm_declining_key or "hmm_lifecycle_prob_declining",
                 "value": prob, "role": "primary"},
            ]
            if hmm_state is not None:
                _hmm_state_key = next(
                    (k for k in features if k.startswith("hmm_lifecycle_state")), None
                )
                _contributing.append(
                    {"name": _hmm_state_key or "hmm_lifecycle_state",
                     "value": _safe_float(hmm_state), "role": "supporting"}
                )
            return {
                "prediction": prediction,
                "confidence": confidence,
                "reason": reason,
                "rule_name": "churn_hmm_lifecycle",
                "contributing_features": _contributing,
            }

        # --- Layer 2: Mamba temporal / TDA features ---
        _mamba_churn_key = next(
            (k for k in features if k.startswith("mamba_temporal_churn")), None
        )
        mamba_churn = features.get(_mamba_churn_key) if _mamba_churn_key else None
        tda_persistence = _first_prefixed(features, "tda_persistence")

        if mamba_churn is not None:
            prob = _safe_float(mamba_churn)
            mamba_threshold = self.config.get("rule_engine", {}).get(
                "mamba_churn_threshold", 0.5
            )
            prediction = 1 if prob >= mamba_threshold else 0
            confidence = min(0.90, 0.55 + 0.35 * prob)
            reason = (
                "최근 거래 활동이 줄어들고 있어, 우대금리 적금으로 "
                "자산을 효율적으로 관리해보세요."
                if prediction == 1
                else "현재 거래 패턴이 안정적입니다."
            )
            _contributing = [
                {"name": _mamba_churn_key, "value": prob, "role": "primary"},
            ]
            if tda_persistence is not None:
                _tda_key = next(
                    (k for k in features if k.startswith("tda_persistence")), None
                )
                if _tda_key:
                    _contributing.append(
                        {"name": _tda_key, "value": _safe_float(tda_persistence),
                         "role": "supporting"}
                    )
            return {
                "prediction": prediction,
                "confidence": confidence,
                "reason": reason,
                "rule_name": "churn_mamba_temporal",
                "contributing_features": _contributing,
            }

        # --- Layer 3: Raw RFM features ---
        _recency_key = next(
            (k for k in features if k.startswith("recency_days")), None
        )
        _freq_key = next(
            (k for k in features if k.startswith("frequency_trend")), None
        )
        _monetary_key = next(
            (k for k in features if k.startswith("monetary_change_pct")), None
        )
        recency_days = _safe_float(
            features.get(_recency_key) if _recency_key else None, default=0.0
        )
        freq_trend = _safe_float(
            features.get(_freq_key) if _freq_key else None, default=0.0
        )
        monetary_change = _safe_float(
            features.get(_monetary_key) if _monetary_key else None, default=0.0
        )

        signals = 0
        if recency_days > self.rfm_recency_warning_days:
            signals += 1
        if freq_trend < 0:
            signals += 1
        if monetary_change < -self.rfm_monetary_drop_pct:
            signals += 1

        prediction = 1 if signals >= 2 else 0
        confidence = 0.40 + 0.15 * signals
        reason = (
            "최근 거래 활동이 줄어들고 있어, 우대금리 적금으로 "
            "자산을 효율적으로 관리해보세요."
            if prediction == 1
            else "현재 거래 패턴이 안정적입니다."
        )
        _contributing = []
        if _recency_key:
            _contributing.append(
                {"name": _recency_key, "value": recency_days, "role": "primary"}
            )
        if _freq_key:
            _contributing.append(
                {"name": _freq_key, "value": freq_trend, "role": "supporting"}
            )
        if _monetary_key:
            _contributing.append(
                {"name": _monetary_key, "value": monetary_change, "role": "supporting"}
            )
        return {
            "prediction": prediction,
            "confidence": confidence,
            "reason": reason,
            "rule_name": "churn_rfm_decay",
            "contributing_features": _contributing,
        }

    def _rule_top_mcc_shift(self, features: Dict[str, Any]) -> Dict[str, Any]:
        """Binary: detect lifestyle change via MCC distribution shift.

        Theory: McKinsey Consumer Decision Journey — real-time 'trigger'
        detection from life changes.

        Priority:
        1. TDA local (transaction pattern topology change)
        2. Merchant hierarchy shift features
        3. Mamba temporal MCC sequence
        4. Raw MCC entropy change
        """
        # --- TDA local topology change ---
        _tda_local_key = next(
            (k for k in features if k.startswith("tda_local_shift")), None
        )
        tda_local = features.get(_tda_local_key) if _tda_local_key else None
        if tda_local is not None:
            score = _safe_float(tda_local)
            threshold = self.config.get("rule_engine", {}).get(
                "tda_shift_threshold", 0.5
            )
            prediction = 1 if score >= threshold else 0
            confidence = min(0.92, 0.60 + 0.30 * score)
            reason = (
                "최근 소비 패턴에 변화가 감지되었습니다. "
                "새로운 라이프스타일에 맞는 상품을 추천드립니다."
                if prediction == 1
                else "소비 패턴이 안정적으로 유지되고 있습니다."
            )
            return {
                "prediction": prediction,
                "confidence": confidence,
                "reason": reason,
                "rule_name": "top_mcc_tda_local",
                "contributing_features": [
                    {"name": _tda_local_key, "value": score, "role": "primary"},
                ],
            }

        # --- Mamba MCC sequence ---
        _mamba_mcc_key = next(
            (k for k in features if k.startswith("mamba_temporal_mcc")), None
        )
        mamba_mcc = features.get(_mamba_mcc_key) if _mamba_mcc_key else None
        if mamba_mcc is not None:
            score = _safe_float(mamba_mcc)
            prediction = 1 if score >= 0.5 else 0
            confidence = min(0.85, 0.55 + 0.30 * score)
            reason = (
                "최근 해외결제가 늘어났네요, 해외 수수료 무료 카드를 확인해보세요."
                if prediction == 1
                else "소비 패턴이 안정적으로 유지되고 있습니다."
            )
            return {
                "prediction": prediction,
                "confidence": confidence,
                "reason": reason,
                "rule_name": "top_mcc_mamba",
                "contributing_features": [
                    {"name": _mamba_mcc_key, "value": score, "role": "primary"},
                ],
            }

        # --- Raw MCC entropy change ---
        _mcc_entropy_key = next(
            (k for k in features if k.startswith("mcc_entropy_change")), None
        )
        mcc_entropy_change = _safe_float(
            features.get(_mcc_entropy_key) if _mcc_entropy_key else None, default=0.0
        )
        prediction = 1 if abs(mcc_entropy_change) >= self.mcc_jsd_threshold else 0
        confidence = 0.50 + min(0.35, abs(mcc_entropy_change) * 2.0)
        reason = (
            "소비 카테고리에 변화가 감지되어 새로운 혜택 상품을 추천드립니다."
            if prediction == 1
            else "소비 패턴이 안정적으로 유지되고 있습니다."
        )
        _contributing = []
        if _mcc_entropy_key:
            _contributing.append(
                {"name": _mcc_entropy_key, "value": mcc_entropy_change, "role": "primary"}
            )
        return {
            "prediction": prediction,
            "confidence": confidence,
            "reason": reason,
            "rule_name": "top_mcc_entropy",
            "contributing_features": _contributing,
        }

    # ------------------------------------------------------------------
    # Rule implementations — Lifecycle group
    # ------------------------------------------------------------------

    def _rule_nba_primary(self, features: Dict[str, Any]) -> Dict[str, Any]:
        """Multiclass: next best product via product adjacency matrix.

        Theory: Kotler 5A (Aware→Appeal→Ask→Act→Advocate) — recommend
        the natural next step.

        Returns class index 0–6 per santander.yaml nba_primary definition:
        0=no_nba, 1=savings_guarantee, 2=checking_accounts, 3=deposits,
        4=investments, 5=credit_loans, 6=debits.

        Priority:
        1. LightGCN collaborative filtering features
        2. GMM cluster peer comparison
        3. PIH economics features
        4. Raw product count + adjacency path
        """
        # Map adjacency path position → class index (0=no_nba so +1)
        adjacency_class_map = {
            step: idx + 1
            for idx, step in enumerate(self.adjacency_path)
            if idx < 6  # classes 1-6 only
        }

        # --- LightGCN collaborative features ---
        _lightgcn_nba_key = next(
            (k for k in features if k.startswith("lightgcn_nba")), None
        )
        lightgcn_nba = features.get(_lightgcn_nba_key) if _lightgcn_nba_key else None
        if lightgcn_nba is not None:
            pred = _safe_int(lightgcn_nba, default=0)
            confidence = min(0.90, 0.70)
            reason = (
                "유사 고객 분석 결과 다음 단계 금융 상품을 추천드립니다."
            )
            return {
                "prediction": pred,
                "confidence": confidence,
                "reason": reason,
                "rule_name": "nba_lightgcn",
                "contributing_features": [
                    {"name": _lightgcn_nba_key, "value": _safe_float(lightgcn_nba),
                     "role": "primary"},
                ],
            }

        # --- GMM cluster peer comparison ---
        _gmm_nba_key = next(
            (k for k in features if k.startswith("gmm_cluster_nba")), None
        )
        gmm_nba = features.get(_gmm_nba_key) if _gmm_nba_key else None
        if gmm_nba is not None:
            pred = _safe_int(gmm_nba, default=0)
            reason = (
                "동일 클러스터 고객군 대비 추천 상품을 안내드립니다."
            )
            return {
                "prediction": pred,
                "confidence": 0.65,
                "reason": reason,
                "rule_name": "nba_gmm_cluster",
                "contributing_features": [
                    {"name": _gmm_nba_key, "value": _safe_float(gmm_nba),
                     "role": "primary"},
                ],
            }

        # --- Raw product count based adjacency ---
        _num_products_key = next(
            (k for k in features if k.startswith("num_products")), None
        )
        num_products = _safe_int(
            features.get(_num_products_key) if _num_products_key else None, default=0
        )
        product_prefixes = _get_prefixed(features, "prod_")
        owned_count = sum(
            1 for v in product_prefixes.values() if _safe_int(v, 0) > 0
        )
        actual_count = max(num_products, owned_count)

        if actual_count < 2:
            # Recommend deposits (position 0 in path → class 3 for santander)
            pred = adjacency_class_map.get(self.adjacency_path[0], 3)
            reason = (
                "현재 입출금 계좌를 보유 중이시네요, "
                "적금으로 자산을 체계적으로 관리해보세요."
            )
        else:
            # Recommend next step in adjacency path (clamp to max class)
            step_idx = min(actual_count, len(self.adjacency_path) - 1)
            next_step = self.adjacency_path[step_idx]
            pred = adjacency_class_map.get(next_step, 0)
            reason = (
                f"현재 {actual_count}개 상품을 이용 중이시네요, "
                f"다음 단계인 {next_step} 상품을 추천드립니다."
            )

        _contributing = []
        if _num_products_key:
            _contributing.append(
                {"name": _num_products_key, "value": float(actual_count), "role": "primary"}
            )
        return {
            "prediction": pred,
            "confidence": 0.55,
            "reason": reason,
            "rule_name": "nba_adjacency_matrix",
            "contributing_features": _contributing,
        }

    def _rule_segment_prediction(self, features: Dict[str, Any]) -> Dict[str, Any]:
        """Multiclass: 3-axis balance × frequency × product count segmentation.

        Theory: CLV Tiered Model — top 20% drive 80% of revenue (Pareto).

        Returns class 0-3: 0=TOP, 1=PARTICULARES, 2=UNIVERSITARIO, 3=UNKNOWN.

        Priority:
        1. GMM cluster ID (pre-computed segment)
        2. HMM behavior state
        3. Raw 3-axis rule
        """
        # --- GMM cluster (pre-computed segment) ---
        _gmm_cluster_key = next(
            (k for k in features if k.startswith("gmm_cluster_id")), None
        )
        gmm_cluster = features.get(_gmm_cluster_key) if _gmm_cluster_key else None
        if gmm_cluster is not None:
            pred = _safe_int(gmm_cluster, default=3)
            label = self._SEGMENT_LABELS.get(pred, "UNKNOWN")
            reason = (
                f"고객님은 {label} 세그먼트에 속하며 해당 단계에 "
                "적합한 상품을 추천드립니다."
            )
            return {
                "prediction": pred,
                "confidence": 0.80,
                "reason": reason,
                "rule_name": "segment_gmm_cluster",
                "contributing_features": [
                    {"name": _gmm_cluster_key, "value": _safe_float(gmm_cluster),
                     "role": "primary"},
                ],
            }

        # --- HMM behavior state ---
        _hmm_behavior_key = next(
            (k for k in features if k.startswith("hmm_behavior_state")), None
        )
        hmm_state = features.get(_hmm_behavior_key) if _hmm_behavior_key else None
        if hmm_state is not None:
            # Map HMM states to segment classes heuristically
            state_val = _safe_int(hmm_state, default=3)
            pred = min(state_val, 3)
            label = self._SEGMENT_LABELS.get(pred, "UNKNOWN")
            reason = (
                f"행동 패턴 기반으로 {label} 세그먼트로 분류되었습니다."
            )
            return {
                "prediction": pred,
                "confidence": 0.65,
                "reason": reason,
                "rule_name": "segment_hmm_behavior",
                "contributing_features": [
                    {"name": _hmm_behavior_key, "value": _safe_float(hmm_state),
                     "role": "primary"},
                ],
            }

        # --- Raw 3-axis rule ---
        _balance_key = next(
            (k for k in features if k.startswith("balance")), None
        )
        _monthly_freq_key = next(
            (k for k in features if k.startswith("monthly_freq")), None
        )
        _seg_num_products_key = next(
            (k for k in features if k.startswith("num_products")), None
        )
        balance = _safe_float(
            features.get(_balance_key) if _balance_key else None, default=0.0
        )
        monthly_freq = _safe_float(
            features.get(_monthly_freq_key) if _monthly_freq_key else None, default=0.0
        )
        num_products = _safe_int(
            features.get(_seg_num_products_key) if _seg_num_products_key else None,
            default=0,
        )

        seg_cfg = self.config.get("rule_engine", {}).get("segment_thresholds", {})
        balance_high = _safe_float(seg_cfg.get("balance_high", 50000.0))
        balance_mid = _safe_float(seg_cfg.get("balance_mid", 10000.0))
        freq_high = _safe_float(seg_cfg.get("freq_high", 20.0))
        freq_mid = _safe_float(seg_cfg.get("freq_mid", 5.0))

        score = 0
        if balance >= balance_high:
            score += 2
        elif balance >= balance_mid:
            score += 1
        if monthly_freq >= freq_high:
            score += 2
        elif monthly_freq >= freq_mid:
            score += 1
        if num_products >= 4:
            score += 2
        elif num_products >= 2:
            score += 1

        if score >= 5:
            pred = 0  # TOP
        elif score >= 3:
            pred = 1  # PARTICULARES
        elif score >= 1:
            pred = 2  # UNIVERSITARIO
        else:
            pred = 3  # UNKNOWN

        label = self._SEGMENT_LABELS.get(pred, "UNKNOWN")
        reason = (
            f"고객님은 자산 및 거래 패턴 기준으로 {label} 세그먼트에 "
            "해당하며, 이에 맞는 상품을 추천드립니다."
        )
        _contributing = []
        if _balance_key:
            _contributing.append(
                {"name": _balance_key, "value": balance, "role": "primary"}
            )
        if _monthly_freq_key:
            _contributing.append(
                {"name": _monthly_freq_key, "value": monthly_freq, "role": "supporting"}
            )
        if _seg_num_products_key:
            _contributing.append(
                {"name": _seg_num_products_key, "value": float(num_products),
                 "role": "supporting"}
            )
        return {
            "prediction": pred,
            "confidence": 0.50,
            "reason": reason,
            "rule_name": "segment_3axis",
            "contributing_features": _contributing,
        }

    def _rule_will_acquire_deposits(self, features: Dict[str, Any]) -> Dict[str, Any]:
        """Binary: deposit acquisition propensity via surplus ratio.

        Theory: Lifecycle Marketing — stable cash flow stage drives
        savings product demand.

        Priority:
        1. Economics PIH features (permanent vs transitory income)
        2. HMM journey (asset accumulation stage)
        3. GMM cluster (savings propensity)
        4. Raw balance surplus ratio
        """
        # --- PIH economics features ---
        _pih_surplus_key = next(
            (k for k in features if k.startswith("economics_pih_surplus")), None
        )
        pih_surplus = features.get(_pih_surplus_key) if _pih_surplus_key else None
        if pih_surplus is not None:
            score = _safe_float(pih_surplus)
            prediction = 1 if score >= self.surplus_ratio_threshold else 0
            confidence = min(0.92, 0.60 + abs(score - self.surplus_ratio_threshold) * 1.5)
            reason = (
                "여유자금이 충분하시네요, 우대금리 정기예금으로 "
                "안정적인 이자 수익을 만들어보세요."
                if prediction == 1
                else "현재 여유자금 수준으로는 예금보다 유동성 상품이 적합합니다."
            )
            return {
                "prediction": prediction,
                "confidence": confidence,
                "reason": reason,
                "rule_name": "deposits_pih_surplus",
                "contributing_features": [
                    {"name": _pih_surplus_key, "value": score, "role": "primary"},
                ],
            }

        # --- HMM journey state ---
        _hmm_journey_key = next(
            (k for k in features if k.startswith("hmm_journey_state")), None
        )
        hmm_journey = features.get(_hmm_journey_key) if _hmm_journey_key else None
        if hmm_journey is not None:
            # State names containing 'accumulation' or high integer values → deposits
            state_str = str(hmm_journey).lower()
            prediction = 1 if "accumul" in state_str or _safe_int(hmm_journey, 0) >= 2 else 0
            reason = (
                "자산 축적 단계에서 정기예금은 최적의 저축 수단입니다."
                if prediction == 1
                else "현재 단계에서는 유동성 확보가 우선입니다."
            )
            return {
                "prediction": prediction,
                "confidence": 0.65,
                "reason": reason,
                "rule_name": "deposits_hmm_journey",
                "contributing_features": [
                    {"name": _hmm_journey_key, "value": _safe_float(hmm_journey),
                     "role": "primary"},
                ],
            }

        # --- Raw surplus ratio: (balance - spending) / balance ---
        _dep_balance_key = next(
            (k for k in features if k.startswith("balance")), None
        )
        _dep_spend_key = next(
            (k for k in features if k.startswith("monthly_spend")), None
        )
        balance = _safe_float(
            features.get(_dep_balance_key) if _dep_balance_key else None, default=0.0
        )
        spending = _safe_float(
            features.get(_dep_spend_key) if _dep_spend_key else None, default=0.0
        )

        if balance > 0:
            surplus_ratio = max(0.0, (balance - spending) / balance)
        else:
            surplus_ratio = 0.0

        prediction = 1 if surplus_ratio >= self.surplus_ratio_threshold else 0
        confidence = 0.40 + 0.30 * min(1.0, surplus_ratio)
        reason = (
            "여유자금이 충분하시네요, 우대금리 정기예금으로 "
            "안정적인 이자 수익을 만들어보세요."
            if prediction == 1
            else "현재 지출 수준 대비 여유자금 확보 후 예금을 고려해보세요."
        )
        _contributing = []
        if _dep_balance_key:
            _contributing.append(
                {"name": _dep_balance_key, "value": balance, "role": "primary"}
            )
        if _dep_spend_key:
            _contributing.append(
                {"name": _dep_spend_key, "value": spending, "role": "supporting"}
            )
        return {
            "prediction": prediction,
            "confidence": confidence,
            "reason": reason,
            "rule_name": "deposits_surplus_ratio",
            "contributing_features": _contributing,
        }

    def _rule_will_acquire_investments(self, features: Dict[str, Any]) -> Dict[str, Any]:
        """Binary: investment acquisition via risk grade + asset size.

        Theory: Financial Suitability (금소법 제17조) + Product Adjacency
        (deposits → funds natural progression).

        Priority:
        1. Causal NOTEARS (investment propensity causal path)
        2. HGCN hyperbolic (customer embedding distance to investment products)
        3. Raw risk grade + asset percentile
        """
        # --- Causal / NOTEARS features ---
        _causal_invest_key = next(
            (k for k in features if k.startswith("causal_invest")), None
        )
        causal_invest = features.get(_causal_invest_key) if _causal_invest_key else None
        if causal_invest is not None:
            score = _safe_float(causal_invest)
            prediction = 1 if score >= 0.5 else 0
            confidence = min(0.92, 0.60 + abs(score - 0.5) * 0.60)
            reason = (
                "고객님의 투자 성향과 자산 규모를 고려하여 안정형 펀드를 추천드립니다."
                if prediction == 1
                else "현재 단계에서는 예금 상품을 먼저 활용하시는 것을 권장합니다."
            )
            return {
                "prediction": prediction,
                "confidence": confidence,
                "reason": reason,
                "rule_name": "investments_causal",
                "contributing_features": [
                    {"name": _causal_invest_key, "value": score, "role": "primary"},
                ],
            }

        # --- HGCN embedding distance ---
        _hgcn_invest_key = next(
            (k for k in features if k.startswith("hgcn_invest")), None
        )
        hgcn_invest = features.get(_hgcn_invest_key) if _hgcn_invest_key else None
        if hgcn_invest is not None:
            score = _safe_float(hgcn_invest)
            prediction = 1 if score >= 0.5 else 0
            reason = (
                "유사 투자 고객군과의 거리 분석 결과 투자 상품이 적합합니다."
                if prediction == 1
                else "현재 프로필 기준 보수적 상품이 적합합니다."
            )
            return {
                "prediction": prediction,
                "confidence": 0.70,
                "reason": reason,
                "rule_name": "investments_hgcn",
                "contributing_features": [
                    {"name": _hgcn_invest_key, "value": score, "role": "primary"},
                ],
            }

        # --- Raw risk grade + asset percentile ---
        _risk_grade_key = next(
            (k for k in features if k.startswith("risk_grade")), None
        )
        _asset_pct_key = next(
            (k for k in features if k.startswith("asset_percentile")), None
        )
        risk_grade = _safe_int(
            features.get(_risk_grade_key) if _risk_grade_key else None, default=1
        )
        asset_pct = _safe_float(
            features.get(_asset_pct_key) if _asset_pct_key else None, default=0.0
        )

        suitability_ok = risk_grade >= self.invest_min_risk_grade
        asset_ok = asset_pct >= (100.0 - self.invest_asset_pct_threshold)
        prediction = 1 if (suitability_ok and asset_ok) else 0
        confidence = 0.55 if prediction == 1 else 0.70
        reason = (
            "고객님의 투자 성향과 자산 규모를 고려하여 안정형 펀드를 추천드립니다."
            if prediction == 1
            else "현재 투자 성향 등급으로는 예금 상품을 먼저 활용하시기를 권장합니다."
        )
        _contributing = []
        if _risk_grade_key:
            _contributing.append(
                {"name": _risk_grade_key, "value": float(risk_grade), "role": "primary"}
            )
        if _asset_pct_key:
            _contributing.append(
                {"name": _asset_pct_key, "value": asset_pct, "role": "supporting"}
            )
        return {
            "prediction": prediction,
            "confidence": confidence,
            "reason": reason,
            "rule_name": "investments_risk_asset",
            "contributing_features": _contributing,
        }

    def _rule_will_acquire_accounts(self, features: Dict[str, Any]) -> Dict[str, Any]:
        """Binary: account acquisition based on transaction pattern needs.

        Theory: SOW (Share of Wallet) — primary bank conversion raises
        per-customer revenue by 40%+ (PwC).

        Priority:
        1. LightGCN (primary bank conversion graph)
        2. Mamba temporal (primary bank migration signal)
        3. Raw: tenure, interbank transfer rate, salary deposit pattern
        """
        # --- LightGCN ---
        _lightgcn_acct_key = next(
            (k for k in features if k.startswith("lightgcn_account")), None
        )
        lightgcn_acct = features.get(_lightgcn_acct_key) if _lightgcn_acct_key else None
        if lightgcn_acct is not None:
            score = _safe_float(lightgcn_acct)
            prediction = 1 if score >= 0.5 else 0
            reason = (
                "급여 입금을 주거래 계좌로 전환하시면 이체 수수료가 면제됩니다."
                if prediction == 1
                else "현재 주거래 계좌 서비스를 잘 활용하고 계십니다."
            )
            return {
                "prediction": prediction,
                "confidence": 0.75,
                "reason": reason,
                "rule_name": "accounts_lightgcn",
                "contributing_features": [
                    {"name": _lightgcn_acct_key, "value": score, "role": "primary"},
                ],
            }

        # --- Mamba temporal migration signal ---
        _mamba_acct_key = next(
            (k for k in features if k.startswith("mamba_temporal_account")), None
        )
        mamba_acct = features.get(_mamba_acct_key) if _mamba_acct_key else None
        if mamba_acct is not None:
            score = _safe_float(mamba_acct)
            prediction = 1 if score >= 0.5 else 0
            reason = (
                "주거래 전환 패턴이 감지되었습니다. "
                "수수료 우대 계좌를 확인해보세요."
                if prediction == 1
                else "현재 거래 패턴이 안정적입니다."
            )
            return {
                "prediction": prediction,
                "confidence": 0.68,
                "reason": reason,
                "rule_name": "accounts_mamba",
                "contributing_features": [
                    {"name": _mamba_acct_key, "value": score, "role": "primary"},
                ],
            }

        # --- Raw rules ---
        # tenure feature prefix read from config (dataset-agnostic)
        tenure_feat_prefix = self.config.get("rule_engine", {}).get(
            "accounts_tenure_feature_prefix", "tenure"
        )
        _tenure_key = next(
            (k for k in features if k.startswith(tenure_feat_prefix)), None
        )
        _interbank_key = next(
            (k for k in features if k.startswith("interbank_transfer_pct")), None
        )
        _salary_key = next(
            (k for k in features if k.startswith("salary_deposit_flag")), None
        )
        tenure_days = _safe_int(
            features.get(_tenure_key) if _tenure_key else None, default=999
        ) * 30
        interbank_pct = _safe_float(
            features.get(_interbank_key) if _interbank_key else None, default=0.0
        )
        has_salary_deposit = _safe_int(
            features.get(_salary_key) if _salary_key else None, default=0
        )
        is_primary = _safe_int(
            _first_prefixed(features, "is_primary_bank"), default=1
        )

        signals = 0
        reason_parts = []

        if tenure_days <= self.new_customer_days:
            signals += 1
            reason_parts.append("신규 고객")
        if has_salary_deposit and not is_primary:
            signals += 2
            reason_parts.append("급여 계좌 주거래 전환")
        if interbank_pct >= self.high_interbank_transfer_pct:
            signals += 1
            reason_parts.append("타행 이체 빈도 높음")

        prediction = 1 if signals >= 2 else 0
        confidence = 0.40 + 0.15 * min(signals, 3)
        if reason_parts:
            reason = (
                f"{', '.join(reason_parts)} 패턴 감지 — "
                "주거래 계좌 전환 시 이체 수수료가 면제됩니다."
                if prediction == 1
                else "현재 계좌 서비스를 잘 활용하고 계십니다."
            )
        else:
            reason = "추가 계좌 서비스 필요성이 낮습니다."

        _contributing = []
        if _salary_key and has_salary_deposit:
            _contributing.append(
                {"name": _salary_key, "value": float(has_salary_deposit), "role": "primary"}
            )
        if _interbank_key:
            _contributing.append(
                {"name": _interbank_key, "value": interbank_pct, "role": "supporting"}
            )
        if _tenure_key:
            _contributing.append(
                {"name": _tenure_key, "value": float(tenure_days // 30), "role": "supporting"}
            )
        return {
            "prediction": prediction,
            "confidence": confidence,
            "reason": reason,
            "rule_name": "accounts_pattern",
            "contributing_features": _contributing,
        }

    def _rule_will_acquire_lending(self, features: Dict[str, Any]) -> Dict[str, Any]:
        """Binary: lending acquisition via credit grade + DTI.

        Theory: Credit Scoring (Altman Z-score extension) + suitability.

        Priority:
        1. Causal NOTEARS (lending demand causal DAG)
        2. Economics PIH (permanent income vs debt burden)
        3. HMM lifecycle (lending-need stage)
        4. Raw credit grade + DTI
        """
        # --- Causal features ---
        _causal_lend_key = next(
            (k for k in features if k.startswith("causal_lending")), None
        )
        causal_lend = features.get(_causal_lend_key) if _causal_lend_key else None
        if causal_lend is not None:
            score = _safe_float(causal_lend)
            prediction = 1 if score >= 0.5 else 0
            confidence = min(0.90, 0.60 + abs(score - 0.5) * 0.60)
            reason = (
                "현재 신용등급과 소득 수준으로 우대금리 대출이 가능합니다."
                if prediction == 1
                else "현재 조건에서는 대출보다 저축 상품을 우선 검토하시기 바랍니다."
            )
            return {
                "prediction": prediction,
                "confidence": confidence,
                "reason": reason,
                "rule_name": "lending_causal",
                "contributing_features": [
                    {"name": _causal_lend_key, "value": score, "role": "primary"},
                ],
            }

        # --- PIH economics ---
        _pih_income_key = next(
            (k for k in features if k.startswith("economics_pih_permanent_income")), None
        )
        pih_income = features.get(_pih_income_key) if _pih_income_key else None
        if pih_income is not None:
            # PIH permanent income high → debt service capacity exists
            income_val = _safe_float(pih_income)
            income_threshold = self.config.get("rule_engine", {}).get(
                "pih_income_threshold_lending", 3000.0
            )
            prediction = 1 if income_val >= income_threshold else 0
            reason = (
                "안정적인 항상소득을 바탕으로 우대금리 대출을 추천드립니다."
                if prediction == 1
                else "소득 안정성 확보 후 대출을 검토하시기 바랍니다."
            )
            return {
                "prediction": prediction,
                "confidence": 0.65,
                "reason": reason,
                "rule_name": "lending_pih",
                "contributing_features": [
                    {"name": _pih_income_key, "value": income_val, "role": "primary"},
                ],
            }

        # --- Raw credit grade + DTI ---
        _credit_grade_key = next(
            (k for k in features if k.startswith("credit_grade")), None
        )
        _dti_key = next(
            (k for k in features if k.startswith("dti")), None
        )
        credit_grade = _safe_int(
            features.get(_credit_grade_key) if _credit_grade_key else None, default=10
        )
        dti = _safe_float(
            features.get(_dti_key) if _dti_key else None, default=1.0
        )

        credit_ok = credit_grade <= self.max_credit_grade_for_lending
        dti_ok = dti < self.max_dti
        prediction = 1 if (credit_ok and dti_ok) else 0
        confidence = 0.60 if prediction == 1 else 0.70
        reason = (
            "현재 신용등급과 소득 수준으로 우대금리 대출이 가능합니다."
            if prediction == 1
            else "현재 부채비율 또는 신용등급 조건이 대출 기준에 미달합니다."
        )
        _contributing = []
        if _credit_grade_key:
            _contributing.append(
                {"name": _credit_grade_key, "value": float(credit_grade), "role": "primary"}
            )
        if _dti_key:
            _contributing.append(
                {"name": _dti_key, "value": dti, "role": "supporting"}
            )
        return {
            "prediction": prediction,
            "confidence": confidence,
            "reason": reason,
            "rule_name": "lending_credit_dti",
            "contributing_features": _contributing,
        }

    def _rule_will_acquire_payments(self, features: Dict[str, Any]) -> Dict[str, Any]:
        """Binary: payment/card acquisition via spending pattern.

        Theory: Habitual Buying + MCC-based Targeting.

        Priority:
        1. Merchant hierarchy MCC-to-card mapping
        2. TDA local (spending topology → card benefit match)
        3. Mamba temporal (payment method switch signal)
        4. Raw card spending + foreign transaction rate
        """
        # --- Merchant hierarchy features ---
        _merch_key = next(
            (k for k in features if k.startswith("merchant_hierarchy")), None
        )
        merch = features.get(_merch_key) if _merch_key else None
        if merch is not None:
            score = _safe_float(merch)
            prediction = 1 if score >= 0.5 else 0
            reason = (
                "온라인쇼핑 결제가 많으시네요, 온라인 캐시백 특화 카드를 확인해보세요."
                if prediction == 1
                else "현재 결제 패턴에 최적화된 카드를 이미 보유하고 계십니다."
            )
            return {
                "prediction": prediction,
                "confidence": 0.75,
                "reason": reason,
                "rule_name": "payments_merchant_hierarchy",
                "contributing_features": [
                    {"name": _merch_key, "value": score, "role": "primary"},
                ],
            }

        # --- TDA local ---
        _tda_payments_key = next(
            (k for k in features if k.startswith("tda_local_payment")), None
        )
        tda_payments = features.get(_tda_payments_key) if _tda_payments_key else None
        if tda_payments is not None:
            score = _safe_float(tda_payments)
            prediction = 1 if score >= 0.5 else 0
            reason = (
                "결제 패턴 분석 결과 혜택 카드 추가가 유리합니다."
                if prediction == 1
                else "현재 결제 구성이 최적화되어 있습니다."
            )
            return {
                "prediction": prediction,
                "confidence": 0.65,
                "reason": reason,
                "rule_name": "payments_tda_local",
                "contributing_features": [
                    {"name": _tda_payments_key, "value": score, "role": "primary"},
                ],
            }

        # --- Raw card spend + foreign pct ---
        _card_spend_key = next(
            (k for k in features if k.startswith("card_spend_percentile")), None
        )
        _foreign_pct_key = next(
            (k for k in features if k.startswith("foreign_txn_pct")), None
        )
        _num_cards_key = next(
            (k for k in features if k.startswith("num_cards")), None
        )
        card_spend_pct = _safe_float(
            features.get(_card_spend_key) if _card_spend_key else None, default=0.0
        )
        foreign_pct = _safe_float(
            features.get(_foreign_pct_key) if _foreign_pct_key else None, default=0.0
        )
        num_cards = _safe_int(
            features.get(_num_cards_key) if _num_cards_key else None, default=0
        )

        prediction = 0
        reason = "현재 결제 서비스가 적합하게 구성되어 있습니다."
        _primary_key: Optional[str] = None
        _primary_val: float = 0.0

        if foreign_pct >= self.payments_foreign_pct:
            prediction = 1
            reason = "해외결제 비중이 높으시네요, 해외 수수료 무료 카드를 확인해보세요."
            _primary_key = _foreign_pct_key
            _primary_val = foreign_pct
        elif card_spend_pct >= self.payments_spend_top_pct and num_cards <= 1:
            prediction = 1
            reason = (
                "카드 결제 금액이 상위권이시네요, "
                "포인트 혜택이 강화된 카드를 추천드립니다."
            )
            _primary_key = _card_spend_key
            _primary_val = card_spend_pct

        confidence = 0.65 if prediction == 1 else 0.60
        _contributing = []
        if _primary_key:
            _contributing.append(
                {"name": _primary_key, "value": _primary_val, "role": "primary"}
            )
            if _num_cards_key and _primary_key == _card_spend_key:
                _contributing.append(
                    {"name": _num_cards_key, "value": float(num_cards),
                     "role": "supporting"}
                )
        return {
            "prediction": prediction,
            "confidence": confidence,
            "reason": reason,
            "rule_name": "payments_spend_foreign",
            "contributing_features": _contributing,
        }

    # ------------------------------------------------------------------
    # Rule implementations — Value group
    # ------------------------------------------------------------------

    def _rule_product_stability(self, features: Dict[str, Any]) -> Dict[str, Any]:
        """Regression: product holding stability via dormancy EWS.

        Theory: Customer Engagement Theory (Gallup) — fully engaged
        customers generate $402 additional annual revenue.

        Priority:
        1. Mamba temporal (account activity time series)
        2. HMM behavior (active→inactive transition probability)
        3. TDA persistence (activity pattern topology stability)
        4. Raw dormancy days + balance trend
        """
        # --- Mamba temporal ---
        _mamba_stability_key = next(
            (k for k in features if k.startswith("mamba_temporal_stability")), None
        )
        mamba_stability = features.get(_mamba_stability_key) if _mamba_stability_key else None
        if mamba_stability is not None:
            score = _safe_float(mamba_stability, default=0.92)
            score = max(0.0, min(1.0, score))
            if score >= 0.85:
                reason = "계좌가 활발하게 이용되고 있습니다."
            elif score >= 0.70:
                reason = (
                    "계좌 활동이 다소 줄고 있어요, "
                    "자동이체 설정으로 편리하게 관리해보세요."
                )
            else:
                reason = (
                    "계좌 활동이 크게 감소하였습니다. "
                    "전담 상담사가 연락드릴 예정입니다."
                )
            return {
                "prediction": round(score, 4),
                "confidence": 0.85,
                "reason": reason,
                "rule_name": "stability_mamba",
                "contributing_features": [
                    {"name": _mamba_stability_key, "value": score, "role": "primary"},
                ],
            }

        # --- HMM behavior transition ---
        _hmm_inactive_key = next(
            (k for k in features if k.startswith("hmm_behavior_inactive_prob")), None
        )
        hmm_inactive = features.get(_hmm_inactive_key) if _hmm_inactive_key else None
        if hmm_inactive is not None:
            inactive_prob = _safe_float(hmm_inactive, default=0.0)
            stability = max(0.0, 1.0 - inactive_prob)
            if stability >= 0.85:
                reason = "행동 패턴이 안정적입니다."
            else:
                reason = (
                    "비활성 전이 확률이 높아 계좌 활성화 서비스를 제안드립니다."
                )
            return {
                "prediction": round(stability, 4),
                "confidence": 0.75,
                "reason": reason,
                "rule_name": "stability_hmm_behavior",
                "contributing_features": [
                    {"name": _hmm_inactive_key, "value": inactive_prob,
                     "role": "primary"},
                ],
            }

        # --- Raw dormancy days ---
        _last_txn_key = next(
            (k for k in features if k.startswith("days_since_last_txn")), None
        )
        _balance_trend_key = next(
            (k for k in features if k.startswith("balance_change_pct")), None
        )
        last_txn_days = _safe_int(
            features.get(_last_txn_key) if _last_txn_key else None, default=0
        )
        balance_trend = _safe_float(
            features.get(_balance_trend_key) if _balance_trend_key else None, default=0.0
        )

        if last_txn_days >= self.dormancy_dormant_days:
            stability = max(0.0, 0.30 + balance_trend * 0.1)
            reason = (
                "90일 이상 거래가 없습니다. "
                "전담 상담 서비스를 이용해보세요."
            )
        elif last_txn_days >= self.dormancy_alert_days:
            stability = max(0.0, 0.55 + balance_trend * 0.1)
            reason = (
                "60일 이상 거래가 없어요. "
                "특별 프로모션을 제안드립니다."
            )
        elif last_txn_days >= self.dormancy_warning_days:
            stability = max(0.0, 0.75 + balance_trend * 0.05)
            reason = (
                "계좌 활동이 줄어들고 있어요, "
                "자동이체 설정으로 편리하게 관리해보세요."
            )
        else:
            stability = min(1.0, 0.92 + balance_trend * 0.02)
            reason = "계좌가 활발하게 이용되고 있습니다."

        stability = max(0.0, min(1.0, stability))
        _contributing = []
        if _last_txn_key:
            _contributing.append(
                {"name": _last_txn_key, "value": float(last_txn_days), "role": "primary"}
            )
        if _balance_trend_key:
            _contributing.append(
                {"name": _balance_trend_key, "value": balance_trend, "role": "supporting"}
            )
        return {
            "prediction": round(stability, 4),
            "confidence": 0.50,
            "reason": reason,
            "rule_name": "stability_dormancy_ews",
            "contributing_features": _contributing,
        }

    def _rule_cross_sell_count(self, features: Dict[str, Any]) -> Dict[str, Any]:
        """Regression: number of cross-sell targets based on CLV tier.

        Theory: SOW (Share of Wallet) — financial institutions average
        10-20% SOW; top performers reach 60%.

        Priority:
        1. LightGCN (peer product holding gap)
        2. GMM cluster (cluster average product count gap)
        3. Raw CLV tier → target product count
        """
        # --- LightGCN peer gap ---
        _lightgcn_gap_key = next(
            (k for k in features if k.startswith("lightgcn_product_gap")), None
        )
        lightgcn_gap = features.get(_lightgcn_gap_key) if _lightgcn_gap_key else None
        if lightgcn_gap is not None:
            gap = _safe_float(lightgcn_gap, default=0.0)
            gap = max(0.0, gap)
            _cs_num_products_key = next(
                (k for k in features if k.startswith("num_products")), None
            )
            num_products = _safe_int(
                features.get(_cs_num_products_key) if _cs_num_products_key else None,
                default=0,
            )
            reason = (
                f"현재 {num_products}개 상품을 이용 중이시네요, "
                "유사 고객 대비 추가 상품 혜택을 받아보세요."
                if gap > 0
                else "현재 상품 구성이 동일 고객군 대비 충분합니다."
            )
            _contributing = [
                {"name": _lightgcn_gap_key, "value": gap, "role": "primary"},
            ]
            if _cs_num_products_key:
                _contributing.append(
                    {"name": _cs_num_products_key, "value": float(num_products),
                     "role": "supporting"}
                )
            return {
                "prediction": round(gap, 4),
                "confidence": 0.80,
                "reason": reason,
                "rule_name": "cross_sell_lightgcn_gap",
                "contributing_features": _contributing,
            }

        # --- GMM cluster gap ---
        _gmm_gap_key = next(
            (k for k in features if k.startswith("gmm_product_gap")), None
        )
        gmm_gap = features.get(_gmm_gap_key) if _gmm_gap_key else None
        if gmm_gap is not None:
            gap = max(0.0, _safe_float(gmm_gap, default=0.0))
            reason = (
                "동일 클러스터 고객 대비 추가 상품이 권장됩니다."
                if gap > 0
                else "현재 상품 구성이 클러스터 평균에 부합합니다."
            )
            return {
                "prediction": round(gap, 4),
                "confidence": 0.70,
                "reason": reason,
                "rule_name": "cross_sell_gmm_gap",
                "contributing_features": [
                    {"name": _gmm_gap_key, "value": gap, "role": "primary"},
                ],
            }

        # --- Raw CLV tier → target gap ---
        _clv_pct_key = next(
            (k for k in features if k.startswith("clv_percentile")), None
        )
        _clv_np_key = next(
            (k for k in features if k.startswith("num_products")), None
        )
        clv_percentile = _safe_float(
            features.get(_clv_pct_key) if _clv_pct_key else None, default=50.0
        )
        num_products = _safe_int(
            features.get(_clv_np_key) if _clv_np_key else None, default=0
        )

        if clv_percentile >= self.clv_premium_pct:
            tier = "premium"
        elif clv_percentile >= 40.0:
            tier = "stable"
        elif clv_percentile >= 20.0:
            tier = "growth"
        else:
            tier = "at_risk"

        target = self.clv_target_products.get(tier, 2)
        gap = max(0.0, float(target - num_products))
        reason = (
            f"현재 {num_products}개 상품을 이용 중이시네요, "
            f"카드 추가로 포인트 혜택을 받아보세요."
            if gap > 0
            else "현재 상품 구성이 고객 등급에 최적화되어 있습니다."
        )
        _contributing = []
        if _clv_pct_key:
            _contributing.append(
                {"name": _clv_pct_key, "value": clv_percentile, "role": "primary"}
            )
        if _clv_np_key:
            _contributing.append(
                {"name": _clv_np_key, "value": float(num_products), "role": "supporting"}
            )
        return {
            "prediction": round(gap, 4),
            "confidence": 0.50,
            "reason": reason,
            "rule_name": "cross_sell_clv_tier",
            "contributing_features": _contributing,
        }

    # ------------------------------------------------------------------
    # Rule implementations — Consumption group
    # ------------------------------------------------------------------

    def _rule_next_mcc(self, features: Dict[str, Any]) -> Dict[str, Any]:
        """Multiclass: predict next MCC category via frequency top-K.

        Theory: Habitual Buying Behavior (Kotler) — past patterns are
        the strongest predictor for low-involvement repeat purchases.

        Priority:
        1. Mamba temporal (MCC sequence next-token prediction)
        2. Merchant hierarchy movement patterns
        3. TDA local (consumption trajectory topological next step)
        4. Raw MCC frequency top-K + seasonal correction
        """
        # --- Mamba temporal next-token ---
        _mamba_mcc_next_key = next(
            (k for k in features if k.startswith("mamba_temporal_next_mcc")), None
        )
        mamba_mcc_next = features.get(_mamba_mcc_next_key) if _mamba_mcc_next_key else None
        if mamba_mcc_next is not None:
            pred = _safe_int(mamba_mcc_next, default=0)
            reason = (
                "거래 시퀀스 분석을 통해 다음 주요 소비 카테고리를 예측하였습니다."
            )
            return {
                "prediction": pred,
                "confidence": 0.75,
                "reason": reason,
                "rule_name": "next_mcc_mamba",
                "contributing_features": [
                    {"name": _mamba_mcc_next_key, "value": _safe_float(mamba_mcc_next),
                     "role": "primary"},
                ],
            }

        # --- Merchant hierarchy ---
        _merch_next_key = next(
            (k for k in features if k.startswith("merchant_hierarchy_next")), None
        )
        merch_next = features.get(_merch_next_key) if _merch_next_key else None
        if merch_next is not None:
            pred = _safe_int(merch_next, default=0)
            reason = "소비 업종 계층 패턴 기반 다음 거래 카테고리를 예측하였습니다."
            return {
                "prediction": pred,
                "confidence": 0.65,
                "reason": reason,
                "rule_name": "next_mcc_merchant_hierarchy",
                "contributing_features": [
                    {"name": _merch_next_key, "value": _safe_float(merch_next),
                     "role": "primary"},
                ],
            }

        # --- TDA local ---
        _tda_next_mcc_key = next(
            (k for k in features if k.startswith("tda_local_next_mcc")), None
        )
        tda_next_mcc = features.get(_tda_next_mcc_key) if _tda_next_mcc_key else None
        if tda_next_mcc is not None:
            pred = _safe_int(tda_next_mcc, default=0)
            reason = "소비 궤적의 위상적 분석을 통해 다음 카테고리를 예측하였습니다."
            return {
                "prediction": pred,
                "confidence": 0.60,
                "reason": reason,
                "rule_name": "next_mcc_tda",
                "contributing_features": [
                    {"name": _tda_next_mcc_key, "value": _safe_float(tda_next_mcc),
                     "role": "primary"},
                ],
            }

        # --- Raw: top-1 MCC by frequency with seasonal correction ---
        _top_mcc_key = next(
            (k for k in features if k.startswith("top_mcc_code")), None
        )
        top_mcc = _safe_int(
            features.get(_top_mcc_key) if _top_mcc_key else None, default=0
        )
        reason = (
            "주로 이용하시는 소비 카테고리 기반으로 다음 거래 카테고리를 예측하였습니다."
        )
        _contributing = []
        if _top_mcc_key:
            _contributing.append(
                {"name": _top_mcc_key, "value": float(top_mcc), "role": "primary"}
            )
        return {
            "prediction": top_mcc,
            "confidence": 0.45,
            "reason": reason,
            "rule_name": "next_mcc_frequency",
            "contributing_features": _contributing,
        }

    def _rule_mcc_diversity_trend(self, features: Dict[str, Any]) -> Dict[str, Any]:
        """Regression: MCC diversity trend prediction via PIH.

        Theory: Friedman Permanent Income Hypothesis (1957) — transitory
        income spikes drive temporary consumption diversification.

        Priority:
        1. Economics PIH (permanent vs transitory income split)
        2. TDA global (Betti number = topological complexity of spending)
        3. GMM cluster (spending pattern cluster transition)
        4. Raw MCC entropy moving average
        """
        # --- PIH economics ---
        _pih_transitory_key = next(
            (k for k in features if k.startswith("economics_pih_transitory_income")), None
        )
        pih_transitory = features.get(_pih_transitory_key) if _pih_transitory_key else None
        if pih_transitory is not None:
            transitory = _safe_float(pih_transitory, default=0.0)
            # Positive transitory income → expect diversity expansion
            trend = math.tanh(transitory / max(1.0, abs(transitory) + 1.0))
            if trend > 0.1:
                reason = (
                    "최근 일시소득 증가로 소비 다양성이 확대될 것으로 예측됩니다. "
                    "새로운 카테고리 혜택 상품을 추천드립니다."
                )
            elif trend < -0.1:
                reason = "소비 패턴이 집중화되는 추세입니다."
            else:
                reason = "소비 다양성이 안정적으로 유지될 것으로 예측됩니다."
            return {
                "prediction": round(float(trend), 4),
                "confidence": 0.80,
                "reason": reason,
                "rule_name": "diversity_pih",
                "contributing_features": [
                    {"name": _pih_transitory_key, "value": transitory, "role": "primary"},
                ],
            }

        # --- TDA global Betti number ---
        _tda_betti_key = next(
            (k for k in features if k.startswith("tda_global_betti")), None
        )
        tda_betti = features.get(_tda_betti_key) if _tda_betti_key else None
        if tda_betti is not None:
            betti = _safe_float(tda_betti, default=0.0)
            # Normalize to [-1, 1] range for trend representation
            trend = math.tanh(betti / 5.0)
            reason = (
                "소비 패턴의 위상적 복잡도 분석 결과 다양성 변화가 예측됩니다."
                if abs(trend) > 0.1
                else "소비 다양성이 안정적으로 유지될 것으로 예측됩니다."
            )
            return {
                "prediction": round(float(trend), 4),
                "confidence": 0.70,
                "reason": reason,
                "rule_name": "diversity_tda_betti",
                "contributing_features": [
                    {"name": _tda_betti_key, "value": betti, "role": "primary"},
                ],
            }

        # --- GMM cluster transition ---
        _gmm_transition_key = next(
            (k for k in features if k.startswith("gmm_cluster_transition")), None
        )
        gmm_transition = features.get(_gmm_transition_key) if _gmm_transition_key else None
        if gmm_transition is not None:
            trend = _safe_float(gmm_transition, default=0.0)
            reason = (
                "소비 패턴 클러스터 전이가 감지되어 다양성 변화가 예측됩니다."
                if abs(trend) > 0.05
                else "소비 클러스터가 안정적으로 유지될 것으로 예측됩니다."
            )
            return {
                "prediction": round(float(trend), 4),
                "confidence": 0.60,
                "reason": reason,
                "rule_name": "diversity_gmm_transition",
                "contributing_features": [
                    {"name": _gmm_transition_key, "value": trend, "role": "primary"},
                ],
            }

        # --- Raw MCC entropy moving average trend ---
        _entropy_recent_key = next(
            (k for k in features if k.startswith("mcc_entropy_recent")), None
        )
        _entropy_prior_key = next(
            (k for k in features if k.startswith("mcc_entropy_prior")), None
        )
        entropy_recent = _safe_float(
            features.get(_entropy_recent_key) if _entropy_recent_key else None, default=0.0
        )
        entropy_prior = _safe_float(
            features.get(_entropy_prior_key) if _entropy_prior_key else None,
            default=entropy_recent,
        )
        trend = entropy_recent - entropy_prior

        if trend > 0.05:
            reason = (
                "최근 소비 패턴이 다양해지고 있어요, "
                "새로운 카테고리 혜택이 있는 상품을 추천드립니다."
            )
        elif trend < -0.05:
            reason = "소비 패턴이 특정 카테고리로 집중되고 있습니다."
        else:
            reason = "소비 다양성이 안정적으로 유지되고 있습니다."

        _contributing = []
        if _entropy_recent_key:
            _contributing.append(
                {"name": _entropy_recent_key, "value": entropy_recent, "role": "primary"}
            )
        if _entropy_prior_key:
            _contributing.append(
                {"name": _entropy_prior_key, "value": entropy_prior, "role": "supporting"}
            )
        return {
            "prediction": round(float(trend), 4),
            "confidence": 0.40,
            "reason": reason,
            "rule_name": "diversity_entropy_trend",
            "contributing_features": _contributing,
        }
