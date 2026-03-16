"""
Template Reason Engine (L1)
============================

Generates recommendation reasons using templates -- no LLM call required.

Key design decisions:
    * Template pool and feature-category mappings come **entirely from
      config** (YAML).  No templates are hardcoded in this module.
    * Deterministic variant selection via ``customer_id`` hash so the same
      customer always sees the same wording (reproducible A/B testing).
    * Task-specific narrative framing from config.
    * Fallback hierarchy:  IG match -> popularity -> minimum_safe.

Config example (``reason.template_engine``)::

    reason:
      template_engine:
        top_k_features: 3
        feature_category_map:
          spend_: spending_pattern
          txn_count_: frequency_pattern
          amt_: spending_pattern
          merchant_: frequency_pattern
          life_stage_: life_stage
          benefit_: benefit_match
        template_pool:
          spending_pattern:
            - "{item_name} offers strong benefits in your primary spending area of {category}."
            - "Based on your {category} spending patterns, {item_name} is a strong fit."
          frequency_pattern:
            - "{item_name} is tailored for frequent {merchant_type} users like you."
          life_stage:
            - "{item_name} is optimised for customers at your life stage."
          benefit_match:
            - "{item_name} delivers {benefit_type} benefits aligned with your profile."
          popularity:
            - "{item_name} is popular among customers with a similar profile."
          minimum_safe:
            - "{item_name} is recommended based on your overall profile analysis."
        task_frames:
          churn:
            frame: retention
            narrative: "We value your loyalty and want to offer you something special."
          ltv:
            frame: growth
            narrative: "Maximise long-term value with this recommendation."
"""

from __future__ import annotations

import hashlib
import logging
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

__all__ = ["TemplateEngine"]


class TemplateEngine:
    """L1 template-based recommendation reason generator.

    All configuration -- templates, feature mappings, task frames -- is
    injected via the *config* dict at construction time.

    Args:
        config: Full pipeline config.  Reads ``config["reason"]["template_engine"]``.
    """

    def __init__(self, config: Dict[str, Any]) -> None:
        te_cfg = config.get("reason", {}).get("template_engine", {})

        self.top_k_features: int = te_cfg.get("top_k_features", 3)

        # Feature prefix -> template category mapping
        self.feature_category_map: Dict[str, str] = te_cfg.get(
            "feature_category_map", {},
        )

        # Template pool: {category: [template_strings]}
        self.template_pool: Dict[str, List[str]] = te_cfg.get(
            "template_pool", {},
        )

        # Task-specific narrative frames
        self.task_frames: Dict[str, Dict[str, str]] = te_cfg.get(
            "task_frames", {},
        )

        # Item metadata (can be provided at init or per-call)
        self.item_metadata: Dict[str, Dict[str, Any]] = te_cfg.get(
            "item_metadata", {},
        )

        logger.info(
            "TemplateEngine initialised: %d categories, %d task frames",
            len(self.template_pool),
            len(self.task_frames),
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def generate_reason(
        self,
        customer_id: str,
        item_id: str,
        ig_top_features: List[Tuple[str, float]],
        segment: str = "WARMSTART",
        task_type: Optional[str] = None,
        item_info: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Generate recommendation reasons for a customer-item pair.

        Args:
            customer_id: Customer identifier (used for deterministic hashing).
            item_id: Item / product identifier.
            ig_top_features: List of ``(feature_name, ig_score)`` sorted by
                             descending importance.
            segment: Customer segment (``WARMSTART`` / ``COLDSTART`` /
                     ``ANONYMOUS``).
            task_type: Optional task type for narrative framing.
            item_info: Optional item metadata dict; falls back to
                       ``self.item_metadata[item_id]``.

        Returns:
            Dict with ``reasons``, ``generation_method``, ``segment``, etc.
        """
        info = item_info or self.item_metadata.get(item_id, {})
        item_name = info.get("name", item_id)

        # Segment dispatch
        cid = str(customer_id)
        if segment == "COLDSTART":
            reasons = self._coldstart_reasons(cid, item_name, info)
        elif segment == "ANONYMOUS":
            reasons = self._anonymous_reasons(cid, item_name)
        else:
            reasons = self._ig_based_reasons(
                cid, ig_top_features, item_name, info, task_type,
            )
            if not reasons:
                reasons = self._popularity_fallback(cid, item_name)

        result: Dict[str, Any] = {
            "customer_id": customer_id,
            "item_id": item_id,
            "segment": segment,
            "reasons": reasons,
            "generation_method": "template_l1",
            "generated_at": datetime.utcnow().isoformat() + "Z",
        }

        # Task-specific narrative frame
        if task_type and task_type in self.task_frames:
            result["task_frame"] = self.task_frames[task_type]

        return result

    # ------------------------------------------------------------------
    # Internal -- IG based
    # ------------------------------------------------------------------

    def _ig_based_reasons(
        self,
        customer_id: str,
        ig_top_features: List[Tuple[str, float]],
        item_name: str,
        item_info: Dict[str, Any],
        task_type: Optional[str],
    ) -> List[Dict[str, Any]]:
        """Map IG top features to template reasons."""
        reasons: List[Dict[str, Any]] = []
        top_n = ig_top_features[: self.top_k_features]

        for rank, (feat_name, ig_score) in enumerate(top_n, start=1):
            category = self._classify_feature(feat_name)
            if category is None:
                continue

            template = self._select_variant(customer_id, category, task_type)
            if template is None:
                continue

            text = self._render_template(template, item_name, item_info)
            reason_type = "primary" if rank == 1 else "supplementary"
            reasons.append({
                "rank": rank,
                "type": reason_type,
                "text": text,
                "feature": feat_name,
                "ig_score": float(ig_score),
                "category": category,
            })

        # Minimum-safe fallback if no IG features matched
        if not reasons:
            tpl = self._select_variant(customer_id, "minimum_safe")
            if tpl:
                reasons.append({
                    "rank": 1,
                    "type": "primary",
                    "text": self._render_template(tpl, item_name, item_info),
                    "feature": "fallback",
                    "ig_score": 0.0,
                    "category": "minimum_safe",
                })

        return reasons

    # ------------------------------------------------------------------
    # Internal -- segment fallbacks
    # ------------------------------------------------------------------

    def _coldstart_reasons(
        self,
        customer_id: str,
        item_name: str,
        item_info: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        """COLDSTART: popularity + optional benefit match."""
        reasons: List[Dict[str, Any]] = []

        tpl = self._select_variant(customer_id, "popularity")
        if tpl:
            reasons.append({
                "rank": 1,
                "type": "primary",
                "text": self._render_template(tpl, item_name, item_info),
                "feature": "coldstart_popularity",
                "ig_score": None,
                "category": "popularity",
            })

        benefit_tpl = self._select_variant(customer_id, "benefit_match")
        if benefit_tpl and item_info.get("benefit_type"):
            reasons.append({
                "rank": 2,
                "type": "supplementary",
                "text": self._render_template(benefit_tpl, item_name, item_info),
                "feature": "coldstart_benefit",
                "ig_score": None,
                "category": "benefit_match",
            })

        return reasons

    def _anonymous_reasons(
        self,
        customer_id: str,
        item_name: str,
    ) -> List[Dict[str, Any]]:
        """ANONYMOUS: generic popularity only."""
        tpl = self._select_variant(customer_id, "popularity")
        text = self._render_template(tpl, item_name, {}) if tpl else item_name
        return [{
            "rank": 1,
            "type": "primary",
            "text": text,
            "feature": "anonymous_popularity",
            "ig_score": None,
            "category": "popularity",
        }]

    def _popularity_fallback(
        self,
        customer_id: str,
        item_name: str,
    ) -> List[Dict[str, Any]]:
        """Final fallback when IG mapping produces nothing."""
        tpl = self._select_variant(customer_id, "popularity")
        text = self._render_template(tpl, item_name, {}) if tpl else item_name
        return [{
            "rank": 1,
            "type": "primary",
            "text": text,
            "feature": "fallback_popularity",
            "ig_score": None,
            "category": "popularity",
        }]

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _classify_feature(self, feat_name: str) -> Optional[str]:
        """Map a feature name to a template category via prefix matching."""
        for prefix, category in self.feature_category_map.items():
            if feat_name.startswith(prefix):
                return category
        return None

    def _select_variant(
        self,
        customer_id: str,
        category: str,
        task_type: Optional[str] = None,
    ) -> Optional[str]:
        """Deterministic variant selection via customer_id hash.

        If *task_type* is provided and has category-specific overrides in
        ``task_frames``, those templates are preferred.
        """
        # Check task-specific template pool first
        if task_type:
            task_cfg = self.task_frames.get(task_type, {})
            task_templates = task_cfg.get("templates", {})
            variants = task_templates.get(category)
            if variants:
                return self._hash_select(customer_id, category, variants, salt=task_type)

        # Fall back to global pool
        variants = self.template_pool.get(category)
        if not variants:
            return None
        return self._hash_select(customer_id, category, variants)

    @staticmethod
    def _hash_select(
        customer_id: str,
        category: str,
        variants: List[str],
        salt: str = "",
    ) -> str:
        """Pick a variant deterministically from *variants*."""
        key = f"{customer_id}:{category}:{salt}"
        digest = hashlib.md5(key.encode()).hexdigest()
        idx = int(digest, 16) % len(variants)
        return variants[idx]

    @staticmethod
    def _render_template(
        template: str,
        item_name: str,
        item_info: Dict[str, Any],
    ) -> str:
        """Safely render a template string.

        Available placeholders:
        ``{item_name}``, ``{category}``, ``{merchant_type}``,
        ``{benefit_type}``, ``{life_stage_desc}``.
        """
        substitutions = {
            "item_name": item_name,
            "category": item_info.get("primary_category", "general"),
            "merchant_type": item_info.get("merchant_type", "merchant"),
            "benefit_type": item_info.get("benefit_type", ""),
            "life_stage_desc": item_info.get("life_stage_desc", ""),
        }
        try:
            return template.format(**substitutions)
        except (KeyError, IndexError) as exc:
            logger.warning("Template render failed: %s", exc)
            return template
