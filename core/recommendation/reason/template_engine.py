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
from typing import Any, Dict, List, Optional, Tuple, TYPE_CHECKING

if TYPE_CHECKING:
    from core.feature.group_config import FeatureGroupConfig

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

        # Task name -> customer-facing product name
        self.item_name_map: Dict[str, str] = te_cfg.get(
            "item_name_map", {},
        )

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
    # Auto-configuration from FeatureGroupConfig
    # ------------------------------------------------------------------

    @classmethod
    def from_feature_groups(
        cls,
        groups: List["FeatureGroupConfig"],
        template_pool: Optional[Dict[str, List[str]]] = None,
        task_frames: Optional[Dict[str, Dict[str, str]]] = None,
        top_k_features: int = 3,
    ) -> "TemplateEngine":
        """Build a TemplateEngine with auto-generated feature-to-category mapping.

        Instead of manually listing feature prefix -> category mappings in
        the config YAML, this reads FeatureGroupConfig objects and derives
        the mapping from each group's output columns and interpretation
        category.

        A per-group template is also auto-registered in the template pool
        (using the group's ``interpretation.template``), so even without a
        manually curated template pool, every feature group has at least one
        template available.

        Manual config still works -- pass ``template_pool`` and
        ``task_frames`` to supplement or override the auto-generated entries.

        Args:
            groups: Ordered list of FeatureGroupConfig instances.
            template_pool: Optional additional templates keyed by category.
                           Merged with (and overrides) auto-generated
                           templates.
            task_frames: Optional task-specific narrative frames.
            top_k_features: Number of top IG features to use for reasons.

        Returns:
            A fully configured TemplateEngine instance.
        """
        from core.feature.group_config import FeatureGroupConfig

        # Auto-generate feature_category_map from output columns
        feature_category_map: Dict[str, str] = {}
        auto_template_pool: Dict[str, List[str]] = {}
        auto_task_frames: Dict[str, Dict[str, str]] = {}

        for group in groups:
            if not group.enabled:
                continue

            category = group.interpretation.category

            # Map each output column prefix to the group's category.
            # We use the column name itself as the prefix key, so exact
            # prefix matching in _classify_feature will work.
            for col in group.output_columns:
                # Use the column name as a prefix key.  Since the
                # template engine uses startswith matching, the full
                # column name is the most specific prefix possible.
                feature_category_map[col] = category

            # Also add the group name as a prefix (catches any features
            # that start with the group name but aren't in output_columns).
            feature_category_map[f"{group.name}_"] = category

            # Register the group's interpretation template in the pool
            if category not in auto_template_pool:
                auto_template_pool[category] = []
            # Convert group template to the template engine format
            # (replace {feature}/{value}/{direction}/{pattern} with
            # {item_name}/{category} placeholders for compatibility).
            auto_template_pool[category].append(
                group.interpretation.template,
            )

            # Build task frames from primary_tasks
            for task in group.interpretation.primary_tasks:
                if task not in auto_task_frames:
                    auto_task_frames[task] = {
                        "frame": group.interpretation.narrative_lens,
                        "narrative": (
                            f"Based on {category.replace('_', ' ')} analysis, "
                            f"this recommendation is tailored to your profile."
                        ),
                    }

        # Ensure fallback categories always exist
        for fallback in ("popularity", "minimum_safe"):
            if fallback not in auto_template_pool:
                auto_template_pool[fallback] = [
                    "{item_name} is recommended based on your overall profile."
                ]

        # Merge: user-provided template_pool overrides auto-generated
        if template_pool:
            for cat, templates in template_pool.items():
                auto_template_pool[cat] = templates

        # Merge: user-provided task_frames overrides auto-generated
        if task_frames:
            auto_task_frames.update(task_frames)

        # Build config dict in the format TemplateEngine expects
        config: Dict[str, Any] = {
            "reason": {
                "template_engine": {
                    "top_k_features": top_k_features,
                    "feature_category_map": feature_category_map,
                    "template_pool": auto_template_pool,
                    "task_frames": auto_task_frames,
                },
            },
        }

        instance = cls(config=config)
        logger.info(
            "TemplateEngine.from_feature_groups: %d feature prefixes mapped, "
            "%d categories in pool, %d task frames",
            len(feature_category_map),
            len(auto_template_pool),
            len(auto_task_frames),
        )
        return instance

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
        task_name: Optional[str] = None,
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
            task_type: Optional task type (binary/multiclass/regression).
            task_name: Optional task name for task_frame lookup
                       (e.g. ``churn_signal``). Falls back to item_id.
            item_info: Optional item metadata dict; falls back to
                       ``self.item_metadata[item_id]``.

        Returns:
            Dict with ``reasons``, ``generation_method``, ``segment``, etc.
        """
        info = item_info or self.item_metadata.get(item_id, {})
        # Resolve customer-facing name: item_name_map > item_info > item_id
        _task_key = task_name or item_id
        item_name = (
            self.item_name_map.get(_task_key)
            or info.get("name")
            or item_id
        )

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

        # Deduplicate identical reason texts
        seen_texts: set = set()
        unique_reasons: list = []
        for r in reasons:
            if r["text"] not in seen_texts:
                seen_texts.add(r["text"])
                unique_reasons.append(r)
        reasons = unique_reasons

        # Apply task_frame wrapping to each reason text
        frame = self.task_frames.get(_task_key, {})
        if frame:
            frame_tpl = frame.get("positive", "{reason}")
            for r in reasons:
                r["text"] = frame_tpl.replace("{reason}", r["text"])

        result: Dict[str, Any] = {
            "customer_id": customer_id,
            "item_id": item_id,
            "segment": segment,
            "reasons": reasons,
            "generation_method": "template_l1",
            "generated_at": datetime.utcnow().isoformat() + "Z",
        }

        return result

    # ------------------------------------------------------------------
    # Internal -- IG based
    # ------------------------------------------------------------------

    def _ig_based_reasons(
        self,
        customer_id: str,
        ig_top_features: list,
        item_name: str,
        item_info: Dict[str, Any],
        task_type: Optional[str],
    ) -> List[Dict[str, Any]]:
        """Map IG top features to template reasons.

        Accepts ig_top_features in two formats:
            - Legacy: ``[(feature_name, ig_score), ...]``
            - Enriched (from InterpretationRegistry):
              ``[(feature_name, ig_score, interpretation_text), ...]``

        When interpretation text is present and non-empty, it is used
        directly as the L1 reason text instead of template rendering.
        """
        reasons: List[Dict[str, Any]] = []
        top_n = ig_top_features[: self.top_k_features]

        for rank, entry in enumerate(top_n, start=1):
            # Support both (name, score) and (name, score, text) tuples
            if len(entry) >= 3:
                feat_name, ig_score, interp_text = entry[0], entry[1], entry[2]
            else:
                feat_name, ig_score = entry[0], entry[1]
                interp_text = ""

            # If InterpretationRegistry provided a text, use it directly
            if interp_text:
                reason_type = "primary" if rank == 1 else "supplementary"
                reasons.append({
                    "rank": rank,
                    "type": reason_type,
                    "text": f"{item_name}: {interp_text}",
                    "feature": feat_name,
                    "ig_score": float(ig_score),
                    "category": "interpretation_registry",
                })
                continue

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

    # ------------------------------------------------------------------
    # Batch generation (Stage B interpretability)
    # ------------------------------------------------------------------

    # AI disclosure notice per Korean financial regulations
    AI_DISCLOSURE_KO = (
        "[AI 자동 생성 안내] 본 추천 사유는 AI 모델의 분석 결과를 바탕으로 "
        "자동 생성되었습니다. 최종 금융 의사결정은 전문 상담사와 상의하시기 바랍니다."
    )

    def generate_batch(
        self,
        ig_attributions: List[Dict[str, Any]],
        product_info: Dict[str, Dict[str, Any]],
        segments: Dict[str, str],
        task_type: Optional[str] = None,
        include_disclosure: bool = True,
    ) -> List[Dict[str, Any]]:
        """Generate reasons for all customers in batch.

        Processes all customer-item pairs using IG Top-K feature
        attribution and template matching.  No LLM calls are required,
        so the full batch completes in minutes rather than hours.

        Args:
            ig_attributions: List of dicts, each with ``customer_id``,
                ``item_id``, and ``ig_top_features`` (list of
                ``(feature_name, ig_score)`` tuples sorted descending).
            product_info: ``{item_id: {name, primary_category, ...}}``.
            segments: ``{customer_id: segment}`` mapping.
            task_type: Optional task type for narrative framing.
            include_disclosure: Whether to append AI disclosure notice
                per Korean financial regulations.

        Returns:
            List of reason dicts (same format as :meth:`generate_reason`).
        """
        results: List[Dict[str, Any]] = []

        for entry in ig_attributions:
            cust_id = entry.get("customer_id", "")
            item_id = entry.get("item_id", "")
            ig_feats = entry.get("ig_top_features", [])
            segment = segments.get(cust_id, "WARMSTART")
            item_info = product_info.get(item_id, {})

            result = self.generate_reason(
                customer_id=cust_id,
                item_id=item_id,
                ig_top_features=ig_feats,
                segment=segment,
                task_type=task_type,
                item_info=item_info,
            )

            # Append AI disclosure notice
            if include_disclosure:
                result["ai_disclosure"] = self.AI_DISCLOSURE_KO

            results.append(result)

        logger.info(
            "Batch generated %d template reasons (task_type=%s)",
            len(results), task_type,
        )
        return results

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

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
