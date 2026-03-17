"""
Full Recommendation Pipeline
==============================

End-to-end orchestrator that wires together every component of the
recommendation intelligence layer:

    model predictions --> scoring --> filtering --> top-K --> reasons --> self-check

All behaviour is config-driven.  The pipeline is the single entry point
for recommendation requests.

Config top-level keys consumed::

    scorer:           # WeightedSumScorer / FDTVSScorer / custom plugin
    filters:          # Per-filter config
    constraint_engine:  # Filter chain + fail_fast
    selector:         # k, diversity_method, lambda
    reason:           # template_engine, reverse_mapper, self_checker
    llm_provider:     # backend, bedrock/openai/dummy sub-config
    pipeline:         # orchestration-level settings
      scorer_name: weighted_sum
      enable_reasons: true
      enable_self_check: true
      enable_reverse_mapping: true
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, TYPE_CHECKING

import numpy as np

from .scorer import AbstractScorer, ScorerRegistry, ScoringResult
from .constraint_engine import ConstraintEngine
from .selector import TopKSelector
from .reason.template_engine import TemplateEngine
from .reason.reverse_mapper import ReverseMapper
from .reason.self_checker import SelfChecker, CheckResult
from .reason.llm_provider import LLMProviderFactory, AbstractLLMProvider

if TYPE_CHECKING:
    from core.feature.group_config import FeatureGroupConfig

logger = logging.getLogger(__name__)

__all__ = ["RecommendationPipeline", "RecommendationResult", "RecommendationItem"]


# ---------------------------------------------------------------------------
# Result data classes
# ---------------------------------------------------------------------------

@dataclass
class RecommendationItem:
    """A single recommendation produced by the pipeline.

    Attributes:
        customer_id: Customer identifier.
        item_id: Recommended item identifier.
        rank: 1-based rank in the final list.
        score: Final priority score.
        score_components: Named scoring sub-components.
        reasons: List of reason dicts (from template engine).
        check_result: Self-checker verdict, if enabled.
        metadata: Extra metadata (timings, filter info, etc.).
    """

    customer_id: str
    item_id: str
    rank: int
    score: float
    score_components: Dict[str, float] = field(default_factory=dict)
    reasons: List[Dict[str, Any]] = field(default_factory=list)
    check_result: Optional[CheckResult] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RecommendationResult:
    """Full pipeline output for one customer.

    Attributes:
        customer_id: Customer identifier.
        items: Ordered list of :class:`RecommendationItem`.
        total_candidates: How many candidates were considered before filtering.
        total_filtered: How many candidates survived filtering.
        elapsed_ms: Wall-clock time for the entire pipeline call (ms).
        metadata: Pipeline-level metadata.
    """

    customer_id: str
    items: List[RecommendationItem] = field(default_factory=list)
    total_candidates: int = 0
    total_filtered: int = 0
    elapsed_ms: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

class RecommendationPipeline:
    """Config-driven end-to-end recommendation pipeline.

    Orchestration flow::

        1. Score all candidate items for the customer.
        2. Filter out ineligible candidates (constraints).
        3. Select top-K with optional diversity.
        4. Generate recommendation reasons (template engine).
        5. Self-check reasons for compliance / injection.
        6. Package results.

    Args:
        config: Full pipeline configuration dict (typically loaded from
                a YAML file).
    """

    def __init__(self, config: Dict[str, Any], audit_store=None) -> None:
        self.config = config
        self._audit_store = audit_store
        pipe_cfg = config.get("pipeline", {})

        # ---- Scorer ----
        scorer_name: str = pipe_cfg.get("scorer_name", "weighted_sum")
        self.scorer: AbstractScorer = ScorerRegistry.create(scorer_name, config)

        # ---- Constraint engine ----
        self.constraint_engine = ConstraintEngine(config)

        # ---- Selector ----
        self.selector = TopKSelector(config)

        # ---- Reason generation ----
        self.enable_reasons: bool = pipe_cfg.get("enable_reasons", True)
        self.template_engine: Optional[TemplateEngine] = None
        if self.enable_reasons:
            self.template_engine = TemplateEngine(config)

        # ---- Reverse mapper ----
        self.enable_reverse_mapping: bool = pipe_cfg.get(
            "enable_reverse_mapping", False,
        )
        self.reverse_mapper: Optional[ReverseMapper] = None
        if self.enable_reverse_mapping:
            self.reverse_mapper = ReverseMapper(config)

        # ---- Self-checker ----
        self.enable_self_check: bool = pipe_cfg.get("enable_self_check", True)
        self.self_checker: Optional[SelfChecker] = None
        if self.enable_self_check:
            llm_provider: Optional[AbstractLLMProvider] = None
            sc_cfg = config.get("reason", {}).get("self_checker", {})
            if sc_cfg.get("enable_llm_check", False):
                try:
                    llm_provider = LLMProviderFactory.create(config)
                except Exception as exc:
                    logger.warning(
                        "LLM provider creation failed, LLM check disabled: %s",
                        exc,
                    )
            self.self_checker = SelfChecker(config, llm_provider=llm_provider)

        logger.info(
            "RecommendationPipeline initialised: scorer=%s, reasons=%s, "
            "self_check=%s, reverse_map=%s",
            scorer_name, self.enable_reasons, self.enable_self_check,
            self.enable_reverse_mapping,
        )

    # ------------------------------------------------------------------
    # Auto-configuration from FeatureGroupConfig
    # ------------------------------------------------------------------

    @classmethod
    def from_feature_groups(
        cls,
        groups: List["FeatureGroupConfig"],
        config: Dict[str, Any],
        template_pool: Optional[Dict[str, List[str]]] = None,
        task_frames: Optional[Dict[str, Dict[str, str]]] = None,
    ) -> "RecommendationPipeline":
        """Build a recommendation pipeline with auto-configured interpretation.

        The ``reverse_mapper`` and ``template_engine`` are auto-configured
        from feature group definitions (the single source of truth),
        ensuring that feature pipeline changes automatically propagate to
        the recommendation reason generation layer.

        Scoring, constraint engine, selector, and self-checker are still
        configured via the standard *config* dict.

        Args:
            groups: Ordered list of FeatureGroupConfig instances.
            config: Full pipeline config dict (for scorer, filters, selector,
                    self-checker, and LLM provider settings).
            template_pool: Optional additional templates for the template
                           engine.  Merged with auto-generated templates.
            task_frames: Optional task-specific narrative frames for the
                         template engine.

        Returns:
            A fully configured RecommendationPipeline with auto-wired
            interpretation components.
        """
        from core.feature.group_config import FeatureGroupConfig

        # Build the pipeline with standard config first
        instance = cls(config)

        # Override the reverse_mapper and template_engine with
        # auto-configured versions from FeatureGroupConfig
        pipe_cfg = config.get("pipeline", {})

        if pipe_cfg.get("enable_reverse_mapping", False):
            instance.reverse_mapper = ReverseMapper.from_feature_groups(groups)
            instance.enable_reverse_mapping = True

        if pipe_cfg.get("enable_reasons", True):
            instance.template_engine = TemplateEngine.from_feature_groups(
                groups,
                template_pool=template_pool,
                task_frames=task_frames,
            )
            instance.enable_reasons = True

        logger.info(
            "RecommendationPipeline.from_feature_groups: auto-configured "
            "reverse_mapper and template_engine from %d feature groups",
            len([g for g in groups if g.enabled]),
        )
        return instance

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def recommend(
        self,
        customer_id: str,
        candidate_items: List[Dict[str, Any]],
        customer_context: Optional[Dict[str, Any]] = None,
        item_features: Optional[Dict[str, np.ndarray]] = None,
    ) -> RecommendationResult:
        """Run the full pipeline for one customer.

        Args:
            customer_id: Customer identifier.
            candidate_items: List of candidate dicts, each containing at
                             least ``item_id`` and ``predictions``
                             (a ``{task: probability}`` dict).
                             May also contain ``item_info`` for reason
                             generation.
            customer_context: Shared context (fatigue, engagement, churn,
                              owned_products, channel, segment, ...).
            item_features: Optional mapping of ``item_id`` to feature
                           vectors for diversity computation.

        Returns:
            :class:`RecommendationResult` with ranked items and reasons.
        """
        t0 = time.perf_counter()
        ctx = customer_context or {}

        # ---- 1. Score ----
        scored: List[Dict[str, Any]] = []
        for cand in candidate_items:
            result: ScoringResult = self.scorer.score(
                customer_id=customer_id,
                item_id=cand["item_id"],
                predictions=cand.get("predictions", {}),
                context=ctx,
            )
            # Merge scored value into filter context so filters
            # (e.g. EligibilityFilter) can see the computed score.
            filter_ctx = {**ctx, "score": result.score}
            scored.append({
                "customer_id": customer_id,
                "item_id": cand["item_id"],
                "score": result.score,
                "score_components": result.components,
                "context": filter_ctx,
                "item_info": cand.get("item_info", {}),
                "ig_top_features": cand.get("ig_top_features", []),
            })

        total_candidates = len(scored)

        # ---- 2. Filter ----
        filtered = self.constraint_engine.apply_batch(scored)
        total_filtered = len(filtered)

        # ---- 3. Select top-K ----
        selected = self.selector.select(filtered, item_features=item_features)

        # ---- 4. Reasons + 5. Self-check ----
        items: List[RecommendationItem] = []
        segment = ctx.get("segment", "WARMSTART")
        task_type = ctx.get("task_type")

        for sel in selected:
            reasons: List[Dict[str, Any]] = []
            check: Optional[CheckResult] = None

            if self.template_engine and self.enable_reasons:
                reason_output = self.template_engine.generate_reason(
                    customer_id=customer_id,
                    item_id=sel["item_id"],
                    ig_top_features=sel.get("ig_top_features", []),
                    segment=segment,
                    task_type=task_type,
                    item_info=sel.get("item_info"),
                )
                reasons = reason_output.get("reasons", [])

                # Self-check the primary reason text
                if self.self_checker and self.enable_self_check and reasons:
                    primary_text = reasons[0].get("text", "")
                    check = self.self_checker.check(
                        reason_text=primary_text,
                        source_context=sel.get("item_info"),
                    )

            meta: Dict[str, Any] = {}
            if self.enable_reverse_mapping and self.reverse_mapper:
                ig_features = sel.get("ig_top_features", [])
                if ig_features:
                    importances = np.array([s for _, s in ig_features])
                    interpretations = self.reverse_mapper.interpret_top_k(
                        importances, k=3, task=task_type,
                    )
                    meta["feature_interpretations"] = [
                        {
                            "feature": interp.feature_name,
                            "group": interp.group_label,
                            "range": interp.range_label,
                            "interpretation": interp.interpretation,
                        }
                        for interp in interpretations
                    ]

            items.append(RecommendationItem(
                customer_id=customer_id,
                item_id=sel["item_id"],
                rank=sel.get("rank", 0),
                score=sel.get("score", 0.0),
                score_components=sel.get("score_components", {}),
                reasons=reasons,
                check_result=check,
                metadata=meta,
            ))

        elapsed = (time.perf_counter() - t0) * 1000.0

        result = RecommendationResult(
            customer_id=customer_id,
            items=items,
            total_candidates=total_candidates,
            total_filtered=total_filtered,
            elapsed_ms=round(elapsed, 2),
            metadata={
                "scorer": type(self.scorer).__name__,
                "diversity_method": self.selector.diversity_method.value,
                "k": self.selector.k,
            },
        )

        if self._audit_store and result:
            self._audit_store.log_event("recommendation", {
                "pk": result.customer_id,
                "customer_id": result.customer_id,
                "total_candidates": result.total_candidates,
                "total_filtered": result.total_filtered,
                "items_returned": len(result.items),
                "elapsed_ms": result.elapsed_ms,
            })

        return result

    def recommend_batch(
        self,
        requests: List[Dict[str, Any]],
    ) -> List[RecommendationResult]:
        """Run the pipeline for multiple customers.

        Each element of *requests* must contain:
        ``customer_id``, ``candidate_items``, and optionally
        ``customer_context`` and ``item_features``.

        Returns:
            List of :class:`RecommendationResult`, one per request.
        """
        results: List[RecommendationResult] = []
        for req in requests:
            result = self.recommend(
                customer_id=req["customer_id"],
                candidate_items=req["candidate_items"],
                customer_context=req.get("customer_context"),
                item_features=req.get("item_features"),
            )
            results.append(result)
        logger.info(
            "recommend_batch: processed %d customers", len(results),
        )
        return results
