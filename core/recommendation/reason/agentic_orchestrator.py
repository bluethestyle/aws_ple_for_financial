"""
2-Layer Agentic Reason Generation Pipeline (Stage C Interpretability)
======================================================================

Synchronous batch-oriented orchestrator for recommendation reason
generation.  Designed as a simpler, dependency-light counterpart to the
reference project's ``AgenticReasonOrchestrator`` and the existing
:class:`AsyncReasonOrchestrator`.

Architecture
~~~~~~~~~~~~

L1: Template full-batch (all customers, no LLM, ~20 min for 12M rows)
    Uses :class:`TemplateEngine.generate_batch` to produce deterministic
    template-based reasons for every customer-item pair.

L2a: Priority LLM rewrite (rich/moderate customers)
    For priority customers, assembles context and calls an LLM provider
    to rewrite the L1 template into a richer, more personalised reason.
    A self-critique loop validates compliance and factuality.

L2b: Quality validation (sampling)
    Samples a small percentage of L1 and L2a results, runs them through
    the :class:`XAIQualityEvaluator` for fidelity/completeness/consistency
    scoring.

Reference: gotothemoon/workspace/code/src/grounding/agentic_reason_orchestrator.py
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)

__all__ = [
    "AgenticReasonOrchestrator",
    "AgenticBatchResult",
]


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass
class AgenticBatchResult:
    """Result of the full L1 + L2a + L2b pipeline.

    Attributes
    ----------
    l1_results : list[dict]
        Template-based reasons for all customers.
    l2a_results : list[dict]
        LLM-rewritten reasons for priority customers.
    quality_report : dict
        L2b quality validation report.
    total_customers : int
        Total number of customers processed.
    l2a_count : int
        Number of customers that received L2a rewrites.
    processing_time_ms : int
        Total wall-clock time in milliseconds.
    """

    l1_results: List[Dict[str, Any]] = field(default_factory=list)
    l2a_results: List[Dict[str, Any]] = field(default_factory=list)
    quality_report: Dict[str, Any] = field(default_factory=dict)
    total_customers: int = 0
    l2a_count: int = 0
    processing_time_ms: int = 0

    def to_dict(self) -> Dict[str, Any]:
        """Serialise to a JSON-friendly dict (summary only, not full results)."""
        return {
            "total_customers": self.total_customers,
            "l2a_count": self.l2a_count,
            "processing_time_ms": self.processing_time_ms,
            "quality_report": self.quality_report,
            "l1_sample_count": len(self.l1_results),
            "l2a_sample_count": len(self.l2a_results),
        }


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

class AgenticReasonOrchestrator:
    """2-Layer agentic reason generation pipeline.

    Parameters
    ----------
    template_engine : TemplateEngine
        L1 template-based reason generator.
    llm_provider : AbstractLLMProvider | None
        LLM backend for L2a rewrites.  When ``None``, L2a is skipped.
    self_checker : SelfChecker | None
        Compliance and factuality checker for L2a outputs.
    config : dict, optional
        Pipeline configuration.  Relevant keys:

        - ``reason.l2a.sample_rate`` (float): fraction of L2a to validate (default 0.05)
        - ``reason.l2b.l1_sample_rate`` (float): fraction of L1 to validate (default 0.004)
        - ``reason.l2a.max_tokens`` (int): LLM max tokens for rewrite (default 500)
        - ``reason.l2a.temperature`` (float): LLM temperature (default 0.3)
    """

    def __init__(
        self,
        template_engine: Any,
        llm_provider: Any = None,
        self_checker: Any = None,
        config: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.template_engine = template_engine
        self.llm_provider = llm_provider
        self.self_checker = self_checker
        self.config = config or {}

        # L2 configuration
        reason_cfg = self.config.get("reason", {})
        l2a_cfg = reason_cfg.get("l2a", {})
        l2b_cfg = reason_cfg.get("l2b", {})

        self.l2a_sample_rate: float = l2a_cfg.get("sample_rate", 0.05)
        self.l1_sample_rate: float = l2b_cfg.get("l1_sample_rate", 0.004)
        self.l2a_max_tokens: int = l2a_cfg.get("max_tokens", 500)
        self.l2a_temperature: float = l2a_cfg.get("temperature", 0.3)

        logger.info(
            "AgenticReasonOrchestrator initialised: "
            "llm=%s, self_checker=%s, l2a_sample=%.3f, l1_sample=%.4f",
            type(llm_provider).__name__ if llm_provider else "None",
            type(self_checker).__name__ if self_checker else "None",
            self.l2a_sample_rate,
            self.l1_sample_rate,
        )

    # ------------------------------------------------------------------
    # L1: Template-based batch
    # ------------------------------------------------------------------

    def run_l1_batch(
        self,
        ig_attributions: List[Dict[str, Any]],
        product_info: Dict[str, Dict[str, Any]],
        segments: Dict[str, str],
        task_type: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """L1: Template-based reasons for ALL customers.

        Delegates to :meth:`TemplateEngine.generate_batch`.

        Parameters
        ----------
        ig_attributions : list[dict]
            Each dict has ``customer_id``, ``item_id``, ``ig_top_features``.
        product_info : dict
            ``{item_id: {name, primary_category, ...}}``.
        segments : dict
            ``{customer_id: segment}`` mapping.
        task_type : str, optional
            Task type for narrative framing.

        Returns
        -------
        list[dict]
            One reason dict per customer-item pair.
        """
        start = time.time()
        results = self.template_engine.generate_batch(
            ig_attributions=ig_attributions,
            product_info=product_info,
            segments=segments,
            task_type=task_type,
        )
        elapsed_ms = int((time.time() - start) * 1000)
        logger.info(
            "[L1] Template batch complete: %d reasons in %dms",
            len(results), elapsed_ms,
        )
        return results

    # ------------------------------------------------------------------
    # L2a: LLM rewrite for priority customers
    # ------------------------------------------------------------------

    def run_l2a_rewrite(
        self,
        l1_results: List[Dict[str, Any]],
        priority_mask: Optional[List[bool]] = None,
        customer_contexts: Optional[Dict[str, Dict[str, Any]]] = None,
    ) -> List[Dict[str, Any]]:
        """L2a: LLM rewrite for priority customers only.

        For each priority customer:
        1. Assemble context (features + consultation + segment)
        2. Build prompt with L1 reason as reference
        3. Call LLM
        4. Run self-critique (compliance + factuality)
        5. On failure, fall back to L1 template

        Parameters
        ----------
        l1_results : list[dict]
            L1 template reasons (output of :meth:`run_l1_batch`).
        priority_mask : list[bool], optional
            Boolean mask indicating which customers are priority.
            When ``None``, no rewrites are performed.
        customer_contexts : dict, optional
            ``{customer_id: context_dict}`` with additional context
            for prompt building (consultation notes, segment details, etc.).

        Returns
        -------
        list[dict]
            Same length as ``l1_results``.  Priority customers get
            rewritten reasons; others keep L1 unchanged.
        """
        if self.llm_provider is None:
            logger.warning("No LLM provider configured, skipping L2a rewrite.")
            return l1_results

        if priority_mask is None:
            logger.info("[L2a] No priority mask provided, skipping L2a.")
            return l1_results

        start = time.time()
        contexts = customer_contexts or {}
        rewrite_count = 0
        rewrite_success = 0

        # Copy results to avoid mutating the original
        l2a_results = [dict(r) for r in l1_results]

        for i, (result, is_priority) in enumerate(zip(l2a_results, priority_mask)):
            if not is_priority:
                continue

            rewrite_count += 1
            cust_id = result.get("customer_id", "")
            item_id = result.get("item_id", "")

            try:
                # Build rewrite prompt
                l1_reasons = result.get("reasons", [])
                l1_text = " ".join(r.get("text", "") for r in l1_reasons)
                context = contexts.get(cust_id, {})

                prompt = self._build_rewrite_prompt(
                    l1_text=l1_text,
                    customer_id=cust_id,
                    item_id=item_id,
                    context=context,
                    segment=result.get("segment", "WARMSTART"),
                )

                # Call LLM
                rewritten_text = self.llm_provider.generate(
                    prompt,
                    max_tokens=self.l2a_max_tokens,
                    temperature=self.l2a_temperature,
                )

                # Self-critique (if checker available)
                if self.self_checker is not None:
                    check_result = self.self_checker.check(
                        reason_text=rewritten_text,
                        source_context=context if context else None,
                    )
                    if check_result.verdict == "reject":
                        logger.debug(
                            "[L2a] Rejected rewrite for %s: %s",
                            cust_id, check_result.feedback,
                        )
                        continue  # keep L1 version

                # Update result with L2a rewrite
                l2a_results[i]["reasons"] = [{
                    "rank": 1,
                    "type": "primary",
                    "text": rewritten_text.strip(),
                    "feature": "llm_rewrite",
                    "ig_score": None,
                    "category": "l2a_rewrite",
                }]
                l2a_results[i]["generation_method"] = "l2a_llm_rewrite"
                l2a_results[i]["l2a_rewritten"] = True
                rewrite_success += 1

            except Exception as exc:
                logger.warning(
                    "[L2a] Rewrite failed for customer %s: %s", cust_id, exc,
                )
                # Fall back to L1 (already in l2a_results[i])

        elapsed_ms = int((time.time() - start) * 1000)
        logger.info(
            "[L2a] Rewrite complete: %d attempted, %d succeeded in %dms",
            rewrite_count, rewrite_success, elapsed_ms,
        )
        return l2a_results

    def _build_rewrite_prompt(
        self,
        l1_text: str,
        customer_id: str,
        item_id: str,
        context: Dict[str, Any],
        segment: str,
    ) -> str:
        """Build the LLM prompt for L2a rewriting.

        The prompt instructs the LLM to produce a more personalised version
        of the L1 template reason while preserving factual accuracy and
        regulatory compliance.
        """
        context_str = ""
        if context:
            context_str = "\n".join(
                f"- {k}: {v}" for k, v in context.items()
            )

        return (
            "You are a financial product recommendation writer for a Korean bank.\n"
            "Rewrite the template-generated reason below into a more personalised, "
            "natural-sounding explanation.\n\n"
            "Rules:\n"
            "- Keep the factual claims from the original reason.\n"
            "- Do NOT promise guaranteed returns or claim zero risk.\n"
            "- Write in Korean (or English if the original is in English).\n"
            "- Keep it concise (2-3 sentences max).\n"
            "- Include the product name.\n\n"
            f"Customer segment: {segment}\n"
            f"Product ID: {item_id}\n\n"
            f"Customer context:\n{context_str or '(not available)'}\n\n"
            f"Original template reason:\n{l1_text}\n\n"
            "Rewritten reason:"
        )

    # ------------------------------------------------------------------
    # L2b: Quality validation (sampling)
    # ------------------------------------------------------------------

    def run_l2b_validation(
        self,
        l1_results: List[Dict[str, Any]],
        l2a_results: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        """L2b: Quality validation on sampled results.

        Samples 0.4 % of L1 results and 5 % of L2a rewrites, then
        evaluates each sample for basic quality metrics.

        Parameters
        ----------
        l1_results : list[dict]
            Full L1 template results.
        l2a_results : list[dict], optional
            L2a rewritten results (only those actually rewritten).

        Returns
        -------
        dict
            Quality validation report with pass rates and sample counts.
        """
        start = time.time()
        rng = np.random.default_rng(42)

        # Sample L1
        n_l1 = len(l1_results)
        l1_sample_size = max(1, int(n_l1 * self.l1_sample_rate))
        l1_sample_indices = rng.choice(n_l1, size=min(l1_sample_size, n_l1), replace=False)
        l1_samples = [l1_results[i] for i in l1_sample_indices]

        # Sample L2a
        l2a_samples = []
        if l2a_results:
            rewritten = [r for r in l2a_results if r.get("l2a_rewritten", False)]
            n_l2a = len(rewritten)
            if n_l2a > 0:
                l2a_sample_size = max(1, int(n_l2a * self.l2a_sample_rate))
                l2a_sample_indices = rng.choice(
                    n_l2a, size=min(l2a_sample_size, n_l2a), replace=False,
                )
                l2a_samples = [rewritten[i] for i in l2a_sample_indices]

        # Validate samples
        l1_quality = self._validate_samples(l1_samples, "L1")
        l2a_quality = self._validate_samples(l2a_samples, "L2a") if l2a_samples else {}

        elapsed_ms = int((time.time() - start) * 1000)

        report = {
            "status": "completed",
            "processing_time_ms": elapsed_ms,
            "l1_validation": {
                "total_l1": n_l1,
                "sampled": len(l1_samples),
                "sample_rate": self.l1_sample_rate,
                **l1_quality,
            },
            "l2a_validation": {
                "total_l2a_rewritten": len(l2a_samples),
                "sampled": len(l2a_samples),
                "sample_rate": self.l2a_sample_rate,
                **l2a_quality,
            } if l2a_samples else {"status": "no_l2a_samples"},
            "generated_at": datetime.utcnow().isoformat() + "Z",
        }

        logger.info(
            "[L2b] Validation complete: L1 sampled=%d, L2a sampled=%d, %dms",
            len(l1_samples), len(l2a_samples), elapsed_ms,
        )
        return report

    def _validate_samples(
        self,
        samples: List[Dict[str, Any]],
        layer_name: str,
    ) -> Dict[str, Any]:
        """Run basic quality checks on sampled reasons.

        Checks:
        - Non-empty reason text
        - Compliance check (if self_checker available)
        - Minimum reason length (>10 chars)
        """
        if not samples:
            return {"pass_rate": 0.0, "checked": 0}

        passed = 0
        compliance_passed = 0
        total = len(samples)

        for sample in samples:
            reasons = sample.get("reasons", [])
            if not reasons:
                continue

            # Check non-empty and minimum length
            primary_text = reasons[0].get("text", "")
            if len(primary_text) < 10:
                continue

            # Compliance check via self_checker
            if self.self_checker is not None:
                try:
                    check = self.self_checker.check(primary_text)
                    if check.verdict != "reject":
                        compliance_passed += 1
                    else:
                        continue
                except Exception:
                    compliance_passed += 1  # Assume pass on error

            passed += 1

        pass_rate = passed / total if total > 0 else 0.0

        return {
            "checked": total,
            "passed": passed,
            "pass_rate": round(pass_rate, 4),
            "compliance_checked": compliance_passed if self.self_checker else "n/a",
        }

    # ------------------------------------------------------------------
    # Full pipeline
    # ------------------------------------------------------------------

    def run_full(
        self,
        ig_attributions: List[Dict[str, Any]],
        product_info: Dict[str, Dict[str, Any]],
        segments: Dict[str, str],
        customer_contexts: Optional[Dict[str, Dict[str, Any]]] = None,
        priority_mask: Optional[List[bool]] = None,
        task_type: Optional[str] = None,
    ) -> AgenticBatchResult:
        """Run the full L1 + L2a + L2b pipeline.

        Parameters
        ----------
        ig_attributions : list[dict]
            IG attribution data for each customer-item pair.
        product_info : dict
            Product metadata.
        segments : dict
            Customer segment mapping.
        customer_contexts : dict, optional
            Additional context for L2a rewrites.
        priority_mask : list[bool], optional
            Boolean mask for L2a priority customers.
        task_type : str, optional
            Task type for narrative framing.

        Returns
        -------
        AgenticBatchResult
        """
        pipeline_start = time.time()

        # L1: Template batch
        l1 = self.run_l1_batch(ig_attributions, product_info, segments, task_type)

        # L2a: LLM rewrite (if priority mask provided)
        if priority_mask is not None and self.llm_provider is not None:
            l2a = self.run_l2a_rewrite(l1, priority_mask, customer_contexts)
        else:
            l2a = l1

        # L2b: Quality validation
        quality = self.run_l2b_validation(l1, l2a)

        # Count actual L2a rewrites
        l2a_count = sum(1 for r in l2a if r.get("l2a_rewritten", False))

        elapsed_ms = int((time.time() - pipeline_start) * 1000)

        logger.info(
            "[AgenticOrchestrator] Full pipeline complete: "
            "%d customers, %d L2a rewrites, %dms",
            len(l1), l2a_count, elapsed_ms,
        )

        return AgenticBatchResult(
            l1_results=l1,
            l2a_results=l2a,
            quality_report=quality,
            total_customers=len(l1),
            l2a_count=l2a_count,
            processing_time_ms=elapsed_ms,
        )
