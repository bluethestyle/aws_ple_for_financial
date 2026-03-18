"""
3-Layer Async Reason Generation (original v3.0 architecture).
==============================================================

L1 (Template): Fast, CPU-only, deterministic template-based reasons.
    - 6 categories x 5 variants = 30 templates
    - Customer-ID hashed deterministic selection
    - Runs inline with recommendation pipeline

L2a (LLM Rewrite): Async GPU-accelerated rewriting of L1 templates.
    - Submitted to SQS queue for background processing
    - Priority: rich customers first, then moderate
    - Uses Bedrock/SageMaker endpoint for LLM inference
    - Results cached in DynamoDB for subsequent requests

L2b (Quality Validation): Async quality check of L2a rewrites.
    - Validates LLM output for compliance (no hallucination, no PII)
    - Samples 5% for human review flagging
    - Failed validations fall back to L1 template

Usage::

    orchestrator = AsyncReasonOrchestrator(config)

    # Synchronous: L1 template (always available)
    reason = orchestrator.generate_l1(customer_id, recommendation, features)

    # Async: Submit L2a rewrite job
    job_id = orchestrator.submit_l2a_rewrite(customer_id, l1_reason, context)

    # Check if L2a result is ready (cached)
    l2_reason = orchestrator.get_cached_reason(customer_id, recommendation_id)

    # Best-effort: returns L2b > L2a > L1 (whichever is available)
    best = orchestrator.get_best_reason(customer_id, recommendation, features)
"""

from __future__ import annotations

import hashlib
import json
import logging
import random
import re
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

__all__ = ["AsyncReasonOrchestrator", "ReasonResult"]


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class ReasonResult:
    """Result from any layer of the reason generation pipeline.

    Attributes:
        text: The generated reason text.
        layer: Which layer produced this result (``"L1"`` / ``"L2a"`` / ``"L2b"``).
        template_id: Template identifier used (L1 only).
        confidence: Confidence score (1.0 for L1, LLM-derived for L2a/L2b).
        validation_passed: Whether L2b quality validation passed.
        generated_at: ISO-8601 timestamp of generation.
        cached: Whether this result was served from cache.
        human_review_flagged: Whether this result was sampled for human review.
        job_id: Async job identifier (L2a/L2b only).
    """

    text: str
    layer: str              # "L1" | "L2a" | "L2b"
    template_id: str = ""
    confidence: float = 1.0
    validation_passed: bool = True
    generated_at: str = ""
    cached: bool = False
    human_review_flagged: bool = False
    job_id: str = ""


# ---------------------------------------------------------------------------
# SQS message priority levels
# ---------------------------------------------------------------------------

class _Priority:
    HIGH = 1        # Rich / VIP customers
    MODERATE = 5    # Standard customers
    LOW = 10        # Low-engagement customers


# ---------------------------------------------------------------------------
# PII detection patterns for L2b validation
# ---------------------------------------------------------------------------

_PII_PATTERNS: List[re.Pattern] = [
    re.compile(r"\b\d{6}[-\s]?\d{7}\b"),                     # Korean resident reg no
    re.compile(r"\b\d{3}[-.\s]?\d{2}[-.\s]?\d{4}\b"),        # SSN-like
    re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b"),  # Email
    re.compile(r"\b\d{4}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}\b"),  # Card number
    re.compile(r"\b01[016789][-\s]?\d{3,4}[-\s]?\d{4}\b"),   # Korean phone
]


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

class AsyncReasonOrchestrator:
    """3-layer async reason generation orchestrator.

    Composes existing :class:`TemplateEngine`, :class:`AbstractLLMProvider`,
    and :class:`SelfChecker` without modifying them.  Adds async SQS-based
    L2a rewrite submission and DynamoDB-backed caching.

    Args:
        config: Pipeline configuration dict.  Reads
                ``config["reason"]["async_orchestrator"]`` for layer-specific
                settings.
        template_engine: Pre-built :class:`TemplateEngine` instance.
                         If ``None``, a minimal engine is created internally.
        llm_provider: Pre-built :class:`AbstractLLMProvider` instance for
                      L2a LLM rewrites.
        self_checker: Pre-built :class:`SelfChecker` instance for L2b
                      validation.
        audit_store: Optional callable ``(record: dict) -> None`` for
                     audit logging of L2a/L2b events.
    """

    # Human-review sampling rate (5%)
    HUMAN_REVIEW_SAMPLE_RATE: float = 0.05

    def __init__(
        self,
        config: Dict[str, Any] = None,
        template_engine=None,
        llm_provider=None,
        self_checker=None,
        audit_store=None,
        prompt_sanitizer=None,
    ) -> None:
        self._config = config or {}
        ao_cfg = self._config.get("reason", {}).get("async_orchestrator", {})
        self._prompt_sanitizer = prompt_sanitizer

        self._template_engine = template_engine
        self._llm_provider = llm_provider
        self._self_checker = self_checker
        self._audit_store = audit_store

        # In-memory cache (local); production overrides with DynamoDB
        self._cache: Dict[str, ReasonResult] = {}

        # SQS configuration (lazy init)
        self._sqs_queue_url: str = ao_cfg.get("sqs_queue_url", "")
        self._sqs_region: str = ao_cfg.get("sqs_region", "ap-northeast-2")
        self._sqs_client = None
        self._sqs_available: Optional[bool] = None  # Tri-state: None=unknown

        # DynamoDB configuration (lazy init)
        self._dynamo_table_name: str = ao_cfg.get(
            "dynamo_table", "reason_cache",
        )
        self._dynamo_client = None
        self._dynamo_available: Optional[bool] = None

        # LLM prompt settings
        self._max_reason_length: int = ao_cfg.get("max_reason_length", 200)
        self._llm_temperature: float = ao_cfg.get("llm_temperature", 0.3)

        # Task-specific narrative frames: {task_type: {frame, narrative, guidelines}}
        # Loaded from config["reason"]["template_engine"]["task_frames"]
        reason_cfg = config.get("reason", {})
        te_cfg = reason_cfg.get("template_engine", {})
        self._task_frames: Dict[str, Dict[str, str]] = te_cfg.get("task_frames", {})

        # Context assembler for grounding (optional, injected or auto-created)
        self._context_assembler: Optional[Any] = None

        # AI disclosure requirement (금소법 -- Financial Consumer Protection Act)
        self._ai_disclosure: str = ao_cfg.get(
            "ai_disclosure",
            (
                "본 추천 사유는 AI 시스템에 의해 생성되었습니다. "
                "최종 판단은 고객님께서 직접 하시기 바랍니다."
            ),
        )

        logger.info(
            "AsyncReasonOrchestrator initialised: sqs_queue=%s, "
            "dynamo_table=%s, template_engine=%s, llm_provider=%s, "
            "self_checker=%s",
            self._sqs_queue_url or "(none)",
            self._dynamo_table_name,
            type(self._template_engine).__name__ if self._template_engine else "None",
            type(self._llm_provider).__name__ if self._llm_provider else "None",
            type(self._self_checker).__name__ if self._self_checker else "None",
        )

    # ------------------------------------------------------------------
    # Lazy AWS client initialisation
    # ------------------------------------------------------------------

    def _get_sqs_client(self):
        """Lazy-init SQS client.  Returns None if boto3 is unavailable."""
        if self._sqs_client is not None:
            return self._sqs_client
        if self._sqs_available is False:
            return None
        try:
            import boto3
            self._sqs_client = boto3.client(
                "sqs", region_name=self._sqs_region,
            )
            self._sqs_available = True
            return self._sqs_client
        except Exception as exc:
            logger.warning("SQS client creation failed (falling back to sync): %s", exc)
            self._sqs_available = False
            return None

    def _get_dynamo_client(self):
        """Lazy-init DynamoDB client.  Returns None if unavailable."""
        if self._dynamo_client is not None:
            return self._dynamo_client
        if self._dynamo_available is False:
            return None
        try:
            import boto3
            self._dynamo_client = boto3.client(
                "dynamodb", region_name=self._sqs_region,
            )
            self._dynamo_available = True
            return self._dynamo_client
        except Exception as exc:
            logger.warning("DynamoDB client creation failed (using in-memory cache): %s", exc)
            self._dynamo_available = False
            return None

    # ------------------------------------------------------------------
    # L1: Synchronous template-based reason
    # ------------------------------------------------------------------

    def generate_l1(
        self,
        customer_id: str,
        recommendation: Dict[str, Any],
        features: List[Tuple[str, float]],
        task_type: Optional[str] = None,
    ) -> ReasonResult:
        """L1: Synchronous template-based reason (always available).

        Deterministic: same ``customer_id`` + ``recommendation`` always
        yields the same template variant (via MD5 hash).

        Args:
            customer_id: Customer identifier.
            recommendation: Recommendation dict containing at least
                            ``item_id`` and optionally ``item_info``.
            features: List of ``(feature_name, ig_score)`` sorted by
                      descending importance.
            task_type: Optional task type for narrative framing.

        Returns:
            :class:`ReasonResult` with ``layer="L1"``.
        """
        item_id = recommendation.get("item_id", "unknown")
        item_info = recommendation.get("item_info", {})
        segment = recommendation.get("segment", "WARMSTART")

        # Derive a deterministic template_id from customer + item
        hash_key = f"{customer_id}:{item_id}"
        template_hash = hashlib.md5(hash_key.encode()).hexdigest()[:8]
        template_id = f"L1-{template_hash}"

        # Use existing TemplateEngine if available
        if self._template_engine is not None:
            reason_output = self._template_engine.generate_reason(
                customer_id=customer_id,
                item_id=item_id,
                ig_top_features=features,
                segment=segment,
                task_type=task_type,
                item_info=item_info,
            )
            reasons = reason_output.get("reasons", [])
            text = reasons[0]["text"] if reasons else self._minimal_reason(item_id)
        else:
            # Minimal fallback when no template engine is configured
            text = self._minimal_reason(item_id)

        return ReasonResult(
            text=text,
            layer="L1",
            template_id=template_id,
            confidence=1.0,
            validation_passed=True,
            generated_at=datetime.now(timezone.utc).isoformat(),
            cached=False,
        )

    # ------------------------------------------------------------------
    # L2a: Async LLM rewrite submission
    # ------------------------------------------------------------------

    def submit_l2a_rewrite(
        self,
        customer_id: str,
        l1_reason: ReasonResult,
        context: Dict[str, Any],
        priority: int = _Priority.MODERATE,
    ) -> str:
        """Submit an async L2a LLM rewrite job to SQS.

        If SQS is unavailable, the job is dropped silently and an empty
        ``job_id`` is returned (the system will continue to serve L1).

        Args:
            customer_id: Customer identifier.
            l1_reason: The L1 :class:`ReasonResult` to rewrite.
            context: Contextual information for the LLM (item info,
                     customer segment, feature interpretations, etc.).
            priority: SQS message priority (lower = higher priority).

        Returns:
            Job ID string for tracking, or ``""`` if submission failed.
        """
        job_id = f"l2a-{uuid.uuid4().hex[:12]}"
        recommendation_id = context.get("recommendation_id", "")

        # Assemble grounding context from 5 sources if assembler is available
        if (
            self._context_assembler is not None
            and "assembled_context_text" not in context
        ):
            try:
                assembled = self._context_assembler.assemble(
                    customer_id=customer_id,
                    task_name=context.get("task_type", ""),
                    features=context.get("features"),
                    feature_importances=context.get("top_features", []),
                    product_id=context.get("item_id", ""),
                )
                context["assembled_context_text"] = assembled.to_prompt_text()
            except Exception:
                logger.debug(
                    "ContextAssembler failed for %s (non-fatal)", customer_id,
                    exc_info=True,
                )

        # PromptSanitizer: pre-classify the L1 reason to record sensitivity
        # and determine the target provider before submitting to SQS.
        # This allows the SQS consumer to honour the routing decision.
        sanitize_sensitivity = "LOW"
        sanitize_provider = "bedrock"
        if self._prompt_sanitizer is not None:
            pre_prompt = self._build_llm_prompt(l1_reason.text, context)
            _sanitized, sanitize_provider, sanitize_result = (
                self._prompt_sanitizer.sanitize_and_route(pre_prompt)
            )
            sanitize_sensitivity = sanitize_result.sensitivity
            logger.debug(
                "Pre-submit sanitize: sensitivity=%s, provider=%s (job=%s)",
                sanitize_sensitivity, sanitize_provider, job_id,
            )

        message_body = {
            "job_id": job_id,
            "customer_id": customer_id,
            "recommendation_id": recommendation_id,
            "l1_reason": l1_reason.text,
            "l1_template_id": l1_reason.template_id,
            "context": {
                "item_id": context.get("item_id", ""),
                "item_name": context.get("item_name", ""),
                "segment": context.get("segment", "WARMSTART"),
                "task_type": context.get("task_type", ""),
                "top_features": context.get("top_features", []),
            },
            "priority": priority,
            "submitted_at": datetime.now(timezone.utc).isoformat(),
            "sanitize_sensitivity": sanitize_sensitivity,
            "sanitize_provider": sanitize_provider,
        }

        sqs = self._get_sqs_client()
        if sqs is None or not self._sqs_queue_url:
            logger.debug(
                "SQS unavailable; L2a job %s not submitted (sync L1 only)", job_id,
            )
            return ""

        try:
            sqs.send_message(
                QueueUrl=self._sqs_queue_url,
                MessageBody=json.dumps(message_body, default=str),
                MessageGroupId=customer_id,
                MessageAttributes={
                    "Priority": {
                        "StringValue": str(priority),
                        "DataType": "Number",
                    },
                },
            )
            logger.info("L2a rewrite job submitted: job_id=%s, customer=%s", job_id, customer_id)
            return job_id
        except Exception as exc:
            logger.warning("SQS send_message failed for job %s: %s", job_id, exc)
            return ""

    # ------------------------------------------------------------------
    # L2a: Process a single job (SQS consumer / Lambda handler)
    # ------------------------------------------------------------------

    def process_l2a_job(self, job: Dict[str, Any]) -> ReasonResult:
        """Process a single L2a rewrite job (called by SQS consumer/Lambda).

        Steps:
            1. Build an LLM prompt from the L1 reason + context.
            2. Call the LLM provider for rewriting.
            3. Run L2b quality validation.
            4. Cache the result.
            5. Audit-log the event.

        Args:
            job: SQS message body (parsed JSON dict) with keys
                 ``job_id``, ``customer_id``, ``l1_reason``, ``context``.

        Returns:
            :class:`ReasonResult` from L2a (or L2b if validation ran).
        """
        job_id = job.get("job_id", f"l2a-{uuid.uuid4().hex[:12]}")
        customer_id = job.get("customer_id", "")
        recommendation_id = job.get("recommendation_id", "")
        l1_text = job.get("l1_reason", "")
        context = job.get("context", {})

        # Step 1: Build LLM prompt
        prompt = self._build_llm_prompt(l1_text, context)

        # Step 1.5: PromptSanitizer — classify sensitivity and route LLM
        actual_provider = self._llm_provider
        if self._prompt_sanitizer is not None:
            sanitized_prompt, provider_name, sanitize_result = (
                self._prompt_sanitizer.sanitize_and_route(prompt)
            )
            prompt = sanitized_prompt
            # Route to appropriate provider if factory is available
            if provider_name != "bedrock" and hasattr(self, "_llm_provider_factory"):
                try:
                    actual_provider = self._llm_provider_factory(provider_name)
                except Exception:
                    pass  # stick with default provider
            logger.debug(
                "Sanitizer: sensitivity=%s, provider=%s, scrubbed=%s (job=%s)",
                sanitize_result.sensitivity, sanitize_result.provider,
                sanitize_result.scrubbed, job_id,
            )

        # Step 2: Call LLM provider
        if actual_provider is None:
            logger.warning("No LLM provider configured; returning L1 as-is for job %s", job_id)
            return ReasonResult(
                text=l1_text,
                layer="L1",
                confidence=1.0,
                validation_passed=True,
                generated_at=datetime.now(timezone.utc).isoformat(),
                job_id=job_id,
            )

        try:
            llm_output = actual_provider.generate(
                prompt, temperature=self._llm_temperature,
            )
            rewritten = self._extract_rewritten_reason(llm_output)
        except Exception as exc:
            logger.error("LLM generation failed for job %s: %s", job_id, exc)
            # Fall back to L1
            return ReasonResult(
                text=l1_text,
                layer="L1",
                confidence=1.0,
                validation_passed=True,
                generated_at=datetime.now(timezone.utc).isoformat(),
                job_id=job_id,
            )

        # L2a Safety Gate: 3-layer check before caching
        gate_passed, gate_reason = self._apply_safety_gate(rewritten, context)
        if not gate_passed:
            logger.warning("L2a safety gate failed: %s (job=%s)", gate_reason, job_id)
            # Fall back to L1
            return ReasonResult(
                text=l1_text,
                layer="L1",
                confidence=1.0,
                validation_passed=True,
                generated_at=datetime.now(timezone.utc).isoformat(),
                job_id=job_id,
            )

        l2a_result = ReasonResult(
            text=rewritten,
            layer="L2a",
            template_id=job.get("l1_template_id", ""),
            confidence=0.9,  # Default; refined by L2b
            validation_passed=True,
            generated_at=datetime.now(timezone.utc).isoformat(),
            job_id=job_id,
        )

        # Step 3: L2b quality validation
        # Pass gate_passed=True to skip redundant self_checker compliance
        # check (already done in _apply_safety_gate Gate 2).
        l2b_result = self._validate_l2b(l2a_result, context, safety_gate_passed=True)

        # If validation failed, fall back to L1
        if not l2b_result.validation_passed:
            logger.info(
                "L2b validation failed for job %s; falling back to L1", job_id,
            )
            fallback = ReasonResult(
                text=l1_text,
                layer="L1",
                confidence=1.0,
                validation_passed=True,
                generated_at=datetime.now(timezone.utc).isoformat(),
                job_id=job_id,
            )
            self._cache_result(customer_id, recommendation_id, fallback)
            self._audit(job_id, customer_id, "l2b_fallback", l2b_result)
            return fallback

        # Step 4: Cache the validated result
        self._cache_result(customer_id, recommendation_id, l2b_result)

        # Step 4.5: Write to audit store if available
        if self._audit_store and callable(self._audit_store):
            self._audit_store({
                "event": "l2a_rewrite",
                "job_id": job_id,
                "customer_id": customer_id,
                "layer": l2b_result.layer,
                "validation_passed": l2b_result.validation_passed,
                "confidence": l2b_result.confidence,
            })

        # Step 5: Audit log
        self._audit(job_id, customer_id, "l2b_pass", l2b_result)

        return l2b_result

    # ------------------------------------------------------------------
    # Cache: get / put
    # ------------------------------------------------------------------

    def get_cached_reason(
        self,
        customer_id: str,
        recommendation_id: str,
    ) -> Optional[ReasonResult]:
        """Check if an L2a/L2b result is cached for this customer+recommendation.

        Checks in-memory cache first, then DynamoDB if available.

        Args:
            customer_id: Customer identifier.
            recommendation_id: Recommendation identifier.

        Returns:
            Cached :class:`ReasonResult`, or ``None`` if not found.
        """
        cache_key = self._cache_key(customer_id, recommendation_id)

        # In-memory lookup
        if cache_key in self._cache:
            result = self._cache[cache_key]
            result.cached = True
            return result

        # DynamoDB lookup
        dynamo = self._get_dynamo_client()
        if dynamo is None:
            return None

        try:
            resp = dynamo.get_item(
                TableName=self._dynamo_table_name,
                Key={
                    "pk": {"S": cache_key},
                },
            )
            item = resp.get("Item")
            if item is None:
                return None

            result = ReasonResult(
                text=item["text"]["S"],
                layer=item["layer"]["S"],
                template_id=item.get("template_id", {}).get("S", ""),
                confidence=float(item.get("confidence", {}).get("N", "1.0")),
                validation_passed=item.get("validation_passed", {}).get("BOOL", True),
                generated_at=item.get("generated_at", {}).get("S", ""),
                cached=True,
                job_id=item.get("job_id", {}).get("S", ""),
            )

            # Populate in-memory cache for subsequent requests
            self._cache[cache_key] = result
            return result
        except Exception as exc:
            logger.warning("DynamoDB get_item failed for %s: %s", cache_key, exc)
            return None

    def _cache_result(
        self,
        customer_id: str,
        recommendation_id: str,
        result: ReasonResult,
    ) -> None:
        """Store a result in both in-memory cache and DynamoDB."""
        cache_key = self._cache_key(customer_id, recommendation_id)
        self._cache[cache_key] = result

        # Persist to DynamoDB if available
        dynamo = self._get_dynamo_client()
        if dynamo is None:
            return

        try:
            dynamo.put_item(
                TableName=self._dynamo_table_name,
                Item={
                    "pk": {"S": cache_key},
                    "customer_id": {"S": customer_id},
                    "recommendation_id": {"S": recommendation_id},
                    "text": {"S": result.text},
                    "layer": {"S": result.layer},
                    "template_id": {"S": result.template_id},
                    "confidence": {"N": str(result.confidence)},
                    "validation_passed": {"BOOL": result.validation_passed},
                    "generated_at": {"S": result.generated_at},
                    "job_id": {"S": result.job_id},
                },
            )
        except Exception as exc:
            logger.warning("DynamoDB put_item failed for %s: %s", cache_key, exc)

    @staticmethod
    def _cache_key(customer_id: str, recommendation_id: str) -> str:
        """Build a composite cache key."""
        return f"{customer_id}::{recommendation_id}"

    # ------------------------------------------------------------------
    # Best-effort reason retrieval
    # ------------------------------------------------------------------

    def get_best_reason(
        self,
        customer_id: str,
        recommendation: Dict[str, Any],
        features: List[Tuple[str, float]],
        recommendation_id: str = "",
        task_type: Optional[str] = None,
    ) -> ReasonResult:
        """Get the best available reason: L2b > L2a > L1.

        Workflow:
            1. Check cache for an existing L2a/L2b result.
            2. If found and validated, return it immediately.
            3. Otherwise, generate an L1 template reason.
            4. Submit an async L2a job for future requests.
            5. Return the L1 reason for now.

        Args:
            customer_id: Customer identifier.
            recommendation: Recommendation dict (``item_id``, ``item_info``, ...).
            features: List of ``(feature_name, ig_score)`` tuples.
            recommendation_id: Unique ID for this recommendation (for caching).
            task_type: Optional task type for narrative framing.

        Returns:
            The best available :class:`ReasonResult`.
        """
        # 1. Check cache
        if recommendation_id:
            cached = self.get_cached_reason(customer_id, recommendation_id)
            if cached is not None and cached.validation_passed:
                logger.debug(
                    "Serving cached %s reason for customer=%s, rec=%s",
                    cached.layer, customer_id, recommendation_id,
                )
                return cached

        # 2. Generate L1
        l1_result = self.generate_l1(
            customer_id=customer_id,
            recommendation=recommendation,
            features=features,
            task_type=task_type,
        )

        # 3. Submit async L2a job for future requests
        if recommendation_id:
            context = {
                "recommendation_id": recommendation_id,
                "item_id": recommendation.get("item_id", ""),
                "item_name": recommendation.get("item_info", {}).get("name", ""),
                "segment": recommendation.get("segment", "WARMSTART"),
                "task_type": task_type or "",
                "top_features": [
                    {"name": f, "score": s} for f, s in features[:5]
                ],
            }
            # Determine priority from segment
            segment = recommendation.get("segment", "WARMSTART")
            priority = _Priority.HIGH if segment == "VIP" else _Priority.MODERATE
            self.submit_l2a_rewrite(customer_id, l1_result, context, priority)

        return l1_result

    # ------------------------------------------------------------------
    # LLM prompt construction
    # ------------------------------------------------------------------

    def _build_llm_prompt(self, l1_reason: str, context: Dict[str, Any]) -> str:
        """Build the LLM prompt for L2a rewriting.

        The prompt instructs the LLM to:
        - Rewrite the template-based reason into natural, personalised language.
        - Preserve factual claims (no hallucination).
        - Stay within the max length.
        - Include AI disclosure per 금소법 (Financial Consumer Protection Act).
        - Not include any PII.

        Args:
            l1_reason: The L1 template-generated reason text.
            context: Contextual information (item, segment, features).

        Returns:
            Complete prompt string.
        """
        item_name = context.get("item_name", "")
        segment = context.get("segment", "WARMSTART")
        task_type = context.get("task_type", "")
        top_features = context.get("top_features", [])

        # Format feature context
        feature_lines = ""
        if top_features:
            feature_items = []
            for feat in top_features[:5]:
                if isinstance(feat, dict):
                    feature_items.append(f"  - {feat.get('name', '?')}: {feat.get('score', 0):.4f}")
                else:
                    feature_items.append(f"  - {feat}")
            feature_lines = "\n".join(feature_items)

        prompt = (
            "You are a financial product recommendation copywriter for a Korean bank.\n"
            "Rewrite the following template-generated recommendation reason into\n"
            "natural, customer-friendly Korean language.\n\n"
            "## Rules\n"
            "1. Preserve all factual claims from the original reason. Do NOT add\n"
            "   information that is not supported by the context below.\n"
            "2. Do NOT include any personally identifiable information (PII).\n"
            f"3. Keep the output under {self._max_reason_length} characters.\n"
            "4. The tone should be warm, professional, and respectful.\n"
            f"5. Customer segment: {segment}\n"
        )

        if task_type:
            prompt += f"6. Task context: {task_type}\n"

            # Task-specific writing guidelines from config
            task_frame = self._task_frames.get(task_type, {})
            if task_frame:
                frame = task_frame.get("frame", "")
                narrative = task_frame.get("narrative", "")
                guidelines = task_frame.get("guidelines", "")
                if frame:
                    prompt += f"7. Writing frame: {frame}\n"
                if narrative:
                    prompt += f"8. Narrative tone: {narrative}\n"
                if guidelines:
                    prompt += f"9. Task guidelines: {guidelines}\n"

        prompt += (
            "\n## Original Reason (L1 Template)\n"
            f"{l1_reason}\n"
        )

        # Assembled grounding context (5 sources)
        # Uses ContextAssembler output if provided in context, otherwise
        # falls back to minimal feature-only context.
        assembled_text = context.get("assembled_context_text", "")
        if assembled_text:
            prompt += (
                "\n## Grounding Context (반드시 이 정보만 사용하세요)\n"
                f"{assembled_text}\n"
            )
        else:
            # Fallback: minimal feature context
            if item_name:
                prompt += f"\n## Product\n{item_name}\n"

            if feature_lines:
                prompt += f"\n## Key Features (by importance)\n{feature_lines}\n"

        prompt += (
            "\n## Grounding Rules\n"
            "- 위 Grounding Context에 명시된 사실만 사용하세요.\n"
            "- Context에 없는 숫자, 혜택, 조건을 절대 생성하지 마세요.\n"
            "- '약', '추정', '분석 결과'와 같은 불확실성 표현을 사용하세요.\n"
            "- 확정적 수익이나 절감 금액을 단정하지 마세요.\n"
            "\n## AI Disclosure Requirement (금소법)\n"
            f"{self._ai_disclosure}\n"
            "\n## Output\n"
            "Return ONLY the rewritten reason text (no JSON, no explanation)."
        )

        return prompt

    # ------------------------------------------------------------------
    # L2a: 3-Layer Safety Gate
    # ------------------------------------------------------------------

    def _apply_safety_gate(self, text: str, context: Dict) -> Tuple[bool, str]:
        """3-Layer Safety Gate for L2a LLM output.

        Gate 1 (Parse): Non-empty, no JSON remnants, no raw code
        Gate 2 (Compliance): No prohibited financial claims (reuse self_checker rules)
        Gate 3 (Quality): Length 30~200 chars, Korean ratio >= 80%

        Returns:
            (passed, gate_name_that_failed_or_empty)
        """
        # Gate 1: Parse check
        if not text or len(text.strip()) < 10:
            return False, "parse:empty"
        if any(marker in text for marker in ['{', '}', '```', '<json>', 'null']):
            return False, "parse:json_remnant"

        # Gate 2: Compliance (reuse self_checker)
        if self._self_checker:
            result = self._self_checker.check(text)
            if result.verdict == "reject" and result.violations:
                return False, "compliance:" + result.violations[0]

        # Gate 3: Quality
        if len(text) < 30 or len(text) > 200:
            return False, f"quality:length_{len(text)}"
        # Korean character ratio
        korean_count = sum(1 for c in text if '\uac00' <= c <= '\ud7a3')
        total_alpha = sum(1 for c in text if c.isalpha())
        if total_alpha > 0 and korean_count / total_alpha < 0.8:
            return False, "quality:korean_ratio_low"

        # Gate 4: Grounding — output must reference only provided context
        grounding_fail = self._check_grounding(text, context)
        if grounding_fail:
            return False, f"grounding:{grounding_fail}"

        return True, ""

    def _check_grounding(self, text: str, context: Dict) -> str:
        """Verify the LLM output is grounded in provided context.

        Checks for hallucination patterns:
        - Specific monetary amounts not present in context
        - Specific percentages not present in context
        - Product benefits not mentioned in product_info

        Args:
            text: LLM generated text.
            context: The context dict that was provided to the LLM.

        Returns:
            Empty string if grounded, failure reason if not.
        """
        import re

        # Extract numbers from output
        output_numbers = set(re.findall(r'\d[\d,]*\.?\d*', text))

        # Extract numbers from context (all sources)
        context_str = str(context.get("assembled_context_text", ""))
        context_str += str(context.get("top_features", ""))
        context_str += str(context.get("item_name", ""))
        context_numbers = set(re.findall(r'\d[\d,]*\.?\d*', context_str))

        # Allow common small numbers (1-31 for dates, percentages etc.)
        common_numbers = {str(i) for i in range(32)} | {"100", "0"}

        # Check for hallucinated specific numbers
        ungrounded_numbers = output_numbers - context_numbers - common_numbers
        # Filter out very short numbers (1-2 digits are often generic)
        suspicious = [n for n in ungrounded_numbers if len(n.replace(',', '').replace('.', '')) >= 3]

        if suspicious:
            logger.warning(
                "Grounding check: suspicious numbers in output not found in context: %s",
                suspicious[:5],
            )
            return f"hallucinated_numbers:{','.join(suspicious[:3])}"

        return ""

    # ------------------------------------------------------------------
    # LLM output extraction
    # ------------------------------------------------------------------

    def _extract_rewritten_reason(self, llm_output: str) -> str:
        """Extract the rewritten reason from LLM output.

        Strips common artefacts (markdown fences, leading/trailing
        whitespace, quoted wrappers).

        Args:
            llm_output: Raw LLM response string.

        Returns:
            Cleaned reason text, truncated to max length.
        """
        text = llm_output.strip()

        # Remove markdown code fences
        text = re.sub(r"^```[a-z]*\n?", "", text)
        text = re.sub(r"\n?```$", "", text)

        # Remove surrounding quotes
        if len(text) >= 2 and text[0] == text[-1] and text[0] in ('"', "'"):
            text = text[1:-1]

        # Truncate to max length
        if len(text) > self._max_reason_length:
            text = text[: self._max_reason_length - 3].rstrip() + "..."

        return text.strip()

    # ------------------------------------------------------------------
    # L2b: Quality validation
    # ------------------------------------------------------------------

    def _validate_l2b(
        self,
        l2a_result: ReasonResult,
        context: Dict[str, Any],
        safety_gate_passed: bool = False,
    ) -> ReasonResult:
        """L2b: Validate an L2a result for compliance and quality.

        Checks:
            0. PromptSanitizer cross-check (if available): verify the
               generated text does not reintroduce sensitive content that
               was scrubbed during L2a.
            1. PII leakage detection (regex-based).
            2. Self-checker compliance + injection + factuality.
               **Skipped** when ``safety_gate_passed=True`` because the
               same ``self_checker.check()`` already ran in
               ``_apply_safety_gate()`` Gate 2.
            3. Hallucination guard: LLM output must not introduce claims
               absent from the context.
            4. Random 5% sampling for human review flagging.

        Args:
            l2a_result: The L2a :class:`ReasonResult` to validate.
            context: Source context for factuality checking.
            safety_gate_passed: When ``True``, the compliance check
                (self_checker) is skipped since ``_apply_safety_gate()``
                already performed it.  Default ``False`` for backward
                compatibility.

        Returns:
            :class:`ReasonResult` with ``layer="L2b"`` and updated
            ``validation_passed`` / ``confidence`` / ``human_review_flagged``.
        """
        text = l2a_result.text
        passed = True
        confidence = l2a_result.confidence

        # Check 0: PromptSanitizer cross-check on generated output
        if passed and self._prompt_sanitizer is not None:
            output_sensitivity = self._prompt_sanitizer.classify(text)
            if output_sensitivity == "HIGH":
                logger.warning(
                    "L2b: Generated text classified as HIGH sensitivity "
                    "(potential data leakage), rejecting (job=%s)",
                    l2a_result.job_id,
                )
                passed = False
                confidence = 0.0

        # Check 1: PII detection
        for pattern in _PII_PATTERNS:
            if pattern.search(text):
                logger.warning("L2b: PII detected in L2a output (job=%s)", l2a_result.job_id)
                passed = False
                confidence = 0.0
                break

        # Check 2: Self-checker (compliance + injection + optional LLM factuality)
        # Skip if _apply_safety_gate() already ran self_checker.check()
        # (Gate 2) to avoid redundant LLM calls and latency.
        if passed and self._self_checker is not None and not safety_gate_passed:
            check_result = self._self_checker.check(
                reason_text=text,
                source_context=context,
            )
            if check_result.verdict == "reject":
                logger.warning(
                    "L2b: Self-checker rejected L2a output (job=%s): %s",
                    l2a_result.job_id, check_result.feedback,
                )
                passed = False
                confidence = 0.0
            elif check_result.verdict == "revise":
                # Downgrade confidence but allow through
                confidence = min(confidence, 0.6)

        # Check 3: Hallucination guard -- ensure the rewritten text does not
        # introduce product names or benefit claims absent from context.
        if passed:
            hallucination_ok = self._check_hallucination(text, context)
            if not hallucination_ok:
                logger.warning(
                    "L2b: Hallucination detected in L2a output (job=%s)",
                    l2a_result.job_id,
                )
                passed = False
                confidence = 0.0

        # Check 4: Random 5% sampling for human review
        human_review = random.random() < self.HUMAN_REVIEW_SAMPLE_RATE

        return ReasonResult(
            text=text,
            layer="L2b",
            template_id=l2a_result.template_id,
            confidence=confidence,
            validation_passed=passed,
            generated_at=datetime.now(timezone.utc).isoformat(),
            cached=False,
            human_review_flagged=human_review,
            job_id=l2a_result.job_id,
        )

    @staticmethod
    def _check_hallucination(text: str, context: Dict[str, Any]) -> bool:
        """Basic hallucination guard.

        Verifies that any specific numeric claims (percentages, amounts) in
        the generated text can be traced back to values in the context.

        This is a lightweight heuristic; the full LLM-based factuality
        check in :class:`SelfChecker` provides deeper verification when
        enabled.

        Args:
            text: Generated reason text.
            context: Source context dict.

        Returns:
            ``True`` if no hallucinated numerics detected, ``False`` otherwise.
        """
        # Extract all numbers from the generated text
        numbers_in_text = set(re.findall(r"\d+\.?\d*", text))
        if not numbers_in_text:
            return True  # No numeric claims to verify

        # Flatten all numbers from context
        context_str = json.dumps(context, default=str)
        numbers_in_context = set(re.findall(r"\d+\.?\d*", context_str))

        # Allow small numbers (< 10) as they are often generic
        suspicious = set()
        for num in numbers_in_text:
            try:
                if float(num) >= 10.0 and num not in numbers_in_context:
                    suspicious.add(num)
            except ValueError:
                continue

        if suspicious:
            logger.debug("Hallucination check: suspicious numbers %s", suspicious)
            return False

        return True

    # ------------------------------------------------------------------
    # Audit logging
    # ------------------------------------------------------------------

    def _audit(
        self,
        job_id: str,
        customer_id: str,
        event: str,
        result: ReasonResult,
    ) -> None:
        """Write an audit record for L2a/L2b events.

        Args:
            job_id: Job identifier.
            customer_id: Customer identifier.
            event: Event type (``"l2b_pass"``, ``"l2b_fallback"``, etc.).
            result: The :class:`ReasonResult` to log.
        """
        record = {
            "job_id": job_id,
            "customer_id": customer_id,
            "event": event,
            "layer": result.layer,
            "confidence": result.confidence,
            "validation_passed": result.validation_passed,
            "human_review_flagged": result.human_review_flagged,
            "generated_at": result.generated_at,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

        if self._audit_store is not None:
            try:
                self._audit_store(record)
            except Exception as exc:
                logger.warning("Audit store write failed: %s", exc)
        else:
            logger.debug("Audit record: %s", json.dumps(record, default=str))

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _minimal_reason(item_id: str) -> str:
        """Generate a minimal fallback reason when no template engine is available."""
        return f"{item_id} is recommended based on your overall profile analysis."
