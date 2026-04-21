"""
Context Assembler -- automatic multi-source context assembly for reason generation.
====================================================================================

Assembles context from 5 structured sources (no vector search needed):

1. **Feature Interpretation** (from ReverseMapper + glossary)
2. **Product Information** (from product catalog)
3. **Consultation History** (from customer records)
4. **Customer Segment Info** (from GMM clustering + demographics)
5. **Regulatory Templates** (from compliance config)

The assembled context is consumed by:

- :meth:`AsyncReasonOrchestrator.generate_l1` (L1 template)
- :meth:`AsyncReasonOrchestrator.submit_l2a_rewrite` (L2a LLM prompt)
- :meth:`RecommendationPipeline._generate_reasons` (inline)

Usage::

    assembler = ContextAssembler(
        reverse_mapper=reverse_mapper,
        product_catalog=product_catalog,
    )
    context = assembler.assemble(
        customer_id="C001",
        task_name="churn",
        features=feature_vector,
        feature_importances=ig_top_k,
        product_id="P100",
    )
    # context is an AssembledContext ready for prompt building
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np

logger = logging.getLogger(__name__)

__all__ = ["ContextAssembler", "AssembledContext"]


# ---------------------------------------------------------------------------
# Default regulatory notes per task type
# ---------------------------------------------------------------------------

_DEFAULT_REGULATORY_NOTES: Dict[str, List[str]] = {
    # Binary classification tasks (CTR / CVR)
    "binary": [
        "AI 기반 추천임을 고지",
    ],
    "ctr": [
        "AI 기반 추천임을 고지",
    ],
    "cvr": [
        "AI 기반 추천임을 고지",
    ],
    # Churn / retention
    "churn": [
        "고객 유지 목적 안내",
    ],
    "retention": [
        "고객 유지 목적 안내",
    ],
    # LTV / regression
    "ltv": [
        "예측값은 참고용이며 실제와 다를 수 있음",
    ],
    "regression": [
        "예측값은 참고용이며 실제와 다를 수 있음",
    ],
    # Multi-class (NBA)
    "multiclass": [
        "상품 적합성 평가 완료 고지",
    ],
    "nba": [
        "상품 적합성 평가 완료 고지",
    ],
    # Contrastive learning
    "contrastive": [
        "AI 선호도 분석 기반 추천",
    ],
}


# ---------------------------------------------------------------------------
# Data class
# ---------------------------------------------------------------------------

@dataclass
class AssembledContext:
    """Multi-source context assembled for recommendation reason generation.

    Each source field is independently populated -- missing sources produce
    empty sections, never errors.

    Attributes:
        customer_id: Customer identifier.
        task_name: Task type that produced the recommendation.
        product_id: Recommended product identifier.
        feature_explanations: Source 1 -- top-K feature interpretations
            sorted by importance.  Each entry is a dict with keys
            ``name``, ``value``, ``text``, ``importance``, and optionally
            ``group`` and ``range_label``.
        product_info: Source 2 -- product catalog information (name,
            category, benefits, etc.).
        consultation_summary: Source 3 -- natural-language summary of
            recent consultations for this customer.
        segment_info: Source 4 -- customer segment information from GMM
            clustering and demographics.
        regulatory_notes: Source 5 -- task-specific regulatory notes
            that must be disclosed in the reason text.
        assembled_at: ISO-8601 timestamp of assembly.
        extra: Arbitrary extra context passed through ``assemble(**extra)``.
    """

    customer_id: str
    task_name: str
    product_id: str

    # Source 1: Feature interpretations (top-k by importance)
    feature_explanations: List[Dict[str, Any]] = field(default_factory=list)
    # e.g. [{"name": "월평균거래건수", "value": 47, "text": "월 평균 47건 거래", "importance": 0.35}]

    # Source 2: Product information
    product_info: Dict[str, Any] = field(default_factory=dict)
    # e.g. {"name": "프리미엄 카드", "category": "신용카드", "benefits": [...]}

    # Source 3: Consultation history summary
    consultation_summary: str = ""
    # e.g. "최근 3개월 내 2회 상담, 주요 문의: 한도 상향"

    # Source 4: Customer segment
    segment_info: Dict[str, Any] = field(default_factory=dict)
    # e.g. {"segment": "VIP", "cluster_id": 5, "churn_risk": 0.3, "ltv_score": 0.8}

    # Source 5: Regulatory requirements
    regulatory_notes: List[str] = field(default_factory=list)
    # e.g. ["AI 기반 추천임을 고지", "원금 손실 가능성 안내 필수"]

    # Source 6 (Sprint 2 S12): Multidisciplinary interpreter insights —
    # each key is a domain label (e.g. "behavioral_economics", "risk_mgmt")
    # and each value is a short natural-language insight derived from
    # features + segment + product. Populated when a
    # `multidisciplinary_interpreter` is attached to the assembler.
    multidisciplinary_insights: Dict[str, str] = field(default_factory=dict)

    # Metadata
    assembled_at: str = ""
    extra: Dict[str, Any] = field(default_factory=dict)

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------

    def to_dict(self) -> Dict[str, Any]:
        """Convert to plain dict for prompt building and JSON serialisation.

        Returns:
            A flat dictionary with all context fields.  Empty sections
            are included as empty values (``[]``, ``{}``, ``""``).
        """
        return {
            "customer_id": self.customer_id,
            "task_name": self.task_name,
            "product_id": self.product_id,
            "feature_explanations": self.feature_explanations,
            "product_info": self.product_info,
            "consultation_summary": self.consultation_summary,
            "segment_info": self.segment_info,
            "regulatory_notes": self.regulatory_notes,
            "multidisciplinary_insights": dict(self.multidisciplinary_insights),
            "assembled_at": self.assembled_at,
            **self.extra,
        }

    def to_prompt_text(self, max_tokens: int = 1500) -> str:
        """Format as structured text for an LLM prompt.

        Each non-empty source is rendered as a labelled section.  The
        output is truncated to approximately *max_tokens* characters
        (a rough proxy for token count -- Korean text averages ~1.5
        chars per token, so the default 1500 char limit maps to roughly
        1000 tokens).

        Args:
            max_tokens: Maximum character length for the output.

        Returns:
            Formatted multi-line string ready for LLM prompt insertion.
        """
        sections: List[str] = []

        # Source 1: Feature explanations
        if self.feature_explanations:
            lines = ["[주요 특성 해석]"]
            for feat in self.feature_explanations:
                name = feat.get("name", "?")
                text = feat.get("text", "")
                importance = feat.get("importance", 0.0)
                if text:
                    lines.append(f"  - {name}: {text} (중요도: {importance:.2f})")
                else:
                    value = feat.get("value", "")
                    lines.append(f"  - {name}: {value} (중요도: {importance:.2f})")
            sections.append("\n".join(lines))

        # Source 2: Product information
        if self.product_info:
            lines = ["[상품 정보]"]
            prod_name = self.product_info.get("name", self.product_id)
            category = self.product_info.get("category", "")
            lines.append(f"  상품명: {prod_name}")
            if category:
                lines.append(f"  카테고리: {category}")
            benefits = self.product_info.get("benefits", [])
            if benefits:
                lines.append(f"  주요 혜택: {', '.join(str(b) for b in benefits[:5])}")
            sections.append("\n".join(lines))

        # Source 3: Consultation history
        if self.consultation_summary:
            sections.append(f"[상담 이력]\n  {self.consultation_summary}")

        # Source 4: Segment info
        if self.segment_info:
            lines = ["[고객 세그먼트]"]
            segment = self.segment_info.get("segment", "")
            if segment:
                lines.append(f"  세그먼트: {segment}")
            cluster_id = self.segment_info.get("cluster_id")
            if cluster_id is not None:
                lines.append(f"  클러스터: {cluster_id}")
            churn_risk = self.segment_info.get("churn_risk")
            if churn_risk is not None:
                lines.append(f"  이탈 위험: {churn_risk:.2f}")
            ltv_score = self.segment_info.get("ltv_score")
            if ltv_score is not None:
                lines.append(f"  LTV 점수: {ltv_score:.2f}")
            sections.append("\n".join(lines))

        # Source 5: Regulatory notes
        if self.regulatory_notes:
            lines = ["[규제 요건]"]
            for note in self.regulatory_notes:
                lines.append(f"  - {note}")
            sections.append("\n".join(lines))

        prompt_text = "\n\n".join(sections)

        # Truncate to approximate token limit
        if len(prompt_text) > max_tokens:
            prompt_text = prompt_text[:max_tokens - 3].rstrip() + "..."

        return prompt_text


# ---------------------------------------------------------------------------
# Assembler
# ---------------------------------------------------------------------------

class ContextAssembler:
    """Multi-source context assembler for recommendation reason generation.

    Each data source is optional -- pass ``None`` (the default) for any
    source that is not available.  Missing sources produce empty sections
    in the assembled context, never exceptions.

    Data sources can be provided as:

    * **A dict** -- used as a direct lookup table (keyed by customer_id
      or product_id).
    * **A callable** -- invoked with the appropriate key and expected to
      return the corresponding value (or ``None``).

    Args:
        reverse_mapper: A :class:`ReverseMapper` instance for feature
            interpretation (Source 1).
        product_catalog: Product info source.  Either a
            ``Dict[str, Dict]`` mapping product_id to info, or a
            callable ``(product_id) -> Dict``.
        consultation_store: Consultation history source.  Either a
            ``Dict[str, Any]`` mapping customer_id to records, or a
            callable ``(customer_id) -> records``.  Records can be a
            string summary or a list of consultation dicts.
        segment_store: Customer segment source.  Either a
            ``Dict[str, Dict]`` mapping customer_id to segment info, or
            a callable ``(customer_id) -> Dict``.
        regulatory_config: Regulatory notes per task type.  A
            ``Dict[str, List[str]]`` mapping task_name to notes.  Falls
            back to built-in defaults when not provided.
    """

    def __init__(
        self,
        reverse_mapper=None,
        product_catalog=None,
        consultation_store=None,
        segment_store=None,
        regulatory_config: Optional[Dict[str, List[str]]] = None,
        multidisciplinary_interpreter=None,
    ) -> None:
        self._reverse_mapper = reverse_mapper
        self._product_catalog = product_catalog
        self._consultation_store = consultation_store
        self._segment_store = segment_store
        self._regulatory_config: Dict[str, List[str]] = (
            regulatory_config if regulatory_config is not None
            else _DEFAULT_REGULATORY_NOTES
        )
        # Sprint 2 S12: optional multidisciplinary interpreter.
        # Contract: any object exposing ``interpret(context_dict) -> dict[str, str]``
        # or a plain callable with the same signature.
        self._multidisciplinary_interpreter = multidisciplinary_interpreter

        logger.info(
            "ContextAssembler initialised: reverse_mapper=%s, "
            "product_catalog=%s, consultation_store=%s, "
            "segment_store=%s, regulatory_config=%d task(s), "
            "multidisciplinary_interpreter=%s",
            type(self._reverse_mapper).__name__ if self._reverse_mapper else "None",
            "dict" if isinstance(self._product_catalog, dict)
            else ("callable" if callable(self._product_catalog) else "None"),
            "dict" if isinstance(self._consultation_store, dict)
            else ("callable" if callable(self._consultation_store) else "None"),
            "dict" if isinstance(self._segment_store, dict)
            else ("callable" if callable(self._segment_store) else "None"),
            len(self._regulatory_config),
            self._multidisciplinary_interpreter is not None,
        )

    # ------------------------------------------------------------------
    # Sprint 2 S12: interpreter attachment
    # ------------------------------------------------------------------

    def attach_interpreter(self, interpreter) -> None:
        """Plug in a multidisciplinary interpreter at runtime.

        Any object with an ``interpret(context_dict) -> Dict[str, str]``
        method, or a bare callable with the same signature, is accepted.
        Passing ``None`` detaches the current interpreter.
        """
        self._multidisciplinary_interpreter = interpreter

    def _run_interpreter(
        self, assembled: "AssembledContext",
    ) -> Dict[str, str]:
        """Call the attached interpreter and normalize its output."""
        interp = self._multidisciplinary_interpreter
        if interp is None:
            return {}
        try:
            if hasattr(interp, "interpret"):
                result = interp.interpret(assembled.to_dict())
            elif callable(interp):
                result = interp(assembled.to_dict())
            else:
                return {}
        except Exception:
            logger.exception(
                "Multidisciplinary interpreter failed for customer=%s",
                assembled.customer_id,
            )
            return {}
        if not isinstance(result, dict):
            return {}
        return {str(k): str(v) for k, v in result.items() if v}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def assemble(
        self,
        customer_id: str,
        task_name: str,
        features: Any = None,
        feature_importances: Any = None,
        product_id: str = "",
        product_info: Optional[Dict[str, Any]] = None,
        cluster_id: int = -1,
        cluster_probs: Any = None,
        **extra_context,
    ) -> AssembledContext:
        """Assemble context from all available sources.

        Each source is independently populated.  Failures in one source
        are logged and do not prevent other sources from being assembled.

        Args:
            customer_id: Customer identifier.
            task_name: Task type (e.g. ``"churn"``, ``"ltv"``, ``"nba"``).
            features: Raw feature vector (numpy array or dict).  Used by
                the ReverseMapper for range classification.
            feature_importances: Feature importance data.  Accepts:
                - ``List[Tuple[str, float]]`` -- (feature_name, score) pairs.
                - ``np.ndarray`` -- 1-D importance scores.
                - ``List[Dict]`` -- pre-interpreted feature dicts.
            product_id: Product identifier for catalog lookup.
            product_info: Override product info instead of catalog lookup.
            cluster_id: GMM cluster assignment (-1 = unknown).
            cluster_probs: Cluster probability vector from GMM.
            **extra_context: Arbitrary additional context passed through
                to :attr:`AssembledContext.extra`.

        Returns:
            A fully populated :class:`AssembledContext`.
        """
        assembled_at = datetime.now(timezone.utc).isoformat()

        # Source 1: Feature explanations
        feature_explanations = self._assemble_feature_explanations(
            features=features,
            feature_importances=feature_importances,
            task_name=task_name,
        )

        # Source 2: Product information
        product = self._assemble_product_info(
            product_id=product_id,
            product_info_override=product_info,
        )

        # Source 3: Consultation history
        consultation_summary = self._assemble_consultation_history(
            customer_id=customer_id,
        )

        # Source 4: Customer segment
        segment_info = self._assemble_segment_info(
            customer_id=customer_id,
            cluster_id=cluster_id,
            cluster_probs=cluster_probs,
        )

        # Source 5: Regulatory notes
        regulatory_notes = self._assemble_regulatory_notes(
            task_name=task_name,
        )

        assembled = AssembledContext(
            customer_id=customer_id,
            task_name=task_name,
            product_id=product_id,
            feature_explanations=feature_explanations,
            product_info=product,
            consultation_summary=consultation_summary,
            segment_info=segment_info,
            regulatory_notes=regulatory_notes,
            assembled_at=assembled_at,
            extra=extra_context if extra_context else {},
        )

        # Source 6 (Sprint 2 S12): multidisciplinary interpretation
        if self._multidisciplinary_interpreter is not None:
            insights = self._run_interpreter(assembled)
            if insights:
                assembled.multidisciplinary_insights = insights

        return assembled

    # ------------------------------------------------------------------
    # Source 1: Feature interpretations via ReverseMapper
    # ------------------------------------------------------------------

    def _assemble_feature_explanations(
        self,
        features: Any,
        feature_importances: Any,
        task_name: str,
        top_k: int = 5,
    ) -> List[Dict[str, Any]]:
        """Assemble feature interpretations from ReverseMapper.

        Handles multiple input formats for ``feature_importances``:

        - ``List[Tuple[str, float]]``: (name, score) pairs from IG top-k.
        - ``np.ndarray``: Raw importance scores (uses ReverseMapper).
        - ``List[Dict]``: Pre-interpreted feature dicts (passed through).
        - ``None``: Returns empty list.

        Args:
            features: Raw feature vector for range classification.
            feature_importances: Importance data in any supported format.
            task_name: Task type for task-aware re-weighting.
            top_k: Number of top features to include.

        Returns:
            List of feature explanation dicts, sorted by importance.
        """
        if feature_importances is None:
            return []

        try:
            # Case 1: Pre-interpreted dicts -- pass through
            if isinstance(feature_importances, list) and feature_importances:
                if isinstance(feature_importances[0], dict):
                    return list(feature_importances[:top_k])

                # Case 2: List of (name, score) tuples
                if (
                    isinstance(feature_importances[0], (tuple, list))
                    and len(feature_importances[0]) == 2
                ):
                    return self._interpret_tuples(
                        feature_importances, features, task_name, top_k,
                    )

            # Case 3: numpy array -- use ReverseMapper
            if isinstance(feature_importances, np.ndarray):
                return self._interpret_array(
                    feature_importances, features, task_name, top_k,
                )

            logger.debug(
                "ContextAssembler: unrecognised feature_importances type %s, "
                "returning empty",
                type(feature_importances).__name__,
            )
            return []

        except Exception as exc:
            logger.warning(
                "ContextAssembler: feature explanation assembly failed: %s", exc,
            )
            return []

    def _interpret_tuples(
        self,
        importances: List[Tuple[str, float]],
        features: Any,
        task_name: str,
        top_k: int,
    ) -> List[Dict[str, Any]]:
        """Interpret (feature_name, importance_score) tuples.

        If a :class:`ReverseMapper` is available, its ``interpret_top_k``
        method is used for full interpretation.  Otherwise, a generic
        fallback is produced.
        """
        # Try ReverseMapper path: convert tuples to array
        if self._reverse_mapper is not None:
            try:
                scores = np.array([s for _, s in importances])
                feature_vector = None
                if features is not None:
                    if isinstance(features, np.ndarray):
                        feature_vector = features
                    elif isinstance(features, dict):
                        # Cannot align dict features with array indices;
                        # skip feature_vector for range classification
                        feature_vector = None

                interpretations = self._reverse_mapper.interpret_top_k(
                    feature_importances=scores,
                    k=top_k,
                    feature_vector=feature_vector,
                    task=task_name,
                )

                return [
                    {
                        "name": interp.feature_name,
                        "value": interp.value,
                        "text": interp.interpretation,
                        "importance": interp.ig_score,
                        "group": interp.group_label,
                        "range_label": interp.range_label,
                    }
                    for interp in interpretations
                ]
            except Exception as exc:
                logger.debug(
                    "ContextAssembler: ReverseMapper interpret_top_k "
                    "failed, falling back to generic: %s", exc,
                )

        # Generic fallback: build explanations from tuples directly
        results: List[Dict[str, Any]] = []
        for name, score in importances[:top_k]:
            value = score
            if features is not None and isinstance(features, dict):
                value = features.get(name, score)
            results.append({
                "name": str(name),
                "value": float(value),
                "text": "",
                "importance": float(abs(score)),
            })
        return results

    def _interpret_array(
        self,
        importances: np.ndarray,
        features: Any,
        task_name: str,
        top_k: int,
    ) -> List[Dict[str, Any]]:
        """Interpret a raw importance array using ReverseMapper."""
        if self._reverse_mapper is not None:
            feature_vector = None
            if isinstance(features, np.ndarray):
                feature_vector = features

            interpretations = self._reverse_mapper.interpret_top_k(
                feature_importances=importances,
                k=top_k,
                feature_vector=feature_vector,
                task=task_name,
            )

            return [
                {
                    "name": interp.feature_name,
                    "value": interp.value,
                    "text": interp.interpretation,
                    "importance": interp.ig_score,
                    "group": interp.group_label,
                    "range_label": interp.range_label,
                }
                for interp in interpretations
            ]

        # No ReverseMapper -- produce minimal output from array
        abs_scores = np.abs(importances)
        top_indices = np.argsort(abs_scores)[::-1][:top_k]
        results: List[Dict[str, Any]] = []
        for idx in top_indices:
            idx = int(idx)
            results.append({
                "name": f"feature_{idx}",
                "value": float(importances[idx]),
                "text": "",
                "importance": float(abs_scores[idx]),
            })
        return results

    # ------------------------------------------------------------------
    # Source 2: Product catalog lookup
    # ------------------------------------------------------------------

    def _assemble_product_info(
        self,
        product_id: str,
        product_info_override: Optional[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Look up product information from the catalog.

        If *product_info_override* is provided, it takes precedence over
        the catalog lookup.

        Args:
            product_id: Product identifier for catalog lookup.
            product_info_override: Explicit product info dict (overrides
                catalog).

        Returns:
            Product info dict, or empty dict if unavailable.
        """
        if product_info_override is not None:
            return product_info_override

        if not product_id or self._product_catalog is None:
            return {}

        try:
            if isinstance(self._product_catalog, dict):
                return self._product_catalog.get(product_id, {})
            if callable(self._product_catalog):
                result = self._product_catalog(product_id)
                return result if isinstance(result, dict) else {}
        except Exception as exc:
            logger.warning(
                "ContextAssembler: product catalog lookup failed for "
                "'%s': %s", product_id, exc,
            )

        return {}

    # ------------------------------------------------------------------
    # Source 3: Consultation history summary
    # ------------------------------------------------------------------

    def _assemble_consultation_history(self, customer_id: str) -> str:
        """Look up and summarise consultation history for a customer.

        The consultation store can return either a pre-built summary
        string or a list of consultation record dicts.  If a list is
        returned, a simple summary is generated.

        Args:
            customer_id: Customer identifier.

        Returns:
            A natural-language summary string, or ``""`` if unavailable.
        """
        if self._consultation_store is None:
            return ""

        try:
            if isinstance(self._consultation_store, dict):
                records = self._consultation_store.get(customer_id)
            elif callable(self._consultation_store):
                records = self._consultation_store(customer_id)
            else:
                return ""

            if records is None:
                return ""

            # Already a summary string
            if isinstance(records, str):
                return records

            # List of consultation records -- build a summary
            if isinstance(records, list) and records:
                count = len(records)
                # Extract topics from records
                topics: List[str] = []
                for rec in records:
                    if isinstance(rec, dict):
                        topic = rec.get("topic") or rec.get("subject") or ""
                        if topic and topic not in topics:
                            topics.append(str(topic))

                summary_parts = [f"총 {count}회 상담"]
                if topics:
                    summary_parts.append(
                        f"주요 문의: {', '.join(topics[:3])}"
                    )
                return ", ".join(summary_parts)

        except Exception as exc:
            logger.warning(
                "ContextAssembler: consultation history lookup failed "
                "for '%s': %s", customer_id, exc,
            )

        return ""

    # ------------------------------------------------------------------
    # Source 4: Customer segment information
    # ------------------------------------------------------------------

    def _assemble_segment_info(
        self,
        customer_id: str,
        cluster_id: int,
        cluster_probs: Any,
    ) -> Dict[str, Any]:
        """Assemble customer segment information.

        Merges data from the segment store with the provided GMM
        clustering results (``cluster_id`` and ``cluster_probs``).

        Args:
            customer_id: Customer identifier.
            cluster_id: GMM cluster assignment (-1 = unknown).
            cluster_probs: Cluster probability vector (numpy array or list).

        Returns:
            Segment info dict, or empty dict if unavailable.
        """
        info: Dict[str, Any] = {}

        # Look up from segment store
        if self._segment_store is not None:
            try:
                if isinstance(self._segment_store, dict):
                    stored = self._segment_store.get(customer_id)
                elif callable(self._segment_store):
                    stored = self._segment_store(customer_id)
                else:
                    stored = None

                if isinstance(stored, dict):
                    info.update(stored)
            except Exception as exc:
                logger.warning(
                    "ContextAssembler: segment store lookup failed for "
                    "'%s': %s", customer_id, exc,
                )

        # Overlay GMM clustering results
        if cluster_id >= 0:
            info["cluster_id"] = cluster_id

        if cluster_probs is not None:
            try:
                if isinstance(cluster_probs, np.ndarray):
                    probs = cluster_probs.tolist()
                elif isinstance(cluster_probs, list):
                    probs = cluster_probs
                else:
                    probs = list(cluster_probs)
                info["cluster_probs"] = probs
                # Derive dominant cluster if not explicitly provided
                if "cluster_id" not in info and probs:
                    info["cluster_id"] = int(np.argmax(probs))
            except Exception as exc:
                logger.debug(
                    "ContextAssembler: cluster_probs processing failed: %s",
                    exc,
                )

        return info

    # ------------------------------------------------------------------
    # Source 5: Task-specific regulatory requirements
    # ------------------------------------------------------------------

    def _assemble_regulatory_notes(self, task_name: str) -> List[str]:
        """Look up task-specific regulatory notes.

        Falls back through progressively broader task categories::

            1. Exact task_name match (e.g. ``"churn"``)
            2. Default catch-all (``"binary"``)

        Args:
            task_name: Task type identifier.

        Returns:
            List of regulatory note strings (may be empty).
        """
        if not task_name:
            return []

        # Exact match
        notes = self._regulatory_config.get(task_name)
        if notes is not None:
            return list(notes)

        # Normalise: strip whitespace and lowercase
        normalised = task_name.strip().lower()
        notes = self._regulatory_config.get(normalised)
        if notes is not None:
            return list(notes)

        # Fallback to "binary" as the most common default
        notes = self._regulatory_config.get("binary")
        if notes is not None:
            return list(notes)

        return []
