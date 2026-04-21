"""
MetadataAggregator — populate HEURISTIC_RULES metadata from runtime sources.

``MetricsDerivedScoreProvider`` consumes a ``(model_version) -> dict`` callable
and applies the ``HEURISTIC_RULES`` transforms to derive FRIA / AI-Risk
dimension scores. Until this module, no production wiring supplied the
backing metadata, so every call collapsed to per-rule fallbacks (0.5) and
``PromotionGate`` behaved as a smoke test instead of a safety floor
(CLAUDE.md §1.16: "Gate enabled=true 전환 전 provider 필수").

This module wires real metadata sources:

====================== =============================================
HEURISTIC_RULES key    Source
====================== =============================================
pii_ratio              DataLineageTracker feature_source_map
human_review_fraction  HumanReviewQueue.summary() + prediction count
customer_count         ModelRegistry manifest (metadata + teacher)
param_count            ModelRegistry manifest (metadata + teacher)
llm_provider_ratio     pipeline.yaml ``llm_provider`` block
disparate_impact_min   FairnessMonitor.get_archive()
reason_coverage        caller-supplied (serving logs) → static source
====================== =============================================

Each source is a thin ``(model_version) -> partial_metadata_dict``
callable. Sources fail closed (log + return {}), so the downstream
heuristic falls back to 0.5 for any dimension whose source is missing
or raised. This matches the CLAUDE.md §1.16 posture of conservative
LIMITED on missing data.

The aggregator itself holds no AWS/DynamoDB dependencies — callers
inject the runtime objects they already have (tracker, monitor, queue,
registry). This keeps import safe in CI without boto3 / credentials.
"""

from __future__ import annotations

import logging
import math
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

logger = logging.getLogger(__name__)

__all__ = [
    "MetadataSource",
    "MetadataAggregatorConfig",
    "MetadataAggregator",
    "build_lineage_source",
    "build_fairness_source",
    "build_registry_source",
    "build_llm_source",
    "build_review_queue_source",
    "build_static_source",
    "build_metadata_aggregator_from_config",
]


# ---------------------------------------------------------------------------
# Types
# ---------------------------------------------------------------------------

MetadataSource = Callable[[str], Mapping[str, Any]]
"""``(model_version) -> partial metadata dict``.

Each source fills a subset of the HEURISTIC_RULES keys (pii_ratio,
human_review_fraction, ...). The aggregator merges source outputs;
later sources override earlier on key conflicts.
"""


@dataclass
class MetadataAggregatorConfig:
    """Aggregator runtime knobs.

    Attributes:
        cache_ttl_seconds: How long a merged metadata dict is reused
            before re-polling the sources. Defaults to 300 s so fairness
            / drift updates roll through quickly without hammering the
            backing stores on every gate evaluation.
        max_cache_entries: Soft cap on per-version entries retained in
            the cache. Oldest entries are evicted when the cap is hit.
    """

    cache_ttl_seconds: float = 300.0
    max_cache_entries: int = 256


# ---------------------------------------------------------------------------
# Aggregator
# ---------------------------------------------------------------------------

class MetadataAggregator:
    """Merge metadata from N sources into a single per-version dict.

    Usage::

        agg = MetadataAggregator([
            build_lineage_source(lineage_tracker),
            build_fairness_source(fairness_monitor),
            build_registry_source(model_registry),
            build_llm_source(pipeline_config),
            build_static_source({"reason_coverage": 0.92}),
        ])
        provider = MetricsDerivedScoreProvider(
            metadata_lookup=agg.get,
            dimensions=FRIA_DIMENSIONS,
        )

    The aggregator is callable::

        provider = MetricsDerivedScoreProvider(
            metadata_lookup=agg, ...,
        )

    so either form works.
    """

    def __init__(
        self,
        sources: Optional[Sequence[MetadataSource]] = None,
        config: Optional[MetadataAggregatorConfig] = None,
    ) -> None:
        self._sources: List[MetadataSource] = list(sources) if sources else []
        self._config = config or MetadataAggregatorConfig()
        self._cache: Dict[str, Tuple[float, Dict[str, Any]]] = {}

    def register_source(self, source: MetadataSource) -> None:
        """Append a source (later sources override earlier on key conflict)."""
        self._sources.append(source)

    def __call__(self, model_version: str) -> Dict[str, Any]:
        return self.get(model_version)

    def get(self, model_version: str) -> Dict[str, Any]:
        """Return the merged metadata dict for ``model_version``.

        On missing sources / empty result, returns ``{}`` — the downstream
        MetricsDerivedScoreProvider will then collapse every heuristic to
        its per-rule fallback (0.5), keeping the gate conservative without
        tripping.
        """
        now = time.time()
        cached = self._cache.get(model_version)
        if cached is not None:
            ts, data = cached
            if now - ts < self._config.cache_ttl_seconds:
                return dict(data)

        aggregated: Dict[str, Any] = {}
        for src in self._sources:
            try:
                partial = src(model_version) or {}
            except Exception:
                logger.exception(
                    "metadata source %r failed for model_version=%s",
                    src, model_version,
                )
                continue
            if not isinstance(partial, Mapping):
                logger.warning(
                    "metadata source %r returned non-mapping (%s) for %s; "
                    "ignoring",
                    src, type(partial).__name__, model_version,
                )
                continue
            aggregated.update(partial)

        self._store(model_version, now, aggregated)
        return dict(aggregated)

    def invalidate(self, model_version: Optional[str] = None) -> None:
        """Drop cached metadata for one (or all) model versions."""
        if model_version is None:
            self._cache.clear()
        else:
            self._cache.pop(model_version, None)

    def sources(self) -> Tuple[MetadataSource, ...]:
        """Return registered sources (read-only)."""
        return tuple(self._sources)

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _store(
        self, model_version: str, ts: float, data: Dict[str, Any],
    ) -> None:
        if len(self._cache) >= self._config.max_cache_entries:
            # Evict oldest entry (insertion order on py3.7+).
            try:
                oldest = next(iter(self._cache))
                self._cache.pop(oldest, None)
            except StopIteration:
                pass
        self._cache[model_version] = (ts, dict(data))


# ---------------------------------------------------------------------------
# Built-in source factories
# ---------------------------------------------------------------------------

def build_lineage_source(tracker: Any) -> MetadataSource:
    """Compute ``pii_ratio`` from a :class:`DataLineageTracker`.

    ``pii_ratio = non_pseudonymized_groups / total_groups``. Higher value
    means more raw-PII feature groups, which maps to higher
    ``data_sensitivity``. When the tracker has no feature map registered,
    returns ``{}`` so the heuristic falls back.
    """

    def _src(_model_version: str) -> Dict[str, Any]:
        try:
            feature_map = getattr(tracker, "feature_source_map", None)
            if not feature_map:
                return {}
            total = len(feature_map)
            if total == 0:
                return {}
            non_pseudo = 0
            for info in feature_map.values():
                if not info.get("pseudonymized", False):
                    non_pseudo += 1
            return {"pii_ratio": float(non_pseudo) / float(total)}
        except Exception:
            logger.exception("build_lineage_source: failed to read tracker")
            return {}

    return _src


def build_fairness_source(
    monitor: Any,
    limit: int = 50,
) -> MetadataSource:
    """Compute ``disparate_impact_min`` from a :class:`FairnessMonitor`.

    Reads the last ``limit`` archived metrics across all protected
    attributes and returns the minimum ``disparate_impact`` observed.
    The heuristic's ``one_minus`` transform then maps "low DI" to "high
    fairness_risk", which is the regulatory-friendly direction (worse
    fairness → stricter promotion verdict).
    """

    def _src(_model_version: str) -> Dict[str, Any]:
        try:
            get_archive = getattr(monitor, "get_archive", None)
            if get_archive is None:
                return {}
            entries = get_archive(limit=limit) or []
            di_values: List[float] = []
            for entry in entries:
                raw = entry.get("disparate_impact")
                if raw is None:
                    continue
                try:
                    di = float(raw)
                except (TypeError, ValueError):
                    continue
                if math.isnan(di) or math.isinf(di):
                    continue
                di_values.append(di)
            if not di_values:
                return {}
            # A DI of 1.0 means perfect parity. Values <1 or >1 both
            # indicate bias — map either direction to a [0, 1] penalty
            # where 1.0 is perfect and any deviation lowers the number.
            # We expose the *most skewed* observation; the ``one_minus``
            # transform then produces the fairness_risk score.
            worst = min(
                (min(v, 1.0 / v) if v > 0 else 0.0) for v in di_values
            )
            return {"disparate_impact_min": float(max(0.0, min(1.0, worst)))}
        except Exception:
            logger.exception("build_fairness_source: failed to read monitor")
            return {}

    return _src


def build_registry_source(
    registry: Any,
    customer_count_keys: Sequence[str] = (
        "customer_count",
        "scope_customer_count",
        "dataset_size",
    ),
    param_count_keys: Sequence[str] = (
        "param_count",
        "teacher_param_count",
        "total_params",
    ),
) -> MetadataSource:
    """Extract ``customer_count`` / ``param_count`` from a ModelRegistry manifest.

    Looks up ``registry.load_manifest(model_version)`` and searches
    ``manifest.metadata`` then ``manifest.teacher_metrics`` for the first
    key in ``customer_count_keys`` / ``param_count_keys``.  Allows
    operators to set either field at packaging time without changing
    the schema.
    """

    def _extract(manifest: Any, keys: Sequence[str]) -> Optional[float]:
        for container_name in ("metadata", "teacher_metrics"):
            container = getattr(manifest, container_name, None) or {}
            for key in keys:
                if key in container:
                    try:
                        return float(container[key])
                    except (TypeError, ValueError):
                        continue
        return None

    def _src(model_version: str) -> Dict[str, Any]:
        try:
            load = getattr(registry, "load_manifest", None)
            if load is None:
                return {}
            manifest = load(model_version)
        except FileNotFoundError:
            return {}
        except Exception:
            logger.exception(
                "build_registry_source: load_manifest failed for %s",
                model_version,
            )
            return {}
        out: Dict[str, Any] = {}
        cc = _extract(manifest, customer_count_keys)
        if cc is not None:
            out["customer_count"] = cc
        pc = _extract(manifest, param_count_keys)
        if pc is not None:
            out["param_count"] = pc
        return out

    return _src


def build_llm_source(
    pipeline_config: Mapping[str, Any],
    agent_slot_baseline: Optional[int] = None,
) -> MetadataSource:
    """Compute ``llm_provider_ratio`` from a pipeline.yaml config dict.

    The ratio is ``configured_slots / baseline``. A ``dummy`` backend
    (no network call) returns 0.0; any cloud backend returns the number
    of models defined under ``llm_provider.<backend>.models`` plus the
    default slot, divided by the baseline.

    When ``agent_slot_baseline`` is omitted, the baseline is the number
    of currently configured slots — so a fully-populated pipeline yields
    1.0 and a dummy backend yields 0.0. Callers who want to penalise a
    partially-populated pipeline can pass an explicit baseline (typically
    pulled from ``compliance.promotion_gate.providers.aggregator`` in
    pipeline.yaml).

    Stable across model versions — the LLM footprint is a pipeline-level
    concern, not per-model.
    """

    llm_cfg = dict((pipeline_config.get("llm_provider") or {}))
    backend = llm_cfg.get("backend", "dummy")

    def _src(_model_version: str) -> Dict[str, Any]:
        try:
            if backend == "dummy" or not backend:
                return {"llm_provider_ratio": 0.0}
            backend_cfg = llm_cfg.get(backend) or {}
            models = backend_cfg.get("models") or {}
            has_default = "default" in backend_cfg
            configured = len(models) + (1 if has_default else 0)
            if configured == 0:
                return {"llm_provider_ratio": 0.0}
            baseline = (
                agent_slot_baseline
                if agent_slot_baseline and agent_slot_baseline > 0
                else configured
            )
            baseline = max(baseline, configured)
            ratio = configured / baseline
            return {"llm_provider_ratio": float(max(0.0, min(1.0, ratio)))}
        except Exception:
            logger.exception("build_llm_source: failed to compute ratio")
            return {}

    return _src


def build_review_queue_source(
    queue: Any,
    total_predictions_fn: Optional[Callable[[str], float]] = None,
) -> MetadataSource:
    """Compute ``human_review_fraction`` from a :class:`HumanReviewQueue`.

    ``human_review_fraction = queue_total / total_predictions``. When no
    ``total_predictions_fn`` is supplied the fraction is undefined
    (queue size alone doesn't tell us the denominator), and we return
    ``{}`` so the heuristic falls back.

    ``total_predictions_fn(model_version)`` should come from the serving
    layer's prediction counter (DynamoDB, CloudWatch, or logs). The
    aggregator does not assume a specific implementation.
    """

    def _src(model_version: str) -> Dict[str, Any]:
        try:
            summary_fn = getattr(queue, "summary", None)
            if summary_fn is None:
                return {}
            summary = summary_fn() or {}
            queued = float(summary.get("total", 0))
            if total_predictions_fn is None:
                return {}
            denom = float(total_predictions_fn(model_version) or 0.0)
            if denom <= 0:
                return {}
            ratio = queued / denom
            return {"human_review_fraction": float(max(0.0, min(1.0, ratio)))}
        except Exception:
            logger.exception("build_review_queue_source: failed")
            return {}

    return _src


def build_static_source(values: Mapping[str, Any]) -> MetadataSource:
    """Emit a fixed mapping regardless of ``model_version``.

    Useful for values that don't have a machine-readable source yet
    (``reason_coverage`` tracked in serving logs, operator-supplied
    override for ``customer_count``, etc.). A copy is made at
    construction time so later mutations do not affect the source.
    """

    snapshot = dict(values)

    def _src(_model_version: str) -> Dict[str, Any]:
        return dict(snapshot)

    return _src


# ---------------------------------------------------------------------------
# High-level builder
# ---------------------------------------------------------------------------

def build_metadata_aggregator_from_config(
    pipeline_config: Mapping[str, Any],
    *,
    lineage_tracker: Any = None,
    fairness_monitor: Any = None,
    model_registry: Any = None,
    review_queue: Any = None,
    total_predictions_fn: Optional[Callable[[str], float]] = None,
    static_overrides: Optional[Mapping[str, Any]] = None,
    aggregator_config: Optional[MetadataAggregatorConfig] = None,
    agent_slot_baseline: Optional[int] = None,
) -> MetadataAggregator:
    """Compose an aggregator from the runtime objects the caller has.

    Every argument is optional. Missing sources simply contribute nothing
    to the merged metadata, and the downstream heuristic falls back to
    0.5 for the corresponding dimension.

    Ordering: lineage → registry → review → fairness → llm → static.
    Later sources override earlier on key conflict, so ``static_overrides``
    is the final operator-escape-hatch layer.
    """

    sources: List[MetadataSource] = []
    if lineage_tracker is not None:
        sources.append(build_lineage_source(lineage_tracker))
    if model_registry is not None:
        sources.append(build_registry_source(model_registry))
    if review_queue is not None:
        sources.append(
            build_review_queue_source(
                review_queue, total_predictions_fn=total_predictions_fn,
            )
        )
    if fairness_monitor is not None:
        sources.append(build_fairness_source(fairness_monitor))
    sources.append(build_llm_source(
        pipeline_config, agent_slot_baseline=agent_slot_baseline,
    ))
    if static_overrides:
        sources.append(build_static_source(static_overrides))
    return MetadataAggregator(sources=sources, config=aggregator_config)
