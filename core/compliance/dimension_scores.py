"""
Dimension score providers for PromotionGate (S7 + M9 follow-up).

`PromotionGate` consumes two dimension-scores providers (one for Korean
FRIA, one for 금감원 AI Risk). Without a real provider the gate falls
back to a conservative 0.5 for every dimension, which collapses every
assessment to LIMITED / medium and defeats the purpose of the gate.

This module ships three providers that cover the typical wiring:

- :class:`ManualScoreProvider`    -- operator-supplied `{model: {dim: score}}`
                                     mapping. The production default until
                                     auto-derivation is audited.
- :class:`DefaultScoreProvider`   -- fixed scalar for every dimension
                                     (tests / smoke).
- :class:`MetricsDerivedScoreProvider` -- heuristic rules that read
                                     fairness / drift / model-card metadata
                                     and emit a score per dimension. Safe
                                     for dev / observability use; should
                                     be audited before being treated as
                                     a safety-floor signal.

All providers implement the same callable contract:
``provider(model_version: str) -> Dict[str, float]``  (values in [0, 1])

so they can be plugged directly into
``PromotionGate(fria_scores_provider=..., ai_risk_scores_provider=...)``.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterable, Mapping, Optional, Sequence

logger = logging.getLogger(__name__)

__all__ = [
    "DimensionScoresProvider",
    "ManualScoreProvider",
    "DefaultScoreProvider",
    "MetricsDerivedScoreProvider",
    "CompositeProvider",
    "HEURISTIC_RULES",
]


# Callable type used by PromotionGate
DimensionScoresProvider = Callable[[str], Dict[str, float]]


# ---------------------------------------------------------------------------
# ManualScoreProvider — operator-supplied exact scores
# ---------------------------------------------------------------------------

class ManualScoreProvider:
    """Return hand-supplied dimension scores per model version.

    Example::

        provider = ManualScoreProvider({
            "v1": {"data_sensitivity": 0.7, "fairness_risk": 0.3},
            "v2": {"data_sensitivity": 0.8, "fairness_risk": 0.4},
        })
        scores = provider("v1")   # missing dims get the default

    Missing dimensions fall back to ``default_score`` (0.5). Unknown
    model versions return a dict of all defaults — this preserves the
    gate's "conservative LIMITED" behaviour for unassessed models.
    """

    def __init__(
        self,
        overrides: Mapping[str, Mapping[str, float]],
        default_score: float = 0.5,
        dimensions: Optional[Sequence[str]] = None,
    ) -> None:
        self._overrides = {
            model: dict(dims) for model, dims in overrides.items()
        }
        self._default = float(default_score)
        self._dimensions = tuple(dimensions) if dimensions else None

    def __call__(self, model_version: str) -> Dict[str, float]:
        per_model = self._overrides.get(model_version)
        if per_model is None:
            # Unknown model version: return empty so :class:`CompositeProvider`
            # can fall through to the next provider. Avoids silently masking
            # a heuristic / metadata-driven layer with conservative defaults.
            return {}
        if self._dimensions:
            return {
                d: float(per_model.get(d, self._default))
                for d in self._dimensions
            }
        return {k: float(v) for k, v in per_model.items()}


# ---------------------------------------------------------------------------
# DefaultScoreProvider — single scalar for every dimension
# ---------------------------------------------------------------------------

class DefaultScoreProvider:
    """Trivial provider that emits the same score for every dimension.

    Useful for tests and for smoke-testing the promotion gate before
    a real provider is wired up. Production callers should prefer
    :class:`ManualScoreProvider` or :class:`MetricsDerivedScoreProvider`.
    """

    def __init__(
        self,
        value: float = 0.5,
        dimensions: Optional[Sequence[str]] = None,
    ) -> None:
        if not (0.0 <= value <= 1.0):
            raise ValueError(f"value={value} must be in [0.0, 1.0]")
        self._value = float(value)
        self._dimensions = tuple(dimensions) if dimensions else None

    def __call__(self, model_version: str) -> Dict[str, float]:
        if self._dimensions:
            return {d: self._value for d in self._dimensions}
        return {}


# ---------------------------------------------------------------------------
# MetricsDerivedScoreProvider — rule-based derivation from model metadata
# ---------------------------------------------------------------------------

# Each rule reads a key path from the per-model metadata dict and returns
# a score in [0, 1]. Missing paths collapse to `fallback`. The caller
# supplies a metadata lookup: (model_version) -> dict[str, Any].
@dataclass
class HeuristicRule:
    path: str                       # dot-delimited key into metadata
    transform: str                  # "identity" | "one_minus" | "log10_ratio"
    fallback: float = 0.5
    ratio_denominator: float = 1.0  # used by log10_ratio

    def apply(self, metadata: Mapping[str, Any]) -> float:
        raw = _walk(metadata, self.path)
        if raw is None:
            return float(self.fallback)
        try:
            v = float(raw)
        except (TypeError, ValueError):
            return float(self.fallback)
        if self.transform == "identity":
            out = v
        elif self.transform == "one_minus":
            out = 1.0 - v
        elif self.transform == "log10_ratio":
            denom = max(self.ratio_denominator, 1.0)
            out = math.log10(max(v, 1.0) + 1.0) / math.log10(denom + 1.0)
        else:
            out = float(self.fallback)
        return max(0.0, min(1.0, out))


# Default heuristic rules — keep simple and auditable. Each maps a
# *compliance dimension* to how to derive it from model metadata.
HEURISTIC_RULES: Dict[str, HeuristicRule] = {
    # higher PII token count → higher data_sensitivity
    "data_sensitivity": HeuristicRule(
        path="pii_ratio", transform="identity", fallback=0.5,
    ),
    # higher human-review fraction → lower automation_level
    "automation_level": HeuristicRule(
        path="human_review_fraction", transform="one_minus", fallback=0.5,
    ),
    # larger customer count → higher scope_of_impact (log-normalised)
    "scope_of_impact": HeuristicRule(
        path="customer_count", transform="log10_ratio", fallback=0.5,
        ratio_denominator=1_000_000,
    ),
    # larger param count → higher model_complexity
    "model_complexity": HeuristicRule(
        path="param_count", transform="log10_ratio", fallback=0.5,
        ratio_denominator=10_000_000,
    ),
    # more LLM providers → higher external_dependency
    "external_dependency": HeuristicRule(
        path="llm_provider_ratio", transform="identity", fallback=0.5,
    ),
    # lower min DI → higher fairness_risk
    "fairness_risk": HeuristicRule(
        path="disparate_impact_min", transform="one_minus", fallback=0.5,
    ),
    # lower reason coverage → higher explainability_gap
    "explainability_gap": HeuristicRule(
        path="reason_coverage", transform="one_minus", fallback=0.5,
    ),
}


class MetricsDerivedScoreProvider:
    """Heuristic-based provider reading runtime metadata.

    Args:
        metadata_lookup: ``(model_version) -> Dict[str, Any]`` returning
            the metadata dict (fairness metrics, model-card fields,
            drift summary, etc.) for the given version. Missing returns
            empty dict; heuristics fall back to the per-rule default.
        rules: Override map ``{dimension: HeuristicRule}``. When ``None``
            the :data:`HEURISTIC_RULES` default is used.
        dimensions: Restrict the output to a specific dimension set.
            When ``None`` all rule dimensions are emitted.
    """

    def __init__(
        self,
        metadata_lookup: Callable[[str], Mapping[str, Any]],
        rules: Optional[Mapping[str, HeuristicRule]] = None,
        dimensions: Optional[Sequence[str]] = None,
    ) -> None:
        self._lookup = metadata_lookup
        self._rules: Dict[str, HeuristicRule] = (
            dict(rules) if rules is not None else dict(HEURISTIC_RULES)
        )
        self._dimensions = (
            tuple(dimensions) if dimensions else tuple(self._rules.keys())
        )

    def __call__(self, model_version: str) -> Dict[str, float]:
        try:
            meta = self._lookup(model_version) or {}
        except Exception:
            logger.exception(
                "metadata_lookup failed for model_version=%s", model_version,
            )
            meta = {}
        out: Dict[str, float] = {}
        for dim in self._dimensions:
            rule = self._rules.get(dim)
            if rule is None:
                out[dim] = 0.5
            else:
                out[dim] = rule.apply(meta)
        return out

    def explain(self, model_version: str) -> Dict[str, Dict[str, Any]]:
        """Return a per-dimension derivation trail for audit.

        For each dimension emits ``{path, raw_value, transform, fallback,
        fallback_used, score}``. ``fallback_used=True`` when the metadata
        path was missing / unparsable and the rule's fallback value was
        used, ``False`` when the raw value drove the score.

        Used by :class:`PromotionGate` to populate ``GateVerdict.details``
        so the HMAC+hash-chain audit log can reproduce why a verdict was
        issued (CLAUDE.md §1.10).
        """
        try:
            meta = self._lookup(model_version) or {}
        except Exception:
            logger.exception(
                "metadata_lookup failed during explain for model_version=%s",
                model_version,
            )
            meta = {}
        trail: Dict[str, Dict[str, Any]] = {}
        for dim in self._dimensions:
            rule = self._rules.get(dim)
            if rule is None:
                trail[dim] = {
                    "path": None, "raw_value": None,
                    "transform": None, "fallback": 0.5,
                    "fallback_used": True, "score": 0.5,
                }
                continue
            raw = _walk(meta, rule.path)
            score = rule.apply(meta)
            fallback_used = raw is None
            if raw is not None:
                try:
                    float(raw)
                except (TypeError, ValueError):
                    fallback_used = True
            trail[dim] = {
                "path": rule.path,
                "raw_value": raw,
                "transform": rule.transform,
                "fallback": rule.fallback,
                "fallback_used": fallback_used,
                "score": score,
            }
        return trail


# ---------------------------------------------------------------------------
# CompositeProvider — layer providers with a win-precedence order
# ---------------------------------------------------------------------------

class CompositeProvider:
    """Combine multiple providers. Earlier providers win on conflicts.

    Useful when an operator wants manual overrides for specific models
    while letting the heuristic provider cover the rest.
    """

    def __init__(self, providers: Sequence[DimensionScoresProvider]) -> None:
        if not providers:
            raise ValueError("CompositeProvider needs at least one provider")
        self._providers = list(providers)

    def __call__(self, model_version: str) -> Dict[str, float]:
        merged: Dict[str, float] = {}
        # Walk backwards so the first provider overrides.
        for p in reversed(self._providers):
            try:
                layer = p(model_version) or {}
            except Exception:
                logger.exception(
                    "provider %r failed for model_version=%s",
                    p, model_version,
                )
                continue
            merged.update(layer)
        return merged


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _walk(data: Any, path: str) -> Any:
    """Traverse a dotted path through nested dicts; return None if missing."""
    cursor: Any = data
    for part in path.split("."):
        if isinstance(cursor, Mapping) and part in cursor:
            cursor = cursor[part]
        else:
            return None
    return cursor
