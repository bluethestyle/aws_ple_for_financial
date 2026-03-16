"""
Recommendation Intelligence Layer
===================================

End-to-end recommendation pipeline: scoring, constraint filtering,
top-K selection, reason generation, and self-verification.

Modules:
    scorer          -- Pluggable scoring system (AbstractScorer + registry)
    constraint_engine -- Constraint-based filtering pipeline
    selector        -- Top-K selection with diversity (DPP / MMR)
    reason          -- Reason generation sub-package (template, reverse mapper, LLM, self-check)
    pipeline        -- Full orchestration pipeline
"""

from .scorer import (
    AbstractScorer,
    ScorerRegistry,
    ScoringResult,
    WeightedSumScorer,
)
from .constraint_engine import (
    AbstractFilter,
    FilterRegistry,
    ConstraintEngine,
    FilterResult,
)
from .selector import TopKSelector, DiversityMethod
from .pipeline import RecommendationPipeline, RecommendationResult

__all__ = [
    # scorer
    "AbstractScorer",
    "ScorerRegistry",
    "ScoringResult",
    "WeightedSumScorer",
    # constraint_engine
    "AbstractFilter",
    "FilterRegistry",
    "ConstraintEngine",
    "FilterResult",
    # selector
    "TopKSelector",
    "DiversityMethod",
    # pipeline
    "RecommendationPipeline",
    "RecommendationResult",
]
