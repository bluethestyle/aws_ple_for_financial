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
    FDTVSScorer,
)
from .constraint_engine import (
    AbstractFilter,
    FilterRegistry,
    ConstraintEngine,
    FilterResult,
)
from .selector import TopKSelector, DiversityMethod
from .pipeline import RecommendationPipeline, RecommendationResult, RecommendationItem
from .audit_archiver import RecommendationAuditArchiver, RecommendationAuditRecord
from .rule_engine import RuleBasedRecommender
from .fallback_router import FallbackRouter
# Sprint 3 additions
from .universe import (
    Campaign,
    CampaignStatus,
    DynamicItemUniverseLoader,
    Item,
    ItemUniverseConfig,
    Product,
    build_item_universe_loader,
)
from .reason.marker_applier import (
    DEFAULT_MARKER_TEXT,
    MarkerApplier,
    MarkerConfig,
    wrap_provider,
)

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
    "RecommendationItem",
    # additional exports
    "FDTVSScorer",
    # audit archiver
    "RecommendationAuditArchiver",
    "RecommendationAuditRecord",
    # layer 3 fallback
    "RuleBasedRecommender",
    "FallbackRouter",
    # Sprint 3 M10: dynamic item universe
    "Campaign",
    "CampaignStatus",
    "DynamicItemUniverseLoader",
    "Item",
    "ItemUniverseConfig",
    "Product",
    "build_item_universe_loader",
    # Sprint 3 M12: LLM marker
    "DEFAULT_MARKER_TEXT",
    "MarkerApplier",
    "MarkerConfig",
    "wrap_provider",
]
