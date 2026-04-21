"""
core.recommendation.universe
============================

Dynamic candidate item universe (Sprint 3 M10). Loads the currently
eligible set of campaigns and products from Parquet tables so that the
recommendation pipeline never sees stale / canceled items.

The loader supports both local Parquet files and s3:// URIs, using the
DuckDB query engine per the data backend policy in CLAUDE.md 3.3.
"""

from core.recommendation.universe.dynamic_loader import (
    Campaign,
    CampaignStatus,
    DynamicItemUniverseLoader,
    Item,
    ItemUniverseConfig,
    Product,
    build_item_universe_loader,
)

__all__ = [
    "Campaign",
    "CampaignStatus",
    "DynamicItemUniverseLoader",
    "Item",
    "ItemUniverseConfig",
    "Product",
    "build_item_universe_loader",
]
