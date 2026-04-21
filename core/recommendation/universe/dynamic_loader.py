"""
DynamicItemUniverseLoader - Sprint 3 M10.

Loads the candidate item universe from Parquet tables using DuckDB
(SQL-first, per CLAUDE.md 3.3). Replaces caller-injected candidate
lists so the pipeline always sees the current set of campaigns and
products.

Data model
----------
- Campaign: lifecycle state machine (planning -> approved -> running ->
  completed / canceled). Only statuses in ``active_statuses`` are kept.
- Product: static catalog.
- Item: unified view returned to the pipeline.

Storage
-------
- Local: .parquet files on disk (tests / batch jobs)
- AWS:   s3:// URIs (DuckDB httpfs extension reads them natively)

Time window
-----------
``load(as_of)`` filters campaigns whose ``[start_date, end_date]`` contain
the given timestamp. When the Parquet lacks date columns, all items in
``active_statuses`` are returned.
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, Iterable, List, Optional, Tuple

logger = logging.getLogger(__name__)

__all__ = [
    "CampaignStatus",
    "Campaign",
    "Product",
    "Item",
    "ItemUniverseConfig",
    "DynamicItemUniverseLoader",
    "build_item_universe_loader",
]


class CampaignStatus(str, Enum):
    PLANNING = "planning"
    APPROVED = "approved"
    RUNNING = "running"
    COMPLETED = "completed"
    CANCELED = "canceled"


@dataclass
class Campaign:
    campaign_id: str
    name: str
    status: CampaignStatus
    start_date: Optional[str] = None   # ISO date
    end_date: Optional[str] = None     # ISO date
    target_segments: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Product:
    product_id: str
    name: str
    category: str = ""
    active: bool = True
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Item:
    """Unified universe view; either a campaign or a product."""
    item_id: str
    item_type: str   # "campaign" | "product"
    name: str
    status: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ItemUniverseConfig:
    campaign_parquet: str = ""
    product_parquet: str = ""
    active_statuses: Tuple[CampaignStatus, ...] = (
        CampaignStatus.APPROVED,
        CampaignStatus.RUNNING,
    )
    cache_ttl_seconds: int = 3600
    enabled: bool = False

    @classmethod
    def from_dict(cls, data: Optional[Dict[str, Any]]) -> "ItemUniverseConfig":
        if not data:
            return cls()
        raw_statuses = data.get("active_statuses",
                                 ["approved", "running"])
        try:
            active = tuple(CampaignStatus(s) for s in raw_statuses)
        except ValueError as exc:
            raise ValueError(
                f"active_statuses contains unknown value: {exc}"
            ) from exc
        return cls(
            campaign_parquet=str(data.get("campaign_parquet", "")),
            product_parquet=str(data.get("product_parquet", "")),
            active_statuses=active,
            cache_ttl_seconds=int(data.get("cache_ttl_seconds", 3600)),
            enabled=bool(data.get("enabled", False)),
        )


# ---------------------------------------------------------------------------
# Loader
# ---------------------------------------------------------------------------

class DynamicItemUniverseLoader:
    """DuckDB-backed Parquet loader with TTL cache."""

    def __init__(
        self,
        config: Optional[ItemUniverseConfig] = None,
        duckdb_conn: Any = None,
    ) -> None:
        self._cfg = config or ItemUniverseConfig()
        self._conn = duckdb_conn
        self._cache: Dict[str, Tuple[float, Any]] = {}
        self._lock = threading.Lock()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def load(self, as_of: Optional[datetime] = None) -> List[Item]:
        """Return the list of eligible items at ``as_of``."""
        if not self._cfg.enabled:
            logger.debug("ItemUniverseLoader disabled; returning []")
            return []
        now = as_of or datetime.now(timezone.utc)
        campaigns = self.get_active_campaigns(as_of=now)
        products = self.get_product_pool()
        out: List[Item] = []
        for c in campaigns:
            out.append(Item(
                item_id=c.campaign_id, item_type="campaign",
                name=c.name, status=c.status.value,
                metadata={
                    "start_date": c.start_date, "end_date": c.end_date,
                    "target_segments": list(c.target_segments),
                    **c.metadata,
                },
            ))
        for p in products:
            out.append(Item(
                item_id=p.product_id, item_type="product",
                name=p.name, status="active" if p.active else "inactive",
                metadata={"category": p.category, **p.metadata},
            ))
        logger.info(
            "ItemUniverse loaded: campaigns=%d products=%d total=%d",
            len(campaigns), len(products), len(out),
        )
        return out

    def get_active_campaigns(
        self, as_of: Optional[datetime] = None,
    ) -> List[Campaign]:
        """Campaigns whose status is active and date window covers `as_of`."""
        all_campaigns = self._load_campaigns()
        now = as_of or datetime.now(timezone.utc)
        active = {s.value for s in self._cfg.active_statuses}
        result: List[Campaign] = []
        for c in all_campaigns:
            if c.status.value not in active:
                continue
            if not self._within_window(c, now):
                continue
            result.append(c)
        return result

    def get_product_pool(self) -> List[Product]:
        products = self._load_products()
        return [p for p in products if p.active]

    # ------------------------------------------------------------------
    # Parquet loaders (with TTL cache)
    # ------------------------------------------------------------------

    def _load_campaigns(self) -> List[Campaign]:
        if not self._cfg.campaign_parquet:
            return []
        cached = self._cache_get(self._cfg.campaign_parquet)
        if cached is not None:
            return list(cached)
        rows = self._query_parquet(self._cfg.campaign_parquet)
        campaigns: List[Campaign] = []
        for row in rows:
            status_raw = str(row.get("status", "")).lower()
            try:
                status = CampaignStatus(status_raw)
            except ValueError:
                logger.warning(
                    "Skipping campaign with unknown status=%s", status_raw,
                )
                continue
            campaigns.append(Campaign(
                campaign_id=str(row.get("campaign_id", "")),
                name=str(row.get("name", "")),
                status=status,
                start_date=row.get("start_date"),
                end_date=row.get("end_date"),
                target_segments=_coerce_list(row.get("target_segments")),
                metadata={
                    k: v for k, v in row.items()
                    if k not in {"campaign_id", "name", "status",
                                 "start_date", "end_date", "target_segments"}
                },
            ))
        self._cache_put(self._cfg.campaign_parquet, campaigns)
        return campaigns

    def _load_products(self) -> List[Product]:
        if not self._cfg.product_parquet:
            return []
        cached = self._cache_get(self._cfg.product_parquet)
        if cached is not None:
            return list(cached)
        rows = self._query_parquet(self._cfg.product_parquet)
        products: List[Product] = []
        for row in rows:
            products.append(Product(
                product_id=str(row.get("product_id", "")),
                name=str(row.get("name", "")),
                category=str(row.get("category", "")),
                active=bool(row.get("active", True)),
                metadata={
                    k: v for k, v in row.items()
                    if k not in {"product_id", "name", "category", "active"}
                },
            ))
        self._cache_put(self._cfg.product_parquet, products)
        return products

    def _query_parquet(self, path: str) -> List[Dict[str, Any]]:
        """Run a DuckDB SELECT * over the Parquet path and return row dicts."""
        conn = self._conn
        close_after = False
        if conn is None:
            try:
                import duckdb
                conn = duckdb.connect(":memory:")
                close_after = True
            except ImportError as exc:
                raise RuntimeError(
                    "duckdb not installed; install `duckdb` or inject a "
                    "connection"
                ) from exc
            if path.lower().startswith("s3://"):
                try:
                    conn.execute("INSTALL httpfs; LOAD httpfs;")
                except Exception:
                    logger.warning(
                        "Could not install/load DuckDB httpfs extension; "
                        "s3:// reads may fail",
                    )
        try:
            result = conn.execute(
                f"SELECT * FROM read_parquet(?)", [path]
            )
            cols = [d[0] for d in result.description]
            return [dict(zip(cols, r)) for r in result.fetchall()]
        except Exception:
            logger.exception("DuckDB query failed for path=%s", path)
            return []
        finally:
            if close_after and conn is not None:
                try:
                    conn.close()
                except Exception:
                    pass

    # ------------------------------------------------------------------
    # Cache
    # ------------------------------------------------------------------

    def _cache_get(self, key: str) -> Optional[Any]:
        with self._lock:
            entry = self._cache.get(key)
            if entry is None:
                return None
            expiry, value = entry
            if time.time() > expiry:
                self._cache.pop(key, None)
                return None
            return value

    def _cache_put(self, key: str, value: Any) -> None:
        with self._lock:
            self._cache[key] = (
                time.time() + self._cfg.cache_ttl_seconds, value,
            )

    def invalidate_cache(self) -> None:
        with self._lock:
            self._cache.clear()

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    def _within_window(
        self, campaign: Campaign, now: datetime,
    ) -> bool:
        s = _parse_date(campaign.start_date)
        e = _parse_date(campaign.end_date)
        if s is None and e is None:
            return True
        if s is not None and now < s:
            return False
        if e is not None and now > e:
            return False
        return True


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _parse_date(value: Optional[Any]) -> Optional[datetime]:
    if value is None:
        return None
    if isinstance(value, datetime):
        return value if value.tzinfo else value.replace(tzinfo=timezone.utc)
    try:
        dt = datetime.fromisoformat(str(value))
        return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)
    except ValueError:
        return None


def _coerce_list(value: Any) -> List[str]:
    if value is None:
        return []
    if isinstance(value, str):
        # comma-separated fallback
        return [s.strip() for s in value.split(",") if s.strip()]
    if isinstance(value, (list, tuple)):
        return [str(v) for v in value]
    return [str(value)]


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def build_item_universe_loader(
    pipeline_config: Optional[Dict[str, Any]] = None,
    duckdb_conn: Any = None,
) -> DynamicItemUniverseLoader:
    """Instantiate from ``serving.item_universe`` block of pipeline.yaml."""
    serving = (pipeline_config or {}).get("serving") or {}
    cfg = ItemUniverseConfig.from_dict(serving.get("item_universe"))
    return DynamicItemUniverseLoader(config=cfg, duckdb_conn=duckdb_conn)
