"""
G7 Membership ingestor.

Classifies membership tiers, computes point balances and utilisation rates.
Korean columns: 고객번호, 멤버십ID, 등급, 적립포인트, 사용포인트.
"""

from __future__ import annotations

import logging
from typing import List

import numpy as np
import pandas as pd

from ..base import AbstractDomainIngestor
from ..registry import DomainRegistry

logger = logging.getLogger(__name__)

_TIER_ORDER = {"platinum": 4, "gold": 3, "silver": 2, "bronze": 1, "basic": 0}
_TIER_NORMALISE = {
    "P": "platinum", "G": "gold", "S": "silver", "B": "bronze",
    "플래티넘": "platinum", "골드": "gold", "실버": "silver", "브론즈": "bronze",
}


@DomainRegistry.register("membership")
class G7MembershipIngestor(AbstractDomainIngestor):
    """Ingest and normalise membership / loyalty data."""

    @property
    def source_name(self) -> str:
        return "membership"

    @property
    def required_columns(self) -> List[str]:
        return ["customer_id", "membership_id"]

    def ingest(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()

        # --- tier classification ---
        tier_col = next((c for c in ["tier", "등급"] if c in out.columns), None)
        if tier_col:
            normalised = out[tier_col].astype(str).str.strip().str.lower()
            out["tier"] = normalised.map(
                lambda v: _TIER_NORMALISE.get(v, v) if v in _TIER_NORMALISE else v
            )
            out["tier_rank"] = out["tier"].map(_TIER_ORDER).fillna(-1).astype(int)

        # --- point balance ---
        earned_col = next((c for c in ["points_earned", "적립포인트"] if c in out.columns), None)
        used_col = next((c for c in ["points_used", "사용포인트"] if c in out.columns), None)
        if earned_col and used_col:
            out["points_earned"] = pd.to_numeric(out[earned_col], errors="coerce").fillna(0)
            out["points_used"] = pd.to_numeric(out[used_col], errors="coerce").fillna(0)
            out["points_balance"] = out["points_earned"] - out["points_used"]
            out["point_utilization_rate"] = np.where(
                out["points_earned"] > 0,
                out["points_used"] / out["points_earned"],
                0.0,
            )
        else:
            logger.warning("Point columns incomplete; skipping balance calc")

        return out
