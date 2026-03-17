"""
Campaign ingestor.

Computes participation rate, response rate, and offer-type distribution.
Korean columns: 고객번호, 캠페인ID, 참여여부, 응답여부, 오퍼유형.
"""

from __future__ import annotations

import logging
from typing import List

import numpy as np
import pandas as pd

from ..base import AbstractDomainIngestor
from ..registry import DomainRegistry

logger = logging.getLogger(__name__)


@DomainRegistry.register("campaign")
class CampaignIngestor(AbstractDomainIngestor):
    """Ingest and aggregate campaign engagement data."""

    @property
    def source_name(self) -> str:
        return "campaign"

    @property
    def required_columns(self) -> List[str]:
        return ["customer_id", "campaign_id"]

    def ingest(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()

        # --- boolean flag normalisation ---
        for flag_candidates, target in [
            (["participated", "참여여부"], "participated"),
            (["responded", "응답여부"], "responded"),
        ]:
            col = next((c for c in flag_candidates if c in out.columns), None)
            if col:
                out[target] = (
                    out[col]
                    .astype(str)
                    .str.strip()
                    .str.upper()
                    .isin(["1", "Y", "TRUE", "YES"])
                )

        # --- offer type standardisation ---
        offer_col = next((c for c in ["offer_type", "오퍼유형"] if c in out.columns), None)
        if offer_col:
            out["offer_type"] = out[offer_col].astype(str).str.strip().str.lower()

        # --- customer-level aggregation ---
        agg_dict: dict = {"campaign_count": ("campaign_id", "nunique")}
        if "participated" in out.columns:
            agg_dict["participation_count"] = ("participated", "sum")
        if "responded" in out.columns:
            agg_dict["response_count"] = ("responded", "sum")

        summary = out.groupby("customer_id").agg(**agg_dict).reset_index()

        if "participation_count" in summary.columns:
            summary["participation_rate"] = np.where(
                summary["campaign_count"] > 0,
                summary["participation_count"] / summary["campaign_count"],
                0.0,
            )
        if "response_count" in summary.columns:
            summary["response_rate"] = np.where(
                summary["campaign_count"] > 0,
                summary["response_count"] / summary["campaign_count"],
                0.0,
            )

        out = out.merge(summary, on="customer_id", how="left")
        return out
