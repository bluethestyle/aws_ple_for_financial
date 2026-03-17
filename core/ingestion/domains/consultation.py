"""
Consultation history ingestor.

Counts visits, categorises topics, computes recency and channel distribution.
Korean columns: 고객번호, 상담일자, 상담유형, 상담채널.
"""

from __future__ import annotations

import logging
from typing import List

import pandas as pd

from ..base import AbstractDomainIngestor
from ..registry import DomainRegistry

logger = logging.getLogger(__name__)

_CHANNEL_MAP = {
    "B": "branch",
    "C": "call_centre",
    "O": "online",
    "M": "mobile",
    "지점": "branch",
    "콜센터": "call_centre",
    "온라인": "online",
    "모바일": "mobile",
}


@DomainRegistry.register("consultation")
class ConsultationIngestor(AbstractDomainIngestor):
    """Ingest and summarise consultation / interaction history."""

    @property
    def source_name(self) -> str:
        return "consultation"

    @property
    def required_columns(self) -> List[str]:
        return ["customer_id"]

    def ingest(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()

        # --- date parsing ---
        date_col = next((c for c in ["consult_date", "상담일자"] if c in out.columns), None)
        if date_col:
            out["consult_date"] = pd.to_datetime(out[date_col], errors="coerce")

        # --- channel standardisation ---
        chan_col = next((c for c in ["channel", "상담채널"] if c in out.columns), None)
        if chan_col:
            out["channel"] = (
                out[chan_col].astype(str).str.strip().map(_CHANNEL_MAP).fillna("other")
            )

        # --- topic categorisation ---
        topic_col = next((c for c in ["consult_type", "상담유형"] if c in out.columns), None)
        if topic_col:
            out["consult_type"] = out[topic_col].astype(str).str.strip().str.lower()

        # --- customer-level summary ---
        agg_dict: dict = {}
        if "consult_date" in out.columns:
            agg_dict["visit_count"] = ("consult_date", "count")
            agg_dict["last_consult_date"] = ("consult_date", "max")
        if "consult_type" in out.columns:
            agg_dict["topic_variety"] = ("consult_type", "nunique")
        if "channel" in out.columns:
            agg_dict["channel_variety"] = ("channel", "nunique")

        if agg_dict:
            summary = out.groupby("customer_id").agg(**agg_dict).reset_index()
            if "last_consult_date" in summary.columns:
                summary["days_since_consult"] = (
                    pd.Timestamp.now() - summary["last_consult_date"]
                ).dt.days
            out = out.merge(summary, on="customer_id", how="left")

        return out
