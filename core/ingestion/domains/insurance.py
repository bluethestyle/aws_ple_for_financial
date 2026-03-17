"""
Insurance ingestor.

Counts policies, aggregates premiums, and summarises claim history.
Korean columns: 고객번호, 보험증권번호, 보험료, 청구금액, 청구건수.
"""

from __future__ import annotations

import logging
from typing import List

import pandas as pd

from ..base import AbstractDomainIngestor
from ..registry import DomainRegistry

logger = logging.getLogger(__name__)

_STATUS_MAP = {
    "A": "active",
    "E": "expired",
    "C": "cancelled",
    "유지": "active",
    "만기": "expired",
    "해지": "cancelled",
}


@DomainRegistry.register("insurance")
class InsuranceIngestor(AbstractDomainIngestor):
    """Ingest and aggregate insurance policy data."""

    @property
    def source_name(self) -> str:
        return "insurance"

    @property
    def required_columns(self) -> List[str]:
        return ["customer_id"]

    def ingest(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()

        # --- status normalisation ---
        status_col = next((c for c in ["policy_status", "보험상태"] if c in out.columns), None)
        if status_col:
            out["policy_status"] = (
                out[status_col].astype(str).str.strip().map(_STATUS_MAP).fillna("unknown")
            )

        # --- premium & claim numerics ---
        for src_candidates, target in [
            (["premium", "보험료"], "premium"),
            (["claim_amount", "청구금액"], "claim_amount"),
            (["claim_count", "청구건수"], "claim_count"),
        ]:
            col = next((c for c in src_candidates if c in out.columns), None)
            if col:
                out[target] = pd.to_numeric(out[col], errors="coerce").fillna(0)

        # --- customer-level aggregation ---
        policy_col = next((c for c in ["policy_no", "보험증권번호"] if c in out.columns), None)
        agg_dict: dict = {}
        if policy_col:
            agg_dict["policy_count"] = (policy_col, "nunique")
        if "premium" in out.columns:
            agg_dict["total_premium"] = ("premium", "sum")
        if "claim_amount" in out.columns:
            agg_dict["total_claim_amount"] = ("claim_amount", "sum")
        if "claim_count" in out.columns:
            agg_dict["total_claim_count"] = ("claim_count", "sum")

        if agg_dict:
            ins_agg = out.groupby("customer_id").agg(**agg_dict).reset_index()
            out = out.merge(ins_agg, on="customer_id", how="left")

        return out
