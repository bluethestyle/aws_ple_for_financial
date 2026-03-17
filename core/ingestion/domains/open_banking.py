"""
Open Banking ingestor.

Counts external accounts and aggregates cross-bank balance data.
Korean columns: 고객번호, 타행코드, 타행계좌번호, 타행잔액.
"""

from __future__ import annotations

import logging
from typing import List

import pandas as pd

from ..base import AbstractDomainIngestor
from ..registry import DomainRegistry

logger = logging.getLogger(__name__)


@DomainRegistry.register("open_banking")
class OpenBankingIngestor(AbstractDomainIngestor):
    """Ingest and aggregate open-banking (cross-bank) data."""

    @property
    def source_name(self) -> str:
        return "open_banking"

    @property
    def required_columns(self) -> List[str]:
        return ["customer_id"]

    def ingest(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()

        # --- external bank code standardisation ---
        bank_col = next((c for c in ["bank_code", "타행코드"] if c in out.columns), None)
        if bank_col:
            out["bank_code"] = out[bank_col].astype(str).str.strip().str.zfill(3)

        # --- balance numeric conversion ---
        bal_col = next((c for c in ["external_balance", "타행잔액"] if c in out.columns), None)
        if bal_col:
            out["external_balance"] = pd.to_numeric(out[bal_col], errors="coerce").fillna(0)

        # --- customer-level aggregation ---
        acct_col = next(
            (c for c in ["external_account_no", "타행계좌번호"] if c in out.columns), None
        )
        agg_dict: dict = {}
        if bank_col:
            agg_dict["external_bank_count"] = ("bank_code", "nunique")
        if acct_col:
            agg_dict["external_account_count"] = (acct_col, "nunique")
        if "external_balance" in out.columns:
            agg_dict["total_external_balance"] = ("external_balance", "sum")
            agg_dict["avg_external_balance"] = ("external_balance", "mean")

        if agg_dict:
            summary = out.groupby("customer_id").agg(**agg_dict).reset_index()
            out = out.merge(summary, on="customer_id", how="left")

        return out
