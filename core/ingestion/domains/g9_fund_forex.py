"""
G9 Fund / Forex ingestor.

Aggregates investment holdings and forex position summaries per customer.
Korean columns: 고객번호, 펀드코드, 보유잔액, 외화통화, 외화잔액.
"""

from __future__ import annotations

import logging
from typing import List

import pandas as pd

from ..base import AbstractDomainIngestor
from ..registry import DomainRegistry

logger = logging.getLogger(__name__)


@DomainRegistry.register("fund_forex")
class G9FundForexIngestor(AbstractDomainIngestor):
    """Ingest and aggregate fund / forex position data."""

    @property
    def source_name(self) -> str:
        return "fund_forex"

    @property
    def required_columns(self) -> List[str]:
        return ["customer_id"]

    def ingest(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()

        # --- fund holding aggregation ---
        fund_col = next((c for c in ["fund_code", "펀드코드"] if c in out.columns), None)
        balance_col = next((c for c in ["holding_balance", "보유잔액"] if c in out.columns), None)
        if fund_col and balance_col:
            out["holding_balance"] = pd.to_numeric(out[balance_col], errors="coerce").fillna(0)
            fund_agg = (
                out.groupby("customer_id")
                .agg(
                    fund_count=(fund_col, "nunique"),
                    total_fund_balance=("holding_balance", "sum"),
                    avg_fund_balance=("holding_balance", "mean"),
                )
                .reset_index()
            )
            out = out.merge(fund_agg, on="customer_id", how="left")
        else:
            logger.warning("Fund columns not found; skipping fund aggregation")

        # --- forex position summary ---
        currency_col = next((c for c in ["currency", "외화통화"] if c in out.columns), None)
        forex_col = next((c for c in ["forex_balance", "외화잔액"] if c in out.columns), None)
        if currency_col and forex_col:
            out["forex_balance"] = pd.to_numeric(out[forex_col], errors="coerce").fillna(0)
            fx_agg = (
                out.groupby("customer_id")
                .agg(
                    forex_currency_count=(currency_col, "nunique"),
                    total_forex_balance=("forex_balance", "sum"),
                )
                .reset_index()
            )
            out = out.merge(fx_agg, on="customer_id", how="left", suffixes=("", "_fx"))
        else:
            logger.warning("Forex columns not found; skipping forex aggregation")

        return out
