"""
Transaction ingestor (largest domain).

Parses dates, deduplicates, maps MCC categories, normalises amount signs,
and computes daily / weekly / monthly aggregation windows.
Korean columns: 고객번호, 거래금액, 거래일자, 거래유형, 가맹점코드.
"""

from __future__ import annotations

import logging
from typing import List

import numpy as np
import pandas as pd

from ..base import AbstractDomainIngestor
from ..registry import DomainRegistry

logger = logging.getLogger(__name__)

_MCC_CATEGORY_MAP = {
    range(1, 1500): "agricultural",
    range(1500, 3000): "contracted_services",
    range(3000, 3300): "airlines",
    range(3300, 3500): "car_rental",
    range(3500, 4000): "lodging",
    range(4000, 4800): "transportation",
    range(4800, 5000): "utilities",
    range(5000, 5600): "retail",
    range(5600, 5700): "clothing",
    range(5700, 5800): "home_furnishing",
    range(5800, 5900): "food_service",
    range(5900, 6000): "drug_stores",
    range(6000, 7000): "financial",
    range(7000, 7300): "personal_services",
    range(7300, 8000): "business_services",
    range(8000, 9000): "professional_services",
    range(9000, 10000): "government",
}

_DEBIT_TYPES = {"출금", "이체출금", "withdrawal", "debit"}


def _map_mcc(code) -> str:
    """Map an integer MCC code to a human-readable category."""
    try:
        code = int(code)
    except (ValueError, TypeError):
        return "unknown"
    for rng, label in _MCC_CATEGORY_MAP.items():
        if code in rng:
            return label
    return "other"


@DomainRegistry.register("transactions")
class TransactionIngestor(AbstractDomainIngestor):
    """Ingest, clean, and aggregate transaction data."""

    @property
    def source_name(self) -> str:
        return "transactions"

    @property
    def required_columns(self) -> List[str]:
        return ["customer_id", "transaction_amount", "transaction_date"]

    def ingest(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()

        # --- date parsing ---
        date_col = next((c for c in ["transaction_date", "거래일자"] if c in out.columns), None)
        if date_col:
            out["transaction_date"] = pd.to_datetime(out[date_col], errors="coerce")
        out.dropna(subset=["transaction_date"], inplace=True)

        # --- deduplication ---
        dedup_cols = ["customer_id", "transaction_date", "transaction_amount"]
        extra = [c for c in ["card_no", "account_no"] if c in out.columns]
        before = len(out)
        out.drop_duplicates(subset=dedup_cols + extra, inplace=True)
        dropped = before - len(out)
        if dropped:
            logger.info("Dropped %d duplicate transactions", dropped)

        # --- amount sign normalisation ---
        amt_col = next((c for c in ["transaction_amount", "거래금액"] if c in out.columns), None)
        if amt_col:
            out["transaction_amount"] = pd.to_numeric(out[amt_col], errors="coerce").abs()
        txn_type_col = next((c for c in ["transaction_type", "거래유형"] if c in out.columns), None)
        if txn_type_col:
            is_debit = out[txn_type_col].astype(str).str.lower().isin(_DEBIT_TYPES)
            out.loc[is_debit, "transaction_amount"] *= -1

        # --- MCC category mapping ---
        mcc_col = next((c for c in ["mcc_code", "가맹점코드"] if c in out.columns), None)
        if mcc_col:
            out["mcc_category"] = out[mcc_col].apply(_map_mcc)

        # --- temporal aggregation features ---
        out["txn_year_month"] = out["transaction_date"].dt.to_period("M")
        out["txn_week"] = out["transaction_date"].dt.isocalendar().week.astype(int)
        out["txn_dow"] = out["transaction_date"].dt.dayofweek

        return out
