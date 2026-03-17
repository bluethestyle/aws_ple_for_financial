"""
Account (products) ingestor.

Normalises the products / accounts table: product-type enrichment,
status normalisation, and multi-account aggregation per customer.
Korean columns: 고객번호, 계좌번호, 상품유형코드, 계좌상태.
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
    "C": "closed",
    "D": "dormant",
    "S": "suspended",
    "1": "active",
    "2": "closed",
    "3": "dormant",
}

_PRODUCT_TYPE_MAP = {
    "01": "savings",
    "02": "checking",
    "03": "fixed_deposit",
    "04": "loan",
    "05": "mortgage",
    "06": "investment",
}


@DomainRegistry.register("account")
class AccountIngestor(AbstractDomainIngestor):
    """Ingest and normalise account / product data."""

    @property
    def source_name(self) -> str:
        return "account"

    @property
    def required_columns(self) -> List[str]:
        return ["customer_id", "account_no"]

    def ingest(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()

        # --- status normalisation ---
        status_col = next((c for c in ["account_status", "계좌상태"] if c in out.columns), None)
        if status_col:
            out["account_status"] = (
                out[status_col].astype(str).str.strip().str.upper().map(_STATUS_MAP).fillna("unknown")
            )

        # --- product type enrichment ---
        ptype_col = next((c for c in ["product_type_code", "상품유형코드"] if c in out.columns), None)
        if ptype_col:
            out["product_type"] = (
                out[ptype_col].astype(str).str.strip().map(_PRODUCT_TYPE_MAP).fillna("other")
            )
        else:
            logger.warning("Product type column not found; skipping enrichment")

        # --- multi-account aggregation ---
        agg = (
            out.groupby("customer_id")
            .agg(
                account_count=("account_no", "nunique"),
                active_account_count=("account_status", lambda s: (s == "active").sum()),
                product_types=("product_type", lambda s: ",".join(sorted(s.dropna().unique())) if "product_type" in out.columns else ""),
            )
            .reset_index()
        )
        out = out.merge(agg, on="customer_id", how="left")

        return out
