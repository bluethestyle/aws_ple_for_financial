"""
Merchant hierarchy ingestor.

Maps MCC codes to L1/L2 hierarchy, classifies brands, and enriches
merchant metadata.
Korean columns: 가맹점번호, MCC코드, 가맹점명, 브랜드구분.
"""

from __future__ import annotations

import logging
from typing import List

import pandas as pd

from ..base import AbstractDomainIngestor
from ..registry import DomainRegistry

logger = logging.getLogger(__name__)

_MCC_L1 = {
    range(1, 1500): "agriculture",
    range(1500, 3000): "contracted_services",
    range(3000, 3500): "travel",
    range(3500, 4000): "lodging",
    range(4000, 5000): "transport_utilities",
    range(5000, 6000): "retail",
    range(6000, 7000): "financial_services",
    range(7000, 8000): "personal_business_services",
    range(8000, 9000): "professional_services",
    range(9000, 10000): "government",
}

_MCC_L2 = {
    range(5800, 5900): "restaurants",
    range(5900, 5950): "grocery",
    range(5200, 5300): "home_supply",
    range(5300, 5400): "wholesale",
    range(5400, 5500): "food_stores",
    range(5600, 5700): "apparel",
    range(5700, 5800): "furniture",
    range(7000, 7100): "lodging_services",
    range(7200, 7300): "laundry_cleaning",
    range(7800, 7900): "entertainment",
}


def _lookup_hierarchy(code, mapping: dict) -> str:
    try:
        code = int(code)
    except (ValueError, TypeError):
        return "unknown"
    for rng, label in mapping.items():
        if code in rng:
            return label
    return "other"


@DomainRegistry.register("merchant")
class MerchantIngestor(AbstractDomainIngestor):
    """Ingest and enrich merchant master / hierarchy data."""

    @property
    def source_name(self) -> str:
        return "merchant"

    @property
    def required_columns(self) -> List[str]:
        return ["merchant_id"]

    def ingest(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()

        # --- MCC hierarchy mapping ---
        mcc_col = next((c for c in ["mcc_code", "MCC코드"] if c in out.columns), None)
        if mcc_col:
            out["mcc_l1"] = out[mcc_col].apply(lambda v: _lookup_hierarchy(v, _MCC_L1))
            out["mcc_l2"] = out[mcc_col].apply(lambda v: _lookup_hierarchy(v, _MCC_L2))
        else:
            logger.warning("MCC code column not found; skipping hierarchy mapping")

        # --- brand classification ---
        brand_col = next((c for c in ["brand_type", "브랜드구분"] if c in out.columns), None)
        if brand_col:
            out["brand_type"] = out[brand_col].astype(str).str.strip().str.lower()
            out["is_franchise"] = out["brand_type"].isin(["franchise", "프랜차이즈", "체인"])

        # --- merchant name cleaning ---
        name_col = next((c for c in ["merchant_name", "가맹점명"] if c in out.columns), None)
        if name_col:
            out["merchant_name"] = (
                out[name_col].astype(str).str.strip().str.replace(r"\s+", " ", regex=True)
            )

        return out
