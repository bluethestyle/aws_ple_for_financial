"""
Card ingestor.

Classifies card types, derives limit/usage ratios, and flags dormant cards.
Korean columns: 고객번호, 카드번호, 카드유형, 카드한도, 이용금액.
"""

from __future__ import annotations

import logging
from typing import List

import numpy as np
import pandas as pd

from ..base import AbstractDomainIngestor
from ..registry import DomainRegistry

logger = logging.getLogger(__name__)

_CARD_TYPE_MAP = {
    "C": "credit",
    "D": "debit",
    "P": "prepaid",
    "H": "hybrid",
    "01": "credit",
    "02": "debit",
    "03": "prepaid",
}


@DomainRegistry.register("card")
class CardIngestor(AbstractDomainIngestor):
    """Ingest and normalise card data."""

    @property
    def source_name(self) -> str:
        return "card"

    @property
    def required_columns(self) -> List[str]:
        return ["customer_id", "card_no"]

    def ingest(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()

        # --- card type classification ---
        ctype_col = next((c for c in ["card_type", "카드유형"] if c in out.columns), None)
        if ctype_col:
            out["card_type"] = (
                out[ctype_col].astype(str).str.strip().str.upper().map(_CARD_TYPE_MAP).fillna("other")
            )

        # --- limit / usage derivation ---
        limit_col = next((c for c in ["card_limit", "카드한도"] if c in out.columns), None)
        usage_col = next((c for c in ["used_amount", "이용금액"] if c in out.columns), None)

        if limit_col and usage_col:
            out["card_limit"] = pd.to_numeric(out[limit_col], errors="coerce")
            out["used_amount"] = pd.to_numeric(out[usage_col], errors="coerce")
            out["utilization_ratio"] = np.where(
                out["card_limit"] > 0,
                out["used_amount"] / out["card_limit"],
                np.nan,
            )
        else:
            logger.warning("Limit/usage columns incomplete; skipping utilisation calc")

        # --- dormancy check (no usage in last 6 months) ---
        last_use_col = next(
            (c for c in ["last_used_date", "최종이용일자"] if c in out.columns), None
        )
        if last_use_col:
            out["last_used_date"] = pd.to_datetime(out[last_use_col], errors="coerce")
            cutoff = pd.Timestamp.now() - pd.DateOffset(months=6)
            out["is_dormant"] = out["last_used_date"] < cutoff
        else:
            out["is_dormant"] = False

        return out
