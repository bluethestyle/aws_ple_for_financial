"""
Marketing response ingestor.

Computes channel-wise response rates (SMS, email, push) and opt-in status.
Korean columns: 고객번호, 채널, 응답여부, 수신동의.
"""

from __future__ import annotations

import logging
from typing import List

import pandas as pd

from ..base import AbstractDomainIngestor
from ..registry import DomainRegistry

logger = logging.getLogger(__name__)

_CHANNEL_NORMALISE = {
    "sms": "sms",
    "SMS": "sms",
    "email": "email",
    "EMAIL": "email",
    "이메일": "email",
    "push": "push",
    "PUSH": "push",
    "푸시": "push",
    "dm": "dm",
    "DM": "dm",
}


@DomainRegistry.register("marketing")
class MarketingIngestor(AbstractDomainIngestor):
    """Ingest and aggregate marketing response data."""

    @property
    def source_name(self) -> str:
        return "marketing"

    @property
    def required_columns(self) -> List[str]:
        return ["customer_id"]

    def ingest(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()

        # --- channel normalisation ---
        chan_col = next((c for c in ["channel", "채널"] if c in out.columns), None)
        if chan_col:
            out["channel"] = (
                out[chan_col].astype(str).str.strip().map(_CHANNEL_NORMALISE).fillna("other")
            )

        # --- response flag ---
        resp_col = next((c for c in ["responded", "응답여부"] if c in out.columns), None)
        if resp_col:
            out["responded"] = (
                out[resp_col].astype(str).str.strip().str.upper().isin(["1", "Y", "TRUE", "YES"])
            )

        # --- opt-in status ---
        optin_col = next((c for c in ["opt_in", "수신동의"] if c in out.columns), None)
        if optin_col:
            out["opt_in"] = (
                out[optin_col].astype(str).str.strip().str.upper().isin(["1", "Y", "TRUE", "YES"])
            )

        # --- channel-wise response pivot ---
        if "channel" in out.columns and "responded" in out.columns:
            pivot = (
                out.groupby(["customer_id", "channel"])["responded"]
                .mean()
                .unstack(fill_value=0.0)
                .add_prefix("response_rate_")
                .reset_index()
            )
            out = out.merge(pivot, on="customer_id", how="left")

        return out
