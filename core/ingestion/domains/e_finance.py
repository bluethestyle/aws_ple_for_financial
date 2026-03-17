"""
E-Finance / Digital channel ingestor.

Derives digital channel usage, app login frequency, and digital transaction ratio.
Korean columns: 고객번호, 로그인일시, 채널유형, 디지털거래건수, 전체거래건수.
"""

from __future__ import annotations

import logging
from typing import List

import numpy as np
import pandas as pd

from ..base import AbstractDomainIngestor
from ..registry import DomainRegistry

logger = logging.getLogger(__name__)

_DIGITAL_CHANNELS = {"app", "mobile", "internet", "모바일", "인터넷뱅킹", "앱"}


@DomainRegistry.register("e_finance")
class EFinanceIngestor(AbstractDomainIngestor):
    """Ingest and aggregate digital / e-finance engagement data."""

    @property
    def source_name(self) -> str:
        return "e_finance"

    @property
    def required_columns(self) -> List[str]:
        return ["customer_id"]

    def ingest(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()

        # --- login timestamp parsing ---
        login_col = next((c for c in ["login_datetime", "로그인일시"] if c in out.columns), None)
        if login_col:
            out["login_datetime"] = pd.to_datetime(out[login_col], errors="coerce")

        # --- channel type standardisation ---
        chan_col = next((c for c in ["channel_type", "채널유형"] if c in out.columns), None)
        if chan_col:
            out["channel_type"] = out[chan_col].astype(str).str.strip().str.lower()
            out["is_digital"] = out["channel_type"].isin(_DIGITAL_CHANNELS)

        # --- customer-level digital engagement ---
        agg_dict: dict = {}
        if "login_datetime" in out.columns:
            agg_dict["login_count"] = ("login_datetime", "count")
            agg_dict["login_days"] = ("login_datetime", lambda s: s.dt.date.nunique())

        # digital transaction ratio
        digi_col = next((c for c in ["digital_txn_count", "디지털거래건수"] if c in out.columns), None)
        total_col = next((c for c in ["total_txn_count", "전체거래건수"] if c in out.columns), None)
        if digi_col and total_col:
            out["digital_txn_count"] = pd.to_numeric(out[digi_col], errors="coerce").fillna(0)
            out["total_txn_count"] = pd.to_numeric(out[total_col], errors="coerce").fillna(0)
            agg_dict["sum_digital_txn"] = ("digital_txn_count", "sum")
            agg_dict["sum_total_txn"] = ("total_txn_count", "sum")

        if agg_dict:
            summary = out.groupby("customer_id").agg(**agg_dict).reset_index()
            if "sum_digital_txn" in summary.columns and "sum_total_txn" in summary.columns:
                summary["digital_txn_ratio"] = np.where(
                    summary["sum_total_txn"] > 0,
                    summary["sum_digital_txn"] / summary["sum_total_txn"],
                    0.0,
                )
            out = out.merge(summary, on="customer_id", how="left")

        return out
