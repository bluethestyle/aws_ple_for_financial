"""
G6 Management code ingestor.

Maps branch codes to regions and standardises management hierarchy codes.
Korean columns: 관리점코드, 지역코드, 영업부코드.
"""

from __future__ import annotations

import logging
from typing import List

import pandas as pd

from ..base import AbstractDomainIngestor
from ..registry import DomainRegistry

logger = logging.getLogger(__name__)

_REGION_MAP = {
    "01": "seoul",
    "02": "gyeonggi",
    "03": "incheon",
    "04": "busan",
    "05": "daegu",
    "06": "daejeon",
    "07": "gwangju",
    "08": "ulsan",
    "09": "sejong",
    "10": "gangwon",
    "11": "chungbuk",
    "12": "chungnam",
    "13": "jeonbuk",
    "14": "jeonnam",
    "15": "gyeongbuk",
    "16": "gyeongnam",
    "17": "jeju",
}


@DomainRegistry.register("management")
class G6ManagementIngestor(AbstractDomainIngestor):
    """Ingest and normalise management / branch metadata."""

    @property
    def source_name(self) -> str:
        return "management"

    @property
    def required_columns(self) -> List[str]:
        return ["branch_code"]

    def ingest(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()

        # --- branch code standardisation (zero-pad to 4 digits) ---
        branch_col = next((c for c in ["branch_code", "관리점코드"] if c in out.columns), None)
        if branch_col:
            out["branch_code"] = out[branch_col].astype(str).str.strip().str.zfill(4)

        # --- region mapping (first 2 digits of branch code) ---
        out["region_code"] = out["branch_code"].str[:2]
        out["region_name"] = out["region_code"].map(_REGION_MAP).fillna("unknown")

        # --- management division code ---
        div_col = next((c for c in ["division_code", "영업부코드"] if c in out.columns), None)
        if div_col:
            out["division_code"] = out[div_col].astype(str).str.strip().str.upper()
        else:
            logger.warning("Division code column not found")

        return out
