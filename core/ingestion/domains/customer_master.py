"""
Customer Master ingestor.

Cleans and enriches the core customer demographics table: age grouping,
income bracket classification, and gender/occupation standardisation.
Korean financial column names: 고객번호, 연령, 성별, 직업코드, 연소득.
"""

from __future__ import annotations

import logging
from typing import List

import numpy as np
import pandas as pd

from ..base import AbstractDomainIngestor
from ..registry import DomainRegistry

logger = logging.getLogger(__name__)

_AGE_BINS = [0, 19, 29, 39, 49, 59, 69, np.inf]
_AGE_LABELS = ["~19", "20s", "30s", "40s", "50s", "60s", "70+"]

_INCOME_BINS = [0, 30_000_000, 50_000_000, 80_000_000, 120_000_000, np.inf]
_INCOME_LABELS = ["low", "mid_low", "mid", "mid_high", "high"]

_GENDER_MAP = {"M": "male", "F": "female", "1": "male", "2": "female"}


@DomainRegistry.register("customer_master")
class CustomerMasterIngestor(AbstractDomainIngestor):
    """Ingest and normalise customer demographics."""

    @property
    def source_name(self) -> str:
        return "customer_master"

    @property
    def required_columns(self) -> List[str]:
        return ["customer_id"]

    def ingest(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()

        # --- age group ---
        age_col = self._resolve_col(out, ["age", "연령"])
        if age_col:
            out["age"] = pd.to_numeric(out[age_col], errors="coerce")
            out["age_group"] = pd.cut(
                out["age"], bins=_AGE_BINS, labels=_AGE_LABELS, right=True
            )

        # --- income bracket ---
        income_col = self._resolve_col(out, ["annual_income", "연소득"])
        if income_col:
            out["annual_income"] = pd.to_numeric(out[income_col], errors="coerce")
            out["income_bracket"] = pd.cut(
                out["annual_income"], bins=_INCOME_BINS, labels=_INCOME_LABELS, right=True
            )

        # --- gender standardisation ---
        gender_col = self._resolve_col(out, ["gender", "성별"])
        if gender_col:
            out["gender"] = out[gender_col].astype(str).str.strip().map(_GENDER_MAP).fillna("unknown")

        # --- occupation code ---
        occ_col = self._resolve_col(out, ["occupation_code", "직업코드"])
        if occ_col:
            out["occupation_code"] = out[occ_col].astype(str).str.strip().str.upper()

        return out

    # ------------------------------------------------------------------
    @staticmethod
    def _resolve_col(df: pd.DataFrame, candidates: list[str]) -> str | None:
        for c in candidates:
            if c in df.columns:
                return c
        logger.warning("None of %s found in columns", candidates)
        return None
