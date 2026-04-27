"""
Economic / Financial Behavior Feature Generator.

Computes income decomposition and financial behavior features from
transactional and account-level data.

Output per row (17D):
  Income Decomposition (8D):
    - salary_stability_index: std of monthly income / mean
    - income_trend_slope: linear regression slope over time
    - income_seasonality: amplitude of seasonal component via FFT
    - income_diversification: entropy across income sources
    - disposable_income_ratio: (income - fixed_expenses) / income
    - savings_rate: savings / income
    - income_volatility: coefficient of variation
    - income_growth_rate: recent vs historical income

  Financial Behavior (9D):
    - spending_elasticity: spending change / income change
    - budget_adherence: actual vs typical spending ratio
    - financial_stress_index: late payments + overdrafts normalised
    - liquidity_ratio: liquid assets / monthly expenses
    - debt_service_ratio: debt payments / income
    - investment_propensity: investment txn count / total txn count
    - digital_payment_ratio: digital txns / total txns
    - cash_preference_index: cash withdrawals / total spending
    - financial_planning_score: regular savings + investment auto-debit presence

Hardware acceleration
---------------------
Pure numpy implementation -- no GPU acceleration required.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from core.data.dataframe import df_backend
from ..generator import AbstractFeatureGenerator, FeatureGeneratorRegistry
from .gpu_utils import (
    _to_numpy_safe,
    _to_pandas_safe,
    _columns_list,
    _select_dtypes_numeric,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class EconomicsConfig:
    """Economics feature generator hyper-parameters."""

    income_decomposition_dim: int = 8
    financial_behavior_dim: int = 9


# ---------------------------------------------------------------------------
# Column-matching helpers
# ---------------------------------------------------------------------------

_INCOME_PATTERNS = re.compile(r"(income|salary|deposit|payment)", re.IGNORECASE)
_SPENDING_PATTERNS = re.compile(
    r"(spending|credit|loan|investment|digital|payment)", re.IGNORECASE
)


def _find_columns(df: Any, pattern: re.Pattern) -> List[str]:
    """Return column names matching *pattern*."""
    return [c for c in _columns_list(df) if pattern.search(c)]


def _safe_divide(numerator: np.ndarray, denominator: np.ndarray) -> np.ndarray:
    """Element-wise division with zero-safe fallback."""
    with np.errstate(divide="ignore", invalid="ignore"):
        result = np.where(
            np.abs(denominator) > 1e-12,
            numerator / denominator,
            0.0,
        )
    return np.nan_to_num(result, nan=0.0, posinf=0.0, neginf=0.0)


def _col_sum(df: Any, cols: List[str]) -> np.ndarray:
    """Row-wise sum of *cols*, returning zeros if none exist."""
    if not cols:
        return np.zeros(len(_to_pandas_safe(df)), dtype=np.float64)
    vals = _to_numpy_safe(df, cols)
    return np.nan_to_num(vals.sum(axis=1), nan=0.0)


def _col_mean(df: Any, cols: List[str]) -> np.ndarray:
    """Row-wise mean of *cols*, returning zeros if none exist."""
    if not cols:
        return np.zeros(len(_to_pandas_safe(df)), dtype=np.float64)
    vals = _to_numpy_safe(df, cols)
    return np.nan_to_num(vals.mean(axis=1), nan=0.0)


def _col_std(df: Any, cols: List[str]) -> np.ndarray:
    """Row-wise std of *cols*, returning zeros if none exist."""
    if not cols:
        return np.zeros(len(_to_pandas_safe(df)), dtype=np.float64)
    vals = _to_numpy_safe(df, cols)
    if vals.shape[1] < 2:
        return np.zeros(vals.shape[0], dtype=np.float64)
    return np.nan_to_num(np.nanstd(vals, axis=1), nan=0.0)


def _col_count_nonzero(df: Any, cols: List[str]) -> np.ndarray:
    """Row-wise count of non-zero values across *cols*."""
    if not cols:
        return np.zeros(len(_to_pandas_safe(df)), dtype=np.float64)
    vals = _to_numpy_safe(df, cols)
    return (np.abs(vals) > 1e-12).sum(axis=1).astype(np.float64)


# ---------------------------------------------------------------------------
# Economics Feature Generator
# ---------------------------------------------------------------------------

@FeatureGeneratorRegistry.register(
    "economics",
    description="Economic / financial behavior features (income decomposition + financial behavior).",
    tags=["economics", "financial", "behavior"],
)
class EconomicsFeatureGenerator(AbstractFeatureGenerator):
    """Economic / financial behavior feature generator.

    Produces 17 features capturing income decomposition (8D) and
    financial behavior patterns (9D) from transactional data.

    The generator scans input columns for income-related and
    spending-related patterns, then derives normalised ratios,
    indices, and scores.  Missing columns are handled gracefully
    by filling with zeros or derived heuristics.

    Parameters
    ----------
    config : EconomicsConfig, optional
        Generator hyper-parameters.
    prefix : str
        Column-name prefix for all output features.
    """

    supports_gpu: bool = False
    required_libraries: List[str] = []

    def __init__(
        self,
        config: Optional[EconomicsConfig] = None,
        prefix: str = "econ",
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.config = config or EconomicsConfig()
        self.prefix = prefix

        # Fitted state
        self._income_cols: List[str] = []
        self._spending_cols: List[str] = []
        self._stress_cols: List[str] = []
        self._liquid_cols: List[str] = []
        self._debt_cols: List[str] = []
        self._invest_cols: List[str] = []
        self._digital_cols: List[str] = []
        self._cash_cols: List[str] = []
        self._all_numeric_cols: List[str] = []
        self._global_income_mean: float = 0.0
        self._global_spending_mean: float = 0.0

    # -- Output description ------------------------------------------------

    @property
    def output_dim(self) -> int:
        return self.config.income_decomposition_dim + self.config.financial_behavior_dim

    @property
    def output_columns(self) -> List[str]:
        p = self.prefix
        return [
            # Income Decomposition (8D)
            f"{p}_salary_stability_index",
            f"{p}_income_trend_slope",
            f"{p}_income_seasonality",
            f"{p}_income_diversification",
            f"{p}_disposable_income_ratio",
            f"{p}_savings_rate",
            f"{p}_income_volatility",
            f"{p}_income_growth_rate",
            # Financial Behavior (9D)
            f"{p}_spending_elasticity",
            f"{p}_budget_adherence",
            f"{p}_financial_stress_index",
            f"{p}_liquidity_ratio",
            f"{p}_debt_service_ratio",
            f"{p}_investment_propensity",
            f"{p}_digital_payment_ratio",
            f"{p}_cash_preference_index",
            f"{p}_financial_planning_score",
        ]

    @classmethod
    def estimated_output_dim(cls, config: Dict[str, Any]) -> int:
        """Estimate output dimensionality from config dict."""
        return config.get("income_decomposition_dim", 8) + config.get(
            "financial_behavior_dim", 9
        )

    # -- Input columns declaration -----------------------------------------

    @property
    def input_cols(self) -> List[str]:
        """Source columns consumed by fit() and generate().

        Before fit(), returns the union of income and spending pattern
        matches seeded to empty lists.  After fit(), returns the full
        set of resolved columns including the dynamically discovered
        stress, liquid, debt, invest, digital, and cash groups.
        Derivable from fitted state with no new __init__ variables.
        """
        seen: Dict[str, None] = {}
        for c in self._income_cols:
            seen[c] = None
        for c in self._spending_cols:
            seen[c] = None
        # Add the ancillary groups stored at fit-time
        for group in (
            self._stress_cols,
            self._liquid_cols,
            self._debt_cols,
            self._invest_cols,
            self._digital_cols,
            self._cash_cols,
            self._all_numeric_cols,
        ):
            for c in group:
                seen[c] = None
        return list(seen.keys())

    # -- Core API ----------------------------------------------------------

    def fit(self, df: Any, **context: Any) -> "EconomicsFeatureGenerator":
        """Fit the generator by resolving relevant columns and global stats.

        Parameters
        ----------
        df : DataFrame
            Training data (pandas, cuDF, or any backend-native type).
        """
        self._income_cols = _find_columns(df, _INCOME_PATTERNS)
        self._spending_cols = _find_columns(df, _SPENDING_PATTERNS)

        logger.info(
            "EconomicsFeatureGenerator.fit: found %d income columns, %d spending columns",
            len(self._income_cols),
            len(self._spending_cols),
        )

        # Discover and cache all ancillary column groups at fit-time so that
        # input_cols is derivable without re-scanning the DataFrame.
        all_cols = _columns_list(df)
        self._stress_cols: List[str] = [
            c for c in all_cols
            if re.search(r"(late|overdue|overdraft|penalty|delinq)", c, re.IGNORECASE)
        ]
        self._liquid_cols: List[str] = [
            c for c in all_cols
            if re.search(r"(liquid|cash|saving|balance|deposit)", c, re.IGNORECASE)
        ]
        self._debt_cols: List[str] = [
            c for c in all_cols
            if re.search(r"(debt|loan|mortgage|emi|installment)", c, re.IGNORECASE)
        ]
        self._invest_cols: List[str] = [
            c for c in all_cols
            if re.search(r"(invest|mutual|stock|bond|fund)", c, re.IGNORECASE)
        ]
        self._digital_cols: List[str] = [
            c for c in all_cols
            if re.search(r"(digital|online|upi|card|electronic|mobile)", c, re.IGNORECASE)
        ]
        self._cash_cols: List[str] = [
            c for c in all_cols
            if re.search(r"(cash|atm|withdraw)", c, re.IGNORECASE)
        ]
        self._all_numeric_cols: List[str] = _select_dtypes_numeric(df)

        # Cache global means for normalisation / fallback heuristics
        income_vals = _col_sum(df, self._income_cols)
        spending_vals = _col_sum(df, self._spending_cols)

        self._global_income_mean = float(np.nanmean(income_vals)) if len(income_vals) > 0 else 0.0
        self._global_spending_mean = float(np.nanmean(spending_vals)) if len(spending_vals) > 0 else 0.0

        self._fitted = True
        return self

    def generate(self, df: Any, **context: Any) -> Any:
        """Generate economic / financial behavior features.

        Parameters
        ----------
        df : DataFrame
            Input data.

        Returns
        -------
        DataFrame
            17-column DataFrame with the same index as *df*.
        """
        if not self._fitted:
            raise RuntimeError(
                "EconomicsFeatureGenerator must be fitted before generate()."
            )

        # Extract all needed columns once as numpy arrays.
        col_arrays = self._input_to_numpy(df, columns=self.input_cols)

        # Derive row count and index from the slim extracted frame.
        n = len(next(iter(col_arrays.values()))) if col_arrays else 0
        _pdf_ref = _to_pandas_safe(df)
        _index = _pdf_ref.index
        del _pdf_ref  # free reference after extracting index
        p = self.prefix

        # Resolve which fitted columns are present in this batch.
        present = set(col_arrays.keys())
        income_cols = [c for c in self._income_cols if c in present]
        spending_cols = [c for c in self._spending_cols if c in present]
        stress_cols = [c for c in self._stress_cols if c in present]
        liquid_cols = [c for c in self._liquid_cols if c in present]
        debt_cols = [c for c in self._debt_cols if c in present]
        invest_cols = [c for c in self._invest_cols if c in present]
        digital_cols = [c for c in self._digital_cols if c in present]
        cash_cols = [c for c in self._cash_cols if c in present]
        all_numeric_cols = [c for c in self._all_numeric_cols if c in present]

        # ------------------------------------------------------------------
        # Helper closures that operate on col_arrays instead of df
        # ------------------------------------------------------------------
        def _arr_sum(cols: List[str]) -> np.ndarray:
            if not cols:
                return np.zeros(n, dtype=np.float64)
            vals = np.stack([col_arrays[c].astype(np.float64) for c in cols], axis=1)
            return np.nan_to_num(vals.sum(axis=1), nan=0.0)

        def _arr_mean(cols: List[str]) -> np.ndarray:
            if not cols:
                return np.zeros(n, dtype=np.float64)
            vals = np.stack([col_arrays[c].astype(np.float64) for c in cols], axis=1)
            return np.nan_to_num(vals.mean(axis=1), nan=0.0)

        def _arr_std(cols: List[str]) -> np.ndarray:
            if not cols:
                return np.zeros(n, dtype=np.float64)
            vals = np.stack([col_arrays[c].astype(np.float64) for c in cols], axis=1)
            if vals.shape[1] < 2:
                return np.zeros(n, dtype=np.float64)
            return np.nan_to_num(np.nanstd(vals, axis=1), nan=0.0)

        def _arr_count_nonzero(cols: List[str]) -> np.ndarray:
            if not cols:
                return np.zeros(n, dtype=np.float64)
            vals = np.stack([col_arrays[c].astype(np.float64) for c in cols], axis=1)
            return (np.abs(vals) > 1e-12).sum(axis=1).astype(np.float64)

        def _arr_matrix(cols: List[str]) -> np.ndarray:
            if not cols:
                return np.zeros((n, 0), dtype=np.float64)
            return np.stack([col_arrays[c].astype(np.float64) for c in cols], axis=1)

        # Aggregate row-level signals
        income_sum = _arr_sum(income_cols)
        income_mean = _arr_mean(income_cols)
        income_std = _arr_std(income_cols)
        spending_sum = _arr_sum(spending_cols)

        results: Dict[str, np.ndarray] = {}

        # =================================================================
        # Income Decomposition (8D)
        # =================================================================

        # 1. Salary stability index: std(monthly income) / mean
        results[f"{p}_salary_stability_index"] = _safe_divide(
            income_std, income_mean
        ).astype(np.float32)

        # 2. Income trend slope: approximate via (last - first) / count
        #    With row-level data we use a proxy: difference between the
        #    last income column value and the first income column value
        #    normalised by the number of income columns.
        if len(income_cols) >= 2:
            first_vals = col_arrays[income_cols[0]].astype(np.float64)
            last_vals = col_arrays[income_cols[-1]].astype(np.float64)
            slope = _safe_divide(last_vals - first_vals, np.full(n, len(income_cols), dtype=np.float64))
        else:
            slope = np.zeros(n, dtype=np.float64)
        results[f"{p}_income_trend_slope"] = slope.astype(np.float32)

        # 3. Income seasonality: amplitude of dominant FFT component
        if len(income_cols) >= 4:
            income_matrix = _arr_matrix(income_cols)
            fft_vals = np.fft.rfft(income_matrix, axis=1)
            # Skip DC component (index 0), take max amplitude
            if fft_vals.shape[1] > 1:
                amplitudes = np.abs(fft_vals[:, 1:])
                seasonality = amplitudes.max(axis=1)
            else:
                seasonality = np.zeros(n, dtype=np.float64)
        else:
            seasonality = np.zeros(n, dtype=np.float64)
        results[f"{p}_income_seasonality"] = seasonality.astype(np.float32)

        # 4. Income diversification: entropy across income sources
        if len(income_cols) >= 2:
            income_matrix = np.abs(_arr_matrix(income_cols))
            row_totals = income_matrix.sum(axis=1, keepdims=True)
            row_totals = np.where(row_totals < 1e-12, 1.0, row_totals)
            probs = income_matrix / row_totals
            entropy = -np.sum(
                np.where(probs > 1e-12, probs * np.log(probs + 1e-300), 0.0),
                axis=1,
            )
        else:
            entropy = np.zeros(n, dtype=np.float64)
        results[f"{p}_income_diversification"] = entropy.astype(np.float32)

        # 5. Disposable income ratio: (income - fixed_expenses) / income
        #    Heuristic: fixed expenses ~ 50% of spending when no explicit column
        fixed_expenses = spending_sum * 0.5
        results[f"{p}_disposable_income_ratio"] = _safe_divide(
            income_sum - fixed_expenses, income_sum
        ).astype(np.float32)

        # 6. Savings rate: savings / income
        #    Heuristic: savings = income - spending
        savings = np.maximum(income_sum - spending_sum, 0.0)
        results[f"{p}_savings_rate"] = _safe_divide(savings, income_sum).astype(np.float32)

        # 7. Income volatility: coefficient of variation
        results[f"{p}_income_volatility"] = _safe_divide(
            income_std, np.abs(income_mean)
        ).astype(np.float32)

        # 8. Income growth rate: recent vs historical
        #    Proxy: second half mean / first half mean - 1
        if len(income_cols) >= 2:
            mid = len(income_cols) // 2
            first_half = _arr_matrix(income_cols[:mid]).mean(axis=1)
            second_half = _arr_matrix(income_cols[mid:]).mean(axis=1)
            growth = _safe_divide(second_half - first_half, np.abs(first_half))
        else:
            growth = np.zeros(n, dtype=np.float64)
        results[f"{p}_income_growth_rate"] = growth.astype(np.float32)

        # =================================================================
        # Financial Behavior (9D)
        # =================================================================

        # 1. Spending elasticity: spending change / income change
        #    Proxy: (spending - global_mean_spending) / (income - global_mean_income)
        income_change = income_sum - self._global_income_mean
        spending_change = spending_sum - self._global_spending_mean
        results[f"{p}_spending_elasticity"] = _safe_divide(
            spending_change, income_change
        ).astype(np.float32)

        # 2. Budget adherence: actual vs typical spending ratio
        results[f"{p}_budget_adherence"] = _safe_divide(
            spending_sum,
            np.full(n, max(self._global_spending_mean, 1e-12), dtype=np.float64),
        ).astype(np.float32)

        # 3. Financial stress index: late payments + overdrafts normalised
        stress_sum = _arr_sum(stress_cols)
        # Normalise by income
        results[f"{p}_financial_stress_index"] = _safe_divide(
            stress_sum, income_sum
        ).astype(np.float32)

        # 4. Liquidity ratio: liquid assets / monthly expenses
        liquid_sum = _arr_sum(liquid_cols)
        monthly_expenses = spending_sum  # proxy
        results[f"{p}_liquidity_ratio"] = _safe_divide(
            liquid_sum, monthly_expenses
        ).astype(np.float32)

        # 5. Debt service ratio: debt payments / income
        debt_sum = _arr_sum(debt_cols)
        results[f"{p}_debt_service_ratio"] = _safe_divide(
            debt_sum, income_sum
        ).astype(np.float32)

        # 6. Investment propensity: investment txn count / total txn count
        invest_count = _arr_count_nonzero(invest_cols)
        total_count = _arr_count_nonzero(
            list(dict.fromkeys(income_cols + spending_cols + invest_cols))
        )
        results[f"{p}_investment_propensity"] = _safe_divide(
            invest_count, total_count
        ).astype(np.float32)

        # 7. Digital payment ratio: digital txns / total txns
        digital_count = _arr_count_nonzero(digital_cols)
        all_txn_count = _arr_count_nonzero(all_numeric_cols)
        results[f"{p}_digital_payment_ratio"] = _safe_divide(
            digital_count, all_txn_count
        ).astype(np.float32)

        # 8. Cash preference index: cash withdrawals / total spending
        cash_sum = _arr_sum(cash_cols)
        results[f"{p}_cash_preference_index"] = _safe_divide(
            cash_sum, spending_sum
        ).astype(np.float32)

        # 9. Financial planning score: regular savings + investment auto-debit presence
        #    Composite: 0.5 * (savings_rate > 0) + 0.5 * (investment_propensity > 0)
        has_savings = (savings > 1e-12).astype(np.float64) * 0.5
        has_invest = (invest_count > 0).astype(np.float64) * 0.5
        results[f"{p}_financial_planning_score"] = (has_savings + has_invest).astype(np.float32)

        return df_backend.from_dict(results, index=_index)
