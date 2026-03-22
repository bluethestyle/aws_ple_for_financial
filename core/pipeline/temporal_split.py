"""
Temporal Splitter
=================

Time-based train/val/test split with configurable gap periods for leakage
prevention.  Also provides sequence truncation for user-level datasets
where sequence columns may extend into the prediction window.

The Santander dataset has 17-month product holding sequences where month 17
is the label month.  The ``split_by_sequence_cutoff`` method truncates these
sequences to months 1-16 and recomputes snapshot-level product columns from
the truncated state, eliminating temporal leakage.

Usage::

    from core.pipeline.temporal_split import TemporalSplitter

    splitter = TemporalSplitter(train_ratio=0.7, val_ratio=0.15, gap_days=30)
    train_df, val_df, test_df = splitter.split(df, date_col="snapshot_date")

    # For user-level data with sequences:
    df_clean = splitter.split_by_sequence_cutoff(df, seq_cols, cutoff_offset=1)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

__all__ = ["TemporalSplitter", "TemporalSplitConfig"]


@dataclass
class TemporalSplitConfig:
    """Configuration for temporal splitting.

    Parameters
    ----------
    enabled : bool
        Whether temporal split is enabled.  If False, falls back to random.
    date_col : str
        Column name containing datetime/date values for ordering.
    gap_days : int
        Number of days gap between train/val and val/test to prevent leakage.
    train_ratio : float
        Fraction of data for training (by temporal ordering).
    val_ratio : float
        Fraction of data for validation.  Remainder goes to test.
    """

    enabled: bool = False
    date_col: str = "snapshot_date"
    gap_days: int = 7
    train_ratio: float = 0.7
    val_ratio: float = 0.15

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "TemporalSplitConfig":
        """Create from a plain dict (e.g. from YAML config)."""
        return cls(
            enabled=d.get("enabled", False),
            date_col=d.get("date_col", "snapshot_date"),
            gap_days=d.get("gap_days", 7),
            train_ratio=d.get("train_ratio", 0.7),
            val_ratio=d.get("val_ratio", 0.15),
        )


class TemporalSplitter:
    """Time-based train/val/test split with gap days for leakage prevention.

    For tabular datasets with a date column, rows are sorted by date and
    split into contiguous temporal segments.  A configurable gap (in days)
    between segments ensures that no future information leaks into training.

    For user-level datasets without a meaningful date ordering (e.g. one row
    per user with sequence columns), use ``split_by_sequence_cutoff`` to
    truncate sequences and ``split_users_temporal`` to split by the user's
    last observed date.

    Parameters
    ----------
    train_ratio : float
        Fraction of data for training.
    val_ratio : float
        Fraction of data for validation.
    gap_days : int
        Number of days gap between splits.
    """

    def __init__(
        self,
        train_ratio: float = 0.7,
        val_ratio: float = 0.15,
        gap_days: int = 7,
    ) -> None:
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.gap_days = gap_days

    # ------------------------------------------------------------------
    # Date-based split (transaction-level or snapshot-level data)
    # ------------------------------------------------------------------

    def split(
        self,
        df: pd.DataFrame,
        date_col: str = "snapshot_date",
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Split by temporal ordering with gap between splits.

        Parameters
        ----------
        df : pd.DataFrame
            Input data with a date column.
        date_col : str
            Column name containing datetime values.

        Returns
        -------
        tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]
            ``(train_df, val_df, test_df)`` in temporal order.
        """
        if date_col not in df.columns:
            logger.warning(
                "TemporalSplitter: date_col '%s' not found. "
                "Falling back to index-based split.",
                date_col,
            )
            return self._split_by_index(df)

        # Sort by date
        df_sorted = df.sort_values(date_col).reset_index(drop=True)
        dates = pd.to_datetime(df_sorted[date_col])

        min_date = dates.min()
        max_date = dates.max()
        total_span = (max_date - min_date).days

        if total_span <= 0:
            logger.warning(
                "TemporalSplitter: all dates identical, falling back to "
                "index-based split."
            )
            return self._split_by_index(df_sorted)

        gap = pd.Timedelta(days=self.gap_days)

        # Compute cutoff dates
        train_end_date = min_date + pd.Timedelta(
            days=int(total_span * self.train_ratio)
        )
        val_start_date = train_end_date + gap
        val_end_date = val_start_date + pd.Timedelta(
            days=int(total_span * self.val_ratio)
        )
        test_start_date = val_end_date + gap

        train_mask = dates <= train_end_date
        val_mask = (dates >= val_start_date) & (dates <= val_end_date)
        test_mask = dates >= test_start_date

        train_df = df_sorted[train_mask].reset_index(drop=True)
        val_df = df_sorted[val_mask].reset_index(drop=True)
        test_df = df_sorted[test_mask].reset_index(drop=True)

        logger.info(
            "TemporalSplitter: train=%d (<=  %s), val=%d (%s - %s), "
            "test=%d (>= %s), gap=%d days, discarded=%d rows in gaps",
            len(train_df), train_end_date.date(),
            len(val_df), val_start_date.date(), val_end_date.date(),
            len(test_df), test_start_date.date(),
            self.gap_days,
            len(df_sorted) - len(train_df) - len(val_df) - len(test_df),
        )

        return train_df, val_df, test_df

    # ------------------------------------------------------------------
    # User-level split by last observed date
    # ------------------------------------------------------------------

    def split_users_temporal(
        self,
        df: pd.DataFrame,
        date_col: str = "snapshot_date",
        user_col: str = "customer_id",
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Split user-level data by each user's snapshot date.

        For datasets with one row per user (like Santander), this splits
        users into train/val/test based on their snapshot_date ordering.

        Parameters
        ----------
        df : pd.DataFrame
            User-level data with date and user columns.
        date_col : str
            Column containing the observation date.
        user_col : str
            Column containing user identifiers.

        Returns
        -------
        tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]
            ``(train_df, val_df, test_df)``
        """
        if date_col not in df.columns:
            logger.warning(
                "TemporalSplitter: date_col '%s' not found for user split. "
                "Falling back to index-based split.",
                date_col,
            )
            return self._split_by_index(df)

        return self.split(df, date_col=date_col)

    # ------------------------------------------------------------------
    # Sequence truncation (key fix for Santander leakage)
    # ------------------------------------------------------------------

    def split_by_sequence_cutoff(
        self,
        df: pd.DataFrame,
        seq_cols: List[str],
        cutoff_offset: int = 1,
        prod_col_prefix: str = "prod_",
        seq_col_prefix: str = "seq_",
    ) -> pd.DataFrame:
        """Truncate sequences to exclude the last N months (leakage fix).

        For user-level data where sequence columns contain the label month:
        - Truncate ``seq_*`` columns to drop the last ``cutoff_offset`` elements
        - Recompute ``prod_*`` columns from the truncated sequence's last element
        - ``nba_label`` stays as-is (it IS the month 17 target)

        This is the **critical fix** for the Santander dataset where:
        - ``seq_*`` columns contain all 17 months INCLUDING month 17 (label)
        - ``prod_*`` columns reflect month 17 holdings (the label state)

        After truncation:
        - ``seq_*`` columns contain months 1-16 only
        - ``prod_*`` columns reflect month 16 state (pre-label)

        Parameters
        ----------
        df : pd.DataFrame
            User-level data with list-valued sequence columns.
        seq_cols : list[str]
            Names of sequence columns to truncate.
        cutoff_offset : int
            Number of elements to drop from the end of each sequence.
        prod_col_prefix : str
            Prefix for product holding columns to recompute.
        seq_col_prefix : str
            Prefix for sequence columns (used for prod recomputation).

        Returns
        -------
        pd.DataFrame
            DataFrame with truncated sequences and recomputed products.
        """
        df = df.copy()
        truncated_count = 0
        recomputed_count = 0

        for col in seq_cols:
            if col not in df.columns:
                logger.warning(
                    "TemporalSplitter: sequence column '%s' not found, skipping",
                    col,
                )
                continue

            # Truncate sequence: drop last cutoff_offset elements
            def _truncate(seq):
                if isinstance(seq, (list, np.ndarray)):
                    if len(seq) > cutoff_offset:
                        return list(seq[:-cutoff_offset])
                    return list(seq)
                return seq

            df[col] = df[col].apply(_truncate)
            truncated_count += 1

            # Recompute corresponding prod_* column from truncated sequence
            # seq_saving -> prod_saving, seq_checking -> prod_checking, etc.
            if col.startswith(seq_col_prefix):
                prod_col = prod_col_prefix + col[len(seq_col_prefix):]
                if prod_col in df.columns:
                    def _last_element(seq):
                        if isinstance(seq, (list, np.ndarray)) and len(seq) > 0:
                            return int(seq[-1])
                        return 0

                    df[prod_col] = df[col].apply(_last_element)
                    recomputed_count += 1

        logger.info(
            "TemporalSplitter: truncated %d sequence columns by %d elements, "
            "recomputed %d product columns from truncated sequences",
            truncated_count, cutoff_offset, recomputed_count,
        )

        return df

    # ------------------------------------------------------------------
    # Index-based fallback split
    # ------------------------------------------------------------------

    def _split_by_index(
        self,
        df: pd.DataFrame,
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Fallback: split by row index position (preserving order).

        Used when no date column is available.  No gap is applied since
        there is no temporal ordering to gap over.

        Parameters
        ----------
        df : pd.DataFrame
            Input data.

        Returns
        -------
        tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]
        """
        n = len(df)
        n_train = int(n * self.train_ratio)
        n_val = int(n * self.val_ratio)

        train_df = df.iloc[:n_train].reset_index(drop=True)
        val_df = df.iloc[n_train:n_train + n_val].reset_index(drop=True)
        test_df = df.iloc[n_train + n_val:].reset_index(drop=True)

        logger.info(
            "TemporalSplitter (index-based): train=%d, val=%d, test=%d",
            len(train_df), len(val_df), len(test_df),
        )

        return train_df, val_df, test_df
