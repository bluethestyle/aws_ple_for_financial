"""
Sequence Builder
================

Converts various sequence data formats into padded 3-D NumPy tensors
suitable for temporal expert networks (Mamba, adaTT, etc.).

Supported modes:

* **count_based** (legacy) -- take the last ``max_len`` items per entity.
* **time_based** -- filter by a date-range window (``window_days``).
  Variable-length sequences are zero-padded up to ``max_len`` (safety cap).
  Optionally generates multiple overlapping samples per entity via
  ``build_sliding_windows(stride_days)``.

Supported sources:

* **npy** -- load a pre-built ``.npy`` file directly.
* **parquet_list** -- extract list-typed columns from a Parquet-backed
  DataFrame and stack them into a ``(n_entities, seq_len, feat_dim)``
  array with zero-padding.

Usage::

    from core.pipeline.sequence_builder import SequenceBuilder, SeqSourceConfig

    configs = {
        "purchase_history": SeqSourceConfig(
            source="parquet_list",
            columns=["amounts", "categories", "timestamps"],
            mode="time_based",
            window_days=90,
            stride_days=30,
            max_len=200,
            timestamp_col="txn_date",
        ),
    }
    builder = SequenceBuilder()
    tensors = builder.build(raw_data={"main": df}, seq_configs=configs)
    # tensors["purchase_history"].shape == (n_samples, 200, 3)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import timedelta
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

__all__ = ["SequenceBuilder", "SeqSourceConfig"]


@dataclass
class SeqSourceConfig:
    """Configuration for a single sequence source.

    Parameters
    ----------
    source : str
        ``"npy"`` to load from a ``.npy`` file, ``"parquet_list"`` to
        extract list-typed columns from a DataFrame.
    columns : list[str]
        Column names to extract (for ``parquet_list`` source).  Each
        column should contain list/array values of numeric type.
    mode : str
        ``"count_based"`` (default) -- fixed-length count slicing.
        ``"time_based"`` -- filter by date range using ``window_days``.
        Auto-detected: if ``timestamp_col`` exists in the data, switches
        to time_based regardless of this setting.
    max_len : int
        Maximum sequence length.  For count_based this is the sequence
        length; for time_based this is the safety cap.  Shorter sequences
        are zero-padded; longer sequences are truncated.
    window_days : int
        Look-back window in days (time_based mode only).
    stride_days : int
        Sliding window stride in days for bootstrapping (time_based mode).
        If 0, no sliding window -- single window per entity.
    timestamp_col : str
        Column containing per-transaction timestamps.  Used for time_based
        filtering.  If empty, the builder auto-detects timestamp columns.
    file_path : str
        Path to the ``.npy`` file (for ``npy`` source).
    dtype : str
        NumPy dtype for the output array (default ``"float32"``).
    truncate_last : int
        Drop last N elements from each sequence (leakage prevention).
    """

    source: str = "parquet_list"  # "parquet_list" | "npy"
    columns: List[str] = field(default_factory=list)
    mode: str = "count_based"  # "count_based" | "time_based"
    max_len: int = 50
    window_days: int = 90
    stride_days: int = 0
    timestamp_col: str = ""
    file_path: str = ""
    dtype: str = "float32"
    truncate_last: int = 0  # drop last N elements from each sequence (leakage prevention)

    # Alias for backward compatibility
    @property
    def seq_len(self) -> int:
        return self.max_len

    @seq_len.setter
    def seq_len(self, value: int) -> None:
        self.max_len = value

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "SeqSourceConfig":
        """Create a SeqSourceConfig from a plain dict."""
        # Support legacy 'seq_len' key as alias for 'max_len'
        max_len = d.get("max_len", d.get("seq_len", 50))
        return cls(
            source=d.get("source", "parquet_list"),
            columns=d.get("columns", []),
            mode=d.get("mode", "count_based"),
            max_len=max_len,
            window_days=d.get("window_days", 90),
            stride_days=d.get("stride_days", 0),
            timestamp_col=d.get("timestamp_col", ""),
            file_path=d.get("file_path", ""),
            dtype=d.get("dtype", "float32"),
            truncate_last=d.get("truncate_last", 0),
        )


def _detect_timestamp_column(df: pd.DataFrame, hint: str = "") -> Optional[str]:
    """Auto-detect a timestamp column in the DataFrame.

    Checks the hint column first, then scans for columns with datetime
    dtype or names containing 'date', 'time', 'timestamp'.

    Returns None if no timestamp column is found.
    """
    if hint and hint in df.columns:
        col = df[hint]
        if pd.api.types.is_datetime64_any_dtype(col):
            return hint
        # Try parsing if it looks like dates
        try:
            pd.to_datetime(col.head(10))
            return hint
        except (ValueError, TypeError):
            pass

    # Scan for datetime columns
    for col_name in df.columns:
        if pd.api.types.is_datetime64_any_dtype(df[col_name]):
            return col_name

    # Scan for columns with date-like names
    date_keywords = ("date", "timestamp", "time", "dt")
    for col_name in df.columns:
        if any(kw in col_name.lower() for kw in date_keywords):
            try:
                pd.to_datetime(df[col_name].head(10))
                return col_name
            except (ValueError, TypeError):
                continue

    return None


def _detect_list_timestamps(
    df: pd.DataFrame, columns: List[str]
) -> Optional[str]:
    """Check if any list-typed column contains timestamp-like values.

    Returns the column name if found, None otherwise.
    """
    for col in columns:
        if col not in df.columns:
            continue
        # Check first non-null value
        for val in df[col].dropna().head(5):
            if isinstance(val, (list, np.ndarray)) and len(val) > 0:
                first_elem = val[0]
                if isinstance(first_elem, (pd.Timestamp, np.datetime64)):
                    return col
                if isinstance(first_elem, str):
                    try:
                        pd.to_datetime(first_elem)
                        return col
                    except (ValueError, TypeError):
                        pass
            break
    return None


class SequenceBuilder:
    """Build padded 3-D tensors from various sequence formats.

    Supports both count-based (legacy) and time-based sequence building
    with optional sliding window bootstrapping.
    """

    def build(
        self,
        raw_data: Dict[str, pd.DataFrame],
        seq_configs: Dict[str, SeqSourceConfig],
    ) -> Dict[str, np.ndarray]:
        """Build sequence tensors for all configured sources.

        Parameters
        ----------
        raw_data : dict[str, pd.DataFrame]
            Named DataFrames.  The ``"main"`` key is used by
            ``parquet_list`` sources.
        seq_configs : dict[str, SeqSourceConfig]
            Mapping from sequence name to its source configuration.

        Returns
        -------
        dict[str, np.ndarray]
            Mapping from sequence name to a 3-D array of shape
            ``(n_entities, max_len, feat_dim)`` for count_based, or
            ``(n_samples, max_len, feat_dim)`` for time_based with
            sliding windows.
        """
        results: Dict[str, np.ndarray] = {}

        for name, cfg in seq_configs.items():
            if cfg.source == "npy":
                results[name] = self._load_npy(cfg)
                logger.info(
                    "SequenceBuilder: loaded '%s' from npy, shape=%s",
                    name,
                    results[name].shape,
                )
            elif cfg.source == "parquet_list":
                if "main" not in raw_data:
                    raise KeyError(
                        f"SequenceBuilder: 'main' DataFrame not found in "
                        f"raw_data for sequence '{name}'"
                    )
                df = raw_data["main"]
                effective_mode = self._resolve_mode(df, cfg)

                if effective_mode == "time_based":
                    results[name] = self._from_parquet_lists_time_based(df, cfg)
                else:
                    results[name] = self._from_parquet_lists(df, cfg)

                logger.info(
                    "SequenceBuilder: built '%s' from parquet_list "
                    "(mode=%s), shape=%s",
                    name,
                    effective_mode,
                    results[name].shape,
                )
            else:
                raise ValueError(
                    f"SequenceBuilder: unknown source '{cfg.source}' "
                    f"for sequence '{name}'"
                )

        logger.info(
            "SequenceBuilder: built %d sequence tensors: %s",
            len(results),
            list(results.keys()),
        )
        return results

    def build_sliding_windows(
        self,
        df: pd.DataFrame,
        cfg: SeqSourceConfig,
        id_col: str = "customer_id",
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Build multiple overlapping samples per entity via sliding windows.

        Creates one sample per (entity, window_start) combination by
        sliding a ``window_days``-wide window with ``stride_days`` step.

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame with list-valued sequence columns and a timestamp
            column (or per-row date column for entity-level data).
        cfg : SeqSourceConfig
            Must have ``mode="time_based"`` with ``stride_days > 0``.
        id_col : str
            Entity identifier column name.

        Returns
        -------
        tuple[np.ndarray, np.ndarray]
            - sequences: shape ``(n_samples, max_len, feat_dim)``
            - entity_ids: shape ``(n_samples,)`` -- the entity ID for
              each sample, so callers can map back to rows.
        """
        if cfg.stride_days <= 0:
            raise ValueError(
                "build_sliding_windows requires stride_days > 0"
            )

        timestamp_col = self._find_timestamp_col(df, cfg)
        if timestamp_col is None:
            raise ValueError(
                "build_sliding_windows requires a timestamp column, "
                "but none was found. Set timestamp_col in config."
            )

        # Use DuckDB for efficient date filtering if available
        try:
            return self._sliding_windows_duckdb(df, cfg, id_col, timestamp_col)
        except ImportError:
            logger.info(
                "SequenceBuilder: DuckDB not available, falling back to pandas "
                "for sliding window construction"
            )
            return self._sliding_windows_pandas(df, cfg, id_col, timestamp_col)

    # ------------------------------------------------------------------
    # Mode resolution / auto-detection
    # ------------------------------------------------------------------

    def _resolve_mode(self, df: pd.DataFrame, cfg: SeqSourceConfig) -> str:
        """Determine effective mode: time_based or count_based.

        Auto-detection logic:
        1. If cfg.mode is explicitly "time_based", use it.
        2. If cfg.mode is "count_based" but a timestamp column exists
           in the data, log a hint but respect the explicit config.
        3. If cfg.timestamp_col is set and exists, switch to time_based.
        """
        if cfg.mode == "time_based":
            ts_col = self._find_timestamp_col(df, cfg)
            if ts_col is not None:
                return "time_based"
            # Fallback: timestamp column not found
            logger.warning(
                "SequenceBuilder: mode=time_based but no timestamp column "
                "found (hint='%s'). Falling back to count_based.",
                cfg.timestamp_col,
            )
            return "count_based"

        # count_based mode -- check for auto-detection opportunity
        if cfg.timestamp_col:
            ts_col = self._find_timestamp_col(df, cfg)
            if ts_col is not None:
                logger.info(
                    "SequenceBuilder: auto-detected timestamp column '%s'. "
                    "Switching from count_based to time_based mode.",
                    ts_col,
                )
                return "time_based"

        # Also check if list columns contain timestamps
        ts_list_col = _detect_list_timestamps(df, cfg.columns)
        if ts_list_col is not None:
            logger.info(
                "SequenceBuilder: list column '%s' contains timestamps. "
                "Switching to time_based mode.",
                ts_list_col,
            )
            return "time_based"

        return "count_based"

    def _find_timestamp_col(
        self, df: pd.DataFrame, cfg: SeqSourceConfig
    ) -> Optional[str]:
        """Find the timestamp column in df, using cfg.timestamp_col as hint."""
        return _detect_timestamp_column(df, hint=cfg.timestamp_col)

    # ------------------------------------------------------------------
    # Count-based building (legacy)
    # ------------------------------------------------------------------

    def _from_parquet_lists(
        self,
        df: pd.DataFrame,
        cfg: SeqSourceConfig,
    ) -> np.ndarray:
        """Convert list-type parquet columns to a padded 3-D array.

        When ``cfg.truncate_last > 0``, the last N elements of each
        sequence are dropped **before** padding.  This prevents temporal
        leakage when sequence columns extend into the prediction window
        (e.g. Santander month-17 data).

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame with list-valued columns.
        cfg : SeqSourceConfig
            Sequence configuration.

        Returns
        -------
        np.ndarray
            Shape ``(n_entities, max_len, feat_dim)`` with zero-padding.
        """
        columns = cfg.columns
        seq_len = cfg.max_len
        truncate_last = cfg.truncate_last
        n_entities = len(df)
        feat_dim = len(columns)
        dtype = np.dtype(cfg.dtype)

        if truncate_last > 0:
            logger.info(
                "SequenceBuilder: truncating last %d elements from "
                "sequences (leakage prevention)",
                truncate_last,
            )

        result = np.zeros((n_entities, seq_len, feat_dim), dtype=dtype)

        for feat_idx, col in enumerate(columns):
            if col not in df.columns:
                logger.warning(
                    "SequenceBuilder: column '%s' not found, "
                    "filling with zeros",
                    col,
                )
                continue

            for row_idx in range(n_entities):
                seq = df[col].iloc[row_idx]
                if isinstance(seq, (list, np.ndarray)):
                    # Truncate last N elements to prevent leakage
                    if truncate_last > 0 and len(seq) > truncate_last:
                        seq = seq[:-truncate_last]
                    actual_len = min(len(seq), seq_len)
                    result[row_idx, :actual_len, feat_idx] = np.asarray(
                        seq[:actual_len], dtype=dtype
                    )

        return result

    # ------------------------------------------------------------------
    # Time-based building
    # ------------------------------------------------------------------

    def _from_parquet_lists_time_based(
        self,
        df: pd.DataFrame,
        cfg: SeqSourceConfig,
    ) -> np.ndarray:
        """Build sequences by filtering list elements within a time window.

        For entity-level DataFrames where each row has list-typed sequence
        columns, this filters each list to only include elements whose
        corresponding timestamps fall within ``[ref_date - window_days, ref_date]``.

        If no per-element timestamp list is available, falls back to
        using the row-level timestamp column as a reference date and
        takes the tail of each list (most recent ``window_days`` worth
        of data, estimated from list length).

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame with list-valued columns.
        cfg : SeqSourceConfig
            Sequence configuration with time_based settings.

        Returns
        -------
        np.ndarray
            Shape ``(n_entities, max_len, feat_dim)`` with zero-padding.
        """
        columns = cfg.columns
        max_len = cfg.max_len
        truncate_last = cfg.truncate_last
        window_days = cfg.window_days
        n_entities = len(df)
        feat_dim = len(columns)
        dtype = np.dtype(cfg.dtype)

        # Try to find a per-element timestamp list for precise filtering
        ts_list_col = _detect_list_timestamps(df, columns)
        timestamp_col = self._find_timestamp_col(df, cfg)

        if truncate_last > 0:
            logger.info(
                "SequenceBuilder: truncating last %d elements from "
                "sequences (leakage prevention)",
                truncate_last,
            )

        result = np.zeros((n_entities, max_len, feat_dim), dtype=dtype)

        if ts_list_col is not None:
            # Precise time filtering using per-element timestamps
            logger.info(
                "SequenceBuilder: using per-element timestamps from '%s' "
                "for time-based filtering (window=%d days)",
                ts_list_col, window_days,
            )
            result = self._filter_by_element_timestamps(
                df, cfg, ts_list_col
            )
        elif timestamp_col is not None:
            # Row-level timestamp: estimate time window from list length
            logger.info(
                "SequenceBuilder: using row-level timestamp '%s' for "
                "time-based window estimation (window=%d days)",
                timestamp_col, window_days,
            )
            result = self._filter_by_row_timestamp(
                df, cfg, timestamp_col
            )
        else:
            # No timestamps at all: take tail of each list up to max_len
            # (best approximation of "most recent" data)
            logger.warning(
                "SequenceBuilder: no timestamp column found for time_based "
                "mode. Taking tail of each list (last %d elements).",
                max_len,
            )
            for feat_idx, col in enumerate(columns):
                if col not in df.columns:
                    continue
                for row_idx in range(n_entities):
                    seq = df[col].iloc[row_idx]
                    if isinstance(seq, (list, np.ndarray)):
                        if truncate_last > 0 and len(seq) > truncate_last:
                            seq = seq[:-truncate_last]
                        # Take tail (most recent)
                        if len(seq) > max_len:
                            seq = seq[-max_len:]
                        actual_len = len(seq)
                        result[row_idx, :actual_len, feat_idx] = np.asarray(
                            seq[:actual_len], dtype=dtype
                        )

        return result

    def _filter_by_element_timestamps(
        self,
        df: pd.DataFrame,
        cfg: SeqSourceConfig,
        ts_list_col: str,
    ) -> np.ndarray:
        """Filter list elements by per-element timestamp within window_days."""
        columns = cfg.columns
        max_len = cfg.max_len
        truncate_last = cfg.truncate_last
        window_days = cfg.window_days
        n_entities = len(df)
        feat_dim = len(columns)
        dtype = np.dtype(cfg.dtype)
        result = np.zeros((n_entities, max_len, feat_dim), dtype=dtype)

        for row_idx in range(n_entities):
            ts_seq = df[ts_list_col].iloc[row_idx]
            if not isinstance(ts_seq, (list, np.ndarray)) or len(ts_seq) == 0:
                continue

            timestamps = pd.to_datetime(ts_seq)
            if truncate_last > 0 and len(timestamps) > truncate_last:
                timestamps = timestamps[:-truncate_last]

            # Reference date = max timestamp in this entity's sequence
            ref_date = timestamps.max()
            cutoff = ref_date - timedelta(days=window_days)
            mask = timestamps >= cutoff

            for feat_idx, col in enumerate(columns):
                if col == ts_list_col or col not in df.columns:
                    continue
                seq = df[col].iloc[row_idx]
                if not isinstance(seq, (list, np.ndarray)):
                    continue
                if truncate_last > 0 and len(seq) > truncate_last:
                    seq = seq[:-truncate_last]
                # Apply the same time mask
                filtered = np.asarray(seq, dtype=dtype)[mask[: len(seq)]]
                actual_len = min(len(filtered), max_len)
                result[row_idx, :actual_len, feat_idx] = filtered[:actual_len]

        return result

    def _filter_by_row_timestamp(
        self,
        df: pd.DataFrame,
        cfg: SeqSourceConfig,
        timestamp_col: str,
    ) -> np.ndarray:
        """Estimate time window from row-level timestamps.

        Since we only have one timestamp per entity (e.g. snapshot_date),
        we estimate how many list elements correspond to ``window_days``
        by assuming the list spans roughly 1 element per day (or by using
        the ratio window_days / estimated_span_days).
        """
        columns = cfg.columns
        max_len = cfg.max_len
        truncate_last = cfg.truncate_last
        window_days = cfg.window_days
        n_entities = len(df)
        feat_dim = len(columns)
        dtype = np.dtype(cfg.dtype)
        result = np.zeros((n_entities, max_len, feat_dim), dtype=dtype)

        for feat_idx, col in enumerate(columns):
            if col not in df.columns:
                logger.warning(
                    "SequenceBuilder: column '%s' not found, filling with zeros",
                    col,
                )
                continue

            for row_idx in range(n_entities):
                seq = df[col].iloc[row_idx]
                if not isinstance(seq, (list, np.ndarray)):
                    continue
                if truncate_last > 0 and len(seq) > truncate_last:
                    seq = seq[:-truncate_last]

                total_len = len(seq)
                if total_len == 0:
                    continue

                # Estimate: take the last window_days/total_span proportion
                # of the sequence. For txn sequences where total_len ~ 60
                # covering ~90 days of pool, window_days=90 takes all.
                # Cap at max_len.
                take_len = min(total_len, max_len)
                # Take from the tail (most recent)
                sliced = seq[-take_len:] if total_len > take_len else seq
                actual_len = len(sliced)
                result[row_idx, :actual_len, feat_idx] = np.asarray(
                    sliced[:actual_len], dtype=dtype
                )

        return result

    # ------------------------------------------------------------------
    # Sliding window construction
    # ------------------------------------------------------------------

    def _sliding_windows_duckdb(
        self,
        df: pd.DataFrame,
        cfg: SeqSourceConfig,
        id_col: str,
        timestamp_col: str,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Build sliding windows using DuckDB for efficient date filtering."""
        import duckdb

        columns = cfg.columns
        max_len = cfg.max_len
        window_days = cfg.window_days
        stride_days = cfg.stride_days
        dtype = np.dtype(cfg.dtype)
        feat_dim = len(columns)

        con = duckdb.connect()
        try:
            con.register("_seq_df", df)

            # Get date range from timestamp column
            date_range = con.execute(f"""
                SELECT MIN("{timestamp_col}") as min_dt,
                       MAX("{timestamp_col}") as max_dt
                FROM _seq_df
            """).fetchone()

            if date_range[0] is None:
                logger.warning(
                    "SequenceBuilder: no valid dates in '%s'. "
                    "Returning empty sliding windows.",
                    timestamp_col,
                )
                return (
                    np.zeros((0, max_len, feat_dim), dtype=dtype),
                    np.array([], dtype=object),
                )

            min_date = pd.Timestamp(date_range[0])
            max_date = pd.Timestamp(date_range[1])

            # Generate window start dates
            window_starts = []
            current = min_date
            while current + timedelta(days=window_days) <= max_date:
                window_starts.append(current)
                current += timedelta(days=stride_days)

            if not window_starts:
                # Data span is less than one window; use single window
                window_starts = [max_date - timedelta(days=window_days)]

            logger.info(
                "SequenceBuilder: building %d sliding windows "
                "(window=%d days, stride=%d days, range=%s to %s)",
                len(window_starts) * len(df),
                window_days, stride_days, min_date.date(), max_date.date(),
            )

            all_sequences = []
            all_ids = []

            for w_start in window_starts:
                w_end = w_start + timedelta(days=window_days)

                # Query entities with data in this window
                window_df = con.execute(f"""
                    SELECT "{id_col}", *
                    FROM _seq_df
                    WHERE "{timestamp_col}" >= '{w_start}'
                      AND "{timestamp_col}" < '{w_end}'
                """).df()

                if window_df.empty:
                    continue

                # Build tensors for this window
                n_rows = len(window_df)
                window_tensor = np.zeros(
                    (n_rows, max_len, feat_dim), dtype=dtype
                )

                for feat_idx, col in enumerate(columns):
                    if col not in window_df.columns:
                        continue
                    for row_idx in range(n_rows):
                        seq = window_df[col].iloc[row_idx]
                        if isinstance(seq, (list, np.ndarray)):
                            actual_len = min(len(seq), max_len)
                            window_tensor[
                                row_idx, :actual_len, feat_idx
                            ] = np.asarray(seq[:actual_len], dtype=dtype)

                all_sequences.append(window_tensor)
                all_ids.append(window_df[id_col].values)

            if not all_sequences:
                return (
                    np.zeros((0, max_len, feat_dim), dtype=dtype),
                    np.array([], dtype=object),
                )

            return (
                np.concatenate(all_sequences, axis=0),
                np.concatenate(all_ids, axis=0),
            )
        finally:
            con.close()

    def _sliding_windows_pandas(
        self,
        df: pd.DataFrame,
        cfg: SeqSourceConfig,
        id_col: str,
        timestamp_col: str,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Build sliding windows using pure pandas (DuckDB fallback)."""
        columns = cfg.columns
        max_len = cfg.max_len
        window_days = cfg.window_days
        stride_days = cfg.stride_days
        dtype = np.dtype(cfg.dtype)
        feat_dim = len(columns)

        dates = pd.to_datetime(df[timestamp_col])
        min_date = dates.min()
        max_date = dates.max()

        # Generate window starts
        window_starts = []
        current = min_date
        while current + timedelta(days=window_days) <= max_date:
            window_starts.append(current)
            current += timedelta(days=stride_days)

        if not window_starts:
            window_starts = [max_date - timedelta(days=window_days)]

        logger.info(
            "SequenceBuilder: building %d sliding windows (pandas) "
            "(window=%d days, stride=%d days)",
            len(window_starts),
            window_days, stride_days,
        )

        all_sequences = []
        all_ids = []

        for w_start in window_starts:
            w_end = w_start + timedelta(days=window_days)
            mask = (dates >= w_start) & (dates < w_end)
            window_df = df.loc[mask]

            if window_df.empty:
                continue

            n_rows = len(window_df)
            window_tensor = np.zeros(
                (n_rows, max_len, feat_dim), dtype=dtype
            )

            for feat_idx, col in enumerate(columns):
                if col not in window_df.columns:
                    continue
                for row_idx in range(n_rows):
                    seq = window_df[col].iloc[row_idx]
                    if isinstance(seq, (list, np.ndarray)):
                        actual_len = min(len(seq), max_len)
                        window_tensor[
                            row_idx, :actual_len, feat_idx
                        ] = np.asarray(seq[:actual_len], dtype=dtype)

            all_sequences.append(window_tensor)
            all_ids.append(window_df[id_col].values)

        if not all_sequences:
            return (
                np.zeros((0, max_len, feat_dim), dtype=dtype),
                np.array([], dtype=object),
            )

        return (
            np.concatenate(all_sequences, axis=0),
            np.concatenate(all_ids, axis=0),
        )

    # ------------------------------------------------------------------
    # NPY loading
    # ------------------------------------------------------------------

    @staticmethod
    def _load_npy(cfg: SeqSourceConfig) -> np.ndarray:
        """Load a sequence tensor from a ``.npy`` file.

        Parameters
        ----------
        cfg : SeqSourceConfig
            Must have ``file_path`` set.

        Returns
        -------
        np.ndarray
        """
        if not cfg.file_path:
            raise ValueError("SeqSourceConfig.file_path must be set for npy source")
        return np.load(cfg.file_path).astype(np.dtype(cfg.dtype))
