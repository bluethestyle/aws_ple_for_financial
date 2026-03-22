"""
Sequence Builder
================

Converts various sequence data formats into padded 3-D NumPy tensors
suitable for temporal expert networks (Mamba, adaTT, etc.).

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
            seq_len=50,
        ),
    }
    builder = SequenceBuilder()
    tensors = builder.build(raw_data={"main": df}, seq_configs=configs)
    # tensors["purchase_history"].shape == (n_rows, 50, 3)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

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
    seq_len : int
        Maximum sequence length.  Shorter sequences are zero-padded;
        longer sequences are truncated.
    file_path : str
        Path to the ``.npy`` file (for ``npy`` source).
    dtype : str
        NumPy dtype for the output array (default ``"float32"``).
    """

    source: str = "parquet_list"  # "parquet_list" | "npy"
    columns: List[str] = field(default_factory=list)
    seq_len: int = 50
    file_path: str = ""
    dtype: str = "float32"

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "SeqSourceConfig":
        """Create a SeqSourceConfig from a plain dict."""
        return cls(
            source=d.get("source", "parquet_list"),
            columns=d.get("columns", []),
            seq_len=d.get("seq_len", 50),
            file_path=d.get("file_path", ""),
            dtype=d.get("dtype", "float32"),
        )


class SequenceBuilder:
    """Build padded 3-D tensors from various sequence formats."""

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
            ``(n_entities, seq_len, feat_dim)``.
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
                results[name] = self._from_parquet_lists(raw_data["main"], cfg)
                logger.info(
                    "SequenceBuilder: built '%s' from parquet_list, shape=%s",
                    name,
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

    def _from_parquet_lists(
        self,
        df: pd.DataFrame,
        cfg: SeqSourceConfig,
    ) -> np.ndarray:
        """Convert list-type parquet columns to a padded 3-D array.

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame with list-valued columns.
        cfg : SeqSourceConfig
            Sequence configuration.

        Returns
        -------
        np.ndarray
            Shape ``(n_entities, seq_len, feat_dim)`` with zero-padding.
        """
        columns = cfg.columns
        seq_len = cfg.seq_len
        n_entities = len(df)
        feat_dim = len(columns)
        dtype = np.dtype(cfg.dtype)

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
                    actual_len = min(len(seq), seq_len)
                    result[row_idx, :actual_len, feat_idx] = np.asarray(
                        seq[:actual_len], dtype=dtype
                    )

        return result

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
