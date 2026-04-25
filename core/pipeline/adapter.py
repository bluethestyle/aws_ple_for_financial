"""DataAdapter ABC -- dataset-specific raw data loading contract."""

from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import logging
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class AdapterMetadata:
    """Metadata returned by adapter about the loaded data."""
    id_col: str = "user_id"
    timestamp_col: Optional[str] = None
    entity_granularity: str = "user"  # "user" | "transaction" | "session"
    num_entities: int = 0
    num_raw_rows: int = 0
    source_files: List[str] = field(default_factory=list)
    backend_used: str = "pandas"  # "cudf" | "duckdb" | "pandas"


@dataclass
class DuckDBAdapterContext:
    """Pandas-free adapter result (CLAUDE.md §3.3).

    Modern adapters expose the loaded data as a DuckDB table on a shared
    connection rather than materialising a 1.4 GB parquet (with LIST
    columns) into a 10-15 GB pandas DataFrame. ``PipelineRunner`` keeps
    the connection open for the duration of all 9 stages.

    Attributes
    ----------
    con : duckdb.DuckDBPyConnection
        Connection that owns ``table_name``. The runner is responsible
        for closing it at the end of the pipeline.
    table_name : str
        Name of the table to query (typically ``"raw"``).
    metadata : AdapterMetadata
        Same fields as the legacy contract — id_col, num_entities, etc.
    extra_tables : Dict[str, str]
        Optional extra-role -> table-name mapping (e.g.
        ``{"transactions": "raw_txn"}``) for adapters that decompose the
        source into multiple entity-level tables.
    """
    con: Any  # duckdb.DuckDBPyConnection (avoid hard import at module load)
    table_name: str
    metadata: AdapterMetadata
    extra_tables: Dict[str, str] = field(default_factory=dict)


class DataAdapter(ABC):
    """Abstract base for dataset-specific data loading.

    Each dataset provides a subclass that implements load_raw().
    The adapter is responsible for:
    - Loading raw files (CSV, Parquet, etc.)
    - Aggregating to entity-level if needed (e.g., transaction -> user)
    - Returning EITHER a dict of DataFrames (legacy) OR a
      :class:`DuckDBAdapterContext` (preferred, no pandas materialisation)

    The adapter is NOT responsible for:
    - Feature engineering (handled by FeatureGroupPipeline)
    - Normalization (handled by transformers)
    - Label derivation (handled by LabelDeriver)
    - Encryption (handled by EncryptionPipeline)
    """

    def __init__(self, config: dict):
        self.config = config
        self._metadata: Optional[AdapterMetadata] = None

    @abstractmethod
    def load_raw(self) -> Union[Dict[str, pd.DataFrame], DuckDBAdapterContext]:
        """Load and return raw data.

        Two return shapes are accepted:

        * Legacy: ``Dict[str, pd.DataFrame]`` with keys like ``"main"``,
          ``"transactions"``. Used by older adapters; the runner will
          register the ``"main"`` frame into a fresh DuckDB connection
          before running the 9 stages.
        * Preferred: :class:`DuckDBAdapterContext` exposing the data on
          an open DuckDB connection so the runner can flow the entire
          pipeline through SQL without ever materialising the LIST
          columns into pandas.
        """
        ...

    @property
    def metadata(self) -> AdapterMetadata:
        if self._metadata is None:
            raise RuntimeError("Call load_raw() before accessing metadata")
        return self._metadata

    def _select_backend(self) -> str:
        """Select best available processing backend."""
        backends = self.config.get("data", {}).get("backend", ["cudf", "duckdb", "pandas"])
        if isinstance(backends, str):
            backends = [backends]
        for b in backends:
            if b == "cudf":
                try:
                    import cudf  # noqa: F401
                    return "cudf"
                except ImportError:
                    continue
            elif b == "duckdb":
                try:
                    import duckdb  # noqa: F401
                    return "duckdb"
                except ImportError:
                    continue
            elif b == "pandas":
                return "pandas"
        return "pandas"


class AdapterRegistry:
    """Registry for dataset adapters."""
    _adapters: Dict[str, type] = {}

    @classmethod
    def register(cls, name: str):
        def decorator(adapter_cls):
            cls._adapters[name] = adapter_cls
            return adapter_cls
        return decorator

    @classmethod
    def build(cls, name: str, config: dict) -> DataAdapter:
        if name not in cls._adapters:
            raise KeyError(f"Adapter '{name}' not registered. Available: {list(cls._adapters.keys())}")
        return cls._adapters[name](config)

    @classmethod
    def list_registered(cls) -> List[str]:
        return list(cls._adapters.keys())
