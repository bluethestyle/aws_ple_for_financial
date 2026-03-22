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


class DataAdapter(ABC):
    """Abstract base for dataset-specific data loading.

    Each dataset provides a subclass that implements load_raw().
    The adapter is responsible for:
    - Loading raw files (CSV, Parquet, etc.)
    - Aggregating to entity-level if needed (e.g., transaction -> user)
    - Returning a dict of DataFrames keyed by role name

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
    def load_raw(self) -> Dict[str, pd.DataFrame]:
        """Load and return raw data as dict of DataFrames.

        Returns:
            Dict with keys like "main", "transactions", "products", etc.
            At minimum must contain "main" key with entity-level DataFrame.
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
