"""
Data layer: schema registry, query engine, DataFrame abstraction, and validation.

Provides config-driven data access with DuckDB as the primary backend,
a unified DataFrame abstraction (DuckDB > cuDF > pandas), schema
validation, and data quality checks.
"""

from .config import DataBackendConfig
from .dataframe import DataFrameBackend, df_backend
from .query_engine import QueryEngine
from .schema_registry import SchemaRegistry
from .validation import DataValidator, ValidationResult

__all__ = [
    "DataBackendConfig",
    "DataFrameBackend",
    "df_backend",
    "SchemaRegistry",
    "QueryEngine",
    "DataValidator",
    "ValidationResult",
]
