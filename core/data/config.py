"""
Data backend configuration.

Defines :class:`DataBackendConfig` -- runtime settings for the DataFrame
abstraction layer.  The config controls which backend is selected, memory
limits, GPU thresholds, and S3 connectivity parameters.

The configuration can be loaded from environment variables, a YAML file,
or constructed programmatically::

    from core.data.config import DataBackendConfig

    # Auto-detect defaults from environment
    cfg = DataBackendConfig()

    # Override for a specific run
    cfg = DataBackendConfig(
        preferred_backend="cudf",
        duckdb_memory_limit="16GB",
        cudf_min_rows=50_000,
    )
"""

from __future__ import annotations

import os
import logging
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

__all__ = ["DataBackendConfig"]


@dataclass
class DataBackendConfig:
    """Runtime settings for the DataFrame abstraction layer.

    Parameters
    ----------
    preferred_backend : str
        Backend selection strategy.

        * ``"duckdb"`` -- always use DuckDB (recommended default).
        * ``"cudf"`` -- always use cuDF (requires RAPIDS GPU toolkit).
        * ``"pandas"`` -- always use pandas (legacy fallback).
        * ``"auto"`` -- try DuckDB first, then cuDF, then pandas.

    cudf_min_rows : int
        Minimum number of rows for cuDF to be selected over DuckDB when
        ``preferred_backend="auto"`` and a GPU is available.  Below this
        threshold the GPU launch overhead is not worth the throughput
        gain, so DuckDB is used instead.

    duckdb_memory_limit : str
        DuckDB per-connection memory limit (e.g. ``"4GB"``).  Controls
        how much RAM DuckDB will use before spilling to disk.

    duckdb_threads : int
        Number of DuckDB worker threads.  ``0`` means auto-detect
        (typically ``os.cpu_count()``).

    duckdb_temp_directory : str or None
        Directory for DuckDB temporary spill files.  ``None`` means
        DuckDB picks its own default.

    s3_region : str
        AWS region for httpfs S3 access (DuckDB) and cuDF S3 reads.

    s3_access_key_id : str or None
        AWS access key.  ``None`` means read from environment or
        instance profile.

    s3_secret_access_key : str or None
        AWS secret key.  ``None`` means read from environment or
        instance profile.
    """

    preferred_backend: str = "duckdb"
    cudf_min_rows: int = 100_000
    duckdb_memory_limit: str = "4GB"
    duckdb_threads: int = 0
    duckdb_temp_directory: Optional[str] = None
    s3_region: str = "ap-northeast-2"
    s3_access_key_id: Optional[str] = None
    s3_secret_access_key: Optional[str] = None

    def __post_init__(self) -> None:
        valid_backends = ("duckdb", "cudf", "pandas", "auto")
        if self.preferred_backend not in valid_backends:
            raise ValueError(
                f"preferred_backend must be one of {valid_backends}, "
                f"got '{self.preferred_backend}'"
            )

    # -- Factory -----------------------------------------------------------

    @classmethod
    def from_env(cls) -> "DataBackendConfig":
        """Construct configuration from environment variables.

        Recognised variables (all optional, fallback to dataclass defaults):

        * ``DF_BACKEND`` -- preferred backend
        * ``CUDF_MIN_ROWS`` -- cuDF row threshold
        * ``DUCKDB_MEMORY_LIMIT`` -- DuckDB memory cap
        * ``DUCKDB_THREADS`` -- DuckDB thread count
        * ``DUCKDB_TEMP_DIR`` -- DuckDB spill directory
        * ``AWS_DEFAULT_REGION`` -- S3 region
        * ``AWS_ACCESS_KEY_ID`` -- S3 access key
        * ``AWS_SECRET_ACCESS_KEY`` -- S3 secret key
        """
        return cls(
            preferred_backend=os.environ.get("DF_BACKEND", "duckdb"),
            cudf_min_rows=int(os.environ.get("CUDF_MIN_ROWS", "100000")),
            duckdb_memory_limit=os.environ.get("DUCKDB_MEMORY_LIMIT", "4GB"),
            duckdb_threads=int(os.environ.get("DUCKDB_THREADS", "0")),
            duckdb_temp_directory=os.environ.get("DUCKDB_TEMP_DIR"),
            s3_region=os.environ.get("AWS_DEFAULT_REGION", "ap-northeast-2"),
            s3_access_key_id=os.environ.get("AWS_ACCESS_KEY_ID"),
            s3_secret_access_key=os.environ.get("AWS_SECRET_ACCESS_KEY"),
        )

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "DataBackendConfig":
        """Construct from a plain dictionary (e.g. parsed YAML section)."""
        return cls(
            preferred_backend=d.get("preferred_backend", "duckdb"),
            cudf_min_rows=int(d.get("cudf_min_rows", 100_000)),
            duckdb_memory_limit=d.get("duckdb_memory_limit", "4GB"),
            duckdb_threads=int(d.get("duckdb_threads", 0)),
            duckdb_temp_directory=d.get("duckdb_temp_directory"),
            s3_region=d.get("s3_region", "ap-northeast-2"),
            s3_access_key_id=d.get("s3_access_key_id"),
            s3_secret_access_key=d.get("s3_secret_access_key"),
        )
