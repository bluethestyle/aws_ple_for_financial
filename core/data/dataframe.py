"""
DataFrame Abstraction Layer -- unified interface over DuckDB, cuDF, and pandas.

This module provides :class:`DataFrameBackend`, a singleton that routes all
DataFrame operations through the best available backend:

1. **DuckDB** (default) -- SQL-native, zero-copy Parquet I/O, automatic
   disk spill for out-of-core processing, direct S3 reads via httpfs.
2. **cuDF** (GPU) -- RAPIDS GPU-accelerated DataFrames for large-scale
   feature engineering when an NVIDIA GPU is available and the data
   exceeds a configurable row threshold.
3. **pandas** (fallback) -- used only when neither DuckDB nor cuDF is
   installed.

All feature engineering, data loading, and transformation code should use
the global ``df_backend`` singleton instead of importing pandas directly::

    from core.data.dataframe import df_backend

    df = df_backend.read_parquet("s3://bucket/data.parquet")
    df = df_backend.query("SELECT * FROM df WHERE amount > 100", df=df)
    arr = df_backend.to_numpy(df)
    df_backend.to_parquet(df, "output.parquet")

The backend is selected once at import time (based on :class:`DataBackendConfig`)
and remains fixed for the process lifetime.  To override, set environment
variables before import or call ``DataFrameBackend.reconfigure()``.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Union

import numpy as np

from .config import DataBackendConfig

logger = logging.getLogger(__name__)

__all__ = ["DataFrameBackend", "df_backend"]


# ---------------------------------------------------------------------------
# Lazy availability flags
# ---------------------------------------------------------------------------

def _check_duckdb() -> bool:
    try:
        import duckdb  # noqa: F401
        return True
    except ImportError:
        return False


def _check_cudf() -> bool:
    try:
        import cudf  # noqa: F401
        return True
    except ImportError:
        return False


def _check_pandas() -> bool:
    try:
        import pandas  # noqa: F401
        return True
    except ImportError:
        return False


# ---------------------------------------------------------------------------
# DataFrameBackend
# ---------------------------------------------------------------------------

class DataFrameBackend:
    """Unified DataFrame abstraction over DuckDB, cuDF, and pandas.

    The backend is chosen at construction time based on
    :class:`DataBackendConfig` and library availability.  All public
    methods accept and return the native DataFrame type of the active
    backend (``duckdb.DuckDBPyRelation`` materialised to pandas,
    ``cudf.DataFrame``, or ``pandas.DataFrame``).

    Parameters
    ----------
    config : DataBackendConfig, optional
        Runtime configuration.  When omitted, defaults are derived
        from environment variables via ``DataBackendConfig.from_env()``.
    """

    # Backend name constants
    DUCKDB = "duckdb"
    CUDF = "cudf"
    PANDAS = "pandas"

    def __init__(self, config: Optional[DataBackendConfig] = None) -> None:
        self._config = config or DataBackendConfig.from_env()
        self._backend: str = self._detect_backend()
        self._duckdb_conn: Any = None  # lazily initialised

        logger.info(
            "DataFrameBackend initialised: backend=%s, "
            "duckdb_memory_limit=%s, cudf_min_rows=%d",
            self._backend,
            self._config.duckdb_memory_limit,
            self._config.cudf_min_rows,
        )

    # -- Backend detection -------------------------------------------------

    def _detect_backend(self) -> str:
        """Select the best available backend based on config and imports.

        Priority when ``preferred_backend="auto"``:
            DuckDB > cuDF > pandas

        When a specific backend is requested and unavailable, falls back
        through the priority chain with a warning.
        """
        pref = self._config.preferred_backend

        if pref == "auto":
            if _check_duckdb():
                return self.DUCKDB
            if _check_cudf():
                return self.CUDF
            if _check_pandas():
                return self.PANDAS
            raise ImportError(
                "No DataFrame backend available. "
                "Install at least one of: duckdb, cudf, pandas."
            )

        # Explicit preference
        if pref == self.DUCKDB:
            if _check_duckdb():
                return self.DUCKDB
            logger.warning(
                "duckdb requested but not installed; falling back."
            )
        elif pref == self.CUDF:
            if _check_cudf():
                return self.CUDF
            logger.warning(
                "cudf requested but not installed; falling back."
            )
        elif pref == self.PANDAS:
            if _check_pandas():
                return self.PANDAS
            logger.warning(
                "pandas requested but not installed; falling back."
            )

        # Fallback chain
        for candidate, checker in [
            (self.DUCKDB, _check_duckdb),
            (self.CUDF, _check_cudf),
            (self.PANDAS, _check_pandas),
        ]:
            if checker():
                logger.warning(
                    "Falling back to '%s' backend.", candidate,
                )
                return candidate

        raise ImportError(
            "No DataFrame backend available. "
            "Install at least one of: duckdb, cudf, pandas."
        )

    # -- DuckDB connection management --------------------------------------

    def _get_duckdb_conn(self) -> Any:
        """Return (and lazily create) the DuckDB connection."""
        if self._duckdb_conn is not None:
            return self._duckdb_conn

        import duckdb

        self._duckdb_conn = duckdb.connect(":memory:")
        cfg = self._config

        self._duckdb_conn.execute(
            f"SET memory_limit='{cfg.duckdb_memory_limit}'"
        )
        if cfg.duckdb_threads > 0:
            self._duckdb_conn.execute(f"SET threads={cfg.duckdb_threads}")
        self._duckdb_conn.execute("SET preserve_insertion_order=false")

        # S3 / httpfs
        s3_key = cfg.s3_access_key_id or os.environ.get("AWS_ACCESS_KEY_ID")
        s3_secret = cfg.s3_secret_access_key or os.environ.get(
            "AWS_SECRET_ACCESS_KEY"
        )
        s3_region = cfg.s3_region or os.environ.get("AWS_DEFAULT_REGION")

        if s3_region or s3_key:
            try:
                self._duckdb_conn.execute("INSTALL httpfs; LOAD httpfs;")
                if s3_region:
                    self._duckdb_conn.execute(
                        f"SET s3_region='{s3_region}'"
                    )
                if s3_key and s3_secret:
                    self._duckdb_conn.execute(
                        f"SET s3_access_key_id='{s3_key}'"
                    )
                    self._duckdb_conn.execute(
                        f"SET s3_secret_access_key='{s3_secret}'"
                    )
                logger.debug("DuckDB httpfs configured (region=%s)", s3_region)
            except Exception as exc:
                logger.warning("httpfs setup skipped: %s", exc)

        # Temp directory for spill
        temp_dir = cfg.duckdb_temp_directory
        if temp_dir:
            os.makedirs(temp_dir, exist_ok=True)
            norm = temp_dir.replace("\\", "/")
            self._duckdb_conn.execute(f"SET temp_directory='{norm}'")

        return self._duckdb_conn

    # -- Properties --------------------------------------------------------

    @property
    def backend_name(self) -> str:
        """Name of the active backend (``"duckdb"``, ``"cudf"``, or ``"pandas"``)."""
        return self._backend

    @property
    def config(self) -> DataBackendConfig:
        """Current configuration snapshot."""
        return self._config

    # -- Reconfiguration ---------------------------------------------------

    def reconfigure(self, config: DataBackendConfig) -> None:
        """Swap configuration and re-detect the backend.

        Closes any existing DuckDB connection before re-initialising.
        """
        self.close()
        self._config = config
        self._backend = self._detect_backend()
        logger.info("DataFrameBackend reconfigured: backend=%s", self._backend)

    def close(self) -> None:
        """Release backend resources (DuckDB connection, etc.)."""
        if self._duckdb_conn is not None:
            try:
                self._duckdb_conn.close()
            except Exception:
                pass
            self._duckdb_conn = None

    # =====================================================================
    # Core I/O operations
    # =====================================================================

    def read_parquet(
        self,
        path: str,
        columns: Optional[List[str]] = None,
    ) -> Any:
        """Read a Parquet file or S3 URI and return a DataFrame.

        Parameters
        ----------
        path : str
            Local file path, glob pattern, or ``s3://`` URI.
        columns : list[str], optional
            Subset of columns to read.  ``None`` reads all columns.

        Returns
        -------
        DataFrame
            Backend-native DataFrame (pandas for DuckDB, cudf.DataFrame
            for cuDF, pandas.DataFrame for pandas backend).
        """
        if self._backend == self.DUCKDB:
            return self._duckdb_read_parquet(path, columns)
        if self._backend == self.CUDF:
            return self._cudf_read_parquet(path, columns)
        return self._pandas_read_parquet(path, columns)

    def read_csv(
        self,
        path: str,
        **kwargs: Any,
    ) -> Any:
        """Read a CSV file and return a DataFrame.

        Parameters
        ----------
        path : str
            Local file path or ``s3://`` URI.
        **kwargs
            Backend-specific keyword arguments (e.g. ``sep``, ``header``).
        """
        if self._backend == self.DUCKDB:
            norm = self._normalise_path(path)
            conn = self._get_duckdb_conn()
            return conn.execute(
                f"SELECT * FROM read_csv_auto('{norm}')"
            ).fetchdf()
        if self._backend == self.CUDF:
            import cudf
            return cudf.read_csv(path, **kwargs)
        import pandas as pd
        return pd.read_csv(path, **kwargs)

    def to_parquet(
        self,
        df: Any,
        path: str,
        *,
        compression: str = "SNAPPY",
        row_group_size: int = 500_000,
    ) -> None:
        """Write a DataFrame as a Parquet file.

        Parameters
        ----------
        df : DataFrame
            Data to write.
        path : str
            Destination file path (local or ``s3://``).
        compression : str
            Parquet compression codec.
        row_group_size : int
            Parquet row-group size.
        """
        if self._backend == self.DUCKDB:
            self._duckdb_to_parquet(df, path, compression, row_group_size)
        elif self._backend == self.CUDF:
            df.to_parquet(path, compression=compression.lower())
        else:
            Path(path).parent.mkdir(parents=True, exist_ok=True)
            df.to_parquet(path, compression=compression.lower(), index=False)

    # =====================================================================
    # SQL query interface
    # =====================================================================

    def query(self, sql: str, **tables: Any) -> Any:
        """Execute a SQL query and return the result as a DataFrame.

        For the DuckDB backend, any keyword arguments are registered as
        virtual tables that can be referenced in the SQL string.  For
        other backends, a temporary DuckDB connection is used.

        Parameters
        ----------
        sql : str
            SQL query string.
        **tables
            Keyword arguments mapping table names to DataFrames.  These
            are registered with DuckDB so the SQL can reference them by
            name.  Example::

                result = df_backend.query(
                    "SELECT a, b FROM t1 JOIN t2 ON t1.id = t2.id",
                    t1=df_one,
                    t2=df_two,
                )

        Returns
        -------
        DataFrame
            Query result.
        """
        if self._backend == self.DUCKDB:
            conn = self._get_duckdb_conn()
            for name, tbl in tables.items():
                conn.register(name, tbl)
            try:
                result = conn.execute(sql).fetchdf()
            finally:
                for name in tables:
                    try:
                        conn.unregister(name)
                    except Exception:
                        pass
            return result

        # For cuDF / pandas: use a temporary DuckDB connection if available
        if _check_duckdb():
            import duckdb
            conn = duckdb.connect(":memory:")
            try:
                # Convert cuDF to pandas for DuckDB registration
                for name, tbl in tables.items():
                    if self._backend == self.CUDF:
                        conn.register(name, tbl.to_pandas())
                    else:
                        conn.register(name, tbl)
                result = conn.execute(sql).fetchdf()
                if self._backend == self.CUDF:
                    import cudf
                    return cudf.DataFrame.from_pandas(result)
                return result
            finally:
                conn.close()

        raise RuntimeError(
            "SQL query requires duckdb. Install it with: pip install duckdb"
        )

    def query_s3(self, sql: str) -> Any:
        """Execute a SQL query that reads directly from S3 Parquet files.

        This is a DuckDB-specific optimisation.  The SQL can reference
        S3 URIs via ``read_parquet('s3://...')``.

        Parameters
        ----------
        sql : str
            SQL query with S3 ``read_parquet()`` references.

        Returns
        -------
        DataFrame
            Query result.

        Raises
        ------
        RuntimeError
            If duckdb is not available.
        """
        if self._backend == self.DUCKDB or _check_duckdb():
            conn = self._get_duckdb_conn()
            return conn.execute(sql).fetchdf()
        raise RuntimeError(
            "query_s3 requires duckdb with httpfs. "
            "Install with: pip install duckdb"
        )

    # =====================================================================
    # DataFrame operations
    # =====================================================================

    def concat(
        self,
        dfs: Sequence[Any],
        axis: int = 0,
    ) -> Any:
        """Concatenate a list of DataFrames.

        Parameters
        ----------
        dfs : sequence of DataFrame
            DataFrames to concatenate.
        axis : int
            ``0`` for row-wise (vertical), ``1`` for column-wise
            (horizontal).

        Returns
        -------
        DataFrame
        """
        if not dfs:
            if self._backend == self.CUDF:
                import cudf
                return cudf.DataFrame()
            import pandas as pd
            return pd.DataFrame()

        if self._backend == self.CUDF:
            import cudf
            return cudf.concat(list(dfs), axis=axis)
        import pandas as pd
        return pd.concat(list(dfs), axis=axis)

    def merge(
        self,
        left: Any,
        right: Any,
        on: Union[str, List[str]],
        how: str = "inner",
    ) -> Any:
        """Merge (join) two DataFrames.

        Parameters
        ----------
        left, right : DataFrame
            DataFrames to merge.
        on : str or list[str]
            Column(s) to join on.
        how : str
            Join type (``"inner"``, ``"left"``, ``"right"``, ``"outer"``).

        Returns
        -------
        DataFrame
        """
        if self._backend == self.DUCKDB:
            # Use DuckDB SQL join for better performance
            conn = self._get_duckdb_conn()
            conn.register("_left", left)
            conn.register("_right", right)
            try:
                if isinstance(on, str):
                    on_clause = f'_left."{on}" = _right."{on}"'
                else:
                    on_clause = " AND ".join(
                        f'_left."{c}" = _right."{c}"' for c in on
                    )
                result = conn.execute(
                    f"SELECT * FROM _left {how.upper()} JOIN _right "
                    f"ON {on_clause}"
                ).fetchdf()
            finally:
                conn.unregister("_left")
                conn.unregister("_right")
            return result
        return left.merge(right, on=on, how=how)

    # =====================================================================
    # Conversion utilities
    # =====================================================================

    def to_numpy(self, df: Any) -> np.ndarray:
        """Convert a DataFrame to a numpy array.

        For cuDF DataFrames, this transfers data from GPU to CPU.  For
        model input in GPU pipelines, prefer :meth:`to_dlpack` instead.

        Parameters
        ----------
        df : DataFrame
            Input DataFrame.

        Returns
        -------
        numpy.ndarray
        """
        if hasattr(df, "to_numpy"):
            return df.to_numpy()
        if hasattr(df, "values"):
            return np.asarray(df.values)
        return np.asarray(df)

    def to_pandas(self, df: Any) -> Any:
        """Convert a DataFrame to pandas (no-op if already pandas).

        Handles cuDF DataFrames regardless of which backend is active --
        if the object *is* a cuDF DataFrame it will be converted via
        ``cudf.DataFrame.to_pandas()``.

        Parameters
        ----------
        df : DataFrame
            Input DataFrame (cudf.DataFrame or pandas.DataFrame).

        Returns
        -------
        pandas.DataFrame
        """
        # Check for cuDF object first (may arrive even when the active
        # backend is not cuDF, e.g. when mixing backends in a pipeline).
        if _check_cudf():
            import cudf
            if isinstance(df, cudf.DataFrame):
                return df.to_pandas()
        return df

    def from_pandas(self, df: Any) -> Any:
        """Convert a pandas DataFrame to the active backend type.

        Parameters
        ----------
        df : pandas.DataFrame
            Input pandas DataFrame.

        Returns
        -------
        DataFrame
            Backend-native DataFrame.
        """
        if self._backend == self.CUDF:
            import cudf
            return cudf.DataFrame.from_pandas(df)
        return df

    def to_cudf(self, df: Any) -> Any:
        """Convert a DataFrame to cuDF.

        Accepts pandas DataFrames, DuckDB relation results, or cuDF
        DataFrames (returned as-is).

        Parameters
        ----------
        df : DataFrame
            Input DataFrame.

        Returns
        -------
        cudf.DataFrame

        Raises
        ------
        RuntimeError
            If cuDF is not available.
        """
        if not _check_cudf():
            raise RuntimeError(
                "to_cudf() requires cuDF. Install RAPIDS cuDF."
            )
        import cudf
        if isinstance(df, cudf.DataFrame):
            return df
        # DuckDB relations expose .df() / .fetchdf(); pandas is the
        # common intermediate.  cuDF.from_pandas handles both cases
        # once the input is a pandas DataFrame.
        import pandas as pd
        if not isinstance(df, pd.DataFrame):
            # Assume DuckDB relation or similar with to_pandas / fetchdf
            if hasattr(df, "fetchdf"):
                df = df.fetchdf()
            elif hasattr(df, "to_pandas"):
                df = df.to_pandas()
        return cudf.DataFrame.from_pandas(df)

    def to_numpy_dict(
        self,
        df: Any,
        columns: Optional[List[str]] = None,
    ) -> Dict[str, np.ndarray]:
        """Extract columns as a ``{name: np.ndarray}`` dict without pandas.

        Chooses the fastest path based on the input type:

        * **cuDF** -- ``df[col].values.get()`` (GPU -> CPU transfer).
        * **DuckDB relation** -- ``.fetchnumpy()`` (zero-copy where
          possible).
        * **pandas** -- ``df[col].values`` (no copy for numeric cols).

        Parameters
        ----------
        df : DataFrame
            Input DataFrame (cuDF, pandas, or DuckDB relation).
        columns : list[str], optional
            Subset of columns.  ``None`` extracts all columns.

        Returns
        -------
        dict[str, numpy.ndarray]
        """
        # -- cuDF path -------------------------------------------------
        if _check_cudf():
            import cudf
            if isinstance(df, cudf.DataFrame):
                cols = columns or list(df.columns)
                return {c: df[c].values.get() for c in cols}

        # -- DuckDB relation path --------------------------------------
        if hasattr(df, "fetchnumpy"):
            np_dict = df.fetchnumpy()
            if columns is not None:
                np_dict = {c: np_dict[c] for c in columns if c in np_dict}
            return np_dict

        # -- pandas fallback -------------------------------------------
        cols = columns or list(df.columns)
        return {c: np.asarray(df[c].values) for c in cols}

    def from_numpy_dict(
        self,
        data: Dict[str, np.ndarray],
        n_rows: Optional[int] = None,
    ) -> Any:
        """Create a DataFrame from a ``{name: np.ndarray}`` dict.

        Returns a cuDF DataFrame when the GPU backend is available;
        otherwise returns a PyArrow Table (zero-copy from numpy).

        Parameters
        ----------
        data : dict[str, numpy.ndarray]
            Column name -> 1-D array mapping.
        n_rows : int, optional
            Expected row count (used for validation only).

        Returns
        -------
        cudf.DataFrame or pyarrow.Table
        """
        if n_rows is not None:
            for name, arr in data.items():
                if len(arr) != n_rows:
                    raise ValueError(
                        f"Column '{name}' has {len(arr)} rows, "
                        f"expected {n_rows}"
                    )

        if _check_cudf():
            import cudf
            return cudf.DataFrame(data)

        import pyarrow as pa
        arrays = [pa.array(arr) for arr in data.values()]
        return pa.table(dict(zip(data.keys(), arrays)))

    def numeric_column_names(self, df: Any) -> List[str]:
        """Return the names of numeric columns without pandas.

        * **cuDF** -- ``select_dtypes(include='number').columns``.
        * **DuckDB** -- ``DESCRIBE`` query filtering numeric SQL types.
        * **pandas** -- ``select_dtypes(include='number').columns``.

        Parameters
        ----------
        df : DataFrame
            Input DataFrame.

        Returns
        -------
        list[str]
        """
        # -- cuDF path -------------------------------------------------
        if _check_cudf():
            import cudf
            if isinstance(df, cudf.DataFrame):
                return list(df.select_dtypes(include="number").columns)

        # -- DuckDB relation path --------------------------------------
        if hasattr(df, "describe"):
            # DuckDB relation objects expose .describe() that returns
            # column metadata; however the safest route is to register
            # and DESCRIBE via SQL when the DuckDB backend is active.
            pass
        if self._backend == self.DUCKDB and _check_duckdb():
            import pandas as pd
            if isinstance(df, pd.DataFrame):
                conn = self._get_duckdb_conn()
                conn.register("_desc_tmp", df)
                try:
                    desc = conn.execute(
                        "DESCRIBE SELECT * FROM _desc_tmp"
                    ).fetchdf()
                finally:
                    conn.unregister("_desc_tmp")
                _numeric_types = {
                    "TINYINT", "SMALLINT", "INTEGER", "BIGINT",
                    "FLOAT", "DOUBLE", "DECIMAL", "HUGEINT",
                    "UTINYINT", "USMALLINT", "UINTEGER", "UBIGINT",
                }
                return [
                    row["column_name"]
                    for _, row in desc.iterrows()
                    if row["column_type"].split("(")[0].upper()
                    in _numeric_types
                ]

        # -- pandas fallback -------------------------------------------
        import pandas as pd
        if isinstance(df, pd.DataFrame):
            return list(df.select_dtypes(include="number").columns)

        # Unknown type -- try generic approach
        if hasattr(df, "columns"):
            return list(df.columns)
        return []

    def to_dlpack(self, df: Any) -> Any:
        """Zero-copy export via DLPack for GPU tensor interop.

        This is a cuDF-specific optimisation.  The DLPack capsule can
        be consumed by ``torch.from_dlpack()`` for zero-copy GPU tensor
        creation::

            capsule = df_backend.to_dlpack(cudf_df)
            tensor = torch.from_dlpack(capsule)

        Parameters
        ----------
        df : cudf.DataFrame
            A cuDF GPU DataFrame.

        Returns
        -------
        DLPack capsule
            Can be passed to ``torch.from_dlpack()`` or
            ``cupy.from_dlpack()``.

        Raises
        ------
        RuntimeError
            If the backend is not cuDF.
        """
        if self._backend != self.CUDF:
            raise RuntimeError(
                "to_dlpack() requires the cuDF backend. "
                "Current backend: %s" % self._backend
            )
        # cuDF Series/DataFrame -> CuPy array -> DLPack
        import cupy
        cp_arr = df.values
        if not isinstance(cp_arr, cupy.ndarray):
            cp_arr = cupy.asarray(cp_arr)
        return cp_arr.toDlpack()

    def empty(self, columns: Optional[List[str]] = None) -> Any:
        """Create an empty DataFrame with the given columns.

        Parameters
        ----------
        columns : list[str], optional
            Column names.  ``None`` returns a completely empty DataFrame.

        Returns
        -------
        DataFrame
        """
        if self._backend == self.CUDF:
            import cudf
            if columns:
                return cudf.DataFrame({c: [] for c in columns})
            return cudf.DataFrame()
        import pandas as pd
        if columns:
            return pd.DataFrame(columns=columns)
        return pd.DataFrame()

    def from_dict(self, data: Dict[str, Any], index: Any = None) -> Any:
        """Create a DataFrame from a dictionary.

        Parameters
        ----------
        data : dict
            Column name -> array-like mapping.
        index : array-like, optional
            Row index.

        Returns
        -------
        DataFrame
        """
        if self._backend == self.CUDF:
            import cudf
            return cudf.DataFrame(data, index=index)
        import pandas as pd
        return pd.DataFrame(data, index=index)

    # =====================================================================
    # Backend-specific I/O helpers (private)
    # =====================================================================

    def _duckdb_read_parquet(
        self,
        path: str,
        columns: Optional[List[str]],
    ) -> Any:
        """Read Parquet via DuckDB, returning a pandas DataFrame."""
        conn = self._get_duckdb_conn()
        norm = self._normalise_path(path)
        col_expr = ", ".join(f'"{c}"' for c in columns) if columns else "*"
        return conn.execute(
            f"SELECT {col_expr} FROM read_parquet('{norm}')"
        ).fetchdf()

    def _cudf_read_parquet(
        self,
        path: str,
        columns: Optional[List[str]],
    ) -> Any:
        """Read Parquet via cuDF."""
        import cudf
        return cudf.read_parquet(path, columns=columns)

    def _pandas_read_parquet(
        self,
        path: str,
        columns: Optional[List[str]],
    ) -> Any:
        """Read Parquet via pandas."""
        import pandas as pd
        return pd.read_parquet(path, columns=columns)

    def _duckdb_to_parquet(
        self,
        df: Any,
        path: str,
        compression: str,
        row_group_size: int,
    ) -> None:
        """Write Parquet via DuckDB."""
        conn = self._get_duckdb_conn()
        norm = self._normalise_path(path)
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        conn.register("_export_df", df)
        try:
            conn.execute(
                f"COPY _export_df TO '{norm}' "
                f"(FORMAT PARQUET, COMPRESSION {compression}, "
                f"ROW_GROUP_SIZE {row_group_size})"
            )
        finally:
            conn.unregister("_export_df")

    @staticmethod
    def _normalise_path(path: str) -> str:
        """Normalise a file path to forward slashes (DuckDB requirement)."""
        if path.startswith("s3://"):
            return path
        return str(Path(path).resolve()).replace("\\", "/")

    # -- Repr --------------------------------------------------------------

    def __repr__(self) -> str:  # pragma: no cover
        return (
            f"DataFrameBackend(backend={self._backend!r}, "
            f"memory_limit={self._config.duckdb_memory_limit!r})"
        )


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------

df_backend = DataFrameBackend()
"""Global singleton :class:`DataFrameBackend` instance.

All modules should import this object rather than creating their own::

    from core.data.dataframe import df_backend

    df = df_backend.read_parquet("data.parquet")
"""
