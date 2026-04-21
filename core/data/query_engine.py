"""
Query Engine — config-driven SQL execution with DuckDB + optional Athena.

DuckDB is the primary backend.  It supports local Parquet files and S3
URIs via the ``httpfs`` extension.  Athena can be configured as a
fallback for queries that must run server-side.

Configuration example (YAML)::

    query_engine:
      backend: duckdb               # duckdb | athena
      duckdb:
        memory_limit: 8GB
        threads: 4
        s3_region: <aws region>     # defaults to AWS_DEFAULT_REGION env var
        s3_access_key_id: ${AWS_ACCESS_KEY_ID}
        s3_secret_access_key: ${AWS_SECRET_ACCESS_KEY}
      athena:
        database: analytics
        workgroup: primary
        output_location: s3://bucket/athena-results/
"""

from __future__ import annotations

import logging
import os
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, Generator, List, Optional, Union

logger = logging.getLogger(__name__)

try:
    import duckdb

    _HAS_DUCKDB = True
except ImportError:
    _HAS_DUCKDB = False

try:
    import pandas as pd

    _HAS_PANDAS = True
except ImportError:
    _HAS_PANDAS = False


class QueryEngine:
    """
    Unified SQL query interface backed by DuckDB (primary) or Athena.

    Parameters
    ----------
    config : dict, optional
        Configuration dictionary.  When omitted, sensible defaults are
        used (in-memory DuckDB, 4 GB memory limit, 4 threads).
    backend : str, optional
        ``"duckdb"`` (default) or ``"athena"``.  Overrides ``config``
        when supplied explicitly.
    """

    # Supported backends
    DUCKDB = "duckdb"
    ATHENA = "athena"

    def __init__(
        self,
        config: Optional[Dict[str, Any]] = None,
        backend: Optional[str] = None,
    ) -> None:
        self._config = config or {}
        self._backend = (
            backend
            or self._config.get("query_engine", {}).get("backend", self.DUCKDB)
        )

        self._conn: Optional[duckdb.DuckDBPyConnection] = None  # type: ignore[name-defined]
        self._athena_client: Any = None

        if self._backend == self.DUCKDB:
            self._init_duckdb()
        elif self._backend == self.ATHENA:
            self._init_athena()
        else:
            raise ValueError(f"Unsupported backend: {self._backend}")

    # ── DuckDB initialisation ─────────────────────────────────────────

    def _init_duckdb(self) -> None:
        if not _HAS_DUCKDB:
            raise ImportError(
                "duckdb is required for the DuckDB backend. "
                "Install it with: pip install duckdb"
            )

        duck_cfg = self._config.get("query_engine", {}).get("duckdb", {})
        memory_limit = duck_cfg.get(
            "memory_limit",
            os.environ.get("DUCKDB_MEMORY_LIMIT", "4GB"),
        )
        threads = duck_cfg.get(
            "threads",
            int(os.environ.get("DUCKDB_THREADS", "4")),
        )

        db_path = duck_cfg.get("database", ":memory:")
        self._conn = duckdb.connect(db_path)
        self._conn.execute(f"SET memory_limit='{memory_limit}'")
        self._conn.execute(f"SET threads={threads}")
        self._conn.execute("SET preserve_insertion_order=false")

        # S3 / httpfs for remote Parquet queries
        s3_region = duck_cfg.get(
            "s3_region", os.environ.get("AWS_DEFAULT_REGION")
        )
        s3_key = duck_cfg.get(
            "s3_access_key_id", os.environ.get("AWS_ACCESS_KEY_ID")
        )
        s3_secret = duck_cfg.get(
            "s3_secret_access_key",
            os.environ.get("AWS_SECRET_ACCESS_KEY"),
        )

        if s3_region or s3_key:
            try:
                self._conn.execute("INSTALL httpfs; LOAD httpfs;")
                if s3_region:
                    self._conn.execute(
                        f"SET s3_region='{s3_region}'"
                    )
                if s3_key and s3_secret:
                    self._conn.execute(
                        f"SET s3_access_key_id='{s3_key}'"
                    )
                    self._conn.execute(
                        f"SET s3_secret_access_key='{s3_secret}'"
                    )
                logger.info("DuckDB httpfs configured (region=%s)", s3_region)
            except Exception as exc:
                logger.warning("httpfs setup skipped: %s", exc)

        # temp directory for spill
        temp_dir = duck_cfg.get("temp_directory")
        if temp_dir:
            os.makedirs(temp_dir, exist_ok=True)
            self._conn.execute(
                f"SET temp_directory='{temp_dir.replace(chr(92), '/')}'"
            )

        logger.info(
            "DuckDB backend ready (memory_limit=%s, threads=%s, db=%s)",
            memory_limit, threads, db_path,
        )

    # ── Athena initialisation ─────────────────────────────────────────

    def _init_athena(self) -> None:
        """Lazy-initialise Athena client via boto3."""
        try:
            import boto3

            athena_cfg = self._config.get("query_engine", {}).get("athena", {})
            region = athena_cfg.get(
                "region", os.environ.get("AWS_DEFAULT_REGION")
            )
            self._athena_client = boto3.client("athena", region_name=region)
            self._athena_database = athena_cfg.get("database", "default")
            self._athena_workgroup = athena_cfg.get("workgroup", "primary")
            self._athena_output = athena_cfg.get(
                "output_location", "s3://aws-athena-query-results/"
            )
            logger.info("Athena backend ready (database=%s)", self._athena_database)
        except ImportError:
            raise ImportError(
                "boto3 is required for the Athena backend. "
                "Install it with: pip install boto3"
            )

    # ── Public API ────────────────────────────────────────────────────

    def query(self, sql: str, **params: Any) -> "pd.DataFrame":
        """Execute *sql* and return a pandas DataFrame.

        Parameters
        ----------
        sql : str
            SQL query string.  For DuckDB, this can reference local
            Parquet files or ``s3://`` URIs directly via ``read_parquet()``.
        **params
            Named parameters forwarded to the backend (DuckDB
            ``execute(sql, params)`` or Athena ``QueryString``
            formatting).

        Returns
        -------
        pandas.DataFrame
        """
        if self._backend == self.DUCKDB:
            return self._query_duckdb(sql, params)
        return self._query_athena(sql, params)

    def execute(self, sql: str, **params: Any) -> Any:
        """Execute *sql* without returning results (DDL / DML).

        Returns the raw backend result object.
        """
        if self._backend == self.DUCKDB:
            return self._conn.execute(sql, params if params else None)
        return self._execute_athena(sql, params)

    def query_parquet(
        self,
        path: str,
        columns: Optional[List[str]] = None,
        where: Optional[str] = None,
        limit: Optional[int] = None,
    ) -> "pd.DataFrame":
        """Convenience wrapper to query a Parquet file or S3 URI.

        Parameters
        ----------
        path : str
            Local file path, glob pattern, or ``s3://`` URI.
        columns : list[str], optional
            Columns to select.  ``None`` means ``SELECT *``.
        where : str, optional
            SQL WHERE clause (without the ``WHERE`` keyword).
        limit : int, optional
            Maximum number of rows to return.

        Returns
        -------
        pandas.DataFrame
        """
        col_expr = ", ".join(f'"{c}"' for c in columns) if columns else "*"
        norm_path = self._normalise_path(path)
        sql = f"SELECT {col_expr} FROM read_parquet('{norm_path}')"
        if where:
            sql += f" WHERE {where}"
        if limit:
            sql += f" LIMIT {limit}"
        return self.query(sql)

    def table_schema(self, path: str) -> List[Dict[str, str]]:
        """Return column names and types for a Parquet file / table.

        Parameters
        ----------
        path : str
            Local path, glob, or ``s3://`` URI.

        Returns
        -------
        list[dict]
            Each dict has keys ``name`` and ``type``.
        """
        norm = self._normalise_path(path)
        rows = self._conn.execute(
            f"DESCRIBE SELECT * FROM read_parquet('{norm}')"
        ).fetchall()
        return [{"name": r[0], "type": r[1]} for r in rows]

    def row_count(self, path: str) -> int:
        """Return the number of rows in a Parquet file."""
        norm = self._normalise_path(path)
        return self._conn.execute(
            f"SELECT COUNT(*) FROM read_parquet('{norm}')"
        ).fetchone()[0]

    # ── Context manager ────────────────────────────────────────────────

    @contextmanager
    def transaction(self) -> Generator["QueryEngine", None, None]:
        """Context manager wrapping a DuckDB transaction.

        Usage::

            with engine.transaction():
                engine.execute("CREATE TABLE ...")
                engine.execute("INSERT INTO ...")
        """
        if self._backend != self.DUCKDB:
            yield self
            return
        self._conn.execute("BEGIN TRANSACTION")
        try:
            yield self
            self._conn.execute("COMMIT")
        except Exception:
            self._conn.execute("ROLLBACK")
            raise

    # ── Cleanup ────────────────────────────────────────────────────────

    def close(self) -> None:
        """Close the backend connection and release resources."""
        if self._conn is not None:
            self._conn.close()
            self._conn = None
        logger.info("QueryEngine closed")

    def __enter__(self) -> "QueryEngine":
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()

    # ── Private helpers ────────────────────────────────────────────────

    @staticmethod
    def _normalise_path(path: str) -> str:
        """Normalise a file path to forward slashes (DuckDB requirement)."""
        if path.startswith("s3://"):
            return path
        return str(Path(path).resolve()).replace("\\", "/")

    def _query_duckdb(
        self, sql: str, params: Dict[str, Any]
    ) -> "pd.DataFrame":
        if not _HAS_PANDAS:
            raise ImportError("pandas is required for query(). Install: pip install pandas")
        result = self._conn.execute(sql, params if params else None)
        return result.fetchdf()

    def _query_athena(
        self, sql: str, params: Dict[str, Any]
    ) -> "pd.DataFrame":
        """Submit query to Athena, wait for completion, and fetch results."""
        import time

        if not _HAS_PANDAS:
            raise ImportError("pandas is required for query()")

        response = self._athena_client.start_query_execution(
            QueryString=sql,
            QueryExecutionContext={"Database": self._athena_database},
            WorkGroup=self._athena_workgroup,
            ResultConfiguration={"OutputLocation": self._athena_output},
        )
        execution_id = response["QueryExecutionId"]
        logger.info("Athena query submitted: %s", execution_id)

        # poll for completion
        while True:
            status = self._athena_client.get_query_execution(
                QueryExecutionId=execution_id
            )
            state = status["QueryExecution"]["Status"]["State"]
            if state in ("SUCCEEDED",):
                break
            if state in ("FAILED", "CANCELLED"):
                reason = status["QueryExecution"]["Status"].get(
                    "StateChangeReason", "unknown"
                )
                raise RuntimeError(
                    f"Athena query {state}: {reason}"
                )
            time.sleep(1)

        # fetch results into a DataFrame
        result_path = status["QueryExecution"]["ResultConfiguration"][
            "OutputLocation"
        ]
        # Use DuckDB to read Athena CSV output from S3
        if self._conn is None:
            self._init_duckdb()
        return self._conn.execute(
            f"SELECT * FROM read_csv_auto('{result_path}')"
        ).fetchdf()

    def _execute_athena(
        self, sql: str, params: Dict[str, Any]
    ) -> Any:
        response = self._athena_client.start_query_execution(
            QueryString=sql,
            QueryExecutionContext={"Database": self._athena_database},
            WorkGroup=self._athena_workgroup,
            ResultConfiguration={"OutputLocation": self._athena_output},
        )
        return response

    # ── Repr ───────────────────────────────────────────────────────────

    def __repr__(self) -> str:  # pragma: no cover
        return f"QueryEngine(backend={self._backend!r})"
