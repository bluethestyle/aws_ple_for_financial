"""
ComplianceSQLHelper (Sprint 2 S6) — DuckDB-over-Parquet compliance audit
SQL queries without Athena.

On-premises the system uses DuckDB as the SQL engine over Parquet
compliance archives. AWS offers several paid alternatives (Athena, S3
Select, Redshift Spectrum), but DuckDB's httpfs extension can read
`s3://` URIs directly at zero infrastructure cost. For the expected
query volume (regulator requests + quarterly reports + ad-hoc incident
review) DuckDB is sufficient and keeps the on-premises experience
intact.

Scope
-----
- Register S3 / local Parquet paths as DuckDB views
- Run arbitrary SQL on the registered views
- Ship a handful of regulator-style convenience queries
  (recent opt-outs, consent changes per user, explanation SLA breaches,
   promotion-gate history)

This helper is a *thin wrapper* — it does not maintain state across
processes and is meant for on-demand use (not a long-running service).
For the online DynamoDB tables use `ComplianceAuditStore` (PartiQL)
instead; for SageMaker Experiments trial components use
`SageMakerComplianceTracker`.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Sequence

logger = logging.getLogger(__name__)

__all__ = [
    "AuditSQLConfig",
    "ComplianceSQLHelper",
    "build_compliance_sql_helper",
]


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class AuditSQLConfig:
    """Driven by ``compliance.audit_sql`` in pipeline.yaml."""

    enabled: bool = True
    paths: Dict[str, str] = None        # view_name -> parquet glob / s3 uri
    default_since_days: int = 30
    install_httpfs: bool = True         # required for s3:// reads

    def __post_init__(self) -> None:
        if self.paths is None:
            self.paths = {}
        if self.default_since_days <= 0:
            raise ValueError("default_since_days must be > 0")

    @classmethod
    def from_dict(cls, data: Optional[Dict[str, Any]]) -> "AuditSQLConfig":
        if not data:
            return cls()
        return cls(
            enabled=bool(data.get("enabled", True)),
            paths=dict(data.get("paths", {})),
            default_since_days=int(data.get("default_since_days", 30)),
            install_httpfs=bool(data.get("install_httpfs", True)),
        )


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

class ComplianceSQLHelper:
    """Zero-infrastructure SQL over compliance Parquet archives."""

    def __init__(
        self,
        config: Optional[AuditSQLConfig] = None,
        duckdb_conn: Any = None,
    ) -> None:
        self._cfg = config or AuditSQLConfig()
        self._conn = duckdb_conn
        self._owns_conn = False
        self._registered: Dict[str, str] = {}

        if self._conn is None:
            try:
                import duckdb  # type: ignore
                self._conn = duckdb.connect(":memory:")
                self._owns_conn = True
            except ImportError as exc:
                raise RuntimeError(
                    "duckdb not installed; install duckdb or inject a conn"
                ) from exc
            if self._cfg.install_httpfs:
                try:
                    self._conn.execute("INSTALL httpfs; LOAD httpfs;")
                except Exception:
                    logger.warning(
                        "DuckDB httpfs extension unavailable — s3:// reads "
                        "may fail. Local Parquet paths still work.",
                    )

        # Bulk-register any paths declared in config
        for name, uri in (self._cfg.paths or {}).items():
            try:
                self.register_view(name, uri)
            except Exception:
                logger.exception(
                    "Failed to register configured view %s -> %s", name, uri,
                )

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def register_view(self, name: str, parquet_uri: str) -> None:
        """Register a Parquet path (glob / s3 URI / local dir) as a view.

        The view is re-creatable so repeat calls overwrite the previous
        binding, which is useful when new batches are added to an
        existing prefix.
        """
        if not name or not name.replace("_", "").isalnum():
            raise ValueError(
                f"view name={name!r} must be [A-Za-z0-9_]+"
            )
        quoted = parquet_uri.replace("'", "''")
        self._conn.execute(
            f"CREATE OR REPLACE VIEW {name} AS "
            f"SELECT * FROM read_parquet('{quoted}')"
        )
        self._registered[name] = parquet_uri
        logger.info("Registered compliance SQL view: %s -> %s",
                    name, parquet_uri)

    def list_views(self) -> Dict[str, str]:
        return dict(self._registered)

    def close(self) -> None:
        if self._owns_conn and self._conn is not None:
            try:
                self._conn.close()
            except Exception:
                pass

    def __enter__(self) -> "ComplianceSQLHelper":
        return self

    def __exit__(self, *_: Any) -> None:
        self.close()

    # ------------------------------------------------------------------
    # Query surface
    # ------------------------------------------------------------------

    def query(
        self,
        sql: str,
        params: Optional[Sequence[Any]] = None,
    ) -> List[Dict[str, Any]]:
        """Execute arbitrary SQL and return row dicts."""
        if not self._cfg.enabled:
            logger.debug("compliance SQL helper disabled; returning []")
            return []
        result = self._conn.execute(sql, params) if params else \
            self._conn.execute(sql)
        cols = [d[0] for d in result.description]
        return [dict(zip(cols, row)) for row in result.fetchall()]

    # ------------------------------------------------------------------
    # Pre-built regulator queries
    # ------------------------------------------------------------------

    def recent_opt_outs(
        self,
        since: Optional[datetime] = None,
        view: str = "opt_out",
    ) -> List[Dict[str, Any]]:
        """Opt-out events recorded since ``since`` (default: 30 days)."""
        cutoff = since or self._default_cutoff()
        return self.query(
            f"SELECT * FROM {view} "
            f"WHERE CAST(timestamp AS TIMESTAMP) >= CAST(? AS TIMESTAMP) "
            f"ORDER BY timestamp DESC",
            [cutoff.isoformat()],
        )

    def consent_changes_for_user(
        self,
        user_id: str,
        since: Optional[datetime] = None,
        view: str = "consent",
    ) -> List[Dict[str, Any]]:
        cutoff = since or self._default_cutoff()
        return self.query(
            f"SELECT * FROM {view} "
            f"WHERE user_id = ? "
            f"AND CAST(timestamp AS TIMESTAMP) >= CAST(? AS TIMESTAMP) "
            f"ORDER BY timestamp DESC",
            [user_id, cutoff.isoformat()],
        )

    def sla_breaches(
        self,
        since: Optional[datetime] = None,
        view: str = "events",
        event_type_column: str = "event_type",
        event_type_value: str = "sla_breach",
    ) -> List[Dict[str, Any]]:
        cutoff = since or self._default_cutoff()
        return self.query(
            f"SELECT * FROM {view} "
            f"WHERE {event_type_column} = ? "
            f"AND CAST(timestamp AS TIMESTAMP) >= CAST(? AS TIMESTAMP) "
            f"ORDER BY timestamp DESC",
            [event_type_value, cutoff.isoformat()],
        )

    def promotion_gate_history(
        self,
        model_version: Optional[str] = None,
        view: str = "promotion_gate",
    ) -> List[Dict[str, Any]]:
        if model_version is not None:
            return self.query(
                f"SELECT * FROM {view} WHERE model_version = ? "
                f"ORDER BY timestamp",
                [model_version],
            )
        return self.query(
            f"SELECT * FROM {view} ORDER BY timestamp"
        )

    def counts_by_column(
        self,
        view: str,
        column: str,
    ) -> Dict[str, int]:
        """Return a `{value: count}` map. Useful for summaries."""
        # Column name is trusted (validated at registration time for views);
        # we still inline via identifier quoting to avoid injection via
        # caller-supplied aliases.
        safe_col = column.replace('"', '""')
        rows = self.query(
            f'SELECT "{safe_col}" AS k, COUNT(*) AS n FROM {view} '
            f'GROUP BY "{safe_col}" ORDER BY n DESC'
        )
        return {str(r.get("k")): int(r.get("n", 0)) for r in rows}

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _default_cutoff(self) -> datetime:
        return datetime.now(timezone.utc) - timedelta(
            days=self._cfg.default_since_days
        )


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def build_compliance_sql_helper(
    pipeline_config: Optional[Dict[str, Any]] = None,
    duckdb_conn: Any = None,
) -> ComplianceSQLHelper:
    """Instantiate from ``compliance.audit_sql`` block of pipeline.yaml."""
    compliance_cfg = (pipeline_config or {}).get("compliance") or {}
    cfg = AuditSQLConfig.from_dict(compliance_cfg.get("audit_sql"))
    return ComplianceSQLHelper(config=cfg, duckdb_conn=duckdb_conn)
