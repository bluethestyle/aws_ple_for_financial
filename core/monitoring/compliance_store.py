"""
Compliance Audit Store backed by Amazon DynamoDB.

Provides append-only audit tables for regulatory compliance events:

- **ks_audit**           Kill-switch activation / deactivation history
- **consent_audit**      Marketing consent changes
- **profiling_audit**    Data-subject profiling rights exercises
- **opt_out_audit**      AI decision opt-out history
- **incident_audit**     Incident records
- **distillation_audit** Distillation model validation pass/fail
- **embedding_audit**    Embedding generation quality records

All writes use DynamoDB ``put_item`` (append-only, no update/delete by design).
Reads support flexible queries via the ``query_table`` helper.
"""

from __future__ import annotations

import json
import logging
import os
import uuid
from datetime import datetime, timezone
from decimal import Decimal
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Table definitions (used for auto-creation and validation)
# ---------------------------------------------------------------------------

# Each table uses ``pk`` (partition key) and ``sk`` (sort key = ISO timestamp).
# This allows efficient time-range queries within a partition.
AUDIT_TABLES: Dict[str, Dict[str, str]] = {
    "ks_audit": {
        "pk_description": "action#level (e.g. ACTIVATE#global)",
        "attributes": "action, key, level, operator_id, reason",
    },
    "consent_audit": {
        "pk_description": "customer_id",
        "attributes": "consent_type, action, old_value, new_value, channel, batch_date",
    },
    "profiling_audit": {
        "pk_description": "customer_id",
        "attributes": "request_id, action_type, fields_json, old_values_json, new_values_json, status, reason, requested_at, processed_at, processor_id",
    },
    "opt_out_audit": {
        "pk_description": "customer_id",
        "attributes": "action, reason, details",
    },
    "incident_audit": {
        "pk_description": "incident_id",
        "attributes": "event_type, severity, source_module, status, description, resolution, resolved_at",
    },
    "distillation_audit": {
        "pk_description": "task_name",
        "attributes": "passed, auc_gap, ranking_corr, teacher_metric, student_metric, metric_name, issues_json, model_version, execution_date",
    },
    "embedding_audit": {
        "pk_description": "embedding_type",
        "attributes": "passed, num_nodes, num_edges, num_covisit_edges, final_loss, convergence_ratio, num_output_users, norm_mean, norm_std, nan_count, origin_collapse_pct, issues_json, config_json, model_version, execution_date",
    },
}


def _decimal_safe(value: Any) -> Any:
    """Convert float values to Decimal for DynamoDB compatibility."""
    if isinstance(value, float):
        return Decimal(str(value))
    if isinstance(value, dict):
        return {k: _decimal_safe(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_decimal_safe(v) for v in value]
    return value


def _from_decimal(value: Any) -> Any:
    """Convert Decimal values back to float for JSON serialization."""
    if isinstance(value, Decimal):
        return float(value)
    if isinstance(value, dict):
        return {k: _from_decimal(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_from_decimal(v) for v in value]
    return value


class ComplianceAuditStore:
    """DynamoDB-backed compliance audit store.

    Parameters
    ----------
    table_prefix : str
        Prefix prepended to each logical table name to form the DynamoDB
        table name (e.g. ``"prod_"`` -> ``"prod_ks_audit"``).
    region : str
        AWS region for DynamoDB.
    create_tables : bool
        If ``True``, create tables on initialization when they do not exist.
    """

    def __init__(
        self,
        table_prefix: Optional[str] = None,
        region: Optional[str] = None,
        create_tables: bool = False,
    ) -> None:
        self.table_prefix = table_prefix or os.environ.get("COMPLIANCE_TABLE_PREFIX", "")
        self.region = region or os.environ.get("AWS_DEFAULT_REGION", "ap-northeast-2")

        try:
            import boto3

            self._dynamodb = boto3.resource("dynamodb", region_name=self.region)
            self._client = boto3.client("dynamodb", region_name=self.region)
        except Exception as exc:
            logger.warning("DynamoDB client init failed: %s", exc)
            self._dynamodb = None
            self._client = None

        if create_tables and self._client is not None:
            self._ensure_tables_exist()

    # ------------------------------------------------------------------
    # Table management
    # ------------------------------------------------------------------

    def _full_table_name(self, logical_name: str) -> str:
        return f"{self.table_prefix}{logical_name}"

    def _ensure_tables_exist(self) -> None:
        """Create DynamoDB tables if they do not exist."""
        existing: set[str] = set()
        try:
            paginator = self._client.get_paginator("list_tables")
            for page in paginator.paginate():
                existing.update(page.get("TableNames", []))
        except Exception as exc:
            logger.warning("Cannot list DynamoDB tables: %s", exc)
            return

        for logical_name in AUDIT_TABLES:
            full_name = self._full_table_name(logical_name)
            if full_name in existing:
                continue
            try:
                self._client.create_table(
                    TableName=full_name,
                    KeySchema=[
                        {"AttributeName": "pk", "KeyType": "HASH"},
                        {"AttributeName": "sk", "KeyType": "RANGE"},
                    ],
                    AttributeDefinitions=[
                        {"AttributeName": "pk", "AttributeType": "S"},
                        {"AttributeName": "sk", "AttributeType": "S"},
                    ],
                    BillingMode="PAY_PER_REQUEST",
                )
                logger.info("Created DynamoDB table: %s", full_name)
            except Exception as exc:
                logger.warning("Failed to create table %s: %s", full_name, exc)

    def _put_item(self, logical_table: str, pk: str, item: Dict[str, Any]) -> None:
        """Write a single item to the specified audit table."""
        if self._dynamodb is None:
            logger.error("DynamoDB not available; audit record dropped for %s", logical_table)
            return

        table = self._dynamodb.Table(self._full_table_name(logical_table))
        ts = datetime.now(timezone.utc).isoformat()
        record = {
            "pk": pk,
            "sk": ts,
            "record_id": str(uuid.uuid4()),
            **_decimal_safe(item),
        }
        # Remove None values (DynamoDB does not accept None)
        record = {k: v for k, v in record.items() if v is not None}
        table.put_item(Item=record)

    # ------------------------------------------------------------------
    # Kill-switch audit
    # ------------------------------------------------------------------

    def log_kill_switch(
        self,
        action: str,
        key: str,
        level: str,
        operator_id: Optional[str] = None,
        reason: Optional[str] = None,
    ) -> None:
        """Log a kill-switch activation or deactivation event.

        Parameters
        ----------
        action : str
            ``"ACTIVATE"`` or ``"DEACTIVATE"``.
        key : str
            Scope key (e.g. ``"global"``, ``"task:cvr"``).
        level : str
            Scope level (``"global"`` / ``"per_task"`` / ``"per_cluster"``).
        operator_id : str, optional
            Identity of the operator.
        reason : str, optional
            Free-text reason for the action.
        """
        self._put_item(
            "ks_audit",
            pk=f"{action}#{level}",
            item={
                "action": action,
                "key": key,
                "level": level,
                "operator_id": operator_id,
                "reason": reason,
            },
        )

    # ------------------------------------------------------------------
    # Consent audit
    # ------------------------------------------------------------------

    def log_consent(
        self,
        customer_id: str,
        action: str,
        consent_type: Optional[str] = None,
        old_value: Optional[str] = None,
        new_value: Optional[str] = None,
        channel: Optional[str] = None,
        batch_date: Optional[str] = None,
    ) -> None:
        """Log a marketing consent change event."""
        self._put_item(
            "consent_audit",
            pk=customer_id,
            item={
                "consent_type": consent_type,
                "action": action,
                "old_value": old_value,
                "new_value": new_value,
                "channel": channel,
                "batch_date": batch_date,
            },
        )

    def log_consent_batch(self, records: List[Dict[str, Any]]) -> None:
        """Batch-write consent audit records.

        Uses DynamoDB ``batch_writer`` for efficient throughput.
        """
        if not records or self._dynamodb is None:
            return
        table = self._dynamodb.Table(self._full_table_name("consent_audit"))
        with table.batch_writer() as batch:
            for r in records:
                ts = datetime.now(timezone.utc).isoformat()
                item = {
                    "pk": r.get("customer_id", "unknown"),
                    "sk": ts,
                    "record_id": str(uuid.uuid4()),
                    "consent_type": r.get("consent_type"),
                    "action": r.get("action"),
                    "old_value": r.get("old_value"),
                    "new_value": r.get("new_value"),
                    "channel": r.get("channel"),
                    "batch_date": r.get("batch_date"),
                }
                item = {k: v for k, v in item.items() if v is not None}
                batch.put_item(Item=_decimal_safe(item))

    # ------------------------------------------------------------------
    # Profiling rights audit
    # ------------------------------------------------------------------

    def log_profiling(
        self,
        request_id: str,
        customer_id: str,
        action_type: str,
        status: str,
        fields_json: Optional[str] = None,
        old_values_json: Optional[str] = None,
        new_values_json: Optional[str] = None,
        reason: Optional[str] = None,
        requested_at: Optional[str] = None,
        processed_at: Optional[str] = None,
        processor_id: Optional[str] = None,
    ) -> None:
        """Log a data-subject profiling rights exercise."""
        self._put_item(
            "profiling_audit",
            pk=customer_id,
            item={
                "request_id": request_id,
                "action_type": action_type,
                "fields_json": fields_json,
                "old_values_json": old_values_json,
                "new_values_json": new_values_json,
                "status": status,
                "reason": reason,
                "requested_at": requested_at,
                "processed_at": processed_at,
                "processor_id": processor_id,
            },
        )

    # ------------------------------------------------------------------
    # Opt-out audit
    # ------------------------------------------------------------------

    def log_opt_out(
        self,
        customer_id: str,
        action: str,
        reason: Optional[str] = None,
        details: Optional[str] = None,
    ) -> None:
        """Log an AI-decision opt-out event."""
        self._put_item(
            "opt_out_audit",
            pk=customer_id,
            item={"action": action, "reason": reason, "details": details},
        )

    # ------------------------------------------------------------------
    # Incident audit
    # ------------------------------------------------------------------

    def log_incident(
        self,
        incident_id: str,
        event_type: str,
        severity: str,
        source_module: Optional[str] = None,
        status: Optional[str] = None,
        description: Optional[str] = None,
        resolution: Optional[str] = None,
        resolved_at: Optional[str] = None,
    ) -> None:
        """Log an incident record."""
        self._put_item(
            "incident_audit",
            pk=incident_id,
            item={
                "event_type": event_type,
                "severity": severity,
                "source_module": source_module,
                "status": status,
                "description": description,
                "resolution": resolution,
                "resolved_at": resolved_at,
            },
        )

    # ------------------------------------------------------------------
    # Distillation audit
    # ------------------------------------------------------------------

    def log_distillation(
        self,
        task_name: str,
        passed: bool,
        auc_gap: Optional[float] = None,
        ranking_corr: Optional[float] = None,
        teacher_metric: Optional[float] = None,
        student_metric: Optional[float] = None,
        metric_name: Optional[str] = None,
        issues_json: Optional[str] = None,
        model_version: Optional[str] = None,
        execution_date: Optional[str] = None,
    ) -> None:
        """Log a distillation model validation result."""
        self._put_item(
            "distillation_audit",
            pk=task_name,
            item={
                "passed": passed,
                "auc_gap": auc_gap,
                "ranking_corr": ranking_corr,
                "teacher_metric": teacher_metric,
                "student_metric": student_metric,
                "metric_name": metric_name,
                "issues_json": issues_json,
                "model_version": model_version,
                "execution_date": execution_date,
            },
        )

    # ------------------------------------------------------------------
    # Embedding audit
    # ------------------------------------------------------------------

    def log_embedding(
        self,
        embedding_type: str,
        passed: bool,
        num_nodes: Optional[int] = None,
        num_edges: Optional[int] = None,
        final_loss: Optional[float] = None,
        convergence_ratio: Optional[float] = None,
        norm_mean: Optional[float] = None,
        norm_std: Optional[float] = None,
        nan_count: Optional[int] = None,
        issues_json: Optional[str] = None,
        config_json: Optional[str] = None,
        model_version: Optional[str] = None,
        execution_date: Optional[str] = None,
    ) -> None:
        """Log an embedding generation quality record."""
        self._put_item(
            "embedding_audit",
            pk=embedding_type,
            item={
                "passed": passed,
                "num_nodes": num_nodes,
                "num_edges": num_edges,
                "final_loss": final_loss,
                "convergence_ratio": convergence_ratio,
                "norm_mean": norm_mean,
                "norm_std": norm_std,
                "nan_count": nan_count,
                "issues_json": issues_json,
                "config_json": config_json,
                "model_version": model_version,
                "execution_date": execution_date,
            },
        )

    # ------------------------------------------------------------------
    # Query helpers
    # ------------------------------------------------------------------

    def query_table(
        self,
        table: str,
        pk_value: Optional[str] = None,
        limit: int = 100,
    ) -> List[Dict[str, Any]]:
        """Query records from an audit table.

        Parameters
        ----------
        table : str
            Logical table name (e.g. ``"ks_audit"``).
        pk_value : str, optional
            Partition key value to filter on.  If ``None``, a scan is
            performed (use sparingly).
        limit : int
            Maximum number of records to return.

        Returns
        -------
        list of dict
            Matching records, newest first.
        """
        if table not in AUDIT_TABLES:
            raise ValueError(f"Unknown audit table: {table}")
        if self._dynamodb is None:
            logger.warning("DynamoDB not available; returning empty result.")
            return []

        ddb_table = self._dynamodb.Table(self._full_table_name(table))

        try:
            if pk_value:
                from boto3.dynamodb.conditions import Key

                response = ddb_table.query(
                    KeyConditionExpression=Key("pk").eq(pk_value),
                    ScanIndexForward=False,  # newest first
                    Limit=limit,
                )
            else:
                response = ddb_table.scan(Limit=limit)

            items = response.get("Items", [])
            return [_from_decimal(item) for item in items]
        except Exception as exc:
            logger.warning("Query on %s failed: %s", table, exc)
            return []

    def query_incidents(
        self,
        severity: Optional[str] = None,
        limit: int = 100,
    ) -> List[Dict[str, Any]]:
        """Query incident records, optionally filtering by severity."""
        if self._dynamodb is None:
            return []
        ddb_table = self._dynamodb.Table(self._full_table_name("incident_audit"))
        try:
            if severity:
                from boto3.dynamodb.conditions import Attr

                response = ddb_table.scan(
                    FilterExpression=Attr("severity").eq(severity),
                    Limit=limit,
                )
            else:
                response = ddb_table.scan(Limit=limit)
            items = response.get("Items", [])
            return [_from_decimal(item) for item in items]
        except Exception as exc:
            logger.warning("Incident query failed: %s", exc)
            return []


# ---------------------------------------------------------------------------
# Module-level singleton accessor
# ---------------------------------------------------------------------------

_default_store: Optional[ComplianceAuditStore] = None


def get_compliance_audit_store(**kwargs: Any) -> ComplianceAuditStore:
    """Return a module-level singleton ``ComplianceAuditStore``."""
    global _default_store
    if _default_store is None:
        _default_store = ComplianceAuditStore(**kwargs)
    return _default_store


__all__ = [
    "ComplianceAuditStore",
    "get_compliance_audit_store",
    "AUDIT_TABLES",
]
