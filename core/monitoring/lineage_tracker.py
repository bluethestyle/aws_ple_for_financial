"""
Data lineage tracker for feature-to-source tracing.

Provides end-to-end lineage from a recommendation decision back through
the features used, to the original source tables and columns.

Features:
- Config-driven feature-to-source mapping (no hardcoded table names)
- Recommendation lineage: top-K IG features -> source data
- LineageRecord with execution_id, inputs, outputs, config_snapshot
- S3 storage for lineage records
"""

from __future__ import annotations

import json
import logging
import os
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Set

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Default feature-source mapping (overridable via config)
# ---------------------------------------------------------------------------

DEFAULT_FEATURE_SOURCE_MAP: Dict[str, Dict[str, Any]] = {
    "spend_": {
        "feature_count": 12,
        "source_tables": ["card_transaction", "card_monthly_summary"],
        "source_columns": ["txn_amount", "merchant_category", "txn_date"],
        "data_group": "transaction",
        "description": "Monthly spending patterns from card transactions",
        "pseudonymized": True,
    },
    "txn_count_": {
        "feature_count": 12,
        "source_tables": ["card_transaction", "account_transaction"],
        "source_columns": ["txn_count", "txn_type", "channel_code"],
        "data_group": "transaction",
        "description": "Transaction count patterns",
        "pseudonymized": True,
    },
    "amt_": {
        "feature_count": 24,
        "source_tables": ["card_transaction", "account_balance", "deposit_summary"],
        "source_columns": ["txn_amount", "balance", "avg_balance"],
        "data_group": "transaction",
        "description": "Transaction amount statistics",
        "pseudonymized": True,
    },
    "merchant_": {
        "feature_count": 48,
        "source_tables": ["merchant_master", "card_transaction"],
        "source_columns": ["merchant_id", "merchant_name", "mcc_code"],
        "data_group": "transaction",
        "description": "Merchant usage patterns",
        "pseudonymized": True,
    },
    "product_": {
        "feature_count": 18,
        "source_tables": ["product_master", "product_holding"],
        "source_columns": ["product_id", "product_type", "holding_status"],
        "data_group": "product",
        "description": "Product holding features",
        "pseudonymized": True,
    },
    "age_": {
        "feature_count": 5,
        "source_tables": ["customer_profile"],
        "source_columns": ["birth_year", "age_group"],
        "data_group": "customer",
        "description": "Age-based features",
        "pseudonymized": True,
    },
    "life_stage_": {
        "feature_count": 6,
        "source_tables": ["customer_profile", "customer_lifecycle"],
        "source_columns": ["life_stage_code", "family_type"],
        "data_group": "customer",
        "description": "Customer life-stage classification",
        "pseudonymized": True,
    },
    "temporal_": {
        "feature_count": None,
        "source_tables": ["card_transaction", "app_log"],
        "source_columns": ["txn_date", "txn_hour", "day_of_week"],
        "data_group": "activity",
        "description": "Temporal usage patterns",
        "pseudonymized": True,
    },
    "benefit_": {
        "feature_count": 12,
        "source_tables": ["product_benefit", "benefit_usage"],
        "source_columns": ["benefit_type", "benefit_amount", "usage_count"],
        "data_group": "product",
        "description": "Product benefit matching features",
        "pseudonymized": False,
    },
}


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class LineageRecord:
    """Represents a single lineage trace."""

    execution_id: str
    traced_at: str
    lineage_type: str           # "feature" | "recommendation" | "batch"
    inputs: Dict[str, Any]      # What was queried
    outputs: Dict[str, Any]     # Trace results
    config_snapshot: Dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# DataLineageTracker
# ---------------------------------------------------------------------------

class DataLineageTracker:
    """Track data lineage from features to source tables.

    Parameters
    ----------
    feature_source_map : dict, optional
        Custom feature-prefix-to-source mapping.  Defaults to
        ``DEFAULT_FEATURE_SOURCE_MAP``.
    s3_bucket : str, optional
        S3 bucket for lineage record storage.
    s3_prefix : str
        Key prefix inside the bucket.
    """

    def __init__(
        self,
        feature_source_map: Optional[Dict[str, Dict[str, Any]]] = None,
        s3_bucket: Optional[str] = None,
        s3_prefix: str = "lineage",
    ) -> None:
        self.feature_source_map = feature_source_map or dict(DEFAULT_FEATURE_SOURCE_MAP)
        self.s3_bucket = s3_bucket or os.environ.get("LINEAGE_S3_BUCKET", "")
        self.s3_prefix = s3_prefix.strip("/")

        self._s3_client = None
        if self.s3_bucket:
            try:
                import boto3

                region = os.environ.get("AWS_DEFAULT_REGION", "ap-northeast-2")
                self._s3_client = boto3.client("s3", region_name=region)
            except Exception as exc:
                logger.warning("S3 client init failed (lineage): %s", exc)

    # ------------------------------------------------------------------
    # Feature-level tracing
    # ------------------------------------------------------------------

    def trace_feature_to_source(self, feature_name: str) -> Dict[str, Any]:
        """Trace a single feature back to its source table(s) and column(s).

        Parameters
        ----------
        feature_name : str
            Feature name (e.g. ``"spend_dining"``).

        Returns
        -------
        dict
            Source information including ``source_tables``, ``source_columns``,
            ``data_group``, etc.  If no mapping is found, returns an
            ``"unmapped"`` status entry.
        """
        for prefix, info in self.feature_source_map.items():
            if feature_name.startswith(prefix):
                return {
                    "feature_name": feature_name,
                    "matched_prefix": prefix,
                    "source_tables": info["source_tables"],
                    "source_columns": info["source_columns"],
                    "data_group": info["data_group"],
                    "description": info["description"],
                    "pseudonymized": info.get("pseudonymized"),
                    "lineage_depth": 3,
                }

        logger.warning("No lineage mapping for feature: %s", feature_name)
        return {
            "feature_name": feature_name,
            "matched_prefix": None,
            "status": "unmapped",
            "note": "No mapping information available.",
        }

    def trace_features_batch(self, feature_names: List[str]) -> List[Dict[str, Any]]:
        """Trace multiple features at once.

        Parameters
        ----------
        feature_names : list of str
            Feature names to trace.

        Returns
        -------
        list of dict
            One trace result per feature.
        """
        return [self.trace_feature_to_source(fn) for fn in feature_names]

    # ------------------------------------------------------------------
    # Sprint 2 S9: extensible feature-to-table mapping
    # ------------------------------------------------------------------

    def register_feature_mapping(
        self,
        prefix: str,
        source_tables: List[str],
        source_columns: List[str],
        data_group: str,
        description: str = "",
        pseudonymized: Optional[bool] = None,
    ) -> None:
        """Register a new feature-prefix-to-source mapping at runtime."""
        if not prefix:
            raise ValueError("prefix must be non-empty")
        entry: Dict[str, Any] = {
            "source_tables": list(source_tables),
            "source_columns": list(source_columns),
            "data_group": data_group,
            "description": description,
        }
        if pseudonymized is not None:
            entry["pseudonymized"] = bool(pseudonymized)
        self.feature_source_map[prefix] = entry
        logger.info(
            "Registered lineage mapping: prefix=%s tables=%s data_group=%s",
            prefix, source_tables, data_group,
        )

    def load_mapping_from_yaml(self, yaml_path: str) -> int:
        """Load a YAML-defined feature mapping catalog.

        Expected YAML shape::

            feature_source_map:
              spend_:
                source_tables: [T_TXN, T_MCC]
                source_columns: [amount, mcc_code]
                data_group: G2_transactions
                description: "Customer spending patterns by MCC"
                pseudonymized: true
              txn_count_: {...}

        Returns the number of prefix entries loaded.
        """
        try:
            import yaml
        except ImportError as exc:
            raise RuntimeError("PyYAML required for load_mapping_from_yaml") from exc

        from pathlib import Path

        p = Path(yaml_path)
        if not p.exists():
            raise FileNotFoundError(yaml_path)
        data = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
        raw_map = (
            data.get("feature_source_map")
            or data.get("lineage", {}).get("feature_source_map")
            or {}
        )
        count = 0
        for prefix, info in raw_map.items():
            self.register_feature_mapping(
                prefix=prefix,
                source_tables=info.get("source_tables", []),
                source_columns=info.get("source_columns", []),
                data_group=info.get("data_group", "unknown"),
                description=info.get("description", ""),
                pseudonymized=info.get("pseudonymized"),
            )
            count += 1
        logger.info(
            "Loaded %d lineage prefix mappings from %s", count, yaml_path,
        )
        return count

    def coverage_report(
        self, feature_names: List[str],
    ) -> Dict[str, Any]:
        """Summarize how many of ``feature_names`` have a mapping."""
        total = len(feature_names)
        unmapped: List[str] = []
        by_group: Dict[str, int] = {}
        by_table: Dict[str, int] = {}
        for fn in feature_names:
            trace = self.trace_feature_to_source(fn)
            if trace.get("status") == "unmapped":
                unmapped.append(fn)
                continue
            grp = trace.get("data_group", "unknown")
            by_group[grp] = by_group.get(grp, 0) + 1
            for tbl in trace.get("source_tables", []):
                by_table[tbl] = by_table.get(tbl, 0) + 1
        mapped = total - len(unmapped)
        return {
            "total_features": total,
            "mapped": mapped,
            "unmapped": unmapped,
            "coverage_rate": (mapped / total) if total else 0.0,
            "by_data_group": by_group,
            "by_source_table": by_table,
        }

    # ------------------------------------------------------------------
    # Recommendation-level tracing
    # ------------------------------------------------------------------

    def trace_recommendation_lineage(
        self,
        customer_id: str,
        recommendation_id: str,
        ig_features: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        """Trace the full lineage of a recommendation decision.

        Parameters
        ----------
        customer_id : str
            Customer identifier.
        recommendation_id : str
            Recommendation or batch identifier.
        ig_features : list of dict, optional
            Top-K Integrated Gradient features, each with ``"name"`` and
            ``"score"`` keys.

        Returns
        -------
        dict
            Complete lineage including per-feature source traces.
        """
        lineage: Dict[str, Any] = {
            "customer_id": customer_id,
            "recommendation_id": recommendation_id,
            "traced_at": datetime.now(timezone.utc).isoformat(),
            "feature_lineage": [],
            "data_groups_involved": [],
            "total_source_tables": [],
        }

        if not ig_features:
            return lineage

        data_groups: Set[str] = set()
        source_tables: Set[str] = set()

        for rank, feat in enumerate(ig_features, 1):
            fname = feat.get("name", "")
            fscore = feat.get("score", 0.0)
            source_trace = self.trace_feature_to_source(fname)
            lineage["feature_lineage"].append({
                "rank": rank,
                "feature_name": fname,
                "ig_score": fscore,
                "source_trace": source_trace,
            })
            dg = source_trace.get("data_group")
            if dg and dg != "unknown":
                data_groups.add(dg)
            for tbl in source_trace.get("source_tables", []):
                source_tables.add(tbl)

        lineage["data_groups_involved"] = sorted(data_groups)
        lineage["total_source_tables"] = sorted(source_tables)
        return lineage

    # ------------------------------------------------------------------
    # Batch lineage report
    # ------------------------------------------------------------------

    def generate_lineage_report(self, batch_date: str) -> Dict[str, Any]:
        """Generate a comprehensive lineage report for all feature groups.

        Parameters
        ----------
        batch_date : str
            Batch date (``YYYY-MM-DD``).

        Returns
        -------
        dict
            Report with per-group lineage and summary statistics.
        """
        all_tables: Set[str] = set()
        pseudonymized_count = 0
        non_pseudonymized_count = 0

        feature_groups: Dict[str, Any] = {}
        for prefix, info in self.feature_source_map.items():
            feature_groups[prefix] = {
                **info,
                "prefix": prefix,
                "source_table_count": len(info["source_tables"]),
            }
            for tbl in info["source_tables"]:
                all_tables.add(tbl)
            if info.get("pseudonymized"):
                pseudonymized_count += 1
            else:
                non_pseudonymized_count += 1

        data_groups_covered = sorted(
            {info["data_group"] for info in self.feature_source_map.values()}
        )

        return {
            "title": "Data Lineage Report",
            "batch_date": batch_date,
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "feature_groups": feature_groups,
            "summary": {
                "total_feature_groups": len(self.feature_source_map),
                "total_source_tables": len(all_tables),
                "pseudonymized_groups": pseudonymized_count,
                "non_pseudonymized_groups": non_pseudonymized_count,
                "data_groups_covered": data_groups_covered,
                "all_source_tables": sorted(all_tables),
            },
        }

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save_lineage(
        self,
        lineage: Dict[str, Any],
        lineage_type: str = "single",
    ) -> LineageRecord:
        """Persist a lineage trace to S3.

        Parameters
        ----------
        lineage : dict
            Lineage data (output of ``trace_*`` or ``generate_lineage_report``).
        lineage_type : str
            ``"single"`` (per-recommendation), ``"batch"``, or ``"report"``.

        Returns
        -------
        LineageRecord
            The persisted record with execution_id.
        """
        execution_id = str(uuid.uuid4())
        now = datetime.now(timezone.utc)

        record = LineageRecord(
            execution_id=execution_id,
            traced_at=now.isoformat(),
            lineage_type=lineage_type,
            inputs={
                "customer_id": lineage.get("customer_id"),
                "recommendation_id": lineage.get("recommendation_id"),
                "batch_date": lineage.get("batch_date"),
            },
            outputs=lineage,
            config_snapshot={
                "feature_source_map_keys": list(self.feature_source_map.keys()),
            },
        )

        # S3 persistence
        if self._s3_client and self.s3_bucket:
            try:
                date_str = now.strftime("%Y-%m-%d")
                s3_key = (
                    f"{self.s3_prefix}/{date_str}/"
                    f"lineage_{lineage_type}_{execution_id}.json"
                )
                body = json.dumps(
                    {
                        "execution_id": record.execution_id,
                        "traced_at": record.traced_at,
                        "lineage_type": record.lineage_type,
                        "inputs": record.inputs,
                        "outputs": record.outputs,
                        "config_snapshot": record.config_snapshot,
                    },
                    ensure_ascii=False,
                    indent=2,
                    default=str,
                ).encode("utf-8")
                self._s3_client.put_object(
                    Bucket=self.s3_bucket,
                    Key=s3_key,
                    Body=body,
                    ContentType="application/json",
                )
                logger.info("Lineage record saved: s3://%s/%s", self.s3_bucket, s3_key)
            except Exception as exc:
                logger.warning("S3 lineage save failed: %s", exc)
        else:
            logger.info("Lineage record created (execution_id=%s) but S3 not configured.", execution_id)

        return record


__all__ = ["DataLineageTracker", "LineageRecord"]
