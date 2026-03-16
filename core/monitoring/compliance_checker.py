"""
Regulatory compliance checker with config-driven compliance items.

Provides automated verification of compliance requirements through
configurable check types:

- ``file_exists``     -- Verify that an implementation file exists
- ``config_range``    -- Verify that a config value is within an acceptable range
- ``module_exists``   -- Verify that a Python module can be imported
- ``endpoint_alive``  -- Verify that a health-check endpoint responds

The compliance registry is loaded from a configuration dict (not hardcoded),
making it adaptable to any regulatory framework.
"""

from __future__ import annotations

import importlib
import json
import logging
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class ComplianceItem:
    """A single compliance check item."""

    item_id: str
    description: str
    regulation: str             # Regulatory reference (generic)
    check_type: str             # "file_exists" | "config_range" | "module_exists" | "endpoint_alive"
    check_target: str           # Path, module name, or URL depending on check_type
    check_params: Dict[str, Any]  # Extra parameters (e.g., valid ranges)
    last_verified: Optional[str] = None
    status: str = "unknown"     # "compliant" | "partial" | "non_compliant" | "unknown"


# ---------------------------------------------------------------------------
# Default compliance registry (example -- override via config)
# ---------------------------------------------------------------------------

DEFAULT_COMPLIANCE_ITEMS: Dict[str, Dict[str, Any]] = {
    "MON-01": {
        "description": "Audit logger with HMAC signing",
        "regulation": "regulatory_authority: audit trail",
        "check_type": "module_exists",
        "check_target": "core.monitoring.audit_logger",
        "check_params": {},
    },
    "MON-02": {
        "description": "Compliance audit store",
        "regulation": "regulatory_authority: audit persistence",
        "check_type": "module_exists",
        "check_target": "core.monitoring.compliance_store",
        "check_params": {},
    },
    "MON-03": {
        "description": "Fairness monitoring (DI/SPD/EOD)",
        "regulation": "regulatory_authority: bias monitoring",
        "check_type": "module_exists",
        "check_target": "core.monitoring.fairness_monitor",
        "check_params": {},
    },
    "MON-04": {
        "description": "Fairness threshold validity",
        "regulation": "regulatory_authority: bias thresholds",
        "check_type": "config_range",
        "check_target": "core.monitoring.fairness_monitor.DEFAULT_THRESHOLDS",
        "check_params": {
            "di_lower": [0.7, 0.9],
            "di_upper": [1.1, 1.3],
            "spd_max": [0.05, 0.15],
            "eod_max": [0.05, 0.15],
        },
    },
    "MON-05": {
        "description": "Herding detection",
        "regulation": "regulatory_authority: market stability",
        "check_type": "module_exists",
        "check_target": "core.monitoring.herding_detector",
        "check_params": {},
    },
    "MON-06": {
        "description": "Data drift detection",
        "regulation": "regulatory_authority: model reliability",
        "check_type": "module_exists",
        "check_target": "core.monitoring.drift_detector",
        "check_params": {},
    },
    "MON-07": {
        "description": "Incident management",
        "regulation": "regulatory_authority: risk management",
        "check_type": "module_exists",
        "check_target": "core.monitoring.incident_reporter",
        "check_params": {},
    },
    "MON-08": {
        "description": "Governance report generation",
        "regulation": "regulatory_authority: governance",
        "check_type": "module_exists",
        "check_target": "core.monitoring.governance_report",
        "check_params": {},
    },
    "MON-09": {
        "description": "Data lineage tracking",
        "regulation": "regulatory_authority: transparency",
        "check_type": "module_exists",
        "check_target": "core.monitoring.lineage_tracker",
        "check_params": {},
    },
}


# ---------------------------------------------------------------------------
# ComplianceChecker
# ---------------------------------------------------------------------------

class ComplianceChecker:
    """Config-driven regulatory compliance checker.

    Parameters
    ----------
    compliance_items : dict, optional
        Custom compliance item definitions.  If ``None``, the default
        registry (``DEFAULT_COMPLIANCE_ITEMS``) is used.
    code_root : str or Path, optional
        Root directory for ``file_exists`` checks.
    s3_bucket : str, optional
        S3 bucket for persisting compliance reports.
    s3_prefix : str
        Key prefix for reports.
    """

    # Supported check types mapped to handler methods
    _CHECK_HANDLERS: Dict[str, str] = {
        "file_exists": "_check_file_exists",
        "config_range": "_check_config_range",
        "module_exists": "_check_module_exists",
        "endpoint_alive": "_check_endpoint_alive",
    }

    def __init__(
        self,
        compliance_items: Optional[Dict[str, Dict[str, Any]]] = None,
        code_root: Optional[str | Path] = None,
        s3_bucket: Optional[str] = None,
        s3_prefix: str = "compliance_reports",
    ) -> None:
        raw_items = compliance_items or dict(DEFAULT_COMPLIANCE_ITEMS)
        self.registry: Dict[str, ComplianceItem] = {}
        for item_id, spec in raw_items.items():
            self.registry[item_id] = ComplianceItem(
                item_id=item_id,
                description=spec.get("description", ""),
                regulation=spec.get("regulation", ""),
                check_type=spec.get("check_type", "file_exists"),
                check_target=spec.get("check_target", ""),
                check_params=spec.get("check_params", {}),
            )

        self.code_root = Path(code_root) if code_root else Path.cwd()
        self.s3_bucket = s3_bucket or os.environ.get("COMPLIANCE_S3_BUCKET", "")
        self.s3_prefix = s3_prefix.strip("/")

        self._s3_client = None
        if self.s3_bucket:
            try:
                import boto3

                region = os.environ.get("AWS_DEFAULT_REGION", "ap-northeast-2")
                self._s3_client = boto3.client("s3", region_name=region)
            except Exception:
                pass

    # ------------------------------------------------------------------
    # Full compliance check
    # ------------------------------------------------------------------

    def run_full_check(self) -> Dict[str, Any]:
        """Execute all registered compliance checks.

        Returns
        -------
        dict
            Result with per-item status, counts, and overall compliance rate.
        """
        checked_at = datetime.now(timezone.utc).isoformat()
        items_result: Dict[str, Any] = {}
        counts = {"compliant": 0, "partial": 0, "non_compliant": 0, "unknown": 0}

        for item_id, item in self.registry.items():
            try:
                handler_name = self._CHECK_HANDLERS.get(item.check_type)
                if not handler_name:
                    status = "unknown"
                    detail = f"Unsupported check type: {item.check_type}"
                else:
                    handler = getattr(self, handler_name)
                    status, detail = handler(item)

                item.status = status
                item.last_verified = checked_at
                items_result[item_id] = {
                    "status": status,
                    "description": item.description,
                    "regulation": item.regulation,
                    "check_type": item.check_type,
                    "check_target": item.check_target,
                    "detail": detail,
                }
                counts[status] = counts.get(status, 0) + 1

                if status != "compliant":
                    logger.warning("Compliance gap: [%s] %s -> %s", item_id, item.description, status)

            except Exception as exc:
                logger.error("Check error for [%s]: %s", item_id, exc)
                items_result[item_id] = {"status": "unknown", "detail": str(exc)}
                counts["unknown"] += 1

        total = len(self.registry)
        compliant = counts["compliant"]
        rate = round(compliant / total, 4) if total > 0 else 0.0

        logger.info(
            "Compliance check complete: %d/%d compliant (%.1f%%)",
            compliant,
            total,
            rate * 100,
        )

        return {
            "checked_at": checked_at,
            "total_items": total,
            "compliant": compliant,
            "partial": counts["partial"],
            "non_compliant": counts["non_compliant"],
            "unknown": counts["unknown"],
            "overall_compliance_rate": rate,
            "items": items_result,
        }

    # ------------------------------------------------------------------
    # Gap analysis
    # ------------------------------------------------------------------

    def get_gap_analysis(self) -> List[Dict[str, Any]]:
        """Return a list of non-compliant or partial items.

        Returns
        -------
        list of dict
            Each entry includes ``item_id``, ``description``, ``status``,
            ``regulation``, and suggested ``action``.
        """
        gaps: List[Dict[str, Any]] = []
        for item_id, item in self.registry.items():
            if item.status in ("non_compliant", "partial", "unknown"):
                action = "Implement or fix" if item.status == "non_compliant" else "Review configuration"
                gaps.append({
                    "item_id": item_id,
                    "description": item.description,
                    "status": item.status,
                    "regulation": item.regulation,
                    "action": f"{action}: {item.check_target}",
                })
        return gaps

    # ------------------------------------------------------------------
    # Report persistence
    # ------------------------------------------------------------------

    def save_report(self, report: Dict[str, Any]) -> Optional[str]:
        """Save a compliance report to S3.

        Returns
        -------
        str or None
            S3 URI if successful, else ``None``.
        """
        if not self._s3_client or not self.s3_bucket:
            logger.warning("S3 not configured; compliance report not saved.")
            return None

        try:
            date_str = datetime.now(timezone.utc).strftime("%Y%m%d")
            s3_key = f"{self.s3_prefix}/{date_str}_compliance_report.json"
            body = json.dumps(report, ensure_ascii=False, indent=2, default=str).encode("utf-8")
            self._s3_client.put_object(
                Bucket=self.s3_bucket,
                Key=s3_key,
                Body=body,
                ContentType="application/json",
            )
            uri = f"s3://{self.s3_bucket}/{s3_key}"
            logger.info("Compliance report saved: %s", uri)
            return uri
        except Exception as exc:
            logger.warning("Failed to save compliance report: %s", exc)
            return None

    # ------------------------------------------------------------------
    # Check type handlers
    # ------------------------------------------------------------------

    def _check_file_exists(self, item: ComplianceItem) -> Tuple[str, str]:
        """Verify that a file exists and is non-empty."""
        path = self.code_root / item.check_target
        try:
            if path.exists() and path.stat().st_size > 0:
                return "compliant", f"File exists: {item.check_target}"
            return "non_compliant", f"File not found or empty: {item.check_target}"
        except Exception as exc:
            return "unknown", f"File check error: {exc}"

    @staticmethod
    def _check_module_exists(item: ComplianceItem) -> Tuple[str, str]:
        """Verify that a Python module can be imported."""
        try:
            importlib.import_module(item.check_target)
            return "compliant", f"Module importable: {item.check_target}"
        except ImportError as exc:
            return "non_compliant", f"Module import failed: {exc}"
        except Exception as exc:
            return "unknown", f"Module check error: {exc}"

    @staticmethod
    def _check_config_range(item: ComplianceItem) -> Tuple[str, str]:
        """Verify that configuration values fall within acceptable ranges.

        The ``check_target`` is a dotted path to a dict (e.g.
        ``"core.monitoring.fairness_monitor.DEFAULT_THRESHOLDS"``).
        ``check_params`` maps config keys to ``[min, max]`` ranges.
        """
        try:
            parts = item.check_target.rsplit(".", 1)
            if len(parts) != 2:
                return "unknown", f"Invalid config path: {item.check_target}"

            module_path, attr_name = parts
            mod = importlib.import_module(module_path)
            actual_values = getattr(mod, attr_name, {})

            if not isinstance(actual_values, dict):
                return "partial", f"Config target is not a dict: {item.check_target}"

            all_valid = True
            issues: List[str] = []
            for key, (min_val, max_val) in item.check_params.items():
                val = actual_values.get(key)
                if val is None:
                    all_valid = False
                    issues.append(f"{key}: missing")
                elif not (min_val <= val <= max_val):
                    all_valid = False
                    issues.append(f"{key}: {val} not in [{min_val}, {max_val}]")

            if all_valid:
                return "compliant", "All config values within acceptable ranges."
            return "partial", f"Config issues: {'; '.join(issues)}"

        except ImportError as exc:
            return "non_compliant", f"Cannot import config module: {exc}"
        except Exception as exc:
            return "unknown", f"Config range check error: {exc}"

    @staticmethod
    def _check_endpoint_alive(item: ComplianceItem) -> Tuple[str, str]:
        """Verify that a health-check endpoint responds with HTTP 2xx."""
        try:
            import urllib.request

            url = item.check_target
            timeout = item.check_params.get("timeout", 5)
            req = urllib.request.Request(url, method="GET")
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                if 200 <= resp.status < 300:
                    return "compliant", f"Endpoint alive: {url} (HTTP {resp.status})"
                return "partial", f"Endpoint returned HTTP {resp.status}: {url}"
        except Exception as exc:
            return "non_compliant", f"Endpoint unreachable: {item.check_target} ({exc})"


__all__ = ["ComplianceChecker", "ComplianceItem"]
