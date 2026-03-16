"""
Monitoring, audit, governance, and compliance infrastructure.

Provides:
- Immutable audit logging (HMAC-SHA256 + hash chain + S3 Object Lock)
- Compliance audit store (DynamoDB backend)
- Fairness monitoring (DI / SPD / EOD)
- Herding / concentration detection (HHI / Gini / Entropy)
- Data drift detection (PSI)
- Incident management with auto-escalation (SNS)
- Governance report generation
- Data lineage tracking
- Regulatory compliance checking
"""

from core.monitoring.audit_logger import AuditLogger
from core.monitoring.compliance_store import ComplianceAuditStore
from core.monitoring.fairness_monitor import FairnessMonitor, FairnessMetrics
from core.monitoring.herding_detector import HerdingDetector, HerdingMetrics
from core.monitoring.drift_detector import DriftDetector, PSICalculator, ConsecutiveDriftTracker
from core.monitoring.incident_reporter import (
    IncidentReporter,
    IncidentRecord,
    IncidentSeverity,
)
from core.monitoring.governance_report import GovernanceReportGenerator, GovernanceReport
from core.monitoring.lineage_tracker import DataLineageTracker, LineageRecord
from core.monitoring.compliance_checker import ComplianceChecker, ComplianceItem

__all__ = [
    "AuditLogger",
    "ComplianceAuditStore",
    "FairnessMonitor",
    "FairnessMetrics",
    "HerdingDetector",
    "HerdingMetrics",
    "DriftDetector",
    "PSICalculator",
    "ConsecutiveDriftTracker",
    "IncidentReporter",
    "IncidentRecord",
    "IncidentSeverity",
    "GovernanceReportGenerator",
    "GovernanceReport",
    "DataLineageTracker",
    "LineageRecord",
    "ComplianceChecker",
    "ComplianceItem",
]
