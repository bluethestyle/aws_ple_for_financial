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
- Financial Risk Impact Assessment (FRIA)
- Privacy Impact Assessment (PIA)
- EU AI Act compliance mapping
- XAI quality evaluation
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
from core.monitoring.fria_evaluator import FRIAEvaluator, FRIAReport, TaskRiskProfile
from core.monitoring.pia_evaluator import PIAEvaluator, PIAReport, PIIInventoryItem, DomainRiskAssessment
from core.monitoring.eu_ai_act_mapper import EUAIActMapper, EUAIActComplianceReport, ArticleComplianceItem
from core.monitoring.xai_quality_evaluator import (
    XAIQualityEvaluator,
    XAIQualityReport,
    ExplanationQualityMetrics,
    ConsistencyResult,
)

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
    "FRIAEvaluator",
    "FRIAReport",
    "TaskRiskProfile",
    "PIAEvaluator",
    "PIAReport",
    "PIIInventoryItem",
    "DomainRiskAssessment",
    "EUAIActMapper",
    "EUAIActComplianceReport",
    "ArticleComplianceItem",
    "XAIQualityEvaluator",
    "XAIQualityReport",
    "ExplanationQualityMetrics",
    "ConsistencyResult",
]
