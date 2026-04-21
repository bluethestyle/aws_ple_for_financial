"""
Compliance Module
=================

Provides:
- Sprint 0 foundation: ComplianceRequest, ComplianceEvent, ComplianceStore, SLATracker
- DynamoDB-backed compliance audit store (7 audit tables)
- In-memory fallback for local testing
- Automated regulatory compliance checker (금소법, 개보법, AI기본법)
- Channel-level marketing consent management
- AI decision opt-out tracking (AI기본법 제31조)
- Data subject profiling rights (개보법 + GDPR)
"""

from core.compliance.ai_opt_out import AIDecisionOptOut, OptOutRecord
from core.compliance.audit_store import (
    ComplianceAuditStore,
    InMemoryAuditStore,
    get_audit_store,
)
from core.compliance.consent_manager import (
    ConsentConfig,
    ConsentManager,
    ConsentRecord,
)
from core.compliance.profiling_rights import (
    ProfilingRightsManager,
    RightsRequest,
)
from core.compliance.regulatory_checker import (
    CheckItem,
    CheckResult,
    RegulatoryComplianceChecker,
)
from core.compliance.ai_risk_classifier import (
    AI_RISK_DIMENSIONS,
    AIRiskAssessment,
    AIRiskClassifier,
    AIRiskConfig,
    build_ai_risk_classifier,
)
from core.compliance.compliance_registry import (
    DEFAULT_CATALOG as COMPLIANCE_CATALOG,
    ComplianceItem,
    ComplianceItemResult,
    ComplianceRegistry,
    load_registry_from_pipeline_yaml,
)
from core.compliance.fria_assessment import (
    FRIA_DIMENSIONS,
    FRIA_RISK_THRESHOLDS,
    FRIAConfig,
    FRIAResult,
    KoreanFRIAAssessor,
    build_fria_assessor,
)
from core.compliance.rights import (
    ExplanationSLATracker,
    OptOutConfig,
    OptOutDecision,
    OptOutManager,
    ProfilingAccessResult,
    ProfilingConfig,
    ProfilingWorkflow,
    build_explanation_sla_tracker,
)
from core.compliance.audit_sql import (
    AuditSQLConfig,
    ComplianceSQLHelper,
    build_compliance_sql_helper,
)
from core.compliance.sagemaker_compliance_tracker import (
    InMemoryTrackerBackend,
    SageMakerComplianceTracker,
    SageMakerTrackerBackend,
    TrackedArtifact,
    TrackingConfig,
    build_sagemaker_compliance_tracker,
)
from core.compliance.sla_tracker import (
    SLADefinition,
    SLAReport,
    SLATracker,
)
from core.compliance.store import (
    ComplianceStore,
    DynamoDBComplianceStore,
    InMemoryComplianceStore,
    S3ParquetComplianceStore,
    build_compliance_store,
)
from core.compliance.types import (
    ComplianceEvent,
    ComplianceRequest,
    EventType,
    RequestStatus,
    RequestType,
    new_event_id,
    new_request_id,
    utcnow,
)

__all__ = [
    # Sprint 0 foundation — types
    "ComplianceRequest",
    "ComplianceEvent",
    "RequestStatus",
    "RequestType",
    "EventType",
    "new_request_id",
    "new_event_id",
    "utcnow",
    # Sprint 0 foundation — store
    "ComplianceStore",
    "InMemoryComplianceStore",
    "DynamoDBComplianceStore",
    "S3ParquetComplianceStore",
    "build_compliance_store",
    # Sprint 0 foundation — SLA
    "SLADefinition",
    "SLATracker",
    "SLAReport",
    # Existing legacy audit store
    "ComplianceAuditStore",
    "InMemoryAuditStore",
    "get_audit_store",
    # Existing regulatory checker
    "CheckItem",
    "CheckResult",
    "RegulatoryComplianceChecker",
    # Existing consent management (extended in Sprint 1 with ConsentConfig)
    "ConsentConfig",
    "ConsentManager",
    "ConsentRecord",
    # Existing AI opt-out (legacy; Sprint 1 rights/OptOutManager preferred)
    "AIDecisionOptOut",
    "OptOutRecord",
    # Existing profiling rights (legacy; rights/ProfilingWorkflow preferred)
    "ProfilingRightsManager",
    "RightsRequest",
    # Sprint 1 rights subpackage
    "OptOutConfig",
    "OptOutDecision",
    "OptOutManager",
    "ProfilingConfig",
    "ProfilingWorkflow",
    "ProfilingAccessResult",
    "ExplanationSLATracker",
    "build_explanation_sla_tracker",
    # Sprint 2: M7 Korean FRIA
    "FRIA_DIMENSIONS",
    "FRIA_RISK_THRESHOLDS",
    "FRIAConfig",
    "FRIAResult",
    "KoreanFRIAAssessor",
    "build_fria_assessor",
    # Sprint 2: M8 Compliance Registry
    "COMPLIANCE_CATALOG",
    "ComplianceItem",
    "ComplianceItemResult",
    "ComplianceRegistry",
    "load_registry_from_pipeline_yaml",
    # Sprint 2: M9 AI Risk Classifier
    "AI_RISK_DIMENSIONS",
    "AIRiskAssessment",
    "AIRiskClassifier",
    "AIRiskConfig",
    "build_ai_risk_classifier",
    # Sprint 2 S5: SageMaker compliance tracker
    "TrackingConfig",
    "TrackedArtifact",
    "InMemoryTrackerBackend",
    "SageMakerTrackerBackend",
    "SageMakerComplianceTracker",
    "build_sagemaker_compliance_tracker",
    # Sprint 2 S6: DuckDB-over-Parquet compliance SQL helper
    "AuditSQLConfig",
    "ComplianceSQLHelper",
    "build_compliance_sql_helper",
]
