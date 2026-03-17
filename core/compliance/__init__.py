"""
Compliance Module
=================

Provides:
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
from core.compliance.consent_manager import ConsentManager, ConsentRecord
from core.compliance.profiling_rights import (
    ProfilingRightsManager,
    RightsRequest,
)
from core.compliance.regulatory_checker import (
    CheckItem,
    CheckResult,
    RegulatoryComplianceChecker,
)

__all__ = [
    # Audit store
    "ComplianceAuditStore",
    "InMemoryAuditStore",
    "get_audit_store",
    # Regulatory checker
    "CheckItem",
    "CheckResult",
    "RegulatoryComplianceChecker",
    # Consent management
    "ConsentManager",
    "ConsentRecord",
    # AI opt-out
    "AIDecisionOptOut",
    "OptOutRecord",
    # Profiling rights
    "ProfilingRightsManager",
    "RightsRequest",
]
