"""
core.compliance.rights
======================

User-facing regulatory rights managers, all built on the Sprint 0 foundation
(ComplianceStore + SLATracker).

Modules:
- opt_out           : AI decision opt-out + explanation request (M4)
- profiling         : Profiling access / correction / deletion (M5)
- explanation_sla   : 10-day explanation SLA tracker (M6)

These are Sprint 1 additions. The legacy modules
`core.compliance.ai_opt_out` and `core.compliance.profiling_rights`
remain in place for backward compatibility; new code should prefer the
managers exposed here.
"""

from core.compliance.rights.explanation_sla import (
    ExplanationSLATracker,
    build_explanation_sla_tracker,
)
from core.compliance.rights.opt_out import (
    OptOutConfig,
    OptOutDecision,
    OptOutManager,
)
from core.compliance.rights.profiling import (
    ProfilingAccessResult,
    ProfilingConfig,
    ProfilingWorkflow,
)

__all__ = [
    "OptOutConfig",
    "OptOutDecision",
    "OptOutManager",
    "ProfilingConfig",
    "ProfilingWorkflow",
    "ProfilingAccessResult",
    "ExplanationSLATracker",
    "build_explanation_sla_tracker",
]
