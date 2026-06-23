"""Regression tests for AIA-004 / AIA-005 fail-closed behaviour.

Guards the fix for the ``self._audit_store`` attribute typo (the store is
saved as ``self._store``) that made the critical bias / performance
monitoring checks fail *open* — raising ``AttributeError``, swallowing it,
and unconditionally returning ``True``. A critical compliance control must
never PASS on the config flag alone; absence of evidence or an audit-store
query failure must fail closed.
"""

from core.compliance.audit_store import InMemoryAuditStore
from core.compliance.regulatory_checker import RegulatoryComplianceChecker


def _checker(store, **overrides):
    cfg = {
        "bias_monitoring_enabled": True,
        "performance_monitoring_enabled": True,
    }
    cfg.update(overrides)
    return RegulatoryComplianceChecker(audit_store=store, config=cfg)


# --------------------------------------------------------------------------
# AIA-004 (bias / fairness monitoring) — critical
# --------------------------------------------------------------------------

def test_aia_004_fails_closed_when_no_evaluation_evidence():
    passed, msg, _ = _checker(InMemoryAuditStore())._check_aia_004_bias()
    assert passed is False
    assert "has not produced any evaluation records" in msg


def test_aia_004_passes_only_with_evaluation_evidence():
    store = InMemoryAuditStore()
    store.log_event("embedding", {"pk": "fairness_evaluation", "summary": "ok"})
    passed, _, details = _checker(store)._check_aia_004_bias()
    assert passed is True
    assert details["recent_fairness_evaluations"] >= 1


def test_aia_004_fails_closed_when_query_raises():
    class _RaisingStore(InMemoryAuditStore):
        def query_events(self, *args, **kwargs):  # noqa: ARG002
            raise RuntimeError("dynamo unavailable")

    passed, msg, details = _checker(_RaisingStore())._check_aia_004_bias()
    assert passed is False
    assert "Failing closed" in msg
    assert "audit_check_error" in details


def test_aia_004_fails_closed_when_store_missing():
    passed, msg, _ = _checker(None)._check_aia_004_bias()
    assert passed is False
    assert "no audit store" in msg


def test_aia_004_respects_disabled_flag():
    passed, msg, _ = _checker(
        InMemoryAuditStore(), bias_monitoring_enabled=False
    )._check_aia_004_bias()
    assert passed is False
    assert "not enabled" in msg


# --------------------------------------------------------------------------
# AIA-005 (model performance monitoring) — high
# --------------------------------------------------------------------------

def test_aia_005_fails_closed_when_no_evaluation_evidence():
    passed, msg, _ = _checker(InMemoryAuditStore())._check_aia_005_performance()
    assert passed is False
    assert "no recent model" in msg


def test_aia_005_passes_only_with_evaluation_evidence():
    store = InMemoryAuditStore()
    store.log_event("distillation", {"pk": "model_evaluation", "metric": "auc"})
    passed, _, details = _checker(store)._check_aia_005_performance()
    assert passed is True
    assert details["recent_evaluation_events"] >= 1


def test_aia_005_fails_closed_when_query_raises():
    class _RaisingStore(InMemoryAuditStore):
        def query_events(self, *args, **kwargs):  # noqa: ARG002
            raise RuntimeError("dynamo unavailable")

    passed, msg, details = _checker(_RaisingStore())._check_aia_005_performance()
    assert passed is False
    assert "Failing closed" in msg
    assert "audit_check_error" in details
