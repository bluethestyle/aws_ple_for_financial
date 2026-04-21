"""
Sprint 1 rights tests: OptOutManager (M4), ProfilingWorkflow (M5),
ExplanationSLATracker (M6).

Run: pytest tests/test_compliance_rights.py -v
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from core.compliance.rights.explanation_sla import (
    ExplanationSLATracker,
    build_explanation_sla_tracker,
)
from core.compliance.rights.opt_out import (
    OptOutConfig,
    OptOutManager,
    VALID_FALLBACKS,
)
from core.compliance.rights.profiling import (
    ProfilingConfig,
    ProfilingWorkflow,
    VALID_DELETION_SCOPES,
)
from core.compliance.store import InMemoryComplianceStore
from core.compliance.types import (
    EventType,
    RequestStatus,
    RequestType,
)


def _fixed(ts: str) -> datetime:
    dt = datetime.fromisoformat(ts)
    return dt if dt.tzinfo is not None else dt.replace(tzinfo=timezone.utc)


@pytest.fixture
def store():
    return InMemoryComplianceStore()


# ---------------------------------------------------------------------------
# M4 - OptOutManager
# ---------------------------------------------------------------------------

class TestOptOutConfig:
    def test_defaults(self):
        cfg = OptOutConfig()
        assert cfg.default_fallback == "rule_based"
        assert cfg.explanation_sla_days == 10
        assert cfg.opt_out_response_days == 30

    def test_rejects_unknown_fallback(self):
        with pytest.raises(ValueError):
            OptOutConfig(default_fallback="teleport")

    def test_from_dict(self):
        cfg = OptOutConfig.from_dict({
            "default_fallback": "human_review",
            "explanation_sla_days": 14,
            "opt_out_response_days": 21,
        })
        assert cfg.default_fallback == "human_review"
        assert cfg.explanation_sla_days == 14
        assert cfg.opt_out_response_days == 21

    def test_from_empty_dict(self):
        assert OptOutConfig.from_dict(None).default_fallback == "rule_based"


class TestOptOut:
    def test_opt_out_records_request_and_event(self, store):
        mgr = OptOutManager(store=store)
        req = mgr.opt_out(
            user_id="u1", reason="user request",
            fallback_type="human_review",
        )
        assert req.status == RequestStatus.PROCESSED
        assert req.metadata["fallback_type"] == "human_review"
        events = store.query_events(user_id="u1")
        assert any(
            e.event_type == EventType.REQUEST_PROCESSED and
            e.payload.get("fallback_type") == "human_review"
            for e in events
        )

    def test_is_opted_out_true_after_opt_out(self, store):
        mgr = OptOutManager(store=store)
        mgr.opt_out(user_id="u1", reason="test")
        assert mgr.is_opted_out("u1") is True

    def test_is_opted_out_false_by_default(self, store):
        mgr = OptOutManager(store=store)
        assert mgr.is_opted_out("fresh_user") is False

    def test_opt_in_revokes_opt_out(self, store):
        mgr = OptOutManager(store=store)
        mgr.opt_out(user_id="u1", reason="test")
        assert mgr.is_opted_out("u1") is True
        mgr.opt_in(user_id="u1", reason="changed mind")
        assert mgr.is_opted_out("u1") is False

    def test_get_decision_returns_fallback(self, store):
        mgr = OptOutManager(store=store)
        mgr.opt_out(user_id="u1", reason="test", fallback_type="disable")
        decision = mgr.get_decision("u1")
        assert decision.is_opted_out is True
        assert decision.fallback_type == "disable"
        assert decision.reason == "test"

    def test_opt_out_rejects_unknown_fallback(self, store):
        mgr = OptOutManager(store=store)
        with pytest.raises(ValueError):
            mgr.opt_out(user_id="u1", reason="x", fallback_type="teleport")


class TestExplanationRequest:
    def test_request_explanation_creates_pending(self, store):
        mgr = OptOutManager(store=store)
        req = mgr.request_explanation(
            user_id="u1", recommendation_id="rec_1", reason="why this?",
        )
        assert req.status == RequestStatus.PENDING
        assert req.request_type == RequestType.EXPLANATION
        assert req.metadata["recommendation_id"] == "rec_1"
        # 10-day deadline by default
        assert (req.sla_deadline - req.submitted_at).days == 10

    def test_mark_explanation_provided_on_time(self, store):
        mgr = OptOutManager(store=store)
        req = mgr.request_explanation(user_id="u1", recommendation_id="rec_1")
        # 5 days later - within SLA
        mgr.mark_explanation_provided(
            req.request_id,
            "because of feature X",
            provided_at=req.submitted_at + timedelta(days=5),
        )
        updated = store.get_request(req.request_id)
        assert updated.status == RequestStatus.PROCESSED
        breaches = store.query_events(event_type=EventType.SLA_BREACH)
        assert breaches == []

    def test_mark_explanation_late_emits_breach(self, store):
        mgr = OptOutManager(store=store)
        req = mgr.request_explanation(user_id="u1", recommendation_id="rec_1")
        mgr.mark_explanation_provided(
            req.request_id,
            "because...",
            provided_at=req.submitted_at + timedelta(days=12),
        )
        breaches = store.query_events(event_type=EventType.SLA_BREACH)
        assert len(breaches) == 1
        assert breaches[0].payload["sla_name"] == "explanation"

    def test_mark_rejects_unknown_id(self, store):
        mgr = OptOutManager(store=store)
        with pytest.raises(KeyError):
            mgr.mark_explanation_provided("req_bogus", "x")

    def test_mark_rejects_wrong_type(self, store):
        mgr = OptOutManager(store=store)
        opt_out_req = mgr.opt_out(user_id="u1", reason="x")
        with pytest.raises(ValueError):
            mgr.mark_explanation_provided(opt_out_req.request_id, "x")

    def test_list_pending_explanations(self, store):
        mgr = OptOutManager(store=store)
        mgr.request_explanation(user_id="u1", recommendation_id="r1")
        mgr.request_explanation(user_id="u2", recommendation_id="r2")
        mgr.opt_out(user_id="u3", reason="z")  # not an explanation
        all_pending = mgr.list_pending_explanations()
        assert len(all_pending) == 2
        u1_pending = mgr.list_pending_explanations(user_id="u1")
        assert len(u1_pending) == 1


# ---------------------------------------------------------------------------
# M5 - ProfilingWorkflow
# ---------------------------------------------------------------------------

class TestProfilingConfig:
    def test_defaults(self):
        cfg = ProfilingConfig()
        assert cfg.sla_days == 30
        assert cfg.warning_days_before == 5

    def test_from_dict(self):
        cfg = ProfilingConfig.from_dict({
            "sla_days": 14, "warning_days_before": 3,
        })
        assert cfg.sla_days == 14
        assert cfg.warning_days_before == 3


class TestProfilingAccess:
    def test_request_access_creates_pending(self, store):
        wf = ProfilingWorkflow(store=store)
        req = wf.request_access(user_id="u1", reason="check my profile")
        assert req.status == RequestStatus.PENDING
        assert req.request_type == RequestType.PROFILING_ACCESS

    def test_fulfill_access_uses_provider(self, store):
        captured = {}

        def provider(uid: str):
            captured["uid"] = uid
            return {"income_tier": "high", "age_bracket": "30s"}

        wf = ProfilingWorkflow(store=store, profile_provider=provider)
        req = wf.request_access(user_id="u1")
        result = wf.fulfill_access(
            req.request_id, disclosed_features=["income_tier", "age_bracket"],
        )
        assert captured["uid"] == "u1"
        assert result.snapshot["income_tier"] == "high"
        assert "income_tier" in result.disclosed_features

        updated = store.get_request(req.request_id)
        assert updated.status == RequestStatus.PROCESSED

    def test_fulfill_access_without_provider_returns_empty(self, store):
        wf = ProfilingWorkflow(store=store)
        req = wf.request_access(user_id="u1")
        result = wf.fulfill_access(req.request_id)
        assert result.snapshot == {}

    def test_fulfill_access_with_failing_provider(self, store):
        def bad_provider(uid: str):
            raise RuntimeError("DB down")

        wf = ProfilingWorkflow(store=store, profile_provider=bad_provider)
        req = wf.request_access(user_id="u1")
        result = wf.fulfill_access(req.request_id)
        assert result.snapshot == {}
        updated = store.get_request(req.request_id)
        assert updated.status == RequestStatus.PROCESSED


class TestProfilingCorrection:
    def test_correction_request_and_fulfill(self, store):
        wf = ProfilingWorkflow(store=store)
        req = wf.request_correction(
            user_id="u1", field_name="income", new_value=50000,
        )
        assert req.request_type == RequestType.PROFILING_CORRECTION
        wf.fulfill_correction(req.request_id, applied=True, notes="verified")
        updated = store.get_request(req.request_id)
        assert updated.status == RequestStatus.PROCESSED


class TestProfilingDeletion:
    def test_deletion_request_and_fulfill(self, store):
        wf = ProfilingWorkflow(store=store)
        req = wf.request_deletion(user_id="u1", scope="profiling_only")
        assert req.request_type == RequestType.PROFILING_DELETION
        wf.fulfill_deletion(req.request_id, deleted=True)
        updated = store.get_request(req.request_id)
        assert updated.status == RequestStatus.PROCESSED

    def test_rejects_unknown_scope(self, store):
        wf = ProfilingWorkflow(store=store)
        with pytest.raises(ValueError):
            wf.request_deletion(user_id="u1", scope="galactic")


class TestProfilingSLABreach:
    def test_late_fulfillment_emits_breach(self, store):
        import core.compliance.rights.profiling as profiling_module
        wf = ProfilingWorkflow(store=store, config=ProfilingConfig(sla_days=5))
        req = wf.request_access(user_id="u1")
        # Simulate a late fulfillment by rewinding the request's deadline.
        # Using the deadline in the store directly keeps the test fast.
        req_stored = store.get_request(req.request_id)
        req_stored.sla_deadline = req_stored.submitted_at - timedelta(days=1)
        store.put_request(req_stored)

        wf.fulfill_access(req.request_id)
        breaches = store.query_events(event_type=EventType.SLA_BREACH)
        assert len(breaches) == 1
        assert breaches[0].payload["sla_name"] == "profiling"


class TestProfilingListing:
    def test_list_pending_for_user(self, store):
        wf = ProfilingWorkflow(store=store)
        wf.request_access(user_id="u1")
        wf.request_correction(user_id="u1", field_name="age", new_value=30)
        wf.request_access(user_id="u2")

        u1_pending = wf.list_pending_for_user("u1")
        assert len(u1_pending) == 2

        all_pending = wf.list_all_pending()
        assert len(all_pending) == 3


# ---------------------------------------------------------------------------
# M6 - ExplanationSLATracker
# ---------------------------------------------------------------------------

class TestExplanationSLATracker:
    def test_inherits_sla_10_days_by_default(self, store):
        tracker = ExplanationSLATracker(store=store)
        req = tracker.open_request(
            user_id="u1", request_type=RequestType.EXPLANATION,
        )
        assert (req.sla_deadline - req.submitted_at).days == 10

    def test_rejects_non_explanation_type(self, store):
        tracker = ExplanationSLATracker(store=store)
        with pytest.raises(ValueError, match="not owned"):
            tracker.open_request(
                user_id="u1", request_type=RequestType.OPT_OUT,
            )

    def test_generate_monthly_report(self, store):
        tracker = ExplanationSLATracker(store=store)
        # seed 3 requests in April 2026
        apr1 = _fixed("2026-04-01T00:00:00+00:00")
        on_time_request = tracker.open_request(
            user_id="u1", request_type=RequestType.EXPLANATION,
            submitted_at=apr1 + timedelta(days=1),
        )
        tracker.mark_processed(
            on_time_request.request_id,
            processed_at=apr1 + timedelta(days=3),
        )
        late_request = tracker.open_request(
            user_id="u2", request_type=RequestType.EXPLANATION,
            submitted_at=apr1 + timedelta(days=2),
        )
        tracker.mark_processed(
            late_request.request_id,
            processed_at=apr1 + timedelta(days=15),
        )
        tracker.open_request(
            user_id="u3", request_type=RequestType.EXPLANATION,
            submitted_at=apr1 + timedelta(days=5),
        )  # still pending

        report = tracker.generate_monthly_report(2026, 4)
        assert report.total_requests == 3
        assert report.processed_on_time == 1
        assert report.processed_late == 1
        assert report.compliance_rate == pytest.approx(0.5)

    def test_summarize_current_state_counts(self, store):
        tracker = ExplanationSLATracker(store=store)
        now = _fixed("2026-04-21T00:00:00+00:00")
        tracker.open_request(
            user_id="u1", request_type=RequestType.EXPLANATION,
            submitted_at=now - timedelta(days=11),  # breached
        )
        tracker.open_request(
            user_id="u2", request_type=RequestType.EXPLANATION,
            submitted_at=now - timedelta(days=9),   # approaching (< 2 days left)
        )
        tracker.open_request(
            user_id="u3", request_type=RequestType.EXPLANATION,
            submitted_at=now - timedelta(days=1),   # healthy
        )
        summary = tracker.summarize_current_state(now=now)
        assert summary["pending"] == 3
        assert summary["breached"] == 1
        assert summary["approaching_deadline"] == 1


class TestExplanationSLAFactory:
    def test_build_with_default_config(self, store):
        tracker = build_explanation_sla_tracker(store)
        assert isinstance(tracker, ExplanationSLATracker)
        req = tracker.open_request(
            user_id="u1", request_type=RequestType.EXPLANATION,
        )
        assert (req.sla_deadline - req.submitted_at).days == 10

    def test_build_with_pipeline_yaml_block(self, store):
        import yaml
        from pathlib import Path

        cfg = yaml.safe_load(Path("configs/pipeline.yaml").read_text(
            encoding="utf-8"
        ))
        tracker = build_explanation_sla_tracker(
            store, config=cfg["compliance"]["sla"],
        )
        req = tracker.open_request(
            user_id="u1", request_type=RequestType.EXPLANATION,
        )
        assert (req.sla_deadline - req.submitted_at).days == 10


# ---------------------------------------------------------------------------
# Cross-cutting: predict.py-style integration
# ---------------------------------------------------------------------------

class TestOptOutPredictIntegration:
    """Simulate how predict.py would use OptOutManager to gate a call."""

    def test_flow_blocks_opted_out_user(self, store):
        mgr = OptOutManager(store=store)
        mgr.opt_out(
            user_id="u1", reason="user request", fallback_type="human_review",
        )
        decision = mgr.get_decision("u1")
        assert decision.is_opted_out
        assert decision.fallback_type == "human_review"

    def test_flow_allows_normal_user(self, store):
        mgr = OptOutManager(store=store)
        decision = mgr.get_decision("u2")
        assert not decision.is_opted_out

    def test_flow_re_allows_after_opt_in(self, store):
        mgr = OptOutManager(store=store)
        mgr.opt_out(user_id="u1", reason="test")
        mgr.opt_in(user_id="u1")
        decision = mgr.get_decision("u1")
        assert not decision.is_opted_out


class TestValidConstants:
    def test_valid_fallbacks_stable(self):
        assert VALID_FALLBACKS == ("rule_based", "human_review", "disable")

    def test_valid_deletion_scopes_stable(self):
        assert VALID_DELETION_SCOPES == ("full", "profiling_only")
