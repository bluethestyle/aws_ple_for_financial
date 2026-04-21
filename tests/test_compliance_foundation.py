"""
Sprint 0 foundation tests: ComplianceRequest / ComplianceEvent dataclasses,
InMemoryComplianceStore round-trip, SLATracker deadline / breach logic.

Run: pytest tests/test_compliance_foundation.py -v
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from core.compliance.store import (
    InMemoryComplianceStore,
    build_compliance_store,
)
from core.compliance.sla_tracker import SLADefinition, SLATracker
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


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _fixed(ts: str) -> datetime:
    return datetime.fromisoformat(ts).replace(tzinfo=timezone.utc) \
        if "+" not in ts and "Z" not in ts else datetime.fromisoformat(ts)


@pytest.fixture
def store() -> InMemoryComplianceStore:
    return InMemoryComplianceStore()


@pytest.fixture
def explanation_sla(store: InMemoryComplianceStore) -> SLATracker:
    definition = SLADefinition(
        name="explanation",
        request_types=(RequestType.EXPLANATION,),
        sla_days=10,
        warning_days_before=2,
    )
    return SLATracker(store=store, definition=definition)


# ---------------------------------------------------------------------------
# ComplianceRequest dataclass
# ---------------------------------------------------------------------------

class TestComplianceRequest:
    def test_basic_construction(self):
        now = utcnow()
        req = ComplianceRequest(
            request_id=new_request_id(),
            user_id="user_42",
            request_type=RequestType.EXPLANATION,
            submitted_at=now,
            sla_deadline=now + timedelta(days=10),
        )
        assert req.status == RequestStatus.PENDING
        assert req.metadata == {}
        assert req.processed_at is None
        assert not req.is_overdue()

    def test_rejects_unknown_request_type(self):
        now = utcnow()
        with pytest.raises(ValueError, match="Unknown request_type"):
            ComplianceRequest(
                request_id="x",
                user_id="u",
                request_type="not_a_real_type",
                submitted_at=now,
                sla_deadline=now + timedelta(days=1),
            )

    def test_rejects_deadline_before_submission(self):
        now = utcnow()
        with pytest.raises(ValueError, match="sla_deadline"):
            ComplianceRequest(
                request_id="x",
                user_id="u",
                request_type=RequestType.EXPLANATION,
                submitted_at=now,
                sla_deadline=now - timedelta(hours=1),
            )

    def test_is_overdue_switches_on_deadline(self):
        now = _fixed("2026-04-21T00:00:00+00:00")
        req = ComplianceRequest(
            request_id="x",
            user_id="u",
            request_type=RequestType.OPT_OUT,
            submitted_at=now,
            sla_deadline=now + timedelta(days=30),
        )
        assert not req.is_overdue(now=now + timedelta(days=15))
        assert req.is_overdue(now=now + timedelta(days=31))

    def test_is_overdue_ignores_processed(self):
        now = utcnow()
        req = ComplianceRequest(
            request_id="x",
            user_id="u",
            request_type=RequestType.EXPLANATION,
            submitted_at=now,
            sla_deadline=now + timedelta(days=1),
            status=RequestStatus.PROCESSED,
            processed_at=now,
        )
        # Deadline is in the past relative to far-future but status=PROCESSED,
        # so is_overdue should be False.
        assert not req.is_overdue(now=now + timedelta(days=365))

    def test_json_roundtrip(self):
        now = utcnow()
        original = ComplianceRequest(
            request_id=new_request_id(),
            user_id="user_7",
            request_type=RequestType.CONSENT_GRANT,
            submitted_at=now,
            sla_deadline=now + timedelta(days=10),
            metadata={"channel": "SMS", "legal_basis": "개보법 §22"},
        )
        raw = original.to_json()
        restored = ComplianceRequest.from_json(raw)
        assert restored.request_id == original.request_id
        assert restored.user_id == original.user_id
        assert restored.metadata == original.metadata
        assert restored.submitted_at == original.submitted_at


class TestComplianceEvent:
    def test_roundtrip(self):
        evt = ComplianceEvent(
            event_id=new_event_id(),
            user_id="user_9",
            event_type=EventType.REQUEST_CREATED,
            timestamp=utcnow(),
            payload={"source": "unit_test"},
            request_id="req_abc",
        )
        restored = ComplianceEvent.from_json(evt.to_json())
        assert restored.event_id == evt.event_id
        assert restored.payload == evt.payload
        assert restored.request_id == evt.request_id


# ---------------------------------------------------------------------------
# InMemoryComplianceStore
# ---------------------------------------------------------------------------

class TestInMemoryComplianceStore:
    def test_put_get_request(self, store):
        now = utcnow()
        req = ComplianceRequest(
            request_id=new_request_id(),
            user_id="u1",
            request_type=RequestType.OPT_OUT,
            submitted_at=now,
            sla_deadline=now + timedelta(days=30),
        )
        store.put_request(req)
        got = store.get_request(req.request_id)
        assert got is not None
        assert got.request_id == req.request_id

    def test_get_unknown_returns_none(self, store):
        assert store.get_request("req_not_there") is None

    def test_list_requests_filters(self, store):
        now = utcnow()
        for uid in ("u1", "u2", "u1"):
            store.put_request(
                ComplianceRequest(
                    request_id=new_request_id(),
                    user_id=uid,
                    request_type=RequestType.EXPLANATION,
                    submitted_at=now,
                    sla_deadline=now + timedelta(days=10),
                )
            )
        store.put_request(
            ComplianceRequest(
                request_id=new_request_id(),
                user_id="u1",
                request_type=RequestType.OPT_OUT,
                submitted_at=now,
                sla_deadline=now + timedelta(days=30),
            )
        )
        u1_all = store.list_requests(user_id="u1")
        assert len(u1_all) == 3

        u1_explanation = store.list_requests(
            user_id="u1", request_type=RequestType.EXPLANATION
        )
        assert len(u1_explanation) == 2

        pending = store.list_pending()
        assert len(pending) == 4  # default status is PENDING

    def test_update_status_transitions(self, store):
        now = utcnow()
        req = ComplianceRequest(
            request_id=new_request_id(),
            user_id="u1",
            request_type=RequestType.EXPLANATION,
            submitted_at=now,
            sla_deadline=now + timedelta(days=10),
        )
        store.put_request(req)
        store.update_request_status(req.request_id, RequestStatus.PROCESSED)
        updated = store.get_request(req.request_id)
        assert updated.status == RequestStatus.PROCESSED
        assert updated.processed_at is not None

    def test_update_rejects_invalid_status(self, store):
        now = utcnow()
        req = ComplianceRequest(
            request_id=new_request_id(),
            user_id="u1",
            request_type=RequestType.EXPLANATION,
            submitted_at=now,
            sla_deadline=now + timedelta(days=10),
        )
        store.put_request(req)
        with pytest.raises(ValueError):
            store.update_request_status(req.request_id, "not_a_status")

    def test_events_query_by_user_and_since(self, store):
        base = _fixed("2026-04-01T00:00:00+00:00")
        for i in range(5):
            store.put_event(
                ComplianceEvent(
                    event_id=new_event_id(),
                    user_id="u1",
                    event_type=EventType.REQUEST_CREATED,
                    timestamp=base + timedelta(days=i),
                )
            )
        store.put_event(
            ComplianceEvent(
                event_id=new_event_id(),
                user_id="u2",
                event_type=EventType.SLA_BREACH,
                timestamp=base + timedelta(days=2),
            )
        )
        all_u1 = store.query_events(user_id="u1")
        assert len(all_u1) == 5

        recent_u1 = store.query_events(
            user_id="u1", since=base + timedelta(days=3)
        )
        assert len(recent_u1) == 2

        breaches = store.query_events(event_type=EventType.SLA_BREACH)
        assert len(breaches) == 1
        assert breaches[0].user_id == "u2"


# ---------------------------------------------------------------------------
# SLATracker
# ---------------------------------------------------------------------------

class TestSLATracker:
    def test_open_request_sets_deadline(self, explanation_sla, store):
        submitted = _fixed("2026-04-21T00:00:00+00:00")
        req = explanation_sla.open_request(
            user_id="u1",
            request_type=RequestType.EXPLANATION,
            submitted_at=submitted,
            metadata={"recommendation_id": "rec_123"},
        )
        assert req.sla_deadline == submitted + timedelta(days=10)
        assert req.status == RequestStatus.PENDING
        # A REQUEST_CREATED event should have been emitted.
        events = store.query_events(user_id="u1")
        assert any(e.event_type == EventType.REQUEST_CREATED for e in events)

    def test_rejects_wrong_request_type(self, explanation_sla):
        with pytest.raises(ValueError, match="not owned"):
            explanation_sla.open_request(
                user_id="u1",
                request_type=RequestType.OPT_OUT,
            )

    def test_mark_processed_on_time_no_breach(self, explanation_sla, store):
        submitted = _fixed("2026-04-21T00:00:00+00:00")
        req = explanation_sla.open_request(
            user_id="u1",
            request_type=RequestType.EXPLANATION,
            submitted_at=submitted,
        )
        explanation_sla.mark_processed(
            req.request_id, processed_at=submitted + timedelta(days=5)
        )
        updated = store.get_request(req.request_id)
        assert updated.status == RequestStatus.PROCESSED
        breaches = store.query_events(event_type=EventType.SLA_BREACH)
        assert breaches == []

    def test_mark_processed_late_emits_breach(self, explanation_sla, store):
        submitted = _fixed("2026-04-21T00:00:00+00:00")
        req = explanation_sla.open_request(
            user_id="u1",
            request_type=RequestType.EXPLANATION,
            submitted_at=submitted,
        )
        explanation_sla.mark_processed(
            req.request_id, processed_at=submitted + timedelta(days=12)
        )
        breaches = store.query_events(event_type=EventType.SLA_BREACH)
        assert len(breaches) == 1
        assert breaches[0].payload["breach_type"] == "late_processing"

    def test_list_approaching_within_warning_window(self, explanation_sla):
        submitted = _fixed("2026-04-21T00:00:00+00:00")
        explanation_sla.open_request(
            user_id="u1",
            request_type=RequestType.EXPLANATION,
            submitted_at=submitted,
        )
        # 9 days in: 1 day before deadline → within 2-day warning
        approaching = explanation_sla.list_approaching(
            now=submitted + timedelta(days=9)
        )
        assert len(approaching) == 1
        # 5 days in: still 5 days to deadline → not yet warning
        assert explanation_sla.list_approaching(
            now=submitted + timedelta(days=5)
        ) == []

    def test_sweep_breaches(self, explanation_sla, store):
        submitted = _fixed("2026-04-21T00:00:00+00:00")
        explanation_sla.open_request(
            user_id="u1",
            request_type=RequestType.EXPLANATION,
            submitted_at=submitted,
        )
        now = submitted + timedelta(days=11)
        breached = explanation_sla.sweep_breaches(now=now)
        assert len(breached) == 1
        breach_events = store.query_events(event_type=EventType.SLA_BREACH)
        assert len(breach_events) == 1
        assert breach_events[0].payload["breach_type"] == "pending_overdue"

    def test_report_compliance_rate(self, explanation_sla, store):
        submitted = _fixed("2026-04-21T00:00:00+00:00")
        # 2 on-time, 1 late
        for delta in (3, 5, 12):
            req = explanation_sla.open_request(
                user_id=f"u_{delta}",
                request_type=RequestType.EXPLANATION,
                submitted_at=submitted,
            )
            explanation_sla.mark_processed(
                req.request_id,
                processed_at=submitted + timedelta(days=delta),
            )
        report = explanation_sla.generate_report(
            window_start=submitted - timedelta(days=1),
            window_end=submitted + timedelta(days=30),
        )
        assert report.total_requests == 3
        assert report.processed_on_time == 2
        assert report.processed_late == 1
        assert report.compliance_rate == pytest.approx(2 / 3)


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

class TestBuildStore:
    def test_in_memory_default(self):
        store = build_compliance_store({"store": {"backend": "in_memory"}})
        assert isinstance(store, InMemoryComplianceStore)

    def test_flat_config_also_works(self):
        store = build_compliance_store({"backend": "in_memory"})
        assert isinstance(store, InMemoryComplianceStore)

    def test_dynamodb_requires_tables(self):
        with pytest.raises(ValueError, match="requests_table"):
            build_compliance_store({"store": {"backend": "dynamodb"}})

    def test_unknown_backend(self):
        with pytest.raises(ValueError, match="Unknown"):
            build_compliance_store({"store": {"backend": "sqlite"}})


# ---------------------------------------------------------------------------
# Config-driven wiring: pipeline.yaml compliance block must be readable
# ---------------------------------------------------------------------------

class TestPipelineYAMLBlock:
    def test_compliance_block_present_and_parseable(self):
        import yaml
        from pathlib import Path

        cfg_path = Path("configs/pipeline.yaml")
        assert cfg_path.exists(), f"Missing {cfg_path}"
        with cfg_path.open("r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        assert "compliance" in cfg, "pipeline.yaml missing compliance: block"
        block = cfg["compliance"]
        assert "store" in block
        assert "sla" in block
        assert block["sla"]["explanation_response_days"] == 10
        assert block["retention"]["default_days"] == 1825

    def test_compliance_block_drives_factory(self):
        import yaml
        from pathlib import Path

        cfg_path = Path("configs/pipeline.yaml")
        with cfg_path.open("r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        store = build_compliance_store(cfg["compliance"])
        # default backend in config is in_memory
        assert isinstance(store, InMemoryComplianceStore)
