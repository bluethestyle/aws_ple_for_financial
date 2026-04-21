"""
Sprint 3 tests: M1 HumanReviewQueue, M2 KillSwitch (in-memory),
M10 DynamicItemUniverseLoader, M11 audit archive columns,
M12 MarkerApplier.

Run: pytest tests/test_sprint3_serving.py -v
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pytest
import yaml

from core.recommendation.audit_archiver import (
    RecommendationAuditArchiver,
    RecommendationAuditRecord,
)
from core.recommendation.reason.marker_applier import (
    DEFAULT_MARKER_TEXT,
    MarkerApplier,
    MarkerConfig,
    wrap_provider,
)
from core.recommendation.universe.dynamic_loader import (
    Campaign,
    CampaignStatus,
    DynamicItemUniverseLoader,
    ItemUniverseConfig,
    Product,
    build_item_universe_loader,
)
from core.serving.kill_switch import KillSwitch
from core.serving.review.human_review_queue import (
    HumanReviewQueue,
    ReviewConfig,
    ReviewItem,
    ReviewState,
    build_human_review_queue,
)


# ---------------------------------------------------------------------------
# M1 - HumanReviewQueue
# ---------------------------------------------------------------------------

class TestReviewConfig:
    def test_defaults(self):
        cfg = ReviewConfig()
        assert cfg.tier_1_sample_rate == 0.05
        assert cfg.tier_2_review_required is True
        assert cfg.tier_3_human_fallback is True
        assert cfg.sla_hours == 24

    def test_rejects_bad_rate(self):
        with pytest.raises(ValueError):
            ReviewConfig(tier_1_sample_rate=1.5)

    def test_rejects_bad_sla(self):
        with pytest.raises(ValueError):
            ReviewConfig(sla_hours=0)

    def test_from_dict(self):
        cfg = ReviewConfig.from_dict({
            "tier_1_sample_rate": 0.1,
            "tier_2_review_required": False,
            "sla_hours": 48,
        })
        assert cfg.tier_1_sample_rate == 0.1
        assert cfg.tier_2_review_required is False
        assert cfg.sla_hours == 48


class TestReviewItem:
    def test_basic_construction(self):
        item = ReviewItem(
            review_id="r1", user_id="u1",
            recommendation_id="rec1", tier=2,
        )
        assert item.state == ReviewState.PENDING
        assert item.created_at != ""

    def test_rejects_bad_tier(self):
        with pytest.raises(ValueError):
            ReviewItem(
                review_id="r1", user_id="u1",
                recommendation_id="rec1", tier=99,
            )


class TestShouldEnqueue:
    def test_tier_2_always_enqueued(self):
        q = HumanReviewQueue(seed=0)
        assert q.should_enqueue(2) is True

    def test_tier_3_always_enqueued(self):
        q = HumanReviewQueue(seed=0)
        assert q.should_enqueue(3) is True

    def test_tier_1_sampled(self):
        # seed=0 → first rng draw is deterministic; sample rate 1.0 → always
        q = HumanReviewQueue(
            config=ReviewConfig(tier_1_sample_rate=1.0), seed=0,
        )
        assert q.should_enqueue(1) is True

    def test_tier_1_not_sampled_when_rate_zero(self):
        q = HumanReviewQueue(
            config=ReviewConfig(tier_1_sample_rate=0.0), seed=0,
        )
        assert q.should_enqueue(1) is False


class TestQueueLifecycle:
    def test_enqueue_returns_item_for_required_tier(self):
        q = HumanReviewQueue(seed=0)
        item = q.enqueue(user_id="u1", recommendation_id="r1", tier=2)
        assert item is not None
        assert item.state == ReviewState.PENDING

    def test_enqueue_returns_none_for_skipped_tier(self):
        q = HumanReviewQueue(
            config=ReviewConfig(tier_1_sample_rate=0.0), seed=0,
        )
        item = q.enqueue(user_id="u1", recommendation_id="r1", tier=1)
        assert item is None

    def test_dequeue_moves_to_in_review(self):
        q = HumanReviewQueue(seed=0)
        item = q.enqueue(user_id="u1", recommendation_id="r1", tier=2)
        got = q.dequeue(reviewer_id="alice", tier=2)
        assert got is not None
        assert got.review_id == item.review_id
        assert got.state == ReviewState.IN_REVIEW
        assert got.reviewer_id == "alice"

    def test_dequeue_empty_returns_none(self):
        q = HumanReviewQueue(seed=0)
        assert q.dequeue(reviewer_id="alice", tier=2) is None

    def test_dequeue_rejects_bad_tier(self):
        q = HumanReviewQueue(seed=0)
        with pytest.raises(ValueError):
            q.dequeue(reviewer_id="alice", tier=5)

    def test_approve_transitions_state(self):
        q = HumanReviewQueue(seed=0)
        item = q.enqueue(user_id="u1", recommendation_id="r1", tier=2)
        q.dequeue(reviewer_id="alice", tier=2)
        updated = q.approve(item.review_id, reviewer_id="alice", reason="ok")
        assert updated.state == ReviewState.APPROVED
        assert updated.disposition_at is not None

    def test_reject_requires_reason(self):
        q = HumanReviewQueue(seed=0)
        item = q.enqueue(user_id="u1", recommendation_id="r1", tier=2)
        q.dequeue(reviewer_id="alice", tier=2)
        with pytest.raises(ValueError):
            q.reject(item.review_id, reviewer_id="alice", reason="")

    def test_modify_applies_modifications(self):
        q = HumanReviewQueue(seed=0)
        item = q.enqueue(user_id="u1", recommendation_id="r1", tier=2)
        q.dequeue(reviewer_id="alice", tier=2)
        updated = q.modify(
            item.review_id, reviewer_id="alice",
            modifications={"rank": 1, "score": 0.9},
            reason="bumped",
        )
        assert updated.state == ReviewState.MODIFIED
        assert updated.modifications == {"rank": 1, "score": 0.9}

    def test_modify_requires_non_empty(self):
        q = HumanReviewQueue(seed=0)
        item = q.enqueue(user_id="u1", recommendation_id="r1", tier=2)
        with pytest.raises(ValueError):
            q.modify(
                item.review_id, reviewer_id="alice",
                modifications={}, reason="x",
            )

    def test_cannot_redispose(self):
        q = HumanReviewQueue(seed=0)
        item = q.enqueue(user_id="u1", recommendation_id="r1", tier=2)
        q.dequeue(reviewer_id="alice", tier=2)
        q.approve(item.review_id, reviewer_id="alice")
        with pytest.raises(ValueError):
            q.reject(item.review_id, reviewer_id="alice", reason="changed")

    def test_list_pending_filters_by_tier(self):
        q = HumanReviewQueue(seed=0)
        q.enqueue(user_id="u1", recommendation_id="r1", tier=2)
        q.enqueue(user_id="u2", recommendation_id="r2", tier=3)
        assert len(q.list_pending()) == 2
        assert len(q.list_pending(tier=2)) == 1
        assert len(q.list_pending(tier=3)) == 1

    def test_summary_counts(self):
        q = HumanReviewQueue(seed=0)
        q.enqueue(user_id="u1", recommendation_id="r1", tier=2)
        item = q.enqueue(user_id="u2", recommendation_id="r2", tier=3)
        q.dequeue(reviewer_id="alice", tier=3)
        q.approve(item.review_id, reviewer_id="alice")
        s = q.summary()
        assert s["total"] == 2
        assert s["by_state"][ReviewState.APPROVED] == 1
        assert s["by_state"][ReviewState.PENDING] == 1


class TestAuditCallback:
    def test_audit_callback_invoked_per_transition(self):
        events = []
        q = HumanReviewQueue(audit_callback=lambda e: events.append(e), seed=0)
        item = q.enqueue(user_id="u1", recommendation_id="r1", tier=2)
        q.dequeue(reviewer_id="alice", tier=2)
        q.approve(item.review_id, reviewer_id="alice")
        actions = [e["action"] for e in events]
        assert "human_review:enqueue" in actions
        assert "human_review:dequeue" in actions
        assert f"human_review:{ReviewState.APPROVED}" in actions

    def test_audit_failure_is_swallowed(self):
        def bad_cb(evt):
            raise RuntimeError("audit down")
        q = HumanReviewQueue(audit_callback=bad_cb, seed=0)
        # Should not raise — failure in audit is never allowed to block
        item = q.enqueue(user_id="u1", recommendation_id="r1", tier=2)
        assert item is not None


class TestReviewQueueFactory:
    def test_build_from_pipeline_yaml(self):
        cfg = yaml.safe_load(
            Path("configs/pipeline.yaml").read_text(encoding="utf-8")
        )
        q = build_human_review_queue(pipeline_config=cfg)
        assert isinstance(q, HumanReviewQueue)
        item = q.enqueue(user_id="u1", recommendation_id="r1", tier=2)
        assert item is not None


# ---------------------------------------------------------------------------
# M2 - KillSwitch (in-memory mode)
# ---------------------------------------------------------------------------

class TestKillSwitchInMemory:
    def test_default_inactive(self):
        ks = KillSwitch(use_dynamo=False)
        state = ks.check()
        assert state.active is False

    def test_activate_and_check_global(self):
        ks = KillSwitch(use_dynamo=False)
        ks.activate(scope="global", reason="hotfix", activated_by="ops")
        state = ks.check()
        assert state.active is True
        assert state.scope == "global"
        assert state.reason == "hotfix"

    def test_task_scope_priority_after_global(self):
        ks = KillSwitch(use_dynamo=False)
        ks.activate(scope="task:ctr", reason="drift", activated_by="ops")
        state = ks.check(task="ctr")
        assert state.active is True
        assert state.scope == "task:ctr"

    def test_deactivate_sets_inactive(self):
        ks = KillSwitch(use_dynamo=False)
        ks.activate(scope="global", reason="hotfix")
        ks.deactivate(scope="global", deactivated_by="ops")
        state = ks.check()
        assert state.active is False

    def test_from_config_reads_backend(self):
        cfg = yaml.safe_load(
            Path("configs/pipeline.yaml").read_text(encoding="utf-8")
        )
        # Override to in_memory for this test
        cfg.setdefault("serving", {}).setdefault("kill_switch", {})
        cfg["serving"]["kill_switch"]["backend"] = "in_memory"
        ks = KillSwitch.from_config(pipeline_config=cfg)
        # Should not touch DynamoDB
        state = ks.check()
        assert state.active is False


# ---------------------------------------------------------------------------
# M10 - DynamicItemUniverseLoader
# ---------------------------------------------------------------------------

class TestItemUniverseConfig:
    def test_defaults_disabled(self):
        cfg = ItemUniverseConfig()
        assert cfg.enabled is False
        assert CampaignStatus.APPROVED in cfg.active_statuses

    def test_from_dict_rejects_unknown_status(self):
        with pytest.raises(ValueError):
            ItemUniverseConfig.from_dict({
                "active_statuses": ["live"],  # not a CampaignStatus
            })


class TestUniverseLoaderDisabled:
    def test_load_returns_empty_when_disabled(self):
        loader = DynamicItemUniverseLoader()
        assert loader.load() == []


class TestUniverseLoaderWithFakeConn:
    """Inject a fake DuckDB-compatible connection."""

    class FakeResult:
        def __init__(self, rows, columns):
            self._rows = rows
            self.description = [(c,) for c in columns]

        def fetchall(self):
            return [tuple(r.get(c) for c in [d[0] for d in self.description])
                    for r in self._rows]

    class FakeConn:
        def __init__(self):
            self._responses: dict = {}

        def configure(self, path, rows, columns):
            self._responses[path] = (rows, columns)

        def execute(self, sql, params=None):
            path = params[0] if params else None
            rows, cols = self._responses.get(path, ([], []))
            return TestUniverseLoaderWithFakeConn.FakeResult(rows, cols)

        def close(self):
            pass

    def _approved_campaign(self, cid="c1", status="approved"):
        return {
            "campaign_id": cid, "name": f"Campaign {cid}",
            "status": status,
            "start_date": "2026-01-01",
            "end_date": "2026-12-31",
            "target_segments": ["premium", "stable"],
        }

    def test_active_campaigns_filtered_by_status(self):
        conn = self.FakeConn()
        conn.configure(
            "campaigns.parquet",
            [
                self._approved_campaign("c1", "approved"),
                self._approved_campaign("c2", "canceled"),
                self._approved_campaign("c3", "running"),
            ],
            ["campaign_id", "name", "status",
             "start_date", "end_date", "target_segments"],
        )
        loader = DynamicItemUniverseLoader(
            config=ItemUniverseConfig(
                enabled=True, campaign_parquet="campaigns.parquet",
            ),
            duckdb_conn=conn,
        )
        active = loader.get_active_campaigns(
            as_of=datetime(2026, 6, 1, tzinfo=timezone.utc)
        )
        assert {c.campaign_id for c in active} == {"c1", "c3"}

    def test_campaign_date_window_filters_out_of_range(self):
        conn = self.FakeConn()
        conn.configure(
            "campaigns.parquet",
            [{
                "campaign_id": "c_old", "name": "old", "status": "approved",
                "start_date": "2025-01-01", "end_date": "2025-12-31",
                "target_segments": [],
            }],
            ["campaign_id", "name", "status",
             "start_date", "end_date", "target_segments"],
        )
        loader = DynamicItemUniverseLoader(
            config=ItemUniverseConfig(
                enabled=True, campaign_parquet="campaigns.parquet",
            ),
            duckdb_conn=conn,
        )
        # as_of is 2026 → campaign window [2025] is out of range
        active = loader.get_active_campaigns(
            as_of=datetime(2026, 6, 1, tzinfo=timezone.utc),
        )
        assert active == []

    def test_load_returns_campaigns_and_products(self):
        conn = self.FakeConn()
        conn.configure(
            "campaigns.parquet",
            [self._approved_campaign("c1", "running")],
            ["campaign_id", "name", "status",
             "start_date", "end_date", "target_segments"],
        )
        conn.configure(
            "products.parquet",
            [
                {"product_id": "p1", "name": "Savings",
                 "category": "deposits", "active": True},
                {"product_id": "p2", "name": "Old", "category": "insurance",
                 "active": False},
            ],
            ["product_id", "name", "category", "active"],
        )
        loader = DynamicItemUniverseLoader(
            config=ItemUniverseConfig(
                enabled=True,
                campaign_parquet="campaigns.parquet",
                product_parquet="products.parquet",
            ),
            duckdb_conn=conn,
        )
        items = loader.load(as_of=datetime(2026, 6, 1, tzinfo=timezone.utc))
        types = [i.item_type for i in items]
        assert "campaign" in types
        assert "product" in types
        # Inactive product filtered out
        product_ids = [i.item_id for i in items if i.item_type == "product"]
        assert "p1" in product_ids
        assert "p2" not in product_ids

    def test_cache_hits_avoid_repeated_queries(self):
        conn = self.FakeConn()
        conn.configure(
            "campaigns.parquet",
            [self._approved_campaign("c1", "approved")],
            ["campaign_id", "name", "status",
             "start_date", "end_date", "target_segments"],
        )
        loader = DynamicItemUniverseLoader(
            config=ItemUniverseConfig(
                enabled=True,
                campaign_parquet="campaigns.parquet",
                cache_ttl_seconds=3600,
            ),
            duckdb_conn=conn,
        )
        loader.get_active_campaigns(
            as_of=datetime(2026, 6, 1, tzinfo=timezone.utc),
        )
        # Mutate underlying response; cache should still return old value
        conn.configure(
            "campaigns.parquet", [],
            ["campaign_id", "name", "status",
             "start_date", "end_date", "target_segments"],
        )
        cached = loader.get_active_campaigns(
            as_of=datetime(2026, 6, 1, tzinfo=timezone.utc),
        )
        assert len(cached) == 1

    def test_invalidate_cache_refetches(self):
        conn = self.FakeConn()
        conn.configure(
            "campaigns.parquet",
            [self._approved_campaign("c1", "approved")],
            ["campaign_id", "name", "status",
             "start_date", "end_date", "target_segments"],
        )
        loader = DynamicItemUniverseLoader(
            config=ItemUniverseConfig(
                enabled=True,
                campaign_parquet="campaigns.parquet",
            ),
            duckdb_conn=conn,
        )
        loader.get_active_campaigns(
            as_of=datetime(2026, 6, 1, tzinfo=timezone.utc),
        )
        loader.invalidate_cache()
        conn.configure(
            "campaigns.parquet", [],
            ["campaign_id", "name", "status",
             "start_date", "end_date", "target_segments"],
        )
        fresh = loader.get_active_campaigns(
            as_of=datetime(2026, 6, 1, tzinfo=timezone.utc),
        )
        assert fresh == []


class TestUniverseLoaderFactory:
    def test_factory_from_pipeline_yaml(self):
        cfg = yaml.safe_load(
            Path("configs/pipeline.yaml").read_text(encoding="utf-8")
        )
        loader = build_item_universe_loader(pipeline_config=cfg)
        # Disabled in repo default
        assert loader.load() == []


# ---------------------------------------------------------------------------
# M11 - Audit Archive extended columns
# ---------------------------------------------------------------------------

class TestAuditArchiveColumns:
    def test_record_accepts_new_columns(self):
        archiver = RecommendationAuditArchiver(local_path="tmp_audit")
        archiver.start_batch("batch_test")
        archiver.record(RecommendationAuditRecord(
            customer_id=1, product_id="p1", task_name="churn",
            rank=1, score=0.8,
            thinking_trace="considered features A,B,C",
            hallucination_flags=["low_entropy"],
            tools_used=["feature_lookup", "llm_rewrite"],
            critique_verdict="pass",
            agent_tier=2,
        ))
        assert len(archiver._buffer) == 1
        row = archiver._buffer[0]
        assert row["thinking_trace"] == "considered features A,B,C"
        assert row["critique_verdict"] == "pass"
        assert row["agent_tier"] == 2
        # List columns flattened to JSON
        assert "low_entropy" in row["hallucination_flags"]
        assert "feature_lookup" in row["tools_used"]

    def test_backward_compat_old_record(self):
        """A record without the new fields must still be writable."""
        archiver = RecommendationAuditArchiver(local_path="tmp_audit")
        archiver.start_batch("batch_legacy")
        archiver.record(RecommendationAuditRecord(
            customer_id=1, product_id="p1", task_name="churn",
            rank=1, score=0.8,
        ))
        row = archiver._buffer[0]
        # Defaults
        assert row["thinking_trace"] == ""
        assert row["agent_tier"] == 0
        assert row["hallucination_flags"] == "[]"
        assert row["tools_used"] == "[]"


# ---------------------------------------------------------------------------
# M12 - MarkerApplier
# ---------------------------------------------------------------------------

class TestMarkerConfig:
    def test_defaults(self):
        cfg = MarkerConfig()
        assert cfg.enabled is True
        assert "AI기본법" in cfg.marker_text
        assert cfg.position == "append"

    def test_rejects_bad_position(self):
        with pytest.raises(ValueError):
            MarkerConfig(position="middle")

    def test_rejects_empty_marker(self):
        with pytest.raises(ValueError):
            MarkerConfig(marker_text="   ")

    def test_from_dict(self):
        cfg = MarkerConfig.from_dict({
            "enabled": False,
            "marker_text": "CUSTOM",
            "position": "prepend",
            "idempotent_key": "CUSTOM",
        })
        assert cfg.enabled is False
        assert cfg.marker_text == "CUSTOM"
        assert cfg.position == "prepend"


class TestMarkerApplier:
    def test_append_marker(self):
        applier = MarkerApplier()
        out = applier.apply("이 상품을 추천합니다.")
        assert out.endswith(DEFAULT_MARKER_TEXT)
        assert "이 상품을 추천합니다." in out

    def test_prepend_marker(self):
        applier = MarkerApplier(
            config=MarkerConfig(position="prepend")
        )
        out = applier.apply("hello")
        assert out.startswith(DEFAULT_MARKER_TEXT)

    def test_disabled_passthrough(self):
        applier = MarkerApplier(config=MarkerConfig(enabled=False))
        assert applier.apply("unchanged") == "unchanged"

    def test_idempotent_does_not_double_apply(self):
        applier = MarkerApplier()
        once = applier.apply("text")
        twice = applier.apply(once)
        assert once == twice

    def test_none_returns_empty_string(self):
        applier = MarkerApplier()
        assert applier.apply(None) == ""

    def test_empty_text_returns_marker_only(self):
        applier = MarkerApplier()
        out = applier.apply("")
        assert out == DEFAULT_MARKER_TEXT

    def test_has_marker_detects_existing(self):
        applier = MarkerApplier()
        marked = applier.apply("abc")
        assert applier.has_marker(marked) is True
        assert applier.has_marker("abc") is False


class TestMarkerFromPipelineYAML:
    def test_loads_pipeline_yaml_config(self):
        cfg = yaml.safe_load(
            Path("configs/pipeline.yaml").read_text(encoding="utf-8")
        )
        applier = MarkerApplier.from_config(cfg)
        assert applier._cfg.enabled is True
        assert "AI기본법" in applier._cfg.marker_text


class TestProviderWrapper:
    class _FakeProvider:
        def __init__(self):
            self.metadata = "static"

        def generate(self, prompt, **kwargs):
            return f"LLM reply to {prompt}"

        def is_available(self):
            return True

    def test_wrap_provider_applies_marker(self):
        wrapped = wrap_provider(self._FakeProvider(), MarkerApplier())
        out = wrapped.generate("show me a recommendation")
        assert out.endswith(DEFAULT_MARKER_TEXT)
        assert "LLM reply to show me a recommendation" in out

    def test_wrap_provider_delegates_unknown_attrs(self):
        wrapped = wrap_provider(self._FakeProvider(), MarkerApplier())
        assert wrapped.metadata == "static"

    def test_wrap_provider_is_available(self):
        wrapped = wrap_provider(self._FakeProvider(), MarkerApplier())
        assert wrapped.is_available() is True
