"""
Sprint 2 S6 tests: ComplianceSQLHelper (DuckDB over Parquet).

Run: pytest tests/test_compliance_sql.py -v
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List

import pytest

from core.compliance.audit_sql import (
    AuditSQLConfig,
    ComplianceSQLHelper,
    build_compliance_sql_helper,
)

duckdb = pytest.importorskip("duckdb")
pq = pytest.importorskip("pyarrow.parquet")
pa = pytest.importorskip("pyarrow")


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _write_parquet(path: Path, rows: List[Dict[str, Any]]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(rows), str(path))
    return path


@pytest.fixture
def opt_out_file(tmp_path: Path) -> Path:
    base = datetime(2026, 4, 1, tzinfo=timezone.utc)
    rows = [
        {"user_id": "u1", "event_type": "opt_out",
         "timestamp": (base + timedelta(days=1)).isoformat(),
         "fallback_type": "human_review"},
        {"user_id": "u2", "event_type": "opt_out",
         "timestamp": (base + timedelta(days=5)).isoformat(),
         "fallback_type": "rule_based"},
        {"user_id": "u3", "event_type": "opt_out",
         "timestamp": (base + timedelta(days=20)).isoformat(),
         "fallback_type": "disable"},
    ]
    return _write_parquet(tmp_path / "opt_out.parquet", rows)


@pytest.fixture
def consent_file(tmp_path: Path) -> Path:
    base = datetime(2026, 4, 1, tzinfo=timezone.utc)
    rows = [
        {"user_id": "u1", "channel": "SMS",
         "timestamp": (base + timedelta(days=2)).isoformat(),
         "action": "grant"},
        {"user_id": "u1", "channel": "EMAIL",
         "timestamp": (base + timedelta(days=5)).isoformat(),
         "action": "grant"},
        {"user_id": "u2", "channel": "SMS",
         "timestamp": (base + timedelta(days=4)).isoformat(),
         "action": "revoke"},
    ]
    return _write_parquet(tmp_path / "consent.parquet", rows)


@pytest.fixture
def events_file(tmp_path: Path) -> Path:
    base = datetime(2026, 4, 1, tzinfo=timezone.utc)
    rows = [
        {"user_id": "u1", "event_type": "sla_breach",
         "timestamp": (base + timedelta(days=11)).isoformat(),
         "sla_name": "explanation"},
        {"user_id": "u2", "event_type": "request_processed",
         "timestamp": (base + timedelta(days=3)).isoformat(),
         "sla_name": "opt_out"},
        {"user_id": "u3", "event_type": "sla_breach",
         "timestamp": (base + timedelta(days=14)).isoformat(),
         "sla_name": "profiling"},
    ]
    return _write_parquet(tmp_path / "events.parquet", rows)


@pytest.fixture
def helper() -> ComplianceSQLHelper:
    return ComplianceSQLHelper()


# ---------------------------------------------------------------------------
# AuditSQLConfig
# ---------------------------------------------------------------------------

class TestAuditSQLConfig:
    def test_defaults(self):
        cfg = AuditSQLConfig()
        assert cfg.enabled is True
        assert cfg.default_since_days == 30
        assert cfg.install_httpfs is True

    def test_rejects_nonpositive_days(self):
        with pytest.raises(ValueError):
            AuditSQLConfig(default_since_days=0)

    def test_from_dict(self):
        cfg = AuditSQLConfig.from_dict({
            "enabled": False,
            "paths": {"opt_out": "local.parquet"},
            "default_since_days": 90,
        })
        assert cfg.enabled is False
        assert cfg.paths == {"opt_out": "local.parquet"}
        assert cfg.default_since_days == 90

    def test_from_empty_dict(self):
        cfg = AuditSQLConfig.from_dict(None)
        assert cfg.default_since_days == 30


# ---------------------------------------------------------------------------
# Lifecycle
# ---------------------------------------------------------------------------

class TestHelperLifecycle:
    def test_register_view_valid_name(self, helper, opt_out_file):
        helper.register_view("opt_out", str(opt_out_file))
        assert "opt_out" in helper.list_views()

    def test_register_view_rejects_bad_name(self, helper, opt_out_file):
        with pytest.raises(ValueError):
            helper.register_view("opt-out drop", str(opt_out_file))

    def test_register_view_idempotent_overwrite(self, helper, opt_out_file):
        helper.register_view("opt_out", str(opt_out_file))
        helper.register_view("opt_out", str(opt_out_file))
        assert helper.list_views()["opt_out"] == str(opt_out_file)

    def test_context_manager_closes(self, opt_out_file):
        with ComplianceSQLHelper() as h:
            h.register_view("opt_out", str(opt_out_file))
            assert "opt_out" in h.list_views()
        # after close, no-op

    def test_config_paths_auto_registered(self, opt_out_file):
        cfg = AuditSQLConfig(paths={"opt_out": str(opt_out_file)})
        h = ComplianceSQLHelper(config=cfg)
        try:
            assert "opt_out" in h.list_views()
        finally:
            h.close()


# ---------------------------------------------------------------------------
# query() surface
# ---------------------------------------------------------------------------

class TestQuerySurface:
    def test_arbitrary_sql(self, helper, opt_out_file):
        helper.register_view("opt_out", str(opt_out_file))
        rows = helper.query("SELECT COUNT(*) AS n FROM opt_out")
        assert rows == [{"n": 3}]

    def test_query_with_params(self, helper, opt_out_file):
        helper.register_view("opt_out", str(opt_out_file))
        rows = helper.query(
            "SELECT user_id FROM opt_out WHERE fallback_type = ?",
            ["rule_based"],
        )
        assert rows == [{"user_id": "u2"}]

    def test_disabled_helper_returns_empty(self, opt_out_file):
        h = ComplianceSQLHelper(config=AuditSQLConfig(enabled=False))
        try:
            h.register_view("opt_out", str(opt_out_file))
            assert h.query("SELECT * FROM opt_out") == []
        finally:
            h.close()


# ---------------------------------------------------------------------------
# Regulator queries
# ---------------------------------------------------------------------------

class TestRegulatorQueries:
    def test_recent_opt_outs_default_window(self, helper, opt_out_file):
        helper.register_view("opt_out", str(opt_out_file))
        # All events are >30d ago relative to today (2026-04-21+), so we
        # pass an explicit since from 2026-04-01 to verify the SQL path.
        rows = helper.recent_opt_outs(
            since=datetime(2026, 4, 1, tzinfo=timezone.utc)
        )
        assert len(rows) == 3

    def test_recent_opt_outs_tight_window(self, helper, opt_out_file):
        helper.register_view("opt_out", str(opt_out_file))
        rows = helper.recent_opt_outs(
            since=datetime(2026, 4, 10, tzinfo=timezone.utc)
        )
        # only u3 at day 20 is after 2026-04-10
        assert len(rows) == 1
        assert rows[0]["user_id"] == "u3"

    def test_consent_changes_for_user(self, helper, consent_file):
        helper.register_view("consent", str(consent_file))
        rows = helper.consent_changes_for_user(
            user_id="u1",
            since=datetime(2026, 4, 1, tzinfo=timezone.utc),
        )
        assert len(rows) == 2
        assert all(r["user_id"] == "u1" for r in rows)

    def test_sla_breaches(self, helper, events_file):
        helper.register_view("events", str(events_file))
        rows = helper.sla_breaches(
            since=datetime(2026, 4, 1, tzinfo=timezone.utc),
        )
        assert len(rows) == 2
        assert {r["sla_name"] for r in rows} == {"explanation", "profiling"}

    def test_promotion_gate_history_without_filter(
        self, helper, tmp_path,
    ):
        path = _write_parquet(tmp_path / "pg.parquet", [
            {"model_version": "v1", "decision": "pass",
             "timestamp": "2026-04-15T00:00:00+00:00"},
            {"model_version": "v2", "decision": "reject",
             "timestamp": "2026-04-20T00:00:00+00:00"},
        ])
        helper.register_view("promotion_gate", str(path))
        rows = helper.promotion_gate_history()
        assert len(rows) == 2

    def test_promotion_gate_history_filtered(self, helper, tmp_path):
        path = _write_parquet(tmp_path / "pg.parquet", [
            {"model_version": "v1", "decision": "pass",
             "timestamp": "2026-04-15T00:00:00+00:00"},
            {"model_version": "v2", "decision": "reject",
             "timestamp": "2026-04-20T00:00:00+00:00"},
        ])
        helper.register_view("promotion_gate", str(path))
        rows = helper.promotion_gate_history(model_version="v2")
        assert rows == [{
            "model_version": "v2", "decision": "reject",
            "timestamp": "2026-04-20T00:00:00+00:00",
        }]

    def test_counts_by_column(self, helper, opt_out_file):
        helper.register_view("opt_out", str(opt_out_file))
        counts = helper.counts_by_column("opt_out", "fallback_type")
        assert counts == {"human_review": 1, "rule_based": 1, "disable": 1}


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

class TestFactory:
    def test_build_from_pipeline_config(self, opt_out_file):
        cfg = {
            "compliance": {
                "audit_sql": {
                    "enabled": True,
                    "paths": {"opt_out": str(opt_out_file)},
                    "default_since_days": 90,
                    "install_httpfs": False,
                }
            }
        }
        h = build_compliance_sql_helper(cfg)
        try:
            assert "opt_out" in h.list_views()
            rows = h.query("SELECT COUNT(*) AS n FROM opt_out")
            assert rows[0]["n"] == 3
        finally:
            h.close()

    def test_build_from_empty_config(self):
        h = build_compliance_sql_helper({})
        try:
            assert h.list_views() == {}
        finally:
            h.close()


# ---------------------------------------------------------------------------
# Multi-view join
# ---------------------------------------------------------------------------

class TestJoinAcrossViews:
    def test_cross_view_join(self, helper, opt_out_file, consent_file):
        helper.register_view("opt_out", str(opt_out_file))
        helper.register_view("consent", str(consent_file))
        rows = helper.query(
            "SELECT o.user_id, COUNT(c.channel) AS consent_count "
            "FROM opt_out o LEFT JOIN consent c USING (user_id) "
            "GROUP BY o.user_id ORDER BY o.user_id"
        )
        by_user = {r["user_id"]: r["consent_count"] for r in rows}
        assert by_user["u1"] == 2   # u1 has SMS + EMAIL
        assert by_user["u2"] == 1   # u2 has SMS
        assert by_user["u3"] == 0   # u3 has no consent row
