"""
Tests for archive-backed metadata sources.

Runtime objects (DataLineageTracker, FairnessMonitor) are empty on a
fresh process start, so ``build_metadata_aggregator_from_config`` also
accepts ``lineage_yaml_path`` and ``fairness_archive_path`` kwargs that
read from the persistent archives. These tests validate that:

1. ``build_lineage_yaml_source`` parses feature_source_map from YAML.
2. ``build_fairness_archive_source`` reads DI values from a Parquet
   archive and returns the worst-case.
3. ``build_metadata_aggregator_from_config`` wires both forms through.
4. Archive paths win over empty runtime objects (no regression of live
   instance contribution).

Run: ``pytest tests/test_metadata_archive_sources.py -v``
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict

import pytest

from core.compliance.metadata_aggregator import (
    build_fairness_archive_source,
    build_lineage_yaml_source,
    build_metadata_aggregator_from_config,
)


# ---------------------------------------------------------------------------
# Lineage YAML source
# ---------------------------------------------------------------------------

def _write_lineage_yaml(path: Path, feature_map: Dict[str, Dict[str, Any]]) -> None:
    import yaml
    path.write_text(
        yaml.safe_dump({"feature_source_map": feature_map}),
        encoding="utf-8",
    )


class TestLineageYamlSource:
    def test_reads_pseudonymized_flags(self, tmp_path):
        p = tmp_path / "feature_groups.yaml"
        _write_lineage_yaml(p, {
            "spend_": {"pseudonymized": True},
            "product_": {"pseudonymized": True},
            "raw_": {"pseudonymized": False},
            "benefit_": {"pseudonymized": False},
        })
        src = build_lineage_yaml_source(str(p))
        assert src("v1") == {"pii_ratio": 0.5}

    def test_accepts_nested_lineage_key(self, tmp_path):
        import yaml
        p = tmp_path / "lineage.yaml"
        p.write_text(yaml.safe_dump({
            "lineage": {
                "feature_source_map": {
                    "a_": {"pseudonymized": True},
                    "b_": {"pseudonymized": False},
                },
            },
        }), encoding="utf-8")
        src = build_lineage_yaml_source(str(p))
        assert src("v1") == {"pii_ratio": 0.5}

    def test_missing_file_returns_empty(self, tmp_path):
        src = build_lineage_yaml_source(str(tmp_path / "nope.yaml"))
        assert src("v1") == {}

    def test_malformed_yaml_returns_empty(self, tmp_path):
        p = tmp_path / "bad.yaml"
        p.write_text(":::\n  invalid", encoding="utf-8")
        src = build_lineage_yaml_source(str(p))
        assert src("v1") == {}

    def test_empty_feature_map_returns_empty(self, tmp_path):
        p = tmp_path / "empty.yaml"
        _write_lineage_yaml(p, {})
        src = build_lineage_yaml_source(str(p))
        assert src("v1") == {}

    def test_all_pseudonymized_gives_zero_ratio(self, tmp_path):
        p = tmp_path / "fg.yaml"
        _write_lineage_yaml(p, {
            "a_": {"pseudonymized": True},
            "b_": {"pseudonymized": True},
        })
        src = build_lineage_yaml_source(str(p))
        assert src("v1") == {"pii_ratio": 0.0}


# ---------------------------------------------------------------------------
# Fairness archive source
# ---------------------------------------------------------------------------

def _write_fairness_parquet(path: Path, rows: list) -> None:
    pa = pytest.importorskip("pyarrow")
    import pyarrow.parquet as pq
    table = pa.Table.from_pylist(rows)
    pq.write_table(table, str(path))


class TestFairnessArchiveSource:
    def test_reads_min_di_from_parquet(self, tmp_path):
        _write_fairness_parquet(tmp_path / "fa.parquet", [
            {"disparate_impact": 0.95, "attribute": "age"},
            {"disparate_impact": 0.70, "attribute": "gender"},
            {"disparate_impact": 1.10, "attribute": "region"},
        ])
        src = build_fairness_archive_source(str(tmp_path / "fa.parquet"))
        out = src("v1")
        assert out["disparate_impact_min"] == pytest.approx(0.70, rel=1e-4)

    def test_penalises_di_above_one(self, tmp_path):
        _write_fairness_parquet(tmp_path / "fa.parquet", [
            {"disparate_impact": 1.5, "attribute": "age"},
        ])
        src = build_fairness_archive_source(str(tmp_path / "fa.parquet"))
        out = src("v1")
        # min(v, 1/v) for v=1.5 → 2/3
        assert out["disparate_impact_min"] == pytest.approx(1.0 / 1.5, rel=1e-4)

    def test_limit_takes_only_tail(self, tmp_path):
        # 100 rows: first 10 have worst DI=0.1, last 5 have DI=0.95
        rows = [{"disparate_impact": 0.1} for _ in range(10)]
        rows += [{"disparate_impact": 0.9} for _ in range(85)]
        rows += [{"disparate_impact": 0.95} for _ in range(5)]
        _write_fairness_parquet(tmp_path / "fa.parquet", rows)
        src = build_fairness_archive_source(
            str(tmp_path / "fa.parquet"), limit=5,
        )
        out = src("v1")
        assert out["disparate_impact_min"] == pytest.approx(0.95, rel=1e-3)

    def test_missing_file_returns_empty(self, tmp_path):
        src = build_fairness_archive_source(str(tmp_path / "nope.parquet"))
        assert src("v1") == {}

    def test_empty_archive_returns_empty(self, tmp_path):
        pa = pytest.importorskip("pyarrow")
        import pyarrow.parquet as pq
        # Write an empty table with the right schema
        table = pa.table({"disparate_impact": pa.array([], type=pa.float64())})
        pq.write_table(table, str(tmp_path / "fa.parquet"))
        src = build_fairness_archive_source(str(tmp_path / "fa.parquet"))
        assert src("v1") == {}

    def test_non_numeric_di_filtered(self, tmp_path):
        # Pyarrow cannot mix types in a column; store as strings and rely
        # on the source's float() coercion + filter. This also exercises
        # the "unparsable" branch explicitly.
        _write_fairness_parquet(tmp_path / "fa.parquet", [
            {"disparate_impact": "bad-value"},
            {"disparate_impact": ""},
            {"disparate_impact": "0.85"},
        ])
        src = build_fairness_archive_source(str(tmp_path / "fa.parquet"))
        assert src("v1") == {"disparate_impact_min": pytest.approx(0.85)}


# ---------------------------------------------------------------------------
# High-level builder with archive paths
# ---------------------------------------------------------------------------

class TestBuilderWithArchivePaths:
    def test_archive_paths_populate_aggregator(self, tmp_path):
        # Lineage YAML: 1/4 non-pseudonymized
        lineage_p = tmp_path / "fg.yaml"
        _write_lineage_yaml(lineage_p, {
            "a_": {"pseudonymized": True},
            "b_": {"pseudonymized": True},
            "c_": {"pseudonymized": True},
            "d_": {"pseudonymized": False},
        })
        # Fairness parquet: worst DI 0.85
        _write_fairness_parquet(tmp_path / "fa.parquet", [
            {"disparate_impact": 0.95},
            {"disparate_impact": 0.85},
            {"disparate_impact": 1.0},
        ])
        cfg = {"llm_provider": {"backend": "dummy"}}
        agg = build_metadata_aggregator_from_config(
            cfg,
            lineage_yaml_path=str(lineage_p),
            fairness_archive_path=str(tmp_path / "fa.parquet"),
        )
        out = agg("v1")
        assert out["pii_ratio"] == pytest.approx(0.25)
        assert out["disparate_impact_min"] == pytest.approx(0.85)

    def test_static_overrides_still_win(self, tmp_path):
        lineage_p = tmp_path / "fg.yaml"
        _write_lineage_yaml(lineage_p, {
            "a_": {"pseudonymized": False},
        })
        cfg = {"llm_provider": {"backend": "dummy"}}
        agg = build_metadata_aggregator_from_config(
            cfg,
            lineage_yaml_path=str(lineage_p),
            static_overrides={"pii_ratio": 0.01},
        )
        # Static override wins (later source)
        assert agg("v1")["pii_ratio"] == 0.01

    def test_runtime_and_archive_compose(self, tmp_path):
        """Runtime tracker empty, archive YAML populated → archive wins."""
        empty_tracker = SimpleNamespace(feature_source_map={})
        lineage_p = tmp_path / "fg.yaml"
        _write_lineage_yaml(lineage_p, {
            "a_": {"pseudonymized": False},
            "b_": {"pseudonymized": True},
        })
        cfg = {"llm_provider": {"backend": "dummy"}}
        agg = build_metadata_aggregator_from_config(
            cfg,
            lineage_tracker=empty_tracker,
            lineage_yaml_path=str(lineage_p),
        )
        # Runtime returns {} (empty map), archive source fills in 0.5
        assert agg("v1")["pii_ratio"] == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# submit_pipeline integration
# ---------------------------------------------------------------------------

class TestSubmitPipelineArchiveWiring:
    def test_pipeline_config_feeds_archive_paths(self, tmp_path, monkeypatch):
        from scripts import submit_pipeline

        lineage_p = tmp_path / "fg.yaml"
        _write_lineage_yaml(lineage_p, {
            "a_": {"pseudonymized": False},
            "b_": {"pseudonymized": True},
        })
        fairness_p = tmp_path / "fa.parquet"
        _write_fairness_parquet(fairness_p, [
            {"disparate_impact": 0.8},
        ])

        pipeline_cfg = {
            "llm_provider": {"backend": "dummy"},
            "compliance": {
                "promotion_gate": {
                    "enabled": True,
                    "providers": {
                        "aggregator": {
                            "sources": {
                                "lineage_yaml_path": str(lineage_p),
                                "fairness_archive_parquet_path": str(fairness_p),
                            },
                        },
                    },
                },
            },
        }

        # Block live FairnessMonitor / DataLineageTracker import so the
        # archive sources are the only signal, but allow submit_pipeline
        # import paths to still work.
        monkeypatch.setattr(
            "core.monitoring.lineage_tracker.DataLineageTracker",
            lambda *a, **kw: SimpleNamespace(feature_source_map={}),
        )
        monkeypatch.setattr(
            "core.monitoring.fairness_monitor.FairnessMonitor",
            lambda *a, **kw: SimpleNamespace(get_archive=lambda **_: []),
        )

        agg = submit_pipeline._build_metadata_aggregator(pipeline_cfg)
        out = agg("v1")
        assert out["pii_ratio"] == pytest.approx(0.5)
        assert out["disparate_impact_min"] == pytest.approx(0.8)

    def test_fairness_fallback_to_monitoring_block(self, tmp_path, monkeypatch):
        """fairness_archive_parquet_path null → fall back to monitoring."""
        from scripts import submit_pipeline

        fairness_p = tmp_path / "fa.parquet"
        _write_fairness_parquet(fairness_p, [
            {"disparate_impact": 0.75},
        ])
        pipeline_cfg = {
            "llm_provider": {"backend": "dummy"},
            "monitoring": {
                "fairness": {
                    "archive_parquet_path": str(fairness_p),
                },
            },
            "compliance": {
                "promotion_gate": {
                    "enabled": True,
                    "providers": {
                        "aggregator": {
                            "sources": {
                                "fairness_archive_parquet_path": None,
                            },
                        },
                    },
                },
            },
        }

        monkeypatch.setattr(
            "core.monitoring.lineage_tracker.DataLineageTracker",
            lambda *a, **kw: SimpleNamespace(feature_source_map={}),
        )
        monkeypatch.setattr(
            "core.monitoring.fairness_monitor.FairnessMonitor",
            lambda *a, **kw: SimpleNamespace(get_archive=lambda **_: []),
        )

        agg = submit_pipeline._build_metadata_aggregator(pipeline_cfg)
        out = agg("v1")
        assert out["disparate_impact_min"] == pytest.approx(0.75)
