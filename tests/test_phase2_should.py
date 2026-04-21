"""
Phase 2 Should tests (S1, S7, S8, S9, S10, S11, S13, S14, S15).

Run: pytest tests/test_phase2_should.py -v
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import numpy as np
import pytest
import yaml


# ---------------------------------------------------------------------------
# S1 - FallbackRouter Layer 4 (Human Fallback)
# ---------------------------------------------------------------------------

class TestS1HumanFallback:
    def _cfg(self, *, human_fallback: bool) -> Dict[str, Any]:
        return {
            "distillation": {
                "teacher_threshold": {
                    "binary_min_auc": 0.60,
                },
                "fidelity": {},
            },
            "rule_engine": {"enabled": False},
            "serving": {"review": {"tier_3_human_fallback": human_fallback}},
            "tasks": [{"name": "task_a", "type": "binary"}],
        }

    def test_layer4_when_enabled(self):
        from core.recommendation.fallback_router import FallbackRouter

        router = FallbackRouter(self._cfg(human_fallback=True))
        layer = router.route(task_name="task_a")
        assert layer == 4

    def test_layer3_when_human_fallback_disabled(self):
        from core.recommendation.fallback_router import FallbackRouter

        router = FallbackRouter(self._cfg(human_fallback=False))
        layer = router.route(task_name="task_a")
        assert layer == 3

    def test_route_all_counts_layer4(self):
        from core.recommendation.fallback_router import FallbackRouter

        router = FallbackRouter(self._cfg(human_fallback=True))
        routing = router.route_all(task_names=["task_a"])
        assert routing["task_a"] == 4


# ---------------------------------------------------------------------------
# S15 - auto_promote=False
# ---------------------------------------------------------------------------

class TestS15AutoPromote:
    def test_default_config_still_auto_promotes(self):
        from core.evaluation.model_competition import (
            CompetitionConfig, ModelCandidate, ModelCompetition,
        )
        comp = ModelCompetition(CompetitionConfig())
        champ = ModelCandidate(
            model_id="champ", model_uri="", model_type="ple_teacher",
            version="v1", trained_at="2026-04-01",
            metrics={"avg_auc": 0.80, "avg_f1": 0.70, "avg_mae": 0.20},
        )
        chal = ModelCandidate(
            model_id="chal", model_uri="", model_type="ple_teacher",
            version="v2", trained_at="2026-04-21",
            metrics={"avg_auc": 0.82, "avg_f1": 0.71, "avg_mae": 0.19},
        )
        result = comp.evaluate(champ, chal)
        assert result.promotion_approved is True

    def test_explicit_auto_promote_false_blocks(self):
        from core.evaluation.model_competition import (
            CompetitionConfig, ModelCandidate, ModelCompetition,
        )
        comp = ModelCompetition(CompetitionConfig(auto_promote=False))
        champ = ModelCandidate(
            model_id="champ", model_uri="", model_type="ple_teacher",
            version="v1", trained_at="2026-04-01",
            metrics={"avg_auc": 0.80, "avg_f1": 0.70, "avg_mae": 0.20},
        )
        chal = ModelCandidate(
            model_id="chal", model_uri="", model_type="ple_teacher",
            version="v2", trained_at="2026-04-21",
            metrics={"avg_auc": 0.82, "avg_f1": 0.71, "avg_mae": 0.19},
        )
        result = comp.evaluate(champ, chal)
        assert result.promotion_approved is False
        assert "auto_promote=False" in result.decision_reason

    def test_pipeline_yaml_sets_auto_promote_false(self):
        cfg = yaml.safe_load(
            Path("configs/pipeline.yaml").read_text(encoding="utf-8")
        )
        from core.evaluation.model_competition import CompetitionConfig
        comp_cfg = CompetitionConfig.from_dict(
            (cfg.get("serving") or {}).get("competition")
        )
        assert comp_cfg.auto_promote is False


# ---------------------------------------------------------------------------
# S10 - EU AI Act Annex IV
# ---------------------------------------------------------------------------

class TestS10AnnexIV:
    def test_12_sections_in_catalog(self):
        from core.compliance.annex_iv_mapper import ANNEX_IV_SECTIONS
        assert len(ANNEX_IV_SECTIONS) == 12
        ids = {s.section_id for s in ANNEX_IV_SECTIONS}
        assert ids == {str(i) for i in range(1, 13)}

    def test_generate_report_runs(self):
        from core.compliance.annex_iv_mapper import AnnexIVMapper
        mapper = AnnexIVMapper(project_root=Path("."))
        report = mapper.generate_report()
        assert report.total_sections == 12
        assert 0.0 <= report.coverage_rate <= 1.0
        assert isinstance(report.gaps, list)

    def test_module_evidence_is_resolved(self):
        from core.compliance.annex_iv_mapper import AnnexIVMapper
        mapper = AnnexIVMapper()
        # Section 11 references HumanReviewQueue module which we know exists.
        report = mapper.generate_report()
        sec11 = next(c for c in report.checks if c.section_id == "11")
        module_checks = [
            e for e in sec11.resolved_evidence if e["type"] == "module"
        ]
        assert any(e["ok"] for e in module_checks)


# ---------------------------------------------------------------------------
# S11 - L2a Safety Gate
# ---------------------------------------------------------------------------

class TestS11SafetyGate:
    def test_pass_on_clean_text(self):
        from core.recommendation.reason.l2a_safety_gate import L2aSafetyGate

        gate = L2aSafetyGate()
        verdict = gate.validate("이 상품을 추천합니다.")
        assert verdict.passed is True
        assert verdict.layer == "all"

    def test_parse_fails_on_too_short(self):
        from core.recommendation.reason.l2a_safety_gate import (
            L2aSafetyGate, ParseConfig, SafetyGateConfig,
        )
        gate = L2aSafetyGate(
            SafetyGateConfig(parse=ParseConfig(min_length=10))
        )
        verdict = gate.validate("short")
        assert verdict.passed is False
        assert verdict.layer == "parse"
        assert verdict.findings[0]["issue"] == "too_short"

    def test_rules_block_banned_phrase(self):
        from core.recommendation.reason.l2a_safety_gate import (
            L2aSafetyGate, RuleConfig, SafetyGateConfig,
        )
        gate = L2aSafetyGate(
            SafetyGateConfig(
                rules=RuleConfig(banned_phrases=("guaranteed",))
            )
        )
        verdict = gate.validate("This is guaranteed to succeed.")
        assert verdict.passed is False
        assert verdict.layer == "rules"
        assert verdict.findings[0]["issue"] == "banned_phrase"

    def test_pii_pattern_is_detected(self):
        from core.recommendation.reason.l2a_safety_gate import (
            L2aSafetyGate, RuleConfig, SafetyGateConfig,
        )
        gate = L2aSafetyGate(
            SafetyGateConfig(
                rules=RuleConfig(
                    pii_patterns=(r"\b\d{6}-\d{7}\b",),
                ),
            )
        )
        verdict = gate.validate("주민번호 900101-1234567 노출 예시")
        assert verdict.passed is False
        assert verdict.findings[0]["issue"] == "pii_leak"

    def test_quality_checker_is_consulted(self):
        from core.recommendation.reason.l2a_safety_gate import (
            L2aSafetyGate, QualityConfig, SafetyGateConfig,
        )

        class _Checker:
            def check(self, text):
                return {"score": 0.1}

        gate = L2aSafetyGate(
            SafetyGateConfig(
                quality=QualityConfig(enabled=True, min_score=0.5),
            ),
            quality_checker=_Checker(),
        )
        verdict = gate.validate("this is long enough")
        assert verdict.passed is False
        assert verdict.layer == "quality"


# ---------------------------------------------------------------------------
# S13 - Suitability filter
# ---------------------------------------------------------------------------

class TestS13Suitability:
    def _make(self, **overrides):
        from core.recommendation.constraint_engine import SuitabilityFilter
        cfg = {
            "require_assessment": True,
            "senior_age_threshold": 65,
            "senior_max_risk_level": 2,
            "low_income_threshold": 30_000_000,
            "low_income_max_risk_level": 3,
        }
        cfg.update(overrides)
        return SuitabilityFilter(cfg)

    def test_match_passes(self):
        f = self._make()
        result = f.evaluate(
            "C001", "PROD_A",
            {"customer_risk_tolerance": 4, "item_risk_level": 3},
        )
        assert result.passed is True

    def test_risk_exceeds_tolerance_fails(self):
        f = self._make()
        result = f.evaluate(
            "C001", "PROD_A",
            {"customer_risk_tolerance": 2, "item_risk_level": 4},
        )
        assert result.passed is False
        assert "§17" in result.reason

    def test_require_assessment_blocks_missing_tolerance(self):
        f = self._make()
        result = f.evaluate(
            "C001", "PROD_A", {"item_risk_level": 2},
        )
        assert result.passed is False

    def test_senior_cap_blocks_risky(self):
        f = self._make()
        result = f.evaluate(
            "C001", "PROD_A",
            {
                "customer_risk_tolerance": 5,
                "item_risk_level": 3,
                "customer_age": 70,
            },
        )
        assert result.passed is False
        assert "senior cap" in result.reason

    def test_low_income_cap_blocks_risky(self):
        f = self._make()
        result = f.evaluate(
            "C001", "PROD_A",
            {
                "customer_risk_tolerance": 5,
                "item_risk_level": 4,
                "customer_income": 20_000_000,
            },
        )
        assert result.passed is False
        assert "low-income" in result.reason

    def test_no_item_risk_level_is_passthrough(self):
        f = self._make()
        result = f.evaluate(
            "C001", "PROD_A", {"customer_risk_tolerance": 1},
        )
        assert result.passed is True


# ---------------------------------------------------------------------------
# S7 - Fairness metrics archive
# ---------------------------------------------------------------------------

class TestS7FairnessArchive:
    def test_archive_in_memory(self):
        from core.monitoring.fairness_monitor import FairnessMonitor
        fm = FairnessMonitor()
        fm.archive_metrics(
            {"attribute": "gender", "di": 0.95}, context={"batch": "b1"},
        )
        fm.archive_metrics(
            {"attribute": "gender", "di": 0.92}, context={"batch": "b2"},
        )
        fm.archive_metrics(
            {"attribute": "age_bracket", "di": 0.88},
        )
        gender_hist = fm.get_archive(attribute="gender")
        assert len(gender_hist) == 2
        assert gender_hist[0]["di"] == 0.95

    def test_archive_with_limit(self):
        from core.monitoring.fairness_monitor import FairnessMonitor
        fm = FairnessMonitor()
        for i in range(5):
            fm.archive_metrics({"attribute": "gender", "di": 0.9 + i * 0.01})
        latest = fm.get_archive(attribute="gender", limit=2)
        assert len(latest) == 2
        assert latest[-1]["di"] == pytest.approx(0.94)

    def test_archive_parquet_roundtrip(self, tmp_path):
        pq = pytest.importorskip("pyarrow.parquet")
        from core.monitoring.fairness_monitor import FairnessMonitor
        path = tmp_path / "fairness_archive.parquet"
        fm = FairnessMonitor()
        fm.archive_metrics(
            {"attribute": "gender", "di": 0.93},
            parquet_path=str(path),
        )
        assert path.exists()
        table = pq.read_table(str(path))
        df = table.to_pylist()
        assert df[0]["di"] == 0.93
        assert df[0]["attribute"] == "gender"


# ---------------------------------------------------------------------------
# S8 - Drift persist + markdown
# ---------------------------------------------------------------------------

class TestS8DriftPersist:
    def _detector(self):
        from core.monitoring.drift_detector import DriftDetector
        return DriftDetector(
            psi_threshold_warning=0.1, psi_threshold_critical=0.25,
        )

    def _result(self):
        return {
            "psi_scores": {"feat_a": 0.05, "feat_b": 0.30, "feat_c": 0.15},
            "warning_features": ["feat_c"],
            "critical_features": ["feat_b"],
            "summary": {
                "total_features": 3, "warning_count": 1,
                "critical_count": 1, "max_psi": 0.30, "avg_psi": 0.167,
                "drift_detected": True,
            },
        }

    def test_archive_writes_parquet(self, tmp_path):
        pq = pytest.importorskip("pyarrow.parquet")
        d = self._detector()
        path = tmp_path / "drift.parquet"
        count = d.archive_result(self._result(), str(path))
        assert count == 3
        rows = pq.read_table(str(path)).to_pylist()
        by_feat = {r["feature"]: r for r in rows}
        assert by_feat["feat_b"]["severity"] == "critical"
        assert by_feat["feat_c"]["severity"] == "warning"
        assert by_feat["feat_a"]["severity"] == "ok"

    def test_archive_appends_on_rerun(self, tmp_path):
        pq = pytest.importorskip("pyarrow.parquet")
        d = self._detector()
        path = tmp_path / "drift.parquet"
        d.archive_result(self._result(), str(path))
        d.archive_result(self._result(), str(path))
        rows = pq.read_table(str(path)).to_pylist()
        assert len(rows) == 6

    def test_markdown_report_includes_feature_list(self):
        d = self._detector()
        report = d.generate_markdown_report(self._result(), title="T1")
        assert "# T1" in report
        assert "feat_b" in report
        assert "Critical features: 1" in report


# ---------------------------------------------------------------------------
# S9 - Lineage extension
# ---------------------------------------------------------------------------

class TestS9Lineage:
    def test_register_feature_mapping(self):
        from core.monitoring.lineage_tracker import DataLineageTracker
        t = DataLineageTracker(feature_source_map={})
        t.register_feature_mapping(
            prefix="txn_count_",
            source_tables=["T_TXN"],
            source_columns=["txn_id"],
            data_group="G2_transactions",
            description="test",
            pseudonymized=True,
        )
        trace = t.trace_feature_to_source("txn_count_monthly")
        assert trace["source_tables"] == ["T_TXN"]
        assert trace["pseudonymized"] is True

    def test_coverage_report(self):
        from core.monitoring.lineage_tracker import DataLineageTracker
        t = DataLineageTracker(feature_source_map={})
        t.register_feature_mapping(
            prefix="spend_", source_tables=["T_TXN"],
            source_columns=["amt"], data_group="G2",
            description="spend",
        )
        features = ["spend_total", "spend_mcc_5411", "unknown_feat"]
        cov = t.coverage_report(features)
        assert cov["total_features"] == 3
        assert cov["mapped"] == 2
        assert cov["unmapped"] == ["unknown_feat"]
        assert cov["by_data_group"]["G2"] == 2

    def test_load_mapping_from_yaml(self, tmp_path):
        from core.monitoring.lineage_tracker import DataLineageTracker
        yaml_text = """
        feature_source_map:
          spend_:
            source_tables: [T_TXN, T_MCC]
            source_columns: [amount, mcc_code]
            data_group: G2_transactions
            description: "Spending patterns"
            pseudonymized: true
          life_stage_:
            source_tables: [T_CUSTOMER]
            source_columns: [birth_year]
            data_group: G1_profile
            description: "Life stage"
        """
        p = tmp_path / "lineage.yaml"
        p.write_text(yaml_text, encoding="utf-8")
        t = DataLineageTracker(feature_source_map={})
        n = t.load_mapping_from_yaml(str(p))
        assert n == 2
        assert t.trace_feature_to_source("spend_mcc")["source_tables"] == [
            "T_TXN", "T_MCC",
        ]


# ---------------------------------------------------------------------------
# S14 - Counterfactual C-C
# ---------------------------------------------------------------------------

class TestS14Counterfactual:
    def test_ips_identity(self):
        from core.evaluation.counterfactual_cc import ips_estimate
        rewards = [1.0, 0.0, 1.0, 0.0]
        p_log = [0.5, 0.5, 0.5, 0.5]
        # New policy == logged: IPS should recover logged mean
        val = ips_estimate(rewards, p_log, p_log)
        assert val == pytest.approx(0.5, abs=1e-9)

    def test_snips_identity(self):
        from core.evaluation.counterfactual_cc import snips_estimate
        rewards = [1.0, 0.0, 1.0, 0.0]
        p_log = [0.5, 0.5, 0.5, 0.5]
        val = snips_estimate(rewards, p_log, p_log)
        assert val == pytest.approx(0.5, abs=1e-9)

    def test_ips_upweights_target_policy(self):
        from core.evaluation.counterfactual_cc import ips_estimate
        # Rewards=1 only when first and third samples. If new policy puts
        # all its mass on those samples, IPS should be higher than baseline.
        rewards = [1.0, 0.0, 1.0, 0.0]
        p_log = [0.5, 0.5, 0.5, 0.5]
        p_new = [0.9, 0.1, 0.9, 0.1]
        val = ips_estimate(rewards, p_log, p_new)
        assert val > 0.5

    def test_rejects_zero_logged_propensity(self):
        from core.evaluation.counterfactual_cc import ips_estimate
        with pytest.raises(ValueError):
            ips_estimate([1.0], [0.0], [0.5])

    def test_evaluator_compare_returns_result(self):
        from core.evaluation.counterfactual_cc import CounterfactualEvaluator
        ev = CounterfactualEvaluator(
            estimator="snips", n_bootstrap=50, alpha=0.1,
        )
        r = [1, 0, 1, 0, 1, 0, 1, 0] * 10
        p_log = [0.5] * len(r)
        p_chal = [0.8, 0.2] * (len(r) // 2)
        result = ev.compare(r, p_log, p_chal)
        assert result.sample_size == len(r)
        assert "CI[" in result.reason

    def test_evaluator_from_config(self):
        from core.evaluation.counterfactual_cc import CounterfactualEvaluator
        ev = CounterfactualEvaluator.from_config({
            "serving": {
                "counterfactual_cc": {
                    "estimator": "ips",
                    "min_lift": 0.05,
                    "n_bootstrap": 200,
                }
            }
        })
        assert ev.estimator == "ips"
        assert ev.min_lift == 0.05
        assert ev.n_bootstrap == 200


# ---------------------------------------------------------------------------
# Smoke: combined imports
# ---------------------------------------------------------------------------

class TestCombinedImports:
    def test_all_phase2_symbols_importable(self):
        from core.compliance.annex_iv_mapper import AnnexIVMapper  # noqa
        from core.evaluation.counterfactual_cc import CounterfactualEvaluator  # noqa
        from core.evaluation.model_competition import CompetitionConfig  # noqa
        from core.monitoring.drift_detector import DriftDetector  # noqa
        from core.monitoring.fairness_monitor import FairnessMonitor  # noqa
        from core.monitoring.lineage_tracker import DataLineageTracker  # noqa
        from core.recommendation.constraint_engine import SuitabilityFilter  # noqa
        from core.recommendation.fallback_router import FallbackRouter  # noqa
        from core.recommendation.reason.l2a_safety_gate import L2aSafetyGate  # noqa
