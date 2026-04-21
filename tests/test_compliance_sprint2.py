"""
Sprint 2 tests: M7 KoreanFRIAAssessor, M8 ComplianceRegistry,
M9 AIRiskClassifier, and PromotionGate integration.

Run: pytest tests/test_compliance_sprint2.py -v
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest
import yaml

from core.compliance.ai_risk_classifier import (
    AIRiskClassifier,
    AIRiskConfig,
    build_ai_risk_classifier,
)
from core.compliance.compliance_registry import (
    ComplianceItem,
    ComplianceRegistry,
    DEFAULT_CATALOG,
    load_registry_from_pipeline_yaml,
)
from core.compliance.fria_assessment import (
    FRIA_DIMENSIONS,
    FRIAConfig,
    KoreanFRIAAssessor,
    build_fria_assessor,
)
from core.compliance.store import InMemoryComplianceStore
from core.compliance.types import EventType
from core.evaluation.promotion_gate import (
    PromotionGate,
    build_promotion_gate,
)


def _fixed(ts: str) -> datetime:
    dt = datetime.fromisoformat(ts)
    return dt if dt.tzinfo is not None else dt.replace(tzinfo=timezone.utc)


@pytest.fixture
def store():
    return InMemoryComplianceStore()


# ---------------------------------------------------------------------------
# M7 KoreanFRIAAssessor
# ---------------------------------------------------------------------------

class TestFRIAConfig:
    def test_defaults(self):
        cfg = FRIAConfig()
        assert cfg.operator_type == "국가기관등"
        assert cfg.retention_days == 1825
        assert set(cfg.dimensions) == set(FRIA_DIMENSIONS)

    def test_rejects_unknown_operator(self):
        with pytest.raises(ValueError):
            FRIAConfig(operator_type="외계인")

    def test_from_dict_pipeline_yaml(self):
        cfg_yaml = yaml.safe_load(
            Path("configs/pipeline.yaml").read_text(encoding="utf-8")
        )
        fria_cfg = FRIAConfig.from_dict(cfg_yaml["compliance"]["fria"])
        assert fria_cfg.retention_days == 1825
        assert set(fria_cfg.dimensions) == set(FRIA_DIMENSIONS)


class TestFRIAAssessor:
    def _default_scores(self, value: float = 0.5):
        return {d: value for d in FRIA_DIMENSIONS}

    def test_evaluate_returns_result(self, store):
        assessor = KoreanFRIAAssessor(store=store)
        r = assessor.evaluate(
            model_version="v1", dimension_scores=self._default_scores(0.5),
        )
        assert r.model_version == "v1"
        assert r.risk_category == "LIMITED"
        assert 0.49 <= r.total_score <= 0.51

    def test_high_risk_blocks_promotion(self, store):
        assessor = KoreanFRIAAssessor(store=store)
        r = assessor.evaluate(
            model_version="v2", dimension_scores=self._default_scores(0.95),
        )
        assert r.risk_category == "UNACCEPTABLE"
        assert r.blocks_promotion() is True

    def test_minimal_risk(self, store):
        assessor = KoreanFRIAAssessor(store=store)
        r = assessor.evaluate(
            model_version="v3", dimension_scores=self._default_scores(0.1),
        )
        assert r.risk_category == "MINIMAL"
        assert r.blocks_promotion() is False

    def test_rejects_missing_dimensions(self, store):
        assessor = KoreanFRIAAssessor(store=store)
        with pytest.raises(ValueError, match="missing keys"):
            assessor.evaluate(
                model_version="v1",
                dimension_scores={"data_sensitivity": 0.5},
            )

    def test_rejects_out_of_range(self, store):
        assessor = KoreanFRIAAssessor(store=store)
        with pytest.raises(ValueError, match="outside"):
            scores = self._default_scores(0.5)
            scores["data_sensitivity"] = 1.5
            assessor.evaluate(model_version="v1", dimension_scores=scores)

    def test_persistence_and_retrieval(self, store):
        assessor = KoreanFRIAAssessor(store=store)
        assessor.evaluate(
            model_version="v1", dimension_scores=self._default_scores(0.4),
        )
        assessor.evaluate(
            model_version="v1", dimension_scores=self._default_scores(0.8),
        )
        history = assessor.list_assessments(model_version="v1")
        assert len(history) == 2
        latest = assessor.latest_for_model("v1")
        assert latest.risk_category == "HIGH"

    def test_retention_expiry_plus_5_years(self, store):
        assessor = KoreanFRIAAssessor(store=store)
        now = _fixed("2026-04-21T00:00:00+00:00")
        r = assessor.evaluate(
            model_version="v1", dimension_scores=self._default_scores(0.3),
            assessed_at=now,
        )
        expected = now + timedelta(days=1825)
        assert r.retention_expiry == expected

    def test_list_expiring(self, store):
        cfg = FRIAConfig(retention_days=10, warning_before_expiry_days=3)
        assessor = KoreanFRIAAssessor(store=store, config=cfg)
        old = _fixed("2026-04-01T00:00:00+00:00")
        # retention_expiry = old + 10 days = 2026-04-11
        assessor.evaluate(
            model_version="v1",
            dimension_scores={d: 0.5 for d in FRIA_DIMENSIONS},
            assessed_at=old,
        )
        # As of 2026-04-09, only 2 days left → within 3-day warning
        expiring = assessor.list_expiring(
            within_days=3,
            now=_fixed("2026-04-09T00:00:00+00:00"),
        )
        assert len(expiring) == 1


# ---------------------------------------------------------------------------
# M9 AIRiskClassifier
# ---------------------------------------------------------------------------

class TestAIRiskConfig:
    def test_defaults_sum_to_one(self):
        cfg = AIRiskConfig()
        total = sum(cfg.dimensions.values())
        assert abs(total - 1.0) < 1e-6

    def test_rejects_nonsum_weights(self):
        with pytest.raises(ValueError):
            AIRiskConfig(dimensions={"data_sensitivity": 0.5})

    def test_from_pipeline_yaml(self):
        cfg_yaml = yaml.safe_load(
            Path("configs/pipeline.yaml").read_text(encoding="utf-8")
        )
        cfg = AIRiskConfig.from_dict(cfg_yaml["compliance"]["ai_risk"])
        assert cfg.grade_thresholds["high"] == 0.70
        assert cfg.grade_thresholds["medium"] == 0.40


class TestAIRiskClassifier:
    def _scores(self, value=0.5):
        return {
            "data_sensitivity": value,
            "automation_level": value,
            "scope_of_impact": value,
            "model_complexity": value,
            "external_dependency": value,
            "fairness_risk": value,
        }

    def test_classify_low(self, store):
        clf = AIRiskClassifier(store=store)
        a = clf.classify(model_version="v1", dimension_scores=self._scores(0.2))
        assert a.grade == "low"
        assert a.prev_grade is None
        assert a.grade_change is False

    def test_classify_medium(self, store):
        clf = AIRiskClassifier(store=store)
        a = clf.classify(model_version="v1", dimension_scores=self._scores(0.5))
        assert a.grade == "medium"

    def test_classify_high(self, store):
        clf = AIRiskClassifier(store=store)
        a = clf.classify(model_version="v1", dimension_scores=self._scores(0.8))
        assert a.grade == "high"

    def test_rejects_missing_dim(self, store):
        clf = AIRiskClassifier(store=store)
        with pytest.raises(ValueError, match="missing keys"):
            clf.classify(
                model_version="v1",
                dimension_scores={"data_sensitivity": 0.5},
            )

    def test_grade_change_detected(self, store):
        clf = AIRiskClassifier(store=store)
        clf.classify(model_version="v1", dimension_scores=self._scores(0.2))
        a2 = clf.classify(
            model_version="v1", dimension_scores=self._scores(0.8),
        )
        assert a2.prev_grade == "low"
        assert a2.grade == "high"
        assert a2.grade_change is True
        assert a2.escalated() is True
        assert a2.requires_additional_approval() is True

    def test_grade_downgrade_no_extra_approval(self, store):
        clf = AIRiskClassifier(store=store)
        clf.classify(model_version="v1", dimension_scores=self._scores(0.8))
        a2 = clf.classify(
            model_version="v1", dimension_scores=self._scores(0.2),
        )
        assert a2.escalated() is False
        assert a2.requires_additional_approval() is False

    def test_factory(self, store):
        cfg_yaml = yaml.safe_load(
            Path("configs/pipeline.yaml").read_text(encoding="utf-8")
        )
        clf = build_ai_risk_classifier(
            store=store, config=cfg_yaml["compliance"]["ai_risk"],
        )
        assert isinstance(clf, AIRiskClassifier)


# ---------------------------------------------------------------------------
# M8 ComplianceRegistry
# ---------------------------------------------------------------------------

class TestComplianceRegistryCatalog:
    def test_catalog_has_36_items(self):
        assert len(DEFAULT_CATALOG) == 36

    def test_catalog_split_18_18(self):
        a = [i for i in DEFAULT_CATALOG if i.group == "A"]
        gap = [i for i in DEFAULT_CATALOG if i.group == "GAP"]
        assert len(a) == 18
        assert len(gap) == 18

    def test_item_ids_unique(self):
        ids = [i.item_id for i in DEFAULT_CATALOG]
        assert len(set(ids)) == len(ids)


class TestComplianceRegistryChecks:
    def test_check_module_exists(self):
        reg = ComplianceRegistry()
        result = reg.check_single("A-02")  # consent_manager module
        assert result.status == "compliant"

    def test_check_missing_module_is_non_compliant(self):
        reg = ComplianceRegistry(items=[
            ComplianceItem(
                item_id="X-99", group="GAP",
                description="fake",
                legal_basis=["test"],
                check_type="module_exists",
                check_params={"module": "core.does.not.exist"},
            ),
        ])
        r = reg.check_single("X-99")
        assert r.status == "non_compliant"

    def test_check_config_key_present(self):
        cfg = {"monitoring": {"governance_report": {"enabled": True}}}
        reg = ComplianceRegistry(items=[
            ComplianceItem(
                item_id="C-01", group="A",
                description="governance flag",
                legal_basis=["test"],
                check_type="config_key",
                check_params={"path": "monitoring.governance_report.enabled"},
            ),
        ], config=cfg)
        assert reg.check_single("C-01").status == "compliant"

    def test_check_config_key_missing(self):
        reg = ComplianceRegistry(items=[
            ComplianceItem(
                item_id="C-02", group="A",
                description="missing flag",
                legal_basis=["test"],
                check_type="config_key",
                check_params={"path": "missing.key"},
            ),
        ], config={})
        assert reg.check_single("C-02").status == "non_compliant"

    def test_check_custom_check(self):
        reg = ComplianceRegistry(items=[
            ComplianceItem(
                item_id="CUST-01", group="GAP",
                description="custom", legal_basis=["test"],
                check_type="custom_check",
                check_params={"check_name": "always_pass"},
            ),
        ])
        from core.compliance.compliance_registry import ComplianceItemResult
        from core.compliance.types import utcnow

        def passer(item):
            return ComplianceItemResult(
                item_id=item.item_id, status="compliant",
                checked_at=utcnow(), message="ok",
            )

        reg.register_check("always_pass", passer)
        assert reg.check_single("CUST-01").status == "compliant"

    def test_unknown_check_type_is_not_applicable(self):
        reg = ComplianceRegistry(items=[
            ComplianceItem(
                item_id="N-01", group="GAP",
                description="n/a", legal_basis=["test"],
                check_type="bogus_type",
                check_params={},
            ),
        ])
        assert reg.check_single("N-01").status == "not_applicable"

    def test_summary_counts(self, store):
        reg = ComplianceRegistry(store=store, items=[
            ComplianceItem(
                item_id="A-01", group="A", description="ok",
                legal_basis=["t"],
                check_type="module_exists",
                check_params={"module": "core.compliance.types"},
                severity="high",
            ),
            ComplianceItem(
                item_id="GAP-01", group="GAP", description="missing",
                legal_basis=["t"],
                check_type="module_exists",
                check_params={"module": "core.does.not.exist"},
                severity="critical",
            ),
        ])
        results = reg.check_all()
        s = reg.summary(results)
        assert s["total"] == 2
        assert s["by_status"]["compliant"] == 1
        assert s["by_status"]["non_compliant"] == 1
        assert "GAP-01" in s["critical_failures"]

    def test_quarterly_report_format(self):
        reg = ComplianceRegistry(items=[DEFAULT_CATALOG[0]])
        report = reg.generate_quarterly_report("2026-Q2")
        assert "Compliance Registry Report 2026-Q2" in report
        assert "Total items: 1" in report


class TestComplianceRegistryLoader:
    def test_load_from_pipeline_yaml(self):
        reg = load_registry_from_pipeline_yaml()
        assert len(reg.items()) == 36


# ---------------------------------------------------------------------------
# Promotion gate
# ---------------------------------------------------------------------------

class TestPromotionGate:
    def test_disabled_gate_skips(self, store):
        gate = PromotionGate(
            fria_assessor=KoreanFRIAAssessor(store=store),
            ai_risk_classifier=AIRiskClassifier(store=store),
            enabled=False,
        )
        v = gate.evaluate(model_version="v1")
        assert v.decision == "skip"

    def test_pass_with_conservative_defaults(self, store):
        gate = PromotionGate(
            fria_assessor=KoreanFRIAAssessor(store=store),
            ai_risk_classifier=AIRiskClassifier(store=store),
            enabled=True,
        )
        v = gate.evaluate(model_version="v1")
        assert v.decision == "pass"
        assert v.fria.risk_category == "LIMITED"

    def test_reject_on_unacceptable_fria(self, store):
        def high(model_version):
            return {d: 0.95 for d in FRIA_DIMENSIONS}

        gate = PromotionGate(
            fria_assessor=KoreanFRIAAssessor(store=store),
            ai_risk_classifier=AIRiskClassifier(store=store),
            enabled=True,
            fria_scores_provider=high,
        )
        v = gate.evaluate(model_version="v_bad")
        assert v.decision == "reject"
        assert "UNACCEPTABLE" in v.reason
        assert v.blocks_promotion is True

    def test_require_approval_on_risk_escalation(self, store):
        low = lambda _: {
            "data_sensitivity": 0.2, "automation_level": 0.2,
            "scope_of_impact": 0.2, "model_complexity": 0.2,
            "external_dependency": 0.2, "fairness_risk": 0.2,
        }
        high = lambda _: {
            "data_sensitivity": 0.95, "automation_level": 0.95,
            "scope_of_impact": 0.95, "model_complexity": 0.95,
            "external_dependency": 0.95, "fairness_risk": 0.95,
        }
        # First pass with low AI risk, establishes prev_grade=low
        gate_low = PromotionGate(
            fria_assessor=KoreanFRIAAssessor(store=store),
            ai_risk_classifier=AIRiskClassifier(store=store),
            enabled=True,
            ai_risk_scores_provider=low,
        )
        v1 = gate_low.evaluate(model_version="v1")
        assert v1.decision == "pass"

        # Second pass with high AI risk → escalation
        gate_high = PromotionGate(
            fria_assessor=KoreanFRIAAssessor(store=store),
            ai_risk_classifier=AIRiskClassifier(store=store),
            enabled=True,
            ai_risk_scores_provider=high,
        )
        v2 = gate_high.evaluate(model_version="v1")
        assert v2.decision == "require_approval"
        assert v2.blocks_promotion is True

    def test_factory_from_pipeline_yaml(self):
        cfg_yaml = yaml.safe_load(
            Path("configs/pipeline.yaml").read_text(encoding="utf-8")
        )
        gate = build_promotion_gate(cfg_yaml)
        # default in repo is disabled → skip
        v = gate.evaluate(model_version="v1")
        assert v.decision == "skip"

    def test_factory_with_override(self):
        cfg = {
            "compliance": {
                "promotion_gate": {"enabled": True, "default_score": 0.3},
                "fria": {},
                "ai_risk": {},
                "store": {"backend": "in_memory"},
            }
        }
        gate = build_promotion_gate(cfg)
        v = gate.evaluate(model_version="v1")
        assert v.decision == "pass"
        assert v.fria.risk_category == "MINIMAL"


# ---------------------------------------------------------------------------
# Audit trail: FRIA + AI Risk leave events in store
# ---------------------------------------------------------------------------

class TestAuditTrail:
    def test_events_written_per_assessment(self, store):
        assessor = KoreanFRIAAssessor(store=store)
        assessor.evaluate(
            model_version="v1",
            dimension_scores={d: 0.5 for d in FRIA_DIMENSIONS},
        )
        clf = AIRiskClassifier(store=store)
        clf.classify(
            model_version="v1",
            dimension_scores={
                "data_sensitivity": 0.5, "automation_level": 0.5,
                "scope_of_impact": 0.5, "model_complexity": 0.5,
                "external_dependency": 0.5, "fairness_risk": 0.5,
            },
        )
        fria_events = store.query_events(
            event_type=EventType.FRIA_ASSESSMENT
        )
        ai_events = store.query_events(
            event_type=EventType.AI_RISK_ASSESSMENT
        )
        assert len(fria_events) == 1
        assert len(ai_events) == 1
