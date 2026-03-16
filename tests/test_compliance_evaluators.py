"""Tests for FRIA, PIA, EU AI Act mapper, and XAI quality evaluator modules."""

import numpy as np
import pytest

from core.monitoring.fria_evaluator import (
    FRIAEvaluator,
    FRIAReport,
    TaskRiskProfile,
    RISK_THRESHOLDS,
    ASSESSMENT_DIMENSIONS,
)
from core.monitoring.pia_evaluator import (
    PIAEvaluator,
    PIAReport,
    PIIInventoryItem,
    DomainRiskAssessment,
    PII_CATEGORIES,
)
from core.monitoring.eu_ai_act_mapper import (
    EUAIActMapper,
    EUAIActComplianceReport,
    ArticleComplianceItem,
    FINANCIAL_HIGH_RISK_INDICATORS,
)
from core.monitoring.xai_quality_evaluator import (
    XAIQualityEvaluator,
    XAIQualityReport,
    ExplanationQualityMetrics,
    ConsistencyResult,
)


# ===================================================================
# FRIAEvaluator tests
# ===================================================================

class TestFRIAEvaluator:
    """Tests for the Financial Risk Impact Assessment evaluator."""

    def setup_method(self):
        self.evaluator = FRIAEvaluator(
            system_name="test-system",
            assessor="test-assessor",
        )

    def test_assess_task_no_metrics(self):
        """Task assessment with no metrics defaults to conservative scores."""
        profile = self.evaluator.assess_task(task_name="cvr_prediction")
        assert isinstance(profile, TaskRiskProfile)
        assert profile.task_name == "cvr_prediction"
        assert profile.aggregate_score > 0.0
        assert profile.risk_level in ("UNACCEPTABLE", "HIGH", "LIMITED", "MINIMAL")
        assert len(profile.dimension_scores) == 5
        for dim in ASSESSMENT_DIMENSIONS:
            assert dim in profile.dimension_scores

    def test_assess_task_minimal_risk(self):
        """Task with good metrics should yield MINIMAL risk."""
        profile = self.evaluator.assess_task(
            task_name="low_risk_task",
            fairness_metrics={"violation_count": 0, "di_value": 1.0},
            transparency_metrics={
                "has_explanations": True,
                "explanation_coverage": 0.95,
                "model_card_exists": True,
            },
            accuracy_metrics={"auc": 0.95, "drift_detected": False, "performance_degradation": 0.0},
            security_metrics={
                "input_validation": True,
                "encryption_enabled": True,
                "access_control_level": "strict",
            },
            oversight_metrics={
                "human_review_rate": 1.0,
                "alert_configured": True,
                "kill_switch_enabled": True,
            },
        )
        assert profile.risk_level == "MINIMAL"
        assert profile.aggregate_score < 0.4
        assert len(profile.remediations) == 0

    def test_assess_task_high_risk(self):
        """Task with violations should yield HIGH or above risk."""
        profile = self.evaluator.assess_task(
            task_name="risky_task",
            fairness_metrics={"violation_count": 5, "di_value": 0.5, "spd_value": 0.3},
            transparency_metrics={"has_explanations": False, "explanation_coverage": 0.0},
            accuracy_metrics={"auc": 0.55, "drift_detected": True, "performance_degradation": 0.2},
            security_metrics={"input_validation": False, "encryption_enabled": False, "access_control_level": "none"},
            oversight_metrics={"human_review_rate": 0.0, "alert_configured": False, "kill_switch_enabled": False},
        )
        assert profile.risk_level in ("HIGH", "UNACCEPTABLE")
        assert profile.aggregate_score >= 0.7
        assert len(profile.remediations) > 0

    def test_generate_report_from_assessments(self):
        """Report generation from raw assessment dicts."""
        report = self.evaluator.generate_report(
            task_assessments=[
                {
                    "task_name": "task_a",
                    "fairness_metrics": {"violation_count": 0, "di_value": 1.0},
                },
                {
                    "task_name": "task_b",
                    "fairness_metrics": {"violation_count": 3, "di_value": 0.6},
                },
            ]
        )
        assert isinstance(report, FRIAReport)
        assert len(report.task_profiles) == 2
        assert report.overall_risk_level in ("UNACCEPTABLE", "HIGH", "LIMITED", "MINIMAL")
        assert report.overall_score >= 0.0
        assert "FRIA" in report.report_id
        assert len(report.regulatory_references) > 0

    def test_generate_report_empty(self):
        """Report with no tasks should still produce valid output."""
        report = self.evaluator.generate_report(task_assessments=[])
        assert report.overall_score == 0.0
        assert len(report.task_profiles) == 0

    def test_to_dict(self):
        """Serialization should produce a JSON-compatible dict."""
        report = self.evaluator.generate_report(
            task_assessments=[{"task_name": "test"}]
        )
        d = self.evaluator.to_dict(report)
        assert isinstance(d, dict)
        assert "report_id" in d
        assert "task_profiles" in d
        assert isinstance(d["task_profiles"], list)

    def test_risk_classification_thresholds(self):
        """Verify risk classification maps correctly."""
        assert self.evaluator._classify_risk(0.95) == "UNACCEPTABLE"
        assert self.evaluator._classify_risk(0.75) == "HIGH"
        assert self.evaluator._classify_risk(0.5) == "LIMITED"
        assert self.evaluator._classify_risk(0.2) == "MINIMAL"


# ===================================================================
# PIAEvaluator tests
# ===================================================================

class TestPIAEvaluator:
    """Tests for the Privacy Impact Assessment evaluator."""

    def setup_method(self):
        self.evaluator = PIAEvaluator(
            system_name="test-system",
            assessor="test-assessor",
            data_regions=["ap-northeast-2"],
        )

    def test_analyze_pii_inventory(self):
        """PII inventory analysis produces correct items."""
        fields = [
            {
                "field_name": "customer_name",
                "pii_category": "direct_identifier",
                "sensitivity_level": "high",
                "purpose": "account identification",
                "retention_days": 365,
                "encrypted": True,
                "anonymized": False,
            },
            {
                "field_name": "age_group",
                "pii_category": "quasi_identifier",
                "sensitivity_level": "low",
                "purpose": "segmentation",
                "retention_days": 90,
                "encrypted": False,
                "anonymized": True,
            },
        ]
        inventory = self.evaluator.analyze_pii_inventory(fields)
        assert len(inventory) == 2
        assert all(isinstance(item, PIIInventoryItem) for item in inventory)
        assert inventory[0].field_name == "customer_name"
        assert inventory[1].anonymized is True

    def test_assess_data_collection_compliant(self):
        """Compliant data collection should have low risk."""
        result = self.evaluator.assess_data_collection({
            "lawful_basis": "consent",
            "consent_obtained": True,
            "purpose_specified": True,
            "data_sources_documented": True,
            "collection_scope": "narrow",
        })
        assert isinstance(result, DomainRiskAssessment)
        assert result.domain == "data_collection"
        assert result.score == 0.0
        assert result.risk_level == "LOW"
        assert len(result.findings) == 0

    def test_assess_data_collection_non_compliant(self):
        """Non-compliant data collection should have high risk."""
        result = self.evaluator.assess_data_collection({})
        assert result.score > 0.5
        assert result.risk_level in ("HIGH", "CRITICAL")
        assert len(result.findings) > 0
        assert len(result.recommendations) > 0

    def test_assess_cross_border_single_jurisdiction(self):
        """Single jurisdiction should have low transfer risk."""
        result = self.evaluator.assess_cross_border_transfer()
        assert result.domain == "cross_border_transfer"
        assert result.score < 0.3
        assert "single jurisdiction" in result.findings[0].lower()

    def test_assess_cross_border_multi_jurisdiction(self):
        """Multi-jurisdiction should flag cross-border risks."""
        evaluator = PIAEvaluator(
            data_regions=["ap-northeast-2", "eu-west-1", "us-east-1"],
        )
        result = evaluator.assess_cross_border_transfer()
        assert result.score > 0.0
        assert any("cross-border" in f.lower() for f in result.findings)

    def test_data_minimization_score(self):
        """Minimization score should reflect anonymization and encryption."""
        inventory = [
            PIIInventoryItem("f1", "direct_identifier", "high", "id", 30, True, True),
            PIIInventoryItem("f2", "quasi_identifier", "low", "seg", 90, True, False),
        ]
        score = self.evaluator.compute_data_minimization_score(inventory)
        assert 0.0 <= score <= 1.0
        assert score > 0.0  # At least some protections in place

    def test_data_minimization_empty(self):
        """Empty inventory = perfect minimization."""
        assert self.evaluator.compute_data_minimization_score([]) == 1.0

    def test_consent_adequacy_full(self):
        """Fully adequate consent should pass."""
        result = PIAEvaluator.check_consent_adequacy({
            "explicit_consent": True,
            "granular_consent": True,
            "withdrawable": True,
            "records_maintained": True,
            "age_verification": True,
            "plain_language": True,
        })
        assert result["is_adequate"] is True
        assert result["score"] == 1.0
        assert len(result["gaps"]) == 0

    def test_consent_adequacy_none(self):
        """No consent mechanisms should fail."""
        result = PIAEvaluator.check_consent_adequacy({})
        assert result["is_adequate"] is False
        assert result["score"] == 0.0
        assert len(result["gaps"]) == 6

    def test_generate_report(self):
        """Full PIA report generation."""
        report = self.evaluator.generate_report(
            pii_fields=[
                {"field_name": "name", "pii_category": "direct_identifier",
                 "sensitivity_level": "high", "purpose": "id",
                 "retention_days": 365, "encrypted": True, "anonymized": False},
            ],
            collection_config={"lawful_basis": "consent", "consent_obtained": True},
            processing_config={"purpose_limitation": True, "processing_logged": True},
            storage_config={"encryption_at_rest": True, "encryption_in_transit": True},
            consent_config={"explicit_consent": True, "withdrawable": True},
        )
        assert isinstance(report, PIAReport)
        assert "PIA" in report.report_id
        assert len(report.pii_inventory) == 1
        assert len(report.domain_assessments) == 6  # 5 domains + minimization
        assert report.overall_score >= 0.0
        assert len(report.regulatory_references) > 0

    def test_to_dict(self):
        """Serialization should produce a JSON-compatible dict."""
        report = self.evaluator.generate_report()
        d = self.evaluator.to_dict(report)
        assert isinstance(d, dict)
        assert "report_id" in d
        assert "domain_assessments" in d
        assert "pii_inventory" in d


# ===================================================================
# EUAIActMapper tests
# ===================================================================

class TestEUAIActMapper:
    """Tests for the EU AI Act compliance mapper."""

    def setup_method(self):
        self.mapper = EUAIActMapper(
            system_name="test-system",
            system_purpose="financial product recommendation",
        )

    def test_classify_financial_service_default_high(self):
        """Financial services AI defaults to HIGH risk."""
        level, rationale = self.mapper.classify_risk()
        assert level == "HIGH"
        assert "financial" in rationale.lower()

    def test_classify_prohibited_practices(self):
        """Manipulative techniques should be UNACCEPTABLE."""
        level, _ = self.mapper.classify_risk({"manipulative_techniques": True})
        assert level == "UNACCEPTABLE"

    def test_classify_social_scoring(self):
        """Social scoring should be UNACCEPTABLE."""
        level, _ = self.mapper.classify_risk({"social_scoring": True})
        assert level == "UNACCEPTABLE"

    def test_classify_high_risk_use_case(self):
        """Known high-risk use case should classify as HIGH."""
        level, rationale = self.mapper.classify_risk({
            "use_cases": ["creditworthiness", "fraud_detection"],
        })
        assert level == "HIGH"
        assert "Annex III" in rationale

    def test_classify_minimal_risk(self):
        """Non-financial, non-autonomous system should be MINIMAL."""
        mapper = EUAIActMapper(domain="other")
        level, _ = mapper.classify_risk({"affects_individuals": False})
        assert level == "MINIMAL"

    def test_evaluate_compliance_no_evidence(self):
        """All items non-compliant when no evidence provided."""
        checklist = self.mapper.evaluate_compliance()
        assert len(checklist) > 0
        assert all(isinstance(item, ArticleComplianceItem) for item in checklist)
        assert all(not item.is_compliant for item in checklist)
        assert all(item.remediation != "" for item in checklist)

    def test_evaluate_compliance_with_evidence(self):
        """Items with evidence should be marked compliant."""
        evidence = {
            "Art.12(1)": {"is_compliant": True, "evidence": "Audit logger deployed."},
            "Art.14(1)": {"is_compliant": True, "evidence": "Human review workflow active."},
        }
        checklist = self.mapper.evaluate_compliance(evidence)
        compliant_ids = [item.article_id for item in checklist if item.is_compliant]
        assert "Art.12(1)" in compliant_ids
        assert "Art.14(1)" in compliant_ids

    def test_generate_report(self):
        """Full EU AI Act compliance report generation."""
        report = self.mapper.generate_report(
            evidence={
                "Art.12(1)": {"is_compliant": True, "evidence": "Audit logger."},
                "Art.10(5)": {"is_compliant": True, "evidence": "Fairness monitor."},
            },
        )
        assert isinstance(report, EUAIActComplianceReport)
        assert "EUAIA" in report.report_id
        assert report.risk_classification == "HIGH"
        assert report.compliant_count == 2
        assert report.non_compliant_count == len(report.article_checklist) - 2
        assert 0.0 <= report.compliance_rate <= 1.0
        assert len(report.remediations) > 0

    def test_remediations_priority_sorted(self):
        """Remediations should be sorted by priority."""
        report = self.mapper.generate_report()
        priorities = [r["priority"] for r in report.remediations]
        priority_order = {"critical": 0, "high": 1, "medium": 2, "low": 3}
        numeric = [priority_order[p] for p in priorities]
        assert numeric == sorted(numeric)

    def test_integrate_existing_compliance(self):
        """Integration with 36-item checker should map to articles."""
        checker_results = {
            "items": {
                "MON-01": {"status": "compliant", "description": "Audit logger"},
                "MON-03": {"status": "non_compliant", "description": "Fairness monitoring"},
                "MON-07": {"status": "compliant", "description": "Incident management"},
            }
        }
        mapping = self.mapper.integrate_existing_compliance(checker_results)
        assert "Art.12(1)" in mapping
        assert mapping["Art.12(1)"]["is_compliant"] is True
        assert "Art.10(5)" in mapping
        assert mapping["Art.10(5)"]["is_compliant"] is False
        assert "Art.62(1)" in mapping
        assert mapping["Art.62(1)"]["is_compliant"] is True

    def test_to_dict(self):
        """Serialization should produce a JSON-compatible dict."""
        report = self.mapper.generate_report()
        d = self.mapper.to_dict(report)
        assert isinstance(d, dict)
        assert "article_checklist" in d
        assert "remediations" in d


# ===================================================================
# XAIQualityEvaluator tests
# ===================================================================

class TestXAIQualityEvaluator:
    """Tests for the XAI quality evaluator."""

    def setup_method(self):
        self.evaluator = XAIQualityEvaluator(system_name="test-system")

    def test_evaluate_faithfulness_correlated(self):
        """Correlated changes should produce high faithfulness."""
        np.random.seed(42)
        n_samples, n_features = 50, 10
        attributions = np.random.rand(n_samples, n_features)
        mask = np.zeros((n_samples, n_features))
        mask[:, :3] = 1.0  # Perturb top 3 features

        original = np.random.rand(n_samples)
        # Perturbed predictions change proportional to masked attributions
        perturbation = (attributions * mask).sum(axis=1) * 0.5
        perturbed = original - perturbation

        score = self.evaluator.evaluate_faithfulness(
            original, perturbed, attributions, mask,
        )
        assert 0.0 <= score <= 1.0
        assert score > 0.5  # Should be reasonably high given correlation

    def test_evaluate_faithfulness_empty(self):
        """Empty input should return 0."""
        assert self.evaluator.evaluate_faithfulness(
            np.array([]), np.array([]), np.array([]).reshape(0, 0), np.array([]).reshape(0, 0),
        ) == 0.0

    def test_evaluate_stability_identical(self):
        """Identical runs should produce stability = 1.0."""
        attrs = np.random.rand(20, 5)
        score = self.evaluator.evaluate_stability(attrs, attrs)
        assert score == 1.0

    def test_evaluate_stability_different(self):
        """Different runs should produce lower stability."""
        np.random.seed(42)
        a1 = np.random.rand(20, 5)
        a2 = np.random.rand(20, 5)
        score = self.evaluator.evaluate_stability(a1, a2)
        assert 0.0 <= score <= 1.0
        assert score < 1.0

    def test_evaluate_stability_shape_mismatch(self):
        """Shape mismatch should return 0."""
        assert self.evaluator.evaluate_stability(
            np.random.rand(10, 5), np.random.rand(10, 3),
        ) == 0.0

    def test_evaluate_comprehensibility_sparse(self):
        """Sparse explanations should score well."""
        # One dominant feature
        attrs = np.zeros((10, 20))
        attrs[:, 0] = 1.0
        score = XAIQualityEvaluator.evaluate_comprehensibility(attrs, max_features_shown=5)
        assert score > 0.5

    def test_evaluate_comprehensibility_with_names(self):
        """Feature names should give a small bonus."""
        attrs = np.random.rand(5, 3)
        names = ["feature_a", "feature_b", "feature_c"]
        score_with = XAIQualityEvaluator.evaluate_comprehensibility(attrs, feature_names=names)
        score_without = XAIQualityEvaluator.evaluate_comprehensibility(attrs)
        assert score_with >= score_without

    def test_check_consistency_identical(self):
        """Identical attributions should be consistent."""
        attrs = np.random.rand(20, 10)
        result = self.evaluator.check_explanation_consistency(
            task_name="test_task",
            attributions_a=attrs,
            attributions_b=attrs,
        )
        assert isinstance(result, ConsistencyResult)
        assert result.is_consistent is True
        assert result.rank_correlation == 1.0
        assert result.top_k_overlap == 1.0

    def test_check_consistency_shape_mismatch(self):
        """Shape mismatch should return inconsistent."""
        result = self.evaluator.check_explanation_consistency(
            task_name="test_task",
            attributions_a=np.random.rand(10, 5),
            attributions_b=np.random.rand(10, 3),
        )
        assert result.is_consistent is False

    def test_evaluate_task_passing(self):
        """Task with good scores should pass."""
        metrics = self.evaluator.evaluate_task(
            task_name="good_task",
            faithfulness_score=0.85,
            stability_score=0.90,
            comprehensibility_score=0.80,
            coverage=0.95,
        )
        assert isinstance(metrics, ExplanationQualityMetrics)
        assert metrics.meets_threshold is True
        assert len(metrics.violations) == 0
        assert metrics.overall_quality > 0.0

    def test_evaluate_task_failing(self):
        """Task with bad scores should fail."""
        metrics = self.evaluator.evaluate_task(
            task_name="bad_task",
            faithfulness_score=0.3,
            stability_score=0.2,
            comprehensibility_score=0.1,
            coverage=0.2,
        )
        assert metrics.meets_threshold is False
        assert len(metrics.violations) == 4

    def test_generate_report(self):
        """Full XAI quality report generation."""
        m1 = self.evaluator.evaluate_task("task_a", 0.9, 0.9, 0.8, 0.95)
        m2 = self.evaluator.evaluate_task("task_b", 0.5, 0.4, 0.3, 0.5)

        report = self.evaluator.generate_report(task_metrics=[m1, m2])
        assert isinstance(report, XAIQualityReport)
        assert "XAI" in report.report_id
        assert report.tasks_passing == 1
        assert report.tasks_failing == 1
        assert 0.0 <= report.aggregate_quality <= 1.0

    def test_generate_report_empty(self):
        """Report with no tasks should still be valid."""
        report = self.evaluator.generate_report()
        assert report.aggregate_quality == 0.0
        assert report.tasks_passing == 0

    def test_to_dict(self):
        """Serialization should produce a JSON-compatible dict."""
        metrics = self.evaluator.evaluate_task("t", 0.8, 0.8, 0.7, 0.9)
        report = self.evaluator.generate_report(task_metrics=[metrics])
        d = self.evaluator.to_dict(report)
        assert isinstance(d, dict)
        assert "task_metrics" in d
        assert "consistency_results" in d


# ===================================================================
# Integration tests
# ===================================================================

class TestCrossModuleIntegration:
    """Tests verifying interaction between evaluator modules."""

    def test_fria_feeds_into_eu_ai_act(self):
        """FRIA risk level should be consistent with EU AI Act classification."""
        fria = FRIAEvaluator()
        profile = fria.assess_task(
            task_name="credit_scoring",
            fairness_metrics={"violation_count": 0, "di_value": 1.0},
        )
        assert profile.risk_level in ("UNACCEPTABLE", "HIGH", "LIMITED", "MINIMAL")

        mapper = EUAIActMapper()
        level, _ = mapper.classify_risk({"use_cases": ["creditworthiness"]})
        assert level == "HIGH"

    def test_pia_consent_gaps_align_with_eu_ai_act(self):
        """PIA consent gaps should map to EU AI Act transparency requirements."""
        pia = PIAEvaluator()
        consent = pia.check_consent_adequacy({})
        assert len(consent["gaps"]) > 0

        mapper = EUAIActMapper()
        checklist = mapper.evaluate_compliance()
        transparency_items = [
            item for item in checklist if "Transparency" in item.article_title
        ]
        assert len(transparency_items) > 0
        assert all(not item.is_compliant for item in transparency_items)

    def test_eu_ai_act_existing_compliance_integration(self):
        """EU AI Act mapper should integrate with existing compliance results."""
        mapper = EUAIActMapper()
        checker_results = {
            "items": {
                "MON-01": {"status": "compliant", "description": "Audit logger"},
                "MON-03": {"status": "compliant", "description": "Fairness monitoring"},
            }
        }
        evidence = mapper.integrate_existing_compliance(checker_results)
        report = mapper.generate_report(evidence=evidence)
        assert report.compliant_count >= 2
