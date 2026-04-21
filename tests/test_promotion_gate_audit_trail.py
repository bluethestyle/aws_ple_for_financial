"""
Audit-trail tests for PromotionGate (CLAUDE.md §1.10).

Validates that:
1. ``MetricsDerivedScoreProvider.explain()`` exposes per-dim derivation
   (path, raw_value, transform, fallback_used, score) for audit.
2. ``GateVerdict.details`` is populated with metadata snapshot + dim
   derivation for every non-skip verdict (pass / reject / require_approval).
3. ``AuditLogger.log_model_promotion(gate_details=...)`` embeds the
   verdict details in the HMAC+hash-chain record.
4. ``_audit_promotion`` in submit_pipeline forwards gate_details.

Run: ``pytest tests/test_promotion_gate_audit_trail.py -v``
"""

from __future__ import annotations

import json
from types import SimpleNamespace
from typing import Any, Dict

import pytest

from core.compliance.ai_risk_classifier import AIRiskClassifier
from core.compliance.dimension_scores import (
    CompositeProvider,
    ManualScoreProvider,
    MetricsDerivedScoreProvider,
)
from core.compliance.fria_assessment import FRIA_DIMENSIONS, KoreanFRIAAssessor
from core.compliance.metadata_aggregator import (
    MetadataAggregator,
    build_static_source,
)
from core.compliance.store import InMemoryComplianceStore
from core.evaluation.promotion_gate import (
    GateVerdict,
    PromotionGate,
    build_promotion_gate,
)
from core.monitoring.audit_logger import AuditLogger


# ---------------------------------------------------------------------------
# explain()
# ---------------------------------------------------------------------------

class TestProviderExplain:
    def test_explain_returns_per_dim_trail(self):
        def lookup(_v: str) -> Dict[str, Any]:
            return {
                "pii_ratio": 0.3,
                "human_review_fraction": 0.7,
                "customer_count": 100,
                "param_count": 1000,
                "llm_provider_ratio": 0.2,
                "disparate_impact_min": 0.9,
                "reason_coverage": 0.8,
            }
        provider = MetricsDerivedScoreProvider(
            metadata_lookup=lookup, dimensions=FRIA_DIMENSIONS,
        )
        trail = provider.explain("v1")
        assert set(trail.keys()) == set(FRIA_DIMENSIONS)
        # data_sensitivity uses pii_ratio (identity): raw=0.3 → score=0.3
        ds = trail["data_sensitivity"]
        assert ds["path"] == "pii_ratio"
        assert ds["raw_value"] == 0.3
        assert ds["transform"] == "identity"
        assert ds["fallback_used"] is False
        assert ds["score"] == pytest.approx(0.3)

    def test_explain_marks_fallback_when_path_missing(self):
        # Metadata missing reason_coverage → explainability_gap uses fallback
        def lookup(_v: str) -> Dict[str, Any]:
            return {"pii_ratio": 0.5}
        provider = MetricsDerivedScoreProvider(
            metadata_lookup=lookup, dimensions=("explainability_gap",),
        )
        trail = provider.explain("v1")
        eg = trail["explainability_gap"]
        assert eg["fallback_used"] is True
        assert eg["raw_value"] is None
        assert eg["score"] == 0.5  # rule.fallback

    def test_explain_marks_fallback_when_raw_unparsable(self):
        def lookup(_v: str) -> Dict[str, Any]:
            return {"pii_ratio": "not-a-number"}
        provider = MetricsDerivedScoreProvider(
            metadata_lookup=lookup, dimensions=("data_sensitivity",),
        )
        trail = provider.explain("v1")
        ds = trail["data_sensitivity"]
        assert ds["fallback_used"] is True
        assert ds["raw_value"] == "not-a-number"

    def test_explain_survives_lookup_exception(self):
        def lookup(_v: str):
            raise RuntimeError("source down")
        provider = MetricsDerivedScoreProvider(
            metadata_lookup=lookup, dimensions=("data_sensitivity",),
        )
        trail = provider.explain("v1")
        assert trail["data_sensitivity"]["fallback_used"] is True


# ---------------------------------------------------------------------------
# GateVerdict.details population
# ---------------------------------------------------------------------------

class TestVerdictDetails:
    def _gate(self, provider):
        store = InMemoryComplianceStore()
        return PromotionGate(
            fria_assessor=KoreanFRIAAssessor(store=store),
            ai_risk_classifier=AIRiskClassifier(store=store),
            enabled=True,
            fria_scores_provider=provider,
            ai_risk_scores_provider=provider,
        )

    def test_details_present_on_pass_verdict(self):
        agg = MetadataAggregator(sources=[build_static_source({
            "pii_ratio": 0.2, "human_review_fraction": 0.8,
            "customer_count": 10, "param_count": 100,
            "llm_provider_ratio": 0.1, "disparate_impact_min": 0.95,
            "reason_coverage": 0.9,
        })])
        provider = MetricsDerivedScoreProvider(
            metadata_lookup=agg, dimensions=FRIA_DIMENSIONS,
        )
        gate = self._gate(provider)
        v = gate.evaluate(model_version="v_pass")
        assert v.decision == "pass"
        assert v.details["model_version"] == "v_pass"
        assert set(v.details["fria_dimension_scores"]) == set(FRIA_DIMENSIONS)
        assert v.details["fria_derivation"] is not None
        assert "pii_ratio" in v.details["metadata_snapshot"]

    def test_details_present_on_reject_verdict(self):
        # Force UNACCEPTABLE via ManualScoreProvider
        manual = ManualScoreProvider(
            {"v_bad": {d: 0.99 for d in FRIA_DIMENSIONS}},
            dimensions=FRIA_DIMENSIONS,
        )
        gate = self._gate(manual)
        v = gate.evaluate(model_version="v_bad")
        assert v.decision == "reject"
        assert v.details["model_version"] == "v_bad"
        # Manual provider has no explain() → derivation trail is None
        assert v.details["fria_derivation"] is None

    def test_details_derivation_walks_composite_layers(self):
        """CompositeProvider with a MetricsDerived layer → explain wired."""
        agg = MetadataAggregator(sources=[build_static_source({
            "pii_ratio": 0.2, "human_review_fraction": 0.8,
            "customer_count": 10, "param_count": 100,
            "llm_provider_ratio": 0.1, "disparate_impact_min": 0.95,
            "reason_coverage": 0.9,
        })])
        manual = ManualScoreProvider({}, dimensions=FRIA_DIMENSIONS)
        derived = MetricsDerivedScoreProvider(
            metadata_lookup=agg, dimensions=FRIA_DIMENSIONS,
        )
        comp = CompositeProvider([manual, derived])
        gate = self._gate(comp)
        v = gate.evaluate(model_version="v_comp")
        assert v.decision == "pass"
        trail = v.details["fria_derivation"]
        assert trail is not None
        assert set(trail.keys()) == set(FRIA_DIMENSIONS)

    def test_details_metadata_snapshot_freezes_inputs(self):
        captured = {"v1": {
            "pii_ratio": 0.3, "human_review_fraction": 0.7,
            "customer_count": 100, "param_count": 1000,
            "llm_provider_ratio": 0.2, "disparate_impact_min": 0.9,
            "reason_coverage": 0.8,
        }}
        def lookup(v):
            return dict(captured[v])
        provider = MetricsDerivedScoreProvider(
            metadata_lookup=lookup, dimensions=FRIA_DIMENSIONS,
        )
        gate = self._gate(provider)
        v = gate.evaluate(model_version="v1")
        snap = v.details["metadata_snapshot"]
        assert snap["pii_ratio"] == 0.3
        # Mutating after verdict must not affect the frozen snapshot
        captured["v1"]["pii_ratio"] = 0.99
        assert v.details["metadata_snapshot"]["pii_ratio"] == 0.3


# ---------------------------------------------------------------------------
# AuditLogger integration
# ---------------------------------------------------------------------------

def _read_audit_lines(tmp_path):
    """Read the jsonl file the AuditLogger wrote under {tmp_path}/{yyyymm}/."""
    files = list(tmp_path.rglob("audit_*.jsonl"))
    assert files, f"expected audit_*.jsonl under {tmp_path}"
    lines = []
    for f in sorted(files):
        lines.extend(f.read_text(encoding="utf-8").splitlines())
    return lines


class TestAuditLoggerGateDetails:
    def test_log_model_promotion_accepts_gate_details(self, tmp_path):
        al = AuditLogger(local_fallback_dir=str(tmp_path))
        gate_details = {
            "model_version": "v1",
            "fria_dimension_scores": {"data_sensitivity": 0.3},
            "metadata_snapshot": {"pii_ratio": 0.3},
        }
        entry = al.log_model_promotion(
            champion_version="v0", challenger_version="v1",
            decision="promote", reason="competition pass",
            gate_details=gate_details,
        )
        assert entry is not None
        lines = _read_audit_lines(tmp_path)
        assert len(lines) == 1
        payload = json.loads(lines[0])
        meta = payload["metadata"]
        assert meta["gate_details"]["model_version"] == "v1"
        assert meta["gate_details"]["metadata_snapshot"]["pii_ratio"] == 0.3

    def test_log_without_gate_details_omits_field(self, tmp_path):
        al = AuditLogger(local_fallback_dir=str(tmp_path))
        al.log_model_promotion(
            champion_version=None, challenger_version="v1",
            decision="bootstrap", reason="first model",
        )
        lines = _read_audit_lines(tmp_path)
        payload = json.loads(lines[0])
        assert "gate_details" not in payload["metadata"]

    def test_hash_chain_intact_with_gate_details(self, tmp_path):
        al = AuditLogger(local_fallback_dir=str(tmp_path))
        al.log_model_promotion(
            champion_version="v0", challenger_version="v1",
            decision="promote", reason="pass",
            gate_details={"k": "v"},
        )
        al.log_model_promotion(
            champion_version="v1", challenger_version="v2",
            decision="reject", reason="fail",
            gate_details={"k2": "v2"},
        )
        lines = _read_audit_lines(tmp_path)
        assert al.verify_chain(lines) is True


# ---------------------------------------------------------------------------
# submit_pipeline._audit_promotion smoke test
# ---------------------------------------------------------------------------

class TestSubmitPipelineAuditWiring:
    def test_audit_promotion_forwards_gate_details(self, monkeypatch):
        from scripts import submit_pipeline

        captured: Dict[str, Any] = {}

        class _FakeLogger:
            def log_model_promotion(self, **kwargs):
                captured.update(kwargs)
                return {"ok": True}

        monkeypatch.setattr(
            "core.monitoring.audit_logger.AuditLogger",
            lambda *a, **kw: _FakeLogger(),
        )

        submit_pipeline._audit_promotion(
            champion_version="v0", challenger_version="v1",
            decision="reject", reason="test",
            gate_details={"model_version": "v1",
                          "fria_dimension_scores": {"data_sensitivity": 0.9}},
        )
        assert captured["gate_details"]["fria_dimension_scores"][
            "data_sensitivity"] == 0.9
