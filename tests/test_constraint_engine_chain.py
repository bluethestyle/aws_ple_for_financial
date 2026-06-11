"""Regression: SuitabilityFilter (금소법 §17) must run in the default chain.

The filter was registered and unit-tested, but the ConstraintEngine default
filter_chain was ["fatigue", "eligibility", "owned_product"] — so the
suitability principle never executed in the recommendation path unless a
deployment explicitly listed it. These tests pin it into the default chain
and verify it actually blocks an unsuitable pair end-to-end.

Run: pytest tests/test_constraint_engine_chain.py -v
"""

from __future__ import annotations

from core.recommendation.constraint_engine import ConstraintEngine, SuitabilityFilter


class TestDefaultChainIncludesSuitability:
    def test_suitability_in_default_chain(self):
        engine = ConstraintEngine(config={})
        names = [type(f).__name__ for f in engine.filters]
        assert "SuitabilityFilter" in names

    def test_default_chain_blocks_risk_exceeds_tolerance(self):
        # fail_fast=False so every filter evaluates and we can assert the
        # suitability verdict specifically (otherwise an earlier filter may
        # short-circuit the chain before suitability runs).
        engine = ConstraintEngine(config={"constraint_engine": {"fail_fast": False}})
        passed, results = engine.apply(
            "C001", "PROD_A",
            {"customer_risk_tolerance": 2, "item_risk_level": 5},
        )
        assert passed is False
        suit = [r for r in results if r.filter_name == "suitability"]
        assert suit and suit[0].passed is False

    def test_no_item_risk_level_is_passthrough(self):
        """Deployments without risk context are unaffected (pass-through)."""
        engine = ConstraintEngine(config={})
        passed, results = engine.apply("C001", "PROD_A", {})
        suit = [r for r in results if r.filter_name == "suitability"]
        # suitability itself does not block when no item_risk_level is present
        assert not suit or suit[0].passed is True

    def test_explicit_chain_override_still_respected(self):
        engine = ConstraintEngine(config={
            "constraint_engine": {"filter_chain": ["fatigue"]},
        })
        names = [type(f).__name__ for f in engine.filters]
        assert "SuitabilityFilter" not in names
