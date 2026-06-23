"""Regression tests for §17 suitability fail-closed behaviour (RuleEngine).

Guards the fix that removed the ``default=99`` customer-grade fallback, which
silently passed every risk-graded product when no risk assessment feature was
present (fail-open). An unassessed customer must now be blocked (금소법 §17).
"""

from core.recommendation.rule_engine import RuleBasedRecommender


def _engine():
    return RuleBasedRecommender(
        config={"rule_engine": {"product_risk_grades": {"will_acquire_investments": 3}}}
    )


def _result():
    return {"prediction": 1, "confidence": 0.9, "layer": 3}


def test_blocks_when_no_customer_grade():
    out = _engine()._apply_suitability({}, "will_acquire_investments", _result())
    assert out["prediction"] == 0
    assert out["rule_name"] == "suitability_unassessed"


def test_blocks_when_grade_unparseable():
    # A credit-category string (e.g. "AAA") is not a 1-5 tolerance → fail-closed.
    out = _engine()._apply_suitability(
        {"risk_grade_x": "AAA"}, "will_acquire_investments", _result()
    )
    assert out["prediction"] == 0


def test_blocks_when_grade_below_required():
    out = _engine()._apply_suitability(
        {"risk_grade_x": 1}, "will_acquire_investments", _result()
    )
    assert out["prediction"] == 0
    assert out["rule_name"] == "suitability_block"


def test_passes_when_grade_meets_required():
    result = _result()
    out = _engine()._apply_suitability(
        {"risk_grade_x": 5}, "will_acquire_investments", result
    )
    assert out is result  # unchanged → passes


def test_no_constraint_for_zero_grade_task():
    result = _result()
    out = _engine()._apply_suitability({}, "some_other_task", result)
    assert out is result  # required_grade 0 → no suitability constraint
