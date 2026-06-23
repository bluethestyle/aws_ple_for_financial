"""Regression tests for the Disparate-Impact insufficient-data guard (#7).

Guards the fix where an empty / single-group recommendation set made
``compute_disparate_impact`` return 0.0 (read as maximal unfairness), raising
a *false* fairness violation that polluted the promotion gate. Absence of data
must return a neutral 1.0 (parity), while a genuine disparity still returns 0.0.
"""

from core.monitoring.fairness_monitor import FairnessMonitor


def _recs(pairs):
    return [{"gender": g, "recommended": r} for g, r in pairs]


def test_di_neutral_when_one_group_missing():
    fm = FairnessMonitor()
    recs = _recs([("M", True), ("M", False)])  # no 'F' members
    assert fm.compute_disparate_impact(recs, "gender", "M", "F") == 1.0


def test_di_neutral_when_no_data():
    fm = FairnessMonitor()
    assert fm.compute_disparate_impact([], "gender", "M", "F") == 1.0


def test_di_detects_real_disparity():
    fm = FairnessMonitor()
    # privileged M rate 1.0, unprivileged F rate 0.0 → genuine 0.0
    recs = _recs([("M", True), ("M", True), ("F", False), ("F", False)])
    assert fm.compute_disparate_impact(recs, "gender", "M", "F") == 0.0


def test_di_parity_when_equal_rates():
    fm = FairnessMonitor()
    recs = _recs([("M", True), ("M", False), ("F", True), ("F", False)])
    assert fm.compute_disparate_impact(recs, "gender", "M", "F") == 1.0
