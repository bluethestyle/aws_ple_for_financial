"""Tests for distillation fidelity gate (Fix 1 + Fix 2, 2026-05-12).

Covers two changes to the fidelity gate after the v14 distillation re-run
revealed two latent defects:

Fix 1 (containers/distillation/fidelity.py::validate_fidelity)
    The gate now accepts an optional ``distill_tasks`` argument and skips
    tasks that were routed via the 3-layer fallback (DIRECT / SKIP). These
    tasks are intentionally trained against hard labels because the
    teacher was below the threshold gate's viability criteria, so scoring
    them *against the teacher* yields meaningless failures.

Fix 2 (core/training/distillation_validator.py::_compute_calibration_gap)
    The metric was previously ``|ECE(teacher) - ECE(student)|`` which
    confounds student calibration quality with teacher calibration
    quality. The new definition is the student's own ECE against
    ground-truth labels — i.e. "is the student's probability score
    trustworthy at decision thresholds?". A well-calibrated student now
    reports a low value regardless of teacher ECE.
"""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict

import json
import numpy as np
import pytest


# ---------------------------------------------------------------------------
# Fix 2 — _compute_calibration_gap now returns student ECE vs ground truth
# ---------------------------------------------------------------------------


def test_calibration_gap_is_student_ece_only() -> None:
    """A perfectly-calibrated student with a poorly-calibrated teacher must
    report a low calibration_gap (close to 0), not the historical
    |ECE(teacher) - ECE(student)| ≈ ECE(teacher).
    """
    from core.training.distillation_validator import DistillationValidator

    rng = np.random.default_rng(0)
    n = 10_000

    # Ground-truth Bernoulli(0.3).
    labels = (rng.random(n) < 0.3).astype(int)

    # Perfectly calibrated student: prediction = empirical rate.
    student = np.full(n, 0.3)

    # Pathologically miscalibrated teacher: always predicts 0.9 regardless
    # of label (focal-loss-like overconfidence). Old formula would give a
    # huge gap; new formula ignores the teacher entirely.
    teacher = np.full(n, 0.9)

    gap = DistillationValidator._compute_calibration_gap(teacher, student, labels)

    # Student is perfectly calibrated (acc=0.3, conf=0.3 in the active bin).
    assert gap < 0.02, f"student-only ECE should be ~0, got {gap:.4f}"


def test_calibration_gap_detects_overconfident_student() -> None:
    """An overconfident student must still register as poorly calibrated
    even when the teacher is also overconfident (no cancellation)."""
    from core.training.distillation_validator import DistillationValidator

    rng = np.random.default_rng(1)
    n = 10_000

    # Base rate 0.2 but both teacher and student predict 0.9 → student ECE
    # ≈ |0.9 - 0.2| = 0.7.
    labels = (rng.random(n) < 0.2).astype(int)
    teacher = np.full(n, 0.9)
    student = np.full(n, 0.9)

    gap = DistillationValidator._compute_calibration_gap(teacher, student, labels)

    # Student is wildly overconfident; old formula returns 0 (teacher and
    # student match), new formula returns student ECE ≈ 0.7.
    assert gap > 0.6, f"overconfident student should fail calibration, got {gap:.4f}"


def test_calibration_gap_threshold_compatible_with_well_calibrated_student() -> None:
    """A student with ECE ≈ 0.03 must pass the default 0.05 threshold."""
    from core.training.distillation_validator import (
        DistillationValidator,
        ValidationCriteria,
    )

    rng = np.random.default_rng(2)
    n = 20_000
    labels = (rng.random(n) < 0.4).astype(int)
    # Student predictions ≈ true probability with small jitter.
    student = np.clip(0.4 + rng.normal(0, 0.05, n), 0.05, 0.95)
    teacher = rng.random(n)  # Random teacher should not affect outcome.

    criteria = ValidationCriteria(max_calibration_gap=0.05)
    validator = DistillationValidator(criteria=criteria)
    result = validator.validate_task(
        task_name="probe",
        task_type="binary",
        teacher_preds=teacher,
        student_preds=student,
        labels=labels,
    )

    assert "calibration_gap" in result.metrics
    cal_failures = [f for f in result.failures if "calibration_gap" in f]
    assert not cal_failures, (
        f"well-calibrated student should pass 0.05 gate, failures={cal_failures}"
    )


# ---------------------------------------------------------------------------
# Fix 1 — validate_fidelity scopes the gate to distill_tasks when given
# ---------------------------------------------------------------------------


class _StubTrainer:
    """Minimal trainer exposing get_soft_labels()."""

    def __init__(self, soft_labels: Dict[str, np.ndarray]) -> None:
        self._soft = soft_labels

    def get_soft_labels(self) -> Dict[str, np.ndarray]:
        return self._soft


class _StubLGBM:
    """Minimal LGBM returning a fixed probability."""

    def __init__(self, p: float) -> None:
        self.p = p

    def predict(self, features: np.ndarray) -> np.ndarray:
        return np.full(features.shape[0], self.p, dtype=np.float64)


def _make_pipeline_config() -> Any:
    """Three binary tasks: a, b, c. Plain SimpleNamespace stand-in."""
    return SimpleNamespace(
        tasks=[
            SimpleNamespace(name="distill_a", type="binary", label_col="distill_a",
                            num_classes=2),
            SimpleNamespace(name="direct_b", type="binary", label_col="direct_b",
                            num_classes=2),
            SimpleNamespace(name="skip_c", type="binary", label_col="skip_c",
                            num_classes=2),
        ]
    )


def _make_inputs(n: int = 1000, seed: int = 7):
    rng = np.random.default_rng(seed)
    features = rng.random((n, 4)).astype(np.float32)
    labels = {
        "distill_a": (rng.random(n) < 0.3).astype(int),
        "direct_b":  (rng.random(n) < 0.5).astype(int),
        "skip_c":    (rng.random(n) < 0.2).astype(int),
    }
    soft = {
        # distill_a: teacher ~ student so AUC gap is small (gate passes)
        "distill_a": np.where(labels["distill_a"] == 1,
                              rng.uniform(0.6, 0.8, n),
                              rng.uniform(0.2, 0.4, n)),
        # direct_b: teacher is near-random (the reason it was routed direct)
        "direct_b": rng.uniform(0.45, 0.55, n),
        # skip_c: teacher random (skipped, but if reached, would fail)
        "skip_c": rng.uniform(0.0, 1.0, n),
    }
    return features, labels, soft


def test_fidelity_gate_scoped_to_distill_tasks(tmp_path: Path) -> None:
    """When distill_tasks=['distill_a'], the gate must score only
    distill_a even though all three tasks have trained students."""
    from containers.distillation.fidelity import validate_fidelity

    features, hard_labels, soft = _make_inputs()
    students = {
        "distill_a": _StubLGBM(p=0.7),
        "direct_b":  _StubLGBM(p=0.9),
        "skip_c":    _StubLGBM(p=0.5),
    }
    trainer = _StubTrainer(soft)

    cfg = SimpleNamespace(use_custom_objective=False)
    distillation_cfg = {
        "fidelity": {
            "binary": {
                "max_auc_gap": 0.10,
                "min_agreement": 0.0,        # relaxed for stub data
                "max_jsd": 1.0,
                "min_ranking_corr": -1.0,
                "max_calibration_gap": 0.5,
            }
        }
    }

    results, report = validate_fidelity(
        pipeline_config=_make_pipeline_config(),
        students=students,
        calibrated_models={},
        features=features,
        hard_labels=hard_labels,
        student_config=cfg,
        distillation_cfg=distillation_cfg,
        trainer=trainer,
        out_dir=tmp_path,
        skip_gate=True,
        distill_tasks=["distill_a"],
    )

    scored = {r.task_name for r in results}
    assert scored == {"distill_a"}, (
        f"only distill_a should be scored under scoping, got {scored}"
    )
    assert report["gate_scope"] == "distilled_tasks_only"
    assert report["scoped_tasks"] == ["distill_a"]
    assert "direct_b" not in report["details"]
    assert "skip_c" not in report["details"]


def test_fidelity_gate_legacy_mode_scores_all(tmp_path: Path) -> None:
    """When distill_tasks=None, all tasks with trained students are scored
    (legacy behaviour preserved for backward compatibility)."""
    from containers.distillation.fidelity import validate_fidelity

    features, hard_labels, soft = _make_inputs()
    students = {
        "distill_a": _StubLGBM(p=0.7),
        "direct_b":  _StubLGBM(p=0.9),
    }
    trainer = _StubTrainer(soft)
    cfg = SimpleNamespace(use_custom_objective=False)
    distillation_cfg = {
        "fidelity": {
            "binary": {
                "max_auc_gap": 0.10,
                "min_agreement": 0.0,
                "max_jsd": 1.0,
                "min_ranking_corr": -1.0,
                "max_calibration_gap": 0.5,
            }
        }
    }

    results, report = validate_fidelity(
        pipeline_config=_make_pipeline_config(),
        students=students,
        calibrated_models={},
        features=features,
        hard_labels=hard_labels,
        student_config=cfg,
        distillation_cfg=distillation_cfg,
        trainer=trainer,
        out_dir=tmp_path,
        skip_gate=True,
        distill_tasks=None,
    )

    scored = {r.task_name for r in results}
    assert scored == {"distill_a", "direct_b"}
    assert report["gate_scope"] == "all_tasks_with_students"
    assert report["scoped_tasks"] is None
