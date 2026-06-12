"""PORT-13 tests — QualityGate degenerate 가드 + schema_changed 경고 게이트.

Run: pytest tests/test_quality_gate_degenerate.py -v
"""

from __future__ import annotations

import pandas as pd
import pyarrow as pa
import pytest

from core.data.quality_gate import QualityGate, QualityGateError, Verdict
from core.data.validation import ValidationResult


class _PassThroughValidator:
    """기존 validator 체크와 분리해 PORT-13 체크만 검증하기 위한 fake."""

    def validate(self, source_name, df, reference_df=None):
        return ValidationResult(source=source_name, checks=[])


def _gate(**config) -> QualityGate:
    return QualityGate(config=config, validation_engine=_PassThroughValidator())


def _df_with_degenerates() -> pd.DataFrame:
    return pd.DataFrame({
        "normal": [1.0, 2.0, 3.0],
        "all_zero": [0.0, 0.0, 0.0],
        "constant": [7, 7, 7],
    })


class TestDegenerateGuard:
    def test_detects_all_zero_and_constant(self):
        result = _gate().evaluate(_df_with_degenerates(), "src")
        check = next(c for c in result.checks if c.name == "degenerate_columns")
        assert check.details["columns"] == {
            "all_zero": "all_zero", "constant": "constant",
        }
        # 기본은 WARN 가시화 — 파이프라인 비차단
        assert result.verdict == Verdict.WARN
        _gate().evaluate_and_block(_df_with_degenerates(), "src")  # no raise

    def test_hard_fail_opt_in_blocks(self):
        gate = _gate(degenerate_hard_fail=True)
        result = gate.evaluate(_df_with_degenerates(), "src")
        assert result.verdict == Verdict.FAIL
        with pytest.raises(QualityGateError):
            gate.evaluate_and_block(_df_with_degenerates(), "src")

    def test_clean_data_passes(self):
        df = pd.DataFrame({"a": [1, 2, 3], "b": [0.0, 1.5, -2.0]})
        result = _gate().evaluate(df, "src")
        assert all(c.name != "degenerate_columns" for c in result.checks)
        assert result.verdict == Verdict.PASS

    def test_pyarrow_table_supported(self):
        table = pa.table({
            "normal": [1.0, 2.0, 3.0],
            "all_zero": [0.0, 0.0, 0.0],
            "constant": ["x", "x", "x"],
        })
        result = _gate().evaluate(table, "src")
        check = next(c for c in result.checks if c.name == "degenerate_columns")
        assert check.details["columns"] == {
            "all_zero": "all_zero", "constant": "constant",
        }

    def test_empty_df_skipped(self):
        result = _gate().evaluate(pd.DataFrame({"a": []}), "src")
        assert all(c.name != "degenerate_columns" for c in result.checks)

    def test_disabled_via_config(self):
        gate = _gate(degenerate_check_enabled=False)
        result = gate.evaluate(_df_with_degenerates(), "src")
        assert all(c.name != "degenerate_columns" for c in result.checks)
        assert result.verdict == Verdict.PASS

    def test_default_config_does_not_leak_between_instances(self):
        # hard_fail 인스턴스가 공유 default set 을 오염시키면 안 된다.
        _gate(degenerate_hard_fail=True)
        result = _gate().evaluate(_df_with_degenerates(), "src")
        assert result.verdict == Verdict.WARN  # 새 인스턴스는 여전히 WARN


class TestSchemaChangedGate:
    def _diff(self, changed=True):
        return {
            "version_a": "v1", "version_b": "v2",
            "schema_changed": changed,
            "columns_added": ["new_col"],
            "columns_removed": [],
        }

    def test_schema_changed_produces_warn(self):
        df = pd.DataFrame({"a": [1, 2, 3]})
        result = _gate().evaluate(df, "src", version_diff=self._diff())
        check = next(c for c in result.checks if c.name == "schema_changed")
        assert check.details["columns_added"] == ["new_col"]
        assert result.verdict == Verdict.WARN

    def test_unchanged_schema_no_check(self):
        df = pd.DataFrame({"a": [1, 2, 3]})
        result = _gate().evaluate(df, "src", version_diff=self._diff(changed=False))
        assert all(c.name != "schema_changed" for c in result.checks)
        assert result.verdict == Verdict.PASS

    def test_no_diff_is_backward_compatible(self):
        df = pd.DataFrame({"a": [1, 2, 3]})
        assert _gate().evaluate(df, "src").verdict == Verdict.PASS
