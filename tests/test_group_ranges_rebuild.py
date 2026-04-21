"""Regression tests for PipelineRunner group-range rebuild (CLAUDE.md §1.7).

The 3-stage normalizer reorders and appends ``{col}_log`` copies, so the
``feature_group_ranges`` computed at Stage 3 no longer match the
post-normalization column order.  ``_rebuild_group_ranges_post_normalization``
remaps each group to the longest contiguous block in the new order and
attaches ``_log`` offspring to their parent group.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from core.pipeline.runner import (
    _longest_contiguous_run,
    _rebuild_group_ranges_post_normalization,
)


def _pipeline(groups):
    registry = SimpleNamespace(enabled_groups=groups)
    return SimpleNamespace(_registry=registry)


# ---------------------------------------------------------------------------
# _longest_contiguous_run
# ---------------------------------------------------------------------------

class TestLongestContiguousRun:
    def test_empty_returns_none(self):
        assert _longest_contiguous_run([]) is None

    def test_single_position(self):
        assert _longest_contiguous_run([7]) == (7, 8)

    def test_fully_contiguous(self):
        assert _longest_contiguous_run([3, 4, 5, 6]) == (3, 7)

    def test_split_picks_longest(self):
        # [0, 1] (len 2) vs [4, 5, 6] (len 3) -> longest is the second run
        assert _longest_contiguous_run([0, 1, 4, 5, 6]) == (4, 7)

    def test_tie_picks_earliest(self):
        # Two runs of length 2: [0, 1] and [4, 5] -> earliest wins
        assert _longest_contiguous_run([0, 1, 4, 5]) == (0, 2)

    def test_unsorted_input_is_handled(self):
        assert _longest_contiguous_run([6, 4, 5, 3]) == (3, 7)

    def test_duplicates_are_deduped(self):
        assert _longest_contiguous_run([2, 2, 3, 4]) == (2, 5)


# ---------------------------------------------------------------------------
# _rebuild_group_ranges_post_normalization
# ---------------------------------------------------------------------------

class TestRebuildGroupRanges:
    def test_contiguous_groups_preserved(self):
        groups = [
            SimpleNamespace(name="g1", output_columns=["a", "b"]),
            SimpleNamespace(name="g2", output_columns=["c", "d"]),
        ]
        feature_cols = ["a", "b", "c", "d"]
        out = _rebuild_group_ranges_post_normalization(
            _pipeline(groups), feature_cols, log_cols_created=[],
        )
        assert out == {"g1": (0, 2), "g2": (2, 4)}

    def test_log_copies_attached_when_contiguous(self):
        # _log copies happen to land right after the group -> still contiguous
        groups = [
            SimpleNamespace(name="money", output_columns=["salary", "debt"]),
        ]
        feature_cols = ["salary", "debt", "salary_log", "debt_log"]
        out = _rebuild_group_ranges_post_normalization(
            _pipeline(groups),
            feature_cols,
            log_cols_created=["salary_log", "debt_log"],
        )
        assert out == {"money": (0, 4)}

    def test_log_copies_at_tail_triggers_longest_block_warning(self, caplog):
        # Normalizer reorders: group 'money' scattered because 'debt' moved
        # to binary block, and _log copies tacked on at the end.
        groups = [
            SimpleNamespace(name="money", output_columns=["salary", "debt"]),
            SimpleNamespace(name="flags", output_columns=["is_active"]),
        ]
        feature_cols = [
            "salary",       # 0 - money
            "is_active",    # 1 - flags
            "debt",         # 2 - money (NON-contiguous with 'salary')
            "salary_log",   # 3 - money _log
            "debt_log",     # 4 - money _log
        ]
        with caplog.at_level("WARNING"):
            out = _rebuild_group_ranges_post_normalization(
                _pipeline(groups),
                feature_cols,
                log_cols_created=["salary_log", "debt_log"],
            )
        # money columns at {0, 2, 3, 4}: longest run is [2, 5) (3 cols).
        assert out["money"] == (2, 5)
        assert out["flags"] == (1, 2)
        assert any(
            "non-contiguous" in rec.message for rec in caplog.records
        ), "Expected a warning about non-contiguous group"

    def test_missing_columns_are_skipped(self):
        groups = [
            SimpleNamespace(name="g1", output_columns=["a", "missing"]),
        ]
        feature_cols = ["a", "b"]
        out = _rebuild_group_ranges_post_normalization(
            _pipeline(groups), feature_cols, log_cols_created=[],
        )
        assert out == {"g1": (0, 1)}

    def test_group_with_zero_resolvable_columns_dropped(self):
        groups = [
            SimpleNamespace(name="g1", output_columns=["x", "y"]),
            SimpleNamespace(name="g_ghost", output_columns=["nope"]),
        ]
        feature_cols = ["x", "y"]
        out = _rebuild_group_ranges_post_normalization(
            _pipeline(groups), feature_cols, log_cols_created=[],
        )
        assert out == {"g1": (0, 2)}
        assert "g_ghost" not in out

    def test_no_registry_returns_empty(self):
        pipeline = SimpleNamespace(_registry=None)
        out = _rebuild_group_ranges_post_normalization(
            pipeline, ["a", "b"], log_cols_created=[],
        )
        assert out == {}

    def test_log_copy_without_parent_in_group_ignored(self):
        # 'mood_log' exists but 'mood' is not in the group -> _log stays
        # orphaned from this group (correct: log copies are attached to
        # the group that owns the BASE column).
        groups = [
            SimpleNamespace(name="money", output_columns=["salary"]),
        ]
        feature_cols = ["salary", "mood_log", "salary_log"]
        out = _rebuild_group_ranges_post_normalization(
            _pipeline(groups),
            feature_cols,
            log_cols_created=["mood_log", "salary_log"],
        )
        # 'salary' at 0, 'salary_log' at 2 -> longest run is either (0,1)
        # or (2,3); tie picks earliest -> (0, 1).
        assert out["money"] == (0, 1)
