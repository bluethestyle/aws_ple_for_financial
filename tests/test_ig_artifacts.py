"""Tests for the serving IG artifact producer/loader (#14)."""

import json

from core.serving.ig_artifacts import (
    IG_TOP_FEATURES_FILE,
    build_serving_ig_top_features,
    load_ig_attributions,
)


def test_build_maps_indices_to_names_and_ranks():
    ig = {"task_a": {0: 0.1, 2: 0.5, 1: 0.3}, "task_b": {"error": "x"}}
    out = build_serving_ig_top_features(ig, feature_names=["income", "age", "dsr"], top_n=2)
    assert out["task_a"] == [["dsr", 0.5], ["age", 0.3]]  # top-2 by importance
    assert "task_b" not in out  # non-dict (error) entry skipped


def test_build_falls_back_to_feature_index_names():
    ig = {"task_a": {0: 0.2, 5: 0.9}}
    out = build_serving_ig_top_features(ig, feature_names=None, top_n=10)
    assert out["task_a"][0] == ["feature_5", 0.9]
    assert out["task_a"][1] == ["feature_0", 0.2]


def test_load_returns_none_when_absent(tmp_path):
    assert load_ig_attributions(tmp_path) is None
    assert load_ig_attributions(None) is None


def test_load_roundtrip(tmp_path):
    payload = {"task_a": [["income", 0.4]]}
    (tmp_path / IG_TOP_FEATURES_FILE).write_text(json.dumps(payload), encoding="utf-8")
    assert load_ig_attributions(tmp_path) == payload


def test_load_handles_corrupt_json(tmp_path):
    (tmp_path / IG_TOP_FEATURES_FILE).write_text("{not json", encoding="utf-8")
    assert load_ig_attributions(tmp_path) is None
