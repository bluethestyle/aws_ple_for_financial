"""Regression for the evidential/SAE dead-config gap (P3-b).

Before this fix, build_ple_config never mapped the model.evidential /
model.sae YAML blocks, so ``enabled: true`` was silently ignored and the
PLEConfig defaults (False) always applied. These tests verify the flags now
flow through, while the pipeline.yaml default stays false (research-gated).

Run: pytest tests/test_config_builder_dead_config.py -v
"""

from __future__ import annotations

from pathlib import Path

import pytest

from core.model.config_builder import build_ple_config


def _label_schema(model_block: dict) -> dict:
    return {
        "tasks": [{"name": "t0", "type": "binary"}],
        "model": model_block,
    }


class TestEvidentialSaeMapping:
    def test_enabled_flags_flow_through(self):
        ls = _label_schema({
            "evidential": {"enabled": True, "kl_lambda": 0.02, "annealing_epochs": 5},
            "sae": {"enabled": True, "weight": 0.05, "expansion_factor": 8},
        })
        cfg = build_ple_config(
            config={}, feature_schema={}, label_schema=ls, input_dim=8, hp={},
        )
        assert cfg.evidential_enabled is True
        assert cfg.evidential_kl_lambda == pytest.approx(0.02)
        assert cfg.evidential_annealing_epochs == 5
        assert cfg.sae_enabled is True
        assert cfg.sae_weight == pytest.approx(0.05)
        assert cfg.sae_expansion_factor == 8

    def test_disabled_when_block_says_false(self):
        ls = _label_schema({
            "evidential": {"enabled": False},
            "sae": {"enabled": False},
        })
        cfg = build_ple_config(
            config={}, feature_schema={}, label_schema=ls, input_dim=8, hp={},
        )
        assert cfg.evidential_enabled is False
        assert cfg.sae_enabled is False

    def test_absent_blocks_keep_defaults_off(self):
        cfg = build_ple_config(
            config={}, feature_schema={}, label_schema=_label_schema({}),
            input_dim=8, hp={},
        )
        assert cfg.evidential_enabled is False
        assert cfg.sae_enabled is False


class TestPipelineYamlDefaultIsResearchGated:
    def test_pipeline_yaml_defaults_false(self):
        import yaml

        cfg = yaml.safe_load(
            Path("configs/pipeline.yaml").read_text(encoding="utf-8")
        )
        model = cfg.get("model", {})
        assert model.get("evidential", {}).get("enabled") is False
        assert model.get("sae", {}).get("enabled") is False
