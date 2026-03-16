"""Tests for the offline policy evaluation module.

Covers:
- PropensityEstimator (fit / predict / clipping)
- OfflinePolicyEvaluator (IPS / SNIPS / DR, bootstrap CI, multi-task)
- ExposureSimulator (cascade / geometric / reciprocal, reward adjustment)
- ModelCompetition (Go/No-Go gating, significance tests, evaluator integration)
"""

from __future__ import annotations

import numpy as np
import pytest

from core.evaluation.propensity import PropensityEstimator
from core.evaluation.policy_evaluator import (
    OfflinePolicyEvaluator,
    OPEResult,
    estimate_ips,
    estimate_snips,
    estimate_dr,
)
from core.evaluation.exposure_simulator import ExposureSimulator
from core.evaluation.model_competition import (
    ModelCompetition,
    CompetitionConfig,
    paired_t_test,
    paired_bootstrap_test,
)


# ═══════════════════════════════════════════════════════════════
# Fixtures
# ═══════════════════════════════════════════════════════════════

@pytest.fixture
def rng():
    return np.random.default_rng(123)


@pytest.fixture
def logged_data(rng):
    """Synthetic logged interaction data."""
    n = 500
    x1 = rng.standard_normal(n)
    x2 = rng.standard_normal(n)
    logit = 0.5 * x1 - 0.3 * x2
    prob = 1.0 / (1.0 + np.exp(-logit))
    actions = (rng.random(n) < prob).astype(int)
    rewards = actions * (0.5 + 0.2 * x1) + rng.standard_normal(n) * 0.1
    return {
        "x1": x1,
        "x2": x2,
        "action": actions,
        "reward": rewards,
    }


# ═══════════════════════════════════════════════════════════════
# PropensityEstimator
# ═══════════════════════════════════════════════════════════════

class TestPropensityEstimator:
    """Tests for propensity score estimation."""

    def test_fit_and_predict(self, logged_data):
        """Fit on logged data and produce propensity scores."""
        est = PropensityEstimator(calibration="none", clip_bounds=(0.05, 0.95))
        est.fit(logged_data, treatment_col="action", feature_cols=["x1", "x2"])
        scores = est.predict(logged_data)

        assert scores.shape == (500,)
        assert scores.min() >= 0.05
        assert scores.max() <= 0.95

    def test_predict_before_fit_raises(self, logged_data):
        """Calling predict before fit should raise RuntimeError."""
        est = PropensityEstimator()
        with pytest.raises(RuntimeError, match="not been fitted"):
            est.predict(logged_data)

    def test_clipping_bounds(self, logged_data):
        """Scores are clipped to configured bounds."""
        est = PropensityEstimator(
            calibration="none",
            clip_bounds=(0.1, 0.9),
        )
        est.fit(logged_data, treatment_col="action", feature_cols=["x1", "x2"])
        scores = est.predict(logged_data)
        assert scores.min() >= 0.1 - 1e-9
        assert scores.max() <= 0.9 + 1e-9

    def test_predict_for_action(self, logged_data):
        """predict_for_action returns per-sample scores for observed actions."""
        est = PropensityEstimator(calibration="none", clip_bounds=(0.01, 0.99))
        est.fit(logged_data, treatment_col="action", feature_cols=["x1", "x2"])
        scores = est.predict_for_action(
            logged_data,
            actions=np.asarray(logged_data["action"]),
        )
        assert scores.shape == (500,)
        assert np.all(scores > 0)


# ═══════════════════════════════════════════════════════════════
# Policy evaluator (IPS / SNIPS / DR)
# ═══════════════════════════════════════════════════════════════

class TestOfflinePolicyEvaluator:
    """Tests for OPE estimators."""

    def test_ips_basic(self):
        """IPS with uniform weights equals mean reward."""
        rewards = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        weights = np.ones(5)
        assert abs(estimate_ips(rewards, weights) - 3.0) < 1e-10

    def test_snips_basic(self):
        """SNIPS with uniform weights equals mean reward."""
        rewards = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        weights = np.ones(5)
        assert abs(estimate_snips(rewards, weights) - 3.0) < 1e-10

    def test_snips_normalisation(self):
        """SNIPS normalises by sum of weights."""
        rewards = np.array([1.0, 0.0, 0.0])
        weights = np.array([3.0, 1.0, 1.0])
        # SNIPS = (3*1 + 1*0 + 1*0) / (3+1+1) = 3/5 = 0.6
        assert abs(estimate_snips(rewards, weights) - 0.6) < 1e-10

    def test_dr_basic(self):
        """DR with perfect reward model equals the model's prediction."""
        rewards = np.array([1.0, 2.0, 3.0])
        weights = np.ones(3)
        reward_model = rewards.copy()  # perfect model
        new_policy_reward = np.array([2.0, 2.0, 2.0])
        # DR = mean(new_policy_reward + w * (rewards - reward_model))
        #    = mean(2 + 1*(r-r)) = mean(2) = 2.0
        result = estimate_dr(rewards, weights, reward_model, new_policy_reward)
        assert abs(result - 2.0) < 1e-10

    def test_evaluate_returns_ope_result(self, rng):
        """evaluate() returns a proper OPEResult with all estimators."""
        n = 200
        rewards = rng.standard_normal(n)
        actions = rng.integers(0, 2, size=n)
        new_probs = np.full(n, 0.5)
        prop_scores = np.full(n, 0.5)
        rm_preds = rng.standard_normal(n)
        np_rp = rng.standard_normal(n)

        evaluator = OfflinePolicyEvaluator(n_bootstrap=100, random_seed=42)
        result = evaluator.evaluate(
            rewards=rewards,
            actions=actions,
            new_policy_probs=new_probs,
            propensity_scores=prop_scores,
            reward_model_preds=rm_preds,
            new_policy_reward_preds=np_rp,
        )

        assert isinstance(result, OPEResult)
        assert "ips" in result.estimates
        assert "snips" in result.estimates
        assert "dr" in result.estimates
        assert result.n_samples == n
        assert result.effective_sample_size > 0

    def test_confidence_intervals(self, rng):
        """Bootstrap CIs should contain the point estimate."""
        n = 300
        rewards = rng.standard_normal(n) + 1.0
        new_probs = np.full(n, 0.5)
        prop_scores = np.full(n, 0.5)

        evaluator = OfflinePolicyEvaluator(
            estimator_types=["ips"],
            n_bootstrap=500,
            random_seed=42,
        )
        result = evaluator.evaluate(
            rewards=rewards,
            actions=rng.integers(0, 2, size=n),
            new_policy_probs=new_probs,
            propensity_scores=prop_scores,
        )

        lo, hi = result.confidence_intervals["ips"]
        assert lo <= result.estimates["ips"] <= hi

    def test_empty_data(self):
        """Empty arrays should return empty result without error."""
        evaluator = OfflinePolicyEvaluator()
        result = evaluator.evaluate(
            rewards=np.array([]),
            actions=np.array([]),
            new_policy_probs=np.array([]),
            propensity_scores=np.array([]),
        )
        assert result.n_samples == 0
        assert result.estimates == {}

    def test_multi_task_evaluation(self, rng):
        """evaluate_multi_task produces per-task results."""
        evaluator = OfflinePolicyEvaluator(
            estimator_types=["ips", "snips"],
            n_bootstrap=50,
        )
        task_data = {
            "ctr": {
                "rewards": rng.standard_normal(100),
                "actions": rng.integers(0, 2, size=100),
                "new_policy_probs": np.full(100, 0.6),
                "propensity_scores": np.full(100, 0.5),
            },
            "revenue": {
                "rewards": rng.standard_normal(100) + 2.0,
                "actions": rng.integers(0, 2, size=100),
                "new_policy_probs": np.full(100, 0.4),
                "propensity_scores": np.full(100, 0.5),
            },
        }
        results = evaluator.evaluate_multi_task(task_data)
        assert "ctr" in results
        assert "revenue" in results
        assert results["ctr"].task_name == "ctr"
        assert results["revenue"].task_name == "revenue"

    def test_dr_skipped_without_reward_model(self, rng):
        """DR is skipped gracefully when reward model predictions are absent."""
        evaluator = OfflinePolicyEvaluator(estimator_types=["ips", "dr"])
        result = evaluator.evaluate(
            rewards=rng.standard_normal(50),
            actions=rng.integers(0, 2, size=50),
            new_policy_probs=np.full(50, 0.5),
            propensity_scores=np.full(50, 0.5),
        )
        assert "ips" in result.estimates
        assert "dr" not in result.estimates


# ═══════════════════════════════════════════════════════════════
# ExposureSimulator
# ═══════════════════════════════════════════════════════════════

class TestExposureSimulator:
    """Tests for position-bias correction."""

    def test_cascade_examination_probs(self):
        """Cascade model produces geometrically decaying probabilities."""
        sim = ExposureSimulator(
            max_positions=5,
            decay_type="cascade",
            cascade_continue_prob=0.8,
        )
        probs = sim.examination_probabilities()
        assert len(probs) == 5
        assert abs(probs[0] - 1.0) < 1e-10  # normalised
        assert abs(probs[1] - 0.8) < 1e-10
        assert abs(probs[2] - 0.64) < 1e-10

    def test_geometric_examination_probs(self):
        """Geometric model produces gamma^k decay."""
        sim = ExposureSimulator(
            max_positions=4,
            decay_type="geometric",
            geometric_decay=0.9,
        )
        probs = sim.examination_probabilities()
        assert abs(probs[0] - 1.0) < 1e-10
        assert abs(probs[1] - 0.9) < 1e-10

    def test_reciprocal_examination_probs(self):
        """Reciprocal model produces 1/k^eta decay."""
        sim = ExposureSimulator(
            max_positions=4,
            decay_type="reciprocal",
            eta=1.0,
        )
        probs = sim.examination_probabilities()
        assert abs(probs[0] - 1.0) < 1e-10
        assert abs(probs[1] - 0.5) < 1e-10
        assert abs(probs[2] - 1.0 / 3.0) < 1e-4

    def test_adjust_rewards(self):
        """Position-1 rewards are unchanged; lower positions are up-weighted."""
        sim = ExposureSimulator(
            max_positions=5,
            decay_type="cascade",
            cascade_continue_prob=0.5,
        )
        rewards = np.array([1.0, 1.0, 1.0])
        positions = np.array([1, 2, 3])
        adjusted = sim.adjust_rewards(rewards, positions)
        assert abs(adjusted[0] - 1.0) < 1e-10
        assert adjusted[1] > adjusted[0]
        assert adjusted[2] > adjusted[1]

    def test_exposure_weights(self):
        """exposure_weights returns inverse examination probabilities."""
        sim = ExposureSimulator(max_positions=5, decay_type="cascade", cascade_continue_prob=0.5)
        positions = np.array([1, 2, 3])
        weights = sim.exposure_weights(positions)
        assert abs(weights[0] - 1.0) < 1e-10
        assert abs(weights[1] - 2.0) < 1e-10
        assert abs(weights[2] - 4.0) < 1e-10

    def test_simulate_exposures(self):
        """Simulated sessions produce the expected output structure."""
        sim = ExposureSimulator(max_positions=10, decay_type="cascade")
        result = sim.simulate_exposures(n_users=50, n_items_per_user=10)
        assert result["examined"].shape == (50, 10)
        assert result["clicked"].shape == (50, 10)
        assert result["positions"].shape == (50, 10)

    def test_estimate_position_bias(self):
        """Empirical position bias estimation from click data."""
        sim = ExposureSimulator(max_positions=5)
        clicks = np.array([1, 1, 1, 0, 0, 1, 0, 0, 0, 0])
        positions = np.array([1, 1, 1, 1, 1, 2, 2, 3, 3, 3])
        bias = sim.estimate_position_bias(clicks, positions)
        assert len(bias) == 5
        # pos-1 should be normalised to 1.0
        assert abs(bias[0] - 1.0) < 1e-10


# ═══════════════════════════════════════════════════════════════
# ModelCompetition
# ═══════════════════════════════════════════════════════════════

class TestModelCompetition:
    """Tests for the automated competition framework."""

    def test_challenger_promoted_when_better(self):
        """Challenger is promoted when it beats champion on all tasks."""
        comp = ModelCompetition(CompetitionConfig(min_improvement=0.005))
        result = comp.run(
            champion_metrics={"ctr": {"auc_roc": 0.80}},
            challenger_metrics={"ctr": {"auc_roc": 0.82}},
            primary_metrics={"ctr": "auc_roc"},
        )
        assert result["promote"] is True

    def test_challenger_rejected_when_worse(self):
        """Challenger is rejected when it loses on primary metric."""
        comp = ModelCompetition(CompetitionConfig(min_improvement=0.005))
        result = comp.run(
            champion_metrics={"ctr": {"auc_roc": 0.85}},
            challenger_metrics={"ctr": {"auc_roc": 0.84}},
            primary_metrics={"ctr": "auc_roc"},
        )
        assert result["promote"] is False

    def test_lower_is_better_metric(self):
        """Lower-is-better metrics (MAE) are handled correctly."""
        comp = ModelCompetition(CompetitionConfig(min_improvement=0.01))
        result = comp.run(
            champion_metrics={"revenue": {"mae": 1.0}},
            challenger_metrics={"revenue": {"mae": 0.95}},
            primary_metrics={"revenue": "mae"},
        )
        assert result["promote"] is True

    def test_multi_task_require_all(self):
        """With require_all_tasks=True, one failing task blocks promotion."""
        comp = ModelCompetition(CompetitionConfig(
            min_improvement=0.005,
            require_all_tasks=True,
        ))
        result = comp.run(
            champion_metrics={
                "ctr": {"auc_roc": 0.80},
                "revenue": {"mae": 1.0},
            },
            challenger_metrics={
                "ctr": {"auc_roc": 0.82},
                "revenue": {"mae": 1.1},  # worse
            },
            primary_metrics={"ctr": "auc_roc", "revenue": "mae"},
        )
        assert result["promote"] is False

    def test_majority_vote_mode(self):
        """With require_all_tasks=False, majority wins."""
        comp = ModelCompetition(CompetitionConfig(
            min_improvement=0.005,
            require_all_tasks=False,
        ))
        result = comp.run(
            champion_metrics={
                "ctr": {"auc_roc": 0.80},
                "revenue": {"mae": 1.0},
                "cvr": {"auc_roc": 0.70},
            },
            challenger_metrics={
                "ctr": {"auc_roc": 0.82},
                "revenue": {"mae": 1.1},  # worse
                "cvr": {"auc_roc": 0.73},
            },
            primary_metrics={"ctr": "auc_roc", "revenue": "mae", "cvr": "auc_roc"},
        )
        assert result["promote"] is True

    def test_paired_t_test_significant(self):
        """Paired t-test detects a significant difference."""
        rng = np.random.default_rng(42)
        champ = rng.standard_normal(1000)
        chall = champ + 0.15 + rng.standard_normal(1000) * 0.05  # clearly better with noise
        t_stat, p_val = paired_t_test(champ, chall)
        assert p_val < 0.001
        assert t_stat > 0

    def test_paired_t_test_not_significant(self):
        """Paired t-test reports high p-value for similar distributions."""
        rng = np.random.default_rng(42)
        champ = rng.standard_normal(100)
        chall = champ + rng.standard_normal(100) * 0.01
        _, p_val = paired_t_test(champ, chall)
        assert p_val > 0.01

    def test_bootstrap_test(self):
        """Paired bootstrap test detects a significant improvement."""
        rng = np.random.default_rng(42)
        champ = rng.standard_normal(200)
        chall = champ + 0.3
        diff, p_val = paired_bootstrap_test(champ, chall, n_bootstrap=500, rng=rng)
        assert diff > 0
        assert p_val < 0.05

    def test_significance_blocks_promotion(self):
        """Insignificant improvement blocks promotion even above threshold."""
        rng = np.random.default_rng(42)
        comp = ModelCompetition(CompetitionConfig(
            min_improvement=0.001,
            significance_level=0.01,
        ))
        # Tiny difference with high noise -> not significant
        champ_samples = rng.standard_normal(50)
        chall_samples = champ_samples + rng.standard_normal(50) * 0.001

        result = comp.run(
            champion_metrics={"ctr": {"auc_roc": 0.80}},
            challenger_metrics={"ctr": {"auc_roc": 0.81}},
            primary_metrics={"ctr": "auc_roc"},
            champion_per_sample={"ctr": champ_samples},
            challenger_per_sample={"ctr": chall_samples},
        )
        assert result["promote"] is False

    def test_run_from_evaluator_reports(self):
        """Integration with ModelEvaluator report format."""
        comp = ModelCompetition(CompetitionConfig(min_improvement=0.005))
        champion_report = {
            "tasks": {
                "ctr": {
                    "metrics": {"auc_roc": 0.80, "f1": 0.70},
                    "primary_metric": "auc_roc",
                },
            },
        }
        challenger_report = {
            "tasks": {
                "ctr": {
                    "metrics": {"auc_roc": 0.82, "f1": 0.72},
                    "primary_metric": "auc_roc",
                },
            },
        }
        result = comp.run_from_evaluator_reports(champion_report, challenger_report)
        assert result["promote"] is True

    def test_no_overlapping_tasks(self):
        """Gracefully handles no overlapping tasks."""
        comp = ModelCompetition()
        result = comp.run(
            champion_metrics={"ctr": {"auc_roc": 0.8}},
            challenger_metrics={"revenue": {"mae": 0.5}},
        )
        assert result["promote"] is False
        assert "No overlapping tasks" in result["summary"]

    def test_ope_comparison_blocks_promotion(self):
        """OPE results can block promotion when challenger is worse."""
        comp = ModelCompetition(CompetitionConfig(
            min_improvement=0.005,
            min_effective_sample_size=10.0,
        ))
        result = comp.run(
            champion_metrics={"ctr": {"auc_roc": 0.80}},
            challenger_metrics={"ctr": {"auc_roc": 0.82}},
            primary_metrics={"ctr": "auc_roc"},
            ope_results={
                "ctr": {
                    "champion": 0.50,
                    "challenger": 0.45,  # worse on OPE
                    "effective_sample_size": 100.0,
                },
            },
        )
        assert result["promote"] is False
