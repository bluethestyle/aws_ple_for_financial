"""
Offline policy evaluation for counterfactual model comparison.

Provides:
- Propensity score estimation (LightGBM + calibration)
- Offline policy estimators (IPS / SNIPS / DR)
- Exposure & position-bias simulation (cascade model)
- Automated model competition framework with Go/No-Go gating
"""

from core.evaluation.propensity import PropensityEstimator, PropensityConfig
from core.evaluation.policy_evaluator import (
    OfflinePolicyEvaluator,
    OPEResult,
    EstimatorType,
)
from core.evaluation.exposure_simulator import ExposureSimulator, ExposureConfig
from core.evaluation.model_competition import (
    ModelCompetition,
    CompetitionConfig,
    ModelCandidate,
    CompetitionResult,
)

__all__ = [
    "PropensityEstimator",
    "PropensityConfig",
    "OfflinePolicyEvaluator",
    "OPEResult",
    "EstimatorType",
    "ExposureSimulator",
    "ExposureConfig",
    "ModelCompetition",
    "CompetitionConfig",
    "ModelCandidate",
    "CompetitionResult",
]
