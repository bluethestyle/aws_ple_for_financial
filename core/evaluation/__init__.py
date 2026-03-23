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
from core.evaluation.integrated_gradients import IntegratedGradients
from core.evaluation.expert_redundancy import (
    ExpertRedundancyAnalyzer,
    RedundancyResult,
)
from core.evaluation.gate_analyzer import GateAnalyzer, GateAnalysisResult
from core.evaluation.multidisciplinary_interpreter import (
    MultidisciplinaryInterpreter,
    FeatureInterpretation as MultiFeatureInterpretation,
)
from core.evaluation.xai_quality_evaluator import XAIQualityEvaluator, XAIReport
from core.evaluation.model_card import ModelCardGenerator
from core.evaluation.template_reason_engine import TemplateReasonEngine
from core.evaluation.counterfactual import (
    CounterfactualEvaluator,
    PolicyEvaluationResult,
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
    "IntegratedGradients",
    "ExpertRedundancyAnalyzer",
    "RedundancyResult",
    "GateAnalyzer",
    "GateAnalysisResult",
    "MultidisciplinaryInterpreter",
    "MultiFeatureInterpretation",
    "XAIQualityEvaluator",
    "XAIReport",
    "ModelCardGenerator",
    "TemplateReasonEngine",
    "CounterfactualEvaluator",
    "PolicyEvaluationResult",
]
