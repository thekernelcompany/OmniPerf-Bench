"""Soft metrics analysis module for agent benchmark runs.

Uses Gemini 3 Pro via OpenRouter to extract academically-relevant metrics
from agent trajectories, logs, and patches.
"""

from .schemas import (
    RunMetadata,
    QuantitativeMetrics,
    QualitativeScores,
    CategoricalMetrics,
    FreeFormAnalysis,
    LLMAnalysisResult,
    RunAnalysis,
    # V4 Patch Quality Analysis schemas
    PatchQualityAnalysis,
    BottleneckTargetCategory,
    BottleneckTargetAnalysis,
    OptimizationTechnique,
    OptimizationTechniqueAnalysis,
    PatchApproachCategory,
    ApproachComparisonAnalysis,
    SpeedupLikelihood,
    SpeedupLikelihoodAnalysis,
    PatchFailureMode,
    FailureModeAnalysis,
    PatchObservations,
)
from .analyzer import SoftMetricsAnalyzer
from .run_loader import RunLoader
from .output_writer import OutputWriter
from .prompts import build_patch_quality_prompt

__all__ = [
    "RunMetadata",
    "QuantitativeMetrics",
    "QualitativeScores",
    "CategoricalMetrics",
    "FreeFormAnalysis",
    "LLMAnalysisResult",
    "RunAnalysis",
    "SoftMetricsAnalyzer",
    "RunLoader",
    "OutputWriter",
    # V4 exports
    "PatchQualityAnalysis",
    "BottleneckTargetCategory",
    "BottleneckTargetAnalysis",
    "OptimizationTechnique",
    "OptimizationTechniqueAnalysis",
    "PatchApproachCategory",
    "ApproachComparisonAnalysis",
    "SpeedupLikelihood",
    "SpeedupLikelihoodAnalysis",
    "PatchFailureMode",
    "FailureModeAnalysis",
    "PatchObservations",
    "build_patch_quality_prompt",
]
