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
)
from .analyzer import SoftMetricsAnalyzer
from .run_loader import RunLoader
from .output_writer import OutputWriter

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
]
