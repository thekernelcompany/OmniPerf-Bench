"""Pydantic schemas for soft metrics analysis.

Defines validated models for academic-grade metrics extraction
from agent benchmark runs. Uses Pydantic for input/output validation.
"""

from __future__ import annotations

import json
from datetime import datetime
from enum import Enum
from typing import List, Dict, Any, Optional

from pydantic import BaseModel, Field, computed_field, model_validator


class ApproachCategory(str, Enum):
    """Categories describing agent approach to the task."""
    SYSTEMATIC = "systematic"
    DIRECT_EDIT = "direct_edit"
    EXPLORATION_HEAVY = "exploration_heavy"
    TRIAL_ERROR = "trial_error"
    MINIMAL = "minimal"
    UNKNOWN = "unknown"


class FailureCategory(str, Enum):
    """Categories of failure reasons."""
    UNDERSTANDING = "understanding"
    EXECUTION = "execution"
    TIMEOUT = "timeout"
    API_ERROR = "api_error"
    NO_PATCH = "no_patch"
    NONE = "none"
    UNKNOWN = "unknown"


class ToolUsagePattern(str, Enum):
    """Patterns of tool usage."""
    BASH_HEAVY = "bash_heavy"
    EDITOR_FOCUSED = "editor_focused"
    READ_HEAVY = "read_heavy"
    BALANCED = "balanced"
    UNKNOWN = "unknown"


class RunMetadata(BaseModel):
    """Metadata extracted from run directory path and files."""
    repo: str = Field(description="Repository name (vllm, sglang)")
    agent: str = Field(description="Agent framework (trae, codex, openhands)")
    model: str = Field(description="Model short name (gpt-5, claude-sonnet-45)")
    model_full: str = Field(default="", description="Full model identifier")
    timestamp: str = Field(description="Run timestamp (YYYY-MM-DD_HH-MM-SS)")
    item_id: str = Field(description="Unique item identifier")
    task_id: str = Field(default="", description="Task identifier")
    commits: Dict[str, str] = Field(default_factory=dict, description="pre/human commit SHAs")
    source_files: Dict[str, str] = Field(default_factory=dict, description="Paths to source artifacts")

    model_config = {"extra": "ignore"}


class ToolDistribution(BaseModel):
    """Distribution of tool calls by type."""
    bash: int = Field(default=0, ge=0, description="Shell command executions")
    editor: int = Field(default=0, ge=0, description="File edit operations")
    read: int = Field(default=0, ge=0, description="File read operations")
    search: int = Field(default=0, ge=0, description="Code search grep/glob operations")
    web_search: int = Field(default=0, ge=0, description="Web search operations")
    other: int = Field(default=0, ge=0, description="Other tool calls (captured below)")
    other_details: Dict[str, int] = Field(default_factory=dict, description="Breakdown of 'other' tool calls")

    @computed_field
    @property
    def total(self) -> int:
        """Total tool calls."""
        return self.bash + self.editor + self.read + self.search + self.web_search + self.other

    model_config = {"extra": "allow"}


class PatchMetrics(BaseModel):
    """Metrics about the generated patch."""
    generated: bool = Field(default=False, description="Whether a patch was produced")
    lines_added: int = Field(default=0, ge=0, description="Lines added in patch")
    lines_removed: int = Field(default=0, ge=0, description="Lines removed in patch")
    hunks: int = Field(default=0, ge=0, description="Number of change hunks")
    files_changed: List[str] = Field(default_factory=list, description="Files modified")
    files_allowed: List[str] = Field(default_factory=list, description="Files allowed to modify")
    files_disallowed: List[str] = Field(default_factory=list, description="Files modified but not allowed")
    compliance_ok: bool = Field(default=True, description="Whether changes comply with constraints")

    @computed_field
    @property
    def total_lines(self) -> int:
        """Total lines changed."""
        return self.lines_added + self.lines_removed


class QuantitativeMetrics(BaseModel):
    """Metrics directly extracted from files (no LLM needed)."""

    # Timing
    duration_s: float = Field(default=0.0, ge=0, description="Total run duration in seconds")
    time_to_first_edit_s: float = Field(default=0.0, ge=0, description="Seconds until first edit")

    # Execution
    step_count: int = Field(default=0, ge=0, description="Number of agent steps")
    iteration_count: int = Field(default=0, ge=0, description="Edit-test iteration cycles")

    # Tool usage
    tool_calls: ToolDistribution = Field(default_factory=ToolDistribution, description="Tool call breakdown")

    # Tokens
    input_tokens: int = Field(default=0, ge=0, description="Total input tokens consumed")
    output_tokens: int = Field(default=0, ge=0, description="Total output tokens generated")

    # Patch
    patch: PatchMetrics = Field(default_factory=PatchMetrics, description="Patch metrics")

    # Errors
    error_count: int = Field(default=0, ge=0, description="Number of errors encountered")
    warning_count: int = Field(default=0, ge=0, description="Number of warnings")
    exception_types: List[str] = Field(default_factory=list, description="Types of exceptions seen")

    # Status
    status: str = Field(default="unknown", description="Final run status")
    returncode: int = Field(default=0, description="Process return code")

    model_config = {"extra": "ignore"}


class QualityScore(BaseModel):
    """A quality score with sub-scores and justification."""
    score: float = Field(default=0.0, ge=0, le=10, description="Overall score (0-10)")
    sub_scores: Dict[str, float] = Field(default_factory=dict, description="Component sub-scores")
    justification: str = Field(default="", description="Explanation for score")


class QualitativeScores(BaseModel):
    """Quality scores from LLM analysis (0-10 scale)."""

    code_understanding: QualityScore = Field(
        default_factory=QualityScore,
        description="How well agent understood the codebase"
    )
    task_alignment: QualityScore = Field(
        default_factory=QualityScore,
        description="How well agent aligned with task requirements"
    )
    approach_quality: QualityScore = Field(
        default_factory=QualityScore,
        description="Quality of agent's problem-solving approach"
    )
    execution_quality: QualityScore = Field(
        default_factory=QualityScore,
        description="Quality of code changes and execution"
    )

    @computed_field
    @property
    def overall_score(self) -> float:
        """Weighted average of all scores."""
        return round((
            self.code_understanding.score * 0.25 +
            self.task_alignment.score * 0.25 +
            self.approach_quality.score * 0.25 +
            self.execution_quality.score * 0.25
        ), 2)

    model_config = {"extra": "ignore"}


class CategoricalMetrics(BaseModel):
    """Categorical classifications from LLM analysis."""
    approach_category: ApproachCategory = Field(
        default=ApproachCategory.UNKNOWN,
        description="How agent approached the task"
    )
    tool_usage_pattern: ToolUsagePattern = Field(
        default=ToolUsagePattern.UNKNOWN,
        description="Pattern of tool usage"
    )
    failure_category: FailureCategory = Field(
        default=FailureCategory.NONE,
        description="Category of failure (if failed)"
    )

    model_config = {"extra": "ignore"}


class FreeFormAnalysis(BaseModel):
    """Free-form text analysis from LLM."""
    summary: str = Field(default="", description="Brief summary of the run")
    key_decisions: List[str] = Field(default_factory=list, description="Important decisions made")
    optimization_techniques: List[str] = Field(default_factory=list, description="Optimization methods used")
    missed_opportunities: List[str] = Field(default_factory=list, description="What could have been better")
    error_recovery_strategy: str = Field(default="", description="How errors were handled")
    strengths: List[str] = Field(default_factory=list, description="What agent did well")
    weaknesses: List[str] = Field(default_factory=list, description="Areas for improvement")
    recommendations: List[str] = Field(default_factory=list, description="Suggestions for improvement")

    model_config = {"extra": "ignore"}


class LLMAnalysisResult(BaseModel):
    """Result from LLM analysis call."""
    model: str = Field(default="google/gemini-3-flash-preview", description="LLM model used")
    provider: str = Field(default="openrouter", description="API provider")
    thinking_enabled: bool = Field(default=True, description="Whether thinking mode was used")

    # Token usage
    prompt_tokens: int = Field(default=0, ge=0)
    completion_tokens: int = Field(default=0, ge=0)
    total_tokens: int = Field(default=0, ge=0)

    # Timing
    timestamp: str = Field(default="")
    duration_s: float = Field(default=0.0, ge=0)

    # Content (stored separately in files)
    prompt: str = Field(default="", exclude=True)
    thinking_content: str = Field(default="", exclude=True)
    response_content: str = Field(default="", exclude=True)

    model_config = {"extra": "ignore"}


# =============================================================================
# V3 Academic-Grade Metrics Models
# =============================================================================

class InteractionMetrics(BaseModel):
    """Per-LLM-interaction metrics extracted from trajectory.json."""
    step_index: int = Field(description="Step number in the agent trajectory")
    timestamp: str = Field(default="", description="ISO timestamp of the interaction")
    elapsed_s: float = Field(default=0.0, ge=0, description="Seconds since run start")

    # Token usage from LLM response
    input_tokens: int = Field(default=0, ge=0, description="Input tokens for this interaction")
    output_tokens: int = Field(default=0, ge=0, description="Output tokens for this interaction")
    cache_read_tokens: int = Field(default=0, ge=0, description="Tokens read from cache")
    cache_creation_tokens: int = Field(default=0, ge=0, description="Tokens used for cache creation")
    reasoning_tokens: int = Field(default=0, ge=0, description="Tokens used for reasoning/thinking")

    # Tool usage
    tool_calls: List[str] = Field(default_factory=list, description="Tool names called in this step")
    tool_success: int = Field(default=0, ge=0, description="Successful tool calls")
    tool_failure: int = Field(default=0, ge=0, description="Failed tool calls")

    model_config = {"extra": "ignore"}


class TrajectoryMetrics(BaseModel):
    """Aggregated metrics from trajectory.json for academic analysis."""
    # Timing
    start_time: str = Field(default="", description="ISO timestamp of run start")
    end_time: str = Field(default="", description="ISO timestamp of run end")
    total_duration_s: float = Field(default=0.0, ge=0, description="Total run duration in seconds")

    # Step timing analysis
    step_count: int = Field(default=0, ge=0, description="Total number of agent steps")
    step_durations: List[float] = Field(default_factory=list, description="Duration per step in seconds")
    avg_step_duration_s: float = Field(default=0.0, ge=0, description="Average step duration")
    max_step_duration_s: float = Field(default=0.0, ge=0, description="Maximum step duration")
    min_step_duration_s: float = Field(default=0.0, ge=0, description="Minimum step duration")

    # Token aggregates
    total_input_tokens: int = Field(default=0, ge=0, description="Sum of input tokens across all interactions")
    total_output_tokens: int = Field(default=0, ge=0, description="Sum of output tokens across all interactions")
    total_cache_read_tokens: int = Field(default=0, ge=0, description="Sum of cache read tokens")
    total_cache_creation_tokens: int = Field(default=0, ge=0, description="Sum of cache creation tokens")
    total_reasoning_tokens: int = Field(default=0, ge=0, description="Sum of reasoning tokens")

    # Tool usage aggregates
    total_tool_calls: int = Field(default=0, ge=0, description="Total tool calls across all steps")
    successful_tool_calls: int = Field(default=0, ge=0, description="Successful tool calls")
    failed_tool_calls: int = Field(default=0, ge=0, description="Failed tool calls")
    tool_success_rate: float = Field(default=0.0, ge=0, le=1, description="Success rate (0-1)")
    tool_call_counts: Dict[str, int] = Field(default_factory=dict, description="Count per tool name")

    # Per-interaction breakdown
    interactions: List[InteractionMetrics] = Field(default_factory=list, description="Per-step metrics")

    model_config = {"extra": "ignore"}

    @computed_field
    @property
    def total_tokens(self) -> int:
        """Total tokens consumed across all interactions."""
        return self.total_input_tokens + self.total_output_tokens


class PatchSimilarityMetrics(BaseModel):
    """Comparison metrics between agent patch and human reference patch."""
    # Availability
    human_patch_available: bool = Field(default=False, description="Whether human patch was found")
    human_patch_source: str = Field(default="", description="Source of human patch (dataset file or commit)")

    # File-level comparison
    agent_files: List[str] = Field(default_factory=list, description="Files modified by agent")
    human_files: List[str] = Field(default_factory=list, description="Files modified by human")
    common_files: List[str] = Field(default_factory=list, description="Files modified by both")
    agent_only_files: List[str] = Field(default_factory=list, description="Files only agent modified")
    human_only_files: List[str] = Field(default_factory=list, description="Files only human modified")
    file_overlap_pct: float = Field(default=0.0, ge=0, le=100, description="% of human files also modified by agent")

    # Line-level statistics
    agent_lines_added: int = Field(default=0, ge=0, description="Lines added by agent")
    agent_lines_removed: int = Field(default=0, ge=0, description="Lines removed by agent")
    human_lines_added: int = Field(default=0, ge=0, description="Lines added by human")
    human_lines_removed: int = Field(default=0, ge=0, description="Lines removed by human")

    # Overlap metrics
    matching_additions: int = Field(default=0, ge=0, description="Agent lines that match human additions")
    matching_removals: int = Field(default=0, ge=0, description="Agent lines that match human removals")
    line_overlap_pct: float = Field(default=0.0, ge=0, le=100, description="% of human lines matched by agent")

    # Semantic similarity
    approach_similarity_score: float = Field(default=0.0, ge=0, le=10, description="Similarity score (0-10)")

    model_config = {"extra": "ignore"}


class LLMRawScores(BaseModel):
    """Raw structured output from LLM analysis (preserved verbatim)."""
    # Quality scores with sub-components
    quality_scores: Dict[str, Any] = Field(default_factory=dict, description="Full quality scores with sub-scores")

    # Categorical classifications
    categorical: Dict[str, str] = Field(default_factory=dict, description="Categorical classifications")

    # Quantitative assessment from LLM
    quantitative_assessment: Dict[str, Any] = Field(default_factory=dict, description="Tool counts, errors, etc.")

    # Tool usage analysis
    tool_usage_analysis: Dict[str, Any] = Field(default_factory=dict, description="Detailed tool usage breakdown")

    # Detailed analysis text
    detailed_analysis: Dict[str, Any] = Field(default_factory=dict, description="Summary, strengths, weaknesses")

    # Recommendations
    recommendations: Dict[str, Any] = Field(default_factory=dict, description="Improvement suggestions")

    model_config = {"extra": "allow"}


class RunAnalysis(BaseModel):
    """Complete analysis for a single run.

    Combines metadata, quantitative metrics, and LLM-extracted
    qualitative metrics into a comprehensive analysis.
    """

    schema_version: str = Field(default="3.0", description="Schema version")

    # Core data
    meta: RunMetadata
    quantitative: QuantitativeMetrics = Field(default_factory=QuantitativeMetrics)
    qualitative: QualitativeScores = Field(default_factory=QualitativeScores)
    categorical: CategoricalMetrics = Field(default_factory=CategoricalMetrics)
    analysis: FreeFormAnalysis = Field(default_factory=FreeFormAnalysis)

    # LLM details
    llm: LLMAnalysisResult = Field(default_factory=LLMAnalysisResult)

    # V3: Academic-grade metrics (optional - populated when available)
    trajectory: Optional[TrajectoryMetrics] = Field(
        default=None,
        description="Detailed trajectory metrics from trajectory.json"
    )
    patch_similarity: Optional[PatchSimilarityMetrics] = Field(
        default=None,
        description="Agent vs human patch comparison metrics"
    )
    llm_raw: Optional[LLMRawScores] = Field(
        default=None,
        description="Raw LLM analysis output (verbatim)"
    )

    # V4: Patch quality analysis (categories + discussion, no scores)
    patch_quality: Optional[PatchQualityAnalysis] = Field(
        default=None,
        description="LLM-based patch quality assessment using categories and discussion"
    )

    # Timestamps
    analyzed_at: str = Field(default_factory=lambda: datetime.now().isoformat())
    analysis_duration_s: float = Field(default=0.0, ge=0)

    model_config = {"extra": "ignore"}

    def to_json(self, indent: int = 2) -> str:
        """Convert to JSON string."""
        return self.model_dump_json(indent=indent)

    def metrics_summary(self) -> Dict[str, Any]:
        """Return clean summary of key metrics for quick reference."""
        summary = {
            "run": {
                "repo": self.meta.repo,
                "agent": self.meta.agent,
                "model": self.meta.model,
                "item_id": self.meta.item_id,
                "task_id": self.meta.task_id,
            },
            "execution": {
                "status": self.quantitative.status,
                "duration_s": round(self.quantitative.duration_s, 1),
                "steps": self.quantitative.step_count,
                "tool_calls": self.quantitative.tool_calls.total,
                "errors": self.quantitative.error_count,
            },
            "tool_breakdown": {
                "bash": self.quantitative.tool_calls.bash,
                "editor": self.quantitative.tool_calls.editor,
                "read": self.quantitative.tool_calls.read,
                "search": self.quantitative.tool_calls.search,
                "other": self.quantitative.tool_calls.other,
            },
            "patch": {
                "generated": self.quantitative.patch.generated,
                "lines_added": self.quantitative.patch.lines_added,
                "lines_removed": self.quantitative.patch.lines_removed,
                "files_changed": len(self.quantitative.patch.files_changed),
                "compliance_ok": self.quantitative.patch.compliance_ok,
            },
            "scores": {
                "code_understanding": self.qualitative.code_understanding.score,
                "task_alignment": self.qualitative.task_alignment.score,
                "approach_quality": self.qualitative.approach_quality.score,
                "execution_quality": self.qualitative.execution_quality.score,
                "overall": self.qualitative.overall_score,
            },
            "classification": {
                "approach": self.categorical.approach_category.value,
                "tool_pattern": self.categorical.tool_usage_pattern.value,
                "failure_reason": self.categorical.failure_category.value,
            },
            "insights": {
                "summary": self.analysis.summary,
                "strengths": self.analysis.strengths[:3],  # Top 3
                "weaknesses": self.analysis.weaknesses[:3],  # Top 3
            },
        }

        # V3: Add trajectory metrics if available
        if self.trajectory:
            summary["trajectory"] = {
                "total_input_tokens": self.trajectory.total_input_tokens,
                "total_output_tokens": self.trajectory.total_output_tokens,
                "total_tokens": self.trajectory.total_tokens,
                "total_reasoning_tokens": self.trajectory.total_reasoning_tokens,
                "avg_step_duration_s": round(self.trajectory.avg_step_duration_s, 2),
                "tool_success_rate": round(self.trajectory.tool_success_rate, 3),
                "interaction_count": len(self.trajectory.interactions),
            }

        # V3: Add patch similarity if available
        if self.patch_similarity:
            summary["patch_similarity"] = {
                "human_patch_available": self.patch_similarity.human_patch_available,
                "file_overlap_pct": round(self.patch_similarity.file_overlap_pct, 1),
                "line_overlap_pct": round(self.patch_similarity.line_overlap_pct, 1),
                "approach_similarity": round(self.patch_similarity.approach_similarity_score, 1),
            }

        return summary

    model_config = {"extra": "ignore"}


class TaskAnalysis(BaseModel):
    """Analysis of the task complexity and domain."""
    domain: str = Field(description="Primary domain (e.g., 'compute', 'memory', 'io', 'concurrency')")
    complexity: str = Field(description="Task complexity rating (low/medium/high/extreme)")
    description: str = Field(description="Detailed description of what the task involves")

    model_config = {"extra": "ignore"}


# =============================================================================
# V4 Patch Quality Analysis (Categories + Discussion, NO SCORES)
# Inspired by GSO Benchmark (arxiv:2505.23671v3)
# =============================================================================


class BottleneckTargetCategory(str, Enum):
    """Categories for bottleneck targeting comparison."""
    SAME_TARGET = "same_target"
    RELATED_TARGET = "related_target"
    DIFFERENT_TARGET = "different_target"
    NO_OPTIMIZATION = "no_optimization"
    OTHER = "other"


class OptimizationTechnique(str, Enum):
    """Optimization technique categories."""
    ALGORITHMIC = "algorithmic"
    MEMORY_OPTIMIZATION = "memory_optimization"
    PARALLELIZATION = "parallelization"
    API_LIBRARY = "api_library"
    LAZY_COMPUTATION = "lazy_computation"
    BATCHING = "batching"
    LOW_LEVEL = "low_level"
    COMPILER_OPTIMIZATION = "compiler_optimization"
    OTHER = "other"  # Must explain in discussion


class PatchApproachCategory(str, Enum):
    """Categories for approach comparison (distinct from existing ApproachCategory)."""
    SAME_APPROACH = "same_approach"
    SIMILAR_APPROACH = "similar_approach"
    VALID_ALTERNATIVE = "valid_alternative"
    PARTIAL_SOLUTION = "partial_solution"
    INEFFECTIVE = "ineffective"
    HARMFUL = "harmful"
    OTHER = "other"


class SpeedupLikelihood(str, Enum):
    """Categories for speedup likelihood opinion."""
    LIKELY_SIMILAR = "likely_similar"
    LIKELY_PARTIAL = "likely_partial"
    UNCERTAIN = "uncertain"
    LIKELY_INEFFECTIVE = "likely_ineffective"
    LIKELY_REGRESSION = "likely_regression"
    OTHER = "other"


class PatchFailureMode(str, Enum):
    """Failure mode categories based on GSO paper taxonomy."""
    LOCALIZATION_FAILURE = "localization_failure"
    TECHNIQUE_MISMATCH = "technique_mismatch"
    INCOMPLETE_IMPLEMENTATION = "incomplete_implementation"
    COMPLEXITY_AVOIDANCE = "complexity_avoidance"
    OVERCOMPLICATED = "overcomplicated"
    NOT_APPLICABLE = "not_applicable"
    OTHER = "other"


class BottleneckTargetAnalysis(BaseModel):
    """Analysis of whether agent targets the same bottleneck as human."""
    category: BottleneckTargetCategory
    human_target: str = Field(description="What bottleneck the human patch targets")
    agent_target: str = Field(description="What bottleneck the agent patch targets")
    discussion: str = Field(description="Explanation of bottleneck comparison")

    model_config = {"extra": "ignore"}


class OptimizationTechniqueAnalysis(BaseModel):
    """Analysis of optimization techniques used by human vs agent."""
    human_techniques: List[OptimizationTechnique] = Field(default_factory=list)
    agent_techniques: List[OptimizationTechnique] = Field(default_factory=list)
    technique_overlap: bool = Field(default=False)
    discussion: str = Field(default="", description="Comparison of techniques used")

    model_config = {"extra": "ignore"}


class ApproachComparisonAnalysis(BaseModel):
    """Analysis of how agent's approach compares to human's."""
    category: PatchApproachCategory
    discussion: str = Field(description="Why this category, key differences")

    model_config = {"extra": "ignore"}


class SpeedupLikelihoodAnalysis(BaseModel):
    """Opinion on likely speedup based on patch analysis."""
    category: SpeedupLikelihood
    discussion: str = Field(description="Reasoning, what would need benchmarking")

    model_config = {"extra": "ignore"}


class FailureModeAnalysis(BaseModel):
    """Analysis of failure mode if agent didn't match human."""
    category: PatchFailureMode
    discussion: str = Field(description="What went wrong, if applicable")

    model_config = {"extra": "ignore"}


class PatchObservations(BaseModel):
    """Free-form observations about patch comparison."""
    key_differences: List[str] = Field(default_factory=list)
    agent_strengths: List[str] = Field(default_factory=list)
    agent_weaknesses: List[str] = Field(default_factory=list)
    benchmark_needed: str = Field(default="", description="What benchmark would verify performance")

    model_config = {"extra": "ignore"}



class LibraryFailureAnalysis(BaseModel):
    """Analysis of which libraries failed and why."""
    responsible_libraries: List[str] = Field(default_factory=list, description="List of libraries that caused issues (e.g. 'pytorch', 'cuda', 'triton')")
    failure_reason: str = Field(default="", description="Detailed reason for the library failure")

    model_config = {"extra": "ignore"}


class PatchQualityAnalysis(BaseModel):
    """LLM-based patch quality analysis - categories and discussion only, no scores.

    This complements the rule-based PatchSimilarityMetrics with LLM-driven
    categorical assessment and expert discussion.
    """
    human_patch_available: bool = Field(default=False)
    analysis_model: str = Field(default="")
    analysis_tokens: int = Field(default=0, ge=0)

    task_analysis: Optional[TaskAnalysis] = Field(default=None)
    library_failure: Optional[LibraryFailureAnalysis] = Field(default=None)
    bottleneck_target: Optional[BottleneckTargetAnalysis] = Field(default=None)
    optimization_techniques: Optional[OptimizationTechniqueAnalysis] = Field(default=None)
    approach_comparison: Optional[ApproachComparisonAnalysis] = Field(default=None)
    speedup_likelihood: Optional[SpeedupLikelihoodAnalysis] = Field(default=None)
    failure_mode: Optional[FailureModeAnalysis] = Field(default=None)
    observations: Optional[PatchObservations] = Field(default=None)

    model_config = {"extra": "ignore"}


# Legacy compatibility aliases
def _legacy_to_dict(obj) -> Dict[str, Any]:
    """Convert Pydantic model to dict (legacy compatibility)."""
    if hasattr(obj, "model_dump"):
        return obj.model_dump()
    return dict(obj)


# For backward compatibility with old dataclass-based code
RunMetadata.to_dict = lambda self: self.model_dump()
QuantitativeMetrics.to_dict = lambda self: self.model_dump()
QualitativeScores.to_dict = lambda self: self.model_dump()
CategoricalMetrics.to_dict = lambda self: self.model_dump()
FreeFormAnalysis.to_dict = lambda self: self.model_dump()
LLMAnalysisResult.to_dict = lambda self: self.model_dump()
RunAnalysis.to_dict = lambda self: self.model_dump()
