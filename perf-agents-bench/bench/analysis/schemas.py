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
    search: int = Field(default=0, ge=0, description="Search/grep operations")
    other: int = Field(default=0, ge=0, description="Other tool calls")

    @computed_field
    @property
    def total(self) -> int:
        """Total tool calls."""
        return self.bash + self.editor + self.read + self.search + self.other

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
    model: str = Field(default="google/gemini-3-pro-preview", description="LLM model used")
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
