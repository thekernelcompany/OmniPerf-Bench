"""Core data models for OmniPerf-Bench."""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional


@dataclass
class CommitExtraction:
    """A single commit extraction from JSON input."""
    commit_hash: str
    parent_hash: str
    message: str
    diff_text: str
    affected_files: List[str] = field(default_factory=list)
    affected_apis: List[str] = field(default_factory=list)
    repo_name: Optional[str] = None

    @classmethod
    def from_json(cls, data: Dict[str, Any]) -> "CommitExtraction":
        """Load from JSON dict (supports multiple JSON formats)."""
        return cls(
            commit_hash=data.get("commit_hash") or data.get("hash", ""),
            parent_hash=data.get("parent_hash") or data.get("parent", ""),
            message=data.get("message") or data.get("commit_message", ""),
            diff_text=data.get("diff_text") or data.get("diff", ""),
            affected_files=data.get("affected_files") or data.get("files_changed", []),
            affected_apis=data.get("affected_apis") or data.get("apis", []),
            repo_name=data.get("repo_name") or data.get("repo"),
        )


@dataclass
class PerformanceTest:
    """A generated performance test."""
    test_id: str
    code: str
    target_api: Optional[str] = None
    generation_model: Optional[str] = None


@dataclass
class TimingResult:
    """Timing results from test execution."""
    test_id: str
    commit: str
    durations: List[float] = field(default_factory=list)
    mean_duration: Optional[float] = None
    std_duration: Optional[float] = None
    error: Optional[str] = None

    def compute_stats(self) -> None:
        """Compute mean and std from durations."""
        if self.durations:
            self.mean_duration = sum(self.durations) / len(self.durations)
            if len(self.durations) > 1:
                variance = sum((d - self.mean_duration) ** 2 for d in self.durations) / len(self.durations)
                self.std_duration = variance ** 0.5


@dataclass
class ExecutionResult:
    """Results from executing tests on a commit."""
    commit_hash: str
    base_results: List[TimingResult] = field(default_factory=list)
    head_results: List[TimingResult] = field(default_factory=list)
    main_results: List[TimingResult] = field(default_factory=list)


@dataclass
class DatasetRecord:
    """A single record in the output dataset (canonical format)."""
    instance_id: str
    repo: str
    base_commit: str
    head_commit: str
    patch: str
    efficiency_test: List[str]
    duration_changes: List[Dict[str, List[float]]]
    human_performance: float
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())

    # Optional fields
    test_patch: Optional[str] = None
    setup_commands: List[str] = field(default_factory=list)
    install_commands: List[str] = field(default_factory=list)
    hints_text: Optional[str] = None
    api: Optional[str] = None
    version: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "instance_id": self.instance_id,
            "repo": self.repo,
            "base_commit": self.base_commit,
            "head_commit": self.head_commit,
            "patch": self.patch,
            "efficiency_test": self.efficiency_test,
            "duration_changes": self.duration_changes,
            "human_performance": self.human_performance,
            "created_at": self.created_at,
            "test_patch": self.test_patch,
            "setup_commands": self.setup_commands,
            "install_commands": self.install_commands,
            "hints_text": self.hints_text,
            "api": self.api,
            "version": self.version,
        }
