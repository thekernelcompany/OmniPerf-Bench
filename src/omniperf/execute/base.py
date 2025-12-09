"""Base executor interface for running performance tests."""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

from omniperf.data import CommitExtraction, PerformanceTest, ExecutionResult


@dataclass
class ExecutorConfig:
    """Configuration for test execution."""
    repo_path: Path
    timeout_seconds: int = 300
    num_runs: int = 3
    python_version: str = "3.12"
    setup_commands: List[str] = field(default_factory=list)
    install_commands: List[str] = field(default_factory=list)
    use_venv: bool = True

    def __post_init__(self):
        self.repo_path = Path(self.repo_path)
        if not self.install_commands:
            self.install_commands = ["pip install -e ."]


class BaseExecutor(ABC):
    """Abstract base class for test executors."""

    def __init__(self, config: ExecutorConfig):
        self.config = config

    @abstractmethod
    def execute(
        self,
        extraction: CommitExtraction,
        tests: List[PerformanceTest],
    ) -> ExecutionResult:
        """Execute tests on base, head, and main commits.

        Args:
            extraction: The commit extraction with hash info
            tests: List of performance tests to run

        Returns:
            ExecutionResult with timing data for all commits
        """
        pass

    @abstractmethod
    def setup_environment(self, commit: str) -> None:
        """Setup environment for a specific commit."""
        pass

    @abstractmethod
    def teardown_environment(self) -> None:
        """Cleanup after execution."""
        pass

    def run_single_test(self, test: PerformanceTest) -> List[float]:
        """Run a single test multiple times and return durations.

        Default implementation - subclasses may override.
        """
        raise NotImplementedError("Subclass must implement run_single_test")
