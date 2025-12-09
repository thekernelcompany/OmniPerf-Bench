"""Local executor - runs tests directly on the local machine."""
from __future__ import annotations

import logging
import os
import re
import subprocess
import tempfile
from pathlib import Path
from typing import List, Optional

from omniperf.data import (
    CommitExtraction,
    PerformanceTest,
    ExecutionResult,
    TimingResult,
)
from .base import BaseExecutor, ExecutorConfig

logger = logging.getLogger(__name__)

# Regex to extract timing from test output
TIMING_PATTERN = re.compile(r"Execution time:\s*([\d.]+)\s*s", re.IGNORECASE)


class LocalExecutor(BaseExecutor):
    """Execute tests locally using subprocess and venv."""

    def __init__(self, config: ExecutorConfig):
        super().__init__(config)
        self._venv_path: Optional[Path] = None
        self._original_commit: Optional[str] = None

    def execute(
        self,
        extraction: CommitExtraction,
        tests: List[PerformanceTest],
    ) -> ExecutionResult:
        """Execute tests on base, head, and optionally main commits."""
        result = ExecutionResult(commit_hash=extraction.commit_hash)

        try:
            # Save original state
            self._original_commit = self._get_current_commit()

            # Run on base (parent) commit
            logger.info(f"Running tests on base commit: {extraction.parent_hash[:8]}")
            self.setup_environment(extraction.parent_hash)
            result.base_results = self._run_tests(tests, extraction.parent_hash)

            # Run on head (optimized) commit
            logger.info(f"Running tests on head commit: {extraction.commit_hash[:8]}")
            self.setup_environment(extraction.commit_hash)
            result.head_results = self._run_tests(tests, extraction.commit_hash)

            # Optionally run on main
            try:
                logger.info("Running tests on main branch")
                self.setup_environment("main")
                result.main_results = self._run_tests(tests, "main")
            except Exception as e:
                logger.warning(f"Failed to run on main: {e}")

        finally:
            self.teardown_environment()

        return result

    def setup_environment(self, commit: str) -> None:
        """Checkout commit and setup venv."""
        repo = self.config.repo_path

        # Checkout the commit
        self._run_cmd(["git", "checkout", commit], cwd=repo)

        # Create/update venv if needed
        if self.config.use_venv:
            if self._venv_path is None:
                self._venv_path = Path(tempfile.mkdtemp(prefix="omniperf_venv_"))
                self._run_cmd(
                    ["python3", "-m", "venv", str(self._venv_path)],
                    cwd=repo
                )

            # Run install commands
            pip = self._venv_path / "bin" / "pip"
            for cmd in self.config.install_commands:
                self._run_cmd(
                    [str(pip)] + cmd.replace("pip ", "").split(),
                    cwd=repo
                )

    def teardown_environment(self) -> None:
        """Restore original commit and cleanup."""
        if self._original_commit:
            try:
                self._run_cmd(
                    ["git", "checkout", self._original_commit],
                    cwd=self.config.repo_path
                )
            except Exception as e:
                logger.warning(f"Failed to restore commit: {e}")

    def _run_tests(
        self,
        tests: List[PerformanceTest],
        commit: str
    ) -> List[TimingResult]:
        """Run all tests and collect timing results."""
        results = []

        for test in tests:
            timing = TimingResult(test_id=test.test_id, commit=commit)

            try:
                for _ in range(self.config.num_runs):
                    duration = self._run_single_test(test)
                    if duration is not None:
                        timing.durations.append(duration)

                timing.compute_stats()

            except Exception as e:
                timing.error = str(e)
                logger.warning(f"Test {test.test_id} failed: {e}")

            results.append(timing)

        return results

    def _run_single_test(self, test: PerformanceTest) -> Optional[float]:
        """Run a single test and extract timing."""
        # Write test to temp file
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".py", delete=False
        ) as f:
            f.write(test.code)
            test_file = f.name

        try:
            # Determine python executable
            if self.config.use_venv and self._venv_path:
                python = str(self._venv_path / "bin" / "python")
            else:
                python = "python3"

            # Run test with timeout
            result = subprocess.run(
                [python, test_file],
                cwd=self.config.repo_path,
                capture_output=True,
                text=True,
                timeout=self.config.timeout_seconds,
            )

            # Extract timing from output
            output = result.stdout + result.stderr
            match = TIMING_PATTERN.search(output)

            if match:
                return float(match.group(1))
            else:
                logger.warning(f"No timing found in output: {output[:200]}")
                return None

        finally:
            os.unlink(test_file)

    def _run_cmd(
        self,
        cmd: List[str],
        cwd: Optional[Path] = None,
        timeout: int = 300
    ) -> str:
        """Run a shell command and return stdout."""
        result = subprocess.run(
            cmd,
            cwd=str(cwd) if cwd else None,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        if result.returncode != 0:
            raise RuntimeError(
                f"Command failed: {' '.join(cmd)}\n"
                f"stderr: {result.stderr}"
            )
        return result.stdout

    def _get_current_commit(self) -> str:
        """Get current HEAD commit hash."""
        return self._run_cmd(
            ["git", "rev-parse", "HEAD"],
            cwd=self.config.repo_path
        ).strip()
