# Architecture Refactor Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Refactor OmniPerf-Bench into a modular, maintainable architecture with pluggable execution backends.

**Architecture:** GSO-inspired module structure with clean separation: data models, input loading (JSON/HuggingFace), test generation, pluggable execution (local/Modal/cloud), and dataset building. Each module has single responsibility.

**Tech Stack:** Python 3.12+, Pydantic for data models, ABC for executor interface, existing LLM clients (OpenAI/Anthropic/Bedrock)

---

## Phase 1: Core Data Models

### Task 1: Create Data Models Module

**Files:**
- Create: `src/omniperf/data/__init__.py`
- Create: `src/omniperf/data/models.py`
- Create: `src/omniperf/__init__.py`

**Step 1: Create directory structure**

```bash
mkdir -p src/omniperf/data
touch src/omniperf/__init__.py
touch src/omniperf/data/__init__.py
```

**Step 2: Write data models**

Create `src/omniperf/data/models.py`:

```python
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
```

**Step 3: Create __init__ exports**

Create `src/omniperf/data/__init__.py`:

```python
"""Data models for OmniPerf-Bench."""
from .models import (
    CommitExtraction,
    PerformanceTest,
    TimingResult,
    ExecutionResult,
    DatasetRecord,
)

__all__ = [
    "CommitExtraction",
    "PerformanceTest",
    "TimingResult",
    "ExecutionResult",
    "DatasetRecord",
]
```

Create `src/omniperf/__init__.py`:

```python
"""OmniPerf-Bench: Performance optimization benchmark framework."""
__version__ = "0.2.0"
```

**Step 4: Verify imports work**

```bash
cd /Users/fortuna/Desktop/Exp/OmniPerf-Bench
PYTHONPATH=src python -c "from omniperf.data import CommitExtraction, DatasetRecord; print('OK')"
```

Expected: `OK`

**Step 5: Commit**

```bash
git add src/omniperf/
git commit -m "feat: add core data models for OmniPerf-Bench

- CommitExtraction: input from JSON files
- PerformanceTest: generated test representation
- TimingResult/ExecutionResult: execution outputs
- DatasetRecord: canonical output format"
```

---

### Task 2: Create Input Loaders Module

**Files:**
- Create: `src/omniperf/input/__init__.py`
- Create: `src/omniperf/input/loaders.py`

**Step 1: Create directory**

```bash
mkdir -p src/omniperf/input
touch src/omniperf/input/__init__.py
```

**Step 2: Write input loaders**

Create `src/omniperf/input/loaders.py`:

```python
"""Input loaders for commit extractions (JSON files and HuggingFace)."""
from __future__ import annotations

import json
import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Iterator, List, Optional

from omniperf.data import CommitExtraction

logger = logging.getLogger(__name__)


class BaseLoader(ABC):
    """Abstract base for input loaders."""

    @abstractmethod
    def load(self) -> List[CommitExtraction]:
        """Load all commit extractions."""
        pass

    @abstractmethod
    def __iter__(self) -> Iterator[CommitExtraction]:
        """Iterate over commit extractions."""
        pass


class JSONDirectoryLoader(BaseLoader):
    """Load commit extractions from a directory of JSON files."""

    def __init__(self, directory: Path, pattern: str = "*.json"):
        self.directory = Path(directory)
        self.pattern = pattern
        self._skip_files = {"extraction_summary.json"}

    def load(self) -> List[CommitExtraction]:
        """Load all JSON files from directory."""
        return list(self)

    def __iter__(self) -> Iterator[CommitExtraction]:
        """Iterate over JSON files, yielding CommitExtraction objects."""
        if not self.directory.exists():
            raise FileNotFoundError(f"Directory not found: {self.directory}")

        json_files = sorted(self.directory.glob(self.pattern))

        for json_file in json_files:
            if json_file.name in self._skip_files:
                continue

            try:
                data = json.loads(json_file.read_text())
                yield CommitExtraction.from_json(data)
            except Exception as e:
                logger.warning(f"Failed to load {json_file}: {e}")
                continue


class JSONFileLoader(BaseLoader):
    """Load commit extractions from a single JSON file (or JSONL)."""

    def __init__(self, filepath: Path):
        self.filepath = Path(filepath)

    def load(self) -> List[CommitExtraction]:
        """Load from file."""
        return list(self)

    def __iter__(self) -> Iterator[CommitExtraction]:
        """Iterate over records in file."""
        if not self.filepath.exists():
            raise FileNotFoundError(f"File not found: {self.filepath}")

        content = self.filepath.read_text()

        # Try JSONL first (one JSON object per line)
        if self.filepath.suffix == ".jsonl":
            for line in content.strip().split("\n"):
                if line.strip():
                    data = json.loads(line)
                    yield CommitExtraction.from_json(data)
        else:
            # Try as single JSON (could be array or single object)
            data = json.loads(content)
            if isinstance(data, list):
                for item in data:
                    yield CommitExtraction.from_json(item)
            else:
                yield CommitExtraction.from_json(data)


class HuggingFaceLoader(BaseLoader):
    """Load commit extractions from HuggingFace dataset."""

    def __init__(self, repo_id: str, split: str = "train", config: Optional[str] = None):
        self.repo_id = repo_id
        self.split = split
        self.config = config
        self._dataset = None

    def _load_dataset(self):
        """Lazy load the HuggingFace dataset."""
        if self._dataset is None:
            try:
                from datasets import load_dataset
            except ImportError:
                raise ImportError("Install datasets: pip install datasets")

            if self.config:
                self._dataset = load_dataset(self.repo_id, self.config, split=self.split)
            else:
                self._dataset = load_dataset(self.repo_id, split=self.split)
        return self._dataset

    def load(self) -> List[CommitExtraction]:
        """Load all records from HuggingFace."""
        return list(self)

    def __iter__(self) -> Iterator[CommitExtraction]:
        """Iterate over HuggingFace dataset records."""
        dataset = self._load_dataset()
        for record in dataset:
            yield CommitExtraction.from_json(dict(record))


def create_loader(source: str) -> BaseLoader:
    """Factory function to create appropriate loader from source string.

    Args:
        source: Can be:
            - Directory path: "/path/to/jsons/"
            - File path: "/path/to/file.json" or ".jsonl"
            - HuggingFace: "hf://repo/name" or "hf://repo/name:split"

    Returns:
        Appropriate loader instance.
    """
    if source.startswith("hf://"):
        # Parse HuggingFace URL: hf://owner/repo or hf://owner/repo:split
        hf_path = source[5:]  # Remove "hf://"
        if ":" in hf_path:
            repo_id, split = hf_path.rsplit(":", 1)
        else:
            repo_id, split = hf_path, "train"
        return HuggingFaceLoader(repo_id, split)

    path = Path(source)
    if path.is_dir():
        return JSONDirectoryLoader(path)
    elif path.is_file():
        return JSONFileLoader(path)
    else:
        raise ValueError(f"Unknown source type: {source}")
```

**Step 3: Create __init__ exports**

Create `src/omniperf/input/__init__.py`:

```python
"""Input loaders for OmniPerf-Bench."""
from .loaders import (
    BaseLoader,
    JSONDirectoryLoader,
    JSONFileLoader,
    HuggingFaceLoader,
    create_loader,
)

__all__ = [
    "BaseLoader",
    "JSONDirectoryLoader",
    "JSONFileLoader",
    "HuggingFaceLoader",
    "create_loader",
]
```

**Step 4: Verify imports work**

```bash
PYTHONPATH=src python -c "from omniperf.input import create_loader; print('OK')"
```

Expected: `OK`

**Step 5: Commit**

```bash
git add src/omniperf/input/
git commit -m "feat: add input loaders for JSON and HuggingFace

- JSONDirectoryLoader: load from directory of JSON files
- JSONFileLoader: load from single JSON/JSONL file
- HuggingFaceLoader: load from HuggingFace datasets
- create_loader factory for automatic source detection"
```

---

### Task 3: Create Executor Interface and Local Executor

**Files:**
- Create: `src/omniperf/execute/__init__.py`
- Create: `src/omniperf/execute/base.py`
- Create: `src/omniperf/execute/local.py`

**Step 1: Create directory**

```bash
mkdir -p src/omniperf/execute
touch src/omniperf/execute/__init__.py
```

**Step 2: Write executor base class**

Create `src/omniperf/execute/base.py`:

```python
"""Base executor interface for running performance tests."""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
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
    setup_commands: List[str] = None
    install_commands: List[str] = None
    use_venv: bool = True

    def __post_init__(self):
        self.repo_path = Path(self.repo_path)
        if self.setup_commands is None:
            self.setup_commands = []
        if self.install_commands is None:
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
```

**Step 3: Write local executor**

Create `src/omniperf/execute/local.py`:

```python
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
                    ["python", "-m", "venv", str(self._venv_path)],
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
                python = "python"

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
```

**Step 4: Create __init__ exports**

Create `src/omniperf/execute/__init__.py`:

```python
"""Execution backends for OmniPerf-Bench."""
from .base import BaseExecutor, ExecutorConfig
from .local import LocalExecutor

__all__ = [
    "BaseExecutor",
    "ExecutorConfig",
    "LocalExecutor",
]
```

**Step 5: Verify imports work**

```bash
PYTHONPATH=src python -c "from omniperf.execute import LocalExecutor, ExecutorConfig; print('OK')"
```

Expected: `OK`

**Step 6: Commit**

```bash
git add src/omniperf/execute/
git commit -m "feat: add pluggable executor interface with local backend

- BaseExecutor: abstract interface for test execution
- ExecutorConfig: configuration dataclass
- LocalExecutor: run tests locally with venv isolation
- Designed for easy addition of Modal/cloud backends"
```

---

### Task 4: Create Test Generator Module

**Files:**
- Create: `src/omniperf/generate/__init__.py`
- Create: `src/omniperf/generate/generator.py`

**Step 1: Create directory**

```bash
mkdir -p src/omniperf/generate
touch src/omniperf/generate/__init__.py
```

**Step 2: Write test generator**

Create `src/omniperf/generate/generator.py`:

```python
"""Performance test generation using LLMs."""
from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import List, Optional

from omniperf.data import CommitExtraction, PerformanceTest

logger = logging.getLogger(__name__)


@dataclass
class GeneratorConfig:
    """Configuration for test generation."""
    provider: str = "openai"  # openai, anthropic, bedrock
    model: str = "gpt-4o-mini"
    temperature: float = 0.1
    max_tokens: int = 4096
    num_samples: int = 1


class TestGenerator:
    """Generate performance tests from commit extractions using LLMs."""

    SYSTEM_PROMPT = """You are an expert at writing performance tests for Python code.
Given a commit that optimizes code, write a performance test that:
1. Exercises the optimized code path
2. Prints timing in format: "Execution time: X.XXXXs"
3. Uses realistic inputs that trigger the optimization
4. Is self-contained and runnable

Output ONLY the Python code, no explanations."""

    USER_PROMPT_TEMPLATE = """Write a performance test for this optimization commit:

Repository: {repo_name}
Commit message: {message}

Affected APIs: {apis}

Diff:
```
{diff}
```

Write a complete, runnable Python test that measures the performance of the optimized code.
The test MUST print timing as: print(f"Execution time: {{duration:.4f}}s")
"""

    def __init__(self, config: GeneratorConfig):
        self.config = config
        self._client = None

    def _get_client(self):
        """Lazy-load the appropriate LLM client."""
        if self._client is not None:
            return self._client

        if self.config.provider == "openai":
            from openai import OpenAI
            self._client = OpenAI()
        elif self.config.provider == "anthropic":
            from anthropic import Anthropic
            self._client = Anthropic()
        elif self.config.provider == "bedrock":
            import boto3
            self._client = boto3.client("bedrock-runtime")
        else:
            raise ValueError(f"Unknown provider: {self.config.provider}")

        return self._client

    def generate(self, extraction: CommitExtraction) -> List[PerformanceTest]:
        """Generate performance tests for a commit extraction."""
        prompt = self.USER_PROMPT_TEMPLATE.format(
            repo_name=extraction.repo_name or "unknown",
            message=extraction.message,
            apis=", ".join(extraction.affected_apis) or "N/A",
            diff=extraction.diff_text[:8000],  # Truncate long diffs
        )

        tests = []
        for i in range(self.config.num_samples):
            try:
                code = self._call_llm(prompt)
                code = self._extract_code(code)

                test = PerformanceTest(
                    test_id=f"{extraction.commit_hash[:8]}_{i}",
                    code=code,
                    target_api=extraction.affected_apis[0] if extraction.affected_apis else None,
                    generation_model=f"{self.config.provider}/{self.config.model}",
                )
                tests.append(test)

            except Exception as e:
                logger.warning(f"Failed to generate test {i}: {e}")

        return tests

    def _call_llm(self, prompt: str) -> str:
        """Call the LLM and return response text."""
        client = self._get_client()

        if self.config.provider == "openai":
            response = client.chat.completions.create(
                model=self.config.model,
                messages=[
                    {"role": "system", "content": self.SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens,
            )
            return response.choices[0].message.content

        elif self.config.provider == "anthropic":
            response = client.messages.create(
                model=self.config.model,
                max_tokens=self.config.max_tokens,
                system=self.SYSTEM_PROMPT,
                messages=[{"role": "user", "content": prompt}],
            )
            return response.content[0].text

        elif self.config.provider == "bedrock":
            import json
            body = json.dumps({
                "anthropic_version": "bedrock-2023-05-31",
                "max_tokens": self.config.max_tokens,
                "system": self.SYSTEM_PROMPT,
                "messages": [{"role": "user", "content": prompt}],
            })
            response = client.invoke_model(
                modelId=self.config.model,
                body=body,
            )
            result = json.loads(response["body"].read())
            return result["content"][0]["text"]

        raise ValueError(f"Unknown provider: {self.config.provider}")

    def _extract_code(self, response: str) -> str:
        """Extract Python code from LLM response."""
        # Try to find code blocks
        if "```python" in response:
            start = response.find("```python") + 9
            end = response.find("```", start)
            if end > start:
                return response[start:end].strip()

        if "```" in response:
            start = response.find("```") + 3
            end = response.find("```", start)
            if end > start:
                return response[start:end].strip()

        # Return as-is if no code blocks
        return response.strip()
```

**Step 3: Create __init__ exports**

Create `src/omniperf/generate/__init__.py`:

```python
"""Test generation for OmniPerf-Bench."""
from .generator import TestGenerator, GeneratorConfig

__all__ = [
    "TestGenerator",
    "GeneratorConfig",
]
```

**Step 4: Verify imports work**

```bash
PYTHONPATH=src python -c "from omniperf.generate import TestGenerator, GeneratorConfig; print('OK')"
```

Expected: `OK`

**Step 5: Commit**

```bash
git add src/omniperf/generate/
git commit -m "feat: add LLM-based test generator module

- TestGenerator: generate performance tests from commit extractions
- GeneratorConfig: provider/model configuration
- Supports OpenAI, Anthropic, and Bedrock backends"
```

---

### Task 5: Create Dataset Builder Module

**Files:**
- Create: `src/omniperf/build/__init__.py`
- Create: `src/omniperf/build/builder.py`

**Step 1: Create directory**

```bash
mkdir -p src/omniperf/build
touch src/omniperf/build/__init__.py
```

**Step 2: Write dataset builder**

Create `src/omniperf/build/builder.py`:

```python
"""Dataset building - combine results into final dataset format."""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

from omniperf.data import (
    CommitExtraction,
    PerformanceTest,
    ExecutionResult,
    DatasetRecord,
)

logger = logging.getLogger(__name__)


@dataclass
class BuilderConfig:
    """Configuration for dataset building."""
    output_dir: Path = Path("data")
    dataset_name: str = "omniperf_dataset"
    min_speedup: float = 1.0  # Minimum speedup to include (1.0 = no filter)
    min_tests: int = 1  # Minimum tests required
    push_to_hf: bool = False
    hf_repo: Optional[str] = None

    def __post_init__(self):
        self.output_dir = Path(self.output_dir)


class DatasetBuilder:
    """Build dataset from execution results."""

    def __init__(self, config: BuilderConfig):
        self.config = config
        self.records: List[DatasetRecord] = []

    def add_result(
        self,
        extraction: CommitExtraction,
        tests: List[PerformanceTest],
        result: ExecutionResult,
    ) -> Optional[DatasetRecord]:
        """Add an execution result to the dataset.

        Returns the DatasetRecord if it passes filters, None otherwise.
        """
        # Calculate speedup
        speedup = self._calculate_speedup(result)

        if speedup is None:
            logger.warning(f"Could not calculate speedup for {extraction.commit_hash}")
            return None

        if speedup < self.config.min_speedup:
            logger.info(f"Skipping {extraction.commit_hash}: speedup {speedup:.2f} < {self.config.min_speedup}")
            return None

        if len(tests) < self.config.min_tests:
            logger.info(f"Skipping {extraction.commit_hash}: only {len(tests)} tests")
            return None

        # Build duration_changes
        duration_changes = self._build_duration_changes(result)

        # Create record
        record = DatasetRecord(
            instance_id=self._make_instance_id(extraction),
            repo=extraction.repo_name or "unknown",
            base_commit=extraction.parent_hash,
            head_commit=extraction.commit_hash,
            patch=extraction.diff_text,
            efficiency_test=[t.code for t in tests],
            duration_changes=duration_changes,
            human_performance=speedup,
            hints_text=extraction.message,
            api=extraction.affected_apis[0] if extraction.affected_apis else None,
        )

        self.records.append(record)
        return record

    def build(self) -> Path:
        """Write dataset to disk and optionally push to HuggingFace.

        Returns path to the output file.
        """
        self.config.output_dir.mkdir(parents=True, exist_ok=True)
        output_path = self.config.output_dir / f"{self.config.dataset_name}.jsonl"

        # Write JSONL
        with open(output_path, "w") as f:
            for record in self.records:
                f.write(json.dumps(record.to_dict()) + "\n")

        logger.info(f"Wrote {len(self.records)} records to {output_path}")

        # Push to HuggingFace if configured
        if self.config.push_to_hf and self.config.hf_repo:
            self._push_to_huggingface(output_path)

        return output_path

    def _calculate_speedup(self, result: ExecutionResult) -> Optional[float]:
        """Calculate speedup ratio (base_time / head_time)."""
        base_times = [
            r.mean_duration for r in result.base_results
            if r.mean_duration is not None
        ]
        head_times = [
            r.mean_duration for r in result.head_results
            if r.mean_duration is not None
        ]

        if not base_times or not head_times:
            return None

        avg_base = sum(base_times) / len(base_times)
        avg_head = sum(head_times) / len(head_times)

        if avg_head <= 0:
            return None

        return avg_base / avg_head

    def _build_duration_changes(self, result: ExecutionResult) -> List[Dict]:
        """Build duration_changes list from execution result."""
        changes = []

        for i, base_result in enumerate(result.base_results):
            change = {
                "base": base_result.durations,
                "head": result.head_results[i].durations if i < len(result.head_results) else [],
            }
            if i < len(result.main_results):
                change["main"] = result.main_results[i].durations
            changes.append(change)

        return changes

    def _make_instance_id(self, extraction: CommitExtraction) -> str:
        """Generate instance ID from extraction."""
        repo = extraction.repo_name or "unknown"
        repo_clean = repo.replace("/", "__")
        return f"{repo_clean}-{extraction.commit_hash[:8]}"

    def _push_to_huggingface(self, filepath: Path) -> None:
        """Push dataset to HuggingFace Hub."""
        try:
            from datasets import Dataset
            from huggingface_hub import HfApi

            # Load as Dataset
            records = [json.loads(line) for line in filepath.read_text().strip().split("\n")]
            dataset = Dataset.from_list(records)

            # Push
            dataset.push_to_hub(self.config.hf_repo)
            logger.info(f"Pushed to HuggingFace: {self.config.hf_repo}")

        except Exception as e:
            logger.error(f"Failed to push to HuggingFace: {e}")
```

**Step 3: Create __init__ exports**

Create `src/omniperf/build/__init__.py`:

```python
"""Dataset building for OmniPerf-Bench."""
from .builder import DatasetBuilder, BuilderConfig

__all__ = [
    "DatasetBuilder",
    "BuilderConfig",
]
```

**Step 4: Verify imports work**

```bash
PYTHONPATH=src python -c "from omniperf.build import DatasetBuilder, BuilderConfig; print('OK')"
```

Expected: `OK`

**Step 5: Commit**

```bash
git add src/omniperf/build/
git commit -m "feat: add dataset builder module

- DatasetBuilder: combine results into JSONL dataset
- BuilderConfig: output configuration
- Speedup filtering and HuggingFace push support"
```

---

### Task 6: Create Main Pipeline and CLI

**Files:**
- Create: `src/omniperf/pipeline.py`
- Create: `src/omniperf/cli.py`

**Step 1: Write pipeline orchestrator**

Create `src/omniperf/pipeline.py`:

```python
"""Main pipeline orchestrating the full dataset generation flow."""
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from omniperf.data import CommitExtraction
from omniperf.input import create_loader
from omniperf.generate import TestGenerator, GeneratorConfig
from omniperf.execute import LocalExecutor, ExecutorConfig
from omniperf.build import DatasetBuilder, BuilderConfig

logger = logging.getLogger(__name__)


@dataclass
class PipelineConfig:
    """Full pipeline configuration."""
    # Input
    input_source: str  # Path or hf:// URL

    # Repository
    repo_path: Path

    # Generation
    llm_provider: str = "openai"
    llm_model: str = "gpt-4o-mini"
    llm_temperature: float = 0.1
    num_test_samples: int = 1

    # Execution
    executor: str = "local"  # local, modal, skypilot
    timeout_seconds: int = 300
    num_runs: int = 3

    # Output
    output_dir: Path = Path("data")
    dataset_name: str = "omniperf_dataset"
    min_speedup: float = 1.0
    push_to_hf: bool = False
    hf_repo: Optional[str] = None

    def __post_init__(self):
        self.repo_path = Path(self.repo_path)
        self.output_dir = Path(self.output_dir)


class Pipeline:
    """Main pipeline for dataset generation."""

    def __init__(self, config: PipelineConfig):
        self.config = config

        # Initialize components
        self.loader = create_loader(config.input_source)

        self.generator = TestGenerator(GeneratorConfig(
            provider=config.llm_provider,
            model=config.llm_model,
            temperature=config.llm_temperature,
            num_samples=config.num_test_samples,
        ))

        self.executor = self._create_executor()

        self.builder = DatasetBuilder(BuilderConfig(
            output_dir=config.output_dir,
            dataset_name=config.dataset_name,
            min_speedup=config.min_speedup,
            push_to_hf=config.push_to_hf,
            hf_repo=config.hf_repo,
        ))

    def _create_executor(self):
        """Create executor based on config."""
        exec_config = ExecutorConfig(
            repo_path=self.config.repo_path,
            timeout_seconds=self.config.timeout_seconds,
            num_runs=self.config.num_runs,
        )

        if self.config.executor == "local":
            return LocalExecutor(exec_config)
        elif self.config.executor == "modal":
            # Placeholder for Modal executor
            raise NotImplementedError("Modal executor not yet implemented")
        elif self.config.executor == "skypilot":
            # Placeholder for SkyPilot executor
            raise NotImplementedError("SkyPilot executor not yet implemented")
        else:
            raise ValueError(f"Unknown executor: {self.config.executor}")

    def run(self) -> Path:
        """Run the full pipeline and return path to output dataset."""
        logger.info("Starting OmniPerf-Bench pipeline")

        # Load extractions
        extractions = self.loader.load()
        logger.info(f"Loaded {len(extractions)} commit extractions")

        # Process each extraction
        for i, extraction in enumerate(extractions):
            logger.info(f"Processing {i+1}/{len(extractions)}: {extraction.commit_hash[:8]}")

            try:
                # Generate tests
                tests = self.generator.generate(extraction)
                if not tests:
                    logger.warning(f"No tests generated for {extraction.commit_hash}")
                    continue
                logger.info(f"Generated {len(tests)} tests")

                # Execute tests
                result = self.executor.execute(extraction, tests)
                logger.info(f"Executed tests on {extraction.commit_hash[:8]}")

                # Add to dataset
                record = self.builder.add_result(extraction, tests, result)
                if record:
                    logger.info(f"Added record: {record.instance_id} (speedup: {record.human_performance:.2f}x)")

            except Exception as e:
                logger.error(f"Failed to process {extraction.commit_hash}: {e}")
                continue

        # Build final dataset
        output_path = self.builder.build()
        logger.info(f"Pipeline complete. Output: {output_path}")

        return output_path
```

**Step 2: Write CLI**

Create `src/omniperf/cli.py`:

```python
"""Command-line interface for OmniPerf-Bench."""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import yaml

from omniperf.pipeline import Pipeline, PipelineConfig


def setup_logging(verbose: bool = False):
    """Configure logging."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler("omniperf.log", mode="w"),
        ],
    )


def load_config(config_path: Path) -> PipelineConfig:
    """Load configuration from YAML file."""
    with open(config_path) as f:
        data = yaml.safe_load(f)

    return PipelineConfig(
        input_source=data.get("input_source") or data.get("extractions_dir", ""),
        repo_path=Path(data.get("repo_path", ".")),
        llm_provider=data.get("llm_provider", "openai"),
        llm_model=data.get("llm_model", "gpt-4o-mini"),
        llm_temperature=data.get("llm_temperature", 0.1),
        num_test_samples=data.get("num_test_samples", 1),
        executor=data.get("executor", "local"),
        timeout_seconds=data.get("timeout_seconds", 300),
        num_runs=data.get("num_runs", 3),
        output_dir=Path(data.get("output_dir", "data")),
        dataset_name=data.get("dataset_name", "omniperf_dataset"),
        min_speedup=data.get("min_speedup", 1.0),
        push_to_hf=data.get("push_to_hf", False),
        hf_repo=data.get("hf_repo"),
    )


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="OmniPerf-Bench: Performance optimization benchmark framework"
    )
    parser.add_argument(
        "config",
        type=Path,
        help="Path to YAML configuration file",
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable verbose logging",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate config without running pipeline",
    )

    args = parser.parse_args()

    setup_logging(args.verbose)
    logger = logging.getLogger(__name__)

    # Load config
    if not args.config.exists():
        logger.error(f"Config file not found: {args.config}")
        sys.exit(1)

    config = load_config(args.config)
    logger.info(f"Loaded config from {args.config}")

    if args.dry_run:
        logger.info("Dry run - config is valid")
        print(f"Input source: {config.input_source}")
        print(f"Repository: {config.repo_path}")
        print(f"Executor: {config.executor}")
        print(f"Output: {config.output_dir}/{config.dataset_name}.jsonl")
        sys.exit(0)

    # Run pipeline
    try:
        pipeline = Pipeline(config)
        output_path = pipeline.run()
        print(f"\nDataset written to: {output_path}")
    except Exception as e:
        logger.exception(f"Pipeline failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
```

**Step 3: Verify CLI works**

```bash
PYTHONPATH=src python -c "from omniperf.cli import main; print('OK')"
```

Expected: `OK`

**Step 4: Commit**

```bash
git add src/omniperf/pipeline.py src/omniperf/cli.py
git commit -m "feat: add pipeline orchestrator and CLI

- Pipeline: coordinates loader -> generator -> executor -> builder
- CLI: YAML config loading and argument parsing
- Supports dry-run mode for config validation"
```

---

### Task 7: Create New Entry Point Script

**Files:**
- Create: `run_omniperf.py`
- Modify: `configs/omniperf.yaml` (new config format)

**Step 1: Create entry point script**

Create `run_omniperf.py`:

```python
#!/usr/bin/env python3
"""
OmniPerf-Bench - Performance optimization benchmark framework.

Usage:
    python run_omniperf.py config.yaml
    python run_omniperf.py config.yaml --dry-run
    python run_omniperf.py config.yaml -v
"""
import sys
from pathlib import Path

# Ensure src is importable
ROOT = Path(__file__).parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from omniperf.cli import main

if __name__ == "__main__":
    main()
```

**Step 2: Create example config**

Create `configs/omniperf.yaml`:

```yaml
# OmniPerf-Bench Configuration
# ============================

# Input: where to load commit extractions from
# Options:
#   - Local directory: "/path/to/jsons/"
#   - Single file: "/path/to/file.jsonl"
#   - HuggingFace: "hf://owner/repo" or "hf://owner/repo:split"
input_source: "inputs/experiments/commit_extractions_with_apis"

# Repository path (the repo being benchmarked)
repo_path: "/path/to/vllm"

# LLM configuration for test generation
llm_provider: "openai"  # openai, anthropic, bedrock
llm_model: "gpt-4o-mini"
llm_temperature: 0.1
num_test_samples: 1

# Execution configuration
executor: "local"  # local, modal (future), skypilot (future)
timeout_seconds: 300
num_runs: 3

# Output configuration
output_dir: "data"
dataset_name: "omniperf_dataset"
min_speedup: 1.0  # Filter: only include if speedup >= this value

# HuggingFace push (optional)
push_to_hf: false
hf_repo: null  # e.g., "username/dataset-name"
```

**Step 3: Make script executable**

```bash
chmod +x run_omniperf.py
```

**Step 4: Test dry-run**

```bash
python run_omniperf.py configs/omniperf.yaml --dry-run
```

Expected output showing config summary.

**Step 5: Commit**

```bash
git add run_omniperf.py configs/omniperf.yaml
git commit -m "feat: add new entry point and example config

- run_omniperf.py: clean entry point replacing monolithic script
- configs/omniperf.yaml: example configuration with documentation
- Supports dry-run mode for validation"
```

---

### Task 8: Update Package Exports

**Files:**
- Modify: `src/omniperf/__init__.py`

**Step 1: Update main __init__.py**

Update `src/omniperf/__init__.py`:

```python
"""
OmniPerf-Bench: Performance optimization benchmark framework.

Modular architecture:
- omniperf.data: Data models (CommitExtraction, DatasetRecord, etc.)
- omniperf.input: Input loaders (JSON, HuggingFace)
- omniperf.generate: LLM-based test generation
- omniperf.execute: Pluggable execution backends
- omniperf.build: Dataset building and export

Quick start:
    python run_omniperf.py configs/omniperf.yaml
"""
__version__ = "0.2.0"

from omniperf.data import (
    CommitExtraction,
    PerformanceTest,
    TimingResult,
    ExecutionResult,
    DatasetRecord,
)
from omniperf.input import create_loader
from omniperf.generate import TestGenerator, GeneratorConfig
from omniperf.execute import LocalExecutor, ExecutorConfig
from omniperf.build import DatasetBuilder, BuilderConfig
from omniperf.pipeline import Pipeline, PipelineConfig

__all__ = [
    # Version
    "__version__",
    # Data models
    "CommitExtraction",
    "PerformanceTest",
    "TimingResult",
    "ExecutionResult",
    "DatasetRecord",
    # Input
    "create_loader",
    # Generation
    "TestGenerator",
    "GeneratorConfig",
    # Execution
    "LocalExecutor",
    "ExecutorConfig",
    # Building
    "DatasetBuilder",
    "BuilderConfig",
    # Pipeline
    "Pipeline",
    "PipelineConfig",
]
```

**Step 2: Verify all imports work**

```bash
PYTHONPATH=src python -c "
from omniperf import (
    Pipeline, PipelineConfig,
    create_loader,
    TestGenerator,
    LocalExecutor,
    DatasetBuilder,
)
print('All imports OK')
print(f'Version: {__import__(\"omniperf\").__version__}')
"
```

Expected:
```
All imports OK
Version: 0.2.0
```

**Step 3: Commit**

```bash
git add src/omniperf/__init__.py
git commit -m "feat: update package exports with all modules

- Export all public classes from submodules
- Add module docstring with quick start
- Version bump to 0.2.0"
```

---

## Phase 2: Cleanup and Migration

### Task 9: Deprecate Old Entry Point

**Files:**
- Modify: `commit_to_dataset.py` (add deprecation notice)

**Step 1: Add deprecation warning to old script**

Add at the top of `commit_to_dataset.py` (after the docstring):

```python
import warnings
warnings.warn(
    "commit_to_dataset.py is deprecated. Use: python run_omniperf.py config.yaml\n"
    "See configs/omniperf.yaml for the new configuration format.",
    DeprecationWarning,
    stacklevel=2,
)
```

**Step 2: Commit**

```bash
git add commit_to_dataset.py
git commit -m "chore: add deprecation warning to old entry point

- Points users to new run_omniperf.py
- Old script still works for backwards compatibility"
```

---

### Task 10: Update Documentation

**Files:**
- Modify: `README.md`

**Step 1: Add new usage section to README.md**

Add after the existing setup section:

```markdown
## New Modular Architecture (v0.2.0)

OmniPerf-Bench now has a clean, modular architecture:

```
src/omniperf/
├── data/       # Data models (CommitExtraction, DatasetRecord)
├── input/      # Loaders (JSON files, HuggingFace)
├── generate/   # LLM test generation
├── execute/    # Pluggable executors (local, Modal, cloud)
├── build/      # Dataset building
├── pipeline.py # Main orchestrator
└── cli.py      # Command-line interface
```

### Quick Start (New)

```bash
# Edit config
cp configs/omniperf.yaml my_config.yaml
# Edit my_config.yaml with your paths

# Run pipeline
python run_omniperf.py my_config.yaml

# Dry run (validate config)
python run_omniperf.py my_config.yaml --dry-run
```

### Configuration

See `configs/omniperf.yaml` for all options:

```yaml
input_source: "inputs/experiments/commit_extractions_with_apis"
repo_path: "/path/to/repo"
llm_provider: "openai"
llm_model: "gpt-4o-mini"
executor: "local"
output_dir: "data"
dataset_name: "my_dataset"
```
```

**Step 2: Commit**

```bash
git add README.md
git commit -m "docs: add new modular architecture to README

- Document new directory structure
- Add quick start for run_omniperf.py
- Show example configuration"
```

---

## Summary

After completing all tasks, you will have:

```
src/omniperf/
├── __init__.py          # Package exports
├── cli.py               # Command-line interface
├── pipeline.py          # Main orchestrator
├── data/
│   ├── __init__.py
│   └── models.py        # CommitExtraction, DatasetRecord, etc.
├── input/
│   ├── __init__.py
│   └── loaders.py       # JSON and HuggingFace loaders
├── generate/
│   ├── __init__.py
│   └── generator.py     # LLM test generation
├── execute/
│   ├── __init__.py
│   ├── base.py          # Executor interface
│   └── local.py         # Local executor
└── build/
    ├── __init__.py
    └── builder.py       # Dataset building

run_omniperf.py          # New entry point
configs/omniperf.yaml    # Example config
```

**Key improvements:**
1. Clean module separation (like GSO)
2. Pluggable execution (easy to add Modal/cloud later)
3. JSON + HuggingFace input support
4. Single config file for everything
5. Deprecation path for old code

---

**Plan complete and saved to `docs/plans/2025-12-09-architecture-refactor.md`. Two execution options:**

**1. Subagent-Driven (this session)** - I dispatch fresh subagent per task, review between tasks, fast iteration

**2. Parallel Session (separate)** - Open new session with executing-plans, batch execution with checkpoints

**Which approach?**