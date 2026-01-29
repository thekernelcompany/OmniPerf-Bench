"""
OmniPerf Benchmark Runner for Claude Code Patches.

This module runs benchmarks from the Ayushnangia/omniperf_v1 HuggingFace dataset
against Claude Code-generated patches, measuring baseline and patched performance.

Usage:
    python -m src.eval.omniperf_benchmark_runner \
        --repo-path /path/to/repos \
        --state-root /path/to/perf-agents-bench/state \
        --output-dir /path/to/results \
        --split vllm  # or sglang

The benchmark flow:
1. Load dataset from HuggingFace (Ayushnangia/omniperf_v1)
2. Match dataset instances to Claude Code patches by commit_hash
3. For each matched instance:
   a. Create git worktree at pre-commit (baseline)
   b. Execute test_script from dataset to get baseline timing
   c. Apply Claude Code's patch
   d. Execute test_script again to get patched timing
   e. Compute speedup and record metrics
4. Generate comprehensive evaluation report
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import shutil
import subprocess
import tempfile
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# Configuration
DEFAULT_TIMEOUT = 600  # 10 minutes per test
DEFAULT_WARMUP_RUNS = 2
HF_DATASET_ID = "Ayushnangia/omniperf_v1"

# Cache directories for ephemeral storage
EPHEMERAL_CACHE_ROOT = Path("/tmp/omniperf_cache")


@dataclass
class BenchmarkResult:
    """Result of running a benchmark."""
    status: str  # success, error, timeout, no_patch, no_test, baseline_failed, patched_failed
    commit_hash: str
    baseline_ms: Optional[float] = None
    patched_ms: Optional[float] = None
    speedup: Optional[float] = None
    improvement: bool = False
    baseline_output: Optional[Dict[str, Any]] = None
    patched_output: Optional[Dict[str, Any]] = None
    error_message: Optional[str] = None
    stdout: str = ""
    stderr: str = ""
    duration_s: float = 0.0
    patch_path: Optional[str] = None
    patch_stats: Optional[Dict[str, Any]] = None


@dataclass
class DatasetInstance:
    """Represents an instance from the OmniPerf dataset."""
    commit_hash: str
    commit_subject: str
    repo: str  # vllm or sglang
    test_script: Optional[str] = None
    perf_command: Optional[str] = None
    files_changed: List[str] = field(default_factory=list)
    pr_url: Optional[str] = None
    stats: Optional[Dict[str, Any]] = None

    @classmethod
    def from_hf_row(cls, row: Dict[str, Any]) -> "DatasetInstance":
        """Create instance from HuggingFace dataset row."""
        return cls(
            commit_hash=row.get("commit_hash", ""),
            commit_subject=row.get("commit_subject", ""),
            repo=row.get("repo", "unknown"),
            test_script=row.get("test_script"),
            perf_command=row.get("perf_command"),
            files_changed=row.get("files_changed", []),
            pr_url=row.get("pr_url"),
            stats=row.get("stats"),
        )


@dataclass
class ClaudeCodePatch:
    """Represents a Claude Code generated patch."""
    patch_path: str
    human_commit: str  # Maps to dataset's commit_hash
    pre_commit: str
    item_id: str
    journal_path: str
    patch_stats: Optional[Dict[str, int]] = None
    agent_duration_s: Optional[float] = None


def load_omniperf_dataset(split: str = "vllm") -> List[DatasetInstance]:
    """
    Load the OmniPerf dataset from HuggingFace.

    Args:
        split: Dataset split to load ("vllm" or "sglang")

    Returns:
        List of DatasetInstance objects
    """
    try:
        from datasets import load_dataset
    except ImportError:
        raise ImportError("datasets library required. Install with: pip install datasets")

    logger.info(f"Loading {HF_DATASET_ID} split={split}...")

    dataset = load_dataset(HF_DATASET_ID, split=split)

    instances = []
    for row in dataset:
        instance = DatasetInstance.from_hf_row(row)
        instances.append(instance)

    logger.info(f"Loaded {len(instances)} instances from {split} split")
    return instances


def discover_claude_code_patches(
    state_root: Path,
    repo_filter: Optional[str] = None
) -> Dict[str, ClaudeCodePatch]:
    """
    Discover Claude Code patches from the state directory.

    Args:
        state_root: Path to perf-agents-bench/state directory
        repo_filter: Optional filter for repository (vllm, sglang)

    Returns:
        Dict mapping human_commit -> ClaudeCodePatch
    """
    runs_dir = state_root / "runs"
    if not runs_dir.exists():
        logger.error(f"Runs directory not found: {runs_dir}")
        return {}

    patches: Dict[str, ClaudeCodePatch] = {}

    # Hierarchical structure: {repo}/claude_code/{model}/{timestamp}/{item_id}/
    for repo_dir in runs_dir.iterdir():
        if not repo_dir.is_dir():
            continue

        repo_name = repo_dir.name
        if repo_filter and repo_name.lower() != repo_filter.lower():
            continue

        claude_code_dir = repo_dir / "claude_code"
        if not claude_code_dir.exists():
            continue

        # Traverse model -> timestamp -> item directories
        for model_dir in claude_code_dir.iterdir():
            if not model_dir.is_dir():
                continue

            for timestamp_dir in model_dir.iterdir():
                if not timestamp_dir.is_dir():
                    continue

                for item_dir in timestamp_dir.iterdir():
                    if not item_dir.is_dir():
                        continue

                    journal_path = item_dir / "journal.json"
                    patch_path = item_dir / "model_patch.diff"

                    if not journal_path.exists():
                        continue

                    try:
                        journal = json.loads(journal_path.read_text())
                        commits = journal.get("commits", {})
                        human_commit = commits.get("human")
                        pre_commit = commits.get("pre")

                        if not human_commit or not pre_commit:
                            continue

                        # Check if patch exists and has content
                        patch_exists = patch_path.exists() and patch_path.stat().st_size > 0

                        # Extract patch stats from journal metrics
                        metrics = journal.get("metrics", {})
                        patch_stats = None
                        if metrics:
                            patch_stats = {
                                "lines_added": metrics.get("patch_size_loc", 0),
                                "files_changed": metrics.get("changed_files_count", 0),
                            }

                        # Extract agent duration
                        claude_code_info = journal.get("claude_code", {})
                        agent_duration = claude_code_info.get("duration_s")

                        patch_info = ClaudeCodePatch(
                            patch_path=str(patch_path) if patch_exists else "",
                            human_commit=human_commit,
                            pre_commit=pre_commit,
                            item_id=item_dir.name,
                            journal_path=str(journal_path),
                            patch_stats=patch_stats,
                            agent_duration_s=agent_duration,
                        )

                        # Use first 8 chars of commit for matching
                        patches[human_commit[:8]] = patch_info
                        # Also store full hash
                        patches[human_commit] = patch_info

                    except Exception as e:
                        logger.warning(f"Error parsing {journal_path}: {e}")
                        continue

    logger.info(f"Discovered {len(patches) // 2} Claude Code patches")  # Divided by 2 since we store both short and full hash
    return patches


def _get_ephemeral_env() -> Dict[str, str]:
    """Get environment with caches redirected to ephemeral storage."""
    EPHEMERAL_CACHE_ROOT.mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()
    env.update({
        "PIP_CACHE_DIR": str(EPHEMERAL_CACHE_ROOT / "pip"),
        "UV_CACHE_DIR": str(EPHEMERAL_CACHE_ROOT / "uv"),
        "TRITON_CACHE_DIR": str(EPHEMERAL_CACHE_ROOT / "triton"),
        "HF_HOME": str(EPHEMERAL_CACHE_ROOT / "huggingface"),
        "TORCH_HOME": str(EPHEMERAL_CACHE_ROOT / "torch"),
        "XDG_CACHE_HOME": str(EPHEMERAL_CACHE_ROOT),
    })
    return env


def create_worktree(repo_path: Path, worktree_path: Path, commit: str) -> bool:
    """Create a git worktree at the specified commit."""
    try:
        # Remove existing worktree if present
        if worktree_path.exists():
            subprocess.run(
                ["git", "worktree", "remove", "--force", str(worktree_path)],
                cwd=repo_path,
                capture_output=True,
            )

        subprocess.run(
            ["git", "worktree", "add", "--detach", str(worktree_path), commit],
            cwd=repo_path,
            check=True,
            capture_output=True,
        )
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"Failed to create worktree: {e.stderr.decode() if e.stderr else e}")
        return False


def apply_patch(worktree_path: Path, patch_path: str) -> bool:
    """Apply a git patch to the worktree."""
    try:
        # First check if patch applies cleanly
        result = subprocess.run(
            ["git", "apply", "--check", patch_path],
            cwd=worktree_path,
            capture_output=True,
        )

        if result.returncode != 0:
            logger.warning(f"Patch check failed: {result.stderr.decode()}")
            # Try with more lenient options
            result = subprocess.run(
                ["git", "apply", "--check", "--ignore-whitespace", patch_path],
                cwd=worktree_path,
                capture_output=True,
            )
            if result.returncode != 0:
                return False

        # Apply the patch
        subprocess.run(
            ["git", "apply", patch_path],
            cwd=worktree_path,
            check=True,
            capture_output=True,
        )
        return True
    except subprocess.CalledProcessError as e:
        logger.warning(f"Failed to apply patch: {e}")
        return False


def run_test_script(
    test_script_content: str,
    worktree_path: Path,
    timeout: int = DEFAULT_TIMEOUT,
    use_pythonpath: bool = True,
) -> Tuple[str, str, int, Optional[Dict[str, Any]]]:
    """
    Run a test script and capture output.

    Args:
        test_script_content: Python script content from dataset
        worktree_path: Path to the git worktree
        timeout: Timeout in seconds
        use_pythonpath: If True, add worktree to PYTHONPATH

    Returns:
        Tuple of (stdout, stderr, return_code, parsed_result)
    """
    # Create temporary script file
    with tempfile.NamedTemporaryFile(
        mode='w',
        suffix='.py',
        delete=False,
        prefix='omniperf_test_'
    ) as f:
        f.write(test_script_content)
        test_script_path = f.name

    try:
        env = _get_ephemeral_env()

        if use_pythonpath:
            paths_to_add = []
            # Check for python/ subdirectory (common in sglang)
            python_sub = worktree_path / "python"
            if python_sub.exists():
                paths_to_add.append(str(python_sub))
            paths_to_add.append(str(worktree_path))

            current_path = os.environ.get("PYTHONPATH", "")
            if current_path:
                env["PYTHONPATH"] = f"{':'.join(paths_to_add)}:{current_path}"
            else:
                env["PYTHONPATH"] = ":".join(paths_to_add)

        result = subprocess.run(
            ["python3", test_script_path],
            cwd=worktree_path,
            capture_output=True,
            text=True,
            timeout=timeout,
            env=env,
        )

        # Parse JSON output from test script
        parsed_result = None
        for line in result.stdout.strip().split("\n"):
            line = line.strip()
            if line.startswith("{") and line.endswith("}"):
                try:
                    parsed_result = json.loads(line)
                    break
                except json.JSONDecodeError:
                    continue

        return result.stdout, result.stderr, result.returncode, parsed_result

    except subprocess.TimeoutExpired:
        return "", "Test timed out", -1, None
    finally:
        # Clean up temporary script
        try:
            os.unlink(test_script_path)
        except OSError:
            pass


def cleanup_worktree(repo_path: Path, worktree_path: Path) -> None:
    """Clean up a git worktree."""
    try:
        subprocess.run(
            ["git", "worktree", "remove", "--force", str(worktree_path)],
            cwd=repo_path,
            capture_output=True,
        )
    except Exception:
        pass

    # Prune orphaned worktrees
    subprocess.run(
        ["git", "worktree", "prune"],
        cwd=repo_path,
        capture_output=True,
    )


class OmniPerfBenchmarkRunner:
    """
    Main benchmark runner for OmniPerf dataset with Claude Code patches.
    """

    def __init__(
        self,
        vllm_repo_path: Path,
        sglang_repo_path: Path,
        state_root: Path,
        output_dir: Path,
        timeout: int = DEFAULT_TIMEOUT,
    ):
        self.vllm_repo_path = Path(vllm_repo_path).resolve()
        self.sglang_repo_path = Path(sglang_repo_path).resolve()
        self.state_root = Path(state_root).resolve()
        self.output_dir = Path(output_dir).resolve()
        self.timeout = timeout

        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Track results
        self.results: List[BenchmarkResult] = []

    def get_repo_path(self, repo_name: str) -> Optional[Path]:
        """Get repository path for the given repo name."""
        if repo_name.lower() == "vllm":
            return self.vllm_repo_path
        elif repo_name.lower() == "sglang":
            return self.sglang_repo_path
        else:
            logger.warning(f"Unknown repo: {repo_name}")
            return None

    def run_benchmark(
        self,
        instance: DatasetInstance,
        patch: Optional[ClaudeCodePatch],
    ) -> BenchmarkResult:
        """
        Run benchmark for a single dataset instance.

        Args:
            instance: Dataset instance with test script
            patch: Optional Claude Code patch to apply

        Returns:
            BenchmarkResult with timing data
        """
        start_time = time.time()
        commit_short = instance.commit_hash[:8]

        logger.info(f"Benchmarking {commit_short}: {instance.commit_subject[:50]}...")

        # Check if test script exists
        if not instance.test_script:
            return BenchmarkResult(
                status="no_test",
                commit_hash=instance.commit_hash,
                error_message="No test_script in dataset",
                duration_s=time.time() - start_time,
            )

        # Check if patch exists
        if not patch or not patch.patch_path:
            return BenchmarkResult(
                status="no_patch",
                commit_hash=instance.commit_hash,
                error_message="No Claude Code patch found for this commit",
                duration_s=time.time() - start_time,
            )

        # Get repository path
        repo_path = self.get_repo_path(instance.repo)
        if not repo_path or not repo_path.exists():
            return BenchmarkResult(
                status="error",
                commit_hash=instance.commit_hash,
                error_message=f"Repository not found: {instance.repo}",
                duration_s=time.time() - start_time,
            )

        # Create temporary worktree
        with tempfile.TemporaryDirectory(prefix="omniperf_worktree_") as tmp_dir:
            worktree_path = Path(tmp_dir) / "worktree"

            try:
                # Create worktree at pre-commit (baseline)
                if not create_worktree(repo_path, worktree_path, patch.pre_commit):
                    return BenchmarkResult(
                        status="error",
                        commit_hash=instance.commit_hash,
                        error_message=f"Failed to create worktree at {patch.pre_commit[:8]}",
                        duration_s=time.time() - start_time,
                    )

                # Run baseline test
                logger.info(f"  Running baseline test...")
                baseline_stdout, baseline_stderr, baseline_rc, baseline_result = run_test_script(
                    instance.test_script,
                    worktree_path,
                    timeout=self.timeout,
                )

                # Check baseline result
                if baseline_result is None:
                    return BenchmarkResult(
                        status="baseline_failed",
                        commit_hash=instance.commit_hash,
                        error_message="Baseline test failed to produce JSON output",
                        stdout=baseline_stdout,
                        stderr=baseline_stderr,
                        duration_s=time.time() - start_time,
                        patch_path=patch.patch_path,
                    )

                baseline_ms = baseline_result.get("avg_ms")
                if baseline_ms is None:
                    error_msg = baseline_result.get("error") or "No avg_ms in baseline output"
                    return BenchmarkResult(
                        status="baseline_failed",
                        commit_hash=instance.commit_hash,
                        error_message=error_msg,
                        baseline_output=baseline_result,
                        stdout=baseline_stdout,
                        stderr=baseline_stderr,
                        duration_s=time.time() - start_time,
                        patch_path=patch.patch_path,
                    )

                logger.info(f"  Baseline: {baseline_ms:.2f}ms")

                # Apply patch
                logger.info(f"  Applying patch...")
                if not apply_patch(worktree_path, patch.patch_path):
                    return BenchmarkResult(
                        status="patch_failed",
                        commit_hash=instance.commit_hash,
                        baseline_ms=baseline_ms,
                        baseline_output=baseline_result,
                        error_message="Failed to apply patch",
                        stdout=baseline_stdout,
                        stderr=baseline_stderr,
                        duration_s=time.time() - start_time,
                        patch_path=patch.patch_path,
                        patch_stats=patch.patch_stats,
                    )

                # Run patched test
                logger.info(f"  Running patched test...")
                patched_stdout, patched_stderr, patched_rc, patched_result = run_test_script(
                    instance.test_script,
                    worktree_path,
                    timeout=self.timeout,
                )

                # Check patched result
                if patched_result is None:
                    return BenchmarkResult(
                        status="patched_failed",
                        commit_hash=instance.commit_hash,
                        baseline_ms=baseline_ms,
                        baseline_output=baseline_result,
                        error_message="Patched test failed to produce JSON output",
                        stdout=baseline_stdout + "\n" + patched_stdout,
                        stderr=baseline_stderr + "\n" + patched_stderr,
                        duration_s=time.time() - start_time,
                        patch_path=patch.patch_path,
                        patch_stats=patch.patch_stats,
                    )

                patched_ms = patched_result.get("avg_ms")
                if patched_ms is None:
                    error_msg = patched_result.get("error") or "No avg_ms in patched output"
                    return BenchmarkResult(
                        status="patched_failed",
                        commit_hash=instance.commit_hash,
                        baseline_ms=baseline_ms,
                        baseline_output=baseline_result,
                        patched_output=patched_result,
                        error_message=error_msg,
                        stdout=baseline_stdout + "\n" + patched_stdout,
                        stderr=baseline_stderr + "\n" + patched_stderr,
                        duration_s=time.time() - start_time,
                        patch_path=patch.patch_path,
                        patch_stats=patch.patch_stats,
                    )

                logger.info(f"  Patched: {patched_ms:.2f}ms")

                # Compute speedup
                speedup = baseline_ms / patched_ms if patched_ms > 0 else None
                improvement = speedup > 1.0 if speedup else False

                if speedup:
                    logger.info(f"  Speedup: {speedup:.3f}x {'(improved)' if improvement else '(regressed)'}")

                return BenchmarkResult(
                    status="success",
                    commit_hash=instance.commit_hash,
                    baseline_ms=baseline_ms,
                    patched_ms=patched_ms,
                    speedup=speedup,
                    improvement=improvement,
                    baseline_output=baseline_result,
                    patched_output=patched_result,
                    stdout=baseline_stdout + "\n" + patched_stdout,
                    stderr=baseline_stderr + "\n" + patched_stderr,
                    duration_s=time.time() - start_time,
                    patch_path=patch.patch_path,
                    patch_stats=patch.patch_stats,
                )

            except subprocess.TimeoutExpired:
                return BenchmarkResult(
                    status="timeout",
                    commit_hash=instance.commit_hash,
                    error_message=f"Test timed out after {self.timeout}s",
                    duration_s=time.time() - start_time,
                    patch_path=patch.patch_path if patch else None,
                )
            except Exception as e:
                logger.exception(f"Error running benchmark for {commit_short}")
                return BenchmarkResult(
                    status="error",
                    commit_hash=instance.commit_hash,
                    error_message=str(e),
                    duration_s=time.time() - start_time,
                    patch_path=patch.patch_path if patch else None,
                )
            finally:
                cleanup_worktree(repo_path, worktree_path)

    def run_all(
        self,
        split: str = "vllm",
        limit: Optional[int] = None,
        commit_filter: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        Run benchmarks for all instances in the dataset.

        Args:
            split: Dataset split to benchmark ("vllm" or "sglang")
            limit: Optional limit on number of instances to process
            commit_filter: Optional list of commit hashes to filter

        Returns:
            Summary report dict
        """
        # Load dataset
        instances = load_omniperf_dataset(split)

        # Discover patches
        patches = discover_claude_code_patches(self.state_root, repo_filter=split)

        # Filter instances
        if commit_filter:
            instances = [
                i for i in instances
                if i.commit_hash[:8] in commit_filter or i.commit_hash in commit_filter
            ]

        if limit:
            instances = instances[:limit]

        logger.info(f"Processing {len(instances)} instances with {len(patches)//2} available patches")

        # Run benchmarks
        for i, instance in enumerate(instances):
            logger.info(f"[{i+1}/{len(instances)}] Processing {instance.commit_hash[:8]}")

            # Find matching patch
            patch = patches.get(instance.commit_hash[:8]) or patches.get(instance.commit_hash)

            result = self.run_benchmark(instance, patch)
            self.results.append(result)

            # Save individual result
            self._save_result(instance, result)

        # Generate summary report
        return self._generate_report(split)

    def _save_result(self, instance: DatasetInstance, result: BenchmarkResult) -> None:
        """Save individual benchmark result."""
        commit_short = instance.commit_hash[:8]
        result_dir = self.output_dir / instance.repo / commit_short
        result_dir.mkdir(parents=True, exist_ok=True)

        # Save result JSON
        result_path = result_dir / "benchmark_result.json"
        with open(result_path, "w") as f:
            json.dump({
                "instance": {
                    "commit_hash": instance.commit_hash,
                    "commit_subject": instance.commit_subject,
                    "repo": instance.repo,
                    "pr_url": instance.pr_url,
                    "files_changed": instance.files_changed,
                },
                "result": asdict(result),
            }, f, indent=2, default=str)

        # Save stdout/stderr
        if result.stdout:
            (result_dir / "stdout.txt").write_text(result.stdout)
        if result.stderr:
            (result_dir / "stderr.txt").write_text(result.stderr)

    def _generate_report(self, split: str) -> Dict[str, Any]:
        """Generate summary report."""
        report = {
            "generated_at": datetime.now().isoformat(),
            "dataset": HF_DATASET_ID,
            "split": split,
            "total_instances": len(self.results),
            "summary": {
                "success": 0,
                "no_patch": 0,
                "no_test": 0,
                "baseline_failed": 0,
                "patched_failed": 0,
                "patch_failed": 0,
                "timeout": 0,
                "error": 0,
            },
            "performance": {
                "improved": 0,
                "regressed": 0,
                "neutral": 0,
                "speedups": [],
                "avg_speedup": None,
                "median_speedup": None,
            },
            "results": [],
        }

        speedups = []

        for result in self.results:
            report["summary"][result.status] = report["summary"].get(result.status, 0) + 1

            result_entry = {
                "commit_hash": result.commit_hash,
                "status": result.status,
                "baseline_ms": result.baseline_ms,
                "patched_ms": result.patched_ms,
                "speedup": result.speedup,
                "improvement": result.improvement,
                "error": result.error_message,
            }
            report["results"].append(result_entry)

            if result.status == "success" and result.speedup is not None:
                speedups.append(result.speedup)
                if result.speedup > 1.05:
                    report["performance"]["improved"] += 1
                elif result.speedup < 0.95:
                    report["performance"]["regressed"] += 1
                else:
                    report["performance"]["neutral"] += 1

        if speedups:
            report["performance"]["speedups"] = sorted(speedups)
            report["performance"]["avg_speedup"] = sum(speedups) / len(speedups)
            sorted_speedups = sorted(speedups)
            report["performance"]["median_speedup"] = sorted_speedups[len(sorted_speedups) // 2]

        # Save report
        report_path = self.output_dir / f"omniperf_report_{split}.json"
        with open(report_path, "w") as f:
            json.dump(report, f, indent=2)

        logger.info(f"Report saved to {report_path}")

        return report


def print_summary(report: Dict[str, Any]) -> None:
    """Print human-readable summary."""
    print("\n" + "=" * 70)
    print(f"OMNIPERF BENCHMARK REPORT - {report['split'].upper()}")
    print("=" * 70)
    print(f"\nDataset: {report['dataset']}")
    print(f"Generated: {report['generated_at']}")
    print(f"\nTotal instances: {report['total_instances']}")

    print("\nStatus breakdown:")
    for status, count in report["summary"].items():
        if count > 0:
            print(f"  {status:20s}: {count}")

    perf = report["performance"]
    print("\nPerformance:")
    print(f"  Improved (>5%):     {perf['improved']}")
    print(f"  Regressed (<5%):    {perf['regressed']}")
    print(f"  Neutral:            {perf['neutral']}")

    if perf["avg_speedup"]:
        print(f"\n  Average speedup:    {perf['avg_speedup']:.3f}x")
        print(f"  Median speedup:     {perf['median_speedup']:.3f}x")


def main():
    parser = argparse.ArgumentParser(
        description="Run OmniPerf benchmarks for Claude Code patches"
    )
    parser.add_argument(
        "--vllm-repo",
        type=Path,
        required=True,
        help="Path to vLLM repository",
    )
    parser.add_argument(
        "--sglang-repo",
        type=Path,
        required=True,
        help="Path to SGLang repository",
    )
    parser.add_argument(
        "--state-root",
        type=Path,
        default=Path("./perf-agents-bench/state"),
        help="Path to perf-agents-bench state directory",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("./omniperf_results"),
        help="Output directory for results",
    )
    parser.add_argument(
        "--split",
        choices=["vllm", "sglang"],
        default="vllm",
        help="Dataset split to benchmark",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=DEFAULT_TIMEOUT,
        help="Timeout per test in seconds",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of instances to process",
    )
    parser.add_argument(
        "--commits",
        type=str,
        nargs="+",
        default=None,
        help="Specific commit hashes to benchmark",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose logging",
    )

    args = parser.parse_args()

    # Setup logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )

    # Create runner
    runner = OmniPerfBenchmarkRunner(
        vllm_repo_path=args.vllm_repo,
        sglang_repo_path=args.sglang_repo,
        state_root=args.state_root,
        output_dir=args.output_dir,
        timeout=args.timeout,
    )

    # Run benchmarks
    report = runner.run_all(
        split=args.split,
        limit=args.limit,
        commit_filter=args.commits,
    )

    # Print summary
    print_summary(report)


if __name__ == "__main__":
    main()
