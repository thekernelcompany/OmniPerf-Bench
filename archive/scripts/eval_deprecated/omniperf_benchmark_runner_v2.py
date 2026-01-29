"""
OmniPerf Benchmark Runner v2 - With Wheel/Compilation Support.

This version supports:
- Pre-built vLLM wheels for faster benchmarking
- SGLang compilation for commits that need C extensions
- Caching of compiled packages to avoid recompilation

Usage:
    python -m src.eval.omniperf_benchmark_runner_v2 \
        --vllm-repo ./vllm \
        --sglang-repo ./sglang \
        --state-root ./perf-agents-bench/state \
        --output-dir ./omniperf_results \
        --split vllm \
        --use-wheels  # Enable wheel/compilation mode
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import re
import shutil
import site
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import urllib.request
import urllib.error

logger = logging.getLogger(__name__)

# Configuration
DEFAULT_TIMEOUT = 600  # 10 minutes per test
HF_DATASET_ID = "Ayushnangia/omniperf_v1"

# Pre-built vLLM wheel URL pattern
VLLM_WHEEL_URL = "https://vllm-wheels.s3.us-west-2.amazonaws.com/{commit}/vllm-1.0.0.dev-cp38-abi3-manylinux1_x86_64.whl"

# Cache directories - use /ephemeral for large files
EPHEMERAL_ROOT = Path("/ephemeral")
EPHEMERAL_CACHE = EPHEMERAL_ROOT / "omniperf_cache"
WHEEL_CACHE = EPHEMERAL_CACHE / "wheels"
BUILD_CACHE = EPHEMERAL_CACHE / "builds"


@dataclass
class BenchmarkResult:
    """Result of running a benchmark."""
    status: str
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
    used_wheel: bool = False
    used_compilation: bool = False


@dataclass
class DatasetInstance:
    """Represents an instance from the OmniPerf dataset."""
    commit_hash: str
    commit_subject: str
    repo: str
    test_script: Optional[str] = None
    perf_command: Optional[str] = None
    files_changed: List[str] = field(default_factory=list)
    pr_url: Optional[str] = None
    stats: Optional[Dict[str, Any]] = None

    @classmethod
    def from_hf_row(cls, row: Dict[str, Any]) -> "DatasetInstance":
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
    human_commit: str
    pre_commit: str
    item_id: str
    journal_path: str
    patch_stats: Optional[Dict[str, int]] = None
    agent_duration_s: Optional[float] = None


def _get_ephemeral_env() -> Dict[str, str]:
    """Get environment with caches on ephemeral storage."""
    EPHEMERAL_CACHE.mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()
    env.update({
        "PIP_CACHE_DIR": str(EPHEMERAL_CACHE / "pip"),
        "UV_CACHE_DIR": str(EPHEMERAL_CACHE / "uv"),
        "TRITON_CACHE_DIR": str(EPHEMERAL_CACHE / "triton"),
        "HF_HOME": str(EPHEMERAL_CACHE / "huggingface"),
        "TORCH_HOME": str(EPHEMERAL_CACHE / "torch"),
        "XDG_CACHE_HOME": str(EPHEMERAL_CACHE),
    })
    return env


def load_omniperf_dataset(split: str = "vllm") -> List[DatasetInstance]:
    """Load the OmniPerf dataset from HuggingFace."""
    try:
        from datasets import load_dataset
    except ImportError:
        raise ImportError("datasets library required")

    logger.info(f"Loading {HF_DATASET_ID} split={split}...")
    dataset = load_dataset(HF_DATASET_ID, split=split)
    instances = [DatasetInstance.from_hf_row(row) for row in dataset]
    logger.info(f"Loaded {len(instances)} instances from {split} split")
    return instances


def discover_claude_code_patches(
    state_root: Path,
    repo_filter: Optional[str] = None
) -> Dict[str, ClaudeCodePatch]:
    """Discover Claude Code patches from state directory."""
    runs_dir = state_root / "runs"
    if not runs_dir.exists():
        return {}

    patches: Dict[str, ClaudeCodePatch] = {}

    for repo_dir in runs_dir.iterdir():
        if not repo_dir.is_dir():
            continue
        if repo_filter and repo_dir.name.lower() != repo_filter.lower():
            continue

        claude_code_dir = repo_dir / "claude_code"
        if not claude_code_dir.exists():
            continue

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

                        patch_exists = patch_path.exists() and patch_path.stat().st_size > 0

                        patch_info = ClaudeCodePatch(
                            patch_path=str(patch_path) if patch_exists else "",
                            human_commit=human_commit,
                            pre_commit=pre_commit,
                            item_id=item_dir.name,
                            journal_path=str(journal_path),
                        )

                        patches[human_commit[:8]] = patch_info
                        patches[human_commit] = patch_info

                    except Exception as e:
                        logger.warning(f"Error parsing {journal_path}: {e}")

    logger.info(f"Discovered {len(patches) // 2} Claude Code patches")
    return patches


class WheelManager:
    """Manages vLLM wheel installation and caching."""

    def __init__(self):
        WHEEL_CACHE.mkdir(parents=True, exist_ok=True)
        self._installed_commit: Optional[str] = None

    def check_wheel_exists(self, commit: str) -> bool:
        """Check if pre-built wheel exists for commit."""
        url = VLLM_WHEEL_URL.format(commit=commit)
        try:
            req = urllib.request.Request(url, method='HEAD')
            urllib.request.urlopen(req, timeout=10)
            return True
        except Exception:
            return False

    def install_wheel(self, commit: str) -> bool:
        """Install pre-built vLLM wheel for commit."""
        if self._installed_commit == commit:
            logger.info(f"Wheel for {commit[:8]} already installed")
            return True

        url = VLLM_WHEEL_URL.format(commit=commit)
        logger.info(f"Installing vLLM wheel for {commit[:8]}...")

        try:
            env = _get_ephemeral_env()
            env["UV_SKIP_WHEEL_FILENAME_CHECK"] = "1"

            result = subprocess.run(
                ["uv", "pip", "install", url, "--reinstall", "--quiet"],
                capture_output=True,
                timeout=600,  # 10 min for large wheels
                env=env,
            )

            if result.returncode == 0:
                self._installed_commit = commit
                logger.info(f"Wheel installed successfully")
                return True
            else:
                logger.warning(f"Wheel install failed: {result.stderr.decode()}")
                return False

        except subprocess.TimeoutExpired:
            logger.warning(f"Wheel install timed out")
            return False
        except Exception as e:
            logger.warning(f"Wheel install error: {e}")
            return False

    def apply_patch_to_site_packages(self, patch_path: str, pkg_name: str = "vllm") -> bool:
        """Apply patch to installed package in site-packages."""
        # Find site-packages location
        site_packages = None
        for sp in site.getsitepackages():
            pkg_path = Path(sp) / pkg_name
            if pkg_path.exists():
                site_packages = Path(sp)
                break

        if not site_packages:
            logger.warning(f"Could not find {pkg_name} in site-packages")
            return False

        # Parse patch to find Python files
        with open(patch_path, 'r') as f:
            patch_content = f.read()

        modified_files = re.findall(r'^diff --git a/(.*) b/', patch_content, re.MULTILINE)
        python_files = [f for f in modified_files if f.endswith('.py') and f.startswith(pkg_name + '/')]

        if not python_files:
            logger.info(f"No Python files in {pkg_name}/ to patch")
            return False

        # Create a modified patch with corrected paths for site-packages
        # The patch has paths like "vllm/foo.py" but we need to apply to site-packages/vllm/foo.py
        # Use --directory to prepend site-packages, but strip the leading 'a/' and 'b/' prefixes properly
        try:
            # Method 1: Use patch command with -p0 and modified paths
            # Create temp patch with full paths
            import tempfile
            with tempfile.NamedTemporaryFile(mode='w', suffix='.patch', delete=False) as tmp:
                # Rewrite patch paths to be relative to site-packages
                modified_patch = patch_content
                for py_file in python_files:
                    # Replace paths like "a/vllm/foo.py" with the site-packages path
                    old_a = f"a/{py_file}"
                    old_b = f"b/{py_file}"
                    new_path = str(site_packages / py_file)
                    modified_patch = modified_patch.replace(old_a, new_path)
                    modified_patch = modified_patch.replace(old_b, new_path)
                tmp.write(modified_patch)
                tmp_path = tmp.name

            # Apply with patch command using -p0 (no path stripping)
            result = subprocess.run(
                ["patch", "-p0", "--forward", "--ignore-whitespace", "-i", tmp_path],
                capture_output=True,
                text=True,
                cwd="/",  # Use root since paths are absolute
            )

            os.unlink(tmp_path)

            if result.returncode == 0:
                logger.info(f"Applied patch to {len(python_files)} files in site-packages")
                return True

            # Method 2: Try git apply with different options
            result = subprocess.run(
                ["git", "apply", "-p1", "--unsafe-paths",
                 f"--directory={site_packages}", patch_path],
                capture_output=True,
                text=True,
            )

            if result.returncode == 0:
                logger.info(f"Applied patch via git apply")
                return True

            logger.warning(f"Failed to apply patch: {result.stderr}")
            return False

        except Exception as e:
            logger.warning(f"Error applying patch: {e}")
            return False

    def restore_wheel(self, commit: str) -> bool:
        """Restore wheel to original state by reinstalling."""
        return self.install_wheel(commit)


class BuildManager:
    """Manages compilation of packages from source."""

    def __init__(self, cache_dir: Path = BUILD_CACHE):
        self.cache_dir = cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def get_cache_key(self, repo_path: Path, commit: str) -> str:
        """Generate cache key for a built package."""
        return f"{repo_path.name}_{commit[:8]}"

    def get_cached_build(self, repo_path: Path, commit: str) -> Optional[Path]:
        """Check if we have a cached build for this commit."""
        cache_key = self.get_cache_key(repo_path, commit)
        cached_path = self.cache_dir / cache_key
        if cached_path.exists():
            return cached_path
        return None

    def build_package(self, worktree_path: Path, repo_name: str, commit: str) -> bool:
        """Build package from source in worktree."""
        logger.info(f"Building {repo_name} from source at {commit[:8]}...")

        env = _get_ephemeral_env()

        try:
            if repo_name == "sglang":
                # SGLang build
                python_dir = worktree_path / "python"
                if python_dir.exists():
                    build_dir = python_dir
                else:
                    build_dir = worktree_path

                # Install in development mode
                result = subprocess.run(
                    ["uv", "pip", "install", "-e", ".", "--quiet"],
                    cwd=build_dir,
                    capture_output=True,
                    timeout=1800,  # 30 min
                    env=env,
                )
            else:
                # vLLM build (fallback if no wheel)
                result = subprocess.run(
                    ["uv", "pip", "install", "-e", ".", "--quiet"],
                    cwd=worktree_path,
                    capture_output=True,
                    timeout=3600,  # 60 min for vLLM
                    env=env,
                )

            if result.returncode == 0:
                logger.info(f"Build completed successfully")
                return True
            else:
                logger.warning(f"Build failed: {result.stderr.decode()}")
                return False

        except subprocess.TimeoutExpired:
            logger.warning(f"Build timed out")
            return False
        except Exception as e:
            logger.warning(f"Build error: {e}")
            return False


def create_worktree(repo_path: Path, worktree_path: Path, commit: str) -> bool:
    """Create a git worktree at the specified commit."""
    try:
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
        result = subprocess.run(
            ["git", "apply", "--check", patch_path],
            cwd=worktree_path,
            capture_output=True,
        )

        if result.returncode != 0:
            result = subprocess.run(
                ["git", "apply", "--check", "--ignore-whitespace", patch_path],
                cwd=worktree_path,
                capture_output=True,
            )
            if result.returncode != 0:
                return False

        subprocess.run(
            ["git", "apply", patch_path],
            cwd=worktree_path,
            check=True,
            capture_output=True,
        )
        return True
    except subprocess.CalledProcessError:
        return False


def run_test_script(
    test_script_content: str,
    worktree_path: Path,
    timeout: int = DEFAULT_TIMEOUT,
    use_pythonpath: bool = True,
) -> Tuple[str, str, int, Optional[Dict[str, Any]]]:
    """Run a test script and capture output."""
    with tempfile.NamedTemporaryFile(
        mode='w', suffix='.py', delete=False, prefix='omniperf_test_'
    ) as f:
        f.write(test_script_content)
        test_script_path = f.name

    try:
        env = _get_ephemeral_env()

        if use_pythonpath:
            paths_to_add = []
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
    subprocess.run(["git", "worktree", "prune"], cwd=repo_path, capture_output=True)


class OmniPerfBenchmarkRunnerV2:
    """Enhanced benchmark runner with wheel/compilation support."""

    def __init__(
        self,
        vllm_repo_path: Path,
        sglang_repo_path: Path,
        state_root: Path,
        output_dir: Path,
        timeout: int = DEFAULT_TIMEOUT,
        use_wheels: bool = False,
        use_compilation: bool = False,
    ):
        self.vllm_repo_path = Path(vllm_repo_path).resolve()
        self.sglang_repo_path = Path(sglang_repo_path).resolve()
        self.state_root = Path(state_root).resolve()
        self.output_dir = Path(output_dir).resolve()
        self.timeout = timeout
        self.use_wheels = use_wheels
        self.use_compilation = use_compilation

        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.results: List[BenchmarkResult] = []

        # Initialize managers
        self.wheel_manager = WheelManager() if use_wheels else None
        self.build_manager = BuildManager() if use_compilation else None

    def get_repo_path(self, repo_name: str) -> Optional[Path]:
        if repo_name.lower() == "vllm":
            return self.vllm_repo_path
        elif repo_name.lower() == "sglang":
            return self.sglang_repo_path
        return None

    def run_benchmark(
        self,
        instance: DatasetInstance,
        patch: Optional[ClaudeCodePatch],
    ) -> BenchmarkResult:
        """Run benchmark for a single instance."""
        start_time = time.time()
        commit_short = instance.commit_hash[:8]

        logger.info(f"Benchmarking {commit_short}: {instance.commit_subject[:50]}...")

        if not instance.test_script:
            return BenchmarkResult(
                status="no_test",
                commit_hash=instance.commit_hash,
                error_message="No test_script in dataset",
                duration_s=time.time() - start_time,
            )

        if not patch or not patch.patch_path:
            return BenchmarkResult(
                status="no_patch",
                commit_hash=instance.commit_hash,
                error_message="No Claude Code patch found",
                duration_s=time.time() - start_time,
            )

        repo_path = self.get_repo_path(instance.repo)
        if not repo_path:
            return BenchmarkResult(
                status="error",
                commit_hash=instance.commit_hash,
                error_message=f"Repository not found: {instance.repo}",
                duration_s=time.time() - start_time,
            )

        # Determine execution strategy
        use_wheel = False
        use_compilation = False

        if instance.repo.lower() == "vllm" and self.use_wheels:
            if self.wheel_manager and self.wheel_manager.check_wheel_exists(patch.pre_commit):
                use_wheel = True
                logger.info(f"  Using pre-built wheel for {patch.pre_commit[:8]}")

        if not use_wheel and self.use_compilation:
            use_compilation = True
            logger.info(f"  Will compile from source")

        with tempfile.TemporaryDirectory(prefix="omniperf_worktree_") as tmp_dir:
            worktree_path = Path(tmp_dir) / "worktree"

            try:
                # Create worktree
                if not create_worktree(repo_path, worktree_path, patch.pre_commit):
                    return BenchmarkResult(
                        status="error",
                        commit_hash=instance.commit_hash,
                        error_message=f"Failed to create worktree at {patch.pre_commit[:8]}",
                        duration_s=time.time() - start_time,
                    )

                # Install/compile if needed
                if use_wheel:
                    if not self.wheel_manager.install_wheel(patch.pre_commit):
                        # Fall back to PYTHONPATH mode
                        use_wheel = False
                        logger.info(f"  Falling back to PYTHONPATH mode")

                if use_compilation and not use_wheel:
                    if not self.build_manager.build_package(worktree_path, instance.repo, patch.pre_commit):
                        # Fall back to PYTHONPATH mode
                        use_compilation = False
                        logger.info(f"  Compilation failed, using PYTHONPATH mode")

                # Run baseline test
                logger.info(f"  Running baseline test...")
                use_pythonpath = not (use_wheel or use_compilation)

                baseline_stdout, baseline_stderr, baseline_rc, baseline_result = run_test_script(
                    instance.test_script,
                    worktree_path,
                    timeout=self.timeout,
                    use_pythonpath=use_pythonpath,
                )

                if baseline_result is None:
                    return BenchmarkResult(
                        status="baseline_failed",
                        commit_hash=instance.commit_hash,
                        error_message="Baseline test failed to produce JSON output",
                        stdout=baseline_stdout,
                        stderr=baseline_stderr,
                        duration_s=time.time() - start_time,
                        patch_path=patch.patch_path,
                        used_wheel=use_wheel,
                        used_compilation=use_compilation,
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
                        used_wheel=use_wheel,
                        used_compilation=use_compilation,
                    )

                logger.info(f"  Baseline: {baseline_ms:.2f}ms")

                # Apply patch
                logger.info(f"  Applying patch...")
                patch_applied = False

                if use_wheel:
                    # Apply to site-packages
                    patch_applied = self.wheel_manager.apply_patch_to_site_packages(
                        patch.patch_path, "vllm"
                    )
                    if not patch_applied:
                        # Try applying to worktree as fallback
                        patch_applied = apply_patch(worktree_path, patch.patch_path)
                else:
                    patch_applied = apply_patch(worktree_path, patch.patch_path)

                if not patch_applied:
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
                        used_wheel=use_wheel,
                        used_compilation=use_compilation,
                    )

                # Run patched test
                logger.info(f"  Running patched test...")
                patched_stdout, patched_stderr, patched_rc, patched_result = run_test_script(
                    instance.test_script,
                    worktree_path,
                    timeout=self.timeout,
                    use_pythonpath=use_pythonpath,
                )

                # Restore wheel if needed
                if use_wheel:
                    self.wheel_manager.restore_wheel(patch.pre_commit)

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
                        used_wheel=use_wheel,
                        used_compilation=use_compilation,
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
                        used_wheel=use_wheel,
                        used_compilation=use_compilation,
                    )

                logger.info(f"  Patched: {patched_ms:.2f}ms")

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
                    used_wheel=use_wheel,
                    used_compilation=use_compilation,
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
        """Run benchmarks for all instances."""
        instances = load_omniperf_dataset(split)
        patches = discover_claude_code_patches(self.state_root, repo_filter=split)

        if commit_filter:
            instances = [
                i for i in instances
                if i.commit_hash[:8] in commit_filter or i.commit_hash in commit_filter
            ]

        if limit:
            instances = instances[:limit]

        logger.info(f"Processing {len(instances)} instances with {len(patches)//2} patches")

        if self.use_wheels:
            logger.info("Wheel mode enabled - will use pre-built vLLM wheels when available")
        if self.use_compilation:
            logger.info("Compilation mode enabled - will compile from source when needed")

        for i, instance in enumerate(instances):
            logger.info(f"[{i+1}/{len(instances)}] Processing {instance.commit_hash[:8]}")

            patch = patches.get(instance.commit_hash[:8]) or patches.get(instance.commit_hash)
            result = self.run_benchmark(instance, patch)
            self.results.append(result)

            self._save_result(instance, result)

        return self._generate_report(split)

    def _save_result(self, instance: DatasetInstance, result: BenchmarkResult) -> None:
        """Save individual result."""
        result_dir = self.output_dir / instance.repo / instance.commit_hash[:8]
        result_dir.mkdir(parents=True, exist_ok=True)

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
            "config": {
                "use_wheels": self.use_wheels,
                "use_compilation": self.use_compilation,
            },
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
            "execution": {
                "used_wheel": 0,
                "used_compilation": 0,
                "used_pythonpath": 0,
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

            if result.used_wheel:
                report["execution"]["used_wheel"] += 1
            elif result.used_compilation:
                report["execution"]["used_compilation"] += 1
            else:
                report["execution"]["used_pythonpath"] += 1

            result_entry = {
                "commit_hash": result.commit_hash,
                "status": result.status,
                "baseline_ms": result.baseline_ms,
                "patched_ms": result.patched_ms,
                "speedup": result.speedup,
                "improvement": result.improvement,
                "error": result.error_message,
                "used_wheel": result.used_wheel,
                "used_compilation": result.used_compilation,
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

    print("\nExecution mode:")
    exec_info = report.get("execution", {})
    print(f"  Used wheels:      {exec_info.get('used_wheel', 0)}")
    print(f"  Used compilation: {exec_info.get('used_compilation', 0)}")
    print(f"  Used PYTHONPATH:  {exec_info.get('used_pythonpath', 0)}")

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
        description="Run OmniPerf benchmarks with wheel/compilation support"
    )
    parser.add_argument("--vllm-repo", type=Path, required=True)
    parser.add_argument("--sglang-repo", type=Path, required=True)
    parser.add_argument("--state-root", type=Path, default=Path("./perf-agents-bench/state"))
    parser.add_argument("--output-dir", type=Path, default=Path("./omniperf_results"))
    parser.add_argument("--split", choices=["vllm", "sglang"], default="vllm")
    parser.add_argument("--timeout", type=int, default=DEFAULT_TIMEOUT)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--commits", type=str, nargs="+", default=None)
    parser.add_argument("--use-wheels", action="store_true",
                        help="Use pre-built vLLM wheels when available")
    parser.add_argument("--use-compilation", action="store_true",
                        help="Compile from source when wheels unavailable")
    parser.add_argument("--verbose", "-v", action="store_true")

    args = parser.parse_args()

    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )

    runner = OmniPerfBenchmarkRunnerV2(
        vllm_repo_path=args.vllm_repo,
        sglang_repo_path=args.sglang_repo,
        state_root=args.state_root,
        output_dir=args.output_dir,
        timeout=args.timeout,
        use_wheels=args.use_wheels,
        use_compilation=args.use_compilation,
    )

    report = runner.run_all(
        split=args.split,
        limit=args.limit,
        commit_filter=args.commits,
    )

    print_summary(report)


if __name__ == "__main__":
    main()
