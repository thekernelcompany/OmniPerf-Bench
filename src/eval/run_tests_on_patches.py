"""
Run test scripts on agent patches in isolated environments.

This module executes test scripts from the HuggingFace dataset against
agent-generated patches, measuring baseline and patched performance.
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import subprocess
import tempfile
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional

from .download_tests import find_test_script, load_test_index
from .run_summary import (
    RunSummary,
    load_summary,
    save_summary,
    add_evaluation_to_summary,
    generate_summary_from_state,
)

logger = logging.getLogger(__name__)

# Test execution settings
DEFAULT_TIMEOUT = 600  # 10 minutes per test
DEFAULT_WARMUP_RUNS = 3

# Pre-built vLLM wheel URL pattern (available for commits since v0.5.3)
VLLM_WHEEL_URL = "https://vllm-wheels.s3.us-west-2.amazonaws.com/{commit}/vllm-1.0.0.dev-cp38-abi3-manylinux1_x86_64.whl"

# Ephemeral cache root for runtime caches (wheel downloads, Triton JIT, etc.)
EPHEMERAL_CACHE_ROOT = Path("/ephemeral/cache")


def _get_ephemeral_cache_env() -> dict:
    """Get environment with caches redirected to /ephemeral.

    This ensures all runtime caches (wheel downloads, Triton JIT compilation,
    HuggingFace, PyTorch) go to ephemeral storage instead of filling up root.
    """
    EPHEMERAL_CACHE_ROOT.mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()
    env.update({
        "PIP_CACHE_DIR": str(EPHEMERAL_CACHE_ROOT / "pip"),
        "UV_CACHE_DIR": str(EPHEMERAL_CACHE_ROOT / "uv"),
        "TRITON_CACHE_DIR": str(EPHEMERAL_CACHE_ROOT / "triton"),
        "HF_HOME": str(EPHEMERAL_CACHE_ROOT / "huggingface"),
        "TORCH_HOME": str(EPHEMERAL_CACHE_ROOT / "torch"),
        "XDG_CACHE_HOME": str(EPHEMERAL_CACHE_ROOT),  # Fallback for misc caches
    })
    return env


@dataclass
class TestResult:
    """Result of running a test script."""
    status: str  # "success", "error", "timeout", "no_test", "no_patch", "patch_failed"
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


@dataclass
class RunMetadata:
    """Metadata for an agent run."""
    run_id: str
    item_id: str
    task_id: str
    human_commit: str
    pre_commit: str
    agent_status: str
    patch_path: Optional[str] = None
    test_script_path: Optional[str] = None
    # Hierarchical path components (optional, for new structure)
    repo: Optional[str] = None
    agent: Optional[str] = None
    model: Optional[str] = None
    timestamp: Optional[str] = None
    # Source directory path (for loading run_summary.json)
    source_dir: Optional[str] = None


class TestRunner:
    """
    Runs test scripts against agent patches in isolated environments.

    Key features:
    - Creates isolated git worktrees for each test
    - Applies agent patches without modifying test scripts
    - Runs baseline (no patch) and patched versions
    - Enforces timeouts and captures all output
    """

    def __init__(
        self,
        repo_path: Path,
        state_root: Path,
        output_dir: Path,
        test_index: Optional[Dict[str, str]] = None,
        timeout: int = DEFAULT_TIMEOUT,
        sglang_repo_path: Optional[Path] = None,
    ):
        """
        Initialize test runner.

        Args:
            repo_path: Path to the main repository (vllm)
            state_root: Path to perf-agents-bench/state directory
            output_dir: Directory to write evaluation results
            test_index: Pre-loaded commit->script index
            timeout: Timeout in seconds for each test execution
            sglang_repo_path: Path to sglang repository (auto-detected if None)
        """
        self.repo_path = Path(repo_path).resolve()
        self.state_root = Path(state_root).resolve()
        self.output_dir = Path(output_dir).resolve()
        self.timeout = timeout

        # Auto-detect sglang repo path
        if sglang_repo_path:
            self.sglang_repo_path = Path(sglang_repo_path).resolve()
        else:
            # Try common locations
            parent = self.repo_path.parent
            sglang_candidates = [parent / "sglang-repo", parent / "sglang"]
            for cand in sglang_candidates:
                if cand.exists() and (cand / ".git").exists():
                    self.sglang_repo_path = cand
                    break
            else:
                self.sglang_repo_path = None

        logger.info(f"vllm repo: {self.repo_path}")
        logger.info(f"sglang repo: {self.sglang_repo_path}")

        # Load test index if not provided
        if test_index is None:
            try:
                self.test_index = load_test_index()
            except FileNotFoundError:
                logger.warning("Test index not found. Run download_and_index_tests() first.")
                self.test_index = {}
        else:
            self.test_index = test_index

        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def discover_runs(self, run_ids: Optional[List[str]] = None) -> List[RunMetadata]:
        """
        Discover agent runs from state directory.

        Supports both flat and hierarchical directory structures:
        - Flat: state/runs/{run_id}/{item_id}/
        - Hierarchical: state/runs/{repo}/{agent}/{model}/{timestamp}/{item_id}/

        Args:
            run_ids: Optional list of specific run IDs to process

        Returns:
            List of RunMetadata for all discovered runs
        """
        runs_dir = self.state_root / "runs"
        if not runs_dir.exists():
            logger.error(f"Runs directory not found: {runs_dir}")
            return []

        discovered = []

        # Detect structure type by checking first-level directories
        first_level = [d for d in runs_dir.iterdir() if d.is_dir()]
        is_hierarchical = any(d.name in ("vllm", "sglang") for d in first_level)

        if is_hierarchical:
            # Hierarchical structure: {repo}/{agent}/{model}/{timestamp}/{item_id}/
            discovered = self._discover_hierarchical(runs_dir, run_ids)
        else:
            # Flat structure: {run_id}/{item_id}/
            discovered = self._discover_flat(runs_dir, run_ids)

        logger.info(f"Discovered {len(discovered)} runs")
        return discovered

    def _discover_flat(self, runs_dir: Path, run_ids: Optional[List[str]] = None) -> List[RunMetadata]:
        """Discover runs from flat directory structure."""
        discovered = []

        # Get run directories to process
        if run_ids:
            run_dirs = [runs_dir / rid for rid in run_ids if (runs_dir / rid).exists()]
        else:
            run_dirs = [d for d in runs_dir.iterdir() if d.is_dir()]

        for run_dir in run_dirs:
            run_id = run_dir.name

            for item_dir in run_dir.iterdir():
                if not item_dir.is_dir():
                    continue

                metadata = self._parse_item_dir(item_dir, run_id)
                if metadata:
                    discovered.append(metadata)

        return discovered

    def _discover_hierarchical(self, runs_dir: Path, run_ids: Optional[List[str]] = None) -> List[RunMetadata]:
        """Discover runs from hierarchical directory structure: {repo}/{agent}/{model}/{timestamp}/{item_id}/"""
        discovered = []

        for repo_dir in runs_dir.iterdir():
            if not repo_dir.is_dir():
                continue
            repo = repo_dir.name

            for agent_dir in repo_dir.iterdir():
                if not agent_dir.is_dir():
                    continue
                agent = agent_dir.name

                for model_dir in agent_dir.iterdir():
                    if not model_dir.is_dir():
                        continue
                    model = model_dir.name

                    for timestamp_dir in model_dir.iterdir():
                        if not timestamp_dir.is_dir():
                            continue
                        timestamp = timestamp_dir.name

                        # Build run_id for filtering
                        run_id = f"{repo}/{agent}/{model}/{timestamp}"

                        # Filter by run_ids if specified
                        if run_ids and not any(rid in run_id for rid in run_ids):
                            continue

                        for item_dir in timestamp_dir.iterdir():
                            if not item_dir.is_dir():
                                continue

                            metadata = self._parse_item_dir(
                                item_dir, run_id,
                                repo=repo, agent=agent, model=model, timestamp=timestamp
                            )
                            if metadata:
                                discovered.append(metadata)

        return discovered

    def _parse_item_dir(
        self,
        item_dir: Path,
        run_id: str,
        repo: Optional[str] = None,
        agent: Optional[str] = None,
        model: Optional[str] = None,
        timestamp: Optional[str] = None,
    ) -> Optional[RunMetadata]:
        """Parse an item directory and return RunMetadata if valid."""
        journal_path = item_dir / "journal.json"
        if not journal_path.exists():
            return None

        try:
            journal = json.loads(journal_path.read_text())
        except Exception as e:
            logger.error(f"Error reading {journal_path}: {e}")
            return None

        commits = journal.get("commits", {})
        human_commit = commits.get("human")
        pre_commit = commits.get("pre")

        if not human_commit or not pre_commit:
            logger.warning(f"Missing commits in {journal_path}")
            return None

        # Check for patch file
        patch_path = item_dir / "model_patch.diff"
        patch_exists = patch_path.exists() and patch_path.stat().st_size > 0

        # Find matching test script
        test_script = find_test_script(human_commit, self.test_index)

        return RunMetadata(
            run_id=run_id,
            item_id=item_dir.name,
            task_id=journal.get("task_id", "unknown"),
            human_commit=human_commit,
            pre_commit=pre_commit,
            agent_status=journal.get("status", "unknown"),
            patch_path=str(patch_path) if patch_exists else None,
            test_script_path=test_script,
            repo=repo,
            agent=agent,
            model=model,
            timestamp=timestamp,
            source_dir=str(item_dir),
        )

    def _get_repo_for_run(self, metadata: RunMetadata) -> Optional[Path]:
        """Determine which repo to use based on run metadata.

        Uses metadata.repo if available (hierarchical structure),
        otherwise falls back to parsing run_id (flat structure).
        """
        # Use repo field directly if available (hierarchical structure)
        if metadata.repo:
            repo_name = metadata.repo.lower()
        else:
            # Fall back to parsing run_id (flat structure)
            repo_name = metadata.run_id.lower()

        if repo_name.startswith("sglang"):
            if self.sglang_repo_path:
                return self.sglang_repo_path
            else:
                logger.warning(f"sglang repo not configured for run {metadata.run_id}")
                return None
        # Default to vllm repo for vllm_*, prefix_caching_opt, moe_align_opt, etc.
        return self.repo_path

    def run_single(self, metadata: RunMetadata) -> TestResult:
        """
        Run test for a single agent run.

        Args:
            metadata: Run metadata

        Returns:
            TestResult with baseline and patched timings
        """
        logger.info(f"Evaluating {metadata.run_id}/{metadata.item_id}")

        start_time = time.time()

        # Check if test script exists
        if not metadata.test_script_path:
            return TestResult(
                status="no_test",
                error_message=f"No test script found for commit {metadata.human_commit[:8]}",
                duration_s=time.time() - start_time,
            )

        # Determine which repo to use
        repo_path = self._get_repo_for_run(metadata)
        if not repo_path:
            return TestResult(
                status="error",
                error_message=f"No repo configured for run type: {metadata.run_id}",
                duration_s=time.time() - start_time,
            )

        # Check CUDA availability
        if not self._check_cuda():
            return TestResult(
                status="error",
                error_message="CUDA not available - GPU required for tests",
                duration_s=time.time() - start_time,
            )

        # Create isolated worktree
        with tempfile.TemporaryDirectory(prefix="eval_worktree_") as tmp_dir:
            worktree_path = Path(tmp_dir) / "worktree"

            try:
                # Create worktree at pre-commit
                self._create_worktree(worktree_path, metadata.pre_commit, repo_path)

                # Determine if we can use a wheel (vLLM only)
                repo_name = metadata.repo.lower() if metadata.repo else metadata.run_id.lower()
                is_vllm = not repo_name.startswith("sglang")
                wheel_installed = False

                if is_vllm and self._check_wheel_exists(metadata.pre_commit):
                    wheel_installed = self._install_vllm_wheel(metadata.pre_commit)

                # If no wheel, use PYTHONPATH overlay (sglang or vllm without wheel)
                use_pythonpath = not wheel_installed

                # Copy test script to isolated location (read-only protection)
                test_script_copy = Path(tmp_dir) / "test_script.py"
                shutil.copy2(metadata.test_script_path, test_script_copy)

                # Run baseline (no patch)
                logger.info("Running baseline test...")
                baseline_output, baseline_err, baseline_retcode = self._run_test(
                    test_script_copy, worktree_path, use_pythonpath=use_pythonpath
                )
                baseline_result = self._parse_test_output(baseline_output)

                # Apply patch if available
                patched_result = None
                patch_applied = False
                patched_output = ""
                patched_err = ""

                if metadata.patch_path:
                    logger.info("Applying agent patch...")

                    if wheel_installed:
                        # Apply patch to site-packages (for wheel mode)
                        patch_applied = self._apply_patch_to_wheel(metadata.patch_path, "vllm")
                    else:
                        # Apply patch to worktree (for PYTHONPATH mode)
                        patch_applied = self._apply_patch(worktree_path, metadata.patch_path)

                    if patch_applied:
                        # Run patched test
                        logger.info("Running patched test...")
                        patched_output, patched_err, patched_retcode = self._run_test(
                            test_script_copy, worktree_path, use_pythonpath=use_pythonpath
                        )
                        patched_result = self._parse_test_output(patched_output)

                        # If wheel was modified, reinstall it to restore original state
                        if wheel_installed:
                            logger.info("Reinstalling wheel to restore original state...")
                            self._install_vllm_wheel(metadata.pre_commit)

                # Compute metrics
                result = self._compute_result(
                    baseline_result,
                    patched_result,
                    patch_applied,
                    metadata,
                    baseline_output + "\n" + (patched_output if patched_result else ""),
                    baseline_err + "\n" + (patched_err if patched_result else ""),
                )
                result.duration_s = time.time() - start_time
                return result

            except subprocess.TimeoutExpired:
                return TestResult(
                    status="timeout",
                    error_message=f"Test timed out after {self.timeout}s",
                    duration_s=time.time() - start_time,
                )
            except Exception as e:
                logger.exception(f"Error running test for {metadata.item_id}")
                return TestResult(
                    status="error",
                    error_message=str(e),
                    duration_s=time.time() - start_time,
                )

    def run_all(
        self,
        run_ids: Optional[List[str]] = None,
        max_workers: int = 1,
    ) -> Dict[str, Dict[str, TestResult]]:
        """
        Run tests for all discovered runs.

        Args:
            run_ids: Optional list of specific run IDs to process
            max_workers: Number of parallel workers (default 1 for GPU)

        Returns:
            Nested dict: {run_id: {item_id: TestResult}}
        """
        runs = self.discover_runs(run_ids)

        if not runs:
            logger.warning("No runs found to evaluate")
            return {}

        results: Dict[str, Dict[str, TestResult]] = {}

        for i, metadata in enumerate(runs):
            logger.info(f"[{i+1}/{len(runs)}] Processing {metadata.run_id}/{metadata.item_id}")

            result = self.run_single(metadata)

            if metadata.run_id not in results:
                results[metadata.run_id] = {}
            results[metadata.run_id][metadata.item_id] = result

            # Save individual result
            self._save_result(metadata, result)

        return results

    def _check_cuda(self) -> bool:
        """Check if CUDA is available."""
        try:
            result = subprocess.run(
                ["python3", "-c", "import torch; print(torch.cuda.is_available())"],
                capture_output=True,
                text=True,
                timeout=30,
            )
            return result.stdout.strip().lower() == "true"
        except Exception:
            return False

    def _check_wheel_exists(self, commit: str) -> bool:
        """Check if pre-built vLLM wheel exists for commit."""
        import urllib.request
        url = VLLM_WHEEL_URL.format(commit=commit)
        try:
            req = urllib.request.Request(url, method='HEAD')
            urllib.request.urlopen(req, timeout=10)
            return True
        except Exception:
            return False

    def _install_vllm_wheel(self, commit: str) -> bool:
        """Install pre-built vLLM wheel for commit.

        Args:
            commit: Full commit hash

        Returns:
            True if installation succeeded, False otherwise
        """
        url = VLLM_WHEEL_URL.format(commit=commit)
        logger.info(f"Installing vllm wheel for {commit[:8]}...")
        try:
            # Use ephemeral cache for wheel downloads + skip version check
            env = _get_ephemeral_cache_env()
            env["UV_SKIP_WHEEL_FILENAME_CHECK"] = "1"
            subprocess.run(
                ["uv", "pip", "install", url, "--reinstall", "--quiet"],
                check=True,
                capture_output=True,
                timeout=300,  # 5 min max for wheel download+install (wheels are ~800MB)
                env=env,
            )
            logger.info(f"Wheel installed successfully")
            return True
        except subprocess.TimeoutExpired:
            logger.warning(f"Wheel install timed out for {commit[:8]}")
            return False
        except subprocess.CalledProcessError as e:
            logger.warning(f"Wheel install failed for {commit[:8]}: {e}")
            return False

    def _create_worktree(self, worktree_path: Path, commit: str, repo_path: Optional[Path] = None) -> None:
        """Create a git worktree at the specified commit."""
        cwd = repo_path or self.repo_path
        subprocess.run(
            ["git", "worktree", "add", "--detach", str(worktree_path), commit],
            cwd=cwd,
            check=True,
            capture_output=True,
        )

    def _install_deps(self, worktree_path: Path, commit: str, run_id: str) -> None:
        """Install dependencies using pre-built wheels when available.

        Strategy:
        - sglang: Use PYTHONPATH overlay (sglang already installed in bench-env)
        - vllm with wheel: Install commit-specific wheel (~30 sec)
        - vllm without wheel: Use PYTHONPATH overlay with base vllm

        Args:
            worktree_path: Path to the git worktree
            commit: The commit hash to install for
            run_id: Run identifier (used to detect sglang vs vllm)
        """
        # sglang: skip install, use PYTHONPATH overlay
        if run_id.lower().startswith("sglang"):
            logger.info(f"sglang run: using PYTHONPATH overlay")
            return

        # vllm: try pre-built wheel first
        if self._check_wheel_exists(commit):
            if self._install_vllm_wheel(commit):
                return  # Success!
            logger.warning(f"Wheel install failed for {commit[:8]}, using PYTHONPATH")
        else:
            logger.info(f"No wheel for {commit[:8]}, using PYTHONPATH overlay")

        # Fallback: rely on base vllm installation + PYTHONPATH
        # PYTHONPATH is set in _run_test() to point to worktree

    def _apply_patch(self, worktree_path: Path, patch_path: str) -> bool:
        """Apply a git patch to the worktree."""
        try:
            result = subprocess.run(
                ["git", "apply", "--check", patch_path],
                cwd=worktree_path,
                capture_output=True,
            )
            if result.returncode != 0:
                logger.warning(f"Patch check failed: {result.stderr.decode()}")
                return False

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

    def _apply_patch_to_wheel(self, patch_path: str, pkg_name: str = "vllm") -> bool:
        """Apply patch directly to installed wheel in site-packages.

        This allows using the compiled wheel while still testing Python patches.

        Args:
            patch_path: Path to the patch file
            pkg_name: Package name (vllm or sglang)

        Returns:
            True if any files were patched
        """
        import re
        import site

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

        # Parse patch to find modified Python files
        with open(patch_path, 'r') as f:
            patch_content = f.read()

        # Find files that match the package (e.g., vllm/*.py)
        modified_files = re.findall(r'^diff --git a/(.*) b/', patch_content, re.MULTILINE)
        python_files = [f for f in modified_files if f.endswith('.py') and f.startswith(pkg_name + '/')]

        if not python_files:
            logger.info(f"No Python files in {pkg_name}/ to patch")
            return False

        # Apply patch to site-packages
        try:
            result = subprocess.run(
                ["git", "apply", "--directory", str(site_packages), patch_path],
                capture_output=True,
                text=True,
            )
            if result.returncode != 0:
                # Try with more lenient options
                result = subprocess.run(
                    ["git", "apply", "--directory", str(site_packages), "--ignore-whitespace", patch_path],
                    capture_output=True,
                    text=True,
                )

            if result.returncode == 0:
                logger.info(f"Applied patch to {len(python_files)} files in site-packages")
                return True
            else:
                logger.warning(f"Failed to apply patch to site-packages: {result.stderr}")
                return False
        except Exception as e:
            logger.warning(f"Error applying patch to site-packages: {e}")
            return False

    def _run_test(
        self, test_script: Path, worktree_path: Path, use_pythonpath: bool = True
    ) -> tuple[str, str, int]:
        """Run a test script and capture output.

        Args:
            test_script: Path to the test script
            worktree_path: Path to the git worktree
            use_pythonpath: If True, set PYTHONPATH to worktree (for PYTHONPATH overlay mode).
                           If False, use installed packages (for wheel mode).
        """
        # Use ephemeral cache for Triton JIT, torch caches, etc.
        env = _get_ephemeral_cache_env()
        if use_pythonpath:
            env["PYTHONPATH"] = str(worktree_path)

        result = subprocess.run(
            ["python3", str(test_script)],
            cwd=worktree_path,
            capture_output=True,
            text=True,
            timeout=self.timeout,
            env=env,
        )
        return result.stdout, result.stderr, result.returncode

    def _parse_test_output(self, output: str) -> Optional[Dict[str, Any]]:
        """Parse JSON output from test script."""
        # Look for JSON object in output (test scripts print JSON summary)
        for line in output.strip().split("\n"):
            line = line.strip()
            if line.startswith("{") and line.endswith("}"):
                try:
                    return json.loads(line)
                except json.JSONDecodeError:
                    continue
        return None

    def _compute_result(
        self,
        baseline: Optional[Dict[str, Any]],
        patched: Optional[Dict[str, Any]],
        patch_applied: bool,
        metadata: RunMetadata,
        stdout: str,
        stderr: str,
    ) -> TestResult:
        """Compute final test result from baseline and patched outputs."""
        if baseline is None:
            return TestResult(
                status="error",
                error_message="Baseline test failed to produce output",
                stdout=stdout,
                stderr=stderr,
            )

        baseline_ms = baseline.get("avg_ms")

        if not patch_applied or metadata.patch_path is None:
            return TestResult(
                status="no_patch",
                baseline_ms=baseline_ms,
                baseline_output=baseline,
                stdout=stdout,
                stderr=stderr,
            )

        if patched is None:
            return TestResult(
                status="patch_failed",
                baseline_ms=baseline_ms,
                baseline_output=baseline,
                error_message="Patched test failed to produce output",
                stdout=stdout,
                stderr=stderr,
            )

        patched_ms = patched.get("avg_ms")

        # Compute speedup
        speedup = None
        improvement = False
        if baseline_ms and patched_ms and patched_ms > 0:
            speedup = baseline_ms / patched_ms
            improvement = speedup > 1.0

        return TestResult(
            status="success",
            baseline_ms=baseline_ms,
            patched_ms=patched_ms,
            speedup=speedup,
            improvement=improvement,
            baseline_output=baseline,
            patched_output=patched,
            stdout=stdout,
            stderr=stderr,
        )

    def _save_result(self, metadata: RunMetadata, result: TestResult) -> None:
        """Save test result to output directory.

        Uses hierarchical structure when metadata has repo/agent/model/timestamp:
        - Hierarchical: {repo}/{agent}/{model}/{timestamp}/{item_id}/
        - Flat fallback: {run_id}/{item_id}/

        Also handles run_summary.json:
        - Stage 2: Load from source dir, add evaluation, save to output
        """
        if metadata.repo and metadata.agent and metadata.model and metadata.timestamp:
            # Hierarchical output: {repo}/{agent}/{model}/{timestamp}/{item_id}/
            result_dir = (
                self.output_dir
                / metadata.repo
                / metadata.agent
                / metadata.model
                / metadata.timestamp
                / metadata.item_id
            )
        else:
            # Flat output fallback: {run_id}/{item_id}/
            result_dir = self.output_dir / metadata.run_id / metadata.item_id
        result_dir.mkdir(parents=True, exist_ok=True)

        # Save result JSON (legacy format for backward compatibility)
        result_path = result_dir / "test_results.json"
        with open(result_path, "w") as f:
            json.dump(
                {
                    "metadata": asdict(metadata) if hasattr(metadata, "__dataclass_fields__") else vars(metadata),
                    "result": asdict(result),
                },
                f,
                indent=2,
                default=str,
            )

        # Handle run_summary.json (Stage 2: load from source, add eval, save)
        summary = None
        if metadata.source_dir:
            source_summary_path = Path(metadata.source_dir) / "run_summary.json"
            summary = load_summary(source_summary_path)

        if summary:
            # Add evaluation results to existing summary
            add_evaluation_to_summary(
                summary,
                status=result.status,
                baseline_ms=result.baseline_ms,
                patched_ms=result.patched_ms,
                speedup=result.speedup,
                improvement=result.improvement,
                error=result.error_message,
            )
        else:
            # Generate summary from scratch (backward compatibility)
            if metadata.source_dir:
                summary = generate_summary_from_state(
                    item_dir=Path(metadata.source_dir),
                    repo=metadata.repo or "unknown",
                    agent=metadata.agent or "unknown",
                    model_hint=metadata.model or "unknown",
                    timestamp=metadata.timestamp or "unknown",
                )
                if summary:
                    add_evaluation_to_summary(
                        summary,
                        status=result.status,
                        baseline_ms=result.baseline_ms,
                        patched_ms=result.patched_ms,
                        speedup=result.speedup,
                        improvement=result.improvement,
                        error=result.error_message,
                    )

        # Save run_summary.json to output directory
        if summary:
            save_summary(summary, result_dir / "run_summary.json")

        # Save stdout/stderr
        if result.stdout:
            (result_dir / "test_stdout.txt").write_text(result.stdout)
        if result.stderr:
            (result_dir / "test_stderr.txt").write_text(result.stderr)

        logger.info(f"Saved results to {result_dir}")


def cleanup_worktrees(repo_path: Path) -> None:
    """Clean up any orphaned git worktrees."""
    subprocess.run(
        ["git", "worktree", "prune"],
        cwd=repo_path,
        capture_output=True,
    )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    # Example usage
    runner = TestRunner(
        repo_path=Path("/root/OmniPerf-Bench/vllm"),
        state_root=Path("/root/OmniPerf-Bench/perf-agents-bench/state"),
        output_dir=Path("/root/OmniPerf-Bench/eval_results"),
    )

    # Discover runs
    runs = runner.discover_runs()
    print(f"Found {len(runs)} runs")

    for r in runs[:5]:
        print(f"  {r.run_id}/{r.item_id}: {r.human_commit[:8]} (test: {'yes' if r.test_script_path else 'no'})")
