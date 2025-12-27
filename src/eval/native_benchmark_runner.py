"""
Native Benchmark Runner for OmniPerf Dataset.

Uses per-commit wheels from https://wheels.vllm.ai/ for accurate benchmarking.
This avoids C extension coupling issues that occur with PYTHONPATH approach.

Runs vLLM/SGLang native benchmarks and collects metrics:
- TTFT (Time To First Token)
- Throughput (tokens/sec, requests/sec)
- Latency (mean, P50, P90, P99)
- ITL (Inter-Token Latency)

Usage:
    python -m src.eval.native_benchmark_runner \
        --vllm-repo ./vllm \
        --sglang-repo ./sglang \
        --state-root ./perf-agents-bench/state \
        --output-dir ./omniperf_results \
        --split vllm \
        --limit 5
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import shutil
import signal
import subprocess
import tempfile
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# Configuration
DEFAULT_TIMEOUT = 3600  # 1 hour per benchmark
HF_DATASET_ID = "Ayushnangia/omniperf_v1"
VLLM_WHEELS_BASE_URL = "https://wheels.vllm.ai"

# Ephemeral storage for large files (model weights, caches)
EPHEMERAL_ROOT = Path("/ephemeral")
EPHEMERAL_HF_HOME = EPHEMERAL_ROOT / "huggingface"
EPHEMERAL_CACHE = EPHEMERAL_ROOT / "cache"
EPHEMERAL_VENV = EPHEMERAL_ROOT / "benchmark_venv"
EPHEMERAL_WORKTREE = EPHEMERAL_ROOT / "benchmark_worktree"


@dataclass
class BenchmarkMetrics:
    """Native benchmark metrics from vLLM/SGLang."""
    # Throughput metrics
    request_throughput: Optional[float] = None  # requests/sec
    token_throughput: Optional[float] = None  # tokens/sec
    output_throughput: Optional[float] = None  # output tokens/sec

    # Latency metrics (ms)
    ttft_mean: Optional[float] = None  # Time To First Token
    ttft_p50: Optional[float] = None
    ttft_p90: Optional[float] = None
    ttft_p99: Optional[float] = None

    tpot_mean: Optional[float] = None  # Time Per Output Token
    tpot_p50: Optional[float] = None
    tpot_p90: Optional[float] = None
    tpot_p99: Optional[float] = None

    itl_mean: Optional[float] = None  # Inter-Token Latency
    itl_p50: Optional[float] = None
    itl_p90: Optional[float] = None
    itl_p99: Optional[float] = None

    e2e_latency_mean: Optional[float] = None  # End-to-end latency
    e2e_latency_p50: Optional[float] = None
    e2e_latency_p90: Optional[float] = None
    e2e_latency_p99: Optional[float] = None

    # For latency benchmarks
    latency_ms: Optional[float] = None

    # Raw output for debugging
    raw_output: str = ""


@dataclass
class BenchmarkResult:
    """Result of running a benchmark."""
    status: str  # success, error, timeout, no_patch, server_failed, benchmark_failed
    commit_hash: str
    benchmark_type: str  # serving, latency, throughput
    baseline_metrics: Optional[BenchmarkMetrics] = None
    human_metrics: Optional[BenchmarkMetrics] = None  # Human's actual commit
    agent_metrics: Optional[BenchmarkMetrics] = None  # Claude Code's patch
    human_improvement: Optional[Dict[str, float]] = None  # human vs baseline
    agent_improvement: Optional[Dict[str, float]] = None  # agent vs baseline
    agent_vs_human: Optional[Dict[str, float]] = None  # agent vs human (positive = agent better)
    error_message: Optional[str] = None
    stdout: str = ""
    stderr: str = ""
    duration_s: float = 0.0
    patch_path: Optional[str] = None
    # Legacy compatibility
    patched_metrics: Optional[BenchmarkMetrics] = None
    improvement: Optional[Dict[str, float]] = None
    perf_command: Optional[str] = None


@dataclass
class DatasetInstance:
    """Represents an instance from the OmniPerf dataset."""
    commit_hash: str
    commit_subject: str
    repo: str
    perf_command: Optional[str] = None
    test_script: Optional[str] = None
    files_changed: List[str] = field(default_factory=list)
    pr_url: Optional[str] = None
    models: List[str] = field(default_factory=list)

    @classmethod
    def from_hf_row(cls, row: Dict[str, Any]) -> "DatasetInstance":
        return cls(
            commit_hash=row.get("commit_hash", ""),
            commit_subject=row.get("commit_subject", ""),
            repo=row.get("repo", "unknown"),
            perf_command=row.get("perf_command"),
            test_script=row.get("test_script"),
            files_changed=row.get("files_changed", []),
            pr_url=row.get("pr_url"),
            models=row.get("models", []),
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


def get_ephemeral_env() -> Dict[str, str]:
    """Get environment with caches on ephemeral storage."""
    EPHEMERAL_HF_HOME.mkdir(parents=True, exist_ok=True)
    EPHEMERAL_CACHE.mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()
    env.update({
        "HF_HOME": str(EPHEMERAL_HF_HOME),
        "TRANSFORMERS_CACHE": str(EPHEMERAL_HF_HOME / "transformers"),
        "HF_DATASETS_CACHE": str(EPHEMERAL_HF_HOME / "datasets"),
        "TORCH_HOME": str(EPHEMERAL_CACHE / "torch"),
        "TRITON_CACHE_DIR": str(EPHEMERAL_CACHE / "triton"),
        "XDG_CACHE_HOME": str(EPHEMERAL_CACHE),
        "PIP_CACHE_DIR": str(EPHEMERAL_CACHE / "pip"),
    })
    # Ensure HF_TOKEN is passed through for gated model access
    hf_token = os.environ.get("HF_TOKEN")
    # Fallback: read from default HF cache location
    if not hf_token:
        token_file = Path.home() / ".cache" / "huggingface" / "token"
        if token_file.exists():
            hf_token = token_file.read_text().strip()
    if hf_token:
        env["HF_TOKEN"] = hf_token
        env["HUGGING_FACE_HUB_TOKEN"] = hf_token
    return env


def get_full_commit_hash(repo_path: Path, short_hash: str) -> Optional[str]:
    """Get full 40-char commit hash from short hash."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", short_hash],
            cwd=repo_path,
            capture_output=True,
            text=True,
            check=True,
        )
        return result.stdout.strip()
    except subprocess.CalledProcessError:
        return None


def discover_wheel_url(repo_path: Path, commit: str) -> Optional[str]:
    """
    Discover vLLM wheel URL for a given commit.

    Uses the nightly wheels repository at https://wheels.vllm.ai/<commit>/vllm/
    Wheel format: vllm-<version>.dev<N>+g<short_hash>-cp38-abi3-manylinux1_x86_64.whl
    """
    import urllib.request
    import urllib.error
    from html.parser import HTMLParser

    # Get full commit hash
    full_hash = get_full_commit_hash(repo_path, commit)
    if not full_hash:
        logger.warning(f"Could not resolve commit {commit}")
        return None

    # Wheels are in the vllm/ subdirectory
    wheel_dir_url = f"{VLLM_WHEELS_BASE_URL}/{full_hash}/vllm/"

    try:
        # Fetch directory listing
        req = urllib.request.Request(wheel_dir_url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=30) as response:
            html = response.read().decode()
    except urllib.error.HTTPError as e:
        if e.code == 404:
            logger.warning(f"No wheel found for commit {commit[:8]} at {wheel_dir_url}")
        else:
            logger.warning(f"HTTP error {e.code} fetching {wheel_dir_url}")
        return None
    except Exception as e:
        logger.warning(f"Error fetching wheel URL: {e}")
        return None

    # Parse HTML to find .whl file
    class WheelLinkParser(HTMLParser):
        def __init__(self):
            super().__init__()
            self.wheel_files = []

        def handle_starttag(self, tag, attrs):
            if tag == "a":
                for name, value in attrs:
                    if name == "href" and value.endswith(".whl"):
                        # Handle relative paths (../) in href
                        whl_name = value.split("/")[-1] if "/" in value else value
                        self.wheel_files.append(whl_name)

    parser = WheelLinkParser()
    parser.feed(html)

    if not parser.wheel_files:
        logger.warning(f"No wheel files found in {wheel_dir_url}")
        return None

    # Prefer manylinux wheel
    for whl in parser.wheel_files:
        if "manylinux" in whl and "x86_64" in whl:
            # URL-encode the + sign in version string
            whl_encoded = whl.replace("+", "%2B")
            # Wheel files are in parent directory (../)
            return f"{VLLM_WHEELS_BASE_URL}/{full_hash}/{whl_encoded}"

    # Fallback to first wheel
    whl = parser.wheel_files[0]
    whl_encoded = whl.replace("+", "%2B")
    return f"{VLLM_WHEELS_BASE_URL}/{full_hash}/{whl_encoded}"


def setup_benchmark_venv(venv_path: Path) -> bool:
    """Create or verify virtual environment for benchmarks."""
    if not venv_path.exists():
        logger.info(f"Creating virtual environment at {venv_path}")
        try:
            subprocess.run(
                ["uv", "venv", "--python", "3.12", str(venv_path)],
                check=True,
                capture_output=True,
            )
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to create venv: {e.stderr.decode() if e.stderr else e}")
            return False
    return True


def install_vllm_wheel(venv_path: Path, wheel_url: str, reinstall: bool = False) -> bool:
    """Install vLLM from wheel URL using uv pip install."""
    # Use system uv with the target venv
    pip_cmd = [
        "uv", "pip", "install",
        "--python", str(venv_path / "bin" / "python"),
        wheel_url,
    ]
    if reinstall:
        pip_cmd.append("--reinstall")

    logger.info(f"Installing wheel: {wheel_url.split('/')[-1]}")

    try:
        result = subprocess.run(
            pip_cmd,
            capture_output=True,
            text=True,
            timeout=600,  # 10 min timeout for installation
            env={**os.environ, "VIRTUAL_ENV": str(venv_path)},
        )
        if result.returncode != 0:
            logger.error(f"Wheel install failed: {result.stderr}")
            return False
        return True
    except subprocess.TimeoutExpired:
        logger.error("Wheel installation timed out")
        return False
    except Exception as e:
        logger.error(f"Error installing wheel: {e}")
        return False


def get_installed_vllm_version(venv_path: Path) -> Optional[str]:
    """Get installed vLLM version in the venv."""
    try:
        result = subprocess.run(
            [str(venv_path / "bin" / "python"), "-c", "import vllm; print(vllm.__version__)"],
            capture_output=True,
            text=True,
            cwd="/ephemeral",  # Avoid path conflicts
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except:
        pass
    return None


def load_omniperf_dataset(split: str = "vllm") -> List[DatasetInstance]:
    """Load the OmniPerf dataset from HuggingFace."""
    try:
        from datasets import load_dataset
    except ImportError:
        raise ImportError("datasets library required. Install with: pip install datasets")

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
        logger.error(f"Runs directory not found: {runs_dir}")
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


def parse_benchmark_output(output: str, benchmark_type: str) -> BenchmarkMetrics:
    """Parse vLLM/SGLang benchmark output to extract metrics."""
    metrics = BenchmarkMetrics(raw_output=output)

    # Common patterns in vLLM benchmark output
    patterns = {
        # Throughput - various formats
        r"Request throughput:\s*([\d.]+)\s*requests/s": "request_throughput",
        r"Output token throughput:\s*([\d.]+)\s*tokens/s": "output_throughput",
        r"Total Token throughput:\s*([\d.]+)\s*tokens/s": "token_throughput",
        # benchmark_throughput.py format: "Throughput: X requests/s, Y tokens/s"
        r"Throughput:\s*([\d.]+)\s*requests/s": "request_throughput",
        r"Throughput:\s*[\d.]+\s*requests/s,\s*([\d.]+)\s*tokens/s": "token_throughput",

        # TTFT
        r"Mean TTFT \(ms\):\s*([\d.]+)": "ttft_mean",
        r"Median TTFT \(ms\):\s*([\d.]+)": "ttft_p50",
        r"P90 TTFT \(ms\):\s*([\d.]+)": "ttft_p90",
        r"P99 TTFT \(ms\):\s*([\d.]+)": "ttft_p99",

        # TPOT (Time Per Output Token)
        r"Mean TPOT \(ms\):\s*([\d.]+)": "tpot_mean",
        r"Median TPOT \(ms\):\s*([\d.]+)": "tpot_p50",
        r"P90 TPOT \(ms\):\s*([\d.]+)": "tpot_p90",
        r"P99 TPOT \(ms\):\s*([\d.]+)": "tpot_p99",

        # ITL (Inter-Token Latency)
        r"Mean ITL \(ms\):\s*([\d.]+)": "itl_mean",
        r"Median ITL \(ms\):\s*([\d.]+)": "itl_p50",
        r"P90 ITL \(ms\):\s*([\d.]+)": "itl_p90",
        r"P99 ITL \(ms\):\s*([\d.]+)": "itl_p99",

        # E2E Latency
        r"Mean E2E Latency \(ms\):\s*([\d.]+)": "e2e_latency_mean",
        r"Median E2E Latency \(ms\):\s*([\d.]+)": "e2e_latency_p50",
        r"P90 E2E Latency \(ms\):\s*([\d.]+)": "e2e_latency_p90",
        r"P99 E2E Latency \(ms\):\s*([\d.]+)": "e2e_latency_p99",

        # Latency benchmark specific
        r"Avg latency:\s*([\d.]+)\s*(ms|seconds)": "latency_ms",
    }

    for pattern, attr in patterns.items():
        match = re.search(pattern, output, re.IGNORECASE)
        if match:
            value = float(match.group(1))
            # Convert seconds to ms if needed
            if attr == "latency_ms" and len(match.groups()) > 1 and match.group(2) == "seconds":
                value *= 1000
            setattr(metrics, attr, value)

    # Also try to parse JSON output (some benchmarks output JSON)
    try:
        # Look for JSON block in output
        json_match = re.search(r'\{[^{}]*"throughput"[^{}]*\}', output, re.DOTALL)
        if json_match:
            data = json.loads(json_match.group())
            if "throughput" in data:
                metrics.token_throughput = data["throughput"]
            if "latency" in data:
                metrics.latency_ms = data["latency"] * 1000 if data["latency"] < 100 else data["latency"]
    except:
        pass

    return metrics


def compute_improvement(baseline: BenchmarkMetrics, patched: BenchmarkMetrics) -> Dict[str, float]:
    """Compute percentage improvement for each metric."""
    improvement = {}

    # Throughput metrics (higher is better)
    for attr in ["request_throughput", "token_throughput", "output_throughput"]:
        b = getattr(baseline, attr)
        p = getattr(patched, attr)
        if b and p and b > 0:
            improvement[attr] = ((p - b) / b) * 100

    # Latency metrics (lower is better, so we invert)
    for attr in ["ttft_mean", "ttft_p50", "ttft_p90", "ttft_p99",
                 "tpot_mean", "tpot_p50", "tpot_p90", "tpot_p99",
                 "itl_mean", "itl_p50", "itl_p90", "itl_p99",
                 "e2e_latency_mean", "e2e_latency_p50", "e2e_latency_p90", "e2e_latency_p99",
                 "latency_ms"]:
        b = getattr(baseline, attr)
        p = getattr(patched, attr)
        if b and p and b > 0:
            # Positive improvement = lower latency
            improvement[attr] = ((b - p) / b) * 100

    return improvement


class ServerManager:
    """Manages vLLM/SGLang server lifecycle."""

    def __init__(self, repo_type: str = "vllm"):
        self.repo_type = repo_type
        self.process: Optional[subprocess.Popen] = None
        self.port = 8000

    def start(self, model: str, worktree_path: Path, venv_path: Path = None, extra_args: List[str] = None, pythonpath_overlay: Path = None) -> bool:
        """Start the inference server.

        NOTE: We do NOT set PYTHONPATH to the worktree because the server needs
        to use the installed vllm wheel (which has compiled _C extensions).
        Only the pythonpath_overlay is used if provided (for agent patches).
        """
        env = get_ephemeral_env()
        # Only set PYTHONPATH for overlay (agent patches), not for worktree
        # Server must use the installed wheel (with compiled _C extensions)
        if pythonpath_overlay:
            env["PYTHONPATH"] = str(pythonpath_overlay)

        # Use venv python if provided, otherwise default python
        python_bin = str(venv_path / "bin" / "python") if venv_path else "python"

        if self.repo_type == "vllm":
            cmd = [
                python_bin, "-m", "vllm.entrypoints.openai.api_server",
                "--model", model,
                "--port", str(self.port),
            ]
        else:  # sglang
            cmd = [
                python_bin, "-m", "sglang.launch_server",
                "--model-path", model,
                "--port", str(self.port),
            ]

        if extra_args:
            cmd.extend(extra_args)

        logger.info(f"Starting server: {' '.join(cmd)}")

        try:
            # IMPORTANT: Don't run from worktree_path - Python would find local vllm source
            # Run from /ephemeral so only the installed wheel is used
            self.process = subprocess.Popen(
                cmd,
                cwd="/ephemeral",
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                preexec_fn=os.setsid,
            )

            # Wait for server to be ready
            return self._wait_for_ready()
        except Exception as e:
            logger.error(f"Failed to start server: {e}")
            return False

    def _wait_for_ready(self, timeout: int = 300) -> bool:
        """Wait for server to be ready."""
        import urllib.request
        import urllib.error

        start = time.time()
        url = f"http://localhost:{self.port}/health"

        while time.time() - start < timeout:
            try:
                urllib.request.urlopen(url, timeout=5)
                logger.info("Server is ready")
                return True
            except (urllib.error.URLError, ConnectionRefusedError):
                time.sleep(2)

            # Check if process died
            if self.process and self.process.poll() is not None:
                logger.error("Server process died")
                return False

        logger.error("Server startup timeout")
        return False

    def stop(self):
        """Stop the server."""
        if self.process:
            try:
                os.killpg(os.getpgid(self.process.pid), signal.SIGTERM)
                self.process.wait(timeout=10)
            except:
                try:
                    os.killpg(os.getpgid(self.process.pid), signal.SIGKILL)
                except:
                    pass
            self.process = None


class NativeBenchmarkRunner:
    """Runs native vLLM/SGLang benchmarks."""

    # Models that require multi-GPU configurations
    LARGE_MODEL_GPU_MAP = {
        "deepseek-ai/DeepSeek-V3": "H100:8",
        "deepseek-ai/DeepSeek-V3-0324": "H100:8",
        "deepseek-ai/DeepSeek-V2": "H100:8",  # 236B MoE needs 8x H100
        "nvidia/Nemotron-4-340B": "H100:8",
        "meta-llama/Llama-4-Scout-17B-16E": "H100:2",
        "meta-llama/Meta-Llama-3-70B": "H100:4",
    }

    def __init__(
        self,
        vllm_repo_path: Path,
        sglang_repo_path: Path,
        state_root: Path,
        output_dir: Path,
        timeout: int = DEFAULT_TIMEOUT,
        use_modal: bool = False,
    ):
        self.vllm_repo_path = Path(vllm_repo_path).resolve()
        self.sglang_repo_path = Path(sglang_repo_path).resolve()
        self.state_root = Path(state_root).resolve()
        self.output_dir = Path(output_dir).resolve()
        self.timeout = timeout
        self.use_modal = use_modal

        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.results: List[BenchmarkResult] = []

        if self.use_modal:
            logger.info("Modal cloud GPU execution enabled for large models")

    def get_repo_path(self, repo_name: str) -> Optional[Path]:
        if repo_name.lower() == "vllm":
            return self.vllm_repo_path
        elif repo_name.lower() == "sglang":
            return self.sglang_repo_path
        return None

    def detect_benchmark_type(self, perf_command: str) -> str:
        """Detect benchmark type from command."""
        if "benchmark_serving" in perf_command or "bench serve" in perf_command:
            return "serving"
        elif "benchmark_latency" in perf_command or "bench latency" in perf_command:
            return "latency"
        elif "benchmark_throughput" in perf_command or "bench throughput" in perf_command:
            return "throughput"
        return "unknown"

    def extract_model_from_command(self, perf_command: str) -> Optional[str]:
        """Extract model name from benchmark command."""
        # Try --model flag
        match = re.search(r'--model[=\s]+["\']?([^\s"\']+)', perf_command)
        if match:
            return match.group(1)
        # Try --model-path flag
        match = re.search(r'--model-path[=\s]+["\']?([^\s"\']+)', perf_command)
        if match:
            return match.group(1)
        return None

    def should_use_modal(self, model: str, perf_command: str) -> Tuple[bool, Optional[str]]:
        """Determine if benchmark should use Modal cloud GPUs.

        Returns:
            Tuple of (should_use_modal, gpu_config)
            gpu_config is like "H100:4" or None
        """
        if not self.use_modal:
            return False, None

        # Check for explicit tensor parallelism in command
        tp_match = re.search(r'(?:-tp|--tensor-parallel-size)\s+(\d+)', perf_command)
        if tp_match:
            tp_size = int(tp_match.group(1))
            if tp_size > 1:
                if tp_size >= 8:
                    return True, "H100:8"
                elif tp_size >= 4:
                    return True, "H100:4"
                elif tp_size >= 2:
                    return True, "H100:2"

        # Check known large models
        if model:
            for pattern, config in self.LARGE_MODEL_GPU_MAP.items():
                if pattern.lower() in model.lower():
                    return True, config

        return False, None

    def run_benchmark_on_modal(
        self,
        wheel_url: str,
        perf_command: str,
        model: str,
        benchmark_type: str,
        gpu_config: str,
    ) -> Tuple[Dict[str, Any], str]:
        """Run benchmark on Modal cloud GPU.

        Returns:
            Tuple of (metrics_dict, error_message)
        """
        try:
            from src.eval.modal_benchmark import run_modal_benchmark
        except ImportError:
            logger.warning("Modal benchmark module not available, falling back to local")
            return {}, "Modal module not available"

        logger.info(f"  Dispatching to Modal with {gpu_config}...")

        try:
            result = run_modal_benchmark(
                wheel_url=wheel_url,
                perf_command=perf_command,
                model=model,
                benchmark_type=benchmark_type,
                gpu_config=gpu_config,
            )

            if result.get("status") == "success":
                return result.get("metrics", {}), ""
            else:
                return {}, result.get("error", "Unknown Modal error")

        except Exception as e:
            logger.error(f"Modal execution failed: {e}")
            return {}, str(e)

    def _run_benchmark_on_modal(
        self,
        instance: "DatasetInstance",
        patch: "ClaudeCodePatch",
        baseline_wheel: str,
        human_wheel: str,
        benchmark_type: str,
        model: str,
        gpu_config: str,
        start_time: float,
    ) -> "BenchmarkResult":
        """Run full 3-way benchmark comparison on Modal cloud GPUs.

        This is used for large models that don't fit on a single GPU.
        Uses the new run_3way_modal_benchmark function that runs all 3 benchmarks
        in a single Modal container for efficiency.
        """
        try:
            # Import the 3-way benchmark function
            import importlib.util
            module_path = Path(__file__).parent / "modal_benchmark.py"
            if module_path.exists():
                spec = importlib.util.spec_from_file_location("modal_benchmark", module_path)
                modal_benchmark = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(modal_benchmark)
                run_3way_modal_benchmark = modal_benchmark.run_3way_modal_benchmark
            else:
                raise ImportError(f"Modal benchmark module not found at {module_path}")
        except Exception as e:
            logger.error(f"Modal module not available: {e}")
            return BenchmarkResult(
                status="error",
                commit_hash=instance.commit_hash,
                benchmark_type=benchmark_type,
                error_message=f"Modal module not available: {e}",
                duration_s=time.time() - start_time,
                perf_command=instance.perf_command,
                patch_path=patch.patch_path,
            )

        # Read agent patch content if available
        agent_patch = None
        if patch.patch_path and Path(patch.patch_path).exists():
            try:
                agent_patch = Path(patch.patch_path).read_text()
                logger.info(f"  Loaded agent patch ({len(agent_patch)} chars)")
            except Exception as e:
                logger.warning(f"  Could not read agent patch: {e}")

        # Run all 3 benchmarks in single Modal container
        logger.info(f"  Running 3-way benchmark on Modal ({gpu_config})...")
        logger.info(f"    Baseline: {baseline_wheel.split('/')[-1][:50]}")
        logger.info(f"    Human: {human_wheel.split('/')[-1][:50]}")
        logger.info(f"    Agent patch: {'Yes' if agent_patch else 'No'}")

        try:
            result = run_3way_modal_benchmark(
                baseline_wheel_url=baseline_wheel,
                human_wheel_url=human_wheel,
                agent_patch=agent_patch,
                perf_command=instance.perf_command,
                model=model,
                gpu_config=gpu_config,
            )
        except Exception as e:
            logger.error(f"Modal execution failed: {e}")
            return BenchmarkResult(
                status="error",
                commit_hash=instance.commit_hash,
                benchmark_type=benchmark_type,
                error_message=f"Modal execution failed: {e}",
                duration_s=time.time() - start_time,
                perf_command=instance.perf_command,
                patch_path=patch.patch_path,
            )

        if result.get("status") != "success":
            return BenchmarkResult(
                status=result.get("status", "error"),
                commit_hash=instance.commit_hash,
                benchmark_type=benchmark_type,
                error_message=result.get("error", "Unknown Modal error"),
                duration_s=time.time() - start_time,
                perf_command=instance.perf_command,
                patch_path=patch.patch_path,
            )

        # Convert Modal result metrics to BenchmarkMetrics objects
        baseline_metrics_dict = result.get("baseline_metrics", {})
        human_metrics_dict = result.get("human_metrics", {})
        agent_metrics_dict = result.get("agent_metrics")

        baseline_metrics = BenchmarkMetrics(
            token_throughput=baseline_metrics_dict.get("output_throughput") or baseline_metrics_dict.get("total_throughput"),
            request_throughput=baseline_metrics_dict.get("request_throughput"),
            ttft_mean=baseline_metrics_dict.get("ttft_mean"),
            ttft_p50=baseline_metrics_dict.get("ttft_median"),
            ttft_p99=baseline_metrics_dict.get("ttft_p99"),
            tpot_mean=baseline_metrics_dict.get("tpot_mean"),
            itl_mean=baseline_metrics_dict.get("itl_mean"),
        )

        human_metrics = BenchmarkMetrics(
            token_throughput=human_metrics_dict.get("output_throughput") or human_metrics_dict.get("total_throughput"),
            request_throughput=human_metrics_dict.get("request_throughput"),
            ttft_mean=human_metrics_dict.get("ttft_mean"),
            ttft_p50=human_metrics_dict.get("ttft_median"),
            ttft_p99=human_metrics_dict.get("ttft_p99"),
            tpot_mean=human_metrics_dict.get("tpot_mean"),
            itl_mean=human_metrics_dict.get("itl_mean"),
        )

        agent_metrics = None
        if agent_metrics_dict:
            agent_metrics = BenchmarkMetrics(
                token_throughput=agent_metrics_dict.get("output_throughput") or agent_metrics_dict.get("total_throughput"),
                request_throughput=agent_metrics_dict.get("request_throughput"),
                ttft_mean=agent_metrics_dict.get("ttft_mean"),
                ttft_p50=agent_metrics_dict.get("ttft_median"),
                ttft_p99=agent_metrics_dict.get("ttft_p99"),
                tpot_mean=agent_metrics_dict.get("tpot_mean"),
                itl_mean=agent_metrics_dict.get("itl_mean"),
            )

        logger.info(f"  Results from Modal:")
        logger.info(f"    Baseline: {baseline_metrics.request_throughput or baseline_metrics.token_throughput}")
        logger.info(f"    Human: {human_metrics.request_throughput or human_metrics.token_throughput}")
        logger.info(f"    Human improvement: {result.get('human_improvement', {})}")
        if agent_metrics:
            logger.info(f"    Agent: {agent_metrics.request_throughput or agent_metrics.token_throughput}")
            logger.info(f"    Agent improvement: {result.get('agent_improvement', {})}")
        else:
            logger.info(f"    Agent: skipped ({result.get('agent_error', 'no patch')})")

        return BenchmarkResult(
            status="success",
            commit_hash=instance.commit_hash,
            benchmark_type=benchmark_type,
            baseline_metrics=baseline_metrics,
            human_metrics=human_metrics,
            agent_metrics=agent_metrics,
            human_improvement=result.get("human_improvement", {}),
            agent_improvement=result.get("agent_improvement"),
            agent_vs_human=result.get("agent_vs_human"),
            duration_s=time.time() - start_time,
            perf_command=instance.perf_command,
            patch_path=patch.patch_path,
        )

    def translate_command(self, perf_command: str, worktree_path: Path) -> str:
        """Translate old benchmark commands to new CLI format if needed."""
        # Replace old script paths with new CLI commands
        cmd = perf_command

        # Handle old vLLM benchmark script paths
        if "benchmarks/benchmark_serving.py" in cmd:
            # Check if worktree has the old script
            old_script = worktree_path / "benchmarks" / "benchmark_serving.py"
            if old_script.exists():
                # Read first line to check if deprecated
                content = old_script.read_text()[:500]
                if "DEPRECATED" in content:
                    # Use new CLI
                    cmd = cmd.replace("python benchmarks/benchmark_serving.py", "vllm bench serve")
                    cmd = cmd.replace("python3 benchmarks/benchmark_serving.py", "vllm bench serve")

        if "benchmarks/benchmark_latency.py" in cmd:
            old_script = worktree_path / "benchmarks" / "benchmark_latency.py"
            if old_script.exists():
                content = old_script.read_text()[:500]
                if "DEPRECATED" in content:
                    cmd = cmd.replace("python benchmarks/benchmark_latency.py", "vllm bench latency")
                    cmd = cmd.replace("python3 benchmarks/benchmark_latency.py", "vllm bench latency")

        return cmd

    def run_benchmark_command(
        self,
        command: str,
        worktree_path: Path,
        benchmark_type: str,
        venv_path: Optional[Path] = None,
        pythonpath_overlay: Optional[Path] = None,
    ) -> Tuple[str, str, int]:
        """Run a benchmark command and capture output."""
        env = get_ephemeral_env()

        # Use venv if provided (wheel-based approach)
        if venv_path:
            env["PATH"] = f"{venv_path}/bin:{env.get('PATH', '')}"
            env["VIRTUAL_ENV"] = str(venv_path)

            # If pythonpath_overlay provided, prepend it to override installed package files
            if pythonpath_overlay:
                env["PYTHONPATH"] = f"{pythonpath_overlay}:{env.get('PYTHONPATH', '')}"
        else:
            # Fallback: Add worktree to PYTHONPATH for imports
            python_path = str(worktree_path)
            if (worktree_path / "python").exists():
                python_path = f"{worktree_path}/python:{python_path}"
            if os.environ.get("PYTHONPATH"):
                python_path = f"{python_path}:{os.environ['PYTHONPATH']}"
            env["PYTHONPATH"] = python_path

        # Translate command if needed
        cmd = self.translate_command(command, worktree_path)

        # Replace 'python' with venv python and make script paths absolute
        if venv_path:
            venv_python = str(venv_path / "bin" / "python")
            import re

            # Replace python commands at start of line or after shell operators
            # This avoids matching 'python' inside file paths
            def replace_python(match):
                prefix = match.group(1) or ""
                py = match.group(2)
                rest = match.group(3)

                # If followed by benchmarks/, make path absolute
                if rest.startswith("benchmarks/"):
                    return f"{prefix}{venv_python} {worktree_path}/{rest}"
                return f"{prefix}{venv_python} {rest}"

            # Match python/python3 at start or after common shell operators (|, &&, ;, etc)
            # Negative lookbehind to avoid matching in paths like /bin/python
            cmd = re.sub(
                r'(^|[|;&]\s*)(python3?)\s+(.+?)(?=$|[|;&])',
                replace_python,
                cmd,
                flags=re.MULTILINE
            )

            # Handle vllm CLI commands
            if cmd.startswith("vllm "):
                cmd = f"{venv_path}/bin/{cmd}"

        logger.info(f"Running: {cmd}")

        # Run from worktree for correct relative paths, but with venv in PATH
        cwd = worktree_path

        try:
            result = subprocess.run(
                cmd,
                shell=True,
                cwd=cwd,
                capture_output=True,
                text=True,
                timeout=self.timeout,
                env=env,
            )
            return result.stdout, result.stderr, result.returncode
        except subprocess.TimeoutExpired:
            return "", "Benchmark timed out", -1

    def prepare_serving_command(self, perf_command: str) -> str:
        """Clean serving benchmark command and add host/port.

        Serving benchmarks are CLIENTS that connect to a running server.
        Server-specific args (--dtype, --tensor-parallel-size) should not be
        passed to the benchmark client.
        """
        cmd = perf_command

        # Remove server-only args that don't belong in benchmark client
        cmd = re.sub(r'--dtype\s+\S+', '', cmd)
        cmd = re.sub(r'--tensor-parallel-size\s+\d+', '', cmd)
        cmd = re.sub(r'--trust-remote-code', '', cmd)
        cmd = re.sub(r'--max-model-len\s+\d+', '', cmd)

        # Add host/port if not present (defaults for vLLM server)
        if '--host' not in cmd and '--base-url' not in cmd:
            cmd += ' --host 127.0.0.1'
        if '--port' not in cmd and '--base-url' not in cmd:
            cmd += ' --port 8000'

        # CRITICAL FIX: benchmark_serving.py defaults to --dataset-name sharegpt
        # which requires --dataset-path. If neither is specified, the benchmark
        # crashes immediately. Add --dataset-name random with default lengths.
        if '--dataset-name' not in cmd and '--dataset-path' not in cmd and '--dataset' not in cmd:
            cmd += ' --dataset-name random'
            # Add default random lengths if not present
            if '--random-input-len' not in cmd:
                cmd += ' --random-input-len 512'
            if '--random-output-len' not in cmd:
                cmd += ' --random-output-len 128'

        # Clean up multiple spaces
        cmd = re.sub(r'\s+', ' ', cmd).strip()

        return cmd

    def run_serving_benchmark(
        self,
        perf_command: str,
        worktree_path: Path,
        venv_path: Path,
        model: str,
        pythonpath_overlay: Path = None,
    ) -> Tuple[str, str, int]:
        """Run a serving benchmark with server lifecycle management.

        For serving benchmarks (benchmark_serving.py / vllm bench serve):
        1. Start the vLLM server with the model
        2. Run the benchmark client against the server
        3. Stop the server

        Args:
            pythonpath_overlay: Optional path to prepend to PYTHONPATH for both
                server and benchmark client (used for agent patch testing)
        """
        server = ServerManager(repo_type="vllm")

        try:
            # Start server with venv python
            logger.info(f"  Starting vLLM server for model: {model}")
            if not server.start(model, worktree_path, venv_path=venv_path, pythonpath_overlay=pythonpath_overlay):
                # Try to capture any error output from the server
                server_stderr = ""
                if server.process:
                    try:
                        _, server_stderr = server.process.communicate(timeout=5)
                    except:
                        pass
                return "", f"Failed to start vLLM server\n{server_stderr}", -1

            # Clean command: remove server-only args, add host/port
            cmd = self.prepare_serving_command(perf_command)
            logger.info(f"  Running benchmark client: {cmd[:100]}...")

            # Run benchmark (client doesn't need overlay, only server does)
            stdout, stderr, rc = self.run_benchmark_command(
                cmd, worktree_path, "serving", venv_path=venv_path
            )
            return stdout, stderr, rc

        finally:
            # Always stop server
            logger.info("  Stopping vLLM server")
            server.stop()

    def create_worktree(self, repo_path: Path, worktree_path: Path, commit: str) -> bool:
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
            logger.error(f"Failed to create worktree: {e}")
            return False

    def apply_patch(self, worktree_path: Path, patch_path: str) -> bool:
        """Apply a git patch to the worktree."""
        try:
            # Check if patch applies
            result = subprocess.run(
                ["git", "apply", "--check", patch_path],
                cwd=worktree_path,
                capture_output=True,
            )

            if result.returncode != 0:
                # Try with lenient options
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
        except subprocess.CalledProcessError:
            return False

    def cleanup_worktree(self, repo_path: Path, worktree_path: Path):
        """Clean up a git worktree."""
        try:
            subprocess.run(
                ["git", "worktree", "remove", "--force", str(worktree_path)],
                cwd=repo_path,
                capture_output=True,
            )
        except:
            pass
        subprocess.run(["git", "worktree", "prune"], cwd=repo_path, capture_output=True)

    def create_agent_overlay(self, worktree_path: Path, patch_path: str) -> Optional[Path]:
        """
        Create an overlay directory with agent's patched files.
        Returns the overlay path or None if patch application fails.
        """
        overlay_path = EPHEMERAL_ROOT / "agent_overlay"
        if overlay_path.exists():
            shutil.rmtree(overlay_path)
        overlay_path.mkdir(parents=True)

        # Parse the patch to get affected files
        try:
            with open(patch_path, 'r') as f:
                patch_content = f.read()

            # Extract file paths from patch (simplified parsing)
            import re
            file_matches = re.findall(r'^diff --git a/(.+?) b/', patch_content, re.MULTILINE)
            if not file_matches:
                # Try unified diff format
                file_matches = re.findall(r'^--- a/(.+)$', patch_content, re.MULTILINE)

            for rel_path in file_matches:
                src_file = worktree_path / rel_path
                if src_file.exists():
                    # Copy original file to overlay
                    dst_file = overlay_path / rel_path
                    dst_file.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(src_file, dst_file)

            # Apply patch to overlay files
            result = subprocess.run(
                ["git", "apply", "--directory", str(overlay_path), patch_path],
                cwd=worktree_path,
                capture_output=True,
            )

            if result.returncode != 0:
                # Try with more lenient options
                result = subprocess.run(
                    ["patch", "-p1", "-d", str(overlay_path), "-i", patch_path],
                    capture_output=True,
                )
                if result.returncode != 0:
                    logger.warning(f"Failed to apply agent patch: {result.stderr.decode()}")
                    return None

            return overlay_path

        except Exception as e:
            logger.warning(f"Error creating agent overlay: {e}")
            return None

    def run_benchmark(
        self,
        instance: DatasetInstance,
        patch: Optional[ClaudeCodePatch],
    ) -> BenchmarkResult:
        """
        Run 3-way benchmark comparison:
        1. Baseline (pre-commit wheel)
        2. Human (human commit wheel - the actual optimization)
        3. Agent (baseline wheel + Claude Code patch overlay)
        """
        start_time = time.time()
        commit_short = instance.commit_hash[:8]

        logger.info(f"Benchmarking {commit_short}: {instance.commit_subject[:50]}...")

        # Check requirements
        if not instance.perf_command:
            return BenchmarkResult(
                status="no_perf_command",
                commit_hash=instance.commit_hash,
                benchmark_type="unknown",
                error_message="No perf_command in dataset",
                duration_s=time.time() - start_time,
            )

        if not patch:
            return BenchmarkResult(
                status="no_patch",
                commit_hash=instance.commit_hash,
                benchmark_type=self.detect_benchmark_type(instance.perf_command),
                error_message="No Claude Code patch found",
                duration_s=time.time() - start_time,
                perf_command=instance.perf_command,
            )

        repo_path = self.get_repo_path(instance.repo)
        if not repo_path:
            return BenchmarkResult(
                status="error",
                commit_hash=instance.commit_hash,
                benchmark_type="unknown",
                error_message=f"Repository not found: {instance.repo}",
                duration_s=time.time() - start_time,
            )

        benchmark_type = self.detect_benchmark_type(instance.perf_command)

        # Discover wheel URLs
        logger.info(f"  Discovering wheels for baseline ({patch.pre_commit[:8]}) and human ({patch.human_commit[:8]})...")
        baseline_wheel = discover_wheel_url(repo_path, patch.pre_commit)
        human_wheel = discover_wheel_url(repo_path, patch.human_commit)

        if not baseline_wheel:
            return BenchmarkResult(
                status="no_wheel",
                commit_hash=instance.commit_hash,
                benchmark_type=benchmark_type,
                error_message=f"No wheel for baseline {patch.pre_commit[:8]}",
                duration_s=time.time() - start_time,
                perf_command=instance.perf_command,
                patch_path=patch.patch_path,
            )

        if not human_wheel:
            return BenchmarkResult(
                status="no_wheel",
                commit_hash=instance.commit_hash,
                benchmark_type=benchmark_type,
                error_message=f"No wheel for human {patch.human_commit[:8]}",
                duration_s=time.time() - start_time,
                perf_command=instance.perf_command,
                patch_path=patch.patch_path,
            )

        # Check if this benchmark requires Modal (multi-GPU)
        model = self.extract_model_from_command(instance.perf_command)
        use_modal, gpu_config = self.should_use_modal(model, instance.perf_command)

        if use_modal:
            logger.info(f"  Large model detected ({model}), using Modal {gpu_config}...")
            return self._run_benchmark_on_modal(
                instance=instance,
                patch=patch,
                baseline_wheel=baseline_wheel,
                human_wheel=human_wheel,
                benchmark_type=benchmark_type,
                model=model,
                gpu_config=gpu_config,
                start_time=start_time,
            )

        # Setup venv
        venv_path = EPHEMERAL_VENV
        if not setup_benchmark_venv(venv_path):
            return BenchmarkResult(
                status="error",
                commit_hash=instance.commit_hash,
                benchmark_type=benchmark_type,
                error_message="Failed to setup venv",
                duration_s=time.time() - start_time,
                perf_command=instance.perf_command,
                patch_path=patch.patch_path,
            )

        worktree_path = EPHEMERAL_WORKTREE
        all_stdout, all_stderr = [], []

        try:
            # Create worktree at pre-commit for benchmark scripts
            if not self.create_worktree(repo_path, worktree_path, patch.pre_commit):
                return BenchmarkResult(
                    status="error",
                    commit_hash=instance.commit_hash,
                    benchmark_type=benchmark_type,
                    error_message=f"Failed to create worktree",
                    duration_s=time.time() - start_time,
                    perf_command=instance.perf_command,
                )

            # ========== 1. BASELINE BENCHMARK ==========
            logger.info(f"  [1/3] Installing baseline wheel...")
            if not install_vllm_wheel(venv_path, baseline_wheel, reinstall=True):
                return BenchmarkResult(
                    status="wheel_install_failed",
                    commit_hash=instance.commit_hash,
                    benchmark_type=benchmark_type,
                    error_message="Failed to install baseline wheel",
                    duration_s=time.time() - start_time,
                    perf_command=instance.perf_command,
                    patch_path=patch.patch_path,
                )

            baseline_version = get_installed_vllm_version(venv_path)
            logger.info(f"  Baseline vLLM: {baseline_version}")

            logger.info(f"  [1/3] Running BASELINE benchmark...")
            if benchmark_type == "serving":
                model = self.extract_model_from_command(instance.perf_command)
                if not model:
                    return BenchmarkResult(
                        status="error",
                        commit_hash=instance.commit_hash,
                        benchmark_type=benchmark_type,
                        error_message="Could not extract model from perf_command for serving benchmark",
                        duration_s=time.time() - start_time,
                        perf_command=instance.perf_command,
                        patch_path=patch.patch_path,
                    )
                baseline_stdout, baseline_stderr, _ = self.run_serving_benchmark(
                    instance.perf_command, worktree_path, venv_path, model
                )
            else:
                baseline_stdout, baseline_stderr, _ = self.run_benchmark_command(
                    instance.perf_command, worktree_path, benchmark_type, venv_path=venv_path,
                )
            all_stdout.append(f"=== BASELINE ===\n{baseline_stdout}")
            all_stderr.append(f"=== BASELINE ===\n{baseline_stderr}")

            baseline_metrics = parse_benchmark_output(baseline_stdout + "\n" + baseline_stderr, benchmark_type)

            if not (baseline_metrics.token_throughput or baseline_metrics.request_throughput or
                    baseline_metrics.ttft_mean or baseline_metrics.latency_ms):
                return BenchmarkResult(
                    status="baseline_failed",
                    commit_hash=instance.commit_hash,
                    benchmark_type=benchmark_type,
                    baseline_metrics=baseline_metrics,
                    error_message="Baseline produced no metrics",
                    stdout="\n".join(all_stdout),
                    stderr="\n".join(all_stderr),
                    duration_s=time.time() - start_time,
                    perf_command=instance.perf_command,
                    patch_path=patch.patch_path,
                )

            logger.info(f"  Baseline: throughput={baseline_metrics.token_throughput or baseline_metrics.request_throughput}")

            # ========== 2. HUMAN BENCHMARK ==========
            logger.info(f"  [2/3] Installing HUMAN wheel...")
            if not install_vllm_wheel(venv_path, human_wheel, reinstall=True):
                return BenchmarkResult(
                    status="wheel_install_failed",
                    commit_hash=instance.commit_hash,
                    benchmark_type=benchmark_type,
                    baseline_metrics=baseline_metrics,
                    error_message="Failed to install human wheel",
                    stdout="\n".join(all_stdout),
                    stderr="\n".join(all_stderr),
                    duration_s=time.time() - start_time,
                    perf_command=instance.perf_command,
                    patch_path=patch.patch_path,
                )

            human_version = get_installed_vllm_version(venv_path)
            logger.info(f"  Human vLLM: {human_version}")

            logger.info(f"  [2/3] Running HUMAN benchmark...")
            if benchmark_type == "serving":
                human_stdout, human_stderr, _ = self.run_serving_benchmark(
                    instance.perf_command, worktree_path, venv_path, model
                )
            else:
                human_stdout, human_stderr, _ = self.run_benchmark_command(
                    instance.perf_command, worktree_path, benchmark_type, venv_path=venv_path,
                )
            all_stdout.append(f"=== HUMAN ===\n{human_stdout}")
            all_stderr.append(f"=== HUMAN ===\n{human_stderr}")

            human_metrics = parse_benchmark_output(human_stdout + "\n" + human_stderr, benchmark_type)

            if not (human_metrics.token_throughput or human_metrics.request_throughput or
                    human_metrics.ttft_mean or human_metrics.latency_ms):
                return BenchmarkResult(
                    status="human_failed",
                    commit_hash=instance.commit_hash,
                    benchmark_type=benchmark_type,
                    baseline_metrics=baseline_metrics,
                    human_metrics=human_metrics,
                    error_message="Human benchmark produced no metrics",
                    stdout="\n".join(all_stdout),
                    stderr="\n".join(all_stderr),
                    duration_s=time.time() - start_time,
                    perf_command=instance.perf_command,
                    patch_path=patch.patch_path,
                )

            human_improvement = compute_improvement(baseline_metrics, human_metrics)
            logger.info(f"  Human: throughput={human_metrics.token_throughput or human_metrics.request_throughput}, improvement={human_improvement}")

            # ========== 3. AGENT BENCHMARK ==========
            agent_metrics = None
            agent_improvement = None
            agent_vs_human = None

            if patch.patch_path:
                logger.info(f"  [3/3] Setting up AGENT benchmark (baseline + patch overlay)...")

                # Reinstall baseline wheel for agent test
                if not install_vllm_wheel(venv_path, baseline_wheel, reinstall=True):
                    logger.warning("  Failed to reinstall baseline for agent test")
                else:
                    # Create overlay with agent's patched files
                    overlay_path = self.create_agent_overlay(worktree_path, patch.patch_path)

                    if overlay_path:
                        logger.info(f"  [3/3] Running AGENT benchmark...")
                        # Run with PYTHONPATH overlay for patched agent code
                        if benchmark_type == "serving":
                            agent_stdout, agent_stderr, _ = self.run_serving_benchmark(
                                instance.perf_command, worktree_path, venv_path, model,
                                pythonpath_overlay=overlay_path,
                            )
                        else:
                            agent_stdout, agent_stderr, _ = self.run_benchmark_command(
                                instance.perf_command, worktree_path, benchmark_type,
                                venv_path=venv_path, pythonpath_overlay=overlay_path,
                            )
                        all_stdout.append(f"=== AGENT ===\n{agent_stdout}")
                        all_stderr.append(f"=== AGENT ===\n{agent_stderr}")

                        agent_metrics = parse_benchmark_output(agent_stdout + "\n" + agent_stderr, benchmark_type)

                        if agent_metrics.token_throughput or agent_metrics.request_throughput or agent_metrics.ttft_mean or agent_metrics.latency_ms:
                            agent_improvement = compute_improvement(baseline_metrics, agent_metrics)
                            agent_vs_human = compute_improvement(human_metrics, agent_metrics)
                            logger.info(f"  Agent: throughput={agent_metrics.token_throughput or agent_metrics.request_throughput}, vs_baseline={agent_improvement}, vs_human={agent_vs_human}")
                        else:
                            logger.warning("  Agent benchmark produced no metrics (patch may modify C extensions)")
                    else:
                        logger.warning("  Could not create agent overlay")
            else:
                logger.info("  [3/3] No agent patch available, skipping")

            return BenchmarkResult(
                status="success",
                commit_hash=instance.commit_hash,
                benchmark_type=benchmark_type,
                baseline_metrics=baseline_metrics,
                human_metrics=human_metrics,
                agent_metrics=agent_metrics,
                human_improvement=human_improvement,
                agent_improvement=agent_improvement,
                agent_vs_human=agent_vs_human,
                stdout="\n".join(all_stdout),
                stderr="\n".join(all_stderr),
                duration_s=time.time() - start_time,
                perf_command=instance.perf_command,
                patch_path=patch.patch_path,
                # Legacy compatibility
                patched_metrics=human_metrics,
                improvement=human_improvement,
            )

        except Exception as e:
            logger.exception(f"Error running benchmark for {commit_short}")
            return BenchmarkResult(
                status="error",
                commit_hash=instance.commit_hash,
                benchmark_type=benchmark_type,
                error_message=str(e),
                duration_s=time.time() - start_time,
                perf_command=instance.perf_command,
                patch_path=patch.patch_path if patch else None,
            )
        finally:
            self.cleanup_worktree(repo_path, worktree_path)

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

        for i, instance in enumerate(instances):
            logger.info(f"[{i+1}/{len(instances)}] {instance.commit_hash[:8]}")

            patch = patches.get(instance.commit_hash[:8]) or patches.get(instance.commit_hash)
            result = self.run_benchmark(instance, patch)
            self.results.append(result)

            self._save_result(instance, result)

        return self._generate_report(split)

    def _save_result(self, instance: DatasetInstance, result: BenchmarkResult):
        """Save individual result."""
        result_dir = self.output_dir / instance.repo / instance.commit_hash[:8]
        result_dir.mkdir(parents=True, exist_ok=True)

        result_path = result_dir / "benchmark_result.json"
        with open(result_path, "w") as f:
            data = {
                "instance": asdict(instance) if hasattr(instance, "__dataclass_fields__") else vars(instance),
                "result": {
                    "status": result.status,
                    "commit_hash": result.commit_hash,
                    "benchmark_type": result.benchmark_type,
                    "baseline_metrics": asdict(result.baseline_metrics) if result.baseline_metrics else None,
                    "patched_metrics": asdict(result.patched_metrics) if result.patched_metrics else None,
                    "improvement": result.improvement,
                    "error_message": result.error_message,
                    "duration_s": result.duration_s,
                    "perf_command": result.perf_command,
                    "patch_path": result.patch_path,
                },
            }
            json.dump(data, f, indent=2, default=str)

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
            "summary": {},
            "metrics": {
                "throughput": {"improved": 0, "regressed": 0, "avg_change": []},
                "ttft": {"improved": 0, "regressed": 0, "avg_change": []},
                "latency": {"improved": 0, "regressed": 0, "avg_change": []},
            },
            "results": [],
        }

        for result in self.results:
            report["summary"][result.status] = report["summary"].get(result.status, 0) + 1

            entry = {
                "commit_hash": result.commit_hash,
                "status": result.status,
                "benchmark_type": result.benchmark_type,
                "improvement": result.improvement,
                "error": result.error_message,
            }
            report["results"].append(entry)

            if result.improvement:
                for key, val in result.improvement.items():
                    if "throughput" in key:
                        if val > 0:
                            report["metrics"]["throughput"]["improved"] += 1
                        else:
                            report["metrics"]["throughput"]["regressed"] += 1
                        report["metrics"]["throughput"]["avg_change"].append(val)
                    elif "ttft" in key:
                        if val > 0:
                            report["metrics"]["ttft"]["improved"] += 1
                        else:
                            report["metrics"]["ttft"]["regressed"] += 1
                        report["metrics"]["ttft"]["avg_change"].append(val)
                    elif "latency" in key or "tpot" in key or "itl" in key:
                        if val > 0:
                            report["metrics"]["latency"]["improved"] += 1
                        else:
                            report["metrics"]["latency"]["regressed"] += 1
                        report["metrics"]["latency"]["avg_change"].append(val)

        # Compute averages
        for metric in ["throughput", "ttft", "latency"]:
            changes = report["metrics"][metric]["avg_change"]
            if changes:
                report["metrics"][metric]["avg_pct"] = sum(changes) / len(changes)
            report["metrics"][metric].pop("avg_change")

        report_path = self.output_dir / f"native_benchmark_report_{split}.json"
        with open(report_path, "w") as f:
            json.dump(report, f, indent=2)

        logger.info(f"Report saved to {report_path}")
        return report


def print_summary(report: Dict[str, Any]):
    """Print human-readable summary."""
    print("\n" + "=" * 70)
    print(f"NATIVE BENCHMARK REPORT - {report['split'].upper()}")
    print("=" * 70)
    print(f"\nDataset: {report['dataset']}")
    print(f"Generated: {report['generated_at']}")
    print(f"\nTotal instances: {report['total_instances']}")

    print("\nStatus breakdown:")
    for status, count in report["summary"].items():
        if count > 0:
            print(f"  {status:20s}: {count}")

    print("\nMetrics:")
    for metric, data in report["metrics"].items():
        if data.get("improved", 0) + data.get("regressed", 0) > 0:
            print(f"\n  {metric.upper()}:")
            print(f"    Improved: {data['improved']}")
            print(f"    Regressed: {data['regressed']}")
            if "avg_pct" in data:
                print(f"    Avg change: {data['avg_pct']:.2f}%")


def main():
    parser = argparse.ArgumentParser(description="Run native vLLM/SGLang benchmarks")
    parser.add_argument("--vllm-repo", type=Path, required=True)
    parser.add_argument("--sglang-repo", type=Path, required=True)
    parser.add_argument("--state-root", type=Path, default=Path("./perf-agents-bench/state"))
    parser.add_argument("--output-dir", type=Path, default=Path("./omniperf_results"))
    parser.add_argument("--split", choices=["vllm", "sglang"], default="vllm")
    parser.add_argument("--timeout", type=int, default=DEFAULT_TIMEOUT)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--commits", nargs="+", default=None)
    parser.add_argument("--verbose", "-v", action="store_true")
    parser.add_argument("--use-modal", action="store_true",
                       help="Use Modal cloud GPUs for large models requiring multi-GPU")

    args = parser.parse_args()

    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )

    runner = NativeBenchmarkRunner(
        vllm_repo_path=args.vllm_repo,
        sglang_repo_path=args.sglang_repo,
        state_root=args.state_root,
        output_dir=args.output_dir,
        timeout=args.timeout,
        use_modal=args.use_modal,
    )

    report = runner.run_all(
        split=args.split,
        limit=args.limit,
        commit_filter=args.commits,
    )

    print_summary(report)


if __name__ == "__main__":
    main()
