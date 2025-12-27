"""
OmniPerf Native Benchmark Runner - Real model benchmarks with TTFT, throughput, latency.

Runs native vLLM/SGLang benchmarks using perf_command from Ayushnangia/omniperf_v1.
Uses actual model weights downloaded to /ephemeral storage.

Metrics collected:
- TTFT (Time To First Token): mean, P50, P90, P99
- Throughput: requests/s, output tokens/s
- Latency: E2E mean, P50, P90, P99
- ITL (Inter-Token Latency)
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import shutil
import signal
import socket
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# Timeouts
SERVER_STARTUP_TIMEOUT = 600  # 10 min for model loading
BENCHMARK_TIMEOUT = 900  # 15 min per benchmark
WARMUP_REQUESTS = 3

# Storage
EPHEMERAL_ROOT = Path("/ephemeral")
MODEL_CACHE = EPHEMERAL_ROOT / "hf_models"
BENCHMARK_CACHE = EPHEMERAL_ROOT / "benchmark_cache"

HF_DATASET_ID = "Ayushnangia/omniperf_v1"

# Model mappings - some models in dataset may need substitution
MODEL_SUBSTITUTIONS = {
    # Large models -> smaller alternatives for H100 80GB
    "deepseek-ai/DeepSeek-V3": "deepseek-ai/deepseek-llm-7b-chat",
    "deepseek-ai/DeepSeek-V3-Base": "deepseek-ai/deepseek-llm-7b-base",
    # Gated models without access -> public alternatives
    "meta-llama/Llama-2-7b-hf": "meta-llama/Llama-2-7b-hf",  # requires HF login
    "meta-llama/Meta-Llama-3.1-8B-Instruct": "meta-llama/Llama-3.1-8B-Instruct",
}

# Models known to work on H100 80GB
FEASIBLE_MODELS = {
    "meta-llama/Llama-3.1-8B-Instruct",
    "meta-llama/Llama-3.1-8B",
    "meta-llama/Meta-Llama-3-8B-Instruct",
    "meta-llama/Llama-2-7b-hf",
    "meta-llama/Llama-2-7b-chat-hf",
    "facebook/opt-125m",
    "facebook/opt-1.3b",
    "facebook/opt-6.7b",
    "google/gemma-2b",
    "google/gemma-7b",
    "mistralai/Mistral-7B-v0.1",
    "mistralai/Mistral-7B-Instruct-v0.1",
    "Qwen/Qwen2-7B",
    "Qwen/Qwen2-7B-Instruct",
    "deepseek-ai/deepseek-llm-7b-chat",
    "deepseek-ai/deepseek-coder-6.7b-instruct",
    "tiiuae/falcon-7b",
    "tiiuae/falcon-7b-instruct",
    "bigscience/bloom-7b1",
    "mosaicml/mpt-7b",
    "EleutherAI/gpt-j-6B",
}


@dataclass
class NativeMetrics:
    """Comprehensive benchmark metrics."""
    completed: int = 0
    failed: int = 0
    total_input_tokens: int = 0
    total_output_tokens: int = 0

    # Throughput
    request_throughput: Optional[float] = None
    output_throughput: Optional[float] = None
    total_throughput: Optional[float] = None

    # TTFT (ms)
    ttft_mean: Optional[float] = None
    ttft_median: Optional[float] = None
    ttft_p50: Optional[float] = None
    ttft_p90: Optional[float] = None
    ttft_p99: Optional[float] = None

    # TPOT (ms)
    tpot_mean: Optional[float] = None
    tpot_p50: Optional[float] = None
    tpot_p90: Optional[float] = None
    tpot_p99: Optional[float] = None

    # ITL (ms)
    itl_mean: Optional[float] = None
    itl_p50: Optional[float] = None
    itl_p90: Optional[float] = None
    itl_p99: Optional[float] = None

    # E2E Latency (ms)
    e2e_mean: Optional[float] = None
    e2e_median: Optional[float] = None
    e2e_p50: Optional[float] = None
    e2e_p90: Optional[float] = None
    e2e_p99: Optional[float] = None

    duration_s: Optional[float] = None

    @classmethod
    def from_vllm(cls, data: Dict[str, Any]) -> "NativeMetrics":
        """Parse vLLM benchmark output."""
        m = cls()
        m.completed = data.get("completed", 0)
        m.failed = data.get("failed", 0)
        m.total_input_tokens = data.get("total_input", 0)
        m.total_output_tokens = data.get("total_output", 0)

        m.request_throughput = data.get("request_throughput")
        m.output_throughput = data.get("output_throughput")
        m.total_throughput = data.get("total_token_throughput")

        m.ttft_mean = data.get("mean_ttft_ms")
        m.ttft_median = data.get("median_ttft_ms")
        m.ttft_p50 = data.get("p50_ttft_ms")
        m.ttft_p90 = data.get("p90_ttft_ms")
        m.ttft_p99 = data.get("p99_ttft_ms")

        m.tpot_mean = data.get("mean_tpot_ms")
        m.tpot_p50 = data.get("p50_tpot_ms")
        m.tpot_p90 = data.get("p90_tpot_ms")
        m.tpot_p99 = data.get("p99_tpot_ms")

        m.itl_mean = data.get("mean_itl_ms")
        m.itl_p50 = data.get("p50_itl_ms")
        m.itl_p90 = data.get("p90_itl_ms")
        m.itl_p99 = data.get("p99_itl_ms")

        m.e2e_mean = data.get("mean_e2el_ms")
        m.e2e_median = data.get("median_e2el_ms")
        m.e2e_p50 = data.get("p50_e2el_ms")
        m.e2e_p90 = data.get("p90_e2el_ms")
        m.e2e_p99 = data.get("p99_e2el_ms")

        m.duration_s = data.get("duration")
        return m

    @classmethod
    def from_sglang(cls, data: Dict[str, Any]) -> "NativeMetrics":
        """Parse SGLang benchmark output."""
        m = cls()
        m.completed = data.get("completed", 0)
        m.total_input_tokens = data.get("total_input_tokens", 0)
        m.total_output_tokens = data.get("total_output_tokens", 0)

        m.request_throughput = data.get("request_throughput")
        m.output_throughput = data.get("output_throughput")

        m.ttft_mean = data.get("mean_ttft_ms")
        m.ttft_median = data.get("median_ttft_ms")
        m.ttft_p50 = data.get("p50_ttft_ms")
        m.ttft_p90 = data.get("p90_ttft_ms")
        m.ttft_p99 = data.get("p99_ttft_ms")

        m.tpot_mean = data.get("mean_tpot_ms")
        m.tpot_p99 = data.get("p99_tpot_ms")

        m.itl_mean = data.get("mean_itl_ms")
        m.itl_p99 = data.get("p99_itl_ms")

        m.e2e_mean = data.get("mean_e2e_latency_ms")
        m.e2e_median = data.get("median_e2e_latency_ms")
        m.e2e_p99 = data.get("p99_e2e_latency_ms")

        m.duration_s = data.get("duration")
        return m


@dataclass
class BenchmarkResult:
    """Native benchmark result."""
    status: str
    commit_hash: str
    benchmark_type: str = ""
    model: str = ""

    baseline: Optional[NativeMetrics] = None
    patched: Optional[NativeMetrics] = None

    # Speedups (>1 = improvement)
    speedup_throughput: Optional[float] = None
    speedup_ttft: Optional[float] = None
    speedup_e2e_p99: Optional[float] = None

    error: Optional[str] = None
    perf_command: Optional[str] = None
    duration_s: float = 0.0

    def compute_speedups(self):
        """Calculate improvement ratios."""
        if not self.baseline or not self.patched:
            return

        # Throughput: higher is better (patched/baseline)
        if self.baseline.request_throughput and self.patched.request_throughput:
            if self.baseline.request_throughput > 0:
                self.speedup_throughput = self.patched.request_throughput / self.baseline.request_throughput

        # TTFT: lower is better (baseline/patched)
        if self.baseline.ttft_mean and self.patched.ttft_mean:
            if self.patched.ttft_mean > 0:
                self.speedup_ttft = self.baseline.ttft_mean / self.patched.ttft_mean

        # E2E P99: lower is better (baseline/patched)
        if self.baseline.e2e_p99 and self.patched.e2e_p99:
            if self.patched.e2e_p99 > 0:
                self.speedup_e2e_p99 = self.baseline.e2e_p99 / self.patched.e2e_p99


@dataclass
class DatasetInstance:
    commit_hash: str
    commit_subject: str
    repo: str
    test_script: Optional[str] = None
    perf_command: Optional[str] = None
    files_changed: List[str] = field(default_factory=list)

    @classmethod
    def from_row(cls, row: Dict[str, Any]) -> "DatasetInstance":
        return cls(
            commit_hash=row.get("commit_hash", ""),
            commit_subject=row.get("commit_subject", ""),
            repo=row.get("repo", "unknown"),
            test_script=row.get("test_script"),
            perf_command=row.get("perf_command"),
            files_changed=row.get("files_changed", []),
        )


@dataclass
class Patch:
    path: str
    human_commit: str
    pre_commit: str
    item_id: str


def find_free_port() -> int:
    """Get available port."""
    with socket.socket() as s:
        s.bind(('', 0))
        return s.getsockname()[1]


def is_port_open(host: str, port: int) -> bool:
    """Check if port is accepting connections."""
    try:
        with socket.create_connection((host, port), timeout=2):
            return True
    except:
        return False


def wait_for_server(host: str, port: int, timeout: int = SERVER_STARTUP_TIMEOUT) -> bool:
    """Wait for server to be ready."""
    start = time.time()
    while time.time() - start < timeout:
        if is_port_open(host, port):
            # Extra wait for full initialization
            time.sleep(5)
            return True
        time.sleep(3)
    return False


def kill_proc_tree(pid: int):
    """Kill process and children."""
    try:
        import psutil
        proc = psutil.Process(pid)
        for child in proc.children(recursive=True):
            try:
                child.kill()
            except:
                pass
        proc.kill()
    except:
        try:
            os.kill(pid, signal.SIGKILL)
        except:
            pass


def get_env(worktree: Path, repo_type: str) -> Dict[str, str]:
    """Get environment with proper paths."""
    env = os.environ.copy()

    paths = [str(worktree)]
    if repo_type == "sglang":
        py_sub = worktree / "python"
        if py_sub.exists():
            paths.insert(0, str(py_sub))

    existing = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = ":".join(paths) + (":" + existing if existing else "")

    # Model cache
    MODEL_CACHE.mkdir(parents=True, exist_ok=True)
    BENCHMARK_CACHE.mkdir(parents=True, exist_ok=True)

    env.update({
        "HF_HOME": str(MODEL_CACHE),
        "TRANSFORMERS_CACHE": str(MODEL_CACHE / "transformers"),
        "HF_DATASETS_CACHE": str(MODEL_CACHE / "datasets"),
        "TORCH_HOME": str(BENCHMARK_CACHE / "torch"),
        "TRITON_CACHE_DIR": str(BENCHMARK_CACHE / "triton"),
    })

    return env


def extract_model_from_command(cmd: str) -> Optional[str]:
    """Extract model name from perf_command."""
    patterns = [
        r'--model[=\s]+["\']?([^\s"\']+)',
        r'--model-path[=\s]+["\']?([^\s"\']+)',
        r'-m[=\s]+["\']?([^\s"\']+)',
    ]
    for p in patterns:
        m = re.search(p, cmd)
        if m:
            return m.group(1).strip('"\'')
    return None


def get_benchmark_type(cmd: str) -> str:
    """Determine benchmark type from command."""
    if "benchmark_serving" in cmd or "bench_serving" in cmd:
        return "serving"
    elif "benchmark_latency" in cmd or "bench_latency" in cmd:
        return "latency"
    elif "benchmark_throughput" in cmd:
        return "throughput"
    elif "bench_one_batch" in cmd:
        return "one_batch"
    return "unknown"


def is_model_feasible(model: str) -> bool:
    """Check if model can run on H100 80GB."""
    if not model:
        return False

    # Check substitutions
    model = MODEL_SUBSTITUTIONS.get(model, model)

    # Check feasibility list
    if model in FEASIBLE_MODELS:
        return True

    # Heuristics for model size
    model_lower = model.lower()
    if any(x in model_lower for x in ["70b", "65b", "40b", "33b", "30b", "180b", "v3"]):
        return False

    # Small models usually work
    if any(x in model_lower for x in ["125m", "350m", "1b", "2b", "3b", "6b", "7b", "8b"]):
        return True

    return False


def substitute_model(model: str) -> str:
    """Get model substitution if needed."""
    return MODEL_SUBSTITUTIONS.get(model, model)


class InferenceServer:
    """Manages vLLM/SGLang inference server."""

    def __init__(self, repo_type: str, worktree: Path):
        self.repo_type = repo_type
        self.worktree = worktree
        self.process = None
        self.port = None
        self.host = "127.0.0.1"

    def start(self, model: str, extra_args: List[str] = None) -> bool:
        """Start server with model."""
        self.port = find_free_port()
        extra_args = extra_args or []

        env = get_env(self.worktree, self.repo_type)

        if self.repo_type == "vllm":
            cmd = [
                sys.executable, "-m", "vllm.entrypoints.openai.api_server",
                "--model", model,
                "--host", self.host,
                "--port", str(self.port),
                "--trust-remote-code",
                *extra_args
            ]
        else:
            cmd = [
                sys.executable, "-m", "sglang.launch_server",
                "--model-path", model,
                "--host", self.host,
                "--port", str(self.port),
                "--trust-remote-code",
                *extra_args
            ]

        logger.info(f"Starting server: {model} on port {self.port}")

        try:
            self.process = subprocess.Popen(
                cmd,
                cwd=self.worktree,
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )

            if wait_for_server(self.host, self.port):
                logger.info("Server ready")
                return True

            logger.error("Server startup timeout")
            self.stop()
            return False

        except Exception as e:
            logger.error(f"Server start failed: {e}")
            return False

    def stop(self):
        """Stop server."""
        if self.process:
            logger.info("Stopping server")
            kill_proc_tree(self.process.pid)
            self.process = None
            self.port = None

    @property
    def endpoint(self) -> str:
        return f"http://{self.host}:{self.port}"


def run_benchmark(
    cmd: str,
    worktree: Path,
    repo_type: str,
    server_endpoint: Optional[str] = None,
    timeout: int = BENCHMARK_TIMEOUT,
) -> Tuple[Optional[Dict], str, str]:
    """Execute benchmark command and parse output."""
    env = get_env(worktree, repo_type)

    # Update endpoint in command if server provided
    if server_endpoint:
        host, port = server_endpoint.replace("http://", "").split(":")
        cmd = re.sub(r'--host[=\s]+[^\s]+', f'--host {host}', cmd)
        cmd = re.sub(r'--port[=\s]+\d+', f'--port {port}', cmd)
        cmd = re.sub(r'--base-url[=\s]+[^\s]+', f'--base-url {server_endpoint}', cmd)

    # Add result saving for vLLM
    result_file = worktree / "bench_result.json"
    if "benchmark_serving" in cmd and "--save-result" not in cmd:
        cmd += f" --save-result --result-filename {result_file}"

    logger.debug(f"Running: {cmd[:120]}...")

    try:
        result = subprocess.run(
            cmd, shell=True, cwd=worktree, env=env,
            capture_output=True, text=True, timeout=timeout
        )

        stdout, stderr = result.stdout, result.stderr

        # Parse output
        parsed = None

        # Check result file
        if result_file.exists():
            try:
                parsed = json.loads(result_file.read_text())
                result_file.unlink()
            except:
                pass

        # Parse from stdout
        if not parsed:
            for line in stdout.split("\n"):
                line = line.strip()
                if line.startswith("{"):
                    try:
                        parsed = json.loads(line)
                        break
                    except:
                        continue

        return parsed, stdout, stderr

    except subprocess.TimeoutExpired:
        return None, "", f"Timeout after {timeout}s"
    except Exception as e:
        return None, "", str(e)


def load_dataset(split: str) -> List[DatasetInstance]:
    """Load HF dataset."""
    from datasets import load_dataset as hf_load

    logger.info(f"Loading {HF_DATASET_ID} split={split}")
    ds = hf_load(HF_DATASET_ID, split=split)

    instances = [DatasetInstance.from_row(r) for r in ds]
    logger.info(f"Loaded {len(instances)} instances")
    return instances


def discover_patches(state_root: Path, repo_filter: str) -> Dict[str, Patch]:
    """Find Claude Code patches."""
    patches = {}
    runs = state_root / "runs"

    if not runs.exists():
        return patches

    for repo_dir in runs.iterdir():
        if not repo_dir.is_dir():
            continue
        if repo_filter and repo_dir.name.lower() != repo_filter.lower():
            continue

        cc_dir = repo_dir / "claude_code"
        if not cc_dir.exists():
            continue

        for model in cc_dir.iterdir():
            if not model.is_dir():
                continue
            for ts in model.iterdir():
                if not ts.is_dir():
                    continue
                for item in ts.iterdir():
                    if not item.is_dir():
                        continue

                    journal = item / "journal.json"
                    patch_file = item / "model_patch.diff"

                    if not journal.exists():
                        continue

                    try:
                        j = json.loads(journal.read_text())
                        commits = j.get("commits", {})
                        human = commits.get("human")
                        pre = commits.get("pre")

                        if not human or not pre:
                            continue

                        has_patch = patch_file.exists() and patch_file.stat().st_size > 0

                        p = Patch(
                            path=str(patch_file) if has_patch else "",
                            human_commit=human,
                            pre_commit=pre,
                            item_id=item.name,
                        )

                        patches[human[:8]] = p
                        patches[human] = p

                    except Exception as e:
                        logger.warning(f"Parse error {journal}: {e}")

    logger.info(f"Found {len(patches)//2} patches")
    return patches


def create_worktree(repo: Path, worktree: Path, commit: str) -> bool:
    """Create git worktree."""
    try:
        if worktree.exists():
            subprocess.run(
                ["git", "worktree", "remove", "--force", str(worktree)],
                cwd=repo, capture_output=True
            )

        subprocess.run(
            ["git", "worktree", "add", "--detach", str(worktree), commit],
            cwd=repo, check=True, capture_output=True
        )
        return True
    except Exception as e:
        logger.error(f"Worktree failed: {e}")
        return False


def apply_patch(worktree: Path, patch_path: str) -> bool:
    """Apply git patch."""
    try:
        r = subprocess.run(
            ["git", "apply", "--check", patch_path],
            cwd=worktree, capture_output=True
        )
        if r.returncode != 0:
            r = subprocess.run(
                ["git", "apply", "--check", "--ignore-whitespace", patch_path],
                cwd=worktree, capture_output=True
            )
            if r.returncode != 0:
                return False

        subprocess.run(
            ["git", "apply", patch_path],
            cwd=worktree, check=True, capture_output=True
        )
        return True
    except:
        return False


def cleanup_worktree(repo: Path, worktree: Path):
    """Remove worktree."""
    try:
        subprocess.run(
            ["git", "worktree", "remove", "--force", str(worktree)],
            cwd=repo, capture_output=True
        )
        subprocess.run(["git", "worktree", "prune"], cwd=repo, capture_output=True)
    except:
        pass


class NativeBenchmarkRunner:
    """Runner for native benchmarks with real models."""

    def __init__(
        self,
        vllm_repo: Path,
        sglang_repo: Path,
        state_root: Path,
        output_dir: Path,
    ):
        self.vllm_repo = Path(vllm_repo).resolve()
        self.sglang_repo = Path(sglang_repo).resolve()
        self.state_root = Path(state_root).resolve()
        self.output_dir = Path(output_dir).resolve()
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.results: List[BenchmarkResult] = []

    def get_repo(self, name: str) -> Optional[Path]:
        if name.lower() == "vllm":
            return self.vllm_repo
        elif name.lower() == "sglang":
            return self.sglang_repo
        return None

    def run_instance(
        self,
        inst: DatasetInstance,
        patch: Optional[Patch],
    ) -> BenchmarkResult:
        """Run benchmark for one instance."""
        start = time.time()
        commit_short = inst.commit_hash[:8]

        logger.info(f"Benchmarking {commit_short}: {inst.commit_subject[:50]}...")

        # Validate
        if not inst.perf_command:
            return BenchmarkResult(
                status="skip", commit_hash=inst.commit_hash,
                error="No perf_command", duration_s=time.time()-start
            )

        # Skip local paths
        if any(x in inst.perf_command for x in ["/data/users/", "/dev/shm/", "/home/"]):
            return BenchmarkResult(
                status="skip", commit_hash=inst.commit_hash,
                error="Local path in command", perf_command=inst.perf_command,
                duration_s=time.time()-start
            )

        if not patch or not patch.path:
            return BenchmarkResult(
                status="no_patch", commit_hash=inst.commit_hash,
                error="No patch found", perf_command=inst.perf_command,
                duration_s=time.time()-start
            )

        # Extract model and check feasibility
        model = extract_model_from_command(inst.perf_command)
        bench_type = get_benchmark_type(inst.perf_command)

        if model and not is_model_feasible(model):
            return BenchmarkResult(
                status="skip", commit_hash=inst.commit_hash,
                benchmark_type=bench_type, model=model,
                error=f"Model too large: {model}",
                perf_command=inst.perf_command,
                duration_s=time.time()-start
            )

        # Apply substitution
        if model:
            actual_model = substitute_model(model)
            if actual_model != model:
                logger.info(f"  Substituting model: {model} -> {actual_model}")
                inst.perf_command = inst.perf_command.replace(model, actual_model)
                model = actual_model

        repo = self.get_repo(inst.repo)
        if not repo or not repo.exists():
            return BenchmarkResult(
                status="error", commit_hash=inst.commit_hash,
                error=f"Repo not found: {inst.repo}",
                duration_s=time.time()-start
            )

        # Create worktree
        with tempfile.TemporaryDirectory(prefix="native_") as tmp:
            worktree = Path(tmp) / "wt"

            try:
                if not create_worktree(repo, worktree, patch.pre_commit):
                    return BenchmarkResult(
                        status="error", commit_hash=inst.commit_hash,
                        error=f"Worktree failed at {patch.pre_commit[:8]}",
                        duration_s=time.time()-start
                    )

                server = None
                endpoint = None

                # Start server for serving benchmarks
                if bench_type == "serving" and model:
                    server = InferenceServer(inst.repo, worktree)
                    if not server.start(model):
                        return BenchmarkResult(
                            status="baseline_failed", commit_hash=inst.commit_hash,
                            benchmark_type=bench_type, model=model,
                            error="Server startup failed",
                            perf_command=inst.perf_command,
                            duration_s=time.time()-start
                        )
                    endpoint = server.endpoint

                try:
                    # Run baseline
                    logger.info("  Running baseline...")
                    baseline_data, stdout, stderr = run_benchmark(
                        inst.perf_command, worktree, inst.repo, endpoint
                    )

                    if not baseline_data:
                        return BenchmarkResult(
                            status="baseline_failed", commit_hash=inst.commit_hash,
                            benchmark_type=bench_type, model=model or "",
                            error=f"Baseline failed: {stderr[:300]}",
                            perf_command=inst.perf_command,
                            duration_s=time.time()-start
                        )

                    # Parse baseline metrics
                    if inst.repo == "vllm":
                        baseline = NativeMetrics.from_vllm(baseline_data)
                    else:
                        baseline = NativeMetrics.from_sglang(baseline_data)

                    logger.info(f"  Baseline: throughput={baseline.request_throughput}, ttft={baseline.ttft_mean}ms")

                    # Stop server before patching
                    if server:
                        server.stop()
                        server = None

                    # Apply patch
                    logger.info("  Applying patch...")
                    if not apply_patch(worktree, patch.path):
                        return BenchmarkResult(
                            status="patch_failed", commit_hash=inst.commit_hash,
                            benchmark_type=bench_type, model=model or "",
                            baseline=baseline, error="Patch failed",
                            perf_command=inst.perf_command,
                            duration_s=time.time()-start
                        )

                    # Restart server for patched run
                    if bench_type == "serving" and model:
                        server = InferenceServer(inst.repo, worktree)
                        if not server.start(model):
                            return BenchmarkResult(
                                status="patched_failed", commit_hash=inst.commit_hash,
                                benchmark_type=bench_type, model=model,
                                baseline=baseline, error="Patched server failed",
                                perf_command=inst.perf_command,
                                duration_s=time.time()-start
                            )
                        endpoint = server.endpoint

                    # Run patched
                    logger.info("  Running patched...")
                    patched_data, stdout, stderr = run_benchmark(
                        inst.perf_command, worktree, inst.repo, endpoint
                    )

                    if not patched_data:
                        return BenchmarkResult(
                            status="patched_failed", commit_hash=inst.commit_hash,
                            benchmark_type=bench_type, model=model or "",
                            baseline=baseline, error=f"Patched failed: {stderr[:300]}",
                            perf_command=inst.perf_command,
                            duration_s=time.time()-start
                        )

                    # Parse patched metrics
                    if inst.repo == "vllm":
                        patched = NativeMetrics.from_vllm(patched_data)
                    else:
                        patched = NativeMetrics.from_sglang(patched_data)

                    logger.info(f"  Patched: throughput={patched.request_throughput}, ttft={patched.ttft_mean}ms")

                    # Build result
                    result = BenchmarkResult(
                        status="success", commit_hash=inst.commit_hash,
                        benchmark_type=bench_type, model=model or "",
                        baseline=baseline, patched=patched,
                        perf_command=inst.perf_command,
                        duration_s=time.time()-start
                    )
                    result.compute_speedups()

                    if result.speedup_throughput:
                        logger.info(f"  Throughput speedup: {result.speedup_throughput:.3f}x")
                    if result.speedup_ttft:
                        logger.info(f"  TTFT speedup: {result.speedup_ttft:.3f}x")

                    return result

                finally:
                    if server:
                        server.stop()

            except Exception as e:
                logger.exception(f"Error: {e}")
                return BenchmarkResult(
                    status="error", commit_hash=inst.commit_hash,
                    error=str(e), perf_command=inst.perf_command,
                    duration_s=time.time()-start
                )
            finally:
                cleanup_worktree(repo, worktree)

    def run_all(
        self,
        split: str = "vllm",
        limit: Optional[int] = None,
        benchmark_types: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """Run all benchmarks."""
        instances = load_dataset(split)
        patches = discover_patches(self.state_root, split)

        # Filter
        filtered = []
        for inst in instances:
            if not inst.perf_command:
                continue

            bt = get_benchmark_type(inst.perf_command)
            if benchmark_types and bt not in benchmark_types:
                continue

            filtered.append(inst)

        if limit:
            filtered = filtered[:limit]

        logger.info(f"Processing {len(filtered)} instances")

        for i, inst in enumerate(filtered):
            logger.info(f"[{i+1}/{len(filtered)}] {inst.commit_hash[:8]}")

            patch = patches.get(inst.commit_hash[:8]) or patches.get(inst.commit_hash)
            result = self.run_instance(inst, patch)
            self.results.append(result)

            # Save individual result
            self._save_result(inst, result)

        return self._generate_report(split)

    def _save_result(self, inst: DatasetInstance, result: BenchmarkResult):
        """Save individual result."""
        rdir = self.output_dir / inst.repo / inst.commit_hash[:8]
        rdir.mkdir(parents=True, exist_ok=True)

        data = {
            "commit_hash": inst.commit_hash,
            "commit_subject": inst.commit_subject,
            "status": result.status,
            "benchmark_type": result.benchmark_type,
            "model": result.model,
            "speedup_throughput": result.speedup_throughput,
            "speedup_ttft": result.speedup_ttft,
            "speedup_e2e_p99": result.speedup_e2e_p99,
            "error": result.error,
            "duration_s": result.duration_s,
        }

        if result.baseline:
            data["baseline"] = asdict(result.baseline)
        if result.patched:
            data["patched"] = asdict(result.patched)

        (rdir / "native_result.json").write_text(json.dumps(data, indent=2))

    def _generate_report(self, split: str) -> Dict[str, Any]:
        """Generate summary."""
        report = {
            "generated_at": datetime.now().isoformat(),
            "dataset": HF_DATASET_ID,
            "split": split,
            "type": "native",
            "total": len(self.results),
            "summary": {},
            "performance": {
                "throughput_improved": 0,
                "throughput_regressed": 0,
                "ttft_improved": 0,
                "ttft_regressed": 0,
                "avg_throughput_speedup": None,
                "avg_ttft_speedup": None,
                "throughput_speedups": [],
                "ttft_speedups": [],
            },
            "results": [],
        }

        throughput_speedups = []
        ttft_speedups = []

        for r in self.results:
            report["summary"][r.status] = report["summary"].get(r.status, 0) + 1

            entry = {
                "commit": r.commit_hash,
                "status": r.status,
                "type": r.benchmark_type,
                "model": r.model,
                "speedup_throughput": r.speedup_throughput,
                "speedup_ttft": r.speedup_ttft,
                "speedup_e2e_p99": r.speedup_e2e_p99,
                "error": r.error,
            }

            if r.baseline:
                entry["baseline_throughput"] = r.baseline.request_throughput
                entry["baseline_ttft"] = r.baseline.ttft_mean
            if r.patched:
                entry["patched_throughput"] = r.patched.request_throughput
                entry["patched_ttft"] = r.patched.ttft_mean

            report["results"].append(entry)

            if r.status == "success":
                if r.speedup_throughput:
                    throughput_speedups.append(r.speedup_throughput)
                    if r.speedup_throughput > 1.05:
                        report["performance"]["throughput_improved"] += 1
                    elif r.speedup_throughput < 0.95:
                        report["performance"]["throughput_regressed"] += 1

                if r.speedup_ttft:
                    ttft_speedups.append(r.speedup_ttft)
                    if r.speedup_ttft > 1.05:
                        report["performance"]["ttft_improved"] += 1
                    elif r.speedup_ttft < 0.95:
                        report["performance"]["ttft_regressed"] += 1

        if throughput_speedups:
            report["performance"]["avg_throughput_speedup"] = sum(throughput_speedups) / len(throughput_speedups)
            report["performance"]["throughput_speedups"] = sorted(throughput_speedups)
        if ttft_speedups:
            report["performance"]["avg_ttft_speedup"] = sum(ttft_speedups) / len(ttft_speedups)
            report["performance"]["ttft_speedups"] = sorted(ttft_speedups)

        # Save
        path = self.output_dir / f"native_report_{split}.json"
        path.write_text(json.dumps(report, indent=2))
        logger.info(f"Report: {path}")

        return report


def print_report(r: Dict[str, Any]):
    """Print summary."""
    print("\n" + "="*70)
    print(f"NATIVE BENCHMARK REPORT - {r['split'].upper()}")
    print("="*70)
    print(f"Generated: {r['generated_at']}")
    print(f"Total: {r['total']}")

    print("\nStatus:")
    for s, c in r["summary"].items():
        print(f"  {s:20s}: {c}")

    p = r["performance"]
    print("\nPerformance:")
    print(f"  Throughput improved:  {p['throughput_improved']}")
    print(f"  Throughput regressed: {p['throughput_regressed']}")
    print(f"  TTFT improved:        {p['ttft_improved']}")
    print(f"  TTFT regressed:       {p['ttft_regressed']}")

    if p["avg_throughput_speedup"]:
        print(f"\n  Avg throughput speedup: {p['avg_throughput_speedup']:.3f}x")
    if p["avg_ttft_speedup"]:
        print(f"  Avg TTFT speedup:       {p['avg_ttft_speedup']:.3f}x")


def main():
    parser = argparse.ArgumentParser(description="Native OmniPerf Benchmarks")
    parser.add_argument("--vllm-repo", type=Path, required=True)
    parser.add_argument("--sglang-repo", type=Path, required=True)
    parser.add_argument("--state-root", type=Path, default=Path("./perf-agents-bench/state"))
    parser.add_argument("--output-dir", type=Path, default=Path("/ephemeral/native_results"))
    parser.add_argument("--split", choices=["vllm", "sglang"], default="vllm")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--benchmark-types", nargs="+", choices=["serving", "latency", "throughput", "one_batch"])
    parser.add_argument("-v", "--verbose", action="store_true")

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s"
    )

    runner = NativeBenchmarkRunner(
        vllm_repo=args.vllm_repo,
        sglang_repo=args.sglang_repo,
        state_root=args.state_root,
        output_dir=args.output_dir,
    )

    report = runner.run_all(
        split=args.split,
        limit=args.limit,
        benchmark_types=args.benchmark_types,
    )

    print_report(report)


if __name__ == "__main__":
    main()
