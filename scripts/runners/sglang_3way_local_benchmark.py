#!/usr/bin/env python3
"""
SGLang True 3-Way Local Docker Benchmark Runner (vLLM-Compatible Output)

Runs actual 3-way benchmarks (baseline vs human vs agent) for SGLang commits
using SEPARATE Docker images for baseline and human commits.

This runs locally using docker and nvidia-container-toolkit.
No cloud infrastructure (Modal) required.

Docker Repository: shikhar481/sglang-images
- Baseline: shikhar481/sglang-images:{parent_commit_hash}
- Human: shikhar481/sglang-images:{human_commit_hash}
- Agent: Baseline image + agent patch overlay

Output Format: vLLM-compatible schema (76 columns) for HuggingFace upload.

Requirements:
- Docker with NVIDIA GPU support (nvidia-container-toolkit)
- NVIDIA GPU(s)
- HuggingFace token for model access (HF_TOKEN env var)

Usage:
    # Dry run to see what would be benchmarked
    python sglang_3way_local_benchmark.py --dry-run

    # Run all candidates
    python sglang_3way_local_benchmark.py

    # Run specific commit
    python sglang_3way_local_benchmark.py --commit 021f76e4

    # Skip agent phase (just baseline vs human)
    python sglang_3way_local_benchmark.py --no-agent

    # Specify agent info for output metadata
    python sglang_3way_local_benchmark.py --agent-name codex --agent-model gpt-5
"""

import os
import sys
import json
import time
import logging
import argparse
import subprocess
import tempfile
import shutil
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any
import re

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# Configuration
DEFAULT_DOCKER_REPO = "shikhar481/sglang-images"
ALT_DOCKER_REPOS = {
    "ayushnangia16": "ayushnangia16/nvidia-sglang-docker",
    "shikhar481": "shikhar481/sglang-images",
}
RESULTS_DIR = Path("omniperf_results_3way_sglang_docker")
CANDIDATES_FILE = Path("/tmp/sglang_3way_candidates.json")

# Benchmark parameters
SERVER_PORT = 30000
SERVER_TIMEOUT = 600  # seconds to wait for server to start
BENCHMARK_TIMEOUT = 900  # seconds for benchmark to complete


def detect_benchmark_mode(perf_command: str) -> str:
    """
    Detect benchmark mode from perf_command.

    Returns: 'serving' or 'throughput'
    """
    if "bench_serving" in perf_command.lower():
        return "serving"
    elif "bench_throughput" in perf_command.lower():
        return "throughput"
    # Default to serving for backward compatibility
    return "serving"


def get_gpu_config() -> str:
    """Get GPU configuration string."""
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if result.returncode == 0:
            gpus = [g.strip() for g in result.stdout.strip().split('\n') if g.strip()]
            if gpus:
                # Get primary GPU name and count
                gpu_name = gpus[0]
                # Simplify name (e.g., "NVIDIA H100 80GB HBM3" -> "H100")
                if "H100" in gpu_name:
                    return f"H100:{len(gpus)}"
                elif "A100" in gpu_name:
                    return f"A100:{len(gpus)}"
                elif "V100" in gpu_name:
                    return f"V100:{len(gpus)}"
                return f"{gpu_name}:{len(gpus)}"
    except Exception:
        pass
    return "GPU:1"


def check_docker_gpu():
    """Check if Docker with GPU support is available."""
    try:
        result = subprocess.run(
            ["docker", "run", "--rm", "--gpus", "all", "nvidia/cuda:12.4.0-base-ubuntu22.04", "nvidia-smi"],
            capture_output=True,
            text=True,
            timeout=60,
        )
        if result.returncode == 0:
            logger.info("Docker GPU support: OK")
            return True
        else:
            logger.error(f"Docker GPU check failed: {result.stderr}")
            return False
    except Exception as e:
        logger.error(f"Docker GPU check failed: {e}")
        return False


def pull_docker_image(image: str) -> bool:
    """Pull Docker image if not present."""
    logger.info(f"Pulling image: {image}")
    try:
        result = subprocess.run(
            ["docker", "pull", image],
            capture_output=True,
            text=True,
            timeout=600,
        )
        if result.returncode == 0:
            logger.info(f"Image pulled successfully: {image}")
            return True
        else:
            logger.error(f"Failed to pull image: {result.stderr}")
            return False
    except Exception as e:
        logger.error(f"Pull failed: {e}")
        return False


def load_candidates() -> List[Dict]:
    """Load benchmark candidates from JSON file."""
    if not CANDIDATES_FILE.exists():
        logger.error(f"Candidates file not found: {CANDIDATES_FILE}")
        logger.info("Run the discovery script first to generate candidates")
        return []

    with open(CANDIDATES_FILE) as f:
        return json.load(f)


def parse_benchmark_output(output: str) -> Dict[str, float]:
    """
    Parse benchmark output to extract metrics.

    Returns metrics with SGLang-native names.
    """
    metrics = {}

    # Patterns match SGLang output format: "Metric (unit):   value"
    patterns = {
        "request_throughput": r"Request throughput \(req/s\):\s*(\d+\.?\d*)",
        "output_token_throughput": r"Output token throughput \(tok/s\):\s*(\d+\.?\d*)",
        "total_token_throughput": r"Total token throughput \(tok/s\):\s*(\d+\.?\d*)",
        "mean_ttft_ms": r"Mean TTFT \(ms\):\s*(\d+\.?\d*)",
        "median_ttft_ms": r"Median TTFT \(ms\):\s*(\d+\.?\d*)",
        "p99_ttft_ms": r"P99 TTFT \(ms\):\s*(\d+\.?\d*)",
        "mean_tpot_ms": r"Mean TPOT \(ms\):\s*(\d+\.?\d*)",
        "median_tpot_ms": r"Median TPOT \(ms\):\s*(\d+\.?\d*)",
        "p99_tpot_ms": r"P99 TPOT \(ms\):\s*(\d+\.?\d*)",
        "mean_itl_ms": r"Mean ITL \(ms\):\s*(\d+\.?\d*)",
        "median_itl_ms": r"Median ITL \(ms\):\s*(\d+\.?\d*)",
        "p99_itl_ms": r"P99 ITL \(ms\):\s*(\d+\.?\d*)",
        "mean_e2e_latency_ms": r"Mean E2E Latency \(ms\):\s*(\d+\.?\d*)",
        "median_e2e_latency_ms": r"Median E2E Latency \(ms\):\s*(\d+\.?\d*)",
    }

    for key, pattern in patterns.items():
        match = re.search(pattern, output, re.IGNORECASE)
        if match:
            try:
                metrics[key] = float(match.group(1))
            except:
                pass

    return metrics


def convert_to_vllm_metrics(metrics: Dict[str, float], prefix: str) -> Dict[str, Any]:
    """
    Convert SGLang metrics to vLLM naming convention.

    Args:
        metrics: SGLang metrics dict
        prefix: 'baseline_', 'human_', or 'agent_'

    Returns dict with vLLM-compatible metric names.
    """
    result = {}

    # TTFT (Time To First Token)
    if "mean_ttft_ms" in metrics:
        result[f"{prefix}ttft_mean"] = metrics["mean_ttft_ms"]
    if "median_ttft_ms" in metrics:
        result[f"{prefix}ttft_median"] = metrics["median_ttft_ms"]
    if "p99_ttft_ms" in metrics:
        result[f"{prefix}ttft_p99"] = metrics["p99_ttft_ms"]

    # TPOT (Time Per Output Token)
    if "mean_tpot_ms" in metrics:
        result[f"{prefix}tpot_mean"] = metrics["mean_tpot_ms"]
    if "median_tpot_ms" in metrics:
        result[f"{prefix}tpot_median"] = metrics["median_tpot_ms"]
    if "p99_tpot_ms" in metrics:
        result[f"{prefix}tpot_p99"] = metrics["p99_tpot_ms"]

    # ITL (Inter-Token Latency)
    if "mean_itl_ms" in metrics:
        result[f"{prefix}itl_mean"] = metrics["mean_itl_ms"]
    if "median_itl_ms" in metrics:
        result[f"{prefix}itl_median"] = metrics["median_itl_ms"]
    if "p99_itl_ms" in metrics:
        result[f"{prefix}itl_p99"] = metrics["p99_itl_ms"]

    # Throughput
    if "request_throughput" in metrics:
        result[f"{prefix}throughput"] = metrics["request_throughput"]

    # Latency
    if "mean_e2e_latency_ms" in metrics:
        result[f"{prefix}latency_avg"] = metrics["mean_e2e_latency_ms"]
    if "median_e2e_latency_ms" in metrics:
        result[f"{prefix}latency_median"] = metrics["median_e2e_latency_ms"]

    return result


def calculate_improvements(
    baseline_metrics: Dict[str, float],
    target_metrics: Dict[str, float],
    prefix: str = ""
) -> Dict[str, Optional[float]]:
    """
    Calculate improvement percentages.

    For latency metrics (ttft, tpot, itl, latency): lower is better
    For throughput metrics: higher is better

    Returns dict with {prefix}improvement_{metric} keys.
    """
    improvements = {}

    latency_keys = ["ttft_mean", "ttft_median", "ttft_p99",
                    "tpot_mean", "tpot_median", "tpot_p99",
                    "itl_mean", "itl_median", "itl_p99",
                    "latency_avg", "latency_median"]

    throughput_keys = ["throughput"]

    def calc_improvement(base_val, new_val, higher_is_better=False):
        """Calculate % improvement. Positive = better."""
        if not base_val or base_val == 0:
            return None
        if higher_is_better:
            # Higher is better (throughput)
            return round(((new_val - base_val) / base_val) * 100, 2)
        else:
            # Lower is better (latency)
            return round(((base_val - new_val) / base_val) * 100, 2)

    # Calculate improvements for latency metrics
    for key in latency_keys:
        base_key = f"baseline_{key}"
        target_key_lookup = [f"human_{key}", f"agent_{key}"]

        if base_key in baseline_metrics:
            for tk in target_key_lookup:
                if tk in target_metrics:
                    improvement = calc_improvement(
                        baseline_metrics[base_key],
                        target_metrics[tk],
                        higher_is_better=False
                    )
                    improvements[f"{prefix}improvement_{key}"] = improvement
                    break

    # Calculate improvements for throughput metrics
    for key in throughput_keys:
        base_key = f"baseline_{key}"
        target_key_lookup = [f"human_{key}", f"agent_{key}"]

        if base_key in baseline_metrics:
            for tk in target_key_lookup:
                if tk in target_metrics:
                    improvement = calc_improvement(
                        baseline_metrics[base_key],
                        target_metrics[tk],
                        higher_is_better=True
                    )
                    improvements[f"{prefix}improvement_{key}"] = improvement
                    break

    return improvements


def run_benchmark_in_docker(
    image: str,
    model: str,
    perf_command: str,
    phase: str,
    agent_patch: str = None,
    hf_token: str = None,
    lora_server_args: str = "",
) -> Dict[str, Any]:
    """
    Run a single benchmark phase inside Docker container.

    Returns dict with metrics, raw_output, server_logs, success, error.
    """
    result = {
        "metrics": {},
        "raw_output": "",
        "server_logs": "",
        "success": False,
        "error": None,
    }

    container_name = f"sglang-benchmark-{phase}-{int(time.time())}"

    # Create temp directory for patch if needed
    patch_dir = None
    if agent_patch and phase == "agent":
        patch_dir = tempfile.mkdtemp(prefix="sglang_patch_")
        patch_file = Path(patch_dir) / "agent.patch"
        patch_file.write_text(agent_patch)

    try:
        # Build docker run command
        # Use --entrypoint "" to override potentially corrupted bash in some images
        docker_cmd = [
            "docker", "run",
            "--rm",
            "--entrypoint", "",
            "--gpus", "all",
            "--name", container_name,
            "--network", "host",  # Use host network for simplicity
            "-e", f"HF_TOKEN={hf_token or ''}",
            "-e", "TRANSFORMERS_CACHE=/root/.cache/huggingface",
        ]

        # Mount HuggingFace cache
        hf_cache = Path.home() / ".cache" / "huggingface"
        if hf_cache.exists():
            docker_cmd.extend(["-v", f"{hf_cache}:/root/.cache/huggingface"])

        # Mount patch directory if agent phase
        if patch_dir:
            docker_cmd.extend(["-v", f"{patch_dir}:/patches:ro"])

        docker_cmd.append(image)

        # Build server launch command with optional LoRA args
        server_cmd = f"python3 -m sglang.launch_server --model-path {model} --port {SERVER_PORT} --host 0.0.0.0"
        if lora_server_args:
            server_cmd += f" {lora_server_args}"
            # LoRA requires radix cache to be disabled for compatibility
            server_cmd += " --disable-radix-cache"

        # Create the benchmark script to run inside container (POSIX sh compatible)
        benchmark_script = f'''
#!/bin/sh
set -e

echo "=== SGLang Benchmark: {phase} ==="
echo "Model: {model}"
echo "Command: {perf_command}"

# Apply patch if agent phase
PATCH_FILE="/patches/agent.patch"
if [ "{phase}" = "agent" ] && [ -f "$PATCH_FILE" ]; then
    echo "Applying agent patch..."
    SGLANG_PATH=$(python3 -c "import sglang; print(sglang.__file__)" 2>/dev/null | xargs dirname | xargs dirname)
    if [ -n "$SGLANG_PATH" ]; then
        cd "$SGLANG_PATH"
        patch -p1 < "$PATCH_FILE" || echo "Patch may have partially failed"
        cd /
    fi
fi

# Start server in background
echo "Starting SGLang server..."
{server_cmd} 2>&1 | tee /tmp/server.log &

SERVER_PID=$!

# Wait for server to be ready
echo "Waiting for server..."
i=0
while [ $i -lt {SERVER_TIMEOUT // 5} ]; do
    if curl -s http://127.0.0.1:{SERVER_PORT}/health > /dev/null 2>&1; then
        echo "Server ready!"
        break
    fi
    sleep 5
    i=$((i + 1))
done

# Check if server is running
if ! curl -s http://127.0.0.1:{SERVER_PORT}/health > /dev/null 2>&1; then
    echo "ERROR: Server failed to start"
    echo "=== SERVER LOGS ==="
    cat /tmp/server.log
    exit 1
fi

# Run benchmark
echo "Running benchmark..."
BENCH_CMD="{perf_command}"
# Add port if not present (POSIX-compatible check)
case "$BENCH_CMD" in
    *"--port"*) ;;
    *) BENCH_CMD="$BENCH_CMD --port {SERVER_PORT}" ;;
esac
# Add host if not present
case "$BENCH_CMD" in
    *"--host"*) ;;
    *) BENCH_CMD="$BENCH_CMD --host 127.0.0.1" ;;
esac

echo "Executing: $BENCH_CMD"
$BENCH_CMD 2>&1 | tee /tmp/benchmark.log
BENCH_EXIT=$?

echo "=== BENCHMARK COMPLETE ==="
echo "Exit code: $BENCH_EXIT"

# Cleanup
kill $SERVER_PID 2>/dev/null || true

# Output results marker
echo "===BENCHMARK_OUTPUT==="
cat /tmp/benchmark.log
echo "===SERVER_LOGS==="
cat /tmp/server.log

exit $BENCH_EXIT
'''

        # Run docker with sh script (use /bin/sh for POSIX compatibility)
        full_cmd = docker_cmd + ["/bin/sh", "-c", benchmark_script]

        logger.info(f"[{phase}] Starting Docker container...")
        logger.debug(f"Command: {' '.join(full_cmd[:10])}...")

        proc = subprocess.run(
            full_cmd,
            capture_output=True,
            text=True,
            timeout=SERVER_TIMEOUT + BENCHMARK_TIMEOUT + 120,
        )

        output = proc.stdout + proc.stderr

        # Parse output
        if "===BENCHMARK_OUTPUT===" in output:
            bench_start = output.index("===BENCHMARK_OUTPUT===")
            bench_end = output.index("===SERVER_LOGS===") if "===SERVER_LOGS===" in output else len(output)
            bench_output = output[bench_start:bench_end]
            result["raw_output"] = bench_output[-10000:]

            metrics = parse_benchmark_output(bench_output)
            if metrics:
                result["metrics"] = metrics
                result["success"] = True
                logger.info(f"[{phase}] Metrics: {metrics}")
            else:
                result["error"] = "No metrics parsed from benchmark output"
                logger.warning(f"[{phase}] No metrics found")

        if "===SERVER_LOGS===" in output:
            server_start = output.index("===SERVER_LOGS===")
            result["server_logs"] = output[server_start:][-5000:]

        if proc.returncode != 0 and not result["success"]:
            result["error"] = f"Docker exited with code {proc.returncode}"
            logger.error(f"[{phase}] {result['error']}")

    except subprocess.TimeoutExpired:
        result["error"] = f"Timeout after {SERVER_TIMEOUT + BENCHMARK_TIMEOUT}s"
        logger.error(f"[{phase}] {result['error']}")
        # Try to kill container
        subprocess.run(["docker", "kill", container_name], capture_output=True)
    except Exception as e:
        result["error"] = str(e)
        logger.error(f"[{phase}] Exception: {e}")
    finally:
        # Cleanup
        if patch_dir:
            shutil.rmtree(patch_dir, ignore_errors=True)
        # Ensure container is removed
        subprocess.run(["docker", "rm", "-f", container_name], capture_output=True)

    return result


def run_3way_benchmark(
    candidate: Dict,
    hf_token: str = None,
    include_agent: bool = True,
    agent_name: str = "claude-code",
    agent_model: str = "sonnet-4.5",
) -> Dict[str, Any]:
    """
    Run true 3-way benchmark using local Docker.

    Returns combined results in vLLM-compatible format.
    """
    # Handle both short and full hashes - use first 8 chars for short version
    human_tag = candidate["human"]  # Tag for Docker image (could be short or full)
    parent_tag = candidate["parent"]  # Tag for Docker image
    human_short = human_tag[:8]  # Always use first 8 chars for short hash
    parent_short = parent_tag[:8]
    model = candidate["model"]
    perf_command = candidate["perf_command"]
    lora_server_args = candidate.get("lora_server_args", "")

    # Detect benchmark mode
    benchmark_mode = detect_benchmark_mode(perf_command)
    gpu_config = get_gpu_config()

    # Read agent patch
    patch_path = candidate.get("patch_path")
    agent_patch = ""
    if patch_path and Path(patch_path).exists():
        agent_patch = Path(patch_path).read_text()

    # Initialize result with vLLM-compatible structure
    result = {
        # Core metadata
        "commit_hash": candidate.get("human_full", human_short),
        "commit_short": human_short,
        "commit_subject": candidate.get("subject", ""),
        "repo": "sglang",
        "perf_command": perf_command,
        "files_changed": candidate.get("files_changed", []),
        "pr_url": candidate.get("pr_url", ""),
        "models": [model],
        "parent_commit": candidate.get("parent_full", parent_short),

        # Agent/benchmark metadata
        "agent_name": agent_name,
        "agent_model": agent_model,
        "benchmark_date": datetime.now().isoformat(),
        "benchmark_mode": benchmark_mode,
        "gpu_config": gpu_config,
        "data_source": "local_docker",

        # Flags per omniperf_v1 schema
        "has_serving": "bench_serving" in perf_command.lower(),
        "has_throughput": "bench_throughput" in perf_command.lower(),
        "has_latency": False,

        # Status
        "status": "error",
        "error": None,
        "duration_s": 0,
    }

    start_time = time.time()

    # Get Docker repo for this candidate (supports per-candidate repos)
    docker_repo = candidate.get("docker_repo", DEFAULT_DOCKER_REPO)

    # Docker images - use full hashes if available for ayushnangia16 repo
    if "ayushnangia16" in docker_repo:
        # ayushnangia16 images use full commit hashes
        human_image_tag = candidate.get("human_full", human_tag)
        parent_image_tag = candidate.get("parent_full", parent_tag)
    else:
        # shikhar481 images use short hashes
        human_image_tag = human_tag
        parent_image_tag = parent_tag

    baseline_image = f"{docker_repo}:{parent_image_tag}"
    human_image = f"{docker_repo}:{human_image_tag}"

    # Pull images first
    logger.info("Pulling Docker images...")
    if not pull_docker_image(baseline_image):
        result["error"] = f"Failed to pull baseline image: {baseline_image}"
        result["duration_s"] = time.time() - start_time
        return result

    if not pull_docker_image(human_image):
        result["error"] = f"Failed to pull human image: {human_image}"
        result["duration_s"] = time.time() - start_time
        return result

    # Run baseline phase
    logger.info(f"\n{'='*60}")
    logger.info("PHASE 1: BASELINE")
    logger.info(f"Image: {baseline_image}")
    logger.info(f"{'='*60}")

    baseline_result = run_benchmark_in_docker(
        image=baseline_image,
        model=model,
        perf_command=perf_command,
        phase="baseline",
        hf_token=hf_token,
        lora_server_args=lora_server_args,
    )

    # Convert baseline metrics to vLLM format
    baseline_vllm = convert_to_vllm_metrics(baseline_result["metrics"], "baseline_")
    result.update(baseline_vllm)
    result["baseline_raw"] = baseline_result["raw_output"]

    # Run human phase
    logger.info(f"\n{'='*60}")
    logger.info("PHASE 2: HUMAN")
    logger.info(f"Image: {human_image}")
    logger.info(f"{'='*60}")

    human_result = run_benchmark_in_docker(
        image=human_image,
        model=model,
        perf_command=perf_command,
        phase="human",
        hf_token=hf_token,
        lora_server_args=lora_server_args,
    )

    # Convert human metrics to vLLM format
    human_vllm = convert_to_vllm_metrics(human_result["metrics"], "human_")
    result.update(human_vllm)
    result["human_raw"] = human_result["raw_output"]

    # Calculate human improvements
    human_improvements = calculate_improvements(baseline_vllm, human_vllm, "human_")
    result.update(human_improvements)

    # Run agent phase (on baseline image with patch)
    if include_agent and agent_patch:
        logger.info(f"\n{'='*60}")
        logger.info("PHASE 3: AGENT")
        logger.info(f"Image: {baseline_image} + patch")
        logger.info(f"{'='*60}")

        agent_result = run_benchmark_in_docker(
            image=baseline_image,
            model=model,
            perf_command=perf_command,
            phase="agent",
            agent_patch=agent_patch,
            hf_token=hf_token,
            lora_server_args=lora_server_args,
        )

        # Convert agent metrics to vLLM format
        agent_vllm = convert_to_vllm_metrics(agent_result["metrics"], "agent_")
        result.update(agent_vllm)
        result["agent_raw"] = agent_result["raw_output"]

        # Calculate agent improvements (vs baseline)
        agent_improvements = calculate_improvements(baseline_vllm, agent_vllm, "agent_")
        result.update(agent_improvements)

        # Calculate agent vs human comparison
        if human_vllm and agent_vllm:
            agent_vs_human = {}
            for key in ["ttft_mean", "ttft_median", "throughput"]:
                human_key = f"human_{key}"
                agent_key = f"agent_{key}"
                if human_key in human_vllm and agent_key in agent_vllm:
                    human_val = human_vllm[human_key]
                    agent_val = agent_vllm[agent_key]
                    if human_val and human_val != 0:
                        is_latency = "ttft" in key or "tpot" in key or "itl" in key
                        if is_latency:
                            # Lower is better, positive means agent is better
                            pct = round(((human_val - agent_val) / human_val) * 100, 2)
                        else:
                            # Higher is better, positive means agent is better
                            pct = round(((agent_val - human_val) / human_val) * 100, 2)
                        agent_vs_human[f"agent_vs_human_{key}"] = pct
            result.update(agent_vs_human)
    else:
        result["agent_raw"] = ""

    # Determine overall status
    if baseline_vllm and human_vllm:
        result["status"] = "success"
    elif human_vllm:
        result["status"] = "partial"
        result["error"] = "Baseline failed"
    elif baseline_vllm:
        result["status"] = "partial"
        result["error"] = "Human failed"
    else:
        result["status"] = "error"
        result["error"] = "All phases failed"

    result["duration_s"] = time.time() - start_time

    return result


def save_result(result: Dict, results_dir: Path):
    """Save benchmark result to JSON file."""
    commit = result.get("commit_short", result.get("commit", "unknown"))
    commit_dir = results_dir / "sglang" / commit
    commit_dir.mkdir(parents=True, exist_ok=True)

    result_file = commit_dir / "benchmark_result.json"
    with open(result_file, 'w') as f:
        json.dump(result, f, indent=2)

    logger.info(f"Saved result to {result_file}")


def main():
    parser = argparse.ArgumentParser(description="Run SGLang 3-way Docker benchmarks locally")
    parser.add_argument("--dry-run", action="store_true", help="Show what would run without executing")
    parser.add_argument("--commit", type=str, help="Run specific commit only")
    parser.add_argument("--skip-existing", action="store_true", help="Skip commits with existing results")
    parser.add_argument("--no-agent", action="store_true", help="Skip agent phase (just baseline vs human)")
    parser.add_argument("--limit", type=int, help="Limit number of benchmarks to run")
    parser.add_argument("--agent-name", type=str, default="claude-code", help="Agent name for metadata (default: claude-code)")
    parser.add_argument("--agent-model", type=str, default="sonnet-4.5", help="Agent model for metadata (default: sonnet-4.5)")
    parser.add_argument("--status", type=str, help="Filter by status (e.g., 'ready', 'completed')")
    parser.add_argument("--include-all", action="store_true", help="Include all candidates regardless of status")
    args = parser.parse_args()

    # Setup
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # Get HF token
    hf_token = os.environ.get("HF_TOKEN", "")
    if not hf_token:
        logger.warning("HF_TOKEN not set - some models may fail to load")

    # Load candidates
    candidates = load_candidates()
    if not candidates:
        logger.error("No candidates to run")
        return

    logger.info(f"Loaded {len(candidates)} candidates")

    # Filter by commit if specified
    if args.commit:
        candidates = [c for c in candidates if c["human"].startswith(args.commit)]
        if not candidates:
            logger.error(f"No candidate matching commit {args.commit}")
            return

    # Filter by status (default: only 'ready' and 'completed')
    if args.status:
        candidates = [c for c in candidates if c.get("status") == args.status]
        logger.info(f"Filtered by status '{args.status}': {len(candidates)} candidates")
    elif not args.include_all:
        # Default: only run ready/completed candidates (skip broken ones)
        runnable_statuses = {"ready", "completed"}
        candidates = [c for c in candidates if c.get("status") in runnable_statuses]
        logger.info(f"Filtered to runnable status (ready/completed): {len(candidates)} candidates")

    # Check for existing results
    if args.skip_existing:
        existing = set()
        for rf in RESULTS_DIR.glob("sglang/*/benchmark_result.json"):
            try:
                r = json.loads(rf.read_text())
                if r.get("status") == "success":
                    existing.add(rf.parent.name)
            except:
                pass
        candidates = [c for c in candidates if c["human"] not in existing]
        logger.info(f"After skipping existing: {len(candidates)} candidates")

    # Apply limit
    if args.limit:
        candidates = candidates[:args.limit]

    if args.dry_run:
        print("\n=== DRY RUN - Would run these 3-way benchmarks ===")
        print(f"Default Docker repo: {DEFAULT_DOCKER_REPO}")
        print(f"Agent phase: {'ENABLED' if not args.no_agent else 'DISABLED'}")
        print(f"Agent name: {args.agent_name}")
        print(f"Agent model: {args.agent_model}")
        print()
        for i, c in enumerate(candidates, 1):
            bench_mode = detect_benchmark_mode(c["perf_command"])
            docker_repo = c.get("docker_repo", DEFAULT_DOCKER_REPO)
            status = c.get("status", "unknown")
            claimed = c.get("claimed_improvement", "")

            # Determine image tags based on repo
            if "ayushnangia16" in docker_repo:
                human_tag = c.get("human_full", c["human"])
                parent_tag = c.get("parent_full", c["parent"])
            else:
                human_tag = c["human"]
                parent_tag = c["parent"]

            print(f"{i}. {c['human']} (parent: {c['parent']})")
            print(f"   Subject: {c['subject'][:60]}...")
            print(f"   Model: {c['model']}")
            print(f"   Mode: {bench_mode}")
            print(f"   Docker repo: {docker_repo}")
            print(f"   Baseline: {docker_repo}:{parent_tag}")
            print(f"   Human: {docker_repo}:{human_tag}")
            print(f"   Status: {status}")
            if claimed:
                print(f"   Claimed: {claimed}")
            if c.get('lora_server_args'):
                print(f"   LoRA args: {c['lora_server_args']}")
            print()
        return

    # Check Docker GPU support
    if not check_docker_gpu():
        logger.error("Docker GPU support not available. Please install nvidia-container-toolkit.")
        return

    # Run benchmarks
    success_count = 0
    error_count = 0

    for i, candidate in enumerate(candidates, 1):
        logger.info(f"\n{'='*60}")
        logger.info(f"[{i}/{len(candidates)}] Running 3-way benchmark for {candidate['human']}")
        logger.info(f"Subject: {candidate['subject'][:60]}...")
        logger.info(f"{'='*60}")

        try:
            result = run_3way_benchmark(
                candidate=candidate,
                hf_token=hf_token,
                include_agent=not args.no_agent,
                agent_name=args.agent_name,
                agent_model=args.agent_model,
            )

            save_result(result, RESULTS_DIR)

            if result["status"] == "success":
                success_count += 1
                logger.info(f"SUCCESS ({result['duration_s']:.1f}s)")
                # Print improvements
                improvements = {k: v for k, v in result.items() if "improvement" in k and v is not None}
                if improvements:
                    logger.info(f"  Improvements: {improvements}")
            else:
                error_count += 1
                logger.error(f"FAILED: {result.get('error', 'Unknown error')}")

        except Exception as e:
            error_count += 1
            logger.error(f"Exception: {e}")

    # Summary
    logger.info(f"\n{'='*60}")
    logger.info("BENCHMARK RUN COMPLETE")
    logger.info(f"  Total: {len(candidates)}")
    logger.info(f"  Success: {success_count}")
    logger.info(f"  Errors: {error_count}")
    logger.info(f"  Results dir: {RESULTS_DIR}")
    logger.info(f"{'='*60}")


if __name__ == "__main__":
    main()
