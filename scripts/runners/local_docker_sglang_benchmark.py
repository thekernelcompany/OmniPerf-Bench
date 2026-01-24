#!/usr/bin/env python3
"""
Local Docker Benchmark Runner for SGLang commits.

Runs benchmarks inside pre-built Docker containers for commits that failed on Modal.
Supports serving, throughput, and latency benchmark types.

ARCHITECTURAL NOTE:
SGLang has ABI compatibility constraints with sgl_kernel and flashinfer.
Unlike vLLM, the Python overlay approach CANNOT work reliably.
Each phase (baseline, human, agent) requires its own Docker image.

Docker Image Strategy (searches both repos with fallback):
1. shikhar481/sglang-images:{commit[:8]}-src, -v2, -v3 (newer, multiple versions)
2. ayushnangia16/nvidia-sglang-docker:{full_40char_hash} (full hashes)

Usage:
    # Run human-only benchmark
    python local_docker_sglang_benchmark.py --commit 09deb20d --human-only

    # Run 3-way benchmark (requires baseline image to exist)
    python local_docker_sglang_benchmark.py --commit 09deb20d --3way

    # List available images for a commit
    python local_docker_sglang_benchmark.py --commit 09deb20d --check-images
"""

import argparse
import json
import os
import re
import subprocess
import sys
import time
import urllib.request
import urllib.error
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional, Dict, Any, List, Tuple

# Configuration - Docker image repositories
SGLANG_DOCKER_REPOS = [
    "shikhar481/sglang-images",       # Newer repo with -src, -v2, -v3 suffixes
    "ayushnangia16/nvidia-sglang-docker",  # Original repo with full hashes
]

# Results directory
RESULTS_DIR = Path("/home/ubuntu/OmniPerf-Bench/omniperf_results_3way_sglang")
OUTPUT_DIR = RESULTS_DIR / "docker_benchmark_results"
BASELINE_OUTPUT_DIR = RESULTS_DIR / "baseline_benchmark_results"
AGENT_OUTPUT_DIR = RESULTS_DIR / "agent_benchmark_results"

# Commit mapping file
COMMIT_MAPPING_FILE = Path("/home/ubuntu/OmniPerf-Bench/src/benchmark/fixes/sglang_commit_mapping.json")

# Agent patches configuration
CLAUDE_CODE_RUNS_DIR = Path("/home/ubuntu/OmniPerf-Bench/perf-agents-bench/state/runs/sglang/claude_code")

# SGLang repo URL
SGLANG_REPO_URL = "https://github.com/sgl-project/sglang.git"


@dataclass
class BenchmarkResult:
    """Benchmark result data."""
    commit_hash: str
    status: str  # success, error, timeout
    benchmark_type: str  # serving, throughput, latency
    model: str
    duration_s: float
    phase: str = "human"  # human, baseline, agent
    error: Optional[str] = None
    # Metrics
    ttft_mean: Optional[float] = None
    ttft_median: Optional[float] = None
    ttft_p99: Optional[float] = None
    tpot_mean: Optional[float] = None
    tpot_median: Optional[float] = None
    tpot_p99: Optional[float] = None
    itl_mean: Optional[float] = None
    itl_median: Optional[float] = None
    itl_p99: Optional[float] = None
    request_throughput: Optional[float] = None
    output_throughput: Optional[float] = None
    input_throughput: Optional[float] = None
    e2e_latency_mean: Optional[float] = None
    raw_output: Optional[str] = None


def get_hf_token() -> str:
    """Get HuggingFace token."""
    token_file = Path.home() / ".cache" / "huggingface" / "token"
    if token_file.exists():
        return token_file.read_text().strip()

    try:
        result = subprocess.run(
            ["python3", "-c", "from huggingface_hub import get_token; print(get_token())"],
            capture_output=True, text=True, timeout=10
        )
        return result.stdout.strip()
    except Exception:
        return ""


def check_docker_image_exists(image_tag: str) -> bool:
    """Check if a Docker image exists on DockerHub or locally."""
    # First check locally
    result = subprocess.run(
        ['docker', 'images', '-q', image_tag],
        capture_output=True, text=True, timeout=10
    )
    if result.stdout.strip():
        return True

    # Check DockerHub
    repo, tag = image_tag.rsplit(":", 1) if ":" in image_tag else (image_tag, "latest")
    url = f"https://hub.docker.com/v2/repositories/{repo}/tags/{tag}"
    try:
        req = urllib.request.Request(url, method='HEAD')
        urllib.request.urlopen(req, timeout=10)
        return True
    except urllib.error.HTTPError as e:
        if e.code == 404:
            return False
        return False
    except Exception:
        return False


def get_sglang_image(commit: str, phase: str = "human") -> Optional[str]:
    """
    Find a Docker image for an SGLang commit.

    Search strategy:
    1. For 'human' phase: Try shikhar481 repo with -src, -v3, -v2, '' suffixes
    2. For 'human' phase: Fall back to ayushnangia16 repo with full hash
    3. For 'baseline' phase: Try baseline-{commit[:12]} tag in both repos

    Returns full image tag or None if not found.
    """
    short = commit[:8]
    full = commit[:40] if len(commit) >= 40 else commit

    if phase == "baseline":
        # Look for baseline-tagged images
        candidates = [
            f"shikhar481/sglang-images:baseline-{commit[:12]}",
            f"ayushnangia16/nvidia-sglang-docker:baseline-{commit[:12]}",
        ]
    else:
        # Human/agent phase - look for commit images
        candidates = [
            # shikhar481 repo (newer, multiple versions per commit)
            f"shikhar481/sglang-images:{short}-src",
            f"shikhar481/sglang-images:{short}-v3",
            f"shikhar481/sglang-images:{short}-v2",
            f"shikhar481/sglang-images:{short}",
            # ayushnangia16 repo (full hashes)
            f"ayushnangia16/nvidia-sglang-docker:{full}",
        ]

    for image_tag in candidates:
        if check_docker_image_exists(image_tag):
            return image_tag

    return None


def find_all_available_images(commit: str) -> Dict[str, List[str]]:
    """Find all available Docker images for a commit across both repos."""
    short = commit[:8]
    full = commit[:40] if len(commit) >= 40 else commit

    available = {"human": [], "baseline": []}

    # Human/commit images
    human_candidates = [
        f"shikhar481/sglang-images:{short}-src",
        f"shikhar481/sglang-images:{short}-v3",
        f"shikhar481/sglang-images:{short}-v2",
        f"shikhar481/sglang-images:{short}",
        f"ayushnangia16/nvidia-sglang-docker:{full}",
    ]

    for image_tag in human_candidates:
        if check_docker_image_exists(image_tag):
            available["human"].append(image_tag)

    # Baseline images
    baseline_candidates = [
        f"shikhar481/sglang-images:baseline-{commit[:12]}",
        f"ayushnangia16/nvidia-sglang-docker:baseline-{commit[:12]}",
    ]

    for image_tag in baseline_candidates:
        if check_docker_image_exists(image_tag):
            available["baseline"].append(image_tag)

    return available


def load_commit_mapping() -> List[Dict[str, Any]]:
    """Load SGLang commit mapping (human commit -> base commit)."""
    if not COMMIT_MAPPING_FILE.exists():
        print(f"ERROR: Commit mapping file not found: {COMMIT_MAPPING_FILE}")
        return []

    with open(COMMIT_MAPPING_FILE) as f:
        data = json.load(f)
        return data.get("commits", [])


def find_commit_info(commit: str, mapping: List[Dict]) -> Optional[Dict]:
    """Find commit info from mapping by short hash."""
    short = commit[:8]
    for entry in mapping:
        if entry.get("human_commit_short", "")[:8] == short or \
           entry.get("human_commit", "")[:8] == short:
            return entry
    return None


def load_agent_patches() -> Dict[str, Dict[str, Any]]:
    """Load agent patches from Claude Code runs."""
    patches = {}

    if not CLAUDE_CODE_RUNS_DIR.exists():
        print(f"WARNING: Claude Code runs directory not found: {CLAUDE_CODE_RUNS_DIR}")
        return patches

    for run_dir in CLAUDE_CODE_RUNS_DIR.glob("*/*/sglang_*"):
        patch_file = run_dir / "model_patch.diff"
        journal_file = run_dir / "journal.json"

        if patch_file.exists() and journal_file.exists():
            try:
                journal = json.loads(journal_file.read_text())
                commits = journal.get("commits", {})
                full_commit = commits.get("human", "")

                if not full_commit:
                    continue

                short_hash = full_commit[:8]
                patch_content = patch_file.read_text()

                if not patch_content.strip():
                    continue

                patches[short_hash] = {
                    "patch_path": str(patch_file),
                    "patch_content": patch_content,
                    "full_commit": full_commit,
                    "parent_commit": commits.get("pre", ""),
                }
            except Exception as e:
                print(f"Warning: Failed to read patch from {run_dir}: {e}")

    return patches


def get_benchmark_type(perf_command: str) -> str:
    """Determine benchmark type from perf_command."""
    if 'bench_serving' in perf_command or 'benchmark_serving' in perf_command:
        return 'serving'
    elif 'bench_throughput' in perf_command or 'benchmark_throughput' in perf_command:
        return 'throughput'
    elif 'bench_latency' in perf_command or 'benchmark_latency' in perf_command:
        return 'latency'
    return 'serving'  # Default to serving


def extract_tp_from_command(perf_command: str) -> int:
    """Extract tensor parallel size from benchmark command."""
    tp_patterns = [
        r'--tp[=\s]+(\d+)',
        r'--tp-size[=\s]+(\d+)',
        r'--tensor-parallel-size[=\s]+(\d+)',
    ]
    for pattern in tp_patterns:
        match = re.search(pattern, perf_command)
        if match:
            return int(match.group(1))
    return 1


def parse_sglang_metrics(output: str) -> Dict[str, float]:
    """Parse metrics from SGLang benchmark output."""
    metrics = {}

    patterns = {
        'ttft_mean': r'Mean TTFT \(ms\):\s+([\d.]+)',
        'ttft_median': r'Median TTFT \(ms\):\s+([\d.]+)',
        'ttft_p99': r'P99 TTFT \(ms\):\s+([\d.]+)',
        'tpot_mean': r'Mean TPOT \(ms\):\s+([\d.]+)',
        'tpot_median': r'Median TPOT \(ms\):\s+([\d.]+)',
        'tpot_p99': r'P99 TPOT \(ms\):\s+([\d.]+)',
        'itl_mean': r'Mean ITL \(ms\):\s+([\d.]+)',
        'itl_median': r'Median ITL \(ms\):\s+([\d.]+)',
        'itl_p99': r'P99 ITL \(ms\):\s+([\d.]+)',
        'request_throughput': r'Request throughput \(req/s\):\s+([\d.]+)',
        'output_throughput': r'Output token throughput \(tok/s\):\s+([\d.]+)',
        'input_throughput': r'Input token throughput \(tok/s\):\s+([\d.]+)',
        'e2e_latency_mean': r'Mean E2E Latency \(ms\):\s+([\d.]+)',
    }

    for key, pattern in patterns.items():
        match = re.search(pattern, output, re.IGNORECASE)
        if match:
            metrics[key] = float(match.group(1))

    return metrics


def extract_model_from_command(perf_command: str) -> Optional[str]:
    """Extract model name from perf_command."""
    patterns = [
        r'--model[=\s]+["\']?([^\s"\']+)',
        r'--model-path[=\s]+["\']?([^\s"\']+)',
    ]
    for pattern in patterns:
        match = re.search(pattern, perf_command)
        if match:
            return match.group(1)
    return None


def run_human_serving_benchmark(
    human_commit: str,
    model: str,
    perf_command: str,
    hf_token: str,
    timeout: int = 1800
) -> BenchmarkResult:
    """Run a serving benchmark for human commit inside Docker."""
    start_time = time.time()

    # Find Docker image
    docker_image = get_sglang_image(human_commit, phase="human")
    if not docker_image:
        return BenchmarkResult(
            commit_hash=human_commit, status='error', benchmark_type='serving',
            model=model, duration_s=0, phase='human',
            error=f'No Docker image found for {human_commit[:8]}'
        )

    print(f"  Using Docker image: {docker_image}")

    # Extract benchmark parameters
    tp_size = extract_tp_from_command(perf_command)

    docker_cmd = f'''
    set -e
    COMMIT="{human_commit}"
    MODEL="{model}"

    # Fix dependencies
    pip install typing_extensions>=4.10.0 --upgrade -q 2>/dev/null || true
    pip install numpy"<2.0" tqdm requests aiohttp -q 2>/dev/null || true

    # Set PYTHONPATH for SGLang (check common paths)
    if [ -d "/sglang/python" ]; then
        export PYTHONPATH="/sglang/python:$PYTHONPATH"
    elif [ -d "/sgl-workspace/sglang/python" ]; then
        export PYTHONPATH="/sgl-workspace/sglang/python:$PYTHONPATH"
    fi

    # Check SGLang version
    echo "=== Checking SGLang installation ==="
    python3 -c "import sglang; print(f'SGLang version: {{sglang.__version__}}'); print(f'SGLang path: {{sglang.__path__}}')" || echo "SGLang import check failed"

    # Start SGLang server
    echo "=== Starting SGLang server ==="
    python3 -m sglang.launch_server \\
        --model-path $MODEL \\
        --port 30000 \\
        --host 0.0.0.0 \\
        --tp-size {tp_size} \\
        --log-level info 2>&1 &
    SERVER_PID=$!

    # Wait for server with better detection
    echo "Waiting for server (pid=$SERVER_PID)..."
    for i in $(seq 1 300); do
        # Check if server is responding
        HEALTH=$(curl -s http://localhost:30000/health 2>/dev/null || true)
        if echo "$HEALTH" | grep -qiE "ok|healthy|true|success"; then
            echo "SERVER_READY_AFTER=${{i}}s"
            break
        fi
        # Also check for v1/models endpoint (OpenAI-compatible)
        MODELS=$(curl -s http://localhost:30000/v1/models 2>/dev/null || true)
        if echo "$MODELS" | grep -qiE "model|data"; then
            echo "SERVER_READY_AFTER=${{i}}s"
            break
        fi
        # Check if server process died
        if ! kill -0 $SERVER_PID 2>/dev/null; then
            echo "SERVER_CRASHED"
            wait $SERVER_PID 2>/dev/null || true
            exit 1
        fi
        sleep 1
    done

    # Verify server
    if ! curl -s http://localhost:30000/health > /dev/null 2>&1; then
        if ! curl -s http://localhost:30000/v1/models > /dev/null 2>&1; then
            echo "SERVER_TIMEOUT"
            kill $SERVER_PID 2>/dev/null || true
            exit 1
        fi
    fi

    # Run benchmark
    echo "=== Running benchmark ==="
    python3 -m sglang.bench_serving \\
        --backend sglang \\
        --host 127.0.0.1 \\
        --port 30000 \\
        --num-prompts 100 \\
        --random-input-len 256 \\
        --random-output-len 64 \\
        2>&1

    echo "BENCHMARK_DONE"
    kill $SERVER_PID 2>/dev/null || true
    '''

    # Determine GPU flag based on TP size
    gpu_flag = f'"device={",".join(str(i) for i in range(tp_size))}"' if tp_size > 1 else 'all'

    try:
        result = subprocess.run(
            [
                'docker', 'run', '--rm',
                '--gpus', gpu_flag,
                '-e', f'HF_TOKEN={hf_token}',
                '-e', f'HUGGING_FACE_HUB_TOKEN={hf_token}',
                '-v', '/home/ubuntu/.cache/huggingface:/root/.cache/huggingface',
                '--shm-size=16g',
                '--entrypoint', 'bash',
                docker_image,
                '-c', docker_cmd
            ],
            capture_output=True, text=True, timeout=timeout
        )

        output = result.stdout + result.stderr
        duration = time.time() - start_time

        if 'SERVER_CRASHED' in output:
            return BenchmarkResult(
                commit_hash=human_commit, status='error', benchmark_type='serving',
                model=model, duration_s=duration, phase='human',
                error='Server crashed during startup',
                raw_output=output[-5000:]
            )

        if 'SERVER_TIMEOUT' in output:
            return BenchmarkResult(
                commit_hash=human_commit, status='error', benchmark_type='serving',
                model=model, duration_s=duration, phase='human',
                error='Server startup timeout',
                raw_output=output[-5000:]
            )

        # Parse metrics
        metrics = parse_sglang_metrics(output)

        if not metrics:
            return BenchmarkResult(
                commit_hash=human_commit, status='error', benchmark_type='serving',
                model=model, duration_s=duration, phase='human',
                error='No metrics in output',
                raw_output=output[-5000:]
            )

        return BenchmarkResult(
            commit_hash=human_commit, status='success', benchmark_type='serving',
            model=model, duration_s=duration, phase='human',
            raw_output=output[-5000:],
            **metrics
        )

    except subprocess.TimeoutExpired:
        return BenchmarkResult(
            commit_hash=human_commit, status='timeout', benchmark_type='serving',
            model=model, duration_s=timeout, phase='human',
            error=f'Benchmark timed out after {timeout}s'
        )
    except Exception as e:
        return BenchmarkResult(
            commit_hash=human_commit, status='error', benchmark_type='serving',
            model=model, duration_s=time.time() - start_time, phase='human',
            error=str(e)
        )


def run_baseline_serving_benchmark(
    base_commit: str,
    model: str,
    perf_command: str,
    hf_token: str,
    timeout: int = 1800
) -> BenchmarkResult:
    """Run a serving benchmark for baseline commit inside Docker."""
    start_time = time.time()

    # Find baseline Docker image
    docker_image = get_sglang_image(base_commit, phase="baseline")
    if not docker_image:
        # Fall back to human image if no baseline image exists
        docker_image = get_sglang_image(base_commit, phase="human")
        if not docker_image:
            return BenchmarkResult(
                commit_hash=base_commit, status='error', benchmark_type='serving',
                model=model, duration_s=0, phase='baseline',
                error=f'No Docker image found for baseline {base_commit[:8]}'
            )

    print(f"  Using baseline Docker image: {docker_image}")

    # Extract benchmark parameters
    tp_size = extract_tp_from_command(perf_command)

    docker_cmd = f'''
    set -e
    COMMIT="{base_commit}"
    MODEL="{model}"

    # Fix dependencies
    pip install typing_extensions>=4.10.0 --upgrade -q 2>/dev/null || true
    pip install numpy"<2.0" tqdm requests aiohttp -q 2>/dev/null || true

    # Set PYTHONPATH for SGLang
    export PYTHONPATH="/sgl-workspace/sglang/python:$PYTHONPATH"

    # Check SGLang version
    python3 -c "import sglang; print(f'SGLang version: {{sglang.__version__}}')" 2>/dev/null || echo "SGLang import check failed"

    # Start SGLang server
    echo "=== Starting BASELINE SGLang server ==="
    python3 -m sglang.launch_server \\
        --model-path $MODEL \\
        --port 30001 \\
        --host 0.0.0.0 \\
        --tp-size {tp_size} \\
        --log-level info 2>&1 &
    SERVER_PID=$!

    # Wait for server
    echo "Waiting for server..."
    for i in $(seq 1 300); do
        if curl -s http://localhost:30001/health 2>/dev/null | grep -q "ok\\|healthy"; then
            echo "BASELINE_SERVER_READY_AFTER=${{i}}s"
            break
        fi
        if ! kill -0 $SERVER_PID 2>/dev/null; then
            echo "BASELINE_SERVER_CRASHED"
            wait $SERVER_PID 2>/dev/null || true
            exit 1
        fi
        sleep 1
    done

    # Verify server
    if ! curl -s http://localhost:30001/health > /dev/null 2>&1; then
        echo "BASELINE_SERVER_TIMEOUT"
        kill $SERVER_PID 2>/dev/null || true
        exit 1
    fi

    # Run benchmark
    echo "=== Running BASELINE benchmark ==="
    python3 -m sglang.bench_serving \\
        --backend sglang \\
        --host 127.0.0.1 \\
        --port 30001 \\
        --num-prompts 100 \\
        --random-input-len 256 \\
        --random-output-len 64 \\
        2>&1

    echo "BASELINE_BENCHMARK_DONE"
    kill $SERVER_PID 2>/dev/null || true
    '''

    gpu_flag = f'"device={",".join(str(i) for i in range(tp_size))}"' if tp_size > 1 else 'all'

    try:
        result = subprocess.run(
            [
                'docker', 'run', '--rm',
                '--gpus', gpu_flag,
                '-e', f'HF_TOKEN={hf_token}',
                '-e', f'HUGGING_FACE_HUB_TOKEN={hf_token}',
                '-v', '/home/ubuntu/.cache/huggingface:/root/.cache/huggingface',
                '--shm-size=16g',
                '--entrypoint', 'bash',
                docker_image,
                '-c', docker_cmd
            ],
            capture_output=True, text=True, timeout=timeout
        )

        output = result.stdout + result.stderr
        duration = time.time() - start_time

        if 'BASELINE_SERVER_CRASHED' in output:
            return BenchmarkResult(
                commit_hash=base_commit, status='error', benchmark_type='serving',
                model=model, duration_s=duration, phase='baseline',
                error='Baseline server crashed during startup',
                raw_output=output[-5000:]
            )

        if 'BASELINE_SERVER_TIMEOUT' in output:
            return BenchmarkResult(
                commit_hash=base_commit, status='error', benchmark_type='serving',
                model=model, duration_s=duration, phase='baseline',
                error='Baseline server startup timeout',
                raw_output=output[-5000:]
            )

        metrics = parse_sglang_metrics(output)

        if not metrics:
            return BenchmarkResult(
                commit_hash=base_commit, status='error', benchmark_type='serving',
                model=model, duration_s=duration, phase='baseline',
                error='No metrics in baseline output',
                raw_output=output[-5000:]
            )

        return BenchmarkResult(
            commit_hash=base_commit, status='success', benchmark_type='serving',
            model=model, duration_s=duration, phase='baseline',
            raw_output=output[-5000:],
            **metrics
        )

    except subprocess.TimeoutExpired:
        return BenchmarkResult(
            commit_hash=base_commit, status='timeout', benchmark_type='serving',
            model=model, duration_s=timeout, phase='baseline',
            error=f'Baseline benchmark timed out after {timeout}s'
        )
    except Exception as e:
        return BenchmarkResult(
            commit_hash=base_commit, status='error', benchmark_type='serving',
            model=model, duration_s=time.time() - start_time, phase='baseline',
            error=str(e)
        )


def run_combined_3way_serving(
    human_commit: str,
    base_commit: str,
    model: str,
    perf_command: str,
    agent_patch_path: Optional[Path],
    hf_token: str,
    timeout: int = 5400
) -> Tuple[BenchmarkResult, BenchmarkResult, Optional[BenchmarkResult]]:
    """
    Run 3-way benchmark: baseline, human, and optionally agent.

    Returns tuple of (baseline_result, human_result, agent_result).
    agent_result is None if no agent patch is provided.
    """
    print(f"\n{'='*60}")
    print(f"3-WAY BENCHMARK: {human_commit[:8]}")
    print(f"{'='*60}")
    print(f"  Human commit:    {human_commit[:8]}")
    print(f"  Baseline commit: {base_commit[:8]}")
    print(f"  Model: {model}")
    print(f"  Agent patch: {'Yes' if agent_patch_path else 'No'}")

    # Check for required images
    human_image = get_sglang_image(human_commit, phase="human")
    baseline_image = get_sglang_image(base_commit, phase="baseline")

    if not human_image:
        human_image = get_sglang_image(human_commit, phase="human")

    print(f"\n  Human image: {human_image or 'NOT FOUND'}")
    print(f"  Baseline image: {baseline_image or 'NOT FOUND'}")

    # Run baseline benchmark
    print(f"\n[1/3] Running BASELINE benchmark...")
    if baseline_image:
        baseline_result = run_baseline_serving_benchmark(
            base_commit, model, perf_command, hf_token, timeout=timeout//3
        )
    else:
        baseline_result = BenchmarkResult(
            commit_hash=base_commit, status='error', benchmark_type='serving',
            model=model, duration_s=0, phase='baseline',
            error='No baseline Docker image available'
        )
    print(f"  Baseline: {baseline_result.status}")

    # Run human benchmark
    print(f"\n[2/3] Running HUMAN benchmark...")
    if human_image:
        human_result = run_human_serving_benchmark(
            human_commit, model, perf_command, hf_token, timeout=timeout//3
        )
    else:
        human_result = BenchmarkResult(
            commit_hash=human_commit, status='error', benchmark_type='serving',
            model=model, duration_s=0, phase='human',
            error='No human Docker image available'
        )
    print(f"  Human: {human_result.status}")

    # Run agent benchmark (if patch provided)
    agent_result = None
    if agent_patch_path and agent_patch_path.exists():
        print(f"\n[3/3] Running AGENT benchmark...")
        agent_result = run_agent_serving_benchmark(
            base_commit, model, perf_command, agent_patch_path, hf_token, timeout=timeout//3
        )
        print(f"  Agent: {agent_result.status}")
    else:
        print(f"\n[3/3] Skipping AGENT benchmark (no patch)")

    return baseline_result, human_result, agent_result


def run_agent_serving_benchmark(
    base_commit: str,
    model: str,
    perf_command: str,
    agent_patch_path: Path,
    hf_token: str,
    timeout: int = 2400
) -> BenchmarkResult:
    """
    Run agent benchmark by applying patch to baseline image.

    NOTE: Due to SGLang's ABI constraints, the Python overlay approach has limitations.
    This function attempts to apply the patch and run, but may fail if the patch
    affects compiled extensions.
    """
    start_time = time.time()

    # Find baseline Docker image for agent
    docker_image = get_sglang_image(base_commit, phase="baseline")
    if not docker_image:
        docker_image = get_sglang_image(base_commit, phase="human")
        if not docker_image:
            return BenchmarkResult(
                commit_hash=base_commit, status='error', benchmark_type='serving',
                model=model, duration_s=0, phase='agent',
                error=f'No Docker image found for agent base {base_commit[:8]}'
            )

    print(f"  Using agent base Docker image: {docker_image}")

    tp_size = extract_tp_from_command(perf_command)

    docker_cmd = f'''
    set -e
    MODEL="{model}"

    # Fix dependencies
    pip install typing_extensions>=4.10.0 --upgrade -q 2>/dev/null || true
    pip install numpy"<2.0" tqdm requests aiohttp -q 2>/dev/null || true

    # Set PYTHONPATH for SGLang
    export PYTHONPATH="/sgl-workspace/sglang/python:$PYTHONPATH"

    # Apply agent patch
    echo "=== Applying agent patch ==="
    cd /sgl-workspace/sglang || cd /opt/sglang || echo "No SGLang source directory found"
    if [ -f /agent_patch.diff ]; then
        if git apply --check /agent_patch.diff 2>/dev/null; then
            git apply /agent_patch.diff
            echo "AGENT_PATCH_APPLIED"
        elif patch -p1 --dry-run < /agent_patch.diff 2>/dev/null; then
            patch -p1 < /agent_patch.diff
            echo "AGENT_PATCH_APPLIED"
        else
            echo "AGENT_PATCH_FAILED"
            exit 1
        fi
    else
        echo "No agent patch file found"
        exit 1
    fi

    # Check SGLang version after patch
    python3 -c "import sglang; print(f'SGLang version: {{sglang.__version__}}')" 2>/dev/null || echo "SGLang import check failed"

    # Start SGLang server with patched code
    echo "=== Starting AGENT SGLang server ==="
    python3 -m sglang.launch_server \\
        --model-path $MODEL \\
        --port 30002 \\
        --host 0.0.0.0 \\
        --tp-size {tp_size} \\
        --log-level info 2>&1 &
    SERVER_PID=$!

    # Wait for server
    echo "Waiting for server..."
    for i in $(seq 1 300); do
        if curl -s http://localhost:30002/health 2>/dev/null | grep -q "ok\\|healthy"; then
            echo "AGENT_SERVER_READY_AFTER=${{i}}s"
            break
        fi
        if ! kill -0 $SERVER_PID 2>/dev/null; then
            echo "AGENT_SERVER_CRASHED"
            wait $SERVER_PID 2>/dev/null || true
            exit 1
        fi
        sleep 1
    done

    # Verify server
    if ! curl -s http://localhost:30002/health > /dev/null 2>&1; then
        echo "AGENT_SERVER_TIMEOUT"
        kill $SERVER_PID 2>/dev/null || true
        exit 1
    fi

    # Run benchmark
    echo "=== Running AGENT benchmark ==="
    python3 -m sglang.bench_serving \\
        --backend sglang \\
        --host 127.0.0.1 \\
        --port 30002 \\
        --num-prompts 100 \\
        --random-input-len 256 \\
        --random-output-len 64 \\
        2>&1

    echo "AGENT_BENCHMARK_DONE"
    kill $SERVER_PID 2>/dev/null || true
    '''

    gpu_flag = f'"device={",".join(str(i) for i in range(tp_size))}"' if tp_size > 1 else 'all'

    try:
        result = subprocess.run(
            [
                'docker', 'run', '--rm',
                '--gpus', gpu_flag,
                '-e', f'HF_TOKEN={hf_token}',
                '-e', f'HUGGING_FACE_HUB_TOKEN={hf_token}',
                '-v', '/home/ubuntu/.cache/huggingface:/root/.cache/huggingface',
                '-v', f'{agent_patch_path}:/agent_patch.diff:ro',
                '--shm-size=16g',
                '--entrypoint', 'bash',
                docker_image,
                '-c', docker_cmd
            ],
            capture_output=True, text=True, timeout=timeout
        )

        output = result.stdout + result.stderr
        duration = time.time() - start_time

        if 'AGENT_PATCH_FAILED' in output:
            return BenchmarkResult(
                commit_hash=base_commit, status='error', benchmark_type='serving',
                model=model, duration_s=duration, phase='agent',
                error='Agent patch failed to apply',
                raw_output=output[-5000:]
            )

        if 'AGENT_SERVER_CRASHED' in output:
            return BenchmarkResult(
                commit_hash=base_commit, status='error', benchmark_type='serving',
                model=model, duration_s=duration, phase='agent',
                error='Agent server crashed (possible ABI mismatch)',
                raw_output=output[-5000:]
            )

        if 'AGENT_SERVER_TIMEOUT' in output:
            return BenchmarkResult(
                commit_hash=base_commit, status='error', benchmark_type='serving',
                model=model, duration_s=duration, phase='agent',
                error='Agent server startup timeout',
                raw_output=output[-5000:]
            )

        metrics = parse_sglang_metrics(output)

        if not metrics:
            return BenchmarkResult(
                commit_hash=base_commit, status='error', benchmark_type='serving',
                model=model, duration_s=duration, phase='agent',
                error='No metrics in agent output',
                raw_output=output[-5000:]
            )

        return BenchmarkResult(
            commit_hash=base_commit, status='success', benchmark_type='serving',
            model=model, duration_s=duration, phase='agent',
            raw_output=output[-5000:],
            **metrics
        )

    except subprocess.TimeoutExpired:
        return BenchmarkResult(
            commit_hash=base_commit, status='timeout', benchmark_type='serving',
            model=model, duration_s=timeout, phase='agent',
            error=f'Agent benchmark timed out after {timeout}s'
        )
    except Exception as e:
        return BenchmarkResult(
            commit_hash=base_commit, status='error', benchmark_type='serving',
            model=model, duration_s=time.time() - start_time, phase='agent',
            error=str(e)
        )


def save_result(result: BenchmarkResult, output_dir: Path):
    """Save benchmark result to JSON file."""
    output_dir.mkdir(parents=True, exist_ok=True)

    filename = f"{result.commit_hash[:8]}_{result.phase}_{result.benchmark_type}.json"
    output_file = output_dir / filename

    with open(output_file, 'w') as f:
        json.dump(asdict(result), f, indent=2)

    print(f"  Saved: {output_file}")


def main():
    parser = argparse.ArgumentParser(description="Local Docker Benchmark Runner for SGLang")
    parser.add_argument("--commit", type=str, required=True, help="Commit hash to benchmark")
    parser.add_argument("--model", type=str, help="Override model (default: from dataset)")
    parser.add_argument("--human-only", action="store_true", help="Only run human benchmark")
    parser.add_argument("--3way", dest="three_way", action="store_true", help="Run 3-way benchmark")
    parser.add_argument("--check-images", action="store_true", help="Check available images and exit")
    parser.add_argument("--timeout", type=int, default=1800, help="Timeout per benchmark (seconds)")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be run without executing")
    args = parser.parse_args()

    commit = args.commit

    # Check available images
    if args.check_images:
        print(f"\nChecking available images for {commit[:8]}...")
        available = find_all_available_images(commit)
        print(f"\nHuman images:")
        for img in available["human"]:
            print(f"  - {img}")
        if not available["human"]:
            print("  (none found)")
        print(f"\nBaseline images:")
        for img in available["baseline"]:
            print(f"  - {img}")
        if not available["baseline"]:
            print("  (none found)")
        return 0

    # Load commit mapping to get base commit and model
    mapping = load_commit_mapping()
    commit_info = find_commit_info(commit, mapping)

    if not commit_info:
        print(f"ERROR: Commit {commit} not found in mapping file")
        return 1

    human_commit = commit_info.get("human_commit", commit)
    base_commit = commit_info.get("base_commit", "")

    # Determine model
    model = args.model
    if not model:
        # Try to extract from perf_command in dataset
        # For now, use a default model
        model = "meta-llama/Llama-3.1-8B-Instruct"

    # Build perf_command (simplified for now)
    perf_command = f"python -m sglang.bench_serving --model {model} --num-prompts 100"

    print(f"\nBenchmark Configuration:")
    print(f"  Human commit:    {human_commit[:12]}")
    print(f"  Base commit:     {base_commit[:12] if base_commit else 'N/A'}")
    print(f"  Model:           {model}")
    print(f"  Mode:            {'3-way' if args.three_way else 'human-only'}")

    if args.dry_run:
        print("\n[DRY RUN] Would execute benchmarks")
        return 0

    # Get HF token
    hf_token = get_hf_token()
    if not hf_token:
        print("WARNING: No HuggingFace token found")

    # Create output directories
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if args.human_only:
        # Run human-only benchmark
        print(f"\n{'='*60}")
        print("HUMAN-ONLY BENCHMARK")
        print(f"{'='*60}")

        result = run_human_serving_benchmark(
            human_commit, model, perf_command, hf_token, timeout=args.timeout
        )

        print(f"\nResult: {result.status}")
        if result.status == 'success':
            print(f"  Request throughput: {result.request_throughput:.2f} req/s" if result.request_throughput else "")
            print(f"  Output throughput:  {result.output_throughput:.2f} tok/s" if result.output_throughput else "")
            print(f"  TTFT mean:          {result.ttft_mean:.2f} ms" if result.ttft_mean else "")
        elif result.error:
            print(f"  Error: {result.error}")

        save_result(result, OUTPUT_DIR)

    elif args.three_way:
        # Run 3-way benchmark
        if not base_commit:
            print("ERROR: No base commit found in mapping for 3-way benchmark")
            return 1

        # Load agent patches
        agent_patches = load_agent_patches()
        agent_patch_info = agent_patches.get(human_commit[:8])
        agent_patch_path = Path(agent_patch_info["patch_path"]) if agent_patch_info else None

        baseline_result, human_result, agent_result = run_combined_3way_serving(
            human_commit, base_commit, model, perf_command,
            agent_patch_path, hf_token, timeout=args.timeout * 3
        )

        # Print summary
        print(f"\n{'='*60}")
        print("BENCHMARK SUMMARY")
        print(f"{'='*60}")
        print(f"  Baseline: {baseline_result.status}")
        print(f"  Human:    {human_result.status}")
        if agent_result:
            print(f"  Agent:    {agent_result.status}")

        if baseline_result.status == 'success' and human_result.status == 'success':
            base_throughput = baseline_result.request_throughput or 0
            human_throughput = human_result.request_throughput or 0
            if base_throughput > 0:
                improvement = (human_throughput - base_throughput) / base_throughput * 100
                print(f"\n  Human improvement: {improvement:+.1f}% throughput")

        # Save results
        save_result(baseline_result, BASELINE_OUTPUT_DIR)
        save_result(human_result, OUTPUT_DIR)
        if agent_result:
            save_result(agent_result, AGENT_OUTPUT_DIR)

    else:
        print("ERROR: Specify --human-only or --3way")
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
