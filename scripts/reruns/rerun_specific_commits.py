#!/usr/bin/env python3
"""
Rerun specific commits with fixed perf_commands using local Docker.

This script reruns benchmark commits that failed due to incorrect perf_commands.

Commits to rerun:
- 2deb029d: BlockManagerV2 prefix caching - brackets in command fixed
- 9f1710f1: MLA prefill performance - wrong CLI args fixed

Usage:
    python scripts/reruns/rerun_specific_commits.py [--dry-run] [--commit COMMIT]
"""

import argparse
import json
import os
import re
import subprocess
import sys
import time
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional

# Docker image sources
HUMAN_IMAGE_PREFIX = "ayushnangia16/nvidia-vllm-docker"
BASELINE_IMAGE_PREFIX = "shikhar481/vllm_fixed_human_images"

# Results directory
RESULTS_DIR = Path("/root/OmniPerf-Bench/omniperf_results_3way_claude_code/results/docker")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# Commits to rerun with fixed configurations
COMMITS_TO_RERUN = {
    '2deb029d': {
        'full_hash': '2deb029d115dadd012ce5ea70487a207cb025493',
        'parent_hash': '029c71de11bc3bcf84a1b3cf9d91e79ab6949799',
        'baseline_image': None,  # No pre-built baseline, need to build from source
        'perf_command': 'python3 benchmarks/benchmark_prefix_caching.py --model neuralmagic/Meta-Llama-3-8B-Instruct-FP8 --output-len 200 --enable-prefix-caching --use-v2-block-manager',
        'model': 'neuralmagic/Meta-Llama-3-8B-Instruct-FP8',
        'subject': '[Performance][BlockManagerV2] Mark prefix cache block as computed after schedule (#7822)',
        'benchmark_type': 'prefix_caching',
        'pr_url': 'https://github.com/vllm-project/vllm/pull/7822',
        'original_issue': 'Brackets [--use-v2-block-manager] left in command',
    },
    '9f1710f1': {
        'full_hash': '9f1710f1ace3535920c0bb6d4cc329c36289080e',
        'parent_hash': 'e642ec962cf2283f9aa44492727e6efc17a32129',
        'baseline_image': 'shikhar481/vllm_fixed_human_images:baseline-e642ec962cf2',  # Pre-built baseline available!
        'perf_command': 'python benchmarks/benchmark_serving.py --model deepseek-ai/DeepSeek-V2-Lite-Chat --random-input-len 8192 --random-output-len 64 --dataset-name random --num-prompts 20 --request-rate 1',
        'model': 'deepseek-ai/DeepSeek-V2-Lite-Chat',
        'subject': 'Fix mla prefill context performance (#13897)',
        'benchmark_type': 'serving',
        'pr_url': 'https://github.com/vllm-project/vllm/pull/13897',
        'original_issue': '--input-len instead of --random-input-len',
    },
}


@dataclass
class BenchmarkResult:
    """Benchmark result data."""
    commit_hash: str
    status: str  # success, error, timeout
    benchmark_type: str
    model: str
    duration_s: float
    version: str  # baseline, human, agent
    error: Optional[str] = None
    metrics: Optional[Dict[str, Any]] = None
    raw_output: Optional[str] = None


def get_hf_token() -> str:
    """Get HuggingFace token."""
    token_file = Path.home() / ".cache" / "huggingface" / "token"
    if token_file.exists():
        return token_file.read_text().strip()
    return os.environ.get("HF_TOKEN", "")


def check_docker_gpu() -> bool:
    """Check if Docker can access GPU."""
    try:
        result = subprocess.run(
            ['docker', 'run', '--rm', '--gpus', 'all', 'nvidia/cuda:12.1.0-base-ubuntu22.04', 'nvidia-smi'],
            capture_output=True, text=True, timeout=60
        )
        return result.returncode == 0
    except Exception as e:
        print(f"ERROR: Docker GPU check failed: {e}")
        return False


def pull_docker_image(image: str) -> bool:
    """Pull Docker image if not available locally."""
    result = subprocess.run(
        ['docker', 'images', '-q', image],
        capture_output=True, text=True, timeout=30
    )
    if result.stdout.strip():
        print(f"  Image {image} already available locally")
        return True

    print(f"  Pulling {image}...")
    result = subprocess.run(
        ['docker', 'pull', image],
        capture_output=True, text=True, timeout=1800
    )
    if result.returncode != 0:
        print(f"  ERROR: Failed to pull {image}: {result.stderr}")
        return False
    return True


def parse_serving_metrics(output: str) -> Dict[str, Any]:
    """Parse metrics from benchmark_serving.py output."""
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
    }
    for key, pattern in patterns.items():
        match = re.search(pattern, output)
        if match:
            metrics[key] = float(match.group(1))
    return metrics


def parse_prefix_caching_metrics(output: str) -> Dict[str, Any]:
    """Parse metrics from benchmark_prefix_caching.py output."""
    metrics = {}

    # Look for elapsed time patterns
    elapsed_match = re.search(r'elapsed[:\s]+([\d.]+)\s*(?:s|sec)', output, re.IGNORECASE)
    if elapsed_match:
        metrics['elapsed_time_s'] = float(elapsed_match.group(1))

    # Look for throughput
    throughput_match = re.search(r'throughput[:\s]+([\d.]+)', output, re.IGNORECASE)
    if throughput_match:
        metrics['throughput'] = float(throughput_match.group(1))

    # Look for tokens/s
    tokens_match = re.search(r'([\d.]+)\s*tokens?/s', output, re.IGNORECASE)
    if tokens_match:
        metrics['tokens_per_s'] = float(tokens_match.group(1))

    # Look for timing info
    for pattern_name, pattern in [
        ('time_s', r'time[:\s]+([\d.]+)\s*s'),
        ('latency_ms', r'latency[:\s]+([\d.]+)\s*ms'),
    ]:
        match = re.search(pattern, output, re.IGNORECASE)
        if match:
            metrics[pattern_name] = float(match.group(1))

    return metrics


def run_prefix_caching_benchmark(
    commit_short: str,
    config: Dict[str, Any],
    version: str,
    hf_token: str,
    timeout: int = 1800,
) -> BenchmarkResult:
    """Run prefix_caching benchmark (standalone, no server needed)."""
    start_time = time.time()

    if version == "baseline":
        # Check if pre-built baseline image exists
        if config.get('baseline_image'):
            return run_prebuilt_baseline_benchmark(commit_short, config, hf_token, timeout)
        else:
            return run_baseline_from_source(commit_short, config, hf_token, timeout)

    # Human version
    image = f"{HUMAN_IMAGE_PREFIX}:{config['full_hash']}"

    if not pull_docker_image(image):
        return BenchmarkResult(
            commit_hash=commit_short, status="error", benchmark_type="prefix_caching",
            model=config['model'], duration_s=time.time() - start_time, version=version,
            error=f"Failed to pull image {image}"
        )

    # Extract benchmark args
    perf_command = config['perf_command']
    bench_args = re.sub(r'python3?\s+benchmarks/benchmark_prefix_caching\.py\s*', '', perf_command)

    docker_cmd = f'''
set -e
COMMIT="{config['full_hash']}"

# Install uv for faster package management
pip install uv -q 2>/dev/null || true

# Clone vLLM repo at specific commit for benchmark scripts
cd /opt
git clone --depth 1 https://github.com/vllm-project/vllm.git vllm_bench 2>/dev/null || true
cd vllm_bench
git fetch --depth 1 origin $COMMIT 2>/dev/null || true
git checkout $COMMIT 2>/dev/null || git checkout -f HEAD

# Install benchmark deps
pip install aiohttp pandas datasets -q 2>/dev/null || true

# Run benchmark
cd /opt/vllm_bench/benchmarks
echo "=== Running prefix_caching benchmark ==="
python3 benchmark_prefix_caching.py {bench_args} 2>&1
echo "=== BENCHMARK_COMPLETE ==="
'''

    print(f"  Running {version} benchmark...")
    try:
        result = subprocess.run(
            [
                'docker', 'run', '--rm', '--gpus', 'all',
                '-e', f'HF_TOKEN={hf_token}',
                '-e', f'HUGGING_FACE_HUB_TOKEN={hf_token}',
                '-v', '/root/.cache/huggingface:/root/.cache/huggingface',
                '--shm-size=16g',
                '--entrypoint', 'bash',
                image, '-c', docker_cmd
            ],
            capture_output=True, text=True, timeout=timeout
        )

        duration = time.time() - start_time
        raw_output = result.stdout + result.stderr

        metrics = parse_prefix_caching_metrics(raw_output)

        if "BENCHMARK_COMPLETE" not in raw_output or result.returncode != 0:
            return BenchmarkResult(
                commit_hash=commit_short, status="error", benchmark_type="prefix_caching",
                model=config['model'], duration_s=duration, version=version,
                error=f"Exit code {result.returncode}", metrics=metrics,
                raw_output=raw_output[-8000:]
            )

        return BenchmarkResult(
            commit_hash=commit_short, status="success", benchmark_type="prefix_caching",
            model=config['model'], duration_s=duration, version=version,
            metrics=metrics, raw_output=raw_output[-8000:]
        )

    except subprocess.TimeoutExpired:
        return BenchmarkResult(
            commit_hash=commit_short, status="timeout", benchmark_type="prefix_caching",
            model=config['model'], duration_s=timeout, version=version,
            error=f"Timeout after {timeout}s"
        )
    except Exception as e:
        return BenchmarkResult(
            commit_hash=commit_short, status="error", benchmark_type="prefix_caching",
            model=config['model'], duration_s=time.time() - start_time, version=version,
            error=str(e)
        )


def run_prebuilt_baseline_benchmark(
    commit_short: str,
    config: Dict[str, Any],
    hf_token: str,
    timeout: int = 1800,
) -> BenchmarkResult:
    """Run benchmark using pre-built baseline image (no build needed!)."""
    start_time = time.time()

    image = config['baseline_image']
    model = config['model']
    benchmark_type = config['benchmark_type']
    perf_command = config['perf_command']
    parent_commit = config['parent_hash']

    print(f"  Using pre-built baseline image: {image}")

    if not pull_docker_image(image):
        return BenchmarkResult(
            commit_hash=commit_short, status="error", benchmark_type=benchmark_type,
            model=model, duration_s=time.time() - start_time, version="baseline",
            error=f"Failed to pull baseline image {image}"
        )

    if benchmark_type == "serving":
        bench_args = re.sub(r'python3?\s+benchmarks/benchmark_serving\.py\s*', '', perf_command)
        docker_cmd = f'''
set -e
MODEL="{model}"
COMMIT="{parent_commit}"

# Install benchmark deps
pip install aiohttp pandas datasets -q 2>/dev/null || true

# Clone vLLM repo at parent commit for benchmark scripts
cd /opt
git clone --depth 1 https://github.com/vllm-project/vllm.git vllm_bench 2>/dev/null || true
cd vllm_bench
git fetch --depth 1 origin $COMMIT 2>/dev/null || true
git checkout $COMMIT 2>/dev/null || git checkout -f HEAD

# Start vLLM server
echo "=== Starting vLLM server ==="
cd /
python3 -m vllm.entrypoints.openai.api_server \\
    --model $MODEL --port 8000 --max-model-len 16384 --disable-log-requests 2>&1 &
SERVER_PID=$!

# Wait for server
echo "Waiting for server..."
for i in $(seq 1 300); do
    if curl -s http://localhost:8000/v1/models 2>/dev/null | grep -q "model"; then
        echo "SERVER_READY after ${{i}}s"
        break
    fi
    if ! kill -0 $SERVER_PID 2>/dev/null; then
        echo "SERVER_CRASHED"
        exit 1
    fi
    sleep 1
done

if ! curl -s http://localhost:8000/v1/models > /dev/null 2>&1; then
    echo "SERVER_TIMEOUT"
    exit 1
fi

# Run benchmark
echo "=== Running serving benchmark ==="
cd /opt/vllm_bench/benchmarks
python3 benchmark_serving.py {bench_args} --port 8000 2>&1
echo "=== BENCHMARK_COMPLETE ==="
kill $SERVER_PID 2>/dev/null || true
'''
    else:  # prefix_caching
        bench_args = re.sub(r'python3?\s+benchmarks/benchmark_prefix_caching\.py\s*', '', perf_command)
        docker_cmd = f'''
set -e
COMMIT="{parent_commit}"

# Install benchmark deps
pip install aiohttp pandas datasets -q 2>/dev/null || true

# Clone vLLM repo at parent commit for benchmark scripts
cd /opt
git clone --depth 1 https://github.com/vllm-project/vllm.git vllm_bench 2>/dev/null || true
cd vllm_bench
git fetch --depth 1 origin $COMMIT 2>/dev/null || true
git checkout $COMMIT 2>/dev/null || git checkout -f HEAD

# Run benchmark
echo "=== Running prefix_caching benchmark ==="
cd /opt/vllm_bench/benchmarks
python3 benchmark_prefix_caching.py {bench_args} 2>&1
echo "=== BENCHMARK_COMPLETE ==="
'''

    try:
        result = subprocess.run(
            [
                'docker', 'run', '--rm', '--gpus', 'all',
                '-e', f'HF_TOKEN={hf_token}',
                '-e', f'HUGGING_FACE_HUB_TOKEN={hf_token}',
                '-v', '/root/.cache/huggingface:/root/.cache/huggingface',
                '--shm-size=16g',
                '--entrypoint', 'bash',
                image, '-c', docker_cmd
            ],
            capture_output=True, text=True, timeout=timeout
        )

        duration = time.time() - start_time
        raw_output = result.stdout + result.stderr

        if "SERVER_CRASHED" in raw_output:
            return BenchmarkResult(
                commit_hash=commit_short, status="error", benchmark_type=benchmark_type,
                model=model, duration_s=duration, version="baseline",
                error="Server crashed during startup", raw_output=raw_output[-8000:]
            )

        if "SERVER_TIMEOUT" in raw_output:
            return BenchmarkResult(
                commit_hash=commit_short, status="error", benchmark_type=benchmark_type,
                model=model, duration_s=duration, version="baseline",
                error="Server startup timeout", raw_output=raw_output[-8000:]
            )

        if benchmark_type == "serving":
            metrics = parse_serving_metrics(raw_output)
        else:
            metrics = parse_prefix_caching_metrics(raw_output)

        if "BENCHMARK_COMPLETE" not in raw_output:
            return BenchmarkResult(
                commit_hash=commit_short, status="error", benchmark_type=benchmark_type,
                model=model, duration_s=duration, version="baseline",
                error="Benchmark did not complete", metrics=metrics,
                raw_output=raw_output[-8000:]
            )

        if not metrics:
            return BenchmarkResult(
                commit_hash=commit_short, status="error", benchmark_type=benchmark_type,
                model=model, duration_s=duration, version="baseline",
                error="No metrics in output", raw_output=raw_output[-8000:]
            )

        return BenchmarkResult(
            commit_hash=commit_short, status="success", benchmark_type=benchmark_type,
            model=model, duration_s=duration, version="baseline",
            metrics=metrics, raw_output=raw_output[-8000:]
        )

    except subprocess.TimeoutExpired:
        return BenchmarkResult(
            commit_hash=commit_short, status="timeout", benchmark_type=benchmark_type,
            model=model, duration_s=timeout, version="baseline",
            error=f"Timeout after {timeout}s"
        )
    except Exception as e:
        return BenchmarkResult(
            commit_hash=commit_short, status="error", benchmark_type=benchmark_type,
            model=model, duration_s=time.time() - start_time, version="baseline",
            error=str(e)
        )


def run_serving_benchmark(
    commit_short: str,
    config: Dict[str, Any],
    version: str,
    hf_token: str,
    timeout: int = 1800,
) -> BenchmarkResult:
    """Run serving benchmark (starts server, then runs benchmark against it)."""
    start_time = time.time()

    if version == "baseline":
        # Check if pre-built baseline image exists
        if config.get('baseline_image'):
            return run_prebuilt_baseline_benchmark(commit_short, config, hf_token, timeout)
        else:
            return run_baseline_from_source(commit_short, config, hf_token, timeout)

    # Human version
    image = f"{HUMAN_IMAGE_PREFIX}:{config['full_hash']}"

    if not pull_docker_image(image):
        return BenchmarkResult(
            commit_hash=commit_short, status="error", benchmark_type="serving",
            model=config['model'], duration_s=time.time() - start_time, version=version,
            error=f"Failed to pull image {image}"
        )

    # Extract benchmark args
    perf_command = config['perf_command']
    bench_args = re.sub(r'python3?\s+benchmarks/benchmark_serving\.py\s*', '', perf_command)
    model = config['model']

    docker_cmd = f'''
set -e
COMMIT="{config['full_hash']}"
MODEL="{model}"

# Install uv for faster package management
pip install uv -q 2>/dev/null || true

# Install benchmark deps
pip install aiohttp pandas datasets -q 2>/dev/null || true

# Clone vLLM repo at specific commit for benchmark scripts
cd /opt
git clone --depth 1 https://github.com/vllm-project/vllm.git vllm_bench 2>/dev/null || true
cd vllm_bench
git fetch --depth 1 origin $COMMIT 2>/dev/null || true
git checkout $COMMIT 2>/dev/null || git checkout -f HEAD
cd /opt/vllm_bench/benchmarks

# Start vLLM server
echo "=== Starting vLLM server ==="
cd /
python3 -m vllm.entrypoints.openai.api_server \
    --model $MODEL --port 8000 --max-model-len 16384 --disable-log-requests 2>&1 &
SERVER_PID=$!

# Wait for server to be ready
echo "Waiting for server..."
for i in $(seq 1 300); do
    if curl -s http://localhost:8000/v1/models 2>/dev/null | grep -q "model"; then
        echo "SERVER_READY after ${{i}}s"
        break
    fi
    if ! kill -0 $SERVER_PID 2>/dev/null; then
        echo "SERVER_CRASHED"
        exit 1
    fi
    sleep 1
done

# Verify server
if ! curl -s http://localhost:8000/v1/models > /dev/null 2>&1; then
    echo "SERVER_TIMEOUT"
    exit 1
fi

# Run benchmark
echo "=== Running serving benchmark ==="
cd /opt/vllm_bench/benchmarks
python3 benchmark_serving.py {bench_args} --port 8000 2>&1
echo "=== BENCHMARK_COMPLETE ==="

kill $SERVER_PID 2>/dev/null || true
'''

    print(f"  Running {version} benchmark (with server)...")
    try:
        result = subprocess.run(
            [
                'docker', 'run', '--rm', '--gpus', 'all',
                '-e', f'HF_TOKEN={hf_token}',
                '-e', f'HUGGING_FACE_HUB_TOKEN={hf_token}',
                '-v', '/root/.cache/huggingface:/root/.cache/huggingface',
                '--shm-size=16g',
                '--entrypoint', 'bash',
                image, '-c', docker_cmd
            ],
            capture_output=True, text=True, timeout=timeout
        )

        duration = time.time() - start_time
        raw_output = result.stdout + result.stderr

        if "SERVER_CRASHED" in raw_output:
            return BenchmarkResult(
                commit_hash=commit_short, status="error", benchmark_type="serving",
                model=model, duration_s=duration, version=version,
                error="Server crashed during startup", raw_output=raw_output[-8000:]
            )

        if "SERVER_TIMEOUT" in raw_output:
            return BenchmarkResult(
                commit_hash=commit_short, status="error", benchmark_type="serving",
                model=model, duration_s=duration, version=version,
                error="Server startup timeout", raw_output=raw_output[-8000:]
            )

        metrics = parse_serving_metrics(raw_output)

        if not metrics:
            return BenchmarkResult(
                commit_hash=commit_short, status="error", benchmark_type="serving",
                model=model, duration_s=duration, version=version,
                error="No metrics in output", raw_output=raw_output[-8000:]
            )

        return BenchmarkResult(
            commit_hash=commit_short, status="success", benchmark_type="serving",
            model=model, duration_s=duration, version=version,
            metrics=metrics, raw_output=raw_output[-8000:]
        )

    except subprocess.TimeoutExpired:
        return BenchmarkResult(
            commit_hash=commit_short, status="timeout", benchmark_type="serving",
            model=config['model'], duration_s=timeout, version=version,
            error=f"Timeout after {timeout}s"
        )
    except Exception as e:
        return BenchmarkResult(
            commit_hash=commit_short, status="error", benchmark_type="serving",
            model=config['model'], duration_s=time.time() - start_time, version=version,
            error=str(e)
        )


def run_baseline_from_source(
    commit_short: str,
    config: Dict[str, Any],
    hf_token: str,
    timeout: int = 7200,  # 2 hours for build + benchmark
) -> BenchmarkResult:
    """Build baseline from source at parent commit and run benchmark."""
    start_time = time.time()

    # Use human image as base (has CUDA runtime, PyTorch, etc.)
    image = f"{HUMAN_IMAGE_PREFIX}:{config['full_hash']}"

    if not pull_docker_image(image):
        return BenchmarkResult(
            commit_hash=commit_short, status="error", benchmark_type=config['benchmark_type'],
            model=config['model'], duration_s=time.time() - start_time, version="baseline",
            error=f"Failed to pull base image {image}"
        )

    parent_commit = config['parent_hash']
    model = config['model']
    benchmark_type = config['benchmark_type']
    perf_command = config['perf_command']

    if benchmark_type == "serving":
        bench_args = re.sub(r'python3?\s+benchmarks/benchmark_serving\.py\s*', '', perf_command)
        benchmark_cmd = f'''
# Start vLLM server
echo "=== Starting vLLM server ==="
python3 -m vllm.entrypoints.openai.api_server \
    --model $MODEL --port 8000 --max-model-len 16384 --disable-log-requests 2>&1 &
SERVER_PID=$!

# Wait for server
echo "Waiting for server..."
for i in $(seq 1 300); do
    if curl -s http://localhost:8000/v1/models 2>/dev/null | grep -q "model"; then
        echo "SERVER_READY after ${{i}}s"
        break
    fi
    if ! kill -0 $SERVER_PID 2>/dev/null; then
        echo "SERVER_CRASHED"
        exit 1
    fi
    sleep 1
done

if ! curl -s http://localhost:8000/v1/models > /dev/null 2>&1; then
    echo "SERVER_TIMEOUT"
    exit 1
fi

# Run benchmark
echo "=== Running serving benchmark ==="
cd /opt/vllm_baseline/benchmarks
python3 benchmark_serving.py {bench_args} --port 8000 2>&1
echo "=== BENCHMARK_COMPLETE ==="
kill $SERVER_PID 2>/dev/null || true
'''
    else:  # prefix_caching
        bench_args = re.sub(r'python3?\s+benchmarks/benchmark_prefix_caching\.py\s*', '', perf_command)
        benchmark_cmd = f'''
# Run benchmark directly (no server needed)
echo "=== Running prefix_caching benchmark ==="
cd /opt/vllm_baseline/benchmarks
python3 benchmark_prefix_caching.py {bench_args} 2>&1
echo "=== BENCHMARK_COMPLETE ==="
'''

    docker_cmd = f'''
set -e
PARENT_COMMIT="{parent_commit}"
MODEL="{model}"

echo "=== BASELINE BUILD: Installing build tools ==="
apt-get update -qq
apt-get install -y -qq cuda-toolkit-12-4 git 2>/dev/null || apt-get install -y -qq git

echo "=== Cloning vLLM at parent commit $PARENT_COMMIT ==="
cd /opt
for attempt in 1 2 3; do
    if git clone --depth 1 https://github.com/vllm-project/vllm.git vllm_baseline 2>&1; then
        break
    fi
    echo "Git clone attempt $attempt failed, retrying..."
    rm -rf vllm_baseline 2>/dev/null
    sleep 5
done

cd vllm_baseline
git fetch --depth 1 origin $PARENT_COMMIT
git checkout $PARENT_COMMIT

# Install uv for faster package management
pip install uv -q 2>/dev/null || true

echo "=== Uninstalling human vLLM and building baseline from source ==="
pip uninstall vllm -y 2>/dev/null || true

# Build with H100 optimization only
export TORCH_CUDA_ARCH_LIST="9.0"
export MAX_JOBS=16
export NVCC_THREADS=2

# Install build dependencies
pip install setuptools wheel packaging ninja cmake -q 2>/dev/null || true

echo "=== Building vLLM from source (this takes 15-30 min) ==="
pip install -e . --no-build-isolation 2>&1

# Install benchmark deps
pip install aiohttp pandas datasets -q 2>/dev/null || true

echo "=== Verifying baseline vLLM installation ==="
python3 -c "import vllm; print(f'vLLM version: {{vllm.__version__}}')"

{benchmark_cmd}
'''

    print(f"  Building and running baseline from source (parent: {parent_commit[:8]})...")
    print(f"  This will take 15-30 minutes for vLLM build...")

    try:
        result = subprocess.run(
            [
                'docker', 'run', '--rm', '--gpus', 'all',
                '-e', f'HF_TOKEN={hf_token}',
                '-e', f'HUGGING_FACE_HUB_TOKEN={hf_token}',
                '-v', '/root/.cache/huggingface:/root/.cache/huggingface',
                '--shm-size=16g',
                '--entrypoint', 'bash',
                image, '-c', docker_cmd
            ],
            capture_output=True, text=True, timeout=timeout
        )

        duration = time.time() - start_time
        raw_output = result.stdout + result.stderr

        if "SERVER_CRASHED" in raw_output:
            return BenchmarkResult(
                commit_hash=commit_short, status="error", benchmark_type=benchmark_type,
                model=model, duration_s=duration, version="baseline",
                error="Server crashed during startup", raw_output=raw_output[-8000:]
            )

        if "SERVER_TIMEOUT" in raw_output:
            return BenchmarkResult(
                commit_hash=commit_short, status="error", benchmark_type=benchmark_type,
                model=model, duration_s=duration, version="baseline",
                error="Server startup timeout", raw_output=raw_output[-8000:]
            )

        if benchmark_type == "serving":
            metrics = parse_serving_metrics(raw_output)
        else:
            metrics = parse_prefix_caching_metrics(raw_output)

        if "BENCHMARK_COMPLETE" not in raw_output:
            return BenchmarkResult(
                commit_hash=commit_short, status="error", benchmark_type=benchmark_type,
                model=model, duration_s=duration, version="baseline",
                error="Benchmark did not complete", metrics=metrics,
                raw_output=raw_output[-8000:]
            )

        if not metrics:
            return BenchmarkResult(
                commit_hash=commit_short, status="error", benchmark_type=benchmark_type,
                model=model, duration_s=duration, version="baseline",
                error="No metrics in output", raw_output=raw_output[-8000:]
            )

        return BenchmarkResult(
            commit_hash=commit_short, status="success", benchmark_type=benchmark_type,
            model=model, duration_s=duration, version="baseline",
            metrics=metrics, raw_output=raw_output[-8000:]
        )

    except subprocess.TimeoutExpired:
        return BenchmarkResult(
            commit_hash=commit_short, status="timeout", benchmark_type=benchmark_type,
            model=model, duration_s=timeout, version="baseline",
            error=f"Timeout after {timeout}s"
        )
    except Exception as e:
        return BenchmarkResult(
            commit_hash=commit_short, status="error", benchmark_type=benchmark_type,
            model=model, duration_s=time.time() - start_time, version="baseline",
            error=str(e)
        )


def run_3way_benchmark(
    commit_short: str,
    config: Dict[str, Any],
    hf_token: str,
    dry_run: bool = False,
    human_only: bool = False
) -> Dict[str, BenchmarkResult]:
    """Run all versions: baseline and human (agent would need patch application)."""
    results = {}

    print(f"\n{'='*60}")
    print(f"Running benchmark for {commit_short}")
    print(f"Subject: {config['subject']}")
    print(f"Original issue: {config['original_issue']}")
    print(f"Benchmark type: {config['benchmark_type']}")
    print(f"{'='*60}")

    versions = ["human"] if human_only else ["baseline", "human"]

    if dry_run:
        for version in versions:
            print(f"\n[{version.upper()}]")
            print(f"  [DRY RUN] Would run: {config['perf_command']}")
            if version == "baseline":
                print(f"  [DRY RUN] Would build from source at parent: {config['parent_hash'][:8]}")
            else:
                print(f"  [DRY RUN] Image: {HUMAN_IMAGE_PREFIX}:{config['full_hash']}")
            results[version] = BenchmarkResult(
                commit_hash=commit_short, status="dry_run",
                benchmark_type=config['benchmark_type'], model=config['model'],
                duration_s=0, version=version
            )
        return results

    benchmark_type = config['benchmark_type']

    for version in versions:
        print(f"\n[{version.upper()}]")

        if benchmark_type == "serving":
            result = run_serving_benchmark(commit_short, config, version, hf_token)
        else:  # prefix_caching
            result = run_prefix_caching_benchmark(commit_short, config, version, hf_token)

        results[version] = result

        if result.status == "success":
            print(f"  Status: SUCCESS")
            print(f"  Duration: {result.duration_s:.1f}s")
            if result.metrics:
                print(f"  Metrics:")
                for k, v in result.metrics.items():
                    print(f"    {k}: {v}")
        else:
            print(f"  Status: {result.status.upper()}")
            if result.error:
                print(f"  Error: {result.error}")

    return results


def save_results(commit_short: str, results: Dict[str, BenchmarkResult]):
    """Save benchmark results to JSON file."""
    output_dir = RESULTS_DIR / commit_short
    output_dir.mkdir(parents=True, exist_ok=True)

    output_file = output_dir / "benchmark_result.json"

    data = {
        "commit": commit_short,
        "timestamp": datetime.now().isoformat(),
        "results": {k: asdict(v) for k, v in results.items()}
    }

    with open(output_file, 'w') as f:
        json.dump(data, f, indent=2)

    print(f"\nResults saved to: {output_file}")


def main():
    parser = argparse.ArgumentParser(description="Rerun specific commits with fixed perf_commands")
    parser.add_argument('--dry-run', action='store_true', help="Don't actually run benchmarks")
    parser.add_argument('--commit', type=str, help="Run only this commit (e.g., 2deb029d)")
    parser.add_argument('--skip-gpu-check', action='store_true', help="Skip Docker GPU check")
    parser.add_argument('--human-only', action='store_true', help="Skip baseline (faster)")
    args = parser.parse_args()

    # Check Docker GPU access
    if not args.skip_gpu_check and not args.dry_run:
        print("Checking Docker GPU access...")
        if not check_docker_gpu():
            print("ERROR: Docker cannot access GPU. Run with --skip-gpu-check to bypass.")
            sys.exit(1)
        print("Docker GPU access OK\n")

    # Get HuggingFace token
    hf_token = get_hf_token()
    if not hf_token and not args.dry_run:
        print("WARNING: No HuggingFace token found. Gated models may fail.")

    # Determine which commits to run
    if args.commit:
        if args.commit not in COMMITS_TO_RERUN:
            print(f"ERROR: Unknown commit {args.commit}")
            print(f"Available: {list(COMMITS_TO_RERUN.keys())}")
            sys.exit(1)
        commits_to_run = {args.commit: COMMITS_TO_RERUN[args.commit]}
    else:
        commits_to_run = COMMITS_TO_RERUN

    # Run benchmarks
    all_results = {}
    for commit_short, config in commits_to_run.items():
        results = run_3way_benchmark(commit_short, config, hf_token,
                                     dry_run=args.dry_run, human_only=args.human_only)
        all_results[commit_short] = results

        if not args.dry_run:
            save_results(commit_short, results)

    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    for commit, results in all_results.items():
        print(f"\n{commit}:")
        for version, result in results.items():
            status_icon = "✓" if result.status == "success" else "✗"
            print(f"  {status_icon} {version}: {result.status}")
            if result.metrics:
                for k, v in result.metrics.items():
                    print(f"      {k}: {v}")


if __name__ == "__main__":
    main()
