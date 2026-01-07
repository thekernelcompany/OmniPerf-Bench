#!/usr/bin/env python3
"""
Local Docker Benchmark Runner for vLLM commits.

Runs benchmarks inside pre-built Docker containers for commits that failed on Modal.
Supports serving, throughput, and latency benchmark types.

Benchmark scripts are obtained by cloning the vLLM repo at the specific commit
(approach #2) for better reproducibility compared to raw GitHub URL downloads.

BASELINE MODE (--baseline):
Runs baseline benchmarks by building vLLM from source at the parent commit.
Uses the human Docker image as base (has CUDA runtime, PyTorch, etc.),
installs CUDA toolkit, clones vLLM at parent commit, and builds from source.
This enables apples-to-apples comparison between baseline and human optimizations.
"""

import json
import os
import re
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Dict, Any, List

# Configuration
DOCKER_IMAGE_PREFIX = "ayushnangia16/nvidia-vllm-docker"
FIXED_IMAGE_PREFIX = "shikhar481/vllm_fixed_human_images"
RESULTS_DIR = Path("/root/OmniPerf-Bench/omniperf_results_3way_claude_code")
FULL_RESULTS_FILE = RESULTS_DIR / "full_results.jsonl"
OUTPUT_DIR = RESULTS_DIR / "docker_benchmark_results"
BASELINE_OUTPUT_DIR = RESULTS_DIR / "baseline_benchmark_results"
BASELINE_MAPPING_FILE = Path("/root/OmniPerf-Bench/baseline_benchmark_mapping.json")

# Commits that need fixed images (from previous analysis)
FIXED_IMAGE_COMMITS = {
    "015069b0", "22dd9c27", "67da5720", "d55e446d", "e493e485",  # aimv2 fix
    "35fad35a", "3092375e", "93e5f3c5", "9d72daf4", "b10e5198",  # V1 engine fix
    "ad8d696a", "b6d10354",  # NumPy fix
}


@dataclass
class BenchmarkResult:
    """Benchmark result data."""
    commit_hash: str
    status: str  # success, error, timeout
    benchmark_type: str  # serving, throughput, latency
    model: str
    duration_s: float
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
    throughput_req_s: Optional[float] = None
    throughput_tok_s: Optional[float] = None
    raw_output: Optional[str] = None


def get_hf_token() -> str:
    """Get HuggingFace token."""
    try:
        result = subprocess.run(
            ["python3", "-c", "from huggingface_hub import get_token; print(get_token())"],
            capture_output=True, text=True, timeout=10
        )
        return result.stdout.strip()
    except Exception:
        return ""


def get_docker_image(commit_short: str, commit_full: str) -> str:
    """Get the appropriate Docker image for a commit."""
    if commit_short in FIXED_IMAGE_COMMITS:
        return f"{FIXED_IMAGE_PREFIX}:{commit_full}"
    return f"{DOCKER_IMAGE_PREFIX}:{commit_full}"


def load_baseline_mapping() -> List[Dict[str, Any]]:
    """Load baseline benchmark mapping (human commit -> parent commit)."""
    if not BASELINE_MAPPING_FILE.exists():
        print(f"ERROR: Baseline mapping file not found: {BASELINE_MAPPING_FILE}")
        print("Run the mapping generator first or create the file.")
        return []

    with open(BASELINE_MAPPING_FILE) as f:
        return json.load(f)


def load_commits_to_run() -> List[Dict[str, Any]]:
    """Load commits that need to be re-run from full_results.jsonl."""
    commits = []
    commits_file = RESULTS_DIR / "commits_to_rerun.txt"

    # Load commit hashes to rerun
    rerun_hashes = set()
    if commits_file.exists():
        with open(commits_file) as f:
            for line in f:
                h = line.strip()
                if h:
                    rerun_hashes.add(h)

    # Load full results and filter
    with open(FULL_RESULTS_FILE) as f:
        for line in f:
            r = json.loads(line)
            short_hash = r['commit_hash'][:8]
            if short_hash in rerun_hashes:
                commits.append(r)

    return commits


def get_benchmark_type(perf_command: str) -> str:
    """Determine benchmark type from perf_command."""
    if 'benchmark_serving' in perf_command:
        return 'serving'
    elif 'benchmark_throughput' in perf_command:
        return 'throughput'
    elif 'benchmark_latency' in perf_command:
        return 'latency'
    elif 'vllm bench serve' in perf_command:
        return 'serving'
    elif 'vllm bench throughput' in perf_command:
        return 'throughput'
    elif 'vllm bench latency' in perf_command:
        return 'latency'
    return 'unknown'


def parse_serving_metrics(output: str) -> Dict[str, float]:
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
        'throughput_req_s': r'Request throughput \(req/s\):\s+([\d.]+)',
        'throughput_tok_s': r'Output token throughput \(tok/s\):\s+([\d.]+)',
    }

    for key, pattern in patterns.items():
        match = re.search(pattern, output)
        if match:
            metrics[key] = float(match.group(1))

    return metrics


def run_serving_benchmark(commit_hash: str, model: str, perf_command: str,
                          hf_token: str, timeout: int = 600) -> BenchmarkResult:
    """Run a serving benchmark inside Docker."""
    start_time = time.time()

    # Adjust perf_command to use random dataset (sharegpt requires local file)
    # Replace sharegpt/sonnet with random dataset
    perf_command = re.sub(r'--dataset-name\s+sharegpt', '--dataset-name random', perf_command)
    perf_command = re.sub(r'--dataset-name\s+sonnet', '--dataset-name random', perf_command)
    perf_command = re.sub(r'--dataset\s+\S+\.json', '', perf_command)  # Remove file paths

    if '--dataset-name' not in perf_command and '--dataset-path' not in perf_command:
        perf_command += ' --dataset-name random'

    # Ensure random dataset params are present
    if '--dataset-name random' in perf_command and '--random-input-len' not in perf_command:
        perf_command += ' --random-input-len 256 --random-output-len 64'

    # Extract just the benchmark args (after benchmark_serving.py)
    bench_args = re.sub(r'python\s+benchmarks/benchmark_serving\.py\s*', '', perf_command)

    docker_cmd = f'''
    set -e
    COMMIT="{commit_hash}"
    MODEL="{model}"

    # Install deps
    pip install aiohttp pandas datasets -q

    # Clone vLLM repo at specific commit for benchmark scripts (approach #2 - more reproducible)
    cd /opt
    git clone --depth 1 https://github.com/vllm-project/vllm.git vllm_bench 2>/dev/null || true
    cd vllm_bench
    git fetch --depth 1 origin $COMMIT 2>/dev/null || true
    git checkout $COMMIT 2>/dev/null || git checkout -f HEAD
    cd /opt/vllm_bench/benchmarks

    # Start server
    cd /
    python3 -m vllm.entrypoints.openai.api_server \
        --model $MODEL --port 8000 --max-model-len 4096 --disable-log-requests 2>&1 &
    SERVER_PID=$!

    # Wait for server
    for i in $(seq 1 180); do
        if curl -s http://localhost:8000/v1/models 2>/dev/null | grep -q "model"; then
            echo "SERVER_READY_AFTER=${{i}}s"
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
    cd /opt/vllm_bench/benchmarks
    python3 benchmark_serving.py {bench_args} --port 8000 2>&1

    kill $SERVER_PID 2>/dev/null || true
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
                f'{DOCKER_IMAGE_PREFIX}:{commit_hash}',
                '-c', docker_cmd
            ],
            capture_output=True, text=True, timeout=timeout
        )

        output = result.stdout + result.stderr
        duration = time.time() - start_time

        if 'SERVER_CRASHED' in output:
            return BenchmarkResult(
                commit_hash=commit_hash, status='error', benchmark_type='serving',
                model=model, duration_s=duration, error='Server crashed during startup',
                raw_output=output[-5000:]
            )

        if 'SERVER_TIMEOUT' in output:
            return BenchmarkResult(
                commit_hash=commit_hash, status='error', benchmark_type='serving',
                model=model, duration_s=duration, error='Server startup timeout',
                raw_output=output[-5000:]
            )

        # Parse metrics
        metrics = parse_serving_metrics(output)

        if not metrics:
            return BenchmarkResult(
                commit_hash=commit_hash, status='error', benchmark_type='serving',
                model=model, duration_s=duration, error='No metrics in output',
                raw_output=output[-5000:]
            )

        return BenchmarkResult(
            commit_hash=commit_hash, status='success', benchmark_type='serving',
            model=model, duration_s=duration, raw_output=output[-5000:],
            **metrics
        )

    except subprocess.TimeoutExpired:
        return BenchmarkResult(
            commit_hash=commit_hash, status='timeout', benchmark_type='serving',
            model=model, duration_s=timeout, error=f'Benchmark timed out after {timeout}s'
        )
    except Exception as e:
        return BenchmarkResult(
            commit_hash=commit_hash, status='error', benchmark_type='serving',
            model=model, duration_s=time.time() - start_time, error=str(e)
        )


def run_throughput_benchmark(commit_hash: str, model: str, perf_command: str,
                             hf_token: str, timeout: int = 600) -> BenchmarkResult:
    """Run a throughput benchmark inside Docker (no server needed)."""
    start_time = time.time()

    # Extract benchmark args
    bench_args = re.sub(r'python\s+benchmarks/benchmark_throughput\.py\s*', '', perf_command)

    docker_cmd = f'''
    set -e
    COMMIT="{commit_hash}"

    # Clone vLLM repo at specific commit for benchmark scripts (approach #2 - more reproducible)
    cd /opt
    git clone --depth 1 https://github.com/vllm-project/vllm.git vllm_bench 2>/dev/null || true
    cd vllm_bench
    git fetch --depth 1 origin $COMMIT 2>/dev/null || true
    git checkout $COMMIT 2>/dev/null || git checkout -f HEAD

    # Run benchmark directly (no server needed)
    cd /opt/vllm_bench/benchmarks
    python3 benchmark_throughput.py {bench_args} 2>&1
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
                f'{DOCKER_IMAGE_PREFIX}:{commit_hash}',
                '-c', docker_cmd
            ],
            capture_output=True, text=True, timeout=timeout
        )

        output = result.stdout + result.stderr
        duration = time.time() - start_time

        # Parse throughput metrics
        throughput_match = re.search(r'Throughput:\s+([\d.]+)\s+requests/s', output)
        tok_throughput_match = re.search(r'([\d.]+)\s+tokens/s', output)

        return BenchmarkResult(
            commit_hash=commit_hash, status='success' if throughput_match else 'error',
            benchmark_type='throughput', model=model, duration_s=duration,
            throughput_req_s=float(throughput_match.group(1)) if throughput_match else None,
            throughput_tok_s=float(tok_throughput_match.group(1)) if tok_throughput_match else None,
            raw_output=output[-5000:],
            error=None if throughput_match else 'No throughput metrics in output'
        )

    except subprocess.TimeoutExpired:
        return BenchmarkResult(
            commit_hash=commit_hash, status='timeout', benchmark_type='throughput',
            model=model, duration_s=timeout, error=f'Benchmark timed out after {timeout}s'
        )
    except Exception as e:
        return BenchmarkResult(
            commit_hash=commit_hash, status='error', benchmark_type='throughput',
            model=model, duration_s=time.time() - start_time, error=str(e)
        )


def run_latency_benchmark(commit_hash: str, model: str, perf_command: str,
                          hf_token: str, timeout: int = 600) -> BenchmarkResult:
    """Run a latency benchmark inside Docker (no server needed)."""
    start_time = time.time()

    # Extract benchmark args
    bench_args = re.sub(r'python\s+benchmarks/benchmark_latency\.py\s*', '', perf_command)

    docker_cmd = f'''
    set -e
    COMMIT="{commit_hash}"

    # Clone vLLM repo at specific commit for benchmark scripts (approach #2 - more reproducible)
    cd /opt
    git clone --depth 1 https://github.com/vllm-project/vllm.git vllm_bench 2>/dev/null || true
    cd vllm_bench
    git fetch --depth 1 origin $COMMIT 2>/dev/null || true
    git checkout $COMMIT 2>/dev/null || git checkout -f HEAD

    # Run benchmark directly
    cd /opt/vllm_bench/benchmarks
    python3 benchmark_latency.py {bench_args} 2>&1
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
                f'{DOCKER_IMAGE_PREFIX}:{commit_hash}',
                '-c', docker_cmd
            ],
            capture_output=True, text=True, timeout=timeout
        )

        output = result.stdout + result.stderr
        duration = time.time() - start_time

        # Parse latency metrics
        latency_match = re.search(r'Avg latency:\s+([\d.]+)\s*(?:ms|seconds)', output)

        return BenchmarkResult(
            commit_hash=commit_hash, status='success' if latency_match else 'error',
            benchmark_type='latency', model=model, duration_s=duration,
            raw_output=output[-5000:],
            error=None if latency_match else 'No latency metrics in output'
        )

    except subprocess.TimeoutExpired:
        return BenchmarkResult(
            commit_hash=commit_hash, status='timeout', benchmark_type='latency',
            model=model, duration_s=timeout, error=f'Benchmark timed out after {timeout}s'
        )
    except Exception as e:
        return BenchmarkResult(
            commit_hash=commit_hash, status='error', benchmark_type='latency',
            model=model, duration_s=time.time() - start_time, error=str(e)
        )


def save_result(result: BenchmarkResult, output_dir: Path = None):
    """Save benchmark result to file."""
    if output_dir is None:
        output_dir = OUTPUT_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    output_file = output_dir / f"{result.commit_hash[:8]}_result.json"

    data = {
        'commit_hash': result.commit_hash,
        'status': result.status,
        'benchmark_type': result.benchmark_type,
        'model': result.model,
        'duration_s': result.duration_s,
        'error': result.error,
        'ttft_mean': result.ttft_mean,
        'ttft_median': result.ttft_median,
        'ttft_p99': result.ttft_p99,
        'tpot_mean': result.tpot_mean,
        'tpot_median': result.tpot_median,
        'tpot_p99': result.tpot_p99,
        'itl_mean': result.itl_mean,
        'itl_median': result.itl_median,
        'itl_p99': result.itl_p99,
        'throughput_req_s': result.throughput_req_s,
        'throughput_tok_s': result.throughput_tok_s,
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
    }

    with open(output_file, 'w') as f:
        json.dump(data, f, indent=2)

    print(f"Saved result to {output_file}")


def run_baseline_serving_benchmark(human_commit: str, parent_commit: str, model: str,
                                   perf_command: str, hf_token: str, timeout: int = 3600) -> BenchmarkResult:
    """Run a baseline serving benchmark by building vLLM from source at parent commit.

    Uses the human Docker image as base (has CUDA runtime, PyTorch, FlashAttn),
    installs CUDA toolkit for compilation, clones vLLM at parent commit,
    uninstalls human vLLM and builds from source.
    """
    start_time = time.time()

    # Get the appropriate Docker image for the human commit
    human_short = human_commit[:8]
    docker_image = get_docker_image(human_short, human_commit)

    # Adjust perf_command for random dataset
    perf_command = re.sub(r'--dataset-name\s+sharegpt', '--dataset-name random', perf_command)
    perf_command = re.sub(r'--dataset-name\s+sonnet', '--dataset-name random', perf_command)
    perf_command = re.sub(r'--dataset\s+\S+\.json', '', perf_command)

    if '--dataset-name' not in perf_command and '--dataset-path' not in perf_command:
        perf_command += ' --dataset-name random'

    if '--dataset-name random' in perf_command and '--random-input-len' not in perf_command:
        perf_command += ' --random-input-len 256 --random-output-len 64'

    # Extract benchmark args
    bench_args = re.sub(r'python\s+benchmarks/benchmark_serving\.py\s*', '', perf_command)

    docker_cmd = f'''
    set -e
    PARENT_COMMIT="{parent_commit}"
    MODEL="{model}"

    echo "=== BASELINE BUILD: Installing CUDA toolkit ==="
    apt-get update -qq
    apt-get install -y -qq cuda-toolkit-12-4

    echo "=== Cloning vLLM at parent commit $PARENT_COMMIT ==="
    cd /opt
    git clone https://github.com/vllm-project/vllm.git vllm_baseline
    cd vllm_baseline
    git checkout $PARENT_COMMIT

    echo "=== Uninstalling human vLLM and building baseline from source ==="
    pip uninstall vllm -y

    # Build with H100 optimization only (SM 9.0)
    export TORCH_CUDA_ARCH_LIST="9.0"
    export MAX_JOBS=8
    pip install -e . --no-build-isolation 2>&1 | tail -20

    echo "=== Verifying baseline vLLM installation ==="
    python3 -c "import vllm; print(f'vLLM version: {{vllm.__version__}}')"

    # Install benchmark deps
    pip install aiohttp pandas datasets -q

    echo "=== Starting vLLM server ==="
    python3 -m vllm.entrypoints.openai.api_server \\
        --model $MODEL --port 8000 --max-model-len 4096 --disable-log-requests 2>&1 &
    SERVER_PID=$!

    # Wait for server
    for i in $(seq 1 300); do
        if curl -s http://localhost:8000/v1/models 2>/dev/null | grep -q "model"; then
            echo "SERVER_READY_AFTER=${{i}}s"
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

    echo "=== Running benchmark ==="
    cd /opt/vllm_baseline/benchmarks
    python3 benchmark_serving.py {bench_args} --port 8000 2>&1

    kill $SERVER_PID 2>/dev/null || true
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
                docker_image,
                '-c', docker_cmd
            ],
            capture_output=True, text=True, timeout=timeout
        )

        output = result.stdout + result.stderr
        duration = time.time() - start_time

        if 'SERVER_CRASHED' in output:
            return BenchmarkResult(
                commit_hash=parent_commit, status='error', benchmark_type='serving',
                model=model, duration_s=duration, error='Server crashed during startup',
                raw_output=output[-10000:]
            )

        if 'SERVER_TIMEOUT' in output:
            return BenchmarkResult(
                commit_hash=parent_commit, status='error', benchmark_type='serving',
                model=model, duration_s=duration, error='Server startup timeout',
                raw_output=output[-10000:]
            )

        # Parse metrics
        metrics = parse_serving_metrics(output)

        if not metrics:
            return BenchmarkResult(
                commit_hash=parent_commit, status='error', benchmark_type='serving',
                model=model, duration_s=duration, error='No metrics in output',
                raw_output=output[-10000:]
            )

        return BenchmarkResult(
            commit_hash=parent_commit, status='success', benchmark_type='serving',
            model=model, duration_s=duration, raw_output=output[-10000:],
            **metrics
        )

    except subprocess.TimeoutExpired:
        return BenchmarkResult(
            commit_hash=parent_commit, status='timeout', benchmark_type='serving',
            model=model, duration_s=timeout, error=f'Benchmark timed out after {timeout}s'
        )
    except Exception as e:
        return BenchmarkResult(
            commit_hash=parent_commit, status='error', benchmark_type='serving',
            model=model, duration_s=time.time() - start_time, error=str(e)
        )


def run_baseline_throughput_benchmark(human_commit: str, parent_commit: str, model: str,
                                      perf_command: str, hf_token: str, timeout: int = 3600) -> BenchmarkResult:
    """Run a baseline throughput benchmark by building vLLM from source at parent commit."""
    start_time = time.time()

    human_short = human_commit[:8]
    docker_image = get_docker_image(human_short, human_commit)

    # Extract benchmark args
    bench_args = re.sub(r'python\s+benchmarks/benchmark_throughput\.py\s*', '', perf_command)
    # Also handle vllm bench throughput format
    bench_args = re.sub(r'vllm\s+bench\s+throughput\s*', '', bench_args)

    docker_cmd = f'''
    set -e
    PARENT_COMMIT="{parent_commit}"

    echo "=== BASELINE BUILD: Installing CUDA toolkit ==="
    apt-get update -qq
    apt-get install -y -qq cuda-toolkit-12-4

    echo "=== Cloning vLLM at parent commit $PARENT_COMMIT ==="
    cd /opt
    git clone https://github.com/vllm-project/vllm.git vllm_baseline
    cd vllm_baseline
    git checkout $PARENT_COMMIT

    echo "=== Uninstalling human vLLM and building baseline from source ==="
    pip uninstall vllm -y

    export TORCH_CUDA_ARCH_LIST="9.0"
    export MAX_JOBS=8
    pip install -e . --no-build-isolation 2>&1 | tail -20

    echo "=== Verifying baseline vLLM installation ==="
    python3 -c "import vllm; print(f'vLLM version: {{vllm.__version__}}')"

    echo "=== Running throughput benchmark ==="
    cd /opt/vllm_baseline/benchmarks
    python3 benchmark_throughput.py {bench_args} 2>&1
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
                docker_image,
                '-c', docker_cmd
            ],
            capture_output=True, text=True, timeout=timeout
        )

        output = result.stdout + result.stderr
        duration = time.time() - start_time

        # Parse throughput metrics
        throughput_match = re.search(r'Throughput:\s+([\d.]+)\s+requests/s', output)
        tok_throughput_match = re.search(r'([\d.]+)\s+tokens/s', output)

        return BenchmarkResult(
            commit_hash=parent_commit, status='success' if throughput_match else 'error',
            benchmark_type='throughput', model=model, duration_s=duration,
            throughput_req_s=float(throughput_match.group(1)) if throughput_match else None,
            throughput_tok_s=float(tok_throughput_match.group(1)) if tok_throughput_match else None,
            raw_output=output[-10000:],
            error=None if throughput_match else 'No throughput metrics in output'
        )

    except subprocess.TimeoutExpired:
        return BenchmarkResult(
            commit_hash=parent_commit, status='timeout', benchmark_type='throughput',
            model=model, duration_s=timeout, error=f'Benchmark timed out after {timeout}s'
        )
    except Exception as e:
        return BenchmarkResult(
            commit_hash=parent_commit, status='error', benchmark_type='throughput',
            model=model, duration_s=time.time() - start_time, error=str(e)
        )


def run_baseline_latency_benchmark(human_commit: str, parent_commit: str, model: str,
                                   perf_command: str, hf_token: str, timeout: int = 3600) -> BenchmarkResult:
    """Run a baseline latency benchmark by building vLLM from source at parent commit."""
    start_time = time.time()

    human_short = human_commit[:8]
    docker_image = get_docker_image(human_short, human_commit)

    bench_args = re.sub(r'python\s+benchmarks/benchmark_latency\.py\s*', '', perf_command)

    docker_cmd = f'''
    set -e
    PARENT_COMMIT="{parent_commit}"

    echo "=== BASELINE BUILD: Installing CUDA toolkit ==="
    apt-get update -qq
    apt-get install -y -qq cuda-toolkit-12-4

    echo "=== Cloning vLLM at parent commit $PARENT_COMMIT ==="
    cd /opt
    git clone https://github.com/vllm-project/vllm.git vllm_baseline
    cd vllm_baseline
    git checkout $PARENT_COMMIT

    echo "=== Uninstalling human vLLM and building baseline from source ==="
    pip uninstall vllm -y

    export TORCH_CUDA_ARCH_LIST="9.0"
    export MAX_JOBS=8
    pip install -e . --no-build-isolation 2>&1 | tail -20

    echo "=== Verifying baseline vLLM installation ==="
    python3 -c "import vllm; print(f'vLLM version: {{vllm.__version__}}')"

    echo "=== Running latency benchmark ==="
    cd /opt/vllm_baseline/benchmarks
    python3 benchmark_latency.py {bench_args} 2>&1
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
                docker_image,
                '-c', docker_cmd
            ],
            capture_output=True, text=True, timeout=timeout
        )

        output = result.stdout + result.stderr
        duration = time.time() - start_time

        latency_match = re.search(r'Avg latency:\s+([\d.]+)\s*(?:ms|seconds)', output)

        return BenchmarkResult(
            commit_hash=parent_commit, status='success' if latency_match else 'error',
            benchmark_type='latency', model=model, duration_s=duration,
            raw_output=output[-10000:],
            error=None if latency_match else 'No latency metrics in output'
        )

    except subprocess.TimeoutExpired:
        return BenchmarkResult(
            commit_hash=parent_commit, status='timeout', benchmark_type='latency',
            model=model, duration_s=timeout, error=f'Benchmark timed out after {timeout}s'
        )
    except Exception as e:
        return BenchmarkResult(
            commit_hash=parent_commit, status='error', benchmark_type='latency',
            model=model, duration_s=time.time() - start_time, error=str(e)
        )


def run_baseline_benchmarks(args):
    """Run baseline benchmarks for commits in baseline_benchmark_mapping.json."""
    print("Loading baseline mapping...")
    mapping = load_baseline_mapping()
    if not mapping:
        return

    print(f"Found {len(mapping)} commits in baseline mapping")

    # Filter by commit if specified
    if args.commit:
        mapping = [m for m in mapping if m['human_commit_short'].startswith(args.commit)]
        print(f"Filtered to {len(mapping)} commits matching {args.commit}")

    # Filter by type if specified
    if args.type:
        mapping = [m for m in mapping if m.get('benchmark_type') == args.type]
        print(f"Filtered to {len(mapping)} {args.type} benchmarks")

    # Apply limit
    if args.limit > 0:
        mapping = mapping[:args.limit]
        print(f"Limited to {len(mapping)} commits")

    if args.dry_run:
        print("\nDry run - would run baseline benchmarks for:")
        for m in mapping:
            print(f"  {m['human_commit_short']} -> parent {m['parent_commit'][:8]}")
            print(f"    Model: {m.get('model', 'N/A')}")
            print(f"    Type: {m.get('benchmark_type', 'unknown')}")
        return

    # Get HF token
    hf_token = get_hf_token()
    if not hf_token:
        print("WARNING: No HuggingFace token found. Gated models will fail.")

    # Run benchmarks
    results = []
    for i, m in enumerate(mapping):
        human_commit = m['human_commit_full']
        parent_commit = m['parent_commit']
        model = m.get('model', '')
        perf_command = m.get('perf_command', '')
        btype = m.get('benchmark_type', 'unknown')

        # Detect benchmark type from command if not set
        if btype == 'unknown':
            btype = get_benchmark_type(perf_command)

        print(f"\n[{i+1}/{len(mapping)}] Running BASELINE for {m['human_commit_short']}")
        print(f"  Parent commit: {parent_commit[:8]}")
        print(f"  Model: {model}")
        print(f"  Type: {btype}")

        # Check if already run
        result_file = BASELINE_OUTPUT_DIR / f"{m['human_commit_short']}_baseline_result.json"
        if result_file.exists():
            print(f"  SKIP: Already have baseline result")
            continue

        # Run appropriate benchmark
        if btype == 'serving':
            result = run_baseline_serving_benchmark(human_commit, parent_commit, model, perf_command, hf_token)
        elif btype == 'throughput':
            result = run_baseline_throughput_benchmark(human_commit, parent_commit, model, perf_command, hf_token)
        elif btype == 'latency':
            result = run_baseline_latency_benchmark(human_commit, parent_commit, model, perf_command, hf_token)
        else:
            print(f"  SKIP: Unknown benchmark type: {btype}")
            continue

        # Save with human commit prefix for easy mapping
        BASELINE_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        result_data = {
            'human_commit': m['human_commit_short'],
            'human_commit_full': human_commit,
            'parent_commit': parent_commit,
            'status': result.status,
            'benchmark_type': result.benchmark_type,
            'model': result.model,
            'duration_s': result.duration_s,
            'error': result.error,
            'ttft_mean': result.ttft_mean,
            'ttft_median': result.ttft_median,
            'ttft_p99': result.ttft_p99,
            'tpot_mean': result.tpot_mean,
            'tpot_median': result.tpot_median,
            'tpot_p99': result.tpot_p99,
            'itl_mean': result.itl_mean,
            'itl_median': result.itl_median,
            'itl_p99': result.itl_p99,
            'throughput_req_s': result.throughput_req_s,
            'throughput_tok_s': result.throughput_tok_s,
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        }
        with open(result_file, 'w') as f:
            json.dump(result_data, f, indent=2)
        print(f"  Saved to {result_file}")

        results.append(result)

        if result.status == 'success':
            print(f"  SUCCESS: {result.duration_s:.1f}s")
            if result.throughput_tok_s:
                print(f"    Throughput: {result.throughput_tok_s:.2f} tok/s")
            if result.ttft_mean:
                print(f"    TTFT: {result.ttft_mean:.2f}ms")
        else:
            print(f"  {result.status.upper()}: {result.error}")

    # Summary
    print("\n" + "="*50)
    print("BASELINE BENCHMARK SUMMARY")
    print("="*50)
    success = sum(1 for r in results if r.status == 'success')
    errors = sum(1 for r in results if r.status == 'error')
    timeouts = sum(1 for r in results if r.status == 'timeout')
    print(f"Success: {success}/{len(results)}")
    print(f"Errors: {errors}/{len(results)}")
    print(f"Timeouts: {timeouts}/{len(results)}")


def main():
    """Main entry point."""
    import argparse
    parser = argparse.ArgumentParser(description='Run Docker-based vLLM benchmarks')
    parser.add_argument('--commit', type=str, help='Run specific commit (short hash)')
    parser.add_argument('--type', type=str, choices=['serving', 'throughput', 'latency'],
                        help='Only run benchmarks of this type')
    parser.add_argument('--limit', type=int, default=0, help='Limit number of commits to run')
    parser.add_argument('--dry-run', action='store_true', help='Show what would be run')
    parser.add_argument('--baseline', action='store_true',
                        help='Run baseline benchmarks (build vLLM from source at parent commit)')
    args = parser.parse_args()

    # If baseline mode, run baseline benchmarks instead
    if args.baseline:
        run_baseline_benchmarks(args)
        return

    print("Loading commits...")
    commits = load_commits_to_run()
    print(f"Found {len(commits)} commits to run")

    # Filter by commit if specified
    if args.commit:
        commits = [c for c in commits if c['commit_hash'].startswith(args.commit)]
        print(f"Filtered to {len(commits)} commits matching {args.commit}")

    # Filter by type if specified
    if args.type:
        commits = [c for c in commits if get_benchmark_type(c.get('perf_command', '')) == args.type]
        print(f"Filtered to {len(commits)} {args.type} benchmarks")

    # Apply limit
    if args.limit > 0:
        commits = commits[:args.limit]
        print(f"Limited to {len(commits)} commits")

    if args.dry_run:
        print("\nDry run - would run:")
        for c in commits:
            btype = get_benchmark_type(c.get('perf_command', ''))
            print(f"  {c['commit_hash'][:8]}: {btype} - {c.get('model', 'N/A')}")
        return

    # Get HF token
    hf_token = get_hf_token()
    if not hf_token:
        print("WARNING: No HuggingFace token found. Gated models will fail.")

    # Run benchmarks
    results = []
    for i, commit in enumerate(commits):
        commit_hash = commit['commit_hash']
        model = commit.get('model', '')
        perf_command = commit.get('perf_command', '')
        btype = get_benchmark_type(perf_command)

        print(f"\n[{i+1}/{len(commits)}] Running {commit_hash[:8]} ({btype})")
        print(f"  Model: {model}")
        print(f"  Command: {perf_command[:80]}...")

        # Check if already run
        result_file = OUTPUT_DIR / f"{commit_hash[:8]}_result.json"
        if result_file.exists():
            print(f"  SKIP: Already have result")
            continue

        # Run appropriate benchmark
        if btype == 'serving':
            result = run_serving_benchmark(commit_hash, model, perf_command, hf_token)
        elif btype == 'throughput':
            result = run_throughput_benchmark(commit_hash, model, perf_command, hf_token)
        elif btype == 'latency':
            result = run_latency_benchmark(commit_hash, model, perf_command, hf_token)
        else:
            print(f"  SKIP: Unknown benchmark type: {btype}")
            continue

        # Save and report
        save_result(result)
        results.append(result)

        if result.status == 'success':
            print(f"  SUCCESS: {result.duration_s:.1f}s")
            if result.ttft_mean:
                print(f"    TTFT: {result.ttft_mean:.2f}ms, TPOT: {result.tpot_mean:.2f}ms")
        else:
            print(f"  {result.status.upper()}: {result.error}")

    # Summary
    print("\n" + "="*50)
    print("SUMMARY")
    print("="*50)
    success = sum(1 for r in results if r.status == 'success')
    errors = sum(1 for r in results if r.status == 'error')
    timeouts = sum(1 for r in results if r.status == 'timeout')
    print(f"Success: {success}/{len(results)}")
    print(f"Errors: {errors}/{len(results)}")
    print(f"Timeouts: {timeouts}/{len(results)}")


if __name__ == '__main__':
    main()
