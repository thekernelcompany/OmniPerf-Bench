#!/usr/bin/env python3
"""
Local Docker Benchmark Runner for vLLM commits.

Runs benchmarks inside pre-built Docker containers for commits that failed on Modal.
Supports serving, throughput, and latency benchmark types.
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
RESULTS_DIR = Path("/root/OmniPerf-Bench/omniperf_results_3way_claude_code")
FULL_RESULTS_FILE = RESULTS_DIR / "full_results.jsonl"
OUTPUT_DIR = RESULTS_DIR / "docker_benchmark_results"


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

    # Get benchmark scripts
    mkdir -p /opt/benchmarks && cd /opt/benchmarks
    curl -sL "https://raw.githubusercontent.com/vllm-project/vllm/$COMMIT/benchmarks/benchmark_serving.py" -o benchmark_serving.py 2>/dev/null || true
    curl -sL "https://raw.githubusercontent.com/vllm-project/vllm/$COMMIT/benchmarks/backend_request_func.py" -o backend_request_func.py 2>/dev/null || true
    curl -sL "https://raw.githubusercontent.com/vllm-project/vllm/$COMMIT/benchmarks/benchmark_dataset.py" -o benchmark_dataset.py 2>/dev/null || true
    curl -sL "https://raw.githubusercontent.com/vllm-project/vllm/$COMMIT/benchmarks/benchmark_utils.py" -o benchmark_utils.py 2>/dev/null || true

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
    cd /opt/benchmarks
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

    # Get benchmark script
    mkdir -p /opt/benchmarks && cd /opt/benchmarks
    curl -sL "https://raw.githubusercontent.com/vllm-project/vllm/$COMMIT/benchmarks/benchmark_throughput.py" -o benchmark_throughput.py 2>/dev/null || true

    # Run benchmark directly (no server needed)
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

    # Get benchmark script
    mkdir -p /opt/benchmarks && cd /opt/benchmarks
    curl -sL "https://raw.githubusercontent.com/vllm-project/vllm/$COMMIT/benchmarks/benchmark_latency.py" -o benchmark_latency.py 2>/dev/null || true

    # Run benchmark directly
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


def save_result(result: BenchmarkResult):
    """Save benchmark result to file."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    output_file = OUTPUT_DIR / f"{result.commit_hash[:8]}_result.json"

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


def main():
    """Main entry point."""
    import argparse
    parser = argparse.ArgumentParser(description='Run Docker-based vLLM benchmarks')
    parser.add_argument('--commit', type=str, help='Run specific commit (short hash)')
    parser.add_argument('--type', type=str, choices=['serving', 'throughput', 'latency'],
                        help='Only run benchmarks of this type')
    parser.add_argument('--limit', type=int, default=0, help='Limit number of commits to run')
    parser.add_argument('--dry-run', action='store_true', help='Show what would be run')
    args = parser.parse_args()

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
