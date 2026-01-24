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

# Agent patches configuration (legacy - single agent)
AGENT_PATCHES_DIR = Path("/root/OmniPerf-Bench/perf-agents-bench/state/runs/vllm/claude_code/default/2025-12-22_21-40-38")
AGENT_OUTPUT_DIR = RESULTS_DIR / "agent_benchmark_results"

# Multi-agent configuration - paths to agent patch directories
AGENT_CONFIGS = {
    "claude_code": "perf-agents-bench/state/runs/vllm/claude_code/default/2025-12-22_21-40-38",
    "codex_gpt5": "perf-agents-bench/state/runs/vllm/codex/gpt-5",
    "trae_gpt5": "perf-agents-bench/state/runs/vllm/trae/gpt-5",
    "trae_sonnet45": "perf-agents-bench/state/runs/vllm/trae/claude-sonnet-45",
    # TRAE specific run paths:
    "trae_gpt5_0123": "perf-agents-bench/state/runs/vllm/trae/gpt-5/2026-01-23_21-19-19",
    "trae_sonnet45_0123": "perf-agents-bench/state/runs/vllm/trae/us-anthropic-claude-sonnet-4-5-20250929-v1-0/2026-01-23_16-40-44",
}

# Output directories per agent type
AGENT_OUTPUT_DIRS = {
    "claude_code": Path("/root/OmniPerf-Bench/omniperf_results_3way_claude_code"),
    "codex_gpt5": Path("/root/OmniPerf-Bench/omniperf_results_3way_codex"),
    "trae_gpt5": Path("/root/OmniPerf-Bench/omniperf_results_3way_trae_gpt5"),
    "trae_sonnet45": Path("/root/OmniPerf-Bench/omniperf_results_3way_trae_sonnet45"),
    # TRAE specific run output dirs:
    "trae_gpt5_0123": Path("/root/OmniPerf-Bench/omniperf_results_3way_trae_gpt5_0123"),
    "trae_sonnet45_0123": Path("/root/OmniPerf-Bench/omniperf_results_3way_trae_sonnet45_0123"),
}


def get_output_dirs(agent_type: str = "claude_code") -> Dict[str, Path]:
    """Get output directories for a specific agent type."""
    base_dir = AGENT_OUTPUT_DIRS.get(agent_type, RESULTS_DIR)
    return {
        'results': base_dir,
        'baseline': base_dir / "baseline_benchmark_results",
        'agent': base_dir / "agent_benchmark_results",
        'docker': base_dir / "docker_benchmark_results",
    }


# Pre-built baseline images on Docker Hub (preferred - no compilation needed)
BASELINE_IMAGE_HUB = "shikhar481/vllm_fixed_human_images"  # Docker Hub images
BASELINE_IMAGE_PREFIX = "vllm-baseline-built"  # Local cache prefix (fallback)

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


def check_baseline_image_exists(parent_commit: str) -> bool:
    """Check if a pre-built baseline image exists (Docker Hub or local)."""
    # First check Docker Hub image (preferred - no compilation)
    hub_image = f"{BASELINE_IMAGE_HUB}:baseline-{parent_commit[:12]}"
    try:
        result = subprocess.run(
            ['docker', 'manifest', 'inspect', hub_image],
            capture_output=True, text=True, timeout=30
        )
        if result.returncode == 0:
            return True
    except:
        pass

    # Fall back to local cache
    local_image = f"{BASELINE_IMAGE_PREFIX}:{parent_commit[:12]}"
    result = subprocess.run(
        ['docker', 'images', '-q', local_image],
        capture_output=True, text=True, timeout=10
    )
    return bool(result.stdout.strip())


def get_baseline_image(parent_commit: str) -> str:
    """Get the baseline image tag (Docker Hub preferred, local fallback)."""
    # First check Docker Hub image
    hub_image = f"{BASELINE_IMAGE_HUB}:baseline-{parent_commit[:12]}"
    try:
        result = subprocess.run(
            ['docker', 'manifest', 'inspect', hub_image],
            capture_output=True, text=True, timeout=30
        )
        if result.returncode == 0:
            return hub_image
    except:
        pass

    # Fall back to local cache
    return f"{BASELINE_IMAGE_PREFIX}:{parent_commit[:12]}"


def save_baseline_image(container_id: str, parent_commit: str) -> bool:
    """Commit a container with built vLLM as a reusable baseline image."""
    image_tag = f"{BASELINE_IMAGE_PREFIX}:{parent_commit[:12]}"
    try:
        result = subprocess.run(
            ['docker', 'commit', container_id, image_tag],
            capture_output=True, text=True, timeout=300
        )
        if result.returncode == 0:
            print(f"  Saved baseline image: {image_tag}")
            return True
        else:
            print(f"  Failed to save baseline image: {result.stderr}")
            return False
    except Exception as e:
        print(f"  Error saving baseline image: {e}")
        return False


def get_hf_token() -> str:
    """Get HuggingFace token."""
    # First try reading from token file directly
    token_file = Path.home() / ".cache" / "huggingface" / "token"
    if token_file.exists():
        return token_file.read_text().strip()

    # Fallback to huggingface_hub
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


def load_baseline_mapping(mapping_file: Path = None) -> List[Dict[str, Any]]:
    """Load baseline benchmark mapping (human commit -> parent commit)."""
    file_path = mapping_file if mapping_file else BASELINE_MAPPING_FILE
    if not file_path.exists():
        print(f"ERROR: Baseline mapping file not found: {file_path}")
        print("Run the mapping generator first or create the file.")
        return []

    with open(file_path) as f:
        return json.load(f)


def load_agent_patch_mapping(agent_type: str = "claude_code") -> Dict[str, Path]:
    """Map human commits to their agent patch paths.

    Supports multiple agent types with different directory structures:
    - claude_code: vllm_core-*/journal.json + model_patch.diff
    - codex_gpt5, trae_gpt5, trae_sonnet45: */run_summary.json + model_patch.diff

    Uses os.walk to traverse all subdirectories and find patches.
    """
    mapping = {}

    # Get agent directory from config
    agent_subdir = AGENT_CONFIGS.get(agent_type)
    if not agent_subdir:
        print(f"WARNING: Unknown agent type: {agent_type}")
        return mapping

    agent_dir = Path("/root/OmniPerf-Bench") / agent_subdir
    if not agent_dir.exists():
        print(f"WARNING: Agent patches directory not found: {agent_dir}")
        return mapping

    # Walk through all directories to find patches
    # Handles both journal.json (claude_code) and run_summary.json (other agents)
    for root, dirs, files in os.walk(agent_dir):
        patch_file = Path(root) / "model_patch.diff"

        # Skip if no patch file or empty patch
        if not patch_file.exists() or patch_file.stat().st_size == 0:
            continue

        # Try run_summary.json first (used by codex, trae)
        summary_file = Path(root) / "run_summary.json"
        journal_file = Path(root) / "journal.json"

        human_commit = None

        if summary_file.exists():
            try:
                with open(summary_file) as f:
                    summary = json.load(f)
                human_commit = summary.get('commits', {}).get('human', '')[:8]
            except Exception as e:
                pass

        # Fallback to journal.json (used by claude_code)
        if not human_commit and journal_file.exists():
            try:
                with open(journal_file) as f:
                    journal = json.load(f)
                human_commit = journal.get("commits", {}).get("human", "")[:8]
            except Exception as e:
                pass

        if human_commit and human_commit not in mapping:
            mapping[human_commit] = patch_file

    return mapping


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
    bench_args = re.sub(r'--dtype\s+\S+', '', bench_args)  # Strip --dtype (not supported in older vLLM)

    docker_cmd = f'''
    set -e
    COMMIT="{commit_hash}"
    MODEL="{model}"

    # Install uv for faster package management
    pip install uv -q

    # Install deps
    uv pip install aiohttp pandas datasets -q --system

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
    bench_args = re.sub(r'--dtype\s+\S+', '', bench_args)  # Strip --dtype (not supported in older vLLM)

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
    bench_args = re.sub(r'--dtype\s+\S+', '', bench_args)  # Strip --dtype (not supported in older vLLM)

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
                                   perf_command: str, hf_token: str, timeout: int = 10800) -> BenchmarkResult:
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
    bench_args = re.sub(r'--dtype\s+\S+', '', bench_args)  # Strip --dtype (not supported in older vLLM)

    docker_cmd = f'''
    set -e
    PARENT_COMMIT="{parent_commit}"
    MODEL="{model}"

    echo "=== BASELINE BUILD: Installing CUDA toolkit ==="
    apt-get update -qq
    apt-get install -y -qq cuda-toolkit-12-4 git

    echo "=== Cloning vLLM at parent commit $PARENT_COMMIT ==="
    cd /opt
    # Retry git clone up to 3 times (network can be flaky)
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
    pip install uv -q

    echo "=== Uninstalling human vLLM and building baseline from source ==="
    uv pip uninstall vllm --system || true

    # Build with H100 optimization only (SM 9.0)
    export TORCH_CUDA_ARCH_LIST="9.0"
    export MAX_JOBS=32  # Increased for RAM headroom, NVCC_THREADS=2 = 16 effective jobs
    export NVCC_THREADS=2  # Multi-threaded nvcc compilation
    # Install build dependencies first
    uv pip install setuptools wheel packaging ninja cmake --system
    # Use regular pip for build since it can see system torch (uv can't)
    pip install -e . --no-build-isolation 2>&1

    # Fix aimv2 config registration conflict with newer transformers (both ovis.py and ovis2.py)
    for ovis_file in /opt/vllm_baseline/vllm/transformers_utils/configs/ovis.py /opt/vllm_baseline/vllm/transformers_utils/configs/ovis2.py; do
        if [ -f "$ovis_file" ] && grep -q 'AutoConfig.register("aimv2"' "$ovis_file" 2>/dev/null; then
            sed -i 's/AutoConfig.register("aimv2", AIMv2Config)/AutoConfig.register("aimv2", AIMv2Config, exist_ok=True)/' "$ovis_file"
        fi
    done

    # Fix pyairports dependency for older vLLM with outlines
    echo "Installing pyairports..."
    pip install pyairports --no-cache-dir 2>&1 || echo "pyairports install warning (may be ok)"

    echo "=== Verifying baseline vLLM installation ==="
    python3 -c "import vllm; print(f'vLLM version: {{vllm.__version__}}')"

    # Install benchmark deps
    uv pip install aiohttp pandas datasets -q --system

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
                                      perf_command: str, hf_token: str, timeout: int = 10800) -> BenchmarkResult:
    """Run a baseline throughput benchmark by building vLLM from source at parent commit."""
    start_time = time.time()

    human_short = human_commit[:8]
    docker_image = get_docker_image(human_short, human_commit)

    # Extract benchmark args
    bench_args = re.sub(r'python\s+benchmarks/benchmark_throughput\.py\s*', '', perf_command)
    # Also handle vllm bench throughput format
    bench_args = re.sub(r'vllm\s+bench\s+throughput\s*', '', bench_args)
    bench_args = re.sub(r'--dtype\s+\S+', '', bench_args)  # Strip --dtype (not supported in older vLLM)

    docker_cmd = f'''
    set -e
    PARENT_COMMIT="{parent_commit}"

    echo "=== BASELINE BUILD: Installing CUDA toolkit ==="
    apt-get update -qq
    apt-get install -y -qq cuda-toolkit-12-4 git

    echo "=== Cloning vLLM at parent commit $PARENT_COMMIT ==="
    cd /opt
    # Retry git clone up to 3 times (network can be flaky)
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
    pip install uv -q

    echo "=== Uninstalling human vLLM and building baseline from source ==="
    uv pip uninstall vllm --system || true

    export TORCH_CUDA_ARCH_LIST="9.0"
    export MAX_JOBS=32  # Increased for RAM headroom, NVCC_THREADS=2 = 16 effective jobs
    export NVCC_THREADS=2  # Multi-threaded nvcc compilation
    # Install build dependencies first
    uv pip install setuptools wheel packaging ninja cmake --system
    # Use regular pip for build since it can see system torch (uv can't)
    pip install -e . --no-build-isolation 2>&1

    # Fix aimv2 config registration conflict with newer transformers (both ovis.py and ovis2.py)
    for ovis_file in /opt/vllm_baseline/vllm/transformers_utils/configs/ovis.py /opt/vllm_baseline/vllm/transformers_utils/configs/ovis2.py; do
        if [ -f "$ovis_file" ] && grep -q 'AutoConfig.register("aimv2"' "$ovis_file" 2>/dev/null; then
            sed -i 's/AutoConfig.register("aimv2", AIMv2Config)/AutoConfig.register("aimv2", AIMv2Config, exist_ok=True)/' "$ovis_file"
        fi
    done

    # Fix pyairports dependency for older vLLM with outlines
    echo "Installing pyairports..."
    pip install pyairports --no-cache-dir 2>&1 || echo "pyairports install warning (may be ok)"

    echo "=== Verifying baseline vLLM installation ==="
    python3 -c "import vllm; print(f'vLLM version: {{vllm.__version__}}')"

    # Install benchmark dependencies
    pip install aiohttp pandas datasets -q

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
                                   perf_command: str, hf_token: str, timeout: int = 10800) -> BenchmarkResult:
    """Run a baseline latency benchmark by building vLLM from source at parent commit."""
    start_time = time.time()

    human_short = human_commit[:8]
    docker_image = get_docker_image(human_short, human_commit)

    bench_args = re.sub(r'python\s+benchmarks/benchmark_latency\.py\s*', '', perf_command)
    bench_args = re.sub(r'--dtype\s+\S+', '', bench_args)  # Strip --dtype (not supported in older vLLM)

    docker_cmd = f'''
    set -e
    PARENT_COMMIT="{parent_commit}"

    echo "=== BASELINE BUILD: Installing CUDA toolkit ==="
    apt-get update -qq
    apt-get install -y -qq cuda-toolkit-12-4 git

    echo "=== Cloning vLLM at parent commit $PARENT_COMMIT ==="
    cd /opt
    # Retry git clone up to 3 times (network can be flaky)
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
    pip install uv -q

    echo "=== Uninstalling human vLLM and building baseline from source ==="
    uv pip uninstall vllm --system || true

    export TORCH_CUDA_ARCH_LIST="9.0"
    export MAX_JOBS=32  # Increased for RAM headroom, NVCC_THREADS=2 = 16 effective jobs
    export NVCC_THREADS=2  # Multi-threaded nvcc compilation
    # Install build dependencies first
    uv pip install setuptools wheel packaging ninja cmake --system
    # Use regular pip for build since it can see system torch (uv can't)
    pip install -e . --no-build-isolation 2>&1

    # Fix aimv2 config registration conflict with newer transformers (both ovis.py and ovis2.py)
    for ovis_file in /opt/vllm_baseline/vllm/transformers_utils/configs/ovis.py /opt/vllm_baseline/vllm/transformers_utils/configs/ovis2.py; do
        if [ -f "$ovis_file" ] && grep -q 'AutoConfig.register("aimv2"' "$ovis_file" 2>/dev/null; then
            sed -i 's/AutoConfig.register("aimv2", AIMv2Config)/AutoConfig.register("aimv2", AIMv2Config, exist_ok=True)/' "$ovis_file"
        fi
    done

    # Fix pyairports dependency for older vLLM with outlines
    echo "Installing pyairports..."
    pip install pyairports --no-cache-dir 2>&1 || echo "pyairports install warning (may be ok)"

    echo "=== Verifying baseline vLLM installation ==="
    python3 -c "import vllm; print(f'vLLM version: {{vllm.__version__}}')"

    # Install benchmark dependencies
    pip install aiohttp pandas datasets -q

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


def build_baseline_image_only(human_commit: str, parent_commit: str, timeout: int = 10800) -> bool:
    """Build a clean baseline image (no benchmarks, no patches).

    This creates a pristine baseline image that can be reused for multiple benchmarks.
    Call this before running benchmarks to pre-build images.
    """
    if check_baseline_image_exists(parent_commit):
        print(f"  Baseline image already exists: {get_baseline_image(parent_commit)}")
        return True

    human_short = human_commit[:8]
    docker_image = get_docker_image(human_short, human_commit)
    container_name = f"baseline-build-{parent_commit[:12]}"

    build_cmd = f'''
    set -e
    PARENT_COMMIT="{parent_commit}"

    echo "=== BASELINE BUILD: Installing CUDA toolkit ==="
    apt-get update -qq
    apt-get install -y -qq cuda-toolkit-12-4 git

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

    pip install uv -q

    echo "=== Uninstalling human vLLM and building baseline from source ==="
    uv pip uninstall vllm --system || true
    uv pip install setuptools wheel packaging ninja cmake --system

    export TORCH_CUDA_ARCH_LIST="9.0"
    export MAX_JOBS=32  # Increased for RAM headroom, NVCC_THREADS=2 = 16 effective jobs
    export NVCC_THREADS=2  # Multi-threaded nvcc compilation
    pip install -e . --no-build-isolation 2>&1

    # Fix aimv2 config registration conflict (both ovis.py and ovis2.py)
    for ovis_file in /opt/vllm_baseline/vllm/transformers_utils/configs/ovis.py /opt/vllm_baseline/vllm/transformers_utils/configs/ovis2.py; do
        if [ -f "$ovis_file" ] && grep -q 'AutoConfig.register("aimv2"' "$ovis_file" 2>/dev/null; then
            sed -i 's/AutoConfig.register("aimv2", AIMv2Config)/AutoConfig.register("aimv2", AIMv2Config, exist_ok=True)/' "$ovis_file"
        fi
    done

    # Fix pyairports dependency for older vLLM with outlines
    echo "Installing pyairports..."
    pip install pyairports --no-cache-dir 2>&1 || echo "pyairports install warning (may be ok)"

    echo "=== Verifying baseline vLLM installation ==="
    python3 -c "import vllm; print(f'vLLM version: {{vllm.__version__}}')"

    # Install benchmark deps
    pip install aiohttp pandas datasets --no-cache-dir

    echo "BUILD_SUCCESS"
    '''

    print(f"  Building clean baseline image for parent {parent_commit[:8]}...")

    try:
        subprocess.run(['docker', 'rm', '-f', container_name], capture_output=True, timeout=30)

        result = subprocess.run(
            [
                'docker', 'run',
                '--name', container_name,
                '--gpus', 'all',
                '--shm-size=16g',
                '--entrypoint', 'bash',
                docker_image,
                '-c', build_cmd
            ],
            capture_output=True, text=True, timeout=timeout
        )

        output = result.stdout + result.stderr

        if "BUILD_SUCCESS" in output:
            print(f"  Build succeeded - committing clean image...")
            if save_baseline_image(container_name, parent_commit):
                subprocess.run(['docker', 'rm', '-f', container_name], capture_output=True, timeout=30)
                return True

        print(f"  Build failed: {output[-2000:]}")
        subprocess.run(['docker', 'rm', '-f', container_name], capture_output=True, timeout=30)
        return False

    except subprocess.TimeoutExpired:
        print(f"  Build timed out after {timeout}s")
        subprocess.run(['docker', 'rm', '-f', container_name], capture_output=True, timeout=30)
        return False
    except Exception as e:
        print(f"  Build error: {e}")
        subprocess.run(['docker', 'rm', '-f', container_name], capture_output=True, timeout=30)
        return False


def run_combined_baseline_agent_serving(human_commit: str, parent_commit: str, model: str,
                                        perf_command: str, agent_patch_path: Path,
                                        hf_token: str, timeout: int = 10800) -> tuple:
    """Run baseline benchmark, then apply agent patch and run again (no rebuild needed).

    Returns tuple of (baseline_result, agent_result).
    Agent patches are Python-only, so no rebuild is required.

    CACHING: If baseline was built before, reuse it (skip 20+ min build time).
    Must call build_baseline_image_only() first to ensure clean cached image.
    """
    start_time = time.time()

    human_short = human_commit[:8]

    # Check if we have a pre-built baseline image (saves 20+ min build time)
    use_cached = check_baseline_image_exists(parent_commit)
    if use_cached:
        docker_image = get_baseline_image(parent_commit)
        print(f"  Using cached baseline image: {docker_image}")
    else:
        docker_image = get_docker_image(human_short, human_commit)
        print(f"  No cached baseline - will build from source")

    # Adjust perf_command for random dataset
    perf_command = re.sub(r'--dataset-name\s+sharegpt', '--dataset-name random', perf_command)
    perf_command = re.sub(r'--dataset-name\s+sonnet', '--dataset-name random', perf_command)
    perf_command = re.sub(r'--dataset\s+\S+\.json', '', perf_command)

    if '--dataset-name' not in perf_command and '--dataset-path' not in perf_command:
        perf_command += ' --dataset-name random'

    if '--dataset-name random' in perf_command and '--random-input-len' not in perf_command:
        perf_command += ' --random-input-len 256 --random-output-len 64'

    bench_args = re.sub(r'python\s+benchmarks/benchmark_serving\.py\s*', '', perf_command)
    bench_args = re.sub(r'--dtype\s+\S+', '', bench_args)  # Strip --dtype (not supported in older vLLM)

    # Build steps - only needed if not using cached image
    build_steps = f'''
    echo "=== BASELINE BUILD: Installing CUDA toolkit ==="
    apt-get update -qq
    apt-get install -y -qq cuda-toolkit-12-4 git

    echo "=== Cloning vLLM at parent commit $PARENT_COMMIT ==="
    cd /opt
    # Retry git clone up to 3 times (network can be flaky)
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

    pip install uv -q

    echo "=== Uninstalling human vLLM and building baseline from source ==="
    uv pip uninstall vllm --system || true

    # Install build dependencies
    uv pip install setuptools wheel packaging ninja cmake --system

    export TORCH_CUDA_ARCH_LIST="9.0"
    export MAX_JOBS=32  # Increased for RAM headroom, NVCC_THREADS=2 = 16 effective jobs
    export NVCC_THREADS=2  # Multi-threaded nvcc compilation
    echo "=== Building vLLM from source ==="
    # Use regular pip for build since it can see system torch (uv can't)
    pip install -e . --no-build-isolation 2>&1

    # Fix aimv2 config registration conflict with newer transformers (both ovis.py and ovis2.py)
    echo "=== Patching compatibility issues ==="
    for ovis_file in /opt/vllm_baseline/vllm/transformers_utils/configs/ovis.py /opt/vllm_baseline/vllm/transformers_utils/configs/ovis2.py; do
        if [ -f "$ovis_file" ] && grep -q 'AutoConfig.register("aimv2"' "$ovis_file" 2>/dev/null; then
            sed -i 's/AutoConfig.register("aimv2", AIMv2Config)/AutoConfig.register("aimv2", AIMv2Config, exist_ok=True)/' "$ovis_file"
            echo "Patched aimv2 registration in $ovis_file"
        fi
    done

    # Fix pyairports dependency for older vLLM with outlines
    echo "Installing pyairports..."
    pip install pyairports --no-cache-dir 2>&1 || echo "pyairports install warning (may be ok)"

    echo "=== Verifying baseline vLLM installation ==="
    python3 -c "import vllm; print(f'vLLM version: {{vllm.__version__}}')"
    echo "BUILD_SUCCESS"

    # Install benchmark deps using pip (not uv) to ensure venv compatibility
    pip install aiohttp pandas datasets -q
    '''

    # For cached image, skip build entirely
    cached_setup = '''
    echo "=== Using cached baseline image (skipping build) ==="
    # Install benchmark deps using pip to ensure venv compatibility
    pip install aiohttp pandas datasets -q
    '''

    docker_cmd = f'''
    set -e
    PARENT_COMMIT="{parent_commit}"
    MODEL="{model}"

    {cached_setup if use_cached else build_steps}

    # ============ COMPATIBILITY FIXES ============
    echo "Applying compatibility fixes..."

    # Fix transformers LogitsWarper issue
    if ! python3 -c "from transformers.generation.logits_process import LogitsWarper" 2>/dev/null; then
        echo "Fixing transformers compatibility (LogitsWarper missing)..."
        pip install 'transformers==4.44.2' -q 2>/dev/null || true
    fi

    # Fix numpy < 2 for outlines compatibility
    pip install 'numpy<2' -q 2>/dev/null || true

    # Fix rope_scaling for Llama-3.1 models (use rope_type instead of type)
    VLLM_DIR=$(python3 -c "import vllm, os; print(os.path.dirname(vllm.__file__))" 2>/dev/null || echo "/opt/vllm_baseline/vllm")
    if [ -d "$VLLM_DIR" ]; then
        echo "Fixing rope_scaling compatibility..."
        find "$VLLM_DIR" -name "*.py" -exec grep -l 'rope_scaling\["type"\]' {{}} \; 2>/dev/null | while read f; do
            sed -i 's/rope_scaling\["type"\]/rope_scaling.get("type", rope_scaling.get("rope_type"))/g' "$f"
        done
    fi

    # Fix outlines.fsm compatibility
    if ! python3 -c "from outlines.fsm.guide import Guide" 2>/dev/null; then
        echo "Fixing outlines.fsm compatibility..."
        pip uninstall outlines -y 2>/dev/null || true
        pip install 'outlines==0.0.34' --no-deps 2>/dev/null || true
    fi

    # If outlines still fails, patch vLLM guided_decoding
    if ! python3 -c "from outlines.fsm.guide import Guide" 2>/dev/null; then
        echo "Patching vLLM guided_decoding..."
        VLLM_DIR=$(python3 -c "import vllm, os; print(os.path.dirname(vllm.__file__))" 2>/dev/null || echo "/opt/vllm_baseline/vllm")
        if [ -f "$VLLM_DIR/model_executor/guided_decoding/__init__.py" ]; then
            cat > "$VLLM_DIR/model_executor/guided_decoding/__init__.py" << 'PATCH'
from typing import Optional
from dataclasses import dataclass
@dataclass
class GuidedDecodingRequest:
    guided_json: Optional[str] = None
    guided_regex: Optional[str] = None
    guided_choice: Optional[list] = None
    guided_grammar: Optional[str] = None
    guided_decoding_backend: Optional[str] = None
    guided_whitespace_pattern: Optional[str] = None
async def get_guided_decoding_logits_processor(*args, **kwargs):
    return None
async def get_local_guided_decoding_logits_processor(*args, **kwargs):
    return None
def get_outlines_guided_decoding_logits_processor(*args, **kwargs):
    return None
PATCH
        fi
    fi

    # ============ BASELINE BENCHMARK ============
    echo "=== Starting vLLM server for BASELINE benchmark ==="
    python3 -m vllm.entrypoints.openai.api_server \\
        --model $MODEL --port 8000 --max-model-len 4096 --disable-log-requests 2>&1 &
    SERVER_PID=$!

    for i in $(seq 1 300); do
        if curl -s http://localhost:8000/v1/models 2>/dev/null | grep -q "model"; then
            echo "BASELINE_SERVER_READY_AFTER=${{i}}s"
            break
        fi
        if ! kill -0 $SERVER_PID 2>/dev/null; then
            echo "BASELINE_SERVER_CRASHED"
            exit 1
        fi
        sleep 1
    done

    if ! curl -s http://localhost:8000/v1/models > /dev/null 2>&1; then
        echo "BASELINE_SERVER_TIMEOUT"
        exit 1
    fi

    echo "=== Running BASELINE benchmark ==="
    cd /opt/vllm_baseline/benchmarks
    python3 benchmark_serving.py {bench_args} --port 8000 2>&1 | tee /tmp/baseline_output.txt
    echo "BASELINE_BENCHMARK_DONE"

    # Stop baseline server
    kill $SERVER_PID 2>/dev/null || true
    sleep 2

    # ============ AGENT BENCHMARK ============
    echo "=== Applying agent patch (Python-only, no rebuild needed) ==="
    cd /opt/vllm_baseline
    if git apply --check /agent_patch.diff 2>/dev/null; then
        git apply /agent_patch.diff
        echo "AGENT_PATCH_APPLIED"
    else
        echo "AGENT_PATCH_FAILED"
        cat /tmp/baseline_output.txt
        exit 0
    fi

    echo "=== Starting vLLM server for AGENT benchmark ==="
    python3 -m vllm.entrypoints.openai.api_server \\
        --model $MODEL --port 8000 --max-model-len 4096 --disable-log-requests 2>&1 &
    SERVER_PID=$!

    for i in $(seq 1 300); do
        if curl -s http://localhost:8000/v1/models 2>/dev/null | grep -q "model"; then
            echo "AGENT_SERVER_READY_AFTER=${{i}}s"
            break
        fi
        if ! kill -0 $SERVER_PID 2>/dev/null; then
            echo "AGENT_SERVER_CRASHED"
            cat /tmp/baseline_output.txt
            exit 0
        fi
        sleep 1
    done

    if ! curl -s http://localhost:8000/v1/models > /dev/null 2>&1; then
        echo "AGENT_SERVER_TIMEOUT"
        cat /tmp/baseline_output.txt
        exit 0
    fi

    echo "=== Running AGENT benchmark ==="
    cd /opt/vllm_baseline/benchmarks
    python3 benchmark_serving.py {bench_args} --port 8000 2>&1 | tee /tmp/agent_output.txt
    echo "AGENT_BENCHMARK_DONE"

    kill $SERVER_PID 2>/dev/null || true

    # Output both results
    echo "=== BASELINE OUTPUT ==="
    cat /tmp/baseline_output.txt
    echo "=== AGENT OUTPUT ==="
    cat /tmp/agent_output.txt
    '''

    # Container name for potential caching
    container_name = f"baseline-build-{parent_commit[:12]}"

    try:
        # Remove any existing container with same name
        subprocess.run(['docker', 'rm', '-f', container_name], capture_output=True, timeout=30)

        result = subprocess.run(
            [
                'docker', 'run',
                '--name', container_name,  # Named container for caching
                '--gpus', 'all',
                '-e', f'HF_TOKEN={hf_token}',
                '-e', f'HUGGING_FACE_HUB_TOKEN={hf_token}',
                '-v', '/root/.cache/huggingface:/root/.cache/huggingface',
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

        # NOTE: We do NOT cache from combined benchmark because agent patch modifies files.
        # Use build_baseline_image_only() to pre-build clean baseline images.

        # Clean up container
        subprocess.run(['docker', 'rm', '-f', container_name], capture_output=True, timeout=30)

        # Parse baseline metrics
        baseline_output = ""
        agent_output = ""
        if "=== BASELINE OUTPUT ===" in output and "=== AGENT OUTPUT ===" in output:
            parts = output.split("=== BASELINE OUTPUT ===")
            if len(parts) > 1:
                baseline_agent = parts[1].split("=== AGENT OUTPUT ===")
                baseline_output = baseline_agent[0] if len(baseline_agent) > 0 else ""
                agent_output = baseline_agent[1] if len(baseline_agent) > 1 else ""

        baseline_metrics = parse_serving_metrics(baseline_output or output)
        agent_metrics = parse_serving_metrics(agent_output) if "AGENT_BENCHMARK_DONE" in output else {}

        # Create baseline result
        baseline_status = 'success' if baseline_metrics else 'error'
        if 'BASELINE_SERVER_CRASHED' in output:
            baseline_status = 'error'
            baseline_error = 'Server crashed during startup'
        elif 'BASELINE_SERVER_TIMEOUT' in output:
            baseline_status = 'error'
            baseline_error = 'Server startup timeout'
        elif not baseline_metrics:
            baseline_error = 'No metrics in output'
        else:
            baseline_error = None

        baseline_result = BenchmarkResult(
            commit_hash=parent_commit, status=baseline_status, benchmark_type='serving',
            model=model, duration_s=duration, error=baseline_error,
            raw_output=baseline_output[-10000:] if baseline_output else output[-10000:],
            **baseline_metrics
        )

        # Create agent result
        agent_result = None
        if "AGENT_PATCH_APPLIED" in output:
            agent_status = 'success' if agent_metrics else 'error'
            if 'AGENT_SERVER_CRASHED' in output:
                agent_status = 'error'
                agent_error = 'Server crashed after patch'
            elif 'AGENT_SERVER_TIMEOUT' in output:
                agent_status = 'error'
                agent_error = 'Server startup timeout after patch'
            elif not agent_metrics:
                agent_error = 'No metrics in agent output'
            else:
                agent_error = None

            agent_result = BenchmarkResult(
                commit_hash=f"{human_short}_agent", status=agent_status, benchmark_type='serving',
                model=model, duration_s=duration, error=agent_error,
                raw_output=agent_output[-10000:] if agent_output else "",
                **agent_metrics
            )
        elif "AGENT_PATCH_FAILED" in output:
            agent_result = BenchmarkResult(
                commit_hash=f"{human_short}_agent", status='error', benchmark_type='serving',
                model=model, duration_s=duration, error='Agent patch failed to apply',
                raw_output=""
            )

        return (baseline_result, agent_result)

    except subprocess.TimeoutExpired:
        # Clean up container on timeout
        subprocess.run(['docker', 'rm', '-f', container_name], capture_output=True, timeout=30)
        return (
            BenchmarkResult(
                commit_hash=parent_commit, status='timeout', benchmark_type='serving',
                model=model, duration_s=timeout, error=f'Benchmark timed out after {timeout}s'
            ),
            None
        )
    except Exception as e:
        # Clean up container on error
        subprocess.run(['docker', 'rm', '-f', container_name], capture_output=True, timeout=30)
        return (
            BenchmarkResult(
                commit_hash=parent_commit, status='error', benchmark_type='serving',
                model=model, duration_s=time.time() - start_time, error=str(e)
            ),
            None
        )


def run_combined_baseline_agent_throughput(human_commit: str, parent_commit: str, model: str,
                                           perf_command: str, agent_patch_path: Path,
                                           hf_token: str, timeout: int = 10800) -> tuple:
    """Run baseline throughput benchmark, then apply agent patch and run again.

    Returns tuple of (baseline_result, agent_result).
    Throughput benchmarks don't need a server - they run directly.
    """
    start_time = time.time()
    human_short = human_commit[:8]

    # Check if we have a pre-built baseline image
    use_cached = check_baseline_image_exists(parent_commit)
    if use_cached:
        docker_image = get_baseline_image(parent_commit)
        print(f"  Using cached baseline image: {docker_image}")
    else:
        docker_image = get_docker_image(human_short, human_commit)
        print(f"  No cached baseline - will build from source")

    # Extract benchmark args
    bench_args = re.sub(r'python\s+benchmarks/benchmark_throughput\.py\s*', '', perf_command)
    bench_args = re.sub(r'vllm\s+bench\s+throughput\s*', '', bench_args)
    bench_args = re.sub(r'--dtype\s+\S+', '', bench_args)

    # Build steps - only needed if not using cached image
    build_steps = f'''
    echo "=== BASELINE BUILD: Installing CUDA toolkit ==="
    apt-get update -qq
    apt-get install -y -qq cuda-toolkit-12-4 git

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

    pip install uv -q
    uv pip uninstall vllm --system || true
    uv pip install setuptools wheel packaging ninja cmake --system

    export TORCH_CUDA_ARCH_LIST="9.0"
    export MAX_JOBS=32
    export NVCC_THREADS=2
    pip install -e . --no-build-isolation 2>&1

    for ovis_file in /opt/vllm_baseline/vllm/transformers_utils/configs/ovis.py /opt/vllm_baseline/vllm/transformers_utils/configs/ovis2.py; do
        if [ -f "$ovis_file" ] && grep -q 'AutoConfig.register("aimv2"' "$ovis_file" 2>/dev/null; then
            sed -i 's/AutoConfig.register("aimv2", AIMv2Config)/AutoConfig.register("aimv2", AIMv2Config, exist_ok=True)/' "$ovis_file"
        fi
    done

    pip install pyairports --no-cache-dir 2>&1 || true
    python3 -c "import vllm; print(f'vLLM version: {{vllm.__version__}}')"
    echo "BUILD_SUCCESS"
    pip install aiohttp pandas datasets -q
    '''

    cached_setup = '''
    echo "=== Using cached baseline image (skipping build) ==="
    pip install aiohttp pandas datasets -q
    '''

    docker_cmd = f'''
    set -e
    PARENT_COMMIT="{parent_commit}"

    {cached_setup if use_cached else build_steps}

    # ============ BASELINE BENCHMARK ============
    echo "=== Running BASELINE throughput benchmark ==="
    cd /opt/vllm_baseline/benchmarks
    python3 benchmark_throughput.py {bench_args} 2>&1 | tee /tmp/baseline_output.txt
    echo "BASELINE_BENCHMARK_DONE"

    # ============ AGENT BENCHMARK ============
    echo "=== Applying agent patch ==="
    cd /opt/vllm_baseline
    if git apply --check /agent_patch.diff 2>/dev/null; then
        git apply /agent_patch.diff
        echo "AGENT_PATCH_APPLIED"
    else
        echo "AGENT_PATCH_FAILED"
        cat /tmp/baseline_output.txt
        exit 0
    fi

    echo "=== Running AGENT throughput benchmark ==="
    cd /opt/vllm_baseline/benchmarks
    python3 benchmark_throughput.py {bench_args} 2>&1 | tee /tmp/agent_output.txt
    echo "AGENT_BENCHMARK_DONE"

    echo "=== BASELINE OUTPUT ==="
    cat /tmp/baseline_output.txt
    echo "=== AGENT OUTPUT ==="
    cat /tmp/agent_output.txt
    '''

    container_name = f"baseline-build-{parent_commit[:12]}"

    try:
        subprocess.run(['docker', 'rm', '-f', container_name], capture_output=True, timeout=30)

        result = subprocess.run(
            [
                'docker', 'run',
                '--name', container_name,
                '--gpus', 'all',
                '-e', f'HF_TOKEN={hf_token}',
                '-e', f'HUGGING_FACE_HUB_TOKEN={hf_token}',
                '-v', '/root/.cache/huggingface:/root/.cache/huggingface',
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

        subprocess.run(['docker', 'rm', '-f', container_name], capture_output=True, timeout=30)

        # Parse outputs
        baseline_output = ""
        agent_output = ""
        if "=== BASELINE OUTPUT ===" in output and "=== AGENT OUTPUT ===" in output:
            parts = output.split("=== BASELINE OUTPUT ===")
            if len(parts) > 1:
                baseline_agent = parts[1].split("=== AGENT OUTPUT ===")
                baseline_output = baseline_agent[0] if len(baseline_agent) > 0 else ""
                agent_output = baseline_agent[1] if len(baseline_agent) > 1 else ""

        # Parse throughput metrics
        def parse_throughput(text):
            throughput_match = re.search(r'Throughput:\s+([\d.]+)\s+requests/s', text)
            tok_match = re.search(r'([\d.]+)\s+tokens/s', text)
            return {
                'throughput_req_s': float(throughput_match.group(1)) if throughput_match else None,
                'throughput_tok_s': float(tok_match.group(1)) if tok_match else None,
            }

        baseline_metrics = parse_throughput(baseline_output or output)
        agent_metrics = parse_throughput(agent_output) if "AGENT_BENCHMARK_DONE" in output else {}

        baseline_status = 'success' if baseline_metrics.get('throughput_req_s') or baseline_metrics.get('throughput_tok_s') else 'error'
        baseline_result = BenchmarkResult(
            commit_hash=parent_commit, status=baseline_status, benchmark_type='throughput',
            model=model, duration_s=duration,
            error=None if baseline_status == 'success' else 'No throughput metrics',
            raw_output=baseline_output[-10000:] if baseline_output else output[-10000:],
            **baseline_metrics
        )

        agent_result = None
        if "AGENT_PATCH_APPLIED" in output:
            agent_status = 'success' if agent_metrics.get('throughput_req_s') or agent_metrics.get('throughput_tok_s') else 'error'
            agent_result = BenchmarkResult(
                commit_hash=f"{human_short}_agent", status=agent_status, benchmark_type='throughput',
                model=model, duration_s=duration,
                error=None if agent_status == 'success' else 'No throughput metrics',
                raw_output=agent_output[-10000:] if agent_output else "",
                **agent_metrics
            )
        elif "AGENT_PATCH_FAILED" in output:
            agent_result = BenchmarkResult(
                commit_hash=f"{human_short}_agent", status='error', benchmark_type='throughput',
                model=model, duration_s=duration, error='Agent patch failed to apply'
            )

        return (baseline_result, agent_result)

    except subprocess.TimeoutExpired:
        subprocess.run(['docker', 'rm', '-f', container_name], capture_output=True, timeout=30)
        return (
            BenchmarkResult(commit_hash=parent_commit, status='timeout', benchmark_type='throughput',
                          model=model, duration_s=timeout, error=f'Timed out after {timeout}s'),
            None
        )
    except Exception as e:
        subprocess.run(['docker', 'rm', '-f', container_name], capture_output=True, timeout=30)
        return (
            BenchmarkResult(commit_hash=parent_commit, status='error', benchmark_type='throughput',
                          model=model, duration_s=time.time() - start_time, error=str(e)),
            None
        )


def run_combined_baseline_agent_latency(human_commit: str, parent_commit: str, model: str,
                                        perf_command: str, agent_patch_path: Path,
                                        hf_token: str, timeout: int = 10800) -> tuple:
    """Run baseline latency benchmark, then apply agent patch and run again.

    Returns tuple of (baseline_result, agent_result).
    Latency benchmarks don't need a server - they run directly.
    """
    start_time = time.time()
    human_short = human_commit[:8]

    use_cached = check_baseline_image_exists(parent_commit)
    if use_cached:
        docker_image = get_baseline_image(parent_commit)
        print(f"  Using cached baseline image: {docker_image}")
    else:
        docker_image = get_docker_image(human_short, human_commit)
        print(f"  No cached baseline - will build from source")

    # Extract benchmark args
    bench_args = re.sub(r'python\s+benchmarks/benchmark_latency\.py\s*', '', perf_command)
    bench_args = re.sub(r'vllm\s+bench\s+latency\s*', '', bench_args)
    bench_args = re.sub(r'--dtype\s+\S+', '', bench_args)
    # Remove environment variable prefixes if present
    bench_args = re.sub(r'VLLM_\w+=\S+\s*', '', bench_args)

    build_steps = f'''
    echo "=== BASELINE BUILD: Installing CUDA toolkit ==="
    apt-get update -qq
    apt-get install -y -qq cuda-toolkit-12-4 git

    cd /opt
    for attempt in 1 2 3; do
        if git clone --depth 1 https://github.com/vllm-project/vllm.git vllm_baseline 2>&1; then
            break
        fi
        rm -rf vllm_baseline 2>/dev/null
        sleep 5
    done
    cd vllm_baseline
    git fetch --depth 1 origin $PARENT_COMMIT
    git checkout $PARENT_COMMIT

    pip install uv -q
    uv pip uninstall vllm --system || true
    uv pip install setuptools wheel packaging ninja cmake --system

    export TORCH_CUDA_ARCH_LIST="9.0"
    export MAX_JOBS=32
    export NVCC_THREADS=2
    pip install -e . --no-build-isolation 2>&1

    for ovis_file in /opt/vllm_baseline/vllm/transformers_utils/configs/ovis.py /opt/vllm_baseline/vllm/transformers_utils/configs/ovis2.py; do
        if [ -f "$ovis_file" ] && grep -q 'AutoConfig.register("aimv2"' "$ovis_file" 2>/dev/null; then
            sed -i 's/AutoConfig.register("aimv2", AIMv2Config)/AutoConfig.register("aimv2", AIMv2Config, exist_ok=True)/' "$ovis_file"
        fi
    done

    pip install pyairports --no-cache-dir 2>&1 || true
    python3 -c "import vllm; print(f'vLLM version: {{vllm.__version__}}')"
    echo "BUILD_SUCCESS"
    pip install aiohttp pandas datasets -q
    '''

    cached_setup = '''
    echo "=== Using cached baseline image (skipping build) ==="
    pip install aiohttp pandas datasets -q
    '''

    docker_cmd = f'''
    set -e
    PARENT_COMMIT="{parent_commit}"

    {cached_setup if use_cached else build_steps}

    # ============ BASELINE BENCHMARK ============
    echo "=== Running BASELINE latency benchmark ==="
    cd /opt/vllm_baseline/benchmarks
    python3 benchmark_latency.py {bench_args} 2>&1 | tee /tmp/baseline_output.txt
    echo "BASELINE_BENCHMARK_DONE"

    # ============ AGENT BENCHMARK ============
    echo "=== Applying agent patch ==="
    cd /opt/vllm_baseline
    if git apply --check /agent_patch.diff 2>/dev/null; then
        git apply /agent_patch.diff
        echo "AGENT_PATCH_APPLIED"
    else
        echo "AGENT_PATCH_FAILED"
        cat /tmp/baseline_output.txt
        exit 0
    fi

    echo "=== Running AGENT latency benchmark ==="
    cd /opt/vllm_baseline/benchmarks
    python3 benchmark_latency.py {bench_args} 2>&1 | tee /tmp/agent_output.txt
    echo "AGENT_BENCHMARK_DONE"

    echo "=== BASELINE OUTPUT ==="
    cat /tmp/baseline_output.txt
    echo "=== AGENT OUTPUT ==="
    cat /tmp/agent_output.txt
    '''

    container_name = f"baseline-build-{parent_commit[:12]}"

    try:
        subprocess.run(['docker', 'rm', '-f', container_name], capture_output=True, timeout=30)

        result = subprocess.run(
            [
                'docker', 'run',
                '--name', container_name,
                '--gpus', 'all',
                '-e', f'HF_TOKEN={hf_token}',
                '-e', f'HUGGING_FACE_HUB_TOKEN={hf_token}',
                '-v', '/root/.cache/huggingface:/root/.cache/huggingface',
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

        subprocess.run(['docker', 'rm', '-f', container_name], capture_output=True, timeout=30)

        # Parse outputs
        baseline_output = ""
        agent_output = ""
        if "=== BASELINE OUTPUT ===" in output and "=== AGENT OUTPUT ===" in output:
            parts = output.split("=== BASELINE OUTPUT ===")
            if len(parts) > 1:
                baseline_agent = parts[1].split("=== AGENT OUTPUT ===")
                baseline_output = baseline_agent[0] if len(baseline_agent) > 0 else ""
                agent_output = baseline_agent[1] if len(baseline_agent) > 1 else ""

        # Parse latency metrics
        def parse_latency(text):
            avg_match = re.search(r'Avg latency:\s+([\d.]+)', text)
            p50_match = re.search(r'P50 latency:\s+([\d.]+)', text)
            p99_match = re.search(r'P99 latency:\s+([\d.]+)', text)
            # Also try alternate format from some benchmark versions
            if not avg_match:
                avg_match = re.search(r'avg:\s+([\d.]+)', text, re.IGNORECASE)
            return {
                'latency_avg': float(avg_match.group(1)) if avg_match else None,
                'latency_p50': float(p50_match.group(1)) if p50_match else None,
                'latency_p99': float(p99_match.group(1)) if p99_match else None,
            }

        baseline_metrics = parse_latency(baseline_output or output)
        agent_metrics = parse_latency(agent_output) if "AGENT_BENCHMARK_DONE" in output else {}

        baseline_status = 'success' if baseline_metrics.get('latency_avg') else 'error'
        baseline_result = BenchmarkResult(
            commit_hash=parent_commit, status=baseline_status, benchmark_type='latency',
            model=model, duration_s=duration,
            error=None if baseline_status == 'success' else 'No latency metrics',
            raw_output=baseline_output[-10000:] if baseline_output else output[-10000:],
        )
        # Store latency metrics in ttft fields (reusing existing structure)
        if baseline_metrics.get('latency_avg'):
            baseline_result.ttft_mean = baseline_metrics['latency_avg']
        if baseline_metrics.get('latency_p99'):
            baseline_result.ttft_p99 = baseline_metrics['latency_p99']

        agent_result = None
        if "AGENT_PATCH_APPLIED" in output:
            agent_status = 'success' if agent_metrics.get('latency_avg') else 'error'
            agent_result = BenchmarkResult(
                commit_hash=f"{human_short}_agent", status=agent_status, benchmark_type='latency',
                model=model, duration_s=duration,
                error=None if agent_status == 'success' else 'No latency metrics',
                raw_output=agent_output[-10000:] if agent_output else "",
            )
            if agent_metrics.get('latency_avg'):
                agent_result.ttft_mean = agent_metrics['latency_avg']
            if agent_metrics.get('latency_p99'):
                agent_result.ttft_p99 = agent_metrics['latency_p99']
        elif "AGENT_PATCH_FAILED" in output:
            agent_result = BenchmarkResult(
                commit_hash=f"{human_short}_agent", status='error', benchmark_type='latency',
                model=model, duration_s=duration, error='Agent patch failed to apply'
            )

        return (baseline_result, agent_result)

    except subprocess.TimeoutExpired:
        subprocess.run(['docker', 'rm', '-f', container_name], capture_output=True, timeout=30)
        return (
            BenchmarkResult(commit_hash=parent_commit, status='timeout', benchmark_type='latency',
                          model=model, duration_s=timeout, error=f'Timed out after {timeout}s'),
            None
        )
    except Exception as e:
        subprocess.run(['docker', 'rm', '-f', container_name], capture_output=True, timeout=30)
        return (
            BenchmarkResult(commit_hash=parent_commit, status='error', benchmark_type='latency',
                          model=model, duration_s=time.time() - start_time, error=str(e)),
            None
        )


def save_agent_result(result: BenchmarkResult, human_commit: str, mapping_entry: dict,
                      output_dir: Path = None):
    """Save agent benchmark result."""
    if output_dir is None:
        output_dir = AGENT_OUTPUT_DIR
    output_dir.mkdir(parents=True, exist_ok=True)
    result_file = output_dir / f"{human_commit}_agent_result.json"

    result_data = {
        'human_commit': human_commit,
        'human_commit_full': mapping_entry['human_commit_full'],
        'parent_commit': mapping_entry['parent_commit'],
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
    print(f"  Agent result saved to {result_file}")


def run_baseline_benchmarks(args):
    """Run baseline benchmarks for commits in baseline_benchmark_mapping.json."""
    # Get agent type and output directories
    agent_type = getattr(args, 'agent_type', 'claude_code')
    output_dirs = get_output_dirs(agent_type)

    print(f"=== Agent Type: {agent_type} ===")
    print(f"Output directory: {output_dirs['results']}")

    print("Loading baseline mapping...")
    mapping_file = Path(args.mapping) if args.mapping else None
    mapping = load_baseline_mapping(mapping_file)
    if not mapping:
        return

    print(f"Found {len(mapping)} commits in baseline mapping")

    # Load agent patch mapping for the specified agent type
    print(f"Loading agent patch mapping for {agent_type}...")
    agent_patches = load_agent_patch_mapping(agent_type)
    print(f"Found {len(agent_patches)} agent patches")

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
        print("\nDry run - would run baseline + agent benchmarks for:")
        for m in mapping:
            human_short = m['human_commit_short']
            has_patch = human_short in agent_patches
            print(f"  {human_short} -> parent {m['parent_commit'][:8]} {'[+AGENT]' if has_patch else ''}")
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
        baseline_output_dir = output_dirs['baseline']
        result_file = baseline_output_dir / f"{m['human_commit_short']}_baseline_result.json"
        if result_file.exists():
            print(f"  SKIP: Already have baseline result")
            continue

        # Check if we have an agent patch for this commit
        human_short = m['human_commit_short']
        agent_patch = agent_patches.get(human_short)

        # Pre-build clean baseline image if needed (before any benchmarks or patches)
        if agent_patch:
            if not check_baseline_image_exists(parent_commit):
                print(f"  Pre-building clean baseline image...")
                if not build_baseline_image_only(human_commit, parent_commit):
                    print(f"  SKIP: Failed to build baseline image")
                    continue

        # Run appropriate benchmark
        agent_result = None
        if btype == 'serving' and agent_patch:
            # Use combined function for serving benchmarks with agent patches
            print(f"  Running combined baseline + agent benchmark")
            result, agent_result = run_combined_baseline_agent_serving(
                human_commit, parent_commit, model, perf_command, agent_patch, hf_token
            )
        elif btype == 'throughput' and agent_patch:
            # Use combined function for throughput benchmarks with agent patches
            print(f"  Running combined baseline + agent throughput benchmark")
            result, agent_result = run_combined_baseline_agent_throughput(
                human_commit, parent_commit, model, perf_command, agent_patch, hf_token
            )
        elif btype == 'latency' and agent_patch:
            # Use combined function for latency benchmarks with agent patches
            print(f"  Running combined baseline + agent latency benchmark")
            result, agent_result = run_combined_baseline_agent_latency(
                human_commit, parent_commit, model, perf_command, agent_patch, hf_token
            )
        elif btype == 'serving':
            result = run_baseline_serving_benchmark(human_commit, parent_commit, model, perf_command, hf_token)
        elif btype == 'throughput':
            result = run_baseline_throughput_benchmark(human_commit, parent_commit, model, perf_command, hf_token)
        elif btype == 'latency':
            result = run_baseline_latency_benchmark(human_commit, parent_commit, model, perf_command, hf_token)
        else:
            print(f"  SKIP: Unknown benchmark type: {btype}")
            continue

        # Save with human commit prefix for easy mapping
        baseline_output_dir.mkdir(parents=True, exist_ok=True)
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
            'raw_output': result.raw_output,  # Include for debugging
        }
        with open(result_file, 'w') as f:
            json.dump(result_data, f, indent=2)
        print(f"  Baseline saved to {result_file}")

        results.append(result)

        if result.status == 'success':
            print(f"  BASELINE SUCCESS: {result.duration_s:.1f}s")
            if result.throughput_tok_s:
                print(f"    Throughput: {result.throughput_tok_s:.2f} tok/s")
            if result.ttft_mean:
                print(f"    TTFT: {result.ttft_mean:.2f}ms")
        else:
            print(f"  BASELINE {result.status.upper()}: {result.error}")

        # Save agent result if available
        if agent_result:
            save_agent_result(agent_result, human_short, m, output_dirs['agent'])
            if agent_result.status == 'success':
                print(f"  AGENT SUCCESS")
                if agent_result.throughput_tok_s:
                    print(f"    Throughput: {agent_result.throughput_tok_s:.2f} tok/s")
                if agent_result.ttft_mean:
                    print(f"    TTFT: {agent_result.ttft_mean:.2f}ms")
            else:
                print(f"  AGENT {agent_result.status.upper()}: {agent_result.error}")

    # Summary
    print("\n" + "="*50)
    print("BASELINE + AGENT BENCHMARK SUMMARY")
    print("="*50)
    success = sum(1 for r in results if r.status == 'success')
    errors = sum(1 for r in results if r.status == 'error')
    timeouts = sum(1 for r in results if r.status == 'timeout')
    print(f"Baseline - Success: {success}/{len(results)}")
    print(f"Baseline - Errors: {errors}/{len(results)}")
    print(f"Baseline - Timeouts: {timeouts}/{len(results)}")


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
    parser.add_argument('--mapping', type=str, default=None,
                        help='Path to baseline mapping JSON file (default: baseline_benchmark_mapping.json)')
    parser.add_argument('--agent-type', type=str,
                        choices=list(AGENT_CONFIGS.keys()),
                        default='claude_code',
                        help='Agent type to benchmark (default: claude_code)')
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
