#!/usr/bin/env python3
"""
SGLang True 3-Way Local Docker Benchmark Runner

Runs actual 3-way benchmarks (baseline vs human vs agent) for SGLang commits
using SEPARATE Docker images for baseline and human commits.

This runs locally using docker and nvidia-container-toolkit.
No cloud infrastructure (Modal) required.

Docker Repository: shikhar481/sglang-images
- Baseline: shikhar481/sglang-images:{parent_commit_hash}
- Human: shikhar481/sglang-images:{human_commit_hash}
- Agent: Baseline image + agent patch overlay

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
DOCKER_REPO = "shikhar481/sglang-images"
RESULTS_DIR = Path("omniperf_results_3way_sglang_docker")
CANDIDATES_FILE = Path("/tmp/sglang_3way_candidates.json")

# Benchmark parameters
SERVER_PORT = 30000
SERVER_TIMEOUT = 600  # seconds to wait for server to start
BENCHMARK_TIMEOUT = 900  # seconds for benchmark to complete


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
    """Parse benchmark output to extract metrics."""
    metrics = {}

    patterns = {
        "request_throughput": r"Request throughput.*?(\d+\.?\d*)\s*req/s",
        "output_token_throughput": r"Output token throughput.*?(\d+\.?\d*)\s*tok/s",
        "total_token_throughput": r"Total token throughput.*?(\d+\.?\d*)\s*tok/s",
        "mean_ttft_ms": r"Mean TTFT.*?(\d+\.?\d*)\s*ms",
        "median_ttft_ms": r"Median TTFT.*?(\d+\.?\d*)\s*ms",
        "mean_tpot_ms": r"Mean TPOT.*?(\d+\.?\d*)\s*ms",
        "median_tpot_ms": r"Median TPOT.*?(\d+\.?\d*)\s*ms",
        "mean_itl_ms": r"Mean ITL.*?(\d+\.?\d*)\s*ms",
        "median_itl_ms": r"Median ITL.*?(\d+\.?\d*)\s*ms",
    }

    for key, pattern in patterns.items():
        match = re.search(pattern, output, re.IGNORECASE)
        if match:
            try:
                metrics[key] = float(match.group(1))
            except:
                pass

    return metrics


def run_benchmark_in_docker(
    image: str,
    model: str,
    perf_command: str,
    phase: str,
    agent_patch: str = None,
    hf_token: str = None,
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
        docker_cmd = [
            "docker", "run",
            "--rm",
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

        # Create the benchmark script to run inside container
        benchmark_script = f'''
#!/bin/bash
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
python3 -m sglang.launch_server \\
    --model-path {model} \\
    --port {SERVER_PORT} \\
    --host 0.0.0.0 \\
    2>&1 | tee /tmp/server.log &

SERVER_PID=$!

# Wait for server to be ready
echo "Waiting for server..."
for i in $(seq 1 {SERVER_TIMEOUT // 5}); do
    if curl -s http://127.0.0.1:{SERVER_PORT}/health > /dev/null 2>&1; then
        echo "Server ready!"
        break
    fi
    sleep 5
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
if [[ "$BENCH_CMD" != *"--port"* ]]; then
    BENCH_CMD="$BENCH_CMD --port {SERVER_PORT}"
fi
if [[ "$BENCH_CMD" != *"--host"* ]]; then
    BENCH_CMD="$BENCH_CMD --host 127.0.0.1"
fi

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

        # Run docker with bash script
        full_cmd = docker_cmd + ["bash", "-c", benchmark_script]

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
) -> Dict[str, Any]:
    """
    Run true 3-way benchmark using local Docker.

    Returns combined results from all three phases.
    """
    human_short = candidate["human"]
    parent_short = candidate["parent"]
    model = candidate["model"]
    perf_command = candidate["perf_command"]

    # Read agent patch
    patch_path = candidate.get("patch_path")
    agent_patch = ""
    if patch_path and Path(patch_path).exists():
        agent_patch = Path(patch_path).read_text()

    result = {
        "commit": human_short,
        "human_commit": candidate.get("human_full", human_short),
        "parent_commit": candidate.get("parent_full", parent_short),
        "model": model,
        "perf_command": perf_command,
        "subject": candidate.get("subject", ""),
        "status": "error",
        "error": None,
        "baseline_metrics": {},
        "human_metrics": {},
        "agent_metrics": {},
        "human_improvement": {},
        "agent_improvement": {},
        "agent_vs_human": {},
        "duration_s": 0,
        "benchmark_mode": "3way_local_docker",
    }

    start_time = time.time()

    # Docker images
    baseline_image = f"{DOCKER_REPO}:{parent_short}"
    human_image = f"{DOCKER_REPO}:{human_short}"

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
    )
    result["baseline_metrics"] = baseline_result["metrics"]
    result["baseline_raw"] = baseline_result["raw_output"]
    result["baseline_server_logs"] = baseline_result.get("server_logs", "")

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
    )
    result["human_metrics"] = human_result["metrics"]
    result["human_raw"] = human_result["raw_output"]
    result["human_server_logs"] = human_result.get("server_logs", "")

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
        )
        result["agent_metrics"] = agent_result["metrics"]
        result["agent_raw"] = agent_result["raw_output"]
        result["agent_server_logs"] = agent_result.get("server_logs", "")

    # Calculate improvements
    baseline = result["baseline_metrics"]
    human = result["human_metrics"]
    agent = result["agent_metrics"]

    def calc_improvement(base_val, new_val, is_throughput=True):
        """Calculate % improvement. Positive = better."""
        if not base_val or base_val == 0:
            return None
        if is_throughput:
            # Higher is better
            return round(((new_val - base_val) / base_val) * 100, 2)
        else:
            # Lower is better (latency)
            return round(((base_val - new_val) / base_val) * 100, 2)

    if baseline and human:
        for key in baseline:
            if key in human:
                is_throughput = "throughput" in key.lower()
                improvement = calc_improvement(baseline[key], human[key], is_throughput)
                if improvement is not None:
                    result["human_improvement"][key] = improvement

    if baseline and agent:
        for key in baseline:
            if key in agent:
                is_throughput = "throughput" in key.lower()
                improvement = calc_improvement(baseline[key], agent[key], is_throughput)
                if improvement is not None:
                    result["agent_improvement"][key] = improvement

    if human and agent:
        for key in human:
            if key in agent:
                pct = round(((agent[key] - human[key]) / human[key]) * 100, 2) if human[key] else 0
                result["agent_vs_human"][key] = pct

    # Determine overall status
    if result["baseline_metrics"] and result["human_metrics"]:
        result["status"] = "success"
    elif result["human_metrics"]:
        result["status"] = "partial"
        result["error"] = "Baseline failed"
    elif result["baseline_metrics"]:
        result["status"] = "partial"
        result["error"] = "Human failed"
    else:
        result["status"] = "error"
        result["error"] = "All phases failed"

    result["duration_s"] = time.time() - start_time

    return result


def save_result(result: Dict, results_dir: Path):
    """Save benchmark result to JSON file."""
    commit = result.get("commit", "unknown")
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
        print(f"Docker repo: {DOCKER_REPO}")
        print(f"Agent phase: {'ENABLED' if not args.no_agent else 'DISABLED'}")
        print()
        for i, c in enumerate(candidates, 1):
            bench_type = "server" if "bench_serving" in c["perf_command"].lower() else "direct"
            print(f"{i}. {c['human']} (parent: {c['parent']})")
            print(f"   Subject: {c['subject']}")
            print(f"   Model: {c['model']}")
            print(f"   Type: {bench_type}")
            print(f"   Baseline: {DOCKER_REPO}:{c['parent']}")
            print(f"   Human: {DOCKER_REPO}:{c['human']}")
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
        logger.info(f"Subject: {candidate['subject']}")
        logger.info(f"{'='*60}")

        try:
            result = run_3way_benchmark(
                candidate=candidate,
                hf_token=hf_token,
                include_agent=not args.no_agent,
            )

            save_result(result, RESULTS_DIR)

            if result["status"] == "success":
                success_count += 1
                logger.info(f"SUCCESS ({result['duration_s']:.1f}s)")
                if result.get("human_improvement"):
                    logger.info(f"  Human improvement: {result['human_improvement']}")
                if result.get("agent_improvement"):
                    logger.info(f"  Agent improvement: {result['agent_improvement']}")
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
