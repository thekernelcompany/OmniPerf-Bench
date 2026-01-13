#!/usr/bin/env python3
"""
Agent-Only Benchmark Runner for OmniPerf.

Runs agent benchmarks using existing baseline/human metrics from Claude Code results.
Applies agent patches to baseline Docker images and runs the benchmark.

Usage:
    python scripts/runners/run_agent_only_benchmark.py --agent codex_gpt5 [--limit N] [--dry-run]
    python scripts/runners/run_agent_only_benchmark.py --agent trae_gpt5 [--skip-existing]
    python scripts/runners/run_agent_only_benchmark.py --agent trae_sonnet45 [--commit COMMIT]
"""

import argparse
import json
import logging
import os
import re
import subprocess
import sys
import tempfile
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Configuration
BASE_DIR = Path(__file__).parent.parent.parent
CC_RESULTS_DIR = BASE_DIR / "omniperf_results_3way_claude_code" / "vllm"
PATCHES_BASE = BASE_DIR / "perf-agents-bench" / "state" / "runs" / "vllm"

AGENT_CONFIGS = {
    "codex_gpt5": {
        "patches_dir": PATCHES_BASE / "codex" / "gpt-5",
        "agent_name": "codex",
        "agent_model": "gpt-5",
        "output_dir": BASE_DIR / "omniperf_results_3way_codex" / "vllm",
    },
    "trae_gpt5": {
        "patches_dir": PATCHES_BASE / "trae" / "gpt-5",
        "agent_name": "trae",
        "agent_model": "gpt-5",
        "output_dir": BASE_DIR / "omniperf_results_3way_trae" / "vllm",
    },
    "trae_sonnet45": {
        "patches_dir": PATCHES_BASE / "trae" / "claude-sonnet-45",
        "agent_name": "trae",
        "agent_model": "claude-sonnet-4.5",
        "output_dir": BASE_DIR / "omniperf_results_3way_trae" / "vllm",
    },
}

# Docker image registries
BASELINE_IMAGE_REGISTRY = "shikhar481/vllm_fixed_human_images"
HUMAN_IMAGE_REGISTRY = "ayushnangia16/nvidia-vllm-docker"

# Available baseline images (from Docker Hub)
AVAILABLE_BASELINES = {
    "7206ce4ce112", "029c71de11bc", "f728ab8e3578", "a869baca73eb", "bc8a8ce5ec37",
    "9bde5ba12709", "f168b8572520", "edc4fa31888b", "ec82c3e388b9", "eb5741ad422f",
    "e642ec962cf2", "dd572c0ab3ef", "d263bd9df7b2", "cf2f084d56a1", "cc466a32903d",
    "c8d70e2437fe", "c34eeec58d3a", "acaea3bb0788", "a6d795d59304", "90f1e55421f1",
    "8d59dbb00044", "8c1e77fb585c", "8bb43b9c9ee8", "88faa466d788", "7d94577138e3",
    "76b494444fd8", "733e7c9e95f5", "67abdbb42fdb", "63d635d17962", "5fc5ce0fe45f",
    "5b8a1fde8422", "526078a96c52", "4ce64e2df486", "4a18fd14ba4a", "3e1c76cf3a87",
    "3d925165f2b1", "3cdfe1f38b2c", "3cd91dc9555e", "3a1e6481586e", "333681408fea",
    "3014c920dae5", "20478c4d3abc", "2f385183f354", "270a5da495d2", "25373b6c6cc2",
    "1da8f0e1ddda", "10904e6d7550", "067c34a15594", "005ae9be6c22", "a4d577b37944",
    "95baec828f3e", "2b04c209ee98", "f508e03e7f2d", "1d35662e6dc1", "f721096d48a7",
    "51f8aa90ad40", "6dd55af6c9dd", "beebf4742af8", "5c04bb8b863b", "70363bccfac1",
    "388596c91437", "0fca3cdcf265", "084a01fd3544", "bd43973522ea", "51e971d39e12",
    "dd2a6a82e3f4", "95a178f86120", "6a11fdfbb8d6", "64172a976c8d", "ebce310b7433",
    "b0e96aaebbfb", "fbefc8a78d22", "f1c852014603", "0032903a5bb7", "36fb68f94792",
    "54600709b6d4"
}

# Setup logging
def setup_logging(agent_key: str) -> logging.Logger:
    log_dir = BASE_DIR / "omniperf_results" / "agent_run_logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"{agent_key}_run_{timestamp}.log"

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout),
        ]
    )
    logger = logging.getLogger(__name__)
    logger.info(f"Log file: {log_file}")
    return logger


def load_claude_code_results() -> Dict[str, Dict]:
    """Load all existing Claude Code benchmark results."""
    results = {}
    if not CC_RESULTS_DIR.exists():
        return results

    for commit_dir in CC_RESULTS_DIR.iterdir():
        if not commit_dir.is_dir():
            continue
        result_file = commit_dir / "benchmark_result.json"
        if result_file.exists():
            try:
                with open(result_file) as f:
                    data = json.load(f)
                    commit_hash = data.get("instance", {}).get("commit_hash", "")
                    if commit_hash:
                        results[commit_hash[:8]] = {
                            "full_commit": commit_hash,
                            "short_commit": commit_hash[:8],
                            "data": data,
                        }
            except Exception as e:
                print(f"Error loading {result_file}: {e}")

    return results


def build_patch_mapping(patches_dir: Path) -> Dict[str, Dict]:
    """Build mapping from commit hash to patch info."""
    mapping = {}

    if not patches_dir.exists():
        return mapping

    for run_summary_path in patches_dir.rglob("run_summary.json"):
        try:
            with open(run_summary_path) as f:
                summary = json.load(f)

            human_commit = summary.get("commits", {}).get("human", "")
            pre_commit = summary.get("commits", {}).get("pre", "")
            agent_status = summary.get("agent", {}).get("status", "")
            patch_generated = summary.get("agent", {}).get("patch_generated", False)

            if not human_commit:
                continue

            short_commit = human_commit[:8]
            patch_path = run_summary_path.parent / "model_patch.diff"

            # Check if patch file exists and is non-empty
            patch_exists = patch_path.exists() and patch_path.stat().st_size > 0

            # Store the mapping (prefer successful runs with patches)
            existing = mapping.get(short_commit)
            if existing is None or (patch_generated and patch_exists and not existing.get("has_patch")):
                mapping[short_commit] = {
                    "full_commit": human_commit,
                    "short_commit": short_commit,
                    "parent_commit": pre_commit,
                    "patch_path": str(patch_path) if patch_exists else None,
                    "has_patch": patch_generated and patch_exists,
                    "agent_status": agent_status,
                    "run_summary_path": str(run_summary_path),
                }
        except Exception as e:
            print(f"Error processing {run_summary_path}: {e}")

    return mapping


def parse_benchmark_output(output: str, benchmark_mode: str) -> Dict[str, float]:
    """Parse benchmark output to extract metrics."""
    metrics = {}

    if benchmark_mode == "serving":
        # Parse vLLM serving benchmark output
        patterns = {
            "ttft_mean": r"Mean TTFT \(ms\):\s+([0-9.]+)",
            "ttft_median": r"Median TTFT \(ms\):\s+([0-9.]+)",
            "ttft_p99": r"P99 TTFT \(ms\):\s+([0-9.]+)",
            "tpot_mean": r"Mean TPOT \(ms\):\s+([0-9.]+)",
            "tpot_median": r"Median TPOT \(ms\):\s+([0-9.]+)",
            "tpot_p99": r"P99 TPOT \(ms\):\s+([0-9.]+)",
            "itl_mean": r"Mean ITL \(ms\):\s+([0-9.]+)",
            "itl_median": r"Median ITL \(ms\):\s+([0-9.]+)",
            "itl_p99": r"P99 ITL \(ms\):\s+([0-9.]+)",
            "throughput": r"Request throughput.*?:\s+([0-9.]+)",
        }
    else:
        # Parse standalone benchmark output (benchmark_latency.py)
        patterns = {
            "latency_avg": r"Avg latency:\s+([0-9.]+)\s+(?:ms|seconds)",
            "throughput": r"Throughput:\s+([0-9.]+)\s+(?:tokens/s|requests/s)",
        }
        # Alternative patterns
        alt_patterns = {
            "latency_avg": r"latency[:\s]+([0-9.]+)",
        }

    for key, pattern in patterns.items():
        match = re.search(pattern, output, re.IGNORECASE)
        if match:
            metrics[key] = float(match.group(1))

    return metrics


def run_docker_benchmark(
    baseline_image: str,
    patch_content: Optional[str],
    perf_command: str,
    gpu_config: str = "H100:1",
    timeout: int = 1800,
    logger: logging.Logger = None,
) -> Tuple[Dict[str, float], str, Optional[str]]:
    """Run benchmark in Docker container with optional patch."""

    # Parse GPU config
    gpu_type, gpu_count = gpu_config.split(":")
    gpu_count = int(gpu_count)

    # Create container name
    container_name = f"agent_bench_{int(time.time())}"

    # Build docker run command
    docker_cmd = [
        "docker", "run",
        "--name", container_name,
        "--gpus", f'"device={",".join(str(i) for i in range(gpu_count))}"',
        "--rm",
        "-e", "HF_TOKEN",
        "-e", "HUGGING_FACE_HUB_TOKEN",
        "--shm-size=16g",
        "--ulimit", "memlock=-1",
        "--network", "host",
    ]

    # Build the command to run inside container
    if patch_content:
        # Apply patch then run benchmark
        inner_cmd = f"""
cd /workspace/vllm && \\
echo '{patch_content}' | base64 -d | git apply --whitespace=nowarn - 2>/dev/null || true && \\
pip install -e . --no-build-isolation 2>/dev/null || true && \\
{perf_command}
"""
    else:
        inner_cmd = f"cd /workspace/vllm && {perf_command}"

    docker_cmd.extend([
        baseline_image,
        "bash", "-c", inner_cmd,
    ])

    if logger:
        logger.info(f"Running: {' '.join(docker_cmd[:15])}...")

    try:
        result = subprocess.run(
            " ".join(docker_cmd),
            shell=True,
            capture_output=True,
            text=True,
            timeout=timeout,
            env={**os.environ},
        )

        output = result.stdout + result.stderr

        # Determine benchmark mode from command
        if "benchmark_serving" in perf_command:
            benchmark_mode = "serving"
        else:
            benchmark_mode = "standalone"

        metrics = parse_benchmark_output(output, benchmark_mode)

        return metrics, output, None

    except subprocess.TimeoutExpired:
        # Kill container if still running
        subprocess.run(["docker", "kill", container_name], capture_output=True)
        return {}, "", "timeout"
    except Exception as e:
        return {}, "", str(e)


def run_local_benchmark(
    baseline_image: str,
    patch_path: Optional[str],
    perf_command: str,
    gpu_config: str = "H100:1",
    timeout: int = 1800,
    logger: logging.Logger = None,
) -> Tuple[Dict[str, float], str, Optional[str]]:
    """Run benchmark locally using Docker with GPU support."""

    # Parse GPU config
    gpu_parts = gpu_config.split(":")
    gpu_count = int(gpu_parts[1]) if len(gpu_parts) > 1 else 1

    # Read patch content if available
    patch_content = None
    if patch_path and Path(patch_path).exists():
        try:
            with open(patch_path, 'r') as f:
                patch_content = f.read()
            if patch_content:
                import base64
                patch_content = base64.b64encode(patch_content.encode()).decode()
        except Exception as e:
            if logger:
                logger.warning(f"Could not read patch: {e}")

    # Generate unique container name
    container_name = f"agent_bench_{int(time.time())}"

    # Build docker command
    gpu_devices = ",".join(str(i) for i in range(gpu_count))

    # Prepare inner command
    if patch_content:
        inner_cmd = f"""
set -e
cd /workspace/vllm 2>/dev/null || cd /vllm 2>/dev/null || cd /app 2>/dev/null || true
echo '{patch_content}' | base64 -d > /tmp/patch.diff
git apply --whitespace=nowarn /tmp/patch.diff 2>&1 || echo "Patch apply warning (continuing)"
pip install -e . --no-build-isolation -q 2>&1 || echo "Reinstall warning (continuing)"
{perf_command}
"""
    else:
        inner_cmd = f"""
set -e
cd /workspace/vllm 2>/dev/null || cd /vllm 2>/dev/null || cd /app 2>/dev/null || true
{perf_command}
"""

    docker_cmd = f'''docker run --name {container_name} \\
        --gpus '"device={gpu_devices}"' \\
        --rm \\
        -e HF_TOKEN="${{HF_TOKEN}}" \\
        -e HUGGING_FACE_HUB_TOKEN="${{HF_TOKEN}}" \\
        --shm-size=16g \\
        --ulimit memlock=-1 \\
        --network host \\
        {baseline_image} \\
        bash -c '{inner_cmd}'
'''

    if logger:
        logger.info(f"Container: {container_name}")
        logger.info(f"Image: {baseline_image}")
        logger.info(f"GPUs: {gpu_devices}")
        logger.info(f"Has patch: {patch_content is not None}")

    start_time = time.time()

    try:
        result = subprocess.run(
            docker_cmd,
            shell=True,
            capture_output=True,
            text=True,
            timeout=timeout,
            env=os.environ.copy(),
        )

        elapsed = time.time() - start_time
        output = result.stdout + "\n" + result.stderr

        # Determine benchmark mode
        if "benchmark_serving" in perf_command:
            benchmark_mode = "serving"
        else:
            benchmark_mode = "standalone"

        metrics = parse_benchmark_output(output, benchmark_mode)

        if logger:
            logger.info(f"Benchmark completed in {elapsed:.1f}s")
            if metrics:
                logger.info(f"Metrics: {metrics}")
            else:
                logger.warning("No metrics parsed from output")
                logger.debug(f"Raw output (last 500 chars): {output[-500:]}")

        return metrics, output, None

    except subprocess.TimeoutExpired:
        subprocess.run(["docker", "kill", container_name], capture_output=True, check=False)
        subprocess.run(["docker", "rm", "-f", container_name], capture_output=True, check=False)
        if logger:
            logger.error(f"Benchmark timed out after {timeout}s")
        return {}, "", "timeout"

    except Exception as e:
        subprocess.run(["docker", "rm", "-f", container_name], capture_output=True, check=False)
        if logger:
            logger.error(f"Benchmark failed: {e}")
        return {}, "", str(e)


def calculate_improvement(baseline: float, optimized: float) -> float:
    """Calculate percentage improvement (positive = better)."""
    if baseline == 0:
        return 0.0
    # For latency metrics, lower is better
    return ((baseline - optimized) / baseline) * 100


def check_baseline_available(parent_commit: str) -> bool:
    """Check if baseline image is available."""
    parent_short = parent_commit[:12] if parent_commit else ""
    return parent_short in AVAILABLE_BASELINES


def run_agent_benchmark(
    commit: str,
    cc_result: Dict,
    patch_info: Dict,
    agent_config: Dict,
    logger: logging.Logger,
    timeout: int = 1800,
) -> Dict[str, Any]:
    """Run a single agent benchmark."""

    cc_data = cc_result["data"]
    instance = cc_data.get("instance", {})
    result = cc_data.get("result", {})

    # Extract existing metrics
    baseline_metrics = result.get("baseline_metrics", {})
    human_metrics = result.get("human_metrics", {})
    parent_commit = result.get("parent_commit", patch_info.get("parent_commit", ""))
    perf_command = result.get("perf_command", instance.get("perf_command", ""))
    gpu_config = result.get("gpu_config", "H100:1")
    benchmark_mode = result.get("benchmark_mode", "standalone")

    # Check baseline availability
    if not check_baseline_available(parent_commit):
        logger.warning(f"  No baseline image available for parent {parent_commit[:12]}")
        return {
            "status": "no_baseline_image",
            "baseline_metrics": baseline_metrics,
            "human_metrics": human_metrics,
            "agent_metrics": None,
            "error": f"No baseline image for parent {parent_commit[:12]}",
        }

    # Get baseline image
    parent_short = parent_commit[:12] if parent_commit else ""
    baseline_image = f"{BASELINE_IMAGE_REGISTRY}:baseline-{parent_short}"

    logger.info(f"  Parent commit: {parent_short}")
    logger.info(f"  Baseline image: {baseline_image}")
    logger.info(f"  GPU config: {gpu_config}")
    logger.info(f"  Benchmark mode: {benchmark_mode}")
    logger.info(f"  Has agent patch: {patch_info.get('has_patch', False)}")

    # Check if we have a patch to apply
    if not patch_info.get("has_patch"):
        logger.warning("  No agent patch available - recording as no_patch")
        return {
            "status": "no_patch",
            "baseline_metrics": baseline_metrics,
            "human_metrics": human_metrics,
            "agent_metrics": None,
            "error": "No agent patch generated",
            "agent_status": patch_info.get("agent_status", "unknown"),
        }

    # Run agent benchmark
    logger.info(f"  Running agent benchmark...")
    start_time = time.time()

    agent_metrics, raw_output, error = run_local_benchmark(
        baseline_image=baseline_image,
        patch_path=patch_info.get("patch_path"),
        perf_command=perf_command,
        gpu_config=gpu_config,
        timeout=timeout,
        logger=logger,
    )

    duration = time.time() - start_time

    if error:
        logger.error(f"  Agent benchmark failed: {error}")
        return {
            "status": "agent_error",
            "baseline_metrics": baseline_metrics,
            "human_metrics": human_metrics,
            "agent_metrics": None,
            "error": error,
            "duration_s": duration,
            "agent_raw": raw_output[-2000:] if raw_output else None,
        }

    if not agent_metrics:
        logger.warning("  No metrics extracted from agent benchmark")
        return {
            "status": "no_metrics",
            "baseline_metrics": baseline_metrics,
            "human_metrics": human_metrics,
            "agent_metrics": None,
            "error": "Could not parse metrics from output",
            "duration_s": duration,
            "agent_raw": raw_output[-2000:] if raw_output else None,
        }

    # Calculate improvements
    human_improvement = {}
    agent_improvement = {}
    agent_vs_human = {}

    for metric in baseline_metrics:
        baseline_val = baseline_metrics.get(metric, 0)
        human_val = human_metrics.get(metric, 0) if human_metrics else 0
        agent_val = agent_metrics.get(metric, 0)

        if baseline_val > 0:
            human_improvement[metric] = calculate_improvement(baseline_val, human_val)
            agent_improvement[metric] = calculate_improvement(baseline_val, agent_val)

        if human_val > 0 and agent_val > 0:
            agent_vs_human[metric] = calculate_improvement(human_val, agent_val)

    logger.info(f"  Agent benchmark completed in {duration:.1f}s")
    logger.info(f"  Agent metrics: {agent_metrics}")

    return {
        "status": "success",
        "baseline_metrics": baseline_metrics,
        "human_metrics": human_metrics,
        "agent_metrics": agent_metrics,
        "human_improvement": human_improvement,
        "agent_improvement": agent_improvement,
        "agent_vs_human": agent_vs_human,
        "duration_s": duration,
        "agent_raw": raw_output[-2000:] if raw_output else None,
        "error": None,
    }


def main():
    parser = argparse.ArgumentParser(description="Agent-Only Benchmark Runner")
    parser.add_argument("--agent", type=str, required=True,
                        choices=list(AGENT_CONFIGS.keys()),
                        help="Agent configuration to run")
    parser.add_argument("--limit", type=int, default=0,
                        help="Limit number of benchmarks to run")
    parser.add_argument("--commit", type=str,
                        help="Run only a specific commit (8 char)")
    parser.add_argument("--skip-existing", action="store_true",
                        help="Skip commits that already have results")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print plan without running")
    parser.add_argument("--timeout", type=int, default=1800,
                        help="Timeout per benchmark in seconds")
    args = parser.parse_args()

    # Setup logging
    logger = setup_logging(args.agent)

    agent_config = AGENT_CONFIGS[args.agent]
    output_dir = agent_config["output_dir"]
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 80)
    logger.info(f"AGENT-ONLY BENCHMARK RUN: {args.agent}")
    logger.info(f"Agent: {agent_config['agent_name']} / {agent_config['agent_model']}")
    logger.info(f"Output: {output_dir}")
    logger.info("=" * 80)

    # Load Claude Code results
    logger.info("\nLoading Claude Code reference results...")
    cc_results = load_claude_code_results()
    logger.info(f"  Found {len(cc_results)} Claude Code results")

    # Build patch mapping
    logger.info(f"\nBuilding patch mapping from {agent_config['patches_dir']}...")
    patch_mapping = build_patch_mapping(agent_config["patches_dir"])
    logger.info(f"  Found {len(patch_mapping)} commit mappings")

    # Find overlap
    overlap_commits = set(cc_results.keys()) & set(patch_mapping.keys())
    logger.info(f"  Overlap with Claude Code: {len(overlap_commits)} commits")

    # Filter commits with patches
    commits_with_patches = [c for c in overlap_commits if patch_mapping[c].get("has_patch")]
    logger.info(f"  Commits with patches: {len(commits_with_patches)}")

    # Build run list
    commits_to_run = sorted(overlap_commits)

    if args.commit:
        if args.commit in commits_to_run:
            commits_to_run = [args.commit]
        else:
            logger.error(f"Commit {args.commit} not found in overlap")
            return

    if args.skip_existing:
        existing = set()
        for d in output_dir.iterdir():
            if d.is_dir() and (d / "benchmark_result.json").exists():
                existing.add(d.name)
        commits_to_run = [c for c in commits_to_run if c not in existing]
        logger.info(f"  After skipping existing: {len(commits_to_run)} commits")

    if args.limit > 0:
        commits_to_run = commits_to_run[:args.limit]
        logger.info(f"  Limited to: {args.limit} commits")

    # Dry run
    if args.dry_run:
        logger.info("\n### DRY RUN - Would process:\n")
        for i, commit in enumerate(commits_to_run, 1):
            patch_info = patch_mapping.get(commit, {})
            has_patch = "✓" if patch_info.get("has_patch") else "✗"
            logger.info(f"  {i}. {commit} [patch: {has_patch}]")
        return

    # Run benchmarks
    results = []
    success_count = 0
    error_count = 0
    no_patch_count = 0
    total = len(commits_to_run)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    for i, commit in enumerate(commits_to_run, 1):
        logger.info(f"\n{'='*80}")
        logger.info(f"[{i}/{total}] BENCHMARK: {commit}")
        logger.info("=" * 80)

        cc_result = cc_results[commit]
        patch_info = patch_mapping.get(commit, {"has_patch": False})

        # Run benchmark
        result = run_agent_benchmark(
            commit=commit,
            cc_result=cc_result,
            patch_info=patch_info,
            agent_config=agent_config,
            logger=logger,
            timeout=args.timeout,
        )

        # Add metadata
        result["commit"] = commit
        result["full_commit"] = cc_result["full_commit"]
        result["agent_name"] = agent_config["agent_name"]
        result["agent_model"] = agent_config["agent_model"]

        # Copy instance data from Claude Code
        result["instance"] = cc_result["data"].get("instance", {})
        result["gpu_config"] = cc_result["data"].get("result", {}).get("gpu_config", "H100:1")
        result["benchmark_mode"] = cc_result["data"].get("result", {}).get("benchmark_mode", "standalone")
        result["parent_commit"] = cc_result["data"].get("result", {}).get("parent_commit", "")
        result["perf_command"] = cc_result["data"].get("result", {}).get("perf_command", "")

        # Track status
        status = result.get("status", "unknown")
        if status == "success":
            success_count += 1
            logger.info(f"  ✓ SUCCESS")
        elif status == "no_patch":
            no_patch_count += 1
            logger.info(f"  ○ NO PATCH")
        else:
            error_count += 1
            logger.error(f"  ✗ {status.upper()}")

        results.append(result)

        # Save individual result
        commit_dir = output_dir / commit
        commit_dir.mkdir(parents=True, exist_ok=True)
        with open(commit_dir / "benchmark_result.json", "w") as f:
            json.dump({
                "instance": result["instance"],
                "result": {k: v for k, v in result.items() if k != "instance"},
            }, f, indent=2, default=str)

        # Save incremental summary
        summary_file = output_dir / f"{args.agent}_run_{timestamp}.json"
        with open(summary_file, "w") as f:
            json.dump({
                "agent": args.agent,
                "timestamp": timestamp,
                "total": total,
                "completed": i,
                "success": success_count,
                "no_patch": no_patch_count,
                "errors": error_count,
                "results": results,
            }, f, indent=2, default=str)

        logger.info(f"  Progress: {success_count} success, {no_patch_count} no_patch, {error_count} errors [{i}/{total}]")

    # Final summary
    logger.info("\n" + "=" * 80)
    logger.info("RUN COMPLETE")
    logger.info("=" * 80)
    logger.info(f"Total commits: {total}")
    logger.info(f"Success: {success_count} ({100*success_count/max(total,1):.1f}%)")
    logger.info(f"No patch: {no_patch_count} ({100*no_patch_count/max(total,1):.1f}%)")
    logger.info(f"Errors: {error_count} ({100*error_count/max(total,1):.1f}%)")
    logger.info(f"Results saved to: {output_dir}")


if __name__ == "__main__":
    main()
