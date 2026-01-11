#!/usr/bin/env python3
"""
Hero Benchmark Runner for OmniPerf vLLM Dataset.

Runs all 40 runnable commits on Modal with comprehensive logging.

Usage:
    python hero_benchmark_runner.py [--limit N] [--start-from COMMIT] [--dry-run]

Prerequisites:
    - modal deploy src/eval/modal_benchmark.py
    - HuggingFace token configured
"""

import json
import logging
import os
import re
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Setup comprehensive logging
log_dir = Path("omniperf_results/hero_run_logs")
log_dir.mkdir(parents=True, exist_ok=True)
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
log_file = log_dir / f"hero_run_{timestamp}.log"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler(sys.stdout),
    ]
)
logger = logging.getLogger(__name__)

# Results directory
RESULTS_DIR = Path("omniperf_results/hero_run")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# Wheel URL pattern
VLLM_WHEEL_URL = "https://vllm-wheels.s3.us-west-2.amazonaws.com/{commit}/vllm-1.0.0.dev-cp38-abi3-manylinux1_x86_64.whl"

# Large models that need multi-GPU
# Pattern matching is case-insensitive and uses substring matching
LARGE_MODEL_GPU_MAP = {
    # DeepSeek models
    "deepseek-v3": "H100:8",
    "deepseek-v2": "H100:4",
    # Mixtral MoE models (47B-176B params)
    "mixtral-8x22b": "H100:4",
    "mixtral-8x7b": "H100:2",
    # Llama 70B+ models
    "llama-3-70b": "H100:4",
    "llama-3.1-70b": "H100:4",
    "llama-3.2-70b": "H100:4",
    "llama-2-70b": "H100:4",
    "meta-llama-3-70b": "H100:4",
    "llama-4-scout-17b-16e": "H100:2",
    # Qwen large models
    "qwen-72b": "H100:4",
    "qwen2-72b": "H100:4",
    "qwen2.5-72b": "H100:4",
    "qwen3-30b": "H100:2",  # 30B A3B MoE needs 2 GPUs
    "qwen3-235b": "H100:8",
    # Other large models
    "nemotron-4-340b": "H100:8",
    "falcon-180b": "H100:8",
    "yi-34b": "H100:2",
    "codellama-70b": "H100:4",
    "dbrx": "H100:4",  # DBRX is a 132B MoE
}

# Commits with special handling
TIMEOUT_ADJUSTMENT = {
    "a3223766": {"request_rate": 20},  # Reduce from 200 to 20
}

# Model name fixes - map incorrect model names to correct HuggingFace paths
MODEL_NAME_FIXES = {
    "Qwen/Qwen3-7B-Instruct": "mergekit-community/Qwen3-7B-Instruct",  # Correct path
    "Qwen/Qwen3-30B-A3B": "Qwen/Qwen3-30B-A3B-Instruct-2507",  # Correct path
}


def validate_wheel_url(url: str) -> bool:
    """Check if wheel URL exists on S3."""
    import requests
    try:
        resp = requests.head(url, timeout=10)
        return resp.status_code == 200
    except Exception:
        return False


def get_parent_commit(commit: str) -> Optional[str]:
    """Get parent commit from vllm repo."""
    try:
        result = subprocess.run(
            ["git", "-C", "vllm", "rev-parse", f"{commit}^"],
            capture_output=True, text=True, timeout=10
        )
        return result.stdout.strip() if result.returncode == 0 else None
    except Exception as e:
        logger.error(f"Failed to get parent commit: {e}")
        return None


def get_gpu_config(model: str, perf_command: str) -> str:
    """Determine GPU requirements."""
    # Check tensor parallelism in command
    tp_match = re.search(r'(?:-tp|--tensor-parallel-size)\s+(\d+)', perf_command)
    if tp_match:
        tp = int(tp_match.group(1))
        if tp >= 8:
            return "H100:8"
        elif tp >= 4:
            return "H100:4"
        elif tp >= 2:
            return "H100:2"

    # Check large models
    model_lower = model.lower()
    for pattern, config in LARGE_MODEL_GPU_MAP.items():
        if pattern in model_lower:
            return config

    return "H100:1"


def adjust_command(commit: str, perf_command: str) -> str:
    """Apply any needed adjustments to the command."""
    cmd = perf_command

    if commit[:8] in TIMEOUT_ADJUSTMENT:
        adj = TIMEOUT_ADJUSTMENT[commit[:8]]
        if "request_rate" in adj:
            old_rate = re.search(r'--request-rate\s+(\d+)', cmd)
            if old_rate:
                cmd = cmd.replace(f"--request-rate {old_rate.group(1)}",
                                 f"--request-rate {adj['request_rate']}")
                logger.info(f"  Adjusted request-rate: {old_rate.group(1)} -> {adj['request_rate']}")

    return cmd


def run_benchmark(
    commit: str,
    parent_commit: str,
    perf_command: str,
    model: str,
    gpu_config: str,
) -> Dict[str, Any]:
    """Run a single benchmark on Modal."""
    from src.eval.modal_benchmark import run_3way_modal_benchmark

    baseline_url = VLLM_WHEEL_URL.format(commit=parent_commit)
    human_url = VLLM_WHEEL_URL.format(commit=commit)

    logger.info(f"  Baseline wheel: {parent_commit[:8]}")
    logger.info(f"  Human wheel: {commit[:8]}")
    logger.info(f"  GPU config: {gpu_config}")
    logger.info(f"  Command preview: {perf_command[:100]}...")

    start_time = time.time()

    try:
        result = run_3way_modal_benchmark(
            baseline_wheel_url=baseline_url,
            human_wheel_url=human_url,
            agent_patch=None,
            perf_command=perf_command,
            model=model,
            gpu_config=gpu_config,
            base_commit=parent_commit,
            human_commit=commit,  # CRITICAL: Enables build-from-source fallback for 404 wheels
        )

        result["duration_s"] = time.time() - start_time
        return result

    except Exception as e:
        logger.error(f"  Benchmark failed with exception: {e}")
        return {
            "status": "exception",
            "error": str(e),
            "duration_s": time.time() - start_time,
        }


def run_single_benchmark_task(
    commit_info: Dict[str, Any],
    ds_map: Dict[str, Any],
    results_dir: Path,
    timestamp: str,
) -> Dict[str, Any]:
    """Run a single benchmark - used for parallel execution.

    This function is self-contained and can be called from ThreadPoolExecutor.
    """
    commit = commit_info["full_commit"]
    short_commit = commit[:8]

    # Get full dataset info
    if commit not in ds_map:
        logger.error(f"[{short_commit}] Commit not found in dataset!")
        return {
            "commit": short_commit,
            "full_commit": commit,
            "status": "error",
            "error": "Commit not found in dataset",
        }

    row = ds_map[commit]
    perf_command = row.get("perf_command", "")

    # Extract model from command
    model_match = re.search(r'--model[=\s]+["\'"]?([^\s"\']+)', perf_command)
    model = model_match.group(1) if model_match else "unknown"

    # Fix model name if needed
    if model in MODEL_NAME_FIXES:
        fixed_model = MODEL_NAME_FIXES[model]
        logger.info(f"[{short_commit}] Fixing model name: {model} -> {fixed_model}")
        perf_command = perf_command.replace(model, fixed_model)
        model = fixed_model

    # Get parent commit
    parent = get_parent_commit(commit)
    if not parent:
        logger.error(f"[{short_commit}] Could not get parent commit!")
        return {
            "commit": short_commit,
            "full_commit": commit,
            "status": "error",
            "error": "Could not get parent commit",
        }

    # Determine GPU config
    gpu_config = get_gpu_config(model, perf_command)

    # Adjust command if needed
    perf_command = adjust_command(commit, perf_command)

    logger.info(f"[{short_commit}] Starting benchmark | Model: {model} | GPU: {gpu_config}")

    # Run benchmark
    result = run_benchmark(commit, parent, perf_command, model, gpu_config)
    result["commit"] = short_commit
    result["full_commit"] = commit
    result["model"] = model
    result["gpu_config"] = gpu_config
    result["subject"] = commit_info["subject"]

    # Save result immediately
    result_file = results_dir / short_commit / "benchmark_result.json"
    result_file.parent.mkdir(parents=True, exist_ok=True)
    with open(result_file, "w") as f:
        json.dump(result, f, indent=2, default=str)

    return result


def main():
    import argparse
    from concurrent.futures import ThreadPoolExecutor, as_completed

    parser = argparse.ArgumentParser(description="Hero Benchmark Runner")
    parser.add_argument("--limit", type=int, default=0, help="Limit number of commits to run")
    parser.add_argument("--start-from", type=str, help="Start from specific commit (8 char)")
    parser.add_argument("--dry-run", action="store_true", help="Print plan without running")
    parser.add_argument("--parallel", "-p", type=int, default=1,
                       help="Number of commits to run in parallel (default: 1)")
    args = parser.parse_args()

    logger.info("=" * 80)
    logger.info("HERO BENCHMARK RUN")
    logger.info(f"Start time: {datetime.now().isoformat()}")
    logger.info(f"Log file: {log_file}")
    logger.info("=" * 80)

    # Load commits
    with open("/tmp/hero_run_commits.json") as f:
        commits = json.load(f)

    logger.info(f"Loaded {len(commits)} commits from plan")

    # Apply filters
    if args.start_from:
        start_idx = next((i for i, c in enumerate(commits) if c["commit"].startswith(args.start_from)), 0)
        commits = commits[start_idx:]
        logger.info(f"Starting from commit {args.start_from} (index {start_idx})")

    if args.limit > 0:
        commits = commits[:args.limit]
        logger.info(f"Limited to {args.limit} commits")

    # Dry run mode
    if args.dry_run:
        logger.info("\n### DRY RUN - Would run the following:\n")
        for i, c in enumerate(commits, 1):
            logger.info(f"{i}. {c['commit']} | {c['model']} | {c['subject'][:40]}")
        return

    # Load dataset for full info
    logger.info("\nLoading HuggingFace dataset...")
    from datasets import load_dataset
    ds = load_dataset("Ayushnangia/omniperf_v1", split="train")
    ds_map = {r["commit_hash"]: r for r in ds}

    # Results tracking
    results = []
    success_count = 0
    error_count = 0
    completed_count = 0
    total = len(commits)
    results_file = RESULTS_DIR / f"hero_run_{timestamp}.json"

    def process_result(result: Dict[str, Any], commit_info: Dict[str, Any]) -> None:
        """Process and log a benchmark result (used in parallel mode)."""
        nonlocal success_count, error_count, completed_count, results

        completed_count += 1
        short_commit = result.get("commit", "unknown")
        status = result.get("status", "unknown")
        duration = result.get("duration_s", 0)
        subject = commit_info["subject"][:60]

        logger.info("")
        logger.info("=" * 80)
        logger.info(f"[{completed_count}/{total}] COMPLETED: {short_commit}")
        logger.info(f"Subject: {subject}")

        if status == "success":
            success_count += 1
            logger.info(f"  ✅ SUCCESS ({duration:.1f}s)")
            if result.get("human_improvement"):
                for metric, value in result["human_improvement"].items():
                    logger.info(f"    {metric}: {value:+.2f}%")
        else:
            error_count += 1
            logger.error(f"  ❌ {status.upper()}: {result.get('error', 'Unknown error')}")

        results.append(result)

        # Save incremental results
        with open(results_file, "w") as f:
            json.dump({
                "timestamp": timestamp,
                "total_commits": total,
                "completed": completed_count,
                "success_count": success_count,
                "error_count": error_count,
                "results": results,
            }, f, indent=2, default=str)

        logger.info(f"  Progress: {success_count} success, {error_count} errors ({completed_count}/{total})")

    # Parallel or sequential execution
    if args.parallel > 1:
        logger.info(f"\n*** PARALLEL MODE: {args.parallel} concurrent benchmarks ***")
        logger.info("Spawning Modal containers in parallel...")

        with ThreadPoolExecutor(max_workers=args.parallel) as executor:
            # Submit all tasks
            future_to_commit = {
                executor.submit(
                    run_single_benchmark_task,
                    commit_info,
                    ds_map,
                    RESULTS_DIR,
                    timestamp,
                ): commit_info
                for commit_info in commits
            }

            # Process as they complete
            for future in as_completed(future_to_commit):
                commit_info = future_to_commit[future]
                try:
                    result = future.result()
                    process_result(result, commit_info)
                except Exception as e:
                    short_commit = commit_info["full_commit"][:8]
                    logger.error(f"Benchmark exception for {short_commit}: {e}")
                    error_count += 1
                    completed_count += 1
    else:
        # Sequential execution (original behavior)
        for i, commit_info in enumerate(commits, 1):
            commit = commit_info["full_commit"]
            short_commit = commit[:8]

            logger.info(f"\n{'='*80}")
            logger.info(f"[{i}/{total}] BENCHMARK: {short_commit}")
            logger.info(f"Subject: {commit_info['subject']}")
            logger.info(f"Model: {commit_info['model']}")
            logger.info("=" * 80)

            # Run using the helper function
            result = run_single_benchmark_task(commit_info, ds_map, RESULTS_DIR, timestamp)

            # Log result
            status = result.get("status", "unknown")
            duration = result.get("duration_s", 0)

            if status == "success":
                success_count += 1
                logger.info(f"  ✅ SUCCESS ({duration:.1f}s)")
                if result.get("human_improvement"):
                    for metric, value in result["human_improvement"].items():
                        logger.info(f"    {metric}: {value:+.2f}%")
            else:
                error_count += 1
                logger.error(f"  ❌ {status.upper()}: {result.get('error', 'Unknown error')}")

            results.append(result)
            completed_count += 1

            # Save incremental results
            with open(results_file, "w") as f:
                json.dump({
                    "timestamp": timestamp,
                    "total_commits": total,
                    "completed": completed_count,
                    "success_count": success_count,
                    "error_count": error_count,
                    "results": results,
                }, f, indent=2, default=str)

            logger.info(f"  Progress: {success_count} success, {error_count} errors ({i}/{total})")

    # Final summary
    logger.info("\n" + "=" * 80)
    logger.info("HERO RUN COMPLETE")
    logger.info("=" * 80)
    logger.info(f"Total commits: {total}")
    logger.info(f"Success: {success_count} ({100*success_count/total:.1f}%)")
    logger.info(f"Errors: {error_count} ({100*error_count/total:.1f}%)")
    logger.info(f"Results saved to: {results_file}")
    logger.info(f"Log file: {log_file}")


if __name__ == "__main__":
    main()
