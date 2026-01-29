#!/usr/bin/env python3
"""
Hero 3-Way Benchmark Runner for OmniPerf vLLM Dataset.

Runs all commits with Claude Code agent patches on Modal:
  - Baseline: Parent commit wheel
  - Human: Commit wheel (human optimization)
  - Agent: Baseline + Claude Code patch

Usage:
    python hero_3way_benchmark.py [--limit N] [--start-from COMMIT] [--dry-run]

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
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Setup comprehensive logging
log_dir = Path("omniperf_results_3way_claude_code/hero_run_logs")
log_dir.mkdir(parents=True, exist_ok=True)
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
log_file = log_dir / f"hero_3way_{timestamp}.log"

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
RESULTS_DIR = Path("omniperf_results_3way_claude_code/vllm")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# Wheel URL pattern
VLLM_WHEEL_URL = "https://vllm-wheels.s3.us-west-2.amazonaws.com/{commit}/vllm-1.0.0.dev-cp38-abi3-manylinux1_x86_64.whl"

# Large models that need multi-GPU
LARGE_MODEL_GPU_MAP = {
    "deepseek-v3": "H100:8",
    "deepseek-v2": "H100:4",
    "nemotron-4-340b": "H100:8",
    "llama-4-scout-17b-16e": "H100:2",
    "meta-llama-3-70b": "H100:4",
    "llama-3.1-70b": "H100:4",
}

# Models that OOM even on max GPU config (skip these)
SKIP_MODELS = [
    "deepseek-ai/deepseek-v3",  # 671B MoE - too large even for 8x H100
    "deepseek-ai/deepseek-v2",  # 236B MoE - OOMs on 4x H100
]

# Invalid commands that cannot be run (hardcoded paths, missing scripts, etc.)
INVALID_COMMAND_PATTERNS = [
    "/data/users/ktong/",  # Hardcoded user path that doesn't exist
    "moe_mem.py",  # Script doesn't exist
    "MODEL",  # Unresolved placeholder
    "BS",  # Unresolved placeholder
    "INPUT_LEN",  # Unresolved placeholder
    "OUTPUT_LEN",  # Unresolved placeholder
    "--dataset-name sharegpt",  # Uses custom ShareGPT split we don't have access to
]

# Model name corrections - fix incorrect model names in the dataset
# HuggingFace model names are case-sensitive and must match exactly
MODEL_NAME_CORRECTIONS = {
    "meta-llama/Llama-3-8B-Instruct": "meta-llama/Meta-Llama-3-8B-Instruct",
    "meta-llama/Llama-3-8B": "meta-llama/Meta-Llama-3-8B",
    "meta-llama/Llama-3-70B-Instruct": "meta-llama/Meta-Llama-3-70B-Instruct",
    "meta-llama/Llama-3-70B": "meta-llama/Meta-Llama-3-70B",
}


def normalize_model_name(model: str) -> str:
    """Normalize model name to correct HuggingFace model ID.

    Some models in the dataset have incorrect names (case sensitivity issues).
    This function corrects them.
    """
    return MODEL_NAME_CORRECTIONS.get(model, model)


def normalize_perf_command(perf_command: str) -> str:
    """Normalize the perf_command by fixing incorrect model names."""
    result = perf_command
    for wrong, correct in MODEL_NAME_CORRECTIONS.items():
        result = result.replace(wrong, correct)
    return result


def should_skip_model(perf_command: str) -> bool:
    """Check if the model in the command should be skipped (known OOM)."""
    cmd_lower = perf_command.lower()
    for model in SKIP_MODELS:
        if model in cmd_lower:
            return True
    return False


def has_invalid_command(perf_command: str) -> bool:
    """Check if the command has invalid patterns (hardcoded paths, placeholders)."""
    for pattern in INVALID_COMMAND_PATTERNS:
        if pattern in perf_command:
            return True
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


def check_wheel_exists(wheel_url: str) -> bool:
    """Check if a wheel URL is accessible."""
    import urllib.request
    try:
        req = urllib.request.Request(wheel_url, method='HEAD')
        with urllib.request.urlopen(req, timeout=10) as response:
            return response.status == 200
    except:
        return False


def is_standard_benchmark(perf_command: str) -> bool:
    """Check if benchmark command uses standard vLLM benchmarks."""
    standard_patterns = [
        "benchmark_serving.py",
        "benchmark_throughput.py",
        "benchmark_latency.py",
        "vllm bench serve",
        "vllm bench throughput",
        "vllm bench latency",
    ]
    cmd_lower = perf_command.lower()
    return any(p in cmd_lower for p in standard_patterns)


def load_existing_benchmarks() -> Dict[str, Dict]:
    """Load existing benchmark results."""
    benchmarks = {}
    for result_dir in RESULTS_DIR.iterdir():
        if not result_dir.is_dir():
            continue
        result_file = result_dir / "benchmark_result.json"
        if result_file.exists():
            try:
                with open(result_file) as f:
                    data = json.load(f)
                commit = result_dir.name
                benchmarks[commit] = data
            except:
                pass
    return benchmarks


# Claude Code run directory with all 96 patches
CLAUDE_CODE_RUN_DIR = Path("ISO-Bench/state/runs/vllm/claude_code/default/2025-12-22_21-40-38")


def load_claude_code_patches() -> Dict[str, str]:
    """Load all Claude Code patches from the run directory.

    Returns mapping of short commit hash -> patch file path.
    """
    commit_to_patch = {}

    if not CLAUDE_CODE_RUN_DIR.exists():
        logger.error(f"Claude Code run directory not found: {CLAUDE_CODE_RUN_DIR}")
        return commit_to_patch

    for item_dir in sorted(CLAUDE_CODE_RUN_DIR.iterdir()):
        if not item_dir.is_dir():
            continue

        journal_file = item_dir / "journal.json"
        patch_file = item_dir / "model_patch.diff"

        if not journal_file.exists() or not patch_file.exists():
            continue

        try:
            with open(journal_file) as f:
                journal = json.load(f)

            commit_hash = journal.get("commits", {}).get("human", "")
            if commit_hash:
                short_hash = commit_hash[:8]
                commit_to_patch[short_hash] = str(patch_file.absolute())
        except Exception as e:
            logger.warning(f"Failed to parse {journal_file}: {e}")

    return commit_to_patch


def run_3way_benchmark(
    commit: str,
    parent_commit: str,
    perf_command: str,
    model: str,
    gpu_config: str,
    agent_patch: Optional[str] = None,
) -> Dict[str, Any]:
    """Run a 3-way benchmark on Modal.

    Now supports Python overlay approach: if wheels aren't available,
    we install a compatible wheel and overlay Python files from git checkout.
    """
    from src.eval.modal_benchmark import run_3way_modal_benchmark

    baseline_url = VLLM_WHEEL_URL.format(commit=parent_commit)
    human_url = VLLM_WHEEL_URL.format(commit=commit)

    logger.info(f"  Baseline commit: {parent_commit[:8]}")
    logger.info(f"  Human commit: {commit[:8]}")
    logger.info(f"  GPU config: {gpu_config}")
    logger.info(f"  Agent patch: {'yes' if agent_patch else 'no'}")
    logger.info(f"  Command preview: {perf_command[:100]}...")

    start_time = time.time()

    try:
        result = run_3way_modal_benchmark(
            baseline_wheel_url=baseline_url,
            human_wheel_url=human_url,
            agent_patch=agent_patch,
            perf_command=perf_command,
            model=model,
            gpu_config=gpu_config,
            base_commit=parent_commit,
            human_commit=commit,  # Pass human commit for overlay fallback
        )

        result["duration_s"] = time.time() - start_time
        return result

    except Exception as e:
        import traceback
        tb = traceback.format_exc()
        error_str = str(e)
        logger.error(f"  Benchmark failed with exception: {e}")
        logger.error(f"  EXCEPTION: {e}")

        return {
            "status": "exception",
            "error": error_str,
            "traceback": tb,
            "duration_s": time.time() - start_time,
        }


def run_single_benchmark(commit_info: Dict, existing: Dict) -> Dict:
    """Run a single benchmark - used for parallel execution."""
    commit = commit_info["full_commit"]
    short_commit = commit_info["commit"]

    # Get parent commit
    parent = commit_info.get("parent_commit")
    if not parent:
        parent = get_parent_commit(commit)
    if not parent:
        return {
            "commit": short_commit,
            "full_commit": commit,
            "status": "error",
            "error": "Could not get parent commit",
        }

    # Load agent patch
    agent_patch = None
    patch_path = commit_info.get("patch_path", "")
    if patch_path and Path(patch_path).exists():
        try:
            with open(patch_path) as f:
                agent_patch = f.read()
        except Exception as e:
            pass

    # Get GPU config
    gpu_config = get_gpu_config(commit_info["model"], commit_info["perf_command"])

    # Run benchmark
    result = run_3way_benchmark(
        commit=commit,
        parent_commit=parent,
        perf_command=commit_info["perf_command"],
        model=commit_info["model"],
        gpu_config=gpu_config,
        agent_patch=agent_patch,
    )

    result["commit"] = short_commit
    result["full_commit"] = commit
    result["parent_commit"] = parent
    result["model"] = commit_info["model"]
    result["gpu_config"] = gpu_config
    result["subject"] = commit_info["subject"]
    result["has_agent_patch"] = agent_patch is not None

    # Preserve instance data
    if short_commit in existing:
        instance_data = existing[short_commit].get("instance", {})
    else:
        instance_data = {
            "commit_hash": commit,
            "commit_subject": commit_info["subject"],
            "perf_command": commit_info["perf_command"],
        }
    result["_instance_data"] = instance_data

    return result


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Hero 3-Way Benchmark Runner")
    parser.add_argument("--limit", type=int, default=0, help="Limit number of commits to run")
    parser.add_argument("--start-from", type=str, help="Start from specific commit (8 char)")
    parser.add_argument("--dry-run", action="store_true", help="Print plan without running")
    parser.add_argument("--skip-existing", action="store_true", help="Skip commits with successful results or failed agent patches")
    parser.add_argument("--parallel", type=int, default=1, help="Number of parallel benchmarks (default: 1)")
    args = parser.parse_args()

    logger.info("=" * 80)
    logger.info("HERO 3-WAY BENCHMARK RUN")
    logger.info(f"Start time: {datetime.now().isoformat()}")
    logger.info(f"Log file: {log_file}")
    logger.info("=" * 80)

    # Load Claude Code patches (all 96 commits)
    logger.info("\nLoading Claude Code patches...")
    claude_patches = load_claude_code_patches()
    logger.info(f"Found {len(claude_patches)} Claude Code patches")

    # Load existing benchmark results (for --skip-existing)
    existing = load_existing_benchmarks()
    logger.info(f"Found {len(existing)} existing benchmark results")

    # Load HuggingFace dataset for full commit info (all 96 commits)
    logger.info("\nLoading HuggingFace dataset...")
    from datasets import load_dataset
    ds = load_dataset("Ayushnangia/omniperf_v1", split="vllm")
    logger.info(f"Dataset has {len(ds)} commits")

    # Build list of commits to run directly from HuggingFace dataset
    commits_to_run = []
    skipped_no_perf = 0
    skipped_non_standard = 0
    skipped_existing = 0
    skipped_oom = 0
    skipped_invalid = 0

    for row in ds:
        full_commit = row["commit_hash"]
        commit_short = full_commit[:8]
        perf_command = row.get("perf_command", "")
        commit_subject = row.get("commit_subject", "")[:50]

        # Skip if no perf_command
        if not perf_command:
            logger.warning(f"Skipping {commit_short} - no perf_command")
            skipped_no_perf += 1
            continue

        # Skip OOM-prone models (too large even for H100:8)
        if should_skip_model(perf_command):
            logger.warning(f"Skipping {commit_short} - model too large (OOM on H100:8)")
            skipped_oom += 1
            continue

        # Skip invalid commands (hardcoded paths, missing scripts, placeholders)
        if has_invalid_command(perf_command):
            logger.warning(f"Skipping {commit_short} - invalid command: {perf_command[:50]}...")
            skipped_invalid += 1
            continue

        # Skip if already successful WITH AGENT RESULTS, or has known version bug, or agent patch failed
        if args.skip_existing and commit_short in existing:
            existing_result = existing[commit_short].get("result", {})
            existing_status = existing_result.get("status", "")
            agent_metrics = existing_result.get("agent_metrics")
            agent_error = existing_result.get("agent_error")
            had_agent_patch = existing_result.get("has_agent_patch", False)

            # Check if agent was attempted and produced valid results
            has_valid_agent = agent_metrics is not None and agent_metrics != {}
            # Check if agent was attempted but FAILED (buggy patch - don't retry)
            agent_attempted_and_failed = had_agent_patch and (agent_error or (agent_metrics is not None and agent_metrics == {}))

            if existing_status == "success" and has_valid_agent:
                logger.info(f"Skipping {commit_short} - already successful with agent results")
                skipped_existing += 1
                continue
            elif existing_status == "success" and agent_attempted_and_failed:
                # Agent patch was tried but failed (crashed, no metrics) - patch is buggy, don't retry
                logger.info(f"Skipping {commit_short} - agent patch attempted but failed (buggy patch)")
                skipped_existing += 1
                continue
            elif existing_status == "success" and not had_agent_patch:
                logger.info(f"Will rerun {commit_short} - success but agent was never attempted")
                # Don't skip - need to run agent
            elif existing_status == "version_bug":
                logger.info(f"Skipping {commit_short} - known vLLM version bug")
                skipped_existing += 1
                continue

        # Check for standard benchmark command
        if not is_standard_benchmark(perf_command):
            logger.warning(f"Skipping {commit_short} - non-standard benchmark: {perf_command[:50]}...")
            skipped_non_standard += 1
            continue

        # Get patch path from Claude Code patches
        patch_path = claude_patches.get(commit_short, "")
        if not patch_path:
            logger.warning(f"Skipping {commit_short} - no Claude Code patch found")
            continue

        # Normalize perf_command to fix incorrect model names
        perf_command = normalize_perf_command(perf_command)

        # Extract model from command
        model_match = re.search(r'--model[=\s]+["\']?([^\s"\']+)', perf_command)
        model = model_match.group(1) if model_match else "unknown"
        # Normalize model name (in case it wasn't in the perf_command pattern)
        model = normalize_model_name(model)

        commits_to_run.append({
            "commit": commit_short,
            "full_commit": full_commit,
            "model": model,
            "perf_command": perf_command,
            "patch_path": patch_path,
            "subject": commit_subject,
        })

    logger.info(f"\nCommit filtering summary:")
    logger.info(f"  Total in dataset: {len(ds)}")
    logger.info(f"  Skipped (no perf_command): {skipped_no_perf}")
    logger.info(f"  Skipped (model OOM): {skipped_oom}")
    logger.info(f"  Skipped (invalid command): {skipped_invalid}")
    logger.info(f"  Skipped (non-standard benchmark): {skipped_non_standard}")
    logger.info(f"  Skipped (already successful): {skipped_existing}")
    logger.info(f"  Ready to run: {len(commits_to_run)}")

    # Pre-flight: Get parent commits and check wheel status (informational only)
    # NOTE: Wheel availability no longer blocks runs - we use Python overlay for missing wheels
    logger.info("\nPre-flight: Getting parent commits and checking wheel status...")
    valid_commits = []
    for c in commits_to_run:
        parent = get_parent_commit(c["full_commit"])
        if not parent:
            logger.warning(f"  {c['commit']} - cannot get parent commit (SKIP)")
            continue

        baseline_url = VLLM_WHEEL_URL.format(commit=parent)
        human_url = VLLM_WHEEL_URL.format(commit=c["full_commit"])

        baseline_ok = check_wheel_exists(baseline_url)
        human_ok = check_wheel_exists(human_url)

        c["parent_commit"] = parent
        c["baseline_wheel_ok"] = baseline_ok
        c["human_wheel_ok"] = human_ok

        # Log status but don't filter - overlay will handle missing wheels
        if baseline_ok and human_ok:
            logger.info(f"  {c['commit']} - both wheels available")
        elif baseline_ok:
            logger.info(f"  {c['commit']} - human wheel missing (will use overlay)")
        elif human_ok:
            logger.info(f"  {c['commit']} - baseline wheel missing (will use overlay)")
        else:
            logger.info(f"  {c['commit']} - no wheels available (will use overlay)")

        valid_commits.append(c)

    commits_to_run = valid_commits
    logger.info(f"\nTotal runnable commits: {len(commits_to_run)}")

    # Apply filters
    if args.start_from:
        start_idx = next((i for i, c in enumerate(commits_to_run)
                         if c["commit"].startswith(args.start_from)), 0)
        commits_to_run = commits_to_run[start_idx:]
        logger.info(f"Starting from commit {args.start_from} (index {start_idx})")

    if args.limit > 0:
        commits_to_run = commits_to_run[:args.limit]
        logger.info(f"Limited to {args.limit} commits")

    # Dry run mode
    if args.dry_run:
        logger.info("\n### DRY RUN - Would run the following:\n")
        for i, c in enumerate(commits_to_run, 1):
            has_patch = "PATCH" if c["patch_path"] and Path(c["patch_path"]).exists() else "no-patch"
            logger.info(f"{i}. {c['commit']} | {c['model'][:30]} | {has_patch} | {c['subject']}")
        return

    # Results tracking
    results = []
    success_count = 0
    error_count = 0
    completed_count = 0

    # Run benchmarks
    total = len(commits_to_run)

    def process_result(result, commit_info):
        """Process and log a benchmark result."""
        nonlocal success_count, error_count, completed_count
        completed_count += 1

        short_commit = result["commit"]
        status = result.get("status", "unknown")
        duration = result.get("duration_s", 0)

        logger.info(f"\n{'='*80}")
        logger.info(f"[{completed_count}/{total}] COMPLETED: {short_commit}")
        logger.info(f"Subject: {commit_info['subject']}")

        if status == "success":
            success_count += 1
            logger.info(f"  SUCCESS ({duration:.1f}s)")

            if result.get("human_improvement"):
                logger.info("  Human vs Baseline:")
                for metric, value in result["human_improvement"].items():
                    logger.info(f"    {metric}: {value:+.2f}%")

            if result.get("agent_improvement"):
                logger.info("  Agent vs Baseline:")
                for metric, value in result["agent_improvement"].items():
                    logger.info(f"    {metric}: {value:+.2f}%")
        else:
            error_count += 1
            error_msg = result.get('error', 'Unknown error')
            logger.error(f"  {status.upper()}: {error_msg}")
            logger.error(f"  === DETAILED ERROR INFO ===")
            logger.error(f"  Commit: {short_commit}")
            logger.error(f"  Model: {commit_info['model']}")
            if result.get("traceback"):
                logger.error(f"  Traceback:\n{result.get('traceback')}")
            logger.error(f"  === END DETAILED ERROR ===")

        logger.info(f"  Progress: {success_count} success, {error_count} errors ({completed_count}/{total})")

        # Save individual result
        result_dir = RESULTS_DIR / short_commit
        result_dir.mkdir(exist_ok=True)
        result_file = result_dir / "benchmark_result.json"
        instance_data = result.pop("_instance_data", {})
        with open(result_file, "w") as f:
            json.dump({"instance": instance_data, "result": result}, f, indent=2, default=str)

        return result

    # Parallel or sequential execution
    if args.parallel > 1:
        logger.info(f"\n*** PARALLEL MODE: {args.parallel} concurrent benchmarks ***\n")
        with ThreadPoolExecutor(max_workers=args.parallel) as executor:
            # Submit all tasks
            future_to_commit = {
                executor.submit(run_single_benchmark, commit_info, existing): commit_info
                for commit_info in commits_to_run
            }

            # Process as they complete
            for future in as_completed(future_to_commit):
                commit_info = future_to_commit[future]
                try:
                    result = future.result()
                    result = process_result(result, commit_info)
                    results.append(result)
                except Exception as e:
                    logger.error(f"Benchmark exception for {commit_info['commit']}: {e}")
                    error_count += 1
                    completed_count += 1
    else:
        # Sequential execution (original behavior)
        for i, commit_info in enumerate(commits_to_run, 1):
            commit = commit_info["full_commit"]
            short_commit = commit_info["commit"]

            logger.info(f"\n{'='*80}")
            logger.info(f"[{i}/{total}] 3-WAY BENCHMARK: {short_commit}")
            logger.info(f"Subject: {commit_info['subject']}")
            logger.info(f"Model: {commit_info['model']}")
            logger.info("=" * 80)

            # Use pre-fetched parent commit
            parent = commit_info.get("parent_commit")
            if not parent:
                parent = get_parent_commit(commit)
            if not parent:
                logger.error(f"  Could not get parent commit!")
                results.append({
                    "commit": short_commit,
                    "status": "error",
                    "error": "Could not get parent commit",
                })
                error_count += 1
                continue

            # Load agent patch if available
            agent_patch = None
            patch_path = commit_info.get("patch_path", "")
            if patch_path and Path(patch_path).exists():
                try:
                    with open(patch_path) as f:
                        agent_patch = f.read()
                    logger.info(f"  Loaded agent patch: {len(agent_patch)} bytes")
                except Exception as e:
                    logger.warning(f"  Could not load patch: {e}")

            # Determine GPU config
            gpu_config = get_gpu_config(commit_info["model"], commit_info["perf_command"])

            # Run benchmark
            result = run_3way_benchmark(
                commit=commit,
                parent_commit=parent,
                perf_command=commit_info["perf_command"],
                model=commit_info["model"],
                gpu_config=gpu_config,
                agent_patch=agent_patch,
            )

            result["commit"] = short_commit
            result["full_commit"] = commit
            result["parent_commit"] = parent
            result["model"] = commit_info["model"]
            result["gpu_config"] = gpu_config
            result["subject"] = commit_info["subject"]
            result["has_agent_patch"] = agent_patch is not None

            # Log result
            status = result.get("status", "unknown")
            duration = result.get("duration_s", 0)

            if status == "success":
                success_count += 1
                logger.info(f"  SUCCESS ({duration:.1f}s)")

                # Log improvements
                if result.get("human_improvement"):
                    logger.info("  Human vs Baseline:")
                    for metric, value in result["human_improvement"].items():
                        logger.info(f"    {metric}: {value:+.2f}%")

                if result.get("agent_improvement"):
                    logger.info("  Agent vs Baseline:")
                    for metric, value in result["agent_improvement"].items():
                        logger.info(f"    {metric}: {value:+.2f}%")

                if result.get("agent_vs_human"):
                    logger.info("  Agent vs Human:")
                    for metric, value in result["agent_vs_human"].items():
                        logger.info(f"    {metric}: {value:+.2f}%")
            else:
                error_count += 1
                error_msg = result.get('error', 'Unknown error')
                logger.error(f"  {status.upper()}: {error_msg}")

                # Log detailed error information
                logger.error(f"  === DETAILED ERROR INFO ===")
                logger.error(f"  Commit: {short_commit} (full: {commit})")
                logger.error(f"  Parent: {parent}")
                logger.error(f"  Model: {commit_info['model']}")
                logger.error(f"  GPU config: {gpu_config}")
                logger.error(f"  Perf command: {commit_info['perf_command'][:100]}...")
                logger.error(f"  Had agent patch: {agent_patch is not None}")
                if result.get("traceback"):
                    logger.error(f"  Traceback:\n{result.get('traceback')}")

                # Log any additional error details from result
                if result.get("baseline_metrics"):
                    logger.error(f"  Baseline metrics: {result.get('baseline_metrics')}")
                if result.get("human_metrics"):
                    logger.error(f"  Human metrics: {result.get('human_metrics')}")
                if result.get("agent_metrics"):
                    logger.error(f"  Agent metrics: {result.get('agent_metrics')}")
                if result.get("agent_error"):
                    logger.error(f"  Agent error: {result.get('agent_error')}")
                if result.get("server_logs"):
                    logger.error(f"  Server logs (last 500 chars): ...{str(result.get('server_logs'))[-500:]}")
                logger.error(f"  === END DETAILED ERROR ===")

            results.append(result)

            # Save individual result
            result_dir = RESULTS_DIR / short_commit
            result_dir.mkdir(exist_ok=True)
            result_file = result_dir / "benchmark_result.json"

            # Preserve instance data from existing
            if short_commit in existing:
                instance_data = existing[short_commit].get("instance", {})
            else:
                instance_data = {
                    "commit_hash": commit,
                    "commit_subject": commit_info["subject"],
                    "perf_command": commit_info["perf_command"],
                }

            with open(result_file, "w") as f:
                json.dump({
                    "instance": instance_data,
                    "result": result,
                }, f, indent=2, default=str)

            # Save aggregate progress
            progress_file = log_dir / f"hero_3way_{timestamp}_progress.json"
            with open(progress_file, "w") as f:
                json.dump({
                    "timestamp": timestamp,
                    "total_commits": total,
                    "completed": i,
                    "success_count": success_count,
                    "error_count": error_count,
                    "results": results,
                }, f, indent=2, default=str)

            logger.info(f"  Progress: {success_count} success, {error_count} errors ({i}/{total})")

    # Final summary
    logger.info("\n" + "=" * 80)
    logger.info("HERO 3-WAY RUN COMPLETE")
    logger.info("=" * 80)
    logger.info(f"Total commits: {total}")
    logger.info(f"Success: {success_count} ({100*success_count/max(total,1):.1f}%)")
    logger.info(f"Errors: {error_count} ({100*error_count/max(total,1):.1f}%)")
    logger.info(f"Log file: {log_file}")


if __name__ == "__main__":
    main()
