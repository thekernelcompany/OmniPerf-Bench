#!/usr/bin/env python3
"""
Hero SGLang 3-Way Benchmark Runner

Runs 3-way benchmarks (baseline vs human vs agent) for SGLang commits
using Claude Code patches from perf-agents-bench.

Usage:
    python hero_sglang_benchmark.py [--start-from N] [--dry-run]
"""

import os
import sys
import json
import logging
import argparse
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from datasets import load_dataset

# Import local Docker benchmark functions (when available)
try:
    from local_docker_sglang_benchmark import (
        run_human_serving_benchmark as local_run_human,
        run_baseline_serving_benchmark as local_run_baseline,
        run_agent_serving_benchmark as local_run_agent,
        run_combined_3way_serving as local_run_3way,
        get_sglang_image,
        get_hf_token,
    )
    LOCAL_DOCKER_AVAILABLE = True
except ImportError:
    LOCAL_DOCKER_AVAILABLE = False

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# Paths
CLAUDE_CODE_RUNS_DIR = Path("perf-agents-bench/state/runs/sglang/claude_code")
RESULTS_DIR = Path("omniperf_results_3way_sglang")
SGLANG_REPO_URL = "https://github.com/sgl-project/sglang.git"

# Default GPU config
DEFAULT_GPU_CONFIG = "H100:1"

# Large models that need multi-GPU
LARGE_MODEL_GPU_MAP = {
    "deepseek-v3": "H100:8",
    "deepseek-v2": "H100:4",
    "deepseek-r1": "H100:8",
    "llama-4": "H100:2",
    "llama-3-70b": "H100:4",
    "llama-3.1-70b": "H100:4",
    "mixtral": "H100:2",
    "qwen2-72b": "H100:4",
}


def extract_tp_from_command(perf_command: str) -> int:
    """Extract tensor parallel size from benchmark command."""
    import re
    tp_patterns = [
        r'--tp[=\s]+(\d+)',
        r'--tp-size[=\s]+(\d+)',
        r'--tensor-parallel-size[=\s]+(\d+)',
        r'-tp[=\s]+(\d+)',
    ]
    for pattern in tp_patterns:
        match = re.search(pattern, perf_command)
        if match:
            return int(match.group(1))
    return 1


def get_gpu_config(model: str, perf_command: str) -> str:
    """Determine GPU config based on model and TP size in command.

    Priority:
    1. TP size explicitly in command
    2. Known large models
    3. Default to H100:1
    """
    # First check TP in command
    tp_size = extract_tp_from_command(perf_command)
    if tp_size >= 8:
        return "H100:8"
    elif tp_size >= 4:
        return "H100:4"
    elif tp_size >= 2:
        return "H100:2"

    # Check for known large models
    model_lower = model.lower() if model else ""
    for pattern, config in LARGE_MODEL_GPU_MAP.items():
        if pattern in model_lower:
            return config

    return DEFAULT_GPU_CONFIG


def extract_model_from_command(perf_command: str) -> Optional[str]:
    """Extract model name from perf_command if present."""
    import re
    # Try --model or --model-path patterns
    patterns = [
        r'--model[=\s]+["\']?([^\s"\']+)',
        r'--model-path[=\s]+["\']?([^\s"\']+)',
    ]
    for pattern in patterns:
        match = re.search(pattern, perf_command)
        if match:
            return match.group(1)
    return None


def find_claude_code_patches() -> Dict[str, Dict[str, Any]]:
    """
    Find all Claude Code patches for SGLang commits.

    Returns:
        Dict mapping short_commit_hash (8 chars) -> {
            patch_path, patch_content, run_summary, full_commit, parent_commit
        }
    """
    patches = {}

    if not CLAUDE_CODE_RUNS_DIR.exists():
        logger.warning(f"Claude Code runs directory not found: {CLAUDE_CODE_RUNS_DIR}")
        return patches

    # Walk through all run directories
    for run_dir in CLAUDE_CODE_RUNS_DIR.glob("*/*/sglang_*"):
        patch_file = run_dir / "model_patch.diff"
        journal_file = run_dir / "journal.json"
        summary_file = run_dir / "run_summary.json"

        if patch_file.exists() and journal_file.exists():
            try:
                # Read journal to get full commit hashes
                journal = json.loads(journal_file.read_text())
                commits = journal.get("commits", {})
                full_commit = commits.get("human", "")
                parent_commit = commits.get("pre", "")

                if not full_commit:
                    # Fallback to extracting from directory name
                    dir_name = run_dir.name
                    parts = dir_name.split("_")
                    if len(parts) >= 3:
                        full_commit = parts[-1]  # Short hash only
                    else:
                        continue

                short_hash = full_commit[:8]

                # Read patch content
                patch_content = patch_file.read_text()

                # Skip empty patches
                if not patch_content.strip():
                    logger.warning(f"Skipping {short_hash} - empty patch file")
                    continue

                # Read run summary if available
                run_summary = None
                if summary_file.exists():
                    run_summary = json.loads(summary_file.read_text())

                patches[short_hash] = {
                    "patch_path": str(patch_file),
                    "patch_content": patch_content,
                    "run_summary": run_summary,
                    "patch_size": len(patch_content),
                    "full_commit": full_commit,
                    "parent_commit": parent_commit,
                    "journal_status": journal.get("status", "unknown"),
                }
            except Exception as e:
                logger.warning(f"Failed to read patch data from {run_dir}: {e}")

    logger.info(f"Found {len(patches)} Claude Code patches")
    return patches


def get_parent_commit(commit_hash: str, repo_path: str = "/tmp/sglang_repo") -> Optional[str]:
    """Get the parent commit hash for a given commit."""
    import subprocess

    # Ensure repo exists
    if not os.path.exists(repo_path):
        result = subprocess.run(
            ["git", "clone", "--depth", "100", SGLANG_REPO_URL, repo_path],
            capture_output=True, text=True, timeout=300
        )
        if result.returncode != 0:
            logger.error(f"Failed to clone repo: {result.stderr}")
            return None

    # Fetch the specific commit if needed
    subprocess.run(
        ["git", "fetch", "origin", commit_hash],
        cwd=repo_path, capture_output=True, timeout=60
    )

    # Get parent
    result = subprocess.run(
        ["git", "rev-parse", f"{commit_hash}^"],
        cwd=repo_path, capture_output=True, text=True, timeout=30
    )

    if result.returncode == 0:
        return result.stdout.strip()
    return None


def load_sglang_dataset() -> List[Dict]:
    """Load SGLang dataset from HuggingFace."""
    logger.info("Loading HuggingFace dataset...")
    ds = load_dataset("Ayushnangia/omniperf_v1", split="sglang")
    logger.info(f"Dataset has {len(ds)} commits")
    return list(ds)


def should_skip_commit(item: Dict) -> Optional[str]:
    """
    Check if a commit should be skipped and return reason.

    Returns:
        None if should run, or skip reason string
    """
    # Check for perf_command
    if not item.get("perf_command"):
        return "no perf_command"

    # Check for models
    if not item.get("models"):
        return "no models specified"

    return None


def check_docker_image_exists(commit: str) -> bool:
    """Check if Docker image exists for this commit."""
    import urllib.request
    import urllib.error

    SGLANG_DOCKER_REPO = "ayushnangia16/sglang-docker"

    for tag in [commit[:40], commit[:12], commit[:8]]:
        url = f"https://hub.docker.com/v2/repositories/{SGLANG_DOCKER_REPO}/tags/{tag}"
        try:
            req = urllib.request.Request(url, method='HEAD')
            urllib.request.urlopen(req, timeout=10)
            return True
        except urllib.error.HTTPError as e:
            if e.code == 404:
                continue
        except Exception:
            continue
    return False


def run_3way_benchmark(
    base_commit: str,
    human_commit: str,
    agent_patch: Optional[str],
    perf_command: str,
    model: str,
    gpu_config: str = DEFAULT_GPU_CONFIG,
    use_wheel: bool = False,  # Default to Docker - wheel approach has sgl_kernel ABI issues
    human_only: bool = True,  # NEW: Only benchmark human commit (baseline/agent fail with overlay)
) -> Dict[str, Any]:
    """
    Run benchmark on Modal.

    IMPORTANT ARCHITECTURAL NOTE:
    SGLang has multi-package dependencies (sglang, sgl-kernel, flashinfer) that must
    be built together for ABI compatibility. The Python overlay approach CANNOT work
    for baseline/agent because:
    - Docker image has compiled extensions (sgl_kernel, flashinfer) for HUMAN commit
    - Overlaying Python files from baseline/agent creates ABI mismatch
    - Server crashes with ImportError or undefined symbol errors

    As of Jan 2026, the only reliable approach is:
    - human_only=True (default): Only benchmark human commit using Docker image as-is
    - human_only=False: Attempt 3-way (will fail for baseline/agent with current infra)

    For true 3-way benchmarks, we would need to build 3 separate Docker images per commit.

    Returns:
        Benchmark result dict with human_metrics always populated (when successful)
        baseline_metrics and agent_metrics will be empty unless separate images are built
    """
    from src.eval.sglang_modal_benchmark import run_3way_modal_benchmark, has_prebuilt_image

    print(f"Running SGLang benchmark on Modal with {gpu_config}...")
    print(f"  Human commit: {human_commit[:8]}")
    print(f"  Model: {model}")
    print(f"  Command: {perf_command[:80]}...")
    print(f"  Mode: {'human-only' if human_only else '3-way (experimental)'}")

    if human_only:
        print(f"  NOTE: Baseline/agent skipped - requires separate Docker images for accurate comparison")

    # For Docker-based approach, check if Docker image exists
    if not use_wheel and not has_prebuilt_image(human_commit):
        return {
            "status": "error",
            "error": f"No Docker image for {human_commit[:8]}. Build with: python tools/build_sglang_images.py --commit {human_commit}",
            "baseline_metrics": {},
            "human_metrics": {},
            "agent_metrics": None,
            "benchmark_mode": "human_only" if human_only else "3way",
        }

    try:
        # If human_only mode, pass None for base_commit and agent_patch
        # This tells the benchmark to skip baseline/agent phases entirely
        result = run_3way_modal_benchmark(
            base_commit=base_commit if not human_only else None,
            human_commit=human_commit,
            agent_patch=agent_patch if not human_only else None,
            perf_command=perf_command,
            model=model,
            gpu_config=gpu_config,
            use_wheel=use_wheel,
        )

        # Add benchmark mode to result
        result["benchmark_mode"] = "human_only" if human_only else "3way"

        # If human_only mode and no baseline/agent, document why
        if human_only:
            if not result.get("baseline_metrics"):
                result["baseline_skip_reason"] = "Human-only mode - baseline requires separate Docker image"
            if not result.get("agent_metrics"):
                result["agent_skip_reason"] = "Human-only mode - agent requires separate Docker image"

        return result

    except Exception as e:
        return {
            "status": "error",
            "error": str(e),
            "baseline_metrics": {},
            "human_metrics": {},
            "agent_metrics": None,
            "benchmark_mode": "human_only" if human_only else "3way",
        }


def run_3way_benchmark_local_docker(
    base_commit: str,
    human_commit: str,
    agent_patch: Optional[str],
    perf_command: str,
    model: str,
    gpu_config: str = DEFAULT_GPU_CONFIG,
    human_only: bool = True,
) -> Dict[str, Any]:
    """
    Run benchmark using local Docker containers instead of Modal.

    This is useful for:
    - Machines with local GPUs
    - Debugging without Modal costs
    - Commits that fail on Modal due to network issues

    Requires local_docker_sglang_benchmark.py to be importable.
    """
    if not LOCAL_DOCKER_AVAILABLE:
        return {
            "status": "error",
            "error": "Local Docker benchmark module not available. Import local_docker_sglang_benchmark failed.",
            "baseline_metrics": {},
            "human_metrics": {},
            "agent_metrics": None,
            "benchmark_mode": "local_docker",
        }

    print(f"Running SGLang benchmark using LOCAL DOCKER...")
    print(f"  Human commit: {human_commit[:8]}")
    print(f"  Model: {model}")
    print(f"  Command: {perf_command[:80]}...")
    print(f"  Mode: {'human-only' if human_only else '3-way'}")

    # Get HuggingFace token
    hf_token = get_hf_token()

    result = {
        "status": "error",
        "baseline_metrics": {},
        "human_metrics": {},
        "agent_metrics": None,
        "benchmark_mode": "local_docker",
        "human_only": human_only,
    }

    try:
        if human_only:
            # Run human-only benchmark
            human_result = local_run_human(
                human_commit=human_commit,
                model=model,
                perf_command=perf_command,
                hf_token=hf_token,
                timeout=1800
            )

            if human_result.status == 'success':
                result["status"] = "success"
                result["human_metrics"] = {
                    "request_throughput": human_result.request_throughput,
                    "output_throughput": human_result.output_throughput,
                    "ttft_mean": human_result.ttft_mean,
                    "ttft_median": human_result.ttft_median,
                    "ttft_p99": human_result.ttft_p99,
                    "tpot_mean": human_result.tpot_mean,
                    "itl_mean": human_result.itl_mean,
                }
            else:
                result["error"] = human_result.error or "Human benchmark failed"

        else:
            # Run 3-way benchmark
            agent_patch_path = None
            if agent_patch:
                # Write patch to temp file
                import tempfile
                with tempfile.NamedTemporaryFile(mode='w', suffix='.diff', delete=False) as f:
                    f.write(agent_patch)
                    agent_patch_path = Path(f.name)

            baseline_result, human_result, agent_result = local_run_3way(
                human_commit=human_commit,
                base_commit=base_commit,
                model=model,
                perf_command=perf_command,
                agent_patch_path=agent_patch_path,
                hf_token=hf_token,
                timeout=5400
            )

            # Convert results
            if baseline_result.status == 'success':
                result["baseline_metrics"] = {
                    "request_throughput": baseline_result.request_throughput,
                    "output_throughput": baseline_result.output_throughput,
                    "ttft_mean": baseline_result.ttft_mean,
                }

            if human_result.status == 'success':
                result["human_metrics"] = {
                    "request_throughput": human_result.request_throughput,
                    "output_throughput": human_result.output_throughput,
                    "ttft_mean": human_result.ttft_mean,
                }

            if agent_result and agent_result.status == 'success':
                result["agent_metrics"] = {
                    "request_throughput": agent_result.request_throughput,
                    "output_throughput": agent_result.output_throughput,
                    "ttft_mean": agent_result.ttft_mean,
                }

            # Determine overall status
            if human_result.status == 'success':
                result["status"] = "success"
            else:
                result["status"] = "error"
                result["error"] = human_result.error or "Benchmark failed"

            # Cleanup temp file
            if agent_patch_path:
                try:
                    agent_patch_path.unlink()
                except:
                    pass

    except Exception as e:
        result["error"] = str(e)

    return result


def save_result(commit_hash: str, result: Dict, results_dir: Path):
    """Save benchmark result to JSON file."""
    commit_dir = results_dir / "sglang" / commit_hash
    commit_dir.mkdir(parents=True, exist_ok=True)

    result_file = commit_dir / "benchmark_result.json"
    with open(result_file, 'w') as f:
        json.dump(result, f, indent=2)

    logger.info(f"Saved result to {result_file}")


def run_single_benchmark(commit_data: Dict, results_dir: Path, use_local_docker: bool = False, human_only: bool = True) -> Dict:
    """Run a single benchmark - used for parallel execution.

    This function is self-contained and can be called from ThreadPoolExecutor.

    Args:
        commit_data: Dict with 'item' and 'patch_info'
        results_dir: Directory to save results
        use_local_docker: If True, use local Docker instead of Modal
        human_only: If True, only run human benchmark (skip baseline/agent)
    """
    item = commit_data["item"]
    patch_info = commit_data["patch_info"]

    commit_hash = item["commit_hash"]
    short_hash = commit_hash[:8]

    # Get parent commit - prefer from patch journal, fall back to git
    parent_commit = patch_info.get("parent_commit")
    if not parent_commit:
        parent_commit = get_parent_commit(commit_hash)

    if not parent_commit:
        return {
            "commit": short_hash,
            "full_commit": commit_hash,
            "status": "error",
            "error": "Could not get parent commit",
            "baseline_metrics": {},
            "human_metrics": {},
            "agent_metrics": None,
        }

    # Get model
    model = None
    if item['models'] and item['models'][0] and item['models'][0] != 'N/A':
        model = item['models'][0]
    if not model:
        model = extract_model_from_command(item['perf_command'])

    if not model:
        return {
            "commit": short_hash,
            "full_commit": commit_hash,
            "status": "error",
            "error": "No model specified",
            "baseline_metrics": {},
            "human_metrics": {},
            "agent_metrics": None,
        }

    # Determine GPU config
    gpu_config = get_gpu_config(model, item["perf_command"])

    # Run benchmark
    start_time = time.time()

    if use_local_docker:
        # Use local Docker benchmark
        result = run_3way_benchmark_local_docker(
            base_commit=parent_commit,
            human_commit=commit_hash,
            agent_patch=patch_info["patch_content"],
            perf_command=item["perf_command"],
            model=model,
            gpu_config=gpu_config,
            human_only=human_only,
        )
    else:
        # Use Modal benchmark (original path)
        result = run_3way_benchmark(
            base_commit=parent_commit,
            human_commit=commit_hash,
            agent_patch=patch_info["patch_content"],
            perf_command=item["perf_command"],
            model=model,
            gpu_config=gpu_config,
            human_only=human_only,
        )

    duration = time.time() - start_time

    # Add metadata
    result["commit"] = short_hash
    result["full_commit"] = commit_hash
    result["parent_commit"] = parent_commit
    result["model"] = model
    result["subject"] = item["commit_subject"]
    result["perf_command"] = item["perf_command"]
    result["duration_s"] = duration
    result["has_agent_patch"] = True
    result["gpu_config"] = gpu_config
    result["use_local_docker"] = use_local_docker

    # Save result immediately
    save_result(short_hash, result, results_dir)

    return result


def main():
    parser = argparse.ArgumentParser(description="Run SGLang 3-way benchmarks")
    parser.add_argument("--start-from", type=int, default=1, help="Start from commit N")
    parser.add_argument("--limit", type=int, help="Limit number of commits to process")
    parser.add_argument("--dry-run", action="store_true", help="Don't run benchmarks, just show what would run")
    parser.add_argument("--commit", type=str, help="Run specific commit only")
    parser.add_argument("--parallel", type=int, default=1, help="Number of parallel benchmarks (default: 1)")
    parser.add_argument("--use-local-docker", action="store_true",
                       help="Run benchmarks in local Docker containers instead of Modal")
    parser.add_argument("--human-only", action="store_true", default=True,
                       help="Only run human benchmark (skip baseline/agent)")
    parser.add_argument("--3way", dest="three_way", action="store_true",
                       help="Run 3-way benchmark (baseline, human, agent)")
    args = parser.parse_args()

    # Check local Docker availability if requested
    if args.use_local_docker and not LOCAL_DOCKER_AVAILABLE:
        print("ERROR: --use-local-docker requested but local_docker_sglang_benchmark module not available")
        print("Make sure local_docker_sglang_benchmark.py is in the same directory")
        sys.exit(1)

    # 3-way overrides human-only
    if args.three_way:
        args.human_only = False

    # Setup
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = RESULTS_DIR
    results_dir.mkdir(parents=True, exist_ok=True)

    # Log file
    log_dir = results_dir / "hero_run_logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / f"hero_sglang_{timestamp}.log"

    # Add file handler to logger
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(logging.Formatter('%(asctime)s | %(levelname)s | %(message)s'))
    logger.addHandler(file_handler)

    logger.info("=" * 80)
    logger.info("HERO SGLANG 3-WAY BENCHMARK RUN")
    logger.info(f"Start time: {datetime.now().isoformat()}")
    logger.info(f"Log file: {log_file}")
    logger.info(f"Execution mode: {'LOCAL DOCKER' if args.use_local_docker else 'Modal'}")
    logger.info(f"Benchmark mode: {'human-only' if args.human_only else '3-way'}")
    logger.info("=" * 80)

    # Load Claude Code patches
    print("Loading Claude Code patches...")
    patches = find_claude_code_patches()
    logger.info(f"Found {len(patches)} Claude Code patches")

    # Load dataset
    print("Loading HuggingFace dataset...")
    dataset = load_sglang_dataset()

    # Check existing results
    existing_results = set()
    for result_file in results_dir.glob("sglang/*/benchmark_result.json"):
        try:
            result = json.loads(result_file.read_text())
            if result.get("status") == "success":
                commit = result_file.parent.name
                existing_results.add(commit)
        except:
            pass
    logger.info(f"Found {len(existing_results)} existing successful results")

    # Filter and prepare commits
    commits_to_run = []
    skipped = {"no_perf_command": 0, "no_models": 0, "no_patch": 0, "already_done": 0}

    for item in dataset:
        commit_hash = item["commit_hash"]
        short_hash = commit_hash[:8]

        # Check skip reasons
        skip_reason = should_skip_commit(item)
        if skip_reason:
            if "perf_command" in skip_reason:
                skipped["no_perf_command"] += 1
            elif "models" in skip_reason:
                skipped["no_models"] += 1
            logger.warning(f"Skipping {short_hash} - {skip_reason}")
            continue

        # Check for agent patch
        if short_hash not in patches:
            skipped["no_patch"] += 1
            logger.warning(f"Skipping {short_hash} - no Claude Code patch")
            continue

        # Check if already done
        if short_hash in existing_results:
            skipped["already_done"] += 1
            logger.info(f"Skipping {short_hash} - already successful")
            continue

        # Get specific commit if requested
        if args.commit and args.commit not in [commit_hash, short_hash]:
            continue

        commits_to_run.append({
            "item": item,
            "patch_info": patches[short_hash],
        })

    # Summary
    logger.info("\nCommit filtering summary:")
    logger.info(f"  Total in dataset: {len(dataset)}")
    logger.info(f"  Skipped (no perf_command): {skipped['no_perf_command']}")
    logger.info(f"  Skipped (no models): {skipped['no_models']}")
    logger.info(f"  Skipped (no patch): {skipped['no_patch']}")
    logger.info(f"  Skipped (already successful): {skipped['already_done']}")
    logger.info(f"  Ready to run: {len(commits_to_run)}")

    if not commits_to_run:
        logger.info("No commits to run!")
        return

    # Apply start-from and limit
    if args.start_from > 1:
        commits_to_run = commits_to_run[args.start_from - 1:]
        logger.info(f"Starting from commit {args.start_from}")

    if args.limit:
        commits_to_run = commits_to_run[:args.limit]
        logger.info(f"Limited to {args.limit} commits")

    # Dry run check
    if args.dry_run:
        print("\n=== DRY RUN - Would run these commits ===")
        for i, commit_data in enumerate(commits_to_run, 1):
            item = commit_data["item"]
            patch_info = commit_data["patch_info"]

            # Determine model (same logic as actual run)
            model = None
            if item['models'] and item['models'][0] and item['models'][0] != 'N/A':
                model = item['models'][0]
            if not model:
                model = extract_model_from_command(item['perf_command'])

            # Determine GPU config
            gpu_config = get_gpu_config(model or "", item['perf_command'])

            # Extract TP size
            tp_size = extract_tp_from_command(item['perf_command'])

            # Determine benchmark type
            bench_type = "server" if "bench_serving" in item['perf_command'].lower() or "benchmark_serving" in item['perf_command'].lower() else "direct"

            print(f"{i}. {item['commit_hash'][:8]} - {item['commit_subject'][:50]}")
            print(f"   Model: {model or 'UNKNOWN'}")
            print(f"   GPU: {gpu_config} | TP: {tp_size} | Type: {bench_type} | Patch: {patch_info['patch_size']}b")
            print(f"   Cmd: {item['perf_command'][:70]}...")
        return

    # Run benchmarks
    success_count = 0
    error_count = 0
    completed_count = 0
    total = len(commits_to_run)

    def process_result(result: Dict, commit_data: Dict) -> Dict:
        """Process and log a benchmark result."""
        nonlocal success_count, error_count, completed_count
        completed_count += 1

        short_hash = result.get("commit", "unknown")
        status = result.get("status", "unknown")
        duration = result.get("duration_s", 0)
        subject = commit_data["item"]["commit_subject"][:60]

        logger.info("")
        logger.info("=" * 80)
        logger.info(f"[{completed_count}/{total}] COMPLETED: {short_hash}")
        logger.info(f"Subject: {subject}")

        if status == "success":
            success_count += 1
            logger.info(f"  SUCCESS ({duration:.1f}s)")

            # Log improvements
            if result.get("human_improvement"):
                logger.info("  Human vs Baseline:")
                for k, v in result["human_improvement"].items():
                    logger.info(f"    {k}: {v:+.2f}%")

            if result.get("agent_improvement"):
                logger.info("  Agent vs Baseline:")
                for k, v in result["agent_improvement"].items():
                    logger.info(f"    {k}: {v:+.2f}%")
        else:
            error_count += 1
            logger.error(f"  ERROR: {result.get('error', 'Unknown error')}")

        logger.info(f"  Progress: {success_count} success, {error_count} errors ({completed_count}/{total})")
        return result

    # Parallel or sequential execution
    if args.parallel > 1:
        logger.info(f"\n*** PARALLEL MODE: {args.parallel} concurrent benchmarks ***")
        logger.info("Spawning Modal containers in parallel...")

        with ThreadPoolExecutor(max_workers=args.parallel) as executor:
            # Submit all tasks
            future_to_commit = {
                executor.submit(run_single_benchmark, commit_data, results_dir): commit_data
                for commit_data in commits_to_run
            }

            # Process as they complete
            for future in as_completed(future_to_commit):
                commit_data = future_to_commit[future]
                try:
                    result = future.result()
                    process_result(result, commit_data)
                except Exception as e:
                    short_hash = commit_data["item"]["commit_hash"][:8]
                    logger.error(f"Benchmark exception for {short_hash}: {e}")
                    error_count += 1
                    completed_count += 1
    else:
        # Sequential execution (original behavior)
        for i, commit_data in enumerate(commits_to_run, 1):
            item = commit_data["item"]
            patch_info = commit_data["patch_info"]

            commit_hash = item["commit_hash"]
            short_hash = commit_hash[:8]

            logger.info("")
            logger.info("=" * 80)
            logger.info(f"[{i}/{total}] 3-WAY BENCHMARK: {short_hash}")
            logger.info(f"Subject: {item['commit_subject'][:60]}")
            logger.info(f"Model: {item['models'][0] if item['models'] else 'N/A'}")
            logger.info("=" * 80)

            # Run benchmark using the helper function
            result = run_single_benchmark(
                commit_data, results_dir,
                use_local_docker=args.use_local_docker,
                human_only=args.human_only
            )

            # Log outcome
            if result["status"] == "success":
                success_count += 1
                logger.info(f"  SUCCESS ({result.get('duration_s', 0):.1f}s)")

                # Log improvements
                if result.get("human_improvement"):
                    logger.info("  Human vs Baseline:")
                    for k, v in result["human_improvement"].items():
                        logger.info(f"    {k}: {v:+.2f}%")

                if result.get("agent_improvement"):
                    logger.info("  Agent vs Baseline:")
                    for k, v in result["agent_improvement"].items():
                        logger.info(f"    {k}: {v:+.2f}%")
            else:
                error_count += 1
                logger.error(f"  ERROR: {result.get('error', 'Unknown error')}")

            logger.info(f"  Progress: {success_count} success, {error_count} errors ({i}/{total})")

    # Final summary
    logger.info("")
    logger.info("=" * 80)
    logger.info("BENCHMARK RUN COMPLETE")
    logger.info(f"  Total processed: {total}")
    logger.info(f"  Successful: {success_count}")
    logger.info(f"  Errors: {error_count}")
    logger.info(f"  Success rate: {success_count/total*100:.1f}%" if total > 0 else "N/A")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()
