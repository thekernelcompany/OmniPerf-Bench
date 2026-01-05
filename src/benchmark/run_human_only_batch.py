#!/usr/bin/env python3
"""
Run benchmarks for all H100 commits in different modes.

Modes:
- human-only: Benchmark only the human commit (default)
- baseline-only: Benchmark only the baseline (parent) commit
- agent-only: Benchmark only the agent patch

This script:
1. Reads commits from sglang_h100_commits.jsonl
2. Skips commits that already have errors logged
3. Runs benchmark for each commit IN PARALLEL
4. Logs errors back to the JSONL file

Usage:
    python -m src.benchmark.run_human_only_batch --limit 10 --parallel 5
    python -m src.benchmark.run_human_only_batch --mode baseline-only
    python -m src.benchmark.run_human_only_batch --mode baseline-only --only-human-success
    python -m src.benchmark.run_human_only_batch --start-from abc123
    python -m src.benchmark.run_human_only_batch --dry-run
"""

import os
import sys
import json
import argparse
import subprocess
import time
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Optional

# Paths
H100_COMMITS_FILE = Path("src/benchmark/plans/sglang_h100_commits.jsonl")
RESULTS_DIR = Path("benchmark_results/sglang")


def load_commits() -> List[Dict]:
    """Load commits from JSONL file."""
    commits = []
    with open(H100_COMMITS_FILE) as f:
        for line in f:
            if line.strip():
                commits.append(json.loads(line))
    return commits


def save_commits(commits: List[Dict]):
    """Save commits back to JSONL file."""
    with open(H100_COMMITS_FILE, 'w') as f:
        for c in commits:
            f.write(json.dumps(c) + '\n')


def get_commit_status(commit_short: str) -> Optional[str]:
    """Check if commit has results or errors."""
    result_dir = RESULTS_DIR / commit_short
    if not result_dir.exists():
        return None

    # Find latest result
    timestamps = sorted(result_dir.iterdir(), reverse=True)
    if not timestamps:
        return None

    summary_file = timestamps[0] / "summary.txt"
    if summary_file.exists():
        content = summary_file.read_text()
        if "Status: success" in content:
            return "success"
        elif "Status: error" in content:
            return "error"
    return None


def run_benchmark(commit_hash: str, mode: str = "human-only", dry_run: bool = False) -> Dict:
    """Run benchmark for a commit in the specified mode."""
    commit_short = commit_hash[:8]

    cmd = [
        sys.executable, "-m", "src.benchmark.run_single_commit",
        commit_hash,
        "--repo", "sglang",
    ]

    # Add mode-specific flag
    if mode == "human-only":
        cmd.append("--human-only")
    elif mode == "baseline-only":
        cmd.append("--baseline-only")
    elif mode == "agent-only":
        cmd.append("--agent-only")

    # Always add --parallel since we use Modal Functions
    cmd.append("--parallel")

    print(f"\n{'='*60}")
    print(f"Running: {' '.join(cmd)}")
    print(f"{'='*60}")

    if dry_run:
        return {"status": "dry_run", "commit": commit_short}

    start_time = time.time()
    try:
        result = subprocess.run(
            cmd,
            capture_output=False,  # Show output in real-time
            timeout=1800,  # 30 min timeout
        )
        duration = time.time() - start_time

        # Check result status
        status = get_commit_status(commit_short)
        return {
            "status": status or ("success" if result.returncode == 0 else "error"),
            "commit": commit_short,
            "duration": duration,
            "returncode": result.returncode,
        }
    except subprocess.TimeoutExpired:
        return {
            "status": "timeout",
            "commit": commit_short,
            "error": "Benchmark timed out after 30 minutes",
        }
    except Exception as e:
        return {
            "status": "error",
            "commit": commit_short,
            "error": str(e),
        }


def main():
    parser = argparse.ArgumentParser(description="Run benchmarks for H100 commits in different modes")
    parser.add_argument("--mode", choices=["human-only", "baseline-only", "agent-only"], default="human-only",
                        help="Benchmark mode (default: human-only)")
    parser.add_argument("--limit", type=int, default=None, help="Max commits to run")
    parser.add_argument("--parallel", type=int, default=10, help="Number of parallel benchmarks (default: 10)")
    parser.add_argument("--start-from", type=str, default=None, help="Start from this commit")
    parser.add_argument("--skip-existing", action="store_true", default=True, help="Skip commits with results")
    parser.add_argument("--skip-errors", action="store_true", default=True, help="Skip commits with logged errors")
    parser.add_argument("--only-human-success", action="store_true",
                        help="Only run on commits that succeeded in human-only mode")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be done")
    args = parser.parse_args()

    # Load commits
    commits = load_commits()
    print(f"Loaded {len(commits)} commits from {H100_COMMITS_FILE}")

    # Filter commits
    to_run = []
    skipped_errors = 0
    skipped_existing = 0
    skipped_no_human_success = 0
    started = args.start_from is None

    for c in commits:
        commit_hash = c["commit_hash"]
        commit_short = commit_hash[:8]

        # Start from specific commit
        if not started:
            if commit_short.startswith(args.start_from[:8]):
                started = True
            else:
                continue

        # Skip commits with logged errors
        if args.skip_errors and c.get("error"):
            skipped_errors += 1
            continue

        # For baseline-only/agent-only, optionally only run on human-success commits
        if args.only_human_success:
            if not c.get("human_only_success"):
                skipped_no_human_success += 1
                continue

        # Skip commits with existing success for this mode
        if args.skip_existing:
            mode_key = args.mode.replace("-", "_")  # e.g., "baseline-only" -> "baseline_only"
            if c.get(f"{mode_key}_success"):
                skipped_existing += 1
                continue

        to_run.append(c)

        if args.limit and len(to_run) >= args.limit:
            break

    print(f"\nMode: {args.mode}")
    print(f"Commits to run: {len(to_run)}")
    print(f"Skipped (logged errors): {skipped_errors}")
    if args.only_human_success:
        print(f"Skipped (no human success): {skipped_no_human_success}")
    print(f"Skipped (existing results): {skipped_existing}")
    print(f"Parallel workers: {args.parallel}")

    if args.dry_run:
        print("\n[DRY RUN] Would run:")
        for i, c in enumerate(to_run, 1):
            print(f"  {i}. {c['commit_short']} - {c.get('commit_subject', 'N/A')[:50]}...")
        return

    # Thread-safe lock for updating commits file
    commits_lock = threading.Lock()

    def run_and_update(commit_data, index, total):
        """Run benchmark and update commit data."""
        commit_hash = commit_data["commit_hash"]
        commit_short = commit_hash[:8]

        print(f"[{index}/{total}] STARTING: {commit_short} - {commit_data.get('commit_subject', 'N/A')[:40]}...")

        result = run_benchmark(commit_hash, mode=args.mode, dry_run=False)

        # Update commit with result
        with commits_lock:
            if result["status"] == "error" or result["status"] == "timeout":
                commit_data["error"] = result.get("error", "Benchmark failed")
                commit_data["error_date"] = datetime.now().strftime("%Y-%m-%d")
                save_commits(commits)
                print(f"[{index}/{total}] ERROR: {commit_short} - {commit_data['error'][:50]}")
            elif result["status"] == "success":
                # Remove any previous error
                if "error" in commit_data:
                    del commit_data["error"]
                if "error_date" in commit_data:
                    del commit_data["error_date"]
                # Use mode-specific success field names
                mode_key = args.mode.replace("-", "_")  # e.g., "baseline-only" -> "baseline_only"
                commit_data[f"{mode_key}_success"] = True
                commit_data[f"{mode_key}_date"] = datetime.now().strftime("%Y-%m-%d")
                save_commits(commits)
                print(f"[{index}/{total}] SUCCESS: {commit_short}")

        return result

    # Run benchmarks in parallel
    results = []
    print(f"\n{'='*60}")
    print(f"Starting {len(to_run)} benchmarks with {args.parallel} parallel workers")
    print(f"{'='*60}\n")

    with ThreadPoolExecutor(max_workers=args.parallel) as executor:
        futures = {
            executor.submit(run_and_update, c, i, len(to_run)): c
            for i, c in enumerate(to_run, 1)
        }

        for future in as_completed(futures):
            try:
                result = future.result()
                results.append(result)
            except Exception as e:
                print(f"Exception in worker: {e}")
                results.append({"status": "error", "error": str(e)})

    # Summary
    print(f"\n{'='*60}")
    print("BATCH SUMMARY")
    print(f"{'='*60}")
    success = sum(1 for r in results if r["status"] == "success")
    errors = sum(1 for r in results if r["status"] in ["error", "timeout"])
    print(f"Total: {len(results)}")
    print(f"Success: {success}")
    print(f"Errors: {errors}")


if __name__ == "__main__":
    main()
