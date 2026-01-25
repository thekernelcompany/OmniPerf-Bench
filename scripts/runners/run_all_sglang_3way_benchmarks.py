#!/usr/bin/env python3
"""
Run 3-way benchmarks for all 14 SGLang commits with Docker images.

This script runs baseline, human, and agent (claude_code + codex) benchmarks
for each commit that has a verified Docker image available.

Usage:
    # Run all benchmarks
    python run_all_sglang_3way_benchmarks.py

    # Run specific commits
    python run_all_sglang_3way_benchmarks.py --commits 187b85b7,6b231325

    # Dry run (check images only)
    python run_all_sglang_3way_benchmarks.py --dry-run

    # Skip already completed benchmarks
    python run_all_sglang_3way_benchmarks.py --skip-existing
"""

import argparse
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any

# The 14 commits with verified Docker images
COMMITS_WITH_IMAGES = [
    "187b85b7f38496653948a2aba546d53c09ada0f3",  # [PD] Optimize custom mem pool usage
    "6b231325b9782555eb8e1cfcf27820003a98382b",  # [PD Perf] replace Queue to FastQueue
    "6cb00c6398126513e37c43dd975d461765fb44c7",  # [PD] Optimize time out logic
    "148254d4db8bf3bffee23710cd1acbd5711ebd1b",  # Improve moe reduce sum kernel
    "2bd18e2d767e3a0f8afb5aff427bc8e6e4d297c0",  # Memory pool: Minor optimize
    "2a754e57b052e249ed4f8572cb6f0069ba6a495e",  # 2x perf for large prefill
    "880221bd3b3e56a4bc2268fe9a9f77f426accf6c",  # Revert transfer batch
    "b1e5a33ae337d20e35e966b8d82a02a913d32689",  # Eliminate stream sync for LoRA
    "c087ddd6865a52634326a05af66429cb5531cd16",  # Refine pre_reorder_triton_kernel
    "da47621ccc4f8e8381f3249257489d5fe32aff1b",  # Minor speedup topk postprocessing
    "dd1012fcbe2a1fb36c44e10c16f8d0bcd8e9da25",  # [PD] Fix potential perf spike
    "ddcf9fe3beacd8aed573c711942194dd02350da4",  # Optimize triton attention mask
    "df7f61ee7d235936e6663f07813d7c03c4ec1603",  # Speed up rebalancing
    "e3ec6bf4b65a50e26e936a96adc7acc618292002",  # Minor speed up block_quant_dequant
]

# Commits without Docker images (skipped)
COMMITS_SKIPPED = [
    "4418f599a54699181b35d89b0def2697cccb721a",  # Fix FA3 DeepSeek prefill regression
    "2a413829f42b8e8433a3e7cfd91cc9cb241cfbc0",  # Add triton version config key
    "5e02330137a1ce44f29cc41a4da5f010c4bffec6",  # [perf] dsv3 bmm fallback to bf16
]

RESULTS_DIR = Path("/root/OmniPerf-Bench/omniperf_results_3way_sglang")
BENCHMARK_SCRIPT = Path("/root/OmniPerf-Bench/scripts/runners/local_docker_sglang_benchmark.py")


def get_existing_results(commit: str) -> Dict[str, bool]:
    """Check which benchmark results already exist for a commit."""
    short = commit[:8]
    results = {}

    # Check human result
    human_file = RESULTS_DIR / "docker_benchmark_results" / f"{short}_human_serving.json"
    results["human"] = human_file.exists()

    # Check baseline result
    baseline_file = RESULTS_DIR / "baseline_benchmark_results" / f"{short}_baseline_serving.json"
    results["baseline"] = baseline_file.exists()

    # Check agent results
    for agent in ["claude_code", "codex"]:
        agent_file = RESULTS_DIR / "agent_benchmark_results" / f"{short}_agent_{agent}_serving.json"
        results[f"agent_{agent}"] = agent_file.exists()

    return results


def run_benchmark(commit: str, timeout: int = 1800, dry_run: bool = False) -> Dict[str, Any]:
    """Run 3-way benchmark for a single commit."""
    short = commit[:8]

    print(f"\n{'='*70}")
    print(f"BENCHMARKING: {short}")
    print(f"{'='*70}")

    cmd = [
        "python", str(BENCHMARK_SCRIPT),
        "--commit", commit,
        "--3way",
        "--agents", "claude_code,codex",
        "--timeout", str(timeout),
    ]

    if dry_run:
        cmd.append("--dry-run")
        print(f"  Command: {' '.join(cmd)}")
        return {"commit": short, "status": "dry_run"}

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout * 5  # Allow extra time for all phases
        )

        output = result.stdout + result.stderr

        # Check for success indicators
        success = "BENCHMARK_DONE" in output or "success" in output.lower()

        return {
            "commit": short,
            "status": "success" if result.returncode == 0 else "error",
            "returncode": result.returncode,
            "output_lines": output.split("\n")[-50:],  # Last 50 lines
        }

    except subprocess.TimeoutExpired:
        return {
            "commit": short,
            "status": "timeout",
            "error": f"Benchmark timed out after {timeout * 5}s"
        }
    except Exception as e:
        return {
            "commit": short,
            "status": "error",
            "error": str(e)
        }


def main():
    parser = argparse.ArgumentParser(description="Run 3-way SGLang benchmarks for all commits")
    parser.add_argument("--commits", type=str, help="Comma-separated list of commit hashes (short or full)")
    parser.add_argument("--timeout", type=int, default=1800, help="Timeout per benchmark phase (seconds)")
    parser.add_argument("--dry-run", action="store_true", help="Check images and show commands without running")
    parser.add_argument("--skip-existing", action="store_true", help="Skip commits with existing results")
    parser.add_argument("--start-from", type=int, default=0, help="Start from commit index (0-based)")
    args = parser.parse_args()

    # Determine which commits to run
    if args.commits:
        commits = []
        for c in args.commits.split(","):
            c = c.strip()
            # Match against known commits
            matching = [full for full in COMMITS_WITH_IMAGES if full.startswith(c)]
            if matching:
                commits.extend(matching)
            else:
                print(f"WARNING: No matching commit found for {c}")
        commits = list(set(commits))  # Remove duplicates
    else:
        commits = COMMITS_WITH_IMAGES[args.start_from:]

    print(f"\n{'='*70}")
    print("SGLang 3-WAY BENCHMARK RUNNER")
    print(f"{'='*70}")
    print(f"Total commits to benchmark: {len(commits)}")
    print(f"Timeout per phase: {args.timeout}s")
    print(f"Skip existing: {args.skip_existing}")
    print(f"Dry run: {args.dry_run}")

    # Create results directory
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # Run benchmarks
    results = []
    start_time = datetime.now()

    for idx, commit in enumerate(commits, start=1):
        short = commit[:8]
        print(f"\n[{idx}/{len(commits)}] Processing {short}...")

        # Check existing results
        existing = get_existing_results(commit)
        if args.skip_existing:
            if existing.get("human") and existing.get("baseline"):
                print(f"  Skipping {short} - results already exist")
                results.append({
                    "commit": short,
                    "status": "skipped",
                    "existing": existing
                })
                continue

        # Run benchmark
        result = run_benchmark(commit, timeout=args.timeout, dry_run=args.dry_run)
        results.append(result)

        # Print progress
        elapsed = (datetime.now() - start_time).total_seconds()
        avg_time = elapsed / idx
        remaining = avg_time * (len(commits) - idx)
        print(f"  Progress: {idx}/{len(commits)} | Elapsed: {elapsed/60:.1f}m | Est. remaining: {remaining/60:.1f}m")

    # Summary
    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")

    success_count = sum(1 for r in results if r.get("status") == "success")
    error_count = sum(1 for r in results if r.get("status") == "error")
    timeout_count = sum(1 for r in results if r.get("status") == "timeout")
    skipped_count = sum(1 for r in results if r.get("status") == "skipped")

    print(f"  Total:    {len(results)}")
    print(f"  Success:  {success_count}")
    print(f"  Error:    {error_count}")
    print(f"  Timeout:  {timeout_count}")
    print(f"  Skipped:  {skipped_count}")

    # Save run log
    log_file = RESULTS_DIR / f"batch_run_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(log_file, "w") as f:
        json.dump({
            "start_time": start_time.isoformat(),
            "end_time": datetime.now().isoformat(),
            "commits": [c[:8] for c in commits],
            "results": results,
        }, f, indent=2)
    print(f"\nRun log saved to: {log_file}")

    return 0 if error_count == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
