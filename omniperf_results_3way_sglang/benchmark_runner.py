#!/usr/bin/env python3
"""
SGLang 3-way Benchmark Runner using Python Overlay Approach

This script runs 3-way benchmarks (baseline, human, agent variants) for SGLang commits.
It uses a working base Docker image and applies patches to measure performance.

Usage:
    python benchmark_runner.py --commit 148254d4 --dry-run
    python benchmark_runner.py --commit 148254d4 --agents claude_code,codex,trae_gpt5
"""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from datetime import datetime

# Configuration
RESULTS_DIR = Path("/root/OmniPerf-Bench/omniperf_results_3way_sglang/benchmark_results")
COMMIT_MAPPING_FILE = Path("/root/OmniPerf-Bench/src/benchmark/fixes/sglang_commit_mapping.json")
AGENT_BASE_DIR = Path("/root/OmniPerf-Bench/perf-agents-bench/state/runs/sglang")

# Agent run paths
AGENT_PATHS = {
    "claude_code": AGENT_BASE_DIR / "claude_code/default/2025-12-23_06-28-44",
    "codex": AGENT_BASE_DIR / "codex/gpt-5/389be848",
    "trae_gpt5": AGENT_BASE_DIR / "trae/gpt-5/2025-11-14_21-05-32",
}

# 14 commits with Docker images (from the plan)
BENCHMARK_COMMITS = [
    "187b85b7", "6b231325", "6cb00c63", "148254d4", "2bd18e2d", "2a754e57",
    "880221bd", "b1e5a33a", "c087ddd6", "da47621c", "dd1012fc", "ddcf9fe3",
    "df7f61ee", "e3ec6bf4"
]


def load_commit_mapping():
    """Load the commit mapping JSON."""
    with open(COMMIT_MAPPING_FILE) as f:
        data = json.load(f)

    mapping = {}
    for commit in data["commits"]:
        short = commit["human_commit"][:8]
        mapping[short] = {
            "human_commit": commit["human_commit"],
            "base_commit": commit["base_commit"],
            "pr_number": commit.get("pr_number"),
            "subject": commit.get("subject")
        }
    return mapping


def find_agent_patch(commit_short: str, agent: str) -> Path | None:
    """Find the patch file for a commit and agent variant."""
    agent_path = AGENT_PATHS.get(agent)
    if not agent_path or not agent_path.exists():
        return None

    # Search for the commit folder
    for folder in agent_path.iterdir():
        if folder.is_dir():
            # Claude code uses: sglang_004_148254d4
            if commit_short in folder.name:
                patch_file = folder / "model_patch.diff"
                if patch_file.exists():
                    return patch_file
            # Codex uses: sglang_core-0001, need to check prompt.json
            elif folder.name.startswith("sglang_core-"):
                prompt_file = folder / "prompt.json"
                if prompt_file.exists():
                    with open(prompt_file) as f:
                        data = json.load(f)
                        human = data.get("commits", {}).get("human", "")
                        if human.startswith(commit_short):
                            patch_file = folder / "model_patch.diff"
                            if patch_file.exists():
                                return patch_file
    return None


def get_human_patch_url(pr_number: str) -> str:
    """Get the GitHub diff URL for a PR."""
    return f"https://patch-diff.githubusercontent.com/raw/sgl-project/sglang/pull/{pr_number}.diff"


def run_benchmark_in_docker(
    commit_short: str,
    variant: str,
    patch_path: str | None,
    base_commit: str,
    dry_run: bool = False
) -> dict:
    """
    Run a benchmark inside Docker.

    Returns a dict with benchmark results or error info.
    """
    result = {
        "commit": commit_short,
        "variant": variant,
        "timestamp": datetime.now().isoformat(),
        "status": "pending"
    }

    if dry_run:
        print(f"  [DRY RUN] Would run {variant} benchmark for {commit_short}")
        if patch_path:
            print(f"  [DRY RUN] Patch: {patch_path}")
        result["status"] = "dry_run"
        return result

    # TODO: Implement actual Docker benchmark execution
    # This would:
    # 1. Start the rebuilt Docker image
    # 2. Apply the patch (if provided)
    # 3. Run bench_latency.py or start server + run client
    # 4. Collect metrics

    result["status"] = "not_implemented"
    return result


def run_3way_benchmark(
    commit_short: str,
    agents: list[str],
    dry_run: bool = False
) -> dict:
    """
    Run a 3-way benchmark for a commit.

    Benchmarks:
    1. Baseline (parent commit)
    2. Human (the optimization commit)
    3. Agent variants (patches applied to baseline)
    """
    mapping = load_commit_mapping()

    if commit_short not in mapping:
        print(f"ERROR: Commit {commit_short} not found in mapping")
        return {"error": f"Commit {commit_short} not found"}

    commit_info = mapping[commit_short]
    base_commit = commit_info["base_commit"]
    human_commit = commit_info["human_commit"]
    pr_number = commit_info.get("pr_number")
    subject = commit_info.get("subject", "Unknown")

    print(f"\n{'='*60}")
    print(f"Commit: {commit_short}")
    print(f"Subject: {subject}")
    print(f"Base: {base_commit[:12]}")
    print(f"Human: {human_commit[:12]}")
    print(f"PR: #{pr_number}")
    print(f"{'='*60}")

    results = {
        "commit": commit_short,
        "human_commit": human_commit,
        "base_commit": base_commit,
        "subject": subject,
        "benchmarks": {}
    }

    # 1. Baseline benchmark
    print("\n[1/N] Running BASELINE benchmark...")
    results["benchmarks"]["baseline"] = run_benchmark_in_docker(
        commit_short, "baseline", None, base_commit, dry_run
    )

    # 2. Human benchmark
    print("\n[2/N] Running HUMAN benchmark...")
    human_patch_url = get_human_patch_url(pr_number) if pr_number else None
    results["benchmarks"]["human"] = run_benchmark_in_docker(
        commit_short, "human", human_patch_url, base_commit, dry_run
    )

    # 3. Agent benchmarks
    for i, agent in enumerate(agents, start=3):
        print(f"\n[{i}/N] Running {agent.upper()} benchmark...")
        patch_path = find_agent_patch(commit_short, agent)
        if patch_path:
            results["benchmarks"][agent] = run_benchmark_in_docker(
                commit_short, agent, str(patch_path), base_commit, dry_run
            )
        else:
            print(f"  WARNING: No patch found for {agent}")
            results["benchmarks"][agent] = {
                "status": "no_patch",
                "error": f"Patch not found for {agent}"
            }

    return results


def main():
    parser = argparse.ArgumentParser(description="SGLang 3-way Benchmark Runner")
    parser.add_argument("--commit", type=str, help="Specific commit to benchmark (8-char hash)")
    parser.add_argument("--all", action="store_true", help="Benchmark all 14 commits")
    parser.add_argument("--agents", type=str, default="claude_code,codex,trae_gpt5",
                        help="Comma-separated list of agent variants")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be done without running")
    parser.add_argument("--list-commits", action="store_true", help="List available commits")
    parser.add_argument("--check-patches", action="store_true", help="Check patch availability")

    args = parser.parse_args()

    # Create results directory
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    agents = [a.strip() for a in args.agents.split(",")]

    if args.list_commits:
        mapping = load_commit_mapping()
        print("\nAvailable commits with Docker images:")
        for commit in BENCHMARK_COMMITS:
            info = mapping.get(commit, {})
            subject = info.get("subject", "Unknown")[:50]
            print(f"  {commit}: {subject}")
        return

    if args.check_patches:
        print("\nPatch availability check:")
        print(f"{'Commit':<12} {'claude_code':<15} {'codex':<15} {'trae_gpt5':<15}")
        print("-" * 60)
        for commit in BENCHMARK_COMMITS:
            patches = []
            for agent in ["claude_code", "codex", "trae_gpt5"]:
                patch = find_agent_patch(commit, agent)
                patches.append("✓" if patch else "✗")
            print(f"{commit:<12} {patches[0]:<15} {patches[1]:<15} {patches[2]:<15}")
        return

    commits_to_run = []
    if args.all:
        commits_to_run = BENCHMARK_COMMITS
    elif args.commit:
        commits_to_run = [args.commit]
    else:
        parser.print_help()
        return

    all_results = []
    for commit in commits_to_run:
        result = run_3way_benchmark(commit, agents, args.dry_run)
        all_results.append(result)

    # Save results
    if not args.dry_run:
        output_file = RESULTS_DIR / f"benchmark_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(output_file, 'w') as f:
            json.dump(all_results, f, indent=2)
        print(f"\nResults saved to: {output_file}")

    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    for result in all_results:
        print(f"\n{result['commit']}: {result.get('subject', '')[:40]}")
        for variant, bench in result.get("benchmarks", {}).items():
            status = bench.get("status", "unknown")
            print(f"  {variant}: {status}")


if __name__ == "__main__":
    main()
