#!/usr/bin/env python3
"""Add missing codex entries to the HuggingFace dataset."""

import json
from datetime import datetime
from pathlib import Path

from datasets import load_dataset, Dataset
from huggingface_hub import HfApi

DATASET_ID = "Inferencebench/claude-code-vllm-benchmarks"
STATE_ROOT = Path("perf-agents-bench/state")

# Missing commits to add
MISSING_COMMITS = [
    {
        "commit_short": "3476ed08",
        "commit_hash": "3476ed0809ec91a3457da0cb90543133a4f4b519",
        "parent_commit": "54600709b6d419fb243ce718a48ab7d40f5c3eb7",
        "has_agent_patch": True,
    },
    {
        "commit_short": "6ce01f30",
        "commit_hash": "6ce01f30667bbae33f112152e07a3b66b841078f",
        "parent_commit": "6a11fdfbb8d6701c7ad38648aead23d8cbe6aac5",
        "has_agent_patch": True,
    },
    {
        "commit_short": "99abb8b6",
        "commit_hash": "99abb8b650c66664cdc84d815b7f306f33bd9881",
        "parent_commit": "3a1e6481586ed7f079275b5d5072a6e246af691e",
        "has_agent_patch": True,
    },
    {
        "commit_short": "fa63e710",
        "commit_hash": "fa63e710c7fbaae3a445f669d3b5ba6b9a4ef412",
        "parent_commit": "2a0309a646b1ed83a0c40974e08c8dc628726d3c",
        "has_agent_patch": True,
    },
]


def create_entry(commit_info: dict) -> dict:
    """Create a dataset entry for a missing commit."""
    return {
        "commit_hash": commit_info["commit_hash"],
        "commit_short": commit_info["commit_short"],
        "commit_subject": None,
        "repo": "vllm-project/vllm",
        "perf_command": None,
        "files_changed": None,
        "pr_url": None,
        "models": None,
        "parent_commit": commit_info["parent_commit"],
        "gpu_config": None,
        "benchmark_mode": None,
        "agent_name": "codex",
        "agent_model": "gpt-5",
        "benchmark_date": datetime.now().strftime("%Y-%m-%d"),
        "model": None,
        "has_agent_patch": commit_info["has_agent_patch"],
        "patch_path": None,
        # All benchmark metrics are None (not yet benchmarked)
        "baseline_ttft_mean": None,
        "baseline_ttft_median": None,
        "baseline_ttft_p99": None,
        "baseline_tpot_mean": None,
        "baseline_tpot_median": None,
        "baseline_tpot_p99": None,
        "baseline_itl_mean": None,
        "baseline_itl_median": None,
        "baseline_itl_p99": None,
        "baseline_latency_avg": None,
        "baseline_throughput": None,
        "human_ttft_mean": None,
        "human_ttft_median": None,
        "human_ttft_p99": None,
        "human_tpot_mean": None,
        "human_tpot_median": None,
        "human_tpot_p99": None,
        "human_itl_mean": None,
        "human_itl_median": None,
        "human_itl_p99": None,
        "human_latency_avg": None,
        "human_throughput": None,
        "agent_ttft_mean": None,
        "agent_ttft_median": None,
        "agent_ttft_p99": None,
        "agent_tpot_mean": None,
        "agent_tpot_median": None,
        "agent_tpot_p99": None,
        "agent_itl_mean": None,
        "agent_itl_median": None,
        "agent_itl_p99": None,
        "agent_latency_avg": None,
        "agent_throughput": None,
        "human_improvement_ttft_mean": None,
        "human_improvement_tpot_mean": None,
        "human_improvement_itl_mean": None,
        "agent_improvement_ttft_mean": None,
        "agent_improvement_tpot_mean": None,
        "agent_improvement_itl_mean": None,
        "agent_vs_human_ttft_mean": None,
        "agent_vs_human_tpot_mean": None,
        "agent_vs_human_itl_mean": None,
        "human_improvement_ttft_median": None,
        "human_improvement_ttft_p99": None,
        "agent_improvement_ttft_median": None,
        "agent_improvement_ttft_p99": None,
        "agent_vs_human_ttft_median": None,
        "agent_vs_human_ttft_p99": None,
        "human_improvement_latency_avg": None,
        "human_improvement_throughput": None,
        "agent_improvement_latency_avg": None,
        "agent_improvement_throughput": None,
        "agent_vs_human_latency_avg": None,
        "agent_vs_human_throughput": None,
        "baseline_raw": None,
        "human_raw": None,
        "agent_raw": None,
        "test_script": None,
        "data_source": "codex_modal",
    }


def main():
    print(f"Loading dataset: {DATASET_ID}")
    ds = load_dataset(DATASET_ID, split="train")

    # Check which commits are already in the dataset
    existing_commits = set(r["commit_short"] for r in ds if r.get("agent_name") == "codex")
    print(f"Existing codex commits: {len(existing_commits)}")

    # Find commits to add
    new_entries = []
    for commit_info in MISSING_COMMITS:
        commit_short = commit_info["commit_short"]
        if commit_short in existing_commits:
            print(f"  {commit_short}: already exists, skipping")
        else:
            print(f"  {commit_short}: adding new entry")
            new_entries.append(create_entry(commit_info))

    if not new_entries:
        print("\nNo new entries to add.")
        return

    print(f"\nAdding {len(new_entries)} new entries...")

    # Convert existing dataset to list and add new entries
    all_data = list(ds)
    all_data.extend(new_entries)

    # Create new dataset
    new_ds = Dataset.from_list(all_data)

    print(f"New dataset size: {len(new_ds)} rows")

    # Verify new entries
    new_codex_commits = set(r["commit_short"] for r in new_ds if r.get("agent_name") == "codex")
    print(f"Codex commits after update: {len(new_codex_commits)}")

    for commit_info in MISSING_COMMITS:
        status = "✓" if commit_info["commit_short"] in new_codex_commits else "✗"
        print(f"  {status} {commit_info['commit_short']}")

    # Push to hub
    print(f"\nPushing to HuggingFace: {DATASET_ID}")
    new_ds.push_to_hub(DATASET_ID, split="train")
    print("Done!")


if __name__ == "__main__":
    main()
