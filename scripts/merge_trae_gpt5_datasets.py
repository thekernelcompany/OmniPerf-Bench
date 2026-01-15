#!/usr/bin/env python3
"""Merge TRAE GPT-5 trajectories from HuggingFace + local-only clean runs."""

from huggingface_hub import hf_hub_download
from pathlib import Path
import json
import shutil

DATASET_ID = "Inferencebench/trae-gpt5-trajectories"
LOCAL_BASE = Path("perf-agents-bench/state/runs/vllm/trae/gpt-5-merged")
GIT_SOURCE = Path("perf-agents-bench/state/runs/vllm/trae/gpt-5")  # All timestamp folders

# 17 clean local-only commits (no "No tool output" bug)
GIT_ONLY_CLEAN = [
    "015069b0", "2a052011", "2deb029d", "2f192835", "3476ed08", "526de822", "660470e5",
    "6d0734c5", "6dd94dbe", "7661e92e", "7c01f706", "80aa7e91", "83450458", "88693683",
    "89a84b0b", "e7b20426", "ec3b5ce9"
]

def main():
    # Clean up existing merged directory
    if LOCAL_BASE.exists():
        print(f"Removing existing {LOCAL_BASE}")
        shutil.rmtree(LOCAL_BASE)

    # Step 1: Download HuggingFace runs (53)
    print("=== Downloading HuggingFace runs ===")
    metadata_path = hf_hub_download(DATASET_ID, "metadata.json", repo_type="dataset")
    with open(metadata_path) as f:
        metadata = json.load(f)

    timestamp = "2025-12-26_15-06-42"  # HuggingFace timestamp
    hf_count = 0

    # Files to download for each commit
    file_names = ["journal.json", "model_patch.diff", "prediction.jsonl", "run_summary.json", "task.txt", "trajectory.json"]

    for item in metadata['data']['vllm']:
        commit = item['commit']
        item_id = f"vllm_gpt5_rerun_{commit}"

        local_dir = LOCAL_BASE / timestamp / item_id
        local_dir.mkdir(parents=True, exist_ok=True)

        # Download files using actual HuggingFace path structure: vllm/<commit>/<file>
        for file_name in file_names:
            hf_path = f"vllm/{commit}/{file_name}"
            try:
                cached_path = hf_hub_download(DATASET_ID, hf_path, repo_type="dataset")
                local_file = local_dir / file_name
                shutil.copy(cached_path, local_file)
            except Exception as e:
                # Some files might not exist (e.g., prediction.jsonl for errors)
                pass

        hf_count += 1
        print(f"  ✓ {item_id} ({item['status']})")

    print(f"\nDownloaded {hf_count} HuggingFace runs")

    # Step 2: Copy clean local-only runs (17) from various timestamps
    print("\n=== Copying clean local-only runs ===")
    found_commits = set()

    for ts_dir in sorted(GIT_SOURCE.iterdir(), reverse=True):  # Latest first
        if not ts_dir.is_dir():
            continue
        for item_dir in ts_dir.iterdir():
            if not item_dir.is_dir():
                continue
            summary_file = item_dir / "run_summary.json"
            if not summary_file.exists():
                continue
            with open(summary_file) as f:
                data = json.load(f)
            commit = data['commits']['human'][:8]

            if commit in GIT_ONLY_CLEAN and commit not in found_commits:
                dest_dir = LOCAL_BASE / ts_dir.name / item_dir.name
                shutil.copytree(item_dir, dest_dir)
                found_commits.add(commit)
                print(f"  ✓ {item_dir.name} from {ts_dir.name}")

    print(f"\nCopied {len(found_commits)} clean local-only runs")

    print(f"\n=== SUMMARY ===")
    print(f"Merged dataset: {LOCAL_BASE}")
    print(f"HuggingFace runs: {hf_count}")
    print(f"Local-only clean runs: {len(found_commits)}")
    print(f"Total: {hf_count + len(found_commits)} runs")

if __name__ == "__main__":
    main()
