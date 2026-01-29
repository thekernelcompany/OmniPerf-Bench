#!/usr/bin/env python3
"""Create rerun plans for TRAE + GPT-5 failed commits."""

import json
from pathlib import Path

print("="*70)
print("Creating TRAE GPT-5 Rerun Plans")
print("="*70)

# Read failed commit lists
vllm_commits_file = Path("/tmp/trae_gpt5_vllm_failed.txt")
sglang_commits_file = Path("/tmp/trae_gpt5_sglang_failed.txt")

if not vllm_commits_file.exists():
    print(f"ERROR: {vllm_commits_file} not found")
    print("Run the extraction script first to generate failed commit lists")
    exit(1)

if not sglang_commits_file.exists():
    print(f"ERROR: {sglang_commits_file} not found")
    print("Run the extraction script first to generate failed commit lists")
    exit(1)

vllm_commits = [c for c in vllm_commits_file.read_text().strip().split("\n") if c]
sglang_commits = [c for c in sglang_commits_file.read_text().strip().split("\n") if c]

print(f"\nLoaded {len(vllm_commits)} vLLM failed commits")
print(f"Loaded {len(sglang_commits)} SGLang failed commits")

# vLLM plan
vllm_plan = {
    "repo": str(Path("/home/ubuntu/OmniPerf-Bench/vllm").resolve()),
    "task_id": "vllm_gpt5_rerun",
    "items": [
        {
            "item_id": f"vllm_gpt5_rerun_{commit[:8]}",
            "human": commit,
            "pre": "",
            "pre_parent_index": 1,
        }
        for idx, commit in enumerate(vllm_commits, 1)
    ]
}

# SGLang plan
sglang_plan = {
    "repo": str(Path("/home/ubuntu/OmniPerf-Bench/sglang").resolve()),
    "task_id": "sglang_gpt5_rerun",
    "items": [
        {
            "item_id": f"sglang_gpt5_rerun_{commit[:8]}",
            "human": commit,
            "pre": "",
            "pre_parent_index": 1,
        }
        for idx, commit in enumerate(sglang_commits, 1)
    ]
}

# Write plans
output_dir = Path("/home/ubuntu/OmniPerf-Bench/perf-agents-bench/state")
output_dir.mkdir(parents=True, exist_ok=True)

vllm_plan_path = output_dir / "plan_trae_gpt5_vllm_rerun.json"
sglang_plan_path = output_dir / "plan_trae_gpt5_sglang_rerun.json"

vllm_plan_path.write_text(json.dumps(vllm_plan, indent=2))
sglang_plan_path.write_text(json.dumps(sglang_plan, indent=2))

print(f"\n{'='*70}")
print("Plans Created Successfully")
print(f"{'='*70}")
print(f"✓ vLLM rerun plan: {vllm_plan_path}")
print(f"  - Commits: {len(vllm_commits)}")
print(f"  - Task ID: vllm_gpt5_rerun")
print(f"\n✓ SGLang rerun plan: {sglang_plan_path}")
print(f"  - Commits: {len(sglang_commits)}")
print(f"  - Task ID: sglang_gpt5_rerun")
print(f"\n{'='*70}")
