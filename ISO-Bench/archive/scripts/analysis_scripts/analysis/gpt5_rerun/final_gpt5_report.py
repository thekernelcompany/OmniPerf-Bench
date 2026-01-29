#!/usr/bin/env python3
import json
from pathlib import Path
from collections import Counter

base_dir = Path("/home/ubuntu/OmniPerf-Bench/perf-agents-bench/state/runs/vllm/trae/gpt-5-2025-08-07")

all_commits = {}
all_runs = []

# Get all journals
for run_dir in sorted(base_dir.glob("2025-*")):
    for journal_file in run_dir.glob("*/journal.json"):
        commit_hash = journal_file.parent.name.replace("vllm_gpt5_rerun_", "")

        try:
            data = json.loads(journal_file.read_text())
            status = data.get("status")

            # Track by commit hash - keep best status
            if commit_hash not in all_commits or status == "success":
                all_commits[commit_hash] = {
                    "status": status,
                    "run": run_dir.name
                }
        except:
            pass

# Count statuses
statuses = Counter(c["status"] for c in all_commits.values())

print("=" * 60)
print("FINAL GPT-5 vLLM RERUN RESULTS")
print("=" * 60)
print()
print(f"Total unique commits attempted: {len(all_commits)}")
print(f"Successes: {statuses['success']} ({statuses['success']/len(all_commits)*100:.1f}%)")
print(f"Errors: {statuses.get('error', 0)}")
print(f"Max steps exceeded: {statuses.get('max_steps_exceeded', 0)}")
print()
print(f"Target was: 53 unique failed commits")
print(f"Success rate: {statuses['success']}/53 = {statuses['success']/53*100:.1f}%")
print()
print("=" * 60)
