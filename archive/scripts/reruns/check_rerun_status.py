#!/usr/bin/env python3
import json
from pathlib import Path
from collections import Counter

run_dir = Path("/home/ubuntu/OmniPerf-Bench/perf-agents-bench/state/runs/vllm/trae/us-anthropic-claude-sonnet-4-5-20250929-v1-0/2025-12-23_22-06-06")

statuses = []
successes_with_patch = 0
successes_no_patch = 0
errors = 0

for journal_file in run_dir.glob("*/journal.json"):
    try:
        with open(journal_file) as f:
            data = json.load(f)
            status = data.get("status", "unknown")
            statuses.append(status)

            if status == "success":
                # Check if patch was generated
                patch_file = journal_file.parent / "model_patch.diff"
                if patch_file.exists() and patch_file.stat().st_size > 0:
                    successes_with_patch += 1
                else:
                    successes_no_patch += 1
            else:
                errors += 1
    except Exception as e:
        print(f"Error reading {journal_file}: {e}")

print("=== TRAE Sonnet 4.5 vLLM Rerun Status ===\n")
print(f"Total commits completed: {len(statuses)}")
print(f"Commits remaining: {91 - len(statuses)}\n")

print("Status breakdown:")
status_counts = Counter(statuses)
for status, count in status_counts.most_common():
    print(f"  {status}: {count}")

print(f"\nDetailed success breakdown:")
print(f"  Success with patch: {successes_with_patch}")
print(f"  Success without patch: {successes_no_patch}")
print(f"  Errors/failures: {errors}")

if len(statuses) > 0:
    success_rate = (successes_with_patch / len(statuses)) * 100
    print(f"\nSuccess rate (with patch): {success_rate:.1f}%")
