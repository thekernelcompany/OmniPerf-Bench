#!/usr/bin/env python3
import json
from pathlib import Path

run_dir = Path("/home/ubuntu/OmniPerf-Bench/perf-agents-bench/state/runs/vllm/trae/gpt-5-2025-08-07/2025-12-26_15-06-42")

empty_patches = []
small_patches = []
good_patches = []

for journal_file in run_dir.glob("*/journal.json"):
    try:
        data = json.loads(journal_file.read_text())
        if data.get("status") != "success":
            continue

        metrics = data.get("metrics", {})
        patch_size = metrics.get("patch_size_loc", 0)
        commit_id = journal_file.parent.name

        if patch_size == 0:
            empty_patches.append(commit_id)
        elif patch_size < 10:
            small_patches.append((commit_id, patch_size))
        else:
            good_patches.append((commit_id, patch_size))
    except:
        pass

print(f"=== Quality Analysis of {len(empty_patches) + len(small_patches) + len(good_patches)} Successful Commits ===\n")
print(f"Empty patches (0 LOC): {len(empty_patches)}")
for c in empty_patches[:5]:
    print(f"  - {c}")
if len(empty_patches) > 5:
    print(f"  ... and {len(empty_patches)-5} more")

print(f"\nSmall patches (<10 LOC): {len(small_patches)}")
for c, size in small_patches[:5]:
    print(f"  - {c}: {size} LOC")
if len(small_patches) > 5:
    print(f"  ... and {len(small_patches)-5} more")

print(f"\nGood patches (>=10 LOC): {len(good_patches)}")
avg = sum(s for _, s in good_patches) / len(good_patches) if good_patches else 0
print(f"  Average size: {avg:.1f} LOC")

print(f"\n=== Summary ===")
total = len(empty_patches) + len(small_patches) + len(good_patches)
print(f"Real optimizations (>0 LOC): {total - len(empty_patches)}/{total} ({(total-len(empty_patches))/total*100:.1f}%)")
print(f"False positives: {len(empty_patches)}/{total} ({len(empty_patches)/total*100:.1f}%)")
