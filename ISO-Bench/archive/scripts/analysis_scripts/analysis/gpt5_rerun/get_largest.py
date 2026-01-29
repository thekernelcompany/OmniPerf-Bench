#!/usr/bin/env python3
import json
from pathlib import Path

run_dir = Path("/home/ubuntu/OmniPerf-Bench/perf-agents-bench/state/runs/vllm/trae/gpt-5-2025-08-07/2025-12-26_15-06-42")

patches = []
for journal_file in run_dir.glob("*/journal.json"):
    try:
        data = json.loads(journal_file.read_text())
        if data.get("status") == "success":
            size = data.get("metrics", {}).get("patch_size_loc", 0)
            patches.append((size, journal_file.parent.name))
    except:
        pass

patches.sort(reverse=True)
print("Top 3 largest patches:")
for size, name in patches[:3]:
    print(f"  {size} LOC - {name}")
