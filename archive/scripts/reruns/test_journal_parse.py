#!/usr/bin/env python3

import json
from pathlib import Path

runs_dir = Path("perf-agents-bench/state/runs")
journals = list(runs_dir.rglob("journal.json"))

print(f"Found {len(journals)} total journals")

trae_journals = []
for journal_path in journals:
    if "trae" in str(journal_path):
        trae_journals.append(journal_path)

print(f"Found {len(trae_journals)} TRAE journals")

if trae_journals:
    sample = trae_journals[0]
    print(f"\nSample journal: {sample}")
    with open(sample, 'r') as f:
        data = json.load(f)
        print(f"Status: {data.get('status')}")
        print(f"Commits: {data.get('commits')}")
        
    error_count = 0
    success_count = 0
    for jp in trae_journals[:100]:
        try:
            with open(jp, 'r') as f:
                data = json.load(f)
                if data.get('status') == 'error':
                    error_count += 1
                elif data.get('status') == 'success':
                    success_count += 1
        except:
            pass
    print(f"\nFirst 100: errors={error_count}, successes={success_count}")

