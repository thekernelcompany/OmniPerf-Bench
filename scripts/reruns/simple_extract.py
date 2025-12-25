#!/usr/bin/env python3
import json
from pathlib import Path
from collections import defaultdict

runs_dir = Path("perf-agents-bench/state/runs")
commits_by_repo = defaultdict(set)
commit_to_pre = {}
commit_to_item = {}

print("Scanning for claude-sonnet-45 journal files...")

count = 0
for journal_path in runs_dir.rglob("journal.json"):
    if "/claude-sonnet-45/" not in str(journal_path):
        continue
    count += 1
    try:
        journal = json.loads(journal_path.read_text())
        parts = journal_path.parts
        repo = None
        for i, p in enumerate(parts):
            if p == "runs" and i + 1 < len(parts):
                repo = parts[i + 1]
                break
        if repo in ["vllm", "sglang"]:
            commits = journal.get("commits", {})
            hc = commits.get("human", "")
            pc = commits.get("pre", "")
            if hc:
                commits_by_repo[repo].add(hc)
                commit_to_item[hc] = journal_path.parent.name
                if pc:
                    commit_to_pre[hc] = pc
    except:
        pass

print(f"Processed {count} journals")
for repo in sorted(commits_by_repo.keys()):
    print(f"{repo}: {len(commits_by_repo[repo])} commits")

output_dir = Path("perf-agents-bench/state")
output_dir.mkdir(parents=True, exist_ok=True)

for repo in sorted(commits_by_repo.keys()):
    commits = sorted(commits_by_repo[repo])
    plan_path = output_dir / f"plan_claude_sonnet45_rerun_{repo}.json"
    items = []
    for idx, commit in enumerate(commits, 1):
        item = {
            "item_id": commit_to_item.get(commit, f"{repo}_core-{idx:04d}"),
            "human": commit,
        }
        pc = commit_to_pre.get(commit, "")
        if pc:
            item["pre"] = pc
            item["pre_parent_index"] = 1
        else:
            item["pre"] = None
        items.append(item)
    
    plan = {
        "repo": f"/home/ubuntu/OmniPerf-Bench/{repo}",
        "task_id": f"{repo}_core",
        "items": items
    }
    
    plan_path.write_text(json.dumps(plan, indent=2))
    print(f"Created {plan_path}: {len(items)} commits")

