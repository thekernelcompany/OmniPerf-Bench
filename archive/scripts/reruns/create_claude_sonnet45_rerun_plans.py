#!/usr/bin/env python3
"""
Extract all commits that were run with claude-sonnet-45 and create rerun plan files.
"""

import json
import sys
from pathlib import Path
from collections import defaultdict

def main():
    runs_dir = Path("perf-agents-bench/state/runs")
    if not runs_dir.exists():
        print(f"Error: {runs_dir} does not exist", file=sys.stderr)
        sys.exit(1)
    
    commits_by_repo = defaultdict(set)
    commit_to_pre = {}
    commit_to_item = {}
    
    print("Scanning journal files for claude-sonnet-45 runs...")
    sys.stdout.flush()
    
    journal_count = 0
    for journal_path in runs_dir.rglob("journal.json"):
        path_str = str(journal_path)
        if "/claude-sonnet-45/" not in path_str:
            continue
        
        journal_count += 1
        if journal_count % 100 == 0:
            print(f"  Processed {journal_count} files...")
            sys.stdout.flush()
        
        try:
            journal = json.loads(journal_path.read_text())
            path_parts = journal_path.parts
            
            repo = None
            for i, part in enumerate(path_parts):
                if part == "runs" and i + 1 < len(path_parts):
                    repo = path_parts[i + 1]
                    break
            
            if repo not in ["vllm", "sglang"]:
                continue
            
            commits = journal.get("commits", {})
            human_commit = commits.get("human", "")
            pre_commit = commits.get("pre", "")
            
            if human_commit:
                commits_by_repo[repo].add(human_commit)
                commit_to_item[human_commit] = journal_path.parent.name
                if pre_commit:
                    commit_to_pre[human_commit] = pre_commit
        except Exception as e:
            print(f"Error reading {journal_path}: {e}")
    
    print(f"\nTotal journal files processed: {journal_count}")
    print(f"\nCommits by repository:")
    for repo in sorted(commits_by_repo.keys()):
        commits = commits_by_repo[repo]
        print(f"  {repo}: {len(commits)} unique commits")
    sys.stdout.flush()
    
    output_dir = Path("perf-agents-bench/state")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    for repo in sorted(commits_by_repo.keys()):
        commits = sorted(commits_by_repo[repo])
        plan_name = f"plan_claude_sonnet45_rerun_{repo}.json"
        plan_path = output_dir / plan_name
        
        items = []
        for idx, commit in enumerate(commits, 1):
            item_id = commit_to_item.get(commit, f"{repo}_core-{idx:04d}")
            pre_commit = commit_to_pre.get(commit, "")
            
            item = {
                "item_id": item_id,
                "human": commit,
            }
            
            if pre_commit:
                item["pre"] = pre_commit
                item["pre_parent_index"] = 1
            else:
                item["pre"] = None
            
            items.append(item)
        
        repo_path = f"/home/ubuntu/OmniPerf-Bench/{repo}"
        
        plan = {
            "repo": repo_path,
            "task_id": f"{repo}_core",
            "items": items
        }
        
        plan_path.write_text(json.dumps(plan, indent=2))
        print(f"Created: {plan_path}")
        print(f"  Repository: {repo}")
        print(f"  Total commits: {len(items)}")
        sys.stdout.flush()

if __name__ == "__main__":
    main()

