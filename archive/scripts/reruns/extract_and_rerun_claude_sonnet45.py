#!/usr/bin/env python3
"""
Extract all commits that were run with claude-sonnet-45 and create rerun plans.
"""

import json
import sys
from pathlib import Path
from collections import defaultdict

def extract_claude_sonnet45_commits():
    """Extract all commits from claude-sonnet-45 runs."""
    runs_dir = Path("perf-agents-bench/state/runs")
    
    commits_by_repo = defaultdict(set)
    commit_to_item = {}
    commit_to_pre = {}
    
    print("Scanning for claude-sonnet-45 journal files...", file=sys.stderr)
    
    journal_count = 0
    for journal_path in runs_dir.rglob("journal.json"):
        path_str = str(journal_path)
        if "/claude-sonnet-45/" in path_str:
            journal_count += 1
            try:
                with open(journal_path, 'r') as f:
                    journal = json.load(f)
                
                path_parts = journal_path.parts
                repo = "unknown"
                for i, part in enumerate(path_parts):
                    if part == "runs" and i + 1 < len(path_parts):
                        repo = path_parts[i + 1]
                        break
                
                commits = journal.get("commits", {})
                human_commit = commits.get("human", "")
                pre_commit = commits.get("pre", "")
                
                if human_commit:
                    commits_by_repo[repo].add(human_commit)
                    commit_to_item[human_commit] = journal_path.parent.name
                    if pre_commit:
                        commit_to_pre[human_commit] = pre_commit
            except Exception as e:
                print(f"Error reading {journal_path}: {e}", file=sys.stderr)
    
    print(f"Found {journal_count} journal files", file=sys.stderr)
    sys.stderr.flush()
    
    output_dir = Path("perf-agents-bench/state")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    for repo, commits in commits_by_repo.items():
        if repo == "unknown":
            continue
            
        plan_name = f"plan_claude_sonnet45_rerun_{repo}.json"
        plan_path = output_dir / plan_name
        
        items = []
        for idx, commit in enumerate(sorted(commits), 1):
            item_id = commit_to_item.get(commit, f"{repo}_core-{idx:04d}")
            pre_commit = commit_to_pre.get(commit, "")
            
            items.append({
                "item_id": item_id,
                "human": commit,
                "pre": pre_commit,
                "pre_parent_index": 1 if pre_commit else None
            })
        
        repo_path = f"/home/ubuntu/OmniPerf-Bench/{repo}" if repo in ["vllm", "sglang"] else repo
        
        plan = {
            "repo": repo_path,
            "task_id": f"{repo}_core",
            "items": items
        }
        
        with open(plan_path, 'w') as f:
            json.dump(plan, f, indent=2)
        
        print(f"\nCreated plan: {plan_path}")
        print(f"  Repository: {repo}")
        print(f"  Total commits: {len(items)}")
        sys.stdout.flush()
    
    return commits_by_repo

if __name__ == "__main__":
    commits_by_repo = extract_claude_sonnet45_commits()
    
    print("\n" + "="*60)
    print("Summary:")
    print("="*60)
    for repo, commits in commits_by_repo.items():
        if repo != "unknown":
            print(f"{repo}: {len(commits)} commits")
    print("\nNext steps:")
    print("1. Review the generated plan files in perf-agents-bench/state/")
    print("2. Run the pipeline using the instructions in README.md")

