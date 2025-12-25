#!/usr/bin/env python3
"""
Extract all commits that were run with claude-sonnet-45 model.
Creates a plan file for rerunning those commits.
"""

import json
import sys
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Set

def find_claude_sonnet45_journals(runs_dir: Path) -> List[tuple]:
    """Find all journal.json files for claude-sonnet-45 runs."""
    journals = []
    for journal_path in runs_dir.rglob("journal.json"):
        path_str = str(journal_path)
        if "/claude-sonnet-45/" in path_str:
            try:
                with open(journal_path, 'r') as f:
                    journal = json.load(f)
                    journals.append((journal_path, journal))
            except Exception as e:
                print(f"Error reading {journal_path}: {e}", file=sys.stderr)
    return journals

def extract_commit_info(journal_path: Path, journal: Dict) -> Dict:
    """Extract commit information from journal."""
    path_parts = journal_path.parts
    
    repo = "unknown"
    model = "unknown"
    run_timestamp = "unknown"
    item_id = journal_path.parent.name
    
    for i, part in enumerate(path_parts):
        if part == "runs":
            if i + 1 < len(path_parts):
                repo = path_parts[i + 1]
        elif part == "claude-sonnet-45":
            model = "claude-sonnet-45"
            if i + 1 < len(path_parts):
                run_timestamp = path_parts[i + 1]
    
    commits = journal.get("commits", {})
    human_commit = commits.get("human", "")
    pre_commit = commits.get("pre", "")
    status = journal.get("status", "unknown")
    
    return {
        "commit": human_commit,
        "pre_commit": pre_commit,
        "status": status,
        "repo": repo,
        "model": model,
        "run_timestamp": run_timestamp,
        "item_id": item_id,
    }

def main():
    runs_dir = Path("perf-agents-bench/state/runs")
    if not runs_dir.exists():
        print(f"Error: {runs_dir} does not exist", file=sys.stderr)
        sys.exit(1)
    
    print("Finding all claude-sonnet-45 runs...", file=sys.stderr)
    sys.stderr.flush()
    
    journals = find_claude_sonnet45_journals(runs_dir)
    print(f"Found {len(journals)} claude-sonnet-45 journal files", file=sys.stderr)
    sys.stderr.flush()
    
    commits_by_repo: Dict[str, Set[str]] = defaultdict(set)
    commit_details: Dict[str, Dict] = {}
    
    for journal_path, journal in journals:
        data = extract_commit_info(journal_path, journal)
        commit = data["commit"]
        repo = data["repo"]
        
        if not commit:
            continue
        
        commits_by_repo[repo].add(commit)
        
        if commit not in commit_details:
            commit_details[commit] = {
                "commit": commit,
                "pre_commit": data["pre_commit"],
                "repo": repo,
                "item_id": data["item_id"],
                "statuses": []
            }
        commit_details[commit]["statuses"].append(data["status"])
    
    print(f"\nFound commits by repository:", file=sys.stderr)
    for repo, commits in commits_by_repo.items():
        print(f"  {repo}: {len(commits)} unique commits", file=sys.stderr)
    
    sys.stderr.flush()
    
    output_dir = Path("perf-agents-bench/state")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    for repo, commits in commits_by_repo.items():
        plan_name = f"plan_claude_sonnet45_rerun_{repo}.json"
        plan_path = output_dir / plan_name
        
        items = []
        for idx, commit in enumerate(sorted(commits), 1):
            details = commit_details[commit]
            item_id = details.get("item_id", f"{repo}_core-{idx:04d}")
            
            items.append({
                "item_id": item_id,
                "human": commit,
                "pre": details.get("pre_commit", ""),
            })
        
        plan = {
            "repo": f"/home/ubuntu/OmniPerf-Bench/{repo}" if repo in ["vllm", "sglang"] else repo,
            "task_id": f"{repo}_core",
            "items": items
        }
        
        with open(plan_path, 'w') as f:
            json.dump(plan, f, indent=2)
        
        print(f"\nCreated plan: {plan_path}")
        print(f"  Total commits: {len(items)}")
        
        status_counts = defaultdict(int)
        for commit in commits:
            for status in commit_details[commit]["statuses"]:
                status_counts[status] += 1
        
        print(f"  Status breakdown:")
        for status, count in sorted(status_counts.items()):
            print(f"    {status}: {count}")
    
    all_commits_list = Path("claude_sonnet45_all_commits.txt")
    with open(all_commits_list, 'w') as f:
        for repo in sorted(commits_by_repo.keys()):
            f.write(f"# {repo} repository\n")
            for commit in sorted(commits_by_repo[repo]):
                f.write(f"{commit}\n")
            f.write("\n")
    print(f"\nCreated commit list: {all_commits_list}")

if __name__ == "__main__":
    main()

