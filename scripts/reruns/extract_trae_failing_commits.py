#!/usr/bin/env python3
"""
Extract all TRAE commits that failed from journal.json files.
Creates a comprehensive list of failing commits with details.
"""

import json
import sys
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Set

def find_all_trae_journals(runs_dir: Path) -> List[tuple]:
    """Find all journal.json files for TRAE runs."""
    journals = []
    for journal_path in runs_dir.rglob("journal.json"):
        path_str = str(journal_path)
        if "/trae/" in path_str:
            try:
                with open(journal_path, 'r') as f:
                    journal = json.load(f)
                    journals.append((journal_path, journal))
            except Exception as e:
                print(f"Error reading {journal_path}: {e}", file=sys.stderr)
    return journals

def extract_commit_from_journal(journal_path: Path, journal: Dict) -> Dict:
    """Extract commit information from journal."""
    path_parts = journal_path.parts
    
    repo = "unknown"
    agent = "unknown"
    model = "unknown"
    run_timestamp = "unknown"
    
    for i, part in enumerate(path_parts):
        if part == "runs":
            if i + 1 < len(path_parts):
                repo = path_parts[i + 1]
        elif part == "trae":
            agent = "trae"
            if i + 1 < len(path_parts):
                model = path_parts[i + 1]
                if i + 2 < len(path_parts):
                    run_timestamp = path_parts[i + 2]
    
    commits = journal.get("commits", {})
    human_commit = commits.get("human", "")
    status = journal.get("status", "unknown")
    error = journal.get("error")
    error_type = journal.get("error_type")
    
    item_id = journal_path.parent.name
    run_id = f"{run_timestamp}/{item_id}"
    
    return {
        "commit": human_commit,
        "status": status,
        "error": error,
        "error_type": error_type,
        "repo": repo,
        "model": model,
        "run_id": run_id,
        "item_id": item_id,
        "returncode": journal.get("trae", {}).get("returncode"),
        "duration_s": journal.get("trae", {}).get("duration_s"),
    }

def analyze_trae_commits(runs_dir: Path) -> Dict:
    """Analyze all TRAE journal files and categorize commits by status."""
    journals = find_all_trae_journals(runs_dir)
    
    print(f"Found {len(journals)} TRAE journal files", file=sys.stderr)
    sys.stderr.flush()
    
    commit_data: Dict[str, List[Dict]] = defaultdict(list)
    all_commits: Set[str] = set()
    
    for journal_path, journal in journals:
        data = extract_commit_from_journal(journal_path, journal)
        commit = data["commit"]
        
        if not commit:
            continue
        
        all_commits.add(commit)
        commit_data[commit].append(data)
    
    failing_commits = []
    commits_with_errors = []
    
    for commit, attempts in commit_data.items():
        statuses = [a["status"] for a in attempts]
        errors = [a for a in attempts if a["status"] == "error"]
        successes = [a for a in attempts if a["status"] == "success"]
        
        has_success = any(s == "success" for s in statuses)
        has_error = any(s == "error" for s in statuses)
        
        if not has_success and has_error:
            failing_commits.append({
                "commit": commit,
                "total_attempts": len(attempts),
                "error_count": len(errors),
                "attempts": attempts
            })
        elif has_success and has_error:
            commits_with_errors.append({
                "commit": commit,
                "total_attempts": len(attempts),
                "success_count": len(successes),
                "error_count": len(errors),
                "error_attempts": errors
            })
    
    return {
        "total_commits": len(all_commits),
        "commits_never_succeeded": len(failing_commits),
        "commits_with_some_errors": len(commits_with_errors),
        "failing_commits": sorted(failing_commits, key=lambda x: x["total_attempts"], reverse=True),
        "commits_with_errors": sorted(commits_with_errors, key=lambda x: x["error_count"], reverse=True),
    }

def main():
    runs_dir = Path("perf-agents-bench/state/runs")
    if not runs_dir.exists():
        print(f"Error: {runs_dir} does not exist", file=sys.stderr)
        sys.exit(1)
    
    print("Analyzing TRAE commits...", file=sys.stderr)
    sys.stderr.flush()
    
    results = analyze_trae_commits(runs_dir)
    
    print("=" * 80)
    print("TRAE COMMIT FAILURE ANALYSIS")
    print("=" * 80)
    print(f"\nTotal unique commits analyzed: {results['total_commits']}")
    print(f"Commits that NEVER succeeded (all attempts failed): {results['commits_never_succeeded']}")
    print(f"Commits with mixed results (some success, some errors): {results['commits_with_some_errors']}")
    
    print("\n" + "=" * 80)
    print("COMMITS THAT NEVER SUCCEEDED (ALL ATTEMPTS FAILED)")
    print("=" * 80)
    
    for i, commit_info in enumerate(results['failing_commits'], 1):
        commit = commit_info['commit']
        attempts = commit_info['total_attempts']
        print(f"\n{i}. {commit}")
        print(f"   Attempts: {attempts} (all failed)")
        print(f"   Error details:")
        for j, attempt in enumerate(commit_info['attempts'], 1):
            model = attempt.get('model', 'unknown')
            run_id = attempt.get('run_id', 'unknown')
            error_msg = attempt.get('error') or 'null'
            error_type = attempt.get('error_type') or 'null'
            duration = attempt.get('duration_s')
            print(f"      {j}. Model: {model}, Run: {run_id}")
            if error_msg and error_msg != 'null':
                print(f"         Error: {error_msg[:150]}")
            if error_type and error_type != 'null':
                print(f"         Error Type: {error_type}")
            if duration:
                print(f"         Duration: {duration:.1f}s")
    
    print("\n" + "=" * 80)
    print("COMMITS WITH MIXED RESULTS (SOME SUCCESS, SOME FAILURES)")
    print("=" * 80)
    
    for i, commit_info in enumerate(results['commits_with_errors'], 1):
        commit = commit_info['commit']
        total = commit_info['total_attempts']
        successes = commit_info['success_count']
        errors = commit_info['error_count']
        print(f"\n{i}. {commit}")
        print(f"   Total attempts: {total} (Successes: {successes}, Errors: {errors})")
        print(f"   Error attempts:")
        for j, attempt in enumerate(commit_info['error_attempts'], 1):
            model = attempt.get('model', 'unknown')
            run_id = attempt.get('run_id', 'unknown')
            error_msg = attempt.get('error') or 'null'
            error_type = attempt.get('error_type') or 'null'
            print(f"      {j}. Model: {model}, Run: {run_id}")
            if error_msg and error_msg != 'null':
                print(f"         Error: {error_msg[:150]}")
            if error_type and error_type != 'null':
                print(f"         Error Type: {error_type}")
    
    output_json = Path("trae_failing_commits_analysis.json")
    with open(output_json, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n\nFull JSON analysis saved to: {output_json}")
    
    failing_only = [c['commit'] for c in results['failing_commits']]
    output_list = Path("trae_failing_commits_list.txt")
    with open(output_list, 'w') as f:
        for commit in failing_only:
            f.write(f"{commit}\n")
    print(f"List of commits that never succeeded saved to: {output_list}")
    
    all_failing_commits = failing_only + [c['commit'] for c in results['commits_with_errors']]
    all_failing_list = Path("trae_all_failing_commits_list.txt")
    with open(all_failing_list, 'w') as f:
        for commit in sorted(set(all_failing_commits)):
            f.write(f"{commit}\n")
    print(f"List of all commits with any failures saved to: {all_failing_list}")

if __name__ == "__main__":
    main()

