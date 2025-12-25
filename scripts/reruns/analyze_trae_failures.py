#!/usr/bin/env python3

import json
import sys
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Set, Tuple

def find_all_journals(runs_dir: Path) -> List[Tuple[Path, Dict]]:
    journals = []
    for journal_path in runs_dir.rglob("journal.json"):
        try:
            with open(journal_path, 'r') as f:
                journal = json.load(f)
                journals.append((journal_path, journal))
        except Exception as e:
            print(f"Error reading {journal_path}: {e}", file=sys.stderr)
    return journals

def analyze_trae_failures(runs_dir: Path) -> Dict:
    journals = find_all_journals(runs_dir)
    
    commit_statuses: Dict[str, List[Dict]] = defaultdict(list)
    all_commits: Set[str] = set()
    
    for journal_path, journal in journals:
        path_parts = journal_path.parts
        
        repo = "unknown"
        agent = "unknown"
        model = "unknown"
        run_id = "unknown"
        
        for i, part in enumerate(path_parts):
            if part == "runs":
                if i + 1 < len(path_parts):
                    repo = path_parts[i + 1]
            elif part == "trae" and i > 0:
                agent = "trae"
                if i + 1 < len(path_parts):
                    model = path_parts[i + 1]
                    if i + 2 < len(path_parts):
                        run_id = path_parts[i + 2]
        
        if agent.lower() != "trae":
            continue
        
        commits = journal.get("commits", {})
        human_commit = commits.get("human", "")
        status = journal.get("status", "unknown")
        error = journal.get("error")
        error_type = journal.get("error_type")
        
        if not human_commit:
            continue
        
        all_commits.add(human_commit)
        
        run_path = str(journal_path.parent)
        item_id = journal_path.parent.name
        
        attempt_info = {
            "status": status,
            "error": error,
            "error_type": error_type,
            "run_path": run_path,
            "run_id": run_id,
            "item_id": item_id,
            "repo": repo,
            "agent": agent,
            "model": model,
            "returncode": journal.get("trae", {}).get("returncode"),
            "duration_s": journal.get("trae", {}).get("duration_s"),
        }
        
        commit_statuses[human_commit].append(attempt_info)
    
    failing_commits = []
    all_failures = []
    commits_with_mixed_results = []
    
    for commit, attempts in commit_statuses.items():
        statuses = [a["status"] for a in attempts]
        errors = [a for a in attempts if a["status"] == "error"]
        successes = [a for a in attempts if a["status"] == "success"]
        
        has_success = any(s == "success" for s in statuses)
        has_error = any(s == "error" for s in statuses)
        
        if not has_success and has_error:
            failing_commits.append({
                "commit": commit,
                "attempts": len(attempts),
                "errors": len(errors),
                "error_details": errors
            })
            all_failures.extend(errors)
        elif has_success and has_error:
            commits_with_mixed_results.append({
                "commit": commit,
                "total_attempts": len(attempts),
                "successes": len(successes),
                "errors": len(errors),
                "error_details": errors
            })
    
    return {
        "total_commits_analyzed": len(all_commits),
        "commits_with_only_failures": len(failing_commits),
        "commits_with_mixed_results": len(commits_with_mixed_results),
        "total_failure_attempts": len(all_failures),
        "failing_commits": sorted(failing_commits, key=lambda x: x["attempts"], reverse=True),
        "commits_with_errors": sorted(commits_with_mixed_results, key=lambda x: x["errors"], reverse=True),
    }

def main():
    runs_dir = Path("perf-agents-bench/state/runs")
    if not runs_dir.exists():
        print(f"Error: {runs_dir} does not exist", file=sys.stderr)
        sys.exit(1)
    
    print("Analyzing TRAE failures...", file=sys.stderr)
    sys.stderr.flush()
    results = analyze_trae_failures(runs_dir)
    print(f"Analysis complete. Found {results['total_commits_analyzed']} commits.", file=sys.stderr)
    sys.stderr.flush()
    
    print("=" * 80)
    print("TRAE COMMIT FAILURE ANALYSIS")
    print("=" * 80)
    print(f"\nTotal commits analyzed: {results['total_commits_analyzed']}")
    print(f"Commits with ONLY failures (never succeeded): {results['commits_with_only_failures']}")
    print(f"Commits with mixed results (some success, some errors): {results['commits_with_mixed_results']}")
    print(f"Total failure attempts: {results['total_failure_attempts']}")
    
    print("\n" + "=" * 80)
    print("COMMITS THAT NEVER SUCCEEDED (FAILED ALL ATTEMPTS)")
    print("=" * 80)
    
    for i, commit_info in enumerate(results['failing_commits'], 1):
        commit = commit_info['commit']
        attempts = commit_info['attempts']
        errors = commit_info['errors']
        print(f"\n{i}. Commit: {commit[:12]}... (full: {commit})")
        print(f"   Total attempts: {attempts}")
        print(f"   All attempts failed: {errors}")
        print(f"   Error details:")
        for j, error_detail in enumerate(commit_info['error_details'], 1):
            error_msg = error_detail.get('error', 'null')
            error_type = error_detail.get('error_type', 'null')
            run_id = error_detail.get('run_id', 'unknown')
            item_id = error_detail.get('item_id', 'unknown')
            model = error_detail.get('model', 'unknown')
            print(f"      {j}. Run: {run_id}/{item_id}, Model: {model}")
            print(f"         Error: {error_msg[:100] if error_msg else 'null'}...")
            print(f"         Error Type: {error_type}")
    
    print("\n" + "=" * 80)
    print("COMMITS WITH MIXED RESULTS (SOME SUCCESS, SOME FAILURES)")
    print("=" * 80)
    
    for i, commit_info in enumerate(results['commits_with_errors'], 1):
        commit = commit_info['commit']
        total = commit_info['total_attempts']
        successes = commit_info['successes']
        errors = commit_info['errors']
        print(f"\n{i}. Commit: {commit[:12]}... (full: {commit})")
        print(f"   Total attempts: {total} (Successes: {successes}, Errors: {errors})")
        print(f"   Error details:")
        for j, error_detail in enumerate(commit_info['error_details'], 1):
            error_msg = error_detail.get('error', 'null')
            error_type = error_detail.get('error_type', 'null')
            run_id = error_detail.get('run_id', 'unknown')
            item_id = error_detail.get('item_id', 'unknown')
            model = error_detail.get('model', 'unknown')
            print(f"      {j}. Run: {run_id}/{item_id}, Model: {model}")
            print(f"         Error: {error_msg[:100] if error_msg else 'null'}...")
            print(f"         Error Type: {error_type}")
    
    output_file = Path("trae_failure_analysis.json")
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n\nFull results saved to: {output_file}")
    
    failing_commits_list = [c['commit'] for c in results['failing_commits']]
    output_list_file = Path("trae_failing_commits_list.txt")
    with open(output_list_file, 'w') as f:
        for commit in failing_commits_list:
            f.write(f"{commit}\n")
    print(f"List of failing commits saved to: {output_list_file}")

if __name__ == "__main__":
    main()

