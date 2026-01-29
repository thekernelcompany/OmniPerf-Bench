#!/usr/bin/env python3
"""
Verify the 31 SGLang commits that were processed during earlier failed runs.
"""
import json
from pathlib import Path
from collections import Counter

# These are the commits we saw in the worktree directory
previous_commits = [
    "021f76e4", "10189d08", "132dad87", "148254d4", "187b85b7",
    "205d5cb4", "23c764b1", "25e1816e", "2854a5ea", "2a413829",
    "2a754e57", "2f427491", "31589e17", "3212c2ad", "4418f599",
    "564a898a", "5e023301", "6a2941f4", "6cb00c63", "86a876d8",
    "8f8f96a6", "912788c0", "915140fd", "9183c23e", "93470a14",
    "9c745d07", "a191a0e4", "a37e1247", "a99801e0", "ab4a83b2",
    "ac971ff6"
]

# Find all SGLang run directories from today
run_dirs = list(Path("/home/ubuntu/OmniPerf-Bench/perf-agents-bench/state/runs").glob("sglan/trae/*/2025-12-24_*"))

print(f"=== SGLang Previous Runs Verification ===\n")
print(f"Found {len(run_dirs)} run directories from today\n")

all_results = []
for run_dir in sorted(run_dirs):
    print(f"Checking: {run_dir.name}")

    for commit_hash in previous_commits:
        item_id = f"sglang_sonnet45_rerun_{commit_hash}"
        journal_file = run_dir / item_id / "journal.json"

        if journal_file.exists():
            try:
                with open(journal_file) as f:
                    data = json.load(f)

                status = data.get("status", "unknown")

                # Check if patch exists
                patch_file = journal_file.parent / "model_patch.diff"
                has_patch = patch_file.exists() and patch_file.stat().st_size > 0

                all_results.append({
                    "commit": commit_hash,
                    "run_dir": run_dir.name,
                    "status": status,
                    "has_patch": has_patch
                })
            except Exception as e:
                print(f"  Error reading {item_id}: {e}")

print(f"\n=== Results Summary ===\n")
print(f"Total results found: {len(all_results)}\n")

if all_results:
    # Group by commit
    by_commit = {}
    for r in all_results:
        commit = r["commit"]
        if commit not in by_commit:
            by_commit[commit] = []
        by_commit[commit].append(r)

    print(f"Unique commits processed: {len(by_commit)}\n")

    # Count statuses
    statuses = Counter([r["status"] for r in all_results])
    print("Status breakdown:")
    for status, count in statuses.most_common():
        print(f"  {status}: {count}")

    # Count patches
    with_patch = sum(1 for r in all_results if r["has_patch"])
    without_patch = len(all_results) - with_patch
    print(f"\nPatch generation:")
    print(f"  With patch: {with_patch}")
    print(f"  Without patch: {without_patch}")

    # Show detailed results
    print(f"\n=== Detailed Results ===\n")
    for commit in sorted(by_commit.keys()):
        results = by_commit[commit]
        # Take the most recent result for each commit
        latest = results[-1]
        status_icon = "✓" if latest["status"] == "success" else "✗"
        patch_icon = "📄" if latest["has_patch"] else "  "
        print(f"{status_icon} {patch_icon} {commit} - {latest['status']}")

    # Summary stats
    success_count = sum(1 for commit, results in by_commit.items()
                       if results[-1]["status"] == "success")
    error_count = sum(1 for commit, results in by_commit.items()
                     if results[-1]["status"] == "error")

    print(f"\n=== Final Summary ===")
    print(f"Total unique commits: {len(by_commit)}")
    print(f"Success: {success_count} ({success_count/len(by_commit)*100:.1f}%)")
    print(f"Error: {error_count} ({error_count/len(by_commit)*100:.1f}%)")
    print(f"Success with patch: {sum(1 for commit, results in by_commit.items() if results[-1]['status'] == 'success' and results[-1]['has_patch'])}")

else:
    print("No results found! The commits may not have been processed yet.")
