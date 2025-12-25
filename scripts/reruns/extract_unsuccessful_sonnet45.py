#!/usr/bin/env python3
"""
Extract all unsuccessful TRAE Sonnet 4.5 commits from the evaluation analysis.
"""
import re
from pathlib import Path

def extract_unsuccessful_commits():
    """Extract unsuccessful commits from DEEP_ANALYSIS.md"""

    analysis_file = Path("perf-agents-bench/eval_results_v2/DEEP_ANALYSIS.md")

    unsuccessful_commits = []
    unsuccessful_types = [
        "AGENT_NO_PATCH",
        "BASELINE_OOM",
        "BASELINE_CUDA_ERROR",
        "BASELINE_IMPORT_ERROR",
        "BASELINE_ATTRIBUTE_ERROR",
        "BASELINE_TYPE_ERROR",
        "BASELINE_RUNTIME_ERROR",
        "BASELINE_ASSERTION",
        "BASELINE_EXCEPTION",
        "BASELINE_UNKNOWN",
        "TEST_IMPORT_ERROR",
        "TARGET_NOT_RESOLVED",
        "OPT_PATH_NOT_HIT",
        "GIT_WORKTREE_FAILED",
        "NO_TEST_SCRIPT",
        "PATCH_INVALID",
        "PATCH_APPLY_FAILED",
        "UNKNOWN",
        "SUCCESS_REGRESSION"  # Include regressions as "unsuccessful"
    ]

    with open(analysis_file) as f:
        content = f.read()

    # Find sections for each unsuccessful type
    for section_name in unsuccessful_types:
        # Find the section
        section_pattern = rf"## {section_name} \((\d+) runs"
        section_match = re.search(section_pattern, content)

        if not section_match:
            continue

        # Extract the section content
        section_start = section_match.end()
        next_section = re.search(r"\n## ", content[section_start:])
        if next_section:
            section_end = section_start + next_section.start()
        else:
            section_end = len(content)

        section_content = content[section_start:section_end]

        # Find all table rows with TRAE + claude-sonnet-45
        # Table format: | item_id | agent | model | repo | commit |
        pattern = r"\|\s*([^\|]+?)\s*\|\s*trae\s*\|\s*claude-sonnet-45\s*\|\s*([^\|]+?)\s*\|\s*`([a-f0-9]+)`\s*\|"

        for match in re.finditer(pattern, section_content):
            item_id = match.group(1).strip()
            repo = match.group(2).strip()
            commit = match.group(3).strip()

            unsuccessful_commits.append({
                'item_id': item_id,
                'repo': repo,
                'commit': commit,
                'reason': section_name
            })

    return unsuccessful_commits

def main():
    print("Extracting unsuccessful TRAE Sonnet 4.5 commits...")

    commits = extract_unsuccessful_commits()

    # Deduplicate by commit hash
    unique_commits = {}
    for c in commits:
        if c['commit'] not in unique_commits:
            unique_commits[c['commit']] = c
        else:
            # Append reason if different
            if c['reason'] not in unique_commits[c['commit']]['reason']:
                unique_commits[c['commit']]['reason'] += f", {c['reason']}"

    # Group by repo
    by_repo = {'vllm': [], 'sglang': []}
    for commit_hash, info in unique_commits.items():
        repo = info['repo']
        if repo in by_repo:
            by_repo[repo].append(info)

    # Print summary
    print(f"\n✓ Found {len(unique_commits)} unique unsuccessful commits for TRAE + Sonnet 4.5")
    print(f"  - vLLM: {len(by_repo['vllm'])} commits")
    print(f"  - SGLang: {len(by_repo['sglang'])} commits")

    # Write to files
    output_dir = Path("perf-agents-bench")

    # Write vLLM commits
    if by_repo['vllm']:
        vllm_file = output_dir / "TRAE_SONNET45_VLLM_UNSUCCESSFUL.txt"
        with open(vllm_file, 'w') as f:
            for info in sorted(by_repo['vllm'], key=lambda x: x['commit']):
                f.write(f"{info['commit']}  # {info['item_id']} - {info['reason']}\n")
        print(f"\n✓ Written vLLM commits to {vllm_file}")

    # Write SGLang commits
    if by_repo['sglang']:
        sglang_file = output_dir / "TRAE_SONNET45_SGLANG_UNSUCCESSFUL.txt"
        with open(sglang_file, 'w') as f:
            for info in sorted(by_repo['sglang'], key=lambda x: x['commit']):
                f.write(f"{info['commit']}  # {info['item_id']} - {info['reason']}\n")
        print(f"✓ Written SGLang commits to {sglang_file}")

    # Write combined commit list (just hashes)
    all_file = output_dir / "TRAE_SONNET45_ALL_UNSUCCESSFUL.txt"
    with open(all_file, 'w') as f:
        for commit_hash in sorted(unique_commits.keys()):
            info = unique_commits[commit_hash]
            f.write(f"{commit_hash}\n")
    print(f"✓ Written all commit hashes to {all_file}")

    # Print breakdown by failure reason
    print("\n=== Breakdown by Failure Reason ===")
    reason_counts = {}
    for info in unique_commits.values():
        reasons = info['reason'].split(', ')
        for reason in reasons:
            reason_counts[reason] = reason_counts.get(reason, 0) + 1

    for reason, count in sorted(reason_counts.items(), key=lambda x: -x[1]):
        print(f"  {reason:30s}: {count:3d} commits")

if __name__ == "__main__":
    main()
