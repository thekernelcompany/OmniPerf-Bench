#!/usr/bin/env python3
"""
Patch Quality Analysis Script

This script compares human patches (from vllm git history) with agent patches
to determine if agents found the SAME optimization or a DIFFERENT one.

Usage:
    source bench-env/bin/activate
    python scripts/analyze_patch_quality.py
"""

import json
import subprocess
from pathlib import Path
from dataclasses import dataclass
from typing import Optional


@dataclass
class PatchAnalysis:
    commit: str
    agent: str
    human_title: str
    human_files_changed: int
    human_lines: str
    agent_files_changed: int
    agent_lines: str
    human_optimization_type: str
    agent_optimization_type: str
    same_optimization: str  # "yes", "no", "maybe", "unknown"
    notes: str


def get_human_patch_info(commit: str, vllm_path: str = "vllm") -> dict:
    """Get human patch information from git."""
    try:
        # Get commit message
        result = subprocess.run(
            ["git", "show", commit, "--stat", "--format=%s"],
            cwd=vllm_path, capture_output=True, text=True
        )
        lines = result.stdout.strip().split('\n')
        title = lines[0] if lines else "Unknown"

        # Parse stats
        files_changed = 0
        insertions = 0
        deletions = 0
        for line in lines:
            if "file changed" in line or "files changed" in line:
                parts = line.split(',')
                for part in parts:
                    if "file" in part:
                        files_changed = int(part.strip().split()[0])
                    elif "insertion" in part:
                        insertions = int(part.strip().split()[0])
                    elif "deletion" in part:
                        deletions = int(part.strip().split()[0])

        return {
            "title": title,
            "files_changed": files_changed,
            "lines": f"+{insertions}/-{deletions}",
            "exists": True
        }
    except Exception as e:
        return {
            "title": f"Error: {e}",
            "files_changed": 0,
            "lines": "N/A",
            "exists": False
        }


def find_agent_patch(commit: str, agent: str) -> Optional[Path]:
    """Find agent patch file for a commit."""
    base_path = Path("/home/raven/coding-mess/kernel-corp/OmniPerf-Bench")

    # Search patterns by agent
    search_paths = {
        "claude-code": [
            base_path / "ISO-Bench/state/runs/vllm/claude_code",
        ],
        "codex": [
            base_path / "ISO-Bench/state/runs/vllm/codex",
        ],
        "trae": [
            base_path / "ISO-Bench/state/runs/vllm/trae",
            base_path / "trae_gpt5_hf_trajectories/vllm" / commit,
        ]
    }

    # Look for patch files
    for search_base in search_paths.get(agent, []):
        if not search_base.exists():
            continue

        # Check direct commit directory
        direct_path = search_base / commit / "model_patch.diff"
        if direct_path.exists():
            return direct_path

        # Search recursively for commit in directory names
        for path in search_base.rglob(f"*{commit}*/model_patch.diff"):
            return path

    return None


def classify_optimization(patch_content: str) -> str:
    """Classify the type of optimization from patch content."""
    if not patch_content or len(patch_content) < 50:
        return "empty_or_minimal"

    content_lower = patch_content.lower()

    # Check for different optimization types
    if "torch.empty" in content_lower and "torch.zeros" in content_lower:
        return "tensor_allocation"
    if "softmax" in content_lower or "log_min_p" in content_lower:
        return "algorithmic_change"
    if ".clear()" in content_lower or "early return" in content_lower:
        return "micro_optimization"
    if "def " in content_lower and patch_content.count("def ") > 2:
        return "architectural_change"
    if "cache" in content_lower or "precompute" in content_lower:
        return "caching_optimization"

    return "other"


def analyze_commits():
    """Analyze patches for key commits."""

    # Key commits from the analysis
    commits_to_analyze = [
        ("e3580537", "claude-code", "[Performance] Enable chunked prefill + prefix caching"),
        ("a3223766", "claude-code", "[Core] Optimize update checks in LogitsProcessor"),
        ("30172b49", "codex", "[V1] Optimize handling of sampling metadata"),
        ("30172b49", "trae", "[V1] Optimize handling of sampling metadata"),
        ("b55ed6ef", "codex", "[V1][Minor] Optimize token_ids_cpu copy"),
        ("b55ed6ef", "trae", "[V1][Minor] Optimize token_ids_cpu copy"),
        ("58eee5f2", "codex", "[PERF] Use faster decode in tokenizer"),
        ("58eee5f2", "trae", "[PERF] Use faster decode in tokenizer"),
    ]

    results = []

    print("=" * 100)
    print("PATCH QUALITY ANALYSIS")
    print("=" * 100)

    for commit, agent, expected_title in commits_to_analyze:
        print(f"\n### {commit} ({agent}) ###")

        # Get human patch info
        human_info = get_human_patch_info(commit)
        print(f"Human: {human_info['title'][:60]}...")
        print(f"  Files: {human_info['files_changed']}, Lines: {human_info['lines']}")

        # Find agent patch
        agent_patch_path = find_agent_patch(commit, agent)
        if agent_patch_path:
            print(f"Agent patch: {agent_patch_path}")
            patch_content = agent_patch_path.read_text()
            agent_lines = patch_content.count('\n')
            agent_files = patch_content.count('diff --git')
            agent_type = classify_optimization(patch_content)
            print(f"  Files: {agent_files}, Lines: ~{agent_lines}")
            print(f"  Type: {agent_type}")
        else:
            print(f"Agent patch: NOT FOUND")
            patch_content = ""
            agent_lines = 0
            agent_files = 0
            agent_type = "not_found"

        # Classify human optimization type from title
        title_lower = human_info['title'].lower()
        if "enable" in title_lower and "together" in title_lower:
            human_type = "architectural_change"
        elif "optimize" in title_lower:
            human_type = "targeted_optimization"
        elif "fix" in title_lower:
            human_type = "bug_fix"
        else:
            human_type = "other"

        # Determine if same optimization
        same = "no"
        if agent_type == "not_found":
            same = "unknown"
        elif agent_type == human_type:
            same = "maybe"
        elif "tokenizer" in human_info['title'].lower() and "_decode" in patch_content:
            same = "maybe"

        results.append(PatchAnalysis(
            commit=commit,
            agent=agent,
            human_title=human_info['title'][:80],
            human_files_changed=human_info['files_changed'],
            human_lines=human_info['lines'],
            agent_files_changed=agent_files,
            agent_lines=f"~{agent_lines}",
            human_optimization_type=human_type,
            agent_optimization_type=agent_type,
            same_optimization=same,
            notes=""
        ))

    # Summary
    print("\n" + "=" * 100)
    print("SUMMARY TABLE")
    print("=" * 100)
    print(f"\n{'Commit':<12} {'Agent':<14} {'Human Type':<22} {'Agent Type':<22} {'Same?':<8}")
    print("-" * 80)

    for r in results:
        print(f"{r.commit:<12} {r.agent:<14} {r.human_optimization_type:<22} {r.agent_optimization_type:<22} {r.same_optimization:<8}")

    # Save results
    output = {
        "patches": [
            {
                "commit": r.commit,
                "agent": r.agent,
                "human_title": r.human_title,
                "human_files": r.human_files_changed,
                "human_lines": r.human_lines,
                "agent_files": r.agent_files_changed,
                "agent_lines": r.agent_lines,
                "human_type": r.human_optimization_type,
                "agent_type": r.agent_optimization_type,
                "same_optimization": r.same_optimization
            }
            for r in results
        ]
    }

    output_path = Path("scripts/patch_quality_results.json")
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    analyze_commits()
