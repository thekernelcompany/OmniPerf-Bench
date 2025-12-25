#!/usr/bin/env python3
"""
Create filtered plans for rerunning unsuccessful TRAE Sonnet 4.5 commits.
"""
import json
from pathlib import Path

def load_commit_list(filepath):
    """Load commit hashes from a text file (ignoring comments)"""
    commits = []
    with open(filepath) as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith('#'):
                # Extract commit hash (before any comment)
                commit = line.split('#')[0].strip()
                if commit:
                    commits.append(commit)
    return commits

def create_plan(repo_name, repo_path, task_id, commits, output_file):
    """Create a plan JSON file for the given commits"""

    items = []
    for commit in commits:
        # Create item_id from repo and commit hash (first 8 chars)
        short_hash = commit[:8]
        item_id = f"{task_id}_{short_hash}"

        items.append({
            "item_id": item_id,
            "human": commit,
            "pre": None  # Will be auto-resolved
        })

    plan = {
        "repo": repo_path,
        "task_id": task_id,
        "items": items
    }

    with open(output_file, 'w') as f:
        json.dump(plan, f, indent=2)

    print(f"✓ Created plan with {len(items)} commits: {output_file}")
    return len(items)

def main():
    print("Creating rerun plans for unsuccessful TRAE Sonnet 4.5 commits...")

    base_dir = Path("/home/ubuntu/OmniPerf-Bench")
    perf_agents = base_dir / "perf-agents-bench"
    state_dir = perf_agents / "state"

    # Load unsuccessful commit lists
    vllm_file = perf_agents / "TRAE_SONNET45_VLLM_UNSUCCESSFUL.txt"
    sglang_file = perf_agents / "TRAE_SONNET45_SGLANG_UNSUCCESSFUL.txt"

    vllm_commits = load_commit_list(vllm_file)
    sglang_commits = load_commit_list(sglang_file)

    print(f"\n✓ Loaded {len(vllm_commits)} vLLM commits")
    print(f"✓ Loaded {len(sglang_commits)} SGLang commits")

    # Create plans
    total_commits = 0

    # vLLM plan
    vllm_plan = state_dir / "plan_trae_sonnet45_vllm_rerun.json"
    total_commits += create_plan(
        repo_name="vllm",
        repo_path=str(base_dir / "vllm"),
        task_id="vllm_sonnet45_rerun",
        commits=vllm_commits,
        output_file=vllm_plan
    )

    # SGLang plan
    sglang_plan = state_dir / "plan_trae_sonnet45_sglang_rerun.json"
    total_commits += create_plan(
        repo_name="sglang",
        repo_path=str(base_dir / "sglang"),
        task_id="sglang_sonnet45_rerun",
        commits=sglang_commits,
        output_file=sglang_plan
    )

    print(f"\n=== Summary ===")
    print(f"Total commits to rerun: {total_commits}")
    print(f"  - vLLM: {len(vllm_commits)} commits")
    print(f"  - SGLang: {len(sglang_commits)} commits")

    print(f"\n=== Next Steps ===")
    print(f"To run the vLLM rerun:")
    print(f"  cd perf-agents-bench")
    print(f"  .venv/bin/python -m bench.cli prepare tasks/vllm.yaml \\")
    print(f"    --from-plan {vllm_plan.relative_to(perf_agents)} \\")
    print(f"    --bench-cfg bench.yaml \\")
    print(f"    --max-workers 1")
    print()
    print(f"To run the SGLang rerun:")
    print(f"  cd perf-agents-bench")
    print(f"  .venv/bin/python -m bench.cli prepare tasks/sglang.yaml \\")
    print(f"    --from-plan {sglang_plan.relative_to(perf_agents)} \\")
    print(f"    --bench-cfg bench.yaml \\")
    print(f"    --max-workers 1")

if __name__ == "__main__":
    main()
