#!/bin/bash
set -e

echo "=========================================="
echo "TRAE Claude Sonnet 4.5 Rerun Setup"
echo "=========================================="
echo ""

cd /home/ubuntu/OmniPerf-Bench

echo "Step 1: Extracting commits from claude-sonnet-45 runs..."
python3 << 'PYTHON_SCRIPT'
import json
from pathlib import Path
from collections import defaultdict

runs_dir = Path("perf-agents-bench/state/runs")
commits_by_repo = defaultdict(set)
commit_to_pre = {}

print("Scanning journal files...", flush=True)

for journal_path in runs_dir.rglob("journal.json"):
    if "/claude-sonnet-45/" not in str(journal_path):
        continue
    
    try:
        journal = json.loads(journal_path.read_text())
        path_parts = journal_path.parts
        
        repo = None
        for i, part in enumerate(path_parts):
            if part == "runs" and i + 1 < len(path_parts):
                repo = path_parts[i + 1]
                break
        
        if not repo:
            continue
        
        commits = journal.get("commits", {})
        human_commit = commits.get("human", "")
        pre_commit = commits.get("pre", "")
        
        if human_commit:
            commits_by_repo[repo].add(human_commit)
            if pre_commit:
                commit_to_pre[human_commit] = pre_commit
    except Exception as e:
        print(f"Error reading {journal_path}: {e}", flush=True)

print(f"\nFound commits by repository:", flush=True)
for repo, commits in sorted(commits_by_repo.items()):
    print(f"  {repo}: {len(commits)} commits", flush=True)

output_dir = Path("perf-agents-bench/state")
output_dir.mkdir(parents=True, exist_ok=True)

for repo, commits in sorted(commits_by_repo.items()):
    plan_name = f"plan_claude_sonnet45_rerun_{repo}.json"
    plan_path = output_dir / plan_name
    
    items = []
    for idx, commit in enumerate(sorted(commits), 1):
        pre_commit = commit_to_pre.get(commit, "")
        item_id = f"{repo}_core-{idx:04d}" if repo == "vllm" else f"{repo}_{idx:03d}_{commit[:8]}"
        
        item = {
            "item_id": item_id,
            "human": commit,
            "pre": pre_commit if pre_commit else None
        }
        if pre_commit:
            item["pre_parent_index"] = 1
        items.append(item)
    
    repo_path = f"/home/ubuntu/OmniPerf-Bench/{repo}"
    
    plan = {
        "repo": repo_path,
        "task_id": f"{repo}_core",
        "items": items
    }
    
    plan_path.write_text(json.dumps(plan, indent=2))
    print(f"\nCreated: {plan_path}")
    print(f"  Commits: {len(items)}")

PYTHON_SCRIPT

echo ""
echo "Step 2: Reviewing generated plans..."
echo ""

for plan_file in perf-agents-bench/state/plan_claude_sonnet45_rerun_*.json; do
    if [ -f "$plan_file" ]; then
        repo=$(basename "$plan_file" | sed 's/plan_claude_sonnet45_rerun_\(.*\)\.json/\1/')
        commit_count=$(python3 -c "import json; print(len(json.load(open('$plan_file'))['items']))")
        echo "  $plan_file: $commit_count commits ($repo)"
    fi
done

echo ""
echo "=========================================="
echo "Setup Complete!"
echo "=========================================="
echo ""
echo "Next steps to run the rerun:"
echo ""
echo "1. Ensure AWS credentials are set:"
echo "   aws sso login --sso-session your-session"
echo "   # OR set AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY"
echo ""
echo "2. Set environment variables:"
echo "   export TRAE_PYTHON=/home/ubuntu/OmniPerf-Bench/bench-env/bin/python"
echo "   export TRAE_CONFIG=/home/ubuntu/OmniPerf-Bench/third-party/trae-agent/trae_config.yaml"
echo "   export AWS_REGION=us-east-1"
echo ""
echo "3. Run the pipeline (example for vllm):"
echo "   cd perf-agents-bench"
echo "   source ../bench-env/bin/activate"
echo "   python -m bench.cli prepare \\"
echo "       tasks/vllm.yaml \\"
echo "       --from-plan state/plan_claude_sonnet45_rerun_vllm.json \\"
echo "       --bench-cfg bench.yaml \\"
echo "       --max-workers 1 \\"
echo "       --resume"
echo ""
echo "4. For SGLang:"
echo "   python -m bench.cli prepare \\"
echo "       tasks/sglang.yaml \\"
echo "       --from-plan state/plan_claude_sonnet45_rerun_sglang.json \\"
echo "       --bench-cfg bench.yaml \\"
echo "       --max-workers 1 \\"
echo "       --resume"
echo ""

