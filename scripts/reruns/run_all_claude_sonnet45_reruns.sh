#!/bin/bash
set -e

echo "=========================================="
echo "Claude Sonnet 4.5 Complete Rerun Script"
echo "=========================================="
echo ""

cd /home/ubuntu/OmniPerf-Bench

echo "Step 1: Creating rerun plan files from existing claude-sonnet-45 runs..."
echo ""

python3 << 'PYTHON_EOF'
import json
from pathlib import Path
from collections import defaultdict

runs_dir = Path("perf-agents-bench/state/runs")
commits_by_repo = defaultdict(set)
commit_to_pre = {}
commit_to_item = {}

print("Scanning journal files...", flush=True)

journal_count = 0
for journal_path in runs_dir.rglob("journal.json"):
    path_str = str(journal_path)
    if "/claude-sonnet-45/" not in path_str:
        continue
    
    journal_count += 1
    if journal_count % 50 == 0:
        print(f"  Processed {journal_count} files...", flush=True)
    
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
        print(f"Error reading {journal_path}: {e}", flush=True)

print(f"\nTotal journal files: {journal_count}", flush=True)
print(f"\nFound commits by repository:", flush=True)
for repo in sorted(commits_by_repo.keys()):
    print(f"  {repo}: {len(commits_by_repo[repo])} unique commits", flush=True)

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
    print(f"\n✓ Created: {plan_path}", flush=True)
    print(f"  Repository: {repo}", flush=True)
    print(f"  Total commits: {len(items)}", flush=True)

PYTHON_EOF

echo ""
echo "Step 2: Verifying plan files were created..."
echo ""

for repo in vllm sglang; do
    plan_file="perf-agents-bench/state/plan_claude_sonnet45_rerun_${repo}.json"
    if [ -f "$plan_file" ]; then
        commit_count=$(python3 -c "import json; print(len(json.load(open('$plan_file'))['items']))" 2>/dev/null || echo "0")
        echo "  ✓ $plan_file: $commit_count commits"
    else
        echo "  ✗ $plan_file: NOT FOUND"
    fi
done

echo ""
echo "=========================================="
echo "Setup Complete!"
echo "=========================================="
echo ""
echo "Next steps:"
echo ""
echo "1. Set AWS credentials:"
echo "   aws sso login --sso-session your-session"
echo "   # OR"
echo "   export AWS_ACCESS_KEY_ID=..."
echo "   export AWS_SECRET_ACCESS_KEY=..."
echo "   export AWS_REGION=us-east-1"
echo ""
echo "2. Set TRAE environment variables:"
echo "   export TRAE_PYTHON=/home/ubuntu/OmniPerf-Bench/bench-env/bin/python"
echo "   export TRAE_CONFIG=/home/ubuntu/OmniPerf-Bench/third-party/trae-agent/trae_config.yaml"
echo ""
echo "3. Run vLLM rerun:"
echo "   cd perf-agents-bench"
echo "   source ../bench-env/bin/activate"
echo "   python -m bench.cli prepare tasks/vllm.yaml --from-plan state/plan_claude_sonnet45_rerun_vllm.json --bench-cfg bench.yaml --max-workers 1 --resume"
echo ""
echo "4. Run SGLang rerun:"
echo "   python -m bench.cli prepare tasks/sglang.yaml --from-plan state/plan_claude_sonnet45_rerun_sglang.json --bench-cfg bench.yaml --max-workers 1 --resume"
echo ""

