#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "=== Setting up Codex vLLM runs ==="
echo "Working directory: $SCRIPT_DIR"

# Check prerequisites
if ! command -v codex &> /dev/null; then
    echo "ERROR: Codex CLI not found. Please install it first."
    exit 1
fi

if [ ! -d "vllm" ]; then
    echo "ERROR: vllm repository not found. Expected at: $SCRIPT_DIR/vllm"
    exit 1
fi

if [ ! -d "bench-env" ]; then
    echo "ERROR: bench-env virtual environment not found."
    exit 1
fi

# Check Codex authentication
echo "Checking Codex authentication..."
if ! codex whoami &> /dev/null; then
    echo "WARNING: Codex CLI not authenticated. You may need to run 'codex login' first."
    echo "Continuing anyway, but authentication may be required during execution."
fi

# Setup Codex home directory for non-interactive runs
if [ ! -d "ISO-Bench/.codex_home" ]; then
    echo "Setting up Codex home directory..."
    mkdir -p ISO-Bench/.codex_home
    if [ -d "$HOME/.codex" ]; then
        echo "Copying Codex state from $HOME/.codex to ISO-Bench/.codex_home"
        rsync -a "$HOME/.codex/" ISO-Bench/.codex_home/.codex/ || true
    else
        echo "WARNING: $HOME/.codex not found. Codex may prompt for authentication during runs."
    fi
fi

# Build plan from all commit JSONs
echo "Building plan from commit JSONs..."
COMMIT_DIR="hf_cache/alpha-vllm-99-commits/vllm_commits_separated"
PLAN_OUT="ISO-Bench/state/plan_codex_full.json"

if [ ! -d "$COMMIT_DIR" ]; then
    echo "ERROR: Commit directory not found: $COMMIT_DIR"
    exit 1
fi

python3 << 'PYEOF'
import json
from pathlib import Path

commit_dir = Path("hf_cache/alpha-vllm-99-commits/vllm_commits_separated")
if not commit_dir.exists():
    raise FileNotFoundError(f"Commit directory not found: {commit_dir}")

items = []
for idx, path in enumerate(sorted(commit_dir.glob("*.json")), 1):
    try:
        data = json.loads(path.read_text())
        commit_hash = data.get("commit_hash") or path.stem
        items.append({
            "item_id": f"vllm_core-{idx:04d}",
            "human": commit_hash,
            "pre": "",
            "pre_parent_index": 1,
        })
    except Exception as e:
        print(f"WARNING: Failed to process {path.name}: {e}")
        continue

if not items:
    raise ValueError("No valid commit items found!")

vllm_path = Path("vllm").resolve()
if not vllm_path.exists():
    raise FileNotFoundError(f"vLLM repository not found at: {vllm_path}")

plan = {
    "repo": str(vllm_path),
    "task_id": "vllm_core",
    "items": items
}

out = Path("ISO-Bench/state/plan_codex_full.json")
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(json.dumps(plan, indent=2))
print(f"✓ Created plan with {len(items)} commits: {out}")
PYEOF

if [ ! -f "$PLAN_OUT" ]; then
    echo "ERROR: Failed to create plan file"
    exit 1
fi

PLAN_COUNT=$(python3 -c "import json; print(len(json.load(open('$PLAN_OUT'))['items']))")
echo "Plan contains $PLAN_COUNT commits"

# Check if tmux session already exists
if tmux has-session -t codex_vllm 2>/dev/null; then
    echo "WARNING: tmux session 'codex_vllm' already exists."
    echo "You can attach with: tmux attach -t codex_vllm"
    echo "Or kill it first with: tmux kill-session -t codex_vllm"
    read -p "Continue anyway? (y/N) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

echo ""
echo "=== Setup complete ==="
echo "Plan file: $PLAN_OUT"
echo "Commits: $PLAN_COUNT"
echo ""
echo "To run the bench harness, execute the run script in tmux:"
echo "  ./run_codex_vllm_bench.sh"



