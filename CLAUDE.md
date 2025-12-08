# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

---

## Quick Mental Model

OmniPerf-Bench is a **performance optimization benchmark framework** with two pipelines:

1. **Dataset Generation**: Extract optimizations from git history → Generate LLM tests → Measure speedups → Create JSONL datasets
2. **Agent Evaluation**: Run AI agents (OpenHands, TRAE, Codex) on optimization tasks → Grade results → Compute Opt@K metrics

**Core Safety Mechanism**: Git worktrees isolate agent execution - agents commit freely without corrupting the main repository.

---

## Essential Commands

### Dataset Generation
```bash
# Main entry point - processes all commits in extractions_dir
python commit_to_dataset.py configs/experiments.yaml

# Single commit (for debugging)
python run_commit_optimization.py \
  --commit-json inputs/experiments/commit_extractions_with_apis/<hash>.json \
  --test-script inputs/experiments/generated_test_generators_v4/<hash8>_test_case_generator.py \
  --repo-path vllm
```

### Agent Evaluation (perf-agents-bench)
```bash
cd perf-agents-bench
source .venv/bin/activate

# Three-phase workflow: Plan → Prepare → Report
.venv/bin/python -m bench.cli plan tasks/vllm.yaml --out state/plan.json
.venv/bin/python -m bench.cli prepare tasks/vllm.yaml --from-plan state/plan.json --bench-cfg bench.yaml
.venv/bin/python -m bench.cli report state/runs/<run_id>

# Other useful commands
.venv/bin/python -m bench.cli doctor --bench-cfg bench.yaml  # Check prerequisites
.venv/bin/python -m bench.cli validate tasks/vllm.yaml       # Validate task config
.venv/bin/python -m bench.cli smoke tasks/vllm.yaml          # Quick smoke test
.venv/bin/python -m bench.cli build tasks/vllm.yaml          # Build Docker images only
```

### Evaluation Harness (Docker-based grading)
```bash
python src/harness/prepare_images.py --dataset_name data/vllm_dataset_with_test.jsonl
python src/harness/opt_at_k.py --prediction_paths predictions.jsonl --k 10
```

### Development Setup
```bash
# Root project
uv venv && source .venv/bin/activate
uv pip install -r requirements.txt

# perf-agents-bench (separate venv)
cd perf-agents-bench
uv venv --python 3.12 .venv
uv pip install -r requirements.txt -p .venv/bin/python

# TRAE agent installation (auto-clones if needed)
./scripts/install_trae_integration.sh
```

### Environment Variables
```bash
# LLM Providers (choose one)
export OPENAI_API_KEY="sk-..."
export ANTHROPIC_API_KEY="sk-ant-..."
# OR AWS Bedrock
export AWS_REGION="us-east-1" && aws sso login

# Agent-specific
export TRAE_PYTHON=/path/to/bench-env/bin/python
export TRAE_CONFIG=/path/to/trae_config.yaml
```

---

## Repository Structure

```
OmniPerf-Bench/
├── commit_to_dataset.py          # Main dataset generation entry
├── run_commit_optimization.py    # Single commit runner
│
├── src/                          # Core framework
│   ├── collect/                  # Dataset generation pipeline
│   ├── harness/                  # Docker-based Opt@K evaluation
│   └── test_scripts/             # LLM test generation
│
├── perf-agents-bench/            # Agent evaluation framework
│   ├── bench/                    # CLI (plan/prepare/report)
│   ├── tasks/                    # Task configs
│   └── state/                    # Plans and journals
│
├── agents/codex/                 # Codex CLI wrapper & config
├── data/                         # Final datasets only
├── inputs/experiments/           # Commit extractions & test generators
├── configs/                      # Pipeline configurations
├── scripts/                      # Analysis & runner scripts
├── docs/                         # Documentation & results
└── third-party/                  # External deps (trae-agent, effibench)
```

---

## Key Workflows

### 1. Dataset Generation Flow
```
Commit JSONs → LLM Test Generation → Execute on base/head/main
→ Measure Speedups → Write JSONL Dataset
```

Entry: `commit_to_dataset.py` with config like:
```yaml
repo_path: "/path/to/vllm"
extractions_dir: "inputs/experiments/commit_extractions_with_apis"
llm_provider: openai  # or anthropic, bedrock
llm_model: gpt-4o-mini
dataset_name: vllm_dataset_with_test
```

### 2. Agent Evaluation Flow
```
Task Config → Plan (resolve commits) → Create Worktrees
→ Execute Agent → Extract Patch → Write Journal
```

**Plan phase** resolves `human_commit` → `pre_commit` (parent):
```bash
.venv/bin/python -m bench.cli plan tasks/vllm.yaml --out state/plan.json
```

**Prepare phase** runs agents in isolated worktrees:
```bash
.venv/bin/python -m bench.cli prepare tasks/vllm.yaml \
  --from-plan state/plan.json \
  --bench-cfg bench.yaml \
  --max-workers 1 \
  --resume
```

### 3. Resume Pattern (Skip Completed Commits)

The `--resume` flag only works within one run. To skip commits from ALL previous runs:

```bash
# Find all successful commits across all runs
grep -r '"result": "success"' state/runs/*/*/journal.json | \
  python -c "
import sys, json
completed = set()
for line in sys.stdin:
    path = line.split(':')[0]
    data = json.load(open(path))
    if data.get('result') == 'success':
        completed.add(data['item_id'])
print('\n'.join(completed))
" > completed_commits.txt

# Filter plan
python -c "
import json
plan = json.load(open('state/plan.json'))
completed = set(open('completed_commits.txt').read().split())
filtered = [item for item in plan if item['id'] not in completed]
json.dump(filtered, open('state/plan_remaining.json', 'w'), indent=2)
print(f'Filtered: {len(plan)} -> {len(filtered)} items')
"

# Run with filtered plan
.venv/bin/python -m bench.cli prepare tasks/vllm.yaml \
  --from-plan state/plan_remaining.json --resume
```

---

## Configuration Layers

1. **Environment Variables**: `OPENAI_API_KEY`, `TRAE_PYTHON`, `TRAE_CONFIG`
2. **bench.yaml** (agent config):
   ```yaml
   agents:
     default: "trae"
     trae:
       cli: "${TRAE_PYTHON}"
       config_file: "${TRAE_CONFIG}"
       time_budget_minutes: 120
   ```
3. **tasks/*.yaml** (task config):
   ```yaml
   id: "vllm_core"
   repo:
     url: "https://github.com/vllm-project/vllm.git"
     human_commit: "abc123..."
   ```
4. **configs/experiments.yaml** (dataset generation)

Environment variables use `${VAR:-default}` syntax.

---

## Agent Integration

Three supported agents with different patterns:

| Agent | Integration | Key Config |
|-------|-------------|------------|
| OpenHands | CLI via `uvx` | `openhands.cli`, `container_image` |
| TRAE | Python package | `trae.cli` (python path), `config_file` |
| Codex CLI | External binary | `codex_cli.cli`, `profile` |

All agents must produce `journal.json` with:
```json
{
  "item_id": "vllm_core-0000",
  "result": "success",  // or "failed", "abandoned"
  "duration_s": 1234.5,
  "commits": ["abc123"],
  "model_patch": "diff --git a/..."
}
```

---

## Git Worktree Isolation

Agents run in isolated worktrees sharing git history:
```
.work/
├── repos/vllm/                 # Base repo (never modified)
└── worktrees/vllm/
    ├── vllm_core-0000/         # Agent 1's workspace
    └── vllm_core-0001/         # Agent 2's workspace
```

**Cleanup stale worktrees:**
```bash
git worktree list
git worktree prune
rm -rf .work/worktrees/vllm/<item_id> && git worktree prune
```

---

## Data Formats

### Commit Extraction JSON (input)
```json
{
  "commit_hash": "abc123...",
  "parent_hash": "def456...",
  "message": "Optimize attention kernels",
  "diff_text": "diff --git a/...",
  "affected_apis": ["vllm.attention.PagedAttention"]
}
```

### Dataset Record JSONL (output)
```json
{
  "instance_id": "OmniPerf-Bench__vllm-abc123",
  "efficiency_test": "def test_performance(): ...",
  "duration_changes": [{"base": [1.24], "head": [0.95], "main": [0.92]}],
  "human_performance": 1.31
}
```

### Journal (agent output)
Located at `state/runs/<run_id>/<item_id>/journal.json`

---

## Troubleshooting

### Import Errors
```bash
# Always run from repo root
cd /path/to/OmniPerf-Bench
PYTHONPATH=src python <script>
```

### TRAE Agent Issues
```bash
# Auto-install (clones if needed)
./scripts/install_trae_integration.sh

# Or manual clone
git clone https://github.com/agokrani/trae-agent.git third-party/trae-agent
```

### Agent Debugging
```bash
# Check journal
cat state/runs/<run_id>/<item_id>/journal.json | jq .

# Check agent logs
cat state/runs/<run_id>/<item_id>/trae_stdout.txt
cat state/runs/<run_id>/<item_id>/trae_stderr.txt

# Inspect worktree (if still exists)
cd .work/worktrees/vllm/<item_id> && git log
```

### AWS Bedrock
```bash
aws sso login
aws bedrock list-foundation-models --region us-east-1 --by-provider anthropic
# Empty? Request model access in AWS Console → Bedrock → Model Access
```

### Test Timing Not Captured
Tests must print: `Execution time: {duration:.4f}s`
Check regex in `src/collect/execute/evaluate.py`

---

## Journal Queries

```bash
# Find successful runs
grep -r '"result": "success"' state/runs/*/*/journal.json

# Find abandoned tasks
grep -r '"result": "abandoned"' state/runs/*/*/journal.json

# Calculate total cost
find state/runs -name journal.json -exec jq -r '.token_usage.total_cost_usd // 0' {} + | \
  awk '{sum+=$1} END {print "Total: $" sum}'

# Find longest tasks
find state/runs -name journal.json -exec jq -r '"\(.duration_s // 0)\t\(.item_id)"' {} + | \
  sort -rn | head -10
```
