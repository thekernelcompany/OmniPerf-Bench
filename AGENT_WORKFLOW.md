# Running TRAE Agent with vLLM and SGLang

**Complete guide for isolated agent execution with commit tracking and testing**

## Overview

The agent workflow provides:
- ✅ **Isolated virtual environments** - Each agent run in separate workspace
- ✅ **Git worktrees** - Agents commit code changes without affecting main repo
- ✅ **No cross-contamination** - Agents can't see other runs
- ✅ **Full testing** - Automated performance testing after agent changes
- ✅ **Both frameworks** - Run on vLLM (64 commits) and SGLang (80 commits)

---

## Setup (One-Time)

### 1. Install TRAE Agent

```bash
# Run interactive installer
./install_trae_integration.sh

# This will:
# - Check Python 3.12+
# - Initialize trae-agent submodule
# - Create virtual environment
# - Install dependencies
# - Configure TRAE settings
```

### 2. Setup perf-agents-bench

```bash
cd perf-agents-bench

# Create virtual environment (Python 3.12+ required)
uv venv --python 3.12 .venv
uv pip install -r requirements.txt -p .venv/bin/python

# Configure environment
cp .env.example .env
# Edit .env:
#   LLM_MODEL=gpt-4o
#   LLM_API_KEY=sk-...
```

### 3. Verify Setup

```bash
cd perf-agents-bench
.venv/bin/python -m bench.cli doctor --bench-cfg bench.yaml

# Should show:
# ✓ Python version: 3.12.x
# ✓ Docker available
# ✓ Git repository initialized
# ✓ Agent configurations valid
```

---

## Workflow: Run TRAE on vLLM

### Step 1: Select Commits

```bash
cd perf-agents-bench

# Create commit list for vLLM
cat > .work/vllm_commits.txt << 'EOF'
f092153fbe349a9a1742940e3703bfcff6aa0a6d parent=1
8d75fe48ca5f46b7af0f5201d8500b9604eed769 parent=1
0ec82edda59aaf5cf3b07aadf4ecce1aa1131add parent=1
EOF

# This specifies:
# - commit_hash: The optimized commit to reproduce
# - parent=N: Go back N commits to get the "before" state
```

### Step 2: Generate Plan

```bash
# Create execution plan
.venv/bin/python -m bench.cli plan \
  tasks/vllm.yaml \
  --commits .work/vllm_commits.txt \
  --out state/plan_vllm.json

# Verify plan
cat state/plan_vllm.json | python -m json.tool

# Shows: (human_commit, pre_commit) pairs
```

### Step 3: Run TRAE Agent

```bash
# Execute TRAE on planned commits
.venv/bin/python -m bench.cli prepare \
  tasks/vllm.yaml \
  --from-plan state/plan_vllm.json \
  --bench-cfg bench.yaml \
  --max-workers 1 \
  --resume

# What happens:
# 1. Creates isolated git worktree for each commit
# 2. Checks out pre-commit state (before optimization)
# 3. Launches TRAE agent in virtual environment
# 4. Agent reads task description, explores code, makes changes
# 5. Agent commits changes to isolated worktree
# 6. System captures all agent output and changes
# 7. Runs performance tests on agent's changes
# 8. Compares against human expert optimization
```

### Step 4: Monitor Progress

```bash
# In another terminal, monitor real-time output
LATEST=$(ls -t state/runs | head -n1)
tail -f state/runs/$LATEST/*/trae_stdout.txt

# Check journal for structured data
cat state/runs/$LATEST/*/journal.json | python -m json.tool
```

### Step 5: Review Results

```bash
# Generate report
LATEST=$(ls -t state/runs | head -n1)
.venv/bin/python -m bench.cli report state/runs/$LATEST

# Output shows:
# - Success rate per commit
# - Performance improvements
# - Agent vs human comparison
# - Time taken per task
# - Error categories for failures
```

### Step 6: Inspect Agent Changes

```bash
# View what the agent changed
LATEST=$(ls -t state/runs | head -n1)
COMMIT_DIR=$(ls -d state/runs/$LATEST/*/ | head -n1)

# See agent's patch
cat ${COMMIT_DIR}/model_patch.diff

# See agent's thought process
cat ${COMMIT_DIR}/trae_stdout.txt

# See performance test results
cat ${COMMIT_DIR}/journal.json | jq '.result'
```

---

## Workflow: Run TRAE on SGLang

### Step 1: Select SGLang Commits

```bash
cd perf-agents-bench

# Create commit list for SGLang
cat > .work/sglang_commits.txt << 'EOF'
a1b2c3d4e5f6g7h8i9j0 parent=1
f1e2d3c4b5a6 parent=1
EOF

# Use actual SGLang commit hashes from:
# ../misc/experiments/sglang_commit_extractions_with_apis/
```

### Step 2: Generate Plan

```bash
.venv/bin/python -m bench.cli plan \
  tasks/sglang.yaml \
  --commits .work/sglang_commits.txt \
  --out state/plan_sglang.json
```

### Step 3: Run TRAE Agent

```bash
.venv/bin/python -m bench.cli prepare \
  tasks/sglang.yaml \
  --from-plan state/plan_sglang.json \
  --bench-cfg bench.yaml \
  --max-workers 1
```

**Same isolated execution as vLLM:**
- Separate git worktrees
- Independent virtual environments
- No interference with vLLM runs

---

## How Isolation Works

### Git Worktrees

Each agent run gets its own git worktree:

```
perf-agents-bench/state/runs/
├── vllm_core-a1b2c3d4/          # Run 1
│   └── vllm_core-0000/          # Task 1 (commit 1)
│       ├── worktree/            # Isolated git worktree
│       │   └── .git            # Separate git state
│       ├── task.txt            # Agent's task description
│       ├── trae_stdout.txt     # Agent's output
│       ├── model_patch.diff    # Agent's code changes
│       └── journal.json        # Execution metadata
└── sglang_core-e5f6g7h8/       # Run 2 (different worktree)
    └── sglang_core-0000/
        └── worktree/            # Completely separate
```

**Benefits:**
- ✅ Agent can commit freely without affecting main repo
- ✅ Multiple agents can run in parallel
- ✅ Failed runs don't corrupt repository
- ✅ Easy to inspect agent's changes
- ✅ Clean rollback after each task

### Virtual Environment Isolation

Each agent run has:
- Separate Python virtual environment
- Independent package installations
- Isolated environment variables
- No shared state between runs

### Process Isolation

Configured in `bench.yaml`:

```yaml
container:
  engine: docker           # Optional: Docker isolation
  platform: linux/amd64
  gpus: none              # Or "all" for GPU access

agents:
  trae:
    cli: /path/to/python  # TRAE's isolated environment
    time_budget_minutes: 120
    args:
      max_steps: 120
```

---

## Running Multiple Agents in Parallel

### Option 1: Sequential (Safe)

```bash
# Run vLLM first
.venv/bin/python -m bench.cli prepare \
  tasks/vllm.yaml \
  --from-plan state/plan_vllm.json \
  --bench-cfg bench.yaml \
  --max-workers 1

# Then run SGLang
.venv/bin/python -m bench.cli prepare \
  tasks/sglang.yaml \
  --from-plan state/plan_sglang.json \
  --bench-cfg bench.yaml \
  --max-workers 1
```

### Option 2: Parallel (Advanced)

```bash
# Terminal 1: Run vLLM
cd perf-agents-bench
.venv/bin/python -m bench.cli prepare \
  tasks/vllm.yaml \
  --from-plan state/plan_vllm.json \
  --bench-cfg bench.yaml &

# Terminal 2: Run SGLang simultaneously
cd perf-agents-bench
.venv/bin/python -m bench.cli prepare \
  tasks/sglang.yaml \
  --from-plan state/plan_sglang.json \
  --bench-cfg bench.yaml &

# Wait for both
wait
```

**Safe because:**
- Separate worktrees
- Different run IDs (vllm_core-* vs sglang_core-*)
- Independent state directories
- No shared file access

---

## Testing Agent Changes

### Automatic Testing

Testing happens automatically during agent execution:

1. **Agent makes changes** → Commits to isolated worktree
2. **System detects commit** → Extracts patch
3. **Performance test runs** → Uses generated test script
4. **Results recorded** → In journal.json
5. **Comparison made** → Agent vs human optimization

### Manual Testing

Inspect and test agent changes manually:

```bash
# Navigate to agent's worktree
LATEST=$(ls -t state/runs | head -n1)
TASK=$(ls -d state/runs/$LATEST/*/ | head -n1)
cd ${TASK}/worktree

# Agent's changes are already committed here
git log -1 --stat

# Run custom tests
pytest tests/

# Check performance manually
python performance_test.py
```

### Test Against Human Baseline

```bash
# Agent's performance is automatically compared:
cat journal.json | jq '.result.performance'

# Shows:
# {
#   "agent_speedup": 1.25,      # Agent's improvement
#   "human_speedup": 1.30,      # Expert's improvement
#   "agent_vs_human": 0.96      # Agent achieved 96% of expert
# }
```

---

## Advanced Configuration

### Customize TRAE Behavior

Edit `bench.yaml`:

```yaml
agents:
  trae:
    time_budget_minutes: 180    # Longer time for complex tasks
    args:
      max_steps: 150           # More exploration steps
      temperature: 0.7         # Creativity vs determinism
      model: "gpt-4o"         # Stronger model
```

### Custom Task Definitions

Edit `tasks/vllm.yaml` or `tasks/sglang.yaml`:

```yaml
optimization_contract:
  goal: "Improve inference throughput by at least 20%"
  constraints:
    - "No public API breakage"
    - "All existing tests must pass"
    - "Memory usage should not increase >10%"

task_template: |
  You are optimizing {repo_name} for better performance.

  Base commit: {pre_commit}
  Target commit: {human_commit}

  Your goal: Reproduce or improve upon the expert's optimization.

  The expert achieved: {human_speedup}x speedup

  Files changed by expert:
  {changed_files}
```

---

## Troubleshooting

### Agent Can't Find Code

**Issue:** Agent explores wrong files
**Solution:** Check task description clarity

```bash
# Review what the agent sees
cat state/runs/*/*/task.txt

# Add more context to task template
```

### Agent Makes No Changes

**Issue:** Agent gives up too early
**Solution:** Increase time budget and steps

```yaml
# In bench.yaml
trae:
  time_budget_minutes: 240  # 4 hours
  args:
    max_steps: 200
```

### Performance Test Fails

**Issue:** Test script has errors
**Solution:** Use pre-generated tests or fix template

```bash
# Check test script
cat misc/experiments/generated_test_generators_v4/HASH_test_case_generator.py

# Run test manually
cd state/runs/*/*/worktree
python ../../test_script.py
```

### Worktree Conflicts

**Issue:** "worktree already exists"
**Solution:** Clean old worktrees

```bash
# Remove old run directories
rm -rf perf-agents-bench/state/runs/OLD_RUN_ID/

# Or clean all completed runs
cd perf-agents-bench
find state/runs -type d -mtime +7 -exec rm -rf {} +
```

---

## Complete Example: vLLM Optimization

```bash
# 1. Setup (one-time)
./install_trae_integration.sh
cd perf-agents-bench
uv venv --python 3.12 .venv
uv pip install -r requirements.txt -p .venv/bin/python

# 2. Select 3 vLLM commits
cat > .work/my_test.txt << 'EOF'
f092153fbe349a9a1742940e3703bfcff6aa0a6d parent=1
8d75fe48ca5f46b7af0f5201d8500b9604eed769 parent=1
0ec82edda59aaf5cf3b07aadf4ecce1aa1131add parent=1
EOF

# 3. Create plan
.venv/bin/python -m bench.cli plan \
  tasks/vllm.yaml \
  --commits .work/my_test.txt \
  --out state/my_plan.json

# 4. Run TRAE (will take 1-2 hours)
.venv/bin/python -m bench.cli prepare \
  tasks/vllm.yaml \
  --from-plan state/my_plan.json \
  --bench-cfg bench.yaml \
  --max-workers 1

# 5. Generate report
LATEST=$(ls -t state/runs | head -n1)
.venv/bin/python -m bench.cli report state/runs/$LATEST

# 6. Review agent's code
cat state/runs/$LATEST/*/model_patch.diff

# Done! Agent ran in isolation, committed changes, tests executed.
```

---

## Summary

✅ **Full agent workflow is preserved**
✅ **TRAE + OpenHands both ready**
✅ **vLLM (64 commits) + SGLang (80 commits)**
✅ **Complete isolation** - git worktrees + virtual envs
✅ **Agents can commit** - changes captured safely
✅ **No cross-contamination** - separate workspaces
✅ **Automated testing** - performance measured automatically
✅ **Easy inspection** - all agent output and diffs saved

**Your production-clean branch has everything you need!**
