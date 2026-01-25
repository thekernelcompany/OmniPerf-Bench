# SGLang Per-Commit Isolated Benchmark Plan

## Core Principle
**Each commit is its own world.** No shared dependencies. Fresh venv per commit.

## Method (from sglang-install.md)
```bash
git checkout <commit>
pip install -e "python"  # Installs THAT commit's dependencies
```

## 13 Target Commits

| # | Commit | PR | Era | Expected Script |
|---|--------|-----|-----|-----------------|
| 1 | 187b85b7 | #7393 | Jan 2025 | bench_one_batch |
| 2 | 6b231325 | #6649 | Jan 2025 | bench_one_batch |
| 3 | 6cb00c63 | #6761 | Jan 2025 | bench_one_batch |
| 4 | 148254d4 | #2705 | Oct 2024 | bench_one_batch |
| 5 | 2bd18e2d | #2901 | Oct 2024 | bench_one_batch |
| 6 | 880221bd | #7968 | Jan 2025 | bench_one_batch |
| 7 | b1e5a33a | #6960 | Jan 2025 | bench_one_batch |
| 8 | c087ddd6 | #6627 | Jan 2025 | bench_one_batch |
| 9 | da47621c | #7058 | Jan 2025 | bench_one_batch |
| 10 | dd1012fc | #6764 | Jan 2025 | bench_one_batch |
| 11 | ddcf9fe3 | #3731 | Nov 2024 | bench_one_batch |
| 12 | df7f61ee | #6812 | Jan 2025 | bench_one_batch |
| 13 | e3ec6bf4 | #6814 | Jan 2025 | bench_one_batch |

**Already done:** 2a754e57 (June 2024, bench_latency)

## Per-Commit Process

For each commit (human_commit, base_commit pair):

### Step 1: Setup Isolated Environment
```bash
COMMIT=<8char>
VENV_DIR=/tmp/sglang-venvs/$COMMIT
rm -rf $VENV_DIR
uv venv $VENV_DIR
source $VENV_DIR/bin/activate
```

### Step 2: Checkout & Install
```bash
cd /path/to/sglang-repo
git reset --hard HEAD
git clean -fdx
git checkout <full_commit_hash>
pip install -e "python"
```

### Step 3: Discover Benchmark Script
```python
# Check what exists at this commit
if exists("python/sglang/bench_one_batch.py"):
    script = "sglang.bench_one_batch"
elif exists("python/sglang/bench_latency.py"):
    script = "sglang.bench_latency"
```

### Step 4: Run Benchmark
```bash
python -m $script \
    --model-path TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
    --batch-size 4 \
    --input-len 512 \
    --output-len 64 \
    --load-format dummy \
    --trust-remote-code \
    --mem-fraction-static 0.8
```

### Step 5: For Agent Variants
```bash
# Checkout base_commit
git checkout <base_commit>
pip install -e "python"

# Apply agent patch
git apply /path/to/agent/model_patch.diff

# Run benchmark
python -m $script ...
```

## 4 Agent Variants Per Commit
- claude_code
- codex
- trae_gpt5
- trae_sonnet45

## Total Benchmarks
- 13 commits x (1 baseline + 1 human + 4 agents) = 78 benchmark runs
- Plus 2a754e57 already done = 6 runs

## Key Insights

1. **Dependencies change per commit** - Oct 2024 needs vllm 0.6.x, Jan 2025 needs different
2. **Benchmark scripts change** - bench_latency.py (old) vs bench_one_batch.py (new)
3. **Flags may differ** - older commits may need --disable-flashinfer
4. **Fresh venv is critical** - prevents dependency conflicts

## Execution Order
1. Start with one commit to validate approach
2. Then batch run remaining commits
3. Save results after each commit (fault tolerance)
