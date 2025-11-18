# OmniPerf-Bench Quickstart Guide

Get started with OmniPerf-Bench in minutes! This guide shows the 3 most common workflows.

## Prerequisites

```bash
# 1. Install dependencies
uv pip install -r requirements.txt
# OR
pip install -r requirements.txt

# 2. Set API key (choose one)
export OPENAI_API_KEY="sk-..."
export ANTHROPIC_API_KEY="sk-ant-..."
# OR use AWS Bedrock: aws sso login

# 3. Clone vLLM repository
git clone https://github.com/vllm-project/vllm.git vllm
```

---

## Workflow 1: Test Single Commit (5 minutes)

**Run agent optimization on one performance commit**

```bash
python run_commit_optimization.py \
  --commit-json misc/experiments/commit_extractions_with_apis/0ec82edda59aaf5cf3b07aadf4ecce1aa1131add.json \
  --test-script misc/experiments/generated_test_generators_v4/0ec82edd_test_case_generator.py \
  --repo-path vllm \
  --cleanup
```

**Expected Output:**
```
✓ Agent optimization completed
✓ Performance comparison: Agent 1.03x speedup vs baseline
✓ Human optimization: 1.06x speedup (agent achieved 97% of human)
```

**What happened:**
- Agent analyzed the commit context
- Generated optimization code
- Performance test measured speedup
- Compared against human expert optimization

---

## Workflow 2: Generate Dataset (30 minutes)

**Convert commit extractions into benchmark dataset**

### Option A: Single Commit (Quick Test)

```bash
# Edit config to use single commit
python commit_to_dataset.py configs/single_commit_config.yaml

# Output: data/single_commit_test.jsonl
```

### Option B: Full Dataset (All Commits)

```bash
# Generate full vLLM dataset (64 commits)
python commit_to_dataset.py configs/experiments.yaml

# Output: data/vllm_dataset_with_test.jsonl (282 problems)
# Log: commit_to_dataset.log
```

**What's generated:**
- Canonical JSONL dataset
- Performance tests for each commit
- Timing measurements (base/head/main)
- Human performance benchmarks

---

## Workflow 3: Run Agent Framework (20 minutes)

**Evaluate OpenHands or TRAE agent on multiple commits**

### Setup (one-time)

```bash
cd perf-agents-bench

# Create virtual environment
uv venv --python 3.12 .venv
uv pip install -r requirements.txt -p .venv/bin/python

# Configure environment
cp .env.example .env
# Edit .env with your API keys
```

### Run Evaluation

```bash
# 1. Create commit list
cat > .work/test_commits.txt << 'EOF'
f092153fbe349a9a1742940e3703bfcff6aa0a6d parent=1
8d75fe48ca5f46b7af0f5201d8500b9604eed769 parent=1
EOF

# 2. Generate plan
.venv/bin/python -m bench.cli plan \
  tasks/vllm.yaml \
  --commits .work/test_commits.txt \
  --out state/plan.json

# 3. Execute agents
.venv/bin/python -m bench.cli prepare \
  tasks/vllm.yaml \
  --from-plan state/plan.json \
  --bench-cfg bench.yaml \
  --max-workers 1

# 4. Generate report
LATEST=$(ls -t state/runs | head -n1)
.venv/bin/python -m bench.cli report state/runs/$LATEST
```

**Expected Output:**
```
✓ 2/2 commits completed
✓ Success rate: 50%
✓ Average time per task: 18.5 minutes
✓ Performance improvements: 1.2x average speedup
```

---

## Common Commands

### Generate Performance Tests

```bash
# Generate tests using LLM
PYTHONPATH=src python src/test_scripts/generate_test_generators.py \
  --extractions-dir misc/experiments/commit_extractions_with_apis \
  --output-dir misc/experiments/generated_test_generators_v4
```

### Run Batch Optimizations

```bash
python batch_commit_optimization.py \
  --commit-dir misc/experiments/commit_extractions_with_apis/ \
  --test-dir misc/experiments/generated_test_generators_v4/ \
  --repo-path vllm \
  --output-dir results/ \
  --max-workers 2
```

### Evaluate Agent Predictions

```bash
# Build Docker images
uv run src/harness/prepare_images.py \
  --dataset_name data/vllm_dataset_with_test.jsonl

# Run evaluation
uv run src/harness/opt_at_k.py \
  --model my_agent \
  --prediction_paths predictions.jsonl \
  --dataset_name data/vllm_dataset_with_test.jsonl \
  --k 5
```

---

## Troubleshooting

### "ModuleNotFoundError: collect"
**Solution:** Always run from repository root
```bash
cd /path/to/OmniPerf-Bench  # Not src/
python commit_to_dataset.py configs/experiments.yaml
```

### "CUDA not available"
**Solution:** Use Docker or CPU-only mode
```bash
# In config YAML:
use_docker: true
# OR
export PROB_DEVICE=cpu
```

### LLM Generation Failures
**Solution:** Use more reliable model
```bash
# In config YAML:
llm_model: gpt-4o  # Instead of gpt-4o-mini
```

### AWS Bedrock Access
**Solution:** Request model access
```bash
aws sso login
aws bedrock list-foundation-models --region us-east-1 --by-provider anthropic
# If empty, go to AWS Console → Bedrock → Model Access
```

---

## Available Data

### Pre-extracted Commits

```bash
# vLLM commits (64 commits)
ls misc/experiments/commit_extractions_with_apis/*.json

# SGLang commits (80 commits)
ls misc/experiments/sglang_commit_extractions_with_apis/*.json
```

### Pre-generated Tests

```bash
# Ready-to-use test scripts
ls misc/experiments/generated_test_generators_v4/*.py
# 0ec82edd_test_case_generator.py
# 2deb029d_test_case_generator.py
# 8aa1485f_test_case_generator.py
```

### Datasets

```bash
# vLLM performance dataset (282 problems)
data/vllm_dataset_with_test.jsonl

# Load from HuggingFace
python -c "
from datasets import load_dataset
ds = load_dataset('Inferencebench/vllm_dataset_with_test', split='test')
print(f'Loaded {len(ds)} instances')
"
```

---

## Next Steps

- **Full documentation:** See `README.md` and `docs/`
- **Architecture details:** Read `docs/omni_commit_architecture.md`
- **Agent framework:** See `perf-agents-bench/README.md`
- **Dataset schema:** Check `docs/dataset_schema.md`
- **Example configs:** Browse `configs/` directory

## Configuration Templates

All example configurations are in `configs/`:
- `experiments.yaml` - Full dataset generation
- `single_commit_config.yaml` - Quick test with 1 commit

Agent framework configs in `perf-agents-bench/`:
- `bench.yaml` - Agent configuration
- `tasks/vllm.yaml` - vLLM task definition
- `tasks/sglang.yaml` - SGLang task definition

---

## Support

- **Issues:** Report at repository issue tracker
- **Documentation:** See `docs/` directory
- **Examples:** Check `configs/` for working configurations
- **Logs:** Review `commit_to_dataset.log`, `run_commit_optimization.log`

**Environment check:**
```bash
cd perf-agents-bench
.venv/bin/python -m bench.cli doctor --bench-cfg bench.yaml
```

This validates Python, Docker, Git, and agent configurations.

---

**Ready to go!** Start with Workflow 1 to test a single commit in 5 minutes.
