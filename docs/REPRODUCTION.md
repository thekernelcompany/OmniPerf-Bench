# Reproducing OmniPerf-Bench Results

This document provides step-by-step instructions to reproduce the benchmark results from the OmniPerf-Bench paper.

## Prerequisites

- Python 3.12+
- Docker (for evaluation harness)
- GPU with CUDA support (for benchmarking)
- Git with LFS enabled
- API keys for LLM providers (see Environment Variables below)

## Environment Variables

```bash
# Required for dataset generation
OPENAI_API_KEY="sk-..."      # OpenAI API key
# OR
ANTHROPIC_API_KEY="sk-..."   # Anthropic API key

# Optional for agent benchmarking
GHAPI_TOKEN="ghp_..."        # GitHub API token
HF_TOKEN="hf_..."            # HuggingFace token for dataset uploads

# Optional for Modal cloud execution
MODAL_TOKEN_ID="..."
MODAL_TOKEN_SECRET="..."

# For AWS Bedrock (optional)
AWS_REGION="us-west-2"
AWS_ACCESS_KEY_ID="..."
AWS_SECRET_ACCESS_KEY="..."
```

## Pipeline 1: Dataset Generation

Generate performance optimization benchmark data from vLLM commits.

### Setup

```bash
# Clone repository with submodules
git clone --recursive https://github.com/thekernelcompany/OmniPerf-Bench.git
cd OmniPerf-Bench

# Create and activate main virtual environment
uv venv && source .venv/bin/activate
uv sync

# Verify submodules are present
ls vllm/ sglang/
```

### Run Dataset Generation

```bash
# Set API key
export OPENAI_API_KEY="your_key"

# Run the main pipeline
python commit_to_dataset.py configs/experiments.yaml

# Expected output:
# - data/vllm_dataset_with_test.jsonl
```

### Configuration Options

Edit `configs/experiments.yaml` to customize:
- `repo_path`: Path to vLLM repository
- `llm_provider`: "openai", "anthropic", or "bedrock"
- `llm_model`: Model name (e.g., "gpt-4o", "claude-3-opus")
- `push_to_hf`: Set to `true` to upload results to HuggingFace

## Pipeline 2: Agent Benchmarking

Run AI agents (OpenHands, TRAE, Codex, Claude Code) on optimization tasks.

### Setup

```bash
cd ISO-Bench

# Create dedicated virtual environment
uv venv --python 3.12 .venv
source .venv/bin/activate
pip install -r requirements.txt

# Copy and edit environment file
cp .env.example .env
# Edit .env with your API keys
```

### Run Agent Benchmarks

```bash
# Step 1: Plan - resolve commits to benchmark
.venv/bin/python -m bench.cli plan tasks/vllm.yaml \
    --commits .work/vllm_commits.txt \
    --out ./state/plan.json

# Step 2: Prepare - run agents on commits
.venv/bin/python -m bench.cli prepare tasks/vllm.yaml \
    --from-plan ./state/plan.json \
    --bench-cfg bench.yaml \
    --max-workers 1 \
    --resume

# Step 3: Report - aggregate results
.venv/bin/python -m bench.cli report state/runs/vllm/$(ls -t state/runs/vllm | head -n1)
```

### Expected Output

Results are saved to `ISO-Bench/state/runs/<repo>/<agent>/<model>/<timestamp>/`:
- `run_summary.json`: Summary metrics
- `journal.json`: Detailed execution log
- `<commit>/patch.diff`: Generated patches

## Pipeline 3: Evaluation Harness

Run performance benchmarks using Docker containers.

### Setup

```bash
# Use main virtual environment
source .venv/bin/activate

# Verify Docker is running
docker info
```

### Run Evaluation

```bash
# Run Opt@K evaluation
uv run src/harness/opt_at_k.py \
    --model gpt-4o \
    --prediction_paths data/predictions.jsonl \
    --timeout 3600 \
    --run_id test_run \
    --k 10

# Expected output: evaluation_results/<run_id>/
```

### Local Docker Benchmarks

For commits that need local evaluation:

```bash
python scripts/runners/local_docker_benchmark.py \
    --commit <commit_hash> \
    --agent-type claude_code \
    --mode human
```

## Verifying Installation

Run these commands to verify your setup:

```bash
# Check dataset pipeline imports
python -c "from commit_to_dataset import *; print('Dataset pipeline OK')"

# Check agent CLI
cd ISO-Bench && .venv/bin/python -m bench.cli --help

# Check evaluation harness
python -c "import sys; sys.path.insert(0, 'src'); from harness.opt_at_k import *; print('Harness OK')"
```

## Troubleshooting

### Common Issues

1. **Missing submodules**: Run `git submodule update --init --recursive`
2. **Docker permission denied**: Add user to docker group or use sudo
3. **CUDA out of memory**: Reduce batch size or use smaller models
4. **API rate limits**: Implement exponential backoff or use different API keys

### Getting Help

- Open an issue: https://github.com/thekernelcompany/OmniPerf-Bench/issues
- Check existing documentation in `docs/`
