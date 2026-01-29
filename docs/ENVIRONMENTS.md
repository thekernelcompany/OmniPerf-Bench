# Virtual Environments

OmniPerf-Bench uses multiple virtual environments for different purposes. This document explains when and how to use each one.

## Overview

| Environment | Location | Purpose |
|-------------|----------|---------|
| `.venv/` | Root | Main OmniPerf-Bench, dataset generation, evaluation harness |
| `bench-env/` | Root | Modal deployment, TRAE integration, cloud execution |
| `perf-agents-bench/.venv/` | Subdirectory | OpenHands agent benchmarking CLI |

## 1. Main Environment (`.venv/`)

**Purpose**: Core OmniPerf-Bench functionality including dataset generation, test script generation, and evaluation harness.

### Setup

```bash
cd OmniPerf-Bench

# Create virtual environment with uv
uv venv
source .venv/bin/activate

# Install dependencies
uv sync

# Or with pip
pip install -r requirements.txt
```

### When to Use

- Running `commit_to_dataset.py`
- Generating test scripts (`src/test_scripts/`)
- Running the evaluation harness (`src/harness/opt_at_k.py`)
- Development and testing

### Key Dependencies

- `torch` - PyTorch for ML operations
- `datasets`, `huggingface-hub` - HuggingFace dataset handling
- `docker` - Docker SDK for container management
- `openai`, `anthropic` - LLM API clients

## 2. Agent Benchmarking Environment (`bench-env/`)

**Purpose**: Running agent benchmarks on Modal cloud, TRAE agent integration, and cloud GPU execution.

### Setup

```bash
cd OmniPerf-Bench

# Create virtual environment
python -m venv bench-env
source bench-env/bin/activate

# Install TRAE requirements
pip install -r trae_requirements.txt

# Configure Modal (if using cloud execution)
modal token new
```

### When to Use

- Deploying Modal benchmarks
- Running TRAE agent experiments
- Cloud-based GPU benchmarking
- HuggingFace dataset uploads

### Key Dependencies

- `modal` - Modal cloud platform SDK
- `openai` - OpenAI API for TRAE
- `anthropic` - Anthropic API for TRAE
- TRAE agent dependencies

## 3. perf-agents-bench Environment (`perf-agents-bench/.venv/`)

**Purpose**: Running the perf-agents-bench CLI for agent planning, preparation, and reporting. Includes OpenHands integration.

### Setup

```bash
cd OmniPerf-Bench/perf-agents-bench

# Create dedicated virtual environment
uv venv --python 3.12 .venv
source .venv/bin/activate

# Install dependencies
uv pip install -r requirements.txt -p .venv/bin/python

# Or with pip
pip install -r requirements.txt
```

### When to Use

- Running `bench.cli plan` - Planning benchmark runs
- Running `bench.cli prepare` - Executing agent benchmarks
- Running `bench.cli report` - Generating reports
- Running `bench.cli analyze` - Soft metrics analysis
- OpenHands agent experiments

### Key Dependencies

- `openhands-ai` - OpenHands agent framework
- `typer` - CLI framework
- `pydantic` - Configuration validation
- `playwright` - Browser automation for OpenHands

## Quick Reference

```bash
# Dataset generation
source .venv/bin/activate
python commit_to_dataset.py configs/experiments.yaml

# Modal/TRAE cloud benchmarks
source bench-env/bin/activate
modal deploy src/eval/modal_benchmark.py

# Agent CLI benchmarking
cd perf-agents-bench
source .venv/bin/activate
python -m bench.cli plan tasks/vllm.yaml --out state/plan.json

# Return to main environment
cd ..
source .venv/bin/activate
```

## Troubleshooting

### Wrong Environment Active

If you see import errors, check which environment is active:

```bash
which python
# Should show the expected .venv path
```

### Dependency Conflicts

Each environment is isolated. If you encounter conflicts:

1. Deactivate current environment: `deactivate`
2. Remove the problematic venv: `rm -rf .venv`
3. Recreate with fresh dependencies

### Python Version Mismatch

OmniPerf-Bench requires Python 3.12+:

```bash
python --version
# Should be 3.12.x or higher

# If not, specify version when creating venv:
uv venv --python 3.12 .venv
```
