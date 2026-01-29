# OmniPerf-Bench

A benchmark framework for evaluating AI agents on real-world software performance optimization tasks.

## Overview

OmniPerf-Bench provides:
- **Performance optimization dataset**: Real-world performance commits extracted from vLLM and SGLang repositories
- **Automated test generation**: LLM-powered creation of performance tests for optimization tasks
- **Evaluation harness**: Docker-based system measuring optimization effectiveness (Opt@K metrics)
- **Agent benchmarking**: Harness for running AI agents (TRAE, Codex, OpenHands, Claude Code) on optimization tasks

Each task provides a codebase with performance bottlenecks, precise performance tests, and requires agents to generate patches that improve runtime efficiency. Success is measured against expert developer optimizations using wall-clock timing comparisons.

## Quick Start

```bash
# 1. Clone with submodules (required)
git clone --recursive git@github.com:thekernelcompany/OmniPerf-Bench.git
cd OmniPerf-Bench

# 2. Install uv package manager
curl -LsSf https://astral.sh/uv/install.sh | sh
source ~/.local/bin/env  # or restart your shell

# 3. Create virtual environment and install ALL dependencies
uv venv && source .venv/bin/activate
uv sync

# 4. Create .env file with your API key
echo 'OPENAI_API_KEY="your_openai_api_key_here"' > .env

# 5. Verify installation
PYTHONPATH=src python -c "from collect.analysis.commits import PerfCommitAnalyzer; print('Setup complete')"

# 6. Run the dataset generation pipeline
python commit_to_dataset.py configs/experiments.yaml
```

**Important**: Step 3 (`uv sync`) installs the `r2e` package from GitHub, which is required for the `PerfCommitAnalyzer` module. If you see `ModuleNotFoundError: No module named 'r2e'`, re-run `uv sync`.

## Available Datasets

```python
from datasets import load_dataset

# Load vLLM performance optimization dataset (282 problems)
vllm_data = load_dataset('Inferencebench/vllm_dataset_with_test', split='test')

# Load GSO-compatible benchmark
gso = load_dataset('gso-bench/gso', split='test')
```

## Repository Structure

```
OmniPerf-Bench/
├── commit_to_dataset.py         # Main entry point: dataset generation
├── pyproject.toml               # Python packaging and dependencies
├── .env                         # API keys (create this file)
├── configs/                     # Configuration files
│   └── experiments.yaml         # Example experiment configuration
│
├── src/                         # Source code
│   ├── collect/                 # Dataset generation pipeline
│   │   ├── analysis/            # Commit analysis (PerfCommitAnalyzer)
│   │   └── generate/            # Test generation
│   ├── harness/                 # Docker-based evaluation system
│   │   └── opt_at_k.py          # Main evaluation runner
│   ├── data/                    # Data models and schemas
│   └── utils/                   # Utility functions
│
├── scripts/                     # Utility scripts
│   ├── runners/                 # Benchmark runner scripts
│   ├── docker/                  # Docker image management
│   └── upload/                  # HuggingFace upload scripts
│
├── perf-agents-bench/           # Agent benchmarking harness
│   ├── bench/                   # CLI: plan → prepare → report
│   └── tasks/                   # Task configs (vllm.yaml, sglang.yaml)
│
├── data/                        # Generated datasets
├── archive/                     # Archived experimental data
│   └── misc/experiments/        # Commit extractions (used by pipeline)
│
├── docs/                        # Documentation
├── vllm/, sglang/               # Git submodules
└── third-party/                 # External dependencies
```

## Core Pipelines

### 1. Dataset Generation

Generate benchmark datasets from performance optimization commits:

```bash
source .venv/bin/activate
python commit_to_dataset.py configs/experiments.yaml
# Output: data/vllm_dataset_with_test.jsonl
```

The pipeline:
1. Reads commit extraction JSONs from `archive/misc/experiments/commit_extractions_with_apis/`
2. Uses `PerfCommitAnalyzer` to process commit metadata
3. Generates performance tests via LLM (OpenAI/Anthropic)
4. Outputs canonical dataset records

### 2. Evaluation Harness

Run model predictions and compute Opt@K metrics:

```bash
source .venv/bin/activate
uv run src/harness/opt_at_k.py \
    --model <model_name> \
    --prediction_paths <predictions.jsonl> \
    --timeout 3600 --run_id <run_id> --k 10
```

### 3. Agent Benchmarking

Run AI agents on optimization tasks:

```bash
cd perf-agents-bench
source .venv/bin/activate

# Plan benchmarks
python -m bench.cli plan tasks/vllm.yaml --out state/plan.json

# Execute agent benchmarks
python -m bench.cli prepare tasks/vllm.yaml --from-plan state/plan.json

# Generate reports
python -m bench.cli report state/runs/<run_id>
```

## Virtual Environments

This repository uses multiple virtual environments:

| Environment | Purpose |
|-------------|---------|
| `.venv/` | Main OmniPerf-Bench, dataset generation, evaluation harness |
| `bench-env/` | Agent benchmarking, Modal cloud execution |
| `perf-agents-bench/.venv/` | OpenHands CLI |

## Environment Variables

Create a `.env` file in the project root:

```bash
# Required for dataset generation (at least one)
OPENAI_API_KEY="sk-..."
# OR
ANTHROPIC_API_KEY="sk-ant-..."

# Optional
OPENROUTER_API_KEY="sk-or-..."    # Alternative LLM provider
HF_TOKEN="hf_..."                  # For dataset uploads
GHAPI_TOKEN="ghp_..."              # For commit extraction
```

## Troubleshooting

### `ModuleNotFoundError: No module named 'r2e'`

The `r2e` package is installed from GitHub. Re-run:
```bash
uv sync
```

### `ModuleNotFoundError: No module named 'collect.analysis.commits'`

Ensure you're running with PYTHONPATH:
```bash
PYTHONPATH=src python commit_to_dataset.py configs/experiments.yaml
```

### `PerfCommitAnalyzer is required but not available`

The module failed to import. Check:
```bash
PYTHONPATH=src python -c "from collect.analysis.commits import PerfCommitAnalyzer"
```

### API key errors

Ensure your `.env` file exists and contains valid keys:
```bash
cat .env  # Should show OPENAI_API_KEY="sk-..."
```

## Documentation

- **[docs/REPRODUCTION.md](docs/REPRODUCTION.md)** - Step-by-step reproduction guide
- **[docs/ENVIRONMENTS.md](docs/ENVIRONMENTS.md)** - Virtual environment setup
- **[docs/dataset_schema.md](docs/dataset_schema.md)** - Dataset schema specification

## Requirements

- Python 3.12+
- Docker (for evaluation harness)
- CUDA-capable GPU (for performance testing)
- Git with LFS support

## Submodules

```bash
# If cloned without --recursive:
git submodule update --init --recursive
```

Included submodules:
- `vllm/` - vLLM fork for benchmarking
- `sglang/` - SGLang for comparison benchmarks
- `third-party/trae-agent/` - TRAE agent integration

## Citation

```bibtex
@misc{omniperf-bench,
  title={OmniPerf-Bench: A Benchmark for Evaluating AI Agents on Software Performance Optimization},
  author={The Kernel Company},
  year={2026},
  url={https://github.com/thekernelcompany/OmniPerf-Bench}
}
```
