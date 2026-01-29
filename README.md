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
# Clone with submodules
git clone --recursive git@github.com:thekernelcompany/OmniPerf-Bench.git
cd OmniPerf-Bench

# Install dependencies
curl -LsSf https://astral.sh/uv/install.sh | sh
uv venv && source .venv/bin/activate
uv sync

# Run the dataset generation pipeline
export OPENAI_API_KEY="your_key"
python commit_to_dataset.py configs/experiments.yaml
```

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
├── configs/                     # Configuration files
│   └── experiments.yaml         # Example experiment configuration
│
├── src/                         # Source code
│   ├── collect/                 # Dataset generation pipeline
│   ├── harness/                 # Docker-based evaluation system
│   │   └── opt_at_k.py          # Main evaluation runner
│   ├── eval/                    # Benchmark runners
│   │   └── modal_benchmark.py   # Modal cloud GPU benchmarks
│   ├── data/                    # Data models and schemas
│   └── test_scripts/            # Test generation utilities
│
├── scripts/                     # Utility scripts
│   ├── runners/                 # Benchmark runner scripts
│   │   ├── local_docker_benchmark.py   # Local Docker benchmarks
│   │   └── hero_3way_benchmark.py      # 3-way comparison runner
│   ├── docker/                  # Docker image management
│   └── upload/                  # HuggingFace upload scripts
│
├── perf-agents-bench/           # Agent benchmarking harness
│   ├── bench/                   # CLI: plan → prepare → report
│   └── tasks/                   # Task configs (vllm.yaml, sglang.yaml)
│
├── data/                        # Generated datasets
│   └── vllm_dataset_with_test.jsonl
│
├── docs/                        # Documentation
│   ├── REPRODUCTION.md          # Step-by-step reproduction guide
│   ├── ENVIRONMENTS.md          # Virtual environment guide
│   └── BENCHMARK_RUNNERS.md     # Canonical runner documentation
│
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

See [docs/ENVIRONMENTS.md](docs/ENVIRONMENTS.md) for details.

## Environment Variables

```bash
# Required for dataset generation
export OPENAI_API_KEY="your_openai_key"
# OR
export ANTHROPIC_API_KEY="your_anthropic_key"

# Optional
export HF_TOKEN="your_huggingface_token"  # For dataset uploads
export GHAPI_TOKEN="your_github_token"    # For commit extraction
```

## Documentation

- **[docs/REPRODUCTION.md](docs/REPRODUCTION.md)** - Step-by-step reproduction guide
- **[docs/ENVIRONMENTS.md](docs/ENVIRONMENTS.md)** - Virtual environment setup
- **[docs/BENCHMARK_RUNNERS.md](docs/BENCHMARK_RUNNERS.md)** - Benchmark runner documentation
- **[docs/dataset_schema.md](docs/dataset_schema.md)** - Dataset schema specification
- **[src/harness/README.md](src/harness/README.md)** - Evaluation harness guide

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
