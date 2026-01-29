# OmniPerf-Bench

A benchmark framework for evaluating AI agents on real-world software performance optimization tasks.

## Overview

OmniPerf-Bench provides:
- **Performance optimization dataset**: Real-world performance commits extracted from vLLM and SGLang repositories
- **Automated test generation**: LLM-powered creation of performance tests for optimization tasks
- **Evaluation harness**: Docker-based system measuring optimization effectiveness (Opt@K metrics)
- **Agent benchmarking (ISO-Bench)**: Harness for running AI agents (TRAE, Codex, OpenHands, Claude Code) on optimization tasks

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
│
├── src/                         # Source code
│   ├── collect/                 # Dataset generation pipeline
│   │   ├── analysis/            # Commit analysis (PerfCommitAnalyzer)
│   │   └── generate/            # Test generation
│   ├── eval/                    # Benchmark runners (Modal, native)
│   ├── harness/                 # Docker-based evaluation (Opt@K)
│   ├── data/                    # Data models and schemas
│   └── utils/                   # Utility functions
│
├── scripts/                     # Utility scripts
│   ├── runners/                 # GPU benchmark runners
│   │   └── run_3way_benchmarks.py    # 3-way comparison runner
│   ├── docker/                  # Docker image management
│   ├── upload/                  # HuggingFace upload scripts
│   ├── tests/                   # Test scripts
│   └── archive/                 # Archived scripts
│
├── ISO-Bench/                   # Agent benchmarking harness
│   ├── bench/                   # CLI: plan → prepare → report
│   ├── tasks/                   # Task configs (vllm.yaml, sglang.yaml)
│   ├── config/                  # Agent-specific configs
│   └── state/                   # Runtime state and run outputs
│
├── data/                        # Datasets
│   ├── vllm_dataset_with_test.jsonl  # Main vLLM dataset
│   ├── Inferencebench.jsonl          # Inference benchmark
│   ├── mappings/                     # Commit mappings
│   └── archive/                      # Archived test data
│
├── docs/                        # Documentation
│   ├── REPRODUCTION.md          # Reproduction guide
│   ├── ENVIRONMENTS.md          # Environment setup
│   ├── AGENTS.md                # Agent documentation
│   ├── dataset_schema.md        # Schema specification
│   └── archive/                 # Archived docs
│
├── third-party/                 # External dependencies
│   ├── trae-agent/              # TRAE agent (submodule)
│   ├── everything_analysis_data/  # Soft metrics analysis (clone separately)
│   └── vllm-lm-eval/            # LM-eval correctness validation (clone separately)
│
├── archive/                     # Archived data and results
│   ├── results/                 # Benchmark results
│   └── misc/                    # Experimental data
│
└── vllm/, sglang/               # Git submodules
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

### 2. Agent Benchmarking (ISO-Bench)

Run AI agents on optimization tasks. Supports **TRAE**, **Claude Code**, **Codex**, and **OpenHands**.

```bash
cd ISO-Bench

# Setup (one-time)
uv venv --python 3.12 .venv
source .venv/bin/activate
uv pip install -r requirements.txt

# Configure environment variables for your agent
export TRAE_PYTHON=/path/to/bench-env/bin/python
export TRAE_CONFIG=/path/to/trae_config.yaml

# Plan benchmarks
python -m bench.cli plan tasks/vllm.yaml --out state/plan.json

# Execute agent benchmarks
python -m bench.cli prepare tasks/vllm.yaml --from-plan state/plan.json --bench-cfg bench.yaml

# Generate reports
python -m bench.cli report state/runs/<run_id>

# Validate configuration
python -m bench.cli doctor --bench-cfg bench.yaml
```

**Switching agents**: Edit `ISO-Bench/bench.yaml` and change `agents.default` to one of:
- `trae` - TRAE agent (requires `bench-env` with trae-agent)
- `claude_code` - Claude Code CLI
- `codex_cli` - Codex CLI
- `openhands` - OpenHands agent

### 3. Performance Benchmarks (3-Way Comparison)

Run GPU benchmarks comparing baseline, human, and agent patches:

**Requirements:**
- Docker installed and running
- NVIDIA GPU (H100 recommended for large models)
- HuggingFace token (for gated models like Llama)

```bash
source .venv/bin/activate

# Dry-run to see what would be executed
python scripts/runners/run_3way_benchmarks.py \
    --agent-type trae_gpt5 \
    --commits 8aa1485f \
    --dry-run

# Run actual benchmark (requires GPU)
python scripts/runners/run_3way_benchmarks.py \
    --agent-type trae_gpt5 \
    --commits 8aa1485f \
    --timeout 900

# Available agent types:
#   claude_code, codex_gpt5, codex_cli, trae_gpt5, trae_sonnet45
```

**What it does:**
1. Pulls pre-built Docker images from DockerHub
2. Runs vLLM serving/throughput benchmarks for:
   - **Baseline**: Parent commit (before optimization)
   - **Human**: Developer's optimized commit
   - **Agent**: AI agent's patch applied to baseline
3. Outputs results to `archive/results/2026-01/omniperf_results_3way_<agent>/`

### 4. Evaluation Harness (Opt@K)

Run model predictions and compute Opt@K metrics:

```bash
source .venv/bin/activate
uv run src/harness/opt_at_k.py \
    --model <model_name> \
    --prediction_paths <predictions.jsonl> \
    --timeout 3600 --run_id <run_id> --k 10
```

### 5. Soft Metrics Analysis

Analyze agent patches using LLM-based evaluation (comparing against human expert optimizations):

```bash
# Clone the analysis repo (one-time setup)
cd third-party
git clone git@github.com:thekernelcompany/everything_analysis_data.git
cd everything_analysis_data
git checkout feature/vllm-sglang-combined

# Set up environment
export OPENROUTER_API_KEY="sk-or-..."

# Run soft metrics analysis
python scripts/generate_soft_metrics.py \
    --agent claude_code \
    --project vllm \
    --model google/gemini-3-flash-preview

# Available agents: trae_sonnet, trae_gpt5, claude_code, codex
```

**What it analyzes:**
- Task understanding and code comprehension
- Optimization approach quality (same target vs different target)
- Technique overlap with human solution
- Speedup likelihood estimation

Results are saved to `soft_metrics/vllm/soft_metrics.json`.

### 6. LM-Eval Correctness Validation

Validate that agent patches don't break model correctness using lm-eval benchmarks:

```bash
# Clone the lm-eval repo (one-time setup)
cd third-party
git clone https://github.com/ayushnangia/vllm-lm-eval.git
cd vllm-lm-eval

# Set up environment
export HF_TOKEN="hf_..."

# Validate setup
./test_setup.sh --quick

# Check commit status
python scripts/check_status.py --good

# Run full evaluation (requires GPU)
./run_quick_test.sh
```

**Requirements:**
- Docker with nvidia-container-toolkit
- NVIDIA GPU with CUDA support
- HuggingFace token for gated models

## Virtual Environments

| Environment | Purpose |
|-------------|---------|
| `.venv/` | Main OmniPerf-Bench, dataset generation, evaluation harness |
| `bench-env/` | Agent benchmarking (TRAE, Codex), Modal cloud execution |
| `ISO-Bench/.venv/` | ISO-Bench CLI, OpenHands |

## Environment Variables

Create a `.env` file in the project root:

```bash
# Required for dataset generation (at least one)
OPENAI_API_KEY="sk-..."
# OR
ANTHROPIC_API_KEY="sk-ant-..."

# For ISO-Bench agents
TRAE_PYTHON="/path/to/bench-env/bin/python"
TRAE_CONFIG="/path/to/third-party/trae-agent/trae_config.yaml"

# Optional
OPENROUTER_API_KEY="sk-or-..."    # Alternative LLM provider
HF_TOKEN="hf_..."                  # For dataset uploads and gated models
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

### ISO-Bench agent not found

Set the appropriate environment variables:
```bash
export TRAE_PYTHON=/absolute/path/to/bench-env/bin/python
export TRAE_CONFIG=/absolute/path/to/trae_config.yaml
```

### API key errors

Ensure your `.env` file exists and contains valid keys:
```bash
cat .env  # Should show OPENAI_API_KEY="sk-..."
```

## Documentation

- **[docs/REPRODUCTION.md](docs/REPRODUCTION.md)** - Step-by-step reproduction guide
- **[docs/ENVIRONMENTS.md](docs/ENVIRONMENTS.md)** - Virtual environment setup
- **[docs/AGENTS.md](docs/AGENTS.md)** - Agent configuration guide
- **[docs/dataset_schema.md](docs/dataset_schema.md)** - Dataset schema specification
- **[ISO-Bench/README.md](ISO-Bench/README.md)** - ISO-Bench documentation

## Requirements

- Python 3.12+
- Docker (for evaluation harness and benchmarks)
- NVIDIA GPU with CUDA support (H100 recommended for large models)
- Git with LFS support
- HuggingFace account (for gated model access)

## Submodules and External Repos

```bash
# If cloned without --recursive:
git submodule update --init --recursive
```

**Included submodules:**
- `vllm/` - vLLM fork for benchmarking
- `sglang/` - SGLang for comparison benchmarks
- `third-party/trae-agent/` - TRAE agent integration

**Additional repos (clone separately for full functionality):**
```bash
# Soft metrics analysis
cd third-party
git clone git@github.com:thekernelcompany/everything_analysis_data.git
cd everything_analysis_data && git checkout feature/vllm-sglang-combined && cd ..

# LM-eval correctness validation
git clone https://github.com/ayushnangia/vllm-lm-eval.git
```

## Citation

```bibtex
@misc{omniperf-bench,
  title={OmniPerf-Bench: A Benchmark for Evaluating AI Agents on Software Performance Optimization},
  author={The Kernel Company},
  year={2026},
  url={https://github.com/thekernelcompany/OmniPerf-Bench}
}
```
