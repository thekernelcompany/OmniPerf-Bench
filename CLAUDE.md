# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

OmniPerf-Bench is a benchmark framework for evaluating language models on software performance optimization tasks. It provides:
- **Dataset generation pipeline**: Extract performance optimization commits from repositories
- **LLM-powered test generation**: Automated creation of performance tests
- **Evaluation harness**: Docker-based system measuring optimization effectiveness (Opt@K metrics)
- **Agent benchmarking**: Harness for running AI agents (TRAE, Codex, OpenHands, Claude Code) on optimization tasks

## Quick Start

```bash
# Clone with submodules (required)
git clone --recursive <repo-url>
cd OmniPerf-Bench

# Install uv and set up environment (Python 3.12+)
curl -LsSf https://astral.sh/uv/install.sh | sh
uv venv && source .venv/bin/activate
uv sync

# Run the main pipeline on example config
export OPENAI_API_KEY="your_key"
python commit_to_dataset.py configs/experiments.yaml
```

## Virtual Environments

This repository has **two separate virtual environments**:

| Environment | Location | Purpose |
|-------------|----------|---------|
| `.venv/` | Root directory | Main OmniPerf-Bench, collection framework, harness |
| `bench-env/` | Root directory | Agent benchmarking (TRAE, Codex), Modal runners |
| `ISO-Bench/.venv/` | Subdirectory | ISO-Bench CLI (OpenHands) |

```bash
# Main environment
source .venv/bin/activate

# Agent benchmarking (for Modal, TRAE, upload scripts)
source bench-env/bin/activate
```

## Common Development Commands

### Environment Setup
```bash
uv venv && source .venv/bin/activate
uv sync

# If submodules missing:
git submodule update --init --recursive
```

### Main Entry Points

**1. Commit-to-Dataset Pipeline** (primary entry point):
```bash
python commit_to_dataset.py configs/experiments.yaml
# Output: data/vllm_dataset_with_test.jsonl
```

**2. Evaluation Harness** (run model predictions):
```bash
uv run src/harness/opt_at_k.py \
    --model <model_name> \
    --prediction_paths <predictions.jsonl> \
    --timeout 3600 --run_id <run_id> --k 10
```

**3. Collection Framework** (multi-stage dataset generation):
```bash
PYTHONPATH=src python src/collect/analysis/commits.py configs/experiments.yaml
PYTHONPATH=src python src/collect/analysis/apis.py <exp_id>
PYTHONPATH=src python src/collect/generate/generate.py configs/experiments.yaml
```

**4. Agent Benchmarking** (ISO-Bench):
```bash
cd ISO-Bench
.venv/bin/python -m bench.cli plan tasks/vllm.yaml --commits .work/vllm_commits.txt --out ./state/plan.json
.venv/bin/python -m bench.cli prepare tasks/vllm.yaml --from-plan ./state/plan.json --bench-cfg bench.yaml --max-workers 1 --resume
.venv/bin/python -m bench.cli report state/runs/<run_id>
```

**5. Soft Metrics Analysis** (ISO-Bench):
```bash
cd ISO-Bench
source .venv/bin/activate
export OPENROUTER_API_KEY="sk-or-v1-..."

# Analyze all runs for an agent
python -m bench.cli analyze --state-root ./state --data-dir ../data \
    --repo vllm --agent trae --model gpt-5
```

## Architecture

### Directory Structure
```
OmniPerf-Bench/
├── README.md                      # Main documentation
├── CLAUDE.md                      # AI coding assistant instructions
├── commit_to_dataset.py           # Main entry point for dataset creation
│
├── src/                           # Source code
│   ├── collect/                   # Dataset generation pipeline
│   ├── harness/                   # Docker-based evaluation system
│   └── test_scripts/              # Test generation and analysis
│
├── ISO-Bench/                     # Agent benchmarking harness
│   ├── bench/                     # CLI: plan → prepare → report
│   ├── tasks/                     # Task configs (vllm.yaml)
│   ├── state/runs/                # Agent run outputs
│   └── state/analysis/            # Soft metrics analysis results
│
├── scripts/                       # Utility scripts
├── data/                          # Generated datasets
├── docs/                          # Documentation
├── configs/                       # Configuration files
├── tools/                         # Tools and patches
│
├── vllm/, sglang/                 # Git submodules for target repos
├── third-party/                   # External deps (effibench, trae-agent)
│
└── archive/                       # Archived files
    ├── docs/                      # Old documentation
    ├── logs/                      # Execution logs
    ├── scripts/                   # Legacy scripts
    └── misc/                      # Experimental data
```

### Key Components

1. **Collection Framework** (`src/collect/`):
   - `analysis/`: LLM-based commit extraction and API identification
   - `generate/`: Performance test generation using LLMs
   - `execute/`: Cloud execution via SkyPilot

2. **Evaluation Harness** (`src/harness/`):
   - `opt_at_k.py`: Main evaluation runner with Opt@K metrics
   - Docker-based isolated test execution

3. **Agent Benchmarking** (`ISO-Bench/`):
   - `bench/cli.py`: Main CLI entry point
   - `bench/pipeline.py`: Agent execution pipeline
   - `bench/analysis/`: Soft metrics analyzer (LLM-as-a-Judge)
   - Supports OpenHands, TRAE, Codex, Claude Code agents
   - See `ISO-Bench/SOFT_METRICS_ANALYSIS.md` for detailed results

### Soft Metrics Analysis Schema

Each `metrics_summary.json` contains:
```json
{
  "run": { "repo": "vllm", "agent": "trae", "model": "gpt-5" },
  "execution": { "status": "success|error", "duration_s": 1791.6, "steps": 41 },
  "patch": { "generated": true, "lines_added": 28, "files_changed": 1 },
  "scores": {
    "code_understanding": 8.7,
    "task_alignment": 9.0,
    "approach_quality": 8.3,
    "execution_quality": 8.5,
    "overall": 8.62
  }
}
```

### Environment Variables
```bash
OPENAI_API_KEY      # LLM access (OpenAI)
ANTHROPIC_API_KEY   # LLM access (Anthropic)
AWS_REGION          # For Bedrock (Claude models)
GHAPI_TOKEN         # GitHub API access
HF_TOKEN            # HuggingFace dataset uploads
MODAL_TOKEN_ID      # Modal cloud GPU access
MODAL_TOKEN_SECRET  # Modal cloud GPU access
```

## Working with the Codebase

- **Package manager**: Uses `uv` for dependency management
- **Python version**: Requires Python 3.12+
- **PYTHONPATH**: Many scripts require `PYTHONPATH=src` when running from root
- **Submodules**: vllm/, sglang/, third-party/trae-agent are git submodules
- **Git LFS**: Repository uses Git LFS for large files (evaluation logs)
- **Docker**: Required for evaluation harness (isolated test execution)
- **Modal**: Used for cloud GPU benchmarking (H100s)