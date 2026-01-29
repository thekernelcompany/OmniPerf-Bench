# Benchmark Runners

This document describes the canonical benchmark runner implementations in OmniPerf-Bench.

## Overview

OmniPerf-Bench provides multiple benchmark runners for different execution environments:

| Runner | Location | Purpose |
|--------|----------|---------|
| Modal Benchmark | `src/eval/modal_benchmark.py` | Cloud GPU execution via Modal |
| Local Docker | `scripts/runners/local_docker_benchmark.py` | Local Docker container benchmarks |
| 3-Way Benchmark | `scripts/runners/hero_3way_benchmark.py` | Baseline/Human/Agent comparison |
| SGLang Modal | `src/eval/sglang_modal_benchmark.py` | SGLang-specific benchmarks |

## 1. Modal Benchmark Runner

**File**: `src/eval/modal_benchmark.py`

**Purpose**: Execute benchmarks on Modal cloud GPUs (H100s) for consistent, reproducible results.

### Usage

```bash
# Deploy to Modal
source bench-env/bin/activate
modal deploy src/eval/modal_benchmark.py

# Run benchmarks
modal run src/eval/modal_benchmark.py::run_benchmark \
    --commit <commit_hash> \
    --benchmark-type serving
```

### Features

- H100 GPU access via Modal
- Automatic Docker image management
- Parallel execution support
- Result persistence to Modal volumes

## 2. Local Docker Benchmark Runner

**File**: `scripts/runners/local_docker_benchmark.py`

**Purpose**: Run benchmarks locally using Docker containers. Useful for commits that fail on Modal or for local debugging.

### Usage

```bash
source .venv/bin/activate

# Run human benchmark
python scripts/runners/local_docker_benchmark.py \
    --commit <commit_hash> \
    --agent-type claude_code \
    --mode human

# Run agent benchmark
python scripts/runners/local_docker_benchmark.py \
    --commit <commit_hash> \
    --agent-type claude_code \
    --mode agent

# Run baseline benchmark
python scripts/runners/local_docker_benchmark.py \
    --commit <commit_hash> \
    --baseline
```

### Supported Agents

- `claude_code` - Claude Code agent patches
- `codex_gpt5` - Codex with GPT-5
- `trae_gpt5` - TRAE with GPT-5
- `trae_sonnet45` - TRAE with Sonnet 4.5

### Features

- Multi-agent support
- Baseline/Human/Agent comparison modes
- Model compatibility overrides for older vLLM versions
- Automatic Docker image selection

## 3. 3-Way Benchmark Runner

**File**: `scripts/runners/hero_3way_benchmark.py`

**Purpose**: Orchestrate complete 3-way comparisons (Baseline vs Human vs Agent) across multiple commits.

### Usage

```bash
source bench-env/bin/activate

# Run 3-way benchmarks on Modal
python scripts/runners/hero_3way_benchmark.py \
    --commits-file vllm_39_benchmarkable_commits.txt \
    --agent-type claude_code
```

### Features

- Batch processing of multiple commits
- Progress tracking and resumption
- Automatic result aggregation
- Modal deployment integration

## 4. SGLang Modal Benchmark

**File**: `src/eval/sglang_modal_benchmark.py`

**Purpose**: Benchmark SGLang-specific optimizations on Modal cloud.

### Usage

```bash
source bench-env/bin/activate
modal deploy src/eval/sglang_modal_benchmark.py

modal run src/eval/sglang_modal_benchmark.py::run_benchmark \
    --commit <sglang_commit>
```

## Evaluation Harness

**File**: `src/harness/opt_at_k.py`

**Purpose**: Compute Opt@K metrics for model predictions.

### Usage

```bash
source .venv/bin/activate

uv run src/harness/opt_at_k.py \
    --model gpt-4o \
    --prediction_paths data/predictions.jsonl \
    --timeout 3600 \
    --run_id evaluation_run \
    --k 10
```

### Metrics

- **Opt@K**: Probability of finding optimal solution in K attempts
- **Speedup**: Performance improvement ratio
- **Functional Correctness**: Test pass rate

## Configuration

### Docker Images

Pre-built images for benchmarking:
- `ayushnangia16/nvidia-vllm-docker:<commit>` - Human optimized vLLM
- `shikhar481/vllm_fixed_human_images:<commit>` - Fixed compatibility images

### Environment Variables

```bash
# HuggingFace token for model access
HF_TOKEN="hf_..."

# Docker Hub credentials (for pushing images)
DOCKER_USERNAME="..."
DOCKER_PASSWORD="..."
```

## Troubleshooting

### Docker Build Failures

Some commits may have build issues. Check:
1. CUDA compatibility
2. PyTorch version requirements
3. vLLM dependency changes

### Modal Execution Timeouts

Increase timeout or use local Docker for long-running benchmarks:
```bash
# Local fallback
python scripts/runners/local_docker_benchmark.py --timeout 7200 ...
```

### Model Compatibility

Older vLLM versions may not support newer models. The runners include automatic model overrides:
- Llama-3.1 -> Llama-3 for older vLLM
- Bamba -> Llama-3 for unsupported architectures
