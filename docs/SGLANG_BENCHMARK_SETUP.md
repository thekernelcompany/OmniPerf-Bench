# SGLang Benchmark Infrastructure

## Overview

This document describes the Docker-based benchmark infrastructure for SGLang performance optimization commits. The benchmark system evaluates Claude Code's ability to replicate human performance optimizations.

## Architecture

### Current Approach: Human-Only Mode (Default)

Due to SGLang's multi-package architecture, true 3-way benchmarks (baseline vs human vs agent) are not currently possible with Python overlay. The system defaults to **human-only mode**:

```
Docker Image (HUMAN commit)
    │
    └── HUMAN: Use Docker image as-is → Works reliably
```

### Why 3-Way Doesn't Work (Technical Details)

SGLang consists of multiple interdependent packages that must be ABI-compatible:

```
┌─────────────────────────────────────────────────────────────────┐
│  SGLang = Multi-Package System                                   │
│                                                                  │
│  sglang (Python) ←→ sgl-kernel (CUDA C++) ←→ flashinfer (CUDA)  │
│        ↑                    ↑                      ↑             │
│        └────────────────────┴──────────────────────┘             │
│                    Must match ABI                                │
└─────────────────────────────────────────────────────────────────┘
```

The Python overlay approach fails because:
- Docker image has compiled extensions (`sgl-kernel`, `flashinfer`) for HUMAN commit
- Overlaying Python files from baseline/agent creates ABI mismatch
- Result: `ImportError`, `undefined symbol`, server crashes

**Current statistics**: 0% baseline/agent success rate with overlay approach.

### Future: True 3-Way Benchmarks

To enable proper 3-way comparisons, we need to either:
1. Build separate Docker images for baseline/human/agent per commit
2. Build `sglang` + `sgl-kernel` wheels together from same commit

## Files

| File | Purpose |
|------|---------|
| `src/eval/sglang_modal_benchmark.py` | Modal benchmark runner with human-only and 3-way modes |
| `tools/build_sglang_images.py` | Docker image builder for SGLang commits |
| `tools/push_sglang_to_hf.py` | Push benchmark results to HuggingFace |
| `hero_sglang_benchmark.py` | Hero runner for batch benchmarks |

## Results

### HuggingFace Dataset

Benchmark results are published to:
- **Dataset**: [`Inferencebench/claude-code-sglang-benchmarks`](https://huggingface.co/datasets/Inferencebench/claude-code-sglang-benchmarks)
- **Format**: Matches vLLM benchmark schema

### Local Results

```
omniperf_results_3way_sglang/
└── sglang/
    └── {commit_hash}/
        └── benchmark_result.json
```

### Current Status (Jan 2026)

| Metric | Value |
|--------|-------|
| Total benchmarks | 41 |
| Successful | 17 (41%) |
| Errors | 24 (59%) |
| Has human metrics | 11 (27%) |
| Has baseline metrics | 0 (0%) - Expected due to architecture |
| Has agent metrics | 0 (0%) - Expected due to architecture |

### Error Categories

| Category | Count | Root Cause | Fix |
|----------|-------|------------|-----|
| IMPORT_ERROR | 7 | sgl_kernel missing sm90 (H100) | Rebuild Docker images |
| GPU_OOM | 7 | Model too large for GPU config | Multi-node or skip |
| OTHER | 4 | CuDNN/Torch compatibility | Pin versions |
| TIMEOUT | 2 | LoRA not configured | Skip LoRA benchmarks |
| NCCL_MISMATCH | 2 | NCCL version conflict | Fix in Docker image |
| MODEL_PATH | 1 | Local path not found | Use HF model ID |

## Docker Image Repository

- **Repository**: `ayushnangia16/sglang-docker`
- **Tag format**: Full commit hash (e.g., `93470a14116a60fe5dd43f0599206e8ccabdc211`)
- **Images available**: 74

## Usage

### 1. Build Docker Images

```bash
# Build all images (requires Docker + DockerHub credentials)
python tools/build_sglang_images.py --all

# Build specific commit
python tools/build_sglang_images.py --commit 93470a14

# Dry run
python tools/build_sglang_images.py --all --dry-run
```

### 2. Run Benchmarks

```bash
# Deploy Modal app first
modal deploy src/eval/sglang_modal_benchmark.py

# Run specific commit (human-only mode, default)
python hero_sglang_benchmark.py --commit 93470a14

# Run all commits
python hero_sglang_benchmark.py

# Run with parallel workers
python hero_sglang_benchmark.py --parallel 10

# Dry run to see what would run
python hero_sglang_benchmark.py --dry-run
```

### 3. Push Results to HuggingFace

```bash
# Requires HF_TOKEN environment variable
python tools/push_sglang_to_hf.py
```

### 4. Direct Modal Usage

```bash
python src/eval/sglang_modal_benchmark.py \
    --base-commit <parent_hash> \
    --human-commit <commit_hash> \
    --perf-command "python -m sglang.bench_serving --backend sglang --model ..." \
    --model meta-llama/Llama-3.1-8B-Instruct
```

## Benchmark Types

SGLang supports both server-based and direct benchmarks:

**Server-based (require running server):**
- `sglang.bench_serving` - Serving benchmark

**Direct (no server needed):**
- `sglang.bench_one_batch` - Single batch benchmark
- `sglang.bench_offline_throughput` - Offline throughput benchmark

The infrastructure automatically detects the type and handles accordingly.

## Key Differences from vLLM

| Aspect | vLLM | SGLang |
|--------|------|--------|
| Pre-built wheels | Yes (S3) | No |
| Docker images | Fallback | Primary |
| Image repo | `ayushnangia16/nvidia-vllm-docker` | `ayushnangia16/sglang-docker` |
| Default port | 8000 | 30000 |
| Benchmark mode | 3-way works | Human-only (due to ABI issues) |
| Server command | `vllm serve` | `python -m sglang.launch_server` |

## Configuration

### GPU Configurations

The system automatically selects GPU config based on model:

| Model Pattern | GPU Config |
|---------------|------------|
| deepseek-v3, deepseek-r1 | H100:8 |
| deepseek-v2, llama-3-70b | H100:4 |
| llama-4, mixtral | H100:2 |
| Default | H100:1 |

### Environment Variables

```bash
export HF_TOKEN="your_huggingface_token"  # For model access and dataset push
export MODAL_TOKEN_ID="..."               # Modal authentication
export MODAL_TOKEN_SECRET="..."
```

## Known Issues

1. **IMPORT_ERROR (sgl_kernel)**: Docker images need sm90 architecture support for H100 GPUs
2. **GPU_OOM for large models**: DeepSeek-V3/R1 require multi-node (16+ GPUs)
3. **Baseline/Agent always empty**: Architectural limitation, not a bug

## TODO

- [x] Build Docker images for commits (74 images available)
- [x] Run hero benchmark on commits (41 completed)
- [x] Push results to HuggingFace
- [ ] Rebuild Docker images with sm90 support (fix 7 IMPORT_ERROR)
- [ ] Implement wheel-based approach with sgl-kernel co-build
- [ ] Add multi-node support for large models
