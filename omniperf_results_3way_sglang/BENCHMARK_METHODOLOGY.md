# SGLang Docker Benchmark Methodology

## Overview

This document describes how SGLang commits are benchmarked using Docker containers. The process involves:
1. Building Docker images for each SGLang commit
2. Running standardized benchmarks
3. Collecting and analyzing metrics

## Benchmark Process

### 1. Docker Image Build

For each SGLang commit, we build a Docker image with the following approach:

**SGLang 0.3.x** (FIXED_DOCKERFILE_03X):
- Base: `nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04`
- Python 3.11
- PyTorch 2.4.0 (CUDA 12.4)
- flashinfer 0.1.6 (SGLang 0.3.x API requires this specific version)
- transformers 4.46.0 (has AutoProcessor)
- numpy <2.0 (numpy 2.x breaks outlines)
- outlines 0.0.46 (has fsm module)

**SGLang 0.4.x+** (FIXED_DOCKERFILE_04X):
- Base: `nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04`
- Python 3.11
- PyTorch: Let SGLang install (uses 2.10.0)
- compressed_tensors <0.13.0 (CRITICAL: v0.13.0 requires transformers.masking_utils which doesn't exist)
- flashinfer: Not pre-installed (torch version mismatch with wheels)
- numpy <2.0

### 2. Benchmark Execution

Each image is benchmarked using `sglang.bench_serving`:

```bash
# Start server
python3 -m sglang.launch_server \
    --model-path TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
    --port 30000 \
    --host 0.0.0.0 \
    --tp-size 1 \
    --mem-fraction-static 0.7 \
    --disable-cuda-graph  # Avoids OOM during graph capture

# Run benchmark
python3 -m sglang.bench_serving \
    --backend sglang \
    --host 127.0.0.1 \
    --port 30000 \
    --dataset-name random \
    --num-prompts 50 \
    --random-input-len 128 \
    --random-output-len 64
```

### 3. Metrics Collected

- **request_throughput** (req/s): Requests processed per second
- **output_throughput** (tok/s): Output tokens generated per second
- **ttft_mean** (ms): Time to first token (mean)
- **ttft_p99** (ms): Time to first token (99th percentile)
- **tpot_mean** (ms): Time per output token (mean)
- **itl_mean** (ms): Inter-token latency (mean)

## Known Issues and Fixes

### Issue 1: compressed_tensors masking_utils (v0.4.x)
**Error**: `ModuleNotFoundError: No module named 'transformers.masking_utils'`

**Root Cause**: compressed_tensors v0.13.0 added `modeling/attention.py` that imports from `transformers.masking_utils`. This module was only added to transformers in October 2025.

**Fix**: Pin `compressed-tensors<0.13.0`

### Issue 2: flashinfer API mismatch (v0.3.x)
**Error**: `AttributeError: module 'flashinfer' has no attribute '_grouped_size_compiled_for_decode_kernels'`

**Root Cause**: SGLang 0.3.x uses flashinfer 0.1.x API, but newer flashinfer 0.2.x changed the API.

**Fix**: Pin `flashinfer==0.1.6` for SGLang 0.3.x

### Issue 3: CUDA OOM during graph capture
**Error**: `CUDA error: out of memory` during CUDA graph capture

**Fix**: Add `--disable-cuda-graph --mem-fraction-static 0.7`

### Issue 4: numpy 2.x breaking outlines
**Error**: `AttributeError: module 'numpy.lib' has no attribute 'function_base'`

**Root Cause**: numpy 2.0 removed `numpy.lib.function_base`

**Fix**: Pin `numpy<2.0`

## File Structure

```
scripts/runners/
├── rebuild_sglang_images.py    # Rebuilds broken images with fixes
├── benchmark_fixed_images.py   # Runs benchmarks on fixed images
├── run_all_sglang_images.py    # Tests all Docker Hub images
└── hero_sglang_benchmark.py    # Main 3-way benchmark orchestrator

omniperf_results_3way_sglang/
├── rebuilt_images/
│   ├── rebuild_results.json     # Build status for each image
│   └── benchmark_results.json   # Benchmark metrics per image
└── BENCHMARK_METHODOLOGY.md     # This file
```

## Docker Repositories

| Repository | Tag Format | Notes |
|------------|------------|-------|
| `shikhar481/sglang-images` | `{8-char-commit}`, `v04x-fixed-{commit}` | Newer fixed images |
| `ayushnangia16/nvidia-sglang-docker` | `{40-char-commit}`, `fixed-{commit}` | Original images |

## Running Benchmarks

### Rebuild and benchmark all broken images:
```bash
python scripts/runners/rebuild_sglang_images.py --all --max 50 --gpu 0
```

### Benchmark fixed images continuously:
```bash
python scripts/runners/benchmark_fixed_images.py --gpus 4,5,6,7 --continuous
```

### Run parallel rebuilds (using multiple GPUs):
```bash
# GPU 0
python rebuild_sglang_images.py --all --max 50 --offset 0 --gpu 0 &
# GPU 1
python rebuild_sglang_images.py --all --max 50 --offset 50 --gpu 1 &
```

## Version Compatibility Matrix

| SGLang Version | torch | flashinfer | compressed_tensors | Status |
|---------------|-------|------------|-------------------|--------|
| 0.3.0-0.3.5   | 2.4.0 | 0.1.6      | any               | OK with FIXED_DOCKERFILE_03X |
| 0.4.0-0.4.2   | 2.4.0 | 0.2.x      | <0.13.0           | OK with FIXED_DOCKERFILE_04X |
| 0.4.3-0.4.6   | 2.10.0| N/A (disabled) | <0.13.0        | OK with FIXED_DOCKERFILE_04X |
