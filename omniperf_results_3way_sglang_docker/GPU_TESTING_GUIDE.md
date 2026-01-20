# GPU Testing Guide for Rebuilt SGLang Images

**Date:** 2026-01-20 (Updated)

## Quick Start (on H100)

```bash
# Pull and test the v0.4.x images (NEW - built from source with CUDA stubs)
docker pull shikhar481/sglang-images:2bd18e2d-src
docker pull shikhar481/sglang-images:d1112d85-src

# Test v0.4.x imports
docker run --rm --gpus all shikhar481/sglang-images:2bd18e2d-src python3 -c "
import torch
print(f'CUDA: {torch.cuda.is_available()}')
print(f'GPU: {torch.cuda.get_device_name(0)}')
import sglang, sgl_kernel
print(f'SGLang {sglang.__version__} - All imports OK')
"

docker run --rm --gpus all shikhar481/sglang-images:d1112d85-src python3 -c "
import torch
print(f'CUDA: {torch.cuda.is_available()}')
print(f'GPU: {torch.cuda.get_device_name(0)}')
import sglang, sgl_kernel
print(f'SGLang {sglang.__version__} - All imports OK')
"
```

## Images Status

### v0.4.x Images (NEW - Built from Source)

| Commit | Version | Status | Notes |
|--------|---------|--------|-------|
| `2bd18e2d-src` | v0.4.1.post6 | **PUSHED** | Memory pool optimization (PR #2901) |
| `d1112d85-src` | v0.4.4.post1 | **PUSHED** | input_embeds endpoint (PR #2797) |
| `79961afa` | v0.4.6.post2 | BLOCKED | Requires torch>=2.6.0 |
| `93470a14` | v0.4.5 | BLOCKED | CMake FetchContent can't access flashinfer fork |
| `3212c2ad` | v0.4.9.post4 | BLOCKED | Requires torch>=2.7.1 |

### v0.3.x Images (GPU-Confirmed Working)

| Commit | Version | Status | Notes |
|--------|---------|--------|-------|
| `62757db6-src` | v0.2.11 | **PUSHED** | Cache disabled overhead |
| `ab4a83b2-src` | v0.3.0 | **PUSHED** | Optimize schedule |
| `c98e84c2-src` | v0.3.2 | **PUSHED** | torch.argmax optimization |
| `2854a5ea-src` | v0.3.1.post3 | **PUSHED** | bench_latency fix |
| `9c064bf7-src` | v0.3.2 | **PUSHED** | LoRA Step 1 |
| `b77a02cd-src` | v0.3.4.post2 | **PUSHED** | Grammar backends |
| `9c745d07-v3` | v0.3.5.post2 | **PUSHED** | (DockerHub v3 image) |
| `10189d08-v3` | v0.3.6 | **PUSHED** | (DockerHub v3 image) |

### v0.3.x Images (Pending GPU Test)

| Commit | Version | Status | Notes |
|--------|---------|--------|-------|
| `e5db40dc-src` | v0.3.3.post1 | PENDING | ORJson serialization |
| `b1709305-src` | v0.3.3.post1 | PENDING | Radix tree optimization |
| `8f8f96a6-src` | v0.3.4.post1 | PENDING | stop_token_ids fix |

## Test Commands

### 1. Test v0.4.x Images (NEW)

```bash
# Test 2bd18e2d (v0.4.1.post6) - Memory pool optimization
docker run --rm --gpus all shikhar481/sglang-images:2bd18e2d-src python3 -c "
import torch
assert torch.cuda.is_available(), 'CUDA not available'
print(f'GPU: {torch.cuda.get_device_name(0)}')

import sglang
print(f'sglang: {sglang.__version__}')

import sgl_kernel
print('sgl_kernel: OK')

import vllm
print(f'vllm: {vllm.__version__}')

print('SUCCESS: All imports OK')
"

# Test d1112d85 (v0.4.4.post1) - input_embeds endpoint
docker run --rm --gpus all shikhar481/sglang-images:d1112d85-src python3 -c "
import torch
assert torch.cuda.is_available(), 'CUDA not available'
print(f'GPU: {torch.cuda.get_device_name(0)}')

import sglang
print(f'sglang: {sglang.__version__}')

import sgl_kernel
print('sgl_kernel: OK')

import vllm
print(f'vllm: {vllm.__version__}')

print('SUCCESS: All imports OK')
"
```

### 2. Test v0.3.x Images

```bash
# Test ab4a83b2-src (v0.3.0)
docker run --rm --gpus all shikhar481/sglang-images:ab4a83b2-src python3 -c "
import torch
assert torch.cuda.is_available(), 'CUDA not available'
print(f'GPU: {torch.cuda.get_device_name(0)}')

import sglang
print('sglang: OK')

import outlines
print('outlines: OK')

from pyairports.airports import AIRPORT_LIST
print(f'pyairports: OK ({len(AIRPORT_LIST)} airports)')

import importlib.metadata
print(f'Version: {importlib.metadata.version(\"sglang\")}')
"
```

### 3. Benchmark Tests

```bash
# Benchmark v0.4.x (use TinyLlama for quick test)
docker run --rm --gpus all -e HF_TOKEN=$HF_TOKEN \
    shikhar481/sglang-images:2bd18e2d-src \
    python3 -m sglang.bench_latency \
    --model TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
    --batch-size 1 --input-len 128 --output-len 32

# Benchmark v0.3.x
docker run --rm --gpus all -e HF_TOKEN=$HF_TOKEN \
    shikhar481/sglang-images:ab4a83b2-src \
    python3 -m sglang.bench_latency \
    --model TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
    --batch-size 1 --input-len 128 --output-len 32
```

## v0.4.x Build Details

The v0.4.x images were built from source using CUDA stubs (no GPU required for building).

### Key Build Configuration

```dockerfile
# CUDA stubs for building without GPU
RUN ln -sf /usr/local/cuda/targets/x86_64-linux/lib/stubs/libcuda.so \
           /usr/lib/x86_64-linux-gnu/libcuda.so

# Build optimizations
ENV MAX_JOBS=32
ENV NVCC_THREADS=2

# Required: git submodules for sgl-kernel dependencies
RUN git clone --recursive https://github.com/sgl-project/sglang.git
RUN git submodule update --init --recursive
```

### Dependency Matrix

| Commit | Version | Torch | FlashInfer | vLLM |
|--------|---------|-------|------------|------|
| 2bd18e2d | v0.4.1.post6 | 2.4.0 | 0.1.6 | 0.6.3.post1 |
| d1112d85 | v0.4.4.post1 | 2.5.1 | 0.2.3 | 0.7.2 |

## Expected Results

### Success Criteria
- [x] 2bd18e2d-src: imports work, sgl_kernel loads
- [x] d1112d85-src: imports work, sgl_kernel loads
- [ ] GPU benchmark runs without errors
- [ ] No ABI mismatch errors for sgl_kernel

### Known Issues
- v0.4.x images are larger (~17GB) due to full build from source
- CUDA warning about stub library is normal when testing without GPU
- vLLM version varies between commits

## Reporting Results

Please report:
1. GPU model (nvidia-smi output)
2. Import test results (pass/fail)
3. Benchmark results or error messages
4. Any segfaults or CUDA errors

## All Available Images

```bash
# List all available images
docker pull shikhar481/sglang-images:2bd18e2d-src  # v0.4.1.post6 (NEW)
docker pull shikhar481/sglang-images:d1112d85-src  # v0.4.4.post1 (NEW)
docker pull shikhar481/sglang-images:62757db6-src  # v0.2.11
docker pull shikhar481/sglang-images:ab4a83b2-src  # v0.3.0
docker pull shikhar481/sglang-images:c98e84c2-src  # v0.3.2
docker pull shikhar481/sglang-images:2854a5ea-src  # v0.3.1.post3
docker pull shikhar481/sglang-images:9c064bf7-src  # v0.3.2
docker pull shikhar481/sglang-images:e5db40dc-src  # v0.3.3.post1
docker pull shikhar481/sglang-images:b1709305-src  # v0.3.3.post1
docker pull shikhar481/sglang-images:b77a02cd-src  # v0.3.4.post2
docker pull shikhar481/sglang-images:8f8f96a6-src  # v0.3.4.post1
```
