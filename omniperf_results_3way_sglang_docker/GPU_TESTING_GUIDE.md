# GPU Testing Guide for Rebuilt SGLang Images

**Date:** 2026-01-19

## Quick Start (on H100)

```bash
# Pull and test the 2 ready images
docker pull shikhar481/sglang-images:2a754e57
docker pull shikhar481/sglang-images:ab4a83b2

# Test basic imports
docker run --rm --gpus all shikhar481/sglang-images:2a754e57 python3 -c "
import torch
print(f'CUDA: {torch.cuda.is_available()}')
print(f'GPU: {torch.cuda.get_device_name(0)}')
import sglang, rpyc, outlines
print('All imports OK')
"

docker run --rm --gpus all shikhar481/sglang-images:ab4a83b2 python3 -c "
import torch
print(f'CUDA: {torch.cuda.is_available()}')
print(f'GPU: {torch.cuda.get_device_name(0)}')
import sglang, outlines
from pyairports.airports import AIRPORT_LIST
print(f'All imports OK, {len(AIRPORT_LIST)} airports')
"
```

## Images Status

| Commit | Version | Status | Notes |
|--------|---------|--------|-------|
| `2a754e57` | 0.1.17 | PUSHED | Fixed: rpyc, outlines |
| `ab4a83b2` | 0.3.0 | PUSHED | Fixed: pyairports mock |
| `79961afa` | 0.4.6.post2 | BLOCKED | Needs GPU rebuild for sgl-kernel |

## Test Commands

### 1. Test 2a754e57 (v0.1.17)

```bash
# Basic import test
docker run --rm --gpus all shikhar481/sglang-images:2a754e57 python3 -c "
import torch
assert torch.cuda.is_available(), 'CUDA not available'
print(f'GPU: {torch.cuda.get_device_name(0)}')

import sglang
print('sglang: OK')

import rpyc
print('rpyc: OK')

import outlines
print('outlines: OK')

import uvloop
print('uvloop: OK')

import importlib.metadata
print(f'Version: {importlib.metadata.version(\"sglang\")}')
"

# Benchmark test (requires HF_TOKEN)
docker run --rm --gpus all -e HF_TOKEN=$HF_TOKEN \
    shikhar481/sglang-images:2a754e57 \
    python3 -m sglang.bench_latency \
    --model meta-llama/Llama-3.1-8B-Instruct \
    --batch-size 1 --input-len 512 --output-len 64
```

### 2. Test ab4a83b2 (v0.3.0)

```bash
# Basic import test
docker run --rm --gpus all shikhar481/sglang-images:ab4a83b2 python3 -c "
import torch
assert torch.cuda.is_available(), 'CUDA not available'
print(f'GPU: {torch.cuda.get_device_name(0)}')

import sglang
print('sglang: OK')

import outlines
print('outlines: OK')

from pyairports.airports import AIRPORT_LIST
print(f'pyairports: OK ({len(AIRPORT_LIST)} airports)')

import uvloop
print('uvloop: OK')

import importlib.metadata
print(f'Version: {importlib.metadata.version(\"sglang\")}')
"

# Benchmark test
docker run --rm --gpus all -e HF_TOKEN=$HF_TOKEN \
    shikhar481/sglang-images:ab4a83b2 \
    python3 -m sglang.bench_latency \
    --model meta-llama/Llama-3.1-8B-Instruct \
    --batch-size 1 --input-len 512 --output-len 64
```

## Rebuild 79961afa on GPU (if needed)

The 79961afa image needs sgl-kernel built from source. This requires GPU for CUDA compilation.

```bash
# Clone and checkout
git clone https://github.com/sgl-project/sglang.git /tmp/sglang_rebuild
cd /tmp/sglang_rebuild
git checkout 79961afa8281f98f380d11db45c8d4b6e66a574f

# Create Dockerfile
cat > Dockerfile << 'EOF'
FROM nvcr.io/nvidia/tritonserver:24.04-py3-min

ENV DEBIAN_FRONTEND=noninteractive

RUN apt update -y && apt install -y \
    python3 python3-pip python3-dev \
    build-essential cmake ninja-build \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /sgl-workspace

# Install build tools and torch FIRST
RUN pip install --upgrade pip setuptools wheel
RUN pip install scikit-build-core cmake ninja
RUN pip install torch --index-url https://download.pytorch.org/whl/cu124

# Copy source
COPY . /sgl-workspace/sglang/
WORKDIR /sgl-workspace/sglang

# Build sgl-kernel from source (requires CUDA)
RUN cd sgl-kernel && pip install . -v

# Install sglang
RUN pip install -e "python[all]" \
    --find-links https://flashinfer.ai/whl/cu124/torch2.6/flashinfer-python

RUN pip install uvloop sentencepiece

WORKDIR /sgl-workspace
EOF

# Build (requires nvidia-docker with GPU access during build)
docker build -t sglang:79961afa-fixed .

# Verify
docker run --rm --gpus all sglang:79961afa-fixed python3 -c "
from sgl_kernel import common_ops
print('sgl_kernel: OK - no ABI mismatch!')
"

# Tag and push if successful
docker tag sglang:79961afa-fixed shikhar481/sglang-images:79961afa
docker push shikhar481/sglang-images:79961afa
```

## Expected Results

### Success Criteria
- [ ] 2a754e57: imports work, benchmark runs
- [ ] ab4a83b2: imports work, pyairports has 28270 airports, benchmark runs
- [ ] 79961afa: (after GPU rebuild) sgl_kernel imports without ABI error

### Known Issues
- 2a754e57 uses vllm 0.5.0 (older, may have different benchmark behavior)
- ab4a83b2 uses vllm 0.5.5
- pyairports is a mock using airportsdata (works but not original package)

## Reporting Results

Please report:
1. GPU model (nvidia-smi output)
2. Import test results (pass/fail)
3. Benchmark results or error messages
4. Any segfaults or CUDA errors
