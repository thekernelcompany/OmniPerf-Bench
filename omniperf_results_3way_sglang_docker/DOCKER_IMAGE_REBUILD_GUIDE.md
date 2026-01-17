# Docker Image Rebuild Guide for SGLang Benchmarks

**Date:** 2026-01-17
**Purpose:** Document all failure reasons and provide rebuild instructions

---

## Executive Summary

Out of 80 commits analyzed, only 2 were successfully benchmarked. This document explains why commits failed and how to fix them when rebuilding Docker images.

---

## Failure Categories Overview

| Category | Count | % | Fixable by Rebuild? |
|----------|-------|---|---------------------|
| No perf_command extracted | 46 | 57.5% | No (need command inference) |
| Multi-GPU required | 23 | 28.8% | No (need infrastructure) |
| No Docker image built | 5 | 6.2% | Yes |
| Broken Docker image (libnuma) | 5 | 6.2% | **Yes** |
| Successfully completed | 2 | 2.5% | N/A |

---

## Category 1: Missing `libnuma.so.1` (MOST COMMON DOCKER ISSUE)

### Symptoms

```
ImportError: libnuma.so.1: cannot open shared object file: No such file or directory
```

This error occurs when importing `sgl_kernel` on H100 GPUs (SM90 architecture).

### Affected Commits

| Commit | PR | Subject |
|--------|-----|---------|
| `2bd18e2d` | #2901 | Memory pool optimization |
| `93470a14` | #5090 | FA3 Code optimization |
| `bb3a3b66` | #137 | JSON decoding for llava |
| `d1112d85` | #2797 | Input embeds endpoint |
| `ddcf9fe3` | #3731 | Triton attention mask |

### Root Cause

Docker images were built without `libnuma-dev` package. The `sgl_kernel` CUDA extension requires `libnuma.so.1` for NUMA-aware memory allocation on multi-socket systems like H100.

### Fix When Rebuilding

Add to Dockerfile:

```dockerfile
# Add BEFORE pip install sglang
RUN apt-get update && apt-get install -y \
    libnuma-dev \
    && rm -rf /var/lib/apt/lists/*
```

Or if using a build script:

```bash
apt-get update && apt-get install -y libnuma-dev
```

### Verification After Rebuild

```bash
docker run --rm --gpus all YOUR_IMAGE python3 -c "import sgl_kernel; print('OK')"
```

Expected output: `OK`

---

## Category 2: OOM / Exit Code 137

### Symptoms

```
Docker exited with code 137
```

Exit code 137 = Container killed (128 + 9 = SIGKILL), typically due to:
- Out of Memory (OOM)
- GPU memory exhaustion
- Resource conflicts with other containers

### Affected Scenarios

This is NOT a Docker image issue - it's a runtime resource issue.

### Root Cause

1. **Other containers using GPU memory** - Multiple SGLang servers can't share a single GPU
2. **Model too large for available memory** - After other processes claim GPU memory
3. **KV cache allocation failure** - SGLang allocates ~54GB KV cache on H100

### Fix

```bash
# Before running benchmark, ensure GPU is free:
docker stop $(docker ps -q)

# Verify GPU is free:
nvidia-smi
# Should show 0 MiB / 81559 MiB used

# Then run benchmark
```

### NOT a Rebuild Issue

This does not require image rebuild - just ensure clean GPU state before running.

---

## Category 3: Missing Docker Images

### Symptoms

```
Error: Image not found on Docker Hub
```

### Affected Commits

5 commits have no corresponding Docker image built.

### Root Cause

- Image was never built
- Build failed during CI/CD
- Image was built but not pushed to Docker Hub

### Fix When Building

Ensure the build script:

1. Uses the correct commit hash (full 40-char hash)
2. Pushes to the correct Docker Hub repo
3. Tags with both short and full hash

```bash
# Build for specific commit
git checkout <commit_hash>
docker build -t ayushnangia16/nvidia-sglang-docker:<full_commit_hash> .
docker push ayushnangia16/nvidia-sglang-docker:<full_commit_hash>

# Also tag with short hash for convenience
docker tag ayushnangia16/nvidia-sglang-docker:<full_commit_hash> \
           ayushnangia16/nvidia-sglang-docker:<short_hash>
docker push ayushnangia16/nvidia-sglang-docker:<short_hash>
```

---

## Category 4: FlashInfer Compatibility

### Symptoms

```
ImportError: cannot import name 'BatchDecodeWithPagedKVCacheWrapper' from 'flashinfer'
```

Or:

```
CUDA error: no kernel image is available for execution on the device
```

### Root Cause

FlashInfer version incompatible with:
- CUDA version in container
- GPU architecture (SM90 for H100)
- SGLang version expectations

### Fix When Rebuilding

```dockerfile
# Pin compatible flashinfer version
RUN pip install flashinfer==0.1.6+cu121torch2.4 -f https://flashinfer.ai/whl/cu121/torch2.4/

# Or build from source for specific architecture
RUN pip install flashinfer --no-build-isolation
```

### Verification

```bash
docker run --rm --gpus all YOUR_IMAGE python3 -c \
    "from flashinfer import BatchDecodeWithPagedKVCacheWrapper; print('OK')"
```

---

## Category 5: Missing Python Dependencies

### Symptoms

Various ImportError messages for packages like:
- `transformers`
- `torch`
- `triton`
- `xgrammar`

### Fix When Rebuilding

```dockerfile
# Ensure all dependencies are installed
RUN pip install --no-cache-dir \
    torch>=2.4.0 \
    transformers>=4.44.0 \
    triton>=3.0.0 \
    xgrammar \
    outlines
```

---

## Category 6: Model Download Failures

### Symptoms

```
OSError: We couldn't connect to 'https://huggingface.co' to load this file
```

Or:

```
Repository Not Found for url: https://huggingface.co/meta-llama/...
401 Client Error: Unauthorized
```

### Root Cause

- HuggingFace token not set
- Gated model access not granted
- Network issues

### Fix at Runtime (Not Rebuild)

```bash
# Set HF token when running container
docker run --rm --gpus all \
    -e HF_TOKEN=hf_your_token_here \
    YOUR_IMAGE ...

# Or mount token file
docker run --rm --gpus all \
    -v ~/.cache/huggingface:/root/.cache/huggingface \
    YOUR_IMAGE ...
```

### Pre-download Models in Image (Alternative)

```dockerfile
ARG HF_TOKEN
RUN huggingface-cli login --token $HF_TOKEN && \
    huggingface-cli download meta-llama/Llama-3.1-8B-Instruct
```

---

## Complete Rebuild Checklist

When rebuilding Docker images, ensure ALL of these are addressed:

### Dockerfile Template

```dockerfile
FROM nvidia/cuda:12.4.1-devel-ubuntu22.04

# 1. System dependencies (CRITICAL: include libnuma-dev)
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    git \
    libnuma-dev \
    && rm -rf /var/lib/apt/lists/*

# 2. Python dependencies
RUN pip install --no-cache-dir \
    torch>=2.4.0 \
    transformers>=4.44.0 \
    triton>=3.0.0

# 3. FlashInfer (architecture-specific)
RUN pip install flashinfer -f https://flashinfer.ai/whl/cu124/torch2.4/

# 4. SGLang from specific commit
ARG COMMIT_HASH
RUN git clone https://github.com/sgl-project/sglang.git /sglang && \
    cd /sglang && \
    git checkout ${COMMIT_HASH} && \
    pip install -e "python[all]"

# 5. Verify sgl_kernel loads
RUN python3 -c "import sgl_kernel; print('sgl_kernel OK')"

# 6. Default command
WORKDIR /sglang
CMD ["python3", "-m", "sglang.launch_server", "--help"]
```

### Build Script

```bash
#!/bin/bash
set -e

COMMIT_HASH=$1
SHORT_HASH=${COMMIT_HASH:0:8}
REPO="ayushnangia16/nvidia-sglang-docker"

echo "Building image for commit: $COMMIT_HASH"

# Build
docker build \
    --build-arg COMMIT_HASH=$COMMIT_HASH \
    -t $REPO:$COMMIT_HASH \
    -t $REPO:$SHORT_HASH \
    .

# Verify before push
echo "Verifying sgl_kernel..."
docker run --rm --gpus all $REPO:$COMMIT_HASH \
    python3 -c "import sgl_kernel; print('VERIFIED')"

# Push both tags
docker push $REPO:$COMMIT_HASH
docker push $REPO:$SHORT_HASH

echo "Successfully built and pushed: $REPO:$SHORT_HASH"
```

---

## Verification Commands

After rebuilding, run these checks:

```bash
IMAGE="ayushnangia16/nvidia-sglang-docker:YOUR_TAG"

# 1. Check sgl_kernel
docker run --rm --gpus all $IMAGE python3 -c "import sgl_kernel; print('OK')"

# 2. Check flashinfer
docker run --rm --gpus all $IMAGE python3 -c "import flashinfer; print('OK')"

# 3. Check SGLang server starts
docker run --rm --gpus all $IMAGE timeout 60 python3 -m sglang.launch_server \
    --model-path meta-llama/Llama-3.1-8B-Instruct \
    --port 30000 &
sleep 45
curl -s http://localhost:30000/health | grep -q "ok" && echo "Server OK"

# 4. Run quick benchmark
docker run --rm --gpus all $IMAGE python3 -m sglang.bench_latency \
    --model meta-llama/Llama-3.1-8B-Instruct \
    --batch-size 1 \
    --input-len 128 \
    --output-len 16
```

---

## Summary Table: What to Fix

| Issue | Fix Location | Action |
|-------|--------------|--------|
| libnuma.so.1 missing | Dockerfile | Add `apt-get install libnuma-dev` |
| Exit 137 (OOM) | Runtime | Stop other containers, free GPU |
| FlashInfer incompatible | Dockerfile | Pin correct version for CUDA/arch |
| Missing image | Build pipeline | Build and push image |
| HF token missing | Runtime | Pass `-e HF_TOKEN=...` |
| Model gated | HuggingFace account | Request access to model |

---

## Priority Commits to Rebuild

These commits have inferred benchmark commands and would be runnable after rebuild:

| Priority | Commit | Claimed Improvement | Current Issue |
|----------|--------|---------------------|---------------|
| 1 | `2bd18e2d` | Memory pool opt | libnuma missing |
| 2 | `93470a14` | FA3 code opt | libnuma missing |
| 3 | `bb3a3b66` | JSON decoding | libnuma missing |
| 4 | `d1112d85` | Input embeds | libnuma missing |
| 5 | `ddcf9fe3` | Triton attention | libnuma missing |

All 5 require the same fix: rebuild with `libnuma-dev` installed.

---

*Generated: 2026-01-17*
