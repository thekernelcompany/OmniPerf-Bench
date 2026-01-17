# Docker Image Rebuild Guide for SGLang Benchmarks

**Date:** 2026-01-17
**Purpose:** Document all failure reasons and provide rebuild instructions

---

## Executive Summary

Out of 80 commits analyzed, only 2 were successfully benchmarked. This document explains why commits failed and how to fix them when rebuilding Docker images.

---

## Failure Categories Overview (Verified 2026-01-17)

| Category | Count | Fixable? |
|----------|-------|----------|
| Successfully completed | 3 | N/A |
| No Docker image built | 3 | Build images |
| FlashInfer incompatible | 1 | Rebuild with correct flashinfer |
| Corrupt binary (arch mismatch) | 1 | Rebuild image |
| VLM not supported | 1 | Rebuild with multimodal support |
| No perf_command extracted | 46 | Need command inference |
| Multi-GPU required | 23 | Need infrastructure |

**Important:** Zero commits failed due to OOM. The exit 137 errors seen earlier were transient GPU resource conflicts, not image issues.

---

## Category 1: Verified Status of Each Candidate (2026-01-17)

After testing with a free GPU (0 MiB used), here's the actual status:

| Commit | sgl_kernel | flashinfer | Benchmark | Actual Issue |
|--------|------------|------------|-----------|--------------|
| `021f76e4` | OK | OK | OK | **COMPLETED** |
| `6fc17596` | OK | OK | OK | **COMPLETED** |
| `79961afa` | OK | BROKEN | - | FlashInfer `BatchDecodeWithPagedKVCacheWrapper` missing |
| `2bd18e2d` | BROKEN | - | - | "cannot execute binary file" (corrupt/arch mismatch) |
| `93470a14` | N/A | N/A | N/A | No Docker image on Hub |
| `bb3a3b66` | N/A | N/A | N/A | No Docker image on Hub |
| `d1112d85` | N/A | N/A | N/A | No Docker image on Hub |
| `ddcf9fe3` | **OK** | OK | **OK** | **COMPLETED** (used bench_serving instead) |
| `3212c2ad` | OK | OK | BROKEN | VLM multimodal processor not registered |

### Key Finding

The original "libnuma" classification was **incorrect**. When tested properly:
- `ddcf9fe3` - sgl_kernel loads fine, but `benchmark/gsm8k/bench_sglang.py` doesn't exist
- `2bd18e2d` - Binary is corrupt, not a libnuma issue
- `79961afa` - sgl_kernel works, but flashinfer has incompatible version

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

## Category 7: VLM (Vision-Language Model) Support

### Symptoms

```
ValueError: Cannot find corresponding multimodal processor registered in sglang for model type `clip_vision_model`
```

Or:

```
NotImplementedError: Multimodal processor not implemented for this model
```

### Affected Commits

| Commit | PR | Model | Issue |
|--------|-----|-------|-------|
| `3212c2ad` | #6003 | llava-hf/llava-1.5-7b-hf | VLM processor not registered |

### Root Cause

The SGLang version in the Docker image was installed without full multimodal support. The `clip_vision_model` processor is not registered because:

1. SGLang was installed with minimal dependencies (`pip install -e "python"` instead of `pip install -e "python[all]"`)
2. The multimodal processors were not properly initialized
3. Missing vision-related dependencies (pillow, torchvision, etc.)

### Fix When Rebuilding

#### Option A: Full Installation with Multimodal Support (Recommended)

```dockerfile
FROM nvidia/cuda:12.4.1-devel-ubuntu22.04

# System dependencies including image processing libs
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    git \
    libnuma-dev \
    libjpeg-dev \
    libpng-dev \
    && rm -rf /var/lib/apt/lists/*

# SGLang with ALL dependencies including multimodal
ARG COMMIT_HASH
RUN git clone https://github.com/sgl-project/sglang.git /sglang && \
    cd /sglang && \
    git checkout ${COMMIT_HASH} && \
    pip install -e "python[all]"

# Explicitly install vision dependencies
RUN pip install --no-cache-dir \
    pillow \
    torchvision \
    transformers[vision]

# FlashInfer for H100
RUN pip install flashinfer -f https://flashinfer.ai/whl/cu124/torch2.4/

WORKDIR /sglang
```

#### Option B: Patch Existing Image

If you have a working image but just need VLM support:

```bash
# Start container interactively
docker run -it --gpus all YOUR_IMAGE bash

# Inside container, install missing deps
pip install pillow torchvision "transformers[vision]"

# Reinstall sglang with all extras
cd /sglang  # or wherever sglang is installed
pip install -e "python[all]"

# Test VLM support
python3 -c "
from sglang.srt.multimodal.processors.llava import LlavaImageProcessor
print('VLM OK')
"

# Commit the changes
# (from host): docker commit CONTAINER_ID new-image:tag
```

### Verification Commands

```bash
IMAGE="your-vlm-enabled-image:tag"

# 1. Check VLM processor registration
docker run --rm --gpus all $IMAGE python3 -c "
from sglang.srt.managers.multimodal_processor import get_mm_processor
print('Multimodal processor module OK')
"

# 2. Check llava specifically
docker run --rm --gpus all $IMAGE python3 -c "
from transformers import LlavaProcessor
print('LlavaProcessor OK')
"

# 3. Test server startup with VLM model
docker run --rm --gpus all -e HF_TOKEN=\$HF_TOKEN $IMAGE \
    timeout 120 python3 -m sglang.launch_server \
    --model-path llava-hf/llava-1.5-7b-hf \
    --port 30000

# 4. Run VLM benchmark
docker run --rm --gpus all -e HF_TOKEN=\$HF_TOKEN $IMAGE \
    python3 -m sglang.bench_serving \
    --backend sglang \
    --model llava-hf/llava-1.5-7b-hf \
    --dataset-name mmmu \
    --num-prompts 50 \
    --request-rate 2
```

### VLM-Specific Benchmark Commands

For VLM commits, use the `mmmu` dataset:

```bash
# Standard VLM benchmark
python3 -m sglang.bench_serving \
    --backend sglang \
    --model llava-hf/llava-1.5-7b-hf \
    --dataset-name mmmu \
    --num-prompts 50 \
    --request-rate 2
```

### VLM Commits to Rebuild

| Commit | PR | Subject | Model |
|--------|-----|---------|-------|
| `3212c2ad` | #6003 | VLM tensor transport (16% faster) | llava-hf/llava-1.5-7b-hf |
| `bb3a3b66` | #137 | Faster JSON decoding for llava | llava-hf/llava-1.5-7b-hf |

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
| VLM not supported | Dockerfile | Use `pip install -e "python[all]"` + vision deps |

---

## Priority Commits to Fix

Based on verified testing (2026-01-17):

| Priority | Commit | Fix Required | Effort |
|----------|--------|--------------|--------|
| 1 | `3212c2ad` | Rebuild with VLM support (`python[all]`) | Medium |
| 2 | `79961afa` | Rebuild with compatible flashinfer | Medium |
| 3 | `93470a14` | Build Docker image | Medium |
| 4 | `bb3a3b66` | Build Docker image + VLM support | Medium |
| 5 | `d1112d85` | Build Docker image | Medium |
| 6 | `2bd18e2d` | Rebuild image (corrupt binary) | Medium |

### Completed Benchmarks (No Fix Needed)

| Commit | PR | Status |
|--------|-----|--------|
| `021f76e4` | #6994 LoRA stream sync | **COMPLETED** |
| `6fc17596` | #5945 FA3 pad operation | **COMPLETED** |
| `ddcf9fe3` | #3731 Triton attention | **COMPLETED** |

### High Priority: 3212c2ad (VLM 16% Improvement)

This commit claims **16% faster VLM inference** (207.7s -> 173.3s). The Docker image exists and sgl_kernel loads, but VLM support is missing.

**Error:**
```
ValueError: Cannot find corresponding multimodal processor registered in sglang for model type `clip_vision_model`
```

**Fix:** Rebuild with full multimodal support:
```bash
# In Dockerfile
pip install -e "python[all]"
pip install pillow torchvision "transformers[vision]"
```

**Benchmark command after fix:**
```bash
python3 -m sglang.bench_serving \
    --backend sglang \
    --model llava-hf/llava-1.5-7b-hf \
    --dataset-name mmmu \
    --num-prompts 50 \
    --request-rate 2
```

---

*Generated: 2026-01-17*
*Updated: 2026-01-17 (Added VLM support section, updated completed benchmarks)*
