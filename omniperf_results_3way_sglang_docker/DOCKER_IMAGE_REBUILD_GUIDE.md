# Docker Image Rebuild Guide for SGLang Benchmarks

**Date:** 2026-01-17
**Last Updated:** 2026-01-17 (Rebuild completed)
**Purpose:** Document all failure reasons and provide rebuild instructions

---

## Executive Summary

Out of 80 commits analyzed, only 2 were successfully benchmarked initially. **On 2026-01-17, 8 images were successfully rebuilt and pushed to `shikhar481/sglang-images`.**

---

## Rebuild Status (2026-01-17)

### Successfully Rebuilt and Pushed

| Commit | Description | SGLang Version | Status |
|--------|-------------|----------------|--------|
| `93470a14116a60fe5dd43f0599206e8ccabdc211` | FA3 Code optimization | 0.4.5 | **REBUILT & PUSHED** |
| `2bd18e2d767e3a0f8afb5aff427bc8e6e4d297c0` | Memory pool optimization | 0.4.5 | **REBUILT & PUSHED** |
| `d1112d8548eb13c842900b3a8d622345f9737759` | Input embeds endpoint | 0.4.4.post1 | **REBUILT & PUSHED** |
| `ddcf9fe3beacd8aed573c711942194dd02350da4` | Triton attention mask | 0.4.3.post2 | **REBUILT & PUSHED** |
| `79961afa8281f98f380d11db45c8d4b6e66a574f` | FlashInfer fix | 0.4.6.post2 | **REBUILT & PUSHED** |
| `3212c2ad3f7e4fb473dc807b4b176020a778ed5b` | VLM tensor transport (16% faster) | 0.4.9.post4 | **REBUILT & PUSHED** |
| `10189d08dde1096f5759316c0a6ff05962714c4b` | sgl_kernel + triton fix | 0.3.6 | **REBUILT & PUSHED** |
| `f4a8987f6904e4909adb473c52b443a62ba5a4b5` | Parent for c087ddd6 baseline | 0.4.6.post5 | **REBUILT & PUSHED** |

### Skipped

| Commit | Reason |
|--------|--------|
| `bb3a3b6675b1844a13ebe368ad693f3dc75b315b` | Old sglang 0.1.11 - dependencies incompatible with torch 2.4 |

### All Rebuilt Images Verified

Each pushed image passed sanity checks:
- torch: 2.4.0+cu124 OK
- flashinfer: OK
- zmq: OK
- sgl_kernel: INSTALLED

### Build Script Used

The working build script is at: `tools/rebuild_sglang_fixed.sh`

Key fixes applied during rebuild:
1. Added `fastapi`, `uvicorn`, `orjson` for newer sglang versions
2. Added `pybase64` for sglang 0.4.9+
3. Used `--constraint torch==2.4.0` to prevent version drift from vllm
4. Used `[srt]` install variant to avoid vllm dependency conflicts
5. Installed `sgl-kernel` from PyPI (pre-built wheel)

---

## Failure Categories Overview (Verified 2026-01-17)

| Category | Count | Fixable? |
|----------|-------|----------|
| Successfully completed | 3 | N/A |
| **sgl_kernel missing** | 4 | ~~Rebuild with proper sgl_kernel~~ **FIXED** |
| FlashInfer incompatible | 1 | ~~Rebuild with correct flashinfer~~ **FIXED** |
| VLM not supported | 1 | ~~Rebuild with multimodal support~~ **FIXED** |
| No perf_command extracted | 46 | Need command inference |
| Multi-GPU required | 23 | Need infrastructure |

**Important:** Zero commits failed due to OOM. The exit 137 errors seen earlier were transient GPU resource conflicts, not image issues.

---

## Category 1: Verified Status of Each Candidate (Updated 2026-01-17)

After testing with a free GPU (0 MiB used), here's the actual status:

| Commit | sgl_kernel | flashinfer | Benchmark | Status |
|--------|------------|------------|-----------|--------|
| `021f76e4` | OK | OK | OK | **COMPLETED** |
| `6fc17596` | OK | OK | OK | **COMPLETED** |
| `ddcf9fe3` | OK | OK | OK | **REBUILT 2026-01-17** |
| `79961afa` | OK | OK | - | **REBUILT 2026-01-17** |
| `3212c2ad` | OK | OK | - | **REBUILT 2026-01-17** |
| `2bd18e2d` | OK | OK | - | **REBUILT 2026-01-17** |
| `93470a14` | OK | OK | - | **REBUILT 2026-01-17** |
| `d1112d85` | OK | OK | - | **REBUILT 2026-01-17** |
| `10189d08` | OK | OK | - | **REBUILT 2026-01-17** |
| `f4a8987f` | OK | OK | - | **REBUILT 2026-01-17** |
| `bb3a3b66` | - | - | - | SKIPPED (sglang 0.1.11 too old) |

### Key Finding (Updated 2026-01-17 - After Rebuild)

The images in `shikhar481/sglang-images` have been **successfully rebuilt** with all dependencies:

```
93470a14... (FA3 Code)        → sgl_kernel: ✅ INSTALLED, all checks pass
2bd18e2d... (Memory pool)     → sgl_kernel: ✅ INSTALLED, all checks pass
d1112d85... (input_embeds)    → sgl_kernel: ✅ INSTALLED, all checks pass
ddcf9fe3... (Triton attention)→ sgl_kernel: ✅ INSTALLED, all checks pass
79961afa... (FlashInfer fix)  → sgl_kernel: ✅ INSTALLED, all checks pass
3212c2ad... (VLM support)     → sgl_kernel: ✅ INSTALLED, all checks pass
10189d08... (triton fix)      → sgl_kernel: ✅ INSTALLED, all checks pass
f4a8987f... (c087ddd6 parent) → sgl_kernel: ✅ INSTALLED, all checks pass
bb3a3b66... (JSON decoding)   → SKIPPED (sglang 0.1.11 incompatible with torch 2.4)
```

**Verification command used:**
```bash
docker run --rm --gpus all --entrypoint python3 \
    shikhar481/sglang-images:<commit> \
    -c "import sgl_kernel; print('OK')"
```

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

## Category 3: sgl_kernel Not Installed (CRITICAL)

### Symptoms

```
ModuleNotFoundError: No module named 'sgl_kernel'
```

Or ABI mismatch errors:
```
ImportError: /usr/local/.../sgl_kernel/sm90/common_ops.abi3.so: undefined symbol: _ZN3c108ListType3getE...
```

### Affected Commits (Updated 2026-01-17 - FIXED)

| Commit | sgl_kernel Status | Additional Issues | Current Status |
|--------|-------------------|-------------------|----------------|
| `2bd18e2d` | ~~❌ Not installed~~ | ~~Also missing `zmq`~~ | **✅ REBUILT** |
| `bb3a3b66` | ❌ Not installed | - | SKIPPED (sglang 0.1.11 too old) |
| `d1112d85` | ~~❌ Not installed~~ | - | **✅ REBUILT** |
| `93470a14` | ~~❌ Not installed~~ | - | **✅ REBUILT** |
| `ddcf9fe3` | ~~❌ ABI mismatch~~ | ~~Symbol undefined~~ | **✅ REBUILT** |

### Root Cause

1. **sgl_kernel not built** - The Docker build skipped the sgl-kernel installation
2. **ABI mismatch** - sgl_kernel was built against a different PyTorch/CUDA version than what's in the container
3. **Missing dependencies** - Some images are missing `pyzmq` and other required packages

### Detailed Error Analysis

```python
# Error from ddcf9fe3 image:
[sgl_kernel] CRITICAL: Could not load any common_ops library!
- ImportError: .../common_ops.abi3.so: undefined symbol: _ZN3c108ListType3getE...

# This means sgl_kernel was compiled with a different PyTorch C++ ABI
# than the PyTorch installed in the container
```

### Fix When Rebuilding

#### Option A: Full Rebuild with Correct Dependencies (Recommended)

```dockerfile
FROM nvidia/cuda:12.4.1-devel-ubuntu22.04

# 1. System dependencies
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    python3-dev \
    git \
    libnuma-dev \
    && rm -rf /var/lib/apt/lists/*

# 2. Install PyTorch FIRST (determines ABI)
RUN pip install --no-cache-dir \
    torch==2.4.0 \
    --index-url https://download.pytorch.org/whl/cu124

# 3. Install FlashInfer (must match CUDA and torch version)
RUN pip install flashinfer -f https://flashinfer.ai/whl/cu124/torch2.4/

# 4. Clone and install SGLang
ARG COMMIT_HASH
RUN git clone https://github.com/sgl-project/sglang.git /sglang && \
    cd /sglang && \
    git checkout ${COMMIT_HASH} && \
    pip install -e "python[all]"

# 5. Install sgl-kernel FROM SOURCE (critical!)
RUN cd /sglang/sgl-kernel && \
    pip install -e . --no-build-isolation

# 6. Install missing runtime dependencies
RUN pip install --no-cache-dir pyzmq

# 7. Verify sgl_kernel loads
RUN python3 -c "import sgl_kernel; print('sgl_kernel: OK')"

WORKDIR /sglang
```

#### Option B: Patch Existing Image

If sglang and flashinfer work but only sgl_kernel is missing:

```bash
# Start container interactively
docker run -it --gpus all YOUR_IMAGE bash

# Inside container
cd /sglang/sgl-kernel  # or wherever sglang is installed
pip install -e . --no-build-isolation

# Install missing deps
pip install pyzmq

# Test
python3 -c "import sgl_kernel; print('OK')"

# Commit from host
# docker commit CONTAINER_ID new-image:tag
```

### Verification Commands

```bash
IMAGE="shikhar481/sglang-images:YOUR_TAG"

# 1. Check sgl_kernel
docker run --rm --gpus all --entrypoint python3 $IMAGE \
    -c "import sgl_kernel; print('sgl_kernel: OK')"

# 2. Check all critical imports
docker run --rm --gpus all --entrypoint python3 $IMAGE -c "
import sys
print('Python:', sys.version)
try:
    import sgl_kernel
    print('sgl_kernel: OK')
except ImportError as e:
    print(f'sgl_kernel: FAIL - {e}')
try:
    import flashinfer
    print('flashinfer: OK')
except ImportError as e:
    print(f'flashinfer: FAIL - {e}')
try:
    import zmq
    print('zmq: OK')
except ImportError as e:
    print(f'zmq: FAIL - {e}')
try:
    from sglang.srt.server_args import ServerArgs
    print('ServerArgs: OK')
except ImportError as e:
    print(f'ServerArgs: FAIL - {e}')
"
```

### Priority for Rebuild

| Commit | Subject | Model | Claimed Improvement |
|--------|---------|-------|---------------------|
| `93470a14` | Refactor and Optimize FA3 Code | Llama-3.1-8B | FA3 optimization |
| `2bd18e2d` | Memory pool optimization | Llama-2-7b | Memory efficiency |
| `bb3a3b66` | JSON decoding for llava (VLM) | llava-1.5-7b | VLM performance |
| `d1112d85` | input_embeds endpoint | gemma-2-2b | Input embedding speed |

---

## Category 4: Missing Docker Images (Historical)

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

## Category 5: FlashInfer Compatibility (Historical)

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

## Category 6: Missing Python Dependencies

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

## Category 7: Model Download Failures

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

## Category 8: VLM (Vision-Language Model) Support (Historical)

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

### VLM Commits Status

| Commit | PR | Subject | Model | Status |
|--------|-----|---------|-------|--------|
| `3212c2ad` | #6003 | VLM tensor transport (16% faster) | llava-hf/llava-1.5-7b-hf | **✅ REBUILT** |
| `bb3a3b66` | #137 | Faster JSON decoding for llava | llava-hf/llava-1.5-7b-hf | SKIPPED (sglang 0.1.11) |

---

## Complete Rebuild Checklist (CRITICAL)

**Every SGLang Docker image MUST have these components verified before pushing.**

### Required Components Checklist

| Component | Required | Verification Command |
|-----------|----------|---------------------|
| ☐ libnuma-dev | System lib | `ldconfig -p \| grep libnuma` |
| ☐ PyTorch | 2.4.0+ with CUDA | `python3 -c "import torch; print(torch.__version__, torch.cuda.is_available())"` |
| ☐ FlashInfer | Matching CUDA/torch | `python3 -c "import flashinfer; print('OK')"` |
| ☐ **sgl_kernel** | **CRITICAL** | `python3 -c "import sgl_kernel; print('OK')"` |
| ☐ pyzmq | Runtime dep | `python3 -c "import zmq; print('OK')"` |
| ☐ SGLang | From commit | `python3 -c "import sglang; print('OK')"` |
| ☐ ServerArgs | Full install | `python3 -c "from sglang.srt.server_args import ServerArgs; print('OK')"` |

### Common Failures We Encountered

| Failure | Symptom | Root Cause |
|---------|---------|------------|
| `ModuleNotFoundError: sgl_kernel` | sgl_kernel not built | Missing `pip install -e sgl-kernel` step |
| `undefined symbol` in sgl_kernel | ABI mismatch | sgl_kernel built with different PyTorch |
| `ModuleNotFoundError: zmq` | Missing pyzmq | Forgot `pip install pyzmq` |
| `libnuma.so.1 not found` | Missing system lib | Forgot `apt-get install libnuma-dev` |
| `BatchDecodeWithPagedKVCacheWrapper` | FlashInfer mismatch | Wrong flashinfer version for CUDA/torch |

### Dockerfile Template (Corrected)

```dockerfile
FROM nvidia/cuda:12.4.1-devel-ubuntu22.04

# 1. System dependencies (CRITICAL: include libnuma-dev)
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    python3-dev \
    git \
    libnuma-dev \
    libjpeg-dev \
    libpng-dev \
    && rm -rf /var/lib/apt/lists/*

# 2. Install PyTorch FIRST (this determines the C++ ABI for sgl_kernel)
RUN pip install --no-cache-dir \
    torch==2.4.0 \
    --index-url https://download.pytorch.org/whl/cu124

# 3. FlashInfer (MUST match CUDA and torch version exactly)
RUN pip install flashinfer -f https://flashinfer.ai/whl/cu124/torch2.4/

# 4. Clone SGLang
ARG COMMIT_HASH
RUN git clone https://github.com/sgl-project/sglang.git /sglang && \
    cd /sglang && \
    git checkout ${COMMIT_HASH}

# 5. Install SGLang with ALL extras (includes VLM support)
RUN cd /sglang && pip install -e "python[all]"

# 6. CRITICAL: Build sgl_kernel from source (ensures ABI compatibility)
RUN cd /sglang/sgl-kernel && \
    pip install -e . --no-build-isolation

# 7. Install runtime dependencies that might be missing
RUN pip install --no-cache-dir \
    pyzmq \
    pillow \
    torchvision

# 8. VERIFICATION STEP (build will fail if any of these fail)
RUN python3 -c "import sgl_kernel; print('✓ sgl_kernel')" && \
    python3 -c "import flashinfer; print('✓ flashinfer')" && \
    python3 -c "import zmq; print('✓ zmq')" && \
    python3 -c "from sglang.srt.server_args import ServerArgs; print('✓ ServerArgs')" && \
    echo "ALL CHECKS PASSED"

WORKDIR /sglang
CMD ["python3", "-m", "sglang.launch_server", "--help"]
```

### Key Differences from Broken Images

| Step | Broken Images | Fixed Images |
|------|---------------|--------------|
| sgl_kernel | Skipped or pre-built wheel | Built from source with `--no-build-isolation` |
| PyTorch | Installed after sgl_kernel | Installed FIRST (determines ABI) |
| pyzmq | Missing | Explicitly installed |
| Verification | None | Build fails if imports fail |

### Build Script (with Full Verification)

```bash
#!/bin/bash
set -e

COMMIT_HASH=$1
SHORT_HASH=${COMMIT_HASH:0:8}
REPO="shikhar481/sglang-images"

if [ -z "$COMMIT_HASH" ]; then
    echo "Usage: $0 <commit_hash>"
    exit 1
fi

echo "=========================================="
echo "Building SGLang image for commit: $COMMIT_HASH"
echo "=========================================="

# Build
docker build \
    --build-arg COMMIT_HASH=$COMMIT_HASH \
    -t $REPO:$COMMIT_HASH \
    -t $REPO:$SHORT_HASH \
    .

echo ""
echo "=========================================="
echo "VERIFICATION (GPU required)"
echo "=========================================="

# Full verification before push
docker run --rm --gpus all --entrypoint /bin/bash $REPO:$COMMIT_HASH -c '
echo "Testing all required components..."
python3 -c "import sgl_kernel; print(\"✓ sgl_kernel\")" || exit 1
python3 -c "import flashinfer; print(\"✓ flashinfer\")" || exit 1
python3 -c "import zmq; print(\"✓ zmq\")" || exit 1
python3 -c "from sglang.srt.server_args import ServerArgs; print(\"✓ ServerArgs\")" || exit 1
python3 -c "import torch; print(f\"✓ torch {torch.__version__} CUDA={torch.cuda.is_available()}\")" || exit 1
echo ""
echo "ALL VERIFICATION CHECKS PASSED ✓"
'

if [ $? -ne 0 ]; then
    echo "❌ VERIFICATION FAILED - DO NOT PUSH"
    exit 1
fi

echo ""
echo "=========================================="
echo "Pushing to Docker Hub"
echo "=========================================="

# Push both tags
docker push $REPO:$COMMIT_HASH
docker push $REPO:$SHORT_HASH

echo ""
echo "=========================================="
echo "SUCCESS: $REPO:$SHORT_HASH"
echo "=========================================="
```

### One-Line Verification Command

Run this on any image to check if it's properly built:

```bash
docker run --rm --gpus all --entrypoint python3 IMAGE:TAG -c "
import sgl_kernel; print('✓ sgl_kernel')
import flashinfer; print('✓ flashinfer')
import zmq; print('✓ zmq')
from sglang.srt.server_args import ServerArgs; print('✓ ServerArgs')
print('ALL OK')
"
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
| **sgl_kernel missing** | Dockerfile | Add `pip install -e sgl-kernel --no-build-isolation` |
| **sgl_kernel ABI mismatch** | Dockerfile | Install PyTorch FIRST, then build sgl_kernel |
| **pyzmq missing** | Dockerfile | Add `pip install pyzmq` |
| libnuma.so.1 missing | Dockerfile | Add `apt-get install libnuma-dev` |
| Exit 137 (OOM) | Runtime | Stop other containers, free GPU |
| FlashInfer incompatible | Dockerfile | Pin correct version for CUDA/arch |
| Missing image | Build pipeline | Build and push image |
| HF token missing | Runtime | Pass `-e HF_TOKEN=...` |
| Model gated | HuggingFace account | Request access to model |
| VLM not supported | Dockerfile | Use `pip install -e "python[all]"` + vision deps |

---

## Priority Commits to Fix

Based on verified testing (2026-01-17 - Latest):

| Priority | Commit | Issue Found | Status |
|----------|--------|-------------|--------|
| 1 | `93470a14` | sgl_kernel missing | **FIXED & PUSHED** |
| 2 | `2bd18e2d` | sgl_kernel missing + no zmq | **FIXED & PUSHED** |
| 3 | `bb3a3b66` | sgl_kernel missing | SKIPPED (sglang 0.1.11 incompatible) |
| 4 | `d1112d85` | sgl_kernel missing | **FIXED & PUSHED** |
| 5 | `79961afa` | FlashInfer incompatible | **FIXED & PUSHED** |
| 6 | `3212c2ad` | VLM processor missing | **FIXED & PUSHED** |
| 7 | `10189d08` | sgl_kernel missing + triton issue | **FIXED & PUSHED** |

**Note:** All priority commits (except bb3a3b66) have been successfully rebuilt and pushed to `shikhar481/sglang-images` on 2026-01-17.

---

## Images Needing Parent Commits (Tested 2026-01-17)

These images work but cannot be benchmarked because their parent/baseline images are missing:

| Human Commit | Status | Parent Commit Needed | Optimization |
|--------------|--------|---------------------|--------------|
| `c087ddd6` | ✅ All checks pass | `f4a8987f6904e4909adb473c52b443a62ba5a4b5` | MoE align block size kernel |

### c087ddd6 Details

**Human commit:** `c087ddd6865a52634326a05af66429cb5531cd16`
**Parent commit:** `f4a8987f6904e4909adb473c52b443a62ba5a4b5`
**Files changed:**
- `python/sglang/srt/layers/moe/ep_moe/kernels.py`
- `benchmark/kernels/fused_moe_triton/benchmark_ep_pre_reorder_triton.py`

**Verification results:**
```
sgl_kernel: OK
flashinfer: OK
zmq: OK
ServerArgs: OK
```

**Parent image status:** ✅ **REBUILT & PUSHED** on 2026-01-17
```bash
shikhar481/sglang-images:f4a8987f6904e4909adb473c52b443a62ba5a4b5
```

### 10189d08 Details (FIXED)

**Human commit:** `10189d08dde1096f5759316c0a6ff05962714c4b`
**Status:** ✅ **REBUILT & PUSHED** on 2026-01-17

**Verification results (after rebuild):**
```
torch: 2.4.0+cu124 OK
sglang: 0.3.6 OK
flashinfer: OK
zmq: OK
sgl_kernel: INSTALLED
```

### Completed Benchmarks (No Fix Needed)

| Commit | PR | Status |
|--------|-----|--------|
| `021f76e4` | #6994 LoRA stream sync | **COMPLETED** |
| `6fc17596` | #5945 FA3 pad operation | **COMPLETED** |
| `ddcf9fe3` | #3731 Triton attention | **COMPLETED** |

### High Priority: 3212c2ad (VLM 16% Improvement) - FIXED

This commit claims **16% faster VLM inference** (207.7s -> 173.3s).

**Status:** ✅ **REBUILT & PUSHED** on 2026-01-17

**Verification results (after rebuild):**
```
torch: 2.4.0+cu124 OK
sglang: 0.4.9.post4 OK
flashinfer: OK
zmq: OK
sgl_kernel: INSTALLED
```

**Benchmark command:**
```bash
docker run --rm --gpus all -e HF_TOKEN=$HF_TOKEN \
    shikhar481/sglang-images:3212c2ad3f7e4fb473dc807b4b176020a778ed5b \
    python3 -m sglang.bench_serving \
    --backend sglang \
    --model llava-hf/llava-1.5-7b-hf \
    --dataset-name mmmu \
    --num-prompts 50 \
    --request-rate 2
```

---

*Generated: 2026-01-17*
*Updated: 2026-01-17 (Rebuild completed - 8 images rebuilt and pushed to shikhar481/sglang-images)*
