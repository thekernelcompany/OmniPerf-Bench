# SGLang Docker Image Build Status

## Overview

Building Docker images at `shikhar481/sglang-images` for SGLang benchmarking. Images build sgl-kernel FROM SOURCE with git submodules to include the `deep_gemm` module (not available in PyPI versions).

## Current Status (2026-01-13)

| Commit | Type | Model | torch | sgl-kernel | Runtime on H100 | Notes |
|--------|------|-------|-------|------------|-----------------|-------|
| d1112d85 | human | gemma-2-2b | 2.5.1 | ✓ builds | **SEGFAULT** | triton 3.1.0 bug |
| 48efec7b | parent | gemma-2-2b | 2.5.1 | ✓ builds | **SEGFAULT** | triton 3.1.0 bug |
| 93470a14 | human | Llama-3.1-8B | N/A | SKIPPED | N/A | Requires deleted flashinfer fork |
| db452760 | parent | Llama-3.1-8B | N/A | SKIPPED | N/A | Requires deleted flashinfer fork |
| 9c088829 | human | Llama-3.1-8B | 2.6.0 | ✗ missing | untested | FA3 SM90 build failure |
| 005aad32 | parent | Llama-3.1-8B | 2.6.0 | ✗ missing | untested | FA3 SM90 build failure |

---

## EXPERIMENTAL FIX: torch 2.6.0 + triton 3.2.0 (2026-01-13)

### New Experimental Image

**Image:** `shikhar481/sglang-images:d1112d85-torch26-novllm`
**Digest:** sha256:2d8fe1b35ad08080205c4c2863a468853f5adac87566360241b6a51f03772869

### Configuration

| Component | Version |
|-----------|---------|
| torch | 2.6.0+cu124 |
| triton | 3.2.0 |
| sgl-kernel | 0.0.5.post2 |
| torchao | 0.12.0 |
| sglang | 0.4.4.post1 |
| deep_gemm | present |
| vllm | **NOT INSTALLED** |

### Key Changes from Original d1112d85

1. **torch 2.6.0** instead of 2.5.1 - gives us triton 3.2.0
2. **No vllm** - vllm 0.7.2 forces torch 2.5.1 + triton 3.1.0, which segfaults
3. **sgl-kernel 0.0.5.post2 builds successfully** with torch 2.6.0

### Why This Might Work

- sgl-kernel 0.0.5.post2 uses `setup.py` (not CMake with FA3)
- No FA3/SM90 build issues
- triton 3.2.0 paired with torch 2.6.0 may fix the H100 segfault

### Dockerfile

`src/benchmark/docker/sglang_commits/Dockerfile.d1112d85-torch26`

### Test This Image

```bash
docker run --rm --gpus all -p 30000:30000 \
  -e HF_TOKEN=<your_token> \
  shikhar481/sglang-images:d1112d85-torch26-novllm \
  python -m sglang.launch_server --model google/gemma-2-2b-it --port 30000
```

**Expected outcome:** Server should start AND handle inference without triton segfault.

---

## Issue #1: Triton 3.1.0 Segfault on H100 (BLOCKING)

### Symptoms

Server starts but crashes on first inference request:
```
Fatal Python error: Segmentation fault
File "triton/compiler/code_generator.py", line 223 in __init__
File "triton/compiler/code_generator.py", line 1294 in ast_to_ttir
File "triton/compiler/compiler.py", line 113 in make_ir
```

### Tested Triton Versions (all with torch 2.5.1)

| Triton | Result |
|--------|--------|
| 2.3.1 | `AttributeError: module 'triton.language' has no attribute 'cast'` |
| 3.0.0 | **SEGFAULT** in code_generator.py:223 |
| 3.1.0 | **SEGFAULT** in code_generator.py:223 (bundled with torch 2.5.1) |
| 3.2.0 | `TypeError: must be called with a dataclass type or instance` (API incompatible) |

### Root Cause Analysis

The triton 3.1.0 segfault occurs because triton 3.2.0 API is NOT backward compatible with torch 2.5.1.
The only way to use triton 3.2.0 is with torch 2.6.0.

### Solution: torch 2.6.0 + no vllm

- torch 2.6.0 bundles triton 3.2.0 natively
- vllm 0.7.2 CANNOT be used - it forces torch 2.5.1
- sgl-kernel 0.0.5.post2 builds successfully with torch 2.6.0

---

## Issue #2: sgl-kernel Build Failure (9c088829/005aad32)

### Symptoms

sgl-kernel 0.0.9.post2 fails to build with FA3 SM90 errors:
```
error: template parameter 'Is_local' is not a type
```

### Root Cause

FlashAttention 3 code in sgl-kernel 0.0.9.post2 requires SM90 (Hopper) at compile time, but Docker build environment has no GPU.

---

## What Was Tested

### 1. Import Verification (PASSED)

**d1112d85 / 48efec7b:**
```
torch: 2.5.1+cu124
torchao: 0.12.0
transformers: 4.48.3
vllm: 0.7.2
sglang: 0.4.4.post1
sgl_kernel: ✓ present
```

**9c088829 / 005aad32:**
```
torch: 2.6.0+cu124
torchao: 0.12.0
sglang: 0.4.5.post3
sgl_kernel: ✗ MISSING
```

### 2. Runtime on H100 (FAILED)

Tested d1112d85/48efec7b images on H100 (SM90):
- Server starts successfully
- Crashes on first inference with triton segfault
- Tested with: `--disable-cuda-graph`, `--dtype float16`, various env vars
- All configurations crash

### 3. Triton Version Testing

Attempted runtime replacement of triton in d1112d85 container:
- triton 2.3.1: API incompatible
- triton 3.0.0: same segfault
- triton 3.2.0: torch 2.5.1 incompatible

---

## Docker Image Digests

| Commit | Digest |
|--------|--------|
| d1112d85 | sha256:339d024e8f0df4c4f440948ff1a6ca2b77e8b42b6fd8a0516795c51d61d9d559 |
| 48efec7b | sha256:e72e01c337dfbe3112d7a174587766ced2adb4ac5aed331103a2c6d1a3f0084f |
| 9c088829 | sha256:294195edec4f70f1aefd7d3f65da8e401971f11a1a82f5e88e73b15c5a6574f0 |
| 005aad32 | sha256:f8ba68b5a837e2a22747077777d8ff059c07b6ba994a138608cf67169110e05d |

---

## Next Steps to Test

1. **Option A: Try torch 2.6.0 with d1112d85**
   - Modify Dockerfile.d1112d85 to use torch 2.6.0
   - See if sgl-kernel 0.0.5.post2 builds
   - Test if triton 3.2.0 fixes the H100 segfault

2. **Option B: Try official SGLang image**
   - Test `lmsysorg/sglang:latest` on H100
   - See what torch/triton versions they use

---

## Reference: Dependency Versions

From pyproject.toml and pytorch/ao#2919:

| Commit | SGLang | sgl-kernel | torch | torchao | vllm |
|--------|--------|------------|-------|---------|------|
| d1112d85 | 0.4.4.post1 | 0.0.5.post2 | 2.5.1 | 0.12.0 | 0.6.4-0.7.2 |
| 9c088829 | 0.4.5.post3 | 0.0.9.post2 | 2.6.0 | 0.12.0 | not needed |
