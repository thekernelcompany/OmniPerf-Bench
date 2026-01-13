# SGLang Docker Image Build Status

## Overview

Building Docker images at `shikhar481/sglang-images` for SGLang benchmarking. Images build sgl-kernel FROM SOURCE with git submodules to include the `deep_gemm` module (not available in PyPI versions).

## Current Status (2026-01-13)

| Commit | Type | Model | torch | sgl-kernel | Runtime on H100 | Notes |
|--------|------|-------|-------|------------|-----------------|-------|
| d1112d85 | human | gemma-2-2b | 2.5.1 | ✓ builds | **WORKS** | Requires torch_native backend (see below) |
| 48efec7b | parent | gemma-2-2b | 2.5.1 | ✓ builds | **WORKS** | Requires torch_native backend (see below) |
| 93470a14 | human | Llama-3.1-8B | N/A | SKIPPED | N/A | Requires deleted flashinfer fork |
| db452760 | parent | Llama-3.1-8B | N/A | SKIPPED | N/A | Requires deleted flashinfer fork |
| 9c088829 | human | Llama-3.1-8B | 2.6.0 | ✗ missing | untested | FA3 SM90 build failure |
| 005aad32 | parent | Llama-3.1-8B | 2.6.0 | ✗ missing | untested | FA3 SM90 build failure |

---

## 3-WAY BENCHMARK RESULTS (2026-01-13)

### d1112d85 / 48efec7b (gemma-2-2b)

Successfully ran 3-way benchmark using `torch_native` backend workaround.

| Phase | Commit | Request Throughput | Output Throughput | TTFT Mean | TTFT Median | TTFT P99 | ITL Mean | ITL Median | E2E Latency |
|-------|--------|--------------------|-------------------|-----------|-------------|----------|----------|------------|-------------|
| **Baseline** | 48efec7b | 1.20 req/s | 268.21 tok/s | 1118.73 ms | 801.55 ms | 5134.72 ms | 129.22 ms | 112.44 ms | 29818.9 ms |
| **Human** | d1112d85 | 1.18 req/s | 263.96 tok/s | 1159.03 ms | 937.09 ms | 5455.85 ms | 132.26 ms | 115.81 ms | 30537.3 ms |

**Human vs Baseline:**
- Request throughput: -1.7% (worse)
- Output throughput: -1.6% (worse)
- Input throughput: -1.6% (worse)
- TTFT mean: +3.6% (worse)
- TTFT median: +16.9% (worse)
- ITL mean: +2.4% (worse)
- E2E latency: +2.4% (worse)

**Benchmark Configuration:**
- Model: `google/gemma-2-2b`
- Prompts: 100
- Backend: `torch_native` (not flashinfer - required workaround)
- Results saved to: `/ephemeral/omniperf_results_3way_sglang_local/d1112d85/`

**Important Notes:**
- Using `torch_native` backend instead of `flashinfer` (required to avoid triton segfault on H100)
- Results may differ significantly from PR author's original testing which likely used flashinfer backend
- The human commit (d1112d85) shows slightly worse performance than baseline in this configuration
- This could be due to the torch_native backend not benefiting from the optimization, or measurement variance

**PR Investigation (PR #2797):**
- Commit d1112d85 corresponds to SGLang PR #2797
- PR author did not specify which GPU was used for testing
- PR used `google/gemma-2-2b` model (same as our benchmark)
- The optimization may only show benefits with flashinfer backend (not torch_native)
- A100 GPUs (SM80) likely work with flashinfer without the triton segfault issue

---

## WORKING CONFIGURATION FOUND! (2026-01-13)

### Solution

SGLang commit d1112d85 **CAN run on H100** with specific flags that avoid triton JIT compilation:

```bash
docker run --rm --gpus all \
  -e HF_TOKEN="$HF_TOKEN" \
  -e TORCH_COMPILE_DISABLE=1 \
  -e TORCHDYNAMO_DISABLE=1 \
  -p 30000:30000 \
  shikhar481/sglang-images:d1112d8548eb13c842900b3a8d622345f9737759 \
  python -m sglang.launch_server \
    --model google/gemma-2-2b-it \
    --port 30000 \
    --dtype float16 \
    --attention-backend torch_native \
    --sampling-backend pytorch \
    --disable-radix-cache \
    --disable-cuda-graph \
    --host 0.0.0.0
```

### Key Flags

| Flag | Purpose |
|------|---------|
| `--dtype float16` | Avoid BFloat16 (sgl_kernel doesn't support it) |
| `--attention-backend torch_native` | Avoid flashinfer (triggers triton JIT segfault) |
| `--sampling-backend pytorch` | Avoid flashinfer sampling |
| `--disable-cuda-graph` | Avoid CUDA graph capture issues |
| `--disable-radix-cache` | Additional stability |
| `TORCH_COMPILE_DISABLE=1` | Disable torch.compile |
| `TORCHDYNAMO_DISABLE=1` | Disable torch dynamo |

### Test Result

```json
{"text":" \n\nI am trying to create a simple website using HTML, CSS, and",
 "meta_info":{"finish_reason":{"type":"length","length":16},
              "prompt_tokens":2,"completion_tokens":16,
              "e2e_latency":1.248}}
```

### Root Cause

The triton segfault was caused by **flashinfer** backend which triggers triton JIT compilation. The triton 3.1.0 compiler has MLIR bugs that crash on H100. Using `torch_native` backends avoids triton entirely.

---

## PREVIOUS ANALYSIS (for reference)

### What We Tested

| Configuration | Result |
|---------------|--------|
| torch 2.5.1 + triton 3.1.0 | SEGFAULT in `code_generator.py:223` |
| torch 2.5.1 + triton 3.0.0 | SEGFAULT (same location) |
| torch 2.5.1 + triton 2.3.1 | API incompatible (`tl.cast` missing) |
| torch 2.5.1 + triton 3.2.0 | API incompatible with torch 2.5.1 |
| torch 2.6.0 + triton 3.2.0 | SEGFAULT (different location, same root cause) |
| torch 2.6.0 + vllm 0.8.0 + triton 3.2.0 | SEGFAULT |

### Root Cause

1. **Triton MLIR bug** (triton-lang/triton#3882): Triton 3.0.0-3.2.0 have threading bugs in the MLIR code generator that cause segfaults on H100 (SM90)
2. **SGLang 0.4.4.post1 kernel code** triggers this bug during JIT compilation
3. **Only triton 3.5.1+** (with torch 2.9.1+) fixes the issue, but that requires SGLang 0.5.7+ which is a different codebase

### Official SGLang Image Works

We verified `lmsysorg/sglang:latest` runs fine on H100:
- torch: 2.9.1+cu129
- triton: 3.5.1
- sglang: 0.5.7

But this doesn't help benchmark commit d1112d85 since it's a completely different version.

### Hardware Compatibility

| GPU | Architecture | d1112d85 Compatible? | Notes |
|-----|--------------|---------------------|-------|
| H100 | SM90 (Hopper) | **YES** (with workaround) | Requires torch_native backend |
| A100 | SM80 (Ampere) | Likely YES | flashinfer should work natively |
| A10/A30 | SM80 (Ampere) | Likely YES | flashinfer should work natively |

### Decision (UPDATED)

~~**SKIP d1112d85/48efec7b** for H100 3-way benchmarking.~~

**RESOLVED**: Successfully ran 3-way benchmark on H100 using `torch_native` backend workaround. See benchmark results above.

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

## BUGS FOUND AND FIXED (2026-01-13)

### Bug #1: Benchmark Script sed Command Corruption

**Location:** `scripts/runners/local_docker_sglang_benchmark.py`

**Symptom:** Benchmark failed with Python SyntaxError:
```
File "/opt/sglang/python/sglang/srt/layers/quantization/fp8_kernel.py", line 39
    try:\n    import deep_gemm\nexcept ImportError:\n    deep_gemm = None
         ^
SyntaxError: unexpected character after line continuation character
```

**Root Cause:** The sed command was using `\\n` which inserted literal backslash-n characters instead of actual newlines:
```bash
# BROKEN - inserts literal \n characters
sed -i 's/import deep_gemm/try:\\n    import deep_gemm\\nexcept ImportError:\\n    deep_gemm = None/' "$FP8_FILE"
```

**Fix:** Replaced sed with Python-based patching that:
1. First checks if `deep_gemm` is already available (skips patching if so)
2. Uses Python string operations with `chr(10)` for proper newline insertion

**Commit:** `c47fd6bb` - fix(sglang): Fix benchmark script for H100 triton workaround

### Bug #2: Missing H100 Workaround Flags

**Location:** `scripts/runners/local_docker_sglang_benchmark.py`

**Symptom:** Server crashed during startup due to triton JIT segfault

**Root Cause:** Benchmark script was not using the H100 workaround flags discovered during manual testing

**Fix:** Added the following flags to the benchmark script:
- `--dtype float16`
- `--attention-backend torch_native`
- `--sampling-backend pytorch`
- `--disable-cuda-graph`
- `--disable-radix-cache`
- `TORCH_COMPILE_DISABLE=1`
- `TORCHDYNAMO_DISABLE=1`

---

## Next Steps (UPDATED)

### Completed
- [x] d1112d85/48efec7b benchmark on H100 (with torch_native workaround)
- [x] Fixed benchmark script bugs

### Pending
1. **Test newer SGLang commits** from ayushnangia/vllm-docker-build (torch 2.6.0+, triton 3.2.0+)
   - These may work with flashinfer backend (not requiring workaround)
   - Commits: 1acca3a2, 021f76e4, 136c6e04, 3212c2ad

2. **Test on A100** (if available)
   - d1112d85/48efec7b should work with flashinfer backend natively
   - Would provide comparison to H100 torch_native results

3. **Run agent phase benchmarks**
   - Current results only include baseline and human phases
   - Agent patches need to be prepared and tested

---

## SGLang Commits from ayushnangia/vllm-docker-build (2026-01-13)

Built from [ayushnangia/vllm-docker-build](https://github.com/ayushnangia/vllm-docker-build/tree/revolution/fixed-dockerfiles) - SGLang commits with sgl-kernel from PyPI and flashinfer built from source.

### Build Approach

| Aspect | Original Approach | These Builds |
|--------|-------------------|--------------|
| sgl-kernel | Build from source | **PyPI wheel** |
| flashinfer | Pre-built wheel | **Build from source** |
| Python | 3.11 | 3.10 |

### All 8 Commits - BUILT AND PUSHED

| Date | Commit | Type | SGLang | torch | triton | sgl-kernel | Image Tag |
|------|--------|------|--------|-------|--------|------------|-----------|
| 2025-05-02 | 1acca3a2 | human | 0.4.6.post2 | 2.6.0 (cu124) | 3.2.0 | 0.1.1 | `1acca3a2-vllm-style` |
| 2025-05-02 | 6ea1e6ac | parent | 0.4.6.post2 | 2.6.0 (cu124) | 3.2.0 | 0.1.1 | `6ea1e6ac-vllm-style` |
| 2025-06-11 | 021f76e4 | human | **0.4.7** | 2.7.1 (cu126) | **3.3.1** | 0.1.7 | `021f76e4` |
| 2025-06-11 | 777688b8 | parent | **0.4.7** | 2.7.1 (cu126) | **3.3.1** | 0.1.7 | `777688b8` |
| 2025-07-08 | 136c6e04 | human | 0.4.9 | 2.7.1 (cu126) | **3.3.1** | 0.2.4 | `136c6e04-vllm-style` |
| 2025-07-08 | a37e1247 | parent | 0.4.9 | 2.7.1 (cu126) | **3.3.1** | 0.2.4 | `a37e1247-vllm-style` |
| 2025-07-26 | 3212c2ad | human | **0.4.9.post4** | 2.7.1 (cu126) | **3.3.1** | 0.2.7 | `3212c2ad` |
| 2025-07-26 | 53475674 | parent | **0.4.9.post4** | 2.7.1 (cu126) | **3.3.1** | 0.2.7 | `53475674` |

### Image Digests

| Commit | Image Tag | Digest |
|--------|-----------|--------|
| 1acca3a2 | `1acca3a2-vllm-style` | sha256:4568bda8298a3bafbaa4551e641cff4c2bde3e60f73593e8e4af89d26fb33c8d |
| 6ea1e6ac | `6ea1e6ac-vllm-style` | sha256:dbc22924b940275260e907505ef3bba700378cf80688bfbf3a9ef89785c729ab |
| 021f76e4 | `021f76e4` | sha256:f8ff5063f2e57fd599838ba3923eea062c54808502b2ee0e4a76ed5d7208712c |
| 777688b8 | `777688b8` | sha256:9ec6deb8d1500fba18e15297300ea6cbbd6ed9a4992615d25951d92f30b0b9a3 |
| 136c6e04 | `136c6e04-vllm-style` | sha256:3a1e3e253874323fe62890a91ade5cc4ac6c3c1bd37cffea32e9e3fec504e6fe |
| a37e1247 | `a37e1247-vllm-style` | sha256:8f264d1bfebc21ab657278682e8cc11e3f749087e98945da6164ffedf0e2f517 |
| 3212c2ad | `3212c2ad` | sha256:08ee30abd1f60d372f4a262bc7b5c58855402e7d3f032dc0dfb37f1d348112df |
| 53475674 | `53475674` | sha256:f9a9c02c91192223f73707127d50fd7d4bce5910d09ae8f8d9b990423f8ffd55 |

### Test Commands

**torch 2.6.0 + triton 3.2.0 (proper CUDA 12.4 match):**
```bash
docker run --rm --gpus all -p 30000:30000 -e HF_TOKEN=<token> \
  shikhar481/sglang-images:1acca3a2-vllm-style \
  python -m sglang.launch_server --model google/gemma-2-2b-it --port 30000
```

**torch 2.7.1 + triton 3.3.1 (newer triton):**
```bash
docker run --rm --gpus all -p 30000:30000 -e HF_TOKEN=<token> \
  shikhar481/sglang-images:136c6e04-vllm-style \
  python -m sglang.launch_server --model meta-llama/Llama-3.1-8B-Instruct --port 30000
```

### CUDA Version Note

- **torch 2.6.0**: Has cu124 wheels - **matches** CUDA 12.4 base image
- **torch 2.7.1**: Only has cu126 wheels - slight mismatch with base image (12.4 vs 12.6)

### Dockerfiles

```
src/benchmark/docker/vllm_commits/
├── Dockerfile.1acca3a2   # 2025-05-02, torch 2.6.0, SGLang 0.4.6.post2
├── Dockerfile.6ea1e6ac   # 2025-05-02, torch 2.6.0, SGLang 0.4.6.post2 (parent)
├── Dockerfile.021f76e4   # 2025-06-11, torch 2.7.1, SGLang 0.4.7
├── Dockerfile.777688b8   # 2025-06-11, torch 2.7.1, SGLang 0.4.7 (parent)
├── Dockerfile.136c6e04   # 2025-07-08, torch 2.7.1, SGLang 0.4.9
├── Dockerfile.a37e1247   # 2025-07-08, torch 2.7.1, SGLang 0.4.9 (parent)
├── Dockerfile.3212c2ad   # 2025-07-26, torch 2.7.1, SGLang 0.4.9.post4
└── Dockerfile.53475674   # 2025-07-26, torch 2.7.1, SGLang 0.4.9.post4 (parent)
```

---

## Reference: All SGLang Versions

| Commit | SGLang | sgl-kernel | torch | triton | flashinfer |
|--------|--------|------------|-------|--------|------------|
| d1112d85 | 0.4.4.post1 | 0.0.5.post2 | 2.5.1 | 3.1.0 | wheel |
| 1acca3a2 | 0.4.6.post2 | 0.1.1 | 2.6.0 | 3.2.0 | 0.2.5 (source) |
| 021f76e4 | 0.4.7 | 0.1.7 | 2.7.1 | 3.3.1 | 0.2.6.post1 (source) |
| 136c6e04 | 0.4.9 | 0.2.4 | 2.7.1 | 3.3.1 | 0.2.7.post1 (source) |
| 3212c2ad | 0.4.9.post4 | 0.2.7 | 2.7.1 | 3.3.1 | 0.2.9rc1 (source) |

---

## Changelog

### 2026-01-13 (Session 2)

**Benchmark Testing and Bug Fixes**

1. **Debugged benchmark script failures**
   - Initial benchmark runs failed with "Server crashed during startup"
   - Manual docker tests worked fine with same configuration
   - Root cause: sed command in benchmark script corrupting Python files

2. **Fixed sed command bug**
   - Old code: `sed -i 's/import deep_gemm/try:\\n...'` (inserts literal `\n`)
   - New code: Python-based patching with `chr(10)` for proper newlines
   - Also added check to skip patching if `deep_gemm` already available

3. **Fixed missing H100 workaround flags in benchmark script**
   - Added `--dtype float16`, `--attention-backend torch_native`, etc.
   - Added environment variables `TORCH_COMPILE_DISABLE=1`, `TORCHDYNAMO_DISABLE=1`

4. **Successfully ran 3-way benchmark**
   - Baseline (48efec7b): 268.21 tok/s, TTFT 1118.73ms
   - Human (d1112d85): 263.96 tok/s, TTFT 1159.03ms
   - Human shows -1.6% throughput vs baseline (with torch_native backend)

5. **Commits pushed**
   - `c47fd6bb` - fix(sglang): Fix benchmark script for H100 triton workaround

### 2026-01-13 (Session 1)

**H100 Workaround Discovery**

1. **Identified triton segfault root cause**
   - flashinfer backend triggers triton JIT compilation
   - triton 3.0.0-3.2.0 have MLIR threading bugs (triton-lang/triton#3882)
   - Crashes occur during `ast_to_ttir` in code_generator.py

2. **Found working configuration**
   - Use `torch_native` backend instead of flashinfer
   - This bypasses triton JIT entirely
   - Server starts and handles inference successfully

3. **Tested various configurations**
   - torch 2.5.1 + triton 3.0.0/3.1.0: SEGFAULT
   - torch 2.5.1 + triton 2.3.1: API incompatible
   - torch 2.5.1 + triton 3.2.0: API incompatible
   - torch 2.6.0 + triton 3.2.0 + vllm 0.8.0: SEGFAULT (still uses flashinfer)
   - torch_native backend: WORKS

4. **Verified official SGLang image**
   - `lmsysorg/sglang:latest` works on H100
   - Uses torch 2.9.1 + triton 3.5.1 (much newer versions)

### Earlier Sessions

- Built Docker images for d1112d85, 48efec7b, 9c088829, 005aad32
- Discovered sgl-kernel build failures for newer commits (FA3 SM90 issues)
- Built images from ayushnangia/vllm-docker-build with alternative approach
