# SGLang Docker Images - H100 Compatibility Report

**Date:** 2026-01-25
**GPU:** NVIDIA H100 PCIe (SM90, Compute Capability 9.0)
**CUDA Version:** 12.4.1

## Summary

The commit-specific SGLang Docker images from `shikhar481/sglang-images` and `ayushnangia16/nvidia-sglang-docker` have **critical compatibility issues** on H100 GPUs due to Triton kernel compilation failures and ABI mismatches.

**Only the official SGLang image works:** `lmsysorg/sglang:v0.4.6.post5-cu124`

## Working Configuration

```
Image: lmsysorg/sglang:v0.4.6.post5-cu124
Model: meta-llama/Llama-3.1-8B-Instruct
GPU: NVIDIA H100 PCIe (SM90)
PyTorch: 2.6.0+cu124
Triton: 3.2.0

Results:
- Request throughput: 49.05 req/s
- Output throughput: 1512.21 tok/s
- Input throughput: 3014.12 tok/s
- TTFT mean: 529.33 ms
- ITL mean: 32.24 ms
```

## Non-Working Images (H100 Incompatible)

### 1. shikhar481/sglang-images:c087ddd6

**Issue:** Triton kernel segfault during JIT compilation

```
Fatal Python error: Segmentation fault
  File "/usr/local/lib/python3.11/dist-packages/triton/compiler/code_generator.py", line 203 in __init__
  ...
  File "/opt/sglang/python/sglang/srt/managers/schedule_batch.py", line 1263 in prepare_for_extend
```

The server starts and loads the model, but crashes on the first inference request when Triton attempts to compile kernels for SM90 (H100).

### 2. shikhar481/sglang-images:v04x-triton-*

**Issue:** Missing dependencies

```
ModuleNotFoundError: No module named 'aiohttp'
```

These images are missing the aiohttp package required for SGLang server.

### 3. shikhar481/sglang-images:v04x-fixed-*

**Issue:** Triton API incompatibility

```
ImportError: cannot import name 'default_cache_dir' from 'triton.runtime.cache'
```

The SGLang code uses a Triton API that doesn't exist in the installed Triton version.

### 4. shikhar481/sglang-images:v05-improved-*

**Issue:** Missing sgl_kernel module

```
ModuleNotFoundError: No module named 'sgl_kernel'
```

### 5. ayushnangia16/nvidia-sglang-docker:*

**Issue:** sgl_kernel ABI mismatch

```
ImportError: undefined symbol: _ZN3c108ListType3getERKNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEENS_4Type24SingletonOrSharedTypePtrIS9_EE
```

The sgl_kernel module was compiled against a different PyTorch/libtorch version.

## Attempted Fixes

1. **Upgrading Triton 3.2.0 -> 3.6.0:** Breaks SGLang imports (API changes)
2. **Downgrading Triton 3.2.0 -> 3.1.0:** Same segfault issue
3. **Installing libnuma-dev:** Doesn't fix the Triton kernel issue
4. **Upgrading sgl_kernel:** Creates API incompatibilities
5. **Disabling CUDA graphs:** Moves segfault to different location

## Root Cause

The Triton kernels in these images were likely built and tested on A100 (SM80) or older GPUs. When running on H100 (SM90), the Triton JIT compiler crashes during kernel compilation with a segmentation fault.

This is a known issue with certain Triton versions when targeting newer GPU architectures that weren't available during the original build.

## Recommendations

1. **For H100 users:** Use the official SGLang image `lmsysorg/sglang:v0.4.6.post5-cu124`

2. **For benchmarking specific commits:** Requires rebuilding Docker images with:
   - PyTorch 2.6.0+ with CUDA 12.4
   - Triton 3.2.0 built with SM90 support
   - sgl_kernel compiled against the same PyTorch version

3. **Alternative approach:** Use Python code overlay on the official image for performance benchmarking (may not capture all optimizations if they involve compiled extensions)

## Images Tested

| Image | SGLang Version | Status | Issue |
|-------|---------------|--------|-------|
| lmsysorg/sglang:v0.4.6.post5-cu124 | 0.4.6.post5 | **WORKS** | - |
| shikhar481/sglang-images:c087ddd6 | 0.4.6.post5 | FAILS | Triton segfault |
| shikhar481/sglang-images:v04x-triton-148254d4db8b | 0.4.1.post3 | FAILS | Missing aiohttp |
| shikhar481/sglang-images:v04x-fixed-dd1012fcbe2a | 0.4.6.post5 | FAILS | Triton API mismatch |
| shikhar481/sglang-images:v05-improved-148254d4db8b | 0.4.1.post3 | FAILS | Missing sgl_kernel |
| ayushnangia16/nvidia-sglang-docker:187b85b7f38... | 0.4.7.post1 | FAILS | sgl_kernel ABI |
| ayushnangia16/nvidia-sglang-docker:6b231325b97... | 0.4.6.post5 | FAILS | transformers issue |
| ayushnangia16/nvidia-sglang-docker:148254d4db8... | Various | FAILS | sgl_kernel ABI |

## Benchmark Results (Official Image)

```json
{
  "model": "meta-llama/Llama-3.1-8B-Instruct",
  "gpu": "NVIDIA H100 PCIe",
  "request_throughput": 49.05,
  "output_throughput": 1512.21,
  "input_throughput": 3014.12,
  "ttft_mean": 529.33,
  "ttft_median": 402.55,
  "itl_mean": 32.24,
  "itl_median": 15.37,
  "e2e_latency_mean": 1491.19
}
```
