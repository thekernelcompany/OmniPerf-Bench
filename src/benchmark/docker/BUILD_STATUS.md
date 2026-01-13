# SGLang Docker Image Build Status

## Overview

Building Docker images at `shikhar481/sglang-images` for SGLang benchmarking. Images build sgl-kernel FROM SOURCE with git submodules to include the `deep_gemm` module (not available in PyPI versions).

## Build Status Summary

| Commit | Type | Model | torch | Build | Runtime | Notes |
|--------|------|-------|-------|-------|---------|-------|
| d1112d85 | human | gemma-2-2b | 2.5.1→2.6.0 | **NEEDS REBUILD** | **H100 BROKEN** | triton 3.1.0 segfault on H100 |
| 48efec7b | parent | gemma-2-2b | 2.5.1→2.6.0 | **NEEDS REBUILD** | **H100 BROKEN** | triton 3.1.0 segfault on H100 |
| 93470a14 | human | Llama-3.1-8B | N/A | **SKIPPED** | N/A | Requires deleted sgl-project/flashinfer fork |
| db452760 | parent | Llama-3.1-8B | N/A | **SKIPPED** | N/A | Requires deleted sgl-project/flashinfer fork |
| 9c088829 | human | Llama-3.1-8B | 2.6.0 | SUCCESS | **MISSING sgl_kernel** | FA3 build failure (SM90 issue) |
| 005aad32 | parent | Llama-3.1-8B | 2.6.0 | SUCCESS | **MISSING sgl_kernel** | FA3 build failure (SM90 issue) |

## CRITICAL FIX #2: Triton 3.1.0 H100 Segfault (2026-01-13)

### Root Cause

torch 2.5.1 bundles triton 3.1.0, which has an MLIR code generator bug that causes segfaults on H100/SM90 GPUs:

```
Fatal Python error: Segmentation fault
File "triton/compiler/code_generator.py", line 223 in __init__
File "triton/compiler/code_generator.py", line 1294 in ast_to_ttir
File "triton/compiler/compiler.py", line 113 in make_ir
```

### Tested Triton Versions

| Triton | torch | Result on H100 |
|--------|-------|----------------|
| 2.3.1 | 2.5.1 | Missing `tl.cast` API - incompatible |
| 3.0.0 | 2.5.1 | **SEGFAULT** in code_generator.py |
| 3.1.0 | 2.5.1 | **SEGFAULT** in code_generator.py |
| 3.2.0 | 2.5.1 | Incompatible (dataclass API change) |
| **3.2.0** | **2.6.0** | **WORKS** - fixes the segfault |

### The Fix

**Upgrade torch from 2.5.1 to 2.6.0** which bundles triton 3.2.0 with the H100 fix.

Updated `Dockerfile.d1112d85`:
- Changed `torch==2.5.1` → `torch==2.6.0`
- Changed flashinfer `torch2.5` → `torch2.6`

---

## Previous Build: torchao 0.12.0 (2026-01-13)

### Images Built (torch 2.5.1 - BROKEN ON H100)

**Docker Image Digests:**

| Commit | Digest |
|--------|--------|
| d1112d85 | sha256:339d024e8f0df4c4f440948ff1a6ca2b77e8b42b6fd8a0516795c51d61d9d559 |
| 48efec7b | sha256:e72e01c337dfbe3112d7a174587766ced2adb4ac5aed331103a2c6d1a3f0084f |
| 9c088829 | sha256:294195edec4f70f1aefd7d3f65da8e401971f11a1a82f5e88e73b15c5a6574f0 |
| 005aad32 | sha256:f8ba68b5a837e2a22747077777d8ff059c07b6ba994a138608cf67169110e05d |

**Note**: These images pass import tests but crash at runtime on H100 due to triton 3.1.0 segfault.

### Verified Configurations

**d1112d85 / 48efec7b (torch 2.5.1):**
```
torch: 2.5.1+cu124
torchao: 0.12.0
transformers: 4.48.3
vllm: 0.7.2
sglang: 0.4.4.post1
ALL IMPORTS SUCCESSFUL!
RUNTIME: SEGFAULT on H100 (triton 3.1.0 bug)
```

**9c088829 / 005aad32 (torch 2.6.0):**
```
torch: 2.6.0+cu124
torchao: 0.12.0
transformers: 4.57.5
sglang: 0.4.5.post3
IMPORTS: FAIL - sgl_kernel missing (FA3 build failure)
```

---

## Correct Configurations (for H100)

| Commit | SGLang | torch | torchao | vllm | transformers | triton |
|--------|--------|-------|---------|------|--------------|--------|
| d1112d85 | 0.4.4.post1 | **2.6.0** | 0.12.0 | 0.6.4-0.7.2 | 4.48.3 | 3.2.0 |
| 48efec7b | 0.4.4.post1 | **2.6.0** | 0.12.0 | 0.6.4-0.7.2 | 4.48.3 | 3.2.0 |
| 9c088829 | 0.4.5.post3 | **2.6.0** | 0.12.0 | Not needed | >=4.40.0 | 3.2.0 |
| 005aad32 | 0.4.5.post3 | **2.6.0** | 0.12.0 | Not needed | >=4.40.0 | 3.2.0 |

**Note**: torch 2.6.0 is required for H100/SM90 because triton 3.2.0 fixes the MLIR segfault bug.

### Dockerfiles Updated

- `Dockerfile.d1112d85`: **torch 2.6.0** (H100 fix), torchao 0.12.0, vllm 0.6.4-0.7.2, transformers 4.48.3, flashinfer torch2.6
- `Dockerfile.9c088829`: torch 2.6.0, torchao 0.12.0, flashinfer torch2.6

### Rebuild Required

Images d1112d85/48efec7b need rebuild with torch 2.6.0 for H100 support.
