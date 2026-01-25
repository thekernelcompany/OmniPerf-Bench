# SGLang Benchmark Session Report - 2026-01-25

## Executive Summary

This session focused on fixing baseline and human benchmarks for all 17 SGLang commits on H100, followed by a critical analysis of agent patch failures.

**Final Results:**
- **Baseline/Human**: 15/17 (88%) successful
- **Best Agent (claude_code)**: 7/17 (41%) successful
- **Worst Agent (trae_gpt5)**: 2/17 (12%) successful

---

## Table of Contents

1. [Initial State](#initial-state)
2. [Implementation Plan](#implementation-plan)
3. [Problems Encountered & Solutions](#problems-encountered--solutions)
4. [Commit Mapping](#commit-mapping)
5. [Complete Results Table](#complete-results-table)
6. [Critical Analysis of Failures](#critical-analysis-of-failures)
7. [Code Changes Made](#code-changes-made)

---

## Initial State

Before this session:
- **7/17 commits (41%)** had successful baseline/human results
- **10 commits** were failing due to various dependency and import issues

### Initial Failure Categories

| Error Category | Affected Commits | Root Cause |
|----------------|------------------|------------|
| sgl_kernel ABI mismatch | 4418f599, 2bd18e2d, 2a413829, 5e023301, 880221bd, ddcf9fe3 | C++ symbol mismatch in `common_ops.abi3.so` |
| Missing deps | 148254d4, 2a754e57 | pyairports/rpyc not installed |
| Server startup failure | 6cb00c63, b1e5a33a | Unknown at the time |

---

## Implementation Plan

### Phase 1: Easy Fixes (HIGH CONFIDENCE)

**1.1 Add setuptools for june_2024 Era**

The june_2024 era uses triton==2.3.0 which requires setuptools.

```python
# Line 772 - added setuptools
extra_deps = ["transformers>=4.40.0", "numpy<2", "pillow", "requests", "tqdm", "rpyc", "setuptools"]
```

**1.2 Force Reinstall Transformers**

vLLM may downgrade transformers, breaking `AutoProcessor` import.

```python
# After vllm install, force reinstall
uv_pip("install", "--reinstall", "transformers>=4.44.0", timeout=120)
```

### Phase 2: sgl_kernel Patch Extension (MEDIUM CONFIDENCE)

**Problem:** The existing `patch_sgl_kernel_import()` only patched `awq.py`, but failures occurred in other files:
- `w8a8_int8.py` - `from sgl_kernel import int8_scaled_mm`
- `fp8_utils.py` - various FP8 ops
- `quantization/__init__.py` - imports configs

**Solution:** Created `patch_sgl_kernel_quantization()` function to wrap all sgl_kernel imports in try/except blocks.

### Phase 3: Server Startup Debugging (LOW CONFIDENCE)

**Changes:**
1. Increased `SERVER_TIMEOUT` from 120s to 300s
2. Added server startup logging to capture actual errors
3. Included server log in error messages

### Phase 4: deep_gemm Patch (Added Mid-Session)

**Problem Discovered:** Commits 205d5cb4 and 1acca3a2 failed with:
```
ModuleNotFoundError: No module named 'deep_gemm.jit'
ModuleNotFoundError: No module named 'deep_gemm.jit_kernels'
```

**Root Cause:** The deep_gemm API changed significantly in sgl-kernel 0.3.7 (switched from Python JIT to C++ JIT). Older commits expect the old interface.

**Solution:** Created `patch_deep_gemm_import()` function that:
1. Handles both top-level and indented imports (inside `if is_cuda():` blocks)
2. Handles multi-line imports with parentheses
3. Properly extracts variable names from `as` aliases

---

## Problems Encountered & Solutions

### Problem 1: sgl_kernel Import Failures

**Symptom:**
```
ImportError: /tmp/sglang-venvs/.../sgl_kernel/sm90/common_ops.abi3.so: undefined symbol: _ZN3c108ListType3get...
```

**Root Cause:** sgl_kernel compiled against different PyTorch ABI than torch 2.5.1.

**Solution:** Patch all files that import from sgl_kernel to make imports optional:
- `awq.py`
- `w8a8_int8.py`
- `fp8_utils.py`
- `quantization/__init__.py`

**Result:** Worked for most commits, but 6cb00c63 and b1e5a33a still fail because the server process itself requires sgl_kernel at startup.

### Problem 2: deep_gemm API Changes

**Symptom:**
```
ModuleNotFoundError: No module named 'deep_gemm.jit.compiler'
```

**Root Cause:** deep_gemm is bundled in sgl-kernel. Version 0.3.7 changed from Python JIT to C++ JIT interface.

**Solution Attempt 1:** Patch only top-level imports (no indentation)
- **Failed:** Imports were inside `if is_cuda():` block (4 spaces indented)

**Solution Attempt 2:** Patch indented imports but not handle multi-line
- **Failed:** Some imports span multiple lines with parentheses

**Solution Attempt 3:** Handle both single and multi-line imports with proper indentation
- **Failed:** `as` aliases not handled correctly (`includes as deep_gemm_includes = None` is invalid syntax)

**Final Solution:** Extract actual variable names from imports with `as` aliases:
```python
def extract_var_names(import_part):
    """Extract variable names, handling 'as' aliases.
    'includes as deep_gemm_includes' -> ['deep_gemm_includes']
    """
    names = []
    for item in import_part.split(','):
        if ' as ' in item:
            names.append(item.split(' as ')[1].strip())
        else:
            names.append(item.strip())
    return names
```

### Problem 3: Syntax Errors in Patched Files

**Symptom:**
```
SyntaxError: 'yield' outside function
```

**Root Cause:** Initial patch function was patching imports INSIDE functions (e.g., context managers with `yield`), breaking the function structure.

**Solution:** Only patch imports that start at the beginning of a line (top-level) or have consistent indentation (inside `if` blocks), not imports inside functions.

### Problem 4: Server Still Crashes After Patches

**Symptom:** For 6cb00c63 and b1e5a33a, even with all patches applied, server dies with:
```
CRITICAL: Could not load any common_ops library!
```

**Root Cause:** These commits have code paths that REQUIRE sgl_kernel's `common_ops` module at server startup - cannot be patched away.

**Resolution:** Accept these 2 commits cannot work without rebuilding sgl_kernel against torch 2.5.1.

---

## Commit Mapping

The 17 commits benchmarked come from the file `src/benchmark/fixes/sglang_commit_mapping.json`:

| # | Commit | PR# | Subject | Era |
|---|--------|-----|---------|-----|
| 1 | c087ddd6 | 6627 | Multi-modal performance optimization | late_2024 |
| 2 | 6b231325 | 6649 | Speculative decoding improvements | late_2024 |
| 3 | dd1012fc | 6764 | KV cache optimization | late_2024 |
| 4 | df7f61ee | 6812 | Attention kernel speedup | late_2024 |
| 5 | e3ec6bf4 | 6814 | Minor speed up block_quant_dequant | late_2024 |
| 6 | da47621c | 7058 | Minor speedup topk postprocessing | late_2024 |
| 7 | a191a0e4 | 6593 | Improve performance of two batch overlap | late_2024 |
| 8 | 31589e17 | 6668 | DeepSeek model optimization | late_2024 |
| 9 | 6cb00c63 | 6761 | [PD] Optimize time out logic for mooncake | late_2024 |
| 10 | 132dad87 | 6922 | Mooncake connection optimization | late_2024 |
| 11 | b1e5a33a | 6960 | Eliminate stream sync for LoRA batch init | late_2024 |
| 12 | 021f76e4 | 6994 | Memory allocation optimization | late_2024 |
| 13 | 2ed68d7a | 7236 | Scheduler optimization | late_2024 |
| 14 | 73b13e69 | 7285 | Batch processing speedup | late_2024 |
| 15 | 187b85b7 | 7393 | Request handling optimization | late_2024 |
| 16 | 205d5cb4 | 6356 | Optimize local attention memory allocation in FlashAttention | late_2024 |
| 17 | 1acca3a2 | 5969 | FA3 speed up: skip len operation | late_2024 |

Each commit has:
- `human_commit`: The actual merged PR commit
- `base_commit`: The commit before the PR (parent)
- Agent patches generated for: claude_code, codex, trae_gpt5, trae_sonnet45

---

## Complete Results Table

```
══════════════════════════════════════════════════════════════════════════════════════════════════════════════════════
                           COMPLETE SGLang BENCHMARK RESULTS (17 Commits × 6 Variants)
══════════════════════════════════════════════════════════════════════════════════════════════════════════════════════

Commit     │ PR#   │   baseline │      human │    claude_code │          codex │      trae_gpt5 │  trae_sonnet45
───────────┼───────┼────────────┼────────────┼────────────────┼────────────────┼────────────────┼───────────────
c087ddd6   │ 6627  │     2675.7 │     2233.7 │         2745.3 │         2266.0 │         2743.6 │         2684.8
6b231325   │ 6649  │     1251.6 │     1275.9 │         1284.6 │         1268.9 │         1268.2 │        TIMEOUT
dd1012fc   │ 6764  │     1187.4 │     1038.4 │         1060.7 │         1043.4 │        TIMEOUT │         1286.2
df7f61ee   │ 6812  │     1277.2 │     1079.0 │         1056.6 │         1056.8 │        TIMEOUT │         1051.4
e3ec6bf4   │ 6814  │     1284.3 │      879.5 │       CONFLICT │       CONFLICT │    EMPTY_PATCH │         1279.3
da47621c   │ 7058  │     1283.0 │     1007.8 │     PATCH_FAIL │     PATCH_FAIL │    EMPTY_PATCH │         1269.8
a191a0e4   │ 6593  │     1282.1 │     1137.8 │     PATCH_FAIL │     PATCH_FAIL │    EMPTY_PATCH │     PATCH_FAIL
31589e17   │ 6668  │     1275.6 │     1279.2 │         1242.7 │     PATCH_FAIL │     PATCH_FAIL │     PATCH_FAIL
6cb00c63   │ 6761  │ ABI_MISMATCH │ ABI_MISMATCH │   ABI_MISMATCH │     PATCH_FAIL │     PATCH_FAIL │   ABI_MISMATCH
132dad87   │ 6922  │     1272.2 │     1015.5 │         1050.0 │     PATCH_FAIL │     PATCH_FAIL │     PATCH_FAIL
b1e5a33a   │ 6960  │ ABI_MISMATCH │ ABI_MISMATCH │     PATCH_FAIL │     PATCH_FAIL │     PATCH_FAIL │   ABI_MISMATCH
021f76e4   │ 6994  │     1295.9 │     1052.6 │     PATCH_FAIL │     PATCH_FAIL │     PATCH_FAIL │     PATCH_FAIL
2ed68d7a   │ 7236  │     1024.2 │     1051.7 │     PATCH_FAIL │     PATCH_FAIL │     PATCH_FAIL │     PATCH_FAIL
73b13e69   │ 7285  │     1037.6 │     1292.1 │     PATCH_FAIL │     PATCH_FAIL │     PATCH_FAIL │     PATCH_FAIL
187b85b7   │ 7393  │     1282.9 │     1039.5 │         1028.5 │     PATCH_FAIL │     PATCH_FAIL │     HTTP_ERROR
205d5cb4   │ 6356  │     1252.4 │     1255.8 │ BUG:zeros→empty │ BUG:zeros→empty │     PATCH_FAIL │ BUG:zeros→empty
1acca3a2   │ 5969  │     1022.5 │     1258.1 │ BUG:zeros→empty │     PATCH_FAIL │     PATCH_FAIL │     PATCH_FAIL
───────────┼───────┼────────────┼────────────┼────────────────┼────────────────┼────────────────┼───────────────
SUCCESS    │       │      15/17 │      15/17 │           7/17 │           4/17 │           2/17 │           5/17
RATE       │       │        88% │        88% │           41% │           24% │           12% │           29%

══════════════════════════════════════════════════════════════════════════════════════════════════════════════════════
```

### Legend

| Code | Meaning |
|------|---------|
| Numbers | Throughput in tokens/second |
| PATCH_FAIL | Git patch failed to apply (context mismatch with base commit) |
| BUG:zeros→empty | Agent replaced `torch.zeros()` with `torch.empty()` causing crashes |
| CONFLICT | Human & agent modified same function differently |
| EMPTY_PATCH | Patch file is 0 bytes |
| TIMEOUT | Benchmark exceeded 600s limit |
| ABI_MISMATCH | sgl_kernel binary incompatible with torch 2.5.1 |
| HTTP_ERROR | Network error during benchmark |

---

## Critical Analysis of Failures

### Category 1: Semantic Bugs (BUG:zeros→empty)

**Affected:** 205d5cb4, 1acca3a2 (claude_code, codex, trae_sonnet45)

**The Bug:**
The agent "optimized" `torch.zeros()` to `torch.empty()` throughout the FlashAttention backend.

```python
# Original (correct)
"cache_seqlens": torch.zeros(max_bs, dtype=torch.int32, device=self.device)

# Agent's "optimization" (WRONG)
"cache_seqlens": torch.empty(max_bs, dtype=torch.int32, device=self.device)
```

**Why It's Wrong:**
- `torch.zeros()`: Allocates AND initializes tensor to all zeros
- `torch.empty()`: Allocates WITHOUT initialization (garbage values)

**Affected Tensors (used as index tables):**
- `cache_seqlens` - sequence length cache
- `cu_seqlens_k` - cumulative sequence lengths
- `page_table` - memory page mappings
- `encoder_page_table`, `encoder_lens_int32`, etc.

**Consequence:**
These tensors are used to index into GPU memory. With garbage values:
- Out-of-bounds memory access
- CUDA segfaults
- Corrupted computation

**Verdict:**
This is a **semantic understanding failure**. The agent:
1. Correctly identified `zeros()` as an optimization target
2. Correctly knew `empty()` is "faster" (no initialization overhead)
3. **FAILED** to understand that zero values are semantically required

This is a classic "optimization that breaks correctness" - optimizing implementation without understanding intent.

### Category 2: Empty/Corrupted Patch Files

**Affected:** trae_gpt5 (multiple commits including e3ec6bf4, da47621c, a191a0e4)

**Root Cause Found:**
Multiple run directories exist per commit. Example for e3ec6bf4:
```
trae/gpt-5/2025-11-14_21-05-32/sglang_071_e3ec6bf4/model_patch.diff  → 0 bytes (EMPTY)
trae/gpt-5/2025-11-16_09-27-51/sglang_071_e3ec6bf4/model_patch.diff  → 2850 bytes (valid)
```

The benchmark finds the empty patch first → "unrecognized input" error.

**Fix Needed:** Update patch finding logic to skip 0-byte files.

### Category 3: Semantic Conflicts (Human & Agent Modified Same Code)

**Affected:** e3ec6bf4 (claude_code, codex)

**What Happened:**
Both human and agent identified the same optimization opportunity in `block_quant_dequant`:

**Human's approach (merged PR):**
```python
# Vectorized with repeat_interleave
x_scale_repeat = x_s.repeat_interleave(block_n, dim=-2).repeat_interleave(block_k, dim=-1)
return (x_q_block.to(torch.float32) * x_scale_repeat).to(dtype)
```

**Agent's approach (patch):**
```python
# Loop optimization with in-place operations
for j in range(n_tiles):
    for i in range(k_tiles):
        x_dq_block[row_start:row_end, col_start:col_end].mul_(x_s[j, i])
```

The patch fails because it expects the original loop structure that the human already replaced.

**Verdict:** This is actually EXPECTED behavior - semantic conflict, not a bug.

### Category 4: ABI Mismatch (Unfixable)

**Affected:** 6cb00c63, b1e5a33a (all variants)

**Error:**
```
CRITICAL: Could not load any common_ops library!
ImportError: .../common_ops.abi3.so: undefined symbol: _ZN3c108ListType3get...
```

**Root Cause:**
sgl_kernel's `common_ops.abi3.so` was compiled against a different PyTorch C++ ABI. The symbol `_ZN3c108ListType3get...` (mangled C++ name for a PyTorch internal function) doesn't exist in torch 2.5.1.

**Resolution:** Cannot fix without rebuilding sgl_kernel from source against torch 2.5.1.

### Category 5: Timeouts (Needs Investigation)

**Affected:** 6b231325/trae_sonnet45, dd1012fc/trae_gpt5, df7f61ee/trae_gpt5

**Possible Causes:**
- Agent patch introduced infinite loop
- Agent patch caused extreme performance regression
- Agent patch caused deadlock in async code

**Status:** Not fully investigated - would need to examine actual patch content.

---

## Code Changes Made

### File: `run_isolated_benchmarks.py`

#### 1. Added setuptools to extra_deps (line ~772)
```python
extra_deps = ["transformers>=4.40.0", "numpy<2", "pillow", "requests", "tqdm", "rpyc", "setuptools"]
```

#### 2. Force reinstall transformers after vllm (line ~776)
```python
uv_pip("install", "--reinstall", "transformers>=4.44.0", timeout=120)
```

#### 3. Increased SERVER_TIMEOUT (line ~103)
```python
SERVER_TIMEOUT = 300  # Was 120
```

#### 4. Added server startup logging (start_sglang_server function)
```python
log_file = venv_path / "server_startup.log"
log_handle = open(log_file, 'w')
proc = subprocess.Popen(..., stderr=log_handle, ...)
```

#### 5. New function: `patch_sgl_kernel_quantization()` (~lines 456-552)
Patches multiple files to make sgl_kernel imports optional:
- `quantization/__init__.py`
- `w8a8_int8.py`
- `fp8_utils.py`

#### 6. New function: `patch_deep_gemm_import()` (~lines 555-668)
Patches `deep_gemm.py` to make deep_gemm imports optional. Features:
- Handles indented imports (inside `if is_cuda():` blocks)
- Handles multi-line imports with parentheses
- Correctly extracts variable names from `as` aliases

#### 7. Updated sgl_kernel handling (~line 888)
```python
patch_sgl_kernel_import()  # Patches awq.py
patch_sgl_kernel_quantization()  # Patches w8a8_int8.py, fp8_utils.py, __init__.py
```

#### 8. Added deep_gemm patching call (~line 992)
```python
patch_deep_gemm_import()
```

#### 9. Enhanced error reporting in run_benchmark()
Server log content now included in error messages when server fails to start.

---

## Summary Statistics

### Before This Session
- Baseline/Human: 7/17 (41%)
- Agent variants: Not systematically tested

### After This Session
| Variant | Success | Rate |
|---------|---------|------|
| baseline | 15/17 | 88% |
| human | 15/17 | 88% |
| claude_code | 7/17 | 41% |
| trae_sonnet45 | 5/17 | 29% |
| codex | 4/17 | 24% |
| trae_gpt5 | 2/17 | 12% |

### Improvement
- Baseline/Human: +8 commits (+47 percentage points)
- Unfixable: 2 commits (6cb00c63, b1e5a33a) due to ABI mismatch

---

## Recommendations

1. **For BUG:zeros→empty failures:**
   - Mark these agent patches as INVALID
   - This represents a fundamental understanding failure by the agent
   - Training data should emphasize tensor initialization semantics

2. **For EMPTY_PATCH failures:**
   - Update `find_agent_patch()` to skip 0-byte files
   - Prefer latest/largest patch when multiple exist

3. **For CONFLICT failures:**
   - These are expected when human and agent target same code
   - Could be interesting to compare performance of different approaches

4. **For ABI_MISMATCH failures:**
   - Would require building sgl_kernel from source
   - Or using a different torch version (risky)

5. **For TIMEOUT failures:**
   - Need deeper investigation of patch content
   - May reveal patterns of problematic "optimizations"

---

## Session Metadata

- **Date:** 2026-01-25
- **GPU:** NVIDIA H100 PCIe (SM90)
- **Platform:** Linux 6.8.0-60-generic
- **Python:** 3.10
- **Torch:** 2.5.1+cu124
- **Branch:** feature/sglang-modal-benchmarks
