# SGLang Docker Image Build Status

## Overview

Building Docker images at `shikhar481/sglang-images` for SGLang benchmarking. Images build sgl-kernel FROM SOURCE with git submodules to include the `deep_gemm` module (not available in PyPI versions).

---

## ✅ VALIDATED 3-WAY BENCHMARK RESULTS (2026-01-14)

**Methodology: CORRECT** - Used exact models and perf_commands from HuggingFace dataset `Ayushnangia/omniperf_v1`.

### Configuration Summary

| Commit | Optimization | Model | Backend | Special Config |
|--------|--------------|-------|---------|----------------|
| **1acca3a2** | FA3 (FlashAttention 3) | `meta-llama/Llama-3.1-8B-Instruct` | flashinfer | - |
| **021f76e4** | LoRA | `meta-llama/Llama-3.1-8B-Instruct` | flashinfer | LoRA adapter + `--disable-radix-cache` |
| **3212c2ad** | Tensor Transport | `OpenGVLab/InternVL2_5-8B` | flashinfer | VLM model |
| **a37e1247** | pybase64 | `Qwen/Qwen2.5-VL-7B-Instruct` | flashinfer | VLM + MMMU dataset |

### Results: Output Throughput (tok/s)

| Commit | Optimization | Baseline | Human | Agent | Human Δ | Agent Δ |
|--------|--------------|----------|-------|-------|---------|---------|
| **1acca3a2** | FA3 | 1083.7 | 1085.6 | 1081.4 | +0.2% | -0.2% |
| **021f76e4** | LoRA | 1058.9 | 1131.7 | 1055.9 | **+6.9%** ✅ | -0.3% |
| **3212c2ad** | Tensor Transport | 3237.3 | 2936.0 | 3119.5 | **-9.3%** ❌ | -3.6% |
| **a37e1247** | pybase64 | 3854.8 | 3856.0 | 3857.0 | +0.0% | +0.1% |

### Results: TTFT Latency (ms, lower is better)

| Commit | Baseline | Human | Agent | Human Δ | Agent Δ |
|--------|----------|-------|-------|---------|---------|
| **1acca3a2** | 917.8 | 938.4 | 992.7 | +2.2% | +8.2% |
| **021f76e4** | 111.1 | 95.3 | 112.6 | **-14.1%** ✅ | +1.4% |
| **3212c2ad** | 7685.5 | 8248.1 | 7641.8 | +7.3% ❌ | **-0.6%** ✅ |
| **a37e1247** | 53.7 | 51.5 | 51.0 | **-4.1%** ✅ | **-5.0%** ✅ |

### Key Findings

1. **021f76e4 (LoRA)**: Human shows clear improvement (+6.9% throughput, -14.1% TTFT). Agent failed to replicate.
2. **3212c2ad (VLM Tensor Transport)**: Human commit **regressed** performance (-9.3%). Agent outperformed human.
3. **a37e1247 (pybase64)**: Agent slightly outperforms human (-5.0% vs -4.1% TTFT improvement).
4. **1acca3a2 (FA3)**: All results within noise range (~0.2%).

### Bugs Fixed During Benchmarking

1. **Empty `lora_args` creating invalid bash** - Restructured script
2. **Double backslash `\\\\` in f-strings** - Simplified string handling
3. **LoRA requires `--disable-radix-cache`** - Added flag
4. **LoRA adapter naming mismatch** - Used `lora=<path>` format
5. **CUDA OOM from zombie containers** - Added cleanup before each phase

### Results Location

```
/root/sglang-images/OmniPerf-Bench/results/sglang_v2/
├── 1acca3a2/
│   ├── baseline_result.json
│   ├── human_result.json
│   └── agent_result.json
├── 021f76e4/
│   ├── baseline_result.json
│   ├── human_result.json
│   └── agent_result.json
├── 3212c2ad/
│   ├── baseline_result.json
│   ├── human_result.json
│   └── agent_result.json
└── a37e1247/
    ├── baseline_result.json
    ├── human_result.json
    └── agent_result.json
```

---

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

## ⚠️ CRITICAL: BENCHMARK METHODOLOGY ISSUES (2026-01-14)

### Executive Summary

**Our benchmark results are INVALID.** We used wrong models and wrong benchmark commands for all commits.

### What Went Wrong

The Docker images are **correct** - they contain SGLang code at the right commits. However:
1. **Models are NOT baked into images** - they're specified at runtime in the benchmark command
2. **We used a generic benchmark command** with `gemma-2-2b` for all commits
3. **The dataset specifies DIFFERENT models and commands** for each commit

### Dataset vs Our Benchmark Comparison

| Commit | What We Used | What Dataset Specifies | Why Our Results Are Invalid |
|--------|--------------|------------------------|----------------------------|
| **021f76e4** | `gemma-2-2b`, no LoRA | `Llama-3.1-8B-Instruct` + **LoRA adapter** | LoRA optimization has NO effect without LoRA |
| **1acca3a2** | `gemma-2-2b`, torch_native | `Llama-3.1-8B-Instruct`, **FA3 backend** | FA3 optimization needs FlashAttention 3 |
| **3212c2ad** | `gemma-2-2b` (text-only) | `OpenGVLab/InternVL2_5-8B` (**VLM**) | VLM optimization needs Vision-Language Model |
| **a37e1247** | `gemma-2-2b` (text-only) | `Qwen/Qwen2.5-VL-7B-Instruct` + **MMMU dataset** | Multimodal optimization needs VLM + multimodal data |

### Correct perf_commands from HuggingFace Dataset

Source: `https://huggingface.co/datasets/Ayushnangia/omniperf_v1`

```bash
# 021f76e4 - LoRA optimization (REQUIRES LoRA adapter configured!)
python3 -m sglang.bench_serving --backend sglang \
  --model meta-llama/Llama-3.1-8B-Instruct \
  --num-prompt 480 --request-rate 8 --lora-name lora

# 1acca3a2 - FA3 (FlashAttention 3) speed optimization
python -m sglang.bench_serving --backend sglang \
  --model meta-llama/Llama-3.1-8B-Instruct --num-prompts 100

# 3212c2ad - VLM tensor transport optimization
python3 -m sglang.bench_serving --backend sglang \
  --model OpenGVLab/InternVL2_5-8B

# a37e1247 - Multimodal pybase64 optimization
python3 -m sglang.bench_serving --backend sglang \
  --model Qwen/Qwen2.5-VL-7B-Instruct \
  --dataset-name mmmu --request-rate 10 --num-prompts 100
```

### Why Results Appeared Wrong

| Commit | Our Result | Explanation |
|--------|------------|-------------|
| **021f76e4** | ~0% change | LoRA optimization has NO effect without LoRA adapter |
| **a37e1247** | -6.3% regression | Multimodal (base64) optimization may hurt text-only workloads |
| **1acca3a2** | +4.2% improvement | Some general speedup that also helps text (lucky accident) |
| **3212c2ad** | +0.3% improvement | VLM tensor optimization shouldn't significantly affect text |

### Key Insight: Images vs Runtime Configuration

```
┌─────────────────────────────────────────────────────────────────────┐
│                    DOCKER IMAGE CONTAINS:                           │
├─────────────────────────────────────────────────────────────────────┤
│ ✓ SGLang code at specific commit                                    │
│ ✓ Dependencies (torch, triton, flashinfer/sgl-kernel)               │
│ ✓ Python environment                                                │
│ ✗ NOT model weights (downloaded at runtime)                         │
│ ✗ NOT benchmark configuration                                       │
└─────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────┐
│                 SPECIFIED AT RUNTIME (benchmark cmd):               │
├─────────────────────────────────────────────────────────────────────┤
│ • Model to use (--model)                                            │
│ • Backend (flashinfer/torch_native)                                 │
│ • LoRA adapters (--lora-name)                                       │
│ • Dataset (--dataset-name)                                          │
│ • Request parameters (--num-prompts, --request-rate)                │
└─────────────────────────────────────────────────────────────────────┘
```

### What Needs To Be Done

1. **Re-run benchmarks** with correct perf_commands from dataset
2. **Download correct models**: Llama-3.1-8B, InternVL2_5-8B, Qwen2.5-VL-7B
3. **Configure LoRA** for 021f76e4 benchmark
4. **Use flashinfer backend** (not torch_native workaround) where possible
5. **Use multimodal datasets** (MMMU) for VLM commits

### Docker Images Status

The images at `shikhar481/sglang-images` are **VALID** and ready to use:

| Image Tag | Status | Can Re-run With Correct Command |
|-----------|--------|--------------------------------|
| `1acca3a2-vllm-style` | ✅ Working | Yes - need Llama-3.1-8B + flashinfer |
| `6ea1e6ac-vllm-style` | ✅ Working | Yes - need Llama-3.1-8B + flashinfer |
| `021f76e4` | ✅ Working | Yes - need Llama-3.1-8B + LoRA |
| `777688b8` | ✅ Working | Yes - need Llama-3.1-8B + LoRA |
| `a37e1247-vllm-style` | ✅ Working | Yes - need Qwen2.5-VL + MMMU |
| `136c6e04-vllm-style` | ✅ Working | Yes - need Qwen2.5-VL + MMMU |
| `3212c2ad` | ✅ Working | Yes - need InternVL2_5-8B |
| `53475674` | ✅ Working | Yes - need InternVL2_5-8B |

---

## ⚠️ INVALID BENCHMARK RESULTS (For Reference Only)

The following results used **wrong models and commands**. They are preserved for reference but should NOT be used for performance analysis.

### vLLM-style Builds (INVALID - Wrong Methodology)

| Commit | Backend | Baseline (tok/s) | Human (tok/s) | Δ Throughput | Δ TTFT | Result | Why Invalid |
|--------|---------|-----------------|---------------|--------------|--------|--------|-------------|
| **1acca3a2** | torch_native | 236.06 | 246.05 | +4.2% | -5.3% | ? | Wrong model, wrong backend |
| **021f76e4** | torch_native | 265.62 | 264.72 | -0.3% | +3.5% | ? | Wrong model, no LoRA |
| **a37e1247** | torch_native | 260.28 | 243.81 | -6.3% | +9.3% | ? | Wrong model (not VLM) |
| **3212c2ad** | flashinfer/fa3 | 2378.12 | 2386.16 | +0.3% | -4.8% | ? | Wrong model (not VLM) |

**Original (invalid) notes:**
- All tests used `google/gemma-2-2b` instead of correct models
- 3212c2ad uses flashinfer backend (SGLang 0.4.9.post4 has a bug with torch_native + Gemma)
- Results saved to: `/root/sglang-images/OmniPerf-Bench/results/sglang/`

### 1acca3a2 / 6ea1e6ac (torch 2.6.0, triton 3.2.0, SGLang 0.4.6.post2)

Using `torch_native` backend.

| Phase | Commit | Request Throughput | Output Throughput | TTFT Mean | TTFT Median | TTFT P99 | ITL Mean | E2E Latency |
|-------|--------|--------------------|-------------------|-----------|-------------|----------|----------|-------------|
| **Baseline** | 6ea1e6ac | 1.06 req/s | 236.06 tok/s | 1506.08 ms | 1067.87 ms | 6235.32 ms | 148.12 ms | 34416.97 ms |
| **Human** | 1acca3a2 | 1.10 req/s | 246.05 tok/s | 1426.90 ms | 1074.69 ms | 5750.40 ms | 141.59 ms | 33082.98 ms |

**Human vs Baseline:**
- Request throughput: **+3.8%** ✅
- Output throughput: **+4.2%** ✅
- TTFT mean: **-5.3%** ✅ (lower is better)
- E2E latency: **-3.9%** ✅

### 021f76e4 / 777688b8 (torch 2.7.1, triton 3.3.1, SGLang 0.4.7)

Using `torch_native` backend.

| Phase | Commit | Request Throughput | Output Throughput | TTFT Mean | TTFT Median | TTFT P99 | ITL Mean | E2E Latency |
|-------|--------|--------------------|-------------------|-----------|-------------|----------|----------|-------------|
| **Baseline** | 777688b8 | 1.19 req/s | 265.62 tok/s | 1267.57 ms | 884.69 ms | 5665.29 ms | 130.13 ms | 30261.28 ms |
| **Human** | 021f76e4 | 1.19 req/s | 264.72 tok/s | 1312.46 ms | 960.74 ms | 5659.75 ms | 131.23 ms | 30407.22 ms |

**Human vs Baseline:**
- Request throughput: 0.0% (no change)
- Output throughput: **-0.3%** (negligible)
- TTFT mean: **+3.5%** (slightly worse)
- E2E latency: +0.5% (negligible)

### a37e1247 / 136c6e04 (torch 2.7.1, triton 3.3.1, SGLang 0.4.9)

Using `torch_native` backend.

| Phase | Commit | Request Throughput | Output Throughput | TTFT Mean | TTFT Median | TTFT P99 | ITL Mean | E2E Latency |
|-------|--------|--------------------|-------------------|-----------|-------------|----------|----------|-------------|
| **Baseline** | 136c6e04 | 1.17 req/s | 260.28 tok/s | 1266.22 ms | 955.16 ms | 5610.31 ms | 133.92 ms | 31020.54 ms |
| **Human** | a37e1247 | 1.09 req/s | 243.81 tok/s | 1384.47 ms | 986.21 ms | 6232.50 ms | 149.65 ms | 34630.54 ms |

**Human vs Baseline:**
- Request throughput: **-6.8%** ❌
- Output throughput: **-6.3%** ❌
- TTFT mean: **+9.3%** ❌ (higher is worse)
- E2E latency: +11.6% ❌

**Note:** The human commit shows regression with torch_native backend. The optimization may be specific to flashinfer backend.

### 3212c2ad / 53475674 (torch 2.7.1, triton 3.3.1, SGLang 0.4.9.post4)

Using **flashinfer/fa3** backend (required - torch_native crashes on this version with Gemma).

| Phase | Commit | Request Throughput | Output Throughput | TTFT Mean | TTFT Median | TTFT P99 | ITL Mean | E2E Latency |
|-------|--------|--------------------|-------------------|-----------|-------------|----------|----------|-------------|
| **Baseline** | 53475674 | 10.67 req/s | 2378.12 tok/s | 386.38 ms | 408.86 ms | 671.14 ms | 7.17 ms | 2002.22 ms |
| **Human** | 3212c2ad | 10.70 req/s | 2386.16 tok/s | 367.67 ms | 373.05 ms | 639.35 ms | 7.12 ms | 1971.28 ms |

**Human vs Baseline:**
- Request throughput: **+0.3%** ✅
- Output throughput: **+0.3%** ✅
- TTFT mean: **-4.8%** ✅ (lower is better)
- E2E latency: **-1.5%** ✅

**Critical Bug Found:**
- SGLang 0.4.9.post4 has a bug with `torch_native` backend + Gemma models
- Error: `TypeError: unsupported operand type(s) for +: 'int' and 'NoneType'` in `chunk_cache.py:89`
- `attention_chunk_size` is `None` when it shouldn't be
- Workaround: Use flashinfer/fa3 backend (works with triton 3.3.1 on H100)

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
1. ~~**Test newer SGLang commits** from ayushnangia/vllm-docker-build (torch 2.6.0+, triton 3.2.0+)~~
   - **COMPLETED (2026-01-14)** - See benchmark results below

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
| 2025-07-08 | a37e1247 | human | 0.4.9 | 2.7.1 (cu126) | **3.3.1** | 0.2.4 | `a37e1247-vllm-style` |
| 2025-07-08 | 136c6e04 | parent | 0.4.9 | 2.7.1 (cu126) | **3.3.1** | 0.2.4 | `136c6e04-vllm-style` |
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

### 2026-01-14 (Session 2) - CRITICAL METHODOLOGY DISCOVERY

**⚠️ DISCOVERED: All benchmark results are INVALID**

1. **Analyzed HuggingFace dataset** (`Ayushnangia/omniperf_v1`)
   - Found correct `perf_command` for each commit
   - Each commit targets DIFFERENT models and configurations
   - Our generic benchmark was fundamentally wrong

2. **Key findings:**
   - **Docker images are CORRECT** - contain SGLang at right commits
   - **Models NOT baked into images** - specified at runtime
   - **We used wrong benchmark commands** for all commits

3. **What each commit actually needs:**
   | Commit | Correct Model | Special Requirements |
   |--------|---------------|---------------------|
   | 021f76e4 | Llama-3.1-8B-Instruct | LoRA adapter |
   | 1acca3a2 | Llama-3.1-8B-Instruct | FA3/flashinfer backend |
   | 3212c2ad | OpenGVLab/InternVL2_5-8B | VLM model |
   | a37e1247 | Qwen/Qwen2.5-VL-7B-Instruct | VLM + MMMU dataset |

4. **Why our results were wrong:**
   - Used `gemma-2-2b` for ALL commits (wrong)
   - Used `torch_native` backend (some commits need flashinfer)
   - Didn't use LoRA for LoRA optimization commit
   - Didn't use VLM models for VLM optimization commits

5. **Action needed:** Re-run with correct perf_commands from dataset

---

### 2026-01-14 (Session 1) - Benchmark Testing (NOW KNOWN INVALID)

**vLLM-style Builds Benchmark Testing** (results invalid - wrong methodology)

1. **Benchmarked all 4 new commit pairs on H100** (with wrong commands)
   - 1acca3a2/6ea1e6ac: +4.2% (invalid - wrong model/backend)
   - 021f76e4/777688b8: ~0% (invalid - no LoRA)
   - a37e1247/136c6e04: -6.3% (invalid - not VLM)
   - 3212c2ad/53475674: +0.3% (invalid - not VLM)

2. **Discovered SGLang 0.4.9.post4 bug**
   - torch_native backend crashes with Gemma models
   - Error: `attention_chunk_size` is None in `chunk_cache.py`
   - Workaround: Use flashinfer/fa3 backend (works with triton 3.3.1)

3. **Updated benchmark script**
   - Added `use_flashinfer` flag to commit configuration
   - Allows per-commit backend selection
   - 3212c2ad uses flashinfer, others use torch_native

4. **Fixed human/parent label swap**
   - Corrected a37e1247/136c6e04 labels based on journal.json
   - a37e1247 is human, 136c6e04 is parent

5. **Results saved to repo** (invalid but preserved)
   - `/root/sglang-images/OmniPerf-Bench/results/sglang/`
   - JSON files for baseline and human phases for all 4 commits

---

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
