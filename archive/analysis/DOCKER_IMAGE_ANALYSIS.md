# Comprehensive Docker Image & Commit Analysis

## Executive Summary

| Metric | Value |
|--------|-------|
| **Total commits analyzed** | 96 |
| **Complete 3-way benchmarks** | 22 |
| **Category 1 & 2 commits investigated** | 14 |
| **Docker images available (human)** | 12 |
| **Docker images available (baseline)** | 52 |
| **Recoverable from Category 1 & 2** | 0 |
| **Agent failure cases discovered** | 2 |

---

## Part 1: Category 1 & 2 PR Investigation

### Category 1: baseline_failed_everywhere (8 commits)

These commits failed in both Claude's environment and our local Docker infrastructure.

| Commit | PR | Model from PR | Benchmark Type | PR Result |
|--------|-----|---------------|----------------|-----------|
| 22dd9c27 | [#20308](https://github.com/vllm-project/vllm/pull/20308) | meta-llama/Llama-3.1-8B-Instruct | Serving | 45.90 req/s |
| 2f192835 | [#3890](https://github.com/vllm-project/vllm/pull/3890) | *No benchmark info provided* | - | - |
| 9a3b8832 | [#19939](https://github.com/vllm-project/vllm/pull/19939) | Qwen/Qwen2.5-VL-3B-Instruct | Serving (vision) | 105→112 req/s |
| bfdb1ba5 | [#3469](https://github.com/vllm-project/vllm/pull/3469) | meta-llama/Llama-2-7b-chat-hf | Latency | Detokenization 13ms→2ms |
| c45f3c3a | [#17](https://github.com/vllm-project/vllm/pull/17) | facebook/opt-13b | Latency | 5.18s→3.49s (-32.6%) |
| d4bc1a4d | No PR | - | - | - |
| e7523c2e | [#18608](https://github.com/vllm-project/vllm/pull/18608) | google/gemma-3-12b-it | Serving | 5.35 req/s |
| ec3b5ce9 | [#1338](https://github.com/vllm-project/vllm/pull/1338) | *No model specified* | Detokenization | 13ms→2ms |

### Category 2: unknown_model (6 commits)

These commits had no model information in our benchmark metadata.

| Commit | PR | Model from PR | Benchmark Type | PR Result |
|--------|-----|---------------|----------------|-----------|
| 3127e975 | [#12212](https://github.com/vllm-project/vllm/pull/12212) | *CI config only* | NOT a perf PR | - |
| 4fb56914 | [#21116](https://github.com/vllm-project/vllm/pull/21116) | DeepSeek-V3-0324 | Serving | 17.63 req/s |
| 526de822 | [#11698](https://github.com/vllm-project/vllm/pull/11698) | Qwen2-7B-Instruct-quantized.w8a8 | Latency | 2.8x speedup |
| 98f47f2a | [#10733](https://github.com/vllm-project/vllm/pull/10733) | facebook/opt-125m | Latency | 227ms→192ms (-15%) |
| ac45c44d | [#21837](https://github.com/vllm-project/vllm/pull/21837) | Qwen/Qwen3-30B-A3B-FP8 | Evaluation | 86% GSM8K |
| baeded25 | [#12601](https://github.com/vllm-project/vllm/pull/12601) | DeepSeek V3/R1 | No explicit benchmarks | - |

### Why These Failed

All failures in our infrastructure were due to OmniPerf microbenchmark harness issues:
- `No module named 'vllm._C'` - CUDA extensions not compiled
- These PRs use different vLLM versions requiring specific CUDA compilation
- The "model" field was incorrectly empty because microbenchmarks don't run model serving

---

## Part 2: Docker Image Availability

### Docker Hub Repository
**Repository**: `shikhar481/vllm_fixed_human_images`

| Image Type | Count |
|------------|-------|
| Human images (commit hashes) | 12 |
| Baseline images (parent commits) | 52 |
| **Total** | **64** |

### Category 1 & 2 Docker Image Check

| Commit | Human Image | Baseline Image | Recoverable? |
|--------|-------------|----------------|--------------|
| 22dd9c27 | YES | **NO** | NO - missing baseline |
| 2f192835 | NO | - | NO |
| 9a3b8832 | NO | - | NO |
| bfdb1ba5 | NO | - | NO |
| c45f3c3a | NO | - | NO |
| d4bc1a4d | NO | - | NO |
| e7523c2e | NO | - | NO |
| ec3b5ce9 | NO | - | NO |
| 3127e975 | NO | - | NO |
| 4fb56914 | NO | - | NO |
| 526de822 | NO | - | NO |
| 98f47f2a | NO | - | NO |
| ac45c44d | NO | - | NO |
| baeded25 | NO | - | NO |

**Result**: 13/14 have no Docker images. 1/14 (22dd9c27) has human image but missing baseline.

### 22dd9c27 Deep Dive

The only Category 1 & 2 commit with a Docker image:

- **PR**: #20308 - Optimize Prefill Attention
- **Model**: meta-llama/Llama-3.1-8B-Instruct
- **Human Docker image**: `22dd9c2730dc1124b9d0ac15fff223d0b8d9020b`
- **Parent commit**: `a6d795d593046abd490b16349bcd9b40feedd334`
- **Baseline Docker image**: `baseline-a6d795d59304` - **DOES NOT EXIST**

**Actual benchmark results:**
- Baseline: TIMEOUT after 3600s
- Human: SERVER CRASHED - `ModuleNotFoundError: No module named 'transformers.models.mllama'`
- Agent: FAILED - baseline image not found

**Conclusion**: Even 22dd9c27 is NOT recoverable.

---

## Part 3: Comprehensive Docker Analysis (All 12 Human Images)

### All Commits with Docker Images

| Commit | Category | Model | Human Img | Baseline Img | Human Result | Agent Result | Status |
|--------|----------|-------|-----------|--------------|--------------|--------------|--------|
| 015069b0 | complete_3way | Qwen/Qwen3-1.7B | YES | YES | 198 tok/s | 198 tok/s | **WORKING** |
| 3092375e | other_failed | Meta-Llama-3-8B-Instruct | YES | YES | CRASHED | CRASHED | Both crashed |
| 35fad35a | human_only | Meta-Llama-3-8B-Instruct | YES | YES | 3173 tok/s | CRASHED | **AGENT BROKE IT** |
| 93e5f3c5 | other_failed | Meta-Llama-3-8B-Instruct | YES | YES | CRASHED* | 3707 tok/s | Human img broken |
| 9d72daf4 | other_failed | Meta-Llama-3-8B-Instruct | YES | YES | CRASHED* | 3674 tok/s | Human img broken |
| ad8d696a | human_only | Meta-Llama-3-8B-Instruct | YES | YES | 2383 tok/s | CRASHED | **AGENT BROKE IT** |
| e493e485 | other_failed | microsoft/phi-1_5 | YES | YES | No metrics | CRASHED | Both failed |
| 22dd9c27 | baseline_failed | Llama-3.1-8B-Instruct | YES | **NO** | CRASHED | Failed | Missing baseline |
| 67da5720 | complete_3way | Qwen/Qwen2.5-7B-Instruct | YES | unknown | None | 4694 tok/s | Partial data |
| b10e5198 | other_failed | Meta-Llama-3-8B-Instruct | YES | **NO** | None | None | Missing baseline |
| b6d10354 | other_failed | Llama-2-70b-hf | YES | **NO** | None | None | Large model |
| d55e446d | other_failed | Meta-Llama-3-8B-Instruct | YES | **NO** | None | None | Missing baseline |

*CRASHED with: `ModuleNotFoundError: No module named 'vllm.vllm_flash_attn.fa_utils'`

### 7 Commits with BOTH Human + Baseline Images

| Commit | Human Result | Agent Result | Interpretation |
|--------|--------------|--------------|----------------|
| **015069b0** | 198 tok/s | 198 tok/s | Already in 22 complete benchmarks |
| **35fad35a** | 3173 tok/s | CRASHED | Agent patch broke server |
| **ad8d696a** | 2383 tok/s | CRASHED | Agent patch broke server |
| **93e5f3c5** | CRASHED | 3707 tok/s | Human Docker image broken |
| **9d72daf4** | CRASHED | 3674 tok/s | Human Docker image broken |
| **3092375e** | CRASHED | CRASHED | Fundamental vLLM issue |
| **e493e485** | No metrics | CRASHED | Both failed |

---

## Part 4: Key Findings

### Agent Failures (Legitimate Data Points)

Two commits show **legitimate agent failures** where Claude's patch broke the server:

| Commit | Human Throughput | Agent Status | Error |
|--------|------------------|--------------|-------|
| **35fad35a** | 3173.74 tok/s | CRASHED | Server crashed after applying patch |
| **ad8d696a** | 2382.51 tok/s | CRASHED | Server crashed after applying patch |

These should be included in analysis as cases where the agent produced broken code.

### Human Image Broken (Infrastructure Issue)

Two commits have broken human Docker images but working agent benchmarks:

| Commit | Human Error | Agent Throughput |
|--------|-------------|------------------|
| **93e5f3c5** | `No module named 'vllm.vllm_flash_attn.fa_utils'` | 3706.71 tok/s |
| **9d72daf4** | `No module named 'vllm.vllm_flash_attn.fa_utils'` | 3673.82 tok/s |

This is an infrastructure issue with the human Docker image build, not a legitimate benchmark result.

### Summary by Category

| Category | Count | Notes |
|----------|-------|-------|
| Already working | 1 | 015069b0 |
| Agent crashed (legitimate failure) | 2 | 35fad35a, ad8d696a |
| Human image broken (infra issue) | 2 | 93e5f3c5, 9d72daf4 |
| Both crashed | 2 | 3092375e, e493e485 |
| Missing baseline image | 4 | 22dd9c27, b10e5198, b6d10354, d55e446d |

---

## Part 5: Final Conclusions

### For Category 1 & 2 (14 commits)

**NONE are recoverable.**

| Reason | Count |
|--------|-------|
| No Docker images at all | 13 |
| Has human but missing baseline | 1 |

### Updated Dataset for Human vs Agent Comparison

| Category | Count |
|----------|-------|
| Complete 3-way benchmarks | 22 |
| Agent crashes (human worked) | +2 |
| **Total usable** | **24** |

### Performance Summary (24 usable commits)

| Outcome | Count | Percentage |
|---------|-------|------------|
| Human wins (>2% faster) | 5 | 20.8% |
| Agent wins (>2% faster) | 4 | 16.7% |
| Tie (within 2%) | 13 | 54.2% |
| Agent crashed | 2 | 8.3% |

---

## Recommendations

1. **Category 1 & 2 commits**: Do not attempt recovery - no viable Docker infrastructure.

2. **Agent crash commits (35fad35a, ad8d696a)**: Include in analysis as legitimate agent failures.

3. **Human image broken (93e5f3c5, 9d72daf4)**: Could potentially be fixed by rebuilding human Docker images, but ROI is low.

4. **Final dataset**: Use the 22 complete 3-way benchmarks plus 2 agent crash cases = 24 total data points.

---

*Generated: 2026-01-10*
*Docker Hub: shikhar481/vllm_fixed_human_images*
