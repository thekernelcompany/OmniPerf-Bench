# Infrastructure Failure Detailed Analysis

## Executive Summary

| Failure Category | Count | Fixable | Notes |
|-----------------|-------|---------|-------|
| Wheel not found | 4 | 3 YES, 1 NO | 1 is 70B model |
| Version bug (#8791) | 5 | NO | vLLM port binding bug |
| Baseline failed | 6 | UNLIKELY | Same issue locally |
| Server failed | 2 | MAYBE | Platform/model specific |
| Broken pipe | 1 | YES | Network timeout |
| Claude success (no local) | 9 | PARTIAL | Only latency data |
| **TOTAL** | **27** | **~5 fixable** | |

---

## Category 1: Wheel Not Found (4 commits)

**Root Cause:** Claude's benchmark infrastructure couldn't find pre-built vLLM wheels for the baseline (parent) commit on S3.

**Fixability:** YES for 3 commits (can run locally with Docker images)

| Commit | Model | Docker Baseline | Fixable | Notes |
|--------|-------|-----------------|---------|-------|
| 9474e89b | huggyllama/llama-7b | 7,184 tok/s | YES | 7B model fits single GPU |
| b6d10354 | meta-llama/Llama-2-70b-hf | 5,213 tok/s | **NO** | 70B - requires multi-GPU |
| e3580537 | neuralmagic/Meta-Llama-3-8B-Instruct-FP8 | 7,441 tok/s | YES | FP8 8B fits single GPU |
| fc7b8d1e | meta-llama/Meta-Llama-3-8B-Instruct | 7,175 tok/s | YES | 8B fits single GPU |

**Action:** Run local 3-way benchmarks for 9474e89b, e3580537, fc7b8d1e

---

## Category 2: Version Bug - Port Binding #8791 (5 commits)

**Root Cause:** vLLM versions 0.6.3-0.6.4 have a known bug (#8791) where the server fails to bind to port during startup, preventing `benchmark_serving.py` from running.

**Fixability:** NO - These commits are from the affected vLLM version range. Would require patching the vLLM source in those Docker images.

| Commit | Model | vLLM Version | PR Title |
|--------|-------|--------------|----------|
| 25ebed2f | meta-llama/Llama-3.1-8B-Instruct | 0.6.4.post2.dev375 | Cache np arange to reduce input preparation overhead |
| 88693683 | meta-llama/Meta-Llama-3-8B | 0.6.4.post2.dev368 | Optimize evictor v1 and v2 performance |
| 9323a315 | meta-llama/Llama-3.2-3B-Instruct | 0.6.4.post2.dev218 | XGrammar support for guided decoding |
| b2e0ad3b | meta-llama/Llama-3.1-8B-Instruct | 0.6.3.post2.dev398 | Reduce peak memory usage of llama |
| f092153f | meta-llama/Llama-3.1-8B-Instruct | 0.6.4.post2.dev330 | Persistent buffers for input preparation |

**Note:** These ARE legitimate performance PRs, just blocked by infrastructure bug. The performance gains they introduce cannot be measured with serving benchmarks in these versions.

---

## Category 3: Baseline Failed - No Metrics (6 commits)

**Root Cause:** The baseline benchmark completed but produced no metrics. Multiple possible causes:
- Server started but crashed during benchmark requests
- V1 engine incompatibilities (`vllm.vllm_flash_attn.fa_utils` missing)
- CUDA TMA incompatibility (`cuTensorMapEncodeTiled` not available)

**Fixability:** UNLIKELY - These same issues occur in local Docker runs

| Commit | Model | Docker Baseline | Issue |
|--------|-------|-----------------|-------|
| 3092375e | Meta-Llama-3-8B-Instruct | 4,450 tok/s | V1 engine issues |
| 83450458 | Meta-Llama-3-8B-Instruct | 7,444 tok/s | Spec decode/ngram issues |
| 93e5f3c5 | Meta-Llama-3-8B-Instruct | 4,912 tok/s | GPU model runner issues |
| 9d72daf4 | Meta-Llama-3-8B-Instruct | 2,343 tok/s | V1 engine issues |
| aea94362 | Meta-Llama-3-8B-Instruct | 3,629 tok/s | Frontend V1 issues |
| b10e5198 | Meta-Llama-3-8B-Instruct | 4,800 tok/s | V1 cache issues |

**Common Pattern:** Many of these are V1 engine optimizations. The V1 engine requires `vllm.vllm_flash_attn.fa_utils` module which is compiled separately and may not be included in the Docker images.

---

## Category 4: Server Failed to Start (2 commits)

**Root Cause:** vLLM server crashes during initialization with XPU platform warnings.

| Commit | Model | Error Pattern | Fixable |
|--------|-------|---------------|---------|
| d55e446d | Meta-Llama-3-8B-Instruct | XPU plugin warnings | MAYBE |
| e493e485 | microsoft/phi-1_5 | XPU plugin warnings | UNLIKELY |

**Analysis:**
- `d55e446d`: Standard 8B model, might be fixable with correct platform flags
- `e493e485`: phi-1_5 has non-standard architecture, may have model-specific issues

---

## Category 5: Broken Pipe (1 commit)

**Root Cause:** Network connection dropped during benchmark execution in Modal cloud environment.

| Commit | Model | Docker Baseline | Fixable |
|--------|-------|-----------------|---------|
| e7b20426 | 01-ai/Yi-1.5-9B-Chat | 6,583 tok/s | YES |

**Action:** Run local 3-way benchmark for e7b20426

---

## Category 6: Claude Success but No Local Data (9 commits)

**Root Cause:** These commits succeeded in Claude's H100 environment but:
1. They used `benchmark_latency.py` which doesn't produce TPOT metrics
2. No Docker baseline images exist locally (docker_throughput=0)

| Commit | Model | Claude Status | Has TPOT? |
|--------|-------|---------------|-----------|
| 3b61cb45 | meta-llama/Llama-3.1-8B-Instruct | success | NO (latency) |
| 4c822298 | meta-llama/Llama-3.1-8B-Instruct | success | NO (latency) |
| 61b8cea3 | meta-llama/Llama-3.2-3B-Instruct | success | NO (throughput) |
| 6dd94dbe | meta-llama/Meta-Llama-3-8B | success | NO (latency) |
| 8c1e77fb | meta-llama/Llama-3.1-8B-Instruct | success | NO (latency) |
| ce6bf3a2 | google/gemma-2b | success | NO (throughput) |
| f26c4aee | meta-llama/Llama-3.1-8B-Instruct | success | NO (latency) |
| fa63e710 | meta-llama/Meta-Llama-3-8B | success | NO (latency) |

**Note:** "Success" means the benchmark ran without errors, but latency/throughput benchmarks don't produce TPOT (Time Per Output Token) metrics that we use for comparison.

---

## Actionable Recovery Plan

### Immediately Fixable (4 commits)
Run local 3-way benchmarks:
```
9474e89b  huggyllama/llama-7b                     (7B model)
e3580537  neuralmagic/Meta-Llama-3-8B-Instruct-FP8 (FP8 8B)
fc7b8d1e  meta-llama/Meta-Llama-3-8B-Instruct     (8B model)
e7b20426  01-ai/Yi-1.5-9B-Chat                    (9B model)
```

### Unfixable (12 commits)
- **5 version_bug:** Port binding issue in vLLM 0.6.3-0.6.4 - cannot run serving benchmarks
- **6 baseline_failed:** V1 engine/CUDA TMA issues - same problems locally
- **1 large model (b6d10354):** 70B model requires multi-GPU setup

### Partial Data Available (9 commits)
Claude dataset shows "success" but only latency benchmark data, no TPOT metrics.

---

## Critical Assessment

**Out of 27 infrastructure failures:**
- **4 (15%)** can definitely be fixed locally
- **5 (19%)** are blocked by vLLM bug #8791 - legitimate performance PRs but unmeasurable
- **6 (22%)** have fundamental V1/CUDA incompatibilities
- **12 (44%)** have various issues (large models, server crashes, missing metrics)

**Recommendation:** Focus on the 4 fixable commits. The version_bug commits are unfortunate - they represent real performance work that cannot be measured due to vLLM infrastructure bugs in those versions.

---

*Generated: 2026-01-10*
