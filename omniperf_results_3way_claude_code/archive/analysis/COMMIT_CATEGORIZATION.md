# Commit Categorization by GPU & Verifiability

## Summary

| GPU Config | Total | Verifiable | Not Verifiable |
|------------|-------|------------|----------------|
| H100:1 | 26 | ~8-10 | ~16-18 |
| H100:2 | 3 | 0 | 3 |
| H100:4 | 1 | 0 | 1 |
| H100:8 | 2 | 0 | 2 |
| **Total** | **32** | **~8-10** | **~22-24** |

---

## H100:8 (2 commits) - ALL NOT VERIFIABLE

| Commit | Model | Status | Reason |
|--------|-------|--------|--------|
| 21d93c14 | Mixtral-8x7B-v0.1 | ❌ NOT VERIFIABLE | Wheel build fails, Docker fallback has Modal infra issues |
| 4fb56914 | DeepSeek-V3-0324 | ❌ NOT VERIFIABLE | Model too large (671B params) - intentionally blocked |

**Assessment**: 8xH100 commits are problematic - one blocked model, one with build issues.

---

## H100:4 (1 commit) - NOT VERIFIABLE

| Commit | Model | Status | Reason |
|--------|-------|--------|--------|
| 379da6dc | Llama-3-70B | ❌ NOT VERIFIABLE | FP8 dtype on vLLM 0.4.2 unstable, baseline fails |

**Assessment**: Only 1 commit, baseline vLLM version doesn't support the feature being tested.

---

## H100:2 (3 commits) - ALL NOT VERIFIABLE

| Commit | Model | Status | Reason |
|--------|-------|--------|--------|
| 0d243f2a | Mixtral-8x7B-Instruct | ❌ NOT VERIFIABLE | **ROCm-specific commit** - targets AMD MI300, won't work on H100 |
| 0ec82edd | Qwen3-30B-A3B | ❌ NOT VERIFIABLE | Qwen3 MoE architecture not supported in baseline vLLM |
| 2a052011 | Mixtral-8x7B-FP8 | ❌ NOT VERIFIABLE | FP8 not supported in baseline vLLM 0.3.3 |

**Assessment**: All 3 have fundamental incompatibility issues (ROCm, unsupported model, missing feature).

---

## H100:1 (26 commits) - MIXED

### ✅ POTENTIALLY VERIFIABLE (~8-10 commits)

These need runner fixes but the underlying benchmark should work:

| Commit | Model | Issue | Fix Needed |
|--------|-------|-------|------------|
| 9474e89b | llama-7b | NO_METRICS - prefix caching | Output parsing fix |
| 6e36f4fa | Llama-3.1-8B | NO_METRICS - chunked prefill | Output parsing fix |
| 9badee53 | Llama-3.2-1B | Missing ShareGPT dataset | Mount dataset in container |
| 2deb029d | Llama-3-8B-FP8 | MODAL_INFRA | Retry (transient) |
| 660470e5 | Llama-3.1-8B | SERVER_FAILED | Investigate server startup |
| 6ce01f30 | Llama-3-8B | SERVER_FAILED | Investigate server startup |
| 6d646d08 | Llama-3-8B | Unknown | Needs investigation |
| 8d75fe48 | Llama-3-8B-FP8 | Unknown | Needs investigation |

### ❌ NOT VERIFIABLE (~16-18 commits)

| Commit | Model | Reason | Category |
|--------|-------|--------|----------|
| 22dd9c27 | Llama-3.1-8B | VLLM_USE_V1=1 not in baseline version | VLLM_VERSION |
| 526de822 | MODEL (placeholder) | Config has `MODEL`, `BS` placeholders | BAD_CONFIG |
| 3a243095 | Llama-3.1-8B | vLLM 0.3.3 server mode unstable | VLLM_VERSION |
| 35fad35a | Llama-3.1-8B | V1 Sampler not in baseline | VLLM_VERSION |
| 22d33bac | Llama-3.1-8B | Frontend async iterators missing | VLLM_VERSION |
| 296f927f | Bamba-9B-v2 | **Mamba2 architecture unsupported** | MODEL_UNSUPPORTED |
| 2f192835 | Llama-3.1-8B | vLLM 0.4.0 server mode issues | VLLM_VERSION |
| 3092375e | Llama-3.1-8B | V1 serialization not in version | VLLM_VERSION |
| 83450458 | Llama-3.1-8B | ngram_prompt_lookup_max=None error | BAD_CONFIG |
| 3476ed08 | Llama-3.1-8B | Block manager v2 version issue | VLLM_VERSION |
| 015069b0 | Qwen3-7B | **Qwen3 ReasoningParser not supported** | MODEL_UNSUPPORTED |
| 67da5720 | Qwen2.5-VL-3B | **Qwen2.5-VL model not supported** | MODEL_UNSUPPORTED |
| 93e5f3c5 | Llama-3.1-8B | Server startup failure | SERVER_FAILED |
| 99abb8b6 | Llama-3.1-8B | Spec decode server failure | SERVER_FAILED |
| aea94362 | Llama-3.2-1B | V1 serving not in version | VLLM_VERSION |
| b6d10354 | Llama-3.1-8B | Unknown - both None | UNKNOWN |
| c45f3c3a | Llama-3.1-8B | Missing wheel URLs | NO_WHEEL |

---

## Root Cause Analysis

### Why so many NOT VERIFIABLE?

| Root Cause | Count | Explanation |
|------------|-------|-------------|
| **VLLM_VERSION** | ~10 | Benchmark tests features (V1, FP8, etc.) that don't exist in baseline vLLM |
| **MODEL_UNSUPPORTED** | 4 | Model (Qwen3, Qwen2.5-VL, Mamba2) not supported in baseline |
| **BAD_CONFIG** | 2-3 | Dataset has placeholders or missing args |
| **ROCm_SPECIFIC** | 1 | Commit targets AMD GPUs, won't work on NVIDIA |
| **MODEL_TOO_LARGE** | 1 | DeepSeek-V3 671B won't fit |

### The Core Problem

**Most benchmarks are testing NEW features against OLD baselines that don't have those features.**

Example:
- Commit adds V1 engine optimization
- Baseline is vLLM 0.3.x which doesn't have V1 engine
- Both baseline AND human fail because benchmark requires V1

---

## Recommendations

### 1. Skip known-bad commits
```python
SKIP_COMMITS = {
    "4fb56914",  # DeepSeek-V3 too large
    "0d243f2a",  # ROCm-specific
    "526de822",  # Placeholder config
    # ... etc
}
```

### 2. Re-evaluate dataset
Many commits are testing features against baselines that can't run them. Need to either:
- Pick different baseline commits that support the feature
- OR only test commits where baseline supports the benchmark

### 3. Focus on H100:1 verifiable subset
Only ~8-10 commits are potentially verifiable. Start with those after fixing:
- Output parsing
- Dataset mounting
- Server startup issues

---

## Final Verdict

**Only ~25-30% of commits are potentially verifiable.** The rest have fundamental issues:
- Wrong baseline versions
- Unsupported models
- ROCm-specific code
- Bad config

This is a **dataset quality issue**, not a runner issue.
