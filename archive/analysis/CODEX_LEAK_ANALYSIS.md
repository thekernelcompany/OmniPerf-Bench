# Codex Benchmark Leak Analysis

**Analysis Date:** 2026-01-20
**Dataset:** [Inferencebench/claude-code-vllm-benchmarks](https://huggingface.co/datasets/Inferencebench/claude-code-vllm-benchmarks)
**Codex Run:** `2025-11-20_11-05-30` (vLLM, gpt-5)

## Executive Summary

This analysis reveals that Codex benchmark tasks contained significant data leakage - the human diff (ground truth solution) was partially or fully included in the task prompt. Despite this contamination, Codex performed poorly, failing to generate patches in 57% of high-leak cases.

## Leak Methodology

Each Codex `task.txt` contains:
1. **`<example_optimization_diff>`** - Partial/full human diff presented as an "example"
2. **`### Human Developer's Approach:`** - Full commit message
3. **`### Files Modified (statistics):`** - Complete diffstat
4. **`## Target Files (ONLY modify these)`** - Exact files to modify

The "example" diff is actually the human solution, just truncated for larger commits.

---

## Overall Statistics

| Metric | Value |
|--------|-------|
| Total Codex tasks analyzed | 99 |
| Total actual diff lines | 28,890 |
| Total shown in task.txt | 2,602 |
| Overall leak rate | **9.0%** |

### Leak Distribution

| Category | Tasks | % of Total |
|----------|-------|------------|
| 100% Full Leak | 16 | 16.2% |
| 75-99% High Leak | 7 | 7.1% |
| 50-74% Medium Leak | 15 | 15.2% |
| <50% Low Leak | 61 | 61.6% |

---

## Detailed Results by Leak Category

### 100% FULL LEAK (16 tasks)

Codex saw the **ENTIRE** human diff in these tasks.

| Task | Commit | Diff (shown/actual) | Patch? | Agent Δ | Human Δ | Verdict | Commit Subject |
|------|--------|---------------------|--------|---------|---------|---------|----------------|
| vllm_core-0006 | 22d33bac | 14/14 | YES | +51.5% | +92.8% | IMPROVED | `merge_async_iterators` fast-path |
| vllm_core-0007 | 22dd9c27 | 23/23 | YES | N/A | N/A | NO METRICS | Optimize Prefill Attention |
| vllm_core-0010 | 299ebb62 | 24/24 | YES | N/A | N/A | NO METRICS | Speed up decode by remove sync |
| vllm_core-0013 | 2f192835 | 12/12 | **NO** | N/A | N/A | **NO PATCH** | latency optimization |
| vllm_core-0026 | 58eee5f2 | 17/17 | YES | N/A | N/A | NO METRICS | Use faster way of decode |
| vllm_core-0029 | 660470e5 | 22/22 | **NO** | N/A | N/A | **NO PATCH** | Optimize evictor-v2 |
| vllm_core-0031 | 6a417b86 | 19/19 | **NO** | N/A | N/A | **NO PATCH** | fix neuron performance issue |
| vllm_core-0035 | 6dd94dbe | 21/21 | **NO** | N/A | +25.0% | **NO PATCH** | fix perf regression |
| vllm_core-0048 | 8c1e77fb | 13/13 | YES | **-27.3%** | +0.9% | **REGRESSED** | Update vllm-flash-attn |
| vllm_core-0051 | 93e5f3c5 | 21/21 | YES | +8.7% | +2.2% | BEAT HUMAN | Optimize Preparing Inputs |
| vllm_core-0059 | 9f1710f1 | 22/22 | YES | +4630%* | -0.1% | SUSPICIOUS | Fix mla prefill context |
| vllm_core-0064 | b10e5198 | 20/20 | **NO** | N/A | N/A | **NO PATCH** | Optimize get_cached_block |
| vllm_core-0065 | b2e0ad3b | 13/13 | YES | +12.3% | +18.6% | IMPROVED | Reduce peak memory usage |
| vllm_core-0090 | ec3b5ce9 | 27/27 | **NO** | N/A | N/A | **NO PATCH** | Improve detokenization |
| vllm_core-0097 | fc542144 | 14/14 | YES | N/A | N/A | NO METRICS | Fix guided decoding bitmask |
| vllm_core-0098 | fc7b8d1e | 26/26 | **NO** | N/A | N/A | **NO PATCH** | e2e overheads reduction |

#### 100% Leak Summary

| Metric | Count | % |
|--------|-------|---|
| Generated patch | 9 | 56% |
| **NO PATCH despite full answer** | 7 | **44%** |
| Actually improved | 4 | 25% |
| **Made it worse** | 1 | **6%** |

---

### 75-99% HIGH LEAK (7 tasks)

Codex saw **most** of the human diff.

| Task | Commit | Diff (shown/actual) | Leak% | Patch? | Agent Δ | Verdict | Commit Subject |
|------|--------|---------------------|-------|--------|---------|---------|----------------|
| vllm_core-0004 | 19d98e0c | 27/28 | 96% | YES | N/A | NO METRICS | Optimize moe intermediate_cache |
| vllm_core-0009 | 296f927f | 26/27 | 96% | **NO** | N/A | **NO PATCH** | Mamba2 Prefill Performance |
| vllm_core-0008 | 25ebed2f | 27/29 | 93% | YES | -0.3% | MATCHED | Cache np arange |
| vllm_core-0069 | b9986454 | 30/33 | 91% | N/A | N/A | NOT TESTED | Fix attention layers unquantized |
| vllm_core-0025 | 526de822 | 27/31 | 87% | **NO** | N/A | **NO PATCH** | Use block size heuristic |
| vllm_core-0039 | 7c01f706 | 27/36 | 75% | **NO** | N/A | **NO PATCH** | Optimize SequenceStatus |
| vllm_core-0072 | bd6028d6 | 27/36 | 75% | **NO** | N/A | **NO PATCH** | Optimized topk for topk=1 |

#### 75-99% Leak Summary

| Metric | Count | % |
|--------|-------|---|
| Benchmarked | 6 | - |
| Generated patch | 2 | 33% |
| **NO PATCH** | 4 | **67%** |
| Actually improved | 0 | 0% |

---

### 50-74% MEDIUM LEAK (15 tasks)

Codex saw a **significant portion** of the human diff.

| Task | Commit | Diff (shown/actual) | Leak% | Patch? | Agent Δ | Verdict | Commit Subject |
|------|--------|---------------------|-------|--------|---------|---------|----------------|
| vllm_core-0085 | e206b543 | 28/39 | 72% | YES | -0.6% | MATCHED | Use xgrammar shared context |
| vllm_core-0084 | dcc6cfb9 | 25/35 | 71% | **NO** | N/A | **NO PATCH** | Tweak MoE Batched silu_mul |
| vllm_core-0061 | ac45c44d | 29/41 | 71% | **NO** | N/A | **NO PATCH** | DeepEPHighThroughput |
| vllm_core-0002 | 0d243f2a | 30/43 | 70% | **NO** | N/A | **NO PATCH** | mi300 mixtral8x7B perf |
| vllm_core-0042 | 83450458 | 28/41 | 68% | **NO** | N/A | **NO PATCH** | Optimize ngram lookup |
| vllm_core-0036 | 6e36f4fa | 28/43 | 65% | **NO** | N/A | **NO PATCH** | improve chunked prefill |
| vllm_core-0053 | 98f47f2a | 27/43 | 63% | YES | +21.1% | BEAT HUMAN | Optimize CPU overheads FlashAttn |
| vllm_core-0077 | ccf02fcb | 27/44 | 61% | **NO** | N/A | **NO PATCH** | Revert Mamba2 Prefill |
| vllm_core-0099 | fe66b347 | 27/44 | 61% | **NO** | N/A | **NO PATCH** | Mamba2 Prefill Tweaks |
| vllm_core-0060 | a3223766 | 27/46 | 59% | **NO** | N/A | **NO PATCH** | Optimize LogitsProcessor |
| vllm_core-0076 | ca7a2d5f | 29/50 | 58% | YES | -19.5% | BEAT HUMAN* | Revert MLA CPU overheads |
| vllm_core-0083 | dae68969 | 29/50 | 58% | **NO** | N/A | **NO PATCH** | Reduce MLA CPU overheads |
| vllm_core-0066 | b55ed6ef | 27/52 | 52% | YES | N/A | NO METRICS | Optimize token_ids_cpu copy |
| vllm_core-0022 | 3b61cb45 | 30/60 | 50% | YES | **-25.2%** | **REGRESSED** | Further reduce CPU overheads |
| vllm_core-0046 | 8aa1485f | 30/60 | 50% | **NO** | N/A | **NO PATCH** | Disable chunked local attention |

#### 50-74% Leak Summary

| Metric | Count | % |
|--------|-------|---|
| Generated patch | 5 | 33% |
| **NO PATCH** | 10 | **67%** |
| Actually improved | 1 | 7% |
| **Made it worse** | 2 | **13%** |

---

## Overall Summary: All Tasks with ≥50% Leak

| Metric | Count | Percentage |
|--------|-------|------------|
| **Total tasks with ≥50% leak** | 38 | 100% |
| Benchmarked | 37 | 97% |
| **Generated any patch** | 16 | **43%** |
| **Failed to generate patch** | 21 | **57%** |
| Actually improved over baseline | 5 | **14%** |
| Made performance worse | 3 | 8% |
| Matched or beat human | 7 | 19% |

---

## Comparison: Leak Rate vs Success Rate

| Leak Category | Tasks | Patch Rate | Improvement Rate | Beat Human Rate |
|---------------|-------|------------|------------------|-----------------|
| 100% Full | 16 | 56% | 25% | 12% |
| 75-99% High | 6 | 33% | 0% | 33% |
| 50-74% Medium | 15 | 33% | 7% | 20% |
| <50% Low | 59 | 29% | 5% | 10% |

**Observation:** Higher leak rates correlate with higher patch generation rates (56% vs 29%), but not with actual performance improvement.

---

## Suspicious Results

The following results appear anomalous and should be investigated:

| Task | Commit | Leak | Agent Δ | Notes |
|------|--------|------|---------|-------|
| vllm_core-0059 | 9f1710f1 | 100% | +4630% | Impossibly high improvement |
| vllm_core-0032 | 6ce01f30 | 21% | +19230% | Impossibly high improvement |

---

## Conclusions

1. **Data Leakage Confirmed:** 38 of 99 tasks (38%) had ≥50% of the human diff leaked in the prompt.

2. **Leakage Did Not Lead to Success:** Despite seeing the full answer:
   - 44% of 100%-leak tasks produced no patch at all
   - Only 25% of 100%-leak tasks actually improved performance
   - 6% made performance worse even with the full answer

3. **Benchmark Validity:** The leaks represent a serious methodological flaw. However, Codex performed so poorly that the contamination did not meaningfully inflate overall results.

4. **Recommendations:**
   - Tasks with ≥50% leak should be flagged/excluded from fair comparisons
   - The task generation pipeline should be fixed to remove human diff from prompts
   - Suspicious results (>1000% improvement) should be investigated as measurement errors

---

## Appendix: Leak Calculation Method

```
Leak % = (lines in <example_optimization_diff>) / (lines in actual git diff) × 100
```

- Actual diff obtained via `git show --format= <commit_hash>` in vLLM repo
- Shown diff extracted from `<example_optimization_diff>` tags in task.txt
- Lines counted after stripping whitespace-only lines
