# OmniPerf-Bench: Evaluating LLM Agents on Real-World Performance Optimization Tasks

## Abstract

We present an empirical evaluation of large language model (LLM) agents on the task of automated software performance optimization using OmniPerf-Bench, a benchmark derived from real-world performance commits in production ML inference systems. We evaluate multiple agent configurations across 905 optimization tasks extracted from vLLM and SGLang repositories, measuring both functional correctness (patch applicability) and performance impact (speedup). Our results reveal significant disparities in agent performance across different codebases: agents achieve 69.5% pass rate on SGLang tasks but only 18.7% on vLLM tasks. Among agents with measurable speedups, Claude Sonnet 4.5 achieves the highest mean improvement (1.09x on vLLM, 1.01x on SGLang), while exhibiting high variance (σ=0.14-0.39). These findings highlight the challenges of automated performance optimization in complex, production-scale ML systems and suggest directions for improving agent architectures.

---

## 1. Introduction

Automated software performance optimization represents a critical frontier for LLM-based coding agents. While recent advances have demonstrated impressive capabilities in code generation, bug fixing, and test synthesis, the task of identifying and implementing performance improvements poses unique challenges:

1. **Performance reasoning requires deep system understanding**: Unlike functional bugs with clear failure signals, performance bottlenecks require understanding execution profiles, memory hierarchies, and algorithmic complexity.

2. **Optimization validity is context-dependent**: A valid optimization in one context may introduce regressions in another, requiring careful analysis of invariants and side effects.

3. **Measurement is noisy**: Performance improvements must be statistically significant above measurement variance, complicating evaluation.

This work contributes:
- A rigorous evaluation of state-of-the-art LLM agents on 905 real-world performance optimization tasks
- Detailed analysis of failure modes and success patterns across different codebase complexities
- Quantitative characterization of speedup distributions achieved by different agent configurations

---

## 2. Experimental Setup

### 2.1 Benchmark Dataset

OmniPerf-Bench comprises performance optimization commits extracted from two production ML inference frameworks:

| Repository | Description | Tasks | Test Coverage |
|------------|-------------|-------|---------------|
| **vLLM** | High-throughput LLM serving | 580 | 96.3% |
| **SGLang** | Structured generation framework | 291 | 92.1% |
| **Other** | Specialized optimizations | 34 | 100% |
| **Total** | | **905** | **96.4%** |

Each task consists of:
- A pre-optimization commit (baseline)
- The human expert's optimization patch (ground truth)
- An LLM-generated performance test script
- Timing measurements for both baseline and optimized versions

### 2.2 Agent Configurations

We evaluate the following agent configurations:

| Agent | Model | Configuration | Tasks |
|-------|-------|---------------|-------|
| **sglang_claude_sonnet45** | Claude Sonnet 4.5 | TRAE framework | 80 |
| **sglang_core** | TRAE (default) | Multiple runs | 211 |
| **vllm_claude_sonnet45** | Claude Sonnet 4.5 | TRAE framework | 99 |
| **vllm_claude_sonnet45_retry** | Claude Sonnet 4.5 | With retry logic | 82 |
| **vllm_core** | TRAE (default) | Multiple runs | 300 |
| **vllm_core_codex** | Codex CLI | Offline mode | 99 |
| **Other specialized** | Various | Task-specific | 34 |

### 2.3 Evaluation Metrics

We report:
- **Pass Rate**: Percentage of tasks where the agent generates a valid, applicable patch
- **Speedup**: Ratio of baseline time to patched time (>1.0 indicates improvement)
- **Improvement Rate**: Percentage of passed tasks achieving speedup >1.0

---

## 3. Main Results

### 3.1 Overall Performance by Agent

**Table 1: Agent Performance Summary**

| Agent | Total | Passed | Pass Rate | Improvements | Regressions | Errors |
|-------|-------|--------|-----------|--------------|-------------|--------|
| sglang_claude_sonnet45 | 80 | 67 | **83.8%** | 14 | 12 | 5 |
| sglang_core | 211 | 135 | **64.0%** | 21 | 29 | 13 |
| vllm_core_codex | 99 | 30 | 30.3% | 10 | 10 | 54 |
| vllm_claude_sonnet45_retry | 82 | 22 | 26.8% | 8 | 4 | 46 |
| vllm_core | 300 | 64 | 21.3% | 14 | 14 | 144 |
| vllm_claude_sonnet45 | 99 | 9 | 9.1% | 5 | 2 | 53 |
| chunked_local_attn_opt | 15 | 1 | 6.7% | 0 | 0 | 0 |
| moe_align_opt | 15 | 0 | 0.0% | 0 | 0 | 15 |
| prefix_caching_opt | 4 | 0 | 0.0% | 0 | 0 | 0 |
| **TOTAL** | **905** | **328** | **36.2%** | **72** | **71** | **330** |

**Key Finding**: SGLang tasks show dramatically higher pass rates (64-84%) compared to vLLM tasks (9-30%), suggesting fundamental differences in task complexity.

### 3.2 Performance by Repository

**Table 2: Repository-Level Comparison**

| Metric | SGLang | vLLM | Ratio |
|--------|--------|------|-------|
| Total Tasks | 291 | 580 | 0.50x |
| Passed Tests | 202 | 125 | 1.62x |
| Pass Rate | **69.4%** | **21.6%** | 3.22x |
| Total Improvements | 35 | 37 | 0.95x |
| Improvement Rate (of passed) | 17.3% | 29.6% | 0.58x |
| Error Rate | 7.9% | 51.0% | 0.15x |

**Observation**: While SGLang has 3.2x higher pass rate, vLLM shows higher improvement rate among passed tests (29.6% vs 17.3%), suggesting that when vLLM patches succeed, they tend to be more impactful.

### 3.3 Speedup Distribution Analysis

**Table 3: Speedup Statistics by Agent**

| Agent | n | Mean | Median | Std | Min | Max | >1.0x |
|-------|---|------|--------|-----|-----|-----|-------|
| vllm_claude_sonnet45 | 7 | **1.091** | 1.024 | 0.142 | 0.985 | 1.310 | 71.4% |
| vllm_claude_sonnet45_retry | 12 | **1.056** | 1.024 | 0.391 | 0.121 | 1.728 | 66.7% |
| vllm_core_codex | 20 | 1.030 | 1.004 | 0.286 | 0.307 | 1.910 | 50.0% |
| sglang_claude_sonnet45 | 26 | 1.012 | 1.001 | 0.067 | 0.883 | 1.230 | 53.8% |
| sglang_core | 50 | 1.000 | 0.999 | 0.034 | 0.902 | 1.159 | 42.0% |
| vllm_core | 28 | 0.939 | 0.998 | 0.369 | 0.183 | 2.170 | 50.0% |

**Key Findings**:
1. **Claude Sonnet 4.5 achieves highest mean speedups** on vLLM (1.09x), but with substantial variance
2. **SGLang shows lower variance** (σ=0.03-0.07) but also lower mean improvements
3. **vLLM patches exhibit bimodal behavior**: some achieve 2x+ speedups while others regress to 0.1-0.3x

---

## 4. Analysis and Ablations

### 4.1 Why Does SGLang Outperform vLLM?

We hypothesize several factors:

**A. Codebase Complexity**
- vLLM: ~150K lines of Python + 30K lines of CUDA kernels
- SGLang: ~80K lines of Python, minimal native code

**B. Optimization Surface**
- vLLM optimizations often require CUDA kernel modifications or complex scheduling changes
- SGLang optimizations are typically Python-level algorithmic improvements

**C. Test Environment Requirements**
- vLLM requires specific CUDA compilation, GPU memory management
- SGLang tests run more reliably in isolated environments

### 4.2 Error Mode Analysis

Of 330 total errors:

| Error Category | Count | Percentage |
|----------------|-------|------------|
| Import/Module errors | ~180 | 54.5% |
| CUDA/GPU errors | ~80 | 24.2% |
| Git worktree errors | ~40 | 12.1% |
| Other | ~30 | 9.1% |

**Import errors dominate**, primarily due to:
1. API changes between vLLM versions
2. Missing compiled extensions (`vllm._C`)
3. Incompatible dependency versions

### 4.3 Patch Quality Analysis

**No-Patch Rate by Agent**:

| Agent | No Patch | Rate |
|-------|----------|------|
| sglang_core (ae58875a) | 42/80 | 52.5% |
| vllm_claude_sonnet45 | 34/99 | 34.3% |
| vllm_core (84ca0ad4) | 20/44 | 45.5% |
| vllm_core (8e54a51a) | 17/32 | 53.1% |

High no-patch rates indicate the agent either:
1. Could not identify an optimization opportunity
2. Generated a patch that failed to apply
3. Determined no optimization was needed

### 4.4 Variance Analysis

**High Variance Agents** (σ > 0.25):
- `vllm_claude_sonnet45_retry`: σ = 0.391
- `vllm_core`: σ = 0.369
- `vllm_core_codex`: σ = 0.286

This high variance suggests:
1. **Inconsistent optimization quality**: Same agent produces both excellent (2x) and harmful (0.1x) patches
2. **Task difficulty heterogeneity**: Some vLLM optimizations are inherently harder
3. **Sensitivity to prompt/context**: Minor differences in task framing lead to dramatically different outcomes

**Low Variance Agents** (σ < 0.1):
- `sglang_core`: σ = 0.034
- `sglang_claude_sonnet45`: σ = 0.067

Lower variance on SGLang suggests more predictable optimization behavior, possibly due to simpler optimization patterns.

---

## 5. Discussion

### 5.1 The Performance Optimization Challenge

Our results reveal that automated performance optimization remains challenging even for state-of-the-art LLM agents:

1. **Overall pass rate of 36.2%** indicates substantial room for improvement
2. **Net neutral speedup** (72 improvements vs 71 regressions) suggests agents often trade performance in unpredictable ways
3. **High error rates on complex codebases** (51% on vLLM) indicate infrastructure and environment challenges

### 5.2 Agent Architecture Implications

**Claude Sonnet 4.5 shows promise** with highest mean speedups, but requires:
- Better error recovery (currently 53% error rate on vLLM)
- Reduced variance through ensemble or verification methods
- Improved handling of compiled dependencies

**TRAE framework (core)** shows:
- Higher reliability on simpler codebases
- Lower variance but also lower impact
- Better suited for Python-level optimizations

### 5.3 Limitations

1. **Test script validity**: LLM-generated tests may not capture all performance-relevant scenarios
2. **Single-run measurements**: Performance measurements have inherent variance
3. **Environment reproducibility**: Different hardware/software configurations may yield different results
4. **Commit selection bias**: Extracted commits may not represent typical optimization opportunities

### 5.4 Recommendations for Future Work

1. **Multi-stage verification**: Agents should verify optimizations against multiple test scenarios
2. **Performance profiling integration**: Agents should use profiling data to guide optimization
3. **Incremental optimization**: Break complex optimizations into verifiable steps
4. **Domain-specific fine-tuning**: Specialized models for CUDA, scheduling, memory management

---

## 6. Conclusion

We presented a comprehensive evaluation of LLM agents on real-world performance optimization tasks from OmniPerf-Bench. Our analysis of 905 tasks across vLLM and SGLang repositories reveals:

1. **Significant performance gap**: 3.2x higher pass rate on SGLang vs vLLM tasks
2. **Claude Sonnet 4.5 achieves best speedups** (mean 1.09x on vLLM) but with high variance
3. **Error handling remains critical**: 36% of runs fail due to environment/import issues
4. **Net neutral optimization**: Similar numbers of improvements and regressions

These findings suggest that while LLM agents show promise for automated performance optimization, significant advances in error recovery, variance reduction, and domain-specific reasoning are needed before reliable deployment in production systems.

---

## Appendix A: Detailed Results

### A.1 Per-Run Statistics

| Run ID | Total | Passed | Rate | Avg Speedup | Improvements |
|--------|-------|--------|------|-------------|--------------|
| sglang_claude_sonnet45-c0645fb7 | 80 | 67 | 83.8% | 1.012 | 14 |
| sglang_core-389be848 | 80 | 69 | 86.3% | 1.005 | 13 |
| sglang_core-ae58875a | 80 | 27 | 33.8% | 0.996 | 5 |
| sglang_core-bd68ff67 | 51 | 39 | 76.5% | 0.994 | 3 |
| vllm_claude_sonnet45-0a51aaa8 | 99 | 9 | 9.1% | 1.091 | 5 |
| vllm_claude_sonnet45_retry-5d58acda | 82 | 22 | 26.8% | 1.056 | 8 |
| vllm_core-9641716f | 60 | 22 | 36.7% | 0.997 | 6 |
| vllm_core-beffe4cd | 49 | 15 | 30.6% | 1.015 | 5 |
| vllm_core-a40b2039 | 49 | 13 | 26.5% | 0.770 | 2 |
| vllm_core_codex-90a1c13f | 99 | 30 | 30.3% | 1.030 | 10 |

### A.2 Speedup Distribution Visualization

```
Speedup Distribution (n=143 measurements)

    0.0-0.5x  ████████ (8)      - Severe regression
    0.5-0.8x  ██████ (6)        - Moderate regression
    0.8-1.0x  ████████████████████████████████████████ (57) - Minor regression/neutral
    1.0-1.2x  ██████████████████████████████████████████████ (61) - Minor improvement
    1.2-1.5x  ████████ (7)      - Moderate improvement
    1.5-2.0x  ███ (3)           - Good improvement
    2.0x+     █ (1)             - Excellent improvement
```

---

*Generated: December 9, 2025*
*Evaluation Framework: OmniPerf-Bench v0.1*
*Total Runtime: ~4 hours on NVIDIA GPU cluster*
