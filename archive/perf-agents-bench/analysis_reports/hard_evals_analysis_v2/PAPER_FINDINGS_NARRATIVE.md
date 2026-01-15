# OmniPerf-Bench: Paper Findings Narrative

**Generated:** 2026-01-14

## Executive Summary

OmniPerf-Bench is a challenging benchmark for evaluating AI agents on real-world performance optimization tasks. This analysis of 96 Claude Code runs on vLLM performance commits reveals:

- **18.8% success rate** (18/96 runs produced effective optimizations)
- **66.7% prediction accuracy** between LLM-as-judge soft metrics and runtime hard metrics
- **Localization failures dominate** (65.4% of failures) - agents optimize the wrong code

---

## 1. Why OmniPerf-Bench is a Good Benchmark

### 1.1 Real-World Provenance

OmniPerf-Bench tasks are derived from actual merged performance PRs in production open-source projects:

| Characteristic | Value |
|----------------|-------|
| Source repository | vLLM (production inference engine) |
| Total commits analyzed | 96 |
| Soft metric coverage | 96/96 (100%) |
| Hard metric coverage | 53/96 (55%) |

Unlike synthetic benchmarks, these are real optimizations written by domain experts and validated through production deployment.

### 1.2 Expert-Validated Ground Truth

Each task has a human expert solution serving as ground truth:
- Human implementations represent months of domain knowledge
- Solutions target actual performance bottlenecks identified through profiling
- Commits passed code review and CI/CD pipelines before merge

### 1.3 Diverse Task Types

Performance optimization spans multiple domains:

| Domain | Count | % of Successes |
|--------|-------|----------------|
| Memory optimization | 9 | 50.0% |
| Compute optimization | 7 | 38.9% |
| Concurrency | 1 | 5.6% |
| Other | 1 | 5.6% |

### 1.4 Measurable Outcomes

Unlike code generation benchmarks that rely solely on test pass rates, OmniPerf-Bench provides:
- **Soft metrics**: LLM-as-judge evaluation of patch quality (scalable, fast)
- **Hard metrics**: Runtime benchmark measurements (objective ground truth)
- **5 metric types**: throughput, TTFT, TPOT, ITL, latency_avg

---

## 2. Why OmniPerf-Bench is Hard

### 2.1 Agent Failure Analysis (GSO-Style)

Following the methodology from GSO (arxiv:2505.23671), we categorize agent failures:

**Total: 96 runs | 18 successes (18.8%) | 78 failures (81.2%)**

#### High-Level Failure Distribution

| Category | Count | % | Description |
|----------|-------|---|-------------|
| **Localization** | 51 | 65.4% | Agent failed to identify the correct performance bottleneck |
| **Mismanage Compute** | 17 | 21.8% | Agent wasted resources or caused catastrophic failures |
| **Avoid Complexity** | 10 | 12.8% | Agent avoided necessary complexity to implement the fix |

#### Sub-Category Breakdown

| Sub-Category | Count | % | Interpretation |
|--------------|-------|---|----------------|
| Misdiagnosed Bottlenecks | 51 | 65.4% | Optimized wrong component; not the critical path |
| Destructive/Runaway | 9 | 11.5% | Corrupted repository or made unbuildable changes |
| Wrong Abstraction Level | 7 | 9.0% | Stayed in Python when CUDA/C++ was required |
| Exploit-Heavy | 5 | 6.4% | Over-engineered with unnecessary complexity |
| Lazy Optimization | 3 | 3.8% | Made superficial changes when deeper work needed |
| Explore-Heavy | 3 | 3.8% | Spent time reading code without acting |

### 2.2 Key Insight: Localization is the Dominant Challenge

Two-thirds of failures (65.4%) are localization failures. This means:
- Agents can understand the task description
- Agents can write valid optimization code
- **But agents cannot reliably identify WHICH code to optimize**

This is particularly challenging in large codebases like vLLM where:
- Multiple files could plausibly be the bottleneck
- Profiling data is not always available in the task context
- Domain expertise is required to reason about performance

### 2.3 Abstraction Level Mismatch

9% of failures involve working at the wrong abstraction level:
- Human wrote CUDA kernel optimizations
- Agent stayed at Python level
- Fundamental capability gap in low-level optimization

---

## 3. Why Both Soft and Hard Metrics Matter

### 3.1 Soft Metrics: Scalable Evaluation

LLM-as-judge soft metrics provide:
- **Fast feedback** without running benchmarks (~seconds vs minutes)
- **Full coverage** (100% of commits have soft metrics)
- **Interpretable categories** explaining WHY an agent failed

**Soft metric categories:**
- `speedup_likelihood`: likely_similar, likely_partial, likely_ineffective, likely_regression
- `bottleneck_target`: same_target, related_target, different_target, no_optimization
- `approach_comparison`: same_approach, similar_approach, valid_alternative, ineffective, harmful
- `failure_mode`: localization_failure, complexity_avoidance, incomplete_implementation, etc.

### 3.2 Hard Metrics: Objective Ground Truth

Runtime benchmarks provide:
- **Objective measurement** of actual performance impact
- **Quantitative comparison** (agent vs human in %)
- **Validation of soft metrics** (do predictions match reality?)

**Coverage:** 42 commits with both human AND agent benchmarks (for comparison)

### 3.3 Correlation Analysis: The Key Finding

**Prediction Accuracy: 66.7%** (26/39 matched commits)

| Soft Prediction | Correct | Total | Accuracy |
|-----------------|---------|-------|----------|
| likely_similar | 7 | 7 | **100%** |
| likely_ineffective | 15 | 18 | **83%** |
| likely_regression | 4 | 8 | 50% |
| likely_partial | 0 | 5 | **0%** |

**Key insights:**
1. **Strong positive predictor**: When soft metrics predict "likely_similar" (agent matches human), they are correct 100% of the time
2. **Good failure detector**: "likely_ineffective" predictions are 83% accurate
3. **Weak partial success detection**: "likely_partial" predictions are never correct - agents either fully succeed or don't

### 3.4 Complementary Information

| Aspect | Soft Metrics | Hard Metrics |
|--------|--------------|--------------|
| Coverage | 100% | 55% |
| Speed | Seconds | Minutes |
| Explains WHY | Yes (failure categories) | No |
| Measures HOW MUCH | No | Yes (±X% performance) |
| Objective | No (LLM judgment) | Yes (runtime) |

**Together they provide:**
- Comprehensive agent evaluation across all tasks (soft)
- Validation on subset with ground truth (hard)
- Actionable insights for agent improvement (both)

---

## 4. When Agents Succeed: Pattern Analysis

### 4.1 Implementation Alignment

Of the 18 successful runs:

| Category | Count | % | Description |
|----------|-------|---|-------------|
| core_match_extras | 10 | 55.6% | Got main optimization, added unnecessary noise |
| identical | 8 | 44.4% | Produced functionally identical solution to human |

**Insight:** When successful, 44% of agent solutions are as clean as human solutions - a notable improvement. The remaining 56% add extra complexity.

### 4.2 Agent Behavior Patterns (Successful Runs)

| Pattern | Count | % | Description |
|---------|-------|---|-------------|
| diff_pollution | 7 | 38.9% | Included unrelated CI/CD/doc changes |
| over_engineering | 6 | 33.3% | Applied 2+ more techniques than needed |
| clean_patch | 5 | 27.8% | Surgical, minimal changes (ideal) |
| scope_creep | 4 | 22.2% | Modified files beyond necessary scope |

### 4.3 Technique Match Analysis

| Match Type | Count | % |
|------------|-------|---|
| exact_match | 10 | 55.6% |
| agent_superset | 8 | 44.4% |

When successful, agents either match human techniques exactly (56%) or use all human techniques plus additional ones (44%). They never use a subset - suggesting agents tend toward "more is better" when uncertain.

---

## 5. Implications for Agent Development

### 5.1 Priority: Improve Localization

With 65% of failures being localization errors, the highest-impact improvement would be:
- Better bottleneck identification
- Integration with profiling tools
- Domain-specific reasoning about performance

### 5.2 Address Abstraction Level Gap

9% of failures involve staying at wrong abstraction level:
- Need capability to write CUDA/C++ when required
- Better recognition of when low-level work is needed

### 5.3 Reduce Noise in Successful Patches

Even successful agents are noisy:
- 39% include unrelated changes (diff pollution)
- 33% over-engineer
- Only 28% produce clean patches

Focus on producing minimal, surgical changes rather than "kitchen sink" approaches.

---

## 6. Summary Statistics

| Metric | Value |
|--------|-------|
| Total runs analyzed | 96 |
| Success rate | 18.8% (18/96) |
| Failure rate | 81.2% (78/96) |
| Soft-hard prediction accuracy | 66.7% (26/39) |
| Hard metrics available | 42 commits |
| Matched for comparison | 39 commits |
| Dominant failure mode | Localization (65.4%) |
| Clean patch rate (successes only) | 27.8% (5/18) |

---

## 7. Conclusion

OmniPerf-Bench reveals that performance optimization remains a significant challenge for AI agents:

1. **The benchmark is realistic**: Real PRs from production codebases with expert solutions
2. **The benchmark is hard**: 81% failure rate, dominated by localization errors
3. **Evaluation is comprehensive**: Soft metrics explain failures, hard metrics validate predictions
4. **Correlation is meaningful**: 67% prediction accuracy shows soft metrics are useful proxies
5. **Room for improvement is clear**: Better localization and reduced over-engineering are priorities

This combination of real-world tasks, dual evaluation methodology, and detailed failure analysis makes OmniPerf-Bench a valuable tool for advancing agent capabilities in performance optimization.
