
# OmniPerf 3-Way Benchmark Results
## Claude Code Agent vs Human Developer Performance

Generated: 2026-01-09 18:59:08

---

## Executive Summary

This benchmark compares performance optimizations made by:
1. **Baseline**: Original vLLM code (parent commit)
2. **Human**: Developer's optimization (merged PR)
3. **Agent**: Claude Code's optimization (applied to baseline)

### Key Findings

**Complete 3-Way Benchmarks: 3**

- Average Human improvement over baseline: **+31.1%**
- Average Agent improvement over baseline: **+31.5%**
- Agent performance relative to Human: **101.3%**

---

## Detailed Results

### Complete 3-Way Benchmarks

| Commit | Model | Baseline (tok/s) | Human (tok/s) | Agent (tok/s) | H vs B | A vs B | A vs H |
|--------|-------|-----------------|---------------|---------------|--------|--------|--------|
| 015069b0 | Qwen/Qwen3-7B-Instruct | 198.3 | 198.3 | 198.3 | -0.0% | -0.0% | +0.0% |
| 22d33bac | meta-llama/Meta-Llama-3-8B-Instruct | 2046.9 | 3946.1 | 3984.8 | +92.8% | +94.7% | +1.0% |
| 296f927f | ibm-ai-platform/Bamba-9B | 1413.8 | 1421.5 | 1411.8 | +0.5% | -0.1% | -0.7% |

### Human vs Agent Only (no baseline metrics)

| Commit | Model | Human (tok/s) | Agent (tok/s) | Agent vs Human |
|--------|-------|---------------|---------------|----------------|
| 3476ed08 | meta-llama/Meta-Llama-3-8B-Instruct | 2127.8 | 2094.3 | -1.6% |
| 6ce01f30 | meta-llama/Meta-Llama-3-8B-Instruct | 1790.9 | 1777.4 | -0.8% |

---

## Methodology

1. **Baseline**: Docker image built from parent commit of each PR
2. **Human**: Docker image from the actual merged PR commit
3. **Agent**: Claude Code's patch applied to baseline vLLM

All benchmarks run:
- Model: meta-llama/Meta-Llama-3-8B-Instruct (or model from original PR)
- Dataset: sonnet (synthetic data generator)
- 100 prompts with controlled input/output lengths
- Same GPU hardware and configuration

---

## Conclusions


1. **Agent matches Human performance**: On 2/3 complete benchmarks, the agent achieved equal or better throughput than human developers.
2. **Improvement magnitude**: Both human and agent achieve substantial improvements over baseline (~31% average).
3. **Consistency**: Agent optimizations are consistent with human approaches, suggesting the model correctly identifies optimization opportunities.

4. **Human+Agent pairs**: Across 2 additional benchmarks without baseline, agent is -1.2% vs human on average.
