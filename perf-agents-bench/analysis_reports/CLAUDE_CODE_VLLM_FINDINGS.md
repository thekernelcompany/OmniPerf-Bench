# Claude Code (Sonnet 4.5) Performance on vLLM Optimization Tasks

## Executive Summary

This report analyzes Claude Code's performance on 44 vLLM optimization tasks, evaluating both actual benchmark performance (hard metrics) and LLM-as-judge evaluations (soft metrics). The key finding is that **soft metrics have no predictive power for actual benchmark performance** - in fact, they often show inverted correlations.

**Key Statistics:**
- **Success Rate (Hard Metrics):** 47.7% (agent performs equal to or better than human)
- **Failure Rate:** 50.0% (agent performs worse OR fails to produce runnable patch)
- **Soft-Hard Correlation:** Essentially zero or inverted

---

## Data Sources

### Hard Metrics
- **Source:** HuggingFace dataset `Inferencebench/claude-code-vllm-benchmarks`
- **Filter:** `agent_name='claude-code'`, `agent_model='sonnet-4.5'`
- **Metrics:** throughput, ttft_mean, tpot_mean, itl_mean, latency_avg
- **Comparison:** H (Human patch), A (Agent patch), B (Baseline)

### Soft Metrics
- **Source:** `perf-agents-bench/state/analysis/vllm/claude_code/sonnet-4.5/vllm_<commit>/`
- **Files Analyzed:**
  - `patch_quality.json` (speedup_likelihood, failure_mode, bottleneck_target, approach_comparison)
  - `patch_similarity.json` (file_overlap_pct, line_overlap_pct, approach_similarity_score)

### Dataset Composition

| Category | Count | Description |
|----------|-------|-------------|
| Total rows in dataset | 93 | All claude-code/sonnet-4.5 entries |
| Commits to ignore | 9 | Wrong performance command used |
| Additional from rerun | 3 | 3476ed08, 6ce01f30, 99abb8b6 |
| **Valid commits** | **44** | Analyzed in this report |

**Breakdown of 44 Valid Commits:**
| Category | Count | Percentage |
|----------|-------|------------|
| H+A+B (full data) | 22 | 50.0% |
| H+A but NO B | 10 | 22.7% |
| Agent failures (H only) | 11 | 25.0% |
| Human failure (B+A only) | 1 | 2.3% |

### Commits Excluded (Wrong Perf Command)
```
ed250545, 8a4e5c5f, f26c4aee, 6d0734c5, 61b8cea3, cf2f084d, 80aa7e91, ca7a2d5f, 8bc68e19
```

### Agent Failure Commits (Patch Generated but Failed to Run)
```
35fad35a, 3b61cb45, 660470e5, 6dd94dbe, 8c1e77fb, 9ed82e70, ad8d696a, ccf02fcb, ce6bf3a2, d7740ea4, e7b20426
```

---

## Part 1: Hard Metrics Analysis (H vs A vs B)

### Overall Performance Distribution

**44 Commits with H Data Overall:**
| Category | Count | Percentage |
|----------|-------|------------|
| Agent beats Human (>2%) | 9 | 20.5% |
| Similar (within 2%) | 12 | 27.3% |
| Agent worse (<-2%) | 11 | 25.0% |
| Agent failed (no A data) | 11 | 25.0% |
| Human failed (B+A only) | 1 | 2.3% |
| **Combined worse+agent_failed** | **22** | **50.0%** |

### Performance Among Successful Patches

**32 Commits with H+A (can calculate A vs H):**
| Category | Count | Percentage |
|----------|-------|------------|
| Agent beats Human (>2%) | 9 | 28.1% |
| Similar (within 2%) | 12 | 37.5% |
| Agent worse (<-2%) | 11 | 34.4% |

### Top Performers (Beats Category)

| Commit | A vs H | Soft Prediction |
|--------|--------|-----------------|
| e3580537 | **+24.43%** | likely_ineffective, localization_failure, ineffective |
| fc7b8d1e | **+17.34%** | likely_partial, not_applicable, similar_approach |
| 6e36f4fa | **+15.35%** | likely_partial, complexity_avoidance, partial_solution |
| a3223766 | **+8.17%** | likely_partial, localization_failure, valid_alternative |
| e206b543 | **+7.96%** | likely_partial, localization_failure, valid_alternative |
| 6a417b86 | **+7.22%** | likely_ineffective, localization_failure, ineffective |

**Note:** The two best performers (e3580537, 6a417b86) were both marked as "likely_ineffective" and "ineffective" by soft metrics!

### Worst Performers

| Commit | A vs H | Soft Prediction |
|--------|--------|-----------------|
| 2deb029d | **-19.21%** | likely_ineffective, localization_failure, ineffective |
| 89a84b0b | **-16.61%** | likely_partial, complexity_avoidance, partial_solution |
| b690e348 | **-16.40%** | likely_ineffective, complexity_avoidance, ineffective |
| 9474e89b | **-7.58%** | likely_partial, not_applicable, similar_approach |

### Rerun Results (2026-01-16)

Three commits were rerun with corrected benchmark commands:

| Commit | Status | A vs H | Category |
|--------|--------|--------|----------|
| 3476ed08 | H+A+B | -5.2% | worse |
| 6ce01f30 | H+A+B | -0.11% | similar |
| 99abb8b6 | B+A only | N/A | human_failure |

### Key Insights

- **50.0%** of commits either failed or performed worse than human patches
- **47.7%** performed equal to or better than human (9 beats + 12 similar)
- **1 commit** where human patch failed but agent succeeded (99abb8b6)
- High variance: range from **-19.21%** to **+24.43%**

---

## Part 2: Soft Metrics Analysis (44 Commits)

### Speedup Likelihood Distribution

| Category | Count | Percentage |
|----------|-------|------------|
| likely_partial | 27 | 61.4% |
| likely_ineffective | 11 | 25.0% |
| likely_similar | 5 | 11.4% |
| uncertain | 1 | 2.3% |

### Failure Mode Distribution

| Category | Count | Percentage |
|----------|-------|------------|
| localization_failure | 15 | 34.1% |
| complexity_avoidance | 14 | 31.8% |
| not_applicable | 11 | 25.0% |
| incomplete_implementation | 4 | 9.1% |

### Bottleneck Target Distribution

| Category | Count | Percentage |
|----------|-------|------------|
| related_target | 22 | 50.0% |
| same_target | 16 | 36.4% |
| different_target | 6 | 13.6% |

### Approach Comparison Distribution

| Category | Count | Percentage |
|----------|-------|------------|
| valid_alternative | 17 | 38.6% |
| partial_solution | 14 | 31.8% |
| ineffective | 9 | 20.5% |
| similar_approach | 5 | 11.4% |

### Patch Similarity Metrics (Averages)

- **Average file_overlap_pct:** 77.9%
- **Average line_overlap_pct:** 3.7%
- **Average approach_similarity_score:** 2.82

### Soft Metrics for 11 Agent Failures

| Commit | Speedup | Failure Mode | Bottleneck | Approach |
|--------|---------|--------------|------------|----------|
| 35fad35a | likely_partial | incomplete_implementation | same_target | partial_solution |
| 3b61cb45 | likely_partial | localization_failure | related_target | valid_alternative |
| 660470e5 | likely_partial | incomplete_implementation | same_target | partial_solution |
| 6dd94dbe | likely_ineffective | localization_failure | different_target | ineffective |
| 8c1e77fb | uncertain | localization_failure | related_target | valid_alternative |
| 9ed82e70 | likely_partial | not_applicable | related_target | valid_alternative |
| ad8d696a | likely_partial | incomplete_implementation | related_target | valid_alternative |
| ccf02fcb | likely_similar | not_applicable | same_target | valid_alternative |
| ce6bf3a2 | likely_ineffective | localization_failure | different_target | ineffective |
| d7740ea4 | likely_partial | localization_failure | related_target | valid_alternative |
| e7b20426 | likely_partial | complexity_avoidance | related_target | partial_solution |

**Critical Finding:** Most failures (9/11) were predicted as "likely_partial" or better, NOT "likely_ineffective". Soft metrics failed to predict actual runtime failures.

---

## Part 3: Hard-Soft Correlation Analysis

### Core Finding: No Predictive Power

Soft metrics show essentially **zero or inverted correlation** with actual benchmark performance.

### Speedup Likelihood vs Actual Outcome (32 H+A commits)

| Soft Prediction | n | Beats | Similar | Worse | Beats % |
|-----------------|---|-------|---------|-------|---------|
| likely_similar | 4 | 0 | 2 | 2 | **0%** |
| likely_partial | 20 | 7 | 7 | 6 | 35% |
| likely_ineffective | 8 | 3 | 3 | 2 | **37.5%** |

**"likely_ineffective" has a HIGHER beats rate (37.5%) than "likely_similar" (0%)!**

### Surprising Cases

| Commit | Soft Prediction | Actual A vs H |
|--------|-----------------|---------------|
| e3580537 | likely_ineffective, ineffective | **+24.43%** (BEST!) |
| 6a417b86 | likely_ineffective, ineffective | **+7.22%** (beats) |
| bc7c4d20 | likely_ineffective | **+2.03%** (beats) |

### Failure Mode vs Actual Outcome (32 H+A commits)

| Failure Mode | n | Beats | Similar | Worse | Beats % |
|--------------|---|-------|---------|-------|---------|
| not_applicable | 9 | 2 | 3 | 4 | 22% |
| complexity_avoidance | 13 | 2 | 6 | 5 | 15% |
| localization_failure | 9 | 4 | 3 | 2 | **44%** |
| incomplete_implementation | 1 | 1 | 0 | 0 | 100% |

**Counter-intuitive:** "localization_failure" has HIGHEST beats rate (44%)!
**"not_applicable" (no failure detected) has one of the WORST outcome rates.**

### Bottleneck Target vs Actual Outcome (32 H+A commits)

| Bottleneck | n | Beats | Similar | Worse | Beats % |
|------------|---|-------|---------|-------|---------|
| same_target | 14 | 3 | 5 | 6 | 21% |
| related_target | 15 | 6 | 5 | 4 | **40%** |
| different_target | 3 | 0 | 2 | 1 | 0% |

**Counter-intuitive:** "related_target" performs BETTER than "same_target"!

### Approach Comparison vs Actual Outcome (32 H+A commits)

| Approach | n | Beats | Similar | Worse | Beats % |
|----------|---|-------|---------|-------|---------|
| similar_approach | 5 | 1 | 1 | 3 | 20% |
| partial_solution | 11 | 4 | 4 | 3 | 36% |
| valid_alternative | 10 | 3 | 5 | 2 | 30% |
| ineffective | 6 | 2 | 2 | 2 | 33% |

**Key finding:** "similar_approach" has WORST outcome (60% worse rate)!
**"ineffective" approach has same beats rate as "partial_solution"!**

---

## Conclusions

### 1. Hard Metrics Tell the Real Story

- **50.0%** of agent patches either failed to run or performed worse than human patches
- When patches do run, there's high variance (-19% to +24%)
- **1 case** where human patch failed but agent succeeded

### 2. Soft Metrics Have No Predictive Power

| Expected | Actual |
|----------|--------|
| "likely_ineffective" = bad | 37.5% beats rate (BEST!) |
| "likely_similar" = good | 0% beats rate (WORST!) |
| "not_applicable" failure = success | 22% beats, 44% worse |
| "localization_failure" = bad | 44% beats rate (BEST!) |
| "similar_approach" = good | 20% beats, 60% worse |

### 3. Why Soft Metrics Fail

1. **Inverted predictions:** "likely_ineffective" beats human more often than "likely_similar"
2. **Best performer missed:** e3580537 (+24.43%) was marked as "likely_ineffective", "ineffective"
3. **"No failure" predicts failure:** "not_applicable" failure mode has worst outcomes
4. **Static analysis blindspot:** LLM judges assess code patterns, not actual runtime performance
5. **Exact match does not equal good:** "similar_approach" has highest worse rate (60%)

### 4. Implications for Evaluation

1. **Soft metrics should NOT be used alone** to evaluate optimization agent performance
2. **Benchmark-based evaluation is essential** for performance optimization tasks
3. **High soft scores can mask complete failures** - 9/11 failed commits had "likely_partial" or better
4. **Counter-intuitive soft metrics** suggest fundamental issues with LLM-as-judge evaluation for this domain

---

## Appendix: Commit Lists

### 22 Commits with H+A+B (Full Data)
```
015069b0, 22d33bac, 296f927f, 299ebb62, 2deb029d, 30172b49, 310aca88, 3476ed08,
4c822298, 58eee5f2, 6a417b86, 6ce01f30, 70b808fe, 98f47f2a, 9f1710f1, a3223766,
b55ed6ef, b690e348, bc7c4d20, fa63e710, fc542144, fe66b347
```

### 10 Commits with H+A but NO B
```
19d98e0c, 3a243095, 6e36f4fa, 7c01f706, 89a84b0b, 9474e89b, 9badee53, e206b543,
e3580537, fc7b8d1e
```

### 11 Agent Failures (H Data Only)
```
35fad35a, 3b61cb45, 660470e5, 6dd94dbe, 8c1e77fb, 9ed82e70, ad8d696a, ccf02fcb,
ce6bf3a2, d7740ea4, e7b20426
```

### 1 Human Failure (B+A Only)
```
99abb8b6
```

---

*Analysis completed: 2026-01-16*
*Data sources: HuggingFace `Inferencebench/claude-code-vllm-benchmarks` + local soft metrics*
