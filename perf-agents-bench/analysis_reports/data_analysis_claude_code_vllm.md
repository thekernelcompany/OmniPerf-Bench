# Claude Code (Sonnet 4.5) vLLM Performance Analysis

## Executive Summary

This analysis examines Claude Code (sonnet-4.5) performance on 44 vLLM optimization tasks, comparing agent-generated patches against human expert patches using both hard metrics (actual benchmark performance) and soft metrics (qualitative code analysis).

**Key Findings:**
- **47.7%** of tasks performed equal or better than human (9 beats + 12 similar)
- **50.0%** either failed or performed worse (11 worse + 11 agent failures)
- **86%** of agent patches targeted the same or related bottleneck as human
- **72%** targeted the exact same files as human
- Soft metrics provide valuable behavioral insights but do not linearly predict benchmark outcomes

---

## Data Sources

| Source | Description |
|--------|-------------|
| **Hard Metrics** | HuggingFace `Inferencebench/claude-code-vllm-benchmarks` |
| **Soft Metrics** | `state/analysis/vllm/claude_code/sonnet-4.5/vllm_<commit>/patch_quality.json` |
| **Patch Similarity** | `state/analysis/vllm/claude_code/sonnet-4.5/vllm_<commit>/patch_similarity.json` |

### Dataset Composition (44 commits)

| Category | Count | Description |
|----------|-------|-------------|
| H+A+B (full data) | 22 | Can compare Agent vs Human vs Baseline |
| H+A (no baseline) | 10 | Can only compare Agent vs Human |
| Agent failures | 11 | Human patch exists, agent patch failed to run |
| Human failure | 1 | Human patch failed, agent succeeded (99abb8b6) |

**Excluded:** 9 commits with wrong benchmark command (ed250545, 8a4e5c5f, f26c4aee, 6d0734c5, 61b8cea3, cf2f084d, 80aa7e91, ca7a2d5f, 8bc68e19)

---

## Part 1: Hard Metrics Analysis

### Overall Performance (44 Commits)

| Category | Count | Percentage |
|----------|-------|------------|
| Agent beats Human (>2%) | 9 | 20.5% |
| Similar (±2%) | 12 | 27.3% |
| Agent worse (<-2%) | 11 | 25.0% |
| Agent failed (no A data) | 11 | 25.0% |
| Human failed (B+A only) | 1 | 2.3% |

### Breakdown: 32 Commits with Both H+A Data

| Category | Count | Percentage |
|----------|-------|------------|
| Beats (>2%) | 9 | 28.1% |
| Similar (±2%) | 12 | 37.5% |
| Worse (<-2%) | 11 | 34.4% |

### Top Performers (Agent Beats Human)

| Commit | A vs H | Bottleneck | Approach |
|--------|--------|------------|----------|
| e3580537 | **+24.43%** | related_target | ineffective |
| fc7b8d1e | +17.34% | same_target | similar_approach |
| 6e36f4fa | +15.35% | related_target | partial_solution |
| a3223766 | +8.17% | related_target | valid_alternative |
| e206b543 | +7.96% | related_target | valid_alternative |
| 6a417b86 | +7.22% | related_target | ineffective |
| bc7c4d20 | +2.03% | same_target | partial_solution |
| 4c822298 | +2.22% | same_target | valid_alternative |
| 98f47f2a | +2.58% | same_target | similar_approach |

### Worst Performers (Agent Worse Than Human)

| Commit | A vs H | Bottleneck | Approach |
|--------|--------|------------|----------|
| 2deb029d | **-19.21%** | different_target | ineffective |
| 89a84b0b | -16.61% | same_target | partial_solution |
| b690e348 | -16.40% | related_target | ineffective |
| 9474e89b | -7.58% | same_target | similar_approach |
| 3476ed08 | -5.20% | related_target | partial_solution |
| 30172b49 | -4.98% | related_target | partial_solution |

### 11 Agent Failures

Commits where agent patch failed to produce runnable benchmarks:
```
35fad35a, 3b61cb45, 660470e5, 6dd94dbe, 8c1e77fb,
9ed82e70, ad8d696a, ccf02fcb, ce6bf3a2, d7740ea4, e7b20426
```

---

## Part 2: Soft Metrics Analysis

### 2.1 File Targeting

**Question: Does the agent target the same files as the human expert?**

| File Overlap | Count | % |
|--------------|-------|---|
| 100% | 23 | 71.9% |
| 50-99% | 4 | 12.5% |
| <50% | 5 | 15.6% |

**Key insight:** In ~72% of cases, the agent correctly identifies the same files as the human expert.

**Detailed breakdown:**
```
100% overlap (23): 015069b0, 19d98e0c, 22d33bac, 296f927f, 299ebb62, 310aca88,
                   3a243095, 58eee5f2, 6a417b86, 6ce01f30, 6e36f4fa, 70b808fe,
                   7c01f706, 89a84b0b, 98f47f2a, 9badee53, 9f1710f1, a3223766,
                   b55ed6ef, e206b543, fc542144, fc7b8d1e, fe66b347

<100% overlap (9): bc7c4d20 (50%), 9474e89b (50%), 4c822298 (66.7%),
                   2deb029d (33.3%), fa63e710 (33.3%), b690e348 (22.2%),
                   3476ed08 (21.1%), 30172b49 (18.8%), e3580537 (9.1%)
```

### 2.2 Bottleneck Identification

**Question: Does the agent identify the same performance bottleneck as the human?**

| Category | Count | % | Meaning |
|----------|-------|---|---------|
| same_target | 16 | 36.4% | Exact same bottleneck identified |
| related_target | 22 | 50.0% | Related area, different specific bottleneck |
| different_target | 6 | 13.6% | Completely missed the bottleneck |

**86.4% of the time, the agent targets the same or related bottleneck.**

#### Examples by Category

**same_target (Excellent Localization):**
- **fc7b8d1e (+17.34%)**: Both identified redundant work in `BlockSpaceManagerV1`
- **fe66b347 (similar)**: Both identified state_indices_tensor loops as bottleneck
- **2deb029d (-19.21%)**: Both targeted block_manager, but agent missed critical `mark_blocks_as_computed` logic

**related_target (Partial Localization):**
- **e3580537 (+24.43%)**: Human did structural chunked prefill integration; agent did low-level Python cleanup
- **6a417b86 (+7.22%)**: Human added +1 to num_gpu_blocks; agent did property caching
- **19d98e0c (similar)**: Human focused on memory reuse; agent did torch.zeros→torch.empty

**different_target (Missed Localization):**
- **22d33bac**: Human optimized `merge_async_iterators`; agent optimized dict lookups elsewhere
- **fa63e710**: Human removed GPU sync point; agent changed torch.zeros→torch.empty

### 2.3 Agent Behavioral Patterns

**Recurring Strengths (from patch_quality.json):**

1. **Standard PyTorch patterns**: "Correctly identified standard PyTorch optimization patterns (empty vs zeros)"
2. **Clean code practices**: "Clean code practices (caching properties)"
3. **Comprehensive coverage**: "Comprehensive coverage of the file (optimized both batch and streaming methods)"
4. **Correct localization**: "Identified the same critical bottleneck"

**Recurring Weaknesses:**

1. **Over-reliance on example patterns** (8+ commits):
   - 6a417b86: "Over-reliance on the provided 'example' patterns rather than analyzing the specific needs"
   - 30172b49: "Followed the 'example optimization' provided in the prompt literally"
   - fa63e710: "Followed the 'example optimization' provided in the prompt literally"
   - 89a84b0b: "Over-reliance on the provided 'example optimization' (torch.zeros)"

2. **Micro-optimizations instead of architectural changes** (6+ commits):
   - 2deb029d: "Over-focused on micro-optimizations instead of architectural bottlenecks"
   - e3580537: "Optimizations are too low-level to impact the high-level task performance"
   - b690e348: "Trivial initialization changes...do not address the primary performance bottleneck"

3. **Missing domain-specific knowledge**:
   - 6a417b86: "Missed the critical domain-specific optimization (+1 block)"
   - b690e348: "Failed to identify the 'torch.vstack' bottleneck"

### 2.4 Approach Comparison

| Category | Count | % | Meaning |
|----------|-------|---|---------|
| valid_alternative | 17 | 38.6% | Different but valid approach |
| partial_solution | 14 | 31.8% | Partially addresses the problem |
| ineffective | 9 | 20.5% | Doesn't meaningfully address bottleneck |
| similar_approach | 5 | 11.4% | Same approach as human |

### 2.5 Speedup Likelihood Predictions

| Category | Count | % |
|----------|-------|---|
| likely_partial | 27 | 61.4% |
| likely_ineffective | 11 | 25.0% |
| likely_similar | 5 | 11.4% |
| uncertain | 1 | 2.3% |

### 2.6 Failure Mode Distribution

| Category | Count | % | Description |
|----------|-------|---|-------------|
| localization_failure | 15 | 34.1% | Targeted wrong area of code |
| complexity_avoidance | 14 | 31.8% | Avoided complex architectural changes |
| not_applicable | 11 | 25.0% | No significant failure mode |
| incomplete_implementation | 4 | 9.1% | Started but didn't finish |

### 2.7 Soft Metrics for 11 Agent Failures

| Commit | speedup_likelihood | failure_mode | bottleneck_target | approach |
|--------|-------------------|--------------|------------------|----------|
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

**Key insight:** Most failures (9/11) were predicted as "likely_partial" or better - these were valid approaches that failed at runtime, not conceptually broken solutions.

---

## Part 3: Correlation Analysis

### 3.1 Speedup Likelihood vs Actual Outcome

| Soft Prediction | n | Beats | Similar | Worse | Beats % |
|-----------------|---|-------|---------|-------|---------|
| likely_similar | 4 | 0 | 2 | 2 | 0% |
| likely_partial | 20 | 6 | 7 | 7 | 30% |
| likely_ineffective | 8 | 3 | 3 | 2 | **37.5%** |

**Counter-intuitive finding:** "likely_ineffective" has a HIGHER beats rate (37.5%) than "likely_similar" (0%).

### 3.2 Case Studies: Why Predictions Miss

#### e3580537 (+24.43%, marked "likely_ineffective")
- **Soft metrics assessment:** Agent did "low-level Python cleanup" while human did "structural chunked prefill integration"
- **Why agent won:** The benchmark was bottlenecked on code paths the agent optimized, not the structural changes
- **Lesson:** LLM judges assess *approach quality* relative to human, not *benchmark fitness*

#### 2deb029d (-19.21%, marked "likely_ineffective")
- **Soft metrics assessment:** Agent missed `mark_blocks_as_computed` logic
- **Why soft metrics were RIGHT:** The benchmark required prefix caching to work (functional correctness)
- **Lesson:** Soft metrics correctly identified the fundamental miss

#### fc7b8d1e (+17.34%, marked "likely_partial")
- **Soft metrics assessment:** Both identified same bottleneck in BlockSpaceManagerV1
- **Why agent won:** Agent went further by also targeting block freeing and prefix caching
- **Lesson:** "same_target" + agent doing MORE can lead to better outcomes

### 3.3 Bottleneck Target vs Outcome

| Bottleneck | n | Beats | Similar | Worse | Beats % |
|------------|---|-------|---------|-------|---------|
| same_target | 12 | 3 | 4 | 5 | 25% |
| related_target | 17 | 6 | 6 | 5 | **35%** |
| different_target | 3 | 0 | 2 | 1 | **0%** |

**Insights:**
- "related_target" outperforms "same_target" because agent may find optimizations human missed
- **"different_target" is a reliable negative signal:** 0% beats rate when agent completely misses bottleneck

### 3.4 File Overlap vs Outcome

| File Overlap | n | Beats | Similar | Worse | Beats % |
|--------------|---|-------|---------|-------|---------|
| 100% | 23 | 5 | 9 | 9 | 22% |
| <100% | 9 | 4 | 3 | 2 | **44%** |

**Counter-intuitive finding:** Lower file overlap correlates with HIGHER beats rate.

**Explanation:** When agent diverges from human file targets, it may find alternative optimization paths that work better for the specific benchmark. The best performer (e3580537, +24.43%) had only 9.1% file overlap.

### 3.5 Failure Mode vs Outcome

| Failure Mode | n | Beats | Similar | Worse | Beats % |
|--------------|---|-------|---------|-------|---------|
| not_applicable | 9 | 2 | 3 | 4 | 22% |
| complexity_avoidance | 12 | 2 | 6 | 4 | 17% |
| localization_failure | 10 | 4 | 3 | 3 | **40%** |
| incomplete_implementation | 1 | 1 | 0 | 0 | 100% |

**Interpretation:**
- "not_applicable" (no failure) has worst outcomes - agent matched human approach but executed worse
- "localization_failure" has high beats rate - agent's "wrong" target might be right for the benchmark
- "complexity_avoidance" → 50% similar rate - agent plays it safe

---

## Key Takeaways

### What Soft Metrics ARE Good For

1. **Understanding agent behavior** - 86% target same or related bottleneck
2. **Assessing file localization** - 72% target exact same files
3. **Identifying recurring patterns** - Consistent strengths and weaknesses
4. **Qualitative diagnosis** - WHY agent made certain choices

### What Soft Metrics DO NOT Predict

1. Actual benchmark performance (no linear correlation)
2. Whether agent will beat human on specific workload
3. Runtime impact of different optimization strategies

### Actionable Recommendations

1. **For deployment:** Do not use soft metrics as sole predictor - always run benchmarks

2. **For agent improvement:** Address identified weaknesses:
   - Reduce over-reliance on example patterns in prompts
   - Encourage architectural thinking over micro-optimizations
   - Improve domain-specific knowledge (e.g., Neuron backend requirements)

3. **For triage:** "different_target" is a reliable warning sign (0% beats rate)

4. **For understanding:** Soft metrics valuable for post-hoc analysis of WHY agent patches performed as they did

---

## Appendix: Commit Lists

### 32 Commits with H+A Data
```
015069b0, 19d98e0c, 22d33bac, 296f927f, 299ebb62, 2deb029d, 30172b49, 310aca88,
3476ed08, 3a243095, 4c822298, 58eee5f2, 6a417b86, 6ce01f30, 6e36f4fa, 70b808fe,
7c01f706, 89a84b0b, 9474e89b, 98f47f2a, 9badee53, 9f1710f1, a3223766, b55ed6ef,
b690e348, bc7c4d20, e206b543, e3580537, fa63e710, fc542144, fc7b8d1e, fe66b347
```

### 22 Commits with H+A+B (Full Data)
```
015069b0, 22d33bac, 296f927f, 299ebb62, 2deb029d, 30172b49, 310aca88, 3476ed08,
4c822298, 58eee5f2, 6a417b86, 6ce01f30, 70b808fe, 98f47f2a, 9f1710f1, a3223766,
b55ed6ef, b690e348, bc7c4d20, fa63e710, fc542144, fe66b347
```

### 11 Agent Failures
```
35fad35a, 3b61cb45, 660470e5, 6dd94dbe, 8c1e77fb, 9ed82e70, ad8d696a, ccf02fcb,
ce6bf3a2, d7740ea4, e7b20426
```

### 9 Excluded Commits (Wrong Benchmark Command)
```
ed250545, 8a4e5c5f, f26c4aee, 6d0734c5, 61b8cea3, cf2f084d, 80aa7e91, ca7a2d5f, 8bc68e19
```
