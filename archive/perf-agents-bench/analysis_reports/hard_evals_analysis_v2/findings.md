# OmniPerf-Bench: Detailed Findings

**Generated:** 2026-01-14
**Dataset:** vLLM performance optimization commits
**Agent:** Claude Code

---

## Overview

| Metric | Value |
|--------|-------|
| Total runs analyzed | 96 |
| Soft metric coverage | 100% (96/96) |
| Hard metric coverage | 44% (42/96) |
| Matched for correlation | 39 commits |

---

## Finding 1: Success Rate — 18.8% (18/96)

**What it measures:** How many agent runs produced optimizations that the LLM judge considered "successful" (failure_mode = "not_applicable").

**What counts as success:**
- Agent identified a valid performance bottleneck (same or valid alternative to human's target)
- Agent implemented an approach that would plausibly improve performance
- The patch is coherent and doesn't break things

**What counts as failure:**
- Optimized wrong code (localization failure)
- Made harmful/regressive changes
- Made superficial changes that won't help
- Over-complicated or destroyed the codebase

**Interpretation:** An 81.2% failure rate means performance optimization is genuinely hard for agents. This isn't a trivial benchmark where agents get 50%+ by default.

---

## Finding 2: Prediction Accuracy — 66.7% (26/39)

This is the **core correlation finding** between soft and hard metrics.

### How It Works

The soft metric `speedup_likelihood` predicts what will happen at runtime:

| Prediction | What it predicts | Correct when |
|------------|------------------|--------------|
| likely_similar | Agent matches human | diff ≥ -2% |
| likely_partial | Agent shows improvement | diff > +2% |
| likely_ineffective | Agent won't improve | diff ≤ +2% |
| likely_regression | Agent will be worse | diff < -2% |

We compare these predictions against actual `agent_vs_human_pct` from runtime benchmarks.

### Accuracy Breakdown

| Prediction | Correct | Total | Accuracy | Interpretation |
|------------|---------|-------|----------|----------------|
| likely_similar | 7 | 7 | **100%** | When LLM says "agent matches human", it's ALWAYS right |
| likely_ineffective | 15 | 18 | **83%** | When LLM says "won't help", it's right 83% of the time |
| likely_regression | 4 | 8 | **50%** | Regression prediction is coin flip |
| likely_partial | 0 | 5 | **0%** | "Partial success" predictions are NEVER right |

### Key Insights

**1. High-confidence predictions are reliable:**

The two confident categories (`likely_similar` + `likely_ineffective`) cover 25/39 = 64% of predictions and are correct 22/25 = **88% of the time**.

| Prediction | Trust Level | Action |
|------------|-------------|--------|
| likely_similar | HIGH | Trust it — agent succeeded |
| likely_ineffective | HIGH | Trust it — agent failed |
| likely_partial | LOW | Need hard metrics to confirm |
| likely_regression | LOW | Need hard metrics to confirm |

**2. The "likely_partial" disaster:**

All 5 "likely_partial" predictions were wrong. The actual outcomes:

| Commit | Predicted | Actual Outcome |
|--------|-----------|----------------|
| 8a4e5c5f3c1d | partial success | +1.7% (marginal) |
| 70b808fe1a63 | partial success | +0.9% (marginal) |
| 8bc68e198c4c | partial success | -1.2% (slight regression) |
| b690e34824fd | partial success | -5.6% (regression) |
| 89a84b0bb7b3 | partial success | -16.6% (significant regression) |

The LLM cannot distinguish between "almost succeeded" and "significantly worse."

**3. False negatives — LLM was too pessimistic:**

Some commits predicted as "likely_regression" actually outperformed:

| Commit | Predicted | Actual |
|--------|-----------|--------|
| e3580537a41a | regression | **+24.4%** |
| fc7b8d1eefcb | regression | **+17.3%** |
| 6e36f4fa6ce6 | regression | **+15.4%** |

The agent's approach looked wrong on paper but worked better at runtime.

**Practical Implication:** Soft metrics are excellent for screening "definitely failed" and "definitely succeeded" cases, but unreliable for edge cases.

---

## Finding 3: GSO Failure Categories (78 failures)

Following the GSO paper (arxiv:2505.23671) methodology, we categorize **why** agents fail.

### High-Level Distribution

| Category | Count | % | Description |
|----------|-------|---|-------------|
| **Localization** | 51 | 65.4% | Agent optimized the WRONG code |
| **Mismanage Compute** | 17 | 21.8% | Wasted resources or caused damage |
| **Avoid Complexity** | 10 | 12.8% | Avoided necessary work |

### Sub-Category Breakdown

**Localization (65.4%)**

| Sub-Category | Count | % | Description |
|--------------|-------|---|-------------|
| Misdiagnosed Bottlenecks | 51 | 65.4% | Optimized wrong component; not the critical path |

Example: Task is to optimize attention computation, agent optimizes tokenizer instead.

**Mismanage Compute (21.8%)**

| Sub-Category | Count | % | Description |
|--------------|-------|---|-------------|
| Destructive/Runaway | 9 | 11.5% | Corrupted repo, deleted files, made unbuildable |
| Exploit-Heavy | 5 | 6.4% | Over-engineered with unnecessary complexity |
| Explore-Heavy | 3 | 3.8% | Spent time reading code without making useful changes |

**Avoid Complexity (12.8%)**

| Sub-Category | Count | % | Description |
|--------------|-------|---|-------------|
| Wrong Abstraction Level | 7 | 9.0% | Stayed in Python when CUDA/C++ was needed |
| Lazy Optimization | 3 | 3.8% | Made superficial changes instead of deep fixes |

### The 65% Localization Insight

This is the most important finding:

1. Agents CAN write valid optimization code
2. Agents CAN understand what performance optimization means
3. **But agents CANNOT reliably identify WHERE the bottleneck is**

This suggests the highest-impact improvement for agents would be better profiling integration or bottleneck reasoning, not better code generation.

---

## Finding 4: Success Pattern Analysis (18 successes)

When agents succeed, how do their solutions compare to humans?

### Implementation Alignment

| Category | Count | % | Description |
|----------|-------|---|-------------|
| identical | 8 | 44.4% | Agent produced functionally same code as human |
| core_match_extras | 10 | 55.6% | Got core optimization right, but added extra noise |

**Interpretation:** Even when successful:
- Only 44% match human elegance
- 56% add unnecessary complexity

### Agent Behavior Patterns

| Pattern | Count | % | Description |
|---------|-------|---|-------------|
| diff_pollution | 7 | 38.9% | Included unrelated CI/CD, docs, config changes |
| over_engineering | 6 | 33.3% | Used 2+ more techniques than needed |
| clean_patch | 5 | 27.8% | Surgical, minimal changes (ideal) |
| scope_creep | 4 | 22.2% | Modified files beyond necessary scope |

**Key insight:** Only **28% produce clean patches**. The majority of successful agents still exhibit problematic behaviors — they throw "everything at the wall" hoping something sticks.

### Technique Match

| Match Type | Count | % |
|------------|-------|---|
| exact_match | 10 | 55.6% |
| agent_superset | 8 | 44.4% |

Agents NEVER use a subset of human techniques. They either match exactly (56%) or use **more** techniques (44%). This confirms the "kitchen sink" approach.

---

## Finding 5: LLM Scores (1-10 scale)

| Dimension | Score |
|-----------|-------|
| Code Understanding | 8.04 |
| Task Alignment | 7.81 |
| Approach Quality | 7.50 |
| Execution Quality | 6.68 |
| Overall | 7.51 |

### The Understanding-Execution Gap

There's a **1.36 point gap** between Code Understanding (8.04) and Execution Quality (6.68).

This means:
- Agents understand the codebase well
- Agents understand what they're supposed to do
- **But they struggle to execute the solution correctly**

This aligns with the localization finding — understanding isn't the problem, correct action is.

---

## Finding 6: Patch Similarity

| Metric | Value |
|--------|-------|
| Avg File Overlap | 12.6% |
| Avg Line Overlap | 5.3% |
| Technique Overlap Rate | 43.8% |

**Interpretation:**

Agent patches are structurally very different from human patches:
- Only 12.6% of files touched are the same
- Only 5.3% of specific lines overlap
- Less than half use the same optimization techniques

This quantifies the localization failure — agents are literally working on different parts of the codebase than humans.

---

## Finding 7: Technique Overlap Correlation

| Overlap | Count | Avg Diff |
|---------|-------|----------|
| technique_overlap=True | 15 | -0.30% |
| technique_overlap=False | 24 | +1.57% |

**Counterintuitive result:** When agents use *different* techniques than humans, they perform slightly *better* on average.

Possible explanations:
1. Statistical noise (small sample size)
2. Selection bias (agents that deviate might be more "creative")
3. Human solutions aren't always optimal
4. Measurement artifact

---

## Finding 8: Domain Distribution (Successes)

| Domain | Count | % |
|--------|-------|---|
| Memory optimization | 9 | 50.0% |
| Compute optimization | 7 | 38.9% |
| Concurrency | 1 | 5.6% |
| Other | 1 | 5.6% |

Agents appear to be better at memory optimizations than concurrency problems.

---

## Finding 9: Speedup Likelihood Distribution (All 96 runs)

| Category | Count | % |
|----------|-------|---|
| likely_ineffective | 41 | 42.7% |
| likely_similar | 23 | 24.0% |
| likely_regression | 21 | 21.9% |
| likely_partial | 9 | 9.4% |
| uncertain | 2 | 2.1% |

**Reading this:**
- 42.7% of patches are judged as "won't help at all"
- Only 24% are judged as matching human quality
- 22% are judged as making things worse
- 9.4% are judged as "partial" (but we know this category is unreliable)

The predicted success rate (likely_similar + likely_partial) is 33.4%, but the actual success rate is 18.8%. This suggests the soft metrics are slightly optimistic overall.

---

## Summary: The Capability Gap

**Agents CAN:**
- Understand code (8.04/10)
- Understand tasks (7.81/10)
- Write valid optimization code

**Agents CANNOT:**
- Reliably identify bottlenecks (65% localization failure)
- Produce minimal, clean patches (only 28%)
- Work at low abstraction levels (9% wrong level)

---

## Summary: Why Both Metrics Matter

| Aspect | Soft Metrics | Hard Metrics |
|--------|--------------|--------------|
| Coverage | 100% | 44% |
| Speed | Seconds | Minutes |
| Explains WHY | Yes (failure categories) | No |
| Measures HOW MUCH | No | Yes (±X% performance) |
| Objective | No (LLM judgment) | Yes (runtime) |

**Correlation:** 67% overall, but:
- 100% for detecting full success (likely_similar)
- 83% for detecting failure (likely_ineffective)
- Unreliable for edge cases (likely_partial, likely_regression)

**Use soft metrics for:** Screening, understanding failure modes, high-coverage evaluation
**Use hard metrics for:** Validation, quantifying impact, ground truth
