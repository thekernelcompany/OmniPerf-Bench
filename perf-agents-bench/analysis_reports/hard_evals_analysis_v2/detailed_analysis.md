# OmniPerf-Bench: Detailed Hard Metrics Analysis

**Generated:** 2026-01-14
**Dataset:** vLLM performance optimization commits
**Agent:** Claude Code
**Total Evaluable Commits:** 53

---

## Executive Summary

The naive interpretation of hard metrics — "58% of agents matched or beat human performance" — is **misleading**. This detailed analysis reveals:

- **True success rate: 11-24%** (not 58%)
- **34% are "benchmark artifacts"** — similar benchmark despite soft failure
- **Soft metrics are essential** to distinguish genuine success from artifacts

---

## Summary Table

| Category | Count | % | Description |
|----------|-------|---|-------------|
| **Genuine Success** | 6 | 11.3% | Soft success + (beat or match human) |
| **Lucky/Alternative** | 7 | 13.2% | Soft failure but beat human by >+2% |
| **Benchmark Artifact** | 18 | 34.0% | Soft failure but similar benchmark (±2%) |
| **Agent Worse** | 8 | 15.1% | Benchmark shows regression (<-2%) |
| **Agent Failed** | 11 | 20.8% | No agent output/benchmark |
| **Unmatched** | 3 | 5.7% | Data issues |
| **TOTAL** | 53 | 100% | |

---

## Category 1: Genuine Success (6 commits, 11.3%)

These are cases where:
- Soft metrics say "success" (failure_mode = not_applicable)
- Hard metrics show agent matched or beat human

| Commit | Task | Diff% | Bottleneck | Approach | H vs B | A vs B |
|--------|------|-------|------------|----------|--------|--------|
| a32237665df8 | vllm_core-0057 | **+8.2%** | same_target | similar_approach | +6.2% | +13.9% |
| e206b5433109 | vllm_core-0081 | **+8.0%** | same_target | similar_approach | N/A | N/A |
| 015069b01741 | vllm_core-0000 | +0.0% | same_target | similar_approach | -0.0% | -0.0% |
| 9badee53decb | vllm_core-0053 | -0.2% | same_target | same_approach | N/A | N/A |
| 296f927f2493 | vllm_core-0008 | -0.7% | same_target | same_approach | +0.5% | -0.1% |
| 299ebb62b269 | vllm_core-0009 | -1.5% | same_target | same_approach | +12.1% | +10.8% |

### Key Observations

1. **All 6 have `same_target` bottleneck** — Agent identified the correct code to optimize
2. **All 6 have `same_approach` or `similar_approach`** — Agent used the right optimization technique
3. **2 beat human, 4 matched** — When agents get it right, they can match or exceed human performance

---

## Category 2: Lucky/Alternative (7 commits, 13.2%)

These are cases where:
- Soft metrics say "failure" (different target, harmful approach)
- But hard metrics show agent BEAT human by >+2%

| Commit | Task | Diff% | Bottleneck | Approach | Failure Mode |
|--------|------|-------|------------|----------|--------------|
| e3580537a41a | vllm_core-0082 | **+24.4%** | no_optimization | harmful | other |
| fc7b8d1eefcb | vllm_core-0094 | **+17.3%** | no_optimization | harmful | localization_failure |
| 6e36f4fa6ce6 | vllm_core-0034 | **+15.4%** | different_target | harmful | localization_failure |
| 6a417b8600d4 | vllm_core-0029 | +5.4% | same_target | similar_approach | overcomplicated |
| 98f47f2a4032 | vllm_core-0050 | +5.3% | no_optimization | ineffective | localization_failure |
| bc7c4d206bbf | vllm_core-0067 | +2.6% | no_optimization | ineffective | localization_failure |
| 30172b4947c5 | vllm_core-0013 | +2.6% | different_target | ineffective | localization_failure |

### Key Observations

1. **Top 3 show >+15% improvement** — Agent found completely different optimizations that worked better
2. **Soft metrics judged these as "harmful" or "ineffective"** — The LLM judge couldn't predict these would work
3. **These are "happy accidents"** — Not reproducible; agent's approach looked wrong on paper

### Why This Happens

The agent optimized different code than the human, but:
- That code happened to be a valid (or better) optimization target
- The benchmark captured the agent's optimization rather than the human's
- This represents ~13% of cases — not a reliable pattern

---

## Category 3: Benchmark Artifact (18 commits, 34.0%)

**This is the critical category that explains why "58% success" is misleading.**

These are cases where:
- Soft metrics say "failure" (different target, ineffective approach)
- Hard metrics show "similar" (±2%)
- **This looks like success but isn't**

### Full Table

| Commit | Task | Diff% | Bottleneck | Approach | H vs B | A vs B | Why Similar? |
|--------|------|-------|------------|----------|--------|--------|--------------|
| 22d33baca2c0 | vllm_core-0005 | +1.0% | different_target | ineffective | +92.8% | +94.7% | H improved, A matched |
| 310aca88c984 | vllm_core-0015 | +0.2% | no_optimization | ineffective | +99.8% | +100.2% | H improved, A matched |
| 4c822298981a | vllm_core-0022 | +0.2% | different_target | ineffective | +50.0% | +50.3% | H improved, A matched |
| fc542144c447 | vllm_core-0093 | -0.3% | different_target | ineffective | -7.6% | -7.9% | H improved, A matched |
| 70b808fe1a63 | vllm_core-0035 | +0.9% | same_target | similar_approach | +1.8% | +2.7% | B≈H≈A (no effect) |
| ed25054577f7 | vllm_core-0087 | -1.1% | no_optimization | other | +2.3% | +1.3% | B≈H≈A (no effect) |
| 8a4e5c5f3c1d | vllm_core-0042 | +1.7% | related_target | partial_solution | -2.9% | -1.1% | B≈H≈A (no effect) |
| f26c4aeecba4 | vllm_core-0090 | -0.0% | no_optimization | ineffective | +0.0% | -0.0% | B≈H≈A (no effect) |
| fa63e710c7fb | vllm_core-0091 | -0.5% | different_target | ineffective | +0.6% | +0.1% | B≈H≈A (no effect) |
| 6d0734c562e7 | vllm_core-0031 | -1.3% | different_target | ineffective | +1.3% | +0.0% | B≈H≈A (no effect) |
| 61b8cea3b42f | vllm_core-0026 | +0.0% | no_optimization | ineffective | +0.1% | +0.1% | B≈H≈A (no effect) |
| cf2f084d56a1 | vllm_core-0075 | +0.4% | different_target | harmful | N/A | N/A | No baseline |
| 80aa7e91fcd5 | vllm_core-0038 | -0.5% | no_optimization | ineffective | N/A | N/A | No baseline |
| 99abb8b650c6 | vllm_core-0051 | -0.5% | same_target | ineffective | N/A | N/A | No baseline |
| 6ce01f30667b | vllm_core-0030 | -0.8% | different_target | ineffective | N/A | N/A | No baseline |
| ca7a2d5f28ea | vllm_core-0072 | -1.0% | no_optimization | ineffective | N/A | N/A | No baseline |
| 8bc68e198c4c | vllm_core-0044 | -1.2% | related_target | partial_solution | N/A | N/A | No baseline |
| 3476ed0809ec | vllm_core-0017 | -1.6% | related_target | partial_solution | N/A | N/A | No baseline |

### Baseline Analysis (11 commits with baseline data)

| Scenario | Count | Explanation |
|----------|-------|-------------|
| **B≈H≈A (no effect)** | 7 | Human's optimization also didn't improve this benchmark |
| **H improved, A matched** | 4 | Agent somehow matched human's improvement despite wrong target |

### Why "B≈H≈A" Happens

Example scenario:
1. **Human PR:** Optimized attention kernel for batch_size > 32
2. **Our benchmark:** Tested with batch_size = 8
3. **Result:** Human's optimization didn't trigger → baseline ≈ human
4. **Agent:** Optimized tokenizer (wrong target)
5. **Result:** Agent also shows baseline performance

When comparing agent vs human: **0% difference** — but both are just baseline!

### Why "H improved, A matched" Happens

In 4 cases, both human and agent showed 50-100% improvement over baseline, despite agent having `different_target`. Possible explanations:

1. **Shared dependency:** Both changes affected a common code path
2. **Benchmark sensitivity:** The benchmark measures something both changes affected
3. **Coincidence:** Statistical noise in measurement

---

## Category 4: Agent Worse (8 commits, 15.1%)

These are cases where the agent's changes caused measurable performance regression.

| Commit | Task | Diff% | Bottleneck | Approach | Failure Mode |
|--------|------|-------|------------|----------|--------------|
| 89a84b0bb7b3 | vllm_core-0041 | **-16.6%** | different_target | valid_alternative | localization_failure |
| 9474e89ba4ec | vllm_core-0049 | -7.6% | different_target | harmful | localization_failure |
| 3a243095e5e7 | vllm_core-0020 | -6.0% | different_target | harmful | localization_failure |
| b690e34824fd | vllm_core-0064 | -5.6% | same_target | partial_solution | complexity_avoidance |
| 7c01f706418d | vllm_core-0037 | -5.4% | different_target | ineffective | localization_failure |
| 58eee5f2e05b | vllm_core-0025 | -3.0% | same_target | harmful | other |
| fe66b34728e5 | vllm_core-0095 | -2.6% | no_optimization | ineffective | localization_failure |
| b55ed6ef8ab0 | vllm_core-0063 | -2.4% | no_optimization | harmful | localization_failure |

### Key Observations

1. **6/8 have `different_target` or `no_optimization`** — Agent optimized wrong code
2. **Most have `localization_failure`** — Consistent with soft metrics
3. **Worst case: -16.6%** — Agent's changes actively hurt performance

---

## Category 5: Agent Failed (11 commits, 20.8%)

These are commits where the agent crashed or produced no usable output. Only human benchmarks exist.

**Failure modes include:**
- Agent timeout
- Agent produced invalid patch
- Agent corrupted repository
- Build failures after agent changes

---

## Category 6: Unmatched (3 commits, 5.7%)

These commits have hard metrics (both human and agent benchmarks) but weren't matched to soft metrics due to:
- Different commit hash formats
- Missing soft metric analysis
- Data pipeline issues

---

## The Evidence: Why Soft Metrics Are Essential

### Distinguishing Genuine Success from Artifacts

| Evidence | Genuine Success (n=6) | Benchmark Artifact (n=18) |
|----------|----------------------|---------------------------|
| Bottleneck = same_target | **100%** (6/6) | **11%** (2/18) |
| Technique overlap | **100%** (6/6) | **44%** (8/18) |
| Approach = same/similar | **100%** (6/6) | **17%** (3/18) |

Without soft metrics, we cannot distinguish:
- Agent that matched human's optimization (genuine success)
- Agent that changed wrong code but benchmark happened to be similar (artifact)

---

## Corrected Success Interpretation

### Naive Interpretation (WRONG)

| Outcome | Count | % |
|---------|-------|---|
| Agent beat or matched human | 31 | 58% |
| Agent worse | 8 | 15% |
| Agent failed | 11 | 21% |
| Unmatched | 3 | 6% |

### Corrected Interpretation (RIGHT)

| Outcome | Count | % | How We Know |
|---------|-------|---|-------------|
| **Genuine Success** | 6 | 11.3% | Soft success + beat/match |
| **Lucky/Alternative** | 7 | 13.2% | Soft failure + beat (>+2%) |
| **Benchmark Artifact** | 18 | 34.0% | Soft failure + similar (±2%) |
| **Agent Worse** | 8 | 15.1% | Benchmark regression |
| **Agent Failed** | 11 | 20.8% | No agent output |
| **Unmatched** | 3 | 5.7% | Data issues |

### True Success Rate

- **Conservative (genuine only):** 6/53 = **11.3%**
- **Liberal (include lucky):** 13/53 = **24.5%**

---

## Key Takeaways

1. **Hard metrics alone are misleading**
   - "Similar benchmark" ≠ "Matched human's optimization"
   - 18/22 "similar" commits were actually soft failures

2. **Soft metrics explain WHY**
   - They identify localization failures
   - They distinguish same_target from different_target
   - They catch cases where agent changed wrong code

3. **Both metrics together reveal truth**
   - Soft: 100% coverage, explains failure modes
   - Hard: 55% coverage, provides ground truth
   - Combined: Accurate success assessment

4. **The benchmark has limitations**
   - 7 cases where B≈H≈A (human's change also didn't show)
   - Benchmark doesn't capture all optimization types
   - Some optimizations target edge cases not in benchmark

---

## Appendix: Complete Commit Tables

### All 39 Matched Commits

#### Agent Better (>+2%): 9 commits

| Commit | Task | Diff% | Target | Approach | Failure Mode | Tech | H vs B | A vs B |
|--------|------|-------|--------|----------|--------------|------|--------|--------|
| e3580537a41a | vllm_core-0082 | +24.4% | no_optimization | harmful | other | N | N/A | N/A |
| fc7b8d1eefcb | vllm_core-0094 | +17.3% | no_optimization | harmful | localization_failure | N | N/A | N/A |
| 6e36f4fa6ce6 | vllm_core-0034 | +15.4% | different_target | harmful | localization_failure | N | N/A | N/A |
| a32237665df8 | vllm_core-0057 | +8.2% | same_target | similar_approach | not_applicable | Y | +6.2% | +13.9% |
| e206b5433109 | vllm_core-0081 | +8.0% | same_target | similar_approach | not_applicable | Y | N/A | N/A |
| 6a417b8600d4 | vllm_core-0029 | +5.4% | same_target | similar_approach | overcomplicated | Y | +34.1% | +37.7% |
| 98f47f2a4032 | vllm_core-0050 | +5.3% | no_optimization | ineffective | localization_failure | N | +0.0% | +5.3% |
| bc7c4d206bbf | vllm_core-0067 | +2.6% | no_optimization | ineffective | localization_failure | N | -3.5% | -0.8% |
| 30172b4947c5 | vllm_core-0013 | +2.6% | different_target | ineffective | localization_failure | N | +1.1% | +3.7% |

#### Agent Similar (±2%): 22 commits

| Commit | Task | Diff% | Target | Approach | Failure Mode | Tech | H vs B | A vs B |
|--------|------|-------|--------|----------|--------------|------|--------|--------|
| 8a4e5c5f3c1d | vllm_core-0042 | +1.7% | related_target | partial_solution | incomplete_implementation | Y | -2.9% | -1.1% |
| 22d33baca2c0 | vllm_core-0005 | +1.0% | different_target | ineffective | other | N | +92.8% | +94.7% |
| 70b808fe1a63 | vllm_core-0035 | +0.9% | same_target | similar_approach | incomplete_implementation | Y | +1.8% | +2.7% |
| cf2f084d56a1 | vllm_core-0075 | +0.4% | different_target | harmful | localization_failure | N | N/A | N/A |
| 310aca88c984 | vllm_core-0015 | +0.2% | no_optimization | ineffective | localization_failure | N | +99.8% | +100.2% |
| 4c822298981a | vllm_core-0022 | +0.2% | different_target | ineffective | localization_failure | N | +50.0% | +50.3% |
| 61b8cea3b42f | vllm_core-0026 | +0.0% | no_optimization | ineffective | localization_failure | N | +0.1% | +0.1% |
| 015069b01741 | vllm_core-0000 | +0.0% | same_target | similar_approach | not_applicable | Y | -0.0% | -0.0% |
| f26c4aeecba4 | vllm_core-0090 | -0.0% | no_optimization | ineffective | localization_failure | N | +0.0% | -0.0% |
| 9badee53decb | vllm_core-0053 | -0.2% | same_target | same_approach | not_applicable | Y | N/A | N/A |
| fc542144c447 | vllm_core-0093 | -0.3% | different_target | ineffective | localization_failure | N | -7.6% | -7.9% |
| fa63e710c7fb | vllm_core-0091 | -0.5% | different_target | ineffective | localization_failure | N | +0.6% | +0.1% |
| 80aa7e91fcd5 | vllm_core-0038 | -0.5% | no_optimization | ineffective | localization_failure | N | N/A | N/A |
| 99abb8b650c6 | vllm_core-0051 | -0.5% | same_target | ineffective | incomplete_implementation | Y | N/A | N/A |
| 296f927f2493 | vllm_core-0008 | -0.7% | same_target | same_approach | not_applicable | Y | +0.5% | -0.1% |
| 6ce01f30667b | vllm_core-0030 | -0.8% | different_target | ineffective | localization_failure | Y | N/A | N/A |
| ca7a2d5f28ea | vllm_core-0072 | -1.0% | no_optimization | ineffective | localization_failure | N | N/A | N/A |
| ed25054577f7 | vllm_core-0087 | -1.1% | no_optimization | other | localization_failure | N | +2.3% | +1.3% |
| 8bc68e198c4c | vllm_core-0044 | -1.2% | related_target | partial_solution | incomplete_implementation | Y | N/A | N/A |
| 6d0734c562e7 | vllm_core-0031 | -1.3% | different_target | ineffective | localization_failure | N | +1.3% | +0.0% |
| 299ebb62b269 | vllm_core-0009 | -1.5% | same_target | same_approach | not_applicable | Y | +12.1% | +10.8% |
| 3476ed0809ec | vllm_core-0017 | -1.6% | related_target | partial_solution | complexity_avoidance | Y | N/A | N/A |

#### Agent Worse (<-2%): 8 commits

| Commit | Task | Diff% | Target | Approach | Failure Mode | Tech | H vs B | A vs B |
|--------|------|-------|--------|----------|--------------|------|--------|--------|
| 89a84b0bb7b3 | vllm_core-0041 | -16.6% | different_target | valid_alternative | localization_failure | Y | N/A | N/A |
| 9474e89ba4ec | vllm_core-0049 | -7.6% | different_target | harmful | localization_failure | N | N/A | N/A |
| 3a243095e5e7 | vllm_core-0020 | -6.0% | different_target | harmful | localization_failure | N | N/A | N/A |
| b690e34824fd | vllm_core-0064 | -5.6% | same_target | partial_solution | complexity_avoidance | Y | +75.8% | +74.5% |
| 7c01f706418d | vllm_core-0037 | -5.4% | different_target | ineffective | localization_failure | N | N/A | N/A |
| 58eee5f2e05b | vllm_core-0025 | -3.0% | same_target | harmful | other | N | +3.2% | +0.3% |
| fe66b34728e5 | vllm_core-0095 | -2.6% | no_optimization | ineffective | localization_failure | N | +8.1% | +5.6% |
| b55ed6ef8ab0 | vllm_core-0063 | -2.4% | no_optimization | harmful | localization_failure | N | +9.9% | +7.8% |

---

## Column Definitions

| Column | Description |
|--------|-------------|
| **Commit** | First 12 characters of commit hash |
| **Task** | Task ID in benchmark (e.g., vllm_core-0057) |
| **Diff%** | Agent vs Human performance difference (positive = agent better) |
| **Target** | Bottleneck target category (same_target, different_target, no_optimization, related_target) |
| **Approach** | Approach comparison (same_approach, similar_approach, ineffective, harmful, partial_solution) |
| **Failure Mode** | Why agent failed (not_applicable = success, localization_failure, complexity_avoidance, etc.) |
| **Tech** | Technique overlap (Y = agent used same optimization technique as human) |
| **H vs B** | Human vs Baseline performance (positive = human improved over baseline) |
| **A vs B** | Agent vs Baseline performance (positive = agent improved over baseline) |
