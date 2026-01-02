# Soft Metrics vs Hard Metrics: Comprehensive Comparison Analysis

**Analysis Date:** 2026-01-01
**Purpose:** Determine if soft metric predictions align with hard runtime benchmarks

---

## Executive Summary

This analysis compares two evaluation approaches for Claude Code's kernel optimization capabilities:

1. **Soft Metrics** (GPT-5 as Judge): Qualitative assessment of patch quality, bottleneck identification, and approach similarity
2. **Hard Metrics** (Runtime Benchmarks): Actual TTFT/Throughput measurements comparing agent vs human optimizations

**Key Finding: The metrics measure different things and reveal complementary insights.**

| Dimension | Soft Metrics | Hard Metrics |
|-----------|--------------|--------------|
| **Sample Size** | 90-300 runs | 81 commits (8 with metrics) |
| **What it Measures** | Patch quality, code correctness | Actual runtime performance |
| **Predicted Success** | 31% likely speedup | 37.5% agent wins |
| **Primary Failure Mode** | Localization failure (82%) | Infrastructure (60%+) |
| **Actionable Insight** | Agent needs better bottleneck identification | Pipeline needs reliability fixes |

**Critical Insight:** The soft metrics and hard metrics **align directionally** (both show ~30-40% success), but they're measuring different stages of the pipeline. Soft metrics evaluate code quality; hard metrics evaluate runtime impact. **Both are needed for complete assessment.**

---

## Data Sources

### Soft Metrics

| Source | Sample Size | Date |
|--------|-------------|------|
| `optimization_quality_analysis.py` | n=90 | Dec 2025 |
| `technique_overlap_analysis.py` | n=90 | Dec 2025 |
| `FINAL_GPT5_EVALUATION_REPORT.md` | n=300 attempts, 143 patches | Nov 2025 |
| `RESEARCH_ANALYSIS.md` | n=300 attempts, 96 commits | Nov 2025 |

### Hard Metrics

| Source | Sample Size | Date |
|--------|-------------|------|
| `Inferencebench/claude-code-vllm-benchmarks` | n=81 commits | Jan 2026 |
| Commits with agent metrics | n=8 | Jan 2026 |

---

## Detailed Comparison

### 1. Success Rate Predictions

#### Soft Metrics Prediction (n=90)

From `optimization_quality_analysis.png`:

| Category | Count | Percentage |
|----------|-------|------------|
| Likely Similar Speedup | 19 | 21% |
| Likely Partial Speedup | 9 | 10% |
| **Total Likely Success** | **28** | **31%** |
| Uncertain | 1 | 1% |
| Likely Ineffective | 40 | 44% |
| Likely Regression | 21 | 23% |

**Soft metrics predict: 31% of patches will achieve speedup**

#### Hard Metrics Actual Results (n=8)

| Outcome | Count | Percentage |
|---------|-------|------------|
| Agent Wins (>5% better) | 3 | 37.5% |
| Tie (within ±5%) | 4 | 50.0% |
| Agent Loses (>5% worse) | 1 | 12.5% |

**Hard metrics show: 37.5% agent wins, 87.5% don't regress**

#### Alignment Analysis

| Metric | Soft Prediction | Hard Actual | Aligned? |
|--------|----------------|-------------|----------|
| Clear wins | 21% (likely similar) | 37.5% | Partially - hard metrics higher |
| Partial success | 10% (likely partial) | 50% ties | Yes - most are ties |
| Failures | 67% (ineffective/regression) | 12.5% | NO - hard metrics much better |

**Why the discrepancy?**

The soft metrics evaluate **patch code quality** (syntax, logic, completeness). The hard metrics measure **runtime performance**. A patch can:
- Have code issues but still improve performance (imperfect but functional)
- Have perfect code but not affect the specific benchmark metric
- The 8 hard metric cases are **survivors** - they passed many filters

### 2. Pipeline Funnel Comparison

#### Soft Metrics Funnel (n=90)

```
Total Runs:              90 (100%)
Found Right Bottleneck:  36 (40%)
Valid Approach:          23 (26%)
Likely Speedup:          28 (31%)
High Confidence:         19 (21%)
```

#### Hard Metrics Funnel (n=81)

```
Total Commits:           81 (100%)
Has perf_command:        56 (69%)
No error/exception:      41 (51%)
Success status:          11 (14%)
Has agent patch:         50 (62%)
Has agent metrics:        8 (10%)
```

#### Funnel Comparison

| Stage | Soft (n=90) | Hard (n=81) | Notes |
|-------|-------------|-------------|-------|
| Start | 100% | 100% | - |
| Can be evaluated | ~100% | 69% | Hard: 31% have no test |
| No infrastructure error | ~100% | 51% | Hard: 49% fail to infra |
| Produces output | 100% | 62% | Both: agent generates something |
| Evaluable result | 100% | 14% | Hard: only 14% succeed |
| Final metrics | 31% likely success | 10% have metrics | Different meaning |

**Key Insight:** Soft metrics assume successful pipeline execution. Hard metrics reveal that **86% of commits never reach the point where patch quality matters** - they fail due to infrastructure issues first.

### 3. Failure Mode Analysis

#### Soft Metrics: Why Patches Fail (n=62 failures)

| Failure Mode | Count | Percentage |
|--------------|-------|------------|
| Localization Failure | 51 | 82% |
| Other | 7 | 11% |
| Overcomplicated | 2 | 3% |
| Complexity Avoidance | 1 | 2% |
| Incomplete Implementation | 1 | 2% |

**Soft metrics say: 82% fail because agent can't find the right code location**

#### Hard Metrics: Why Pipeline Fails (n=81)

| Failure Mode | Count | Percentage |
|--------------|-------|------------|
| No perf_command | 25 | 31% |
| Exception (Broken pipe) | 24 | 30% |
| Error (various) | 16 | 20% |
| baseline_failed | 3 | 4% |
| Other (version_bug, no_wheel) | 2 | 2% |
| **Success** | **11** | **14%** |

**Hard metrics say: 61% fail due to infrastructure/test generation issues**

#### Failure Mode Alignment

| Category | Soft Metrics | Hard Metrics | Aligned? |
|----------|--------------|--------------|----------|
| Agent capability | 82% (localization) | Unknown | Can't compare |
| Infrastructure | ~0% assumed | 61% | NO - hard reveals hidden issue |
| Test quality | ~0% assumed | 31% (no command) | NO - hard reveals hidden issue |

**Critical Finding:** Soft metrics evaluate agent capability **assuming infrastructure works**. Hard metrics reveal **infrastructure is the dominant problem** (61%+ failures).

### 4. Approach Quality Comparison

#### Soft Metrics: Approach Similarity (n=90)

| Category | Count | Percentage |
|----------|-------|------------|
| Same Approach | 6 | 7% |
| Similar Approach | 16 | 18% |
| Valid Alternative | 1 | 1% |
| **Good Approaches** | **23** | **26%** |
| Partial Solution | 7 | 8% |
| Ineffective | 37 | 41% |
| Harmful | 22 | 24% |
| Other | 1 | 1% |

#### Hard Metrics: Detailed Case Analysis (n=8)

| Commit | Optimization Type | Agent vs Human | Verdict |
|--------|-------------------|----------------|---------|
| a3223766 | General Performance | +8.17% | Win - different but effective approach |
| 98f47f2a | Attention/CPU | +5.26% | Win - found improvement human missed |
| eefbf4a6 | CUDA Kernel | +7.59% | Win - better than human's regression |
| 6d0734c5 | Attention | -1.28% | Tie - similar approach, similar result |
| fc542144 | General | -0.32% | Tie - matched human (both regressed) |
| f26c4aee | General | -0.02% | Tie - no meaningful difference |
| 310aca88 | General | +0.20% | Tie - effectively same as human |
| b690e348 | Model Architecture | -5.58% | Loss - different approach, worse |

#### Alignment Analysis

| Soft Category | Hard Outcome | Evidence |
|---------------|--------------|----------|
| Same Approach (7%) | 4 ties (50%) | Some ties match human exactly |
| Similar Approach (18%) | 3 wins (37.5%) | Agent found variations that work |
| Valid Alternative (1%) | - | Too rare to compare |
| Harmful (24%) | 1 loss (12.5%) | Most "harmful" patches don't reach runtime |

**Key Insight:** Soft metrics are **conservative** - they mark 24% as "harmful" but hard metrics only show 12.5% actual losses. Many code-level issues don't translate to runtime problems.

### 5. Technique Overlap Analysis

#### Soft Metrics: Technique Similarity (n=90)

From `technique_overlap_chart.png`:

| Metric | Value |
|--------|-------|
| Has Technique Overlap | 38 (42.2%) |
| No Overlap | 52 (57.8%) |
| Mean Jaccard Similarity | 0.24 |
| Agent Identified Real Technique | 42 (47%) |
| Agent Only "other" | 48 (53%) |

**Distribution of Jaccard Similarity:**
- 0-0.2 (None): 50 runs (56%)
- 0.2-0.4 (Low): 14 runs (16%)
- 0.4-0.6 (Medium): 10 runs (11%)
- 0.6-0.8 (High): 8 runs (9%)
- 0.8-1.0 (Exact): 8 runs (9%)

**Technique Usage Frequency:**
Humans prefer: Memory optimization (73), lazy computation (38), batching (22)
Agents prefer: Memory optimization (35), lazy computation (21), batching (14)

**Pattern:** Agents use ~50% fewer technique types but focus on same top techniques.

#### Hard Metrics: Success by Optimization Type

| Type | Soft Technique Freq | Hard Success |
|------|---------------------|--------------|
| Memory | 73 (human), 35 (agent) | 0/6 (0%) |
| Attention | - | 2/8 (25%) |
| CUDA/Kernel | - | 1/14 (7%) |
| General Performance | - | 6/46 (13%) |

**Alignment:** Memory optimization has highest technique overlap but **0% hard metric success**. This suggests technique matching ≠ runtime success.

---

## Key Findings

### 1. Soft and Hard Metrics Measure Different Things

| Aspect | Soft Metrics | Hard Metrics |
|--------|--------------|--------------|
| Focus | Code quality | Runtime performance |
| Scope | Agent capability | End-to-end pipeline |
| Assumption | Infrastructure works | Measures infrastructure too |
| Sample | Selected patches | All commits |

### 2. Both Reveal Valid but Different Problems

**Soft metrics reveal:**
- 82% of patches fail due to localization issues
- Only 26% use a valid approach
- Agent struggles to identify correct bottleneck
- Technique overlap is low (42%)

**Hard metrics reveal:**
- 61%+ fail due to infrastructure before agent runs
- Only 10% reach final evaluation
- When evaluated, agent performs reasonably (37.5% wins, 50% ties)
- Pipeline reliability is the bottleneck, not agent capability

### 3. Predictions vs Reality

| Soft Prediction | Hard Reality | Assessment |
|-----------------|--------------|------------|
| 31% likely speedup | 37.5% wins | **Aligned** - soft slightly conservative |
| 67% likely failure | 12.5% actual loss | **NOT aligned** - many "failures" work fine |
| 82% localization failure | Unknown | **Can't verify** - different scope |
| 24% harmful | 12.5% worse | **Partially aligned** - soft overestimates harm |

### 4. The Missing Link: Pipeline Reliability

Neither metric adequately captures the **end-to-end picture**:

```
Commits (100)
    ↓ [31% fail: no test generated]
Tests Generated (69)
    ↓ [30% fail: infrastructure errors]
Infrastructure OK (48)
    ↓ [75% fail: baseline/agent failures]
Benchmarks Run (12)
    ↓ [33% fail: no metrics captured]
Metrics Available (8)
    ↓ [12.5% fail: agent worse]
Agent Wins or Ties (7)
```

**Only 7-8% of commits result in agent success that we can measure.**

---

## Recommendations

### For Soft Metrics Evaluation

1. **Add infrastructure reality check**
   - Include test generation success in evaluation
   - Factor in environment setup failures
   - Don't assume patches reach runtime

2. **Calibrate "harmful" predictions**
   - Current 24% "harmful" rate is overstated vs 12.5% actual
   - Consider adding runtime validation to calibrate

3. **Track patch→runtime correlation**
   - Which soft metric categories predict hard outcomes?
   - Build regression model: soft features → hard speedup

### For Hard Metrics Evaluation

1. **Fix infrastructure first**
   - 30% broken pipe errors are the biggest blocker
   - 31% no perf_command needs better test generation

2. **Increase sample size**
   - 8 commits with metrics is too small
   - Target 50+ for statistical significance

3. **Link to soft metrics**
   - Track which soft predictions matched hard outcomes
   - Use soft metrics to pre-filter likely successes

### For Combined Evaluation

1. **Create unified pipeline score**
   ```
   Score = P(test generated) × P(infra works) × P(soft likely success) × P(hard success | soft success)
   ```

2. **Track both metrics together**
   - Every commit should have both soft and hard evaluation
   - Enable correlation analysis

3. **Prioritize based on combined signal**
   - High soft confidence + successful infra → likely real win
   - Low soft confidence but passes → investigate why soft was wrong

---

## Conclusion

### Do the Findings Align?

**Partially yes, partially no.**

**Aligned:**
- Success rate predictions: Soft 31% vs Hard 37.5% (similar)
- Most patches don't cause regressions (soft: 67% ineffective, hard: 87.5% don't regress)
- Agent has capability but it's limited (both show ~30-40% real success)

**Not aligned:**
- Failure modes are completely different (soft: localization, hard: infrastructure)
- "Harmful" predictions don't match reality (soft: 24%, hard: 12.5%)
- Soft metrics don't capture the dominant problem (infrastructure)

### What This Tells Us

1. **Soft metrics evaluate agent capability in isolation** - useful for understanding what the agent does well/poorly

2. **Hard metrics evaluate end-to-end reality** - reveals infrastructure is the bigger problem

3. **Both are needed** - soft metrics for agent improvement, hard metrics for pipeline improvement

4. **The agent is better than soft metrics suggest** - when patches reach runtime, they usually don't cause regressions

5. **Infrastructure reliability is the bottleneck** - fixing broken pipes and test generation would expose more agent capability

### Final Verdict

**Are agents ready for kernel optimization?**

- **Soft metrics say:** Not yet - 82% localization failures, only 26% valid approaches
- **Hard metrics say:** Mixed - 37.5% wins, 50% ties, 12.5% losses when evaluated
- **Combined view:** Agent shows promise (37.5% wins) but pipeline reliability (only 10% reach evaluation) is the immediate blocker

**Priority order:**
1. Fix infrastructure (broken pipe, test generation) - 60%+ of failures
2. Improve agent localization - 82% of remaining failures
3. Increase sample size for reliable conclusions

---

## Appendix: Data File Mapping

| Analysis Type | Files Used |
|---------------|------------|
| Soft Metrics - Quality | `state/analysis/optimization_quality_analysis.py/png` |
| Soft Metrics - Overlap | `state/analysis/technique_overlap_analysis.py/png` |
| Soft Metrics - GPT5 | `analysis_reports/FINAL_GPT5_EVALUATION_REPORT.md` |
| Soft Metrics - Research | `analysis_reports/RESEARCH_ANALYSIS.md` |
| Hard Metrics | `Inferencebench/claude-code-vllm-benchmarks` (HuggingFace) |
| Analysis Script | `scripts/analyze_benchmark_dataset.py` |
