# Full Analysis Report (v5)

## Executive Summary

This report analyzes 4 AI coding agents (Claude Code, Codex, TRAE-Sonnet, TRAE-GPT) on 39 vLLM optimization tasks, evaluating both actual benchmark performance (hard metrics) and LLM-as-judge evaluations (soft metrics).

**Key Findings:**
- **Claude Code** has highest patch success rate (76.9%) but moderate performance success (53.3%)
- **TRAE-Sonnet** has lowest patch success (43.6%) but highest beats rate when patches work (41.2%)
- **Soft metrics show weak/inverted correlations** with actual outcomes across all agents

---

## Part 1: Dataset Overview

| Category | Count |
|----------|-------|
| Total commits in dataset | 53 |
| Excluded (wrong perf command) | 9 |
| Excluded (corrupted Docker) | 1 |
| Excluded (vLLM incompatibility) | 4 |
| **Final valid commits** | **39** |

---

## Part 2: Patch Success Rates

| Agent | Valid Patches | Patch Failures | Success Rate |
|-------|---------------|----------------|--------------|
| Claude Code | 30 | 9 | **76.9%** |
| Codex | 17 | 22 | 43.6% |
| TRAE (Sonnet) | 17 | 22 | 43.6% |
| TRAE (GPT) | 12 | 27 | 30.8% |

---

## Part 3: Hard Metrics Analysis (Agent vs Human)

| Agent | Comparable | Beats | Similar | Worse | Success Rate |
|-------|------------|-------|---------|-------|--------------|
| Claude Code | 30 | 7 (23%) | 9 (30%) | 14 (47%) | **53.3%** |
| Codex | 17 | 6 (35%) | 2 (12%) | 9 (53%) | 47.1% |
| TRAE (Sonnet) | 17 | 7 (41%) | 1 (6%) | 9 (53%) | 47.1% |
| TRAE (GPT) | 12 | 4 (33%) | 1 (8%) | 7 (58%) | 41.7% |

### Top Performers

**Claude Code:**
- e3580537: +22.3% ttft
- 22d33bac: +17.2% ttft
- a3223766: +8.2% ttft

**Codex:**
- 30172b49: +46.9% ttft
- b55ed6ef: +42.8% ttft
- 58eee5f2: +25.8% ttft

**TRAE (Sonnet):**
- 30172b49: +45.7% ttft
- b55ed6ef: +43.7% ttft
- 58eee5f2: +25.4% ttft

### Worst Performers

| Agent | Commit | Regression |
|-------|--------|------------|
| Claude Code | 89a84b0b | -73.1% ttft |
| Codex | fc542144 | -1659.8% ttft |
| TRAE (Sonnet) | fc542144 | -1633.4% ttft |
| TRAE (GPT) | 70b808fe | -1056.3% ttft |

---

## Part 4: Soft Metrics Analysis

### Speedup Likelihood Distribution

| Category | Claude Code | Codex | TRAE-Sonnet | TRAE-GPT |
|----------|-------------|-------|-------------|----------|
| likely_partial | 61.5% | 41.0% | 17.6% | 37.0% |
| likely_ineffective | 25.6% | 7.7% | 64.7% | 33.3% |
| likely_similar | 10.3% | 43.6% | 14.7% | 22.2% |

### Failure Mode Distribution

| Category | Claude Code | Codex | TRAE-Sonnet | TRAE-GPT |
|----------|-------------|-------|-------------|----------|
| not_applicable | 23.1% | 51.3% | 20.6% | 40.7% |
| localization_failure | 33.3% | 17.9% | 11.8% | 22.2% |
| complexity_avoidance | 33.3% | 5.1% | 17.6% | 11.1% |
| incomplete_implementation | 10.3% | 20.5% | 47.1% | 22.2% |

---

## Part 5: Hard-Soft Correlation Analysis

### Claude Code (n=30)

| Speedup Prediction | n | Beats | Worse | Beats% |
|--------------------|---|-------|-------|--------|
| likely_ineffective | 9 | 4 | 2 | **44.4%** |
| likely_partial | 17 | 3 | 11 | 17.6% |
| likely_similar | 4 | 0 | 1 | 0.0% |

**INVERTED:** "likely_ineffective" has highest beats rate!

### Codex (n=17)

| Speedup Prediction | n | Beats | Worse | Beats% |
|--------------------|---|-------|-------|--------|
| likely_partial | 8 | 3 | 3 | 37.5% |
| likely_regression | 2 | 1 | 1 | 50.0% |
| likely_similar | 7 | 2 | 5 | 28.6% |

### TRAE-Sonnet (n=14)

| Speedup Prediction | n | Beats | Worse | Beats% |
|--------------------|---|-------|-------|--------|
| likely_ineffective | 10 | 4 | 5 | **40.0%** |
| likely_partial | 2 | 1 | 1 | 50.0% |

**INVERTED:** "likely_ineffective" has 40% beats rate!

---

## Part 6: Key Insights

### 1. Patch Success vs Performance Success

| Agent | Patch Success | Performance Success | Gap |
|-------|---------------|---------------------|-----|
| Claude Code | 76.9% | 53.3% | -23.6% |
| Codex | 43.6% | 47.1% | +3.5% |
| TRAE-Sonnet | 43.6% | 47.1% | +3.5% |
| TRAE-GPT | 30.8% | 41.7% | +10.9% |

### 2. Soft Metrics Are Poor Predictors

- "likely_ineffective" often beats human
- "likely_similar" often performs worse
- LLM-as-judge should NOT be used as proxy for benchmark performance

### 3. High Variance

| Agent | Best | Worst |
|-------|------|-------|
| Claude Code | +22.3% | -73.1% |
| Codex | +46.9% | -1659.8% |
| TRAE-Sonnet | +45.7% | -1633.4% |
| TRAE-GPT | +43.1% | -1056.3% |

---

## Conclusions

1. **Claude Code most reliable** for generating working patches (76.9%)
2. **Performance inconsistent** across all agents (41-53% success)
3. **Soft metrics unreliable** - often inverted correlations
4. **High variance universal** - all agents can win big or fail catastrophically
5. **Codex/TRAE-Sonnet identical** - same success/fail patterns

---

*Generated: 2026-01-19*
*Data source: HuggingFace `Inferencebench/claude-code-vllm-benchmarks`*
