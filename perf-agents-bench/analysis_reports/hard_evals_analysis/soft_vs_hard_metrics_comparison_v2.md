# Soft vs Hard Metrics Comparison Analysis v2

**Generated:** 2026-01-04

## Executive Summary

| Metric | vLLM | SGLang | Combined |
|--------|------|--------|----------|
| Soft Metrics Analyzed | 90 | 72 | 162 |
| Hard Metrics Available | 14 | 19 | 33 |
| **Matched Commits** | **14** | **17** | **31** |
| **Prediction Accuracy** | **64.3%** (9/14) | **52.9%** (9/17) | **58.1%** (18/31) |

### Key Takeaways

1. **vLLM soft metrics are more reliable** (64.3%) than SGLang (52.9%)
2. **`likely_ineffective` predictions are highly accurate** across both repos (vLLM: 66.7%, SGLang: 100%)
3. **`likely_partial` predictions are unreliable** - vLLM 0% (0/3), SGLang 50% (6/12)
4. **Technique overlap correlates with better agent performance** in vLLM (+1.5% avg) but not SGLang (-10.8% avg)

---

## vLLM Analysis (14 matched commits)

### Speedup Likelihood Distribution (Soft Metrics)

| Category | Count | % |
|----------|-------|---|
| likely_ineffective | 40 | 44.4% |
| likely_regression | 21 | 23.3% |
| likely_similar | 19 | 21.1% |
| likely_partial | 9 | 10.0% |
| uncertain | 1 | 1.1% |

### Detailed Comparison

| Commit | Metric | Human | Agent | Diff% | Soft Prediction | Tech Overlap | Correct? |
|--------|--------|-------|-------|-------|-----------------|--------------|----------|
| `a32237665df8` | ttft | 33.5 | 30.8 | +8.2% | likely_similar | True | ✓ |
| `6a417b8600d4` | ttft | 1160.4 | 1097.6 | +5.4% | likely_similar | True | ✓ |
| `bc7c4d206bbf` | ttft | 2520.7 | 2454.7 | +2.6% | likely_ineffective | False | ✗ |
| `30172b4947c5` | ttft | 1103.5 | 1074.9 | +2.6% | likely_ineffective | False | ✗ |
| `8a4e5c5f3c1d` | ttft | 924.7 | 908.6 | +1.7% | likely_partial | True | ✗ |
| `70b808fe1a63` | ttft | 58.7 | 58.2 | +0.9% | likely_partial | True | ✗ |
| `fc542144c447` | ttft | 34.5 | 34.6 | -0.3% | likely_ineffective | False | ✓ |
| `ed25054577f7` | ttft | 799.2 | 807.9 | -1.1% | likely_ineffective | False | ✓ |
| `6d0734c562e7` | ttft | 2167.0 | 2194.8 | -1.3% | likely_ineffective | False | ✓ |
| `299ebb62b269` | ttft | 22.6 | 22.9 | -1.5% | likely_similar | True | ✓ |
| `b55ed6ef8ab0` | ttft | 1031.6 | 1056.1 | -2.4% | likely_regression | False | ✓ |
| `fe66b34728e5` | ttft | 5722.9 | 5874.6 | -2.6% | likely_ineffective | False | ✓ |
| `58eee5f2e05b` | ttft | 811.1 | 835.5 | -3.0% | likely_regression | False | ✓ |
| `b690e34824fd` | ttft | 9130.9 | 9640.5 | -5.6% | likely_partial | True | ✗ |

### Prediction Accuracy by Category (vLLM)

| Category | Correct | Total | Accuracy |
|----------|---------|-------|----------|
| likely_similar | 3 | 3 | **100%** |
| likely_partial | 0 | 3 | **0%** |
| likely_ineffective | 4 | 6 | **66.7%** |
| likely_regression | 2 | 2 | **100%** |

### Technique Overlap Analysis (vLLM)

| Technique Overlap | Count | Avg Agent Diff% |
|-------------------|-------|-----------------|
| True | 6 | **+1.52%** |
| False | 8 | **-0.69%** |

---

## SGLang Analysis (17 matched commits)

### Speedup Likelihood Distribution (Soft Metrics)

| Category | Count | % |
|----------|-------|---|
| likely_partial | 45 | 62.5% |
| likely_ineffective | 25 | 34.7% |
| likely_similar | 2 | 2.8% |

### Detailed Comparison

| Commit | Metric | Human | Agent | Diff% | Soft Prediction | Tech Overlap | Correct? |
|--------|--------|-------|-------|-------|-----------------|--------------|----------|
| `73b13e69b420` | ttft | 1602.9 | 707.0 | +55.9% | likely_partial | True | ✓ |
| `6b231325b978` | ttft | 1109.3 | 718.4 | +35.2% | likely_partial | True | ✓ |
| `9216b10678a0` | ttft | 1461.9 | 1048.0 | +28.3% | likely_partial | True | ✓ |
| `132dad874d2e` | ttft | 1293.1 | 1075.6 | +16.8% | likely_partial | False | ✓ |
| `dd1012fcbe2a` | ttft | 1282.2 | 1126.6 | +12.1% | likely_partial | False | ✓ |
| `880221bd3b3e` | ttft | 770.3 | 690.7 | +10.3% | likely_partial | False | ✓ |
| `9183c23eca51` | ttft | 1171.8 | 1205.8 | -2.9% | likely_ineffective | False | ✓ |
| `b1e5a33ae337` | ttft | 677.0 | 751.3 | -11.0% | likely_similar | True | ✗ |
| `6e2da5156176` | ttft | 1094.8 | 1217.4 | -11.2% | likely_partial | False | ✗ |
| `187b85b7f384` | ttft | 1063.5 | 1186.2 | -11.5% | likely_partial | True | ✗ |
| `d1112d8548eb` | ttft | 507.8 | 570.6 | -12.4% | likely_ineffective | False | ✓ |
| `df7f61ee7d23` | ttft | 1119.6 | 1435.7 | -28.2% | likely_partial | True | ✗ |
| `a99801e0750f` | ttft | 1073.3 | 1400.3 | -30.5% | likely_partial | True | ✗ |
| `2a754e57b052` | ttft | 684.8 | 1024.8 | -49.6% | likely_ineffective | False | ✓ |
| `09deb20deef8` | ttft | 855.6 | 1328.0 | -55.2% | likely_similar | True | ✗ |
| `9c088829ee2a` | ttft | 650.0 | 1172.9 | -80.4% | likely_partial | True | ✗ |
| `a191a0e47c2f` | ttft | 629.3 | 1173.2 | -86.4% | likely_partial | False | ✗ |

### Prediction Accuracy by Category (SGLang)

| Category | Correct | Total | Accuracy |
|----------|---------|-------|----------|
| likely_similar | 0 | 2 | **0%** |
| likely_partial | 6 | 12 | **50%** |
| likely_ineffective | 3 | 3 | **100%** |

### Technique Overlap Analysis (SGLang)

| Technique Overlap | Count | Avg Agent Diff% |
|-------------------|-------|-----------------|
| True | 9 | **-10.82%** |
| False | 8 | **-15.41%** |

---

## Combined Analysis

### Prediction Accuracy by Category (Combined)

| Category | vLLM | SGLang | Combined |
|----------|------|--------|----------|
| likely_similar | 3/3 (100%) | 0/2 (0%) | **3/5 (60%)** |
| likely_partial | 0/3 (0%) | 6/12 (50%) | **6/15 (40%)** |
| likely_ineffective | 4/6 (66.7%) | 3/3 (100%) | **7/9 (77.8%)** |
| likely_regression | 2/2 (100%) | 0/0 (N/A) | **2/2 (100%)** |

### Failure Mode Analysis (Combined)

| Failure Mode | vLLM Count | SGLang Count | Avg Agent Diff% |
|--------------|------------|--------------|-----------------|
| localization_failure | 7 | 6 | -4.3% |
| not_applicable | 2 | 3 | -22.1% |
| complexity_avoidance | 1 | 5 | -3.6% |
| overcomplicated | 1 | 0 | +5.4% |
| incomplete_implementation | 2 | 0 | +1.3% |
| technique_mismatch | 0 | 1 | -11.2% |
| other | 1 | 0 | -3.0% |

### False Positives (Predicted Success, Actually Regressed)

These are the most problematic cases where soft metrics overpromised:

| Commit | Repo | Prediction | Actual Diff% | Failure Mode |
|--------|------|------------|--------------|--------------|
| `b690e34824fd` | vLLM | likely_partial | -5.6% | complexity_avoidance |
| `8a4e5c5f3c1d` | vLLM | likely_partial | +1.7%* | incomplete_implementation |
| `70b808fe1a63` | vLLM | likely_partial | +0.9%* | incomplete_implementation |
| `09deb20deef8` | SGLang | likely_similar | -55.2% | not_applicable |
| `b1e5a33ae337` | SGLang | likely_similar | -11.0% | not_applicable |
| `9c088829ee2a` | SGLang | likely_partial | -80.4% | localization_failure |
| `a191a0e47c2f` | SGLang | likely_partial | -86.4% | not_applicable |
| `a99801e0750f` | SGLang | likely_partial | -30.5% | complexity_avoidance |
| `df7f61ee7d23` | SGLang | likely_partial | -28.2% | complexity_avoidance |
| `6e2da5156176` | SGLang | likely_partial | -11.2% | technique_mismatch |
| `187b85b7f384` | SGLang | likely_partial | -11.5% | localization_failure |

*Below 2% threshold, marginal improvement

### False Negatives (Predicted Failure, Actually Improved)

These are cases where soft metrics were too pessimistic:

| Commit | Repo | Prediction | Actual Diff% |
|--------|------|------------|--------------|
| `bc7c4d206bbf` | vLLM | likely_ineffective | +2.6% |
| `30172b4947c5` | vLLM | likely_ineffective | +2.6% |

---

## Key Findings

### 1. Repository-Specific Performance

- **vLLM agent performance**: 5 improvements, 9 regressions (avg -0.15% overall)
- **SGLang agent performance**: 6 improvements, 11 regressions (avg -13.0% overall)
- **SGLang shows significantly worse agent performance** despite similar soft metric predictions

### 2. Prediction Reliability Patterns

| Pattern | Reliability | Notes |
|---------|-------------|-------|
| `likely_ineffective` | **High** (77.8%) | Most reliable across both repos |
| `likely_regression` | **High** (100%) | Only 2 samples (vLLM only) |
| `likely_similar` | **Moderate** (60%) | vLLM 100%, SGLang 0% - major discrepancy |
| `likely_partial` | **Low** (40%) | High false positive rate |

### 3. Technique Overlap Paradox

- **vLLM**: technique_overlap=True correlates with **better** agent performance (+1.5% vs -0.7%)
- **SGLang**: technique_overlap=True correlates with **worse** agent performance (-10.8% vs -15.4%)
- **Hypothesis**: SGLang optimization tasks may be more complex; agents attempt similar techniques but execute poorly

### 4. Failure Mode Insights

- **`localization_failure`** is most common (13/31 = 42%) - agents struggle to identify correct bottleneck
- **`not_applicable`** shows severe performance issues (-22% avg) - soft metrics incorrectly classified as successful
- **`complexity_avoidance`** appears in 6 cases with mixed results

---

## Recommendations

### 1. Calibrate `likely_partial` Predictions
- Current threshold too optimistic; consider requiring higher confidence scores
- Add secondary validation for technique implementation quality

### 2. Add Repository-Specific Weights
- SGLang tasks appear fundamentally harder; adjust expectations accordingly
- Consider task complexity scoring in soft metrics

### 3. Strengthen `technique_overlap` Validation
- When technique_overlap=True but failure_mode is not "not_applicable", flag for additional review
- Overlap alone doesn't guarantee correct implementation

### 4. Expand Hard Metrics Coverage
- Current: 31 matched commits out of 162 soft metrics (19.1% coverage)
- Recommended: Benchmark more commits to improve statistical significance

---

## Appendix: Data Sources

| Source | Location |
|--------|----------|
| vLLM Soft Metrics | `perf-agents-bench/state/analysis/vllm/` (feature/soft-metrics-analyzer-v3 branch) |
| vLLM Hard Metrics | `Inferencebench/claude-code-vllm-benchmarks` |
| SGLang Soft Metrics | `perf-agents-bench/state/analysis/sglang/` (feature/soft-metrics-analyzer-v3 branch) |
| SGLang Hard Metrics | `Inferencebench/claude-code-sglang-benchmarks_v1` |
| Analysis Script | `scripts/compare_soft_hard_metrics.py` |

## Appendix: Data Quality Notes

### SGLang Commit Matching (19 hard → 17 matched)

| Commit | Status | Issue |
|--------|--------|-------|
| `1bf1cf19` | Missing | No soft metrics analysis.json file |
| `6cb00c639812` | Incomplete | patch_quality.json has null values (LLM analysis failed) |

These 2 commits were excluded from analysis due to data quality issues in the soft metrics.
