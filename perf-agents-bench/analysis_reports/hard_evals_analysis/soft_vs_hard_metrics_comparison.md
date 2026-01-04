# Soft vs Hard Metrics Comparison Analysis

**Generated:** 2026-01-02 16:16:08

## Overview

| Metric | Value |
|--------|-------|
| Soft metrics (vllm) | 90 |
| Hard metrics (HuggingFace) | 8 |
| Matched commits | 8 |
| **Prediction Accuracy** | **62.5%** (5/8) |

## Speedup Likelihood Distribution (Soft Metrics)

| Category | Count | % |
|----------|-------|---|
| likely_ineffective | 40 | 44.4% |
| likely_regression | 21 | 23.3% |
| likely_similar | 19 | 21.1% |
| likely_partial | 9 | 10.0% |
| uncertain | 1 | 1.1% |


## Detailed Comparison

| Commit | Metric | Human | Agent | Diff% | Soft Prediction | Tech Overlap | Correct? |
|--------|--------|-------|-------|-------|-----------------|--------------|----------|
| `a32237665df8` | ttft | 33.5 | 30.8 | +8.2% | likely_similar | True | ✓ |
| `eefbf4a68b7b` | ttft | 2649.9 | 2448.7 | +7.6% | likely_ineffective | False | ✗ |
| `98f47f2a4032` | throughput | 972.5 | 1023.7 | +5.3% | likely_ineffective | False | ✗ |
| `310aca88c984` | throughput | 102.1 | 102.3 | +0.2% | likely_ineffective | False | ✓ |
| `f26c4aeecba4` | throughput | 818.9 | 818.7 | -0.0% | likely_ineffective | False | ✓ |
| `fc542144c447` | ttft | 34.5 | 34.6 | -0.3% | likely_ineffective | False | ✓ |
| `6d0734c562e7` | ttft | 2167.0 | 2194.8 | -1.3% | likely_ineffective | False | ✓ |
| `b690e34824fd` | ttft | 9130.9 | 9640.5 | -5.6% | likely_partial | True | ✗ |


## Prediction Accuracy by Category

- **likely_similar**: 1/1 correct
- **likely_partial**: 0/1 correct
- **likely_ineffective**: 4/6 correct


## Key Findings

1. **Good predictive accuracy** (62.5%): Soft metrics are reasonably reliable predictors of hard metric outcomes.
2. **2 false negatives**: Soft metrics predicted failure but agent actually improved.
3. **1 false positives**: Soft metrics predicted success but agent actually regressed.
