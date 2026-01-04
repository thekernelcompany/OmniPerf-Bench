# Soft vs Hard Metrics Comparison Analysis

**Generated:** 2026-01-04 11:08:07

## Overview

| Metric | Value |
|--------|-------|
| Soft metrics (vllm) | 90 |
| Hard metrics (HuggingFace) | 14 |
| Matched commits | 14 |
| **Prediction Accuracy** | **64.3%** (9/14) |

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


## Prediction Accuracy by Category

- **likely_similar**: 3/3 correct
- **likely_partial**: 0/3 correct
- **likely_ineffective**: 4/6 correct
- **likely_regression**: 2/2 correct


## Key Findings

1. **Good predictive accuracy** (64.3%): Soft metrics are reasonably reliable predictors of hard metric outcomes.
2. **2 false negatives**: Soft metrics predicted failure but agent actually improved.
3. **1 false positives**: Soft metrics predicted success but agent actually regressed.
