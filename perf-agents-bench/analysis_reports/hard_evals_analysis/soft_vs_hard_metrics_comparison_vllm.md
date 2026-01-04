# Soft vs Hard Metrics Comparison Analysis

**Generated:** 2026-01-04 21:56:50

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

| Commit | Metric | Baseline | Human | Agent | H vs B | A vs B | A vs H | Soft Prediction | Correct? |
|--------|--------|----------|-------|-------|--------|--------|--------|-----------------|----------|
| `a32237665df8` | ttft | 35.8 | 33.5 | 30.8 | +6.2% | +13.9% | +8.2% | likely_similar | ✓ |
| `6a417b8600d4` | ttft | 1762.0 | 1160.4 | 1097.6 | +34.1% | +37.7% | +5.4% | likely_similar | ✓ |
| `bc7c4d206bbf` | ttft | 2435.9 | 2520.7 | 2454.7 | -3.5% | -0.8% | +2.6% | likely_ineffective | ✗ |
| `30172b4947c5` | ttft | 1115.7 | 1103.5 | 1074.9 | +1.1% | +3.7% | +2.6% | likely_ineffective | ✗ |
| `8a4e5c5f3c1d` | ttft | 898.6 | 924.7 | 908.6 | -2.9% | -1.1% | +1.7% | likely_partial | ✗ |
| `70b808fe1a63` | ttft | 59.8 | 58.7 | 58.2 | +1.8% | +2.7% | +0.9% | likely_partial | ✗ |
| `fc542144c447` | ttft | 32.0 | 34.5 | 34.6 | -7.6% | -7.9% | -0.3% | likely_ineffective | ✓ |
| `ed25054577f7` | ttft | 818.3 | 799.2 | 807.9 | +2.3% | +1.3% | -1.1% | likely_ineffective | ✓ |
| `6d0734c562e7` | ttft | 2194.9 | 2167.0 | 2194.8 | +1.3% | +0.0% | -1.3% | likely_ineffective | ✓ |
| `299ebb62b269` | ttft | 25.7 | 22.6 | 22.9 | +12.1% | +10.8% | -1.5% | likely_similar | ✓ |
| `b55ed6ef8ab0` | ttft | 1145.2 | 1031.6 | 1056.1 | +9.9% | +7.8% | -2.4% | likely_regression | ✓ |
| `fe66b34728e5` | ttft | 6225.6 | 5722.9 | 5874.6 | +8.1% | +5.6% | -2.6% | likely_ineffective | ✓ |
| `58eee5f2e05b` | ttft | 838.1 | 811.1 | 835.5 | +3.2% | +0.3% | -3.0% | likely_regression | ✓ |
| `b690e34824fd` | ttft | 37803.3 | 9130.9 | 9640.5 | +75.8% | +74.5% | -5.6% | likely_partial | ✗ |


## Prediction Accuracy by Category

- **likely_similar**: 3/3 correct
- **likely_partial**: 0/3 correct
- **likely_ineffective**: 4/6 correct
- **likely_regression**: 2/2 correct


## Key Findings

1. **Good predictive accuracy** (64.3%): Soft metrics are reasonably reliable predictors of hard metric outcomes.
2. **2 false negatives**: Soft metrics predicted failure but agent actually improved.
3. **1 false positives**: Soft metrics predicted success but agent actually regressed.


## GSO Failure Category Analysis

*Based on arxiv:2505.23671 Figure 7 methodology - categorizing model failures only*

**Summary:** 90 total runs | 15 successes (16.7%) | 75 failures (83.3%)

### Failure High-Level Distribution (n=75)

| Category | Count | % |
|----------|-------|---|
| Localization | 50 | 66.7% |
| Mismanage Compute | 15 | 20.0% |
| Avoid Complexity | 10 | 13.3% |


### Failure Sub-Category Breakdown

| Sub-Category | Count | % |
|--------------|-------|---|
| Misdiagnosed Bottlenecks | 50 | 66.7% |
| Destructive/Runaway | 8 | 10.7% |
| Wrong Abstraction Level | 7 | 9.3% |
| Exploit-Heavy | 4 | 5.3% |
| Lazy Optimization | 3 | 4.0% |
| Explore-Heavy | 3 | 4.0% |
