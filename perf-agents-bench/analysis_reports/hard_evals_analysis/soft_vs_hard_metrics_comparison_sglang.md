# Soft vs Hard Metrics Comparison Analysis

**Generated:** 2026-01-04 11:08:33

## Overview

| Metric | Value |
|--------|-------|
| Soft metrics (vllm) | 72 |
| Hard metrics (HuggingFace) | 19 |
| Matched commits | 17 |
| **Prediction Accuracy** | **52.9%** (9/17) |

## Speedup Likelihood Distribution (Soft Metrics)

| Category | Count | % |
|----------|-------|---|
| likely_partial | 45 | 62.5% |
| likely_ineffective | 25 | 34.7% |
| likely_similar | 2 | 2.8% |


## Detailed Comparison

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


## Prediction Accuracy by Category

- **likely_similar**: 0/2 correct
- **likely_partial**: 6/12 correct
- **likely_ineffective**: 3/3 correct


## Key Findings

1. **Moderate predictive accuracy** (52.9%): Soft metrics have room for improvement in predicting hard outcomes.
3. **8 false positives**: Soft metrics predicted success but agent actually regressed.
