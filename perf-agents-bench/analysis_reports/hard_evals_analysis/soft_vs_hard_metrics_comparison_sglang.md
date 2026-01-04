# Soft vs Hard Metrics Comparison Analysis

**Generated:** 2026-01-04 21:57:15

## Overview

| Metric | Value |
|--------|-------|
| Soft metrics (vllm) | 72 |
| Hard metrics (HuggingFace) | 0 |
| Matched commits | 0 |
| **Prediction Accuracy** | **0.0%** (0/0) |

## Speedup Likelihood Distribution (Soft Metrics)

| Category | Count | % |
|----------|-------|---|
| likely_partial | 45 | 62.5% |
| likely_ineffective | 25 | 34.7% |
| likely_similar | 2 | 2.8% |


## Detailed Comparison

| Commit | Metric | Baseline | Human | Agent | H vs B | A vs B | A vs H | Soft Prediction | Correct? |
|--------|--------|----------|-------|-------|--------|--------|--------|-----------------|----------|


## Prediction Accuracy by Category



## Key Findings

1. **Moderate predictive accuracy** (0.0%): Soft metrics have room for improvement in predicting hard outcomes.


## GSO Failure Category Analysis

*Based on arxiv:2505.23671 Figure 7 methodology - categorizing model failures only*

**Summary:** 72 total runs | 11 successes (15.3%) | 61 failures (84.7%)

### Failure High-Level Distribution (n=61)

| Category | Count | % |
|----------|-------|---|
| Avoid Complexity | 35 | 57.4% |
| Localization | 24 | 39.3% |
| Mismanage Compute | 2 | 3.3% |


### Failure Sub-Category Breakdown

| Sub-Category | Count | % |
|--------------|-------|---|
| Lazy Optimization | 34 | 55.7% |
| Misdiagnosed Bottlenecks | 19 | 31.1% |
| Less Impactful | 5 | 8.2% |
| Explore-Heavy | 2 | 3.3% |
| Wrong Abstraction Level | 1 | 1.6% |
