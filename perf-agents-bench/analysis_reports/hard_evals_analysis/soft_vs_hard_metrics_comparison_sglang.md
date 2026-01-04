# Soft vs Hard Metrics Comparison Analysis

**Generated:** 2026-01-05 00:27:05

## Overview

| Metric | Value |
|--------|-------|
| Soft metrics (vllm) | 72 |
| Hard metrics (HuggingFace) | 14 |
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


### Failure Category Definitions

#### High-Level Categories

| Category | Description |
|----------|-------------|
| **Localization** | Agent failed to correctly identify the performance bottleneck. Either targeted the wrong component entirely, or made changes that were technically valid but addressed a less critical bottleneck than what the human identified. |
| **Avoid Complexity** | Agent recognized the bottleneck but avoided the necessary complexity to fix it properly. Made superficial or incomplete changes rather than implementing the full solution, often staying at a higher abstraction level (e.g., Python) when lower-level work (e.g., CUDA/C++) was required. |
| **Mismanage Compute** | Agent wasted computational resources through inefficient exploration, over-engineered solutions, or catastrophic failures. Includes cases where the agent destroyed/corrupted the repository or spent excessive time exploring without producing actionable optimizations. |


#### Sub-Category Definitions

| Sub-Category | Description |
|--------------|-------------|
| **Misdiagnosed Bottlenecks** | Agent optimized the wrong component. Failed to identify the actual performance bottleneck that the human targeted. May have made valid optimizations, but to code that wasn't the critical path. |
| **Less Impactful** | Agent identified a related bottleneck but chose a less impactful optimization target. The changes were valid but addressed a secondary concern rather than the primary performance issue. |
| **Lazy Optimization** | Agent made superficial changes that avoid the deeper work required. Typically involves config tweaks, parameter adjustments, or minor refactors when algorithmic or architectural changes were needed. |
| **Wrong Abstraction Level** | Agent worked at the wrong level of the stack. Common pattern: staying in Python when the human wrote CUDA kernels, or modifying high-level APIs when low-level implementation changes were required. |
| **Exploit-Heavy** | Agent over-engineered the solution with unnecessary complexity. Added excessive abstractions, unnecessary features, or convoluted logic when a simpler approach would have sufficed. |
| **Explore-Heavy** | Agent spent most of its steps examining the codebase (reading files, searching) without converging on actionable optimizations. High explore-to-edit ratio (>40% read/search, ≤2 edits) with no success. |
| **Destructive/Runaway** | Catastrophic failure where agent corrupted or destroyed the repository. Includes cases of runaway edits, deletion of critical files, or changes that made the codebase unbuildable/unusable. |


## Success Analysis (Human vs Agent)

**Total Successes:** 11 (15.3% success rate)

### Implementation Alignment

| Category | Count | % | Description |
|----------|-------|---|-------------|
| alternative_technique | 5 | 45.5% | Same bottleneck, different valid solution method |
| core_match_extras | 3 | 27.3% | Got main optimization, added extra changes/noise |
| alternative_target | 3 | 27.3% | Different bottleneck, still valid optimization |


### Agent Behavior Patterns

| Pattern | Count | % |
|---------|-------|---|
| clean_patch | 9 | 81.8% |
| over_engineering | 2 | 18.2% |


### Technique Distribution

| Human Techniques | Count | Agent Techniques | Count |
|------------------|-------|------------------|-------|
| api_library | 6 | memory_optimization | 10 |
| memory_optimization | 5 | api_library | 6 |
| lazy_computation | 4 | lazy_computation | 4 |
| algorithmic | 3 | compiler_optimization | 2 |
| compiler_optimization | 1 | batching | 2 |
| other | 1 | algorithmic | 1 |
| parallelization | 1 |  |  |


### Task Domains

| Domain | Count | % |
|--------|-------|---|
| compute | 5 | 45.5% |
| memory | 4 | 36.4% |
| io | 1 | 9.1% |
| algorithmic | 1 | 9.1% |


### Success Category Definitions

#### Implementation Alignment Categories

| Category | Description |
|----------|-------------|
| **identical** | Agent produced functionally identical code to the human solution. Same optimization technique, same target, same implementation approach. This is the ideal outcome - agent perfectly replicated human expertise. |
| **core_match_extras** | Agent correctly identified and implemented the core optimization, but included additional unnecessary changes. Common patterns: unrelated CI/CD modifications, extra file touches, documentation changes, or redundant refactoring alongside the main fix. The signal is correct but noisy. |
| **alternative_technique** | Agent targeted the same performance bottleneck as the human but used a different valid optimization technique. Example: human used argmax fast-path, agent used single-mask operation - both valid solutions to the same problem. Shows agent creativity while maintaining correctness. |
| **alternative_target** | Agent optimized a different bottleneck than the human, but the optimization was still valid and produced measurable improvement. Example: human fixed benchmark logic, agent optimized serialization. Both are legitimate performance wins, just different problem interpretations. |


#### Agent Behavior Patterns

| Pattern | Description |
|---------|-------------|
| **over_engineering** | Agent applied significantly more optimization techniques than the human (2+ additional techniques). Suggests a 'shotgun approach' - trying many optimizations hoping one works, rather than surgical precision. May indicate uncertainty about which technique will be effective. |
| **diff_pollution** | Agent included unrelated changes in the diff: CI/CD pipeline modifications, Dockerfile changes, BuildKite configs, documentation updates, or build scripts. These changes don't contribute to the optimization and add noise to the patch. |
| **scope_creep** | Agent modified files beyond the necessary scope. Examples: touching C++/CUDA files when human stayed in Python, modifying test files unnecessarily, or making changes to unrelated modules. Increases review burden and risk. |
| **clean_patch** | Agent produced a surgical, minimal patch similar to what a human would write. No unnecessary changes, focused scope, appropriate technique selection. This is the ideal behavior pattern - efficient and precise. |
