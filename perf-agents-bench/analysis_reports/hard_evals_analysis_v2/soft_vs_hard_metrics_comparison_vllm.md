# Soft vs Hard Metrics Comparison Analysis

**Generated:** 2026-01-13 11:46:32

## Overview

| Metric | Value |
|--------|-------|
| Soft metrics (vllm) | 90 |
| Hard metrics (HuggingFace) | 42 |
| Matched commits | 37 |
| **Prediction Accuracy** | **67.6%** (25/37) |

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
| `e3580537a41a` | throughput | N/A | 2496.9 | 3107.0 | N/A | N/A | +24.4% | likely_regression | ✗ |
| `fc7b8d1eefcb` | throughput | N/A | 2214.0 | 2598.0 | N/A | N/A | +17.3% | likely_regression | ✗ |
| `6e36f4fa6ce6` | throughput | N/A | 2413.6 | 2784.2 | N/A | N/A | +15.4% | likely_regression | ✗ |
| `a32237665df8` | ttft | 35.8 | 33.5 | 30.8 | +6.2% | +13.9% | +8.2% | likely_similar | ✓ |
| `e206b5433109` | throughput | N/A | 3105.1 | 3352.1 | N/A | N/A | +8.0% | likely_similar | ✓ |
| `6a417b8600d4` | ttft | 1762.0 | 1160.4 | 1097.6 | +34.1% | +37.7% | +5.4% | likely_similar | ✓ |
| `98f47f2a4032` | throughput | 972.5 | 972.5 | 1023.7 | +0.0% | +5.3% | +5.3% | likely_ineffective | ✗ |
| `bc7c4d206bbf` | ttft | 2435.9 | 2520.7 | 2454.7 | -3.5% | -0.8% | +2.6% | likely_ineffective | ✗ |
| `30172b4947c5` | ttft | 1115.7 | 1103.5 | 1074.9 | +1.1% | +3.7% | +2.6% | likely_ineffective | ✗ |
| `8a4e5c5f3c1d` | ttft | 898.6 | 924.7 | 908.6 | -2.9% | -1.1% | +1.7% | likely_partial | ✗ |
| `70b808fe1a63` | ttft | 59.8 | 58.7 | 58.2 | +1.8% | +2.7% | +0.9% | likely_partial | ✗ |
| `cf2f084d56a1` | throughput | N/A | 2443.1 | 2451.8 | N/A | N/A | +0.4% | likely_regression | ✗ |
| `310aca88c984` | throughput | 51.1 | 102.1 | 102.3 | +99.8% | +100.2% | +0.2% | likely_ineffective | ✓ |
| `4c822298981a` | throughput | 102.1 | 153.2 | 153.5 | +50.0% | +50.3% | +0.2% | likely_ineffective | ✓ |
| `015069b01741` | throughput | 198.3 | 198.3 | 198.3 | -0.0% | -0.0% | +0.0% | likely_similar | ✓ |
| `f26c4aeecba4` | throughput | 818.8 | 818.9 | 818.7 | +0.0% | -0.0% | -0.0% | likely_ineffective | ✓ |
| `9badee53decb` | throughput | N/A | 3424.2 | 3417.1 | N/A | N/A | -0.2% | likely_similar | ✓ |
| `fc542144c447` | ttft | 32.0 | 34.5 | 34.6 | -7.6% | -7.9% | -0.3% | likely_ineffective | ✓ |
| `fa63e710c7fb` | latency | 1331.7 | 1323.8 | 1329.9 | +0.6% | +0.1% | -0.5% | likely_ineffective | ✓ |
| `80aa7e91fcd5` | throughput | N/A | 2178.3 | 2167.7 | N/A | N/A | -0.5% | likely_ineffective | ✓ |
| `99abb8b650c6` | throughput | N/A | 3736.7 | 3716.5 | N/A | N/A | -0.5% | likely_ineffective | ✓ |
| `296f927f2493` | throughput | 1413.8 | 1421.5 | 1411.8 | +0.5% | -0.1% | -0.7% | likely_similar | ✓ |
| `6ce01f30667b` | throughput | N/A | 1790.9 | 1777.4 | N/A | N/A | -0.8% | likely_ineffective | ✓ |
| `ca7a2d5f28ea` | throughput | N/A | 2376.7 | 2353.8 | N/A | N/A | -1.0% | likely_ineffective | ✓ |
| `ed25054577f7` | ttft | 818.3 | 799.2 | 807.9 | +2.3% | +1.3% | -1.1% | likely_ineffective | ✓ |
| `8bc68e198c4c` | throughput | N/A | 1979.4 | 1956.6 | N/A | N/A | -1.2% | likely_partial | ✗ |
| `6d0734c562e7` | ttft | 2194.9 | 2167.0 | 2194.8 | +1.3% | +0.0% | -1.3% | likely_ineffective | ✓ |
| `299ebb62b269` | ttft | 25.7 | 22.6 | 22.9 | +12.1% | +10.8% | -1.5% | likely_similar | ✓ |
| `3476ed0809ec` | throughput | N/A | 2127.8 | 2094.3 | N/A | N/A | -1.6% | likely_ineffective | ✓ |
| `b55ed6ef8ab0` | ttft | 1145.2 | 1031.6 | 1056.1 | +9.9% | +7.8% | -2.4% | likely_regression | ✓ |
| `fe66b34728e5` | ttft | 6225.6 | 5722.9 | 5874.6 | +8.1% | +5.6% | -2.6% | likely_ineffective | ✓ |
| `58eee5f2e05b` | ttft | 838.1 | 811.1 | 835.5 | +3.2% | +0.3% | -3.0% | likely_regression | ✓ |
| `7c01f706418d` | throughput | N/A | 2229.4 | 2109.4 | N/A | N/A | -5.4% | likely_ineffective | ✓ |
| `b690e34824fd` | ttft | 37803.3 | 9130.9 | 9640.5 | +75.8% | +74.5% | -5.6% | likely_partial | ✗ |
| `3a243095e5e7` | throughput | N/A | 2518.8 | 2366.8 | N/A | N/A | -6.0% | likely_regression | ✓ |
| `9474e89ba4ec` | throughput | N/A | 3086.4 | 2852.5 | N/A | N/A | -7.6% | likely_regression | ✓ |
| `89a84b0bb7b3` | throughput | N/A | 3558.5 | 2967.5 | N/A | N/A | -16.6% | likely_partial | ✗ |


## Prediction Accuracy by Category

- **likely_similar**: 7/7 correct
- **likely_partial**: 0/5 correct
- **likely_ineffective**: 14/17 correct
- **likely_regression**: 4/8 correct


## Key Findings

1. **Good predictive accuracy** (67.6%): Soft metrics are reasonably reliable predictors of hard metric outcomes.
2. **6 false negatives**: Soft metrics predicted failure but agent actually improved.
3. **2 false positives**: Soft metrics predicted success but agent actually regressed.


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

**Total Successes:** 15 (16.7% success rate)

### Implementation Alignment

| Category | Count | % | Description |
|----------|-------|---|-------------|
| core_match_extras | 10 | 66.7% | Got main optimization, added extra changes/noise |
| identical | 5 | 33.3% | Agent produced exactly same code as human |


### Agent Behavior Patterns

| Pattern | Count | % |
|---------|-------|---|
| over_engineering | 6 | 40.0% |
| diff_pollution | 5 | 33.3% |
| scope_creep | 4 | 26.7% |
| clean_patch | 4 | 26.7% |


### Technique Distribution

| Human Techniques | Count | Agent Techniques | Count |
|------------------|-------|------------------|-------|
| memory_optimization | 9 | memory_optimization | 13 |
| lazy_computation | 7 | lazy_computation | 10 |
| api_library | 6 | api_library | 7 |
| batching | 4 | batching | 5 |
| other | 2 | low_level | 4 |
| algorithmic | 1 | compiler_optimization | 2 |
| low_level | 1 | parallelization | 2 |
| parallelization | 1 | other | 2 |
|  |  | algorithmic | 1 |


### Task Domains

| Domain | Count | % |
|--------|-------|---|
| compute | 7 | 46.7% |
| memory | 6 | 40.0% |
| concurrency | 1 | 6.7% |
| other | 1 | 6.7% |


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
