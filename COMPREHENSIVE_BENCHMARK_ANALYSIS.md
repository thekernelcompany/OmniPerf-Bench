# Comprehensive vLLM Benchmark Analysis (Critical Review)

## Executive Summary

| Metric | Value |
|--------|-------|
| **Total commits analyzed** | 96 |
| **Local 3-way benchmarks** | 22 (22.9%) |
| **Local human-only** | 8 (8.3%) |
| **Claude-only 3-way** | 14 (14.6%) |
| **Usable total** | **44 (45.8%)** |
| **Large models (multi-GPU)** | 14 (14.6%) |
| **Agent crashes server** | 6 (6.3%) |
| **Version bug #8791** | 5 (5.2%) |
| **Baseline failures** | 8 (8.3%) |
| **Other failures** | 19 (19.8%) |
| **Unknown models** | 6 (6.2%) |

---

## Data Source Comparison

| Source | Methodology | Metrics | Hardware |
|--------|-------------|---------|----------|
| **Local benchmarks** | HTTP direct (100 prompts, 64 tok output) | Throughput (tok/s) | Single GPU |
| **Claude dataset** | Full serving benchmark | TPOT (ms), TTFT (ms) | H100 (1-8 GPU) |

**Critical note:** Metrics from different sources are NOT directly comparable. Local uses throughput; Claude uses latency metrics.

---

## Local 3-Way Benchmarks (22 commits)

These have locally-run human AND agent benchmarks with comparable metrics.

| Commit | Model | Baseline (tok/s) | Human (tok/s) | Agent (tok/s) | Agent vs Human |
|--------|-------|------------------|---------------|---------------|----------------|
| 015069b0 | Qwen/Qwen3-1.7B | 1428 | 198 | 198 | +0.0% |
| 19d98e0c | deepseek-ai/DeepSeek-Coder-V2-Lite- | - | 2358 | 2341 | -0.8% |
| 22d33bac | meta-llama/Meta-Llama-3-8B-Instruct | 3026 | 3946 | 3985 | +1.0% |
| 296f927f | ibm-ai-platform/Bamba-9B | 813 | 1421 | 1412 | -0.7% |
| 3476ed08 | meta-llama/Meta-Llama-3-8B-Instruct | 6225 | 2128 | 2094 | -1.6% |
| 3a243095 | meta-llama/Meta-Llama-3-8B-Instruct | 7125 | 2519 | 2367 | -6.0% |
| 6ce01f30 | meta-llama/Meta-Llama-3-8B | 7062 | 1791 | 1777 | -0.8% |
| 6e36f4fa | meta-llama/Meta-Llama-3-8B-Instruct | 7855 | 2414 | 2784 | +15.4% |
| 7c01f706 | meta-llama/Meta-Llama-3-8B-Instruct | 6214 | 2229 | 2109 | -5.4% |
| 80aa7e91 | meta-llama/Meta-Llama-3-8B-Instruct | 6370 | 2178 | 2168 | -0.5% |
| 89a84b0b | Qwen/Qwen1.5-0.5B | 6004 | 3559 | 2967 | -16.6% |
| 8bc68e19 | meta-llama/Meta-Llama-3-8B-Instruct | 6874 | 1979 | 1957 | -1.2% |
| **9474e89b** | huggyllama/llama-7b | 7184 | 3086 | 2852 | **-7.6%** |
| 99abb8b6 | meta-llama/Meta-Llama-3-8B-Instruct | 2408 | 3737 | 3717 | -0.5% |
| 9badee53 | meta-llama/Meta-Llama-3-8B-Instruct | 8058 | 3424 | 3417 | -0.2% |
| ca7a2d5f | deepseek-ai/DeepSeek-Coder-V2-Lite- | 8308 | 2377 | 2354 | -1.0% |
| cf2f084d | meta-llama/Meta-Llama-3-8B-Instruct | 7080 | 2443 | 2452 | +0.4% |
| e206b543 | meta-llama/Meta-Llama-3-8B-Instruct | 8148 | 3105 | 3352 | +8.0% |
| **e3580537** | neuralmagic/Meta-Llama-3-8B-FP8 | 7441 | 2497 | 3107 | **+24.4%** |
| **fc7b8d1e** | meta-llama/Meta-Llama-3-8B-Instruct | 7175 | 2214 | 2598 | **+17.3%** |

### Local Performance Summary

| Outcome | Count | % |
|---------|-------|---|
| Human wins (>2%) | 5 | 22.7% |
| Agent wins (>2%) | 4 | 18.2% |
| Tie (±2%) | 13 | 59.1% |

---

## Local Human-Only (8 commits)

Human benchmark succeeded but agent failed.

| Commit | Model | Human (tok/s) | Agent Status |
|--------|-------|---------------|--------------|
| 2deb029d | neuralmagic/Meta-Llama-3-8B-Instruc | 3095 | not_run |
| 35fad35a | meta-llama/Meta-Llama-3-8B-Instruct | 3173 | not_run |
| 660470e5 | meta-llama/Meta-Llama-3-8B-Instruct | 2250 | not_run |
| 9ed82e70 | meta-llama/Meta-Llama-3-8B-Instruct | 2117 | failed |
| 9f1710f1 | deepseek-ai/DeepSeek-V2-Lite-Chat | 2408 | not_run |
| ad8d696a | meta-llama/Meta-Llama-3-8B-Instruct | 2383 | not_run |
| ccf02fcb | ibm-ai-platform/Bamba-9B | 1152 | not_run |
| d7740ea4 | meta-llama/Meta-Llama-3-8B-Instruct | 2011 | not_run |

---

## Claude Dataset Only (14 commits)

These succeeded in Claude's H100 environment. Metrics are TPOT (lower = better).

| Commit | Model | Human TPOT (ms) | Agent TPOT (ms) | Agent vs Human |
|--------|-------|-----------------|-----------------|----------------|
| 299ebb62 | Qwen/Qwen2.5-1.5B-Instruct | 4.20 | 4.30 | -2.4% |
| 30172b49 | meta-llama/Llama-3.1-8B-Instru | 27.01 | 27.06 | -0.2% |
| 58eee5f2 | meta-llama/Llama-3.1-8B-Instru | 20.41 | 18.64 | +8.7% |
| 6a417b86 | meta-llama/Llama-3.1-8B-Instru | 30.05 | 27.85 | +7.3% |
| 6d0734c5 | mistralai/Mistral-7B-Instruct- | 84.91 | 84.08 | +1.0% |
| 70b808fe | Qwen/Qwen2-VL-7B | 10.25 | 9.96 | +2.8% |
| 8a4e5c5f | meta-llama/Llama-3.1-8B-Instru | 20.54 | 20.49 | +0.2% |
| a3223766 | facebook/opt-125m | - | - | - |
| b55ed6ef | meta-llama/Llama-3.1-8B-Instru | 31.13 | 30.92 | +0.7% |
| b690e348 | ibm-ai-platform/Bamba-9B-v2 | 69.80 | 85.02 | -21.8% |
| bc7c4d20 | meta-llama/Llama-3.1-8B-Instru | 41.47 | 40.80 | +1.6% |
| ed250545 | meta-llama/Llama-3.1-8B-Instru | 18.86 | 18.67 | +1.0% |
| fc542144 | meta-llama/Llama-3.1-8B-Instru | 8.04 | 8.17 | -1.6% |
| fe66b347 | ibm-ai-platform/Bamba-9B | 71.34 | 74.58 | -4.5% |

### Claude Performance Summary

| Outcome | Count | % |
|---------|-------|---|
| Human wins (>2%) | 3 | 21.4% |
| Agent wins (>2%) | 3 | 21.4% |
| Tie (±2%) | 8 | 57.1% |


---

## Large Models - Multi-GPU Required (13 commits)

| Commit | Model | Why |
|--------|-------|-----|
| 0d243f2a | mistralai/Mixtral-8x7B-Instruct-v0.1 | 8x7B MoE |
| 0ec82edd | Qwen/Qwen3-30B-A3B | 30B params |
| 21d93c14 | mistralai/Mixtral-8x7B-v0.1 | 8x7B MoE |
| 310aca88 | meta-llama/Meta-Llama-3-70B | 70B params |
| 379da6dc | meta-llama/Meta-Llama-3-70B | 70B params |
| 7661e92e | nvidia/Nemotron-4-340B-Instruct | 340B params |
| 8aa1485f | meta-llama/Llama-4-Scout-17B-16E-Instruct | 17B x 16 experts |
| bd6028d6 | RedHatAI/Llama-4-Scout-17B-16E-Instruct-FP8-dynamic | 17B x 16 experts |
| c0569dbc | Qwen/Qwen3-30B-A3B-FP8 | 30B params |
| dae68969 | deepseek-ai/DeepSeek-R1 | 671B+ MoE |
| dcc6cfb9 | Qwen/Qwen3-30B-A3B-FP8 | 30B params |
| eefbf4a6 | Qwen/Qwen3-30B-A3B-FP8 | 30B params |
| fb0acb6c | deepseek-ai/DeepSeek-R1 | 671B+ MoE |

---

## Baseline Failures (8 commits)

Failed in both Claude and local environments.

| Commit | Model | Claude Status |
|--------|-------|---------------|
| 22dd9c27 | meta-llama/Meta-Llama-3-8B-Instruct | baseline_failed |
| 2f192835 | meta-llama/Llama-3.1-8B-Instruct | error |
| 9a3b8832 | Qwen/Qwen2.5-VL-3B-Instruct | error |
| bfdb1ba5 | meta-llama/Llama-2-7b-chat-hf | error |
| c45f3c3a | facebook/opt-13b | error |
| d4bc1a4d | facebook/opt-125m | error |
| e7523c2e | google/gemma-3-12b-it | error |
| ec3b5ce9 | meta-llama/Llama-3.1-8B-Instruct | error |

---

## Other Failures (30 commits)

Various issues preventing benchmarking.

| Commit | Model | Docker Baseline | Claude Status |
|--------|-------|-----------------|---------------|
| 25ebed2f | meta-llama/Llama-3.1-8B-Instru | - | version_bug |
| 2a052011 | Qwen/Qwen2.5-7B-Instruct | 6623 | error |
| 3092375e | meta-llama/Meta-Llama-3-8B-Ins | 4450 | baseline_failed |
| 3b61cb45 | meta-llama/Llama-3.1-8B-Instru | - | success |
| 4c822298 | meta-llama/Llama-3.1-8B-Instru | - | success |
| 61b8cea3 | meta-llama/Llama-3.2-3B-Instru | - | success |
| 67da5720 | Qwen/Qwen2.5-7B-Instruct | 6421 | exception |
| 6d646d08 | meta-llama/Meta-Llama-3-8B | 8040 | error |
| 6dd94dbe | meta-llama/Meta-Llama-3-8B | - | success |
| 83450458 | meta-llama/Meta-Llama-3-8B-Ins | 7444 | baseline_failed |
| 88693683 | meta-llama/Meta-Llama-3-8B | - | version_bug |
| 8c1e77fb | meta-llama/Llama-3.1-8B-Instru | - | success |
| 8d75fe48 | neuralmagic/Meta-Llama-3-8B-In | 6174 | error |
| 9323a315 | meta-llama/Llama-3.2-3B-Instru | - | version_bug |
| 93e5f3c5 | meta-llama/Meta-Llama-3-8B-Ins | 4912 | baseline_failed |

*... and 15 more commits*

---

## Infrastructure Failure Analysis

### Failure Breakdown (27 commits with "other failures")

| Failure Category | Count | Fixable | Root Cause |
|-----------------|-------|---------|------------|
| Wheel not found | 4 | 3 YES | Missing vLLM wheels on S3 |
| Version bug (#8791) | 5 | NO | Port binding bug in vLLM 0.6.3-0.6.4 |
| Baseline failed (no metrics) | 6 | NO | V1 engine/CUDA TMA incompatibility |
| Server failed to start | 2 | MAYBE | Platform-specific issues |
| Broken pipe | 1 | YES | Network timeout (retry locally) |
| Claude success only (latency) | 9 | PARTIAL | No TPOT metrics for latency benchmarks |

### Version Bug #8791 (5 commits - UNFIXABLE)

These legitimate performance PRs cannot be benchmarked due to vLLM port binding bug:

| Commit | vLLM Version | PR Description |
|--------|--------------|----------------|
| 25ebed2f | 0.6.4.post2.dev375 | Cache np arange for input preparation |
| 88693683 | 0.6.4.post2.dev368 | Optimize evictor v1/v2 performance |
| 9323a315 | 0.6.4.post2.dev218 | XGrammar guided decoding support |
| b2e0ad3b | 0.6.3.post2.dev398 | Reduce peak memory usage |
| f092153f | 0.6.4.post2.dev330 | Persistent buffers for input prep |

### Actionable Recoveries (4 commits)

Can run local 3-way benchmarks:
- **9474e89b** - huggyllama/llama-7b (7B)
- **e3580537** - neuralmagic/Meta-Llama-3-8B-Instruct-FP8
- **fc7b8d1e** - meta-llama/Meta-Llama-3-8B-Instruct
- **e7b20426** - 01-ai/Yi-1.5-9B-Chat (9B)

---

## Critical Observations

1. **Only 17 commits (17.7%) have complete local 3-way benchmarks**
   - This is the most reliable data for comparing human vs agent

2. **Claude dataset adds 14 more commits with 3-way TPOT data**
   - Different methodology (H100, full serving benchmark)
   - Metrics not directly comparable to local throughput

3. **Performance comparison (Local 3-way only):**
   - Human wins: 3/17 (17.6%)
   - Agent wins: 2/17 (11.8%)
   - Tie: 12/17 (70.6%)

4. **~13 commits require multi-GPU** (13.5% of total)

5. **~27 commits have infrastructure issues** - breakdown above

6. **5 commits blocked by vLLM bug #8791** - legitimate performance work that cannot be measured

7. **4 commits are recoverable** - can add to usable dataset with local runs

---

*Generated: 2026-01-10*
*See INFRASTRUCTURE_FAILURE_DETAILED_ANALYSIS.md for full breakdown*
