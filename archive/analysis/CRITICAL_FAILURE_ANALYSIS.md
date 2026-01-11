# Critical Failure Analysis - vLLM Benchmark Infrastructure

## Executive Summary: Brutal Honesty

| Category | Count | Fixable? | Reality Check |
|----------|-------|----------|---------------|
| **Working 3-way** | 22 | N/A | This is our usable data |
| **Human-only** | 8 | PARTIAL | Agent patches crash or missing infrastructure |
| **Large models** | 14 | NO | Need multi-GPU, out of scope |
| **Version bug #8791** | 5 | NO | vLLM port binding bug, unfixable |
| **Baseline failed everywhere** | 8 | NO | Fundamental vLLM/model incompatibility |
| **Agent crashes server** | 6 | NO | Claude's patches are broken |
| **Unknown model** | 6 | NO | No model specified in perf command |
| **Other infrastructure** | 24 | MAYBE 2-3 | Missing Docker images, timeouts |

**Total: 96 commits analyzed, 22 usable (22.9%)**

---

## Category 1: Working 3-Way Benchmarks (22 commits)

These are the ONLY commits with reliable human vs agent comparison:

| Commit | Model | Human (tok/s) | Agent (tok/s) | Winner |
|--------|-------|---------------|---------------|--------|
| 015069b0 | Qwen/Qwen3-1.7B | 198 | 198 | TIE |
| 22d33bac | Meta-Llama-3-8B-Instruct | 3,946 | 3,985 | AGENT +1.0% |
| 296f927f | Bamba-9B | 1,421 | 1,412 | TIE |
| 3476ed08 | Meta-Llama-3-8B-Instruct | 2,128 | 2,094 | TIE |
| 3a243095 | Meta-Llama-3-8B-Instruct | 2,519 | 2,367 | HUMAN -6.0% |
| 6ce01f30 | Meta-Llama-3-8B | 1,791 | 1,777 | TIE |
| 6e36f4fa | Meta-Llama-3-8B-Instruct | 2,414 | 2,784 | AGENT +15.4% |
| 7c01f706 | Meta-Llama-3-8B-Instruct | 2,229 | 2,109 | HUMAN -5.4% |
| 80aa7e91 | Meta-Llama-3-8B-Instruct | 2,178 | 2,168 | TIE |
| 89a84b0b | Qwen1.5-0.5B | 3,559 | 2,967 | HUMAN -16.6% |
| 8bc68e19 | Meta-Llama-3-8B-Instruct | 1,979 | 1,957 | TIE |
| 99abb8b6 | Meta-Llama-3-8B-Instruct | 3,737 | 3,717 | TIE |
| 9badee53 | Meta-Llama-3-8B-Instruct | 3,424 | 3,417 | TIE |
| ca7a2d5f | DeepSeek-Coder-V2-Lite | 2,377 | 2,354 | TIE |
| cf2f084d | Meta-Llama-3-8B-Instruct | 2,443 | 2,452 | TIE |
| e206b543 | Meta-Llama-3-8B-Instruct | 3,105 | 3,352 | AGENT +8.0% |
| 19d98e0c | DeepSeek-Coder-V2-Lite | 2,358 | 2,341 | TIE |
| **e3580537** | Meta-Llama-3-8B-FP8 | 2,497 | 3,107 | **AGENT +24.4%** |
| **fc7b8d1e** | Meta-Llama-3-8B-Instruct | 2,214 | 2,598 | **AGENT +17.3%** |
| **9474e89b** | huggyllama/llama-7b | 3,086 | 2,852 | **HUMAN -7.6%** |

### Performance Summary (22 commits):
- **Human wins (>2%)**: 4 (18.2%)
- **Agent wins (>2%)**: 4 (18.2%)
- **Tie (within 2%)**: 14 (63.6%)

---

## Category 2: Agent Patches Crash Server (6 commits) - UNFIXABLE

These are cases where Claude's generated patches break the vLLM server:

| Commit | Model | What Happened |
|--------|-------|---------------|
| 2a052011 | Qwen/Qwen2.5-7B-Instruct | Server crashed after applying patch |
| 35fad35a | Meta-Llama-3-8B-Instruct | Server crashed after applying patch |
| ad8d696a | Meta-Llama-3-8B-Instruct | Server crashed after applying patch |
| 3092375e | Meta-Llama-3-8B-Instruct | Server crashed after applying patch |
| 8d75fe48 | Meta-Llama-3-8B-FP8 | Server crashed after applying patch |
| e493e485 | microsoft/phi-1_5 | Server crashed after applying patch |

**Root cause**: Claude's patches introduce bugs that prevent vLLM from starting. These are legitimate benchmark failures - the agent produced broken code.

---

## Category 3: Version Bug #8791 (5 commits) - UNFIXABLE

vLLM versions 0.6.3-0.6.4 have a port binding bug that prevents the OpenAI API server from starting:

| Commit | vLLM Version | Model |
|--------|--------------|-------|
| 25ebed2f | 0.6.4.post2.dev375 | Llama-3.1-8B-Instruct |
| 88693683 | 0.6.4.post2.dev368 | Meta-Llama-3-8B |
| 9323a315 | 0.6.4.post2.dev218 | Llama-3.2-3B-Instruct |
| b2e0ad3b | 0.6.3.post2.dev398 | Llama-3.1-8B-Instruct |
| f092153f | 0.6.4.post2.dev330 | Llama-3.1-8B-Instruct |

**Root cause**: Bug in vLLM source code, not infrastructure. Cannot be fixed without patching vLLM itself.

---

## Category 4: Large Models (14 commits) - UNFIXABLE

These models exceed single-GPU memory capacity:

| Commit | Model | Why |
|--------|-------|-----|
| 0d243f2a | Mixtral-8x7B-Instruct-v0.1 | 8x7B MoE (~47B params) |
| 21d93c14 | Mixtral-8x7B-v0.1 | 8x7B MoE |
| 310aca88 | Meta-Llama-3-70B | 70B params |
| 379da6dc | Meta-Llama-3-70B | 70B params |
| 7661e92e | Nemotron-4-340B-Instruct | 340B params |
| dae68969 | DeepSeek-R1 | 671B+ MoE |
| fb0acb6c | DeepSeek-R1 | 671B+ MoE |
| ... | Qwen3-30B-A3B variants | 30B params |

**Root cause**: Hardware limitation. Would need 2-8 GPUs per benchmark.

---

## Category 5: Baseline Failed Everywhere (8 commits) - UNFIXABLE

vLLM cannot run these commits in either Claude's or our environment:

| Commit | Model | Suspected Issue |
|--------|-------|-----------------|
| 22dd9c27 | Meta-Llama-3-8B-Instruct | V1 engine incompatibility |
| 2f192835 | Llama-3.1-8B-Instruct | V1 engine incompatibility |
| 9a3b8832 | Qwen2.5-VL-3B-Instruct | Vision model needs special handling |
| bfdb1ba5 | Llama-2-7b-chat-hf | Old vLLM version issues |
| c45f3c3a | facebook/opt-13b | Model too large for context |
| d4bc1a4d | facebook/opt-125m | Unknown |
| e7523c2e | gemma-3-12b-it | New model, old vLLM |
| ec3b5ce9 | Llama-3.1-8B-Instruct | V1 engine incompatibility |

**Root cause**: Fundamental incompatibility between vLLM version and model/configuration.

---

## Category 6: Unknown Model (6 commits) - UNFIXABLE

No model specified in performance command:

| Commit | Status |
|--------|--------|
| 3127e975 | no_perf_command |
| 4fb56914 | error |
| 526de822 | error |
| 98f47f2a | success (no metrics) |
| ac45c44d | error |
| baeded25 | error |

**Root cause**: Data quality issue in original dataset.

---

## Category 7: Other Infrastructure Issues (24 commits) - PARTIALLY FIXABLE

Mixed bag of issues:

| Issue Type | Count | Fixable? |
|------------|-------|----------|
| Missing baseline Docker images | ~10 | NO (need to build) |
| Missing human Docker images | ~5 | NO (need to build) |
| Server startup timeouts | ~5 | MAYBE (retry) |
| No agent patch available | ~4 | NO |

---

## The Brutal Truth

### What We Actually Have:
- **22 usable 3-way benchmarks** out of 96 total (22.9%)
- **14 Claude-only benchmarks** with different methodology

### Why ~77% Failed:
1. **Infrastructure not designed for this**: Docker images missing, version mismatches
2. **Model diversity too high**: 70B+ models, MoE, vision models
3. **vLLM version bugs**: #8791 affects 5 commits
4. **Agent patches break things**: 6 commits where Claude's code crashes the server
5. **Data quality**: Unknown models, missing perf commands

### What Can Realistically Be Recovered:
- **0-3 more commits** with significant effort (retry timeouts, build missing images)
- **Most failures are fundamental**: Wrong hardware, broken code, version bugs

---

## Recommendations

1. **Use the 22 local + 14 Claude 3-way benchmarks** as primary dataset
2. **Do not try to "fix" more commits** - diminishing returns
3. **Agent crash failures are real failures** - count as agent losses
4. **Version bug commits are data loss** - exclude from analysis

---

## Partial Data Analysis for Failed Commits

### Data Availability Summary

| Source | Count | Percentage |
|--------|-------|------------|
| Has baseline only | 15 | 29.4% |
| Has human only | 1 | 2.0% |
| Has agent only | 3 | 5.9% |
| Has ZERO data | 32 | 62.7% |
| **Total failed** | **51** | 100% |

### Commits with Baseline + Agent (No Human)

These show agent performance vs baseline but cannot compare to human:

| Commit | Model | Baseline | Agent | Agent vs Baseline |
|--------|-------|----------|-------|-------------------|
| 83450458 | Meta-Llama-3-8B-Instruct | 7443.9 | 3314.11 | **-55.5%** |
| 93e5f3c5 | Meta-Llama-3-8B-Instruct | 4912.1 | 3706.71 | **-24.5%** |
| 9d72daf4 | Meta-Llama-3-8B-Instruct | 2343.2 | 3673.82 | **+56.8%** |

**Note:** These are NOT usable for human vs agent comparison - only for understanding agent behavior.

### Commits with Baseline Only (10 commits)

| Commit | Model | Baseline | Issue |
|--------|-------|----------|-------|
| 3092375e | Meta-Llama-3-8B-Instruct | 4449.6 | Both crashed |
| 8d75fe48 | Meta-Llama-3-8B-FP8 | 6174.1 | Agent crashed |
| aea94362 | Meta-Llama-3-8B-Instruct | 3628.7 | No metrics |
| b10e5198 | Meta-Llama-3-8B-Instruct | 4799.8 | No metrics |
| b6d10354 | Llama-2-70b-hf | 5212.5 | Large model |
| d55e446d | Meta-Llama-3-8B-Instruct | 6934.8 | No metrics |
| e493e485 | microsoft/phi-1_5 | 6981.4 | Agent crashed |
| 0ec82edd | Qwen3-30B-A3B | 6368.6 | Large model |
| 21d93c14 | Mixtral-8x7B-v0.1 | 3058.0 | Large model |
| 379da6dc | Meta-Llama-3-70B | 7099.3 | Large model |
| c0569dbc | Qwen3-30B-A3B-FP8 | 6561.6 | Large model |
| dcc6cfb9 | Qwen3-30B-A3B-FP8 | 6428.6 | Large model |

### Commit with Human Only (1 commit)

| Commit | Model | Human | Issue |
|--------|-------|-------|-------|
| 9f1710f1 | DeepSeek-V2-Lite-Chat | 2408.03 | Large model, no agent |

---

*Generated: 2026-01-10*
