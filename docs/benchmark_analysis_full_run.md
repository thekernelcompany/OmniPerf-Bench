# OmniPerf Benchmark Full Run Analysis

**Run Date**: 2025-12-26
**Dataset**: Ayushnangia/omniperf_v1 (vLLM split)
**Total Commits**: 96
**Hardware**: Single NVIDIA H100 80GB

## Executive Summary

| Status | Count | Percentage |
|--------|-------|------------|
| Success | 1 | 1.0% |
| baseline_failed | 42 | 43.8% |
| no_perf_command | 37 | 38.5% |
| no_wheel | 16 | 16.7% |

---

## Detailed Failure Analysis

### Category A: No perf_command in Dataset (37 commits)

These PRs have no benchmark command defined in the dataset - they are performance-related but don't have reproducible benchmark specifications.

| # | Commit | PR Title |
|---|--------|----------|
| 1 | 3127e975 | [CI/Build] Make pre-commit faster |
| 2 | b55ed6ef | [V1][Minor] Optimize token_ids_cpu copy |
| 3 | 25ebed2f | [V1][Minor] Cache np arange to reduce input prep |
| 4 | 88693683 | [Performance][Core] Optimize event loop perf |
| 5 | f092153f | [V1] Use persistent buffers for input prep |
| 6 | 3b61cb45 | [V1] Further reduce CPU overheads in flash-attn |
| 7 | 8c1e77fb | [Kernel] Update vllm-flash-attn version |
| 8 | 83450458 | [Performance][Spec Decode] Optimize ngram lookup |
| 9 | 6e36f4fa | improve chunked prefill performance |
| 10 | e3580537 | [Performance] Enable chunked prefill and prefix cache |
| 11 | fc7b8d1e | [Performance] e2e overheads reduction |
| 12 | 6ce01f30 | [Performance] Optimize `get_seqs` |
| 13 | 89a84b0b | [Core] Use array to speedup padding |
| 14 | 9ed82e70 | [Misc] Small perf improvements |
| 15 | 8bc68e19 | [Frontend][Core] Automatically detect vLLM |
| 16 | d7740ea4 | [Core] Optimize sampler get_logprobs |
| 17 | ad8d696a | [Core] Scheduler perf fix |
| 18 | 2f192835 | [Core] latency optimization |
| 19 | b6d10354 | [Kernel] Layernorm performance optimization |
| 20 | 3a243095 | Optimize `_get_ranks` in Sampler |
| 21 | cf2f084d | Dynamic scheduler delay to improve ITL |
| 22 | ec3b5ce9 | Improve detokenization performance |
| 23 | 58eee5f2 | [PERF] Use faster decode in tokenizer |
| 24 | ed250545 | [Core] Introduce popleft_n and append_n |
| 25 | 8a4e5c5f | [V1][P/D] Enhance Performance |
| 26 | e493e485 | [V0][Bugfix] Fix parallel sampling regression |
| 27 | 3092375e | [V1][Performance] Custom serialization |
| 28 | 93e5f3c5 | [Perf] Optimize Preparing Inputs |
| 29 | b10e5198 | [V1][Minor] Optimize get_cached_block |
| 30 | 35fad35a | [V1][Sampler] Faster top-k implementation |
| 31 | 9d72daf4 | [V1][Perf] Simpler request output queues |
| 32 | 22d33bac | [FrontEnd][Perf] merge_async_iterators fast-path |
| 33 | dae68969 | [Perf] Reduce MLA CPU overheads in V1 |
| 34 | e206b543 | [v0][Core] Use xgrammar shared context |
| 35 | 6a417b86 | fix neuron performance issue |
| 36 | 4c822298 | [V1][Spec Decode] Optimize N-gram matching |
| 37 | 30172b49 | [V1] Optimize handling of sampling metadata |

**Action Required**: Dataset curation - these commits need benchmark commands added.

---

### Category B: No Wheel Available (16 commits)

Pre-built wheels missing from wheels.vllm.ai - older commits are not archived.

| # | Commit | Missing Baseline Commit |
|---|--------|------------------------|
| 1 | f26c4aee | 8936316d |
| 2 | 9323a315 | 3257d449 |
| 3 | 98f47f2a | 8c1e77fb |
| 4 | b2e0ad3b | 4a18fd14 |
| 5 | 3476ed08 | 54600709 |
| 6 | 7c01f706 | 51e971d3 |
| 7 | 80aa7e91 | bd439735 |
| 8 | 8d75fe48 | 388596c9 |
| 9 | 379da6dc | ebce310b |
| 10 | 2a052011 | 36fb68f9 |
| 11 | bfdb1ba5 | cf2f084d |
| 12 | 9474e89b | 20478c4d |
| 13 | 21d93c14 | f1c85201 |
| 14 | c45f3c3a | 7a7929ab |
| 15 | d4bc1a4d | b56b6ca0 |
| 16 | 0d243f2a | (human wheel missing) |

**Action Required**: Build wheels from source or use newer commits with available wheels.

---

### Category C: Model Too Large for Single GPU (7 commits)

These models exceed the memory capacity of a single H100 80GB.

| # | Commit | Model | Parameters | Required GPUs |
|---|--------|-------|------------|---------------|
| 1 | baeded25 | deepseek-ai/DeepSeek-V3 | 671B MoE | 8x H100 |
| 2 | ac45c44d | deepseek-ai/DeepSeek-V2 | 236B MoE | 4x H100 |
| 3 | 8aa1485f | meta-llama/Llama-4-Scout-17B-16E | 17B x 16 experts | 2x H100 |
| 4 | 4fb56914 | deepseek-ai/DeepSeek-V3-0324 | 671B MoE | 8x H100 |
| 5 | 7661e92e | nvidia/Nemotron-4-340B | 340B | 8x H100 |
| 6 | ac45c44d | deepseek-ai/DeepSeek-V2 | 236B MoE | 4x H100 |
| 7 | 4fb56914 | deepseek-ai/DeepSeek-V3-0324 | 671B MoE | 8x H100 |

**Action Required**: Use Modal with multi-GPU instances (H100:4 or H100:8).

---

### Category D: Multi-GPU Requirement in Command (2 commits)

The benchmark command explicitly requires multiple GPUs via `-tp` flag.

| # | Commit | Command | Required GPUs |
|---|--------|---------|---------------|
| 1 | 310aca88 | `--model Meta-Llama-3-70B -tp 4` | 4 GPUs |
| 2 | bd6028d6 | `--tensor-parallel-size 2` | 2 GPUs |

**Action Required**: Use Modal with appropriate GPU count.

---

### Category E: Server Startup Failures (13 commits)

Server process died during initialization within seconds.

| # | Commit | Model | Duration | Likely Cause |
|---|--------|-------|----------|--------------|
| 1 | fc542144 | Llama-3.1-8B-Instruct | 12s | CUDA initialization |
| 2 | aea94362 | Llama-3.1-8B-Instruct | 12s | CUDA initialization |
| 3 | 6d646d08 | Llama-3-8B-Instruct | 12s | vLLM 0.5.5 compatibility |
| 4 | 660470e5 | Llama-3.1-8B-Instruct | 6s | vLLM 0.5.4 doesn't support Llama-3.1 |
| 5 | 22dd9c27 | Llama-3.1-8B-Instruct | 10s | Initialization failure |
| 6 | 9a3b8832 | Qwen2.5-VL-3B-Instruct | 10s | VL model requires special config |
| 7 | e7523c2e | gemma-3-12b-it | 6s | Model architecture compatibility |
| 8 | d55e446d | Llama-3-8B | 6s | Initialization failure |
| 9 | 67da5720 | Qwen2.5-7B-Instruct | 6s | Initialization failure |
| 10 | 015069b0 | Qwen3-7B-Instruct | 6s | Qwen3 not supported by this vLLM version |
| 11 | bc7c4d20 | Llama-3.1-8B-Instruct | 8s | Initialization failure |
| 12 | 99abb8b6 | Llama-3.1-8B-Instruct | 6s | Initialization failure |
| 13 | 9badee53 | Llama-3.2-1B-Instruct | 12s | Initialization failure |

**Action Required**:
- Use models compatible with the specific vLLM version being tested
- Add `--trust-remote-code` for models that require it
- Consider using Modal for cleaner environment

---

### Category F: Benchmark Returned No Parseable Output (13 commits)

Server started successfully (evidenced by 70-180s startup times), benchmark ran, but returned no metrics.

| # | Commit | Model | Server Ready Time | Benchmark Duration |
|---|--------|-------|-------------------|-------------------|
| 1 | b690e348 | ibm-ai-platform/Bamba-9B-v2 | 107s | 6s |
| 2 | eefbf4a6 | Qwen/Qwen3-30B-A3B-FP8 | 146s | 6s |
| 3 | 61b8cea3 | meta-llama/Meta-Llama-3-8B-Instruct | 74s | 6s |
| 4 | e7b20426 | 01-ai/Yi-1.5-9B-Chat | 116s | 6s |
| 5 | 6d0734c5 | mistralai/Mistral-7B-Instruct-v0.3 | 112s | 6s |
| 6 | dcc6cfb9 | Qwen/Qwen3-30B-A3B-FP8 | 173s | 6s |
| 7 | c0569dbc | Qwen/Qwen3-30B-A3B-FP8 | 184s | 6s |
| 8 | 296f927f | ibm-ai-platform/Bamba-9B | 108s | 7s |
| 9 | ccf02fcb | ibm-ai-platform/Bamba-9B | 58s | 7s |
| 10 | fe66b347 | ibm-ai-platform/Bamba-9B | 90s | 7s |
| 11 | fb0acb6c | deepseek-ai/DeepSeek-V2-Lite | 82s | 6s |
| 12 | ca7a2d5f | deepseek-ai/DeepSeek-Coder-V2-Lite | 70s | 6s |
| 13 | 9f1710f1 | deepseek-ai/DeepSeek-V2-Lite-Chat | 68s | (28000 input tokens) |

**Observation**: The 6-7 second benchmark duration suggests the client connects to the server but immediately exits or fails to produce valid output.

**Hypotheses**:
1. Output format changed in newer vLLM versions - metric parsing regex doesn't match
2. Missing required arguments (`--dataset-name`, `--tokenizer`, etc.)
3. Client authentication or connection issue

**Action Required**: Manual debugging to capture raw benchmark output.

---

### Category G: Benchmark Timeouts (3 commits)

Aggressive benchmark parameters caused the 1-hour timeout to be exceeded.

| # | Commit | Model | Parameters | Issue |
|---|--------|-------|------------|-------|
| 1 | a3223766 | facebook/opt-125m | `--request-rate 200 --random-input-len 700` | 200 req/s creates massive backlog |
| 2 | 299ebb62 | Qwen/Qwen2.5-1.5B | `--request-rate 1 --random-input-len 1000 --random-output-len 100` | 100 prompts × 1100 tokens each |
| 3 | 70b808fe | Qwen/Qwen2-VL-7B | VL model + random dataset | Vision-language processing is slow |

**Action Required**: Reduce benchmark parameters or increase timeout.

---

### Category H: Invalid Command Syntax (4 commits)

Dataset contains malformed or invalid benchmark commands.

| # | Commit | Issue |
|---|--------|-------|
| 1 | 526de822 | Placeholders `BS`, `INPUT_LEN`, `OUTPUT_LEN`, `MODEL` not replaced |
| 2 | 2deb029d | Literal `[--use-v2-block-manager]` with brackets in command |
| 3 | 19d98e0c | Script `moe_mem.py` doesn't exist in worktree |
| 4 | fa63e710 | Hardcoded path `/data/users/ktong/llama/llm_8b_oss` doesn't exist |

**Action Required**: Fix commands in dataset.

---

### Category I: Success (1 commit)

| Commit | Model | Baseline Throughput | Human Throughput | Change |
|--------|-------|---------------------|------------------|--------|
| ce6bf3a2 | google/gemma-2b | 28,338.48 tok/s | 28,180.41 tok/s | -0.56% |

This was a `benchmark_throughput.py` run with `--input-len 256 --output-len 256`.

---

## Summary by Root Cause

| Root Cause | Count | Fixable? |
|------------|-------|----------|
| Dataset missing perf_command | 37 | Dataset curation needed |
| Model too large for 1 GPU | 9 | Modal multi-GPU |
| No wheel available | 16 | Build from source |
| Server startup failure | 13 | Version compatibility |
| No parseable benchmark output | 13 | Debug metric parsing |
| Benchmark timeout | 3 | Reduce parameters |
| Invalid command syntax | 4 | Dataset fixes |
| **Success** | **1** | - |

## Recommendations

### Immediate Actions (High Impact)
1. **Investigate metric parsing** - 13 commits have working servers but no output
2. **Deploy Modal** - 9 commits need multi-GPU (estimated cost: ~$50-100 for full run)

### Medium-Term Actions
1. **Dataset cleanup** - Fix 4 invalid commands, add 37 missing perf_commands
2. **Build wheels from source** - For 16 commits with missing wheels

### Architecture Improvements
1. **Model-GPU mapping** - Automatically detect required GPU count from model config
2. **Timeout scaling** - Scale timeout based on model size and input length
3. **Fallback models** - Use smaller proxy models when target model unavailable
