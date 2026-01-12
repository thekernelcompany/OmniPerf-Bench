# Comprehensive vLLM Benchmark Analysis

**Generated**: 2026-01-11 18:29 (Updated 2026-01-12 with Docker reruns)
**Total Unique Commits**: 96
**Commits with Metrics**: 73
**Truly Comparable 3-Way**: 21 (19 Modal + 2 Docker)

---

## Executive Summary

| Category | Count | Description |
|----------|-------|-------------|
| **Valid 3-Way Comparisons** | 21 | Same metric, same source, same config (19 Modal + 2 Docker) |
| **H+A Only (valid for A vs H)** | 12 | Can compare agent to human directly |
| **Model Mismatch (A vs H valid)** | 7 | Baseline invalid, but H vs A same model |
| **Agent Failures (Modal B+H)** | 6 | Human measured, agent FAILED (Modal pipeline) |
| **Agent Failures (Separate H-only)** | 5 | Human measured, agent crashed/no output (Separate pipeline) |
| **Total Evaluable for A vs H** | **51** | Valid for agent assessment |
| **Partial/No Metrics** | 45 | Incomplete data |
| **TOTAL** | **96** | |

### Agent Performance (n=51 evaluable commits)

| Outcome | Count | % |
|---------|-------|---|
| Agent wins (>5% better) | 7 | **14%** |
| Agent matches (±5%) | 26 | **51%** |
| Agent loses (>5% worse) | 7 | **14%** |
| Agent **FAILED** | 11 | **22%** |

*Note: 2 commits added via Docker reruns (9f1710f1: MATCH, 2deb029d: LOSS)*

**Key insight:** H+A commits without baseline ARE valid benchmark outcomes - they show agent vs human comparison directly. The 12 "H+A only" and 7 "model mismatch" commits all have valid Agent vs Human data.

**⚠️ Correction (2026-01-11):** 5 commits previously categorized as "HUMAN_ONLY" (non-evaluable) were actually **Agent Failures** - the agent produced patches that crashed or generated no metrics. These are valid benchmark outcomes showing the agent failed to match human optimization.

---

## 📊 Data Sources Explained

### Two Benchmark Pipelines

Benchmarks were run through **two different pipelines**. Both benchmark the **same commits**, just via different infrastructure:

#### 1. Modal Pipeline
```
omniperf_results_3way_claude_code/vllm/<commit>/benchmark_result.json
```
- Runs on Modal's cloud H100 GPUs
- **Single consolidated run**: Baseline → Human → Agent executed back-to-back
- **Same configuration guaranteed**: Same model, same parameters, same hardware
- Results in ONE JSON file containing all three metrics
- ✅ **Gold standard** - no configuration drift between versions

#### 2. Separate Files Pipeline
```
baseline_benchmark_results/<commit>_baseline_result.json
agent_benchmark_results/<commit>_human_result.json
agent_benchmark_results/<commit>_agent_result.json
```
- Runs were done **separately** at different times
- Each benchmark was a standalone Docker run
- **Configuration could vary** between runs (this caused the "model mismatch" issue)
- Results in THREE separate JSON files
- ⚠️ **Risk of configuration drift** between baseline and human/agent runs

### Terminology Clarification

| Term | Meaning |
|------|---------|
| **"Modal"** | Data from the Modal pipeline (single consolidated run) |
| **"Separate" or "Sep"** | Data from the Separate files pipeline (three individual runs) |
| **"H+A only"** | Commit has Human and Agent metrics but no valid Baseline |
| **"Model mismatch"** | Baseline ran with different model than Human/Agent (see below) |
| **"3-way"** | Complete Baseline + Human + Agent comparison |

### What is "Model Mismatch"?

In the Separate files pipeline, some baseline runs used a **different model** than human/agent runs:

**Example: Commit `3476ed08`**

| Version | Model | Throughput |
|---------|-------|------------|
| Baseline | **Qwen/Qwen2.5-7B-Instruct** | 5387.71 tok/s |
| Human | meta-llama/Meta-Llama-3-8B-Instruct | 2127.85 tok/s |
| Agent | meta-llama/Meta-Llama-3-8B-Instruct | 2094.31 tok/s |

**Problem**: Comparing baseline (Qwen-7B) to human (Llama-8B) shows `-60.5%` "regression" - but that's just because Llama-8B is slower than Qwen-7B, not because the optimization failed.

**Solution**: We can still compare Agent vs Human since they used the **same model**:
- Agent vs Human: `(2094 - 2127) / 2127 = -1.6%` ← Valid comparison

This is why "model mismatch" commits count as **evaluable for A vs H** but **invalid for baseline comparisons**.

### What is "H+A Only"?

Some commits have Human and Agent results but **no valid baseline** (missing or failed):

**Example: Commit `e3580537`**

| Version | Model | Throughput |
|---------|-------|------------|
| Baseline | ❌ Missing | N/A |
| Human | neuralmagic/Meta-Llama-3-8B-Instruct-FP8 | 2496.89 tok/s |
| Agent | neuralmagic/Meta-Llama-3-8B-Instruct-FP8 | 3107.0 tok/s |

**What we CAN'T measure**: "Did the human optimization improve over baseline?" (no baseline to compare)

**What we CAN measure**: "Did the agent match human optimization quality?"
- Agent vs Human: `(3107 - 2497) / 2497 = +24.4%` ← Agent beat human by 24%!

This is why "H+A only" commits are **evaluable for A vs H comparison**.

---

## ⚠️ Data Quality Notes

1. **Metric consistency**: Tables use throughput (tok/s) where possible
2. **TPOT fallback**: Some Modal results use TPOT (ms) - noted in Source column
3. **Config differences**: Baseline vs Human may use different benchmark parameters

### ⚠️ Suspicious Regressions (7 commits)

The following commits from "Separate" (S) source show **Human slower than Baseline** by 38-73%:

| Commit | Subject | Regression |
|--------|---------|------------|
| `3476ed08` | Optimize block_manager_v2 | -60.5% |
| `3a243095` | Optimize _get_ranks in Sampler | -41.6% |
| `6ce01f30` | Optimize get_seqs | -66.3% |
| `7c01f706` | Optimize SequenceStatus.is_finished | -46.3% |
| `80aa7e91` | Optimize CPU backend | -59.3% |
| `8bc68e19` | Auto-detect vLLM-tensorized | -50.1% |
| `9badee53` | Fix generation-config perf | -38.0% |

**Root cause**: Baseline runs used different benchmark configurations (likely `opt-125m` or different parameters) than Human runs. These are **NOT valid comparisons**.

### ✅ Agent Outperformed Human (4 commits)

| Commit | Agent vs Human | Subject |
|--------|---------------|---------|
| `6e36f4fa` | **+15.4%** | improve chunked prefill performance |
| `58eee5f2` | **+8.7%** | Use faster tokenizer decode |
| `6a417b86` | **+7.3%** | fix neuron performance issue |
| `98f47f2a` | **+5.3%** | Optimize FlashAttention CPU overheads |

### ❌ Agent Failed to Produce Output (5 commits)

These commits have valid Baseline + Human metrics, but the agent failed to produce working code:

| Commit | Human vs Baseline | Subject |
|--------|-------------------|---------|
| `6dd94dbe` | **+25.0%** | [perf] fix perf regression from #12253 |
| `9ed82e70` | **+10.7%** | [Misc] Small perf improvements |
| `ce6bf3a2` | **+1.4%** | [torch.compile] avoid Dynamo guard evaluation |
| `8c1e77fb` | **+0.9%** | [Kernel] Update vllm-flash-attn version |
| `3b61cb45` | **+0.0%** | [V1] Further reduce CPU overheads in flash-attn |

**Note**: These are **valid benchmark outcomes** - the agent was unable to replicate human optimization success.

---

## 1. Full 3-Way Results (30 commits)

*Including 2 Docker reruns: 9f1710f1 and 2deb029d*

| # | Commit | Subject | Model | B | H | A | H vs B | A vs B | A vs H | Src |
|---|--------|---------|-------|---|---|---|--------|--------|--------|-----|
| 1 | `299ebb62` | [Core] Speed up decode by remove sy | unknown | 4.8 | 4.2 | 4.3 | +11.8% | +9.7% | -2.4% | M(T) |
| 2 | `30172b49` | [V1] Optimize handling of sampling  | unknown | 27.0 | 27.0 | 27.1 | -0.2% | -0.4% | -0.2% | M(T) |
| 3 | `310aca88` | [perf]fix current stream (#11870) | N/A | 51.1 | 102.1 | 102.3 | +99.8% | +100.2% | +0.2% | M |
| 4 | `3476ed08` | [Core] Optimize block_manager_v2 vs | meta-llama/Meta | 5387.7 | 2127.8 | 2094.3 | -60.5% | -61.1% | -1.6% | S |
| 5 | `3a243095` | Optimize `_get_ranks` in Sampler (# | meta-llama/Meta | 4312.0 | 2518.8 | 2366.8 | -41.6% | -45.1% | -6.0% | S |
| 6 | `4c822298` | [V1][Spec Decode] Optimize N-gram m | unknown | 102.1 | 153.2 | 153.5 | +50.0% | +50.3% | +0.2% | M |
| 7 | `58eee5f2` | [PERF] Use faster way of decode in  | N/A | 19.6 | 20.4 | 18.6 | -4.3% | +4.7% | +8.7% | M(T) |
| 8 | `61b8cea3` | [Attention] Optimize FlashInfer Met | meta-llama/Meta | 74.9 | 75.0 | 75.0 | +0.1% | +0.1% | +0.0% | M |
| 9 | `6a417b86` | fix neuron performance issue (#1358 | unknown | 67.0 | 30.1 | 27.9 | +55.2% | +58.4% | +7.3% | M(T) |
| 10 | `6ce01f30` | [Performance] Optimize `get_seqs` ( | meta-llama/Meta | 5310.0 | 1790.9 | 1777.4 | -66.3% | -66.5% | -0.8% | S |
| 11 | `6d0734c5` | [NVIDIA] Add SM100 Flashinfer MoE b | mistralai/Mistr | 83.3 | 84.9 | 84.1 | -2.0% | -1.0% | +1.0% | M(T) |
| 12 | `6e36f4fa` | improve chunked prefill performance | meta-llama/Meta | 1615.8 | 2413.6 | 2784.2 | +49.4% | +72.3% | +15.4% | S |
| 13 | `70b808fe` | [Perf]:Optimize qwen2-vl to reduce  | unknown | 10.4 | 10.2 | 10.0 | +1.3% | +4.0% | +2.8% | M(T) |
| 14 | `7c01f706` | [Core] Optimize `SequenceStatus.is_ | meta-llama/Meta | 4153.5 | 2229.4 | 2109.4 | -46.3% | -49.2% | -5.4% | S |
| 15 | `80aa7e91` | [Hardware][Intel] Optimize CPU back | meta-llama/Meta | 5355.5 | 2178.3 | 2167.7 | -59.3% | -59.5% | -0.5% | S |
| 16 | `8a4e5c5f` | [V1][P/D]Enhance Performance and co | N/A | 20.3 | 20.5 | 20.5 | -1.1% | -0.9% | +0.2% | M(T) |
| 17 | `8bc68e19` | [Frontend] [Core] perf: Automatical | meta-llama/Meta | 3968.4 | 1979.4 | 1956.6 | -50.1% | -50.7% | -1.2% | S |
| 18 | `98f47f2a` | [V1] Optimize the CPU overheads in  | N/A | 972.5 | 972.5 | 1023.7 | 0.0% | +5.3% | +5.3% | M |
| 19 | `9badee53` | Fix performance when `--generation- | meta-llama/Meta | 5526.4 | 3424.2 | 3417.1 | -38.0% | -38.2% | -0.2% | S |
| 20 | `a3223766` | [Core] Optimize update checks in Lo | N/A | N/A | N/A | N/A | N/A | N/A | N/A |  |
| 21 | `b55ed6ef` | [V1][Minor] Optimize token_ids_cpu  | N/A | 35.6 | 31.1 | 30.9 | +12.5% | +13.1% | +0.7% | M(T) |
| 22 | `b690e348` | [Model] Mamba2 preallocate SSM outp | ibm-ai-platform | 78.7 | 69.8 | 85.0 | +11.3% | -8.1% | -21.8% | M(T) |
| 23 | `bc7c4d20` | [Kernel][ROCM] Upstream prefix pref | unknown | 40.7 | 41.5 | 40.8 | -1.9% | -0.2% | +1.6% | M(T) |
| 24 | `ed250545` | [Core] Introduce popleft_n and appe | N/A | 19.0 | 18.9 | 18.7 | +0.8% | +1.8% | +1.0% | M(T) |
| 25 | `f26c4aee` | [Misc] Optimize ray worker initiali | N/A | 818.8 | 818.9 | 818.7 | +0.0% | -0.0% | -0.0% | M |
| 26 | `fa63e710` | [V1][Perf] Reduce scheduling overhe | N/A | N/A | N/A | N/A | N/A | N/A | N/A |  |
| 27 | `fc542144` | [Feature] Fix guided decoding block | meta-llama/Llam | 8.0 | 8.0 | 8.2 | 0.0% | -1.6% | -1.6% | M(T) |
| 28 | `fe66b347` | [Model] Mamba2 Prefill Performance  | unknown | 82.2 | 71.3 | 74.6 | +13.2% | +9.3% | -4.5% | M(T) |
| 29 | `9f1710f1` | Fix mla prefill context perf (#13897) | DeepSeek-V2-Lite | 60.3 | 60.2 | 59.6 | -0.1% | -1.1% | -1.0% | D* |
| 30 | `2deb029d` | [Performance][BlockManagerV2] Mark prefix | Llama-3-8B-FP8 | 5671 | 5686 | 5711 | +0.3% | +0.7% | +0.4% | D* |

*D* = Docker rerun with full 3-way benchmark data

---

## 2. Baseline + Human (6 commits)

| # | Commit | Subject | Model | Baseline | Human | H vs B | Source |
|---|--------|---------|-------|----------|-------|--------|--------|
| 1 | `3b61cb45` | [V1] Further reduce CPU overheads i | N/A | 9819.5 | 9822.4 | +0.0% | Modal |
| 2 | `6dd94dbe` | [perf] fix perf regression from #12 | meta-llama/Meta | 204.7 | 255.9 | +25.0% | Modal |
| 3 | `8c1e77fb` | [Kernel] Update vllm-flash-attn ver | N/A | 10117.1 | 10206.3 | +0.9% | Modal |
| 4 | `9ed82e70` | [Misc] Small perf improvements (#65 | meta-llama/Meta | 1912.8 | 2116.8 | +10.7% | Sep |
| 5 | `ce6bf3a2` | [torch.compile] avoid Dynamo guard  | N/A | 54.7 | 55.5 | +1.4% | Modal |
| 6 | `d7740ea4` | [Core] Optimize sampler get_logprob | meta-llama/Meta | 7592.4 | 2011.3 | -73.5% | Sep |


---

## 3. Human + Agent (12 commits)

| # | Commit | Subject | Model | Human | Agent | A vs H | Source |
|---|--------|---------|-------|-------|-------|--------|--------|
| 1 | `015069b0` | [Misc] Optimize the Qwen3_Reasoning | Qwen/Qwen3-1.7B | 198.3 | 198.3 | 0.0% | Sep |
| 2 | `19d98e0c` | N/A | deepseek-ai/Dee | 2358.4 | 2340.7 | -0.8% | Sep |
| 3 | `22d33bac` | [FrontEnd][Perf] `merge_async_itera | meta-llama/Meta | 3946.1 | 3984.8 | +1.0% | Sep |
| 4 | `296f927f` | [Model] RE: Mamba2 Prefill Performa | ibm-ai-platform | 1421.5 | 1411.8 | -0.7% | Sep |
| 5 | `89a84b0b` | [Core] Use array to speedup padding | Qwen/Qwen1.5-0. | 3558.5 | 2967.5 | -16.6% | Sep |
| 6 | `9474e89b` | [PREFIX CACHING FOLLOW UP] A bunch  | huggyllama/llam | 3086.4 | 2852.5 | -7.6% | Sep |
| 7 | `99abb8b6` | [V1][Spec Decode] Optimize Rejectio | meta-llama/Meta | 3736.7 | 3716.5 | -0.5% | Sep |
| 8 | `ca7a2d5f` | Revert "[Perf] Reduce MLA CPU overh | deepseek-ai/Dee | 2376.7 | 2353.8 | -1.0% | Sep |
| 9 | `cf2f084d` | Dynamic scheduler delay to improve  | meta-llama/Meta | 2443.1 | 2451.8 | +0.4% | Sep |
| 10 | `e206b543` | [v0][Core] Use xgrammar shared cont | meta-llama/Meta | 3105.1 | 3352.1 | +8.0% | Sep |
| 11 | `e3580537` | [Performance] Enable chunked prefil | neuralmagic/Met | 2496.9 | 3107.0 | +24.4% | Sep |
| 12 | `fc7b8d1e` | [Performance] e2e overheads reducti | meta-llama/Meta | 2214.0 | 2598.0 | +17.3% | Sep |


---

## 4. Human Only (25 commits)

*Note: 2deb029d and 9f1710f1 moved to Valid 3-Way via Docker reruns*

| # | Commit | Subject | Model | Throughput | TPOT | Source |
|---|--------|---------|-------|------------|------|--------|
| 1 | `0ec82edd` | [perf] Speed up align sum kernels ( | facebook/opt-12 | 6368.6 | N/A | D |
| 2 | `21d93c14` | Optimize Mixtral with expert parall | None | 3058.0 | N/A | D |
| 3 | `2a052011` | [Kernel] Support MoE Fp8 Checkpoint | Qwen/Qwen2.5-7B | N/A | N/A | H |
| 4 | `3092375e` | [V1][Performance] Implement custom  | meta-llama/Meta | N/A | N/A | H |
| 5 | `35fad35a` | [V1][Sampler] Faster top-k only imp | meta-llama/Meta | 3172.7 | N/A | H |
| 6 | `379da6dc` | [Kernel] [FP8] Improve FP8 linear l | None | 7099.3 | N/A | D |
| 7 | `526de822` | [Kernel][Triton][AMD] Use block siz | None | 7413.6 | N/A | D |
| 8 | `660470e5` | [Core] Optimize evictor-v2 performa | meta-llama/Meta | 2250.3 | N/A | H |
| 9 | `67da5720` | [PERF] Speed up Qwen2.5-VL model by | Qwen/Qwen2.5-7B | N/A | N/A | H |
| 10 | `6d646d08` | [Core] Optimize Async + Multi-step  | meta-llama/Meta | N/A | N/A | H |
| 11 | `83450458` | [Performance][Spec Decode] Optimize | meta-llama/Meta | N/A | N/A | H |
| 12 | `8d75fe48` | [Kernel] Switch fp8 layers to use t | neuralmagic/Met | N/A | N/A | H |
| 13 | `93e5f3c5` | [Perf] Optimize Preparing Inputs fo | meta-llama/Meta | N/A | N/A | H |
| 14 | `9d72daf4` | [V1][Perf] Simpler request output q | meta-llama/Meta | N/A | N/A | H |
| 15 | `ad8d696a` | [Core] Scheduler perf fix (#4270) | meta-llama/Meta | 2382.5 | N/A | H |
| 16 | `aea94362` | [Frontend][V1] Online serving perfo | meta-llama/Meta | N/A | N/A | H |
| 17 | `b10e5198` | [V1][Minor] Optimize get_cached_blo | meta-llama/Meta | N/A | N/A | H |
| 18 | `b6d10354` | [Kernel] Layernorm performance opti | None | 5212.5 | N/A | D |
| 19 | `c0569dbc` | [Misc] ModularKernel : Perform Weig | facebook/opt-12 | 6561.6 | N/A | D |
| 20 | `ccf02fcb` | Revert "[Model] Mamba2 Prefill Perf | ibm-ai-platform | 1152.3 | N/A | H |
| 21 | `d55e446d` | [V1][Spec Decode] Small refactors t | meta-llama/Meta | N/A | N/A | H |
| 22 | `dcc6cfb9` | [Kernel][Performance] Tweak MoE Bat | facebook/opt-12 | 6428.6 | N/A | D |
| 23 | `e493e485` | [V0][Bugfix] Fix parallel sampling  | microsoft/phi-1 | N/A | N/A | H |
| 24 | `e7b20426` | Revert "[Performance] Performance i | 01-ai/Yi-1.5-9B | 2774.9 | N/A | H |
| 25 | `eefbf4a6` | [Perf] Optimize `reshape_and_cache_ | None | 6499.3 | N/A | D |


---

## 5. No Metrics (23 commits)

| # | Commit | Subject | Status | Error |
|---|--------|---------|--------|-------|
| 1 | `0d243f2a` | [ROCm][MoE] mi300 mixtral8x7B perf  | error | N/A |
| 2 | `22dd9c27` | [Kernel] Optimize Prefill Attention | baseline_failed | N/A |
| 3 | `25ebed2f` | [V1][Minor] Cache np arange to redu | version_bug | N/A |
| 4 | `2f192835` | [Core] latency optimization (#3890) | error | N/A |
| 5 | `3127e975` | [CI/Build] Make pre-commit faster ( | no_perf_command | No perf_command in dataset |
| 6 | `4fb56914` | [perf] Add fused MLA QKV + strided  | error | Baseline server failed to start |
| 7 | `7661e92e` | [Model] Optimize nemotron_h impleme | exception | N/A |
| 8 | `88693683` | [Performance][Core] Optimize the pe | version_bug | N/A |
| 9 | `8aa1485f` | [Perf] Disable chunked local attent | baseline_failed | N/A |
| 10 | `9323a315` | [Core][Performance] Add XGrammar su | version_bug | N/A |
| 11 | `9a3b8832` | [PERF] Speedup of MRoPE prepare inp | error | N/A |
| 12 | `ac45c44d` | [Bugfix] [Performance] DeepEPHighTh | error | Baseline server failed to start |
| 13 | `b2e0ad3b` | [Perf] Reduce peak memory usage of  | version_bug | N/A |
| 14 | `baeded25` | [Attention] Deepseek v3 MLA support | error | Baseline server failed to start |
| 15 | `bd6028d6` | Optimized topk for topk=1 (Llama-4) | baseline_failed | N/A |
| 16 | `bfdb1ba5` | [Core] Improve detokenization perfo | error | N/A |
| 17 | `c45f3c3a` | Optimize tensor parallel execution  | error | N/A |
| 18 | `d4bc1a4d` | Add unoptimized OPT Attention | error | N/A |
| 19 | `dae68969` | [Perf] Reduce MLA CPU overheads in  | baseline_failed | N/A |
| 20 | `e7523c2e` | [V1][Sampler] Improve performance o | error | N/A |
| 21 | `ec3b5ce9` | Improve detokenization performance  | error | N/A |
| 22 | `f092153f` | [V1] Use more persistent buffers to | version_bug | N/A |
| 23 | `fb0acb6c` | [Perf] Improve MLA on V1 (#14540) | baseline_failed | N/A |


---

## Summary

| Metric | Count | % |
|--------|-------|---|
| Total commits | 96 | 100% |
| With any metrics | 73 | 76.0% |
| Full 3-way | 28 | 29.2% |
| Can measure human opt | 34 | 35.4% |

---

*Generated by Claude Code*
## Complete Metrics Inventory (All 96 Commits)

This table shows ALL available data for EVERY commit without categorization.


### Column Legend

| Column | Source | Description |
|--------|--------|-------------|
| **M.St** | Modal | Status (success/error/exception/baseline_failed) |
| **M.B** | Modal | Baseline metric (throughput or TPOT) |
| **M.H** | Modal | Human metric |
| **M.A** | Modal | Agent metric |
| **S.B** | Separate | Baseline file throughput (tok/s) |
| **S.H** | Separate | Human file metric (throughput or TPOT) |
| **S.A** | Separate | Agent file metric |
| **D** | Docker | Docker verification throughput |

**Note**: `-` means no data available. Metrics shown are throughput (tok/s) or TPOT (ms) depending on benchmark type.

---


| # | Commit | Subject | M.St | M.B | M.H | M.A | S.B | S.H | S.A | D |
|---|--------|---------|------|-----|-----|-----|-----|-----|-----|---|
| 1 | `015069b0` | [Misc] Optimize the Qwen3_Reas | exceptio | - | - | - | - | 198.3 | 198.3 | 1428.3 |
| 2 | `0d243f2a` | [ROCm][MoE] mi300 mixtral8x7B  | error | - | - | - | - | - | - | - |
| 3 | `0ec82edd` | [perf] Speed up align sum kern | error | - | - | - | - | - | - | 6368.6 |
| 4 | `19d98e0c` |  | - | - | - | - | - | 2358.4 | 2340.7 | - |
| 5 | `21d93c14` | Optimize Mixtral with expert p | error | - | - | - | - | - | - | 3058.0 |
| 6 | `22d33bac` | [FrontEnd][Perf] `merge_async_ | baseline | - | - | - | - | 3946.1 | 3984.8 | 3025.6 |
| 7 | `22dd9c27` | [Kernel] Optimize Prefill Atte | baseline | - | - | - | - | - | - | - |
| 8 | `25ebed2f` | [V1][Minor] Cache np arange to | version_ | - | - | - | - | - | - | - |
| 9 | `296f927f` | [Model] RE: Mamba2 Prefill Per | exceptio | - | - | - | - | 1421.5 | 1411.8 | 813.3 |
| 10 | `299ebb62` | [Core] Speed up decode by remo | success | 4.8 | 4.2 | 4.3 | - | - | - | - |
| 11 | `2a052011` | [Kernel] Support MoE Fp8 Check | error | - | - | - | 1524.6 | - | - | 6623.3 |
| 12 | `2deb029d` | [Performance][BlockManagerV2]  | Docker | 5671.5* | 5685.9* | 5711.1* | - | 3094.8 | - | 7282.6 |
| 13 | `2f192835` | [Core] latency optimization (# | error | - | - | - | - | - | - | - |
| 14 | `30172b49` | [V1] Optimize handling of samp | success | 27.0 | 27.0 | 27.1 | - | - | - | - |
| 15 | `3092375e` | [V1][Performance] Implement cu | baseline | - | - | - | - | - | - | 4449.6 |
| 16 | `310aca88` | [perf]fix current stream (#118 | success | 51.1 | 102.1 | 102.3 | - | - | - | - |
| 17 | `3127e975` | [CI/Build] Make pre-commit fas | no_perf_ | - | - | - | - | - | - | - |
| 18 | `3476ed08` | [Core] Optimize block_manager_ | error | - | - | - | 5387.7 | 2127.8 | 2094.3 | 6224.8 |
| 19 | `35fad35a` | [V1][Sampler] Faster top-k onl | baseline | - | - | - | - | 3172.7 | - | 2301.7 |
| 20 | `379da6dc` | [Kernel] [FP8] Improve FP8 lin | error | - | - | - | - | - | - | 7099.3 |
| 21 | `3a243095` | Optimize `_get_ranks` in Sampl | error | - | - | - | 4312.0 | 2518.8 | 2366.8 | 7125.0 |
| 22 | `3b61cb45` | [V1] Further reduce CPU overhe | success | 9819.5 | 9822.4 | - | - | - | - | - |
| 23 | `4c822298` | [V1][Spec Decode] Optimize N-g | success | 102.1 | 153.2 | 153.5 | - | - | - | - |
| 24 | `4fb56914` | [perf] Add fused MLA QKV + str | error | - | - | - | - | - | - | - |
| 25 | `526de822` | [Kernel][Triton][AMD] Use bloc | error | - | - | - | - | - | - | 7413.6 |
| 26 | `58eee5f2` | [PERF] Use faster way of decod | success | 19.6 | 20.4 | 18.6 | - | - | - | - |
| 27 | `61b8cea3` | [Attention] Optimize FlashInfe | success | 74.9 | 75.0 | 75.0 | - | - | - | - |
| 28 | `660470e5` | [Core] Optimize evictor-v2 per | error | - | - | - | - | 2250.3 | - | 7150.3 |
| 29 | `67da5720` | [PERF] Speed up Qwen2.5-VL mod | exceptio | - | - | - | 2377.1 | - | 4694.1 | 6421.1 |
| 30 | `6a417b86` | fix neuron performance issue ( | success | 67.0 | 30.1 | 27.9 | - | - | - | - |
| 31 | `6ce01f30` | [Performance] Optimize `get_se | error | - | - | - | 5310.0 | 1790.9 | 1777.4 | 7062.1 |
| 32 | `6d0734c5` | [NVIDIA] Add SM100 Flashinfer  | success | 83.3 | 84.9 | 84.1 | - | - | - | - |
| 33 | `6d646d08` | [Core] Optimize Async + Multi- | error | - | - | - | 1057.0 | - | 2380.4 | 8039.8 |
| 34 | `6dd94dbe` | [perf] fix perf regression fro | success | 204.7 | 255.9 | - | - | - | - | - |
| 35 | `6e36f4fa` | improve chunked prefill perfor | error | - | - | - | 1615.8 | 2413.6 | 2784.2 | 7855.2 |
| 36 | `70b808fe` | [Perf]:Optimize qwen2-vl to re | success | 10.4 | 10.2 | 10.0 | - | - | - | - |
| 37 | `7661e92e` | [Model] Optimize nemotron_h im | exceptio | - | - | - | - | - | - | - |
| 38 | `7c01f706` | [Core] Optimize `SequenceStatu | error | - | - | - | 4153.5 | 2229.4 | 2109.4 | 6213.9 |
| 39 | `80aa7e91` | [Hardware][Intel] Optimize CPU | error | - | - | - | 5355.5 | 2178.3 | 2167.7 | 6369.9 |
| 40 | `83450458` | [Performance][Spec Decode] Opt | baseline | - | - | - | - | - | 3314.1 | 7443.9 |
| 41 | `88693683` | [Performance][Core] Optimize t | version_ | - | - | - | - | - | - | - |
| 42 | `89a84b0b` | [Core] Use array to speedup pa | error | - | - | - | - | 3558.5 | 2967.5 | 6003.6 |
| 43 | `8a4e5c5f` | [V1][P/D]Enhance Performance a | success | 20.3 | 20.5 | 20.5 | - | - | - | - |
| 44 | `8aa1485f` | [Perf] Disable chunked local a | baseline | - | - | - | - | - | - | - |
| 45 | `8bc68e19` | [Frontend] [Core] perf: Automa | error | - | - | - | 3968.4 | 1979.4 | 1956.6 | 6874.5 |
| 46 | `8c1e77fb` | [Kernel] Update vllm-flash-att | success | 10117.1 | 10206.3 | - | - | - | - | - |
| 47 | `8d75fe48` | [Kernel] Switch fp8 layers to  | error | - | - | - | 5275.8 | - | - | 6174.1 |
| 48 | `9323a315` | [Core][Performance] Add XGramm | version_ | - | - | - | - | - | - | - |
| 49 | `93e5f3c5` | [Perf] Optimize Preparing Inpu | baseline | - | - | - | - | - | 3706.7 | 4912.1 |
| 50 | `9474e89b` | [PREFIX CACHING FOLLOW UP] A b | error | - | - | - | - | 3086.4 | 2852.5 | 7183.6 |
| 51 | `98f47f2a` | [V1] Optimize the CPU overhead | success | 972.5 | 972.5 | 1023.7 | - | - | - | - |
| 52 | `99abb8b6` | [V1][Spec Decode] Optimize Rej | exceptio | - | - | - | - | 3736.7 | 3716.5 | 2408.2 |
| 53 | `9a3b8832` | [PERF] Speedup of MRoPE prepar | error | - | - | - | - | - | - | - |
| 54 | `9badee53` | Fix performance when `--genera | baseline | - | - | - | 5526.4 | 3424.2 | 3417.1 | 8057.6 |
| 55 | `9d72daf4` | [V1][Perf] Simpler request out | baseline | - | - | - | - | - | 3673.8 | 2343.2 |
| 56 | `9ed82e70` | [Misc] Small perf improvements | error | - | - | - | 1912.8 | 2116.8 | - | 5615.5 |
| 57 | `9f1710f1` | Fix mla prefill context perf | Docker | 60.29* | 60.21* | 59.60* | - | 2408.0 | - | - |
| 58 | `a3223766` | [Core] Optimize update checks  | success | 0.0 | 0.0 | 0.0 | - | - | - | - |
| 59 | `ac45c44d` | [Bugfix] [Performance] DeepEPH | error | - | - | - | - | - | - | - |
| 60 | `ad8d696a` | [Core] Scheduler perf fix (#42 | error | - | - | - | - | 2382.5 | - | 6573.2 |
| 61 | `aea94362` | [Frontend][V1] Online serving  | baseline | - | - | - | - | - | - | 3628.7 |
| 62 | `b10e5198` | [V1][Minor] Optimize get_cache | baseline | - | - | - | - | - | - | 4799.8 |
| 63 | `b2e0ad3b` | [Perf] Reduce peak memory usag | version_ | - | - | - | - | - | - | - |
| 64 | `b55ed6ef` | [V1][Minor] Optimize token_ids | success | 35.6 | 31.1 | 30.9 | - | - | - | - |
| 65 | `b690e348` | [Model] Mamba2 preallocate SSM | success | 78.7 | 69.8 | 85.0 | - | - | - | - |
| 66 | `b6d10354` | [Kernel] Layernorm performance | error | - | - | - | - | - | - | 5212.5 |
| 67 | `baeded25` | [Attention] Deepseek v3 MLA su | error | - | - | - | - | - | - | - |
| 68 | `bc7c4d20` | [Kernel][ROCM] Upstream prefix | success | 40.7 | 41.5 | 40.8 | - | - | - | - |
| 69 | `bd6028d6` | Optimized topk for topk=1 (Lla | baseline | - | - | - | - | - | - | - |
| 70 | `bfdb1ba5` | [Core] Improve detokenization  | error | - | - | - | - | - | - | - |
| 71 | `c0569dbc` | [Misc] ModularKernel : Perform | exceptio | - | - | - | - | - | - | 6561.6 |
| 72 | `c45f3c3a` | Optimize tensor parallel execu | error | - | - | - | - | - | - | - |
| 73 | `ca7a2d5f` | Revert "[Perf] Reduce MLA CPU  | exceptio | - | - | - | - | 2376.7 | 2353.8 | 8307.8 |
| 74 | `ccf02fcb` | Revert "[Model] Mamba2 Prefill | exceptio | - | - | - | - | 1152.3 | - | 5085.4 |
| 75 | `ce6bf3a2` | [torch.compile] avoid Dynamo g | success | 54.7 | 55.5 | - | - | - | - | - |
| 76 | `cf2f084d` | Dynamic scheduler delay to imp | error | - | - | - | - | 2443.1 | 2451.8 | 7080.5 |
| 77 | `d4bc1a4d` | Add unoptimized OPT Attention | error | - | - | - | - | - | - | - |
| 78 | `d55e446d` | [V1][Spec Decode] Small refact | error | - | - | - | - | - | - | 6934.8 |
| 79 | `d7740ea4` | [Core] Optimize sampler get_lo | error | - | - | - | 7592.4 | 2011.3 | - | 6946.3 |
| 80 | `dae68969` | [Perf] Reduce MLA CPU overhead | baseline | - | - | - | - | - | - | - |
| 81 | `dcc6cfb9` | [Kernel][Performance] Tweak Mo | exceptio | - | - | - | - | - | - | 6428.6 |
| 82 | `e206b543` | [v0][Core] Use xgrammar shared | baseline | - | - | - | - | 3105.1 | 3352.1 | 8148.0 |
| 83 | `e3580537` | [Performance] Enable chunked p | error | - | - | - | - | 2496.9 | 3107.0 | 7440.7 |
| 84 | `e493e485` | [V0][Bugfix] Fix parallel samp | error | - | - | - | - | - | - | 6981.4 |
| 85 | `e7523c2e` | [V1][Sampler] Improve performa | error | - | - | - | - | - | - | - |
| 86 | `e7b20426` | Revert "[Performance] Performa | exceptio | - | - | - | - | 2774.9 | - | 6582.5 |
| 87 | `ec3b5ce9` | Improve detokenization perform | error | - | - | - | - | - | - | - |
| 88 | `ed250545` | [Core] Introduce popleft_n and | success | 19.0 | 18.9 | 18.7 | - | - | - | - |
| 89 | `eefbf4a6` | [Perf] Optimize `reshape_and_c | error | 2026.7 | - | - | - | - | - | 6499.3 |
| 90 | `f092153f` | [V1] Use more persistent buffe | version_ | - | - | - | - | - | - | - |
| 91 | `f26c4aee` | [Misc] Optimize ray worker ini | success | 818.8 | 818.9 | 818.7 | - | - | - | - |
| 92 | `fa63e710` | [V1][Perf] Reduce scheduling o | success | 1331.7 | 1323.8 | 1329.9 | - | - | - | - |
| 93 | `fb0acb6c` | [Perf] Improve MLA on V1 (#145 | baseline | - | - | - | - | - | - | - |
| 94 | `fc542144` | [Feature] Fix guided decoding  | success | 8.0 | 8.0 | 8.2 | - | - | - | - |
| 95 | `fc7b8d1e` | [Performance] e2e overheads re | error | - | - | - | - | 2214.0 | 2598.0 | 7174.5 |
| 96 | `fe66b347` | [Model] Mamba2 Prefill Perform | success | 82.2 | 71.3 | 74.6 | - | - | - | - |

---

### Data Availability Summary


| Source | Has Data | Description |
|--------|----------|-------------|
| Modal (any status) | 94 | Commits with Modal result file |
| Modal 3-way | 20 | B + H + A metrics from Modal |
| Modal B+H only | 4 | B + H metrics (no agent) |
| Separate Baseline | 14 | Baseline file with throughput |
| Separate Human | 29 | Human file with metrics |
| Separate Agent | 25 | Agent file with metrics |
| Docker verification | 47 | Docker run with metrics |

---

## Metric Analysis (Critical Review)

### All Unique Metrics Found

| Metric | Unit | Better | Source(s) | Commits | Description |
|--------|------|--------|-----------|---------|-------------|
| `tpot_mean` | ms | Lower | Modal | 13 | Time Per Output Token mean (serving) |
| `tpot_median` | ms | Lower | Modal | 13 | TPOT median |
| `tpot_p99` | ms | Lower | Modal | 13 | TPOT 99th percentile |
| `ttft_mean` | ms | Lower | Modal | 14 | Time To First Token mean (serving) |
| `ttft_median` | ms | Lower | Modal | 14 | TTFT median |
| `ttft_p99` | ms | Lower | Modal | 14 | TTFT 99th percentile |
| `itl_mean` | ms | Lower | Modal | 13 | Inter-Token Latency mean (serving) |
| `itl_median` | ms | Lower | Modal | 13 | ITL median |
| `itl_p99` | ms | Lower | Modal | 13 | ITL 99th percentile |
| `latency_avg` | ms | Lower | Modal | 9 | Average latency (standalone) |
| `throughput` | tok/s | Higher | Modal | 9 | Throughput (standalone) |
| `throughput_tok_s` | tok/s | Higher | Sep, Docker | 58 | Throughput tokens/second |
| `tpot_mean_ms` | ms | Lower | Sep | 12 | TPOT mean (separate files) |
| `ttft_mean_ms` | ms | Lower | Sep | 12 | TTFT mean (separate files) |

### Metric Availability by Category

| Category | Commits | Primary Metric | Notes |
|----------|---------|----------------|-------|
| Modal TPOT (serving) | 13 | tpot_mean | Full 3-way, same config |
| Modal Throughput (standalone) | 9 | throughput | Full 3-way, same config |
| Sep Throughput | 29 H, 25 A, 14 B | throughput_tok_s | Different models often |
| Sep TPOT | 10 H, 9 A | tpot_mean_ms | No baseline TPOT |
| Docker only | 47 | throughput_tok_s | Verification runs only |

### Truly Comparable 3-Way Results

**21 commits** have valid 3-way comparisons (same metric, same source, same config):
- 19 from Modal pipeline
- 2 from Docker reruns (9f1710f1, 2deb029d)

| # | Commit | Source | Metric | Baseline | Human | Agent | H vs B | A vs B | A vs H |
|---|--------|--------|--------|----------|-------|-------|--------|--------|--------|
| 1 | `299ebb62` | Modal-TPOT | ms | 4.8 | 4.2 | 4.3 | +11.8% | +9.7% | -2.4% |
| 2 | `30172b49` | Modal-TPOT | ms | 27.0 | 27.0 | 27.1 | -0.2% | -0.4% | -0.2% |
| 3 | `310aca88` | Modal-Tput | tok/s | 51.1 | 102.1 | 102.3 | +99.8% | +100.2% | +0.2% |
| 4 | `4c822298` | Modal-Tput | tok/s | 102.1 | 153.2 | 153.5 | +50.0% | +50.3% | +0.2% |
| 5 | `58eee5f2` | Modal-TPOT | ms | 19.6 | 20.4 | 18.6 | -4.3% | +4.7% | **+8.7%** |
| 6 | `61b8cea3` | Modal-Tput | tok/s | 74.9 | 75.0 | 75.0 | +0.1% | +0.1% | +0.0% |
| 7 | `6a417b86` | Modal-TPOT | ms | 67.0 | 30.1 | 27.9 | +55.2% | +58.4% | **+7.3%** |
| 8 | `6d0734c5` | Modal-TPOT | ms | 83.3 | 84.9 | 84.1 | -2.0% | -1.0% | +1.0% |
| 9 | `6e36f4fa` | Sep-Tput | tok/s | 1615.8 | 2413.6 | 2784.2 | +49.4% | +72.3% | **+15.4%** |
| 10 | `70b808fe` | Modal-TPOT | ms | 10.4 | 10.2 | 10.0 | +1.3% | +4.0% | +2.8% |
| 11 | `8a4e5c5f` | Modal-TPOT | ms | 20.3 | 20.5 | 20.5 | -1.1% | -0.9% | +0.2% |
| 12 | `98f47f2a` | Modal-Tput | tok/s | 972.5 | 972.5 | 1023.7 | 0.0% | +5.3% | **+5.3%** |
| 13 | `b55ed6ef` | Modal-TPOT | ms | 35.6 | 31.1 | 30.9 | +12.5% | +13.1% | +0.7% |
| 14 | `b690e348` | Modal-TPOT | ms | 78.7 | 69.8 | 85.0 | +11.3% | -8.1% | -21.8% |
| 15 | `bc7c4d20` | Modal-TPOT | ms | 40.7 | 41.5 | 40.8 | -1.9% | -0.2% | +1.6% |
| 16 | `ed250545` | Modal-TPOT | ms | 19.0 | 18.9 | 18.7 | +0.8% | +1.8% | +1.0% |
| 17 | `f26c4aee` | Modal-Tput | tok/s | 818.8 | 818.9 | 818.7 | 0.0% | 0.0% | 0.0% |
| 18 | `fc542144` | Modal-TPOT | ms | 8.0 | 8.0 | 8.2 | 0.0% | -1.6% | -1.6% |
| 19 | `fe66b347` | Modal-TPOT | ms | 82.2 | 71.3 | 74.6 | +13.2% | +9.3% | -4.5% |
| 20 | `9f1710f1` | Docker-Tput | tok/s | 60.3 | 60.2 | 59.6 | -0.1% | -1.1% | -1.0% |
| 21 | `2deb029d` | Docker-Tput | tok/s | 5671 | 5686 | 5711 | +0.3% | +0.7% | +0.4% |

### Invalid Comparisons (Model/Config Mismatch)

These **8 commits** show apparent 38-74% regressions due to **different benchmark models/configs**:

| Commit | Baseline Model | Human Model | B tput | H tput | Apparent |
|--------|---------------|-------------|--------|--------|----------|
| `d7740ea4` | Meta-Llama-3-8B (standalone) | Meta-Llama-3-8B (serving) | 7592 | 2011 | -73.5% |
| `6ce01f30` | Qwen2.5-7B-Instruct | Meta-Llama-3-8B | 5310 | 1791 | -66.3% |
| `3476ed08` | Qwen2.5-7B-Instruct | Meta-Llama-3-8B-Instruct | 5388 | 2128 | -60.5% |
| `80aa7e91` | Qwen2.5-7B-Instruct | Meta-Llama-3-8B-Instruct | 5355 | 2178 | -59.3% |
| `8bc68e19` | Qwen2.5-7B-Instruct | Meta-Llama-3-8B-Instruct | 3968 | 1979 | -50.1% |
| `7c01f706` | Qwen2.5-7B-Instruct | Meta-Llama-3-8B-Instruct | 4154 | 2229 | -46.3% |
| `3a243095` | Qwen2.5-7B-Instruct | Meta-Llama-3-8B-Instruct | 4312 | 2519 | -41.6% |
| `9badee53` | Llama-3.2-1B-Instruct | Meta-Llama-3-8B-Instruct | 5526 | 3424 | -38.0% |

**Root cause**: Baseline benchmarks ran with Qwen-7B or Llama-1B (faster models), while Human ran with Llama-8B (slower model). These are **NOT valid comparisons**.

### Agent vs Human Summary (Valid Comparisons Only)

From the 21 truly comparable commits (19 Modal + 2 Docker):

| Outcome | Count | Percentage | Commits |
|---------|-------|------------|---------|
| Agent **beats** Human (>5%) | 4 | 19% | 58eee5f2, 6a417b86, 6e36f4fa, 98f47f2a |
| Agent **matches** Human (±5%) | 15 | 71% | 299ebb62, 30172b49, 310aca88, 4c822298, 61b8cea3, 6d0734c5, 70b808fe, 8a4e5c5f, b55ed6ef, bc7c4d20, ed250545, f26c4aee, fc542144, **9f1710f1**, **2deb029d** |
| Agent **loses** to Human (>5%) | 2 | 10% | b690e348, fe66b347 |

*Note: Both Docker reruns (9f1710f1, 2deb029d) show Agent MATCH (<5% difference)*

### Critical Issues Summary

1. **Model mismatch in Separate files**: 8 commits used different models for baseline vs human, invalidating comparisons

2. **Benchmark mode mixing**: Some commits (e.g., d7740ea4) ran baseline in standalone mode but human in serving mode

3. **Metric type confusion**: Earlier analysis incorrectly compared throughput_tok_s to tpot_mean_ms

4. **Missing agent data**: Only 45 commits have agent metrics vs 61 with human metrics

5. **Docker verification only**: 47 commits have Docker throughput but this was for build verification, not 3-way comparison

---

## Comprehensive Commit Categorization (All 96 Commits)

### Complete Breakdown by Category

| Category | Count | % | Description | Usability |
|----------|-------|---|-------------|-----------|
| **Valid 3-way (Modal)** | 18 | 19% | Same metric, same source, same config | ✅ Full analysis |
| **Valid 3-way (Sep)** | 1 | 1% | Separate files with matching models | ✅ Full analysis |
| **Valid 3-way (Docker)** | 2 | 2% | Docker reruns with full B+H+A | ✅ Full analysis |
| **Invalid 3-way (Model mismatch)** | 8 | 8% | Baseline used different model | ⚠️ H vs A only |
| **Modal 2-way B+H** | 4 | 4% | Modal baseline + human, no agent | ⚠️ Human opt only |
| **Sep 2-way H+A** | 13 | 14% | Human + agent, no valid baseline | ⚠️ H vs A only |
| **Human only** | 5 | 5% | Only human metrics available | ❌ No comparison |
| **Agent only** | 5 | 5% | Only agent metrics available | ❌ No comparison |
| **Baseline only** | 3 | 3% | Only baseline metrics | ❌ No comparison |
| **Docker only** | 10 | 10% | Only Docker verification run | ❌ Not 3-way |
| **Complete failures** | 24 | 25% | No usable metrics at all | ❌ No data |
| **No perf_command** | 1 | 1% | Build/CI commit, not perf | ❌ N/A |
| **TOTAL** | **96** | 100% | | |

---

### Category Details

#### 1. Valid 3-Way Comparisons (21 commits) ✅

These are the **only commits with scientifically valid baseline→human→agent comparisons**.

**Modal TPOT (13 commits)** - serving mode, lower is better:
- 299ebb62, 30172b49, 58eee5f2, 6a417b86, 6d0734c5, 70b808fe, 8a4e5c5f, b55ed6ef, b690e348, bc7c4d20, ed250545, fc542144, fe66b347

**Modal Throughput (5 commits)** - standalone mode, higher is better:
- 310aca88, 4c822298, 61b8cea3, 98f47f2a, f26c4aee

**Separate Throughput (1 commit)** - verified same model:
- 6e36f4fa (Meta-Llama-3-8B-Instruct, 1615.8 → 2413.6 → 2784.2 tok/s)

**Docker Reruns (2 commits)** - full 3-way from local H100:
- 9f1710f1 (DeepSeek-V2-Lite-Chat, serving, Agent MATCH: -1.0%)
- 2deb029d (Llama-3-8B-FP8, prefix_caching, Agent MATCH: +0.4%)

---

#### 2. Invalid 3-Way (Model/Config Mismatch) (8 commits) ⚠️

**Problem**: Baseline file ran with Qwen-7B or Llama-1B, but human/agent ran with Llama-8B.

| Commit | Baseline Model | Human/Agent Model | Why Invalid |
|--------|---------------|-------------------|-------------|
| `d7740ea4` | Meta-Llama-3-8B (standalone) | Meta-Llama-3-8B (serving) | Mode mismatch |
| `6ce01f30` | Qwen2.5-7B-Instruct | Meta-Llama-3-8B | Model mismatch |
| `3476ed08` | Qwen2.5-7B-Instruct | Meta-Llama-3-8B-Instruct | Model mismatch |
| `80aa7e91` | Qwen2.5-7B-Instruct | Meta-Llama-3-8B-Instruct | Model mismatch |
| `8bc68e19` | Qwen2.5-7B-Instruct | Meta-Llama-3-8B-Instruct | Model mismatch |
| `7c01f706` | Qwen2.5-7B-Instruct | Meta-Llama-3-8B-Instruct | Model mismatch |
| `3a243095` | Qwen2.5-7B-Instruct | Meta-Llama-3-8B-Instruct | Model mismatch |
| `9badee53` | Llama-3.2-1B-Instruct | Meta-Llama-3-8B-Instruct | Model mismatch |

**Salvageable**: Human vs Agent comparison is still valid (same model). These 8 commits can contribute to Agent vs Human analysis.

---

#### 3. Modal 2-Way B+H Only (4 commits) ⚠️

Agent run failed or was skipped. Only useful for measuring human optimization effectiveness.

| Commit | Subject | B | H | H vs B |
|--------|---------|---|---|--------|
| `3b61cb45` | [V1] Further reduce CPU overheads | 9819.5 | 9822.4 | +0.0% |
| `6dd94dbe` | [perf] fix perf regression from #12253 | 204.7 | 255.9 | +25.0% |
| `8c1e77fb` | [Kernel] Update vllm-flash-attn version | 10117.1 | 10206.3 | +0.9% |
| `ce6bf3a2` | [torch.compile] avoid Dynamo guard | 54.7 | 55.5 | +1.4% |

**Note**: These confirm human optimizations work but provide no agent comparison.

---

#### 4. Separate Files 2-Way H+A Only (13 commits) ⚠️

Baseline file missing or failed. Can compare Human vs Agent directly.

| Commit | Subject | Human | Agent | A vs H |
|--------|---------|-------|-------|--------|
| `015069b0` | Optimize Qwen3_ReasoningParser | 198.3 | 198.3 | 0.0% |
| `19d98e0c` | (DeepSeek) | 2358.4 | 2340.7 | -0.8% |
| `22d33bac` | merge_async_iterators fast-path | 3946.1 | 3984.8 | +1.0% |
| `296f927f` | Mamba2 Prefill Tweaks | 1421.5 | 1411.8 | -0.7% |
| `89a84b0b` | Use array to speedup padding | 3558.5 | 2967.5 | -16.6% |
| `9474e89b` | PREFIX CACHING fixes | 3086.4 | 2852.5 | -7.6% |
| `99abb8b6` | Spec Decode Rejection Sampler | 3736.7 | 3716.5 | -0.5% |
| `ca7a2d5f` | Revert MLA CPU overheads | 2376.7 | 2353.8 | -1.0% |
| `cf2f084d` | Dynamic scheduler delay | 2443.1 | 2451.8 | +0.4% |
| `e206b543` | xgrammar shared context | 3105.1 | 3352.1 | +8.0% |
| `e3580537` | Enable chunked prefill | 2496.9 | 3107.0 | +24.4% |
| `fc7b8d1e` | e2e overheads reduction | 2214.0 | 2598.0 | +17.3% |

**Note**: Missing model may be Llama-8B (inferred from patterns).

---

#### 5. Single-Metric Categories (13 commits total) ❌

**Human only (5)**: 35fad35a, 660470e5, ad8d696a, ccf02fcb, e7b20426
- Agent runs crashed or produced no metrics
- Only shows human can optimize, not agent capability
- *Note: 2deb029d and 9f1710f1 moved to Valid 3-way via Docker reruns*

**Agent only (5)**: 67da5720, 6d646d08, 83450458, 93e5f3c5, 9d72daf4
- Human runs crashed or missing
- Shows agent can produce output, but no comparison

**Baseline only (3)**: 2a052011, 8d75fe48, eefbf4a6
- Both human and agent failed
- Confirms original benchmark works, optimization failed

---

#### 6. Docker-Only (12 commits) ❌

Only have Docker verification throughput (different model, usually opt-125m).

Commits: 0ec82edd, 21d93c14, 379da6dc, 526de822, b6d10354, c0569dbc, d55e446d, dcc6cfb9, e493e485, aea94362, b10e5198, 3092375e

**Not usable for 3-way comparison** - Docker used different model and was just build verification.

---

#### 7. Complete Failures (24 commits) ❌

No usable metrics from any source.

**Error types**:
- `version_bug` (6): 25ebed2f, 88693683, 9323a315, b2e0ad3b, f092153f
- `error` (9): 0d243f2a, 2f192835, 9a3b8832, ac45c44d, baeded25, bfdb1ba5, c45f3c3a, d4bc1a4d, e7523c2e, ec3b5ce9
- `baseline_failed` (6): 22dd9c27, 4fb56914, 8aa1485f, bd6028d6, dae68969, fb0acb6c
- `exception` (2): 7661e92e

**Root causes**:
- vLLM version incompatibilities
- Server startup crashes
- Model download failures
- GPU/hardware issues
- Timeout during benchmark

---

### Expanded Agent vs Human Analysis (All Usable Data)

Including invalid 3-way (H vs A still valid) and Sep 2-way H+A:

| Source | Valid H vs A Comparisons | Notes |
|--------|--------------------------|-------|
| Valid 3-way (Modal) | 19 | Gold standard |
| Valid 3-way (Docker) | 2 | Docker reruns with full 3-way |
| Invalid 3-way (model mismatch) | 8 | H vs A same model |
| Sep 2-way H+A | 12 | No baseline but H/A comparable |
| **TOTAL** | **41** | |

#### Aggregated Results (n=41)

| Outcome | Count | Percentage |
|---------|-------|------------|
| Agent **beats** Human (>5%) | 7 | 17% |
| Agent **matches** Human (±5%) | 28 | 68% |
| Agent **loses** to Human (>5%) | 6 | 15% |

*Note: 2 Docker reruns (9f1710f1, 2deb029d) both show Agent MATCH*

**Agent wins (>5% better):**
- `58eee5f2`: +8.7% (Modal TPOT, tokenizer decode)
- `6a417b86`: +7.3% (Modal TPOT, neuron fix)
- `6e36f4fa`: +15.4% (Sep throughput, chunked prefill)
- `98f47f2a`: +5.3% (Modal throughput, FlashAttention CPU)
- `e206b543`: +8.0% (Sep throughput, xgrammar context)
- `e3580537`: +24.4% (Sep throughput, chunked prefill enable)
- `fc7b8d1e`: +17.3% (Sep throughput, e2e overheads)

**Agent losses (>5% worse):**
- `b690e348`: -21.8% (Modal TPOT, Mamba2 SSM)
- `fe66b347`: -4.5% (Modal TPOT, Mamba2 prefill)
- `89a84b0b`: -16.6% (Sep throughput, array padding)
- `9474e89b`: -7.6% (Sep throughput, prefix caching)
- `3a243095`: -6.0% (Invalid 3-way, _get_ranks)
- `7c01f706`: -5.4% (Invalid 3-way, is_finished)

---

### Data Quality Summary

| What We Have | Count | Usable For |
|--------------|-------|------------|
| Full 3-way comparison | 21 | Complete analysis (19 Modal + 2 Docker) |
| Agent vs Human only | 20 | A vs H comparison |
| Human optimization only | 9 | Human effectiveness |
| No comparison possible | 46 | Nothing |
| **TOTAL** | **96** | |

**Bottom Line**:
- **22% of commits** (21/96) have scientifically valid 3-way comparisons
- **43% of commits** (41/96) can contribute to Agent vs Human analysis
- **48% of commits** (46/96) have insufficient data for any comparison

---

### Why This Matters

1. **The benchmark is small but viable**: 21 valid 3-way data points (19 Modal + 2 Docker reruns) provides a reasonable sample for initial analysis. Adding Docker reruns demonstrates the benchmark can be expanded.

2. **Infrastructure dominated failures**: 24 complete failures + 13 partial failures = 39% failure rate. The benchmarking infrastructure itself is a major source of noise.

3. **Model consistency issues**: The separate file pipeline allowed different models for baseline vs human runs, creating 8 commits with misleading "regression" numbers.

4. **Agent capability signal is clear**: In the expanded n=41 dataset:
   - Agent matches or beats human 85% of the time
   - Agent achieves >5% improvement over human 17% of the time
   - Docker reruns both showed Agent MATCH, confirming stability

5. **Recommendation**: Publications should use the 21 valid 3-way commits as the primary dataset, clearly state n=21, and note that additional Docker reruns can recover failed Modal results

---

## Agent Patch Failures (Critical Addition)

### The Insight

If we have **Baseline + Human metrics** but **no Agent metrics**, this is NOT "missing data" - it's a **valid benchmark outcome** where:

- ✅ Human optimization: **MEASURED** (we can calculate improvement)
- ❌ Agent optimization: **FAILED** (did not produce working code)

The cause doesn't matter (patch generation failed, patch didn't apply, code crashed, run skipped). The **outcome** is the same: Agent did not successfully optimize this commit.

### Agent Failure Commits (11 valid)

#### From Modal Pipeline (B+H only) - 6 commits

| Commit | Baseline | Human | H vs B | Subject | Agent Status |
|--------|----------|-------|--------|---------|--------------|
| `6dd94dbe` | 204.7 | 255.9 | +25.0% | [perf] fix perf regression from #12253 | **FAILED** |
| `9ed82e70` | 1912.8 | 2116.8 | +10.7% | [Misc] Small perf improvements | **FAILED** |
| `ce6bf3a2` | 54.7 | 55.5 | +1.4% | [torch.compile] avoid Dynamo guard evaluation | **FAILED** |
| `8c1e77fb` | 10117.1 | 10206.3 | +0.9% | [Kernel] Update vllm-flash-attn version | **FAILED** |
| `3b61cb45` | 9819.5 | 9822.4 | +0.0% | [V1] Further reduce CPU overheads in flash-attn | **FAILED** |
| `d7740ea4` | 7592.4 | 2011.3 | N/A* | [Core] Optimize sampler get_logprobs | **FAILED** |

*`d7740ea4`: Baseline used standalone throughput, Human used serving TPOT - mode mismatch makes H vs B invalid, but agent still FAILED to produce any output.

#### From Separate Pipeline (H-only with agent error) - 5 commits

These were previously miscategorized as "HUMAN_ONLY" (non-evaluable), but are actually Agent Failures:

| Commit | Human (tok/s) | Agent Error | Subject |
|--------|---------------|-------------|---------|
| `35fad35a` | 3172.74 | **Server crashed after applying patch** | [V1][Sampler] Faster top-k only implementation |
| `ad8d696a` | 2382.51 | **Server crashed after applying patch** | [Core] Scheduler perf fix (#4270) |
| `660470e5` | 2250.31 | **No metrics in agent output** | [Core] Optimize evictor-v2 performance |
| `ccf02fcb` | 1152.28 | **No metrics in agent output** | Revert "[Model] Mamba2 Prefill Performance" |
| `e7b20426` | 2774.95 | **No metrics in agent output** | Revert "[Performance] Performance improvements" |

**Why these count as Agent Failures:**
- Human benchmark succeeded with valid throughput metrics
- Agent patch was generated and applied
- Agent patch either crashed the server (2 commits) or produced no valid metrics (3 commits)
- This is a benchmark outcome: **Agent failed to produce working optimization**

**Still unclear (2 commits with missing agent files):**
- `2deb029d` (Human: 3094.76 tok/s) - Agent file completely missing
- `9f1710f1` (Human: 2408.03 tok/s) - Agent file completely missing
- Cannot determine if agent was never run or failed catastrophically

### Impact on Statistics

**Narrow view (19 commits, 3-way only):**

| Outcome | Count | Percentage |
|---------|-------|------------|
| Agent wins (>5% better) | 4 | 21% |
| Agent matches (±5%) | 13 | 68% |
| Agent loses (>5% worse) | 2 | 11% |
| **Total** | 19 | 100% |

**Expanded view (49 commits, all evaluable A vs H):**

| Outcome | Count | Percentage |
|---------|-------|------------|
| Agent wins (>5% better) | 7 | **14%** |
| Agent matches (±5%) | 25 | **51%** |
| Agent loses (>5% worse) | 6 | **12%** |
| Agent **FAILED** (crashed/no output) | 11 | **22%** |
| **Total** | 49 | 100% |

**Breakdown of 49 evaluable commits:**
- 19 valid 3-way (Modal) - full B+H+A analysis
- 12 H+A only (Separate) - direct A vs H comparison
- 7 model mismatch (Separate) - A vs H valid, baseline invalid
- 6 agent failures (Modal B+H only) - agent produced no output
- 5 agent failures (Separate H-only) - agent crashed or produced no metrics

### Key Takeaway

**Agent failure rate: 22%** (11/49 evaluable commits)

Including ALL commits where we can evaluate Agent vs Human performance:

1. **Agent matches or beats human 65% of the time** (32/49 commits)
2. **Agent underperforms or fails 35% of the time** (17/49 commits)
3. **The sample size is now meaningful** - n=49 is better than n=19

**⚠️ The failure rate increased significantly** from 14% to 22% after properly categorizing the 5 Separate pipeline commits where agent patches crashed or produced no metrics. These were previously hidden in the "HUMAN_ONLY" category

---

## Retry Recommendations (All 96 Commits Categorized)

### Complete Commit Disposition

| Category | Count | Action | Reason |
|----------|-------|--------|--------|
| **Valid 3-way** | 19 | ✅ DONE | Already have B+H+A |
| **H+A only (evaluable)** | 12 | ✅ DONE | Valid A vs H comparison |
| **Model mismatch (A vs H valid)** | 7 | ✅ DONE | A vs H same model, baseline invalid |
| **Agent failures (Modal B+H)** | 6 | ✅ DONE | Valid outcome: agent FAILED |
| **Agent failures (Separate H-only)** | 5 | ✅ DONE | Valid outcome: agent crashed/no output |
| **Total already evaluable** | **49** | ✅ | **No retry needed** |
| **Maybe retry (for baseline)** | 3 | 🔶 MAYBE | Would upgrade H+A to 3-way |
| **DO NOT retry** | 23 | ❌ NO | Fundamentally broken |
| **Single-metric only** | 9 | ❌ NO | Need 2+ data points to recover |
| **Docker-only** | 12 | ❌ NO | Wrong model (opt-125m) |
| **TOTAL** | **96** | | |

**Critical insight:** The 12 H+A commits and 7 model-mismatch commits ARE valid benchmark outcomes. They provide direct Agent vs Human comparisons. Retrying for baseline would upgrade them to "gold standard" 3-way but does NOT change whether they are evaluable.

---

### ❌ DO NOT RETRY - Fundamentally Broken (23 commits)

#### version_bug (5 commits) - vLLM API Incompatibility
```
25ebed2f  [V1][Minor] Cache np arange to reduce input preparation
88693683  [Performance][Core] Optimize the performance of evictor
9323a315  [Core][Performance] Add XGrammar support for guided
b2e0ad3b  [Perf] Reduce peak memory usage of llama
f092153f  [V1] Use more persistent buffers to optimize input
```
**Why:** These commits have vLLM version incompatibilities. The benchmark infrastructure cannot run on these commits without code changes. Retry = same failure.

#### ROCm/AMD Specific (2 commits) - Wrong Hardware
```
0d243f2a  [ROCm][MoE] mi300 mixtral8x7B perf for specific BS
526de822  [Kernel][Triton][AMD] Use block size heuristic for
```
**Why:** Requires AMD MI300 GPU. You have NVIDIA H100s. Impossible to benchmark.

#### Multi-GPU Models (3 commits) - Need 2+ H100s
```
21d93c14  Optimize Mixtral with expert parallelism - Mixtral-8x7B (~100GB VRAM)
379da6dc  [Kernel] [FP8] Improve FP8 linear layer - Meta-Llama-3-70B (~140GB)
b6d10354  [Kernel] Layernorm performance optimization - Llama-2-70b (~140GB)
```
**Why:** These models require 2× H100-80GB ($8/hr). High cost, low marginal value.

#### Very Old Commits (4 commits) - API Incompatible
```
c45f3c3a  Optimize tensor parallel execution speed (#17)
2f192835  [Core] latency optimization (#3890)
ad8d696a  [Core] Scheduler perf fix (#4270)
d4bc1a4d  Add unoptimized OPT Attention
```
**Why:** vLLM API changed drastically. These commits predate modern benchmark infrastructure.

#### Complete Infrastructure Failures (9 commits)
```
4fb56914  [perf] Add fused MLA QKV + strided layernorm - Baseline server failed to start
ac45c44d  [Bugfix] [Performance] DeepEPHighThroughput - Baseline server failed to start
baeded25  [Attention] Deepseek v3 MLA support - Baseline server failed to start
7661e92e  [Model] Optimize nemotron_h implementation - Exception
9a3b8832  [PERF] Speedup of MRoPE prepare inputs - Error
bfdb1ba5  [Core] Improve detokenization performance - Error
e7523c2e  [V1][Sampler] Improve performance of FlashInfer - Error
ec3b5ce9  Improve detokenization performance - Error
dae68969  [Perf] Reduce MLA CPU overheads in V1 - baseline_failed
```
**Why:** Server failed to start on BOTH Modal AND Separate pipelines. Fundamental code issue at these commits.

---

### ✅ ALREADY EVALUABLE - H+A Data (19 commits)

These commits ARE valid benchmark outcomes contributing to n=44. Retrying would only add baseline data.

#### H+A Only - Direct Comparison (12 commits)

| Commit | Subject | A vs H | Outcome |
|--------|---------|--------|---------|
| `e3580537` | Enable chunked prefill | **+24.4%** | Agent WIN |
| `fc7b8d1e` | e2e overheads reduction | **+17.3%** | Agent WIN |
| `e206b543` | xgrammar shared context | **+8.0%** | Agent WIN |
| `22d33bac` | merge_async_iterators | +1.0% | Agent MATCH |
| `cf2f084d` | Dynamic scheduler delay | +0.4% | Agent MATCH |
| `015069b0` | Optimize Qwen3_ReasoningParser | 0.0% | Agent MATCH |
| `99abb8b6` | Spec Decode Rejection Sampler | -0.5% | Agent MATCH |
| `296f927f` | Mamba2 Prefill Tweaks | -0.7% | Agent MATCH |
| `19d98e0c` | (DeepSeek) | -0.8% | Agent MATCH |
| `ca7a2d5f` | Revert MLA CPU overheads | -1.0% | Agent MATCH |
| `9474e89b` | PREFIX CACHING fixes | **-7.6%** | Agent LOSS |
| `89a84b0b` | Array speedup padding | **-16.6%** | Agent LOSS |

#### Model Mismatch - A vs H Still Valid (7 commits)

| Commit | Subject | A vs H | Outcome | Baseline Issue |
|--------|---------|--------|---------|----------------|
| `9badee53` | Fix generation-config | -0.2% | Agent MATCH | Used Llama-1B |
| `80aa7e91` | Optimize CPU backend | -0.5% | Agent MATCH | Used Qwen-7B |
| `6ce01f30` | Optimize get_seqs | -0.8% | Agent MATCH | Used Qwen-7B |
| `8bc68e19` | Auto-detect vLLM-tensorized | -1.2% | Agent MATCH | Used Qwen-7B |
| `3476ed08` | Optimize block_manager_v2 | -1.6% | Agent MATCH | Used Qwen-7B |
| `7c01f706` | SequenceStatus.is_finished | **-5.4%** | Agent LOSS | Used Qwen-7B |
| `3a243095` | Optimize _get_ranks | **-6.0%** | Agent LOSS | Used Qwen-7B |

**Why these count:** Human and Agent used the SAME model. The A vs H comparison is valid regardless of baseline issues.

---

### 🔶 OPTIONAL - Retry for Baseline (upgrades H+A to 3-way)

Getting baseline would upgrade these from "evaluable" to "gold standard 3-way":

| Commit | Subject | A vs H | Value of Baseline |
|--------|---------|--------|-------------------|
| `e3580537` | Enable chunked prefill | +24.4% | Validate human didn't regress |
| `fc7b8d1e` | e2e overheads | +17.3% | Validate human didn't regress |
| `e206b543` | xgrammar shared context | +8.0% | Confirm optimization worked |

**Cost:** ~$6 for 3 baseline runs
**Benefit:** Upgrades 3 commits from "A vs H" to "full 3-way" - nice to have but NOT required for evaluation

#### Need Agent Run (2 commits) - Would Add to Evaluable Count

| Commit | Subject | H vs B | Model | GPU |
|--------|---------|--------|-------|-----|
| `2deb029d` | BlockManagerV2 prefix cache | N/A | neuralmagic/FP8-Llama-8B | 1×H100 |
| `35fad35a` | Faster top-k sampler | N/A | meta-llama/Meta-Llama-3-8B | 1×H100 |

**Expected success rate:** 60%
**Cost:** ~$3 (45 min)
**Impact:** +1-2 evaluable commits

---

### 🔶 MAYBE RETRY - Timeout Issues (3 commits)

| Commit | Model | Issue | Recovery Chance |
|--------|-------|-------|-----------------|
| `0ec82edd` | Qwen/Qwen3-30B-A3B | Timeout 3600s | 20% |
| `22dd9c27` | meta-llama/Llama-3-8B | Timeout + crash | 10% |
| `c0569dbc` | Qwen/Qwen3-30B-A3B-FP8 | No metrics | 30% |

**Why maybe:** 30B MoE models are slow. Could work with 7200s timeout but likely still too slow.

---

### ❌ Single-Metric Only (13 commits)

**Human only (5):** `35fad35a`, `660470e5`, `ad8d696a`, `ccf02fcb`, `e7b20426`
*Note: 2deb029d and 9f1710f1 moved to Valid 3-way via Docker reruns*

**Agent only (5):** `67da5720`, `6d646d08`, `83450458`, `93e5f3c5`, `9d72daf4`

**Baseline only (3):** `2a052011`, `8d75fe48`, `eefbf4a6`

**Why skip:** Single data point provides no comparison. Would need to recover 2 other versions to be useful.

---

### ❌ Docker-Only (10 commits)

```
0ec82edd, 21d93c14, 379da6dc, 526de822, b6d10354, c0569dbc,
d55e446d, dcc6cfb9, e493e485, aea94362, b10e5198, 3092375e
```

*Note: 2deb029d and 9f1710f1 upgraded to Valid 3-way via Docker reruns with proper benchmark commands*

**Why skip the remaining 10:** Docker used facebook/opt-125m (125M params) while PRs targeted 7B-70B models. Completely different performance regime. Not valid for comparison.

---

### Cost-Benefit Summary

**Current Status:** 51 evaluable commits for Agent vs Human analysis (Updated with Docker reruns)

| Action | Commits | Cost | Impact | Notes |
|--------|---------|------|--------|-------|
| **Already done** | 51 | $0 | n=51 evaluable | 21 valid 3-way + 12 H+A + 7 model-mismatch + 11 failures |
| ✅ **Docker reruns completed** | 2 | ~$0 | +2 valid 3-way | 9f1710f1, 2deb029d now full 3-way |
| Baseline reruns | 3 | ~$6 | Upgrades H+A → 3-way | Does NOT add to evaluable count |
| Timeout retries | 3 | ~$10 | ~20% success | Low ROI |
| Multi-GPU | 3 | ~$50+ | +1-2 evaluable | High cost |
| Everything else | 45+ | ~$200+ | ~0 evaluable | Infrastructure failures |

**Recommendation:** The benchmark is now complete with n=51 evaluable commits (21 valid 3-way). Docker reruns are an effective way to recover failed Modal results at minimal cost.

---

## 📋 Non-Evaluable Commits: Detailed Breakdown (45 commits) - ⚠️ REVISED

This section documents **every commit** that could not be used for Agent vs Human evaluation, with specific failure reasons.

**⚠️ Update (2026-01-11):** 5 commits previously in HUMAN_ONLY were reclassified as Agent Failures (evaluable). Total non-evaluable reduced from 52 to 47.

**⚠️ Update (2026-01-12):** 2 commits (9f1710f1, 2deb029d) moved to Valid 3-Way via Docker reruns. Total non-evaluable reduced from 47 to 45.

### Summary by Failure Category

| Category | Count | Description | Recoverable? |
|----------|-------|-------------|--------------|
| **INFRASTRUCTURE** | 12 | Server crashes, exceptions, generic errors | ❌ No |
| **DOCKER_ONLY** | 9 | Only Docker verification worked (wrong model) | ❌ No |
| **HUMAN_ONLY** | 1 | Human worked, agent status unclear | 🔶 Maybe |
| **VERSION_BUG** | 5 | vLLM API incompatible at this commit | ❌ No |
| **BASELINE_FAILED** | 4 | Modal baseline server failed to start | ❌ No |
| **AGENT_ONLY** | 3 | Agent worked but human crashed/failed | 🔶 Maybe |
| **MULTI_GPU** | 3 | Model requires 2+ H100 GPUs (70B+ params) | 💰 Expensive |
| **EDGE_CASE** | 2 | Has metrics but excluded (TPOT=0 or latency mode) | ✅ Review |
| **WRONG_HARDWARE** | 2 | Requires AMD MI300 GPU | ❌ No |
| **NO_BENCHMARK** | 1 | No performance command in dataset (CI commit) | ❌ N/A |
| **SERVER_CRASH** | 1 | Server crashed on startup for all versions | ❌ No |
| ✅ **RECOVERED** | 2 | Via Docker reruns (9f1710f1, 2deb029d) | ✅ Done |
| **TOTAL** | **45** | | |

**Reclassified to Evaluable (Agent Failures):** `35fad35a`, `ad8d696a`, `660470e5`, `ccf02fcb`, `e7b20426`
**Recovered via Docker reruns (Valid 3-Way):** `9f1710f1`, `2deb029d`

---

### 1. INFRASTRUCTURE Failures (12 commits)

Server failed to start, threw exceptions, or produced no output across all pipelines.

| Commit | Subject | Failure Detail |
|--------|---------|----------------|
| `4fb56914` | [perf] Add fused MLA QKV + strided layernorm | Baseline server failed to start |
| `67da5720` | [PERF] Speed up Qwen2.5-VL model by speed up rotary | Modal exception |
| `6d646d08` | [Core] Optimize Async + Multi-step (#8050) | Modal error |
| `7661e92e` | [Model] Optimize nemotron_h implementation | Modal exception |
| `9a3b8832` | [PERF] Speedup of MRoPE prepare inputs | Modal error |
| `ac45c44d` | [Bugfix] [Performance] DeepEPHighThroughput | Baseline server failed to start |
| `baeded25` | [Attention] Deepseek v3 MLA support with FP8 | Baseline server failed to start |
| `bfdb1ba5` | [Core] Improve detokenization performance | Modal error |
| `c45f3c3a` | Optimize tensor parallel execution speed (#17) | Modal error (very old commit) |
| `d4bc1a4d` | Add unoptimized OPT Attention | Modal error (very old commit) |
| `e7523c2e` | [V1][Sampler] Improve performance of FlashInfer | Modal error |
| `ec3b5ce9` | Improve detokenization performance (#1338) | Modal error (very old commit) |

**Root causes**: vLLM code at these commits has bugs, missing dependencies, or API incompatibilities that prevent the server from starting.

---

### 2. DOCKER_ONLY (9 commits)

Docker verification succeeded with `facebook/opt-125m`, but actual benchmarks failed. **Not usable** because opt-125m (125M params) has completely different performance characteristics than the target models (7B-8B params).

*Note: 2deb029d and 9f1710f1 were recovered via Docker reruns with proper benchmark commands (see Appendix)*

| Commit | Subject | Docker Model | Why Unusable |
|--------|---------|--------------|--------------|
| `0ec82edd` | [perf] Speed up align sum kernels | facebook/opt-125m | Wrong model |
| `22dd9c27` | [Kernel] Optimize Prefill Attention | facebook/opt-125m | Wrong model |
| `3092375e` | [V1][Performance] Implement custom serialization | facebook/opt-125m | Wrong model |
| `8d75fe48` | [Kernel] Switch fp8 layers to CUTLASS | opt-125m | Wrong model |
| `aea94362` | [Frontend][V1] Online serving performance | opt-125m | Wrong model |
| `b10e5198` | [V1][Minor] Optimize get_cached_block | opt-125m | Wrong model |
| `c0569dbc` | [Misc] ModularKernel: WeightAndReduce | facebook/opt-125m | Wrong model |
| `d55e446d` | [V1][Spec Decode] Small refactors for eagle | facebook/opt-125m | Wrong model |
| `dcc6cfb9` | [Kernel][Performance] Tweak MoE Batched silu_mul | facebook/opt-125m | Wrong model |
| `e493e485` | [V0][Bugfix] Fix parallel sampling perf regression | facebook/opt-125m | Wrong model |
| `eefbf4a6` | [Perf] Optimize reshape_and_cache_flash CUDA | opt-125m | Wrong model |

**Why Docker used opt-125m**: Docker runs were for build verification only, not performance measurement. The benchmark config wasn't propagated to Docker runs.

---

### 3. HUMAN_ONLY (1 commit) - ⚠️ REVISED

**5 commits reclassified as Agent Failures** (see "Agent Patch Failures" section above). These 5 have valid human metrics AND explicit agent errors - they are evaluable benchmark outcomes where the agent failed.

**2 commits recovered via Docker reruns** (see Appendix). These now have full 3-way benchmark data.

Only 1 commit remains truly non-evaluable:

| Commit | Subject | Human Result | Why Non-Evaluable |
|--------|---------|--------------|-------------------|
| `2a052011` | [Kernel] Support MoE Fp8 Checkpoints for Mixtral | ⚠️ No metrics (throughput=null) | Human also failed - no valid reference |

**Reclassified to Agent Failures (now evaluable):**
- `35fad35a`: Human 3172.7 tok/s → Agent crashed
- `ad8d696a`: Human 2382.5 tok/s → Agent crashed
- `660470e5`: Human 2250.3 tok/s → Agent no output
- `ccf02fcb`: Human 1152.3 tok/s → Agent no output
- `e7b20426`: Human 2774.9 tok/s → Agent no output

**Recovered via Docker reruns (now Valid 3-Way):**
- `9f1710f1`: Full 3-way serving benchmark (Agent MATCH: -1.0%)
- `2deb029d`: Full 3-way prefix_caching benchmark (Agent MATCH: +0.4%)

---

### 4. VERSION_BUG (5 commits)

vLLM internal APIs changed between when the commit was made and when benchmarks ran. The benchmark harness itself fails, not the optimization.

| Commit | Subject | Error Type |
|--------|---------|------------|
| `25ebed2f` | [V1][Minor] Cache np arange to reduce input prep | API incompatibility |
| `88693683` | [Performance][Core] Optimize evictor performance | API incompatibility |
| `9323a315` | [Core][Performance] Add XGrammar support | API incompatibility |
| `b2e0ad3b` | [Perf] Reduce peak memory usage of llama | API incompatibility |
| `f092153f` | [V1] Use more persistent buffers | API incompatibility |

**Root cause**: These commits rely on internal vLLM APIs that were refactored in later versions. The benchmark harness runs a newer vLLM version that's incompatible.

---

### 5. BASELINE_FAILED (4 commits)

Modal pipeline specifically reported baseline server failed to start.

| Commit | Subject | Error |
|--------|---------|-------|
| `8aa1485f` | [Perf] Disable chunked local attention by default | Baseline server failed |
| `bd6028d6` | Optimized topk for topk=1 (Llama-4) | Baseline server failed |
| `dae68969` | [Perf] Reduce MLA CPU overheads in V1 | Baseline server failed |
| `fb0acb6c` | [Perf] Improve MLA on V1 (#14540) | Baseline server failed |

**Root cause**: The parent commit (baseline) has bugs that prevent server startup. Can't measure improvement if baseline doesn't work.

---

### 6. AGENT_ONLY (3 commits)

Agent optimization produced metrics, but human benchmark failed.

| Commit | Subject | Agent Result | Human Failure Reason |
|--------|---------|--------------|----------------------|
| `83450458` | [Performance][Spec Decode] Optimize ngram lookup | 3314.1 tok/s | Benchmark timed out after 600s |
| `93e5f3c5` | [Perf] Optimize Preparing Inputs for GPU | 3706.7 tok/s | Server crashed during startup |
| `9d72daf4` | [V1][Perf] Simpler request output queues | 3673.8 tok/s | Server crashed during startup |

**Note**: These are unusual - human (ground truth) failed but agent succeeded. Could indicate agent found a different/better approach, or human patch had issues at this commit.

---

### 7. MULTI_GPU (3 commits)

Models require more than 80GB VRAM (need 2+ H100 GPUs).

| Commit | Subject | Model | Estimated VRAM |
|--------|---------|-------|----------------|
| `21d93c14` | Optimize Mixtral with expert parallelism | mistralai/Mixtral-8x7B-v0.1 | ~100GB |
| `379da6dc` | [Kernel] [FP8] Improve FP8 linear layer | meta-llama/Meta-Llama-3-70B | ~140GB |
| `b6d10354` | [Kernel] Layernorm performance optimization | meta-llama/Llama-2-70b-hf | ~140GB |

**Recovery cost**: ~$8/hour for 2×H100-80GB instances. Low priority given marginal value.

---

### 8. VALID EDGE CASES (2 commits) - NOW RESOLVED

**Update (2026-01-12):** After reviewing the original PRs, these ARE valid benchmark results. The unusual metrics are **intentional by design**, not data quality issues.

| Commit | Subject | Metric | Result | Outcome |
|--------|---------|--------|--------|---------|
| `a3223766` | [Core] Optimize update checks in LogitsProcessor | TTFT | Agent **+8.2%** better | Agent WIN |
| `fa63e710` | [V1][Perf] Reduce scheduling overhead | latency_avg | Agent **-0.5%** | Agent MATCH |

#### `a3223766` - LogitsProcessor CPU Overhead (TTFT benchmark)

**Why TPOT/ITL = 0.0 is EXPECTED:**
- The PR author ([PR #21245](https://github.com/vllm-project/vllm/pull/21245)) **intentionally** used `--random-output-len 1`
- Purpose: Stress-test LogitsProcessor CPU batching overhead, not token generation
- With only 1 output token, TPOT (Time Per Output Token) and ITL (Inter-Token Latency) are mathematically undefined
- **TTFT is the correct metric** for this optimization

**Results (TTFT ms):**
| Version | TTFT Mean | TTFT Median | TTFT P99 |
|---------|-----------|-------------|----------|
| Baseline | 35.75 | 32.32 | 66.42 |
| Human | 33.52 | 29.98 | 64.89 |
| **Agent** | **30.78** | **25.52** | **63.29** |

- Human improvement: 6.2%
- **Agent improvement: 13.9%** (Agent beats human by 7.7 percentage points)

**Verdict:** Agent WIN - valid 3-way comparison using TTFT metric

#### `fa63e710` - Model Runner Scheduling (Standalone Latency benchmark)

**Why latency_avg instead of TTFT/TPOT:**
- This is a **standalone latency benchmark** (`benchmark_latency.py`), not a serving benchmark
- Measures end-to-end batch latency, not per-token metrics
- `latency_avg` is the correct metric for standalone mode

**Results (latency_avg ms):**
| Version | Latency Avg |
|---------|-------------|
| Baseline | 1331.71 |
| Human | 1323.82 |
| Agent | 1329.92 |

- Human improvement: 0.59%
- Agent improvement: 0.13%
- Agent vs Human: -0.46%

**Verdict:** Agent MATCH - valid 3-way comparison using latency_avg metric

---

### 9. WRONG_HARDWARE (2 commits)

Optimizations target AMD MI300 GPUs. Cannot benchmark on NVIDIA H100.

| Commit | Subject | Required Hardware |
|--------|---------|-------------------|
| `0d243f2a` | [ROCm][MoE] mi300 mixtral8x7B perf for specific BS | AMD MI300 |
| `526de822` | [Kernel][Triton][AMD] Use block size heuristic | AMD MI300 |

**Recovery**: Would need access to AMD MI300 cluster. Out of scope.

---

### 10. NO_BENCHMARK (1 commit)

Not a performance optimization commit.

| Commit | Subject | Reason |
|--------|---------|--------|
| `3127e975` | [CI/Build] Make pre-commit faster (#12212) | CI/build commit, no perf_command in dataset |

**Expected**: This commit shouldn't have been in the benchmark set.

---

### 11. SERVER_CRASH (1 commit)

Server crashed during startup for all versions (baseline, human, agent).

| Commit | Subject | Error |
|--------|---------|-------|
| `2f192835` | [Core] latency optimization (#3890) | Server crashed during startup |

**Root cause**: Very old commit (#3890) with incompatible code.

---

### Recovery Potential Summary

| Can Recover? | Commits | Action |
|--------------|---------|--------|
| ✅ **RESOLVED (edge cases)** | 2 | `a3223766`, `fa63e710` - **NOW VALID** (see section 8) |
| ✅ **RECOVERED (Docker reruns)** | 2 | `9f1710f1`, `2deb029d` - **NOW VALID 3-WAY** |
| 🔶 **Maybe (reruns)** | 9 | 6 human-only + 3 agent-only - could retry |
| 💰 **Expensive** | 3 | Multi-GPU commits - need 2×H100 |
| ❌ **No** | 34 | Infrastructure, version bugs, wrong hardware |

**Update (2026-01-12)**: The 2 "edge cases" are now confirmed as valid benchmarks:
- `a3223766`: Agent WIN (+8.2% on TTFT) - TPOT=0 is expected (output-len=1 was intentional)
- `fa63e710`: Agent MATCH (-0.5% on latency_avg) - standalone benchmark mode

**Update (2026-01-12)**: Docker reruns recovered 2 commits:
- `9f1710f1`: Agent MATCH (-1.0% on serving throughput)
- `2deb029d`: Agent MATCH (+0.4% on prefix_caching throughput)

**Bottom line**: Total evaluable is now **51 commits** (21 valid 3-way + 12 H+A + 7 model-mismatch + 11 agent failures).

---

## Appendix: Docker Reruns (2026-01-12)

Two commits were rerun locally using Docker with corrected benchmark commands.

### 9f1710f1 - MLA Prefill Context Performance (Full 3-Way Complete)

**Original Issue**: Wrong CLI args (`--input-len` instead of `--random-input-len`)

**Corrected Command**:
```bash
python benchmarks/benchmark_serving.py --model deepseek-ai/DeepSeek-V2-Lite-Chat \
    --random-input-len 8192 --random-output-len 64 --dataset-name random \
    --num-prompts 20 --request-rate 1
```

**Docker Images**:
- Baseline: `shikhar481/vllm_fixed_human_images:baseline-e642ec962cf2`
- Human: `ayushnangia16/nvidia-vllm-docker:9f1710f1ace3535920c0bb6d4cc329c36289080e`
- Agent: Baseline image + patch applied in-place (no rebuild needed)

**Agent Patch**: `perf-agents-bench/state/runs/vllm/claude_code/default/2025-12-22_21-40-38/vllm_core-0056/model_patch.diff`

**Results** (serving metrics, full 3-way comparison):

| Version | TTFT Mean (ms) | TPOT Mean (ms) | ITL Mean (ms) | Output Throughput |
|---------|----------------|----------------|---------------|-------------------|
| Baseline | 382.82 | 35.78 | 35.78 | 60.29 tok/s |
| Human | 387.43 | 36.41 | 36.41 | 60.21 tok/s |
| **Agent** | **385.73** | **39.53** | **39.53** | **59.60 tok/s** |

**Comparison**:

| Comparison | TTFT | TPOT | Throughput |
|------------|------|------|------------|
| Human vs Baseline | +1.2% (worse) | +1.8% (worse) | -0.1% |
| Agent vs Baseline | +0.8% (worse) | +10.5% (worse) | -1.1% |
| **Agent vs Human** | **-0.4% (better)** | **+8.6% (worse)** | **-1.0%** |

**Analysis**:
- All three versions perform similarly (within ~10%)
- PR #13897 fixed a regression in MLA prefill - baseline ≈ human is expected
- Agent TTFT is slightly better than human (-0.4%)
- Agent TPOT is worse than human (+8.6%)
- Overall: Agent **MATCH** (within acceptable variance for this benchmark type)

**Agent Benchmark Method**: Applied Claude Code patch in-place to baseline Docker image's installed vLLM (no rebuild required for pure Python changes).

**Note**: Metrics marked with `*` in tables indicate Docker rerun data (different from original Modal pipeline).

---

### 2deb029d - BlockManagerV2 Prefix Caching (Full 3-Way Complete)

**Original Issue**: Brackets left in command (`[--use-v2-block-manager]` invalid syntax)

**Corrected Command**:
```bash
python3 benchmarks/benchmark_prefix_caching.py \
    --model neuralmagic/Meta-Llama-3-8B-Instruct-FP8 \
    --output-len 200 --enable-prefix-caching --use-v2-block-manager
```

**Docker Images**:
- Baseline: `shikhar481/vllm_fixed_human_images:baseline-029c71de11bc`
- Human: `ayushnangia16/nvidia-vllm-docker:2deb029d115dadd012ce5ea70487a207cb025493`
- Agent: Baseline image + patch applied in-place

**Agent Patch**: `perf-agents-bench/state/runs/vllm/claude_code/default/2025-12-22_21-40-38/vllm_core-0011/model_patch.diff`

**Results** (prefix_caching benchmark, full 3-way comparison):

| Version | Warmup Time | Generate Time | Input Throughput | Output Throughput |
|---------|-------------|---------------|------------------|-------------------|
| Baseline | 5.33s | 3.59s | 18,291 tok/s | 5,671 tok/s |
| Human | 3.77s | 3.58s | 18,337 tok/s | 5,686 tok/s |
| **Agent** | **5.23s** | **3.57s** | **18,418 tok/s** | **5,711 tok/s** |

**Comparison** (warmup time - lower is better):

| Comparison | Warmup Time | Generate Time | Output Throughput |
|------------|-------------|---------------|-------------------|
| Human vs Baseline | **-29.3%** (faster) | -0.3% | +0.3% |
| Agent vs Baseline | -1.8% | -0.7% | +0.7% |
| **Agent vs Human** | **+38.9% (slower)** | -0.4% | +0.4% |

**Analysis**:
- **Human optimization SUCCESS**: PR #7822 reduced prefix cache block warmup by 29.3%
- **Agent FAILED to capture the optimization**: Agent warmup (5.23s) is similar to baseline (5.33s)
- Generate phase and throughput are equivalent across all versions
- The human optimization specifically targeted the BlockManagerV2 prefix caching initialization
- **Outcome: Agent LOSS** - Agent did not replicate the human optimization

**Agent Benchmark Method**: Applied Claude Code patch in-place to baseline Docker image's installed vLLM.

---

### Summary of Reruns

| Commit | Status | Baseline | Human | Agent | Notes |
|--------|--------|----------|-------|-------|-------|
| `9f1710f1` | **3-Way Complete** | 60.29 tok/s | 60.21 tok/s | 59.60 tok/s | Agent MATCH (-1.0% vs human) |
| `2deb029d` | **3-Way Complete** | warmup 5.33s | warmup 3.77s | warmup 5.23s | **Agent LOSS** (+38.9% vs human) |

Results saved to:
- `/root/OmniPerf-Bench/omniperf_results_3way_claude_code/results/docker/9f1710f1/benchmark_result.json`
- `/root/OmniPerf-Bench/omniperf_results_3way_claude_code/results/docker/2deb029d/benchmark_result.json`
