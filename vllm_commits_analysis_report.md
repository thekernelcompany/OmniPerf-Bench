# vLLM Commits Analysis Report

Analysis of 14 vLLM performance commits for benchmark suitability.

## Summary

| Status | Count | Percentage |
|--------|-------|------------|
| Usable as-is | 1 | 7% |
| Fixable with effort | 4 | 29% |
| Unusable | 9 | 64% |

### Key Metrics

| Metric | Count | Percentage |
|--------|-------|------------|
| Shows positive perf change | 3 | 21% |
| correct_domain = yes | 3 | 21% |
| has_baseline = true | 7 | 50% |
| Shows regression | 9 | 64% |

---

## Usable Commit

### `70b808fe1a63` — vllm_core-0035

**Subject:** Optimize qwen2-vl to reduce cudaMemcpyAsync

| Field | Value | Assessment |
|-------|-------|------------|
| PR | [#14377](https://github.com/vllm-project/vllm/pull/14377) | |
| Hardware | H100 | Available |
| Perf Change | +0.9% | Positive |
| target_type | same_target | Correct |
| approach_type | similar_approach | Correct |
| correct_domain | yes | |
| has_baseline | true | |
| invalid | null | No issues |

**Verdict:** Only fully valid commit. Benchmark matches optimization target.

---

## Fixable Commits

| # | Commit | Instance ID | Subject | Issue | Fix Required | Perf Change |
|---|--------|-------------|---------|-------|--------------|-------------|
| 5 | fa63e710c7fb | vllm_core-0091 | Reduce scheduling overhead after cuda sync | Effect size (-0.5%) within noise | Increase `--num-iters`, reduce variance, run more trials | -0.5% |
| 10 | 99abb8b650c6 | vllm_core-0051 | Optimize Rejection Sampler with Triton Kernels | Missing space in command | Fix: `'[ngram]' --ngram` (add space) | -0.5% |
| 11 | 6ce01f30667b | vllm_core-0030 | Optimize `get_seqs` | Benchmark doesn't stress sequence management | Use corrected command with 1000 prompts via throughput benchmark | -0.8% |
| 14 | 3476ed0809ec | vllm_core-0017 | Optimize block_manager_v2 vs v1 | Different commands needed for baseline vs test | Baseline: no flag; Test: `--use-v2-block-manager` | -1.6% |

### Fix Details

| Commit | Original Command | Corrected Command |
|--------|------------------|-------------------|
| vllm_core-0091 | `VLLM_USE_V1=1 python benchmarks/benchmark_latency.py --model meta-llama/Llama-3-8B --tensor-parallel-size 1 --input-len 1000 --batch-size 32` | Same, but add `--num-iters 100` and run multiple trials |
| vllm_core-0051 | `--speculative-model '[ngram]'--ngram_prompt_lookup_min 5` | `--speculative-model '[ngram]' --ngram_prompt_lookup_min 5` |
| vllm_core-0030 | `python benchmarks/benchmark_serving.py --model meta-llama/Llama-3-8B --backend vllm --num-prompts 100` | `python3 benchmarks/benchmark_throughput.py --backend vllm --model meta-llama/Meta-Llama-3-8B-Instruct --dataset-name random --input-len 1024 --output-len 256 --num-prompts 1000` |
| vllm_core-0017 | Same command for both | **Baseline:** `python benchmarks/benchmark_latency.py --model facebook/opt-125m --input-len 1536 --output-len 50 --batch-size 8`<br>**Test:** `python benchmarks/benchmark_latency.py --model facebook/opt-125m --input-len 1536 --output-len 50 --batch-size 8 --use-v2-block-manager` |

### Effort Estimate

| Commit | Effort | Notes |
|--------|--------|-------|
| vllm_core-0051 | Trivial | One character fix |
| vllm_core-0030 | Low | Swap benchmark command |
| vllm_core-0091 | Medium | Need to tune iterations + statistical analysis |
| vllm_core-0017 | Medium | Requires harness change to support different baseline/test commands |

---

## Unusable Commits

| # | Commit | Instance ID | Subject | Primary Issue | Hardware | Perf Change |
|---|--------|-------------|---------|---------------|----------|-------------|
| 2 | ed25054577f7 | vllm_core-0087 | Introduce popleft_n/append_n in FreeKVCacheBlockQueue | Micro-optimization benchmarked with macro-level serving test | H100 | -1.1% |
| 3 | 8a4e5c5f3c1d | vllm_core-0042 | Enhance P2pNcclConnector for P/D | Disaggregated serving not supported in harness | H100 | +1.7% |
| 4 | f26c4aeecba4 | vllm_core-0090 | Optimize ray worker initialization | Needs 4×H100 (TP4) | H100-TP4 | -0.0% |
| 6 | 6d0734c562e7 | vllm_core-0031 | Add SM100 Flashinfer MoE fp8 backend | SM100 = Blackwell arch, H100 is SM90 | H100 | -1.3% |
| 7 | 61b8cea3b42f | vllm_core-0026 | Optimize FlashInfer MetadataBuilder | B200-specific optimization | H100 | +0.0% |
| 8 | cf2f084d56a1 | vllm_core-0075 | Dynamic scheduler delay for ITL | Harmful tradeoff: hides ITL vs TTFT regression | H100 | +0.4% |
| 9 | 80aa7e91fcd5 | vllm_core-0038 | Optimize CPU backend | Intel CPU optimization tested on GPU | Intel-CPU | -0.5% |
| 12 | ca7a2d5f28ea | vllm_core-0072 | Revert MLA CPU overheads | Revert commit + needs 2×H100 (TP2) | H100-TP2 | -1.0% |
| 13 | 8bc68e198c4c | vllm_core-0044 | Auto-detect tensorized model | Needs tensorizer setup + S3 backend | H100 | -1.2% |

### Failure Categories

| Category | Count | Commits |
|----------|-------|---------|
| Wrong hardware arch | 3 | 0031, 0026, 0038 |
| Multi-GPU required | 2 | 0090, 0072 |
| Special infra needed | 2 | 0042, 0044 |
| Wrong benchmark level | 1 | 0087 |
| Harmful/misleading | 1 | 0075 |

---

## Detailed Analysis of Unusable Commits

### `ed25054577f7` — vllm_core-0087
**Subject:** Introduce popleft_n and append_n in FreeKVCacheBlockQueue

- **PR:** [#21222](https://github.com/vllm-project/vllm/pull/21222)
- **Issue:** Generic serving benchmark doesn't exercise KV cache block pool operations. The optimization is micro-level (data structure operations), but benchmark is macro-level (full serving).
- **Corrected command:** `python benchmarks/kv_cache/benchmark_block_pool.py`

### `8a4e5c5f3c1d` — vllm_core-0042
**Subject:** Enhance Performance for P2pNcclConnector (Prefill/Decode disaggregation)

- **PR:** [#20906](https://github.com/vllm-project/vllm/pull/20906)
- **Issue:** P/D disaggregation requires multi-node setup with separate prefill and decode instances. Standard single-node benchmark can't exercise this path.

### `f26c4aeecba4` — vllm_core-0090
**Subject:** Optimize ray worker initialization time

- **PR:** [#11275](https://github.com/vllm-project/vllm/pull/11275)
- **Issue:** Ray distributed execution only kicks in with TP>1. Single GPU benchmark doesn't use Ray workers at all. Even the benchmark command specifies `--tensor-parallel-size 4`.

### `6d0734c562e7` — vllm_core-0031
**Subject:** Add SM100 Flashinfer MoE blockscale fp8 backend

- **PR:** [#20645](https://github.com/vllm-project/vllm/pull/20645)
- **Issue:** SM100 = Blackwell architecture (B100/B200). H100 is SM90. This optimization literally cannot run on H100. The benchmark command also uses wrong model (Mistral-7B instead of MoE model like DeepSeek-R1).

### `61b8cea3b42f` — vllm_core-0026
**Subject:** Optimize FlashInfer MetadataBuilder Build call

- **PR:** [#21137](https://github.com/vllm-project/vllm/pull/21137)
- **Issue:** B200-specific optimization. FlashInfer metadata building differences may only manifest on Blackwell architecture.

### `cf2f084d56a1` — vllm_core-0075
**Subject:** Dynamic scheduler delay to improve ITL performance

- **PR:** [#3279](https://github.com/vllm-project/vllm/pull/3279)
- **Issue:** This PR intentionally trades TTFT (time to first token) for ITL (inter-token latency). A single aggregate benchmark hides this tradeoff. Need separate TTFT and ITL measurements to properly evaluate. Marked `harmful` because naive benchmarking could approve a regression.

### `80aa7e91fcd5` — vllm_core-0038
**Subject:** Optimize CPU backend and add more performance tips

- **PR:** [#4971](https://github.com/vllm-project/vllm/pull/4971)
- **Issue:** Intel CPU optimization tested on GPU. These are entirely different code paths. The benchmark is measuring literally nothing related to the commit.

### `ca7a2d5f28ea` — vllm_core-0072
**Subject:** Revert "[Perf] Reduce MLA CPU overheads in V1"

- **PR:** [#14471](https://github.com/vllm-project/vllm/pull/14471)
- **Issue:** This is a *revert* commit, so expecting "optimization" is wrong. MLA (Multi-head Latent Attention) is DeepSeek-specific. The corrected command shows TP2 requirement. Also, measuring a revert as "optimization" is conceptually wrong.

### `8bc68e198c4c` — vllm_core-0044
**Subject:** Automatically detect vLLM-tensorized model, update tensorizer

- **PR:** [#4208](https://github.com/vllm-project/vllm/pull/4208)
- **Issue:** Tensorizer is a model serialization format for faster loading. To benchmark this, you need:
  1. Pre-tensorized model weights
  2. S3/storage backend configured
  3. Compare load times, not inference times

  Generic inference benchmark doesn't measure model loading optimization.

---

## Recommendations

1. **Discard 9 unusable commits** — they'll produce noise, not signal
2. **Fix 4 fixable commits** — varying effort from trivial to medium
3. **Investigate why domain detection failed** — the `correct_domain=no` cases need root cause analysis
4. **Add hardware constraint filtering early** — reject commits targeting unavailable hardware before LLM analysis
5. **Validate benchmark commands against commit diffs** — ensure the modified code paths are actually exercised
6. **Consider narrower scope** — focus on single-H100 latency/throughput optimizations only

---

## Fixable Commits Rerun Results (2026-01-16)

We executed the 4 fixable commits with corrected benchmark commands. Results uploaded to `Inferencebench/claude-code-vllm-benchmarks`.

### Rerun Summary

| Commit | Task ID | Status | Baseline | Human | Agent | Human Imp. | Agent Imp. |
|--------|---------|--------|----------|-------|-------|------------|------------|
| 3476ed08 | vllm_core-0017 | All 3 succeeded | 169.20 ms | 175.44 ms | 184.16 ms | -3.69% | -8.85% |
| 6ce01f30 | vllm_core-0030 | All 3 succeeded | 9.18 req/s | 9.21 req/s | 9.20 req/s | +0.33% | +0.22% |
| 99abb8b6 | vllm_core-0051 | All 3 succeeded | 2174.04 ms | 2179.97 ms | 2186.58 ms | -0.27% | -0.58% |
| fa63e710 | vllm_core-0091 | Not run | N/A | N/A | N/A | N/A | N/A |

### Detailed Results

#### 3476ed08 — Optimize block_manager_v2 vs v1

**Benchmark:** Latency (asymmetric commands)
**Model:** facebook/opt-125m

| Variant | Command | Latency (ms) |
|---------|---------|--------------|
| Baseline | `--model facebook/opt-125m --input-len 1536 --output-len 50 --batch-size 8` | 169.20 |
| Human | Same + `--use-v2-block-manager` | 175.44 |
| Agent | Same + `--use-v2-block-manager` | 184.16 |

**Analysis:** The v2 block manager shows worse latency than v1 in this benchmark. Both human and agent patches regress, with agent performing worse (-8.85% vs -3.69%).

#### 6ce01f30 — Optimize get_seqs

**Benchmark:** Throughput (1000 prompts)
**Model:** meta-llama/Meta-Llama-3-8B-Instruct

| Variant | Throughput (req/s) | Tokens/s |
|---------|-------------------|----------|
| Baseline | 9.18 | 11,745 |
| Human | 9.21 | 11,787 |
| Agent | 9.20 | 11,770 |

**Analysis:** Marginal improvements (~0.2-0.3%). The optimization has minimal measurable impact at this scale.

#### 99abb8b6 — Optimize Rejection Sampler with Triton Kernels

**Benchmark:** Latency with speculative decoding
**Model:** meta-llama/Llama-3.1-8B-Instruct

| Variant | Status | Latency (ms) | Throughput (tok/s) |
|---------|--------|--------------|-------------------|
| Baseline | Success | 2174.04 | 2635.10 |
| Human | Success | 2179.97 | 2634.0 |
| Agent | Success | 2186.58 | 2635.6 |

**Analysis:** All variants show similar performance. Human shows -0.27% latency regression, Agent shows -0.58% regression. The Triton kernel optimization has minimal measurable impact in this benchmark configuration.

**Note:** The original human Docker image was missing the `vllm.benchmarks` module. It was rebuilt as `shikhar481/vllm_fixed_human_images:human-99abb8b650c66664cdc84d815b7f306f33bd9881` with vLLM installed from source to include benchmarks.

#### fa63e710 — Reduce scheduling overhead (NOT RUN)

**Reason:** Baseline Docker image `shikhar481/vllm_fixed_human_images:baseline-2a0309a646b1` does not exist.

---

### Docker Image Fix for 99abb8b6 (RESOLVED)

The original human image (`ayushnangia16/nvidia-vllm-docker:99abb8b6...`) was missing the `vllm.benchmarks` module because it was built with `pip install vllm` instead of from source.

**Resolution:** Rebuilt the image as `shikhar481/vllm_fixed_human_images:human-99abb8b650c66664cdc84d815b7f306f33bd9881` with vLLM installed from source to include the benchmarks module.

---

### Files Created

| File | Description |
|------|-------------|
| `scripts/reruns/rerun_4_fixable_commits.py` | Main benchmark runner script |
| `scripts/upload/upload_fixable_reruns.py` | HuggingFace upload script (76-column schema) |
| `omniperf_results_3way_claude_code/reruns/4_fixable/*.json` | Raw result files |

---

### Next Steps

1. **Build missing baseline image** for fa63e710 (parent: 2a0309a646b1)
2. **Consider statistical significance tests** for small improvements (<1%)
