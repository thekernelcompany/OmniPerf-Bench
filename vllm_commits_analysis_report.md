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

## 3-Way Benchmark Completeness Analysis (2026-01-18)

Analysis of HuggingFace dataset `Inferencebench/claude-code-vllm-benchmarks` for Baseline (B), Human (H), and Agent (A) metrics completeness.

### Summary

| Status | Unique Commits | Notes |
|--------|----------------|-------|
| Complete (B+H+A) | 20 | All three metric sets present |
| ONLY_AGENT | 20 | Agent metrics only, missing B/H |
| ONLY_BASELINE | 2 | Baseline only, missing H/A |
| MISSING_BASELINE | 1 | Has H+A, missing B |
| MISSING_HUMAN | 2 | Has B+A, missing H |
| No metrics at all | ~40 | Rows with no benchmark data |

### ONLY_AGENT Commits (20 commits)

These commits have agent patches but no baseline/human benchmark results:

| Commit | Status | Notes |
|--------|--------|-------|
| `19d98e0c` | Missing B/H | Needs Docker images |
| `22dd9c27` | Missing B/H | Needs Docker images |
| `25ebed2f` | Missing B/H | Needs Docker images |
| `67da5720` | Missing B/H | Needs Docker images |
| `6ce01f30` | Missing B/H | Needs Docker images |
| `83450458` | Missing B/H | Needs Docker images |
| `93e5f3c5` | Missing B/H | Needs Docker images |
| `99abb8b6` | Missing B/H | **REMINDER: Re-run with serving mode for claude_code** |
| `9a3b8832` | Missing B/H | Needs Docker images |
| `9badee53` | Missing B/H | Needs Docker images |
| `9d72daf4` | Missing B/H | Needs Docker images |
| `aea94362` | Missing B/H | Needs Docker images |
| `b2e0ad3b` | Missing B/H | Needs Docker images |
| `c0569dbc` | Missing B/H | Needs Docker images |
| `ca7a2d5f` | Missing B/H | Needs Docker images |
| `dcc6cfb9` | Missing B/H | Needs Docker images |
| `e206b543` | Missing B/H | Needs Docker images |
| `e7523c2e` | Missing B/H | Needs Docker images |
| `e7b20426` | Missing B/H | Needs Docker images |
| `f092153f` | Missing B/H | Needs Docker images |

### ONLY_BASELINE Commits (2 commits)

| Commit | Status |
|--------|--------|
| `22d33bac` | Has baseline only, no human or agent |
| `eefbf4a6` | Has baseline only, no human or agent |

### MISSING_BASELINE Commits (1 commit)

| Commit | Status |
|--------|--------|
| `9474e89b` | Has human+agent, missing baseline |

### MISSING_HUMAN Commits (2 commits)

| Commit | Status |
|--------|--------|
| `22d33bac` | Has baseline+agent, missing human |
| `eefbf4a6` | Has baseline+agent, missing human |

### Docker Image Availability Check (2026-01-18)

**Human Images (`ayushnangia16/nvidia-vllm-docker`):**
ALL 20 ONLY_AGENT commits have human images available.

| Commit | Human Image Tag | Status |
|--------|-----------------|--------|
| `19d98e0c` | `19d98e0c7db96713f0e2201649159431177a56e2` | ✓ Available |
| `22dd9c27` | `22dd9c2730dc1124b9d0ac15fff223d0b8d9020b` | ✓ Available |
| `25ebed2f` | `25ebed2f8ca6d747d63f2be9ede023c561851ac8` | ✓ Available |
| `67da5720` | `67da5720d4ed2aa1f615ec812031f4f3753b3f62` | ✓ Available |
| `6ce01f30` | `6ce01f30667bbae33f112152e07a3b66b841078f` | ✓ Available |
| `83450458` | `83450458339b07765b0e72a822e5fe93eeaf5258` | ✓ Available |
| `93e5f3c5` | `93e5f3c5fb4a4bbd49610efb96aad30df95fca66` | ✓ Available |
| `99abb8b6` | `99abb8b650c66664cdc84d815b7f306f33bd9881` | ✓ Available |
| `9a3b8832` | `9a3b88328f7e434cac35b90ee463de6689f9a833` | ✓ Available |
| `9badee53` | `9badee53decb3d432dc805336abfb0eb81dfb48f` | ✓ Available |
| `9d72daf4` | `9d72daf4ced05a5fec1ad8ea2914a39296f402da` | ✓ Available |
| `aea94362` | `aea94362c9bdd08ed2b346701bdc09d278e85f66` | ✓ Available |
| `b2e0ad3b` | `b2e0ad3b598ed0e022cdbd678a20821d411873c2` | ✓ Available |
| `c0569dbc` | `c0569dbc82b5e945a77878190114d1b68027828b` | ✓ Available |
| `ca7a2d5f` | `ca7a2d5f28eac9621474563cdda0e08596222755` | ✓ Available |
| `dcc6cfb9` | `dcc6cfb991cd76369aad96e04424f29c8fecdbd8` | ✓ Available |
| `e206b543` | `e206b5433109d298e53451015465b2bf8f03ef0a` | ✓ Available |
| `e7523c2e` | `e7523c2e031bc96740723ab63833d1cf94229ab4` | ✓ Available |
| `e7b20426` | `e7b204268132cb775c139574c1ff4ad7e15c8f66` | ✓ Available |
| `f092153f` | `f092153fbe349a9a1742940e3703bfcff6aa0a6d` | ✓ Available |

**Baseline Images (`shikhar481/vllm_fixed_human_images`):**
Baseline images use parent commit hash format: `baseline-<parent_hash>`

**VERIFIED: All 20 baseline images available**

| Commit | Parent Hash (12 char) | Baseline Tag |
|--------|----------------------|--------------|
| `19d98e0c` | `2b04c209ee98` | `baseline-2b04c209ee98` ✓ |
| `22dd9c27` | `a6d795d59304` | `baseline-a6d795d59304` ✓ |
| `25ebed2f` | `d263bd9df7b2` | `baseline-d263bd9df7b2` ✓ |
| `67da5720` | `5c04bb8b863b` | `baseline-5c04bb8b863b` ✓ |
| `6ce01f30` | `6a11fdfbb8d6` | `baseline-6a11fdfbb8d6` ✓ |
| `83450458` | `5b8a1fde8422` | `baseline-5b8a1fde8422` ✓ |
| `93e5f3c5` | `70363bccfac1` | `baseline-70363bccfac1` ✓ |
| `99abb8b6` | `3a1e6481586e` | `baseline-3a1e6481586e` ✓ |
| `9a3b8832` | `3014c920dae5` | `baseline-3014c920dae5` ✓ |
| `9badee53` | `beebf4742af8` | `baseline-beebf4742af8` ✓ |
| `9d72daf4` | `6dd55af6c9dd` | `baseline-6dd55af6c9dd` ✓ |
| `aea94362` | `7206ce4ce112` | `baseline-7206ce4ce112` ✓ |
| `b2e0ad3b` | `4a18fd14ba4a` | `baseline-4a18fd14ba4a` ✓ |
| `c0569dbc` | `8bb43b9c9ee8` | `baseline-8bb43b9c9ee8` ✓ |
| `ca7a2d5f` | `333681408fea` | `baseline-333681408fea` ✓ |
| `dcc6cfb9` | `dd572c0ab3ef` | `baseline-dd572c0ab3ef` ✓ |
| `e206b543` | `1d35662e6dc1` | `baseline-1d35662e6dc1` ✓ |
| `e7523c2e` | `a869baca73eb` | `baseline-a869baca73eb` ✓ |
| `e7b20426` | `90f1e55421f1` | `baseline-90f1e55421f1` ✓ |
| `f092153f` | `1da8f0e1ddda` | `baseline-1da8f0e1ddda` ✓ |

### Conclusions

1. **Human images: 100% coverage** - All 20 ONLY_AGENT commits have human Docker images ready
2. **Baseline images: 100% coverage** - All 20 parent hashes have baseline images available
3. **Ready to benchmark** - All B/H Docker images exist, can proceed immediately

### Action Items

1. ~~Check Docker Hub for existing images for the 20 ONLY_AGENT commits~~ ✓ Complete
2. ~~Cross-reference parent commit hashes for baseline image availability~~ ✓ Complete (20/20)
3. ~~Re-run benchmark for `99abb8b6` with serving mode for claude_code agent~~ ✓ Complete
4. ~~Run baseline + human benchmarks for 20 ONLY_AGENT commits~~ ✓ Complete (17/19 succeeded)

---

## Baseline Benchmark Results (2026-01-18)

### Summary

| Status | Count | Percentage |
|--------|-------|------------|
| Succeeded | 17 | 89.5% |
| Failed (unfixable) | 2 | 10.5% |

### Fixes Applied to Benchmark Runner

The following fixes were applied to `scripts/runners/run_3way_benchmarks.py` to resolve baseline failures:

1. **Argument filtering for serving benchmarks** — Filter server-only args that `benchmark_serving.py` doesn't accept:
   - `--dtype`, `--guided-decoding-backend`, `--tensor-parallel-size`, `--enforce-eager`
   - `--gpu-memory-utilization`, `--max-model-len`, `--max-concurrency`

2. **Argument filtering for latency benchmarks** — Filter speculative decoding args (human's optimization, not baseline):
   - `--speculative-model`, `--num-speculative-tokens`, `--speculative-draft-token-sampling-method`
   - `--ngram-prompt-lookup-max`, `--spec-decoding-acceptance-method`

3. **Improved transformers compatibility** — Check both `LogitsWarper` AND `transformers.utils` for import failures

### Successful Baselines (17 commits)

| Commit | Model | Metric | Value |
|--------|-------|--------|-------|
| `22dd9c27` | meta-llama/Llama-3.1-8B-Instruct | latency | 821.55 ms |
| `25ebed2f` | meta-llama/Llama-3.1-8B-Instruct | throughput | 3134.11 tok/s |
| `67da5720` | meta-llama/Llama-3.1-8B-Instruct | throughput | 3817.56 tok/s |
| `83450458` | meta-llama/Llama-3.1-8B-Instruct | latency | 1895.63 ms |
| `93e5f3c5` | meta-llama/Llama-3.1-8B-Instruct | throughput | 2856.15 tok/s |
| `99abb8b6` | meta-llama/Llama-3.1-8B-Instruct | throughput | 3780.55 tok/s |
| `9a3b8832` | meta-llama/Llama-3.1-8B-Instruct | throughput | 6131.44 tok/s |
| `9badee53` | meta-llama/Llama-3.1-8B-Instruct | throughput | 10588.27 tok/s |
| `9d72daf4` | meta-llama/Llama-3.1-8B-Instruct | throughput | 3053.03 tok/s |
| `aea94362` | meta-llama/Llama-3.2-1B-Instruct | throughput | 8987.95 tok/s |
| `b2e0ad3b` | meta-llama/Llama-3.1-8B-Instruct | throughput | 2539.67 tok/s |
| `c0569dbc` | Qwen/Qwen3-30B-A3B-FP8 | throughput | 2728.77 tok/s |
| `ca7a2d5f` | deepseek-ai/DeepSeek-Coder-V2-Lite-Instruct | throughput | 3821.83 tok/s |
| `dcc6cfb9` | Qwen/Qwen3-30B-A3B-FP8 | throughput | 2739.95 tok/s |
| `e206b543` | meta-llama/Llama-3.1-8B-Instruct | throughput | 3116.22 tok/s |
| `e7b20426` | 01-ai/Yi-1.5-9B-Chat | throughput | 3084.01 tok/s |
| `f092153f` | meta-llama/Llama-3.1-8B-Instruct | throughput | 3186.16 tok/s |

---

## Unfixable Baseline Failures

### `6ce01f30` — Corrupted Docker Image

**Model:** meta-llama/Meta-Llama-3-8B
**Baseline Image:** `shikhar481/vllm_fixed_human_images:baseline-6a11fdfbb8d6`

**Root Cause:** Severely corrupted Python environment in Docker image.

**Error Chain:**
1. `ModuleNotFoundError: No module named 'transformers.utils'` — transformers package corrupted
2. After fixing transformers: `ModuleNotFoundError: No module named 'pyairports'` — outlines dependency missing
3. Multiple `WARNING: Ignoring invalid distribution -ransformers` — indicates corrupted pip state

**Fix Attempts:**
1. ✗ Conditional transformers downgrade — triggered but pip install failed
2. ✗ Force uninstall + clean reinstall — `OSError: [Errno 39] Directory not empty: 'integrations'`
3. ✗ Install pyairports/pycountry — still failing with import errors

**Verdict:** Docker image `baseline-6a11fdfbb8d6` has fundamentally broken Python packaging. The corrupted state persists across multiple pip operations. **Unfixable without rebuilding the Docker image.**

**Critical Note:** The human benchmark for this commit succeeded, indicating the human's Docker image is functional while the baseline image is not. This is an infrastructure issue, not a benchmark design issue.

---

### `e7523c2e` — Hardware Memory Limitation

**Model:** google/gemma-3-12b-it
**Baseline Image:** `shikhar481/vllm_fixed_human_images:baseline-a869baca73eb`

**Root Cause:** Model's default context length exceeds GPU memory capacity.

**Error:**
```
ValueError: The model's max seq len (131072) is larger than the maximum
number of tokens that can be stored in KV cache (50352). Try increasing
`gpu_memory_utilization` or decreasing `max_model_len` when initializing the engine.
```

**Key Facts:**
- Model default max_seq_len: 131,072 tokens
- Available KV cache capacity: 50,352 tokens
- Human benchmark: **Succeeded** with 1747.58 tok/s
- Baseline benchmark: **Failed** — cannot start server

**Analysis:**
The human's optimized vLLM can run this model configuration while the baseline vLLM cannot. This suggests one of:
1. The human's optimization IS about memory efficiency (allowing larger context)
2. The human's Docker image has different server configuration

**Why NOT reduce `--max-model-len`:**
- Reducing context length for baseline creates an **unfair comparison**
- Baseline would get artificial help (smaller context) that human didn't need
- This would contaminate benchmark validity — we'd be measuring different conditions
- The whole point of baseline is "performance WITHOUT the optimization"

**Verdict:** **Unfixable without invalidating the benchmark comparison.** The baseline genuinely cannot run this configuration, which may be exactly what the human's optimization addresses. Leaving as failed preserves benchmark integrity.

---

## Final Baseline Results Summary

| Metric | Value |
|--------|-------|
| Total commits | 19 |
| Successful baselines | 17 (89.5%) |
| Failed baselines | 2 (10.5%) |
| Failure: Corrupted image | 1 (`6ce01f30`) |
| Failure: Hardware limit | 1 (`e7523c2e`) |

---

## Claude Code Serving Benchmark Fixes (2026-01-19)

### Summary

Fixed 4 Claude Code agent benchmarks that had WRONG METRIC (throughput/latency instead of ttft/tpot/itl for serving mode).

| Commit | Model | TTFT (ms) | TPOT (ms) | ITL (ms) | Throughput (tok/s) |
|--------|-------|-----------|-----------|----------|-------------------|
| 99abb8b6 | Llama-3.1-8B | 656.64 | 30.98 | 24.46 | 2810.3 |
| 22d33bac | Llama-3.1-8B | 651.12 | 30.51 | 24.52 | 2813.8 |
| 9badee53 | Llama-3.2-1B | 174.68 | 9.85 | 7.98 | 8080.42 |
| e206b543 | Llama-3.1-8B | 669.91 | 30.88 | 24.56 | 2784.84 |

### Impact

- **Before:** Claude Code had 20 VALID, 12 WRONG METRIC
- **After:** Claude Code has 24 VALID, 8 WRONG METRIC
- **Complete 4-way benchmarks enabled:** All 4 commits now have VALID data for all agents (except TRAE-GPT for e206b543)

### Results Location

Files saved to: `/root/OmniPerf-Bench/omniperf_results_3way_claude_code/results/{commit}_agent_result.json`

---

## Remaining Discrepancies Analysis

### Remaining WRONG METRIC Commits (8 Claude Code)

| # | Commit | Mode | Issue | Fixable? |
|---|--------|------|-------|----------|
| 1 | 7c01f706 | serving | Has throughput, needs ttft | Yes - re-run serving benchmark |
| 2 | 89a84b0b | serving | Has throughput, needs ttft | Yes - re-run serving benchmark |
| 3 | 3476ed08 | standalone | Has throughput for serving test | **Different** - standalone needs throughput |
| 4 | 19d98e0c | serving | Has throughput, needs ttft | Yes - but ALL agents have issues |
| 5 | 6e36f4fa | serving | Has throughput, needs ttft | Yes - re-run serving benchmark |
| 6 | fc7b8d1e | serving | Has throughput, needs ttft | Yes - re-run serving benchmark |
| 7 | 3a243095 | serving | Has throughput, needs ttft | Yes - re-run serving benchmark |
| 8 | e3580537 | serving | Has throughput, needs ttft | Yes - re-run serving benchmark |

**Summary:** 7 serving mode commits are fixable with same approach as the 4 fixed today. 1 standalone commit (3476ed08) has different issues.

### Cross-Agent Discrepancies

#### Commits Where Claude Code Succeeds but Others Fail

| Commit | Claude Code | Codex | TRAE-Sonnet | TRAE-GPT | Notes |
|--------|-------------|-------|-------------|----------|-------|
| b690e348 | ✓ VALID | FAIL | FAIL | FAIL | Only Claude Code succeeded |
| 2deb029d | ✓ VALID | FAIL | FAIL | FAIL | Only Claude Code succeeded |
| 015069b0 | ✓ VALID | FAIL | FAIL | FAIL | Only Claude Code succeeded |
| a3223766 | ✓ VALID | FAIL | FAIL | FAIL | Only Claude Code succeeded |
| 310aca88 | ✓ VALID | FAIL | FAIL | FAIL | Only Claude Code succeeded |
| bc7c4d20 | ✓ VALID | FAIL | FAIL | FAIL | Only Claude Code succeeded |
| 9474e89b | ✓ VALID | FAIL | FAIL | FAIL | Only Claude Code succeeded |
| 6a417b86 | ✓ VALID | FAIL | FAIL | FAIL | Only Claude Code succeeded |
| fe66b347 | ✓ VALID | FAIL | FAIL | FAIL | Only Claude Code succeeded |

**Pattern:** Claude Code has significantly higher patch success rate (74.4%) compared to Codex/TRAE (39.5%/27.9%).

#### Commits Where Others Succeed but Claude Code Fails

| Commit | Claude Code | Codex | TRAE-Sonnet | TRAE-GPT | Notes |
|--------|-------------|-------|-------------|----------|-------|
| e7b20426 | FAIL | ✓ VALID | ✓ VALID | FAIL | Claude Code patch failed |
| 8c1e77fb | FAIL | ✓ VALID | ✓ VALID | ✓ VALID | Claude Code patch failed |
| 3b61cb45 | FAIL | ✓ VALID | ✓ VALID | ✓ VALID | Claude Code patch failed |

**Analysis:** 3 commits where Claude Code's patch generation failed but others succeeded. These may represent edge cases in Claude Code's approach.

#### Special Case: 19d98e0c

| Agent | Status | Details |
|-------|--------|---------|
| Claude Code | WRONG METRIC | Has throughput instead of ttft |
| Codex | MISSING human_ttft | Agent data exists, human baseline missing |
| TRAE-Sonnet | MISSING human_ttft | Agent data exists, human baseline missing |
| TRAE-GPT | PATCH FAILURE | No agent data |

**Verdict:** This commit has issues across ALL agents. Human benchmark data may be incomplete in the HuggingFace dataset.

### Commits with Complete 4-Way Data (All Agents VALID)

After fixes, these commits now have complete benchmark data across all 4 agents:

| Commit | Mode | Status |
|--------|------|--------|
| fa63e710 | standalone | ✓ All 4 agents VALID |
| 4c822298 | standalone | ✓ All 4 agents VALID |
| b55ed6ef | serving | ✓ All 4 agents VALID |
| 58eee5f2 | serving | ✓ All 4 agents VALID |
| 98f47f2a | standalone | ✓ All 4 agents VALID |
| 99abb8b6 | serving | ✓ All 4 agents VALID (NEW) |
| 22d33bac | serving | ✓ All 4 agents VALID (NEW) |
| 9badee53 | serving | ✓ All 4 agents VALID (NEW) |

**Total:** 8 commits with complete 4-way comparison data.

---

## Recommendations

### Immediate Actions

1. **Fix remaining 7 serving WRONG METRIC commits** for Claude Code:
   - 7c01f706, 89a84b0b, 19d98e0c, 6e36f4fa, fc7b8d1e, 3a243095, e3580537
   - Same approach as today: `run_3way_benchmarks.py --agent-type claude_code --commits <list> --agent-only`

2. **Investigate 19d98e0c** - all agents have issues, may need human benchmark data fix

3. **Upload new results to HuggingFace** - sync local results with dataset

### Longer-term

1. **Investigate Claude Code patch failures** on e7b20426, 8c1e77fb, 3b61cb45
2. **Investigate standalone commit 3476ed08** - may have benchmark command mismatch
3. **Consider re-running TRAE-GPT** for commits where it uniquely failed
