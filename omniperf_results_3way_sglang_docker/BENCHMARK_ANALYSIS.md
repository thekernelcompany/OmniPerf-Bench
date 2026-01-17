# SGLang 3-Way Benchmark Analysis

**Date:** 2026-01-17 (Updated)
**GPU:** NVIDIA H100 PCIe (80GB)
**Benchmark Type:** 3-way comparison (Baseline vs Human vs Agent)

---

## Executive Summary

- **Total Commits:** 80
- **Successfully Benchmarked:** 2 (2.5%)
- **Primary Blockers:** Missing benchmark commands (57.5%), Multi-GPU requirements (28.8%), Broken Docker images (6.2%)

### Update 2026-01-17

Benchmark results for:
- **021f76e4**: Complete 3-way benchmark (Baseline + Human + Agent) - **Human +15-20% improvement**
- **6fc17596**: Complete benchmark (Baseline + Human) - **Micro-optimization (<1% macro impact)**

---

## Filtering Funnel

```
80 Total Commits
    │
    ├── 46 (57.5%) ✗ No perf_command
    │                 └─ No benchmark command found in PR extraction
    │
    └── 34 (42.5%) With perf_command
            │
            ├── 23 (28.8%) ✗ Multi-GPU required
            │                 └─ Models: DeepSeek, Llama-70B/405B, Qwen-72B, etc.
            │
            └── 11 (13.8%) Single-GPU compatible
                    │
                    ├── 5 (6.2%) ✗ No Docker image
                    │              └─ Image never built or not on DockerHub
                    │
                    └── 6 (7.5%) Docker images available
                            │
                            ├── 5 (6.2%) ✗ Broken Docker image
                            │              └─ Missing libnuma.so.1 (H100 incompatible)
                            │
                            └── 1 (1.25%) ✓ COMPLETED
```

---

## Stage-by-Stage Breakdown

### Stage 1: Extraction Data

| Metric | Count | Percentage |
|--------|-------|------------|
| Total commits extracted | 80 | 100% |
| With perf_command | 34 | 42.5% |
| **WITHOUT perf_command** | **46** | **57.5%** |

### Stage 2: GPU Requirements (of 34 with perf_command)

| Metric | Count | Percentage |
|--------|-------|------------|
| Single-GPU compatible | 11 | 32.4% |
| **Multi-GPU required** | **23** | **67.6%** |

### Stage 3: Docker Image Availability

| Metric | Count |
|--------|-------|
| Total Docker images on Hub | 67 |
| Working images (with libnuma) | 4 (short-hash only) |
| **Broken images (missing deps)** | **63** |

### Stage 4: Final Runnable Candidates

| Metric | Count |
|--------|-------|
| Valid candidates (both images exist) | 6 |
| Working (both images functional) | 1 |
| **Broken (libnuma.so.1 missing)** | **5** |

---

## Final 6 Candidates - Detailed Status

| # | Commit | PR Subject | Model | Status | Reason |
|---|--------|-----------|-------|--------|--------|
| 1 | `021f76e4` | LoRAManager stream sync optimization (#6994) | Llama-3.1-8B-Instruct | ✓ **COMPLETED** | Both images work |
| 2 | `2bd18e2d` | Memory pool optimization (#2901) | Llama-2-7b-hf | ✗ Failed | libnuma.so.1 missing |
| 3 | `93470a14` | FA3 Code optimization (#5090) | Llama-3.1-8B-Instruct | ✗ Failed | libnuma.so.1 missing |
| 4 | `bb3a3b66` | JSON decoding for llava (#137) | llava-1.5-7b-hf | ✗ Failed | libnuma.so.1 missing |
| 5 | `d1112d85` | Input embeds endpoint (#2797) | gemma-2-2b | ✗ Failed | libnuma.so.1 missing |
| 6 | `ddcf9fe3` | Triton attention mask (#3731) | Llama-2-7b-chat-hf | ✗ Failed | libnuma.so.1 missing |

---

## Root Cause Analysis

| Blocker | Count | % of Total | Fix Required |
|---------|-------|------------|--------------|
| No perf_command | 46 | 57.5% | Extract benchmark commands from PRs |
| Multi-GPU models | 23 | 28.8% | Need multi-GPU infrastructure |
| No Docker image | 5 | 6.2% | Build missing images |
| Broken images (libnuma) | 5 | 6.2% | Rebuild with `libnuma-dev` |
| **Completed** | **1** | **1.25%** | - |

### Docker Image Issue Details

The Docker images were built in two batches:

1. **Short-hash images** (4 images: `021f76e4`, `777688b8`, `3212c2ad`, `53475674`)
   - Built on: 2026-01-13
   - Status: **Working** - includes all required dependencies

2. **Full-hash images** (63 images)
   - Built on: 2026-01-10 to 2026-01-16
   - Status: **Broken** - missing `libnuma.so.1` library
   - Error: `ImportError: libnuma.so.1: cannot open shared object file`
   - Impact: `sgl_kernel` fails to load on H100 GPUs (SM90 architecture)

---

## Successful Benchmark Result

### Commit Details

| Field | Value |
|-------|-------|
| Commit | `021f76e4f49861b2e9ea9ccff06a46d577e3c548` |
| Subject | [Perf] Refactor LoRAManager to eliminate stream syncs and redundant computations |
| PR | https://github.com/sgl-project/sglang/pull/6994 |
| Model | meta-llama/Llama-3.1-8B-Instruct |
| Parent Commit | `777688b8929c877e4e28c2eac208d776abe4c3af` |

### Benchmark Configuration

| Parameter | Value |
|-----------|-------|
| Benchmark Command | `python3 -m sglang.bench_serving --backend sglang --num-prompt 480 --request-rate 8 --lora-name lora` |
| LoRA Adapter | `algoprog/fact-generation-llama-3.1-8b-instruct-lora` |
| GPU | H100:1 |
| Duration | 413.8 seconds |

### Performance Results

#### Human vs Baseline (Lower is better for latency, higher for throughput)

| Metric | Baseline | Human | Improvement |
|--------|----------|-------|-------------|
| TTFT Mean (ms) | 121.01 | 102.62 | **+15.2%** |
| TTFT Median (ms) | 110.92 | 91.51 | **+17.5%** |
| TTFT P99 (ms) | 368.65 | 344.42 | **+6.6%** |
| ITL Mean (ms) | 56.17 | 45.66 | **+18.7%** |
| ITL Median (ms) | 50.86 | 40.78 | **+19.8%** |
| ITL P99 (ms) | 203.79 | 145.77 | **+28.5%** |
| E2E Latency Mean (ms) | 10263.01 | 8347.08 | **+18.7%** |
| E2E Latency Median (ms) | 6631.04 | 5157.15 | **+22.2%** |
| Throughput (req/s) | 5.78 | 6.21 | **+7.4%** |

#### Agent vs Baseline

| Metric | Baseline | Agent | Improvement |
|--------|----------|-------|-------------|
| TTFT Mean (ms) | 121.01 | 123.42 | **-1.99%** |
| TTFT Median (ms) | 110.92 | 114.48 | **-3.21%** |
| ITL Mean (ms) | 56.17 | 56.54 | **-0.66%** |
| Throughput (req/s) | 5.78 | 5.77 | **-0.17%** |

#### Agent vs Human

| Metric | Comparison |
|--------|------------|
| TTFT Mean | Agent **20.3% worse** than Human |
| TTFT Median | Agent **25.1% worse** than Human |
| Throughput | Agent **7.1% worse** than Human |

### Key Finding

The **human commit achieved significant performance improvements** (15-18% across latency metrics), but the **agent patch did not replicate these optimizations**. The agent's patch resulted in performance similar to (or slightly worse than) the baseline.

---

## Updated Results (2026-01-17)

### Commit 021f76e4 - Complete 3-Way Results

Re-ran with clean GPU resources. All three phases completed successfully.

#### Final Baseline vs Human Comparison

| Metric | Baseline | Human | Improvement |
|--------|----------|-------|-------------|
| TTFT Mean (ms) | 123.46 | 104.05 | **+15.72%** |
| TTFT Median (ms) | 114.48 | 91.57 | **+20.01%** |
| TTFT P99 (ms) | 367.42 | 356.25 | +3.04% |
| ITL Mean (ms) | 56.47 | 45.68 | **+19.11%** |
| ITL Median (ms) | 51.23 | 40.77 | **+20.42%** |
| ITL P99 (ms) | 176.77 | 151.11 | **+14.52%** |
| E2E Latency Mean (ms) | 10319.50 | 8353.01 | **+19.06%** |
| E2E Latency Median (ms) | 6573.65 | 5151.11 | **+21.64%** |
| Throughput (req/s) | 5.78 | 6.21 | **+7.44%** |

#### Human vs Agent Comparison

| Metric | Human | Agent | Human Advantage |
|--------|-------|-------|-----------------|
| TTFT Mean (ms) | 111.62 | 122.94 | **+10.1%** |
| TTFT Median (ms) | 93.61 | 115.28 | **+23.1%** |
| TTFT P99 (ms) | 437.64 | 365.99 | -16.4% |
| ITL Mean (ms) | 45.61 | 57.14 | **+25.3%** |
| ITL Median (ms) | 40.68 | 51.52 | **+26.6%** |
| ITL P99 (ms) | 155.24 | 186.43 | **+20.1%** |
| Throughput (req/s) | 6.21 | 5.73 | **+8.4%** |
| E2E Latency Mean (ms) | 8347.61 | 10441.25 | **+25.1%** |

**Conclusion:** Human PR significantly outperforms the AI-generated patch by 10-25% across most metrics.

---

### Commit 6fc17596 - FA3 Pad Optimization (Complete)

**PR:** [sgl-project/sglang#5945](https://github.com/sgl-project/sglang/pull/5945)
**Subject:** Optimize FA3 pad operation (71% faster - 35us -> 10us)
**Docker Repo:** ayushnangia16/nvidia-sglang-docker

#### Status
- **Baseline:** SUCCESS
- **Human:** SUCCESS
- **Agent:** N/A (no agent patch available)

#### Baseline vs Human Comparison

| Metric | Baseline | Human | Improvement |
|--------|----------|-------|-------------|
| TTFT Mean (ms) | 62.68 | 62.45 | **+0.37%** |
| TTFT Median (ms) | 41.15 | 41.29 | -0.34% |
| TTFT P99 (ms) | 425.72 | 423.90 | +0.43% |
| ITL Mean (ms) | 13.72 | 13.71 | **+0.07%** |
| ITL Median (ms) | 12.10 | 12.12 | -0.17% |
| ITL P99 (ms) | 42.84 | 42.74 | +0.23% |
| E2E Latency Mean (ms) | 2904.94 | 2902.56 | +0.08% |
| E2E Latency Median (ms) | 1816.55 | 1828.42 | -0.65% |
| Throughput (req/s) | 5.90 | 5.90 | **0.0%** |

#### Key Finding

The **claimed 71% improvement (35us -> 10us)** refers to a **micro-operation** (the pad operation within FA3). At the macro level of a full serving benchmark:
- The improvement is **< 1%** and within measurement noise
- A 25us savings per operation is negligible compared to:
  - Overall E2E latency (~2900ms)
  - ITL (~13ms)
  - TTFT (~62ms)

This demonstrates the difference between **micro-benchmarks** (specific operation timing) and **macro-benchmarks** (end-to-end performance). While the PR's optimization is real and validated by the 71% micro-benchmark improvement, it doesn't significantly impact overall serving performance.

---

## Recommendations

### To Increase Benchmark Coverage

1. **Rebuild Docker Images** (High Impact)
   - Add `libnuma-dev` to base image: `apt-get install -y libnuma-dev`
   - Would enable 5 additional candidates (6x increase)

2. **Use Inferred Benchmark Commands** (COMPLETED - 2026-01-16)
   - **30 commits** have inferred benchmark commands from code analysis
   - **7 Tier-1 commits** with quantified improvements (21-71% faster)
   - **10 Multi-GPU required** - need infrastructure upgrade
   - **6 skipped** - reverts or infrastructure-only changes

3. **Multi-GPU Infrastructure** (High Effort)
   - Set up multi-GPU benchmarking capability
   - Would enable 10 additional candidates from the 46 analyzed

### Priority Order

| Priority | Action | Commits Enabled | Effort | Status |
|----------|--------|-----------------|--------|--------|
| 1 | Fix Docker images (libnuma) | +5 | Low | Pending |
| 2 | Use inferred perf_commands | +30 | Low | **READY** |
| 3 | Multi-GPU setup | +10 | High | Pending |

### Benchmark Command Inference Method

Commands were inferred by:
1. **Analyzing files changed** in each PR (e.g., `flashattention_backend.py` → FA3 decode)
2. **Reading PR context** for claimed improvements and test scenarios
3. **Mapping to SGLang benchmark scripts** (`bench_latency` vs `bench_serving`)
4. **Setting appropriate parameters** based on optimization target:
   - FA3 decode → large batch, long context
   - Prefill → single batch, 8K input, minimal output
   - Scheduler → high concurrency (request-rate 16)
   - LoRA → with adapter, disable cache
   - JSON/API → high RPS, short outputs

---

## Files Changed in Successful Commit

```
python/sglang/srt/lora/lora_manager.py
python/sglang/srt/lora/mem_pool.py
```

---

## Extracted Benchmark Commands (46 PRs Analyzed)

**Analysis Method:** Code changes (files modified) + GitHub PR context + SGLang benchmark script knowledge

### Summary

| Category | Count | Percentage |
|----------|-------|------------|
| **Inferred Commands** (code + PR analysis) | 30 | 65.2% |
| **Multi-GPU Required** | 10 | 21.7% |
| **Skip (Reverts/Infrastructure)** | 6 | 13.0% |

---

### Tier 1: High-Impact Optimizations (Quantified Improvements)

#### `79961afa` | PR #6077 - FA3 pad operations (21% faster)
**Claimed:** 530us -> 418us for FA3 metadata init
```bash
python3 -m sglang.bench_latency --model meta-llama/Llama-3.1-8B-Instruct --batch-size 64 --input-len 2048 --output-len 256
```
**Reason:** FA3 decode path - large batch needed to see 100us+ impact

---

#### `6fc17596` | PR #5945 - FA3 pad operation (71% faster)
**Claimed:** 35us -> 10us for pad operation
```bash
python3 -m sglang.bench_latency --model meta-llama/Llama-3.1-8B-Instruct --batch-size 64 --input-len 2048 --output-len 256
```
**Reason:** FA3 decode path optimization

---

#### `2a754e57` | PR #579 - 2x prefill performance
**Claimed:** 2x performance improvement for large prefill
```bash
python3 -m sglang.bench_latency --model meta-llama/Llama-3.1-8B-Instruct --batch-size 1 --input-len 8192 --output-len 1
```
**Reason:** LARGE PREFILL benchmark - 8K input tokens, minimal output

---

#### `b1e5a33a` | PR #6960 - LoRA stream sync (13% ITL improvement)
**Claimed:** 13% ITL@P95 at request rate 8
```bash
python3 -m sglang.bench_serving --backend sglang --model meta-llama/Llama-3.1-8B-Instruct --num-prompt 480 --request-rate 8 --disable-radix-cache
# Server: --lora-paths lora=algoprog/fact-generation-llama-3.1-8b-instruct-lora
```
**Reason:** LoRA serving optimization

---

#### `9216b106` | PR #394 - Scheduler (40% faster)
**Claimed:** 90.094s -> 53.505s, cache hit 24% -> 74%
```bash
python3 -m sglang.bench_serving --backend sglang --model meta-llama/Llama-3.1-8B-Instruct --num-prompt 300 --request-rate 16
```
**Reason:** LPM scheduler priority - HIGH parallelism workload

---

#### `3212c2ad` | PR #6003 - VLM tensor transport (16% faster)
**Claimed:** 207.7s -> 173.3s on MMMU benchmark
```bash
python3 -m sglang.bench_serving --backend sglang --model llava-hf/llava-1.5-7b-hf --num-prompt 100 --request-rate 2
```
**Reason:** VLM CUDA IPC optimization

---

#### `c087ddd6` | PR #6627 - Triton kernel (10-15% faster)
**Claimed:** 8% micro-benchmark (0.026ms -> 0.024ms)
```bash
python benchmark/kernels/fused_moe_triton/benchmark_ep_pre_reorder_triton.py --hidden-size 1024
```
**Reason:** EXPLICIT benchmark script in PR

---

### Tier 2: Scheduler/Batching Optimizations

| Commit | PR | Subject | Benchmark Command | Reason |
|--------|-----|---------|-------------------|--------|
| `ab4a83b2` | #1339 | Optimize schedule | `bench_serving --num-prompt 300 --request-rate 16` | PrefillAdder optimization |
| `10189d08` | #2171 | CPU affinity | `bench_serving --num-prompt 200 --request-rate 16` | NUMA impact at high load |
| `62757db6` | #1010 | Cache disabled | `bench_serving --num-prompt 200 --request-rate 8 --disable-radix-cache` | Cache disabled scenario |

---

### Tier 3: Streaming/TTFT Optimizations

| Commit | PR | Subject | Benchmark Command | Server Args |
|--------|-----|---------|-------------------|-------------|
| `ac971ff6` | #658 | stream_interval | `bench_serving --num-prompt 100 --request-rate 4 --output-len 512` | `--stream-interval 1` |
| `6f560c76` | #117 | First token latency | `bench_serving --num-prompt 100 --request-rate 4 --output-len 256` | - |

---

### Tier 4: FA3/Attention Optimizations

| Commit | PR | Subject | Benchmark Command |
|--------|-----|---------|-------------------|
| `1acca3a2` | #5969 | FA3 len() removal | `bench_latency --batch-size 64 --input-len 2048 --output-len 256` |
| `fbcbb263` | #1765 | KV buffer fix | `bench_latency --batch-size 32 --input-len 1024 --output-len 256` |

---

### Tier 5: Memory/Cache Optimizations

| Commit | PR | Subject | Benchmark Command |
|--------|-----|---------|-------------------|
| `b1709305` | #1697 | Radix tree | `bench_serving --num-prompt 500 --request-rate 10` |
| `09deb20d` | #420 | Logits memory | `bench_serving --num-prompt 300 --request-rate 10` |
| `564a898a` | #619 | Mem indices | `bench_serving --num-prompt 300 --request-rate 10` |
| `f06e90c2` | #440 | Retract optimize | `bench_serving --num-prompt 300 --request-rate 10` |

---

### Tier 6: LoRA Optimizations

| Commit | PR | Subject | Benchmark Command | Server Args |
|--------|-----|---------|-------------------|-------------|
| `9c064bf7` | #1587 | LoRA Step 1 | `bench_serving --num-prompt 480 --request-rate 8 --disable-radix-cache` | LoRA adapter |

---

### Tier 7: Sampling/Decode Optimizations

| Commit | PR | Subject | Benchmark Command |
|--------|-----|---------|-------------------|
| `8f8f96a6` | #1773 | stop_token_ids fix | `bench_latency --batch-size 64 --input-len 512 --output-len 128` |
| `c98e84c2` | #1589 | torch.argmax | `bench_latency --batch-size 64 --input-len 256 --output-len 256` |
| `2854a5ea` | #1496 | bench_latency fix | `bench_latency --batch-size 32 --input-len 512 --output-len 128` |

---

### Tier 8: Server/API Optimizations

| Commit | PR | Subject | Benchmark Command | Reason |
|--------|-----|---------|-------------------|--------|
| `e5db40dc` | #1694 | ORJson | `bench_serving --num-prompt 500 --request-rate 30 --output-len 64` | SHORT outputs, HIGH RPS |

---

### Tier 9: Constrained Decoding

| Commit | PR | Subject | Benchmark Command | Server Args |
|--------|-----|---------|-------------------|-------------|
| `9c745d07` | #2056 | xgrammar | `bench_serving --num-prompt 100 --request-rate 4` | `--grammar-backend xgrammar` |
| `b77a02cd` | #1752 | Grammar backends | `bench_serving --num-prompt 100 --request-rate 4` | `--grammar-backend xgrammar` |

---

### Tier 10: Other Single-GPU Optimizations

| Commit | PR | Subject | Benchmark Command |
|--------|-----|---------|-------------------|
| `e3ec6bf4` | #6814 | FP8 quant | `bench_latency --batch-size 32 --input-len 1024 --output-len 128` |
| `9183c23e` | #2695 | Weights update | `bench_latency --batch-size 32 --input-len 512 --output-len 128` |
| `6a2941f4` | #625 | TP overhead | `bench_serving --num-prompt 200 --request-rate 10` |
| `e88dd482` | #6038 | VLM CI | `bench_serving --model llava-hf/llava-1.5-7b-hf --num-prompt 100` |
| `1bf1cf19` | #375 | fork(1) | `bench_serving --num-prompt 200 --request-rate 8` (reverted in #412) |

**Note:** All commands use `python3 -m sglang.` prefix and `--model meta-llama/Llama-3.1-8B-Instruct` unless specified.

---

### Multi-GPU Required (10)

| Commit | PR | Subject | Infrastructure |
|--------|-----|---------|----------------|
| `132dad87` | #6922 | PD transfer queue | Mooncake disaggregation |
| `148254d4` | #2705 | MoE reduce sum | MoE models (DeepSeek) |
| `2a413829` | #5955 | Triton MoE config | H20/B200 MoE |
| `2ed68d7a` | #7236 | PD batch transfer | 3P+9D nodes, **8-12x faster** |
| `6b231325` | #6649 | PD FastQueue | Mooncake |
| `6cb00c63` | #6761 | PD timeout | Mooncake |
| `880221bd` | #7968 | **REVERT** batch transfer | NVLink bugs |
| `9c088829` | #5786 | **REVERT** NCCL | Caused problems |
| `da47621c` | #7058 | MoE topk | torch.compile for MoE |
| `dd1012fc` | #6764 | PD tracker gc | O(n)->O(1) theoretical |

---

### Skip - Infrastructure Only (6)

| Commit | PR | Reason |
|--------|-----|--------|
| `6e2da515` | #6178 | time.perf_counter - timing accuracy only |
| `a191a0e4` | #6593 | Two-batch overlap - complex TBO |
| `a99801e0` | #8133 | TokenToKVPool - PD disaggregation |
| `73b13e69` | #7285 | DP attn scheduling - speculative decoding |
| `df7f61ee` | #6812 | Non-static dispatch rebalancing |
| `187b85b7` | #7393 | PD mem pool - Mooncake version bump |

---

## Priority Order for Single-GPU H100

| Priority | Commit | PR | Claimed | Command |
|----------|--------|-----|---------|---------|
| 1 | `79961afa` | #6077 | **21% FA3** | `bench_latency --batch-size 64 --input-len 2048` |
| 2 | `6fc17596` | #5945 | **71% pad** | `bench_latency --batch-size 64 --input-len 2048` |
| 3 | `2a754e57` | #579 | **2x prefill** | `bench_latency --input-len 8192 --output-len 1` |
| 4 | `9216b106` | #394 | **40% sched** | `bench_serving --num-prompt 300 --request-rate 16` |
| 5 | `b1e5a33a` | #6960 | **13% LoRA** | `bench_serving` with LoRA adapter |
| 6 | `3212c2ad` | #6003 | **16% VLM** | `bench_serving` with llava model |
| 7 | `c087ddd6` | #6627 | **10-15% kernel** | Explicit benchmark script |

---

## Model Requirements

| Model | Used In | GPU Memory |
|-------|---------|------------|
| meta-llama/Llama-3.1-8B-Instruct | Most commits | ~16GB |
| llava-hf/llava-1.5-7b-hf | VLM commits (#6003, #6038) | ~14GB |

All models fit on single H100 (80GB).

---

## Appendix: All 23 Multi-GPU Commits (from original extraction)

<details>
<summary>Click to expand</summary>

1. `25c83fff` - Performing Vocabulary Parallelism for LM Head across At
2. `27168308` - Speed up when having padding tokens in DeepEP (#6175)
3. `2f427491` - Fix topk inference performance reduce (#6474)
4. `31589e17` - Speed up when having padding tokens two-batch overlap (
5. `5239d795` - Speedup shared expert weight construction by avoid clon
6. `5e023301` - [perf] dsv3 bmm fallback to bf16 (#5662)
7. `6b7038ba` - Speedup warmup when DP > 1 (#4695)
8. `a73c4df4` - Add optimized native kernels in sgl-kernel (#5150)
9. `adca585b` - [DeepEP] Reduce routed scaling overhead (#5277)
10. `f0653886` - Expert distribution recording without overhead for EPLB

</details>

---

*Generated by SGLang 3-Way Benchmark Runner*
