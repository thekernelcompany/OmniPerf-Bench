# Pass@k GPU Benchmark Status

**Last updated:** 2026-03-29
**Machine:** H100 80GB PCIe, Docker + nvidia-container-toolkit
**Script:** `scripts/runners/run_pass_at_k_benchmarks.py`
**Results HF repo:** `Inferencebench/pass-at-k-benchmark-results`
**Patches HF repo:** `Inferencebench/pass-at-k-samples`

---

## HF Data Reality

The `Inferencebench/pass-at-k-samples` repo has two types of parquet files:

1. **Consolidated files** (`data/vllm/train.parquet`, `data/sglang/train.parquet`) — contain Bedrock-auth failure rows from early collection. The `datasets.load_dataset()` call picks these up and shows ~98% empty patches. **These are stale/misleading.**

2. **Per-sample shards** (`data/vllm_core-0003_s0_1774698610.parquet`, etc.) — 4000+ individual files, one per collection run. These contain the real data with valid patches. **The benchmark script reads these directly via HuggingFace API, bypassing `load_dataset`.**

---

## vLLM Benchmarkable Samples (from per-sample shards)

| Agent | Model | Total Shards | With Patch | Empty | Items | Items w/ Patches |
|-------|-------|-------------|-----------|-------|-------|-----------------|
| claude_code | sonnet | 211 | 187 | 24 | 38 | 26 |
| codex_cli | gpt-5 | 675 | 228 | 447 | 85 | 29 |
| claude_code | claude_model-claude-sonnet-4-5 | 84 | 84 | 0 | 13 | 13 |
| **Total** | | **970** | **499** | **471** | | |

### claude_code/claude_model-claude-sonnet-4-5 (COMPLETE)

All 84 vLLM samples benchmarked. 69 success, 7 benchmark_failed, 8 image_not_found.

| Task | OK | Fail | Avg Metric | Notes |
|------|---:|-----:|-----------|-------|
| vllm_core-0001 | 0 | 6 | — | Local model path in perf_command |
| vllm_core-0002 | 0 | 8 | — | Missing Docker image (baseline-0e74d797ce86) |
| vllm_core-0003 | 7 | 1 | 2124.8 tok/s | 1 server crash |
| vllm_core-0004 | 8 | 0 | 2891.3 tok/s | |
| vllm_core-0005 | 8 | 0 | 2885.0 tok/s | |
| vllm_core-0006 | 8 | 0 | 1177.4 (latency ms) | |
| vllm_core-0007 | 8 | 0 | 8160.0 tok/s | |
| vllm_core-0008 | 8 | 0 | 2889.1 tok/s | |
| vllm_core-0009 | 8 | 0 | 6099.7 tok/s | |
| vllm_core-0010 | 4 | 0 | 5447.6 tok/s | Only 4 samples on HF |
| vllm_core-0011 | 1 | 0 | 5396.9 tok/s | Only 1 sample on HF |
| vllm_core-0012 | 8 | 0 | 2039.1 tok/s | |
| vllm_core-0013 | 1 | 0 | 2884.8 tok/s | Only 1 sample on HF |

### claude_code/sonnet — 187 samples with patches, ~9 benchmarked

These were invisible to `load_dataset()` because the consolidated train.parquet overwrote them.
The shard-based loader found 187 valid patches across 26 vLLM items.

### codex_cli/gpt-5 — 228 samples with patches, ~1 benchmarked

Most codex runs failed due to usage limits (see `codex_cli_pass_at_k_status.md`).
228 shards have actual patches. Many tasks overlap with claude_code — shared container setup saves time.

---

## SGLang Status

| Agent | Model | Total Shards | With Patch | Notes |
|-------|-------|-------------|-----------|-------|
| claude_code | sonnet | ~80 | ~80 | Not benchmarked — SGLang needs Modal (ABI mismatch with Docker overlay) |
| claude_code | claude_model-claude-sonnet-4-5 | 90 | 90 | Not benchmarked — same reason |

**SGLang cannot be benchmarked with local Docker.** The multi-package dependency chain (sglang + sgl_kernel + flashinfer) requires ABI-compatible builds. Applying a Python patch to an existing Docker image crashes with ImportError/undefined symbol. Modal wheel-based approach is the only viable path.

---

## Missing Docker Images (vLLM)

11 tasks failed due to missing baseline Docker images on `shikhar481/vllm_fixed_human_images`:

| Task | Parent Commit | Image Tag | Agents Affected |
|------|--------------|-----------|-----------------|
| vllm_core-0001 | 88f6ba3281f7 | baseline-88f6ba3281f7 | both |
| vllm_core-0015 | a732900efc4e | baseline-a732900efc4e | both (70B, needs tp=4) |
| vllm_core-0029 | d3ea50113c08 | baseline-d3ea50113c08 | both (Neuron worker) |
| vllm_core-0033 | 0e74d797ce86 | baseline-0e74d797ce86 | both (buildable) |
| vllm_core-0038 | bd43973522ea | baseline-bd43973522ea | codex only |
| vllm_core-0039 | 5b8a1fde8422 | baseline-5b8a1fde8422 | codex only |
| vllm_core-0040 | 6d917d0eebd0 | baseline-6d917d0eebd0 | codex only |
| vllm_core-0041 | 084a01fd3544 | baseline-084a01fd3544 | both |
| vllm_core-0042 | 76b494444fd8 | baseline-76b494444fd8 | codex only |
| vllm_core-0043 | 89ac266b262f | baseline-89ac266b262f | codex only |
| vllm_core-0044 | 0fca3cdcf265 | baseline-0fca3cdcf265 | codex only |

## Benchmark Failures — Root Cause Analysis

87 samples across 8 tasks failed during actual benchmarking. Detailed analysis:

### Fixable (56 samples)

| Task | Samples | Root Cause | Fix |
|------|---------|-----------|-----|
| **vllm_core-0017** | 16 (both agents) | `ModuleNotFoundError: No module named 'lark'` | Add `pip install lark` to container setup |
| **vllm_core-0018** | 16 (both agents) | `$VLLM_PYTHON` empty — `/tmp/run_benchmark.sh: line 34: -m: command not found` | Fix `bench_env.sh` — Python not found in this container's layout |
| **vllm_core-0014** | 2 (claude only) | Same `$VLLM_PYTHON` empty issue | Same fix as 0018 |
| **vllm_core-0022** | 16 (both agents) | `403 Access to gated model meta-llama/Llama-3.2-1B-Instruct` | Need HF token with Llama-3.2 access grant |
| **vllm_core-0004** | 8 (codex only) | `ImportError: MegaBlocks not found` for Mixtral MoE model | Add `pip install megablocks` to container setup |

### Not fixable on single H100 (12 samples)

| Task | Samples | Root Cause |
|------|---------|-----------|
| **vllm_core-0010** | 12 (both agents) | `torch.cuda.OutOfMemoryError` — Mixtral model too large for single H100 80GB |

### Genuinely bad patches (17 samples)

| Task | Samples | Root Cause |
|------|---------|-----------|
| **vllm_core-0034** | 16 (both agents) | Server starts but returns `Internal Server Error` — agent patches broke model serving |
| **vllm_core-0025** | 1 (codex only) | `ImportError: cannot import name 'MistralTokenizer'` — patch broke import chain |

## Tasks with no benchmark mapping (29 samples)

3 tasks have commits not present in `benchmark_mode_mapping.json` (no perf_command known):

| Task | Samples |
|------|---------|
| vllm_core-0016 | 13 (both agents) |
| vllm_core-0023 | 8 (codex only) |
| vllm_core-0024 | 8 (codex only) |

---

## Benchmark Infrastructure

- **Docker images:** `shikhar481/vllm_fixed_human_images` (92 tags: baseline-* and human-*)
- **Model cache:** `/ephemeral/huggingface_cache` (mounted into containers)
- **Docker data root:** `/ephemeral/docker` (700GB ephemeral disk)
- **Results:** `results/pass_at_k_benchmarks/agent_benchmark_results.jsonl` (append-only, crash-safe)
- **HF results:** `Inferencebench/pass-at-k-benchmark-results`

## How to Run

```bash
# Both agents in one pass (shared container per task)
python3 scripts/runners/run_pass_at_k_benchmarks.py \
    --agents claude_code/sonnet codex_cli/gpt-5 \
    --timeout 900 --resume

# Single agent
python3 scripts/runners/run_pass_at_k_benchmarks.py \
    --agents claude_code/claude_model-claude-sonnet-4-5 \
    --timeout 900 --resume

# Specific tasks
python3 scripts/runners/run_pass_at_k_benchmarks.py \
    --agents claude_code/sonnet codex_cli/gpt-5 \
    --items vllm_core-0000 vllm_core-0003 \
    --timeout 900 --resume

# Dry run
python3 scripts/runners/run_pass_at_k_benchmarks.py \
    --agents claude_code/sonnet codex_cli/gpt-5 \
    --timeout 900 --resume --dry-run
```
