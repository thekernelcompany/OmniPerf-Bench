# Pass@k Data Reconciliation & Benchmark Status

**Date:** 2026-03-30
**HF patches repo:** `Inferencebench/pass-at-k-samples` (4329 per-sample shards)
**HF results repo:** `Inferencebench/pass-at-k-benchmark-results`
**ISO-Bench scope:** 54 tasks (39 vllm + 15 sglang) x 8 samples = 432 per agent

---

## 1. HF Shard Inventory

The HF repo has per-sample parquet shards (the real data) plus stale consolidated files. `load_dataset()` merges all files and gives wrong numbers — must read shards directly via HF API.

**IMPORTANT:** Item IDs in HF shards use a different numbering scheme than `plan_iso.json`. Must match by `human_commit` hash, not item_id.

**IMPORTANT:** `claude_code/sonnet` and `claude_code/claude_model-claude-sonnet-4-5` are the SAME agent (Claude Code Sonnet 4.5) — different collection runs with different model name strings. They are stored in different formats:
- `claude_code/sonnet`: 538 per-sample shards (old format `item_sN_ts.parquet`)
- `claude_code/claude_model-claude-sonnet-4-5`: 18 standard shards (`train-NNNNN.parquet`)
- Together they cover all 54 ISO-Bench tasks with overlap on some

### Per-agent shard counts (vLLM)

| Agent | Model | Shards | Unique (item,sample) |
|-------|-------|--------|---------------------|
| claude_code | sonnet | 367 | 295 |
| codex_cli | gpt-5 | 2144 | 675 |
| trae | gpt-5 | 21 | 21 |

### Per-agent shard counts (SGLang)

| Agent | Model | Shards | Unique (item,sample) |
|-------|-------|--------|---------------------|
| claude_code | sonnet | 165 | 138 |
| codex_cli | gpt-5 | 1577 | 648 |
| trae | gpt-5 | 53 | 53 |

---

## 2. Deduplicated GPU Benchmark Results (ISO-Bench 54-task scope only)

All benchmarks on NVIDIA H100 80GB PCIe. vLLM only (SGLang not yet benchmarked).

### Claude Code (Sonnet 4.5) — merging sonnet + claude-sonnet-4-5

| Metric | Count | /432 target |
|--------|------:|------------:|
| Attempted | 234 | 54% |
| **Success** | **121** | **28%** |
| Benchmark failed | 86 | — |
| Image not found | 27 | — |
| Not attempted | 198 | 46% |
| **Tasks w/ success** | **16/54** | — |
| **Effective rate** | **121/207** | **58%** |

### Codex CLI (GPT-5)

| Metric | Count | /432 target |
|--------|------:|------------:|
| Attempted | 311 | 72% |
| **Success** | **133** | **31%** |
| Benchmark failed | 81 | — |
| Image not found | 81 | — |
| Empty patch | 16 | — |
| Not attempted | 121 | 28% |
| **Tasks w/ success** | **18/54** | — |
| **Effective rate** | **133/214** | **62%** |

### Combined unique successful benchmarks: 278 across 24/54 tasks

---

## 3. What's NOT benchmarked

### A. SGLang — 0 benchmarks (15 tasks, 120 samples per agent)

**PYTHONPATH overlay approach DOES NOT WORK.** Tested 2026-03-30: the worktree's Python code (at baseline commit) has cross-dependency version mismatches with the system-installed packages (e.g., worktree imports `triton.runtime.cache.default_cache_dir` which doesn't exist in the installed triton). This is the same ABI/API mismatch problem documented in `hero_sglang_benchmark.py`.

SGLang benchmarking requires one of:
1. **Modal** — wheel-based build per commit (existing: `sglang_modal_benchmark.py`)
2. **Docker images per commit** — build from `tools/build_sglang_images.py`
3. **Full isolated venv per commit** — `pip install sglang` in a fresh venv per baseline

All 15 SGLang tasks have `perf_command` and model info in `Lossfunk/ISO-Bench` sglang split. Mapping file created: `data/mappings/sglang_benchmark_mode_mapping.json`.

### B. vLLM benchmark failures — root cause analysis

| Task | Samples | Root Cause | Fixable? |
|------|---------|-----------|----------|
| vllm_core-0017 | 16 | `ModuleNotFoundError: lark` | Yes — `pip install lark` |
| vllm_core-0018 | 16 | `$VLLM_PYTHON` empty | Yes — fix Python detection |
| vllm_core-0014 | 2 | Same `$VLLM_PYTHON` issue | Yes |
| vllm_core-0022 | 16 | Gated Llama-3.2 model (403) | Yes — HF token access |
| vllm_core-0004 | 8 | `MegaBlocks not found` | Yes — `pip install megablocks` |
| vllm_core-0010 | 12 | CUDA OOM (Mixtral) | No — needs multi-GPU |
| vllm_core-0034 | 16 | Internal Server Error | No — bad patches |
| vllm_core-0041 | 16 | All patches crash vLLM | No — bad patches |
| vllm_core-0025 | 1 | Import error from patch | No — bad patch |

### C. Missing Docker images (13 truly missing)

These baseline images don't exist on `shikhar481/vllm_fixed_human_images`:
baseline-88f6ba3281f7, baseline-a732900efc4e, baseline-d3ea50113c08, baseline-0e74d797ce86, baseline-6d917d0eebd0, baseline-89ac266b262f, baseline-51c31bc10ca7, baseline-f67e9e9f221e, baseline-802329dee9e5, baseline-7a7929abe8e2, baseline-b56b6ca0d650, baseline-8936316d587c, baseline-4a18fd14ba4a

### D. Missing benchmark mapping

3 tasks have commits not in `benchmark_mode_mapping.json`: vllm_core-0016, vllm_core-0023, vllm_core-0024

---

## 4. Timeline

| Time | Event |
|------|-------|
| 2026-03-29 03:30 | claude_code/claude-sonnet-4-5 run (84 vLLM samples, 69 success) |
| 2026-03-29 10:50 | Combined run: claude_code/sonnet + codex_cli/gpt-5 (877 vLLM samples) |
| 2026-03-29 17:17 | First run complete (221 success, disk full killed 35 tasks) |
| 2026-03-29 19:00 | Targeted retry for 35 disk-full tasks (+47 success) |
| 2026-03-29 21:35 | All vLLM runs complete |
| 2026-03-30 | SGLang benchmark — PENDING (worktree-based, no Docker needed) |

---

## 5. Infrastructure

### vLLM benchmarking (Docker-based)
- Docker images: `shikhar481/vllm_fixed_human_images` (baseline-* and human-*)
- Persistent container per task, `docker exec` for each sample
- Model cache: `/ephemeral/huggingface_cache`
- Script: `scripts/runners/run_pass_at_k_benchmarks.py`

### SGLang benchmarking (worktree-based, no Docker)
- Git submodule: `sglang/` → `https://github.com/sgl-project/sglang.git`
- Approach: `git worktree` at baseline commit + PYTHONPATH overlay for patches
- Server: `python -m sglang.launch_server`
- Benchmark: `sglang.bench_serving` (from `Lossfunk/ISO-Bench` perf_command)
- Existing support: `src/eval/native_benchmark_runner.py` (ServerManager + worktree)

## 6. How to run

```bash
# vLLM benchmarks (Docker-based)
python3 scripts/runners/run_pass_at_k_benchmarks.py \
    --agents claude_code/sonnet codex_cli/gpt-5 \
    --timeout 900 --resume

# SGLang benchmarks (worktree-based) — TODO
# Needs script update to support worktree approach
```
