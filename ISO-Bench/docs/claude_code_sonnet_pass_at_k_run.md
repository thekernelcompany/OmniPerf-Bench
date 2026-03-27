# Claude Code Sonnet 4.5 — Pass@K Run Summary

**Date**: 2026-03-27
**Branch**: `icml/rebuttal`
**Agent**: `claude_code` / **Model**: `sonnet` (Claude Sonnet 4.5)
**Config**: `bench_claude_code.yaml`

## Scope

Ran pass@k sampling (8 samples per task) for Claude Code Sonnet 4.5 on both vllm and sglang task sets.

### Tasks

| Repo | Planned tasks | Total tasks run | Samples per task | Total samples |
|------|--------------|-----------------|------------------|---------------|
| **vllm** | 39 (plan_iso.json) | 89 (incl. extras) | 8 (7 for core-0014) | 711 |
| **sglang** | 15 (sglang_plan_iso.json) | 15 | 8 | 120 |

## What was done

### 1. Identified incomplete prior runs

Prior batch runs (2026-03-26) had completed most tasks but left gaps:

**vllm** — 5 tasks missing entirely, 2 incomplete:
- Missing (0/8): `vllm_core-{0000, 0003, 0005, 0008, 0009}`
- Incomplete: `vllm_core-0011` (2/8), `vllm_core-0013` (7/8)

**sglang** — 8 tasks missing, 3 incomplete:
- Missing (0/8): `sglang_core-{0000, 0003, 0005, 0006, 0008, 0017, 0033, 0047}`
- Incomplete: `sglang_core-0071` (3/8), `sglang_core-0027` (1/8), `sglang_core-0019` (1/8)

### 2. Resumed remaining runs

Launched two parallel `run_pass_at_k.py` processes targeting only the missing/incomplete items:

```bash
# VLLM (7 items × 8 samples = 56 runs)
.venv/bin/python scripts/run_pass_at_k.py \
    --task tasks/vllm.yaml --plan state/plan_iso.json \
    --bench-cfg bench_claude_code.yaml --n 8 \
    --items vllm_core-0000 vllm_core-0003 vllm_core-0005 vllm_core-0008 \
            vllm_core-0009 vllm_core-0011 vllm_core-0013

# SGLang (11 items × 8 samples = 88 runs)
.venv/bin/python scripts/run_pass_at_k.py \
    --task tasks/sglang.yaml --plan state/sglang_plan_iso.json \
    --bench-cfg bench_claude_code.yaml --n 8 \
    --items sglang_core-0000 sglang_core-0003 sglang_core-0005 sglang_core-0006 \
            sglang_core-0008 sglang_core-0017 sglang_core-0019 sglang_core-0027 \
            sglang_core-0033 sglang_core-0047 sglang_core-0071
```

Both ran in local-only mode (no `--hf-repo`), so artifacts stayed in `state/runs/`.

### 3. Run results

| Repo | Samples queued | Completed | Failed | Duration |
|------|---------------|-----------|--------|----------|
| **vllm** | 56 | 56 | 0 | ~3h |
| **sglang** | 88 | 88 | 0 | ~4.75h |

Average pace: ~3 min/sample. Zero failures across 144 runs.

### 4. Deduplication and HF push

Since runs were local-only and partially-complete items were re-run with full n=8 (creating some duplicate task+sample pairs), a deduplication script was written (`scripts/push_local_to_hf.py`).

**Deduplication stats:**
- vllm: 924 total local runs → 711 unique (task_id, sample_index) pairs (dropped 213 dupes)
- sglang: 125 total local runs → 120 unique pairs (dropped 5 dupes)
- Strategy: keep earliest run per (task_id, sample_index)

**Pushed to**: `Inferencebench/pass-at-k-samples`
- `data/vllm/train.parquet` — 711 rows, 25 columns
- `data/sglang/train.parquet` — 120 rows, 25 columns

Post-push verification confirmed exact row counts and zero duplicates in both files.

Note: The HF repo also contains pre-existing per-sample shard files from other agents (codex_cli/gpt-5). Our consolidated parquet files are stored separately under `data/vllm/` and `data/sglang/` subdirectories and do not conflict.

## Files changed

- `ISO-Bench/scripts/push_local_to_hf.py` — New script for deduplicated local→HF push
- `ISO-Bench/state/runs/vllm/claude_code/sonnet/` — 56 new run directories
- `ISO-Bench/state/runs/sglan/claude_code/sonnet/` — 88 new run directories
