# Commit Analysis Status (v2)

## Overview

This document tracks the status of 43 valid commits for Claude Code, Codex, and TRAE agents.

**Dataset:** `Inferencebench/claude-code-vllm-benchmarks`

**Excluded:**
- 9 commits with wrong perf command
- 1 commit (6ce01f30) with documented data issue

## Benchmark Modes and Metrics

Per CLAUDE.md guidelines:
- **Serving mode:** ttft_mean, tpot_mean, itl_mean (lower is better)
- **Standalone mode:** throughput, latency_avg (throughput higher is better, latency lower is better)
- **prefix_caching:** Uses throughput metrics (output toks/s) - higher is better

## Status Definitions

| Status | Description |
|--------|-------------|
| ✓ VALID | Has correct metrics for comparison per benchmark mode |
| WRONG METRIC | Has agent data but wrong metric (e.g., throughput instead of ttft for serving) |
| MISSING human_ttft | Agent has ttft but human_ttft missing from dataset |
| PATCH FAILURE | No agent metrics (patch generation or benchmark failure) |

## Complete Commit List (43 commits)

| Commit | Mode | Claude Code | Codex | TRAE (Sonnet) | TRAE (GPT) |
|--------|------|-------------|-------|---------------|------------|
| 7c01f706 | serving | PATCH FAILURE | PATCH FAILURE | PATCH FAILURE | PATCH FAILURE |
| ad8d696a | serving | PATCH FAILURE | PATCH FAILURE | PATCH FAILURE | PATCH FAILURE |
| d7740ea4 | standalone | PATCH FAILURE | PATCH FAILURE | PATCH FAILURE | PATCH FAILURE |
| 660470e5 | serving | PATCH FAILURE | PATCH FAILURE | PATCH FAILURE | PATCH FAILURE |
| 89a84b0b | serving | ✓ VALID | PATCH FAILURE | PATCH FAILURE | PATCH FAILURE |
| 3476ed08 | standalone | ✓ VALID | PATCH FAILURE | PATCH FAILURE | PATCH FAILURE |
| 19d98e0c | serving | ✓ VALID | MISSING human_ttft | MISSING human_ttft | PATCH FAILURE |
| fa63e710 | standalone | ✓ VALID | ✓ VALID | ✓ VALID | ✓ VALID |
| b690e348 | serving | ✓ VALID | PATCH FAILURE | PATCH FAILURE | PATCH FAILURE |
| 6e36f4fa | serving | ✓ VALID | PATCH FAILURE | PATCH FAILURE | PATCH FAILURE |
| fc7b8d1e | serving | PATCH FAILURE | PATCH FAILURE | PATCH FAILURE | PATCH FAILURE |
| 30172b49 | serving | ✓ VALID | ✓ VALID | ✓ VALID | PATCH FAILURE |
| ce6bf3a2 | None | PATCH FAILURE | PATCH FAILURE | PATCH FAILURE | PATCH FAILURE |
| 2deb029d | prefix_caching | ✓ VALID | PATCH FAILURE | PATCH FAILURE | PATCH FAILURE |
| 4c822298 | standalone | ✓ VALID | ✓ VALID | ✓ VALID | ✓ VALID |
| b55ed6ef | serving | ✓ VALID | ✓ VALID | ✓ VALID | ✓ VALID |
| 015069b0 | serving | ✓ VALID | PATCH FAILURE | PATCH FAILURE | PATCH FAILURE |
| e7b20426 | serving | PATCH FAILURE | ✓ VALID | ✓ VALID | PATCH FAILURE |
| ccf02fcb | None | PATCH FAILURE | PATCH FAILURE | PATCH FAILURE | PATCH FAILURE |
| 35fad35a | serving | PATCH FAILURE | PATCH FAILURE | PATCH FAILURE | PATCH FAILURE |
| a3223766 | serving | ✓ VALID | PATCH FAILURE | PATCH FAILURE | PATCH FAILURE |
| 99abb8b6 | serving | ✓ VALID | ✓ VALID | ✓ VALID | ✓ VALID |
| 310aca88 | standalone | ✓ VALID | PATCH FAILURE | PATCH FAILURE | PATCH FAILURE |
| 58eee5f2 | serving | ✓ VALID | ✓ VALID | ✓ VALID | ✓ VALID |
| 9f1710f1 | serving | ✓ VALID | ✓ VALID | ✓ VALID | PATCH FAILURE |
| 3a243095 | serving | PATCH FAILURE | PATCH FAILURE | PATCH FAILURE | PATCH FAILURE |
| 6dd94dbe | standalone | PATCH FAILURE | PATCH FAILURE | PATCH FAILURE | PATCH FAILURE |
| 8c1e77fb | standalone | PATCH FAILURE | ✓ VALID | ✓ VALID | ✓ VALID |
| bc7c4d20 | serving | ✓ VALID | PATCH FAILURE | PATCH FAILURE | PATCH FAILURE |
| e3580537 | serving | PATCH FAILURE | PATCH FAILURE | PATCH FAILURE | PATCH FAILURE |
| 9474e89b | standalone | ✓ VALID | PATCH FAILURE | PATCH FAILURE | PATCH FAILURE |
| 98f47f2a | standalone | ✓ VALID | ✓ VALID | ✓ VALID | ✓ VALID |
| fc542144 | serving | ✓ VALID | ✓ VALID | ✓ VALID | PATCH FAILURE |
| 3b61cb45 | standalone | PATCH FAILURE | ✓ VALID | ✓ VALID | ✓ VALID |
| 22d33bac | serving | ✓ VALID | ✓ VALID | ✓ VALID | ✓ VALID |
| 299ebb62 | serving | ✓ VALID | ✓ VALID | ✓ VALID | PATCH FAILURE |
| 9badee53 | serving | ✓ VALID | ✓ VALID | ✓ VALID | ✓ VALID |
| 9ed82e70 | serving | PATCH FAILURE | PATCH FAILURE | PATCH FAILURE | PATCH FAILURE |
| 6a417b86 | serving | ✓ VALID | PATCH FAILURE | PATCH FAILURE | PATCH FAILURE |
| 296f927f | serving | ✓ VALID | PATCH FAILURE | PATCH FAILURE | ✓ VALID |
| e206b543 | serving | ✓ VALID | ✓ VALID | ✓ VALID | PATCH FAILURE |
| 70b808fe | serving | ✓ VALID | PATCH FAILURE | PATCH FAILURE | ✓ VALID |
| fe66b347 | serving | ✓ VALID | PATCH FAILURE | PATCH FAILURE | PATCH FAILURE |

## Summary

| Status | Claude Code | Codex | TRAE (Sonnet) | TRAE (GPT) |
|--------|-------------|-------|---------------|------------|
| ✓ VALID | 28 | 16 | 16 | 12 |
| WRONG METRIC | 0 | 0 | 0 | 0 |
| MISSING human_ttft | 0 | 1 | 1 | 0 |
| PATCH FAILURE | 15 | 26 | 26 | 31 |
| **TOTAL** | **43** | **43** | **43** | **43** |

## Key Observations

1. **Claude Code** has NO remaining WRONG METRIC issues (all resolved):
   - `3476ed08` - was incorrectly marked; actually has valid latency_avg=184.16ms for standalone/latency benchmark

   **7 serving commits fixed on 2026-01-19:** 99abb8b6, 22d33bac, 9badee53, e206b543, 89a84b0b, 19d98e0c, 6e36f4fa. **4 revealed as PATCH FAILURE:** 7c01f706, fc7b8d1e, 3a243095, e3580537

   **1 prefix_caching commit fixed on 2026-01-19:** 2deb029d (throughput=5446.28 toks/s)

2. **Codex and TRAE (Sonnet)** have identical status counts (16 VALID, 26 PATCH FAILURE) because they succeed/fail on exactly the same commits. The actual metric values differ, but the success pattern is identical.

3. **TRAE (GPT)** has the highest patch failure rate (31/43 = 72.1%)

4. **All agents fail** on 12 commits: 7c01f706, ad8d696a, d7740ea4, 660470e5, fc7b8d1e, ce6bf3a2, ccf02fcb, 35fad35a, 3a243095, 6dd94dbe, e3580537, 9ed82e70

5. **Interesting cases where agents differ:**
   - e7b20426: Claude Code PATCH FAILURE, but Codex/TRAE-Sonnet ✓ VALID
   - 8c1e77fb, 3b61cb45: Claude Code PATCH FAILURE, but all others ✓ VALID
   - 296f927f, 70b808fe: Claude Code ✓ VALID, TRAE-GPT ✓ VALID, but Codex/TRAE-Sonnet PATCH FAILURE

## Agent Comparison

| Metric | Claude Code | Codex | TRAE (Sonnet) | TRAE (GPT) |
|--------|-------------|-------|---------------|------------|
| Valid for comparison | 28 (65.1%) | 16 (37.2%) | 16 (37.2%) | 12 (27.9%) |
| Patch success rate | 65.1% (28/43) | 39.5% (17/43) | 39.5% (17/43) | 27.9% (12/43) |
| Wrong metric issues | 0 | 0 | 0 | 0 |

---

## Session Findings (2026-01-19)

### Standalone Benchmark Analysis

Not all standalone benchmarks produce the same metrics. Analysis of the 43 commits reveals:

#### Latency-Only Benchmarks
| Commit | perf_command | human_latency_avg | human_throughput | Notes |
|--------|--------------|-------------------|------------------|-------|
| **3476ed08** | `benchmark_latency.py --model facebook/opt-125m` | 175.44 ms | N/A | **Only latency, no throughput** |

#### Latency Benchmarks with Both Metrics
Most `benchmark_latency.py` benchmarks produce both latency_avg AND throughput:
| Commit | human_latency_avg | human_throughput |
|--------|-------------------|------------------|
| 310aca88 | 4311.03 ms | 102.1 tok/s |
| 4c822298 | 1078.50 ms | 153.2 tok/s |
| 98f47f2a | 262.10 ms | 972.5 tok/s |
| fa63e710 | 3202.98 ms | 12775.4 tok/s |
| 3b61cb45 | 1706.33 ms | 9822.4 tok/s |
| 8c1e77fb | 1655.66 ms | 10206.3 tok/s |
| 6dd94dbe | 1022.52 ms | 255.9 tok/s |

#### Throughput-Only Benchmarks
| Commit | perf_command | human_throughput |
|--------|--------------|------------------|
| 9474e89b | `benchmark_throughput_cache.py` | 3086.41 tok/s |
| d7740ea4 | `benchmark_throughput.py` | 2011.32 tok/s |

#### Prefix Caching Benchmark
| Commit | perf_command | Metric Type |
|--------|--------------|-------------|
| 2deb029d | `benchmark_prefix_caching.py` | throughput (output toks/s) |

### Technical Fixes Applied

#### 1. 2deb029d (prefix_caching)
- **Problem:** `parent_commit: null` in mapping, wrong model name, missing parser
- **Fixes:**
  - Added parent_commit: `029c71de11bc3bcf84a1b3cf9d91e79ab6949799`
  - Changed model from `neuralmagic/Meta-Llama-3-8B-Instruct-FP8` (internal) to `RedHatAI/Meta-Llama-3-8B-Instruct-FP8` (public)
  - Removed `[--use-v2-block-manager]` brackets from perf_command
  - Added `parse_prefix_caching_metrics()` to runner to extract throughput from progress bar output
- **Result:** throughput=5446.28 toks/s (comparable to human's 5685.86 toks/s)

#### 2. 3476ed08 (standalone/latency)
- **Problem:** Incorrectly marked as WRONG METRIC
- **Analysis:** This is a latency-only benchmark (`benchmark_latency.py`). Claude Code has valid `agent_latency_avg=184.16ms` vs human's `175.44ms`.
- **Result:** Status corrected to ✓ VALID

#### 3. exports/full_results.jsonl Override
- **Problem:** Old perf_command with brackets in exports file was overriding benchmark_mode_mapping.json
- **Fix:** Updated exports/full_results.jsonl to match corrected mapping

### Agent Performance Comparison (3476ed08 - Latency Benchmark)

| Metric | Baseline | Human | Claude Code | Codex | TRAE |
|--------|----------|-------|-------------|-------|------|
| latency_avg (ms) | 169.20 | 175.44 | **184.16** | N/A | N/A |

**Note:** Lower latency is better. Claude Code's patch resulted in 5% higher latency than human optimization. Other agents failed to produce patches.

### Data Quality Issues Identified

1. **Mode=None commits** (ce6bf3a2, ccf02fcb): Benchmark mode not properly classified in dataset
2. **19d98e0c**: Missing `human_ttft` prevents comparison for Codex/TRAE-Sonnet
3. **HuggingFace naming inconsistency**: Some entries use `claude-code` (hyphen), others use `claude_code` (underscore)

### Files Modified This Session

1. `commit_v2.md` - Updated status tracking and analysis
2. `scripts/runners/run_3way_benchmarks.py` - Added `parse_prefix_caching_metrics()`
3. `data/mappings/benchmark_mode_mapping.json` - Fixed 2deb029d entry
4. `omniperf_results_3way_claude_code/exports/full_results.jsonl` - Fixed perf_command override
5. `omniperf_results_3way_claude_code/results/2deb029d_agent_result.json` - New benchmark result

### HuggingFace Update Required (Manual Upload)

**Status:** Prepared but requires write access to push

**Location:** `omniperf_results_3way_claude_code/hf_export/`

**Changes to push:**

| Commit | Update Type | Details |
|--------|-------------|---------|
| 99abb8b6 | Add serving metrics | ttft=656.64ms, tpot=30.98ms, itl=24.46ms |
| 22d33bac | Add serving metrics | ttft=651.12ms, tpot=30.51ms, itl=24.52ms |
| 9badee53 | Add serving metrics | ttft=174.68ms, tpot=9.85ms, itl=7.98ms |
| e206b543 | Add serving metrics | ttft=669.91ms, tpot=30.88ms, itl=24.56ms |
| 89a84b0b | Add serving metrics | ttft=356.05ms, tpot=23.73ms, itl=28.48ms |
| 19d98e0c | Add serving metrics | ttft=1099.58ms, tpot=35.95ms, itl=35.89ms |
| 6e36f4fa | Add serving metrics | ttft=1011.46ms, tpot=30.52ms, itl=33.53ms |
| 2deb029d | Fix benchmark_mode | standalone → prefix_caching |

**Files ready for upload:**
- `updated_dataset.parquet` (500 KB) - Full dataset
- `updated_rows.csv` (86 KB) - Just the 8 modified rows
- `UPLOAD_INSTRUCTIONS.md` - Upload guide

---

*Generated: 2026-01-19*
*Data source: HuggingFace `Inferencebench/claude-code-vllm-benchmarks`*
