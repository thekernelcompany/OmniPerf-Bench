# Commit Analysis Status (v5)

## Overview

This document tracks the status of 39 valid commits for Claude Code, Codex, and TRAE agents.

**Dataset:** `Inferencebench/claude-code-vllm-benchmarks`

**Excluded (14 total):**
- 9 commits with wrong perf command
- 1 commit (`6ce01f30`) with corrupted Docker image
- 4 commits unbenchmarkable due to vLLM version incompatibility or non-standard benchmarks:
  - `7c01f706` - vLLM ~0.4.x, Unknown RoPE scaling type llama3
  - `3a243095` - vLLM ~0.3.x, Unknown RoPE scaling type llama3
  - `ce6bf3a2` - TPU-specific optimization, cannot benchmark on GPU
  - `ccf02fcb` - Mamba2-specific, uses lm_eval (accuracy test, not throughput/latency)

## Benchmark Modes and Metrics

Per CLAUDE.md guidelines:
- **Serving mode:** ttft_mean, tpot_mean, itl_mean (lower is better)
- **Standalone mode:** throughput OR latency_avg
- **prefix_caching:** Treated as standalone (throughput metric)

## Status Definitions

| Status | Description |
|--------|-------------|
| ✓ VALID | Has correct metrics for comparison per benchmark mode |
| PATCH FAILURE | No agent metrics (patch generation or benchmark failure) |

## Data Borrowing Rules

Human metrics are borrowed from `claude_code` rows when missing for other agents. Use `agent_name.isin(['claude-code', 'claude_code'])` for Claude Code.

## Exceptions and Special Cases

| Commit | Issue |
|--------|-------|
| `3476ed08` | Latency-only benchmark - uses `latency_avg` instead of `throughput` (expected) |

## Complete Commit List (39 commits)

| Commit | Mode | Claude Code | Codex | TRAE (Sonnet) | TRAE (GPT) |
|--------|------|-------------|-------|---------------|------------|
| ad8d696a | serving | PATCH FAILURE | PATCH FAILURE | PATCH FAILURE | PATCH FAILURE |
| d7740ea4 | standalone | PATCH FAILURE | PATCH FAILURE | PATCH FAILURE | PATCH FAILURE |
| 660470e5 | serving | PATCH FAILURE | PATCH FAILURE | PATCH FAILURE | PATCH FAILURE |
| 89a84b0b | serving | ✓ VALID | PATCH FAILURE | PATCH FAILURE | PATCH FAILURE |
| 3476ed08 | standalone | ✓ VALID | PATCH FAILURE | PATCH FAILURE | PATCH FAILURE |
| 19d98e0c | serving | ✓ VALID | ✓ VALID | ✓ VALID | PATCH FAILURE |
| fa63e710 | standalone | ✓ VALID | ✓ VALID | ✓ VALID | ✓ VALID |
| b690e348 | serving | ✓ VALID | PATCH FAILURE | PATCH FAILURE | PATCH FAILURE |
| 6e36f4fa | serving | ✓ VALID | PATCH FAILURE | PATCH FAILURE | PATCH FAILURE |
| fc7b8d1e | serving | ✓ VALID | PATCH FAILURE | PATCH FAILURE | PATCH FAILURE |
| 30172b49 | serving | ✓ VALID | ✓ VALID | ✓ VALID | PATCH FAILURE |
| 2deb029d | prefix_caching | ✓ VALID | PATCH FAILURE | PATCH FAILURE | PATCH FAILURE |
| 4c822298 | standalone | ✓ VALID | ✓ VALID | ✓ VALID | ✓ VALID |
| b55ed6ef | serving | ✓ VALID | ✓ VALID | ✓ VALID | ✓ VALID |
| 015069b0 | serving | ✓ VALID | PATCH FAILURE | PATCH FAILURE | PATCH FAILURE |
| e7b20426 | serving | PATCH FAILURE | ✓ VALID | ✓ VALID | PATCH FAILURE |
| 35fad35a | serving | PATCH FAILURE | PATCH FAILURE | PATCH FAILURE | PATCH FAILURE |
| a3223766 | serving | ✓ VALID | PATCH FAILURE | PATCH FAILURE | PATCH FAILURE |
| 99abb8b6 | serving | ✓ VALID | ✓ VALID | ✓ VALID | ✓ VALID |
| 310aca88 | standalone | ✓ VALID | PATCH FAILURE | PATCH FAILURE | PATCH FAILURE |
| 58eee5f2 | serving | ✓ VALID | ✓ VALID | ✓ VALID | ✓ VALID |
| 9f1710f1 | serving | ✓ VALID | ✓ VALID | ✓ VALID | PATCH FAILURE |
| 6dd94dbe | standalone | PATCH FAILURE | PATCH FAILURE | PATCH FAILURE | PATCH FAILURE |
| 8c1e77fb | standalone | PATCH FAILURE | ✓ VALID | ✓ VALID | ✓ VALID |
| bc7c4d20 | serving | ✓ VALID | PATCH FAILURE | PATCH FAILURE | PATCH FAILURE |
| e3580537 | serving | ✓ VALID | PATCH FAILURE | PATCH FAILURE | PATCH FAILURE |
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
| ✓ VALID | 30 | 17 | 17 | 12 |
| PATCH FAILURE | 9 | 22 | 22 | 27 |
| **TOTAL** | **39** | **39** | **39** | **39** |

## Key Observations

1. **Claude Code** has 30 valid commits for comparison - highest among all agents (76.9%)

2. **Codex and TRAE (Sonnet)** have identical status counts (17 VALID, 22 PATCH FAILURE) because they succeed/fail on exactly the same commits

3. **TRAE (GPT)** has the highest patch failure rate (27/39 = 69.2%)

4. **All agents fail** on 6 commits: ad8d696a, d7740ea4, 660470e5, 35fad35a, 6dd94dbe, 9ed82e70

5. **Interesting cases where agents differ:**
   - `e7b20426`: Claude Code PATCH FAILURE, but Codex/TRAE-Sonnet ✓ VALID
   - `8c1e77fb`, `3b61cb45`: Claude Code PATCH FAILURE, but all others ✓ VALID
   - `296f927f`, `70b808fe`: Only Claude Code and TRAE-GPT ✓ VALID

## Agent Comparison

| Metric | Claude Code | Codex | TRAE (Sonnet) | TRAE (GPT) |
|--------|-------------|-------|---------------|------------|
| ✓ VALID | 30 (76.9%) | 17 (43.6%) | 17 (43.6%) | 12 (30.8%) |
| Patch failures | 9 (23.1%) | 22 (56.4%) | 22 (56.4%) | 27 (69.2%) |

## Changes from v4

- **Excluded 4 additional commits** due to vLLM version incompatibility or non-standard benchmarks
- **prefix_caching treated as standalone** (throughput is valid metric)
- **No WRONG METRIC category** - all commits are either ✓ VALID or PATCH FAILURE
- Claude Code: 29 → **30** VALID (2deb029d now valid as standalone)
- Total commits: 43 → **39**

---

*Generated: 2026-01-19*
*Data source: HuggingFace `Inferencebench/claude-code-vllm-benchmarks`*
