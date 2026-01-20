# Commit Analysis Status (v4)

## Overview

This document tracks the status of 43 valid commits for Claude Code, Codex, and TRAE agents.

**Dataset:** `Inferencebench/claude-code-vllm-benchmarks`

**Excluded:**
- 9 commits with wrong perf command
- 1 commit (`6ce01f30`) with corrupted Docker image

## Benchmark Modes and Metrics

Per CLAUDE.md guidelines:
- **Serving mode:** ttft_mean, tpot_mean, itl_mean (lower is better)
- **Standalone mode:** throughput OR latency_avg
- **prefix_caching:** Treated as serving

## Status Definitions

| Status | Description |
|--------|-------------|
| ✓ VALID | Has correct metrics for comparison per benchmark mode |
| WRONG METRIC | Has agent data but wrong metric (e.g., throughput instead of ttft for serving) |
| PATCH FAILURE | No agent metrics (patch generation or benchmark failure) |

## Data Borrowing Rules

Human metrics are borrowed from `claude_code` rows when missing for other agents. Use `agent_name.isin(['claude-code', 'claude_code'])` for Claude Code.

## Exceptions and Special Cases

| Commit | Issue |
|--------|-------|
| `3476ed08` | Latency-only benchmark - uses `latency_avg` instead of `throughput` (expected) |
| `ce6bf3a2` | `benchmark_mode = None` - mode not classified |
| `ccf02fcb` | `benchmark_mode = None` - mode not classified |

## WRONG METRIC Commits (3)

| Commit | Mode | Issue |
|--------|------|-------|
| `7c01f706` | serving | agent has throughput (2109.36) instead of ttft |
| `2deb029d` | prefix_caching | agent has throughput (5711.09) instead of ttft |
| `3a243095` | serving | agent has throughput (2366.75) instead of ttft |

## Complete Commit List (43 commits)

| Commit | Mode | Claude Code | Codex | TRAE (Sonnet) | TRAE (GPT) |
|--------|------|-------------|-------|---------------|------------|
| 7c01f706 | serving | WRONG METRIC | PATCH FAILURE | PATCH FAILURE | PATCH FAILURE |
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
| ce6bf3a2 | None | PATCH FAILURE | PATCH FAILURE | PATCH FAILURE | PATCH FAILURE |
| 2deb029d | prefix_caching | WRONG METRIC | PATCH FAILURE | PATCH FAILURE | PATCH FAILURE |
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
| 3a243095 | serving | WRONG METRIC | PATCH FAILURE | PATCH FAILURE | PATCH FAILURE |
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
| ✓ VALID | 29 | 17 | 17 | 12 |
| WRONG METRIC | 3 | 0 | 0 | 0 |
| PATCH FAILURE | 11 | 26 | 26 | 31 |
| **TOTAL** | **43** | **43** | **43** | **43** |

## Key Observations

1. **Claude Code** has 29 valid commits for comparison - highest among all agents (67.4%)

2. **Claude Code** has 3 commits with WRONG METRIC: `7c01f706`, `2deb029d`, `3a243095`

3. **Codex and TRAE (Sonnet)** have identical status counts (17 VALID, 26 PATCH FAILURE) because they succeed/fail on exactly the same commits

4. **TRAE (GPT)** has the highest patch failure rate (31/43 = 72.1%)

5. **All agents fail** on 8 commits: ad8d696a, d7740ea4, 660470e5, ce6bf3a2, ccf02fcb, 35fad35a, 6dd94dbe, 9ed82e70

6. **Interesting cases where agents differ:**
   - `e7b20426`: Claude Code PATCH FAILURE, but Codex/TRAE-Sonnet ✓ VALID
   - `8c1e77fb`, `3b61cb45`: Claude Code PATCH FAILURE, but all others ✓ VALID
   - `296f927f`, `70b808fe`: Only Claude Code and TRAE-GPT ✓ VALID

## Agent Comparison

| Metric | Claude Code | Codex | TRAE (Sonnet) | TRAE (GPT) |
|--------|-------------|-------|---------------|------------|
| ✓ VALID | 29 (67.4%) | 17 (39.5%) | 17 (39.5%) | 12 (27.9%) |
| Has agent data | 32 (74.4%) | 17 (39.5%) | 17 (39.5%) | 12 (27.9%) |
| Patch failures | 11 (25.6%) | 26 (60.5%) | 26 (60.5%) | 31 (72.1%) |

**Note:** "Has agent data" = ✓ VALID + WRONG METRIC

---

*Generated: 2026-01-19*
*Data source: HuggingFace `Inferencebench/claude-code-vllm-benchmarks`*
