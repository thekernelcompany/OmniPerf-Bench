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
- **prefix_caching:** Treated as serving

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
| 7c01f706 | serving | WRONG METRIC | PATCH FAILURE | PATCH FAILURE | PATCH FAILURE |
| ad8d696a | serving | PATCH FAILURE | PATCH FAILURE | PATCH FAILURE | PATCH FAILURE |
| d7740ea4 | standalone | PATCH FAILURE | PATCH FAILURE | PATCH FAILURE | PATCH FAILURE |
| 660470e5 | serving | PATCH FAILURE | PATCH FAILURE | PATCH FAILURE | PATCH FAILURE |
| 89a84b0b | serving | WRONG METRIC | PATCH FAILURE | PATCH FAILURE | PATCH FAILURE |
| 3476ed08 | standalone | WRONG METRIC | PATCH FAILURE | PATCH FAILURE | PATCH FAILURE |
| 19d98e0c | serving | WRONG METRIC | MISSING human_ttft | MISSING human_ttft | PATCH FAILURE |
| fa63e710 | standalone | ✓ VALID | ✓ VALID | ✓ VALID | ✓ VALID |
| b690e348 | serving | ✓ VALID | PATCH FAILURE | PATCH FAILURE | PATCH FAILURE |
| 6e36f4fa | serving | WRONG METRIC | PATCH FAILURE | PATCH FAILURE | PATCH FAILURE |
| fc7b8d1e | serving | WRONG METRIC | PATCH FAILURE | PATCH FAILURE | PATCH FAILURE |
| 30172b49 | serving | ✓ VALID | ✓ VALID | ✓ VALID | PATCH FAILURE |
| ce6bf3a2 | None | PATCH FAILURE | PATCH FAILURE | PATCH FAILURE | PATCH FAILURE |
| 2deb029d | standalone | ✓ VALID | PATCH FAILURE | PATCH FAILURE | PATCH FAILURE |
| 4c822298 | standalone | ✓ VALID | ✓ VALID | ✓ VALID | ✓ VALID |
| b55ed6ef | serving | ✓ VALID | ✓ VALID | ✓ VALID | ✓ VALID |
| 015069b0 | serving | ✓ VALID | PATCH FAILURE | PATCH FAILURE | PATCH FAILURE |
| e7b20426 | serving | PATCH FAILURE | ✓ VALID | ✓ VALID | PATCH FAILURE |
| ccf02fcb | None | PATCH FAILURE | PATCH FAILURE | PATCH FAILURE | PATCH FAILURE |
| 35fad35a | serving | PATCH FAILURE | PATCH FAILURE | PATCH FAILURE | PATCH FAILURE |
| a3223766 | serving | ✓ VALID | PATCH FAILURE | PATCH FAILURE | PATCH FAILURE |
| 99abb8b6 | serving | WRONG METRIC | ✓ VALID | ✓ VALID | ✓ VALID |
| 310aca88 | standalone | ✓ VALID | PATCH FAILURE | PATCH FAILURE | PATCH FAILURE |
| 58eee5f2 | serving | ✓ VALID | ✓ VALID | ✓ VALID | ✓ VALID |
| 9f1710f1 | serving | ✓ VALID | ✓ VALID | ✓ VALID | PATCH FAILURE |
| 3a243095 | serving | WRONG METRIC | PATCH FAILURE | PATCH FAILURE | PATCH FAILURE |
| 6dd94dbe | standalone | PATCH FAILURE | PATCH FAILURE | PATCH FAILURE | PATCH FAILURE |
| 8c1e77fb | standalone | PATCH FAILURE | ✓ VALID | ✓ VALID | ✓ VALID |
| bc7c4d20 | serving | ✓ VALID | PATCH FAILURE | PATCH FAILURE | PATCH FAILURE |
| e3580537 | serving | WRONG METRIC | PATCH FAILURE | PATCH FAILURE | PATCH FAILURE |
| 9474e89b | standalone | ✓ VALID | PATCH FAILURE | PATCH FAILURE | PATCH FAILURE |
| 98f47f2a | standalone | ✓ VALID | ✓ VALID | ✓ VALID | ✓ VALID |
| fc542144 | serving | ✓ VALID | ✓ VALID | ✓ VALID | PATCH FAILURE |
| 3b61cb45 | standalone | PATCH FAILURE | ✓ VALID | ✓ VALID | ✓ VALID |
| 22d33bac | serving | WRONG METRIC | ✓ VALID | ✓ VALID | ✓ VALID |
| 299ebb62 | serving | ✓ VALID | ✓ VALID | ✓ VALID | PATCH FAILURE |
| 9badee53 | serving | WRONG METRIC | ✓ VALID | ✓ VALID | ✓ VALID |
| 9ed82e70 | serving | PATCH FAILURE | PATCH FAILURE | PATCH FAILURE | PATCH FAILURE |
| 6a417b86 | serving | ✓ VALID | PATCH FAILURE | PATCH FAILURE | PATCH FAILURE |
| 296f927f | serving | ✓ VALID | PATCH FAILURE | PATCH FAILURE | ✓ VALID |
| e206b543 | serving | WRONG METRIC | ✓ VALID | ✓ VALID | PATCH FAILURE |
| 70b808fe | serving | ✓ VALID | PATCH FAILURE | PATCH FAILURE | ✓ VALID |
| fe66b347 | serving | ✓ VALID | PATCH FAILURE | PATCH FAILURE | PATCH FAILURE |

## Summary

| Status | Claude Code | Codex | TRAE (Sonnet) | TRAE (GPT) |
|--------|-------------|-------|---------------|------------|
| ✓ VALID | 20 | 16 | 16 | 12 |
| WRONG METRIC | 12 | 0 | 0 | 0 |
| MISSING human_ttft | 0 | 1 | 1 | 0 |
| PATCH FAILURE | 11 | 26 | 26 | 31 |
| **TOTAL** | **43** | **43** | **43** | **43** |

## Key Observations

1. **Claude Code** has 12 commits with WRONG METRIC - these have throughput/latency data but not ttft (the correct metric for serving mode)

2. **Codex and TRAE (Sonnet)** have identical status counts (16 VALID, 26 PATCH FAILURE) because they succeed/fail on exactly the same commits. The actual metric values differ, but the success pattern is identical.

3. **TRAE (GPT)** has the highest patch failure rate (31/43 = 72.1%)

4. **All agents fail** on 8 commits: ad8d696a, d7740ea4, 660470e5, ce6bf3a2, ccf02fcb, 35fad35a, 6dd94dbe, 9ed82e70

5. **Interesting cases where agents differ:**
   - e7b20426: Claude Code PATCH FAILURE, but Codex/TRAE-Sonnet ✓ VALID
   - 8c1e77fb, 3b61cb45: Claude Code PATCH FAILURE, but all others ✓ VALID
   - 296f927f, 70b808fe: Claude Code ✓ VALID, TRAE-GPT ✓ VALID, but Codex/TRAE-Sonnet PATCH FAILURE
   - Multiple commits: Claude Code WRONG METRIC, but Codex/TRAE ✓ VALID

## Agent Comparison

| Metric | Claude Code | Codex | TRAE (Sonnet) | TRAE (GPT) |
|--------|-------------|-------|---------------|------------|
| Valid for comparison | 20 (46.5%) | 16 (37.2%) | 16 (37.2%) | 12 (27.9%) |
| Patch success rate | 74.4% (32/43) | 39.5% (17/43) | 39.5% (17/43) | 27.9% (12/43) |
| Wrong metric issues | 12 | 0 | 0 | 0 |

**Note:** Claude Code's "patch success rate" includes WRONG METRIC cases where a patch ran but produced the wrong metric type.

---

*Generated: 2026-01-19*
*Data source: HuggingFace `Inferencebench/claude-code-vllm-benchmarks`*
