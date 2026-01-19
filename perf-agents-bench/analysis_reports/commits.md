# Commit Analysis Status

## Overview

This document tracks the status of 43 valid commits for both Claude Code and Codex agents.

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

| Commit | Mode | Claude Code | Codex |
|--------|------|-------------|-------|
| 7c01f706 | serving | WRONG METRIC | PATCH FAILURE |
| ad8d696a | serving | PATCH FAILURE | PATCH FAILURE |
| d7740ea4 | standalone | PATCH FAILURE | PATCH FAILURE |
| 660470e5 | serving | PATCH FAILURE | PATCH FAILURE |
| 89a84b0b | serving | WRONG METRIC | PATCH FAILURE |
| 3476ed08 | standalone | ✓ VALID | PATCH FAILURE |
| 19d98e0c | serving | WRONG METRIC | MISSING human_ttft |
| fa63e710 | standalone | ✓ VALID | ✓ VALID |
| b690e348 | serving | ✓ VALID | PATCH FAILURE |
| 6e36f4fa | serving | WRONG METRIC | PATCH FAILURE |
| fc7b8d1e | serving | WRONG METRIC | PATCH FAILURE |
| 30172b49 | serving | ✓ VALID | ✓ VALID |
| ce6bf3a2 | None | PATCH FAILURE | PATCH FAILURE |
| 2deb029d | standalone | ✓ VALID | PATCH FAILURE |
| 4c822298 | standalone | ✓ VALID | ✓ VALID |
| b55ed6ef | serving | ✓ VALID | ✓ VALID |
| 015069b0 | serving | ✓ VALID | PATCH FAILURE |
| e7b20426 | serving | PATCH FAILURE | ✓ VALID |
| ccf02fcb | None | PATCH FAILURE | PATCH FAILURE |
| 35fad35a | serving | PATCH FAILURE | PATCH FAILURE |
| a3223766 | serving | ✓ VALID | PATCH FAILURE |
| 99abb8b6 | serving | WRONG METRIC | ✓ VALID |
| 310aca88 | standalone | ✓ VALID | PATCH FAILURE |
| 58eee5f2 | serving | ✓ VALID | ✓ VALID |
| 9f1710f1 | serving | ✓ VALID | ✓ VALID |
| 3a243095 | serving | WRONG METRIC | PATCH FAILURE |
| 6dd94dbe | standalone | PATCH FAILURE | PATCH FAILURE |
| 8c1e77fb | standalone | PATCH FAILURE | ✓ VALID |
| bc7c4d20 | serving | ✓ VALID | PATCH FAILURE |
| e3580537 | serving | WRONG METRIC | PATCH FAILURE |
| 9474e89b | standalone | ✓ VALID | PATCH FAILURE |
| 98f47f2a | standalone | ✓ VALID | ✓ VALID |
| fc542144 | serving | ✓ VALID | ✓ VALID |
| 3b61cb45 | standalone | PATCH FAILURE | ✓ VALID |
| 22d33bac | serving | WRONG METRIC | ✓ VALID |
| 299ebb62 | serving | ✓ VALID | ✓ VALID |
| 9badee53 | serving | WRONG METRIC | ✓ VALID |
| 9ed82e70 | serving | PATCH FAILURE | PATCH FAILURE |
| 6a417b86 | serving | ✓ VALID | PATCH FAILURE |
| 296f927f | serving | ✓ VALID | PATCH FAILURE |
| e206b543 | serving | WRONG METRIC | ✓ VALID |
| 70b808fe | serving | ✓ VALID | PATCH FAILURE |
| fe66b347 | serving | ✓ VALID | PATCH FAILURE |

## Summary

| Status | Claude Code | Codex |
|--------|-------------|-------|
| ✓ VALID | 21 | 16 |
| WRONG METRIC | 11 | 0 |
| MISSING human_ttft | 0 | 1 |
| PATCH FAILURE | 11 | 26 |
| **TOTAL** | **43** | **43** |

## Key Observations

1. **Claude Code** has 11 serving mode commits with WRONG METRIC - these have throughput data but not ttft (the correct metric for serving mode)

2. **Codex** has higher patch failure rate (26 vs 11) but no wrong metric issues

3. **Only 1 commit** (19d98e0c) has missing human data for Codex

4. **Both agents fail** on 10 commits (ad8d696a, d7740ea4, 660470e5, ce6bf3a2, ccf02fcb, 35fad35a, 6dd94dbe, 9ed82e70 + partial overlaps)

5. **Interesting cases:**
   - e7b20426: Claude Code PATCH FAILURE, Codex ✓ VALID
   - 8c1e77fb: Claude Code PATCH FAILURE, Codex ✓ VALID
   - 3b61cb45: Claude Code PATCH FAILURE, Codex ✓ VALID
   - 99abb8b6, 22d33bac, 9badee53, e206b543: Claude Code WRONG METRIC, Codex ✓ VALID

---

*Generated: 2026-01-19*
*Data source: HuggingFace `Inferencebench/claude-code-vllm-benchmarks`*
