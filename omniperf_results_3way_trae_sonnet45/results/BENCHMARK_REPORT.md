# OmniPerf Benchmark Results - Agent Comparison Report

**Date:** January 15, 2026
**Benchmark System:** vLLM Performance Optimization Tasks

---

## Executive Summary

This report documents benchmark results for three AI agents attempting vLLM performance optimization tasks. The results reveal **significant quality issues** across all agents, with success rates ranging from 11-35%.

### Key Finding: Low Success Rates Across All Agents

| Agent | Total Commits | Successful | Success Rate | Serving | Latency | Throughput |
|-------|---------------|------------|--------------|---------|---------|------------|
| **TRAE Sonnet 4.5** | 96 | 34 | **35.4%** | 27 | 6 | 1 |
| **Codex GPT-5** | 96 | 30 | **31.2%** | 24 | 6 | 0 |
| **TRAE GPT-5** | 42* | 11 | **26.2%** | 5 | 5 | 1 |

*TRAE GPT-5 has incomplete coverage (42/96 commits benchmarked)

---

## Critical Analysis

### 1. Fundamental Data Quality Issues

**6 commits have missing `parent_commit` in the benchmark mapping:**
- baeded25, ac45c44d, 526de822, 3127e975, 2deb029d, 4fb56914

These commits **cannot be benchmarked** because the baseline docker image tag requires the parent commit hash. This is a fundamental limitation in the benchmark infrastructure.

### 2. High Failure Rates

**Codex GPT-5 Failure Breakdown (66 failures):**
| Reason | Count | % of Failures |
|--------|-------|---------------|
| Server crashed after patch | 30 | 45.5% |
| No metrics in output | 29 | 43.9% |
| Missing parent_commit | 6 | 9.1% |
| Timeout (900s) | 1 | 1.5% |

**TRAE Sonnet 4.5 Failure Breakdown (62 failures):**
| Reason | Count | % of Failures |
|--------|-------|---------------|
| No metrics in output | 28 | 45.2% |
| Server crashed after patch | 28 | 45.2% |
| Missing parent_commit | 6 | 9.7% |

**Root Causes:**
1. **Server crashes (45%)**: Agent patches introduce syntax errors, runtime exceptions, or break vLLM server startup
2. **No metrics (44%)**: Server runs but agent changes break the benchmark measurement path
3. **Data issues (10%)**: Missing commit metadata prevents benchmarking

### 3. Incomplete TRAE GPT-5 Coverage

TRAE GPT-5 only produced 42 results vs 96 expected. Investigation shows:
- TRAE GPT-5 patch directory: `trae/gpt-5` contains 303 patch files
- However, only 42 patches map to the 96 benchmark commits
- **54 commits (56%) have no TRAE GPT-5 patches available**

This incomplete coverage makes cross-agent comparison unreliable.

### 4. Benchmark Type Distribution

**Most successful benchmarks are "serving" type** (TTFT/TPOT/ITL metrics):

| Benchmark Type | Total Success | Codex | TRAE S4.5 | TRAE G5 |
|----------------|---------------|-------|-----------|---------|
| Serving | 56 | 24 | 27 | 5 |
| Latency | 17 | 6 | 6 | 5 |
| Throughput | 2 | 0 | 1 | 1 |
| **Total** | **75** | **30** | **34** | **11** |

"Throughput" benchmarks have near-zero success, suggesting systematic issues with throughput measurement or agent patches for throughput-focused optimizations.

---

## Infrastructure Issues Encountered

### Disk Space Management
The benchmark runs consumed significant disk space due to Docker images (~15-30GB each). Required multiple cleanup cycles:
- 9 disk cleanup operations during TRAE Sonnet 4.5 run
- Disk usage peaked at 95% multiple times
- Total images processed: ~100+ baseline images

### Initial TRAE Sonnet 4.5 Run
First run failed 49 commits due to disk full errors. Required:
1. Full disk cleanup (`docker system prune -af`)
2. Backup of 8 successful results
3. Complete re-run (captured this time)

---

## What These Results Mean

### For Agent Developers
- **~65% of patches cause server crashes or produce no metrics** - agents need better validation before proposing changes
- **Throughput optimizations rarely work** - may indicate agents don't understand throughput benchmark requirements
- **Serving benchmarks are most successful** - suggests agents better understand HTTP serving workloads

### For Benchmark Infrastructure
- **Missing metadata for 6.25% of commits** - data pipeline needs fixes
- **TRAE GPT-5 coverage gap** - patch generation pipeline incomplete
- **No automated patch validation** - many failures could be caught before benchmark

### For Research Validity
- **Low success rates limit statistical significance** - 30 successes out of 96 attempts means small sample
- **Cross-agent comparison unreliable for TRAE GPT-5** - only 42/96 coverage
- **Server crash rate suggests patch quality > infrastructure issues**

---

## Recommendations

1. **Add pre-benchmark patch validation**: Test patch syntax, import resolution, basic startup before full benchmark
2. **Fix missing parent_commit data**: 6 commits permanently unbenchmarkable without fix
3. **Complete TRAE GPT-5 patch coverage**: 54 commits missing patches
4. **Investigate throughput benchmark failures**: Near-zero success needs root cause analysis
5. **Add patch quality metrics**: Track syntax errors, crash types, partial successes

---

## File Locations

| Agent | Results Directory |
|-------|-------------------|
| TRAE Sonnet 4.5 | `/root/OmniPerf-Bench/omniperf_results_3way_trae_sonnet45/results/` |
| Codex GPT-5 | `/root/OmniPerf-Bench/omniperf_results_3way_codex/results/` |
| TRAE GPT-5 | `/root/OmniPerf-Bench/omniperf_results_3way_trae_gpt5/results/` |

---

*Report generated by automated benchmark analysis*
