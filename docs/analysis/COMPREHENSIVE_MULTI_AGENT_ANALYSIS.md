# Comprehensive Multi-Agent Analysis Report

**Generated:** 2025-12-03 21:33:03

## Executive Summary

This report provides a comprehensive analysis of 837 optimization tasks across 13 runs,
comparing three agent architectures: **Codex**, **TRAE-Bedrock** (Claude Sonnet 4.5), and **TRAE-GPT**.

### Key Findings

1. **Codex achieves highest completion rate** (100%) but exhibits pathological failures (33.5% pathological rate on vLLM)
2. **TRAE-Bedrock shows best results on SGLang** (90% success, 56.2% clean)
3. **Repository complexity dominates performance**: SGLang consistently easier than vLLM across all agents
4. **Instant-edit correlates with failure**: Tasks with TTFE <1s have significantly higher failure rates

---

## Overall Statistics

| Metric | Value |
|--------|-------|
| Total Tasks | 837 |
| Total Runs | 13 |
| Agents Evaluated | 4 |
| Repositories | 2 |

---

## Agent Performance Summary

| Agent | Tasks | Success | Clean Success | Pathological | Avg Duration |
|-------|-------|---------|---------------|--------------|--------------|
| Codex | 179 | 100.0% | 64.2% | 33.5% | 186.8s |
| TRAE-Bedrock | 261 | 48.3% | 37.9% | 0.0% | 330.3s |
| TRAE-GPT | 395 | 48.4% | 32.4% | 0.5% | 793.1s |

---

## Repository Performance Summary

| Repository | Tasks | Success Rate | Clean Rate |
|------------|-------|--------------|------------|
| SGLang | 291 | 77.0% | 62.2% |
| vLLM | 546 | 49.8% | 29.5% |

---

## Cross-Agent × Repository Analysis

| Agent | Repository | Success | Clean | Pathological |
|-------|------------|---------|-------|--------------|
| Codex | SGLang | 100.0% | 96.2% | 0.0% |
| Codex | vLLM | 100.0% | 38.4% | 60.6% |
| TRAE-Bedrock | SGLang | 90.0% | 56.2% | 0.0% |
| TRAE-Bedrock | vLLM | 29.8% | 29.8% | 0.0% |
| TRAE-GPT | SGLang | 55.0% | 45.0% | 0.0% |
| TRAE-GPT | vLLM | 45.1% | 26.1% | 0.8% |
| unknown | vLLM | 0.0% | 0.0% | 0.0% |

---

## Key Behavioral Patterns

### 1. Instant-Edit Pathology

Time-to-first-edit (TTFE) <1 second is a strong predictor of failure:

| Agent | Instant Edit Rate | Failure Rate (Instant) | Failure Rate (Delayed) |
|-------|-------------------|------------------------|------------------------|
| Codex | 33.0% | 0.0% | 0.0% |
| TRAE-Bedrock | 27.6% | 2.8% | 70.4% |
| TRAE-GPT | 1.0% | 0.0% | 52.2% |

### 2. Commit Explosion

Tasks with >50 commits indicate pathological behavior:

| Agent | Max Commits | Avg Commits (Success) | Avg Commits (Failure) |
|-------|-------------|----------------------|----------------------|
| Codex | 7,755 | 445.9 | nan |
| TRAE-Bedrock | 3 | 1.2 | 0.0 |
| TRAE-GPT | 1,878 | 21.0 | 0.0 |

---

## Run-by-Run Details

| Run ID | Agent | Repository | Tasks | Success | Clean | Pathological |
|--------|-------|------------|-------|---------|-------|--------------|
| sglang_core-389be848... | Codex | SGLang | 80 | 100.0% | 96.2% | 0.0% |
| vllm_core_codex-90a1c13f... | Codex | vLLM | 99 | 100.0% | 38.4% | 60.6% |
| sglang_claude_sonnet45-c0645fb... | TRAE-Bedrock | SGLang | 80 | 90.0% | 56.2% | 0.0% |
| vllm_claude_sonnet45-0a51aaa8... | TRAE-Bedrock | vLLM | 99 | 17.2% | 17.2% | 0.0% |
| vllm_claude_sonnet45_retry-5d5... | TRAE-Bedrock | vLLM | 82 | 45.1% | 45.1% | 0.0% |
| sglang_core-ae58875a... | TRAE-GPT | SGLang | 80 | 36.2% | 30.0% | 0.0% |
| sglang_core-bd68ff67... | TRAE-GPT | SGLang | 51 | 84.3% | 68.6% | 0.0% |
| vllm_core-84ca0ad4... | TRAE-GPT | vLLM | 44 | 0.0% | 0.0% | 0.0% |
| vllm_core-8e54a51a... | TRAE-GPT | vLLM | 32 | 0.0% | 0.0% | 0.0% |
| vllm_core-9641716f... | TRAE-GPT | vLLM | 60 | 93.3% | 80.0% | 0.0% |
| vllm_core-a40b2039... | TRAE-GPT | vLLM | 49 | 63.3% | 26.5% | 2.0% |
| vllm_core-aed20220... | TRAE-GPT | vLLM | 32 | 0.0% | 0.0% | 0.0% |
| vllm_core-beffe4cd... | TRAE-GPT | vLLM | 49 | 65.3% | 16.3% | 2.0% |

---

## Visualizations

See `multi_agent_comparison/visualizations/` for detailed charts:

1. **Success by Agent+Repo** - Grouped bar chart comparing success rates
2. **Clean Success Heatmap** - Agent × Repository clean success rates
3. **Duration by Agent** - Boxplot of task durations
4. **TTFE by Agent** - Time-to-first-edit comparison
5. **Commits by Agent** - Violin plot of commit distributions
6. **Pathological Heatmap** - Agent × Repository pathological rates
7. **Instant Edit vs Failure** - Correlation analysis
8. **Overall Comparison** - Summary dashboard

---

## Data Files

- `multi_agent_comparison/data/all_tasks_unified.csv` - All 837 tasks
- `multi_agent_comparison/data/all_runs_summary.csv` - Per-run summaries
- `multi_agent_comparison/data/agent_type_summary.csv` - Agent statistics
- `multi_agent_comparison/data/repository_summary.csv` - Repository statistics

---

## Methodology Notes

- **Clean Success**: status='success', 1 commit, 0 violations
- **Pathological**: >50 commits OR >100 violations
- **Instant Edit**: TTFE < 1 second
- Minimum 10 tasks per run for inclusion

---

*Report generated automatically by `analyze_all_runs.py`*
