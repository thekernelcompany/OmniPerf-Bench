# OmniPerf-Bench: All Visualizations

**Generated:** 2025-12-03

This folder contains all visualizations and data from the comprehensive multi-agent analysis of 837 optimization tasks across 13 runs.

---

## Chart Labels

All charts include detailed labels with:
- **Agent Framework**: Codex, TRAE Agent
- **Model**: Claude (Codex CLI), Claude Sonnet 4.5 (AWS Bedrock), GPT-4o (OpenAI)
- **Repository**: VLLM, SGLANG
- **Statistics**: Task count, success rate, clean success rate

Example title format:
```
Duration Distribution
Codex (Codex CLI (Claude)) | SGLANG | 80 tasks | 100% success
```

---

## Folder Structure

```
all_visualizations/
├── codex/                    # Codex CLI agent (24 charts)
│   ├── codex_vllm_*.png      # vLLM tasks (99 tasks, 100% success, 38.4% clean)
│   └── codex_sglang_*.png    # SGLang tasks (80 tasks, 100% success, 96.2% clean)
│
├── trae_bedrock/             # TRAE with Claude Sonnet 4.5 (36 charts)
│   ├── claude_sglang_90pct_*.png    # SGLang (80 tasks, 90% success)
│   ├── claude_vllm_retry_45pct_*.png # vLLM retry (82 tasks, 45.1% success)
│   └── claude_vllm_17pct_*.png      # vLLM first (99 tasks, 17.2% success)
│
├── trae_gpt/                 # TRAE with GPT-4o (60 charts)
│   ├── gpt_vllm_93pct_*.png        # vLLM best run (60 tasks, 93.3% success)
│   ├── gpt_sglang_84pct_*.png      # SGLang (51 tasks, 84.3% success)
│   ├── gpt_sglang_36pct_*.png      # SGLang (80 tasks, 36.2% success)
│   ├── gpt_vllm_63pct_*.png        # vLLM (49 tasks, 63.3% success)
│   └── gpt_vllm_65pct_*.png        # vLLM (49 tasks, 65.3% success)
│
├── comparative/              # Cross-agent comparisons (8 charts)
│   ├── comparison_01_success_by_agent_repo.png
│   ├── comparison_02_clean_success_heatmap.png
│   ├── comparison_03_duration_by_agent.png
│   ├── comparison_04_ttfe_by_agent.png
│   ├── comparison_05_commits_by_agent.png
│   ├── comparison_06_pathological_heatmap.png
│   ├── comparison_07_instant_edit_failure.png
│   └── comparison_08_overall_comparison.png
│
├── data/                     # Unified data exports
│   ├── all_tasks_unified.csv       # All 837 tasks
│   ├── all_runs_summary.csv        # Per-run summaries
│   ├── agent_type_summary.csv      # By agent type
│   ├── repository_summary.csv      # By repository
│   └── *_summary.json              # Per-run statistics
│
└── README.md                 # This file
```

---

## Agent Summary

| Agent | Model | Framework | Repository | Tasks | Success | Clean | Pathological |
|-------|-------|-----------|------------|-------|---------|-------|--------------|
| **Codex** | Claude (Codex CLI) | Codex | vLLM | 99 | 100% | 38.4% | 60.6% |
| **Codex** | Claude (Codex CLI) | Codex | SGLang | 80 | 100% | 96.2% | 0% |
| **TRAE-Bedrock** | Claude Sonnet 4.5 | TRAE Agent | SGLang | 80 | 90% | 56.2% | 0% |
| **TRAE-Bedrock** | Claude Sonnet 4.5 | TRAE Agent | vLLM | 181 | 17-45% | 17-45% | 0% |
| **TRAE-GPT** | GPT-4o | TRAE Agent | vLLM | 234 | 0-93% | 0-80% | 0-2% |
| **TRAE-GPT** | GPT-4o | TRAE Agent | SGLang | 131 | 36-84% | 30-69% | 0% |

---

## Key Visualizations

### Most Important Charts

1. **`comparative/comparison_08_overall_comparison.png`** - Summary dashboard with all metrics
2. **`comparative/comparison_02_clean_success_heatmap.png`** - Agent × Repo performance matrix
3. **`comparative/comparison_06_pathological_heatmap.png`** - Failure mode analysis
4. **`codex/codex_vllm_04_success_categories.png`** - Codex pathological breakdown

### Per-Run Chart Index

Each run has 12 standardized charts with detailed labels:

| # | Chart | Description |
|---|-------|-------------|
| 01 | duration_distribution | Task duration histogram |
| 02 | duration_vs_commits | Scatter: duration vs commits |
| 03 | time_to_first_edit | TTFE distribution (bimodal analysis) |
| 04 | success_categories | Pie chart of outcomes |
| 05 | patch_size_distribution | Lines of code changed |
| 06 | files_changed_distribution | Number of files modified |
| 07 | commits_distribution | Commit count histogram |
| 08 | violations_analysis | Violations by range |
| 09 | commits_vs_violations | Correlation scatter |
| 10 | efficiency | LOC per minute |
| 11 | clean_vs_pathological | Boxplot comparison |
| 12 | performance_over_time | Timeline view |

---

## Key Findings

### 1. The 58-Point Gap
- Codex on SGLang: **96.2%** clean success
- Codex on vLLM: **38.4%** clean success
- Same agent, same config, **58 percentage point difference**

### 2. Pathological Behavior
- Codex on vLLM: 60.6% pathological (>50 commits or >100 violations)
- Max commits: **7,755** on a single task
- TRAE agents: 0% pathological (controlled behavior)

### 3. Claude Sonnet 4.5 Performance (TRAE-Bedrock)
- Best on SGLang: **90%** success rate
- vLLM challenging: 17-45% success
- No commit explosion (max 3 commits)

### 4. GPT-4o Performance (TRAE-GPT)
- High variance: 0-93% success across runs
- Best vLLM run: **93.3%** success (60 tasks)
- Longer task duration (avg 793s vs 187s for Codex)

### 5. Instant-Edit Pathology
- TTFE < 1 second correlates with higher failure rates
- Codex: 33% instant edits, 59.6% instant on vLLM (pathological)
- TRAE-GPT: Only 1% instant edits

---

## Model Mapping

| Agent Type | Model Used | Provider |
|------------|------------|----------|
| Codex | Claude (via Codex CLI) | Anthropic |
| TRAE-Bedrock | Claude Sonnet 4.5 | AWS Bedrock |
| TRAE-GPT | GPT-4o | OpenAI |

---

## Data Files

| File | Rows | Description |
|------|------|-------------|
| `all_tasks_unified.csv` | 837 | Every task from every run |
| `all_runs_summary.csv` | 13 | One row per run |
| `agent_type_summary.csv` | 3 | Aggregated by agent |
| `repository_summary.csv` | 2 | Aggregated by repo |

---

## Usage

```python
import pandas as pd

# Load all tasks
df = pd.read_csv('data/all_tasks_unified.csv')

# Filter by agent
codex_df = df[df['agent_subtype'] == 'Codex']

# Filter by repository
sglang_df = df[df['repository'] == 'SGLang']

# Success rates by agent and repo
print(df.groupby(['agent_subtype', 'repository'])['success'].mean())
```

---

*Generated by OmniPerf-Bench analysis scripts*
