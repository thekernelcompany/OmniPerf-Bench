# Analysis Report: vllm_claude_sonnet45-0a51aaa8

**Generated:** 2025-12-03 21:32:39

## Run Metadata

| Property | Value |
|----------|-------|
| Run ID | `vllm_claude_sonnet45-0a51aaa8` |
| Agent | trae-bedrock |
| Repository | vllm |
| Total Tasks | 99 |

## Summary Statistics

### Success Rates

| Metric | Value |
|--------|-------|
| Total Tasks | 99 |
| Success Rate | 17.2% (17/99) |
| Clean Success Rate | 17.2% (17) |
| Pathological Rate | 0.0% (0) |
| Failed | 82 |

### Duration

| Metric | Value |
|--------|-------|
| Mean | 248.7s |
| Median | 175.9s |
| Std Dev | 182.1s |
| Range | 91.9s - 974.7s |

### Commits

| Metric | Value |
|--------|-------|
| Mean | 0.2 |
| Median | 0 |
| Max | 1 |
| Single Commit Rate | 17.2% |

### Violations

| Metric | Value |
|--------|-------|
| Mean | 0.0 |
| Median | 0 |
| Max | 0 |
| Zero Violations Rate | 100.0% |

### Time to First Edit

| Metric | Value |
|--------|-------|
| Mean | 273.4s |
| Median | 248.7s |
| Instant Edit Rate (<1s) | 0.0% (0 tasks) |

## Visualizations

| Chart | Description |
|-------|-------------|
| [01_duration_distribution.png](visualizations/01_duration_distribution.png) | Duration histogram |
| [02_duration_vs_commits.png](visualizations/02_duration_vs_commits.png) | Duration vs commits scatter |
| [03_time_to_first_edit.png](visualizations/03_time_to_first_edit.png) | TTFE distribution |
| [04_success_categories.png](visualizations/04_success_categories.png) | Success categories pie |
| [05_patch_size_distribution.png](visualizations/05_patch_size_distribution.png) | Patch size histogram |
| [06_files_changed_distribution.png](visualizations/06_files_changed_distribution.png) | Files changed histogram |
| [07_commits_distribution.png](visualizations/07_commits_distribution.png) | Commit count histogram |
| [08_violations_analysis.png](visualizations/08_violations_analysis.png) | Violations by range |
| [09_commits_vs_violations.png](visualizations/09_commits_vs_violations.png) | Commits vs violations scatter |
| [10_efficiency.png](visualizations/10_efficiency.png) | LOC/minute efficiency |
| [11_clean_vs_pathological.png](visualizations/11_clean_vs_pathological.png) | Clean vs pathological comparison |
| [12_performance_over_time.png](visualizations/12_performance_over_time.png) | Performance timeline |

## Data Files

- `data/vllm_claude_sonnet45_0a51aaa8_metrics_all.csv` - All tasks
- `data/vllm_claude_sonnet45_0a51aaa8_metrics_clean.csv` - Clean successes only
- `data/vllm_claude_sonnet45_0a51aaa8_metrics_anomalous.csv` - Pathological/failed only
- `data/vllm_claude_sonnet45_0a51aaa8_summary.json` - Summary statistics
- `data/vllm_claude_sonnet45_0a51aaa8_outliers.json` - Outlier details
