# Analysis Report: vllm_core-9641716f

**Generated:** 2025-12-03 21:32:44

## Run Metadata

| Property | Value |
|----------|-------|
| Run ID | `vllm_core-9641716f` |
| Agent | trae-gpt |
| Repository | vllm |
| Total Tasks | 60 |

## Summary Statistics

### Success Rates

| Metric | Value |
|--------|-------|
| Total Tasks | 60 |
| Success Rate | 93.3% (56/60) |
| Clean Success Rate | 80.0% (48) |
| Pathological Rate | 0.0% (0) |
| Failed | 4 |

### Duration

| Metric | Value |
|--------|-------|
| Mean | 1201.4s |
| Median | 1194.5s |
| Std Dev | 455.3s |
| Range | 168.9s - 2277.9s |

### Commits

| Metric | Value |
|--------|-------|
| Mean | 1.1 |
| Median | 1 |
| Max | 6 |
| Single Commit Rate | 80.0% |

### Violations

| Metric | Value |
|--------|-------|
| Mean | 0.1 |
| Median | 0 |
| Max | 1 |
| Zero Violations Rate | 90.0% |

### Time to First Edit

| Metric | Value |
|--------|-------|
| Mean | 899.7s |
| Median | 918.6s |
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

- `data/vllm_core_9641716f_metrics_all.csv` - All tasks
- `data/vllm_core_9641716f_metrics_clean.csv` - Clean successes only
- `data/vllm_core_9641716f_metrics_anomalous.csv` - Pathological/failed only
- `data/vllm_core_9641716f_summary.json` - Summary statistics
- `data/vllm_core_9641716f_outliers.json` - Outlier details
