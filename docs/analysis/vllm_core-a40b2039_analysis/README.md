# Analysis Report: vllm_core-a40b2039

**Generated:** 2025-12-03 21:32:47

## Run Metadata

| Property | Value |
|----------|-------|
| Run ID | `vllm_core-a40b2039` |
| Agent | trae-gpt |
| Repository | vllm |
| Total Tasks | 49 |

## Summary Statistics

### Success Rates

| Metric | Value |
|--------|-------|
| Total Tasks | 49 |
| Success Rate | 63.3% (31/49) |
| Clean Success Rate | 26.5% (13) |
| Pathological Rate | 2.0% (1) |
| Failed | 18 |

### Duration

| Metric | Value |
|--------|-------|
| Mean | 1192.0s |
| Median | 1138.7s |
| Std Dev | 817.4s |
| Range | 105.1s - 2664.6s |

### Commits

| Metric | Value |
|--------|-------|
| Mean | 39.3 |
| Median | 1 |
| Max | 1,878 |
| Single Commit Rate | 26.5% |

### Violations

| Metric | Value |
|--------|-------|
| Mean | 46.4 |
| Median | 0 |
| Max | 2,260 |
| Zero Violations Rate | 65.3% |

### Time to First Edit

| Metric | Value |
|--------|-------|
| Mean | 1009.4s |
| Median | 1063.8s |
| Instant Edit Rate (<1s) | 4.1% (2 tasks) |

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

- `data/vllm_core_a40b2039_metrics_all.csv` - All tasks
- `data/vllm_core_a40b2039_metrics_clean.csv` - Clean successes only
- `data/vllm_core_a40b2039_metrics_anomalous.csv` - Pathological/failed only
- `data/vllm_core_a40b2039_summary.json` - Summary statistics
- `data/vllm_core_a40b2039_outliers.json` - Outlier details
