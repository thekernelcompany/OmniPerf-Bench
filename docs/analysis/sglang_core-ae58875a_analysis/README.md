# Analysis Report: sglang_core-ae58875a

**Generated:** 2025-12-03 21:32:35

## Run Metadata

| Property | Value |
|----------|-------|
| Run ID | `sglang_core-ae58875a` |
| Agent | trae-gpt |
| Repository | sglang |
| Total Tasks | 80 |

## Summary Statistics

### Success Rates

| Metric | Value |
|--------|-------|
| Total Tasks | 80 |
| Success Rate | 36.2% (29/80) |
| Clean Success Rate | 30.0% (24) |
| Pathological Rate | 0.0% (0) |
| Failed | 51 |

### Duration

| Metric | Value |
|--------|-------|
| Mean | 753.9s |
| Median | 226.5s |
| Std Dev | 825.5s |
| Range | 143.1s - 3600.1s |

### Commits

| Metric | Value |
|--------|-------|
| Mean | 0.4 |
| Median | 0 |
| Max | 2 |
| Single Commit Rate | 30.0% |

### Violations

| Metric | Value |
|--------|-------|
| Mean | 0.4 |
| Median | 0 |
| Max | 30 |
| Zero Violations Rate | 93.8% |

### Time to First Edit

| Metric | Value |
|--------|-------|
| Mean | 1071.1s |
| Median | 1083.9s |
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

- `data/sglang_core_ae58875a_metrics_all.csv` - All tasks
- `data/sglang_core_ae58875a_metrics_clean.csv` - Clean successes only
- `data/sglang_core_ae58875a_metrics_anomalous.csv` - Pathological/failed only
- `data/sglang_core_ae58875a_summary.json` - Summary statistics
- `data/sglang_core_ae58875a_outliers.json` - Outlier details
