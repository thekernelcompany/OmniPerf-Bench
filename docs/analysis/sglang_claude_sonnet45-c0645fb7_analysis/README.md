# Analysis Report: sglang_claude_sonnet45-c0645fb7

**Generated:** 2025-12-03 21:32:31

## Run Metadata

| Property | Value |
|----------|-------|
| Run ID | `sglang_claude_sonnet45-c0645fb7` |
| Agent | trae-bedrock |
| Repository | sglang |
| Total Tasks | 80 |

## Summary Statistics

### Success Rates

| Metric | Value |
|--------|-------|
| Total Tasks | 80 |
| Success Rate | 90.0% (72/80) |
| Clean Success Rate | 56.2% (45) |
| Pathological Rate | 0.0% (0) |
| Failed | 8 |

### Duration

| Metric | Value |
|--------|-------|
| Mean | 365.8s |
| Median | 250.3s |
| Std Dev | 237.2s |
| Range | 132.9s - 1179.5s |

### Commits

| Metric | Value |
|--------|-------|
| Mean | 1.2 |
| Median | 1 |
| Max | 3 |
| Single Commit Rate | 60.0% |

### Violations

| Metric | Value |
|--------|-------|
| Mean | 0.6 |
| Median | 0 |
| Max | 30 |
| Zero Violations Rate | 77.5% |

### Time to First Edit

| Metric | Value |
|--------|-------|
| Mean | 3.3s |
| Median | 0.0s |
| Instant Edit Rate (<1s) | 90.0% (72 tasks) |

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

- `data/sglang_claude_sonnet45_c0645fb7_metrics_all.csv` - All tasks
- `data/sglang_claude_sonnet45_c0645fb7_metrics_clean.csv` - Clean successes only
- `data/sglang_claude_sonnet45_c0645fb7_metrics_anomalous.csv` - Pathological/failed only
- `data/sglang_claude_sonnet45_c0645fb7_summary.json` - Summary statistics
- `data/sglang_claude_sonnet45_c0645fb7_outliers.json` - Outlier details
