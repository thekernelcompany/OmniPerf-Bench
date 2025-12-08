# Analysis Report: vllm_core_codex-90a1c13f

**Generated:** 2025-12-03 21:32:53

## Run Metadata

| Property | Value |
|----------|-------|
| Run ID | `vllm_core_codex-90a1c13f` |
| Agent | codex |
| Repository | vllm |
| Total Tasks | 99 |

## Summary Statistics

### Success Rates

| Metric | Value |
|--------|-------|
| Total Tasks | 99 |
| Success Rate | 100.0% (99/99) |
| Clean Success Rate | 38.4% (38) |
| Pathological Rate | 60.6% (60) |
| Failed | 0 |

### Duration

| Metric | Value |
|--------|-------|
| Mean | 187.7s |
| Median | 171.4s |
| Std Dev | 66.7s |
| Range | 101.7s - 515.1s |

### Commits

| Metric | Value |
|--------|-------|
| Mean | 805.4 |
| Median | 2 |
| Max | 7,755 |
| Single Commit Rate | 40.4% |

### Violations

| Metric | Value |
|--------|-------|
| Mean | 1141.4 |
| Median | 1004 |
| Max | 2,970 |
| Zero Violations Rate | 38.4% |

### Time to First Edit

| Metric | Value |
|--------|-------|
| Mean | 65.7s |
| Median | 0.1s |
| Instant Edit Rate (<1s) | 59.6% (59 tasks) |

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

- `data/vllm_core_codex_90a1c13f_metrics_all.csv` - All tasks
- `data/vllm_core_codex_90a1c13f_metrics_clean.csv` - Clean successes only
- `data/vllm_core_codex_90a1c13f_metrics_anomalous.csv` - Pathological/failed only
- `data/vllm_core_codex_90a1c13f_summary.json` - Summary statistics
- `data/vllm_core_codex_90a1c13f_outliers.json` - Outlier details
