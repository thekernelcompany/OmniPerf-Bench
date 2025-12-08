#!/usr/bin/env python3
"""
Comprehensive Analysis Script for Codex vLLM Agent Results

Analyzes all 99 vLLM optimization tasks executed by Codex agent,
generates visualizations, computes statistics, and exports data.

Usage:
    python analyze_codex_results.py

Outputs:
    - 12 publication-quality visualizations (300 DPI)
    - CSV exports of all metrics
    - JSON summary statistics
    - Comprehensive analysis report data
"""

import json
import os
from pathlib import Path
from typing import Dict, List, Any, Tuple
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings

warnings.filterwarnings('ignore')

# Configuration
RESULTS_DIR = Path("perf-agents-bench/state/runs/vllm_core_codex-90a1c13f")
OUTPUT_DIR = Path("docs/codex_analysis")
VIZ_DIR = OUTPUT_DIR / "visualizations"
DATA_DIR = OUTPUT_DIR / "data"

# Create output directories
VIZ_DIR.mkdir(parents=True, exist_ok=True)
DATA_DIR.mkdir(parents=True, exist_ok=True)

# Visualization settings
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.size'] = 10
sns.set_palette("husl")


def parse_journal(journal_path: Path) -> Dict[str, Any]:
    """Parse a journal.json file and extract all metrics."""
    try:
        with open(journal_path) as f:
            data = json.load(f)

        metrics = data.get('metrics', {})
        codex_cli = data.get('codex_cli', {})

        # Extract task number from directory name (e.g., vllm_core-0001 -> 1)
        import re
        task_num_match = re.search(r'-(\d+)$', journal_path.parent.name)
        task_number = int(task_num_match.group(1)) if task_num_match else 0

        return {
            'task_id': data.get('task_id', 'unknown'),
            'task_number': task_number,
            'status': data.get('status', 'unknown'),
            'duration_s': codex_cli.get('duration_s', 0),
            'returncode': codex_cli.get('returncode', -1),
            'time_to_first_edit_s': metrics.get('time_to_first_edit_s', 0),
            'commit_count': metrics.get('commit_count', 0),
            'patch_size_loc': metrics.get('patch_size_loc', 0),
            'changed_files_count': metrics.get('changed_files_count', 0),
            'violations_count': metrics.get('violations_count', 0),
            'pre_commit': data.get('commits', {}).get('pre', ''),
            'human_commit': data.get('commits', {}).get('human', ''),
        }
    except Exception as e:
        print(f"Error parsing {journal_path}: {e}")
        return None


def load_all_results() -> pd.DataFrame:
    """Load all journal.json files and create a comprehensive DataFrame."""
    print("Loading all Codex results...")

    results = []
    task_dirs = sorted([d for d in RESULTS_DIR.iterdir() if d.is_dir()])

    for task_dir in task_dirs:
        journal_path = task_dir / "journal.json"
        if journal_path.exists():
            parsed = parse_journal(journal_path)
            if parsed:
                results.append(parsed)

    if not results:
        raise ValueError("No valid journal files found!")

    df = pd.DataFrame(results)

    # Add computed fields
    df['success'] = df['status'] == 'success'
    df['loc_per_minute'] = (df['patch_size_loc'] / (df['duration_s'] / 60)).replace([np.inf, -np.inf], 0)
    df['instant_edit'] = df['time_to_first_edit_s'] < 1.0

    # Categorize tasks
    def categorize_task(row):
        if row['commit_count'] == 1 and row['violations_count'] == 0:
            return 'clean_success'
        elif row['commit_count'] > 50 or row['violations_count'] > 100:
            return 'pathological_failure'
        elif row['violations_count'] > 0:
            return 'success_with_violations'
        else:
            return 'normal_success'

    df['category'] = df.apply(categorize_task, axis=1)

    print(f"Loaded {len(df)} tasks")
    print(f"  Clean success: {len(df[df['category'] == 'clean_success'])}")
    print(f"  Normal success: {len(df[df['category'] == 'normal_success'])}")
    print(f"  Success with violations: {len(df[df['category'] == 'success_with_violations'])}")
    print(f"  Pathological failures: {len(df[df['category'] == 'pathological_failure'])}")

    return df


def compute_statistics(df: pd.DataFrame) -> Dict[str, Any]:
    """Compute comprehensive statistics for the dataset."""
    stats = {
        'overview': {
            'total_tasks': len(df),
            'success_rate': (df['success'].sum() / len(df) * 100),
            'clean_success_rate': (len(df[df['category'] == 'clean_success']) / len(df) * 100),
            'pathological_failure_rate': (len(df[df['category'] == 'pathological_failure']) / len(df) * 100),
        },
        'duration': {
            'mean': df['duration_s'].mean(),
            'median': df['duration_s'].median(),
            'std': df['duration_s'].std(),
            'min': df['duration_s'].min(),
            'max': df['duration_s'].max(),
            'q25': df['duration_s'].quantile(0.25),
            'q75': df['duration_s'].quantile(0.75),
        },
        'time_to_first_edit': {
            'mean': df['time_to_first_edit_s'].mean(),
            'median': df['time_to_first_edit_s'].median(),
            'instant_edit_count': df['instant_edit'].sum(),
            'instant_edit_rate': (df['instant_edit'].sum() / len(df) * 100),
        },
        'commits': {
            'mean': df['commit_count'].mean(),
            'median': df['commit_count'].median(),
            'max': df['commit_count'].max(),
            'single_commit_rate': (len(df[df['commit_count'] == 1]) / len(df) * 100),
        },
        'patch_size': {
            'mean': df['patch_size_loc'].mean(),
            'median': df['patch_size_loc'].median(),
            'total': df['patch_size_loc'].sum(),
            'max': df['patch_size_loc'].max(),
        },
        'files_changed': {
            'mean': df['changed_files_count'].mean(),
            'median': df['changed_files_count'].median(),
            'max': df['changed_files_count'].max(),
        },
        'violations': {
            'mean': df['violations_count'].mean(),
            'median': df['violations_count'].median(),
            'max': df['violations_count'].max(),
            'zero_violations_rate': (len(df[df['violations_count'] == 0]) / len(df) * 100),
        },
        'efficiency': {
            'mean_loc_per_minute': df['loc_per_minute'].mean(),
            'median_loc_per_minute': df['loc_per_minute'].median(),
        },
        'correlations': {
            'time_vs_commits': df['duration_s'].corr(df['commit_count']),
            'commits_vs_violations': df['commit_count'].corr(df['violations_count']),
            'ttfe_vs_commits': df['time_to_first_edit_s'].corr(df['commit_count']),
        },
        'outliers': {
            'max_commits_task': df.loc[df['commit_count'].idxmax()]['task_number'],
            'max_commits_value': df['commit_count'].max(),
            'max_violations_task': df.loc[df['violations_count'].idxmax()]['task_number'],
            'max_violations_value': df['violations_count'].max(),
            'max_files_task': df.loc[df['changed_files_count'].idxmax()]['task_number'],
            'max_files_value': df['changed_files_count'].max(),
        }
    }

    return stats


def create_visualizations(df: pd.DataFrame):
    """Generate all 12 publication-quality visualizations."""
    print("\nGenerating visualizations...")

    # 1. Duration Distribution
    plt.figure(figsize=(10, 6))
    plt.hist(df['duration_s'], bins=30, edgecolor='black', alpha=0.7)
    plt.axvline(df['duration_s'].mean(), color='red', linestyle='--', label=f'Mean: {df["duration_s"].mean():.1f}s')
    plt.axvline(df['duration_s'].median(), color='green', linestyle='--', label=f'Median: {df["duration_s"].median():.1f}s')
    plt.xlabel('Duration (seconds)')
    plt.ylabel('Frequency')
    plt.title('Distribution of Task Execution Duration')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(VIZ_DIR / '01_duration_distribution.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("  ✓ Duration distribution")

    # 2. Duration vs Commits Scatter
    plt.figure(figsize=(10, 6))
    scatter = plt.scatter(df['duration_s'], df['commit_count'],
                         c=df['violations_count'], cmap='YlOrRd',
                         alpha=0.6, s=100, edgecolors='black', linewidth=0.5)
    plt.colorbar(scatter, label='Violations Count')
    plt.xlabel('Duration (seconds)')
    plt.ylabel('Commit Count')
    plt.title('Task Duration vs Commit Count (colored by violations)')
    plt.yscale('log')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(VIZ_DIR / '02_duration_vs_commits.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("  ✓ Duration vs commits scatter")

    # 3. Time to First Edit Bimodal Distribution
    plt.figure(figsize=(10, 6))
    plt.hist(df['time_to_first_edit_s'], bins=50, edgecolor='black', alpha=0.7)
    plt.axvline(1.0, color='red', linestyle='--', linewidth=2, label='1s threshold (failure predictor)')
    instant = len(df[df['time_to_first_edit_s'] < 1.0])
    plt.text(0.5, plt.ylim()[1] * 0.9, f'{instant} tasks (<1s)', fontsize=12, color='red')
    plt.xlabel('Time to First Edit (seconds)')
    plt.ylabel('Frequency')
    plt.title('Time to First Edit Distribution (Bimodal Pattern)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(VIZ_DIR / '03_time_to_first_edit.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("  ✓ Time to first edit distribution")

    # 4. Success Categories Pie Chart
    plt.figure(figsize=(10, 8))
    category_counts = df['category'].value_counts()
    colors = ['#2ecc71', '#3498db', '#f39c12', '#e74c3c']
    explode = [0.05 if cat == 'pathological_failure' else 0 for cat in category_counts.index]

    plt.pie(category_counts.values, labels=[
        f"{cat.replace('_', ' ').title()}\n({count} tasks, {count/len(df)*100:.1f}%)"
        for cat, count in zip(category_counts.index, category_counts.values)
    ], autopct='%1.1f%%', startangle=90, colors=colors, explode=explode)
    plt.title('Task Success Categories', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(VIZ_DIR / '04_success_categories.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("  ✓ Success categories pie chart")

    # 5. Patch Size Distribution
    plt.figure(figsize=(10, 6))
    plt.hist(df['patch_size_loc'], bins=40, edgecolor='black', alpha=0.7)
    plt.axvline(df['patch_size_loc'].mean(), color='red', linestyle='--', label=f'Mean: {df["patch_size_loc"].mean():.0f} LOC')
    plt.axvline(df['patch_size_loc'].median(), color='green', linestyle='--', label=f'Median: {df["patch_size_loc"].median():.0f} LOC')
    plt.xlabel('Patch Size (Lines of Code)')
    plt.ylabel('Frequency')
    plt.title('Distribution of Patch Sizes')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(VIZ_DIR / '05_patch_size_distribution.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("  ✓ Patch size distribution")

    # 6. Files Changed Distribution (Log Scale)
    plt.figure(figsize=(10, 6))
    plt.hist(df['changed_files_count'], bins=40, edgecolor='black', alpha=0.7)
    plt.xlabel('Files Changed Count')
    plt.ylabel('Frequency')
    plt.title('Distribution of Files Changed (showing outliers)')
    plt.yscale('log')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(VIZ_DIR / '06_files_changed_distribution.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("  ✓ Files changed distribution")

    # 7. Commits Distribution (Log Scale)
    plt.figure(figsize=(10, 6))
    plt.hist(df['commit_count'], bins=50, edgecolor='black', alpha=0.7)
    plt.xlabel('Commit Count')
    plt.ylabel('Frequency (log scale)')
    plt.title('Distribution of Commit Counts')
    plt.yscale('log')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(VIZ_DIR / '07_commits_distribution.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("  ✓ Commits distribution")

    # 8. Violations Analysis Bar Chart
    plt.figure(figsize=(12, 6))
    violation_bins = pd.cut(df['violations_count'],
                           bins=[-1, 0, 10, 100, 1000, 10000],
                           labels=['0', '1-10', '11-100', '101-1000', '>1000'])
    violation_counts = violation_bins.value_counts().sort_index()

    bars = plt.bar(range(len(violation_counts)), violation_counts.values,
                   color=['#2ecc71', '#3498db', '#f39c12', '#e67e22', '#e74c3c'])
    plt.xticks(range(len(violation_counts)), violation_counts.index)
    plt.xlabel('Violations Count Range')
    plt.ylabel('Number of Tasks')
    plt.title('Violation Count Distribution by Range')

    for i, (bar, count) in enumerate(zip(bars, violation_counts.values)):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                str(count), ha='center', va='bottom', fontweight='bold')

    plt.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig(VIZ_DIR / '08_violations_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("  ✓ Violations analysis")

    # 9. Violation Correlation Scatter
    plt.figure(figsize=(10, 6))
    plt.scatter(df['commit_count'], df['violations_count'],
               alpha=0.6, s=100, edgecolors='black', linewidth=0.5, c='coral')
    plt.xlabel('Commit Count')
    plt.ylabel('Violations Count')
    plt.title(f'Commit Count vs Violations (correlation: {df["commit_count"].corr(df["violations_count"]):.3f})')
    plt.xscale('log')
    plt.yscale('log')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(VIZ_DIR / '09_violation_correlation.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("  ✓ Violation correlation")

    # 10. LOC per Minute Efficiency
    plt.figure(figsize=(10, 6))
    valid_efficiency = df[df['loc_per_minute'] < 1000]['loc_per_minute']  # Remove extreme outliers for visualization
    plt.hist(valid_efficiency, bins=30, edgecolor='black', alpha=0.7)
    plt.axvline(valid_efficiency.mean(), color='red', linestyle='--',
               label=f'Mean: {valid_efficiency.mean():.1f} LOC/min')
    plt.axvline(valid_efficiency.median(), color='green', linestyle='--',
               label=f'Median: {valid_efficiency.median():.1f} LOC/min')
    plt.xlabel('Lines of Code per Minute')
    plt.ylabel('Frequency')
    plt.title('Code Generation Efficiency (LOC/minute, outliers >1000 excluded)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(VIZ_DIR / '10_efficiency_histogram.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("  ✓ Efficiency histogram")

    # 11. Clean vs Anomalous Comparison
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle('Clean Success vs Pathological Failure Comparison', fontsize=16, fontweight='bold')

    clean_df = df[df['category'] == 'clean_success']
    pathological_df = df[df['category'] == 'pathological_failure']

    metrics = [
        ('duration_s', 'Duration (s)'),
        ('time_to_first_edit_s', 'Time to First Edit (s)'),
        ('commit_count', 'Commit Count'),
        ('patch_size_loc', 'Patch Size (LOC)'),
        ('changed_files_count', 'Files Changed'),
        ('violations_count', 'Violations')
    ]

    for idx, (metric, label) in enumerate(metrics):
        ax = axes[idx // 3, idx % 3]
        data_to_plot = [clean_df[metric].dropna(), pathological_df[metric].dropna()]
        bp = ax.boxplot(data_to_plot, labels=['Clean', 'Pathological'], patch_artist=True)
        bp['boxes'][0].set_facecolor('#2ecc71')
        bp['boxes'][1].set_facecolor('#e74c3c')
        ax.set_ylabel(label)
        ax.set_title(label)
        ax.grid(True, alpha=0.3, axis='y')
        if metric in ['commit_count', 'violations_count', 'changed_files_count']:
            ax.set_yscale('log')

    plt.tight_layout()
    plt.savefig(VIZ_DIR / '11_clean_vs_anomalous.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("  ✓ Clean vs anomalous comparison")

    # 12. Performance Over Time
    plt.figure(figsize=(14, 6))
    df_sorted = df.sort_values('task_number')

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), sharex=True)

    # Duration over time
    ax1.plot(df_sorted['task_number'], df_sorted['duration_s'],
            marker='o', alpha=0.6, linewidth=1, markersize=4)
    ax1.axhline(df['duration_s'].mean(), color='red', linestyle='--', alpha=0.7, label='Mean')
    ax1.set_ylabel('Duration (seconds)')
    ax1.set_title('Task Performance Over Execution Sequence')
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    # Commits over time (log scale)
    colors = ['#2ecc71' if c == 1 else '#e74c3c' if c > 50 else '#3498db'
             for c in df_sorted['commit_count']]
    ax2.scatter(df_sorted['task_number'], df_sorted['commit_count'],
               c=colors, alpha=0.6, s=50, edgecolors='black', linewidth=0.5)
    ax2.set_ylabel('Commit Count (log scale)')
    ax2.set_xlabel('Task Number (chronological order)')
    ax2.set_yscale('log')
    ax2.grid(True, alpha=0.3)

    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#2ecc71', label='1 commit (clean)'),
        Patch(facecolor='#3498db', label='2-50 commits'),
        Patch(facecolor='#e74c3c', label='>50 commits (pathological)')
    ]
    ax2.legend(handles=legend_elements)

    plt.tight_layout()
    plt.savefig(VIZ_DIR / '12_performance_over_time.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("  ✓ Performance over time")

    print(f"\nAll visualizations saved to {VIZ_DIR}/")


def convert_to_native_types(obj):
    """Convert numpy types to native Python types for JSON serialization."""
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {key: convert_to_native_types(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_native_types(item) for item in obj]
    return obj


def export_data(df: pd.DataFrame, stats: Dict[str, Any]):
    """Export data to CSV and JSON formats."""
    print("\nExporting data...")

    # Export full dataset
    df.to_csv(DATA_DIR / 'codex_metrics_all.csv', index=False)
    print(f"  ✓ Exported: codex_metrics_all.csv ({len(df)} rows)")

    # Export by category
    clean_df = df[df['category'] == 'clean_success']
    clean_df.to_csv(DATA_DIR / 'codex_metrics_clean.csv', index=False)
    print(f"  ✓ Exported: codex_metrics_clean.csv ({len(clean_df)} rows)")

    anomalous_df = df[df['category'] == 'pathological_failure']
    anomalous_df.to_csv(DATA_DIR / 'codex_metrics_anomalous.csv', index=False)
    print(f"  ✓ Exported: codex_metrics_anomalous.csv ({len(anomalous_df)} rows)")

    # Export summary statistics (convert numpy types)
    stats_native = convert_to_native_types(stats)
    with open(DATA_DIR / 'codex_summary_statistics.json', 'w') as f:
        json.dump(stats_native, f, indent=2)
    print(f"  ✓ Exported: codex_summary_statistics.json")

    # Export outliers (convert numpy types)
    outliers = {
        'max_commits': df.nlargest(10, 'commit_count')[
            ['task_number', 'commit_count', 'duration_s', 'violations_count']
        ].to_dict('records'),
        'max_violations': df.nlargest(10, 'violations_count')[
            ['task_number', 'violations_count', 'commit_count', 'changed_files_count']
        ].to_dict('records'),
        'max_files': df.nlargest(10, 'changed_files_count')[
            ['task_number', 'changed_files_count', 'violations_count', 'commit_count']
        ].to_dict('records'),
        'instant_edits': df[df['instant_edit']][
            ['task_number', 'time_to_first_edit_s', 'commit_count', 'violations_count']
        ].to_dict('records'),
    }

    outliers_native = convert_to_native_types(outliers)
    with open(DATA_DIR / 'codex_outliers.json', 'w') as f:
        json.dump(outliers_native, f, indent=2)
    print(f"  ✓ Exported: codex_outliers.json")

    print(f"\nAll data files saved to {DATA_DIR}/")


def print_summary(df: pd.DataFrame, stats: Dict[str, Any]):
    """Print a summary of key findings."""
    print("\n" + "="*80)
    print("CODEX ANALYSIS SUMMARY")
    print("="*80)

    print(f"\nDataset Overview:")
    print(f"  Total Tasks: {stats['overview']['total_tasks']}")
    print(f"  Reported Success Rate: {stats['overview']['success_rate']:.1f}%")
    print(f"  Clean Success Rate: {stats['overview']['clean_success_rate']:.1f}%")
    print(f"  Pathological Failure Rate: {stats['overview']['pathological_failure_rate']:.1f}%")

    print(f"\nExecution Metrics:")
    print(f"  Mean Duration: {stats['duration']['mean']:.1f}s")
    print(f"  Median Duration: {stats['duration']['median']:.1f}s")
    print(f"  Duration Range: {stats['duration']['min']:.1f}s - {stats['duration']['max']:.1f}s")

    print(f"\nCode Metrics:")
    print(f"  Total LOC Changed: {stats['patch_size']['total']:,}")
    print(f"  Mean Patch Size: {stats['patch_size']['mean']:.0f} LOC")
    print(f"  Median Patch Size: {stats['patch_size']['median']:.0f} LOC")
    print(f"  Mean Efficiency: {stats['efficiency']['mean_loc_per_minute']:.1f} LOC/min")

    print(f"\nCommit Patterns:")
    print(f"  Mean Commits: {stats['commits']['mean']:.1f}")
    print(f"  Median Commits: {stats['commits']['median']:.0f}")
    print(f"  Max Commits: {stats['commits']['max']:,}")
    print(f"  Single Commit Rate: {stats['commits']['single_commit_rate']:.1f}%")

    print(f"\nQuality Metrics:")
    print(f"  Zero Violations Rate: {stats['violations']['zero_violations_rate']:.1f}%")
    print(f"  Mean Violations: {stats['violations']['mean']:.1f}")
    print(f"  Max Violations: {stats['violations']['max']:,}")

    print(f"\nKey Findings:")
    print(f"  Instant Edit Rate (<1s): {stats['time_to_first_edit']['instant_edit_rate']:.1f}%")
    print(f"  Commits vs Violations Correlation: {stats['correlations']['commits_vs_violations']:.3f}")
    print(f"  Time to First Edit vs Commits Correlation: {stats['correlations']['ttfe_vs_commits']:.3f}")

    print(f"\nOutliers:")
    print(f"  Task {stats['outliers']['max_commits_task']:04d}: {stats['outliers']['max_commits_value']:,} commits")
    print(f"  Task {stats['outliers']['max_violations_task']:04d}: {stats['outliers']['max_violations_value']:,} violations")
    print(f"  Task {stats['outliers']['max_files_task']:04d}: {stats['outliers']['max_files_value']:,} files changed")

    print("\n" + "="*80)


def main():
    """Main execution function."""
    print("="*80)
    print("CODEX vLLM AGENT ANALYSIS")
    print("="*80)
    print(f"Results Directory: {RESULTS_DIR}")
    print(f"Output Directory: {OUTPUT_DIR}")
    print("="*80)

    # Load data
    df = load_all_results()

    # Compute statistics
    print("\nComputing statistics...")
    stats = compute_statistics(df)

    # Generate visualizations
    create_visualizations(df)

    # Export data
    export_data(df, stats)

    # Print summary
    print_summary(df, stats)

    print(f"\n✓ Analysis complete!")
    print(f"  Visualizations: {VIZ_DIR}/")
    print(f"  Data exports: {DATA_DIR}/")
    print(f"\nNext step: Review the comprehensive report in docs/CODEX_ANALYSIS_REPORT.md")


if __name__ == '__main__':
    main()
