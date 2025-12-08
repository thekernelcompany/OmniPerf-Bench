#!/usr/bin/env python3
"""
Universal Run Analyzer for OmniPerf-Bench Agent Results

Analyzes any agent run (Codex, TRAE-Bedrock, TRAE-GPT) and generates
standardized visualizations, statistics, and exports.

Usage:
    python analyze_run.py <run_directory>
    python analyze_run.py perf-agents-bench/state/runs/vllm_core_codex-90a1c13f
    python analyze_run.py --all  # Analyze all major runs

Outputs per run:
    - 12 publication-quality visualizations (300 DPI)
    - CSV exports (all, clean, anomalous)
    - JSON summary statistics
    - Markdown analysis report
"""

import json
import os
import re
import sys
import argparse
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings

warnings.filterwarnings('ignore')

# Base directories
BASE_DIR = Path(__file__).parent
RUNS_DIR = BASE_DIR / "perf-agents-bench" / "state" / "runs"
DOCS_DIR = BASE_DIR / "docs"


def detect_agent_type(journal_data: Dict) -> str:
    """Detect the agent type from journal data."""
    if 'codex_cli' in journal_data:
        return 'codex'
    elif 'trae' in journal_data:
        # Check run directory name for bedrock vs gpt
        return 'trae'
    elif 'openhands' in journal_data:
        return 'openhands'
    return 'unknown'


def detect_repository(journal_data: Dict, run_name: str) -> str:
    """Detect the repository from journal data or run name."""
    task_id = journal_data.get('task_id', '')

    if 'sglang' in task_id.lower() or 'sglang' in run_name.lower():
        return 'sglang'
    elif 'vllm' in task_id.lower() or 'vllm' in run_name.lower():
        return 'vllm'
    elif 'moe' in task_id.lower():
        return 'vllm-moe'
    elif 'chunked' in task_id.lower():
        return 'vllm-attn'
    elif 'prefix' in task_id.lower():
        return 'vllm-prefix'
    return 'unknown'


def get_agent_subtype(run_name: str, agent_type: str) -> str:
    """Get more specific agent subtype."""
    if agent_type == 'codex':
        return 'codex'
    elif agent_type == 'trae':
        if 'claude_sonnet' in run_name.lower():
            return 'trae-bedrock'
        else:
            return 'trae-gpt'
    return agent_type


def get_detailed_label(metadata: Dict, df: pd.DataFrame) -> Dict[str, str]:
    """Generate detailed labels for visualizations."""
    agent = metadata['agent_subtype']
    repo = metadata['repository'].upper()
    total = len(df)
    success_rate = df['success'].mean() * 100
    clean_rate = df['clean_success'].mean() * 100 if 'clean_success' in df.columns else (len(df[df['category'] == 'clean_success']) / len(df) * 100)

    # Determine model name
    if agent == 'codex':
        model = "Codex CLI (Claude)"
        framework = "Codex"
    elif agent == 'trae-bedrock':
        model = "Claude Sonnet 4.5"
        framework = "TRAE Agent"
    elif agent == 'trae-gpt':
        model = "GPT-4o"
        framework = "TRAE Agent"
    else:
        model = "Unknown"
        framework = agent

    return {
        'short': f"{framework} on {repo}",
        'full': f"{framework} ({model}) on {repo}",
        'with_stats': f"{framework} ({model}) on {repo}\n{total} tasks | {success_rate:.1f}% success | {clean_rate:.1f}% clean",
        'title_line1': f"{framework} ({model})",
        'title_line2': f"Repository: {repo} | Tasks: {total} | Success: {success_rate:.1f}% | Clean: {clean_rate:.1f}%",
        'model': model,
        'framework': framework,
        'repo': repo,
        'success_rate': success_rate,
        'clean_rate': clean_rate,
    }


def parse_journal(journal_path: Path) -> Optional[Dict[str, Any]]:
    """Parse a journal.json file and extract all metrics."""
    try:
        with open(journal_path) as f:
            data = json.load(f)

        metrics = data.get('metrics', {})

        # Detect agent type and get duration/returncode
        agent_type = detect_agent_type(data)
        agent_data = data.get('codex_cli', data.get('trae', data.get('openhands', {})))

        # Extract task number from directory name
        task_num_match = re.search(r'-(\d+)$', journal_path.parent.name)
        task_number = int(task_num_match.group(1)) if task_num_match else 0

        return {
            'task_id': data.get('task_id', 'unknown'),
            'task_number': task_number,
            'task_dir': journal_path.parent.name,
            'status': data.get('status', 'unknown'),
            'agent_type': agent_type,
            'duration_s': agent_data.get('duration_s', 0),
            'returncode': agent_data.get('returncode', -1),
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


def load_run_results(run_dir: Path) -> Tuple[pd.DataFrame, Dict[str, str]]:
    """Load all journal.json files from a run and create a DataFrame."""
    results = []
    task_dirs = sorted([d for d in run_dir.iterdir() if d.is_dir()])

    for task_dir in task_dirs:
        journal_path = task_dir / "journal.json"
        if journal_path.exists():
            parsed = parse_journal(journal_path)
            if parsed:
                results.append(parsed)

    if not results:
        raise ValueError(f"No valid journal files found in {run_dir}!")

    df = pd.DataFrame(results)

    # Metadata
    run_name = run_dir.name
    agent_type = df['agent_type'].iloc[0] if len(df) > 0 else 'unknown'
    agent_subtype = get_agent_subtype(run_name, agent_type)
    repository = detect_repository(results[0], run_name) if results else 'unknown'

    metadata = {
        'run_name': run_name,
        'run_dir': str(run_dir),
        'agent_type': agent_type,
        'agent_subtype': agent_subtype,
        'repository': repository,
        'total_tasks': len(df),
    }

    # Add computed fields
    df['success'] = df['status'] == 'success'
    df['loc_per_minute'] = (df['patch_size_loc'] / (df['duration_s'] / 60)).replace([np.inf, -np.inf], 0).fillna(0)
    df['instant_edit'] = df['time_to_first_edit_s'] < 1.0

    # Categorize tasks
    def categorize_task(row):
        if not row['success']:
            return 'failed'
        elif row['commit_count'] == 1 and row['violations_count'] == 0:
            return 'clean_success'
        elif row['commit_count'] > 50 or row['violations_count'] > 100:
            return 'pathological_failure'
        elif row['violations_count'] > 0:
            return 'success_with_violations'
        else:
            return 'normal_success'

    df['category'] = df.apply(categorize_task, axis=1)

    return df, metadata


def compute_statistics(df: pd.DataFrame, metadata: Dict) -> Dict[str, Any]:
    """Compute comprehensive statistics for the dataset."""
    success_df = df[df['success']]

    stats = {
        'metadata': metadata,
        'overview': {
            'total_tasks': len(df),
            'success_count': df['success'].sum(),
            'success_rate': (df['success'].sum() / len(df) * 100) if len(df) > 0 else 0,
            'clean_success_count': len(df[df['category'] == 'clean_success']),
            'clean_success_rate': (len(df[df['category'] == 'clean_success']) / len(df) * 100) if len(df) > 0 else 0,
            'pathological_count': len(df[df['category'] == 'pathological_failure']),
            'pathological_rate': (len(df[df['category'] == 'pathological_failure']) / len(df) * 100) if len(df) > 0 else 0,
            'failed_count': len(df[df['category'] == 'failed']),
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
            'instant_edit_rate': (df['instant_edit'].sum() / len(df) * 100) if len(df) > 0 else 0,
        },
        'commits': {
            'mean': df['commit_count'].mean(),
            'median': df['commit_count'].median(),
            'max': df['commit_count'].max(),
            'single_commit_rate': (len(df[df['commit_count'] == 1]) / len(df) * 100) if len(df) > 0 else 0,
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
            'zero_violations_rate': (len(df[df['violations_count'] == 0]) / len(df) * 100) if len(df) > 0 else 0,
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
    }

    # Add outliers if data exists
    if len(df) > 0:
        stats['outliers'] = {
            'max_commits_task': int(df.loc[df['commit_count'].idxmax()]['task_number']),
            'max_commits_value': int(df['commit_count'].max()),
            'max_violations_task': int(df.loc[df['violations_count'].idxmax()]['task_number']),
            'max_violations_value': int(df['violations_count'].max()),
            'max_files_task': int(df.loc[df['changed_files_count'].idxmax()]['task_number']),
            'max_files_value': int(df['changed_files_count'].max()),
        }

    return stats


def create_visualizations(df: pd.DataFrame, metadata: Dict, output_dir: Path):
    """Generate all 12 publication-quality visualizations."""
    viz_dir = output_dir / "visualizations"
    viz_dir.mkdir(parents=True, exist_ok=True)

    # Visualization settings
    plt.rcParams['figure.dpi'] = 300
    plt.rcParams['savefig.dpi'] = 300
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.size'] = 10
    sns.set_palette("husl")

    run_name = metadata['run_name']

    # Get detailed labels for titles
    labels = get_detailed_label(metadata, df)
    title_line1 = labels['title_line1']
    title_line2 = labels['title_line2']

    def make_title(chart_name: str) -> str:
        """Create a two-line title with chart name, agent info, and stats."""
        return f"{chart_name}\n{title_line1} | {labels['repo']} | {len(df)} tasks | {labels['success_rate']:.0f}% success"

    print("\nGenerating visualizations...")

    # 1. Duration Distribution
    plt.figure(figsize=(10, 6))
    plt.hist(df['duration_s'], bins=30, edgecolor='black', alpha=0.7)
    plt.axvline(df['duration_s'].mean(), color='red', linestyle='--', label=f'Mean: {df["duration_s"].mean():.1f}s')
    plt.axvline(df['duration_s'].median(), color='green', linestyle='--', label=f'Median: {df["duration_s"].median():.1f}s')
    plt.xlabel('Duration (seconds)')
    plt.ylabel('Frequency')
    plt.title(make_title('Duration Distribution'), fontsize=11, fontweight='bold')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(viz_dir / '01_duration_distribution.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("  1/12 Duration distribution")

    # 2. Duration vs Commits Scatter
    plt.figure(figsize=(10, 6))
    scatter = plt.scatter(df['duration_s'], df['commit_count'],
                         c=df['violations_count'], cmap='YlOrRd',
                         alpha=0.6, s=100, edgecolors='black', linewidth=0.5)
    plt.colorbar(scatter, label='Violations Count')
    plt.xlabel('Duration (seconds)')
    plt.ylabel('Commit Count')
    plt.title(make_title('Duration vs Commits'), fontsize=11, fontweight='bold')
    if df['commit_count'].max() > 10:
        plt.yscale('log')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(viz_dir / '02_duration_vs_commits.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("  2/12 Duration vs commits")

    # 3. Time to First Edit Distribution
    plt.figure(figsize=(10, 6))
    plt.hist(df['time_to_first_edit_s'], bins=50, edgecolor='black', alpha=0.7)
    plt.axvline(1.0, color='red', linestyle='--', linewidth=2, label='1s threshold')
    instant = len(df[df['time_to_first_edit_s'] < 1.0])
    plt.text(0.5, plt.ylim()[1] * 0.9, f'{instant} tasks (<1s)', fontsize=12, color='red')
    plt.xlabel('Time to First Edit (seconds)')
    plt.ylabel('Frequency')
    plt.title(make_title('Time to First Edit'), fontsize=11, fontweight='bold')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(viz_dir / '03_time_to_first_edit.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("  3/12 Time to first edit")

    # 4. Success Categories Pie Chart
    plt.figure(figsize=(10, 8))
    category_counts = df['category'].value_counts()
    colors = {
        'clean_success': '#2ecc71',
        'normal_success': '#3498db',
        'success_with_violations': '#f39c12',
        'pathological_failure': '#e74c3c',
        'failed': '#95a5a6'
    }
    pie_colors = [colors.get(cat, '#cccccc') for cat in category_counts.index]

    plt.pie(category_counts.values, labels=[
        f"{cat.replace('_', ' ').title()}\n({count}, {count/len(df)*100:.1f}%)"
        for cat, count in zip(category_counts.index, category_counts.values)
    ], autopct='%1.1f%%', startangle=90, colors=pie_colors)
    plt.title(make_title('Task Categories'), fontsize=11, fontweight='bold')
    plt.tight_layout()
    plt.savefig(viz_dir / '04_success_categories.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("  4/12 Success categories")

    # 5. Patch Size Distribution
    plt.figure(figsize=(10, 6))
    plt.hist(df['patch_size_loc'], bins=40, edgecolor='black', alpha=0.7)
    plt.axvline(df['patch_size_loc'].mean(), color='red', linestyle='--', label=f'Mean: {df["patch_size_loc"].mean():.0f}')
    plt.axvline(df['patch_size_loc'].median(), color='green', linestyle='--', label=f'Median: {df["patch_size_loc"].median():.0f}')
    plt.xlabel('Patch Size (LOC)')
    plt.ylabel('Frequency')
    plt.title(make_title('Patch Size Distribution'), fontsize=11, fontweight='bold')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(viz_dir / '05_patch_size_distribution.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("  5/12 Patch size")

    # 6. Files Changed Distribution
    plt.figure(figsize=(10, 6))
    plt.hist(df['changed_files_count'], bins=40, edgecolor='black', alpha=0.7)
    plt.xlabel('Files Changed')
    plt.ylabel('Frequency')
    plt.title(make_title('Files Changed Distribution'), fontsize=11, fontweight='bold')
    if df['changed_files_count'].max() > 100:
        plt.yscale('log')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(viz_dir / '06_files_changed_distribution.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("  6/12 Files changed")

    # 7. Commits Distribution
    plt.figure(figsize=(10, 6))
    plt.hist(df['commit_count'], bins=50, edgecolor='black', alpha=0.7)
    plt.xlabel('Commit Count')
    plt.ylabel('Frequency')
    plt.title(make_title('Commit Distribution'), fontsize=11, fontweight='bold')
    if df['commit_count'].max() > 10:
        plt.yscale('log')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(viz_dir / '07_commits_distribution.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("  7/12 Commits distribution")

    # 8. Violations Analysis
    plt.figure(figsize=(12, 6))
    violation_bins = pd.cut(df['violations_count'],
                           bins=[-1, 0, 10, 100, 1000, float('inf')],
                           labels=['0', '1-10', '11-100', '101-1000', '>1000'])
    violation_counts = violation_bins.value_counts().sort_index()

    bars = plt.bar(range(len(violation_counts)), violation_counts.values,
                   color=['#2ecc71', '#3498db', '#f39c12', '#e67e22', '#e74c3c'])
    plt.xticks(range(len(violation_counts)), violation_counts.index)
    plt.xlabel('Violations Range')
    plt.ylabel('Number of Tasks')
    plt.title(make_title('Violations Distribution'), fontsize=11, fontweight='bold')

    for bar, count in zip(bars, violation_counts.values):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                str(count), ha='center', va='bottom', fontweight='bold')

    plt.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig(viz_dir / '08_violations_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("  8/12 Violations analysis")

    # 9. Commits vs Violations Correlation
    plt.figure(figsize=(10, 6))
    plt.scatter(df['commit_count'], df['violations_count'],
               alpha=0.6, s=100, edgecolors='black', linewidth=0.5, c='coral')
    plt.xlabel('Commit Count')
    plt.ylabel('Violations Count')
    corr = df['commit_count'].corr(df['violations_count'])
    plt.title(f"{make_title('Commits vs Violations')}\nCorrelation: r={corr:.3f}", fontsize=11, fontweight='bold')
    if df['commit_count'].max() > 10:
        plt.xscale('log')
    if df['violations_count'].max() > 10:
        plt.yscale('log')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(viz_dir / '09_commits_vs_violations.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("  9/12 Commits vs violations")

    # 10. Efficiency (LOC per minute)
    plt.figure(figsize=(10, 6))
    valid_efficiency = df[df['loc_per_minute'] < 1000]['loc_per_minute']
    if len(valid_efficiency) > 0:
        plt.hist(valid_efficiency, bins=30, edgecolor='black', alpha=0.7)
        plt.axvline(valid_efficiency.mean(), color='red', linestyle='--',
                   label=f'Mean: {valid_efficiency.mean():.1f}')
        plt.xlabel('LOC per Minute')
        plt.ylabel('Frequency')
        plt.title(make_title('Code Generation Efficiency'), fontsize=11, fontweight='bold')
        plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(viz_dir / '10_efficiency.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("  10/12 Efficiency")

    # 11. Clean vs Pathological Comparison
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle(make_title('Clean vs Pathological Comparison'), fontsize=12, fontweight='bold')

    clean_df = df[df['category'] == 'clean_success']
    pathological_df = df[df['category'] == 'pathological_failure']

    metrics = [
        ('duration_s', 'Duration (s)'),
        ('time_to_first_edit_s', 'TTFE (s)'),
        ('commit_count', 'Commits'),
        ('patch_size_loc', 'Patch Size'),
        ('changed_files_count', 'Files Changed'),
        ('violations_count', 'Violations')
    ]

    for idx, (metric, label) in enumerate(metrics):
        ax = axes[idx // 3, idx % 3]
        data_to_plot = []
        labels_to_use = []

        if len(clean_df) > 0:
            data_to_plot.append(clean_df[metric].dropna())
            labels_to_use.append('Clean')
        if len(pathological_df) > 0:
            data_to_plot.append(pathological_df[metric].dropna())
            labels_to_use.append('Patho')

        if data_to_plot:
            bp = ax.boxplot(data_to_plot, labels=labels_to_use, patch_artist=True)
            if len(bp['boxes']) >= 1:
                bp['boxes'][0].set_facecolor('#2ecc71')
            if len(bp['boxes']) >= 2:
                bp['boxes'][1].set_facecolor('#e74c3c')

        ax.set_ylabel(label)
        ax.set_title(label)
        ax.grid(True, alpha=0.3, axis='y')
        if metric in ['commit_count', 'violations_count', 'changed_files_count']:
            max_val = df[metric].max()
            if max_val > 10:
                ax.set_yscale('log')

    plt.tight_layout()
    plt.savefig(viz_dir / '11_clean_vs_pathological.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("  11/12 Clean vs pathological")

    # 12. Performance Over Time
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), sharex=True)
    fig.suptitle(make_title('Performance Over Sequence'), fontsize=12, fontweight='bold')
    df_sorted = df.sort_values('task_number')

    ax1.plot(df_sorted['task_number'], df_sorted['duration_s'],
            marker='o', alpha=0.6, linewidth=1, markersize=4)
    ax1.axhline(df['duration_s'].mean(), color='red', linestyle='--', alpha=0.7, label='Mean')
    ax1.set_ylabel('Duration (s)')
    ax1.set_title('Duration per Task')
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    # Color by commit count
    colors = ['#2ecc71' if c == 1 else '#e74c3c' if c > 50 else '#3498db'
             for c in df_sorted['commit_count']]
    ax2.scatter(df_sorted['task_number'], df_sorted['commit_count'],
               c=colors, alpha=0.6, s=50, edgecolors='black', linewidth=0.5)
    ax2.set_ylabel('Commits')
    ax2.set_xlabel('Task Number')
    if df['commit_count'].max() > 10:
        ax2.set_yscale('log')
    ax2.grid(True, alpha=0.3)

    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#2ecc71', label='1 commit'),
        Patch(facecolor='#3498db', label='2-50'),
        Patch(facecolor='#e74c3c', label='>50')
    ]
    ax2.legend(handles=legend_elements)

    plt.tight_layout()
    plt.savefig(viz_dir / '12_performance_over_time.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("  12/12 Performance over time")

    print(f"\nVisualizations saved to {viz_dir}/")


def convert_to_native_types(obj):
    """Convert numpy types to native Python types for JSON serialization."""
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, np.bool_):
        return bool(obj)
    elif isinstance(obj, dict):
        return {key: convert_to_native_types(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_native_types(item) for item in obj]
    elif pd.isna(obj):
        return None
    return obj


def export_data(df: pd.DataFrame, stats: Dict[str, Any], metadata: Dict, output_dir: Path):
    """Export data to CSV and JSON formats."""
    data_dir = output_dir / "data"
    data_dir.mkdir(parents=True, exist_ok=True)

    prefix = metadata['run_name'].replace('-', '_')

    print("\nExporting data...")

    # Full dataset
    df.to_csv(data_dir / f'{prefix}_metrics_all.csv', index=False)
    print(f"  {prefix}_metrics_all.csv ({len(df)} rows)")

    # Clean success only
    clean_df = df[df['category'] == 'clean_success']
    clean_df.to_csv(data_dir / f'{prefix}_metrics_clean.csv', index=False)
    print(f"  {prefix}_metrics_clean.csv ({len(clean_df)} rows)")

    # Anomalous (pathological + failed)
    anomalous_df = df[df['category'].isin(['pathological_failure', 'failed'])]
    anomalous_df.to_csv(data_dir / f'{prefix}_metrics_anomalous.csv', index=False)
    print(f"  {prefix}_metrics_anomalous.csv ({len(anomalous_df)} rows)")

    # Summary statistics
    stats_native = convert_to_native_types(stats)
    with open(data_dir / f'{prefix}_summary.json', 'w') as f:
        json.dump(stats_native, f, indent=2)
    print(f"  {prefix}_summary.json")

    # Outliers
    if len(df) > 0:
        outliers = {
            'max_commits': df.nlargest(min(10, len(df)), 'commit_count')[
                ['task_number', 'task_dir', 'commit_count', 'duration_s', 'violations_count']
            ].to_dict('records'),
            'max_violations': df.nlargest(min(10, len(df)), 'violations_count')[
                ['task_number', 'task_dir', 'violations_count', 'commit_count']
            ].to_dict('records'),
            'instant_edits': df[df['instant_edit']][
                ['task_number', 'task_dir', 'time_to_first_edit_s', 'commit_count', 'category']
            ].to_dict('records'),
        }
        outliers_native = convert_to_native_types(outliers)
        with open(data_dir / f'{prefix}_outliers.json', 'w') as f:
            json.dump(outliers_native, f, indent=2)
        print(f"  {prefix}_outliers.json")

    print(f"\nData exported to {data_dir}/")


def generate_report(df: pd.DataFrame, stats: Dict[str, Any], metadata: Dict, output_dir: Path):
    """Generate a markdown analysis report."""
    report_path = output_dir / "README.md"

    overview = stats['overview']
    duration = stats['duration']
    commits = stats['commits']
    violations = stats['violations']
    ttfe = stats['time_to_first_edit']

    report = f"""# Analysis Report: {metadata['run_name']}

**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Run Metadata

| Property | Value |
|----------|-------|
| Run ID | `{metadata['run_name']}` |
| Agent | {metadata['agent_subtype']} |
| Repository | {metadata['repository']} |
| Total Tasks | {metadata['total_tasks']} |

## Summary Statistics

### Success Rates

| Metric | Value |
|--------|-------|
| Total Tasks | {overview['total_tasks']} |
| Success Rate | {overview['success_rate']:.1f}% ({overview['success_count']}/{overview['total_tasks']}) |
| Clean Success Rate | {overview['clean_success_rate']:.1f}% ({overview['clean_success_count']}) |
| Pathological Rate | {overview['pathological_rate']:.1f}% ({overview['pathological_count']}) |
| Failed | {overview['failed_count']} |

### Duration

| Metric | Value |
|--------|-------|
| Mean | {duration['mean']:.1f}s |
| Median | {duration['median']:.1f}s |
| Std Dev | {duration['std']:.1f}s |
| Range | {duration['min']:.1f}s - {duration['max']:.1f}s |

### Commits

| Metric | Value |
|--------|-------|
| Mean | {commits['mean']:.1f} |
| Median | {commits['median']:.0f} |
| Max | {commits['max']:,} |
| Single Commit Rate | {commits['single_commit_rate']:.1f}% |

### Violations

| Metric | Value |
|--------|-------|
| Mean | {violations['mean']:.1f} |
| Median | {violations['median']:.0f} |
| Max | {violations['max']:,} |
| Zero Violations Rate | {violations['zero_violations_rate']:.1f}% |

### Time to First Edit

| Metric | Value |
|--------|-------|
| Mean | {ttfe['mean']:.1f}s |
| Median | {ttfe['median']:.1f}s |
| Instant Edit Rate (<1s) | {ttfe['instant_edit_rate']:.1f}% ({ttfe['instant_edit_count']} tasks) |

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

- `data/{metadata['run_name'].replace('-', '_')}_metrics_all.csv` - All tasks
- `data/{metadata['run_name'].replace('-', '_')}_metrics_clean.csv` - Clean successes only
- `data/{metadata['run_name'].replace('-', '_')}_metrics_anomalous.csv` - Pathological/failed only
- `data/{metadata['run_name'].replace('-', '_')}_summary.json` - Summary statistics
- `data/{metadata['run_name'].replace('-', '_')}_outliers.json` - Outlier details
"""

    with open(report_path, 'w') as f:
        f.write(report)

    print(f"\nReport saved to {report_path}")


def analyze_run(run_dir: Path, output_base: Path = None):
    """Analyze a single run and generate all outputs."""
    if not run_dir.exists():
        print(f"Error: Run directory not found: {run_dir}")
        return None

    print("=" * 80)
    print(f"ANALYZING: {run_dir.name}")
    print("=" * 80)

    # Load data
    df, metadata = load_run_results(run_dir)

    print(f"\nRun: {metadata['run_name']}")
    print(f"Agent: {metadata['agent_subtype']}")
    print(f"Repository: {metadata['repository']}")
    print(f"Tasks: {len(df)}")
    print(f"Success: {df['success'].sum()}/{len(df)} ({df['success'].mean()*100:.1f}%)")
    print(f"Clean: {len(df[df['category'] == 'clean_success'])} ({len(df[df['category'] == 'clean_success'])/len(df)*100:.1f}%)")

    # Setup output directory
    if output_base is None:
        output_base = DOCS_DIR
    output_dir = output_base / f"{metadata['run_name']}_analysis"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Compute statistics
    stats = compute_statistics(df, metadata)

    # Generate outputs
    create_visualizations(df, metadata, output_dir)
    export_data(df, stats, metadata, output_dir)
    generate_report(df, stats, metadata, output_dir)

    print("\n" + "=" * 80)
    print(f"Analysis complete: {output_dir}")
    print("=" * 80)

    return stats


def get_major_runs(min_tasks: int = 10) -> List[Path]:
    """Get list of major runs (with at least min_tasks tasks)."""
    major_runs = []

    for run_dir in sorted(RUNS_DIR.iterdir()):
        if not run_dir.is_dir() or run_dir.name.startswith('.'):
            continue

        journal_count = len(list(run_dir.glob("*/journal.json")))
        if journal_count >= min_tasks:
            major_runs.append(run_dir)

    return major_runs


def main():
    parser = argparse.ArgumentParser(description='Analyze OmniPerf-Bench agent runs')
    parser.add_argument('run_dir', nargs='?', help='Run directory to analyze')
    parser.add_argument('--all', action='store_true', help='Analyze all major runs')
    parser.add_argument('--list', action='store_true', help='List available runs')
    parser.add_argument('--min-tasks', type=int, default=10, help='Minimum tasks for --all')
    parser.add_argument('--output', type=str, help='Output base directory')

    args = parser.parse_args()

    if args.list:
        print("Available runs:\n")
        for run_dir in get_major_runs(args.min_tasks):
            journal_count = len(list(run_dir.glob("*/journal.json")))
            print(f"  {run_dir.name} ({journal_count} tasks)")
        return

    if args.all:
        runs = get_major_runs(args.min_tasks)
        print(f"Analyzing {len(runs)} runs with >= {args.min_tasks} tasks\n")

        results = []
        for run_dir in runs:
            try:
                stats = analyze_run(run_dir)
                if stats:
                    results.append(stats)
            except Exception as e:
                print(f"Error analyzing {run_dir.name}: {e}")

        print(f"\n\nCompleted: {len(results)}/{len(runs)} runs")
        return

    if args.run_dir:
        run_path = Path(args.run_dir)
        if not run_path.is_absolute():
            run_path = Path.cwd() / run_path

        output_base = Path(args.output) if args.output else None
        analyze_run(run_path, output_base)
    else:
        parser.print_help()


if __name__ == '__main__':
    main()
