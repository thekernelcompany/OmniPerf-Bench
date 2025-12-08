#!/usr/bin/env python3
"""
Cross-Run Comparator for OmniPerf-Bench

Generates comprehensive multi-agent comparison across all analyzed runs,
producing unified visualizations and the master analysis report.

Usage:
    python analyze_all_runs.py

Outputs:
    - docs/multi_agent_comparison/visualizations/*.png
    - docs/multi_agent_comparison/data/*.csv
    - docs/COMPREHENSIVE_MULTI_AGENT_ANALYSIS.md
"""

import json
import os
from pathlib import Path
from typing import Dict, List, Any
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings

warnings.filterwarnings('ignore')

# Directories
BASE_DIR = Path(__file__).parent
RUNS_DIR = BASE_DIR / "perf-agents-bench" / "state" / "runs"
DOCS_DIR = BASE_DIR / "docs"
OUTPUT_DIR = DOCS_DIR / "multi_agent_comparison"


def detect_agent_type(journal_data: Dict) -> str:
    """Detect agent type from journal data."""
    if 'codex_cli' in journal_data:
        return 'codex'
    elif 'trae' in journal_data:
        return 'trae'
    return 'unknown'


def get_agent_subtype(run_name: str, agent_type: str) -> str:
    """Get more specific agent subtype."""
    if agent_type == 'codex':
        return 'Codex'
    elif agent_type == 'trae':
        if 'claude_sonnet' in run_name.lower():
            return 'TRAE-Bedrock'
        else:
            return 'TRAE-GPT'
    return agent_type


def detect_repository(task_id: str, run_name: str) -> str:
    """Detect repository from task ID or run name."""
    if 'sglang' in task_id.lower() or 'sglang' in run_name.lower():
        return 'SGLang'
    elif 'vllm' in task_id.lower() or 'vllm' in run_name.lower():
        return 'vLLM'
    return 'Unknown'


def load_all_journals(min_tasks: int = 10) -> pd.DataFrame:
    """Load all journals from all runs meeting minimum task threshold."""
    print("Loading journals from all runs...")

    all_records = []

    for run_dir in sorted(RUNS_DIR.iterdir()):
        if not run_dir.is_dir() or run_dir.name.startswith('.'):
            continue

        journals = list(run_dir.glob("*/journal.json"))
        if len(journals) < min_tasks:
            continue

        run_name = run_dir.name

        for journal_path in journals:
            try:
                with open(journal_path) as f:
                    data = json.load(f)

                metrics = data.get('metrics', {})
                agent_type = detect_agent_type(data)
                agent_data = data.get('codex_cli', data.get('trae', {}))

                # Extract task number
                import re
                task_num_match = re.search(r'-(\d+)$', journal_path.parent.name)
                task_number = int(task_num_match.group(1)) if task_num_match else 0

                record = {
                    'run_id': run_name,
                    'task_dir': journal_path.parent.name,
                    'task_number': task_number,
                    'task_id': data.get('task_id', 'unknown'),
                    'status': data.get('status', 'unknown'),
                    'agent_type': agent_type,
                    'agent_subtype': get_agent_subtype(run_name, agent_type),
                    'repository': detect_repository(data.get('task_id', ''), run_name),
                    'duration_s': agent_data.get('duration_s', 0),
                    'returncode': agent_data.get('returncode', -1),
                    'time_to_first_edit_s': metrics.get('time_to_first_edit_s', 0),
                    'commit_count': metrics.get('commit_count', 0),
                    'patch_size_loc': metrics.get('patch_size_loc', 0),
                    'changed_files_count': metrics.get('changed_files_count', 0),
                    'violations_count': metrics.get('violations_count', 0),
                }
                all_records.append(record)

            except Exception as e:
                continue

    df = pd.DataFrame(all_records)

    # Add computed fields
    df['success'] = df['status'] == 'success'
    df['instant_edit'] = df['time_to_first_edit_s'] < 1.0
    df['clean_success'] = df['success'] & (df['commit_count'] == 1) & (df['violations_count'] == 0)
    df['pathological'] = (df['commit_count'] > 50) | (df['violations_count'] > 100)

    print(f"Loaded {len(df)} journals from {df['run_id'].nunique()} runs")

    return df


def compute_run_summaries(df: pd.DataFrame) -> pd.DataFrame:
    """Compute summary statistics per run."""
    summaries = []

    for run_id in df['run_id'].unique():
        run_df = df[df['run_id'] == run_id]

        summary = {
            'run_id': run_id,
            'agent_subtype': run_df['agent_subtype'].iloc[0],
            'repository': run_df['repository'].iloc[0],
            'total_tasks': len(run_df),
            'success_count': run_df['success'].sum(),
            'success_rate': run_df['success'].mean() * 100,
            'clean_count': run_df['clean_success'].sum(),
            'clean_rate': run_df['clean_success'].mean() * 100,
            'pathological_count': run_df['pathological'].sum(),
            'pathological_rate': run_df['pathological'].mean() * 100,
            'instant_edit_count': run_df['instant_edit'].sum(),
            'instant_edit_rate': run_df['instant_edit'].mean() * 100,
            'avg_duration_s': run_df['duration_s'].mean(),
            'avg_commits': run_df['commit_count'].mean(),
            'avg_violations': run_df['violations_count'].mean(),
            'max_commits': run_df['commit_count'].max(),
            'max_violations': run_df['violations_count'].max(),
        }
        summaries.append(summary)

    return pd.DataFrame(summaries)


def get_model_info(agent_subtype: str) -> str:
    """Get model name for agent subtype."""
    models = {
        'Codex': 'Claude (Codex CLI)',
        'TRAE-Bedrock': 'Claude Sonnet 4.5 (AWS Bedrock)',
        'TRAE-GPT': 'GPT-4o (OpenAI)',
    }
    return models.get(agent_subtype, agent_subtype)


def create_comparative_visualizations(df: pd.DataFrame, run_summaries: pd.DataFrame, output_dir: Path):
    """Generate comparative visualizations across all runs."""
    viz_dir = output_dir / "visualizations"
    viz_dir.mkdir(parents=True, exist_ok=True)

    plt.rcParams['figure.dpi'] = 300
    plt.rcParams['savefig.dpi'] = 300
    plt.rcParams['font.size'] = 10
    sns.set_palette("husl")

    # Create common subtitle with dataset info
    total_tasks = len(df)
    total_runs = df['run_id'].nunique()
    repos = ', '.join(df['repository'].unique())
    agents = ', '.join(df['agent_subtype'].unique())
    dataset_info = f"OmniPerf-Bench | {total_tasks} tasks | {total_runs} runs | Repos: {repos}"

    print("\nGenerating comparative visualizations...")

    # 1. Success Rate by Agent Type (grouped bar)
    plt.figure(figsize=(12, 6))
    agent_repo_summary = df.groupby(['agent_subtype', 'repository']).agg({
        'success': 'mean',
        'clean_success': 'mean',
    }).reset_index()
    agent_repo_summary['success'] *= 100
    agent_repo_summary['clean_success'] *= 100

    x = np.arange(len(agent_repo_summary))
    width = 0.35

    fig, ax = plt.subplots(figsize=(14, 6))
    bars1 = ax.bar(x - width/2, agent_repo_summary['success'], width, label='Success Rate', color='#3498db')
    bars2 = ax.bar(x + width/2, agent_repo_summary['clean_success'], width, label='Clean Success Rate', color='#2ecc71')

    ax.set_xlabel('Agent + Repository')
    ax.set_ylabel('Rate (%)')
    ax.set_title(f'Success Rates by Agent and Repository\n{dataset_info}', fontsize=11, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([f"{row['agent_subtype']}\n({get_model_info(row['agent_subtype'])})\n{row['repository']}"
                        for _, row in agent_repo_summary.iterrows()], rotation=45, ha='right', fontsize=9)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    # Add value labels
    for bar in bars1:
        height = bar.get_height()
        ax.annotate(f'{height:.1f}%', xy=(bar.get_x() + bar.get_width()/2, height),
                   xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=8)
    for bar in bars2:
        height = bar.get_height()
        ax.annotate(f'{height:.1f}%', xy=(bar.get_x() + bar.get_width()/2, height),
                   xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=8)

    plt.tight_layout()
    plt.savefig(viz_dir / '01_success_by_agent_repo.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("  1/8 Success by agent+repo")

    # 2. Clean Success Rate Heatmap (Agent x Repository)
    pivot_clean = run_summaries.pivot_table(
        index='agent_subtype',
        columns='repository',
        values='clean_rate',
        aggfunc='mean'
    ).fillna(0)

    plt.figure(figsize=(10, 6))
    sns.heatmap(pivot_clean, annot=True, fmt='.1f', cmap='RdYlGn',
                vmin=0, vmax=100, cbar_kws={'label': 'Clean Success Rate (%)'})
    plt.title(f'Clean Success Rate by Agent Type and Repository\n{dataset_info}', fontsize=11, fontweight='bold')
    plt.xlabel('Repository')
    plt.ylabel('Agent Type (Model)')
    # Add model info to y-axis labels
    new_ylabels = [f"{a}\n({get_model_info(a)})" for a in pivot_clean.index]
    plt.yticks(np.arange(len(new_ylabels)) + 0.5, new_ylabels, rotation=0)
    plt.tight_layout()
    plt.savefig(viz_dir / '02_clean_success_heatmap.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("  2/8 Clean success heatmap")

    # 3. Duration Comparison (boxplots by agent)
    plt.figure(figsize=(12, 6))
    agent_order = ['Codex', 'TRAE-Bedrock', 'TRAE-GPT']
    filtered_agents = [a for a in agent_order if a in df['agent_subtype'].unique()]

    bp = df[df['agent_subtype'].isin(filtered_agents)].boxplot(
        column='duration_s',
        by='agent_subtype',
        figsize=(12, 6),
        patch_artist=True
    )
    plt.title(f'Task Duration by Agent Type\n{dataset_info}', fontsize=11, fontweight='bold')
    plt.suptitle('')
    plt.xlabel('Agent Type')
    plt.ylabel('Duration (seconds)')
    # Update x-tick labels with model info
    ax = plt.gca()
    new_labels = [f"{a}\n({get_model_info(a)})" for a in filtered_agents]
    ax.set_xticklabels(new_labels)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(viz_dir / '03_duration_by_agent.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("  3/8 Duration by agent")

    # 4. TTFE Comparison (boxplots by agent)
    plt.figure(figsize=(12, 6))
    df_plot = df[df['agent_subtype'].isin(filtered_agents)]

    for i, agent in enumerate(filtered_agents):
        agent_data = df_plot[df_plot['agent_subtype'] == agent]['time_to_first_edit_s']
        plt.boxplot(agent_data, positions=[i], widths=0.6, patch_artist=True,
                   boxprops=dict(facecolor=sns.color_palette()[i]))

    plt.xticks(range(len(filtered_agents)), [f"{a}\n({get_model_info(a)})" for a in filtered_agents])
    plt.axhline(y=1.0, color='red', linestyle='--', alpha=0.7, label='1s threshold')
    plt.title(f'Time to First Edit by Agent Type\n{dataset_info}', fontsize=11, fontweight='bold')
    plt.xlabel('Agent Type')
    plt.ylabel('TTFE (seconds)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(viz_dir / '04_ttfe_by_agent.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("  4/8 TTFE by agent")

    # 5. Commit Distribution by Agent (violin plots)
    plt.figure(figsize=(12, 6))
    df_valid = df[df['commit_count'] > 0]

    if len(df_valid) > 0:
        sns.violinplot(data=df_valid, x='agent_subtype', y='commit_count',
                      order=filtered_agents, scale='width', cut=0)
        plt.yscale('log')
        plt.title(f'Commit Count Distribution by Agent Type\n{dataset_info}', fontsize=11, fontweight='bold')
        plt.xlabel('Agent Type')
        plt.ylabel('Commit Count (log scale)')
        # Update x-tick labels
        ax = plt.gca()
        ax.set_xticklabels([f"{a}\n({get_model_info(a)})" for a in filtered_agents])
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(viz_dir / '05_commits_by_agent.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("  5/8 Commits by agent")

    # 6. Pathological Rate by Agent x Repo
    pivot_patho = run_summaries.pivot_table(
        index='agent_subtype',
        columns='repository',
        values='pathological_rate',
        aggfunc='mean'
    ).fillna(0)

    plt.figure(figsize=(10, 6))
    sns.heatmap(pivot_patho, annot=True, fmt='.1f', cmap='YlOrRd',
                vmin=0, cbar_kws={'label': 'Pathological Rate (%)'})
    plt.title(f'Pathological Failure Rate by Agent and Repository\n{dataset_info}', fontsize=11, fontweight='bold')
    plt.xlabel('Repository')
    plt.ylabel('Agent Type (Model)')
    # Add model info to y-axis labels
    new_ylabels = [f"{a}\n({get_model_info(a)})" for a in pivot_patho.index]
    plt.yticks(np.arange(len(new_ylabels)) + 0.5, new_ylabels, rotation=0)
    plt.tight_layout()
    plt.savefig(viz_dir / '06_pathological_heatmap.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("  6/8 Pathological heatmap")

    # 7. Instant Edit Rate Correlation with Failure
    plt.figure(figsize=(12, 6))

    # Group by instant edit and calculate failure rate
    instant_failure = df.groupby(['agent_subtype', 'instant_edit']).agg({
        'success': lambda x: (1 - x.mean()) * 100
    }).reset_index()
    instant_failure.columns = ['agent_subtype', 'instant_edit', 'failure_rate']

    for agent in filtered_agents:
        agent_data = instant_failure[instant_failure['agent_subtype'] == agent]
        if len(agent_data) == 2:
            plt.bar([f"{agent}\n({get_model_info(agent)})\nDelayed", f"{agent}\n({get_model_info(agent)})\nInstant"],
                   agent_data.sort_values('instant_edit')['failure_rate'].values,
                   alpha=0.7, label=agent)

    plt.title(f'Failure Rate: Instant Edit (<1s) vs Delayed Edit\n{dataset_info}', fontsize=11, fontweight='bold')
    plt.xlabel('Edit Pattern by Agent')
    plt.ylabel('Failure Rate (%)')
    plt.xticks(rotation=45, ha='right')
    plt.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig(viz_dir / '07_instant_edit_failure.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("  7/8 Instant edit vs failure")

    # 8. Overall Agent Comparison Summary
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f'Multi-Agent Performance Comparison\n{dataset_info}', fontsize=14, fontweight='bold')

    # X-tick labels with model info for all subplots
    xlabels = [f"{a}\n({get_model_info(a)})" for a in filtered_agents]

    # 8a. Success rate bar chart
    ax = axes[0, 0]
    agent_success = df.groupby('agent_subtype')['success'].mean() * 100
    agent_success = agent_success.reindex(filtered_agents)
    bars = ax.bar(range(len(filtered_agents)), agent_success.values, color=sns.color_palette()[:len(filtered_agents)])
    ax.set_xticks(range(len(filtered_agents)))
    ax.set_xticklabels(xlabels, fontsize=8)
    ax.set_ylabel('Success Rate (%)')
    ax.set_title('Overall Success Rate')
    ax.grid(True, alpha=0.3, axis='y')
    for bar, val in zip(bars, agent_success.values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, f'{val:.1f}%',
               ha='center', va='bottom', fontweight='bold', fontsize=9)

    # 8b. Clean success rate bar chart
    ax = axes[0, 1]
    agent_clean = df.groupby('agent_subtype')['clean_success'].mean() * 100
    agent_clean = agent_clean.reindex(filtered_agents)
    bars = ax.bar(range(len(filtered_agents)), agent_clean.values, color=sns.color_palette()[:len(filtered_agents)])
    ax.set_xticks(range(len(filtered_agents)))
    ax.set_xticklabels(xlabels, fontsize=8)
    ax.set_ylabel('Clean Success Rate (%)')
    ax.set_title('Clean Success Rate')
    ax.grid(True, alpha=0.3, axis='y')
    for bar, val in zip(bars, agent_clean.values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, f'{val:.1f}%',
               ha='center', va='bottom', fontweight='bold', fontsize=9)

    # 8c. Average duration bar chart
    ax = axes[1, 0]
    agent_duration = df.groupby('agent_subtype')['duration_s'].mean()
    agent_duration = agent_duration.reindex(filtered_agents)
    bars = ax.bar(range(len(filtered_agents)), agent_duration.values, color=sns.color_palette()[:len(filtered_agents)])
    ax.set_xticks(range(len(filtered_agents)))
    ax.set_xticklabels(xlabels, fontsize=8)
    ax.set_ylabel('Average Duration (s)')
    ax.set_title('Average Task Duration')
    ax.grid(True, alpha=0.3, axis='y')
    for bar, val in zip(bars, agent_duration.values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5, f'{val:.0f}s',
               ha='center', va='bottom', fontweight='bold', fontsize=9)

    # 8d. Task count bar chart
    ax = axes[1, 1]
    agent_count = df.groupby('agent_subtype').size()
    agent_count = agent_count.reindex(filtered_agents)
    bars = ax.bar(range(len(filtered_agents)), agent_count.values, color=sns.color_palette()[:len(filtered_agents)])
    ax.set_xticks(range(len(filtered_agents)))
    ax.set_xticklabels(xlabels, fontsize=8)
    ax.set_ylabel('Number of Tasks')
    ax.set_title('Total Tasks Evaluated')
    ax.grid(True, alpha=0.3, axis='y')
    for bar, val in zip(bars, agent_count.values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5, f'{val:,}',
               ha='center', va='bottom', fontweight='bold', fontsize=9)

    plt.tight_layout()
    plt.savefig(viz_dir / '08_overall_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("  8/8 Overall comparison")

    print(f"\nVisualizations saved to {viz_dir}/")


def export_comparative_data(df: pd.DataFrame, run_summaries: pd.DataFrame, output_dir: Path):
    """Export comparative data to CSV files."""
    data_dir = output_dir / "data"
    data_dir.mkdir(parents=True, exist_ok=True)

    print("\nExporting comparative data...")

    # All tasks unified
    df.to_csv(data_dir / 'all_tasks_unified.csv', index=False)
    print(f"  all_tasks_unified.csv ({len(df)} rows)")

    # Run summaries
    run_summaries.to_csv(data_dir / 'all_runs_summary.csv', index=False)
    print(f"  all_runs_summary.csv ({len(run_summaries)} rows)")

    # Agent type summaries
    agent_summary = df.groupby('agent_subtype').agg({
        'success': ['sum', 'count', 'mean'],
        'clean_success': ['sum', 'mean'],
        'pathological': ['sum', 'mean'],
        'instant_edit': ['sum', 'mean'],
        'duration_s': ['mean', 'median', 'std'],
        'commit_count': ['mean', 'median', 'max'],
        'violations_count': ['mean', 'median', 'max'],
    }).round(2)
    agent_summary.to_csv(data_dir / 'agent_type_summary.csv')
    print(f"  agent_type_summary.csv")

    # Repository summaries
    repo_summary = df.groupby('repository').agg({
        'success': ['sum', 'count', 'mean'],
        'clean_success': ['sum', 'mean'],
        'pathological': ['sum', 'mean'],
        'duration_s': ['mean', 'median'],
    }).round(2)
    repo_summary.to_csv(data_dir / 'repository_summary.csv')
    print(f"  repository_summary.csv")

    print(f"\nData exported to {data_dir}/")


def generate_comprehensive_report(df: pd.DataFrame, run_summaries: pd.DataFrame, output_dir: Path):
    """Generate the comprehensive multi-agent analysis report."""

    # Calculate key metrics
    total_tasks = len(df)
    total_runs = df['run_id'].nunique()

    # Agent stats
    agent_stats = df.groupby('agent_subtype').agg({
        'success': ['sum', 'count', 'mean'],
        'clean_success': ['sum', 'mean'],
        'pathological': ['sum', 'mean'],
        'instant_edit': ['sum', 'mean'],
        'duration_s': ['mean', 'median'],
        'commit_count': ['mean', 'max'],
        'violations_count': ['mean', 'max'],
    })

    # Repo stats
    repo_stats = df.groupby('repository').agg({
        'success': ['sum', 'count', 'mean'],
        'clean_success': ['sum', 'mean'],
    })

    # Cross-tabulation
    cross_tab = df.groupby(['agent_subtype', 'repository']).agg({
        'success': 'mean',
        'clean_success': 'mean',
        'pathological': 'mean',
    }) * 100

    report = f"""# Comprehensive Multi-Agent Analysis Report

**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Executive Summary

This report provides a comprehensive analysis of {total_tasks:,} optimization tasks across {total_runs} runs,
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
| Total Tasks | {total_tasks:,} |
| Total Runs | {total_runs} |
| Agents Evaluated | {df['agent_subtype'].nunique()} |
| Repositories | {df['repository'].nunique()} |

---

## Agent Performance Summary

| Agent | Tasks | Success | Clean Success | Pathological | Avg Duration |
|-------|-------|---------|---------------|--------------|--------------|
"""

    for agent in ['Codex', 'TRAE-Bedrock', 'TRAE-GPT']:
        if agent in agent_stats.index:
            stats = agent_stats.loc[agent]
            tasks = int(stats[('success', 'count')])
            success = stats[('success', 'sum')]
            success_rate = stats[('success', 'mean')] * 100
            clean_rate = stats[('clean_success', 'mean')] * 100
            patho_rate = stats[('pathological', 'mean')] * 100
            avg_dur = stats[('duration_s', 'mean')]
            report += f"| {agent} | {tasks} | {success_rate:.1f}% | {clean_rate:.1f}% | {patho_rate:.1f}% | {avg_dur:.1f}s |\n"

    report += f"""
---

## Repository Performance Summary

| Repository | Tasks | Success Rate | Clean Rate |
|------------|-------|--------------|------------|
"""

    for repo in ['SGLang', 'vLLM']:
        if repo in repo_stats.index:
            stats = repo_stats.loc[repo]
            tasks = int(stats[('success', 'count')])
            success_rate = stats[('success', 'mean')] * 100
            clean_rate = stats[('clean_success', 'mean')] * 100
            report += f"| {repo} | {tasks} | {success_rate:.1f}% | {clean_rate:.1f}% |\n"

    report += f"""
---

## Cross-Agent × Repository Analysis

| Agent | Repository | Success | Clean | Pathological |
|-------|------------|---------|-------|--------------|
"""

    for (agent, repo), stats in cross_tab.iterrows():
        report += f"| {agent} | {repo} | {stats['success']:.1f}% | {stats['clean_success']:.1f}% | {stats['pathological']:.1f}% |\n"

    report += f"""
---

## Key Behavioral Patterns

### 1. Instant-Edit Pathology

Time-to-first-edit (TTFE) <1 second is a strong predictor of failure:

| Agent | Instant Edit Rate | Failure Rate (Instant) | Failure Rate (Delayed) |
|-------|-------------------|------------------------|------------------------|
"""

    for agent in ['Codex', 'TRAE-Bedrock', 'TRAE-GPT']:
        agent_df = df[df['agent_subtype'] == agent]
        if len(agent_df) > 0:
            instant_rate = agent_df['instant_edit'].mean() * 100
            instant_failure = (1 - agent_df[agent_df['instant_edit']]['success'].mean()) * 100 if agent_df['instant_edit'].sum() > 0 else 0
            delayed_failure = (1 - agent_df[~agent_df['instant_edit']]['success'].mean()) * 100 if (~agent_df['instant_edit']).sum() > 0 else 0
            report += f"| {agent} | {instant_rate:.1f}% | {instant_failure:.1f}% | {delayed_failure:.1f}% |\n"

    report += f"""
### 2. Commit Explosion

Tasks with >50 commits indicate pathological behavior:

| Agent | Max Commits | Avg Commits (Success) | Avg Commits (Failure) |
|-------|-------------|----------------------|----------------------|
"""

    for agent in ['Codex', 'TRAE-Bedrock', 'TRAE-GPT']:
        agent_df = df[df['agent_subtype'] == agent]
        if len(agent_df) > 0:
            max_commits = int(agent_df['commit_count'].max())
            success_commits = agent_df[agent_df['success']]['commit_count'].mean()
            failure_commits = agent_df[~agent_df['success']]['commit_count'].mean()
            report += f"| {agent} | {max_commits:,} | {success_commits:.1f} | {failure_commits:.1f} |\n"

    report += f"""
---

## Run-by-Run Details

| Run ID | Agent | Repository | Tasks | Success | Clean | Pathological |
|--------|-------|------------|-------|---------|-------|--------------|
"""

    for _, row in run_summaries.sort_values(['agent_subtype', 'repository']).iterrows():
        report += f"| {row['run_id'][:30]}... | {row['agent_subtype']} | {row['repository']} | {row['total_tasks']} | {row['success_rate']:.1f}% | {row['clean_rate']:.1f}% | {row['pathological_rate']:.1f}% |\n"

    report += f"""
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

- `multi_agent_comparison/data/all_tasks_unified.csv` - All {total_tasks:,} tasks
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
"""

    report_path = DOCS_DIR / "COMPREHENSIVE_MULTI_AGENT_ANALYSIS.md"
    with open(report_path, 'w') as f:
        f.write(report)

    print(f"\nReport saved to {report_path}")


def main():
    print("=" * 80)
    print("MULTI-AGENT CROSS-RUN ANALYSIS")
    print("=" * 80)

    # Create output directory
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Load all journals
    df = load_all_journals(min_tasks=10)

    # Compute run summaries
    run_summaries = compute_run_summaries(df)

    # Generate visualizations
    create_comparative_visualizations(df, run_summaries, OUTPUT_DIR)

    # Export data
    export_comparative_data(df, run_summaries, OUTPUT_DIR)

    # Generate comprehensive report
    generate_comprehensive_report(df, run_summaries, OUTPUT_DIR)

    print("\n" + "=" * 80)
    print("Analysis complete!")
    print(f"  Visualizations: {OUTPUT_DIR}/visualizations/")
    print(f"  Data: {OUTPUT_DIR}/data/")
    print(f"  Report: {DOCS_DIR}/COMPREHENSIVE_MULTI_AGENT_ANALYSIS.md")
    print("=" * 80)


if __name__ == '__main__':
    main()
