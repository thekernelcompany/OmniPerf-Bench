#!/usr/bin/env python3
"""
TRAE Deep Dive Analysis Script

Analyzes TRAE agent execution trajectories to understand:
1. Silent failures (rc=1 but status=success)
2. Token usage and rate limiting
3. Early error patterns
4. Tool call strategies
5. Failure correction mechanisms

Usage:
    python analyze_trae_deep_dive.py

Outputs:
    - Comprehensive statistics on all 5 dimensions
    - Visualizations showing patterns
    - CSV exports for further analysis
    - JSON summaries
"""

import json
import re
from pathlib import Path
from typing import Dict, List, Any, Tuple, Optional
from collections import defaultdict, Counter
from dataclasses import dataclass
from datetime import datetime

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

warnings.filterwarnings('ignore')

# Configuration
TRAE_VLLM_DIR = Path("perf-agents-bench/state/runs/vllm_core-9641716f")
TRAE_SGLANG_DIR = Path("perf-agents-bench/state/runs/sglang_core-ae58875a")
OUTPUT_DIR = Path("docs/trae_analysis")
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


@dataclass
class TaskData:
    """Container for task-level metrics"""
    task_id: str
    task_number: int
    repo: str

    # From journal.json
    status: str
    returncode: int
    duration_s: float
    time_to_first_edit_s: float
    commit_count: int
    violations_count: int

    # From trajectory.json
    total_interactions: int
    total_tool_calls: int
    total_tokens: int
    input_tokens: int
    output_tokens: int
    reasoning_tokens: int

    # Computed
    is_silent_failure: bool
    tokens_per_second: float


@dataclass
class InteractionData:
    """Container for interaction-level data"""
    task_id: str
    interaction_idx: int
    timestamp: float
    tool_name: str
    tool_args: Dict
    had_error: bool
    error_message: Optional[str]
    input_tokens: int
    output_tokens: int
    reasoning_tokens: int


def load_trajectory(traj_path: Path) -> Optional[Dict]:
    """Load and parse trajectory.json"""
    try:
        with open(traj_path) as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading {traj_path}: {e}")
        return None


def load_journal(journal_path: Path) -> Optional[Dict]:
    """Load and parse journal.json"""
    try:
        with open(journal_path) as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading {journal_path}: {e}")
        return None


def parse_task(task_dir: Path, repo: str) -> Optional[TaskData]:
    """Parse a single task directory"""
    journal_path = task_dir / "journal.json"
    traj_path = task_dir / "trajectory.json"

    if not journal_path.exists():
        return None

    journal = load_journal(journal_path)
    if not journal:
        return None

    # Extract task number
    task_num_match = re.search(r'-(\d+)$', task_dir.name)
    task_number = int(task_num_match.group(1)) if task_num_match else 0

    # Get basic metrics from journal
    metrics = journal.get('metrics', {})
    trae_cli = journal.get('trae_cli', {})

    # Load trajectory if exists
    total_interactions = 0
    total_tool_calls = 0
    total_tokens = 0
    input_tokens = 0
    output_tokens = 0
    reasoning_tokens = 0

    if traj_path.exists():
        traj = load_trajectory(traj_path)
        if traj and 'llm_interactions' in traj:
            interactions = traj['llm_interactions']
            total_interactions = len(interactions)

            for inter in interactions:
                # Count tool calls
                if 'tool_call' in inter and inter['tool_call']:
                    total_tool_calls += 1

                # Sum tokens
                if 'token_usage' in inter and inter['token_usage']:
                    usage = inter['token_usage']
                    input_tokens += usage.get('input_tokens', 0)
                    output_tokens += usage.get('output_tokens', 0)
                    reasoning_tokens += usage.get('reasoning_tokens', 0)

            total_tokens = input_tokens + output_tokens + reasoning_tokens

    # Determine if silent failure
    status = journal.get('status', 'unknown')
    returncode = trae_cli.get('returncode', 0)
    is_silent_failure = (status == 'success' and returncode != 0)

    duration = trae_cli.get('duration_s', 0)
    tokens_per_second = total_tokens / duration if duration > 0 else 0

    return TaskData(
        task_id=task_dir.name,
        task_number=task_number,
        repo=repo,
        status=status,
        returncode=returncode,
        duration_s=duration,
        time_to_first_edit_s=metrics.get('time_to_first_edit_s', 0),
        commit_count=metrics.get('commit_count', 0),
        violations_count=metrics.get('violations_count', 0),
        total_interactions=total_interactions,
        total_tool_calls=total_tool_calls,
        total_tokens=total_tokens,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        reasoning_tokens=reasoning_tokens,
        is_silent_failure=is_silent_failure,
        tokens_per_second=tokens_per_second,
    )


def load_all_tasks() -> Tuple[List[TaskData], List[InteractionData]]:
    """Load all TRAE tasks from both vLLM and SGLang"""
    print("Loading TRAE tasks...")

    tasks = []
    interactions = []

    # Load vLLM tasks
    if TRAE_VLLM_DIR.exists():
        vllm_dirs = sorted([d for d in TRAE_VLLM_DIR.iterdir() if d.is_dir()])
        print(f"  Found {len(vllm_dirs)} vLLM task directories")

        for task_dir in vllm_dirs:
            task = parse_task(task_dir, "vllm")
            if task:
                tasks.append(task)

                # Load interactions from trajectory
                traj_path = task_dir / "trajectory.json"
                if traj_path.exists():
                    traj = load_trajectory(traj_path)
                    if traj and 'llm_interactions' in traj:
                        for idx, inter in enumerate(traj['llm_interactions']):
                            tool_call = inter.get('tool_call', {})
                            tool_result = inter.get('tool_result', {})
                            token_usage = inter.get('token_usage', {})

                            if tool_call:
                                interactions.append(InteractionData(
                                    task_id=task.task_id,
                                    interaction_idx=idx,
                                    timestamp=inter.get('timestamp_start_s', 0),
                                    tool_name=tool_call.get('name', 'unknown'),
                                    tool_args=tool_call.get('args', {}),
                                    had_error=bool(tool_result.get('error')),
                                    error_message=tool_result.get('error'),
                                    input_tokens=token_usage.get('input_tokens', 0),
                                    output_tokens=token_usage.get('output_tokens', 0),
                                    reasoning_tokens=token_usage.get('reasoning_tokens', 0),
                                ))

    # Load SGLang tasks
    if TRAE_SGLANG_DIR.exists():
        sglang_dirs = sorted([d for d in TRAE_SGLANG_DIR.iterdir() if d.is_dir()])
        print(f"  Found {len(sglang_dirs)} SGLang task directories")

        for task_dir in sglang_dirs:
            task = parse_task(task_dir, "sglang")
            if task:
                tasks.append(task)

                # Load interactions (same as vLLM)
                traj_path = task_dir / "trajectory.json"
                if traj_path.exists():
                    traj = load_trajectory(traj_path)
                    if traj and 'llm_interactions' in traj:
                        for idx, inter in enumerate(traj['llm_interactions']):
                            tool_call = inter.get('tool_call', {})
                            tool_result = inter.get('tool_result', {})
                            token_usage = inter.get('token_usage', {})

                            if tool_call:
                                interactions.append(InteractionData(
                                    task_id=task.task_id,
                                    interaction_idx=idx,
                                    timestamp=inter.get('timestamp_start_s', 0),
                                    tool_name=tool_call.get('name', 'unknown'),
                                    tool_args=tool_call.get('args', {}),
                                    had_error=bool(tool_result.get('error')),
                                    error_message=tool_result.get('error'),
                                    input_tokens=token_usage.get('input_tokens', 0),
                                    output_tokens=token_usage.get('output_tokens', 0),
                                    reasoning_tokens=token_usage.get('reasoning_tokens', 0),
                                ))

    print(f"Loaded {len(tasks)} tasks, {len(interactions)} interactions")
    return tasks, interactions


def analyze_silent_failures(tasks: List[TaskData]) -> Dict[str, Any]:
    """Analyze silent failure patterns"""
    print("\nAnalyzing silent failures...")

    by_repo = defaultdict(lambda: {'total': 0, 'silent': 0, 'clean': 0})

    for task in tasks:
        by_repo[task.repo]['total'] += 1
        if task.is_silent_failure:
            by_repo[task.repo]['silent'] += 1
        elif task.returncode == 0:
            by_repo[task.repo]['clean'] += 1

    results = {}
    for repo, counts in by_repo.items():
        results[repo] = {
            'total_tasks': counts['total'],
            'silent_failures': counts['silent'],
            'clean_success': counts['clean'],
            'silent_failure_rate': counts['silent'] / counts['total'] * 100 if counts['total'] > 0 else 0,
            'clean_success_rate': counts['clean'] / counts['total'] * 100 if counts['total'] > 0 else 0,
        }

    return results


def analyze_token_usage(tasks: List[TaskData]) -> Dict[str, Any]:
    """Analyze token usage patterns"""
    print("\nAnalyzing token usage...")

    by_repo = defaultdict(list)

    for task in tasks:
        if task.total_tokens > 0:
            by_repo[task.repo].append({
                'total_tokens': task.total_tokens,
                'input_tokens': task.input_tokens,
                'output_tokens': task.output_tokens,
                'reasoning_tokens': task.reasoning_tokens,
                'tokens_per_second': task.tokens_per_second,
                'duration_s': task.duration_s,
            })

    results = {}
    for repo, data in by_repo.items():
        df = pd.DataFrame(data)
        results[repo] = {
            'total_tasks': len(df),
            'total_tokens_sum': int(df['total_tokens'].sum()),
            'mean_tokens_per_task': float(df['total_tokens'].mean()),
            'median_tokens_per_task': float(df['total_tokens'].median()),
            'max_tokens': int(df['total_tokens'].max()),
            'mean_tokens_per_second': float(df['tokens_per_second'].mean()),
            'input_output_ratio': float(df['input_tokens'].sum() / df['output_tokens'].sum()) if df['output_tokens'].sum() > 0 else 0,
            'reasoning_ratio': float(df['reasoning_tokens'].sum() / df['total_tokens'].sum()) if df['total_tokens'].sum() > 0 else 0,
        }

    return results


def analyze_tool_calls(interactions: List[InteractionData]) -> Dict[str, Any]:
    """Analyze tool call patterns"""
    print("\nAnalyzing tool calls...")

    by_repo = defaultdict(lambda: defaultdict(int))
    by_tool = defaultdict(int)
    error_by_tool = defaultdict(int)

    for inter in interactions:
        repo = 'vllm' if 'vllm' in inter.task_id else 'sglang'
        by_repo[repo][inter.tool_name] += 1
        by_tool[inter.tool_name] += 1

        if inter.had_error:
            error_by_tool[inter.tool_name] += 1

    results = {
        'by_repository': dict(by_repo),
        'total_by_tool': dict(by_tool),
        'errors_by_tool': dict(error_by_tool),
        'total_interactions': len(interactions),
    }

    return results


def analyze_early_errors(tasks: List[TaskData], interactions: List[InteractionData]) -> Dict[str, Any]:
    """Analyze errors in first 5 minutes"""
    print("\nAnalyzing early errors...")

    # Group interactions by task
    by_task = defaultdict(list)
    for inter in interactions:
        by_task[inter.task_id].append(inter)

    early_error_tasks = []
    no_early_error_tasks = []

    for task in tasks:
        task_inters = sorted(by_task[task.task_id], key=lambda x: x.interaction_idx)

        # Check first 5 minutes
        has_early_error = False
        for inter in task_inters:
            if inter.timestamp <= 300:  # 5 minutes
                if inter.had_error:
                    has_early_error = True
                    break

        if has_early_error:
            early_error_tasks.append(task)
        else:
            no_early_error_tasks.append(task)

    # Calculate success rates
    early_error_success = sum(1 for t in early_error_tasks if t.status == 'success' and t.returncode == 0)
    no_early_error_success = sum(1 for t in no_early_error_tasks if t.status == 'success' and t.returncode == 0)

    results = {
        'tasks_with_early_errors': len(early_error_tasks),
        'tasks_without_early_errors': len(no_early_error_tasks),
        'early_error_success_rate': early_error_success / len(early_error_tasks) * 100 if early_error_tasks else 0,
        'no_early_error_success_rate': no_early_error_success / len(no_early_error_tasks) * 100 if no_early_error_tasks else 0,
    }

    return results


def analyze_failure_correction(interactions: List[InteractionData]) -> Dict[str, Any]:
    """Analyze self-correction patterns"""
    print("\nAnalyzing failure correction...")

    # Group by task and file
    file_edits = defaultdict(lambda: defaultdict(list))

    for inter in interactions:
        if inter.tool_name == 'str_replace_based_edit_tool' and 'path' in inter.tool_args:
            file_path = inter.tool_args['path']
            file_edits[inter.task_id][file_path].append(inter)

    multi_edit_count = 0
    max_edits = 0
    total_files_edited = 0

    for task_id, files in file_edits.items():
        for file_path, edits in files.items():
            total_files_edited += 1
            edit_count = len(edits)

            if edit_count > 1:
                multi_edit_count += 1
                max_edits = max(max_edits, edit_count)

    results = {
        'total_files_edited': total_files_edited,
        'files_edited_multiple_times': multi_edit_count,
        'multi_edit_rate': multi_edit_count / total_files_edited * 100 if total_files_edited > 0 else 0,
        'max_edits_to_single_file': max_edits,
    }

    return results


def create_visualizations(tasks: List[TaskData], interactions: List[InteractionData]):
    """Generate all visualizations"""
    print("\nGenerating visualizations...")

    df_tasks = pd.DataFrame([vars(t) for t in tasks])

    # 1. Silent failures by repository
    plt.figure(figsize=(10, 6))
    silent_data = df_tasks.groupby(['repo', 'is_silent_failure']).size().unstack(fill_value=0)
    silent_data.plot(kind='bar', stacked=True, color=['#2ecc71', '#e74c3c'])
    plt.title('Silent Failures by Repository')
    plt.xlabel('Repository')
    plt.ylabel('Task Count')
    plt.legend(['Clean Success (rc=0)', 'Silent Failure (rc≠0)'])
    plt.tight_layout()
    plt.savefig(VIZ_DIR / '01_silent_failures.png', dpi=300)
    plt.close()
    print("  ✓ Silent failures")

    # 2. Token usage distribution
    plt.figure(figsize=(10, 6))
    for repo in df_tasks['repo'].unique():
        repo_data = df_tasks[df_tasks['repo'] == repo]
        plt.hist(repo_data['total_tokens'], bins=30, alpha=0.5, label=repo)
    plt.xlabel('Total Tokens')
    plt.ylabel('Frequency')
    plt.title('Token Usage Distribution by Repository')
    plt.legend()
    plt.tight_layout()
    plt.savefig(VIZ_DIR / '02_token_distribution.png', dpi=300)
    plt.close()
    print("  ✓ Token distribution")

    # 3. Tool call breakdown
    df_inter = pd.DataFrame([vars(i) for i in interactions])
    tool_counts = df_inter['tool_name'].value_counts()

    plt.figure(figsize=(10, 6))
    tool_counts.plot(kind='bar')
    plt.title('Tool Call Frequency')
    plt.xlabel('Tool')
    plt.ylabel('Count')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(VIZ_DIR / '03_tool_calls.png', dpi=300)
    plt.close()
    print("  ✓ Tool calls")

    # 4. Tokens per second
    plt.figure(figsize=(10, 6))
    for repo in df_tasks['repo'].unique():
        repo_data = df_tasks[(df_tasks['repo'] == repo) & (df_tasks['tokens_per_second'] > 0)]
        plt.scatter(repo_data['duration_s'], repo_data['tokens_per_second'], alpha=0.6, label=repo)
    plt.xlabel('Duration (seconds)')
    plt.ylabel('Tokens per Second')
    plt.title('Token Processing Rate')
    plt.legend()
    plt.tight_layout()
    plt.savefig(VIZ_DIR / '04_tokens_per_second.png', dpi=300)
    plt.close()
    print("  ✓ Tokens per second")

    print(f"\nAll visualizations saved to {VIZ_DIR}/")


def export_data(tasks: List[TaskData], interactions: List[InteractionData], analyses: Dict):
    """Export analysis results"""
    print("\nExporting data...")

    # Export tasks
    df_tasks = pd.DataFrame([vars(t) for t in tasks])
    df_tasks.to_csv(DATA_DIR / 'trae_tasks.csv', index=False)
    print(f"  ✓ Exported: trae_tasks.csv ({len(df_tasks)} rows)")

    # Export interactions
    df_inter = pd.DataFrame([vars(i) for i in interactions])
    df_inter.to_csv(DATA_DIR / 'trae_interactions.csv', index=False)
    print(f"  ✓ Exported: trae_interactions.csv ({len(df_inter)} rows)")

    # Export analysis summaries
    with open(DATA_DIR / 'trae_analysis_summary.json', 'w') as f:
        json.dump(analyses, f, indent=2)
    print(f"  ✓ Exported: trae_analysis_summary.json")

    print(f"\nAll data saved to {DATA_DIR}/")


def print_summary(analyses: Dict):
    """Print analysis summary"""
    print("\n" + "="*80)
    print("TRAE ANALYSIS SUMMARY")
    print("="*80)

    print("\n1. SILENT FAILURES:")
    for repo, data in analyses['silent_failures'].items():
        print(f"\n  {repo.upper()}:")
        print(f"    Total tasks: {data['total_tasks']}")
        print(f"    Silent failures: {data['silent_failures']} ({data['silent_failure_rate']:.1f}%)")
        print(f"    Clean success: {data['clean_success']} ({data['clean_success_rate']:.1f}%)")

    print("\n2. TOKEN USAGE:")
    for repo, data in analyses['token_usage'].items():
        print(f"\n  {repo.upper()}:")
        print(f"    Tasks analyzed: {data['total_tasks']}")
        print(f"    Total tokens: {data['total_tokens_sum']:,}")
        print(f"    Mean per task: {data['mean_tokens_per_task']:,.0f}")
        print(f"    Median per task: {data['median_tokens_per_task']:,.0f}")
        print(f"    Max tokens: {data['max_tokens']:,}")
        print(f"    Tokens/second: {data['mean_tokens_per_second']:.1f}")

    print("\n3. TOOL CALLS:")
    print(f"    Total interactions: {analyses['tool_calls']['total_interactions']:,}")
    print(f"    Tool breakdown:")
    for tool, count in sorted(analyses['tool_calls']['total_by_tool'].items(), key=lambda x: x[1], reverse=True):
        errors = analyses['tool_calls']['errors_by_tool'].get(tool, 0)
        print(f"      {tool}: {count:,} ({errors} errors)")

    print("\n4. EARLY ERRORS:")
    print(f"    Tasks with early errors: {analyses['early_errors']['tasks_with_early_errors']}")
    print(f"    Success rate: {analyses['early_errors']['early_error_success_rate']:.1f}%")
    print(f"    Tasks without early errors: {analyses['early_errors']['tasks_without_early_errors']}")
    print(f"    Success rate: {analyses['early_errors']['no_early_error_success_rate']:.1f}%")

    print("\n5. FAILURE CORRECTION:")
    print(f"    Total files edited: {analyses['failure_correction']['total_files_edited']}")
    print(f"    Files edited multiple times: {analyses['failure_correction']['files_edited_multiple_times']}")
    print(f"    Multi-edit rate: {analyses['failure_correction']['multi_edit_rate']:.1f}%")
    print(f"    Max edits to single file: {analyses['failure_correction']['max_edits_to_single_file']}")

    print("\n" + "="*80)


def main():
    """Main execution"""
    print("="*80)
    print("TRAE DEEP DIVE ANALYSIS")
    print("="*80)
    print(f"Output Directory: {OUTPUT_DIR}")
    print("="*80)

    # Load data
    tasks, interactions = load_all_tasks()

    if not tasks:
        print("No tasks found! Check directory paths.")
        return

    # Run analyses
    analyses = {
        'silent_failures': analyze_silent_failures(tasks),
        'token_usage': analyze_token_usage(tasks),
        'tool_calls': analyze_tool_calls(interactions),
        'early_errors': analyze_early_errors(tasks, interactions),
        'failure_correction': analyze_failure_correction(interactions),
    }

    # Generate visualizations
    create_visualizations(tasks, interactions)

    # Export data
    export_data(tasks, interactions, analyses)

    # Print summary
    print_summary(analyses)

    print(f"\n✓ Analysis complete!")
    print(f"  Visualizations: {VIZ_DIR}/")
    print(f"  Data exports: {DATA_DIR}/")


if __name__ == '__main__':
    main()
