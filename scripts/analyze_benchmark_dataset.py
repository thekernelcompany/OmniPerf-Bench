#!/usr/bin/env python3
"""
Comprehensive analysis script for Claude Code vLLM Benchmark dataset.

This script loads the Hugging Face dataset and provides:
- Pipeline funnel analysis (success rates at each stage)
- Status and error breakdown
- Agent vs Human performance comparison
- Optimization type analysis
- Exportable metrics and reports

Usage:
    python scripts/analyze_benchmark_dataset.py
    python scripts/analyze_benchmark_dataset.py --output-dir ./reports
    python scripts/analyze_benchmark_dataset.py --format json
"""

import argparse
import json
import sys
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Optional

import pandas as pd
import numpy as np

try:
    from datasets import load_dataset
except ImportError:
    print("Error: 'datasets' package not installed. Run: pip install datasets")
    sys.exit(1)


DATASET_NAME = "Inferencebench/claude-code-vllm-benchmarks"


@dataclass
class PipelineFunnel:
    """Tracks how many commits pass each stage of the benchmark pipeline."""
    total_commits: int = 0
    has_perf_command: int = 0
    no_error_exception: int = 0
    success_status: int = 0
    has_agent_patch: int = 0
    has_agent_metrics: int = 0

    def as_dict(self):
        return {
            "total_commits": self.total_commits,
            "has_perf_command": self.has_perf_command,
            "has_perf_command_pct": self.has_perf_command / self.total_commits * 100 if self.total_commits else 0,
            "no_error_exception": self.no_error_exception,
            "no_error_exception_pct": self.no_error_exception / self.total_commits * 100 if self.total_commits else 0,
            "success_status": self.success_status,
            "success_status_pct": self.success_status / self.total_commits * 100 if self.total_commits else 0,
            "has_agent_patch": self.has_agent_patch,
            "has_agent_patch_pct": self.has_agent_patch / self.total_commits * 100 if self.total_commits else 0,
            "has_agent_metrics": self.has_agent_metrics,
            "has_agent_metrics_pct": self.has_agent_metrics / self.total_commits * 100 if self.total_commits else 0,
        }


@dataclass
class AgentPerformance:
    """Summary of agent performance vs human baseline."""
    total_comparisons: int = 0
    agent_wins: int = 0  # Agent > 5% better than human
    agent_ties: int = 0  # Within 5%
    agent_losses: int = 0  # Agent > 5% worse than human

    # Detailed case info
    cases: list = field(default_factory=list)

    def win_rate(self) -> float:
        if self.total_comparisons == 0:
            return 0.0
        return self.agent_wins / self.total_comparisons

    def as_dict(self):
        return {
            "total_comparisons": self.total_comparisons,
            "agent_wins": self.agent_wins,
            "agent_ties": self.agent_ties,
            "agent_losses": self.agent_losses,
            "win_rate": self.win_rate(),
            "loss_rate": self.agent_losses / self.total_comparisons if self.total_comparisons else 0,
            "tie_rate": self.agent_ties / self.total_comparisons if self.total_comparisons else 0,
        }


@dataclass
class BenchmarkAnalysis:
    """Complete analysis results."""
    dataset_name: str
    analysis_timestamp: str
    funnel: PipelineFunnel
    status_distribution: dict
    error_distribution: dict
    gpu_distribution: dict
    optimization_types: dict
    agent_performance: AgentPerformance
    recommendations: list

    def as_dict(self):
        return {
            "dataset_name": self.dataset_name,
            "analysis_timestamp": self.analysis_timestamp,
            "funnel": self.funnel.as_dict(),
            "status_distribution": self.status_distribution,
            "error_distribution": self.error_distribution,
            "gpu_distribution": self.gpu_distribution,
            "optimization_types": self.optimization_types,
            "agent_performance": self.agent_performance.as_dict(),
            "detailed_cases": self.agent_performance.cases,
            "recommendations": self.recommendations,
        }


def categorize_optimization(commit_subject: str) -> str:
    """Categorize a commit by its optimization type based on subject line."""
    subject_lower = commit_subject.lower() if commit_subject else ""

    if 'cuda' in subject_lower or 'kernel' in subject_lower:
        return "CUDA/Kernel"
    elif 'attention' in subject_lower or 'flash' in subject_lower or 'attn' in subject_lower:
        return "Attention"
    elif 'cache' in subject_lower or 'memory' in subject_lower or 'evictor' in subject_lower:
        return "Memory/Cache"
    elif 'mamba' in subject_lower or 'moe' in subject_lower or 'fp8' in subject_lower:
        return "Model Architecture"
    elif 'serv' in subject_lower or 'batch' in subject_lower:
        return "Serving"
    else:
        return "General Performance"


def analyze_pipeline_funnel(df: pd.DataFrame) -> PipelineFunnel:
    """Analyze how many commits pass each stage of the benchmark pipeline."""
    funnel = PipelineFunnel()
    funnel.total_commits = len(df)
    funnel.has_perf_command = df['perf_command'].notna().sum()
    funnel.no_error_exception = (~df['status'].isin(['exception', 'error'])).sum()
    funnel.success_status = (df['status'] == 'success').sum()
    funnel.has_agent_patch = (df['has_agent_patch'] == True).sum()

    agent_cols = ['agent_ttft_mean', 'agent_throughput', 'agent_latency_avg']
    funnel.has_agent_metrics = df[agent_cols].notna().any(axis=1).sum()

    return funnel


def analyze_agent_performance(df: pd.DataFrame) -> AgentPerformance:
    """Analyze agent performance vs human baseline.

    Calculates metrics from raw values to ensure correctness.
    - For TTFT/latency: improvement = (baseline - new) / baseline * 100 (lower is better)
    - For Throughput: improvement = (new - baseline) / baseline * 100 (higher is better)
    """
    perf = AgentPerformance()

    # Get cases with agent metrics
    agent_cols = ['agent_ttft_mean', 'agent_throughput', 'agent_latency_avg']
    has_agent = df[df[agent_cols].notna().any(axis=1)].copy()

    for idx, row in has_agent.iterrows():
        # Check which metrics are available and calculate from raw values
        has_ttft = (pd.notna(row.get('baseline_ttft_mean')) and
                    pd.notna(row.get('human_ttft_mean')) and
                    pd.notna(row.get('agent_ttft_mean')))
        has_throughput = (pd.notna(row.get('baseline_throughput')) and
                          pd.notna(row.get('human_throughput')) and
                          pd.notna(row.get('agent_throughput')))

        if has_ttft:
            baseline_val = row['baseline_ttft_mean']
            human_val = row['human_ttft_mean']
            agent_val = row['agent_ttft_mean']
            metric_name = 'TTFT'

            # For latency, lower is better
            human_improvement = (baseline_val - human_val) / baseline_val * 100 if baseline_val != 0 else 0
            agent_vs_human = (human_val - agent_val) / human_val * 100 if human_val != 0 else 0

        elif has_throughput:
            baseline_val = row['baseline_throughput']
            human_val = row['human_throughput']
            agent_val = row['agent_throughput']
            metric_name = 'Throughput'

            # For throughput, higher is better
            human_improvement = (human_val - baseline_val) / baseline_val * 100 if baseline_val != 0 else 0
            agent_vs_human = (agent_val - human_val) / human_val * 100 if human_val != 0 else 0
        else:
            continue

        perf.total_comparisons += 1

        # Determine verdict (5% threshold)
        if agent_vs_human > 5:
            verdict = "AGENT_WINS"
            perf.agent_wins += 1
        elif agent_vs_human < -5:
            verdict = "AGENT_LOSES"
            perf.agent_losses += 1
        else:
            verdict = "TIE"
            perf.agent_ties += 1

        perf.cases.append({
            "commit_hash": row['commit_hash'],
            "commit_subject": row['commit_subject'],
            "optimization_type": categorize_optimization(row['commit_subject']),
            "metric": metric_name,
            "baseline": float(baseline_val) if pd.notna(baseline_val) else None,
            "human": float(human_val) if pd.notna(human_val) else None,
            "agent": float(agent_val) if pd.notna(agent_val) else None,
            "human_improvement_pct": float(human_improvement),
            "agent_vs_human_pct": float(agent_vs_human),
            "verdict": verdict,
        })

    return perf


def analyze_optimization_types(df: pd.DataFrame) -> dict:
    """Analyze success rates by optimization type."""
    results = {}

    for _, row in df.iterrows():
        opt_type = categorize_optimization(row['commit_subject'])
        if opt_type not in results:
            results[opt_type] = {"total": 0, "success": 0}
        results[opt_type]["total"] += 1
        if row['status'] == 'success':
            results[opt_type]["success"] += 1

    # Calculate success rates
    for opt_type in results:
        total = results[opt_type]["total"]
        success = results[opt_type]["success"]
        results[opt_type]["success_rate"] = success / total if total > 0 else 0

    return results


def generate_recommendations(analysis: BenchmarkAnalysis) -> list:
    """Generate actionable recommendations based on analysis."""
    recommendations = []

    # Pipeline issues
    funnel = analysis.funnel
    if funnel.has_perf_command / funnel.total_commits < 0.8:
        recommendations.append({
            "category": "Data Quality",
            "priority": "HIGH",
            "issue": f"Only {funnel.has_perf_command}/{funnel.total_commits} commits have perf_command",
            "recommendation": "Improve test generation to create perf commands for more commits"
        })

    # Error analysis
    broken_pipe_count = analysis.error_distribution.get('[Errno 32] Broken pipe', 0)
    if broken_pipe_count > 5:
        recommendations.append({
            "category": "Infrastructure",
            "priority": "HIGH",
            "issue": f"{broken_pipe_count} commits failed with 'Broken pipe' error",
            "recommendation": "Investigate GPU server stability or subprocess handling"
        })

    # Agent performance
    perf = analysis.agent_performance
    if perf.total_comparisons > 0:
        if perf.win_rate() < 0.4:
            recommendations.append({
                "category": "Agent Quality",
                "priority": "MEDIUM",
                "issue": f"Agent win rate is only {perf.win_rate()*100:.1f}% (wins: {perf.agent_wins}, ties: {perf.agent_ties}, losses: {perf.agent_losses})",
                "recommendation": "Agent needs improvement to consistently beat human optimizations"
            })

        if perf.agent_wins > 0:
            winning_cases = [c for c in perf.cases if c['verdict'] == 'AGENT_WINS']
            opt_types = set(c['optimization_type'] for c in winning_cases)
            recommendations.append({
                "category": "Agent Strengths",
                "priority": "INFO",
                "issue": f"Agent excels at: {', '.join(opt_types)}",
                "recommendation": "Focus agent development on these successful patterns"
            })

    # Low success rate overall
    if funnel.success_status / funnel.total_commits < 0.2:
        recommendations.append({
            "category": "Pipeline Reliability",
            "priority": "HIGH",
            "issue": f"Only {funnel.success_status}/{funnel.total_commits} runs succeeded",
            "recommendation": "Investigate and fix infrastructure/environment issues"
        })

    return recommendations


def run_analysis(dataset_name: str = DATASET_NAME) -> BenchmarkAnalysis:
    """Run complete analysis on the benchmark dataset."""
    print(f"Loading dataset: {dataset_name}")
    ds = load_dataset(dataset_name, split='train')
    df = pd.DataFrame(ds)
    print(f"Loaded {len(df)} records")

    # Run all analyses
    funnel = analyze_pipeline_funnel(df)
    agent_perf = analyze_agent_performance(df)
    opt_types = analyze_optimization_types(df)

    # Status distribution
    status_dist = df['status'].value_counts().to_dict()

    # Error distribution
    error_df = df[df['error'].notna()]
    error_dist = error_df['error'].value_counts().to_dict() if len(error_df) > 0 else {}

    # GPU distribution
    gpu_df = df[df['gpu_config'].notna()]
    gpu_dist = gpu_df['gpu_config'].value_counts().to_dict() if len(gpu_df) > 0 else {}

    analysis = BenchmarkAnalysis(
        dataset_name=dataset_name,
        analysis_timestamp=datetime.now().isoformat(),
        funnel=funnel,
        status_distribution=status_dist,
        error_distribution=error_dist,
        gpu_distribution=gpu_dist,
        optimization_types=opt_types,
        agent_performance=agent_perf,
        recommendations=[],
    )

    # Generate recommendations based on analysis
    analysis.recommendations = generate_recommendations(analysis)

    return analysis


def print_report(analysis: BenchmarkAnalysis):
    """Print a formatted text report."""
    print("\n" + "="*80)
    print("CLAUDE CODE vLLM BENCHMARK ANALYSIS REPORT")
    print("="*80)
    print(f"Dataset: {analysis.dataset_name}")
    print(f"Generated: {analysis.analysis_timestamp}")

    # Pipeline Funnel
    print("\n" + "-"*80)
    print("PIPELINE FUNNEL")
    print("-"*80)
    f = analysis.funnel
    print(f"  Total commits:        {f.total_commits:4d} (100.0%)")
    print(f"  Has perf_command:     {f.has_perf_command:4d} ({f.has_perf_command/f.total_commits*100:5.1f}%)")
    print(f"  No error/exception:   {f.no_error_exception:4d} ({f.no_error_exception/f.total_commits*100:5.1f}%)")
    print(f"  Success status:       {f.success_status:4d} ({f.success_status/f.total_commits*100:5.1f}%)")
    print(f"  Has agent patch:      {f.has_agent_patch:4d} ({f.has_agent_patch/f.total_commits*100:5.1f}%)")
    print(f"  Has agent metrics:    {f.has_agent_metrics:4d} ({f.has_agent_metrics/f.total_commits*100:5.1f}%)")

    # Status Distribution
    print("\n" + "-"*80)
    print("STATUS DISTRIBUTION")
    print("-"*80)
    for status, count in sorted(analysis.status_distribution.items(), key=lambda x: -x[1]):
        pct = count / f.total_commits * 100
        print(f"  {status:20s}: {count:3d} ({pct:5.1f}%)")

    # Error Distribution
    if analysis.error_distribution:
        print("\n" + "-"*80)
        print("ERROR TYPES")
        print("-"*80)
        for error, count in sorted(analysis.error_distribution.items(), key=lambda x: -x[1])[:5]:
            print(f"  {error[:50]:50s}: {count:3d}")

    # Agent Performance
    print("\n" + "-"*80)
    print("AGENT vs HUMAN PERFORMANCE")
    print("-"*80)
    ap = analysis.agent_performance
    if ap.total_comparisons > 0:
        print(f"  Total comparisons: {ap.total_comparisons}")
        print(f"  Agent WINS  (>5% better):  {ap.agent_wins:2d} ({ap.agent_wins/ap.total_comparisons*100:5.1f}%)")
        print(f"  Agent TIES  (within 5%):   {ap.agent_ties:2d} ({ap.agent_ties/ap.total_comparisons*100:5.1f}%)")
        print(f"  Agent LOSES (<5% worse):   {ap.agent_losses:2d} ({ap.agent_losses/ap.total_comparisons*100:5.1f}%)")

        print("\n  Detailed Cases:")
        for case in ap.cases:
            verdict_emoji = {"AGENT_WINS": "🟢", "TIE": "🟡", "AGENT_LOSES": "🔴"}[case['verdict']]
            print(f"\n    {verdict_emoji} {case['commit_hash'][:8]}: {case['commit_subject'][:50]}")
            print(f"       Type: {case['optimization_type']} | Metric: {case['metric']}")
            print(f"       Baseline: {case['baseline']:.2f} → Human: {case['human']:.2f} → Agent: {case['agent']:.2f}")
            print(f"       Agent vs Human: {case['agent_vs_human_pct']:+.2f}%")
    else:
        print("  No agent vs human comparisons available")

    # Optimization Types
    print("\n" + "-"*80)
    print("SUCCESS BY OPTIMIZATION TYPE")
    print("-"*80)
    for opt_type, stats in sorted(analysis.optimization_types.items(), key=lambda x: -x[1]['total']):
        print(f"  {opt_type:20s}: {stats['success']:2d}/{stats['total']:2d} ({stats['success_rate']*100:5.1f}%)")

    # Recommendations
    print("\n" + "-"*80)
    print("RECOMMENDATIONS")
    print("-"*80)
    for rec in analysis.recommendations:
        priority_emoji = {"HIGH": "🔴", "MEDIUM": "🟡", "INFO": "🟢", "LOW": "⚪"}
        emoji = priority_emoji.get(rec['priority'], "⚪")
        print(f"\n  {emoji} [{rec['priority']}] {rec['category']}")
        print(f"     Issue: {rec['issue']}")
        print(f"     Action: {rec['recommendation']}")

    # Final Verdict
    print("\n" + "="*80)
    print("FINAL VERDICT: ARE AGENTS READY FOR KERNEL OPTIMIZATION?")
    print("="*80)

    if ap.total_comparisons == 0:
        print("\n  ⚠️  INSUFFICIENT DATA")
        print("     Cannot draw conclusions without agent vs human comparisons.")
    elif ap.win_rate() >= 0.6:
        print("\n  ✅ PROMISING")
        print(f"     Agent wins {ap.win_rate()*100:.0f}% of the time.")
        print("     Agents show potential for kernel optimization tasks.")
    elif ap.win_rate() >= 0.4:
        print("\n  🟡 MIXED RESULTS")
        print(f"     Agent wins {ap.win_rate()*100:.0f}% of the time.")
        print("     Agents can sometimes match or beat human optimizations,")
        print("     but are not consistently reliable for production use.")
    else:
        print("\n  ❌ NOT READY")
        print(f"     Agent wins only {ap.win_rate()*100:.0f}% of the time.")
        print("     Significant improvements needed before agents can")
        print("     reliably perform kernel optimization tasks.")

    print("\n" + "="*80)


def main():
    parser = argparse.ArgumentParser(description="Analyze Claude Code vLLM Benchmark dataset")
    parser.add_argument("--dataset", default=DATASET_NAME, help="HuggingFace dataset name")
    parser.add_argument("--output-dir", type=Path, help="Directory to save reports")
    parser.add_argument("--format", choices=["text", "json", "both"], default="text",
                       help="Output format")
    args = parser.parse_args()

    # Run analysis
    analysis = run_analysis(args.dataset)

    # Output
    if args.format in ["text", "both"]:
        print_report(analysis)

    if args.output_dir:
        args.output_dir.mkdir(parents=True, exist_ok=True)

        if args.format in ["json", "both"]:
            json_path = args.output_dir / "benchmark_analysis.json"
            with open(json_path, 'w') as f:
                json.dump(analysis.as_dict(), f, indent=2, default=str)
            print(f"\nJSON report saved to: {json_path}")

        if args.format in ["text", "both"]:
            # Save text report too
            from io import StringIO
            import sys

            old_stdout = sys.stdout
            sys.stdout = StringIO()
            print_report(analysis)
            text_report = sys.stdout.getvalue()
            sys.stdout = old_stdout

            text_path = args.output_dir / "benchmark_analysis.txt"
            with open(text_path, 'w') as f:
                f.write(text_report)
            print(f"Text report saved to: {text_path}")


if __name__ == "__main__":
    main()
