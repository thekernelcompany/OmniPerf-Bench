#!/usr/bin/env python3
"""
Compare soft metrics (LLM-as-judge) with hard metrics (runtime benchmarks).

This script:
1. Loads soft metrics from patch_quality.json files in state/analysis/vllm/
2. Loads hard metrics from HuggingFace dataset
3. Matches commits by hash
4. Compares predictions vs actual outcomes
5. Generates alignment analysis

Usage:
    python scripts/compare_soft_hard_metrics.py
    python scripts/compare_soft_hard_metrics.py --output-dir ./reports
"""

import argparse
import json
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, List, Any, Tuple

import numpy as np

try:
    from datasets import load_dataset
except ImportError:
    print("Error: 'datasets' package not installed. Run: pip install datasets")
    sys.exit(1)


HARD_METRICS_DATASET = "Inferencebench/claude-code-vllm-benchmarks"


# GSO-style failure category mapping (arxiv:2505.23671)
GSO_MAPPING = {
    "localization_failure": ("Localization", "Misdiagnosed Bottlenecks"),
    "technique_mismatch": ("Localization", "Less Impactful"),
    "complexity_avoidance": ("Avoid Complexity", "Lazy Optimization"),
    "incomplete_implementation": ("Avoid Complexity", "Wrong Abstraction Level"),
    "overcomplicated": ("Mismanage Compute", "Exploit-Heavy"),
    "not_applicable": ("Success", "N/A"),
    # "other" = catastrophic failures where agent destroyed/corrupted the repository
    "other": ("Mismanage Compute", "Destructive/Runaway"),
}


def detect_explore_heavy(tool_calls: Dict, failure_mode: str) -> bool:
    """
    Detect explore-heavy behavior from tool_calls data.

    GSO defines explore-heavy as: "spend most of their steps examining
    the codebase without converging on actionable optimizations"

    Detection criteria:
    - explore_ratio = (read + search) / total > 0.40 (above 1.5x mean of 0.26)
    - AND editor <= 2 (few actual edits made)
    - AND failure_mode != "not_applicable" (not a success)
    """
    if not tool_calls or failure_mode == "not_applicable":
        return False
    total = tool_calls.get("total", 0)
    if total == 0:
        return False
    explore_ratio = (tool_calls.get("read", 0) + tool_calls.get("search", 0)) / total
    editor_count = tool_calls.get("editor", 0)
    return explore_ratio > 0.40 and editor_count <= 2


def map_to_gso_category(failure_mode: str, tool_calls: Optional[Dict] = None) -> Tuple[str, str]:
    """
    Map failure_mode to GSO (high_level, subcategory), with explore-heavy override.

    Returns tuple of (high_level_category, subcategory).
    """
    # Check for explore-heavy first (overrides other categories)
    if tool_calls and detect_explore_heavy(tool_calls, failure_mode):
        return ("Mismanage Compute", "Explore-Heavy")
    return GSO_MAPPING.get(failure_mode, ("Unknown", "Unknown"))


@dataclass
class SoftMetrics:
    """Soft metrics from patch_quality.json."""
    item_id: str
    speedup_likelihood: str
    failure_mode: str
    technique_overlap: bool
    bottleneck_target: str
    approach_comparison: str
    tool_calls: Optional[Dict] = None  # For explore-heavy detection


@dataclass
class HardMetrics:
    """Hard metrics from runtime benchmarks."""
    status: str
    metric_type: str  # "throughput", "ttft", "latency"
    baseline_value: Optional[float]
    human_value: float
    agent_value: float
    agent_vs_human_pct: float  # positive = agent better
    human_vs_baseline_pct: Optional[float] = None  # positive = human improved over baseline
    agent_vs_baseline_pct: Optional[float] = None  # positive = agent improved over baseline

    def baseline_str(self) -> str:
        """Format baseline value for display."""
        return f"{self.baseline_value:.1f}" if self.baseline_value is not None else "N/A"

    def human_improvement_str(self) -> str:
        """Format human improvement over baseline."""
        return f"{self.human_vs_baseline_pct:+.1f}%" if self.human_vs_baseline_pct is not None else "N/A"

    def agent_improvement_str(self) -> str:
        """Format agent improvement over baseline."""
        return f"{self.agent_vs_baseline_pct:+.1f}%" if self.agent_vs_baseline_pct is not None else "N/A"


@dataclass
class MatchedCommit:
    """A commit with both soft and hard metrics."""
    commit_hash: str
    soft: SoftMetrics
    hard: HardMetrics
    prediction_correct: bool


def load_soft_metrics(analysis_dir: Path) -> Dict[str, SoftMetrics]:
    """
    Load soft metrics from patch_quality.json files.

    Returns dict keyed by commit hash (first 12 chars).
    """
    soft_metrics = {}

    for pq_file in analysis_dir.rglob("patch_quality.json"):
        try:
            # Get commit hash from corresponding analysis.json
            analysis_file = pq_file.parent / "analysis.json"
            if not analysis_file.exists():
                continue

            with open(analysis_file) as f:
                analysis = json.load(f)

            commit_hash = analysis.get("meta", {}).get("commits", {}).get("human", "")
            if not commit_hash or commit_hash == "unknown":
                continue

            # Load patch quality metrics
            with open(pq_file) as f:
                pq = json.load(f)

            short_hash = commit_hash[:12]
            # Get tool_calls from analysis.json for explore-heavy detection
            tool_calls = analysis.get("quantitative", {}).get("tool_calls")

            soft_metrics[short_hash] = SoftMetrics(
                item_id=analysis.get("meta", {}).get("item_id", "unknown"),
                speedup_likelihood=pq.get("speedup_likelihood", {}).get("category", "unknown"),
                failure_mode=pq.get("failure_mode", {}).get("category", "unknown"),
                technique_overlap=pq.get("optimization_techniques", {}).get("technique_overlap", False),
                bottleneck_target=pq.get("bottleneck_target", {}).get("category", "unknown"),
                approach_comparison=pq.get("approach_comparison", {}).get("category", "unknown"),
                tool_calls=tool_calls,
            )

        except Exception as e:
            print(f"Warning: Failed to load {pq_file}: {e}", file=sys.stderr)

    return soft_metrics


def load_hard_metrics(dataset_name: str = HARD_METRICS_DATASET) -> Dict[str, HardMetrics]:
    """
    Load hard metrics from HuggingFace dataset.

    Returns dict keyed by commit hash (first 12 chars).
    Only includes commits with both agent and human metrics.
    """
    print(f"Loading hard metrics from {dataset_name}...")
    ds = load_dataset(dataset_name, split='train')

    hard_metrics = {}

    for row in ds:
        commit_hash = row['commit_hash'][:12]

        # Check for throughput metrics (higher is better) - unified schema uses throughput
        if row.get('agent_throughput') is not None and row.get('human_throughput') is not None:
            baseline_val = row.get('baseline_throughput')
            human_val = row['human_throughput']
            agent_val = row['agent_throughput']
            # Positive means agent is better
            diff_pct = ((agent_val - human_val) / human_val) * 100 if human_val != 0 else 0
            # For throughput, higher is better
            human_vs_baseline = ((human_val - baseline_val) / baseline_val) * 100 if baseline_val else None
            agent_vs_baseline = ((agent_val - baseline_val) / baseline_val) * 100 if baseline_val else None

            hard_metrics[commit_hash] = HardMetrics(
                status=row['status'],
                metric_type="throughput",
                baseline_value=baseline_val,
                human_value=human_val,
                agent_value=agent_val,
                agent_vs_human_pct=diff_pct,
                human_vs_baseline_pct=human_vs_baseline,
                agent_vs_baseline_pct=agent_vs_baseline,
            )
            continue

        # Check for TTFT metrics (lower is better) - unified schema uses ttft_mean
        if row.get('agent_ttft_mean') is not None and row.get('human_ttft_mean') is not None:
            baseline_val = row.get('baseline_ttft_mean')
            human_val = row['human_ttft_mean']
            agent_val = row['agent_ttft_mean']
            # Invert so positive means agent is better (lower TTFT)
            diff_pct = -((agent_val - human_val) / human_val) * 100 if human_val != 0 else 0
            # For TTFT, lower is better, so invert (positive = improved)
            human_vs_baseline = -((human_val - baseline_val) / baseline_val) * 100 if baseline_val else None
            agent_vs_baseline = -((agent_val - baseline_val) / baseline_val) * 100 if baseline_val else None

            hard_metrics[commit_hash] = HardMetrics(
                status=row['status'],
                metric_type="ttft",
                baseline_value=baseline_val,
                human_value=human_val,
                agent_value=agent_val,
                agent_vs_human_pct=diff_pct,
                human_vs_baseline_pct=human_vs_baseline,
                agent_vs_baseline_pct=agent_vs_baseline,
            )
            continue

        # Check for TPOT metrics (lower is better) - unified schema uses tpot_mean
        if row.get('agent_tpot_mean') is not None and row.get('human_tpot_mean') is not None:
            baseline_val = row.get('baseline_tpot_mean')
            human_val = row['human_tpot_mean']
            agent_val = row['agent_tpot_mean']
            # Invert so positive means agent is better (lower TPOT)
            diff_pct = -((agent_val - human_val) / human_val) * 100 if human_val != 0 else 0
            # For TPOT, lower is better, so invert (positive = improved)
            human_vs_baseline = -((human_val - baseline_val) / baseline_val) * 100 if baseline_val else None
            agent_vs_baseline = -((agent_val - baseline_val) / baseline_val) * 100 if baseline_val else None

            hard_metrics[commit_hash] = HardMetrics(
                status=row['status'],
                metric_type="tpot",
                baseline_value=baseline_val,
                human_value=human_val,
                agent_value=agent_val,
                agent_vs_human_pct=diff_pct,
                human_vs_baseline_pct=human_vs_baseline,
                agent_vs_baseline_pct=agent_vs_baseline,
            )

    return hard_metrics


def evaluate_prediction(soft: SoftMetrics, hard: HardMetrics, threshold: float = 2.0) -> bool:
    """
    Evaluate if soft metric prediction was correct.

    Prediction logic:
    - likely_similar/likely_partial: Predicts agent WILL improve over human (diff > threshold)
    - likely_ineffective: Predicts agent will NOT improve (diff <= threshold)
    - likely_regression: Predicts agent will be WORSE (diff < -threshold)

    Args:
        soft: Soft metrics with speedup_likelihood prediction
        hard: Hard metrics with actual agent vs human diff
        threshold: Percentage threshold for "improvement" (default 2%)

    Returns:
        True if prediction was correct
    """
    diff = hard.agent_vs_human_pct
    prediction = soft.speedup_likelihood

    if prediction == "likely_similar":
        # Predicts agent matches or exceeds human
        return diff >= -threshold

    elif prediction == "likely_partial":
        # Predicts agent shows some improvement
        return diff > threshold

    elif prediction == "likely_ineffective":
        # Predicts agent won't improve over human
        return diff <= threshold

    elif prediction == "likely_regression":
        # Predicts agent will be worse
        return diff < -threshold

    else:  # uncertain, unknown
        return False  # Can't evaluate


def match_and_evaluate(
    soft_metrics: Dict[str, SoftMetrics],
    hard_metrics: Dict[str, HardMetrics]
) -> List[MatchedCommit]:
    """Match commits and evaluate predictions."""
    matches = []

    for commit_hash in set(soft_metrics.keys()) & set(hard_metrics.keys()):
        soft = soft_metrics[commit_hash]
        hard = hard_metrics[commit_hash]

        correct = evaluate_prediction(soft, hard)

        matches.append(MatchedCommit(
            commit_hash=commit_hash,
            soft=soft,
            hard=hard,
            prediction_correct=correct,
        ))

    return matches


def compute_gso_distribution(soft_metrics: Dict[str, SoftMetrics]) -> Dict:
    """
    Compute GSO-style failure category distribution from soft metrics.

    GSO Figure 7 only categorizes MODEL FAILURES, so we exclude successes
    (failure_mode="not_applicable") from this distribution.

    Returns dict with high-level and subcategory distributions.
    """
    high_level_counts = Counter()
    subcategory_counts = Counter()
    success_count = 0
    total_runs = len(soft_metrics)

    for sm in soft_metrics.values():
        # Skip successes - GSO only categorizes failures
        if sm.failure_mode == "not_applicable":
            success_count += 1
            continue

        high, sub = map_to_gso_category(sm.failure_mode, sm.tool_calls)
        high_level_counts[high] += 1
        subcategory_counts[sub] += 1

    failure_count = total_runs - success_count

    return {
        "high_level": dict(high_level_counts.most_common()),
        "subcategory": dict(subcategory_counts.most_common()),
        "total_runs": total_runs,
        "success_count": success_count,
        "failure_count": failure_count,
    }


def generate_ascii_bar(count: int, total: int, max_width: int = 20) -> str:
    """Generate ASCII bar chart representation."""
    if total == 0:
        return ""
    width = int((count / total) * max_width)
    return "█" * width


def print_report(
    soft_metrics: Dict[str, SoftMetrics],
    hard_metrics: Dict[str, HardMetrics],
    matches: List[MatchedCommit]
):
    """Print detailed comparison report."""

    print("\n" + "="*90)
    print("SOFT vs HARD METRICS COMPARISON REPORT")
    print("="*90)

    print("\n" + "-"*90)
    print("DATASET OVERVIEW")
    print("-"*90)
    print(f"  Soft metrics (vllm patch_quality.json): {len(soft_metrics)}")
    print(f"  Hard metrics (HuggingFace dataset):     {len(hard_metrics)}")
    print(f"  Matched commits:                        {len(matches)}")

    # Soft metrics distribution
    print("\n" + "-"*90)
    print("SOFT METRICS DISTRIBUTION (speedup_likelihood)")
    print("-"*90)
    speedup_counts = Counter(s.speedup_likelihood for s in soft_metrics.values())
    for cat, count in speedup_counts.most_common():
        print(f"  {cat}: {count} ({count/len(soft_metrics)*100:.1f}%)")

    # Detailed comparison table
    print("\n" + "-"*120)
    print("DETAILED COMPARISON (matched commits with hard metrics)")
    print("-"*120)
    print(f"{'Commit':<14} {'Metric':<7} {'Baseline':>10} {'Human':>10} {'Agent':>10} {'H vs B':>8} {'A vs B':>8} {'A vs H':>8} {'Soft Prediction':<20} {'OK?'}")
    print("-"*120)

    for m in sorted(matches, key=lambda x: -x.hard.agent_vs_human_pct):
        correct_symbol = "✓" if m.prediction_correct else "✗"
        baseline_str = m.hard.baseline_str()
        h_vs_b = m.hard.human_improvement_str()
        a_vs_b = m.hard.agent_improvement_str()
        print(f"{m.commit_hash:<14} {m.hard.metric_type:<7} {baseline_str:>10} {m.hard.human_value:>10.1f} {m.hard.agent_value:>10.1f} {h_vs_b:>8} {a_vs_b:>8} {m.hard.agent_vs_human_pct:>+7.1f}% {m.soft.speedup_likelihood:<20} {correct_symbol}")

    # Prediction accuracy
    print("\n" + "-"*90)
    print("PREDICTION ACCURACY")
    print("-"*90)

    correct = sum(1 for m in matches if m.prediction_correct)
    total = len(matches)
    accuracy = correct / total * 100 if total > 0 else 0

    print(f"  Correct predictions: {correct}/{total} ({accuracy:.1f}%)")

    # Breakdown by prediction type
    print("\n  By prediction type:")
    for pred_type in ["likely_similar", "likely_partial", "likely_ineffective", "likely_regression"]:
        pred_matches = [m for m in matches if m.soft.speedup_likelihood == pred_type]
        if pred_matches:
            pred_correct = sum(1 for m in pred_matches if m.prediction_correct)
            print(f"    {pred_type}: {pred_correct}/{len(pred_matches)} correct")

    # Technique overlap analysis
    print("\n" + "-"*90)
    print("TECHNIQUE OVERLAP ANALYSIS")
    print("-"*90)

    overlap_true = [m for m in matches if m.soft.technique_overlap]
    overlap_false = [m for m in matches if not m.soft.technique_overlap]

    if overlap_true:
        avg_diff_true = np.mean([m.hard.agent_vs_human_pct for m in overlap_true])
        print(f"  technique_overlap=True  (n={len(overlap_true)}): avg diff = {avg_diff_true:+.2f}%")
    if overlap_false:
        avg_diff_false = np.mean([m.hard.agent_vs_human_pct for m in overlap_false])
        print(f"  technique_overlap=False (n={len(overlap_false)}): avg diff = {avg_diff_false:+.2f}%")

    # GSO Failure Category Analysis
    print("\n" + "-"*90)
    print("GSO FAILURE CATEGORY ANALYSIS (arxiv:2505.23671 Figure 7 style)")
    print("-"*90)

    gso_dist = compute_gso_distribution(soft_metrics)
    total_runs = gso_dist["total_runs"]
    success_count = gso_dist["success_count"]
    failure_count = gso_dist["failure_count"]

    print(f"\nTotal runs: {total_runs} | Successes: {success_count} ({success_count/total_runs*100:.1f}%) | Failures: {failure_count} ({failure_count/total_runs*100:.1f}%)")

    print(f"\nFailure High-Level Distribution (n={failure_count}):")
    for category, count in gso_dist["high_level"].items():
        pct = count / failure_count * 100 if failure_count > 0 else 0
        bar = generate_ascii_bar(count, failure_count)
        print(f"  {category:<20} {count:>4} ({pct:>5.1f}%)  {bar}")

    print(f"\nFailure Sub-Category Breakdown:")
    for subcategory, count in gso_dist["subcategory"].items():
        pct = count / failure_count * 100 if failure_count > 0 else 0
        print(f"  {subcategory:<25} {count:>4} ({pct:>5.1f}%)")

    print("\n" + "="*90)


def generate_markdown_report(
    soft_metrics: Dict[str, SoftMetrics],
    hard_metrics: Dict[str, HardMetrics],
    matches: List[MatchedCommit],
    output_path: Path
):
    """Generate markdown report."""

    correct = sum(1 for m in matches if m.prediction_correct)
    accuracy = correct / len(matches) * 100 if matches else 0

    report = f"""# Soft vs Hard Metrics Comparison Analysis

**Generated:** {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

## Overview

| Metric | Value |
|--------|-------|
| Soft metrics (vllm) | {len(soft_metrics)} |
| Hard metrics (HuggingFace) | {len(hard_metrics)} |
| Matched commits | {len(matches)} |
| **Prediction Accuracy** | **{accuracy:.1f}%** ({correct}/{len(matches)}) |

## Speedup Likelihood Distribution (Soft Metrics)

"""
    speedup_counts = Counter(s.speedup_likelihood for s in soft_metrics.values())
    report += "| Category | Count | % |\n|----------|-------|---|\n"
    for cat, count in speedup_counts.most_common():
        report += f"| {cat} | {count} | {count/len(soft_metrics)*100:.1f}% |\n"

    report += """

## Detailed Comparison

| Commit | Metric | Baseline | Human | Agent | H vs B | A vs B | A vs H | Soft Prediction | Correct? |
|--------|--------|----------|-------|-------|--------|--------|--------|-----------------|----------|
"""
    for m in sorted(matches, key=lambda x: -x.hard.agent_vs_human_pct):
        correct_symbol = "✓" if m.prediction_correct else "✗"
        baseline_str = m.hard.baseline_str()
        h_vs_b = m.hard.human_improvement_str()
        a_vs_b = m.hard.agent_improvement_str()
        report += f"| `{m.commit_hash}` | {m.hard.metric_type} | {baseline_str} | {m.hard.human_value:.1f} | {m.hard.agent_value:.1f} | {h_vs_b} | {a_vs_b} | {m.hard.agent_vs_human_pct:+.1f}% | {m.soft.speedup_likelihood} | {correct_symbol} |\n"

    report += f"""

## Prediction Accuracy by Category

"""
    for pred_type in ["likely_similar", "likely_partial", "likely_ineffective", "likely_regression"]:
        pred_matches = [m for m in matches if m.soft.speedup_likelihood == pred_type]
        if pred_matches:
            pred_correct = sum(1 for m in pred_matches if m.prediction_correct)
            report += f"- **{pred_type}**: {pred_correct}/{len(pred_matches)} correct\n"

    report += """

## Key Findings

"""
    if accuracy >= 60:
        report += f"1. **Good predictive accuracy** ({accuracy:.1f}%): Soft metrics are reasonably reliable predictors of hard metric outcomes.\n"
    else:
        report += f"1. **Moderate predictive accuracy** ({accuracy:.1f}%): Soft metrics have room for improvement in predicting hard outcomes.\n"

    # Analyze false negatives/positives
    false_negatives = [m for m in matches if not m.prediction_correct and
                       m.soft.speedup_likelihood in ["likely_ineffective", "likely_regression"] and
                       m.hard.agent_vs_human_pct > 2]
    false_positives = [m for m in matches if not m.prediction_correct and
                       m.soft.speedup_likelihood in ["likely_similar", "likely_partial"] and
                       m.hard.agent_vs_human_pct < -2]

    if false_negatives:
        report += f"2. **{len(false_negatives)} false negatives**: Soft metrics predicted failure but agent actually improved.\n"
    if false_positives:
        report += f"3. **{len(false_positives)} false positives**: Soft metrics predicted success but agent actually regressed.\n"

    # GSO Failure Category Analysis
    gso_dist = compute_gso_distribution(soft_metrics)
    total_runs = gso_dist["total_runs"]
    success_count = gso_dist["success_count"]
    failure_count = gso_dist["failure_count"]

    report += f"""

## GSO Failure Category Analysis

*Based on arxiv:2505.23671 Figure 7 methodology - categorizing model failures only*

**Summary:** {total_runs} total runs | {success_count} successes ({success_count/total_runs*100:.1f}%) | {failure_count} failures ({failure_count/total_runs*100:.1f}%)

### Failure High-Level Distribution (n={failure_count})

| Category | Count | % |
|----------|-------|---|
"""
    for category, count in gso_dist["high_level"].items():
        pct = count / failure_count * 100 if failure_count > 0 else 0
        report += f"| {category} | {count} | {pct:.1f}% |\n"

    report += f"""

### Failure Sub-Category Breakdown

| Sub-Category | Count | % |
|--------------|-------|---|
"""
    for subcategory, count in gso_dist["subcategory"].items():
        pct = count / failure_count * 100 if failure_count > 0 else 0
        report += f"| {subcategory} | {count} | {pct:.1f}% |\n"

    with open(output_path, 'w') as f:
        f.write(report)

    print(f"Saved markdown report to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Compare soft and hard metrics")
    parser.add_argument("--analysis-dir", type=Path,
                       default=Path("perf-agents-bench/state/analysis/vllm"),
                       help="Path to vllm analysis directory with patch_quality.json files")
    parser.add_argument("--hard-dataset", default=HARD_METRICS_DATASET,
                       help="HuggingFace dataset for hard metrics")
    parser.add_argument("--output-dir", type=Path,
                       default=Path("perf-agents-bench/analysis_reports/hard_evals_analysis"),
                       help="Output directory for reports")
    parser.add_argument("--format", choices=["text", "markdown", "json", "all"], default="all")
    parser.add_argument("--output-suffix", default="",
                       help="Suffix for output files (e.g., '_sglang' for soft_vs_hard_metrics_comparison_sglang.md)")
    args = parser.parse_args()

    # Load metrics
    print("Loading soft metrics from patch_quality.json files...")
    soft_metrics = load_soft_metrics(args.analysis_dir)
    print(f"Loaded {len(soft_metrics)} soft metric runs")

    hard_metrics = load_hard_metrics(args.hard_dataset)
    print(f"Loaded {len(hard_metrics)} hard metric commits")

    # Match and evaluate
    matches = match_and_evaluate(soft_metrics, hard_metrics)
    print(f"Matched {len(matches)} commits with both metrics")

    # Output
    if args.format in ["text", "all"]:
        print_report(soft_metrics, hard_metrics, matches)

    if args.output_dir:
        args.output_dir.mkdir(parents=True, exist_ok=True)

        if args.format in ["markdown", "all"]:
            md_path = args.output_dir / f"soft_vs_hard_metrics_comparison{args.output_suffix}.md"
            generate_markdown_report(soft_metrics, hard_metrics, matches, md_path)

        if args.format in ["json", "all"]:
            json_path = args.output_dir / f"soft_hard_comparison{args.output_suffix}.json"
            gso_dist = compute_gso_distribution(soft_metrics)
            output = {
                "generated_at": datetime.now().isoformat(),
                "soft_metrics_count": len(soft_metrics),
                "hard_metrics_count": len(hard_metrics),
                "matched_count": len(matches),
                "prediction_accuracy": sum(1 for m in matches if m.prediction_correct) / len(matches) if matches else 0,
                "gso_distribution": gso_dist,
                "matches": [
                    {
                        "commit": m.commit_hash,
                        "soft_speedup_likelihood": m.soft.speedup_likelihood,
                        "soft_technique_overlap": m.soft.technique_overlap,
                        "soft_failure_mode": m.soft.failure_mode,
                        "hard_metric_type": m.hard.metric_type,
                        "hard_baseline_value": m.hard.baseline_value,
                        "hard_human_value": m.hard.human_value,
                        "hard_agent_value": m.hard.agent_value,
                        "hard_human_vs_baseline_pct": m.hard.human_vs_baseline_pct,
                        "hard_agent_vs_baseline_pct": m.hard.agent_vs_baseline_pct,
                        "hard_agent_vs_human_pct": m.hard.agent_vs_human_pct,
                        "prediction_correct": m.prediction_correct,
                    }
                    for m in matches
                ]
            }
            with open(json_path, 'w') as f:
                json.dump(output, f, indent=2)
            print(f"Saved JSON to {json_path}")


if __name__ == "__main__":
    main()
