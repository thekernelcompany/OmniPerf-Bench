#!/usr/bin/env python3
"""
Plot speedup likelihood distribution histogram for SGLang and vLLM.

This script loads speedup_likelihood categories from patch_quality.json files
and plots a grouped bar chart comparing the distribution across repositories.

Usage:
    python scripts/plot_speedup_distribution.py
    python scripts/plot_speedup_distribution.py --hard-only  # Only commits with hard metrics
    python scripts/plot_speedup_distribution.py --output-dir ./plots
"""

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

try:
    from datasets import load_dataset
    HAS_DATASETS = True
except ImportError:
    HAS_DATASETS = False


# Color palette for repos
REPO_COLORS = {
    "sglang": "#2ecc71",  # Green
    "vllm": "#3498db",    # Blue
}

# HuggingFace datasets for hard metrics
HARD_METRICS_DATASETS = {
    "vllm": "Inferencebench/claude-code-vllm-benchmarks",
    "sglang": "Inferencebench/claude-code-sglang-benchmarks_v1",
}

# Speedup likelihood ordering (for consistent x-axis)
SPEEDUP_ORDER = [
    "likely_similar",
    "likely_partial",
    "uncertain",
    "likely_ineffective",
    "likely_regression",
]


def load_speedup_from_comparison(comparison_file: Path) -> Counter:
    """Load speedup_likelihood counts from comparison JSON (hard metrics only)."""
    counts = Counter()

    if not comparison_file.exists():
        print(f"Warning: Comparison file not found: {comparison_file}", file=sys.stderr)
        return counts

    try:
        with open(comparison_file) as f:
            data = json.load(f)
        for match in data.get("matches", []):
            category = match.get("soft_speedup_likelihood", "unknown")
            counts[category] += 1
    except Exception as e:
        print(f"Warning: Failed to load {comparison_file}: {e}", file=sys.stderr)

    return counts


def has_valid_hard_metrics(row) -> bool:
    """Check if row has both human and agent metrics for any metric type."""
    metric_pairs = [
        ('human_throughput', 'agent_throughput'),
        ('human_ttft_mean', 'agent_ttft_mean'),
        ('human_tpot_mean', 'agent_tpot_mean'),
        ('human_itl_mean', 'agent_itl_mean'),
        ('human_latency_avg', 'agent_latency_avg'),
    ]
    for h_col, a_col in metric_pairs:
        if row.get(h_col) is not None and row.get(a_col) is not None:
            return True
    return False


def load_hard_metrics_commits(repo: str) -> set:
    """Load commit hashes that have hard metrics from HuggingFace dataset."""
    if not HAS_DATASETS:
        print("Warning: 'datasets' package not installed. Run: pip install datasets", file=sys.stderr)
        return set()

    dataset_name = HARD_METRICS_DATASETS.get(repo)
    if not dataset_name:
        print(f"Warning: No hard metrics dataset for repo: {repo}", file=sys.stderr)
        return set()

    try:
        ds = load_dataset(dataset_name, split="train")
        commits = set()
        for row in ds:
            # Check for ANY valid metric (not just ttft_mean)
            if not has_valid_hard_metrics(row):
                continue

            # Try different column names for commit hash
            commit = row.get("commit_hash") or row.get("commit_short") or row.get("human_commit")
            if commit:
                # Normalize to first 12 chars for matching
                commits.add(commit[:12])
        return commits
    except Exception as e:
        print(f"Warning: Failed to load dataset {dataset_name}: {e}", file=sys.stderr)
        return set()


def load_speedup_with_hard_metrics(analysis_dir: Path, repo: str) -> Counter:
    """Load speedup_likelihood only for commits that have hard metrics."""
    counts = Counter()

    # Get commits with hard metrics
    hard_commits = load_hard_metrics_commits(repo)
    if not hard_commits:
        print(f"  No hard metrics commits found for {repo}", file=sys.stderr)
        return counts

    print(f"  Found {len(hard_commits)} commits with hard metrics for {repo}")

    # Load soft metrics and match
    repo_dir = analysis_dir / repo
    if not repo_dir.exists():
        print(f"Warning: Directory not found: {repo_dir}", file=sys.stderr)
        return counts

    matched = 0
    for pq_file in repo_dir.rglob("patch_quality.json"):
        try:
            # Get commit hash from analysis.json in same directory
            analysis_file = pq_file.parent / "analysis.json"
            if not analysis_file.exists():
                continue

            with open(analysis_file) as f:
                analysis = json.load(f)

            commit = analysis.get("meta", {}).get("commits", {}).get("human", "")
            if not commit:
                continue

            # Normalize and check if in hard metrics
            commit_short = commit[:12]
            if commit_short not in hard_commits:
                continue

            # Load speedup likelihood
            with open(pq_file) as f:
                data = json.load(f)

            speedup = data.get("speedup_likelihood")
            if speedup is None:
                category = "unknown"
            else:
                category = speedup.get("category", "unknown")

            counts[category] += 1
            matched += 1

        except Exception as e:
            continue  # Skip silently for cleaner output

    print(f"  Matched {matched} commits with soft metrics")
    return counts


def load_speedup_counts(analysis_dir: Path, repo: str) -> Counter:
    """Load speedup_likelihood counts from patch_quality.json files."""
    counts = Counter()
    repo_dir = analysis_dir / repo

    if not repo_dir.exists():
        print(f"Warning: Directory not found: {repo_dir}", file=sys.stderr)
        return counts

    for pq_file in repo_dir.rglob("patch_quality.json"):
        try:
            with open(pq_file) as f:
                data = json.load(f)
            speedup = data.get("speedup_likelihood")
            if speedup is None:
                category = "unknown"
            else:
                category = speedup.get("category", "unknown")
            counts[category] += 1
        except Exception as e:
            print(f"Warning: Failed to load {pq_file}: {e}", file=sys.stderr)

    return counts


def plot_speedup_histogram(
    sglang_counts: Counter,
    vllm_counts: Counter,
    output_path: Path = None,
    hard_only: bool = False,
) -> plt.Figure:
    """Create grouped bar chart comparing speedup likelihood distributions."""
    fig, ax = plt.subplots(figsize=(12, 6))

    # Filter to only categories that exist in the data
    all_cats = set(sglang_counts.keys()) | set(vllm_counts.keys())
    categories = [c for c in SPEEDUP_ORDER if c in all_cats]
    # Add any unknown categories at the end
    categories.extend([c for c in sorted(all_cats) if c not in SPEEDUP_ORDER])

    x = np.arange(len(categories))
    width = 0.35

    # SGLang bars
    sglang_vals = [sglang_counts.get(cat, 0) for cat in categories]
    bars1 = ax.bar(
        x - width/2, sglang_vals, width,
        label=f"SGLang (n={sum(sglang_counts.values())})",
        color=REPO_COLORS["sglang"], alpha=0.8
    )

    # vLLM bars
    vllm_vals = [vllm_counts.get(cat, 0) for cat in categories]
    bars2 = ax.bar(
        x + width/2, vllm_vals, width,
        label=f"vLLM (n={sum(vllm_counts.values())})",
        color=REPO_COLORS["vllm"], alpha=0.8
    )

    # Add count labels on bars
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            if height > 0:
                ax.annotate(
                    f"{int(height)}",
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha="center", va="bottom", fontsize=10
                )

    # Labels and formatting
    ax.set_xlabel("Speedup Likelihood Category", fontsize=12)
    ax.set_ylabel("Number of Commits", fontsize=12)
    title = "Speedup Likelihood Distribution: SGLang vs vLLM"
    if hard_only:
        title += " (Hard Metrics Only)"
    ax.set_title(title, fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels([c.replace("_", " ").title() for c in categories], rotation=45, ha="right")
    ax.legend(loc="upper right")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()

    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        print(f"Saved: {output_path}")

    return fig


def main():
    parser = argparse.ArgumentParser(
        description="Plot speedup likelihood distribution for SGLang and vLLM"
    )
    parser.add_argument(
        "--analysis-dir", type=Path,
        default=Path("perf-agents-bench/state/analysis"),
        help="Path to analysis directory"
    )
    parser.add_argument(
        "--comparison-dir", type=Path,
        default=Path("perf-agents-bench/analysis_reports/hard_evals_analysis"),
        help="Path to comparison JSON files (for --hard-only)"
    )
    parser.add_argument(
        "--output-dir", type=Path,
        default=Path("perf-agents-bench/analysis_reports/plots"),
        help="Output directory for plots"
    )
    parser.add_argument(
        "--hard-only", action="store_true",
        help="Only include commits that have hard metrics available"
    )
    parser.add_argument(
        "--no-show", action="store_true",
        help="Don't display plot interactively"
    )
    args = parser.parse_args()

    # Load data
    if args.hard_only:
        print("Loading speedup likelihood data (hard metrics only)...")
        print("Loading hard metrics from HuggingFace datasets...")
        sglang_counts = load_speedup_with_hard_metrics(args.analysis_dir, "sglang")
        vllm_counts = load_speedup_with_hard_metrics(args.analysis_dir, "vllm")
    else:
        print("Loading speedup likelihood data...")
        sglang_counts = load_speedup_counts(args.analysis_dir, "sglang")
        vllm_counts = load_speedup_counts(args.analysis_dir, "vllm")

    print(f"  SGLang: {sum(sglang_counts.values())} runs")
    print(f"  vLLM: {sum(vllm_counts.values())} runs")

    # Print summary
    print("\nSpeedup Likelihood Distribution:")
    print("-" * 50)
    print(f"{'Category':<25} {'SGLang':>10} {'vLLM':>10}")
    print("-" * 50)
    all_cats = set(sglang_counts.keys()) | set(vllm_counts.keys())
    for cat in SPEEDUP_ORDER:
        if cat in all_cats:
            print(f"{cat:<25} {sglang_counts.get(cat, 0):>10} {vllm_counts.get(cat, 0):>10}")
    for cat in sorted(all_cats - set(SPEEDUP_ORDER)):
        print(f"{cat:<25} {sglang_counts.get(cat, 0):>10} {vllm_counts.get(cat, 0):>10}")
    print("-" * 50)

    # Plot
    if args.hard_only:
        output_path = args.output_dir / "speedup_likelihood_histogram_hard_only.png"
    else:
        output_path = args.output_dir / "speedup_likelihood_histogram.png"
    plot_speedup_histogram(sglang_counts, vllm_counts, output_path, hard_only=args.hard_only)

    if not args.no_show:
        plt.show()


if __name__ == "__main__":
    main()
