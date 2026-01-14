#!/usr/bin/env python3
"""
Plot correlation between soft metrics (LLM predictions) and hard metrics (runtime outcomes).

Generates:
1. Box plot: Distribution of agent_vs_human_pct by speedup_likelihood category
2. Bar chart: Prediction accuracy by category
3. Combined figure for paper

Usage:
    python scripts/plot_soft_hard_correlation.py
    python scripts/plot_soft_hard_correlation.py --input path/to/soft_hard_comparison.json
"""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# Color scheme
COLORS = {
    'likely_similar': '#2ecc71',      # Green - success
    'likely_partial': '#f39c12',      # Orange - partial
    'likely_ineffective': '#e74c3c',  # Red - ineffective
    'likely_regression': '#9b59b6',   # Purple - regression
    'uncertain': '#95a5a6',           # Gray - uncertain
}

# Order for display
CATEGORY_ORDER = [
    'likely_similar',
    'likely_partial',
    'likely_ineffective',
    'likely_regression',
    'uncertain',
]

# Human-readable labels
CATEGORY_LABELS = {
    'likely_similar': 'Likely\nSimilar',
    'likely_partial': 'Likely\nPartial',
    'likely_ineffective': 'Likely\nIneffective',
    'likely_regression': 'Likely\nRegression',
    'uncertain': 'Uncertain',
}


def load_comparison_data(json_path: Path) -> dict:
    """Load comparison data from JSON file."""
    with open(json_path) as f:
        return json.load(f)


def plot_outcome_distribution(matches: list, output_path: Path):
    """
    Plot box plot showing distribution of agent_vs_human_pct by prediction category.

    This shows: for each soft metric prediction, what was the actual runtime outcome?
    """
    # Group outcomes by prediction
    outcomes_by_category = {cat: [] for cat in CATEGORY_ORDER}

    for match in matches:
        cat = match['soft_speedup_likelihood']
        diff = match['hard_agent_vs_human_pct']
        if cat in outcomes_by_category:
            outcomes_by_category[cat].append(diff)

    # Filter to categories with data
    categories = [c for c in CATEGORY_ORDER if outcomes_by_category[c]]
    data = [outcomes_by_category[c] for c in categories]
    colors = [COLORS[c] for c in categories]
    labels = [CATEGORY_LABELS[c] for c in categories]

    # Create figure
    fig, ax = plt.subplots(figsize=(10, 6))

    # Box plot
    bp = ax.boxplot(data, patch_artist=True, labels=labels)

    # Color the boxes
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    # Add individual points
    for i, (cat_data, color) in enumerate(zip(data, colors)):
        x = np.random.normal(i + 1, 0.04, size=len(cat_data))
        ax.scatter(x, cat_data, alpha=0.6, color=color, edgecolor='black', s=50, zorder=3)

    # Add horizontal line at 0 (human baseline)
    ax.axhline(y=0, color='black', linestyle='--', linewidth=1, alpha=0.5, label='Human baseline')

    # Add threshold lines
    ax.axhline(y=2, color='green', linestyle=':', linewidth=1, alpha=0.5, label='Success threshold (+2%)')
    ax.axhline(y=-2, color='red', linestyle=':', linewidth=1, alpha=0.5, label='Regression threshold (-2%)')

    # Labels and title
    ax.set_xlabel('Soft Metric Prediction (speedup_likelihood)', fontsize=12)
    ax.set_ylabel('Actual Outcome: Agent vs Human (%)', fontsize=12)
    ax.set_title('Runtime Outcomes by LLM Prediction Category\n(vLLM, n=39 matched commits)', fontsize=14)

    # Add count annotations
    for i, (cat, cat_data) in enumerate(zip(categories, data)):
        ax.annotate(f'n={len(cat_data)}', xy=(i + 1, ax.get_ylim()[1]),
                   ha='center', va='bottom', fontsize=10, color='gray')

    ax.legend(loc='upper right', fontsize=9)
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()


def plot_prediction_accuracy(matches: list, output_path: Path):
    """
    Plot bar chart showing prediction accuracy by category.
    """
    # Calculate accuracy by category
    accuracy_data = {}

    for cat in CATEGORY_ORDER:
        cat_matches = [m for m in matches if m['soft_speedup_likelihood'] == cat]
        if cat_matches:
            correct = sum(1 for m in cat_matches if m['prediction_correct'])
            total = len(cat_matches)
            accuracy_data[cat] = {
                'correct': correct,
                'total': total,
                'accuracy': correct / total * 100
            }

    # Filter to categories with data
    categories = [c for c in CATEGORY_ORDER if c in accuracy_data]
    accuracies = [accuracy_data[c]['accuracy'] for c in categories]
    totals = [accuracy_data[c]['total'] for c in categories]
    corrects = [accuracy_data[c]['correct'] for c in categories]
    colors = [COLORS[c] for c in categories]
    labels = [CATEGORY_LABELS[c].replace('\n', ' ') for c in categories]

    # Create figure
    fig, ax = plt.subplots(figsize=(10, 6))

    # Bar chart
    x = np.arange(len(categories))
    bars = ax.bar(x, accuracies, color=colors, alpha=0.8, edgecolor='black')

    # Add accuracy labels on bars
    for i, (bar, acc, corr, tot) in enumerate(zip(bars, accuracies, corrects, totals)):
        height = bar.get_height()
        ax.annotate(f'{acc:.0f}%\n({corr}/{tot})',
                   xy=(bar.get_x() + bar.get_width() / 2, height),
                   ha='center', va='bottom', fontsize=11, fontweight='bold')

    # Labels and title
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=11)
    ax.set_xlabel('Soft Metric Prediction Category', fontsize=12)
    ax.set_ylabel('Prediction Accuracy (%)', fontsize=12)
    ax.set_title('Soft Metric Prediction Accuracy vs Hard Metrics\n(vLLM, n=39 matched commits)', fontsize=14)

    ax.set_ylim(0, 120)  # Leave room for labels
    ax.axhline(y=50, color='gray', linestyle='--', linewidth=1, alpha=0.5, label='Random baseline')
    ax.legend(loc='upper right')
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()


def plot_combined_figure(matches: list, output_path: Path):
    """
    Create combined figure with both plots side by side for paper.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # === Left plot: Outcome distribution ===
    outcomes_by_category = {cat: [] for cat in CATEGORY_ORDER}
    for match in matches:
        cat = match['soft_speedup_likelihood']
        diff = match['hard_agent_vs_human_pct']
        if cat in outcomes_by_category:
            outcomes_by_category[cat].append(diff)

    categories = [c for c in CATEGORY_ORDER if outcomes_by_category[c]]
    data = [outcomes_by_category[c] for c in categories]
    colors = [COLORS[c] for c in categories]
    labels = [CATEGORY_LABELS[c] for c in categories]

    bp = ax1.boxplot(data, patch_artist=True, labels=labels)
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    for i, (cat_data, color) in enumerate(zip(data, colors)):
        x = np.random.normal(i + 1, 0.04, size=len(cat_data))
        ax1.scatter(x, cat_data, alpha=0.6, color=color, edgecolor='black', s=40, zorder=3)

    ax1.axhline(y=0, color='black', linestyle='--', linewidth=1, alpha=0.5)
    ax1.axhline(y=2, color='green', linestyle=':', linewidth=1, alpha=0.3)
    ax1.axhline(y=-2, color='red', linestyle=':', linewidth=1, alpha=0.3)

    ax1.set_xlabel('Soft Metric Prediction', fontsize=11)
    ax1.set_ylabel('Agent vs Human (%)', fontsize=11)
    ax1.set_title('(a) Runtime Outcomes by Prediction', fontsize=12, fontweight='bold')
    ax1.grid(axis='y', alpha=0.3)

    # === Right plot: Accuracy ===
    accuracy_data = {}
    for cat in CATEGORY_ORDER:
        cat_matches = [m for m in matches if m['soft_speedup_likelihood'] == cat]
        if cat_matches:
            correct = sum(1 for m in cat_matches if m['prediction_correct'])
            total = len(cat_matches)
            accuracy_data[cat] = {'correct': correct, 'total': total, 'accuracy': correct / total * 100}

    categories2 = [c for c in CATEGORY_ORDER if c in accuracy_data]
    accuracies = [accuracy_data[c]['accuracy'] for c in categories2]
    totals = [accuracy_data[c]['total'] for c in categories2]
    corrects = [accuracy_data[c]['correct'] for c in categories2]
    colors2 = [COLORS[c] for c in categories2]
    labels2 = [CATEGORY_LABELS[c].replace('\n', ' ') for c in categories2]

    x = np.arange(len(categories2))
    bars = ax2.bar(x, accuracies, color=colors2, alpha=0.8, edgecolor='black')

    for bar, acc, corr, tot in zip(bars, accuracies, corrects, totals):
        height = bar.get_height()
        ax2.annotate(f'{acc:.0f}%\n({corr}/{tot})',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    ha='center', va='bottom', fontsize=10, fontweight='bold')

    ax2.set_xticks(x)
    ax2.set_xticklabels(labels2, fontsize=10)
    ax2.set_xlabel('Soft Metric Prediction', fontsize=11)
    ax2.set_ylabel('Accuracy (%)', fontsize=11)
    ax2.set_title('(b) Prediction Accuracy', fontsize=12, fontweight='bold')
    ax2.set_ylim(0, 120)
    ax2.axhline(y=50, color='gray', linestyle='--', linewidth=1, alpha=0.5)
    ax2.grid(axis='y', alpha=0.3)

    plt.suptitle('Soft vs Hard Metrics Correlation (vLLM, n=39)', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Plot soft vs hard metrics correlation")
    parser.add_argument("--input", type=Path,
                       default=Path("perf-agents-bench/analysis_reports/hard_evals_analysis_v2/soft_hard_comparison_vllm.json"),
                       help="Path to comparison JSON file")
    parser.add_argument("--output-dir", type=Path,
                       default=Path("perf-agents-bench/analysis_reports/hard_evals_analysis_v2/plots"),
                       help="Output directory for plots")
    args = parser.parse_args()

    # Load data
    print(f"Loading data from {args.input}...")
    data = load_comparison_data(args.input)
    matches = data['matches']
    print(f"Loaded {len(matches)} matched commits")

    # Create output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Generate plots
    plot_outcome_distribution(matches, args.output_dir / "soft_hard_outcome_distribution.png")
    plot_prediction_accuracy(matches, args.output_dir / "soft_hard_prediction_accuracy.png")
    plot_combined_figure(matches, args.output_dir / "soft_hard_correlation_combined.png")

    print("\nDone!")


if __name__ == "__main__":
    main()
