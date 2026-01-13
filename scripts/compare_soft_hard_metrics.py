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

# GSO High-Level Category Descriptions
GSO_HIGH_LEVEL_DESC = {
    "Localization": (
        "Agent failed to correctly identify the performance bottleneck. "
        "Either targeted the wrong component entirely, or made changes that "
        "were technically valid but addressed a less critical bottleneck than "
        "what the human identified."
    ),
    "Avoid Complexity": (
        "Agent recognized the bottleneck but avoided the necessary complexity "
        "to fix it properly. Made superficial or incomplete changes rather than "
        "implementing the full solution, often staying at a higher abstraction "
        "level (e.g., Python) when lower-level work (e.g., CUDA/C++) was required."
    ),
    "Mismanage Compute": (
        "Agent wasted computational resources through inefficient exploration, "
        "over-engineered solutions, or catastrophic failures. Includes cases "
        "where the agent destroyed/corrupted the repository or spent excessive "
        "time exploring without producing actionable optimizations."
    ),
}

# GSO Sub-Category Descriptions
GSO_SUBCATEGORY_DESC = {
    "Misdiagnosed Bottlenecks": (
        "Agent optimized the wrong component. Failed to identify the actual "
        "performance bottleneck that the human targeted. May have made valid "
        "optimizations, but to code that wasn't the critical path."
    ),
    "Less Impactful": (
        "Agent identified a related bottleneck but chose a less impactful "
        "optimization target. The changes were valid but addressed a secondary "
        "concern rather than the primary performance issue."
    ),
    "Lazy Optimization": (
        "Agent made superficial changes that avoid the deeper work required. "
        "Typically involves config tweaks, parameter adjustments, or minor "
        "refactors when algorithmic or architectural changes were needed."
    ),
    "Wrong Abstraction Level": (
        "Agent worked at the wrong level of the stack. Common pattern: staying "
        "in Python when the human wrote CUDA kernels, or modifying high-level "
        "APIs when low-level implementation changes were required."
    ),
    "Exploit-Heavy": (
        "Agent over-engineered the solution with unnecessary complexity. Added "
        "excessive abstractions, unnecessary features, or convoluted logic when "
        "a simpler approach would have sufficed."
    ),
    "Explore-Heavy": (
        "Agent spent most of its steps examining the codebase (reading files, "
        "searching) without converging on actionable optimizations. High "
        "explore-to-edit ratio (>40% read/search, ≤2 edits) with no success."
    ),
    "Destructive/Runaway": (
        "Catastrophic failure where agent corrupted or destroyed the repository. "
        "Includes cases of runaway edits, deletion of critical files, or changes "
        "that made the codebase unbuildable/unusable."
    ),
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
    # New fields for success analysis
    human_techniques: List[str] = field(default_factory=list)
    agent_techniques: List[str] = field(default_factory=list)
    task_domain: str = "unknown"
    key_differences: List[str] = field(default_factory=list)


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
                # New fields for success analysis
                human_techniques=pq.get("optimization_techniques", {}).get("human_techniques", []),
                agent_techniques=pq.get("optimization_techniques", {}).get("agent_techniques", []),
                task_domain=pq.get("task_analysis", {}).get("domain", "unknown"),
                key_differences=pq.get("observations", {}).get("key_differences", []),
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
                status=row.get('status', 'completed'),
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
                status=row.get('status', 'completed'),
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
                status=row.get('status', 'completed'),
                metric_type="tpot",
                baseline_value=baseline_val,
                human_value=human_val,
                agent_value=agent_val,
                agent_vs_human_pct=diff_pct,
                human_vs_baseline_pct=human_vs_baseline,
                agent_vs_baseline_pct=agent_vs_baseline,
            )
            continue

        # Check for ITL metrics (lower is better)
        if row.get('agent_itl_mean') is not None and row.get('human_itl_mean') is not None:
            baseline_val = row.get('baseline_itl_mean')
            human_val = row['human_itl_mean']
            agent_val = row['agent_itl_mean']
            # Invert so positive means agent is better (lower ITL)
            diff_pct = -((agent_val - human_val) / human_val) * 100 if human_val != 0 else 0
            # For ITL, lower is better, so invert (positive = improved)
            human_vs_baseline = -((human_val - baseline_val) / baseline_val) * 100 if baseline_val else None
            agent_vs_baseline = -((agent_val - baseline_val) / baseline_val) * 100 if baseline_val else None

            hard_metrics[commit_hash] = HardMetrics(
                status=row.get('status', 'completed'),
                metric_type="itl",
                baseline_value=baseline_val,
                human_value=human_val,
                agent_value=agent_val,
                agent_vs_human_pct=diff_pct,
                human_vs_baseline_pct=human_vs_baseline,
                agent_vs_baseline_pct=agent_vs_baseline,
            )
            continue

        # Check for latency_avg metrics (lower is better) - standalone mode
        if row.get('agent_latency_avg') is not None and row.get('human_latency_avg') is not None:
            baseline_val = row.get('baseline_latency_avg')
            human_val = row['human_latency_avg']
            agent_val = row['agent_latency_avg']
            # Invert so positive means agent is better (lower latency)
            diff_pct = -((agent_val - human_val) / human_val) * 100 if human_val != 0 else 0
            # For latency, lower is better, so invert (positive = improved)
            human_vs_baseline = -((human_val - baseline_val) / baseline_val) * 100 if baseline_val else None
            agent_vs_baseline = -((agent_val - baseline_val) / baseline_val) * 100 if baseline_val else None

            hard_metrics[commit_hash] = HardMetrics(
                status=row.get('status', 'completed'),
                metric_type="latency",
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


# Success Analysis Categories - Implementation Alignment
IMPLEMENTATION_ALIGNMENT = {
    "identical": "Agent produced exactly same code as human",
    "core_match_extras": "Got main optimization, added extra changes/noise",
    "alternative_technique": "Same bottleneck, different valid solution method",
    "alternative_target": "Different bottleneck, still valid optimization",
}

IMPLEMENTATION_ALIGNMENT_DESC = {
    "identical": (
        "Agent produced functionally identical code to the human solution. "
        "Same optimization technique, same target, same implementation approach. "
        "This is the ideal outcome - agent perfectly replicated human expertise."
    ),
    "core_match_extras": (
        "Agent correctly identified and implemented the core optimization, but "
        "included additional unnecessary changes. Common patterns: unrelated CI/CD "
        "modifications, extra file touches, documentation changes, or redundant "
        "refactoring alongside the main fix. The signal is correct but noisy."
    ),
    "alternative_technique": (
        "Agent targeted the same performance bottleneck as the human but used a "
        "different valid optimization technique. Example: human used argmax fast-path, "
        "agent used single-mask operation - both valid solutions to the same problem. "
        "Shows agent creativity while maintaining correctness."
    ),
    "alternative_target": (
        "Agent optimized a different bottleneck than the human, but the optimization "
        "was still valid and produced measurable improvement. Example: human fixed "
        "benchmark logic, agent optimized serialization. Both are legitimate "
        "performance wins, just different problem interpretations."
    ),
}

# Success Analysis Categories - Agent Behavior Flags
AGENT_BEHAVIOR_DESC = {
    "over_engineering": (
        "Agent applied significantly more optimization techniques than the human "
        "(2+ additional techniques). Suggests a 'shotgun approach' - trying many "
        "optimizations hoping one works, rather than surgical precision. May indicate "
        "uncertainty about which technique will be effective."
    ),
    "diff_pollution": (
        "Agent included unrelated changes in the diff: CI/CD pipeline modifications, "
        "Dockerfile changes, BuildKite configs, documentation updates, or build scripts. "
        "These changes don't contribute to the optimization and add noise to the patch."
    ),
    "scope_creep": (
        "Agent modified files beyond the necessary scope. Examples: touching C++/CUDA "
        "files when human stayed in Python, modifying test files unnecessarily, or "
        "making changes to unrelated modules. Increases review burden and risk."
    ),
    "clean_patch": (
        "Agent produced a surgical, minimal patch similar to what a human would write. "
        "No unnecessary changes, focused scope, appropriate technique selection. "
        "This is the ideal behavior pattern - efficient and precise."
    ),
}


def classify_success_alignment(sm: SoftMetrics) -> str:
    """
    Classify successful commit's implementation alignment.

    Categories:
    - identical: Agent produced exactly same code (same_approach)
    - core_match_extras: Got main optimization, added extras (similar_approach)
    - alternative_technique: Same target, different method (valid_alternative + same_target)
    - alternative_target: Different bottleneck, still valid (valid_alternative + related/different target)
    """
    approach = sm.approach_comparison
    bottleneck = sm.bottleneck_target

    if approach == "same_approach":
        return "identical"
    elif approach == "similar_approach":
        return "core_match_extras"
    elif approach == "valid_alternative":
        if bottleneck == "same_target":
            return "alternative_technique"
        else:  # related_target, different_target, other
            return "alternative_target"
    return "unknown"


def detect_agent_behavior_flags(sm: SoftMetrics) -> List[str]:
    """
    Detect agent behavior patterns from techniques and observations.

    Flags (can co-occur):
    - over_engineering: Agent used significantly more techniques than human
    - diff_pollution: Included unrelated CI/CD/build changes (detected from key_differences)
    - scope_creep: Modified extra files (detected from key_differences)
    - clean_patch: Surgical, minimal changes (no issues detected)
    """
    flags = []

    human_set = set(sm.human_techniques)
    agent_set = set(sm.agent_techniques)

    # Over-engineering: agent used 2+ more techniques than human
    if len(agent_set) > len(human_set) + 1:
        flags.append("over_engineering")

    # Check key_differences for pollution/scope creep indicators
    diff_text = " ".join(sm.key_differences).lower()

    pollution_keywords = ["ci/cd", "buildkite", ".github", "dockerfile", "unrelated",
                          "documentation", "build script", "thousands of lines"]
    if any(kw in diff_text for kw in pollution_keywords):
        flags.append("diff_pollution")

    scope_keywords = ["additional file", "extra file", "modified c++", "cuda kernel",
                      "modified files", "touched", "beyond"]
    if any(kw in diff_text for kw in scope_keywords):
        flags.append("scope_creep")

    # If no issues, it's a clean patch
    if not flags:
        flags.append("clean_patch")

    return flags


def compute_success_analysis(soft_metrics: Dict[str, SoftMetrics]) -> Dict:
    """
    Analyze successful commits: implementation alignment and agent behavior.

    Returns detailed breakdown of how agent succeeded and what patterns emerged.
    """
    successes = [sm for sm in soft_metrics.values()
                 if sm.failure_mode == "not_applicable"]

    if not successes:
        return {"total_successes": 0}

    # Alignment categories
    alignment_counts = Counter()
    # Behavior flags (can stack)
    behavior_counts = Counter()
    # Technique counts
    human_technique_counts = Counter()
    agent_technique_counts = Counter()
    # Domain distribution
    domain_counts = Counter()
    # Technique match analysis
    technique_match = {
        "exact_match": 0,       # Same techniques
        "agent_superset": 0,    # Agent used all human techniques + more
        "agent_subset": 0,      # Agent used subset of human techniques
        "partial_overlap": 0,   # Some overlap
        "no_overlap": 0,        # Completely different
    }

    for sm in successes:
        # Classification
        alignment = classify_success_alignment(sm)
        alignment_counts[alignment] += 1

        # Behavior flags
        flags = detect_agent_behavior_flags(sm)
        for flag in flags:
            behavior_counts[flag] += 1

        # Techniques
        for t in sm.human_techniques:
            human_technique_counts[t] += 1
        for t in sm.agent_techniques:
            agent_technique_counts[t] += 1

        # Domain
        domain_counts[sm.task_domain] += 1

        # Technique match
        human_set = set(sm.human_techniques)
        agent_set = set(sm.agent_techniques)
        if human_set == agent_set:
            technique_match["exact_match"] += 1
        elif human_set <= agent_set and human_set:
            technique_match["agent_superset"] += 1
        elif agent_set <= human_set and agent_set:
            technique_match["agent_subset"] += 1
        elif human_set & agent_set:
            technique_match["partial_overlap"] += 1
        else:
            technique_match["no_overlap"] += 1

    return {
        "total_successes": len(successes),
        "implementation_alignment": dict(alignment_counts.most_common()),
        "agent_behavior_flags": dict(behavior_counts.most_common()),
        "human_techniques": dict(human_technique_counts.most_common()),
        "agent_techniques": dict(agent_technique_counts.most_common()),
        "task_domains": dict(domain_counts.most_common()),
        "technique_match": technique_match,
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

    # Success Analysis
    print("\n" + "-"*90)
    print("SUCCESS ANALYSIS (Human vs Agent Techniques)")
    print("-"*90)

    success_analysis = compute_success_analysis(soft_metrics)
    total_successes = success_analysis.get("total_successes", 0)

    if total_successes > 0:
        print(f"\nTotal Successes: {total_successes} ({total_successes/total_runs*100:.1f}% success rate)")

        print(f"\nImplementation Alignment:")
        for alignment, count in success_analysis["implementation_alignment"].items():
            pct = count / total_successes * 100
            bar = generate_ascii_bar(count, total_successes)
            desc = IMPLEMENTATION_ALIGNMENT.get(alignment, "")
            print(f"  {alignment:<22} {count:>3} ({pct:>5.1f}%)  {bar}  {desc}")

        print(f"\nAgent Behavior Patterns:")
        for flag, count in success_analysis["agent_behavior_flags"].items():
            pct = count / total_successes * 100
            bar = generate_ascii_bar(count, total_successes)
            print(f"  {flag:<22} {count:>3} ({pct:>5.1f}%)  {bar}")

        print(f"\nTechnique Distribution:")
        print(f"  {'Human':<30} {'Agent':<30}")
        human_techs = list(success_analysis["human_techniques"].items())
        agent_techs = list(success_analysis["agent_techniques"].items())
        max_rows = max(len(human_techs), len(agent_techs))
        for i in range(max_rows):
            h_str = f"{human_techs[i][0]}: {human_techs[i][1]}" if i < len(human_techs) else ""
            a_str = f"{agent_techs[i][0]}: {agent_techs[i][1]}" if i < len(agent_techs) else ""
            print(f"  {h_str:<30} {a_str:<30}")

        print(f"\nTechnique Match (per commit):")
        for match_type, count in success_analysis["technique_match"].items():
            if count > 0:
                pct = count / total_successes * 100
                print(f"  {match_type:<20} {count:>3} ({pct:>5.1f}%)")

        print(f"\nTask Domains:")
        for domain, count in success_analysis["task_domains"].items():
            pct = count / total_successes * 100
            print(f"  {domain:<15} {count:>3} ({pct:>5.1f}%)")
    else:
        print("\n  No successful commits found.")

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

    # GSO Category Definitions
    report += """

### Failure Category Definitions

#### High-Level Categories

| Category | Description |
|----------|-------------|
"""
    for cat, desc in GSO_HIGH_LEVEL_DESC.items():
        report += f"| **{cat}** | {desc} |\n"

    report += """

#### Sub-Category Definitions

| Sub-Category | Description |
|--------------|-------------|
"""
    for subcat, desc in GSO_SUBCATEGORY_DESC.items():
        report += f"| **{subcat}** | {desc} |\n"

    # Success Analysis
    success_analysis = compute_success_analysis(soft_metrics)
    total_successes = success_analysis.get("total_successes", 0)

    if total_successes > 0:
        report += f"""

## Success Analysis (Human vs Agent)

**Total Successes:** {total_successes} ({total_successes/total_runs*100:.1f}% success rate)

### Implementation Alignment

| Category | Count | % | Description |
|----------|-------|---|-------------|
"""
        for alignment, count in success_analysis["implementation_alignment"].items():
            pct = count / total_successes * 100
            desc = IMPLEMENTATION_ALIGNMENT.get(alignment, "")
            report += f"| {alignment} | {count} | {pct:.1f}% | {desc} |\n"

        report += """

### Agent Behavior Patterns

| Pattern | Count | % |
|---------|-------|---|
"""
        for flag, count in success_analysis["agent_behavior_flags"].items():
            pct = count / total_successes * 100
            report += f"| {flag} | {count} | {pct:.1f}% |\n"

        report += """

### Technique Distribution

| Human Techniques | Count | Agent Techniques | Count |
|------------------|-------|------------------|-------|
"""
        human_techs = list(success_analysis["human_techniques"].items())
        agent_techs = list(success_analysis["agent_techniques"].items())
        max_rows = max(len(human_techs), len(agent_techs))
        for i in range(max_rows):
            h_name = human_techs[i][0] if i < len(human_techs) else ""
            h_count = human_techs[i][1] if i < len(human_techs) else ""
            a_name = agent_techs[i][0] if i < len(agent_techs) else ""
            a_count = agent_techs[i][1] if i < len(agent_techs) else ""
            report += f"| {h_name} | {h_count} | {a_name} | {a_count} |\n"

        report += """

### Task Domains

| Domain | Count | % |
|--------|-------|---|
"""
        for domain, count in success_analysis["task_domains"].items():
            pct = count / total_successes * 100
            report += f"| {domain} | {count} | {pct:.1f}% |\n"

        # Success Category Definitions
        report += """

### Success Category Definitions

#### Implementation Alignment Categories

| Category | Description |
|----------|-------------|
"""
        for cat, desc in IMPLEMENTATION_ALIGNMENT_DESC.items():
            report += f"| **{cat}** | {desc} |\n"

        report += """

#### Agent Behavior Patterns

| Pattern | Description |
|---------|-------------|
"""
        for pattern, desc in AGENT_BEHAVIOR_DESC.items():
            report += f"| **{pattern}** | {desc} |\n"

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
            success_analysis = compute_success_analysis(soft_metrics)
            output = {
                "generated_at": datetime.now().isoformat(),
                "soft_metrics_count": len(soft_metrics),
                "hard_metrics_count": len(hard_metrics),
                "matched_count": len(matches),
                "prediction_accuracy": sum(1 for m in matches if m.prediction_correct) / len(matches) if matches else 0,
                "gso_distribution": gso_dist,
                "success_analysis": success_analysis,
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
