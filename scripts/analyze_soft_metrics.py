#!/usr/bin/env python3
"""
Analyze soft metrics from the state/analysis directory.

Extracts and aggregates LLM-as-judge evaluations including:
- Bottleneck targeting accuracy
- Approach quality
- Speedup likelihood predictions
- Technique overlap
- Code understanding scores

Usage:
    python scripts/analyze_soft_metrics.py
    python scripts/analyze_soft_metrics.py --repo vllm
    python scripts/analyze_soft_metrics.py --output-dir ./reports --format json
"""

import argparse
import json
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, List, Any

import numpy as np


@dataclass
class SoftMetricsRun:
    """Single run's soft metrics."""
    item_id: str
    commit_hash: str
    repo: str
    agent: str
    model: str

    # Execution metrics
    status: str
    duration_s: float
    tool_calls: int
    errors: int

    # Patch metrics
    patch_generated: bool
    lines_added: int
    lines_removed: int
    files_changed: int

    # LLM scores (1-10)
    code_understanding: float
    task_alignment: float
    approach_quality: float
    execution_quality: float
    overall_score: float

    # Patch quality categories
    bottleneck_target: str  # same_target, related_target, different_target, no_optimization
    approach_comparison: str  # same_approach, similar_approach, valid_alternative, ineffective, harmful
    speedup_likelihood: str  # likely_similar, likely_partial, uncertain, likely_ineffective, likely_regression
    failure_mode: str  # localization_failure, technique_mismatch, incomplete_implementation, etc.

    # Patch similarity
    file_overlap_pct: float
    line_overlap_pct: float
    approach_similarity: float

    # Technique analysis
    human_techniques: List[str] = field(default_factory=list)
    agent_techniques: List[str] = field(default_factory=list)
    technique_overlap: bool = False


@dataclass
class SoftMetricsAnalysis:
    """Aggregate soft metrics analysis."""
    repo: str
    total_runs: int

    # Score distributions
    avg_code_understanding: float
    avg_task_alignment: float
    avg_approach_quality: float
    avg_execution_quality: float
    avg_overall: float

    # Category distributions
    bottleneck_distribution: Dict[str, int]
    approach_distribution: Dict[str, int]
    speedup_distribution: Dict[str, int]
    failure_distribution: Dict[str, int]

    # Similarity metrics
    avg_file_overlap: float
    avg_line_overlap: float
    technique_overlap_rate: float

    # Technique analysis
    human_technique_counts: Dict[str, int]
    agent_technique_counts: Dict[str, int]

    # All runs for detailed analysis
    runs: List[SoftMetricsRun] = field(default_factory=list)


def get_score(data: dict, key: str, default: float = 0) -> float:
    """Extract score from potentially nested structure.

    Handles both flat values (e.g., {"key": 5.0}) and nested
    structures (e.g., {"key": {"score": 5.0, ...}}).
    """
    val = data.get(key, default)
    if isinstance(val, dict):
        return val.get("score", default)
    return val if val is not None else default


def load_soft_metrics(analysis_dir: Path, repo_filter: str = None) -> List[SoftMetricsRun]:
    """Load all soft metrics from analysis directory."""
    runs = []

    # Find all analysis.json files
    for analysis_file in analysis_dir.rglob("analysis.json"):
        try:
            with open(analysis_file) as f:
                data = json.load(f)

            meta = data.get("meta", {})
            repo = meta.get("repo", "unknown")

            # Apply repo filter
            if repo_filter and repo != repo_filter:
                continue

            # Extract metrics
            quant = data.get("quantitative", {})
            qual = data.get("qualitative", {})
            cat = data.get("categorical", {})
            patch_sim = data.get("patch_similarity", {})

            # Get patch quality from separate file if available
            patch_quality_file = analysis_file.parent / "patch_quality.json"
            patch_quality = {}
            if patch_quality_file.exists():
                with open(patch_quality_file) as f:
                    patch_quality = json.load(f)

            run = SoftMetricsRun(
                item_id=meta.get("item_id", "unknown"),
                commit_hash=meta.get("commits", {}).get("human", "unknown"),
                repo=repo,
                agent=meta.get("agent", "unknown"),
                model=meta.get("model", "unknown"),

                # Execution
                status=quant.get("status", "unknown"),
                duration_s=quant.get("duration_s", 0),
                tool_calls=quant.get("tool_calls", 0),
                errors=quant.get("errors", 0),

                # Patch
                patch_generated=quant.get("patch_generated", False),
                lines_added=quant.get("lines_added", 0),
                lines_removed=quant.get("lines_removed", 0),
                files_changed=quant.get("files_changed", 0),

                # Scores (handles both flat and nested {"score": x} format)
                code_understanding=get_score(qual, "code_understanding", 0),
                task_alignment=get_score(qual, "task_alignment", 0),
                approach_quality=get_score(qual, "approach_quality", 0),
                execution_quality=get_score(qual, "execution_quality", 0),
                overall_score=get_score(qual, "overall_score", get_score(qual, "overall", 0)),

                # Categories from patch_quality.json
                bottleneck_target=patch_quality.get("bottleneck_target", {}).get("category", cat.get("bottleneck_target", "unknown")),
                approach_comparison=patch_quality.get("approach_comparison", {}).get("category", cat.get("approach_comparison", "unknown")),
                speedup_likelihood=patch_quality.get("speedup_likelihood", {}).get("category", cat.get("speedup_likelihood", "unknown")),
                failure_mode=patch_quality.get("failure_mode", {}).get("category", cat.get("failure_mode", "unknown")),

                # Similarity
                file_overlap_pct=patch_sim.get("file_overlap_pct", 0),
                line_overlap_pct=patch_sim.get("line_overlap_pct", 0),
                approach_similarity=patch_sim.get("approach_similarity", 0),

                # Techniques
                human_techniques=patch_quality.get("optimization_techniques", {}).get("human_techniques", []),
                agent_techniques=patch_quality.get("optimization_techniques", {}).get("agent_techniques", []),
                technique_overlap=patch_quality.get("optimization_techniques", {}).get("technique_overlap", False),
            )
            runs.append(run)

        except Exception as e:
            print(f"Warning: Failed to load {analysis_file}: {e}", file=sys.stderr)

    return runs


def analyze_soft_metrics(runs: List[SoftMetricsRun], repo: str = "all") -> SoftMetricsAnalysis:
    """Analyze soft metrics and compute aggregates."""
    if not runs:
        return SoftMetricsAnalysis(
            repo=repo, total_runs=0,
            avg_code_understanding=0, avg_task_alignment=0,
            avg_approach_quality=0, avg_execution_quality=0, avg_overall=0,
            bottleneck_distribution={}, approach_distribution={},
            speedup_distribution={}, failure_distribution={},
            avg_file_overlap=0, avg_line_overlap=0, technique_overlap_rate=0,
            human_technique_counts={}, agent_technique_counts={},
            runs=[]
        )

    # Calculate averages
    avg_code_understanding = np.mean([r.code_understanding for r in runs])
    avg_task_alignment = np.mean([r.task_alignment for r in runs])
    avg_approach_quality = np.mean([r.approach_quality for r in runs])
    avg_execution_quality = np.mean([r.execution_quality for r in runs])
    avg_overall = np.mean([r.overall_score for r in runs])

    # Category distributions
    bottleneck_dist = Counter(r.bottleneck_target for r in runs)
    approach_dist = Counter(r.approach_comparison for r in runs)
    speedup_dist = Counter(r.speedup_likelihood for r in runs)
    failure_dist = Counter(r.failure_mode for r in runs if r.failure_mode != "unknown")

    # Similarity metrics
    avg_file_overlap = np.mean([r.file_overlap_pct for r in runs])
    avg_line_overlap = np.mean([r.line_overlap_pct for r in runs])
    technique_overlap_rate = sum(1 for r in runs if r.technique_overlap) / len(runs)

    # Technique counts
    human_techniques = Counter()
    agent_techniques = Counter()
    for r in runs:
        human_techniques.update(r.human_techniques)
        agent_techniques.update(r.agent_techniques)

    return SoftMetricsAnalysis(
        repo=repo,
        total_runs=len(runs),
        avg_code_understanding=avg_code_understanding,
        avg_task_alignment=avg_task_alignment,
        avg_approach_quality=avg_approach_quality,
        avg_execution_quality=avg_execution_quality,
        avg_overall=avg_overall,
        bottleneck_distribution=dict(bottleneck_dist),
        approach_distribution=dict(approach_dist),
        speedup_distribution=dict(speedup_dist),
        failure_distribution=dict(failure_dist),
        avg_file_overlap=avg_file_overlap,
        avg_line_overlap=avg_line_overlap,
        technique_overlap_rate=technique_overlap_rate,
        human_technique_counts=dict(human_techniques),
        agent_technique_counts=dict(agent_techniques),
        runs=runs
    )


def print_report(analysis: SoftMetricsAnalysis):
    """Print formatted analysis report."""
    print("\n" + "="*80)
    print(f"SOFT METRICS ANALYSIS REPORT - {analysis.repo.upper()}")
    print("="*80)
    print(f"Total runs analyzed: {analysis.total_runs}")

    # Scores
    print("\n" + "-"*80)
    print("LLM SCORES (1-10 scale)")
    print("-"*80)
    print(f"  Code Understanding:  {analysis.avg_code_understanding:.2f}")
    print(f"  Task Alignment:      {analysis.avg_task_alignment:.2f}")
    print(f"  Approach Quality:    {analysis.avg_approach_quality:.2f}")
    print(f"  Execution Quality:   {analysis.avg_execution_quality:.2f}")
    print(f"  Overall:             {analysis.avg_overall:.2f}")

    # Bottleneck targeting
    print("\n" + "-"*80)
    print("BOTTLENECK TARGETING")
    print("-"*80)
    for cat, count in sorted(analysis.bottleneck_distribution.items(), key=lambda x: -x[1]):
        pct = count / analysis.total_runs * 100
        print(f"  {cat:25s}: {count:3d} ({pct:5.1f}%)")

    # Approach quality
    print("\n" + "-"*80)
    print("APPROACH COMPARISON")
    print("-"*80)
    for cat, count in sorted(analysis.approach_distribution.items(), key=lambda x: -x[1]):
        pct = count / analysis.total_runs * 100
        print(f"  {cat:25s}: {count:3d} ({pct:5.1f}%)")

    # Speedup likelihood
    print("\n" + "-"*80)
    print("SPEEDUP LIKELIHOOD")
    print("-"*80)
    for cat, count in sorted(analysis.speedup_distribution.items(), key=lambda x: -x[1]):
        pct = count / analysis.total_runs * 100
        print(f"  {cat:25s}: {count:3d} ({pct:5.1f}%)")

    # Calculate success metrics
    likely_success = sum(analysis.speedup_distribution.get(k, 0)
                        for k in ["likely_similar", "likely_partial"])
    success_rate = likely_success / analysis.total_runs * 100 if analysis.total_runs else 0
    print(f"\n  ** Predicted Success Rate: {success_rate:.1f}% **")

    # Failure modes
    if analysis.failure_distribution:
        print("\n" + "-"*80)
        print("FAILURE MODES (for non-successful runs)")
        print("-"*80)
        for cat, count in sorted(analysis.failure_distribution.items(), key=lambda x: -x[1]):
            total_failures = sum(analysis.failure_distribution.values())
            pct = count / total_failures * 100 if total_failures else 0
            print(f"  {cat:25s}: {count:3d} ({pct:5.1f}%)")

    # Similarity metrics
    print("\n" + "-"*80)
    print("PATCH SIMILARITY")
    print("-"*80)
    print(f"  Avg File Overlap:      {analysis.avg_file_overlap:.1f}%")
    print(f"  Avg Line Overlap:      {analysis.avg_line_overlap:.1f}%")
    print(f"  Technique Overlap Rate: {analysis.technique_overlap_rate*100:.1f}%")

    # Techniques
    print("\n" + "-"*80)
    print("TECHNIQUE USAGE")
    print("-"*80)
    print("Human techniques:")
    for tech, count in sorted(analysis.human_technique_counts.items(), key=lambda x: -x[1])[:5]:
        print(f"    {tech:25s}: {count:3d}")
    print("Agent techniques:")
    for tech, count in sorted(analysis.agent_technique_counts.items(), key=lambda x: -x[1])[:5]:
        print(f"    {tech:25s}: {count:3d}")

    print("\n" + "="*80)


def export_commits(runs: List[SoftMetricsRun], output_file: Path):
    """Export commit hashes for matching with hard metrics."""
    commits = []
    for r in runs:
        commits.append({
            "commit_hash": r.commit_hash,
            "item_id": r.item_id,
            "repo": r.repo,
            "overall_score": r.overall_score,
            "speedup_likelihood": r.speedup_likelihood,
            "bottleneck_target": r.bottleneck_target,
            "approach_comparison": r.approach_comparison,
        })

    with open(output_file, 'w') as f:
        json.dump(commits, f, indent=2)
    print(f"Exported {len(commits)} commits to {output_file}")


def main():
    parser = argparse.ArgumentParser(description="Analyze soft metrics from LLM-as-judge evaluations")
    parser.add_argument("--analysis-dir", type=Path,
                       default=Path("perf-agents-bench/state/analysis"),
                       help="Path to analysis directory")
    parser.add_argument("--repo", "-r", choices=["vllm", "sglang", "all"], default="all",
                       help="Filter by repository")
    parser.add_argument("--output-dir", type=Path, help="Output directory for reports")
    parser.add_argument("--format", choices=["text", "json", "both"], default="text")
    parser.add_argument("--export-commits", type=Path, help="Export commit list for matching")
    args = parser.parse_args()

    # Load metrics
    repo_filter = None if args.repo == "all" else args.repo
    runs = load_soft_metrics(args.analysis_dir, repo_filter)

    if not runs:
        print(f"No soft metrics found in {args.analysis_dir}")
        return

    # Analyze
    analysis = analyze_soft_metrics(runs, args.repo)

    # Output
    if args.format in ["text", "both"]:
        print_report(analysis)

    if args.output_dir:
        args.output_dir.mkdir(parents=True, exist_ok=True)

        if args.format in ["json", "both"]:
            # Convert to JSON-serializable format
            output = {
                "repo": analysis.repo,
                "total_runs": analysis.total_runs,
                "scores": {
                    "avg_code_understanding": analysis.avg_code_understanding,
                    "avg_task_alignment": analysis.avg_task_alignment,
                    "avg_approach_quality": analysis.avg_approach_quality,
                    "avg_execution_quality": analysis.avg_execution_quality,
                    "avg_overall": analysis.avg_overall,
                },
                "distributions": {
                    "bottleneck": analysis.bottleneck_distribution,
                    "approach": analysis.approach_distribution,
                    "speedup": analysis.speedup_distribution,
                    "failure": analysis.failure_distribution,
                },
                "similarity": {
                    "avg_file_overlap": analysis.avg_file_overlap,
                    "avg_line_overlap": analysis.avg_line_overlap,
                    "technique_overlap_rate": analysis.technique_overlap_rate,
                },
                "techniques": {
                    "human": analysis.human_technique_counts,
                    "agent": analysis.agent_technique_counts,
                },
            }
            json_path = args.output_dir / f"soft_metrics_{args.repo}.json"
            with open(json_path, 'w') as f:
                json.dump(output, f, indent=2)
            print(f"Saved JSON report to {json_path}")

    if args.export_commits:
        export_commits(runs, args.export_commits)


if __name__ == "__main__":
    main()
