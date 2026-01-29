#!/usr/bin/env python3
"""
Benchmark Data Verification Script (v2 - Correlation-Based Validity)

This script verifies the integrity of benchmark data from the HuggingFace dataset
(Inferencebench/claude-code-vllm-benchmarks) using CORRELATION ANALYSIS.

Validity is determined by whether agent TTFT values correlate with human TTFT values:
- High correlation (r > 0.7): Agent ran same benchmark config as human -> VALID
- Low correlation (r < 0.7): Agent ran different benchmark config -> INVALID

Usage:
    source bench-env/bin/activate
    python scripts/verify_benchmark_data.py
"""

import json
from collections import defaultdict
from dataclasses import dataclass
from typing import Optional
import numpy as np
from datasets import load_dataset

CORRELATION_THRESHOLD = 0.7

# README caveats - commits with known data quality issues
README_CAVEATS = {
    "infrastructure_failures": ["6ce01f30", "e7523c2e"],
    "partial_data_issues": ["83450458", "19d98e0c"],
    "universal_failures": ["7c01f706", "ad8d696a", "d7740ea4", "660470e5", "fc7b8d1e",
                           "ce6bf3a2", "ccf02fcb", "35fad35a", "3a243095", "6dd94dbe",
                           "e3580537", "9ed82e70"],
    "unbenchmarkable": ["3a243095", "7c01f706", "80aa7e91", "8bc68e19", "9ed82e70",
                        "ad8d696a", "cf2f084d"],
    "non_standard": ["ccf02fcb", "ce6bf3a2"]
}

ALL_PROBLEM_COMMITS = set()
for commits in README_CAVEATS.values():
    ALL_PROBLEM_COMMITS.update(commits)


@dataclass
class ComparisonResult:
    commit: str
    agent: str
    baseline_ttft: Optional[float]
    human_ttft: Optional[float]
    agent_ttft: Optional[float]
    improvement_pct: Optional[float]
    has_baseline: bool
    comparison_type: str  # "full" (B/H/A) or "partial" (H/A only)
    has_caveat: bool = False
    caveat_reasons: list = None


def load_hf_dataset():
    """Load the HuggingFace dataset."""
    print("Loading HuggingFace dataset...")
    ds = load_dataset("Inferencebench/claude-code-vllm-benchmarks", split="train")
    print(f"Loaded {len(ds)} rows")
    return ds


def calculate_correlation(x, y):
    """Calculate Pearson correlation coefficient."""
    if len(x) < 3:
        return None
    return np.corrcoef(x, y)[0, 1]


def analyze_agent_validity(ds):
    """Analyze validity of each agent's data using correlation."""
    agent_data = defaultdict(lambda: {"human": [], "agent": [], "baseline": [], "commits": []})

    for row in ds:
        if row.get('benchmark_mode') != 'serving':
            continue

        agent = row.get('agent_name', '')
        human_ttft = row.get('human_ttft_mean')
        agent_ttft = row.get('agent_ttft_mean')
        baseline_ttft = row.get('baseline_ttft_mean')
        commit = row.get('commit_short', '')

        if human_ttft is not None and agent_ttft is not None:
            agent_data[agent]["human"].append(human_ttft)
            agent_data[agent]["agent"].append(agent_ttft)
            agent_data[agent]["baseline"].append(baseline_ttft)
            agent_data[agent]["commits"].append(commit)

    print("\n" + "=" * 80)
    print("CORRELATION-BASED VALIDITY ANALYSIS")
    print("=" * 80)
    print(f"\nValidity threshold: r > {CORRELATION_THRESHOLD}")
    print("Low correlation = agent ran different benchmark config than human\n")

    validity_results = {}

    for agent in sorted(agent_data.keys()):
        data = agent_data[agent]
        n = len(data["human"])

        if n < 3:
            validity_results[agent] = {
                "n": n,
                "correlation": None,
                "is_valid": False,
                "reason": "insufficient_data"
            }
            continue

        human = np.array(data["human"])
        agent_arr = np.array(data["agent"])

        corr = calculate_correlation(agent_arr, human)
        is_valid = corr > CORRELATION_THRESHOLD if corr else False

        # Calculate distribution stats
        agent_cv = np.std(agent_arr) / np.mean(agent_arr) if np.mean(agent_arr) > 0 else 0
        human_cv = np.std(human) / np.mean(human) if np.mean(human) > 0 else 0

        validity_results[agent] = {
            "n": n,
            "correlation": corr,
            "is_valid": is_valid,
            "reason": "valid" if is_valid else "config_mismatch",
            "agent_range": (float(np.min(agent_arr)), float(np.max(agent_arr))),
            "human_range": (float(np.min(human)), float(np.max(human))),
            "agent_cv": agent_cv,
            "human_cv": human_cv
        }

        status = "VALID" if is_valid else "CONFIG MISMATCH"
        print(f"\n{agent}:")
        print(f"  Comparisons: {n}")
        print(f"  Correlation (r): {corr:.3f}" if corr else "  Correlation: N/A")
        print(f"  Status: {status}")
        print(f"  Agent TTFT range: {np.min(agent_arr):.1f} - {np.max(agent_arr):.1f} ms (CV={agent_cv:.2f})")
        print(f"  Human TTFT range: {np.min(human):.1f} - {np.max(human):.1f} ms (CV={human_cv:.2f})")

        if not is_valid and corr:
            print(f"  ⚠ LOW CORRELATION: Agent values don't track human values")

    return validity_results, agent_data


def get_caveat_reasons(commit):
    """Get list of caveat reasons for a commit."""
    reasons = []
    for caveat_type, commits in README_CAVEATS.items():
        if commit in commits:
            reasons.append(caveat_type)
    return reasons


def extract_valid_comparisons(ds, validity_results):
    """Extract all valid comparisons for valid agents."""
    comparisons = []

    for row in ds:
        if row.get('benchmark_mode') != 'serving':
            continue

        agent = row.get('agent_name', '')
        if agent not in validity_results or not validity_results[agent]["is_valid"]:
            continue

        human = row.get('human_ttft_mean')
        agent_ttft = row.get('agent_ttft_mean')
        baseline = row.get('baseline_ttft_mean')
        commit = row.get('commit_short', '')

        if human is None or agent_ttft is None:
            continue

        improvement = ((human - agent_ttft) / human) * 100
        caveat_reasons = get_caveat_reasons(commit)

        comparisons.append(ComparisonResult(
            commit=commit,
            agent=agent,
            baseline_ttft=baseline,
            human_ttft=human,
            agent_ttft=agent_ttft,
            improvement_pct=improvement,
            has_baseline=baseline is not None,
            comparison_type="full" if baseline is not None else "partial",
            has_caveat=len(caveat_reasons) > 0,
            caveat_reasons=caveat_reasons
        ))

    return comparisons


def main():
    ds = load_hf_dataset()

    # Analyze validity by agent
    validity_results, agent_data = analyze_agent_validity(ds)

    # Extract valid comparisons
    comparisons = extract_valid_comparisons(ds, validity_results)

    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)

    print(f"\n{'Agent':<15} {'Comparisons':>12} {'Correlation':>12} {'Status':>15}")
    print("-" * 55)

    for agent, results in sorted(validity_results.items()):
        corr_str = f"{results['correlation']:.3f}" if results['correlation'] else "N/A"
        status = "VALID" if results["is_valid"] else "INVALID"
        print(f"{agent:<15} {results['n']:>12} {corr_str:>12} {status:>15}")

    # Show valid comparisons
    if comparisons:
        # Separate clean vs flagged
        clean = [c for c in comparisons if not c.has_caveat]
        flagged = [c for c in comparisons if c.has_caveat]

        full = [c for c in clean if c.comparison_type == "full"]
        partial = [c for c in clean if c.comparison_type == "partial"]

        print(f"\nTotal comparisons: {len(comparisons)}")
        print(f"  Flagged by README caveats: {len(flagged)}")
        print(f"  Clean (no caveats): {len(clean)} ({len(full)} full, {len(partial)} partial)")

        # Show flagged commits
        if flagged:
            print(f"\n--- FLAGGED COMMITS (README caveats) ---")
            for c in sorted(flagged, key=lambda x: -x.improvement_pct):
                reasons = ", ".join(c.caveat_reasons) if c.caveat_reasons else "unknown"
                print(f"  {c.commit}: {c.improvement_pct:+.2f}% - {reasons}")

        # Sort clean by improvement
        clean.sort(key=lambda x: -x.improvement_pct)

        print(f"\n--- CLEAN COMPARISONS ---")
        print(f"{'Commit':<12} {'Agent':<14} {'Human':>10} {'Agent':>10} {'Improv':>10} {'Type':>8}")
        print("-" * 70)

        for c in clean:
            print(f"{c.commit:<12} {c.agent:<14} {c.human_ttft:>8.1f}ms {c.agent_ttft:>8.1f}ms {c.improvement_pct:>+8.2f}% {c.comparison_type:>8}")

        # Recalculate full after sorting
        full = [c for c in clean if c.comparison_type == "full"]

        # Statistics for clean full comparisons only
        if full:
            beats = sum(1 for c in full if c.improvement_pct > 1)
            similar = sum(1 for c in full if -1 <= c.improvement_pct <= 1)
            worse = sum(1 for c in full if c.improvement_pct < -1)

            print(f"\n--- CLEAN FULL COMPARISONS STATS ---")
            print(f"Results: {beats} beats ({beats/len(full)*100:.0f}%), {similar} similar ({similar/len(full)*100:.0f}%), {worse} worse ({worse/len(full)*100:.0f}%)")
            print(f"Average improvement: {np.mean([c.improvement_pct for c in full]):+.2f}%")
            print(f"Best: {full[0].commit} ({full[0].improvement_pct:+.2f}%)")
            worst = min(full, key=lambda x: x.improvement_pct)
            print(f"Worst: {worst.commit} ({worst.improvement_pct:+.2f}%)")

    # Prepare clean comparisons for output
    clean = [c for c in comparisons if not c.has_caveat]
    flagged = [c for c in comparisons if c.has_caveat]

    # Save results (convert numpy types to Python types for JSON)
    output = {
        "methodology": "correlation_based_validity_plus_readme_caveats",
        "correlation_threshold": CORRELATION_THRESHOLD,
        "readme_caveats": README_CAVEATS,
        "validity_by_agent": {
            agent: {
                "n": int(r["n"]),
                "correlation": float(r["correlation"]) if r["correlation"] is not None else None,
                "is_valid": bool(r["is_valid"]),
                "reason": r["reason"],
                "agent_range": r.get("agent_range"),
                "human_range": r.get("human_range")
            }
            for agent, r in validity_results.items()
        },
        "flagged_comparisons": [
            {
                "commit": c.commit,
                "agent": c.agent,
                "improvement_pct": c.improvement_pct,
                "caveat_reasons": c.caveat_reasons
            }
            for c in flagged
        ],
        "clean_comparisons": [
            {
                "commit": c.commit,
                "agent": c.agent,
                "baseline_ttft": c.baseline_ttft,
                "human_ttft": c.human_ttft,
                "agent_ttft": c.agent_ttft,
                "improvement_pct": c.improvement_pct,
                "comparison_type": c.comparison_type
            }
            for c in clean
        ]
    }

    output_path = "scripts/benchmark_validation_results.json"
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
