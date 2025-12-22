"""
Aggregate and report evaluation results.

This module collects test results from the output directory and
generates summary statistics and reports.
"""

from __future__ import annotations

import json
import logging
from collections import defaultdict
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class AgentSummary:
    """Summary statistics for an agent's runs."""
    agent_name: str
    run_id: str
    total_commits: int = 0
    tests_available: int = 0
    tests_run: int = 0
    tests_passed: int = 0
    tests_failed: int = 0
    no_test_available: int = 0
    no_patch: int = 0
    patch_failed: int = 0
    timeouts: int = 0
    errors: int = 0
    # Performance metrics
    avg_speedup: Optional[float] = None
    median_speedup: Optional[float] = None
    commits_with_improvement: int = 0
    commits_with_regression: int = 0
    commits_neutral: int = 0
    # Detailed speedups
    speedups: List[float] = None

    def __post_init__(self):
        if self.speedups is None:
            self.speedups = []


def aggregate_results(output_dir: Path) -> Dict[str, AgentSummary]:
    """
    Aggregate results from all runs in the output directory.

    Supports both flat structure (run_id/item/) and hierarchical structure
    (repo/agent/model/timestamp/item/).

    Args:
        output_dir: Directory containing evaluation results

    Returns:
        Dict mapping run_id -> AgentSummary
    """
    output_dir = Path(output_dir)
    if not output_dir.exists():
        logger.warning(f"Output directory not found: {output_dir}")
        return {}

    summaries: Dict[str, AgentSummary] = {}

    # Find all run_summary.json files (works for hierarchical structure)
    run_summaries = list(output_dir.rglob("run_summary.json"))

    if run_summaries:
        # Hierarchical structure - group by agent/model
        return _aggregate_from_run_summaries(run_summaries)

    # Fall back to legacy flat structure
    return _aggregate_from_flat_structure(output_dir)


def _aggregate_from_run_summaries(run_summary_paths: List[Path]) -> Dict[str, AgentSummary]:
    """Aggregate from run_summary.json files in hierarchical structure."""
    # Group by agent/model
    grouped: Dict[str, List[Dict]] = defaultdict(list)

    for path in run_summary_paths:
        try:
            data = json.loads(path.read_text())
            meta = data.get("meta", {})
            agent = meta.get("agent", "unknown")
            model = meta.get("model", "unknown")
            key = f"{agent}/{model}"
            grouped[key].append(data)
        except Exception as e:
            logger.error(f"Error reading {path}: {e}")
            continue

    summaries = {}
    for key, items in grouped.items():
        agent_name = key.split("/")[0]
        summary = AgentSummary(agent_name=agent_name, run_id=key)
        speedups = []

        for data in items:
            summary.total_commits += 1
            ev = data.get("evaluation", {})
            agent_info = data.get("agent", {})
            status = ev.get("status", "unknown")
            patch_generated = agent_info.get("patch_generated", False)

            # Check for test availability based on error message
            error_msg = ev.get("error", "") or ""
            no_test = "No test script found" in error_msg

            if no_test:
                summary.no_test_available += 1
            elif not patch_generated:
                summary.no_patch += 1
                summary.tests_available += 1
            elif status == "success":
                summary.tests_available += 1
                summary.tests_run += 1

                speedup = ev.get("speedup")
                baseline_ms = ev.get("baseline_ms")
                patched_ms = ev.get("patched_ms")

                if speedup is not None:
                    summary.tests_passed += 1
                    speedups.append(speedup)
                    if speedup > 1.05:  # >5% improvement
                        summary.commits_with_improvement += 1
                    elif speedup < 0.95:  # >5% regression
                        summary.commits_with_regression += 1
                    else:
                        summary.commits_neutral += 1
                elif baseline_ms is None:
                    # Status "success" but no baseline timing - test didn't run properly
                    # This happens with import errors, opt path not triggered, etc.
                    summary.tests_failed += 1
                elif patched_ms is None:
                    # Baseline ran but patched didn't produce timing
                    summary.tests_failed += 1
                else:
                    # Both timings exist but speedup is None (shouldn't happen)
                    summary.tests_passed += 1

            elif status == "patch_failed":
                summary.patch_failed += 1
                summary.tests_available += 1
                summary.tests_run += 1
                summary.tests_failed += 1

            elif status == "timeout":
                summary.timeouts += 1
                summary.tests_available += 1

            elif status == "error" or status == "baseline_failed":
                summary.errors += 1
                summary.tests_available += 1

            elif status == "no_patch":
                summary.no_patch += 1
                summary.tests_available += 1

        # Compute aggregate metrics
        if speedups:
            summary.speedups = sorted(speedups)
            summary.avg_speedup = sum(speedups) / len(speedups)
            summary.median_speedup = speedups[len(speedups) // 2]

        summaries[key] = summary

    return summaries


def _aggregate_from_flat_structure(output_dir: Path) -> Dict[str, AgentSummary]:
    """Aggregate from legacy flat directory structure."""
    summaries: Dict[str, AgentSummary] = {}

    for run_dir in output_dir.iterdir():
        if not run_dir.is_dir():
            continue

        run_id = run_dir.name

        # Extract agent name from run_id (e.g., "vllm_claude_sonnet45-0a51aaa8" -> "claude_sonnet45")
        agent_name = _extract_agent_name(run_id)

        summary = AgentSummary(
            agent_name=agent_name,
            run_id=run_id,
        )

        speedups = []

        for item_dir in run_dir.iterdir():
            if not item_dir.is_dir():
                continue

            result_path = item_dir / "test_results.json"
            if not result_path.exists():
                continue

            try:
                data = json.loads(result_path.read_text())
                result = data.get("result", {})
            except Exception as e:
                logger.error(f"Error reading {result_path}: {e}")
                continue

            summary.total_commits += 1
            status = result.get("status", "unknown")

            if status == "success":
                summary.tests_passed += 1
                summary.tests_run += 1
                summary.tests_available += 1

                speedup = result.get("speedup")
                if speedup is not None:
                    speedups.append(speedup)
                    if speedup > 1.0:
                        summary.commits_with_improvement += 1
                    elif speedup < 1.0:
                        summary.commits_with_regression += 1
                    else:
                        summary.commits_neutral += 1

            elif status == "no_test":
                summary.no_test_available += 1

            elif status == "no_patch":
                summary.no_patch += 1
                summary.tests_available += 1
                summary.tests_run += 1

            elif status == "patch_failed":
                summary.patch_failed += 1
                summary.tests_available += 1
                summary.tests_run += 1
                summary.tests_failed += 1

            elif status == "timeout":
                summary.timeouts += 1
                summary.tests_available += 1

            elif status == "error":
                summary.errors += 1
                summary.tests_available += 1

        # Compute aggregate metrics
        if speedups:
            summary.speedups = sorted(speedups)
            summary.avg_speedup = sum(speedups) / len(speedups)
            summary.median_speedup = speedups[len(speedups) // 2]

        summaries[run_id] = summary

    return summaries


def _extract_agent_name(run_id: str) -> str:
    """Extract agent name from run_id."""
    # Format: "vllm_claude_sonnet45-0a51aaa8" or "sglang_core-389be848"
    parts = run_id.rsplit("-", 1)
    if len(parts) > 1:
        name_part = parts[0]
        # Remove repo prefix
        if name_part.startswith("vllm_"):
            return name_part[5:]
        if name_part.startswith("sglang_"):
            return name_part[7:]
        return name_part
    return run_id


def generate_report(
    summaries: Dict[str, AgentSummary],
    output_path: Optional[Path] = None,
) -> Dict[str, Any]:
    """
    Generate a comprehensive evaluation report.

    Args:
        summaries: Dict of run_id -> AgentSummary
        output_path: Optional path to save report JSON

    Returns:
        Report as dict
    """
    report = {
        "generated_at": datetime.now().isoformat(),
        "total_runs": len(summaries),
        "runs": {},
        "aggregate": {
            "total_commits": 0,
            "total_tests_available": 0,
            "total_tests_run": 0,
            "total_tests_passed": 0,
            "total_improvements": 0,
            "total_regressions": 0,
            "all_speedups": [],
        },
    }

    for run_id, summary in summaries.items():
        run_report = asdict(summary)
        # Remove the full speedups list from individual reports (too verbose)
        run_report["speedup_count"] = len(summary.speedups) if summary.speedups else 0
        del run_report["speedups"]

        report["runs"][run_id] = run_report

        # Aggregate
        report["aggregate"]["total_commits"] += summary.total_commits
        report["aggregate"]["total_tests_available"] += summary.tests_available
        report["aggregate"]["total_tests_run"] += summary.tests_run
        report["aggregate"]["total_tests_passed"] += summary.tests_passed
        report["aggregate"]["total_improvements"] += summary.commits_with_improvement
        report["aggregate"]["total_regressions"] += summary.commits_with_regression
        if summary.speedups:
            report["aggregate"]["all_speedups"].extend(summary.speedups)

    # Compute overall metrics
    all_speedups = report["aggregate"]["all_speedups"]
    if all_speedups:
        report["aggregate"]["overall_avg_speedup"] = sum(all_speedups) / len(all_speedups)
        sorted_speedups = sorted(all_speedups)
        report["aggregate"]["overall_median_speedup"] = sorted_speedups[len(sorted_speedups) // 2]
        report["aggregate"]["speedup_p95"] = sorted_speedups[int(len(sorted_speedups) * 0.95)]
        report["aggregate"]["speedup_p5"] = sorted_speedups[int(len(sorted_speedups) * 0.05)]
    else:
        report["aggregate"]["overall_avg_speedup"] = None
        report["aggregate"]["overall_median_speedup"] = None

    # Remove verbose speedups list from final report
    del report["aggregate"]["all_speedups"]

    # Save if path provided
    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(report, f, indent=2)
        logger.info(f"Report saved to {output_path}")

    return report


def print_summary(summaries: Dict[str, AgentSummary]) -> None:
    """Print a human-readable summary to stdout."""
    print("\n" + "=" * 70)
    print("EVALUATION SUMMARY")
    print("=" * 70)

    for run_id, summary in sorted(summaries.items()):
        print(f"\n{run_id}")
        print("-" * 50)
        print(f"  Total commits:      {summary.total_commits}")
        print(f"  Tests available:    {summary.tests_available}")
        print(f"  Tests passed:       {summary.tests_passed}")
        print(f"  No test available:  {summary.no_test_available}")
        print(f"  No patch:           {summary.no_patch}")
        print(f"  Patch failed:       {summary.patch_failed}")
        print(f"  Timeouts:           {summary.timeouts}")
        print(f"  Errors:             {summary.errors}")
        print()
        print(f"  Improvements:       {summary.commits_with_improvement}")
        print(f"  Regressions:        {summary.commits_with_regression}")
        print(f"  Neutral:            {summary.commits_neutral}")
        if summary.avg_speedup:
            print(f"  Avg speedup:        {summary.avg_speedup:.3f}x")
            print(f"  Median speedup:     {summary.median_speedup:.3f}x")

    # Overall summary
    total = sum(s.total_commits for s in summaries.values())
    total_improvements = sum(s.commits_with_improvement for s in summaries.values())
    total_regressions = sum(s.commits_with_regression for s in summaries.values())

    print("\n" + "=" * 70)
    print("OVERALL")
    print("=" * 70)
    print(f"  Total commits evaluated:  {total}")
    print(f"  Total improvements:       {total_improvements}")
    print(f"  Total regressions:        {total_regressions}")

    all_speedups = []
    for s in summaries.values():
        if s.speedups:
            all_speedups.extend(s.speedups)

    if all_speedups:
        avg = sum(all_speedups) / len(all_speedups)
        print(f"  Overall avg speedup:      {avg:.3f}x")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    # Example usage
    output_dir = Path("/root/OmniPerf-Bench/eval_results")

    if output_dir.exists():
        summaries = aggregate_results(output_dir)
        print_summary(summaries)

        report = generate_report(
            summaries,
            output_path=output_dir / "evaluation_report.json",
        )
        print(f"\nReport saved to {output_dir / 'evaluation_report.json'}")
    else:
        print(f"No results found in {output_dir}")
