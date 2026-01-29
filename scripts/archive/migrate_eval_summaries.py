#!/usr/bin/env python3
"""
Migration script to generate run_summary.json for existing eval_results.

Usage:
    python scripts/migrate_eval_summaries.py [--eval-dir PATH] [--state-dir PATH] [--dry-run]

This script:
1. Walks through eval_results_v2 directories
2. Reads existing test_results.json for evaluation data
3. Loads corresponding run_summary.json from state/runs (if exists)
4. Merges evaluation data into the summary
5. Saves run_summary.json to eval_results
"""

import argparse
import json
import logging
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from eval.run_summary import (
    RunSummary,
    load_summary,
    save_summary,
    add_evaluation_to_summary,
    generate_summary_from_state,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def migrate_eval_results(eval_dir: Path, state_dir: Path, dry_run: bool = False) -> dict:
    """
    Generate run_summary.json for existing eval_results.

    Returns dict with counts: {"success": N, "failed": M, "skipped": K}
    """
    stats = {"success": 0, "failed": 0, "skipped": 0}

    # Walk hierarchical structure: {repo}/{agent}/{model}/{timestamp}/{item_id}/
    for repo_dir in eval_dir.iterdir():
        if not repo_dir.is_dir() or repo_dir.name in ("README.md", "evaluation_report.json"):
            continue
        repo = repo_dir.name

        for agent_dir in repo_dir.iterdir():
            if not agent_dir.is_dir():
                continue
            agent = agent_dir.name

            for model_dir in agent_dir.iterdir():
                if not model_dir.is_dir():
                    continue
                model = model_dir.name

                for timestamp_dir in model_dir.iterdir():
                    if not timestamp_dir.is_dir():
                        continue
                    timestamp = timestamp_dir.name

                    for item_dir in timestamp_dir.iterdir():
                        if not item_dir.is_dir():
                            continue
                        item_id = item_dir.name

                        # Skip if run_summary.json already exists
                        summary_path = item_dir / "run_summary.json"
                        if summary_path.exists():
                            stats["skipped"] += 1
                            continue

                        # Read test_results.json for evaluation data
                        test_results_path = item_dir / "test_results.json"
                        if not test_results_path.exists():
                            stats["skipped"] += 1
                            continue

                        try:
                            test_results = json.loads(test_results_path.read_text())
                            eval_result = test_results.get("result", {})
                        except Exception as e:
                            logger.warning(f"Failed to read {test_results_path}: {e}")
                            stats["failed"] += 1
                            continue

                        # Try to load run_summary.json from state/runs
                        state_item_dir = state_dir / repo / agent / model / timestamp / item_id
                        summary = load_summary(state_item_dir / "run_summary.json")

                        if not summary:
                            # Generate from scratch if state doesn't have it
                            if state_item_dir.exists():
                                summary = generate_summary_from_state(
                                    item_dir=state_item_dir,
                                    repo=repo,
                                    agent=agent,
                                    model_hint=model,
                                    timestamp=timestamp,
                                )

                        if not summary:
                            # Create minimal summary from test_results metadata
                            metadata = test_results.get("metadata", {})
                            from eval.run_summary import RunSummary, RunMeta, CommitInfo, AgentInfo
                            summary = RunSummary(
                                meta=RunMeta(
                                    repo=repo,
                                    agent=agent,
                                    model=model,
                                    model_full=model,
                                    timestamp=timestamp,
                                    task_id=metadata.get("task_id", "unknown"),
                                    item_id=item_id,
                                ),
                                commits=CommitInfo(
                                    human=metadata.get("human_commit", ""),
                                    pre=metadata.get("pre_commit", ""),
                                ),
                                agent=AgentInfo(
                                    status=metadata.get("agent_status", "unknown"),
                                    patch_generated=metadata.get("patch_path") is not None,
                                ),
                            )

                        # Add evaluation results
                        add_evaluation_to_summary(
                            summary,
                            status=eval_result.get("status", "unknown"),
                            baseline_ms=eval_result.get("baseline_ms"),
                            patched_ms=eval_result.get("patched_ms"),
                            speedup=eval_result.get("speedup"),
                            improvement=eval_result.get("improvement", False),
                            error=eval_result.get("error_message"),
                        )

                        if dry_run:
                            logger.info(f"Would generate: {summary_path}")
                            stats["success"] += 1
                        else:
                            save_summary(summary, summary_path)
                            stats["success"] += 1

    return stats


def main():
    parser = argparse.ArgumentParser(
        description="Generate run_summary.json for existing eval_results"
    )
    parser.add_argument(
        "--eval-dir",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "ISO-Bench" / "eval_results_v2",
        help="Path to eval_results directory",
    )
    parser.add_argument(
        "--state-dir",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "ISO-Bench" / "state" / "runs",
        help="Path to state/runs directory",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only count, don't generate summaries",
    )
    args = parser.parse_args()

    eval_dir = args.eval_dir.resolve()
    state_dir = args.state_dir.resolve()

    if not eval_dir.exists():
        logger.error(f"Eval directory not found: {eval_dir}")
        sys.exit(1)

    logger.info(f"Migrating eval results in: {eval_dir}")
    logger.info(f"Using state dir: {state_dir}")

    stats = migrate_eval_results(eval_dir, state_dir, args.dry_run)

    logger.info(f"Migration complete:")
    logger.info(f"  - Generated: {stats['success']}")
    logger.info(f"  - Failed: {stats['failed']}")
    logger.info(f"  - Skipped: {stats['skipped']}")


if __name__ == "__main__":
    main()
