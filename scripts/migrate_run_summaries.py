#!/usr/bin/env python3
"""
Migration script to generate run_summary.json for existing runs.

Usage:
    python scripts/migrate_run_summaries.py [--state-dir PATH] [--dry-run]

This script walks through all runs in state/runs/ and generates
run_summary.json files for those that don't have one yet.
"""

import argparse
import logging
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from eval.run_summary import migrate_state_runs

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(
        description="Generate run_summary.json for existing runs"
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
        help="Only count runs, don't generate summaries",
    )
    args = parser.parse_args()

    state_dir = args.state_dir.resolve()

    if not state_dir.exists():
        logger.error(f"State directory not found: {state_dir}")
        sys.exit(1)

    logger.info(f"Migrating runs in: {state_dir}")

    if args.dry_run:
        # Count runs that would be migrated
        count = 0
        skipped = 0
        for repo_dir in state_dir.iterdir():
            if not repo_dir.is_dir():
                continue
            for agent_dir in repo_dir.iterdir():
                if not agent_dir.is_dir():
                    continue
                for model_dir in agent_dir.iterdir():
                    if not model_dir.is_dir():
                        continue
                    for timestamp_dir in model_dir.iterdir():
                        if not timestamp_dir.is_dir():
                            continue
                        for item_dir in timestamp_dir.iterdir():
                            if not item_dir.is_dir():
                                continue
                            if (item_dir / "run_summary.json").exists():
                                skipped += 1
                            elif (item_dir / "journal.json").exists():
                                count += 1
        logger.info(f"Dry run: would generate {count} summaries, {skipped} already exist")
    else:
        stats = migrate_state_runs(state_dir)
        logger.info(f"Migration complete:")
        logger.info(f"  - Generated: {stats['success']}")
        logger.info(f"  - Failed: {stats['failed']}")
        logger.info(f"  - Skipped (already exist): {stats['skipped']}")


if __name__ == "__main__":
    main()
