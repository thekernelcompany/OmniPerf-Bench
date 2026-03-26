#!/usr/bin/env python3
"""
Retry pushing samples that completed but failed HF upload.

Finds state dirs with artifacts (model_patch.diff, journal.json) that
weren't nuked (i.e., push failed), builds HF rows, and pushes them
with exponential backoff to avoid contention with running pass@k jobs.

Safe to run alongside pass@k — only reads existing state dirs, doesn't
run agents or touch worktrees.

Usage:
    python scripts/retry_failed_pushes.py --hf-repo Inferencebench/pass-at-k-samples --dry-run
    python scripts/retry_failed_pushes.py --hf-repo Inferencebench/pass-at-k-samples
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import shutil
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("retry_push")

ISO_BENCH_ROOT = Path(__file__).resolve().parents[1]

# Import helpers from run_pass_at_k
import sys
sys.path.insert(0, str(ISO_BENCH_ROOT))
from scripts.run_pass_at_k import (
    collect_artifacts,
    build_hf_row,
    push_single_row,
    resolve_hf_token,
    fetch_completed_samples,
)


def find_unpushed_state_dirs(state_root: Path) -> List[Tuple[Path, str, int, str]]:
    """
    Find state dirs that have artifacts but weren't nuked (push failed).

    Returns list of (item_dir, item_id, sample_idx, run_id).
    """
    results = []
    runs_dir = state_root / "runs"
    if not runs_dir.exists():
        return results

    # Walk all run dirs looking for journal.json
    for journal_path in sorted(runs_dir.rglob("journal.json")):
        item_dir = journal_path.parent
        item_id = item_dir.name

        # Extract sample index from the run_id directory name (e.g., 2026-03-26_12-30-16_s6)
        run_ts_dir = item_dir.parent
        run_ts_name = run_ts_dir.name
        match = re.search(r"_s(\d+)$", run_ts_name)
        if not match:
            continue
        sample_idx = int(match.group(1))

        # Reconstruct run_id from path: runs/<run_id>/<item_id>
        # run_id is everything between runs/ and /<item_id>
        rel = item_dir.relative_to(runs_dir)
        parts = list(rel.parts)
        if len(parts) >= 2:
            run_id = "/".join(parts[:-1])
        else:
            continue

        results.append((item_dir, item_id, sample_idx, run_id))

    return results


def main():
    parser = argparse.ArgumentParser(description="Retry failed HF pushes from local state dirs.")
    parser.add_argument("--hf-repo", required=True, help="HuggingFace repo")
    parser.add_argument("--hf-token", default=None, help="HuggingFace token")
    parser.add_argument("--state-root", default="./state", help="State root")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--delay", type=float, default=5.0,
                        help="Seconds between pushes to reduce contention (default: 5)")
    parser.add_argument("--max-retries", type=int, default=3,
                        help="Max retries per sample (default: 3)")
    args = parser.parse_args()

    state_root = Path(args.state_root).resolve()
    hf_token = resolve_hf_token(args.hf_token)
    if not hf_token:
        log.error("No HF token found")
        sys.exit(1)

    # Find unpushed state dirs
    unpushed = find_unpushed_state_dirs(state_root)
    log.info(f"Found {len(unpushed)} local state dirs with artifacts")

    if not unpushed:
        log.info("Nothing to retry")
        return

    # Check what's already on HF to skip duplicates
    log.info("Checking HF for already-pushed samples...")
    already_done = fetch_completed_samples(args.hf_repo, hf_token)
    log.info(f"Found {len(already_done)} samples already on HF")

    # Filter out already-pushed
    to_push = []
    for item_dir, item_id, sample_idx, run_id in unpushed:
        if (item_id, sample_idx) in already_done:
            log.info(f"  [{item_id}][s{sample_idx}] already on HF — will nuke local state")
            if not args.dry_run:
                # Safe to nuke — it's already on HF
                run_dir = item_dir.parent
                shutil.rmtree(run_dir, ignore_errors=True)
                # Clean empty parents
                for parent in run_dir.parents:
                    if parent == state_root / "runs":
                        break
                    try:
                        parent.rmdir()
                    except OSError:
                        break
                log.info(f"  [{item_id}][s{sample_idx}] local state nuked (was already on HF)")
        else:
            to_push.append((item_dir, item_id, sample_idx, run_id))

    log.info(f"Samples to push: {len(to_push)}")

    if args.dry_run:
        for item_dir, item_id, sample_idx, run_id in to_push:
            log.info(f"  DRY RUN: would push [{item_id}][s{sample_idx}] run_id={run_id}")
        return

    # Push with backoff
    pushed = 0
    failed = 0
    for i, (item_dir, item_id, sample_idx, run_id) in enumerate(to_push):
        tag = f"[{item_id}][s{sample_idx}]"
        log.info(f"{tag} ({i+1}/{len(to_push)}) Collecting artifacts...")

        artifacts = collect_artifacts(item_dir)
        if not artifacts:
            log.warning(f"{tag} Empty artifacts, skipping")
            continue

        row = build_hf_row(item_id, sample_idx, run_id, artifacts)

        # Push with retries and exponential backoff
        success = False
        for attempt in range(args.max_retries):
            try:
                log.info(f"{tag} Pushing (attempt {attempt+1}/{args.max_retries})...")
                push_single_row(row, args.hf_repo, hf_token)
                log.info(f"{tag} Push confirmed")
                success = True
                break
            except Exception as e:
                wait = args.delay * (2 ** attempt)
                log.warning(f"{tag} Push failed: {e} — waiting {wait:.0f}s before retry")
                time.sleep(wait)

        if success:
            # Nuke local state
            run_dir = item_dir.parent
            shutil.rmtree(run_dir, ignore_errors=True)
            for parent in run_dir.parents:
                if parent == state_root / "runs":
                    break
                try:
                    parent.rmdir()
                except OSError:
                    break
            log.info(f"{tag} Local state nuked")
            pushed += 1
        else:
            log.error(f"{tag} All retries exhausted")
            failed += 1

        # Delay between pushes to reduce contention with running jobs
        if i < len(to_push) - 1:
            log.debug(f"Waiting {args.delay}s before next push...")
            time.sleep(args.delay)

    log.info(f"Done: {pushed} pushed, {failed} failed, {len(to_push)} total")


if __name__ == "__main__":
    main()
