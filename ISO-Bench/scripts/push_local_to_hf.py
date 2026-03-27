#!/usr/bin/env python3
"""
Push deduplicated local pass@k samples to HuggingFace.

Walks state/runs/{repo}/claude_code/sonnet/, deduplicates by (task_id, sample_idx)
keeping the earliest run, builds HF rows, and uploads as consolidated parquet
files per repo (vllm/sglang).

Usage:
    # Dry run — show what would be pushed
    .venv/bin/python scripts/push_local_to_hf.py --dry-run

    # Push to HF
    .venv/bin/python scripts/push_local_to_hf.py --hf-repo ISO-Bench/pass-at-k-samples
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import sys
import tempfile
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("push_to_hf")

ISO_BENCH_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ISO_BENCH_ROOT / "scripts"))

# Re-use artifact collection and row building from run_pass_at_k
from run_pass_at_k import ARTIFACT_FILES, collect_artifacts, build_hf_row, resolve_hf_token


def scan_local_runs(
    state_root: Path,
    repo_dir: str,
    agent: str = "claude_code",
    model: str = "sonnet",
) -> Dict[Tuple[str, int], List[Tuple[str, Path]]]:
    """
    Scan local run directories and return mapping of
    (task_id, sample_idx) -> [(run_dir_name, item_dir_path), ...]
    sorted by timestamp (earliest first).
    """
    base = state_root / "runs" / repo_dir / agent / model
    if not base.exists():
        return {}

    samples: Dict[Tuple[str, int], List[Tuple[str, Path]]] = defaultdict(list)

    for run_dir in sorted(os.listdir(base)):
        full = base / run_dir
        if not full.is_dir():
            continue
        m = re.search(r"_s(\d+)$", run_dir)
        if not m:
            continue
        sample_idx = int(m.group(1))

        for task_dir in os.listdir(full):
            task_path = full / task_dir
            summary = task_path / "run_summary.json"
            if task_path.is_dir() and summary.exists():
                samples[(task_dir, sample_idx)].append((run_dir, task_path))

    return samples


def deduplicate(
    samples: Dict[Tuple[str, int], List[Tuple[str, Path]]],
) -> Dict[Tuple[str, int], Path]:
    """
    For each (task_id, sample_idx), keep the EARLIEST run (first timestamp).
    Returns mapping of (task_id, sample_idx) -> item_dir_path.
    """
    deduped = {}
    total_dupes = 0
    for key, runs in sorted(samples.items()):
        # Already sorted by timestamp (run_dir names are timestamps)
        deduped[key] = runs[0][1]
        if len(runs) > 1:
            total_dupes += len(runs) - 1
    log.info(f"Deduplicated: {len(deduped)} unique samples, dropped {total_dupes} duplicates")
    return deduped


def build_rows(
    deduped: Dict[Tuple[str, int], Path],
    repo_dir: str,
) -> List[Dict[str, Any]]:
    """Build HF dataset rows from deduplicated samples."""
    rows = []
    for (task_id, sample_idx), item_dir in sorted(deduped.items()):
        run_id = f"{repo_dir}/claude_code/sonnet/{item_dir.parent.name}"
        artifacts = collect_artifacts(item_dir)
        row = build_hf_row(task_id, sample_idx, run_id, artifacts)
        rows.append(row)
    return rows


def check_hf_existing(
    repo_id: str,
    token: str,
    repo_subdir: str,
) -> Set[Tuple[str, int]]:
    """Check what's already on HF to avoid re-uploading."""
    from huggingface_hub import HfApi

    api = HfApi(token=token)
    existing: Set[Tuple[str, int]] = set()

    try:
        files = api.list_repo_files(repo_id, repo_type="dataset")
    except Exception:
        # Repo doesn't exist yet
        return existing

    # Check for per-sample shards in data/
    for f in files:
        basename = f.split("/")[-1]
        m = re.match(r"^(.+)_s(\d+)_\d+\.parquet$", basename)
        if m:
            existing.add((m.group(1), int(m.group(2))))

    # Check consolidated parquet files
    if any(f.endswith(".parquet") and "train" in f for f in files):
        try:
            import pandas as pd
            from huggingface_hub import hf_hub_download

            for f in files:
                if f.endswith(".parquet") and repo_subdir in f:
                    try:
                        local = hf_hub_download(repo_id, f, repo_type="dataset", token=token)
                        df = pd.read_parquet(local, columns=["item_id", "sample_index"])
                        for _, row in df.iterrows():
                            iid = row.get("item_id", "")
                            sidx = row.get("sample_index")
                            if iid and sidx is not None:
                                existing.add((str(iid), int(sidx)))
                    except Exception:
                        continue
        except ImportError:
            pass

    return existing


def push_consolidated(
    rows: List[Dict[str, Any]],
    repo_id: str,
    token: str,
    repo_subdir: str,
) -> bool:
    """Push rows as a single consolidated parquet file."""
    import pandas as pd
    from huggingface_hub import HfApi

    api = HfApi(token=token)

    # Create repo if needed
    api.create_repo(repo_id, repo_type="dataset", exist_ok=True)

    df = pd.DataFrame(rows)
    log.info(f"Built DataFrame: {len(df)} rows, {len(df.columns)} columns")

    with tempfile.TemporaryDirectory() as tmpdir:
        parquet_path = Path(tmpdir) / "train.parquet"
        df.to_parquet(parquet_path, index=False)
        file_size_mb = parquet_path.stat().st_size / (1024 * 1024)
        log.info(f"Parquet file: {file_size_mb:.1f} MB")

        api.upload_file(
            path_or_fileobj=str(parquet_path),
            path_in_repo=f"data/{repo_subdir}/train.parquet",
            repo_id=repo_id,
            repo_type="dataset",
            commit_message=f"Upload {repo_subdir} pass@k samples ({len(rows)} rows, claude_code/sonnet)",
        )

    return True


def verify_push(
    repo_id: str,
    token: str,
    repo_subdir: str,
    expected_count: int,
) -> bool:
    """Verify the push by reading back and checking row count."""
    import pandas as pd
    from huggingface_hub import hf_hub_download

    try:
        local = hf_hub_download(
            repo_id,
            f"data/{repo_subdir}/train.parquet",
            repo_type="dataset",
            token=token,
        )
        df = pd.read_parquet(local)
        actual = len(df)
        unique_pairs = df.groupby(["item_id", "sample_index"]).ngroups

        if actual != expected_count:
            log.error(f"VERIFICATION FAILED: expected {expected_count} rows, got {actual}")
            return False
        if unique_pairs != actual:
            log.error(f"VERIFICATION FAILED: {actual} rows but only {unique_pairs} unique (item_id, sample_index) pairs — DUPLICATES!")
            return False

        log.info(f"VERIFIED: {actual} rows, {unique_pairs} unique pairs, no duplicates")
        return True
    except Exception as e:
        log.error(f"Verification failed: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description="Push deduplicated local samples to HF")
    parser.add_argument("--hf-repo", default=None, help="HuggingFace repo ID")
    parser.add_argument("--hf-token", default=None, help="HuggingFace token")
    parser.add_argument("--dry-run", action="store_true", help="Show stats, don't push")
    parser.add_argument("--state-root", default="./state", help="State root directory")
    parser.add_argument("--verify", action="store_true", default=True, help="Verify after push")
    args = parser.parse_args()

    state_root = Path(args.state_root).resolve()

    repo_configs = [
        ("vllm", "vllm"),
        ("sglan", "sglang"),  # local dir is "sglan", HF subdir is "sglang"
    ]

    for repo_dir, hf_subdir in repo_configs:
        log.info(f"\n{'='*60}")
        log.info(f"Processing {hf_subdir}")
        log.info(f"{'='*60}")

        # Scan
        samples = scan_local_runs(state_root, repo_dir)
        if not samples:
            log.warning(f"No local runs found for {repo_dir}")
            continue

        total_runs = sum(len(v) for v in samples.values())
        log.info(f"Found {len(samples)} unique (task, sample) pairs from {total_runs} total runs")

        # Deduplicate
        deduped = deduplicate(samples)

        # Stats
        tasks = defaultdict(set)
        for (task_id, sidx) in deduped:
            tasks[task_id].add(sidx)

        log.info(f"Tasks: {len(tasks)}")
        for task_id in sorted(tasks):
            indices = sorted(tasks[task_id])
            status = "OK" if len(indices) == 8 else f"INCOMPLETE ({len(indices)}/8)"
            log.info(f"  {task_id}: {len(indices)} samples {status}")

        if args.dry_run:
            log.info(f"DRY RUN — would push {len(deduped)} rows to {args.hf_repo}/data/{hf_subdir}/")
            continue

        if not args.hf_repo:
            log.error("--hf-repo required for push")
            sys.exit(1)

        token = resolve_hf_token(args.hf_token)
        if not token:
            log.error("No HF token found")
            sys.exit(1)

        # Check what's already on HF
        log.info("Checking HF for existing data...")
        existing = check_hf_existing(args.hf_repo, token, hf_subdir)
        if existing:
            log.info(f"Found {len(existing)} samples already on HF")
            overlap = set(deduped.keys()) & existing
            if overlap:
                log.warning(f"{len(overlap)} samples overlap with HF — these will be REPLACED (full file upload)")

        # Build rows
        log.info("Building HF rows...")
        rows = build_rows(deduped, repo_dir)
        log.info(f"Built {len(rows)} rows")

        # Sanity check: no duplicate (item_id, sample_index) in rows
        seen = set()
        for row in rows:
            key = (row["item_id"], row["sample_index"])
            if key in seen:
                log.error(f"DUPLICATE in rows: {key} — aborting!")
                sys.exit(1)
            seen.add(key)
        log.info(f"Dedup sanity check passed: {len(seen)} unique rows")

        # Push
        log.info(f"Pushing {len(rows)} rows to {args.hf_repo}/data/{hf_subdir}/...")
        ok = push_consolidated(rows, args.hf_repo, token, hf_subdir)
        if not ok:
            log.error("Push failed!")
            sys.exit(1)
        log.info("Push succeeded")

        # Verify
        if args.verify:
            log.info("Verifying push...")
            verified = verify_push(args.hf_repo, token, hf_subdir, len(rows))
            if not verified:
                log.error("VERIFICATION FAILED — data on HF may be corrupt!")
                sys.exit(1)

    log.info("\nAll done!")


if __name__ == "__main__":
    main()
