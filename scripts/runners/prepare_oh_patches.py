#!/usr/bin/env python3
"""Flatten the HF OH-Sonnet-4.5 dataset into the layout run_3way_benchmarks.py expects.

Source layout (from HF dataset Inferencebench/iso-bench-openhands-sonnet45-rebuttal):
    runs/{repo}/{timestamp}/{task_id}/{run_summary.json, journal.json, model_patch.diff, ...}

Destination layout (one of AGENT_CONFIGS values in run_3way_benchmarks.py):
    {dst}/{task_id}/{run_summary.json, journal.json, model_patch.diff}

Picks latest-timestamp non-empty model_patch.diff per task; falls back to latest empty
if all timestamps for a task are empty (so the runner sees the task and reports 0-improvement).
"""

import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
SRC = ROOT / "ISO-Bench/state/runs/oh54_hf/runs"
DST_VLLM = ROOT / "ISO-Bench/state/runs/vllm/openhands_sonnet45/flat"
DST_SGLANG = ROOT / "ISO-Bench/state/runs/sglang/openhands_sonnet45/flat"


def pick_latest_per_task(repo: str) -> dict:
    """Strict latest-timestamp wins, even if the latest patch is empty.

    Rationale: each timestamp is a retry. An agent whose *final* retry produced
    an empty diff has functionally given up on that task; counting an earlier
    successful retry would overstate the agent's hard-metric score.
    """
    by_task = {}
    for patch in (SRC / repo).rglob("model_patch.diff"):
        ts, task = patch.parts[-3], patch.parts[-2]
        size = patch.stat().st_size
        prev = by_task.get(task)
        if prev is None or ts > prev[0]:
            by_task[task] = (ts, size, patch)
    return by_task


def main():
    for repo, dst in [("vllm", DST_VLLM), ("sglang", DST_SGLANG)]:
        chosen = pick_latest_per_task(repo)
        dst.mkdir(parents=True, exist_ok=True)
        nonempty = 0
        empty = []
        for task, (ts, size, patch) in chosen.items():
            td = dst / task
            td.mkdir(exist_ok=True)
            shutil.copy(patch, td / "model_patch.diff")
            for sibling in ("run_summary.json", "journal.json"):
                src = patch.parent / sibling
                if src.exists():
                    shutil.copy(src, td / sibling)
            if size > 0:
                nonempty += 1
            else:
                empty.append(task)
        print(f"{repo}: wrote {len(chosen)} task dirs to {dst}  (nonempty={nonempty} empty={empty})")


if __name__ == "__main__":
    main()
