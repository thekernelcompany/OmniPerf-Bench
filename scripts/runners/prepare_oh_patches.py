#!/usr/bin/env python3
"""Flatten an HF OpenHands rebuttal dataset into the layout the hard-metrics
runners expect.

Source layout (HF dataset Inferencebench/iso-bench-openhands-{sonnet45,gpt5}-rebuttal):
    runs/{repo}/{timestamp}/{task_id}/{run_summary.json, journal.json, model_patch.diff, ...}

Destination layout (one of AGENT_CONFIGS values in run_3way_benchmarks.py;
also what run_vllm_native.py / run_sglang_*.py read from):
    {dst}/{task_id}/{run_summary.json, journal.json, model_patch.diff}

Picks strict latest-timestamp per task. If the latest run produced an empty
patch, that empty patch wins — an agent whose final retry gave up shouldn't be
credited with an earlier retry's diff.

Usage:
    # sonnet45 (default, preserves prior behavior)
    python prepare_oh_patches.py

    # gpt5
    python prepare_oh_patches.py --agent openhands_gpt5 \
        --src ISO-Bench/state/runs/oh54_gpt5_hf/runs

    # explicit override
    python prepare_oh_patches.py \
        --src <path/to/runs> \
        --dst-vllm <path> --dst-sglang <path>

Filtering: any task dir whose name matches --skip-suffix (default `-smoke`)
is dropped — keeps the smoke-run residual out of the fanout.
"""

import argparse
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent

DEFAULTS = {
    "openhands_sonnet45": {
        "src": ROOT / "ISO-Bench/state/runs/oh54_hf/runs",
        "dst_vllm": ROOT / "ISO-Bench/state/runs/vllm/openhands_sonnet45/flat",
        "dst_sglang": ROOT / "ISO-Bench/state/runs/sglang/openhands_sonnet45/flat",
    },
    "openhands_gpt5": {
        "src": ROOT / "ISO-Bench/state/runs/oh54_gpt5_hf/runs",
        "dst_vllm": ROOT / "ISO-Bench/state/runs/vllm/openhands_gpt5/flat",
        "dst_sglang": ROOT / "ISO-Bench/state/runs/sglang/openhands_gpt5/flat",
    },
}


def pick_latest_per_task(src_repo: Path, skip_suffix: str) -> dict:
    """Strict latest-timestamp wins, even if the latest patch is empty.

    Rationale: each timestamp is a retry. An agent whose *final* retry produced
    an empty diff has functionally given up on that task; counting an earlier
    successful retry would overstate the agent's hard-metric score.
    """
    by_task = {}
    for patch in src_repo.rglob("model_patch.diff"):
        ts, task = patch.parts[-3], patch.parts[-2]
        if skip_suffix and task.endswith(skip_suffix):
            continue
        size = patch.stat().st_size
        prev = by_task.get(task)
        if prev is None or ts > prev[0]:
            by_task[task] = (ts, size, patch)
    return by_task


def flatten(src: Path, dst_vllm: Path, dst_sglang: Path, skip_suffix: str) -> None:
    for repo, dst in [("vllm", dst_vllm), ("sglang", dst_sglang)]:
        repo_src = src / repo
        if not repo_src.exists():
            print(f"{repo}: SKIP — {repo_src} does not exist")
            continue
        chosen = pick_latest_per_task(repo_src, skip_suffix)
        dst.mkdir(parents=True, exist_ok=True)
        nonempty = 0
        empty = []
        for task, (ts, size, patch) in chosen.items():
            td = dst / task
            td.mkdir(exist_ok=True)
            shutil.copy(patch, td / "model_patch.diff")
            for sibling in ("run_summary.json", "journal.json"):
                ssrc = patch.parent / sibling
                if ssrc.exists():
                    shutil.copy(ssrc, td / sibling)
            if size > 0:
                nonempty += 1
            else:
                empty.append(task)
        print(f"{repo}: wrote {len(chosen)} task dirs to {dst}  "
              f"(nonempty={nonempty} empty={empty})")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--agent", default="openhands_sonnet45",
                   choices=sorted(DEFAULTS.keys()),
                   help="Agent profile (selects default src/dst paths).")
    p.add_argument("--src", type=Path, default=None,
                   help="Override source `runs/` dir (HF dataset layout).")
    p.add_argument("--dst-vllm", type=Path, default=None,
                   help="Override vLLM destination flat dir.")
    p.add_argument("--dst-sglang", type=Path, default=None,
                   help="Override SGLang destination flat dir.")
    p.add_argument("--skip-suffix", default="-smoke",
                   help="Drop task IDs ending with this suffix (default: -smoke).")
    args = p.parse_args()

    d = DEFAULTS[args.agent]
    src = args.src or d["src"]
    dst_vllm = args.dst_vllm or d["dst_vllm"]
    dst_sglang = args.dst_sglang or d["dst_sglang"]

    print(f"agent       : {args.agent}")
    print(f"src         : {src}")
    print(f"dst_vllm    : {dst_vllm}")
    print(f"dst_sglang  : {dst_sglang}")
    print(f"skip_suffix : {args.skip_suffix!r}")
    print()

    flatten(src, dst_vllm, dst_sglang, args.skip_suffix)


if __name__ == "__main__":
    main()
