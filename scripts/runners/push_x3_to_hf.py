#!/usr/bin/env python3
"""Append X3 throughput re-bench artifacts (2026-05-05) to the existing
OH hard-metrics HF datasets — one upload per agent's repo, all under a
`x3_2026-05-05/` prefix so nothing existing is overwritten.

Both repos get the complete X3 picture (full results + doc + runner) so
either is self-contained for paper reproducibility.
"""
import os, sys
from pathlib import Path
from huggingface_hub import HfApi

ROOT = Path(__file__).resolve().parent.parent.parent
X3   = ROOT / "archive/results/2026-05-x3"
PREFIX = "x3_2026-05-05"

REPOS = [
    "Inferencebench/iso-bench-openhands-sonnet45-hard-metrics",
    "Inferencebench/iso-bench-openhands-gpt5-hard-metrics",
]

# Token from default HF cache (shikhar007). The gated token at
# ~/.config/omniperf/hf_token_gated is for downloading gated models, NOT for
# uploading; uploading uses shikhar007's standard write-scoped token.
TOKEN = os.environ.get("HF_TOKEN")
if not TOKEN:
    p = Path.home() / ".cache/huggingface/token"
    if p.exists():
        TOKEN = p.read_text().strip()
if not TOKEN:
    sys.exit("ERROR: set HF_TOKEN env var or `hf auth login` first")

api = HfApi(token=TOKEN)

# The whole X3 dir is small (~316 KB). Upload it identically to both repos.
# .gitignore in the repo root excludes *.log so we don't have logs locally;
# nothing to ignore except junk.
for repo in REPOS:
    print(f"\n=== Uploading X3 to {repo}/{PREFIX}/ ===")
    api.upload_folder(
        folder_path=str(X3),
        repo_id=repo,
        repo_type="dataset",
        path_in_repo=PREFIX,
        token=TOKEN,
        ignore_patterns=["*.tmp", "*.lock", "__pycache__/*"],
        commit_message="X3 throughput re-bench 2026-05-05 — append (do not overwrite May 5 fanout)",
    )
    # Also stash the runner + mapping + driver alongside so reproducibility is in-repo
    extras = [
        ROOT / "scripts/runners/run_vllm_native.py",
        ROOT / "scripts/runners/run_x3_batch.sh",
        ROOT / "scripts/runners/x3_summarize.py",
        ROOT / "scripts/runners/x3_extract_4c822298.py",
        ROOT / "data/mappings/vllm_oh_mapping_x3.json",
    ]
    for src in extras:
        if not src.exists():
            print(f"  skip missing: {src}")
            continue
        api.upload_file(
            path_or_fileobj=str(src),
            repo_id=repo,
            repo_type="dataset",
            path_in_repo=f"{PREFIX}/_repro/{src.name}",
            token=TOKEN,
        )
        print(f"  + _repro/{src.name}")
    print(f"  -> https://huggingface.co/datasets/{repo}/tree/main/{PREFIX}")
