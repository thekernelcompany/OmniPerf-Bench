#!/usr/bin/env python3
"""Push the OH GPT-5 hard-metrics result JSONs + summary + worker logs to HF.

Creates dataset repo Inferencebench/iso-bench-openhands-gpt5-hard-metrics
with everything needed to reproduce or analyze the run.
"""
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
RESULTS_DIR = ROOT / "archive/results/2026-05/iso_bench_results_3way_openhands_gpt5/results"
SGLANG_RESULTS_DIR = ROOT / "archive/results/2026-05/iso_bench_results_3way_openhands_gpt5_sglang/results"
SUMMARY_PATH = ROOT / "archive/results/2026-05/iso_bench_results_3way_openhands_gpt5/SUMMARY.json"
LOGS_DIRS = [
    ROOT / "logs/oh_gpt5_smoke",
    ROOT / "logs/oh_gpt5_fanout",
    ROOT / "logs/oh_gpt5_sglang_smoke",
    ROOT / "logs/oh_gpt5_sglang_fanout",
]

REPO = "Inferencebench/iso-bench-openhands-gpt5-hard-metrics"

# Token must come from env (HF_TOKEN) or ~/.cache/huggingface/token; never hardcoded.
TOKEN = os.environ.get("HF_TOKEN")
if not TOKEN:
    p = Path.home() / ".cache/huggingface/token"
    TOKEN = p.read_text().strip() if p.exists() else ""
if not TOKEN:
    print("ERROR: set HF_TOKEN env var or `huggingface-cli login` first")
    sys.exit(1)

# 1. Ensure SUMMARY.json is fresh
print("Building SUMMARY.json...")
import subprocess
subprocess.run([sys.executable, str(ROOT / "scripts/runners/build_summary.py"),
                "--agent", "openhands_gpt5"], check=True)

# 2. Push to HF
from huggingface_hub import HfApi, create_repo

api = HfApi(token=TOKEN)
try:
    create_repo(repo_id=REPO, repo_type="dataset", exist_ok=True, token=TOKEN, private=True)
    print(f"Repo ready: {REPO}")
except Exception as e:
    print(f"create_repo issue: {e}")

# vLLM result JSONs
print("Uploading vLLM results...")
api.upload_folder(
    folder_path=str(RESULTS_DIR.parent),
    repo_id=REPO, repo_type="dataset", path_in_repo="results_vllm",
    token=TOKEN, ignore_patterns=["*.tmp", "*.lock"],
)

# SGLang result JSONs
if SGLANG_RESULTS_DIR.exists():
    print("Uploading SGLang results...")
    api.upload_folder(
        folder_path=str(SGLANG_RESULTS_DIR.parent),
        repo_id=REPO, repo_type="dataset", path_in_repo="results_sglang",
        token=TOKEN, ignore_patterns=["*.tmp", "*.lock"],
    )

# Worker logs (small, useful for postmortem)
for log_dir in LOGS_DIRS:
    if log_dir.exists():
        print(f"Uploading {log_dir.name}...")
        api.upload_folder(
            folder_path=str(log_dir),
            repo_id=REPO, repo_type="dataset", path_in_repo=f"logs/{log_dir.name}",
            token=TOKEN,
        )

# Key scripts + mappings + ISO-Bench parquet + doc
for src in [
    ROOT / "scripts/runners/run_vllm_native.py",
    ROOT / "scripts/runners/run_sglang_native.py",
    ROOT / "scripts/runners/prepare_oh_patches.py",
    ROOT / "scripts/runners/build_summary.py",
    ROOT / "data/mappings/vllm_oh_mapping.json",
    ROOT / "data/mappings/sglang_oh_mapping.json",
    ROOT / "data/iso_bench/vllm/train.parquet",
    ROOT / "data/iso_bench/sglang/train.parquet",
    ROOT / "docs/HARD_METRICS_OH_GPT5_FINAL.md",
    ROOT / "docs/HARD_METRICS_OH_SONNET45_RUNBOOK.md",
]:
    if src.exists():
        rel = src.relative_to(ROOT)
        api.upload_file(path_or_fileobj=str(src), repo_id=REPO, repo_type="dataset",
                        path_in_repo=str(rel), token=TOKEN)
        print(f"  uploaded {rel}")

# Agent patches (input data) — flat dir layouts
for repo_label, flat in [("vllm", ROOT / "ISO-Bench/state/runs/vllm/openhands_gpt5/flat"),
                          ("sglang", ROOT / "ISO-Bench/state/runs/sglang/openhands_gpt5/flat")]:
    if flat.exists():
        print(f"Uploading {repo_label} agent patches...")
        api.upload_folder(
            folder_path=str(flat),
            repo_id=REPO, repo_type="dataset",
            path_in_repo=f"agent_patches_{repo_label}",
            token=TOKEN,
        )

print(f"\nDone. https://huggingface.co/datasets/{REPO}")
