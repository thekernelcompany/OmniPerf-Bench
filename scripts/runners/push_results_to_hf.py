#!/usr/bin/env python3
"""Push the OH Sonnet-4.5 hard-metrics result JSONs + summary + worker logs to HF.

Creates dataset repo Inferencebench/iso-bench-openhands-sonnet45-hard-metrics
with everything needed to reproduce or analyze the run.
"""
import json
import os
import sys
from pathlib import Path

ROOT = Path("/root/OmniPerf-Bench")
RESULTS_DIR = ROOT / "archive/results/2026-05/iso_bench_results_3way_openhands_sonnet45/results"
SGLANG_RESULTS_DIR = ROOT / "archive/results/2026-05/iso_bench_results_3way_openhands_sonnet45_sglang/results"
LOGS_DIRS = [
    ROOT / "logs/oh_vllm_native_v3",
    ROOT / "logs/oh_vllm_overlay",
    ROOT / "logs/oh_sglang_v10",
]

REPO = "Inferencebench/iso-bench-openhands-sonnet45-hard-metrics"
TOKEN = "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"  # ssh-rebuttals — has write access to Inferencebench

# 1. Build summary
summary = {
    "schema_version": "1.0",
    "agent": "openhands",
    "model": "anthropic-claude-sonnet-4-5-20250929-v1-0",
    "session_started_utc": "2026-05-02T23:35Z",
    "results_pushed_utc": __import__('datetime').datetime.utcnow().isoformat() + "Z",
    "infrastructure": {
        "provider": "Prime Intellect (8x H100 80GB, unprivileged container — CAP_SYS_ADMIN absent)",
        "container_runtime": "Native uv venv per commit (no docker for most). Three execution paths: (1) wheels.vllm.ai/<parent>/vllm-1.0.0.dev-cp38-abi3.whl install for commits whose parent has a wheel; (2) PyPI overlay (pip install vllm==<parent_version> + apply agent patch on installed tree) for 4 no-wheel commits; (3) docker via udocker for the 11 oldest vLLM <0.6 commits (pre-cumem.py — no PRoot bug).",
        "notes": "udocker docker→shim at /usr/local/bin/docker for path (3). Path (1) is the main runner: scripts/runners/run_vllm_native.py. Path (2) reuses (1)'s code via setup_venv_overlay().",
    },
    "buckets": {"success_native": [], "success_overlay": [], "success_docker": [], "success_multi_gpu": [],
                "no_wheel_unrecovered": [], "server_crash": [], "parse_fail": [], "oom": [], "other": []},
    "totals": {},
    "known_caveats": [
        "Final result: 35/38 vLLM (92%) + 11/14 SGLang (79%) = 46/52 (88.5%) combined.",
        "MODEL_OVERRIDES applied to ~23 of 38 commits (Llama-3.1 -> Meta-Llama-3, Bamba -> Meta-Llama-3) "
        "because older vLLM versions in matching baselines don't support Llama-3.1's RoPE scaling. "
        "Within each commit, HUMAN and BASELINE use the same override, so the comparison is apples-to-apples; "
        "model-substituted commits aren't directly comparable across commits.",
        "Per-commit dep mapping in data/mappings/vllm_oh_mapping.json (transformers_pin, vllm_pypi_version, "
        "outlines_pin) — vLLM versions span 0.3.3 to 0.7+ and one transformers/outlines pin doesn't fit all.",
        "Unrecovered (3 vLLM): 9474e89b (vllm 0.3.3 benchmark_throughput.py EngineArgs ABI mismatch); "
        "e3580537 (FP8 model TransferEncodingError — likely real agent regression on FP8 + prefix-caching path); "
        "e7b20426 (Llama-4-Maverick-17B-128E OOM even at -tp 4 on 4xH100 — model genuinely too big).",
        "Unrecovered (3 SGLang): 187b85b7 (real agent regression: deque->list .popleft()); "
        "1acca3a2 (NCCL ABI mismatch in baseline image); 205d5cb4 (real OOM Llama-4-Scout).",
    ],
}

def categorize(results_dir, repo_label):
    for f in sorted(results_dir.glob("*_agent_result.json")):
        r = json.load(open(f))
        c = f.name.split("_")[0]
        entry_base = {"repo": repo_label, "commit": c, "model": r.get("model")}
        if r.get("status") == "success":
            runner = r.get("runner", "docker")
            metrics = r.get("metrics", {})
            entry = {**entry_base, "metrics": metrics, "duration_s": r.get("duration_s"),
                     "runner": runner}
            # Bucket by runner type
            if runner == "native" and r.get("perf_command", "").count("-tp") and "tensor-parallel" in str(r.get("perf_command","")):
                summary["buckets"]["success_multi_gpu"].append(entry)
            elif runner == "native" and "vllm_pypi_version" in r.get("perf_command", "") + str(r):
                # Heuristic: if metrics have ttft AND we're in the no-wheel set, it's overlay
                summary["buckets"]["success_native"].append(entry)
            elif runner == "native":
                # Check if commit was overlay-installed (PyPI fallback)
                overlay_commits = {"3476ed08", "ad8d696a", "310aca88"}
                if c in overlay_commits:
                    summary["buckets"]["success_overlay"].append(entry)
                else:
                    summary["buckets"]["success_native"].append(entry)
            else:
                summary["buckets"]["success_docker"].append(entry)
        else:
            e = r.get("error", "?")
            entry = {**entry_base, "error": e}
            if "no wheel" in e.lower():
                bucket = "no_wheel_unrecovered"
            elif "OOM" in e or "out of memory" in e.lower():
                bucket = "oom"
            elif "Server" in e and ("crash" in e.lower() or "timeout" in e.lower()):
                bucket = "server_crash"
            elif "No metrics" in e or "No latency" in e or "No throughput" in e:
                bucket = "parse_fail"
            else:
                bucket = "other"
            summary["buckets"][bucket].append(entry)


categorize(RESULTS_DIR, "vllm")
if SGLANG_RESULTS_DIR.exists():
    categorize(SGLANG_RESULTS_DIR, "sglang")

summary["totals"] = {k: len(v) for k, v in summary["buckets"].items()}
summary["totals"]["all"] = sum(summary["totals"].values())

summary_path = ROOT / "archive/results/2026-05/iso_bench_results_3way_openhands_sonnet45/SUMMARY.json"
summary_path.write_text(json.dumps(summary, indent=2, default=str))
print(f"Summary written: {summary_path} (totals: {summary['totals']})")

# 2. Push to HF
from huggingface_hub import HfApi, create_repo

api = HfApi(token=TOKEN)
try:
    create_repo(repo_id=REPO, repo_type="dataset", exist_ok=True, token=TOKEN, private=True)
    print(f"Repo ready: {REPO}")
except Exception as e:
    print(f"create_repo issue: {e}")

# Push each piece. upload_folder is the most efficient path.
print("Uploading vLLM results dir + summary...")
api.upload_folder(
    folder_path=str(RESULTS_DIR.parent),
    repo_id=REPO,
    repo_type="dataset",
    path_in_repo="results_vllm",
    token=TOKEN,
    ignore_patterns=["*.tmp", "*.lock"],
)
if SGLANG_RESULTS_DIR.exists():
    print("Uploading SGLang results dir...")
    api.upload_folder(
        folder_path=str(SGLANG_RESULTS_DIR.parent),
        repo_id=REPO,
        repo_type="dataset",
        path_in_repo="results_sglang",
        token=TOKEN,
        ignore_patterns=["*.tmp", "*.lock"],
    )

# Worker logs (small, useful for postmortem)
for log_dir in LOGS_DIRS:
    if log_dir.exists():
        api.upload_folder(
            folder_path=str(log_dir),
            repo_id=REPO,
            repo_type="dataset",
            path_in_repo=f"logs/{log_dir.name}",
            token=TOKEN,
        )

# Push key scripts (the runner + per-commit mapping)
for src in [ROOT / "scripts/runners/run_vllm_native.py",
            ROOT / "scripts/runners/run_sglang_benchmarks.py",
            ROOT / "scripts/runners/run_3way_benchmarks.py",
            ROOT / "scripts/runners/prepare_oh_patches.py",
            ROOT / "scripts/runners/checkpoint_progress.sh",
            ROOT / "data/mappings/vllm_oh_mapping.json",
            ROOT / "data/mappings/sglang_oh_mapping.json",
            ROOT / "docs/HARD_METRICS_OH_SONNET45_RUNBOOK.md"]:
    if src.exists():
        api.upload_file(path_or_fileobj=str(src), repo_id=REPO, repo_type="dataset",
                        path_in_repo=f"scripts/{src.name}", token=TOKEN)
        print(f"  uploaded {src.name}")
# udocker shim is still relevant for the 11 docker-restored commits
if Path("/usr/local/bin/docker").exists():
    api.upload_file(path_or_fileobj="/usr/local/bin/docker", repo_id=REPO, repo_type="dataset",
                    path_in_repo="scripts/docker_udocker_shim.py", token=TOKEN)

print(f"\nDone. https://huggingface.co/datasets/{REPO}")
