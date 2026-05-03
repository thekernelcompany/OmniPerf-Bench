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
LOGS_DIR_FANOUT = ROOT / "logs/oh_fanout/vllm"
LOGS_DIR_RETRY = ROOT / "logs/oh_retry3/vllm"

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
        "provider": "Prime Intellect (8x H100 80GB)",
        "container_runtime": "udocker (PRoot mode P1) — Prime Intellect pod lacked CAP_SYS_ADMIN, so dockerd unusable",
        "notes": "Custom docker→udocker shim at /usr/local/bin/docker; runner unchanged otherwise.",
    },
    "buckets": {"success": [], "no_image": [], "server_crash": [], "parse_fail": [], "other": []},
    "totals": {},
    "known_caveats": [
        "23 of 38 commits (~60%) used MODEL_OVERRIDES (Llama-3.1 -> Meta-Llama-3, Bamba -> Meta-Llama-3) "
        "because the older vLLMs in those baseline images don't support Llama-3.1's RoPE scaling. "
        "Hard metrics for those tasks are on a substitute model.",
        "4 baseline images are unbuildable (per runbook): a732900efc4e, d3ea50113c08, 0e74d797ce86, f67e9e9f221e. "
        "0e74d797ce86 is permanently unbuildable (vLLM 0.4.0 / CUDA 11).",
        "19 commits hit a 'Server crashed after applying patch' failure inside the udocker container — "
        "root cause is a PRoot ptrace ↔ multiprocessing.spawn interaction that breaks dlopen of "
        "torchvision-bundled libcudart.<hash>.so.12 in vLLM's spawn-child engine process. Multiple fix "
        "attempts (force-copy host libcudart, --execmode=F1 fakechroot, pip --force-reinstall torchvision) "
        "did not recover any. Path forward: re-run on a host with privileged Docker.",
        "5 commits hit 'No metrics in agent output' / 'No latency metrics' / 'No throughput metrics' parse "
        "failures because their baseline image's benchmark_serving.py predates --dataset-name random "
        "support. Recoverable via parser fallback to sharegpt (partial fix in repo, not validated).",
    ],
}

for f in sorted(RESULTS_DIR.glob("*_agent_result.json")):
    r = json.load(open(f))
    c = f.name.split("_")[0]
    if r.get("status") == "success":
        summary["buckets"]["success"].append({
            "commit": c,
            "model": r.get("model"),
            "metrics": r.get("metrics", {}),
            "duration_s": r.get("duration_s"),
        })
    else:
        e = r.get("error", "?")
        if "No image" in e:
            bucket = "no_image"
        elif "Server crashed" in e:
            bucket = "server_crash"
        elif "No metrics" in e or "No latency" in e or "No throughput" in e:
            bucket = "parse_fail"
        else:
            bucket = "other"
        summary["buckets"][bucket].append({"commit": c, "model": r.get("model"), "error": e})

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
print("Uploading results dir + summary + worker logs...")
api.upload_folder(
    folder_path=str(RESULTS_DIR.parent),
    repo_id=REPO,
    repo_type="dataset",
    path_in_repo="results_3way",
    token=TOKEN,
    ignore_patterns=["*.tmp", "*.lock"],
)

# Worker logs (small, useful for postmortem)
for log_dir in [LOGS_DIR_FANOUT, LOGS_DIR_RETRY]:
    if log_dir.exists():
        api.upload_folder(
            folder_path=str(log_dir),
            repo_id=REPO,
            repo_type="dataset",
            path_in_repo=f"logs/{log_dir.parent.name}",
            token=TOKEN,
        )

# Push key scripts (the udocker shim + prepare/fanout/retry/checkpoint)
api.upload_file(path_or_fileobj="/usr/local/bin/docker", repo_id=REPO, repo_type="dataset",
                path_in_repo="scripts/docker_udocker_shim.py", token=TOKEN)
for src in [ROOT / "scripts/runners/prepare_oh_patches.py",
            ROOT / "scripts/runners/run_oh_fanout.sh",
            ROOT / "scripts/runners/run_oh_retry.sh",
            ROOT / "scripts/runners/checkpoint_progress.sh",
            ROOT / "scripts/runners/run_3way_benchmarks.py",
            ROOT / "docs/HARD_METRICS_OH_SONNET45_RUNBOOK.md"]:
    if src.exists():
        api.upload_file(path_or_fileobj=str(src), repo_id=REPO, repo_type="dataset",
                        path_in_repo=f"scripts/{src.name}", token=TOKEN)
        print(f"  uploaded {src.name}")

print(f"\nDone. https://huggingface.co/datasets/{REPO}")
