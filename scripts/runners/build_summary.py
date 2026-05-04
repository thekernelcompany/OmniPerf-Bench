#!/usr/bin/env python3
"""Build SUMMARY.json aggregating all per-commit results for a given agent.

Usage:
    python build_summary.py --agent openhands_gpt5

Writes SUMMARY.json adjacent to the per-agent results dir tree.
"""
import argparse
import datetime
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent

AGENT_LAYOUT = {
    "openhands_sonnet45": {
        "model": "anthropic-claude-sonnet-4-5-20250929-v1-0",
        "vllm_results": ROOT / "archive/results/2026-05/iso_bench_results_3way_openhands_sonnet45/results",
        "sglang_results": ROOT / "archive/results/2026-05/iso_bench_results_3way_openhands_sonnet45_sglang/results",
        "summary_root": ROOT / "archive/results/2026-05/iso_bench_results_3way_openhands_sonnet45",
    },
    "openhands_gpt5": {
        "model": "openai-gpt-5",
        "vllm_results": ROOT / "archive/results/2026-05/iso_bench_results_3way_openhands_gpt5/results",
        "sglang_results": ROOT / "archive/results/2026-05/iso_bench_results_3way_openhands_gpt5_sglang/results",
        "summary_root": ROOT / "archive/results/2026-05/iso_bench_results_3way_openhands_gpt5",
    },
}


def categorize(rdir: Path, repo_label: str, buckets: dict, hardware_bound_set: set) -> None:
    if not rdir.exists():
        return
    for f in sorted(rdir.glob("*_agent_result.json")):
        r = json.loads(f.read_text())
        c = f.name[:8]
        if c in hardware_bound_set:
            buckets["hardware_bound"].append({"repo": repo_label, "commit": c,
                                               "model": r.get("model"),
                                               "reason": hardware_bound_set[c]})
            continue
        entry_base = {"repo": repo_label, "commit": c, "model": r.get("model"),
                      "perf_command": r.get("perf_command", "")[:200]}
        if r.get("status") == "success":
            metrics = r.get("metrics", {})
            entry = {**entry_base, "metrics": metrics, "duration_s": r.get("duration_s"),
                     "runner": r.get("runner", "?"),
                     "iso_override": r.get("iso_bench_override_applied", False)}
            # Bucket by runner type / model class
            cvd = r.get("perf_command", "")
            if "-tp 2" in cvd or " -tp 4" in cvd:
                buckets["success_multi_gpu"].append(entry)
            elif r.get("runner") == "sglang_native":
                if "bench_one_batch" in r.get("perf_command", ""):
                    buckets["success_sglang_one_batch"].append(entry)
                else:
                    buckets["success_sglang_native"].append(entry)
            elif r.get("runner") == "native":
                # Heuristic: overlay if raw_output mentions "PyPI"
                raw = r.get("raw_output", "")
                if "from PyPI" in raw or "Overlay (PyPI install)" in raw:
                    buckets["success_overlay"].append(entry)
                else:
                    buckets["success_native"].append(entry)
            else:
                buckets["other_success"].append(entry)
        else:
            entry = {**entry_base, "error": r.get("error", "?"),
                     "_note": r.get("_note", "")}
            buckets["agent_regression_or_infra"].append(entry)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--agent", default="openhands_gpt5",
                   choices=sorted(AGENT_LAYOUT.keys()))
    args = p.parse_args()
    layout = AGENT_LAYOUT[args.agent]

    HW_BOUND = {
        "e7b20426": "Llama-4-Maverick (~400 GB) doesn't fit on 2xH100 (or 8xH100); needs 16xH100 / 8xH200",
        "6dd94dbe": "parent vLLM 0.4.0/CUDA 11 baseline unbuildable on CUDA 12+ host (sonnet45 also dropped)",
    }

    summary = {
        "schema_version": "1.0",
        "agent": "openhands",
        "model": layout["model"],
        "results_pushed_utc": datetime.datetime.utcnow().isoformat() + "Z",
        "infrastructure": {
            "provider": "Local 2x H100 80GB PCIe (driver 535.183.06, CUDA 12.2)",
            "container_runtime": "Native uv venv per commit (no docker). Three execution paths:\n"
                "  (1) wheels.vllm.ai/<parent>/vllm-1.0.0.dev-cp38-abi3.whl install (preferred);\n"
                "  (2) PyPI overlay `pip install vllm==<parent_version>` for no-wheel commits;\n"
                "  (3) Native SGLang via `pip install sglang[all]==<tag-from-submodule>` + patch -p2.",
            "hf_cache": "/ephemeral/huggingface_cache",
            "uv_cache":  "/ephemeral/uv_cache",
            "venvs_root": "/ephemeral/{native_venvs, native_sglang_venvs}",
            "perf_command_authority": "ISO-Bench/ISO-Bench HF dataset (mirrored at data/iso_bench/{vllm,sglang}/train.parquet)",
        },
        "buckets": {
            "success_native": [],
            "success_overlay": [],
            "success_sglang_native": [],
            "success_sglang_one_batch": [],
            "success_multi_gpu": [],
            "agent_regression_or_infra": [],
            "hardware_bound": [],
            "other_success": [],
        },
        "totals": {},
    }

    categorize(layout["vllm_results"], "vllm", summary["buckets"], HW_BOUND)
    categorize(layout["sglang_results"], "sglang", summary["buckets"], HW_BOUND)

    # HW-bound commits that don't have result files (intentional skip).
    # Look up their model + perf_command from local mapping for documentation.
    for c, why in HW_BOUND.items():
        if any(c == e["commit"] for e in summary["buckets"]["hardware_bound"]):
            continue  # already counted from result file
        for repo, mp_path in [("vllm", "data/mappings/vllm_oh_mapping.json"),
                               ("sglang", "data/mappings/sglang_oh_mapping.json")]:
            mp = json.loads((ROOT / mp_path).read_text())
            if c in mp:
                summary["buckets"]["hardware_bound"].append({
                    "repo": repo, "commit": c,
                    "model": mp[c].get("model"),
                    "reason": why,
                    "_note": "Not benchmarked; documented hardware constraint.",
                })
                break

    summary["totals"] = {k: len(v) for k, v in summary["buckets"].items()}
    summary["totals"]["all_buckets_sum"] = sum(summary["totals"].values())
    success_count = sum(len(v) for k, v in summary["buckets"].items() if k.startswith("success_") or k == "other_success")
    summary["totals"]["success"] = success_count
    summary["totals"]["fail"] = len(summary["buckets"]["agent_regression_or_infra"])
    summary["totals"]["hardware_bound"] = len(summary["buckets"]["hardware_bound"])

    summary["known_caveats"] = [
        f"Final result: {success_count}/{summary['totals']['all_buckets_sum']} successful "
        f"({success_count/summary['totals']['all_buckets_sum']*100:.1f}%); "
        f"{summary['totals']['fail']} legitimate agent/infra failures; "
        f"{summary['totals']['hardware_bound']} hardware-bound (not benchmarked).",
        "MODEL_OVERRIDES applied at runtime: meta-llama/Llama-3.1-* → meta-llama/Meta-Llama-3-*, "
        "ibm-ai-platform/Bamba-9B* → meta-llama/Meta-Llama-3-8B-Instruct (older vLLM compat); "
        "ISO-Bench typo fixes Llama-3-8B → Meta-Llama-3-8B + Qwen3-7B-Instruct → Qwen2.5-7B-Instruct.",
        "ISO-Bench is authoritative for perf_command + --model; local mapping holds dep overrides only.",
        "310aca88 was run with -tp 2 instead of ISO-Bench's -tp 4 (only 2 GPUs available); "
        "absolute latency NOT directly comparable to runs at -tp 4. Runtime tp-rewrite based on visible GPUs.",
        "6e36f4fa patch reconstructed from trajectory.json (4 of 7 str_replace ops applied; 3 had "
        "literal +/- diff markers in old_str, would have been no-ops at agent runtime). See "
        "ISO-Bench/state/runs/vllm/openhands_gpt5/flat/vllm_core-0034/RECONSTRUCTED.txt.",
        "1acca3a2 + 205d5cb4 use sglang 0.4.7 substitution (instead of 0.4.6.post3/post5) to avoid "
        "torch 2.6 register_constant ABI break. 1acca3a2 succeeded; 205d5cb4 hit a deeper "
        "compressed-tensors WNA16 MoE dependency tangle even with vllm co-installed.",
        "31589e17 server-hang and 205d5cb4 dependency-tangle errors are saved in result JSONs as "
        "manually-marked status=error with _note field documenting the reason. Both have empty "
        "raw_output but full provenance in the error message.",
    ]

    out_path = layout["summary_root"] / "SUMMARY.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary, indent=2, default=str))
    print(f"Wrote: {out_path}")
    print(f"Totals: {summary['totals']}")


if __name__ == "__main__":
    main()
