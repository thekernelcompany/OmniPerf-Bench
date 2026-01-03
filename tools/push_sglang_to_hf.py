#!/usr/bin/env python3
"""
Push SGLang benchmark results to HuggingFace.

Similar format to: https://huggingface.co/datasets/Inferencebench/claude-code-vllm-benchmarks
"""

import json
import os
from pathlib import Path
from datetime import date
from datasets import Dataset
from huggingface_hub import HfApi

# Configuration
RESULTS_DIR = Path("/home/ubuntu/OmniPerf-Bench/omniperf_results_3way_sglang/sglang")
HF_REPO = "Inferencebench/claude-code-sglang-benchmarks"
AGENT_NAME = "claude-code"
AGENT_MODEL = "claude-sonnet-4-20250514"
REPO_NAME = "sgl-project/sglang"


def determine_benchmark_mode(perf_command: str) -> str:
    """Determine benchmark mode from command."""
    if not perf_command:
        return "unknown"
    cmd_lower = perf_command.lower()
    if "bench_serving" in cmd_lower:
        return "serving"
    elif "bench_one_batch" in cmd_lower:
        return "standalone"
    elif "bench_offline" in cmd_lower:
        return "offline"
    return "serving"


def load_results():
    """Load all benchmark results."""
    results = []

    for result_dir in RESULTS_DIR.iterdir():
        if not result_dir.is_dir():
            continue

        result_file = result_dir / "benchmark_result.json"
        if not result_file.exists():
            continue

        try:
            with open(result_file) as f:
                data = json.load(f)

            # Transform to dataset schema
            row = {
                # Identity
                "commit_hash": data.get("full_commit", data.get("commit", "")),
                "commit_subject": data.get("subject", ""),
                "repo": REPO_NAME,

                # Configuration
                "model": data.get("model", ""),
                "gpu_config": data.get("gpu_config", "H100:1"),
                "benchmark_mode": determine_benchmark_mode(data.get("perf_command", "")),
                "perf_command": data.get("perf_command", ""),

                # Status
                "status": data.get("status", "error"),
                "error": data.get("error", "") or "",
                "duration_s": data.get("duration_s", 0.0),
                "has_agent_patch": data.get("has_agent_patch", False),

                # Installation methods
                "install_method": data.get("install_method", "docker"),

                # Baseline metrics (empty in most SGLang runs due to overlay issues)
                "baseline_ttft_mean": data.get("baseline_metrics", {}).get("ttft_mean"),
                "baseline_ttft_median": data.get("baseline_metrics", {}).get("ttft_median"),
                "baseline_tpot_mean": data.get("baseline_metrics", {}).get("tpot_mean"),
                "baseline_itl_mean": data.get("baseline_metrics", {}).get("itl_mean"),
                "baseline_request_throughput": data.get("baseline_metrics", {}).get("request_throughput"),
                "baseline_output_throughput": data.get("baseline_metrics", {}).get("output_throughput"),

                # Human metrics
                "human_ttft_mean": data.get("human_metrics", {}).get("ttft_mean"),
                "human_ttft_median": data.get("human_metrics", {}).get("ttft_median"),
                "human_ttft_p99": data.get("human_metrics", {}).get("ttft_p99"),
                "human_tpot_mean": data.get("human_metrics", {}).get("tpot_mean"),
                "human_tpot_median": data.get("human_metrics", {}).get("tpot_median"),
                "human_tpot_p99": data.get("human_metrics", {}).get("tpot_p99"),
                "human_itl_mean": data.get("human_metrics", {}).get("itl_mean"),
                "human_itl_median": data.get("human_metrics", {}).get("itl_median"),
                "human_itl_p99": data.get("human_metrics", {}).get("itl_p99"),
                "human_request_throughput": data.get("human_metrics", {}).get("request_throughput"),
                "human_output_throughput": data.get("human_metrics", {}).get("output_throughput"),
                "human_input_throughput": data.get("human_metrics", {}).get("input_throughput"),
                "human_e2e_latency_mean": data.get("human_metrics", {}).get("e2e_latency_mean"),
                "human_e2e_latency_median": data.get("human_metrics", {}).get("e2e_latency_median"),

                # Agent metrics
                "agent_ttft_mean": data.get("agent_metrics", {}).get("ttft_mean") if data.get("agent_metrics") else None,
                "agent_ttft_median": data.get("agent_metrics", {}).get("ttft_median") if data.get("agent_metrics") else None,
                "agent_tpot_mean": data.get("agent_metrics", {}).get("tpot_mean") if data.get("agent_metrics") else None,
                "agent_itl_mean": data.get("agent_metrics", {}).get("itl_mean") if data.get("agent_metrics") else None,
                "agent_request_throughput": data.get("agent_metrics", {}).get("request_throughput") if data.get("agent_metrics") else None,
                "agent_output_throughput": data.get("agent_metrics", {}).get("output_throughput") if data.get("agent_metrics") else None,

                # Improvement metrics
                "human_improvement_tpot_mean": data.get("human_improvement", {}).get("tpot_mean") if data.get("human_improvement") else None,
                "human_improvement_throughput": data.get("human_improvement", {}).get("request_throughput") if data.get("human_improvement") else None,
                "agent_improvement_tpot_mean": data.get("agent_improvement", {}).get("tpot_mean") if data.get("agent_improvement") else None,
                "agent_improvement_throughput": data.get("agent_improvement", {}).get("request_throughput") if data.get("agent_improvement") else None,
                "agent_vs_human_tpot_mean": data.get("agent_vs_human", {}).get("tpot_mean") if data.get("agent_vs_human") else None,

                # Meta
                "agent_name": AGENT_NAME,
                "agent_model": AGENT_MODEL,
                "benchmark_date": str(date.today()),
                "parent_commit": data.get("parent_commit", ""),
            }

            results.append(row)
            print(f"Loaded: {data.get('commit', 'unknown')} - {data.get('status', 'unknown')}")

        except Exception as e:
            print(f"Error loading {result_dir}: {e}")

    return results


def main():
    print("Loading SGLang benchmark results...")
    results = load_results()
    print(f"\nLoaded {len(results)} results")

    # Count by status
    status_counts = {}
    for r in results:
        status = r["status"]
        status_counts[status] = status_counts.get(status, 0) + 1
    print(f"Status distribution: {status_counts}")

    # Create dataset
    print("\nCreating HuggingFace dataset...")
    dataset = Dataset.from_list(results)
    print(f"Dataset: {dataset}")
    print(f"Features: {dataset.features}")

    # Push to HuggingFace
    print(f"\nPushing to {HF_REPO}...")
    dataset.push_to_hub(
        HF_REPO,
        private=False,
        token=os.environ.get("HF_TOKEN"),
    )

    print(f"\nSuccess! Dataset available at: https://huggingface.co/datasets/{HF_REPO}")


if __name__ == "__main__":
    main()
