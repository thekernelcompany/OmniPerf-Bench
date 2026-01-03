#!/usr/bin/env python3
"""
Upload Claude Code vLLM benchmark results to HuggingFace.

This script collects all benchmark results from the omniperf_results_3way_claude_code/vllm
directory and uploads them to HuggingFace as a dataset.
"""

import json
import glob
import os
from datetime import datetime
from pathlib import Path

def collect_benchmark_results(results_dir: str) -> list[dict]:
    """Collect all benchmark results from the results directory."""
    results = []

    for result_file in glob.glob(os.path.join(results_dir, "*/benchmark_result.json")):
        try:
            with open(result_file) as f:
                data = json.load(f)

            instance = data.get("instance", {})
            result = data.get("result", {})

            # Flatten the structure for HuggingFace dataset
            row = {
                # Instance metadata
                "commit_hash": instance.get("commit_hash"),
                "commit_subject": instance.get("commit_subject"),
                "repo": instance.get("repo", "vllm"),
                "perf_command": instance.get("perf_command"),
                "files_changed": instance.get("files_changed", []),
                "pr_url": instance.get("pr_url"),
                "models": instance.get("models", []),

                # Result metadata
                "status": result.get("status"),
                "gpu_config": result.get("gpu_config"),
                "benchmark_mode": result.get("benchmark_mode"),
                "patch_type": result.get("patch_type"),
                "duration_s": result.get("duration_s"),
                "error": result.get("error"),
                "error_message": result.get("error_message"),

                # Version info
                "baseline_version": result.get("baseline_version"),
                "human_version": result.get("human_version"),
                "model": result.get("model"),
                "has_agent_patch": result.get("has_agent_patch"),

                # Baseline metrics - Serving benchmark (before optimization)
                "baseline_ttft_mean": None,
                "baseline_ttft_median": None,
                "baseline_ttft_p99": None,
                "baseline_tpot_mean": None,
                "baseline_tpot_median": None,
                "baseline_tpot_p99": None,
                "baseline_itl_mean": None,
                "baseline_itl_median": None,
                "baseline_itl_p99": None,

                # Baseline metrics - Latency/Throughput benchmark
                "baseline_latency_avg": None,
                "baseline_throughput": None,

                # Human metrics - Serving benchmark (ground truth optimization)
                "human_ttft_mean": None,
                "human_ttft_median": None,
                "human_ttft_p99": None,
                "human_tpot_mean": None,
                "human_tpot_median": None,
                "human_tpot_p99": None,
                "human_itl_mean": None,
                "human_itl_median": None,
                "human_itl_p99": None,

                # Human metrics - Latency/Throughput benchmark
                "human_latency_avg": None,
                "human_throughput": None,

                # Agent metrics - Serving benchmark (Claude Code optimization)
                "agent_ttft_mean": None,
                "agent_ttft_median": None,
                "agent_ttft_p99": None,
                "agent_tpot_mean": None,
                "agent_tpot_median": None,
                "agent_tpot_p99": None,
                "agent_itl_mean": None,
                "agent_itl_median": None,
                "agent_itl_p99": None,

                # Agent metrics - Latency/Throughput benchmark
                "agent_latency_avg": None,
                "agent_throughput": None,

                # Improvement metrics - Serving
                "human_improvement_ttft_mean": None,
                "human_improvement_tpot_mean": None,
                "human_improvement_itl_mean": None,
                "agent_improvement_ttft_mean": None,
                "agent_improvement_tpot_mean": None,
                "agent_improvement_itl_mean": None,
                "agent_vs_human_ttft_mean": None,
                "agent_vs_human_tpot_mean": None,
                "agent_vs_human_itl_mean": None,

                # Improvement metrics - Latency/Throughput
                "human_improvement_latency_avg": None,
                "human_improvement_throughput": None,
                "agent_improvement_latency_avg": None,
                "agent_improvement_throughput": None,
                "agent_vs_human_latency_avg": None,
                "agent_vs_human_throughput": None,

                # Raw benchmark outputs
                "baseline_raw": result.get("baseline_raw"),
                "human_raw": result.get("human_raw"),
                "agent_raw": result.get("agent_raw"),

                # Test script
                "test_script": instance.get("test_script"),
            }

            # Extract baseline metrics
            baseline = result.get("baseline_metrics", {})
            if baseline:
                # Serving benchmark metrics
                row["baseline_ttft_mean"] = baseline.get("ttft_mean")
                row["baseline_ttft_median"] = baseline.get("ttft_median")
                row["baseline_ttft_p99"] = baseline.get("ttft_p99")
                row["baseline_tpot_mean"] = baseline.get("tpot_mean")
                row["baseline_tpot_median"] = baseline.get("tpot_median")
                row["baseline_tpot_p99"] = baseline.get("tpot_p99")
                row["baseline_itl_mean"] = baseline.get("itl_mean")
                row["baseline_itl_median"] = baseline.get("itl_median")
                row["baseline_itl_p99"] = baseline.get("itl_p99")
                # Latency/Throughput benchmark metrics
                row["baseline_latency_avg"] = baseline.get("latency_avg")
                row["baseline_throughput"] = baseline.get("throughput")

            # Extract human metrics
            human = result.get("human_metrics", {})
            if human:
                # Serving benchmark metrics
                row["human_ttft_mean"] = human.get("ttft_mean")
                row["human_ttft_median"] = human.get("ttft_median")
                row["human_ttft_p99"] = human.get("ttft_p99")
                row["human_tpot_mean"] = human.get("tpot_mean")
                row["human_tpot_median"] = human.get("tpot_median")
                row["human_tpot_p99"] = human.get("tpot_p99")
                row["human_itl_mean"] = human.get("itl_mean")
                row["human_itl_median"] = human.get("itl_median")
                row["human_itl_p99"] = human.get("itl_p99")
                # Latency/Throughput benchmark metrics
                row["human_latency_avg"] = human.get("latency_avg")
                row["human_throughput"] = human.get("throughput")

            # Extract agent metrics
            agent = result.get("agent_metrics", {})
            if agent:
                # Serving benchmark metrics
                row["agent_ttft_mean"] = agent.get("ttft_mean")
                row["agent_ttft_median"] = agent.get("ttft_median")
                row["agent_ttft_p99"] = agent.get("ttft_p99")
                row["agent_tpot_mean"] = agent.get("tpot_mean")
                row["agent_tpot_median"] = agent.get("tpot_median")
                row["agent_tpot_p99"] = agent.get("tpot_p99")
                row["agent_itl_mean"] = agent.get("itl_mean")
                row["agent_itl_median"] = agent.get("itl_median")
                row["agent_itl_p99"] = agent.get("itl_p99")
                # Latency/Throughput benchmark metrics
                row["agent_latency_avg"] = agent.get("latency_avg")
                row["agent_throughput"] = agent.get("throughput")

            # Extract improvement metrics
            human_imp = result.get("human_improvement", {})
            if human_imp:
                # Serving benchmark improvements
                row["human_improvement_ttft_mean"] = human_imp.get("ttft_mean")
                row["human_improvement_tpot_mean"] = human_imp.get("tpot_mean")
                row["human_improvement_itl_mean"] = human_imp.get("itl_mean")
                # Latency/Throughput improvements
                row["human_improvement_latency_avg"] = human_imp.get("latency_avg")
                row["human_improvement_throughput"] = human_imp.get("throughput")

            agent_imp = result.get("agent_improvement", {})
            if agent_imp:
                # Serving benchmark improvements
                row["agent_improvement_ttft_mean"] = agent_imp.get("ttft_mean")
                row["agent_improvement_tpot_mean"] = agent_imp.get("tpot_mean")
                row["agent_improvement_itl_mean"] = agent_imp.get("itl_mean")
                # Latency/Throughput improvements
                row["agent_improvement_latency_avg"] = agent_imp.get("latency_avg")
                row["agent_improvement_throughput"] = agent_imp.get("throughput")

            agent_vs_human = result.get("agent_vs_human", {})
            if agent_vs_human:
                # Serving benchmark comparisons
                row["agent_vs_human_ttft_mean"] = agent_vs_human.get("ttft_mean")
                row["agent_vs_human_tpot_mean"] = agent_vs_human.get("tpot_mean")
                row["agent_vs_human_itl_mean"] = agent_vs_human.get("itl_mean")
                # Latency/Throughput comparisons
                row["agent_vs_human_latency_avg"] = agent_vs_human.get("latency_avg")
                row["agent_vs_human_throughput"] = agent_vs_human.get("throughput")

            results.append(row)

        except Exception as e:
            print(f"Error processing {result_file}: {e}")
            continue

    return results


def upload_to_huggingface(results: list[dict], repo_id: str, token: str = None):
    """Upload results to HuggingFace as a dataset."""
    from datasets import Dataset
    from huggingface_hub import HfApi

    # Create dataset
    dataset = Dataset.from_list(results)

    # Add metadata
    dataset.info.description = """
    Claude Code vLLM Performance Benchmark Results

    This dataset contains performance benchmark results from running Claude Code
    (Anthropic's coding agent) on vLLM performance optimization tasks.

    Each row represents a benchmark run for a specific vLLM commit, comparing:
    - Baseline: Performance before the optimization commit
    - Human: Performance with the human-authored optimization
    - Agent: Performance with Claude Code's optimization attempt

    Key metrics include:
    - TTFT: Time to First Token (ms) - latency until first token generation
    - TPOT: Time per Output Token (ms) - average time per generated token
    - ITL: Inter-token Latency (ms) - time between consecutive tokens

    All benchmarks were run on Modal cloud with H100 GPUs.
    """

    # Push to hub
    dataset.push_to_hub(
        repo_id,
        token=token,
        commit_message=f"Upload Claude Code vLLM benchmark results ({len(results)} commits)"
    )

    print(f"Successfully uploaded {len(results)} benchmark results to {repo_id}")
    return dataset


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Upload Claude Code vLLM benchmark results to HuggingFace")
    parser.add_argument("--repo-id", default="Inferencebench/claude-code-vllm-benchmarks",
                        help="HuggingFace repo ID (default: Inferencebench/claude-code-vllm-benchmarks)")
    parser.add_argument("--results-dir", default="vllm",
                        help="Directory containing benchmark results (default: vllm)")
    parser.add_argument("--token", default=None,
                        help="HuggingFace token (default: use HF_TOKEN env var)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Collect results but don't upload")
    parser.add_argument("--save-json", default=None,
                        help="Save results to JSON file instead of uploading")
    args = parser.parse_args()

    # Get token
    token = args.token or os.environ.get("HF_TOKEN")

    # Collect results
    script_dir = Path(__file__).parent
    results_dir = script_dir / args.results_dir

    print(f"Collecting benchmark results from {results_dir}...")
    results = collect_benchmark_results(str(results_dir))

    # Print summary
    status_counts = {}
    for r in results:
        status = r.get("status", "unknown")
        status_counts[status] = status_counts.get(status, 0) + 1

    print(f"\nCollected {len(results)} benchmark results:")
    for status, count in sorted(status_counts.items(), key=lambda x: -x[1]):
        print(f"  {status}: {count}")

    # Count successful results with metrics
    with_serving_metrics = sum(1 for r in results if r.get("baseline_ttft_mean") is not None)
    with_latency_metrics = sum(1 for r in results if r.get("baseline_latency_avg") is not None)
    with_any_metrics = sum(1 for r in results if r.get("baseline_ttft_mean") is not None or r.get("baseline_latency_avg") is not None)
    print(f"\nResults with serving metrics (TTFT/TPOT/ITL): {with_serving_metrics}")
    print(f"Results with latency/throughput metrics: {with_latency_metrics}")
    print(f"Total results with hard metrics: {with_any_metrics}")

    if args.save_json:
        with open(args.save_json, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nSaved results to {args.save_json}")
        return

    if args.dry_run:
        print("\nDry run - not uploading to HuggingFace")
        return

    if not token:
        print("\nError: No HuggingFace token provided. Set HF_TOKEN env var or use --token")
        return

    # Upload
    print(f"\nUploading to HuggingFace: {args.repo_id}...")
    upload_to_huggingface(results, args.repo_id, token)


if __name__ == "__main__":
    main()
