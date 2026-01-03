#!/usr/bin/env python3
"""
Push SGLang benchmark results to HuggingFace using unified schema.

Uses the unified schema from unified_schema.py to ensure consistency
with vLLM benchmark results.
"""

import json
import os
from pathlib import Path
from datetime import date

from unified_schema import UNIFIED_COLUMNS, normalize_row, get_schema_info


# Configuration
RESULTS_DIR = Path(__file__).parent.parent / "omniperf_results_3way_sglang" / "sglang"
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


def extract_metrics(metrics_dict: dict, prefix: str) -> dict:
    """Extract metrics from a nested dict and flatten with prefix."""
    result = {}
    if not metrics_dict:
        return result

    # Latency metrics
    result[f"{prefix}_ttft_mean"] = metrics_dict.get("ttft_mean")
    result[f"{prefix}_ttft_median"] = metrics_dict.get("ttft_median")
    result[f"{prefix}_tpot_mean"] = metrics_dict.get("tpot_mean")
    result[f"{prefix}_itl_mean"] = metrics_dict.get("itl_mean")

    # Throughput metrics - SGLang uses "request_throughput", map to "throughput"
    result[f"{prefix}_throughput"] = metrics_dict.get("request_throughput")
    result[f"{prefix}_output_throughput"] = metrics_dict.get("output_throughput")

    # E2E latency metrics
    result[f"{prefix}_e2e_latency_mean"] = metrics_dict.get("e2e_latency_mean")
    result[f"{prefix}_e2e_latency_median"] = metrics_dict.get("e2e_latency_median")

    return result


def load_results(results_dir: Path = None) -> list[dict]:
    """Load all benchmark results from the results directory."""
    if results_dir is None:
        results_dir = RESULTS_DIR

    results = []

    for result_dir in results_dir.iterdir():
        if not result_dir.is_dir():
            continue

        result_file = result_dir / "benchmark_result.json"
        if not result_file.exists():
            continue

        try:
            with open(result_file) as f:
                data = json.load(f)

            # Build row with unified schema
            row = {
                # Metadata
                "commit_hash": data.get("full_commit") or data.get("commit", ""),
                "commit_subject": data.get("subject", ""),
                "repo": REPO_NAME,
                "model": data.get("model", ""),
                "gpu_config": data.get("gpu_config", "H100:1"),
                "benchmark_mode": determine_benchmark_mode(data.get("perf_command", "")),
                "perf_command": data.get("perf_command", ""),
                "status": data.get("status", "error"),
                "error": data.get("error") or "",
                "duration_s": data.get("duration_s", 0.0),
                "has_agent_patch": data.get("has_agent_patch", False),
                "agent_name": AGENT_NAME,
                "agent_model": AGENT_MODEL,
                "benchmark_date": str(date.today()),
            }

            # Extract metrics for each variant
            baseline_metrics = data.get("baseline_metrics", {})
            human_metrics = data.get("human_metrics", {})
            agent_metrics = data.get("agent_metrics") or {}

            row.update(extract_metrics(baseline_metrics, "baseline"))
            row.update(extract_metrics(human_metrics, "human"))
            row.update(extract_metrics(agent_metrics, "agent"))

            # Normalize to unified schema (fills missing columns with None)
            normalized_row = normalize_row(row, source="sglang")
            results.append(normalized_row)

            print(f"Loaded: {data.get('commit', 'unknown')[:8]} - {data.get('status', 'unknown')}")

        except Exception as e:
            print(f"Error loading {result_dir}: {e}")

    return results


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Push SGLang benchmark results to HuggingFace")
    parser.add_argument("--repo-id", default=HF_REPO,
                        help=f"HuggingFace repo ID (default: {HF_REPO})")
    parser.add_argument("--results-dir", type=Path, default=RESULTS_DIR,
                        help=f"Directory containing benchmark results (default: {RESULTS_DIR})")
    parser.add_argument("--token", default=None,
                        help="HuggingFace token (default: use HF_TOKEN env var)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Collect results but don't upload")
    parser.add_argument("--save-json", type=Path, default=None,
                        help="Save results to JSON file")
    args = parser.parse_args()

    # Print schema info
    schema_info = get_schema_info()
    print(f"Using unified schema v{schema_info['version']} ({schema_info['total_columns']} columns)")
    print(f"Column groups: {schema_info['column_groups']}")
    print()

    # Load results
    print(f"Loading SGLang benchmark results from {args.results_dir}...")
    results = load_results(args.results_dir)

    # Print summary
    status_counts = {}
    for r in results:
        status = r.get("status", "unknown")
        status_counts[status] = status_counts.get(status, 0) + 1

    print(f"\nLoaded {len(results)} benchmark results:")
    for status, count in sorted(status_counts.items(), key=lambda x: -x[1]):
        print(f"  {status}: {count}")

    # Count results with metrics
    with_latency = sum(1 for r in results if r.get("baseline_ttft_mean") is not None)
    with_throughput = sum(1 for r in results if r.get("baseline_throughput") is not None)
    with_human = sum(1 for r in results if r.get("human_ttft_mean") is not None)
    with_agent = sum(1 for r in results if r.get("agent_ttft_mean") is not None)

    print(f"\nMetrics availability:")
    print(f"  With baseline latency metrics: {with_latency}")
    print(f"  With baseline throughput metrics: {with_throughput}")
    print(f"  With human metrics: {with_human}")
    print(f"  With agent metrics: {with_agent}")

    if args.save_json:
        with open(args.save_json, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nSaved results to {args.save_json}")
        return

    if args.dry_run:
        print("\nDry run - not uploading to HuggingFace")
        # Print sample row
        if results:
            print("\nSample row columns:")
            for col in UNIFIED_COLUMNS:
                val = results[0].get(col)
                print(f"  {col}: {val}")
        return

    # Get token
    token = args.token or os.environ.get("HF_TOKEN")
    if not token:
        print("\nError: No HuggingFace token provided. Set HF_TOKEN env var or use --token")
        return

    # Create and push dataset
    from datasets import Dataset
    print(f"\nCreating HuggingFace dataset...")
    dataset = Dataset.from_list(results)
    print(f"Dataset: {dataset}")
    print(f"Features: {list(dataset.features.keys())}")

    print(f"\nPushing to {args.repo_id}...")
    dataset.push_to_hub(
        args.repo_id,
        private=False,
        token=token,
        commit_message=f"Update with unified schema v{schema_info['version']} ({len(results)} results)"
    )

    print(f"\nSuccess! Dataset available at: https://huggingface.co/datasets/{args.repo_id}")


if __name__ == "__main__":
    main()
