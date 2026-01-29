#!/usr/bin/env python3
"""
Upload TRAE GPT-5 vLLM benchmark results to HuggingFace.

Schema v5 compatible with: Inferencebench/claude-code-vllm-benchmarks

This agent has partial baseline/human data in:
- results/*_human_result.json
- baseline_benchmark_results/*_baseline_result.json

Usage:
    # Dry run to see what would be uploaded
    python upload_to_hf.py --dry-run

    # Save to JSON for inspection
    python upload_to_hf.py --save-json exports/trae_gpt5_results.json

    # Upload to HuggingFace
    python upload_to_hf.py --repo-id "Inferencebench/trae-gpt5-vllm-benchmarks"
"""

import sys
from pathlib import Path

# Add scripts/upload to path for unified_agent_upload
script_dir = Path(__file__).parent.parent / "scripts" / "upload"
sys.path.insert(0, str(script_dir))

from unified_agent_upload import (
    collect_agent_results,
    enrich_with_master_dataset,
    upload_to_huggingface,
)


# Agent configuration
AGENT_NAME = "trae"
AGENT_MODEL = "gpt-5"
DATA_SOURCE = "trae_gpt5"
DEFAULT_REPO_ID = "Inferencebench/trae-gpt5-vllm-benchmarks"


def main():
    import argparse
    import json
    import os

    parser = argparse.ArgumentParser(
        description="Upload TRAE GPT-5 benchmark results to HuggingFace (Schema v5)"
    )
    parser.add_argument("--repo-id", default=DEFAULT_REPO_ID,
                        help=f"HuggingFace repo ID (default: {DEFAULT_REPO_ID})")
    parser.add_argument("--token", default=None,
                        help="HuggingFace token (default: use HF_TOKEN env var)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Collect results but don't upload")
    parser.add_argument("--save-json", default=None,
                        help="Save results to JSON file")
    parser.add_argument("--enrich-metadata", action="store_true",
                        help="Enrich with master dataset metadata")

    args = parser.parse_args()

    # Paths relative to this script
    script_path = Path(__file__).parent
    results_dir = script_path / "results"
    baseline_dir = script_path / "baseline_benchmark_results"
    master_dataset = script_path.parent / "data" / "vllm_dataset_with_test.jsonl"

    print(f"Collecting TRAE GPT-5 results from {results_dir}...")
    print(f"Agent: {AGENT_NAME} ({AGENT_MODEL})")
    print(f"Baseline dir: {baseline_dir}")

    # Collect results (TRAE GPT-5 has partial baseline data)
    results = collect_agent_results(
        results_dir=str(results_dir),
        baseline_dir=str(baseline_dir) if baseline_dir.exists() else None,
        agent_name=AGENT_NAME,
        agent_model=AGENT_MODEL,
        data_source=DATA_SOURCE,
    )

    print(f"Collected {len(results)} results")

    # Enrich with master dataset
    if args.enrich_metadata:
        results = enrich_with_master_dataset(results, str(master_dataset))

    # Print summary
    print(f"\n{'='*60}")
    print(f"Summary: {len(results)} benchmark results")
    print(f"{'='*60}")

    with_agent_metrics = sum(1 for r in results if r.get("agent_ttft_mean") or r.get("agent_latency_avg"))
    with_human_metrics = sum(1 for r in results if r.get("human_ttft_mean") or r.get("human_latency_avg"))
    with_baseline_metrics = sum(1 for r in results if r.get("baseline_ttft_mean") or r.get("baseline_latency_avg"))

    print(f"\nMetrics coverage:")
    print(f"  With agent metrics: {with_agent_metrics}")
    print(f"  With human metrics: {with_human_metrics}")
    print(f"  With baseline metrics: {with_baseline_metrics}")

    # Count improvements (where we can calculate them)
    with_improvements = sum(1 for r in results if r.get("agent_improvement_ttft_mean") is not None)
    print(f"  With improvement calculations: {with_improvements}")

    # Count by benchmark mode
    mode_counts = {}
    for r in results:
        mode = r.get("benchmark_mode") or "unknown"
        mode_counts[mode] = mode_counts.get(mode, 0) + 1
    print(f"\nBy benchmark mode:")
    for mode, count in sorted(mode_counts.items(), key=lambda x: -x[1]):
        print(f"  {mode}: {count}")

    # Save to JSON
    if args.save_json:
        save_path = Path(args.save_json)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        with open(save_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nSaved results to {args.save_json}")
        return

    # Dry run
    if args.dry_run:
        print("\nDry run - not uploading to HuggingFace")
        if results:
            print(f"\nSchema columns ({len(results[0])} total)")
        return

    # Upload
    token = args.token or os.environ.get("HF_TOKEN")
    if not token:
        try:
            from huggingface_hub import HfFolder
            token = HfFolder.get_token()
        except Exception:
            pass

    if not token:
        print("\nError: No HuggingFace token. Set HF_TOKEN env var or use --token")
        return

    print(f"\nUploading to HuggingFace: {args.repo_id}...")
    upload_to_huggingface(results, args.repo_id, AGENT_NAME, AGENT_MODEL, token)


if __name__ == "__main__":
    main()
