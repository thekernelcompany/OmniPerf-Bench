#!/usr/bin/env python3
"""
Upload TRAE Sonnet 4.5 vLLM benchmark results to HuggingFace.

Schema v5 compatible with: Inferencebench/claude-code-vllm-benchmarks

Note: This agent has agent-only results (no baseline/human data).

Usage:
    # Dry run to see what would be uploaded
    python upload_to_hf.py --dry-run

    # Save to JSON for inspection
    python upload_to_hf.py --save-json exports/trae_sonnet45_results.json

    # Upload to HuggingFace
    python upload_to_hf.py --repo-id "Inferencebench/trae-sonnet45-vllm-benchmarks"
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
AGENT_MODEL = "sonnet-4.5"
DATA_SOURCE = "trae_sonnet45"
DEFAULT_REPO_ID = "Inferencebench/trae-sonnet45-vllm-benchmarks"


def main():
    import argparse
    import json
    import os

    parser = argparse.ArgumentParser(
        description="Upload TRAE Sonnet 4.5 benchmark results to HuggingFace (Schema v5)"
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
    master_dataset = script_path.parent / "data" / "vllm_dataset_with_test.jsonl"

    print(f"Collecting TRAE Sonnet 4.5 results from {results_dir}...")
    print(f"Agent: {AGENT_NAME} ({AGENT_MODEL})")

    # Collect results (no baseline/human data for Sonnet 4.5)
    results = collect_agent_results(
        results_dir=str(results_dir),
        baseline_dir=None,  # No baseline data available
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
    print(f"\nWith agent metrics: {with_agent_metrics}")

    # Note: Sonnet 4.5 has no baseline/human data
    print("Note: TRAE Sonnet 4.5 results are agent-only (no baseline/human comparison)")

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
            print(f"\nSchema columns ({len(results[0])} total):")
            non_null_cols = [col for col in sorted(results[0].keys()) if results[0].get(col) is not None]
            null_cols = [col for col in sorted(results[0].keys()) if results[0].get(col) is None]
            print(f"  Non-null columns ({len(non_null_cols)}): {', '.join(non_null_cols[:10])}...")
            print(f"  Null columns ({len(null_cols)}): baseline_*, human_*, improvement_* (expected)")
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
