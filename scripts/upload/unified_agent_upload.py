#!/usr/bin/env python3
"""
Unified Agent Upload Script for OmniPerf-Bench Results.

Transforms agent benchmark results (Codex, TRAE, etc.) to the 76-column
HuggingFace schema established by Claude Code.

Schema v5 compatible with: Inferencebench/claude-code-vllm-benchmarks

Usage:
    # Codex GPT-5
    python unified_agent_upload.py \
        --results-dir ../omniperf_results_3way_codex/results \
        --agent-name codex --agent-model gpt-5 \
        --repo-id Inferencebench/codex-vllm-benchmarks \
        --dry-run

    # TRAE GPT-5 (with baseline/human data)
    python unified_agent_upload.py \
        --results-dir ../omniperf_results_3way_trae_gpt5/results \
        --baseline-dir ../omniperf_results_3way_trae_gpt5/baseline_benchmark_results \
        --agent-name trae --agent-model gpt-5 \
        --repo-id Inferencebench/trae-gpt5-vllm-benchmarks \
        --dry-run

    # TRAE Sonnet 4.5
    python unified_agent_upload.py \
        --results-dir ../omniperf_results_3way_trae_sonnet45/results \
        --agent-name trae --agent-model claude-sonnet-4-5-20250514 \
        --repo-id Inferencebench/trae-sonnet45-vllm-benchmarks \
        --dry-run
"""

import json
import os
from pathlib import Path
from typing import Optional


# =============================================================================
# Schema Definition (76 columns - matches Claude Code HuggingFace schema)
# =============================================================================

SCHEMA_COLUMNS = [
    # Identity & Metadata (18 columns)
    "commit_hash", "commit_short", "commit_subject", "repo", "perf_command",
    "files_changed", "pr_url", "models", "parent_commit", "gpu_config",
    "benchmark_mode", "agent_name", "agent_model", "benchmark_date",
    "model", "has_agent_patch", "patch_path", "data_source",

    # Baseline metrics - Serving (9 columns)
    "baseline_ttft_mean", "baseline_ttft_median", "baseline_ttft_p99",
    "baseline_tpot_mean", "baseline_tpot_median", "baseline_tpot_p99",
    "baseline_itl_mean", "baseline_itl_median", "baseline_itl_p99",

    # Baseline metrics - Standalone (2 columns)
    "baseline_latency_avg", "baseline_throughput",

    # Human metrics - Serving (9 columns)
    "human_ttft_mean", "human_ttft_median", "human_ttft_p99",
    "human_tpot_mean", "human_tpot_median", "human_tpot_p99",
    "human_itl_mean", "human_itl_median", "human_itl_p99",

    # Human metrics - Standalone (2 columns)
    "human_latency_avg", "human_throughput",

    # Agent metrics - Serving (9 columns)
    "agent_ttft_mean", "agent_ttft_median", "agent_ttft_p99",
    "agent_tpot_mean", "agent_tpot_median", "agent_tpot_p99",
    "agent_itl_mean", "agent_itl_median", "agent_itl_p99",

    # Agent metrics - Standalone (2 columns)
    "agent_latency_avg", "agent_throughput",

    # Improvement metrics - Serving mean (9 columns)
    "human_improvement_ttft_mean", "human_improvement_tpot_mean", "human_improvement_itl_mean",
    "agent_improvement_ttft_mean", "agent_improvement_tpot_mean", "agent_improvement_itl_mean",
    "agent_vs_human_ttft_mean", "agent_vs_human_tpot_mean", "agent_vs_human_itl_mean",

    # Improvement metrics - Serving median/p99 (6 columns)
    "human_improvement_ttft_median", "human_improvement_ttft_p99",
    "agent_improvement_ttft_median", "agent_improvement_ttft_p99",
    "agent_vs_human_ttft_median", "agent_vs_human_ttft_p99",

    # Improvement metrics - Standalone (6 columns)
    "human_improvement_latency_avg", "human_improvement_throughput",
    "agent_improvement_latency_avg", "agent_improvement_throughput",
    "agent_vs_human_latency_avg", "agent_vs_human_throughput",

    # Raw outputs (3 columns)
    "baseline_raw", "human_raw", "agent_raw",

    # Test script (1 column)
    "test_script",
]


def create_empty_row() -> dict:
    """Create a row with all 76 columns initialized to None."""
    return {col: None for col in SCHEMA_COLUMNS}


# =============================================================================
# Field Mapping: Agent Result Format -> HuggingFace Schema
# =============================================================================

def map_agent_metrics(metrics: dict, prefix: str = "agent") -> dict:
    """
    Map nested metrics{} object from agent results to flat HuggingFace columns.

    Agent format:
        metrics.ttft_mean_ms -> {prefix}_ttft_mean
        metrics.ttft_median_ms -> {prefix}_ttft_median
        ...
    """
    if not metrics:
        return {}

    mapping = {
        "ttft_mean_ms": f"{prefix}_ttft_mean",
        "ttft_median_ms": f"{prefix}_ttft_median",
        "ttft_p99_ms": f"{prefix}_ttft_p99",
        "tpot_mean_ms": f"{prefix}_tpot_mean",
        "tpot_median_ms": f"{prefix}_tpot_median",
        "tpot_p99_ms": f"{prefix}_tpot_p99",
        "itl_mean_ms": f"{prefix}_itl_mean",
        "itl_median_ms": f"{prefix}_itl_median",
        "itl_p99_ms": f"{prefix}_itl_p99",
        "output_token_throughput_tok_s": f"{prefix}_throughput",
        "throughput_tok_s": f"{prefix}_throughput",  # alternate key
        "latency_avg_ms": f"{prefix}_latency_avg",
    }

    result = {}
    for src_key, dst_key in mapping.items():
        if src_key in metrics and metrics[src_key] is not None:
            result[dst_key] = metrics[src_key]

    return result


def map_baseline_flat_metrics(data: dict, prefix: str = "baseline") -> dict:
    """
    Map flat baseline result format (TRAE baseline_benchmark_results/).

    Baseline format (flat fields):
        ttft_mean -> {prefix}_ttft_mean
        throughput_tok_s -> {prefix}_throughput
        ...
    """
    if not data:
        return {}

    mapping = {
        "ttft_mean": f"{prefix}_ttft_mean",
        "ttft_median": f"{prefix}_ttft_median",
        "ttft_p99": f"{prefix}_ttft_p99",
        "tpot_mean": f"{prefix}_tpot_mean",
        "tpot_median": f"{prefix}_tpot_median",
        "tpot_p99": f"{prefix}_tpot_p99",
        "itl_mean": f"{prefix}_itl_mean",
        "itl_median": f"{prefix}_itl_median",
        "itl_p99": f"{prefix}_itl_p99",
        "throughput_tok_s": f"{prefix}_throughput",
        "throughput_req_s": None,  # not mapped
    }

    result = {}
    for src_key, dst_key in mapping.items():
        if dst_key and src_key in data and data[src_key] is not None:
            result[dst_key] = data[src_key]

    return result


# =============================================================================
# Improvement Calculation
# =============================================================================

def calc_improvement(baseline_val: float, target_val: float) -> Optional[float]:
    """Calculate % improvement for latency metrics (lower is better)."""
    if baseline_val and target_val and baseline_val > 0:
        return ((baseline_val - target_val) / baseline_val) * 100
    return None


def calc_throughput_improvement(baseline_val: float, target_val: float) -> Optional[float]:
    """Calculate % improvement for throughput (higher is better)."""
    if baseline_val and target_val and baseline_val > 0:
        return ((target_val - baseline_val) / baseline_val) * 100
    return None


def calculate_improvements(row: dict) -> dict:
    """Calculate all improvement metrics from baseline/human/agent values."""
    updates = {}

    # Human vs Baseline - Serving
    for metric in ["ttft_mean", "ttft_median", "ttft_p99", "tpot_mean", "itl_mean"]:
        baseline = row.get(f"baseline_{metric}")
        human = row.get(f"human_{metric}")
        if baseline and human:
            updates[f"human_improvement_{metric}"] = calc_improvement(baseline, human)

    # Human vs Baseline - Standalone
    baseline_lat = row.get("baseline_latency_avg")
    human_lat = row.get("human_latency_avg")
    if baseline_lat and human_lat:
        updates["human_improvement_latency_avg"] = calc_improvement(baseline_lat, human_lat)

    baseline_tput = row.get("baseline_throughput")
    human_tput = row.get("human_throughput")
    if baseline_tput and human_tput:
        updates["human_improvement_throughput"] = calc_throughput_improvement(baseline_tput, human_tput)

    # Agent vs Baseline - Serving
    for metric in ["ttft_mean", "ttft_median", "ttft_p99", "tpot_mean", "itl_mean"]:
        baseline = row.get(f"baseline_{metric}")
        agent = row.get(f"agent_{metric}")
        if baseline and agent:
            updates[f"agent_improvement_{metric}"] = calc_improvement(baseline, agent)

    # Agent vs Baseline - Standalone
    agent_lat = row.get("agent_latency_avg")
    if baseline_lat and agent_lat:
        updates["agent_improvement_latency_avg"] = calc_improvement(baseline_lat, agent_lat)

    agent_tput = row.get("agent_throughput")
    if baseline_tput and agent_tput:
        updates["agent_improvement_throughput"] = calc_throughput_improvement(baseline_tput, agent_tput)

    # Agent vs Human - Serving
    for metric in ["ttft_mean", "ttft_median", "ttft_p99", "tpot_mean", "itl_mean"]:
        human = row.get(f"human_{metric}")
        agent = row.get(f"agent_{metric}")
        if human and agent:
            updates[f"agent_vs_human_{metric}"] = calc_improvement(human, agent)

    # Agent vs Human - Standalone
    if human_lat and agent_lat:
        updates["agent_vs_human_latency_avg"] = calc_improvement(human_lat, agent_lat)
    if human_tput and agent_tput:
        updates["agent_vs_human_throughput"] = calc_throughput_improvement(human_tput, agent_tput)

    return updates


# =============================================================================
# Result Collection
# =============================================================================

def collect_agent_results(
    results_dir: str,
    baseline_dir: Optional[str],
    agent_name: str,
    agent_model: str,
    data_source: str,
) -> list[dict]:
    """
    Collect all agent results from a directory.

    Args:
        results_dir: Path to directory with *_agent_result.json files
        baseline_dir: Optional path to baseline results directory
        agent_name: Name of agent (e.g., "codex", "trae")
        agent_model: Model used (e.g., "gpt-5", "claude-sonnet-4-5-20250514")
        data_source: Source identifier (e.g., "codex_modal", "trae_gpt5")

    Returns:
        List of rows conforming to 76-column schema
    """
    results = []
    results_path = Path(results_dir)
    baseline_path = Path(baseline_dir) if baseline_dir else None

    # Collect all agent result files
    agent_files = list(results_path.glob("*_agent_result.json"))
    print(f"Found {len(agent_files)} agent result files")

    for agent_file in agent_files:
        try:
            # Extract commit from filename
            commit_short = agent_file.stem.replace("_agent_result", "")

            with open(agent_file) as f:
                agent_data = json.load(f)

            # Skip if status is not success and no metrics
            status = agent_data.get("status", "unknown")
            metrics = agent_data.get("metrics", {})

            # Create base row with all columns
            row = create_empty_row()

            # Map identity fields
            row["commit_hash"] = agent_data.get("human_commit_full") or commit_short
            row["commit_short"] = agent_data.get("human_commit") or commit_short[:8]
            row["parent_commit"] = agent_data.get("parent_commit")
            row["repo"] = "vllm-project/vllm"
            row["model"] = agent_data.get("model")
            row["agent_name"] = agent_name
            row["agent_model"] = agent_model
            row["data_source"] = data_source

            # Extract benchmark date from timestamp
            timestamp = agent_data.get("timestamp", "")
            if timestamp:
                row["benchmark_date"] = timestamp[:10]

            # Determine benchmark mode from metrics
            if metrics:
                if "ttft_mean_ms" in metrics:
                    row["benchmark_mode"] = "serving"
                elif "latency_avg_ms" in metrics:
                    row["benchmark_mode"] = "standalone"

            # Map agent metrics
            if status == "success" and metrics:
                row["has_agent_patch"] = True
                row.update(map_agent_metrics(metrics, "agent"))
                row["agent_raw"] = agent_data.get("raw_output")
            else:
                row["has_agent_patch"] = False

            # Try to load human result (same commit)
            human_file = results_path / f"{commit_short}_human_result.json"
            if human_file.exists():
                with open(human_file) as f:
                    human_data = json.load(f)
                if human_data.get("status") == "success" and human_data.get("metrics"):
                    row.update(map_agent_metrics(human_data["metrics"], "human"))
                    row["human_raw"] = human_data.get("raw_output")

            # Try to load baseline result
            if baseline_path:
                baseline_file = baseline_path / f"{commit_short}_baseline_result.json"
                if baseline_file.exists():
                    with open(baseline_file) as f:
                        baseline_data = json.load(f)
                    if baseline_data.get("status") == "success":
                        # Baseline uses flat format, not nested metrics
                        row.update(map_baseline_flat_metrics(baseline_data, "baseline"))
                        row["baseline_raw"] = baseline_data.get("raw_output")

            # Calculate improvement metrics
            row.update(calculate_improvements(row))

            results.append(row)

        except Exception as e:
            print(f"Error processing {agent_file}: {e}")
            continue

    return results


def enrich_with_master_dataset(
    results: list[dict],
    master_dataset_path: str,
) -> list[dict]:
    """
    Enrich results with metadata from the master dataset.

    Master dataset fields:
        - head_commit -> match with commit_hash
        - gt_commit_message -> commit_subject
        - patch -> can extract files_changed
        - repo
    """
    if not os.path.exists(master_dataset_path):
        print(f"Master dataset not found: {master_dataset_path}")
        return results

    # Load master dataset
    master_data = {}
    with open(master_dataset_path) as f:
        for line in f:
            try:
                entry = json.loads(line)
                commit = entry.get("head_commit", "")
                if commit:
                    master_data[commit] = entry
                    # Also index by short hash
                    master_data[commit[:8]] = entry
            except Exception:
                continue

    print(f"Loaded {len(master_data) // 2} entries from master dataset")

    # Enrich each result
    enriched_count = 0
    for row in results:
        commit = row.get("commit_hash", "")
        commit_short = row.get("commit_short", "")

        master_entry = master_data.get(commit) or master_data.get(commit_short)
        if not master_entry:
            continue

        enriched_count += 1

        # Extract commit subject from message
        commit_msg = master_entry.get("gt_commit_message", "")
        if commit_msg and not row.get("commit_subject"):
            # First line of commit message
            row["commit_subject"] = commit_msg.split("\n")[0][:200]

        # Extract files changed from patch
        patch = master_entry.get("patch", "")
        if patch and not row.get("files_changed"):
            files = []
            for line in patch.split("\n"):
                if line.startswith("+++ b/") or line.startswith("--- a/"):
                    path = line.split()[-1]
                    if path.startswith("a/") or path.startswith("b/"):
                        path = path[2:]
                    if path and path not in files:
                        files.append(path)
            row["files_changed"] = files

        # Try to extract test script
        eff_test = master_entry.get("efficiency_test", [])
        if eff_test and not row.get("test_script"):
            if isinstance(eff_test, list) and len(eff_test) > 0:
                row["test_script"] = eff_test[0] if isinstance(eff_test[0], str) else None

    print(f"Enriched {enriched_count} results with master dataset metadata")
    return results


# =============================================================================
# Upload to HuggingFace
# =============================================================================

def upload_to_huggingface(
    results: list[dict],
    repo_id: str,
    agent_name: str,
    agent_model: str,
    token: Optional[str] = None,
):
    """Upload results to HuggingFace as a dataset."""
    from datasets import Dataset

    # Create dataset
    dataset = Dataset.from_list(results)

    # Add metadata
    dataset.info.description = f"""
{agent_name.upper()} vLLM Performance Benchmark Results (Schema v5)

This dataset contains performance benchmark results from running {agent_name}
agent ({agent_model}) on vLLM performance optimization tasks.

## Benchmark Types

Each row represents a benchmark run for a specific vLLM commit, comparing:
- **Baseline**: Performance before the optimization commit (when available)
- **Human**: Performance with the human-authored optimization (when available)
- **Agent**: Performance with {agent_name}'s optimization attempt

## Metrics

### Serving Benchmarks (benchmark_mode="serving")
- TTFT: Time to First Token (ms) - latency until first token generation
- TPOT: Time per Output Token (ms) - average time per generated token
- ITL: Inter-token Latency (ms) - time between consecutive tokens
- Each metric has mean, median, and p99 variants

### Standalone Benchmarks (benchmark_mode="standalone")
- latency_avg: Average request latency (ms)
- throughput: Tokens per second

## Improvement Metrics
- human_improvement_*: Percentage improvement of human patch over baseline
- agent_improvement_*: Percentage improvement of agent patch over baseline
- agent_vs_human_*: How agent compares to human (positive = agent better)

Note: Improvement columns are null when baseline/human data is unavailable.

## Schema Version
v5 - 76 columns, compatible with Inferencebench/claude-code-vllm-benchmarks
"""

    # Push to hub
    dataset.push_to_hub(
        repo_id,
        token=token,
        commit_message=f"Upload {agent_name} vLLM benchmark results ({len(results)} commits) - Schema v5"
    )

    print(f"Successfully uploaded {len(results)} benchmark results to {repo_id}")
    return dataset


# =============================================================================
# Main
# =============================================================================

def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Upload agent benchmark results to HuggingFace (Schema v5)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Codex GPT-5 (agent-only, no baseline/human)
  python unified_agent_upload.py \\
      --results-dir ../omniperf_results_3way_codex/results \\
      --agent-name codex --agent-model gpt-5 \\
      --data-source codex_modal \\
      --dry-run

  # TRAE GPT-5 (with baseline data)
  python unified_agent_upload.py \\
      --results-dir ../omniperf_results_3way_trae_gpt5/results \\
      --baseline-dir ../omniperf_results_3way_trae_gpt5/baseline_benchmark_results \\
      --agent-name trae --agent-model gpt-5 \\
      --data-source trae_gpt5 \\
      --enrich-metadata \\
      --dry-run
"""
    )

    parser.add_argument("--results-dir", required=True,
                        help="Directory containing *_agent_result.json files")
    parser.add_argument("--baseline-dir", default=None,
                        help="Optional directory containing baseline results")
    parser.add_argument("--agent-name", required=True,
                        help="Name of the agent (e.g., codex, trae)")
    parser.add_argument("--agent-model", required=True,
                        help="Model used by the agent (e.g., gpt-5)")
    parser.add_argument("--data-source", default=None,
                        help="Data source identifier (default: {agent_name}_modal)")
    parser.add_argument("--repo-id", default=None,
                        help="HuggingFace repo ID for upload")
    parser.add_argument("--token", default=None,
                        help="HuggingFace token (default: use HF_TOKEN env var)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Collect results but don't upload")
    parser.add_argument("--save-json", default=None,
                        help="Save results to JSON file")
    parser.add_argument("--enrich-metadata", action="store_true",
                        help="Enrich results with master dataset metadata")
    parser.add_argument("--master-dataset", default="data/vllm_dataset_with_test.jsonl",
                        help="Path to master dataset for enrichment")

    args = parser.parse_args()

    # Set defaults
    data_source = args.data_source or f"{args.agent_name}_modal"

    # Collect results
    print(f"Collecting results from {args.results_dir}...")
    print(f"Agent: {args.agent_name} ({args.agent_model})")

    results = collect_agent_results(
        results_dir=args.results_dir,
        baseline_dir=args.baseline_dir,
        agent_name=args.agent_name,
        agent_model=args.agent_model,
        data_source=data_source,
    )

    print(f"Collected {len(results)} results")

    # Enrich with master dataset
    if args.enrich_metadata:
        # Find master dataset relative to script or current dir
        master_path = args.master_dataset
        if not os.path.exists(master_path):
            # Try relative to script location
            script_dir = Path(__file__).parent
            alt_path = script_dir.parent.parent / master_path
            if alt_path.exists():
                master_path = str(alt_path)

        results = enrich_with_master_dataset(results, master_path)

    # Print summary
    print(f"\n{'='*60}")
    print(f"Summary: {len(results)} benchmark results")
    print(f"{'='*60}")

    # Count by status
    with_agent_metrics = sum(1 for r in results if r.get("agent_ttft_mean") or r.get("agent_latency_avg"))
    with_human_metrics = sum(1 for r in results if r.get("human_ttft_mean") or r.get("human_latency_avg"))
    with_baseline_metrics = sum(1 for r in results if r.get("baseline_ttft_mean") or r.get("baseline_latency_avg"))

    print(f"\nMetrics coverage:")
    print(f"  With agent metrics: {with_agent_metrics}")
    print(f"  With human metrics: {with_human_metrics}")
    print(f"  With baseline metrics: {with_baseline_metrics}")

    # Count by benchmark mode
    mode_counts = {}
    for r in results:
        mode = r.get("benchmark_mode") or "unknown"
        mode_counts[mode] = mode_counts.get(mode, 0) + 1
    print(f"\nBy benchmark mode:")
    for mode, count in sorted(mode_counts.items(), key=lambda x: -x[1]):
        print(f"  {mode}: {count}")

    # Date range
    dates = [r.get("benchmark_date") for r in results if r.get("benchmark_date")]
    if dates:
        print(f"\nDate range: {min(dates)} to {max(dates)}")

    # Save to JSON
    if args.save_json:
        with open(args.save_json, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nSaved results to {args.save_json}")
        return

    # Dry run
    if args.dry_run:
        print("\nDry run - not uploading to HuggingFace")
        if results:
            print(f"\nSchema columns ({len(results[0])} total):")
            for col in sorted(results[0].keys()):
                val = results[0].get(col)
                val_type = type(val).__name__ if val is not None else "None"
                print(f"  - {col}: {val_type}")
        return

    # Upload
    if not args.repo_id:
        print("\nError: --repo-id required for upload")
        return

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
    upload_to_huggingface(
        results,
        args.repo_id,
        args.agent_name,
        args.agent_model,
        token,
    )


if __name__ == "__main__":
    main()
