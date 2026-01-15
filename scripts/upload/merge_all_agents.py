#!/usr/bin/env python3
"""
Merge All Agent Results into a Single HuggingFace Dataset.

Combines results from:
- Claude Code (existing HuggingFace data)
- Codex GPT-5
- TRAE GPT-5
- TRAE Sonnet 4.5

into a unified multi-agent benchmark dataset.

Usage:
    # Dry run
    python merge_all_agents.py --dry-run

    # Save merged dataset
    python merge_all_agents.py --save-json exports/unified_agents.json

    # Upload to HuggingFace (extends existing dataset)
    python merge_all_agents.py --repo-id "Inferencebench/claude-code-vllm-benchmarks"
"""

import json
import os
from pathlib import Path
from typing import Optional

from unified_agent_upload import (
    collect_agent_results,
    enrich_with_master_dataset,
    SCHEMA_COLUMNS,
    create_empty_row,
)


def load_existing_hf_dataset(repo_id: str) -> list[dict]:
    """Load existing dataset from HuggingFace."""
    try:
        from datasets import load_dataset
        print(f"Loading existing dataset from {repo_id}...")
        ds = load_dataset(repo_id, split="train")
        results = [dict(row) for row in ds]
        print(f"  Loaded {len(results)} existing rows")
        return results
    except Exception as e:
        print(f"  Could not load existing dataset: {e}")
        return []


def collect_claude_code_results(base_dir: Path) -> list[dict]:
    """Collect Claude Code results using the existing upload script logic."""
    claude_dir = base_dir / "omniperf_results_3way_claude_code"

    if not claude_dir.exists():
        print(f"Claude Code results dir not found: {claude_dir}")
        return []

    # Import the Claude Code collector
    import sys
    sys.path.insert(0, str(claude_dir))

    try:
        from upload_to_hf import (
            collect_benchmark_results,
            collect_separate_results,
            collect_docker_results,
            merge_results,
        )

        modal_dir = claude_dir / "results" / "modal"
        docker_dir = claude_dir / "results" / "docker"
        separate_dir = claude_dir / "results"

        print(f"Collecting Claude Code results from {claude_dir}...")

        modal_results = collect_benchmark_results(
            str(modal_dir),
            agent_name="claude-code",
            agent_model="sonnet-4.5"
        )
        print(f"  Modal: {len(modal_results)}")

        separate_results = collect_separate_results(
            str(separate_dir),
            agent_name="claude-code",
            agent_model="sonnet-4.5"
        )
        print(f"  Separate: {len(separate_results)}")

        docker_results = collect_docker_results(
            str(docker_dir),
            agent_name="claude-code",
            agent_model="sonnet-4.5"
        )
        print(f"  Docker: {len(docker_results)}")

        merged = merge_results(modal_results, separate_results, docker_results)
        print(f"  Merged: {len(merged)}")

        return merged

    except ImportError as e:
        print(f"  Could not import Claude Code upload script: {e}")
        return []
    finally:
        sys.path.pop(0)


def merge_agent_datasets(
    *agent_results_lists: list[dict],
    dedupe_by: str = "commit_short",
    prefer_agent: Optional[str] = None,
) -> list[dict]:
    """
    Merge multiple agent result lists, handling duplicates.

    Strategy:
    - Each agent's results are kept separately (different agent_name + agent_model)
    - Same commit can appear multiple times (once per agent configuration)
    - If prefer_agent is set, that agent's results take priority for same commit+agent
    """
    merged = []
    seen = set()  # (commit_short, agent_name, agent_model) tuples

    for results in agent_results_lists:
        for row in results:
            commit = row.get(dedupe_by, "")[:8]
            agent = row.get("agent_name", "unknown")
            model = row.get("agent_model", "unknown")
            key = (commit, agent, model)

            if key in seen:
                # Duplicate - skip unless this is preferred agent
                if prefer_agent and agent == prefer_agent:
                    # Replace existing
                    merged = [r for r in merged if not (
                        r.get(dedupe_by, "")[:8] == commit and
                        r.get("agent_name") == agent and
                        r.get("agent_model") == model
                    )]
                    merged.append(row)
                continue

            seen.add(key)
            merged.append(row)

    return merged


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Merge all agent results into unified HuggingFace dataset"
    )
    parser.add_argument("--repo-id", default="Inferencebench/claude-code-vllm-benchmarks",
                        help="HuggingFace repo ID")
    parser.add_argument("--token", default=None,
                        help="HuggingFace token")
    parser.add_argument("--dry-run", action="store_true",
                        help="Don't upload, just show summary")
    parser.add_argument("--save-json", default=None,
                        help="Save merged results to JSON")
    parser.add_argument("--include-claude-code", action="store_true", default=True,
                        help="Include Claude Code results (default: True)")
    parser.add_argument("--skip-claude-code", action="store_true",
                        help="Skip Claude Code results (use existing HF data)")
    parser.add_argument("--enrich-metadata", action="store_true",
                        help="Enrich with master dataset metadata")

    args = parser.parse_args()

    # Base directory (repo root)
    script_dir = Path(__file__).parent
    base_dir = script_dir.parent.parent
    master_dataset = base_dir / "data" / "vllm_dataset_with_test.jsonl"

    all_results = []

    # 1. Claude Code results
    if not args.skip_claude_code:
        claude_results = collect_claude_code_results(base_dir)
        all_results.append(claude_results)
        print(f"Claude Code: {len(claude_results)} results")
    else:
        # Load from existing HuggingFace dataset
        existing = load_existing_hf_dataset(args.repo_id)
        claude_results = [r for r in existing if r.get("agent_name") == "claude-code"]
        all_results.append(claude_results)
        print(f"Claude Code (from HF): {len(claude_results)} results")

    # 2. Codex GPT-5 results
    codex_dir = base_dir / "omniperf_results_3way_codex" / "results"
    if codex_dir.exists():
        print(f"\nCollecting Codex GPT-5 results...")
        codex_results = collect_agent_results(
            results_dir=str(codex_dir),
            baseline_dir=None,
            agent_name="codex",
            agent_model="gpt-5",
            data_source="codex_modal",
        )
        all_results.append(codex_results)
        print(f"Codex GPT-5: {len(codex_results)} results")

    # 3. TRAE GPT-5 results
    trae_gpt5_dir = base_dir / "omniperf_results_3way_trae_gpt5" / "results"
    trae_gpt5_baseline = base_dir / "omniperf_results_3way_trae_gpt5" / "baseline_benchmark_results"
    if trae_gpt5_dir.exists():
        print(f"\nCollecting TRAE GPT-5 results...")
        trae_gpt5_results = collect_agent_results(
            results_dir=str(trae_gpt5_dir),
            baseline_dir=str(trae_gpt5_baseline) if trae_gpt5_baseline.exists() else None,
            agent_name="trae",
            agent_model="gpt-5",
            data_source="trae_gpt5",
        )
        all_results.append(trae_gpt5_results)
        print(f"TRAE GPT-5: {len(trae_gpt5_results)} results")

    # 4. TRAE Sonnet 4.5 results
    trae_sonnet_dir = base_dir / "omniperf_results_3way_trae_sonnet45" / "results"
    if trae_sonnet_dir.exists():
        print(f"\nCollecting TRAE Sonnet 4.5 results...")
        trae_sonnet_results = collect_agent_results(
            results_dir=str(trae_sonnet_dir),
            baseline_dir=None,
            agent_name="trae",
            agent_model="sonnet-4.5",
            data_source="trae_sonnet45",
        )
        all_results.append(trae_sonnet_results)
        print(f"TRAE Sonnet 4.5: {len(trae_sonnet_results)} results")

    # Merge all results
    print(f"\nMerging all agent results...")
    merged = merge_agent_datasets(*all_results, prefer_agent="claude-code")
    print(f"Total merged: {len(merged)} results")

    # Optionally enrich with metadata
    if args.enrich_metadata:
        merged = enrich_with_master_dataset(merged, str(master_dataset))

    # Summary by agent
    print(f"\n{'='*60}")
    print(f"Merged Dataset Summary: {len(merged)} total results")
    print(f"{'='*60}")

    agent_counts = {}
    for r in merged:
        agent = f"{r.get('agent_name', 'unknown')} ({r.get('agent_model', 'unknown')})"
        agent_counts[agent] = agent_counts.get(agent, 0) + 1

    print("\nBy agent:")
    for agent, count in sorted(agent_counts.items(), key=lambda x: -x[1]):
        print(f"  {agent}: {count}")

    # Metrics coverage
    with_metrics = sum(1 for r in merged if r.get("agent_ttft_mean") or r.get("agent_latency_avg"))
    with_baseline = sum(1 for r in merged if r.get("baseline_ttft_mean") or r.get("baseline_latency_avg"))
    print(f"\nMetrics coverage:")
    print(f"  With agent metrics: {with_metrics}")
    print(f"  With baseline metrics: {with_baseline}")

    # By benchmark mode
    mode_counts = {}
    for r in merged:
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
            json.dump(merged, f, indent=2)
        print(f"\nSaved merged results to {args.save_json}")
        return

    # Dry run
    if args.dry_run:
        print("\nDry run - not uploading to HuggingFace")
        print(f"Schema: {len(SCHEMA_COLUMNS)} columns")
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
        print("\nError: No HuggingFace token. Set HF_TOKEN or use --token")
        return

    print(f"\nUploading to HuggingFace: {args.repo_id}...")

    from datasets import Dataset
    dataset = Dataset.from_list(merged)

    dataset.info.description = """
Multi-Agent vLLM Performance Benchmark Results (Schema v5)

This dataset contains performance benchmark results from multiple AI agents
on vLLM performance optimization tasks.

## Agents Included

| Agent | Model | Description |
|-------|-------|-------------|
| claude-code | sonnet-4.5 | Anthropic Claude Code agent |
| codex | gpt-5 | OpenAI Codex agent |
| trae | gpt-5 | TRAE agent with GPT-5 |
| trae | sonnet-4.5 | TRAE agent with Claude Sonnet 4.5 |

## Benchmark Types

Each row represents a benchmark run for a specific vLLM commit, comparing:
- **Baseline**: Performance before the optimization commit
- **Human**: Performance with the human-authored optimization (ground truth)
- **Agent**: Performance with the agent's optimization attempt

## Filtering by Agent

```python
from datasets import load_dataset
ds = load_dataset("Inferencebench/claude-code-vllm-benchmarks", split="train")

# Filter by agent
claude_code = ds.filter(lambda x: x["agent_name"] == "claude-code")
codex = ds.filter(lambda x: x["agent_name"] == "codex")
trae_gpt5 = ds.filter(lambda x: x["agent_name"] == "trae" and x["agent_model"] == "gpt-5")
trae_sonnet = ds.filter(lambda x: x["agent_name"] == "trae" and x["agent_model"] == "sonnet-4.5")
```

## Metrics

### Serving Benchmarks (benchmark_mode="serving")
- TTFT: Time to First Token (ms)
- TPOT: Time per Output Token (ms)
- ITL: Inter-token Latency (ms)

### Standalone Benchmarks (benchmark_mode="standalone")
- latency_avg: Average request latency (ms)
- throughput: Tokens per second

## Schema Version
v5 - 76 columns, multi-agent support
"""

    dataset.push_to_hub(
        args.repo_id,
        token=token,
        commit_message=f"Add multi-agent benchmark results ({len(merged)} total rows)"
    )
    print(f"Successfully uploaded {len(merged)} results to {args.repo_id}")


if __name__ == "__main__":
    main()
