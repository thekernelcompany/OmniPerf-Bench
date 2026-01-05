#!/usr/bin/env python3
"""
Push SGLang benchmark results to HuggingFace.

Parses results from benchmark_results/sglang/{commit}/{timestamp}/ structure.
Merges results from multiple runs per commit (human_only, baseline_only, agent_only).

Similar format to: https://huggingface.co/datasets/Inferencebench/claude-code-vllm-benchmarks
"""

import json
import os
from pathlib import Path
from datetime import date
from collections import defaultdict
from typing import Dict, List, Any, Optional
from datasets import Dataset
from huggingface_hub import HfApi

# Dataset README template
DATASET_README = '''---
license: apache-2.0
task_categories:
  - text-generation
language:
  - en
tags:
  - benchmark
  - performance
  - llm-serving
  - sglang
  - inference
  - claude-code
pretty_name: Claude Code SGLang Performance Benchmarks
size_categories:
  - n<1K
---

# Claude Code SGLang Performance Benchmarks

This dataset contains performance benchmark results comparing human-authored performance optimizations against Claude Code agent-generated patches for the [SGLang](https://github.com/sgl-project/sglang) LLM serving framework.

## Dataset Description

Each row represents a performance-related commit from the SGLang repository, with benchmark metrics for:

1. **Baseline**: Performance before the optimization (parent commit)
2. **Human**: Performance after applying the human-authored PR patch
3. **Agent**: Performance after applying the Claude Code agent-generated patch

### Key Metrics

| Metric | Description | Unit |
|--------|-------------|------|
| `ttft_mean` | Time to First Token (mean) | seconds |
| `tpot_mean` | Time per Output Token (mean) | seconds |
| `itl_mean` | Inter-Token Latency (mean) | seconds |
| `e2e_latency_mean` | End-to-End Latency (mean) | seconds |
| `request_throughput` | Requests per second | req/s |
| `output_throughput` | Output tokens per second | tokens/s |
| `input_throughput` | Input tokens per second | tokens/s |

### Improvement Calculations

- **Latency metrics**: `improvement = (baseline - comparison) / baseline * 100` (positive = faster)
- **Throughput metrics**: `improvement = (comparison - baseline) / baseline * 100` (positive = higher throughput)

## Dataset Structure

### All 75 Columns

#### Identity (5 columns)
| Column | Description |
|--------|-------------|
| `commit_hash` | Full 40-character commit hash |
| `commit_short` | Short 8-character commit hash |
| `commit_subject` | Commit message subject line |
| `pr_url` | Link to the GitHub pull request |
| `repo` | Repository name (sgl-project/sglang) |

#### Configuration (4 columns)
| Column | Description |
|--------|-------------|
| `model` | Model used for benchmarking (e.g., meta-llama/Llama-3.1-8B-Instruct) |
| `gpu_config` | GPU configuration (H100:1) |
| `perf_command` | Benchmark command executed |
| `benchmark_mode` | Status: complete, partial, or human_only |

#### Status Flags (5 columns)
| Column | Description |
|--------|-------------|
| `status` | Overall benchmark status |
| `has_baseline` | Whether baseline metrics are available |
| `has_human` | Whether human patch metrics are available |
| `has_agent` | Whether agent patch metrics are available |
| `has_agent_patch` | Whether an agent patch was generated |

#### Baseline Metrics (13 columns)
| Column | Description |
|--------|-------------|
| `baseline_ttft_mean` | Time to First Token - mean (seconds) |
| `baseline_ttft_median` | Time to First Token - median (seconds) |
| `baseline_ttft_p99` | Time to First Token - 99th percentile (seconds) |
| `baseline_tpot_mean` | Time per Output Token - mean (seconds) |
| `baseline_tpot_median` | Time per Output Token - median (seconds) |
| `baseline_tpot_p99` | Time per Output Token - 99th percentile (seconds) |
| `baseline_itl_mean` | Inter-Token Latency - mean (seconds) |
| `baseline_itl_median` | Inter-Token Latency - median (seconds) |
| `baseline_itl_p99` | Inter-Token Latency - 99th percentile (seconds) |
| `baseline_request_throughput` | Requests per second |
| `baseline_output_throughput` | Output tokens per second |
| `baseline_input_throughput` | Input tokens per second |
| `baseline_e2e_latency_mean` | End-to-end latency - mean (seconds) |

#### Human Metrics (13 columns)
| Column | Description |
|--------|-------------|
| `human_ttft_mean` | Time to First Token - mean (seconds) |
| `human_ttft_median` | Time to First Token - median (seconds) |
| `human_ttft_p99` | Time to First Token - 99th percentile (seconds) |
| `human_tpot_mean` | Time per Output Token - mean (seconds) |
| `human_tpot_median` | Time per Output Token - median (seconds) |
| `human_tpot_p99` | Time per Output Token - 99th percentile (seconds) |
| `human_itl_mean` | Inter-Token Latency - mean (seconds) |
| `human_itl_median` | Inter-Token Latency - median (seconds) |
| `human_itl_p99` | Inter-Token Latency - 99th percentile (seconds) |
| `human_request_throughput` | Requests per second |
| `human_output_throughput` | Output tokens per second |
| `human_input_throughput` | Input tokens per second |
| `human_e2e_latency_mean` | End-to-end latency - mean (seconds) |

#### Agent Metrics (13 columns)
| Column | Description |
|--------|-------------|
| `agent_ttft_mean` | Time to First Token - mean (seconds) |
| `agent_ttft_median` | Time to First Token - median (seconds) |
| `agent_ttft_p99` | Time to First Token - 99th percentile (seconds) |
| `agent_tpot_mean` | Time per Output Token - mean (seconds) |
| `agent_tpot_median` | Time per Output Token - median (seconds) |
| `agent_tpot_p99` | Time per Output Token - 99th percentile (seconds) |
| `agent_itl_mean` | Inter-Token Latency - mean (seconds) |
| `agent_itl_median` | Inter-Token Latency - median (seconds) |
| `agent_itl_p99` | Inter-Token Latency - 99th percentile (seconds) |
| `agent_request_throughput` | Requests per second |
| `agent_output_throughput` | Output tokens per second |
| `agent_input_throughput` | Input tokens per second |
| `agent_e2e_latency_mean` | End-to-end latency - mean (seconds) |

#### Human vs Baseline Improvements (6 columns)
Percentage improvement (positive = human is better)
| Column | Description |
|--------|-------------|
| `human_improvement_ttft_mean` | TTFT improvement % |
| `human_improvement_tpot_mean` | TPOT improvement % |
| `human_improvement_itl_mean` | ITL improvement % |
| `human_improvement_e2e_latency_mean` | E2E latency improvement % |
| `human_improvement_request_throughput` | Request throughput improvement % |
| `human_improvement_output_throughput` | Output throughput improvement % |

#### Agent vs Baseline Improvements (6 columns)
Percentage improvement (positive = agent is better)
| Column | Description |
|--------|-------------|
| `agent_improvement_ttft_mean` | TTFT improvement % |
| `agent_improvement_tpot_mean` | TPOT improvement % |
| `agent_improvement_itl_mean` | ITL improvement % |
| `agent_improvement_e2e_latency_mean` | E2E latency improvement % |
| `agent_improvement_request_throughput` | Request throughput improvement % |
| `agent_improvement_output_throughput` | Output throughput improvement % |

#### Agent vs Human Improvements (6 columns)
Percentage difference (positive = agent outperforms human)
| Column | Description |
|--------|-------------|
| `agent_vs_human_ttft_mean` | TTFT difference % |
| `agent_vs_human_tpot_mean` | TPOT difference % |
| `agent_vs_human_itl_mean` | ITL difference % |
| `agent_vs_human_e2e_latency_mean` | E2E latency difference % |
| `agent_vs_human_request_throughput` | Request throughput difference % |
| `agent_vs_human_output_throughput` | Output throughput difference % |

#### Metadata (4 columns)
| Column | Description |
|--------|-------------|
| `agent_name` | Agent used (claude-code) |
| `agent_model` | Model powering the agent (claude-sonnet-4-20250514) |
| `benchmark_date` | Date benchmarks were run |
| `parent_commit` | Base commit for patches |

## Benchmark Setup

- **Hardware**: NVIDIA H100 GPU (1x)
- **Platform**: [Modal](https://modal.com) cloud infrastructure
- **Benchmark Tool**: `sglang.bench_serving`
- **Workload**: 100 prompts per benchmark run

## Usage

```python
from datasets import load_dataset

# Load the dataset
ds = load_dataset("Inferencebench/claude-code-sglang-benchmarks_v1")

# Filter for complete benchmarks (all 3 phases)
complete = ds.filter(lambda x: x["status"] == "complete")

# Analyze human improvements
for row in complete["train"]:
    print(f"{row['commit_short']}: Human TTFT improvement: {row['human_improvement_ttft_mean']:.1f}%")
```

## Known Limitations

Some commits could not be benchmarked due to:
- **Broken Docker images**: Early SGLang commits missing dependencies (torch, etc.)
- **Wrong script paths**: Commits referencing non-existent benchmark scripts
- **Private models**: Commits requiring access to private model weights
- **Hardware requirements**: Commits requiring multi-GPU setups (TP8, etc.)

## Citation

If you use this dataset, please cite:

```bibtex
@dataset{claude_code_sglang_benchmarks,
  title={Claude Code SGLang Performance Benchmarks},
  author={Inferencebench},
  year={2026},
  url={https://huggingface.co/datasets/Inferencebench/claude-code-sglang-benchmarks_v1}
}
```

## License

Apache 2.0
'''

# Configuration
RESULTS_DIR = Path("benchmark_results/sglang")
HF_REPO = "Inferencebench/claude-code-sglang-benchmarks_v1"
AGENT_NAME = "claude-code"
AGENT_MODEL = "claude-sonnet-4-20250514"
REPO_NAME = "sgl-project/sglang"


def calculate_improvement(baseline_val: float, comparison_val: float) -> Optional[float]:
    """Calculate percentage improvement (positive = comparison is better/lower).

    For latency metrics: lower is better, so improvement = (baseline - comparison) / baseline * 100
    For throughput metrics: higher is better, so improvement = (comparison - baseline) / baseline * 100
    """
    if baseline_val is None or comparison_val is None or baseline_val == 0:
        return None
    return round((baseline_val - comparison_val) / baseline_val * 100, 2)


def calculate_throughput_improvement(baseline_val: float, comparison_val: float) -> Optional[float]:
    """Calculate percentage improvement for throughput (higher is better)."""
    if baseline_val is None or comparison_val is None or baseline_val == 0:
        return None
    return round((comparison_val - baseline_val) / baseline_val * 100, 2)


def load_all_runs() -> Dict[str, List[Dict]]:
    """Load all benchmark runs, grouped by commit."""

    runs_by_commit = defaultdict(list)

    if not RESULTS_DIR.exists():
        print(f"Results directory not found: {RESULTS_DIR}")
        return runs_by_commit

    # Walk through all commit directories
    for commit_dir in RESULTS_DIR.iterdir():
        if not commit_dir.is_dir():
            continue

        commit_short = commit_dir.name

        # Walk through all timestamp directories for this commit
        for timestamp_dir in commit_dir.iterdir():
            if not timestamp_dir.is_dir():
                continue

            metrics_file = timestamp_dir / "metrics.json"
            metadata_file = timestamp_dir / "metadata.json"

            if not metrics_file.exists():
                continue

            try:
                with open(metrics_file) as f:
                    metrics = json.load(f)

                metadata = {}
                if metadata_file.exists():
                    with open(metadata_file) as f:
                        metadata = json.load(f)

                run = {
                    "commit_short": commit_short,
                    "timestamp": timestamp_dir.name,
                    "metrics": metrics,
                    "metadata": metadata,
                }

                runs_by_commit[commit_short].append(run)

            except Exception as e:
                print(f"Error loading {timestamp_dir}: {e}")

    return runs_by_commit


def get_best_run_by_mode(runs: List[Dict]) -> Dict[str, Optional[Dict]]:
    """Get the latest successful run for each benchmark mode."""

    # Sort runs by timestamp (newest first)
    sorted_runs = sorted(runs, key=lambda r: r["timestamp"], reverse=True)

    best_by_mode = {
        "baseline": None,
        "human": None,
        "agent": None,
    }

    for run in sorted_runs:
        metrics = run["metrics"]
        status = metrics.get("status", "error")
        mode = metrics.get("benchmark_mode", "")

        if status != "success":
            continue

        # Baseline metrics (can come from baseline_only or parallel_3way)
        if best_by_mode["baseline"] is None:
            baseline_metrics = metrics.get("baseline_metrics")
            if baseline_metrics and len(baseline_metrics) > 0:
                best_by_mode["baseline"] = run

        # Human metrics (can come from human_only or parallel_3way)
        if best_by_mode["human"] is None:
            human_metrics = metrics.get("human_metrics")
            if human_metrics and len(human_metrics) > 0:
                best_by_mode["human"] = run

        # Agent metrics (can come from agent_only or parallel_3way)
        if best_by_mode["agent"] is None:
            agent_metrics = metrics.get("agent_metrics")
            if agent_metrics and len(agent_metrics) > 0:
                best_by_mode["agent"] = run

    return best_by_mode


def merge_runs_to_row(commit_short: str, runs: List[Dict]) -> Optional[Dict]:
    """Merge multiple runs for a commit into a single dataset row."""

    best_runs = get_best_run_by_mode(runs)

    # Need at least one successful run
    if not any(best_runs.values()):
        return None

    # Get metadata from the first available run
    metadata = {}
    for run in [best_runs["human"], best_runs["baseline"], best_runs["agent"]]:
        if run and run.get("metadata"):
            metadata = run["metadata"]
            break

    # Extract commit info
    dataset_row = metadata.get("dataset_row", {})
    agent_patch_info = metadata.get("agent_patch_info", {})

    # Get metrics from each phase
    baseline_metrics = {}
    human_metrics = {}
    agent_metrics = {}

    if best_runs["baseline"]:
        baseline_metrics = best_runs["baseline"]["metrics"].get("baseline_metrics", {})

    if best_runs["human"]:
        human_metrics = best_runs["human"]["metrics"].get("human_metrics", {})

    if best_runs["agent"]:
        agent_metrics = best_runs["agent"]["metrics"].get("agent_metrics", {})

    # Calculate improvements
    human_improvement = {}
    agent_improvement = {}
    agent_vs_human = {}

    if baseline_metrics and human_metrics:
        # Latency metrics (lower is better)
        for key in ["ttft_mean", "tpot_mean", "itl_mean", "e2e_latency_mean"]:
            human_improvement[key] = calculate_improvement(
                baseline_metrics.get(key), human_metrics.get(key)
            )
        # Throughput metrics (higher is better)
        for key in ["request_throughput", "output_throughput", "input_throughput"]:
            human_improvement[key] = calculate_throughput_improvement(
                baseline_metrics.get(key), human_metrics.get(key)
            )

    if baseline_metrics and agent_metrics:
        # Latency metrics
        for key in ["ttft_mean", "tpot_mean", "itl_mean", "e2e_latency_mean"]:
            agent_improvement[key] = calculate_improvement(
                baseline_metrics.get(key), agent_metrics.get(key)
            )
        # Throughput metrics
        for key in ["request_throughput", "output_throughput", "input_throughput"]:
            agent_improvement[key] = calculate_throughput_improvement(
                baseline_metrics.get(key), agent_metrics.get(key)
            )

    if human_metrics and agent_metrics:
        # Latency metrics
        for key in ["ttft_mean", "tpot_mean", "itl_mean", "e2e_latency_mean"]:
            agent_vs_human[key] = calculate_improvement(
                human_metrics.get(key), agent_metrics.get(key)
            )
        # Throughput metrics
        for key in ["request_throughput", "output_throughput", "input_throughput"]:
            agent_vs_human[key] = calculate_throughput_improvement(
                human_metrics.get(key), agent_metrics.get(key)
            )

    # Determine overall status
    has_baseline = bool(baseline_metrics)
    has_human = bool(human_metrics)
    has_agent = bool(agent_metrics)

    status = "partial"
    if has_baseline and has_human and has_agent:
        status = "complete"
    elif has_human:
        status = "human_only"

    # Get model from dataset row
    models = dataset_row.get("models", [])
    model = models[0] if models else ""

    # Build the row
    row = {
        # Identity
        "commit_hash": metadata.get("commit", dataset_row.get("commit_hash", "")),
        "commit_short": commit_short,
        "commit_subject": dataset_row.get("commit_subject", ""),
        "pr_url": dataset_row.get("pr_url", ""),
        "repo": REPO_NAME,

        # Configuration
        "model": model,
        "gpu_config": "H100:1",
        "benchmark_mode": status,
        "perf_command": dataset_row.get("perf_command", ""),

        # Status
        "status": status,
        "has_baseline": has_baseline,
        "has_human": has_human,
        "has_agent": has_agent,
        "has_agent_patch": agent_patch_info.get("found", False),

        # Baseline metrics
        "baseline_ttft_mean": baseline_metrics.get("ttft_mean"),
        "baseline_ttft_median": baseline_metrics.get("ttft_median"),
        "baseline_ttft_p99": baseline_metrics.get("ttft_p99"),
        "baseline_tpot_mean": baseline_metrics.get("tpot_mean"),
        "baseline_tpot_median": baseline_metrics.get("tpot_median"),
        "baseline_tpot_p99": baseline_metrics.get("tpot_p99"),
        "baseline_itl_mean": baseline_metrics.get("itl_mean"),
        "baseline_itl_median": baseline_metrics.get("itl_median"),
        "baseline_itl_p99": baseline_metrics.get("itl_p99"),
        "baseline_request_throughput": baseline_metrics.get("request_throughput"),
        "baseline_output_throughput": baseline_metrics.get("output_throughput"),
        "baseline_input_throughput": baseline_metrics.get("input_throughput"),
        "baseline_e2e_latency_mean": baseline_metrics.get("e2e_latency_mean"),

        # Human metrics
        "human_ttft_mean": human_metrics.get("ttft_mean"),
        "human_ttft_median": human_metrics.get("ttft_median"),
        "human_ttft_p99": human_metrics.get("ttft_p99"),
        "human_tpot_mean": human_metrics.get("tpot_mean"),
        "human_tpot_median": human_metrics.get("tpot_median"),
        "human_tpot_p99": human_metrics.get("tpot_p99"),
        "human_itl_mean": human_metrics.get("itl_mean"),
        "human_itl_median": human_metrics.get("itl_median"),
        "human_itl_p99": human_metrics.get("itl_p99"),
        "human_request_throughput": human_metrics.get("request_throughput"),
        "human_output_throughput": human_metrics.get("output_throughput"),
        "human_input_throughput": human_metrics.get("input_throughput"),
        "human_e2e_latency_mean": human_metrics.get("e2e_latency_mean"),

        # Agent metrics
        "agent_ttft_mean": agent_metrics.get("ttft_mean") if agent_metrics else None,
        "agent_ttft_median": agent_metrics.get("ttft_median") if agent_metrics else None,
        "agent_ttft_p99": agent_metrics.get("ttft_p99") if agent_metrics else None,
        "agent_tpot_mean": agent_metrics.get("tpot_mean") if agent_metrics else None,
        "agent_tpot_median": agent_metrics.get("tpot_median") if agent_metrics else None,
        "agent_tpot_p99": agent_metrics.get("tpot_p99") if agent_metrics else None,
        "agent_itl_mean": agent_metrics.get("itl_mean") if agent_metrics else None,
        "agent_itl_median": agent_metrics.get("itl_median") if agent_metrics else None,
        "agent_itl_p99": agent_metrics.get("itl_p99") if agent_metrics else None,
        "agent_request_throughput": agent_metrics.get("request_throughput") if agent_metrics else None,
        "agent_output_throughput": agent_metrics.get("output_throughput") if agent_metrics else None,
        "agent_input_throughput": agent_metrics.get("input_throughput") if agent_metrics else None,
        "agent_e2e_latency_mean": agent_metrics.get("e2e_latency_mean") if agent_metrics else None,

        # Human vs Baseline improvements
        "human_improvement_ttft_mean": human_improvement.get("ttft_mean"),
        "human_improvement_tpot_mean": human_improvement.get("tpot_mean"),
        "human_improvement_itl_mean": human_improvement.get("itl_mean"),
        "human_improvement_e2e_latency_mean": human_improvement.get("e2e_latency_mean"),
        "human_improvement_request_throughput": human_improvement.get("request_throughput"),
        "human_improvement_output_throughput": human_improvement.get("output_throughput"),

        # Agent vs Baseline improvements
        "agent_improvement_ttft_mean": agent_improvement.get("ttft_mean"),
        "agent_improvement_tpot_mean": agent_improvement.get("tpot_mean"),
        "agent_improvement_itl_mean": agent_improvement.get("itl_mean"),
        "agent_improvement_e2e_latency_mean": agent_improvement.get("e2e_latency_mean"),
        "agent_improvement_request_throughput": agent_improvement.get("request_throughput"),
        "agent_improvement_output_throughput": agent_improvement.get("output_throughput"),

        # Agent vs Human improvements
        "agent_vs_human_ttft_mean": agent_vs_human.get("ttft_mean"),
        "agent_vs_human_tpot_mean": agent_vs_human.get("tpot_mean"),
        "agent_vs_human_itl_mean": agent_vs_human.get("itl_mean"),
        "agent_vs_human_e2e_latency_mean": agent_vs_human.get("e2e_latency_mean"),
        "agent_vs_human_request_throughput": agent_vs_human.get("request_throughput"),
        "agent_vs_human_output_throughput": agent_vs_human.get("output_throughput"),

        # Meta
        "agent_name": AGENT_NAME,
        "agent_model": AGENT_MODEL,
        "benchmark_date": str(date.today()),
        "parent_commit": agent_patch_info.get("parent_commit", ""),
    }

    return row


def load_results() -> List[Dict]:
    """Load all benchmark results, merging by commit."""

    print(f"Loading results from {RESULTS_DIR}...")

    runs_by_commit = load_all_runs()
    print(f"Found {len(runs_by_commit)} commits with benchmark runs")

    results = []
    for commit_short, runs in runs_by_commit.items():
        row = merge_runs_to_row(commit_short, runs)
        if row:
            results.append(row)
            print(f"  {commit_short}: baseline={row['has_baseline']}, human={row['has_human']}, agent={row['has_agent']}")

    return results


def main():
    print("=" * 60)
    print("SGLang Benchmark Results -> HuggingFace")
    print("=" * 60)

    results = load_results()
    print(f"\nTotal rows: {len(results)}")

    if not results:
        print("No results to upload!")
        return

    # Statistics
    complete = sum(1 for r in results if r["status"] == "complete")
    has_baseline = sum(1 for r in results if r["has_baseline"])
    has_human = sum(1 for r in results if r["has_human"])
    has_agent = sum(1 for r in results if r["has_agent"])

    print(f"\nStatistics:")
    print(f"  Complete (baseline + human + agent): {complete}")
    print(f"  Has baseline: {has_baseline}")
    print(f"  Has human: {has_human}")
    print(f"  Has agent: {has_agent}")

    # Show sample improvements for complete rows
    print(f"\nSample improvements (complete rows):")
    for r in results:
        if r["status"] == "complete":
            print(f"\n  {r['commit_short']}: {r['commit_subject'][:50]}...")
            print(f"    Human vs Baseline:")
            print(f"      TTFT: {r['human_improvement_ttft_mean']:.1f}%" if r['human_improvement_ttft_mean'] else "      TTFT: N/A")
            print(f"      TPOT: {r['human_improvement_tpot_mean']:.1f}%" if r['human_improvement_tpot_mean'] else "      TPOT: N/A")
            print(f"      Throughput: {r['human_improvement_request_throughput']:.1f}%" if r['human_improvement_request_throughput'] else "      Throughput: N/A")
            print(f"    Agent vs Baseline:")
            print(f"      TTFT: {r['agent_improvement_ttft_mean']:.1f}%" if r['agent_improvement_ttft_mean'] else "      TTFT: N/A")
            print(f"      TPOT: {r['agent_improvement_tpot_mean']:.1f}%" if r['agent_improvement_tpot_mean'] else "      TPOT: N/A")
            print(f"      Throughput: {r['agent_improvement_request_throughput']:.1f}%" if r['agent_improvement_request_throughput'] else "      Throughput: N/A")

    # Create dataset
    print("\n" + "=" * 60)
    print("Creating HuggingFace dataset...")
    dataset = Dataset.from_list(results)
    print(f"Dataset: {dataset}")
    print(f"Features: {list(dataset.features.keys())[:10]}... ({len(dataset.features)} total)")

    # Push to HuggingFace
    print(f"\nPushing to {HF_REPO}...")
    token = os.environ.get("HF_TOKEN")
    dataset.push_to_hub(
        HF_REPO,
        private=False,
        token=token,
    )

    # Upload README
    print("Uploading README...")
    api = HfApi()
    api.upload_file(
        path_or_fileobj=DATASET_README.encode("utf-8"),
        path_in_repo="README.md",
        repo_id=HF_REPO,
        repo_type="dataset",
        token=token,
    )

    print(f"\nSuccess! Dataset available at: https://huggingface.co/datasets/{HF_REPO}")


if __name__ == "__main__":
    main()
