#!/usr/bin/env python3
"""
Upload Claude Code vLLM benchmark results to HuggingFace.

This script collects all benchmark results from the omniperf_results_3way_claude_code/results
directory and uploads them to HuggingFace as a dataset.

Schema v5: Merged Modal + Separate + Docker pipeline data (76 columns).
- data_source column tracks provenance: modal, separate, merged, docker
- Docker results (local reruns) have highest priority
- Merges Separate pipeline data to fill gaps in Modal results
"""

import json
import glob
import os
from datetime import datetime
from pathlib import Path

# Default agent configuration - update these for your runs
DEFAULT_AGENT_NAME = "claude-code"
DEFAULT_AGENT_MODEL = "claude-sonnet-4-20250514"


def collect_benchmark_results(results_dir: str, agent_name: str = None, agent_model: str = None) -> list[dict]:
    """Collect all benchmark results from the results directory.

    Args:
        results_dir: Path to directory containing commit folders with benchmark_result.json
        agent_name: Name of the agent that generated patches (e.g., "claude-code")
        agent_model: Model used by the agent (e.g., "claude-sonnet-4-20250514")
    """
    results = []
    agent_name = agent_name or DEFAULT_AGENT_NAME
    agent_model = agent_model or DEFAULT_AGENT_MODEL

    for result_file in glob.glob(os.path.join(results_dir, "*/benchmark_result.json")):
        try:
            with open(result_file) as f:
                data = json.load(f)

            instance = data.get("instance", {})
            result = data.get("result", {})

            # Extract benchmark date from file modification time
            file_mtime = os.path.getmtime(result_file)
            benchmark_date = datetime.fromtimestamp(file_mtime).strftime("%Y-%m-%d")

            # Flatten the structure for HuggingFace dataset
            row = {
                # Instance metadata
                "commit_hash": instance.get("commit_hash") or result.get("full_commit"),
                "commit_short": result.get("commit") or (instance.get("commit_hash") or "")[:8],
                "commit_subject": instance.get("commit_subject") or result.get("subject"),
                "repo": instance.get("repo", "vllm-project/vllm"),
                "perf_command": instance.get("perf_command") or result.get("perf_command"),
                "files_changed": instance.get("files_changed", []),
                "pr_url": instance.get("pr_url"),
                "models": instance.get("models", []),

                # Git metadata
                "parent_commit": result.get("parent_commit"),

                # Result metadata
                "gpu_config": result.get("gpu_config"),
                "benchmark_mode": result.get("benchmark_mode"),

                # Agent metadata (for multi-agent comparisons)
                "agent_name": agent_name,
                "agent_model": agent_model,
                "benchmark_date": benchmark_date,

                # Model and patch info
                "model": result.get("model"),
                "has_agent_patch": result.get("has_agent_patch"),
                "patch_path": result.get("patch_path"),

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

                # Improvement metrics - Serving (mean)
                "human_improvement_ttft_mean": None,
                "human_improvement_tpot_mean": None,
                "human_improvement_itl_mean": None,
                "agent_improvement_ttft_mean": None,
                "agent_improvement_tpot_mean": None,
                "agent_improvement_itl_mean": None,
                "agent_vs_human_ttft_mean": None,
                "agent_vs_human_tpot_mean": None,
                "agent_vs_human_itl_mean": None,

                # Improvement metrics - Serving (median/p99)
                "human_improvement_ttft_median": None,
                "human_improvement_ttft_p99": None,
                "agent_improvement_ttft_median": None,
                "agent_improvement_ttft_p99": None,
                "agent_vs_human_ttft_median": None,
                "agent_vs_human_ttft_p99": None,

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

                # Data source tracking (for merged datasets)
                "data_source": "modal",
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
                # Serving benchmark improvements (mean)
                row["human_improvement_ttft_mean"] = human_imp.get("ttft_mean")
                row["human_improvement_tpot_mean"] = human_imp.get("tpot_mean")
                row["human_improvement_itl_mean"] = human_imp.get("itl_mean")
                # Serving benchmark improvements (median/p99)
                row["human_improvement_ttft_median"] = human_imp.get("ttft_median")
                row["human_improvement_ttft_p99"] = human_imp.get("ttft_p99")
                # Latency/Throughput improvements
                row["human_improvement_latency_avg"] = human_imp.get("latency_avg")
                row["human_improvement_throughput"] = human_imp.get("throughput")

            agent_imp = result.get("agent_improvement", {})
            if agent_imp:
                # Serving benchmark improvements (mean)
                row["agent_improvement_ttft_mean"] = agent_imp.get("ttft_mean")
                row["agent_improvement_tpot_mean"] = agent_imp.get("tpot_mean")
                row["agent_improvement_itl_mean"] = agent_imp.get("itl_mean")
                # Serving benchmark improvements (median/p99)
                row["agent_improvement_ttft_median"] = agent_imp.get("ttft_median")
                row["agent_improvement_ttft_p99"] = agent_imp.get("ttft_p99")
                # Latency/Throughput improvements
                row["agent_improvement_latency_avg"] = agent_imp.get("latency_avg")
                row["agent_improvement_throughput"] = agent_imp.get("throughput")

            agent_vs_human = result.get("agent_vs_human", {})
            if agent_vs_human:
                # Serving benchmark comparisons (mean)
                row["agent_vs_human_ttft_mean"] = agent_vs_human.get("ttft_mean")
                row["agent_vs_human_tpot_mean"] = agent_vs_human.get("tpot_mean")
                row["agent_vs_human_itl_mean"] = agent_vs_human.get("itl_mean")
                # Serving benchmark comparisons (median/p99)
                row["agent_vs_human_ttft_median"] = agent_vs_human.get("ttft_median")
                row["agent_vs_human_ttft_p99"] = agent_vs_human.get("ttft_p99")
                # Latency/Throughput comparisons
                row["agent_vs_human_latency_avg"] = agent_vs_human.get("latency_avg")
                row["agent_vs_human_throughput"] = agent_vs_human.get("throughput")

            results.append(row)

        except Exception as e:
            print(f"Error processing {result_file}: {e}")
            continue

    return results


def collect_docker_results(docker_dir: str, agent_name: str = None, agent_model: str = None) -> list[dict]:
    """Collect benchmark results from the Docker pipeline (local reruns).

    The Docker pipeline stores results in commit folders with this structure:
    - docker/{commit}/benchmark_result.json

    Args:
        docker_dir: Path to results/docker directory
        agent_name: Name of the agent
        agent_model: Model used by the agent
    """
    results = []
    agent_name = agent_name or DEFAULT_AGENT_NAME
    agent_model = agent_model or DEFAULT_AGENT_MODEL

    docker_path = Path(docker_dir)
    if not docker_path.exists():
        return results

    for result_file in docker_path.glob("*/benchmark_result.json"):
        try:
            with open(result_file) as f:
                data = json.load(f)

            commit = data.get("commit", result_file.parent.name)
            timestamp = data.get("timestamp", "")
            benchmark_date = timestamp[:10] if timestamp else None

            # Docker format: results.baseline/human/agent
            results_data = data.get("results", {})
            baseline_data = results_data.get("baseline", {})
            human_data = results_data.get("human", {})
            agent_data = results_data.get("agent", {})

            # Determine benchmark mode from the data
            benchmark_type = baseline_data.get("benchmark_type") or human_data.get("benchmark_type", "unknown")

            # Build row with Schema v4 format
            row = {
                "commit_hash": baseline_data.get("commit_hash") or human_data.get("commit_hash") or commit,
                "commit_short": commit[:8],
                "commit_subject": None,  # Not in Docker format
                "repo": "vllm-project/vllm",
                "perf_command": None,  # Not in Docker format
                "files_changed": [],
                "pr_url": None,
                "models": [],
                "parent_commit": None,
                "gpu_config": "H100:1",  # Docker runs on local H100
                "benchmark_mode": benchmark_type,
                "agent_name": agent_name,
                "agent_model": agent_model,
                "benchmark_date": benchmark_date,
                "model": baseline_data.get("model") or human_data.get("model"),
                "has_agent_patch": bool(agent_data),
                "patch_path": None,
                "data_source": "docker",

                # Initialize all metric columns to None
                "baseline_ttft_mean": None, "baseline_ttft_median": None, "baseline_ttft_p99": None,
                "baseline_tpot_mean": None, "baseline_tpot_median": None, "baseline_tpot_p99": None,
                "baseline_itl_mean": None, "baseline_itl_median": None, "baseline_itl_p99": None,
                "baseline_latency_avg": None, "baseline_throughput": None,
                "human_ttft_mean": None, "human_ttft_median": None, "human_ttft_p99": None,
                "human_tpot_mean": None, "human_tpot_median": None, "human_tpot_p99": None,
                "human_itl_mean": None, "human_itl_median": None, "human_itl_p99": None,
                "human_latency_avg": None, "human_throughput": None,
                "agent_ttft_mean": None, "agent_ttft_median": None, "agent_ttft_p99": None,
                "agent_tpot_mean": None, "agent_tpot_median": None, "agent_tpot_p99": None,
                "agent_itl_mean": None, "agent_itl_median": None, "agent_itl_p99": None,
                "agent_latency_avg": None, "agent_throughput": None,

                # Improvement metrics
                "human_improvement_ttft_mean": None, "human_improvement_tpot_mean": None,
                "human_improvement_itl_mean": None, "agent_improvement_ttft_mean": None,
                "agent_improvement_tpot_mean": None, "agent_improvement_itl_mean": None,
                "agent_vs_human_ttft_mean": None, "agent_vs_human_tpot_mean": None,
                "agent_vs_human_itl_mean": None, "human_improvement_ttft_median": None,
                "human_improvement_ttft_p99": None, "agent_improvement_ttft_median": None,
                "agent_improvement_ttft_p99": None, "agent_vs_human_ttft_median": None,
                "agent_vs_human_ttft_p99": None, "human_improvement_latency_avg": None,
                "human_improvement_throughput": None, "agent_improvement_latency_avg": None,
                "agent_improvement_throughput": None, "agent_vs_human_latency_avg": None,
                "agent_vs_human_throughput": None,

                "baseline_raw": baseline_data.get("raw_output"),
                "human_raw": human_data.get("raw_output"),
                "agent_raw": agent_data.get("raw_output"),
                "test_script": None,
            }

            # Map metrics based on benchmark type
            def map_docker_metrics(metrics: dict, prefix: str):
                """Map Docker metrics to Schema v4 format."""
                if not metrics:
                    return
                # Serving benchmark metrics
                if "ttft_mean" in metrics:
                    row[f"{prefix}_ttft_mean"] = metrics.get("ttft_mean")
                    row[f"{prefix}_ttft_median"] = metrics.get("ttft_median")
                    row[f"{prefix}_ttft_p99"] = metrics.get("ttft_p99")
                    row[f"{prefix}_tpot_mean"] = metrics.get("tpot_mean")
                    row[f"{prefix}_tpot_median"] = metrics.get("tpot_median")
                    row[f"{prefix}_tpot_p99"] = metrics.get("tpot_p99")
                    row[f"{prefix}_itl_mean"] = metrics.get("itl_mean")
                    row[f"{prefix}_itl_median"] = metrics.get("itl_median")
                    row[f"{prefix}_itl_p99"] = metrics.get("itl_p99")
                    row[f"{prefix}_throughput"] = metrics.get("output_throughput")
                # Prefix caching benchmark metrics
                elif "warmup_time_s" in metrics:
                    # Map warmup_time to latency (in ms)
                    row[f"{prefix}_latency_avg"] = metrics.get("warmup_time_s", 0) * 1000
                    row[f"{prefix}_throughput"] = metrics.get("output_throughput")

            map_docker_metrics(baseline_data.get("metrics", {}), "baseline")
            map_docker_metrics(human_data.get("metrics", {}), "human")
            map_docker_metrics(agent_data.get("metrics", {}), "agent")

            # Calculate improvement metrics if we have baseline data
            def calc_improvement(baseline_val, target_val):
                """Calculate % improvement (positive = better)."""
                if baseline_val and target_val and baseline_val > 0:
                    # For latency/time metrics: lower is better
                    return ((baseline_val - target_val) / baseline_val) * 100
                return None

            def calc_throughput_improvement(baseline_val, target_val):
                """Calculate throughput % improvement (higher = better)."""
                if baseline_val and target_val and baseline_val > 0:
                    return ((target_val - baseline_val) / baseline_val) * 100
                return None

            # Human improvements
            if row.get("baseline_ttft_mean") and row.get("human_ttft_mean"):
                row["human_improvement_ttft_mean"] = calc_improvement(row["baseline_ttft_mean"], row["human_ttft_mean"])
            if row.get("baseline_tpot_mean") and row.get("human_tpot_mean"):
                row["human_improvement_tpot_mean"] = calc_improvement(row["baseline_tpot_mean"], row["human_tpot_mean"])
            if row.get("baseline_latency_avg") and row.get("human_latency_avg"):
                row["human_improvement_latency_avg"] = calc_improvement(row["baseline_latency_avg"], row["human_latency_avg"])
            if row.get("baseline_throughput") and row.get("human_throughput"):
                row["human_improvement_throughput"] = calc_throughput_improvement(row["baseline_throughput"], row["human_throughput"])

            # Agent improvements
            if row.get("baseline_ttft_mean") and row.get("agent_ttft_mean"):
                row["agent_improvement_ttft_mean"] = calc_improvement(row["baseline_ttft_mean"], row["agent_ttft_mean"])
            if row.get("baseline_tpot_mean") and row.get("agent_tpot_mean"):
                row["agent_improvement_tpot_mean"] = calc_improvement(row["baseline_tpot_mean"], row["agent_tpot_mean"])
            if row.get("baseline_latency_avg") and row.get("agent_latency_avg"):
                row["agent_improvement_latency_avg"] = calc_improvement(row["baseline_latency_avg"], row["agent_latency_avg"])
            if row.get("baseline_throughput") and row.get("agent_throughput"):
                row["agent_improvement_throughput"] = calc_throughput_improvement(row["baseline_throughput"], row["agent_throughput"])

            # Agent vs Human
            if row.get("human_ttft_mean") and row.get("agent_ttft_mean"):
                row["agent_vs_human_ttft_mean"] = calc_improvement(row["human_ttft_mean"], row["agent_ttft_mean"])
            if row.get("human_tpot_mean") and row.get("agent_tpot_mean"):
                row["agent_vs_human_tpot_mean"] = calc_improvement(row["human_tpot_mean"], row["agent_tpot_mean"])
            if row.get("human_throughput") and row.get("agent_throughput"):
                row["agent_vs_human_throughput"] = calc_throughput_improvement(row["human_throughput"], row["agent_throughput"])

            results.append(row)

        except Exception as e:
            print(f"Error processing Docker result {result_file}: {e}")
            continue

    return results


def collect_separate_results(separate_dir: str, agent_name: str = None, agent_model: str = None) -> list[dict]:
    """Collect benchmark results from the Separate pipeline (baseline + human + agent files).

    The Separate pipeline stores results in three separate file types:
    - *_baseline_result.json in separate_baseline/
    - *_human_result.json in separate_agent/
    - *_agent_result.json in separate_agent/

    Args:
        separate_dir: Path to results directory containing separate_baseline/ and separate_agent/
        agent_name: Name of the agent
        agent_model: Model used by the agent
    """
    results = []
    agent_name = agent_name or DEFAULT_AGENT_NAME
    agent_model = agent_model or DEFAULT_AGENT_MODEL

    baseline_dir = Path(separate_dir) / "separate_baseline"
    agent_dir = Path(separate_dir) / "separate_agent"

    if not baseline_dir.exists() and not agent_dir.exists():
        return results

    # Collect all unique commits from human results
    commits = set()
    for f in agent_dir.glob("*_human_result.json"):
        commit = f.name.replace("_human_result.json", "")
        commits.add(commit)

    # Field mapping from Separate format to Schema v4
    def map_metrics(metrics: dict, prefix: str) -> dict:
        """Map Separate pipeline metrics to Schema v4 format."""
        if not metrics:
            return {}
        return {
            f"{prefix}_ttft_mean": metrics.get("mean_ttft_ms"),
            f"{prefix}_ttft_median": metrics.get("median_ttft_ms"),
            f"{prefix}_ttft_p99": metrics.get("p99_ttft_ms"),
            f"{prefix}_tpot_mean": metrics.get("mean_tpot_ms"),
            f"{prefix}_tpot_median": metrics.get("median_tpot_ms"),
            f"{prefix}_tpot_p99": metrics.get("p99_tpot_ms"),
            f"{prefix}_itl_mean": metrics.get("mean_itl_ms"),
            f"{prefix}_itl_median": metrics.get("median_itl_ms"),
            f"{prefix}_itl_p99": metrics.get("p99_itl_ms"),
            f"{prefix}_throughput": metrics.get("output_token_throughput_tok_s"),
        }

    for commit in commits:
        try:
            # Load human result (required)
            human_file = agent_dir / f"{commit}_human_result.json"
            if not human_file.exists():
                continue
            with open(human_file) as f:
                human_data = json.load(f)

            # Load agent result (optional)
            agent_file = agent_dir / f"{commit}_agent_result.json"
            agent_data = {}
            if agent_file.exists():
                with open(agent_file) as f:
                    agent_data = json.load(f)

            # Load baseline result (optional)
            baseline_file = baseline_dir / f"{commit}_baseline_result.json"
            baseline_data = {}
            if baseline_file.exists():
                with open(baseline_file) as f:
                    baseline_data = json.load(f)

            # Extract timestamp for benchmark date
            timestamp = human_data.get("timestamp", "")
            benchmark_date = timestamp[:10] if timestamp else None

            # Get commit hash - use file-based commit as fallback since many files have None
            commit_hash = human_data.get("commit_full") or human_data.get("commit") or commit

            # Build row
            row = {
                "commit_hash": commit_hash,
                "commit_short": commit[:8],  # Use filename-based commit
                "commit_subject": None,  # Not in Separate format
                "repo": "vllm-project/vllm",
                "perf_command": baseline_data.get("benchmark_command"),
                "files_changed": [],
                "pr_url": None,
                "models": [],
                "parent_commit": human_data.get("parent_commit"),
                "gpu_config": None,
                "benchmark_mode": human_data.get("benchmark_type"),
                "agent_name": agent_name,
                "agent_model": agent_model,
                "benchmark_date": benchmark_date,
                "model": human_data.get("model"),
                "has_agent_patch": bool(agent_data),
                "patch_path": None,
                "data_source": "separate",

                # Initialize all metric columns to None
                "baseline_ttft_mean": None, "baseline_ttft_median": None, "baseline_ttft_p99": None,
                "baseline_tpot_mean": None, "baseline_tpot_median": None, "baseline_tpot_p99": None,
                "baseline_itl_mean": None, "baseline_itl_median": None, "baseline_itl_p99": None,
                "baseline_latency_avg": None, "baseline_throughput": None,
                "human_ttft_mean": None, "human_ttft_median": None, "human_ttft_p99": None,
                "human_tpot_mean": None, "human_tpot_median": None, "human_tpot_p99": None,
                "human_itl_mean": None, "human_itl_median": None, "human_itl_p99": None,
                "human_latency_avg": None, "human_throughput": None,
                "agent_ttft_mean": None, "agent_ttft_median": None, "agent_ttft_p99": None,
                "agent_tpot_mean": None, "agent_tpot_median": None, "agent_tpot_p99": None,
                "agent_itl_mean": None, "agent_itl_median": None, "agent_itl_p99": None,
                "agent_latency_avg": None, "agent_throughput": None,

                # Improvement metrics (not calculated for Separate)
                "human_improvement_ttft_mean": None, "human_improvement_tpot_mean": None,
                "human_improvement_itl_mean": None, "agent_improvement_ttft_mean": None,
                "agent_improvement_tpot_mean": None, "agent_improvement_itl_mean": None,
                "agent_vs_human_ttft_mean": None, "agent_vs_human_tpot_mean": None,
                "agent_vs_human_itl_mean": None, "human_improvement_ttft_median": None,
                "human_improvement_ttft_p99": None, "agent_improvement_ttft_median": None,
                "agent_improvement_ttft_p99": None, "agent_vs_human_ttft_median": None,
                "agent_vs_human_ttft_p99": None, "human_improvement_latency_avg": None,
                "human_improvement_throughput": None, "agent_improvement_latency_avg": None,
                "agent_improvement_throughput": None, "agent_vs_human_latency_avg": None,
                "agent_vs_human_throughput": None,

                "baseline_raw": None, "human_raw": None, "agent_raw": None,
                "test_script": None,
            }

            # Map metrics
            if baseline_data.get("metrics"):
                row.update(map_metrics(baseline_data["metrics"], "baseline"))
            if human_data.get("metrics"):
                row.update(map_metrics(human_data["metrics"], "human"))
            if agent_data.get("metrics"):
                row.update(map_metrics(agent_data["metrics"], "agent"))

            results.append(row)

        except Exception as e:
            print(f"Error processing Separate commit {commit}: {e}")
            continue

    return results


def merge_results(modal_results: list[dict], separate_results: list[dict], docker_results: list[dict] = None) -> list[dict]:
    """Merge Modal + Separate + Docker pipeline data. Priority: Docker > Modal > Separate.

    Strategy:
    1. All Modal results are included (data_source="modal")
    2. For Modal results without metrics, fill from Separate if available (data_source="merged")
    3. Separate-only commits are added (data_source="separate")
    4. Docker results replace failed Modal results (data_source="docker")

    Note: Uses short commit hash (first 8 chars) for matching since Modal uses full hash
    and Separate uses short hash in filenames.
    """
    merged = {}
    docker_results = docker_results or []

    def short_hash(commit: str) -> str:
        """Normalize to short hash for matching."""
        return commit[:8] if commit else ""

    def has_valid_metrics(row: dict) -> bool:
        """Check if a row has valid benchmark metrics."""
        return (
            row.get("baseline_ttft_mean") is not None or
            row.get("baseline_throughput") is not None or
            row.get("baseline_latency_avg") is not None
        )

    # Add all Modal results, indexed by short hash
    for r in modal_results:
        commit = r.get("commit_hash")
        if commit:
            r["data_source"] = "modal"
            merged[short_hash(commit)] = r.copy()

    # Process Separate results
    for r in separate_results:
        commit = r.get("commit_hash")
        if not commit:
            continue

        commit_key = short_hash(commit)
        if commit_key in merged:
            modal_row = merged[commit_key]
            # Check if Modal has no metrics
            if not has_valid_metrics(modal_row):
                # Fill metrics from Separate
                metric_keys = [k for k in r.keys() if k.startswith(("baseline_", "human_", "agent_")) and k not in ("baseline_raw", "human_raw", "agent_raw")]
                for metric_key in metric_keys:
                    if r.get(metric_key) is not None:
                        modal_row[metric_key] = r[metric_key]
                modal_row["data_source"] = "merged"
        else:
            # New commit from Separate only
            merged[commit_key] = r.copy()

    # Process Docker results - highest priority, replaces failed Modal/Separate results
    for r in docker_results:
        commit = r.get("commit_hash")
        if not commit:
            continue

        commit_key = short_hash(commit)
        if commit_key in merged:
            existing_row = merged[commit_key]
            # Docker replaces if it has metrics and existing doesn't, OR if Docker has full 3-way data
            docker_has_metrics = has_valid_metrics(r)
            existing_has_metrics = has_valid_metrics(existing_row)

            if docker_has_metrics and (not existing_has_metrics or r.get("has_agent_patch")):
                # Keep metadata from existing, but replace metrics with Docker data
                for key in r.keys():
                    if r.get(key) is not None:
                        existing_row[key] = r[key]
                existing_row["data_source"] = "docker"
        else:
            # New commit from Docker only
            merged[commit_key] = r.copy()

    return list(merged.values())


def upload_to_huggingface(results: list[dict], repo_id: str, token: str = None):
    """Upload results to HuggingFace as a dataset."""
    from datasets import Dataset
    from huggingface_hub import HfApi

    # Create dataset
    dataset = Dataset.from_list(results)

    # Add metadata
    dataset.info.description = """
Claude Code vLLM Performance Benchmark Results (Schema v4)

This dataset contains performance benchmark results from running Claude Code
(Anthropic's coding agent) on vLLM performance optimization tasks.

## Benchmark Types

Each row represents a benchmark run for a specific vLLM commit, comparing:
- **Baseline**: Performance before the optimization commit
- **Human**: Performance with the human-authored optimization (ground truth)
- **Agent**: Performance with Claude Code's optimization attempt

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

## Data Sources (data_source column)
- **modal**: Data from Modal H100 pipeline (primary, same config for B/H/A)
- **separate**: Data from Separate pipeline only (human + agent files)
- **merged**: Modal metadata with Separate metrics (fills gaps in Modal)
- **docker**: Data from local Docker reruns (highest priority, replaces failed Modal)

## Infrastructure
All benchmarks were run on H100 GPUs.

## Schema Version
v5 - Merged Modal + Separate + Docker pipeline data (76 columns). Added Docker rerun support.
"""

    # Push to hub
    dataset.push_to_hub(
        repo_id,
        token=token,
        commit_message=f"Upload Claude Code vLLM benchmark results ({len(results)} commits) - Schema v5"
    )

    print(f"Successfully uploaded {len(results)} benchmark results to {repo_id}")
    return dataset


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Upload Claude Code vLLM benchmark results to HuggingFace (Schema v4)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Dry run to see what would be uploaded
  python upload_to_hf.py --dry-run

  # Merge Modal + Separate pipeline data (recommended)
  python upload_to_hf.py --merge-separate --dry-run

  # Save to JSON for inspection
  python upload_to_hf.py --save-json results.json

  # Upload merged data to HuggingFace
  python upload_to_hf.py --merge-separate --repo-id "Inferencebench/claude-code-vllm-benchmarks"

  # Upload to a different repo
  python upload_to_hf.py --repo-id "myorg/my-benchmarks"
"""
    )
    parser.add_argument("--repo-id", default="Inferencebench/claude-code-vllm-benchmarks",
                        help="HuggingFace repo ID (default: Inferencebench/claude-code-vllm-benchmarks)")
    parser.add_argument("--results-dir", default="results/modal",
                        help="Directory containing benchmark results (default: results/modal)")
    parser.add_argument("--token", default=None,
                        help="HuggingFace token (default: use HF_TOKEN env var)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Collect results but don't upload")
    parser.add_argument("--save-json", default=None,
                        help="Save results to JSON file instead of uploading")
    parser.add_argument("--agent-name", default=DEFAULT_AGENT_NAME,
                        help=f"Name of the agent (default: {DEFAULT_AGENT_NAME})")
    parser.add_argument("--agent-model", default=DEFAULT_AGENT_MODEL,
                        help=f"Model used by the agent (default: {DEFAULT_AGENT_MODEL})")
    parser.add_argument("--merge-separate", action="store_true",
                        help="Merge Separate pipeline data to fill gaps in Modal results")
    parser.add_argument("--merge-docker", action="store_true",
                        help="Merge Docker pipeline data (local reruns) - highest priority")
    parser.add_argument("--merge-all", action="store_true",
                        help="Merge all data sources: Modal + Separate + Docker")
    args = parser.parse_args()

    # Get token (check env var, then HuggingFace cache)
    token = args.token or os.environ.get("HF_TOKEN")
    if not token:
        try:
            from huggingface_hub import HfFolder
            token = HfFolder.get_token()
        except Exception:
            pass

    # Collect results
    script_dir = Path(__file__).parent
    results_dir = script_dir / args.results_dir

    print(f"Collecting Modal benchmark results from {results_dir}...")
    print(f"Agent: {args.agent_name} ({args.agent_model})")
    modal_results = collect_benchmark_results(
        str(results_dir),
        agent_name=args.agent_name,
        agent_model=args.agent_model
    )
    print(f"  Modal results: {len(modal_results)}")

    # Optionally merge with other data sources
    merge_separate = args.merge_separate or args.merge_all
    merge_docker = args.merge_docker or args.merge_all

    separate_results = []
    docker_results = []

    if merge_separate:
        separate_dir = script_dir / "results"
        print(f"\nCollecting Separate pipeline results from {separate_dir}...")
        separate_results = collect_separate_results(
            str(separate_dir),
            agent_name=args.agent_name,
            agent_model=args.agent_model
        )
        print(f"  Separate results: {len(separate_results)}")

    if merge_docker:
        docker_dir = script_dir / "results" / "docker"
        print(f"\nCollecting Docker pipeline results from {docker_dir}...")
        docker_results = collect_docker_results(
            str(docker_dir),
            agent_name=args.agent_name,
            agent_model=args.agent_model
        )
        print(f"  Docker results: {len(docker_results)}")

    if merge_separate or merge_docker:
        print("\nMerging all data sources...")
        results = merge_results(modal_results, separate_results, docker_results)

        # Count by data source
        source_counts = {}
        for r in results:
            src = r.get("data_source", "unknown")
            source_counts[src] = source_counts.get(src, 0) + 1
        print(f"  By data source: {source_counts}")
    else:
        results = modal_results

    # Print summary
    benchmark_mode_counts = {}
    for r in results:
        mode = r.get("benchmark_mode", "unknown")
        benchmark_mode_counts[mode] = benchmark_mode_counts.get(mode, 0) + 1

    print(f"\n{'='*60}")
    print(f"Collected {len(results)} benchmark results")
    print(f"{'='*60}")

    print(f"\nBy benchmark mode:")
    for mode, count in sorted(benchmark_mode_counts.items(), key=lambda x: -x[1]):
        print(f"  {mode}: {count}")

    # Count successful results with metrics
    with_serving_metrics = sum(1 for r in results if r.get("baseline_ttft_mean") is not None)
    with_latency_metrics = sum(1 for r in results if r.get("baseline_latency_avg") is not None)
    with_any_metrics = sum(1 for r in results if r.get("baseline_ttft_mean") is not None or r.get("baseline_latency_avg") is not None)

    print(f"\nMetrics coverage:")
    print(f"  Serving metrics (TTFT/TPOT/ITL): {with_serving_metrics}")
    print(f"  Standalone metrics (latency/throughput): {with_latency_metrics}")
    print(f"  Total with hard metrics: {with_any_metrics}")

    # Show date range
    dates = [r.get("benchmark_date") for r in results if r.get("benchmark_date")]
    if dates:
        print(f"\nDate range: {min(dates)} to {max(dates)}")

    if args.save_json:
        with open(args.save_json, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nSaved results to {args.save_json}")
        return

    if args.dry_run:
        print("\nDry run - not uploading to HuggingFace")
        # Show schema
        if results:
            print(f"\nSchema columns ({len(results[0])} total):")
            for col in sorted(results[0].keys()):
                print(f"  - {col}")
        return

    if not token:
        print("\nError: No HuggingFace token provided. Set HF_TOKEN env var or use --token")
        return

    # Upload
    print(f"\nUploading to HuggingFace: {args.repo_id}...")
    upload_to_huggingface(results, args.repo_id, token)


if __name__ == "__main__":
    main()
