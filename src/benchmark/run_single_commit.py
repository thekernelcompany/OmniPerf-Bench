#!/usr/bin/env python3
"""
Run SGLang/vLLM benchmark for a single commit.

Automatically fetches perf_command, model from HuggingFace dataset,
finds agent patch from perf-agents-bench, and saves results.

Usage:
    python -m src.benchmark.run_single_commit 09deb20d --repo sglang
    python -m src.benchmark.run_single_commit abc12345 --repo vllm
"""

import os
import sys
import json
import argparse
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, Optional, Any

import pandas as pd

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# Paths
CLAUDE_CODE_RUNS_DIR = {
    "sglang": Path("perf-agents-bench/state/runs/sglang/claude_code"),
    "vllm": Path("perf-agents-bench/state/runs/vllm/claude_code"),
}
RESULTS_DIR = Path("benchmark_results")

# Local parquet dataset paths (ground truth)
LOCAL_PARQUET_PATHS = {
    "sglang": Path("data/omniperf_v1_hf_latest/sglang/train-00000-of-00001.parquet"),
    "vllm": Path("data/omniperf_v1_hf_latest/vllm/train-00000-of-00001.parquet"),
}


def find_dataset_row(commit: str, repo: str) -> Optional[Dict]:
    """Find the dataset row for a commit from local parquet file."""
    parquet_path = LOCAL_PARQUET_PATHS.get(repo)

    if not parquet_path or not parquet_path.exists():
        logger.error(f"Local parquet not found: {parquet_path}")
        logger.error(f"Make sure data/omniperf_v1_hf_latest/{repo}/ exists with the parquet file")
        return None

    logger.info(f"Loading local parquet: {parquet_path}")
    df = pd.read_parquet(parquet_path)

    commit_short = commit[:8]
    for _, row in df.iterrows():
        row_commit = row.get("commit_hash", "")
        if row_commit and (row_commit.startswith(commit_short) or commit_short in row_commit):
            logger.info(f"Found commit in dataset: {row_commit[:12]}")
            return row.to_dict()

    logger.error(f"Commit {commit_short} not found in {repo} parquet ({len(df)} rows)")
    return None


def find_agent_patch(commit: str, repo: str) -> Optional[Dict]:
    """Find agent patch from perf-agents-bench runs."""
    runs_dir = CLAUDE_CODE_RUNS_DIR.get(repo)
    if not runs_dir or not runs_dir.exists():
        logger.warning(f"Claude Code runs directory not found: {runs_dir}")
        return None

    commit_short = commit[:8]

    # Search for patch file matching this commit (3 levels: config/date/instance)
    for run_dir in runs_dir.glob(f"**/*{commit_short}*"):
        patch_file = run_dir / "model_patch.diff"
        journal_file = run_dir / "journal.json"

        if patch_file.exists():
            patch_content = patch_file.read_text()
            if not patch_content.strip():
                logger.warning(f"Empty patch file at {patch_file}")
                continue

            # Get parent commit from journal
            parent_commit = None
            if journal_file.exists():
                try:
                    journal = json.loads(journal_file.read_text())
                    parent_commit = journal.get("commits", {}).get("pre", "")
                except:
                    pass

            logger.info(f"Found agent patch: {patch_file} ({len(patch_content)} bytes)")
            return {
                "patch_content": patch_content,
                "patch_path": str(patch_file),
                "parent_commit": parent_commit,
                "run_dir": str(run_dir),
            }

    logger.warning(f"No agent patch found for {commit_short}")
    return None


def get_parent_commit(commit: str) -> Optional[str]:
    """Get parent commit from git."""
    import subprocess

    # Try to get from local sglang repo
    for repo_path in ["/tmp/sglang_repo", "sglang", "../sglang"]:
        if os.path.exists(repo_path):
            result = subprocess.run(
                ["git", "rev-parse", f"{commit}^"],
                cwd=repo_path, capture_output=True, text=True, timeout=30
            )
            if result.returncode == 0:
                return result.stdout.strip()

    return None


def run_benchmark(
    commit: str,
    repo: str,
    dataset_row: Dict,
    agent_patch: Optional[Dict],
    gpu_config: str = "H100:1",
    human_only: bool = False,
    baseline_only: bool = False,
    agent_only: bool = False,
    parallel: bool = False,
) -> Dict[str, Any]:
    """Run the benchmark on Modal.

    Args:
        commit: Commit hash (short or full)
        repo: Repository name (sglang or vllm)
        dataset_row: Row from HuggingFace dataset
        agent_patch: Agent patch info (if found)
        gpu_config: GPU configuration (H100:1, H100:2, etc.)
        human_only: Only benchmark human commit (no baseline/agent)
        baseline_only: Only benchmark baseline commit (no human/agent)
        agent_only: Only benchmark agent commit (requires agent patch)
        parallel: Run 3-way benchmark in parallel (3 GPUs, ~3x faster)
    """

    # Use full commit hash from dataset (needed for Docker image lookup)
    full_commit = dataset_row.get("commit_hash", commit)

    perf_command = dataset_row.get("perf_command", "")
    models = dataset_row.get("models", [])
    model = models[0] if isinstance(models, list) and models else models

    if not perf_command:
        return {"error": "No perf_command in dataset", "status": "error"}

    if not model or model == "N/A":
        return {"error": "No model specified in dataset", "status": "error"}

    logger.info(f"Model: {model}")
    logger.info(f"Command: {perf_command[:100]}...")
    logger.info(f"GPU config: {gpu_config}")
    logger.info(f"Parallel mode: {parallel}")

    # Determine base commit
    base_commit = None
    if not human_only or baseline_only or agent_only:
        if agent_patch and agent_patch.get("parent_commit"):
            base_commit = agent_patch["parent_commit"]
        else:
            base_commit = get_parent_commit(full_commit)

    # Get patch content
    patch_content = None
    if agent_patch:
        patch_content = agent_patch.get("patch_content")

    # Get human PR patch for reverse-patch approach
    human_patch = dataset_row.get("diff_text", "")
    if human_patch:
        logger.info(f"Human PR patch: {len(human_patch)} bytes")

    logger.info(f"Full commit: {full_commit}")
    logger.info(f"Base commit: {base_commit}")

    if repo == "sglang":
        if baseline_only:
            # Run only baseline benchmark using the dedicated function
            from src.benchmark.modal.sglang_benchmark import (
                run_baseline_benchmark,
                SGLANG_DOCKER_REPO,
                get_prebuilt_commit,
                has_prebuilt_image,
                GPU_CONFIGS,
            )
            import modal

            if not base_commit:
                return {"error": "No base commit found for baseline-only mode", "status": "error"}

            if not has_prebuilt_image(base_commit):
                return {"error": f"No Docker image for base commit {base_commit[:8]}", "status": "error"}

            full_base_commit = get_prebuilt_commit(base_commit)
            base_docker_image = f"{SGLANG_DOCKER_REPO}:{full_base_commit}"

            logger.info(f"Using BASELINE-ONLY benchmark")
            logger.info(f"Base Docker image: {base_docker_image}")

            # Parse GPU config
            gpu_cfg = GPU_CONFIGS.get(gpu_config, GPU_CONFIGS["H100:1"])

            # Enable Modal output and run
            modal.enable_output()

            try:
                run_baseline_fn = modal.Function.from_name("sglang-benchmark", "run_baseline_benchmark")
                logger.info("Found deployed run_baseline_benchmark function")
            except modal.exception.NotFoundError:
                return {"error": "Modal app not deployed. Run: modal deploy src/benchmark/modal/sglang_benchmark.py", "status": "error"}

            call = run_baseline_fn.spawn(
                docker_image_tag=base_docker_image,
                commit=base_commit,
                perf_command=perf_command,
                model=model,
                gpu_type=gpu_cfg["gpu"],
                gpu_count=gpu_cfg["count"],
                sandbox_timeout=gpu_cfg["timeout"],
            )

            phase_result = call.get(timeout=gpu_cfg["timeout"] + 300)

            result = {
                "status": phase_result.get("status", "error"),
                "baseline_metrics": phase_result.get("metrics", {}),
                "human_metrics": {},
                "agent_metrics": None,
                "baseline_raw": phase_result.get("raw_output", ""),
                "human_raw": "",
                "agent_raw": "",
                "error": phase_result.get("error"),
                "benchmark_mode": "baseline_only",
                "duration_s": phase_result.get("duration_s", 0),
            }

        elif agent_only:
            # Run only agent benchmark using the dedicated function
            from src.benchmark.modal.sglang_benchmark import (
                run_agent_benchmark,
                SGLANG_DOCKER_REPO,
                get_prebuilt_commit,
                has_prebuilt_image,
                GPU_CONFIGS,
            )
            import modal

            # Agent-only requires both base commit and agent patch
            if not base_commit:
                return {"error": "No base commit found for agent-only mode", "status": "error"}

            if not patch_content:
                return {"error": "No agent patch found for agent-only mode. Agent patch is required.", "status": "error"}

            if not has_prebuilt_image(base_commit):
                return {"error": f"No Docker image for base commit {base_commit[:8]}", "status": "error"}

            full_base_commit = get_prebuilt_commit(base_commit)
            base_docker_image = f"{SGLANG_DOCKER_REPO}:{full_base_commit}"

            logger.info(f"Using AGENT-ONLY benchmark")
            logger.info(f"Base Docker image: {base_docker_image}")
            logger.info(f"Agent patch size: {len(patch_content)} bytes")

            # Parse GPU config
            gpu_cfg = GPU_CONFIGS.get(gpu_config, GPU_CONFIGS["H100:1"])

            # Enable Modal output and run
            modal.enable_output()

            try:
                run_agent_fn = modal.Function.from_name("sglang-benchmark", "run_agent_benchmark")
                logger.info("Found deployed run_agent_benchmark function")
            except modal.exception.NotFoundError:
                return {"error": "Modal app not deployed. Run: modal deploy src/benchmark/modal/sglang_benchmark.py", "status": "error"}

            call = run_agent_fn.spawn(
                docker_image_tag=base_docker_image,
                commit=base_commit,
                perf_command=perf_command,
                model=model,
                agent_patch=patch_content,
                gpu_type=gpu_cfg["gpu"],
                gpu_count=gpu_cfg["count"],
                sandbox_timeout=gpu_cfg["timeout"],
            )

            phase_result = call.get(timeout=gpu_cfg["timeout"] + 300)

            result = {
                "status": phase_result.get("status", "error"),
                "baseline_metrics": {},
                "human_metrics": {},
                "agent_metrics": phase_result.get("metrics", {}),
                "baseline_raw": "",
                "human_raw": "",
                "agent_raw": phase_result.get("raw_output", ""),
                "error": phase_result.get("error"),
                "benchmark_mode": "agent_only",
                "duration_s": phase_result.get("duration_s", 0),
            }

        elif parallel:
            # Uses spawn/get pattern with Modal functions
            # Works for both 3-way and human-only (let it complete, don't ESC)
            from src.benchmark.modal.sglang_benchmark import run_3way_benchmark_docker_parallel
            phases = 1 if human_only else 3
            logger.info(f"Using PARALLEL benchmark ({phases} phase{'s' if phases > 1 else ''}, Modal Functions)")
            result = run_3way_benchmark_docker_parallel(
                human_commit=full_commit,
                base_commit=base_commit,
                agent_patch=patch_content,
                human_patch=human_patch,  # Pass PR patch for reverse-patch approach
                perf_command=perf_command,
                model=model,
                gpu_config=gpu_config,
            )
        else:
            from src.benchmark.modal.sglang_benchmark import run_3way_benchmark_docker
            result = run_3way_benchmark_docker(
                human_commit=full_commit,
                base_commit=base_commit,
                agent_patch=patch_content,
                human_patch=human_patch,  # Pass PR patch for reverse-patch approach
                perf_command=perf_command,
                model=model,
                gpu_config=gpu_config,
            )
    else:
        if parallel and not human_only:
            from src.benchmark.modal.vllm_benchmark import run_3way_modal_benchmark_prebuilt_parallel
            logger.info("Using PARALLEL 3-way benchmark (3 GPUs)")
            result = run_3way_modal_benchmark_prebuilt_parallel(
                baseline_commit=base_commit or full_commit,
                human_commit=full_commit,
                agent_patch=patch_content,
                perf_command=perf_command,
                model=model,
                gpu_config=gpu_config,
            )
        else:
            from src.benchmark.modal.vllm_benchmark import run_3way_modal_benchmark_prebuilt
            result = run_3way_modal_benchmark_prebuilt(
                baseline_commit=base_commit or full_commit,
                human_commit=full_commit,
                agent_patch=patch_content,
                perf_command=perf_command,
                model=model,
                gpu_config=gpu_config,
            )

    return result


def save_results(
    commit: str,
    repo: str,
    dataset_row: Dict,
    agent_patch: Optional[Dict],
    result: Dict,
):
    """Save benchmark results to organized folder structure.

    Creates:
        benchmark_results/{repo}/{full_commit}/{timestamp}/
            ├── metrics.json          # Parsed metrics (small, structured)
            ├── metadata.json         # Run metadata
            ├── baseline_raw.txt      # Full raw output (no truncation)
            ├── human_raw.txt         # Full raw output (no truncation)
            ├── agent_raw.txt         # Full raw output (no truncation)
            ├── agent_patch.diff      # The patch file
            ├── perf_command.txt      # The benchmark command
            ├── commit_proof.json     # Docker commit verification
            └── summary.txt           # Human-readable summary
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Use full commit hash from dataset for folder name (not short)
    full_commit = dataset_row.get("commit_hash", commit)
    commit_short = commit[:12]  # For display only

    # Create organized folder structure with FULL commit hash
    run_dir = RESULTS_DIR / repo / full_commit / timestamp
    run_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Saving results to: {run_dir}")

    # 1. Save metrics.json (parsed metrics only, no raw outputs)
    metrics_data = {
        "status": result.get("status"),
        "baseline_metrics": result.get("baseline_metrics", {}),
        "human_metrics": result.get("human_metrics", {}),
        "agent_metrics": result.get("agent_metrics"),
        "human_improvement": result.get("human_improvement", {}),
        "agent_improvement": result.get("agent_improvement"),
        "agent_vs_human": result.get("agent_vs_human"),
        "error": result.get("error"),
        "duration_s": result.get("duration_s"),
        "gpu_config": result.get("gpu_config"),
        "benchmark_mode": result.get("benchmark_mode"),
        "install_method": result.get("install_method"),
    }
    (run_dir / "metrics.json").write_text(json.dumps(metrics_data, indent=2, default=str))

    # 2. Save metadata.json
    metadata = {
        "commit": commit,
        "commit_full": full_commit,
        "commit_short": commit_short,
        "repo": repo,
        "timestamp": timestamp,
        "dataset_row": {
            "commit_hash": dataset_row.get("commit_hash"),
            "pr_url": dataset_row.get("pr_url"),
            "commit_subject": dataset_row.get("commit_subject"),
            "models": dataset_row.get("models"),
            "has_serving": dataset_row.get("has_serving"),
            "has_latency": dataset_row.get("has_latency"),
            "has_throughput": dataset_row.get("has_throughput"),
        },
        "agent_patch_info": {
            "found": agent_patch is not None,
            "patch_path": agent_patch.get("patch_path") if agent_patch else None,
            "patch_size": len(agent_patch.get("patch_content", "")) if agent_patch else 0,
            "parent_commit": agent_patch.get("parent_commit") if agent_patch else None,
            "run_dir": agent_patch.get("run_dir") if agent_patch else None,
        },
    }
    (run_dir / "metadata.json").write_text(json.dumps(metadata, indent=2, default=str))

    # 2b. Save commit_proof.json (Docker container commit verification)
    commit_proof = {
        "expected_commit": full_commit,
        "docker_proof": result.get("docker_commit_proof", {}),
        "verification": {
            "human": result.get("human_commit_proof", {}),
            "baseline": result.get("baseline_commit_proof", {}),
            "agent": result.get("agent_commit_proof", {}),
        },
    }
    (run_dir / "commit_proof.json").write_text(json.dumps(commit_proof, indent=2, default=str))

    # 3. Save perf_command.txt (full command, no truncation)
    perf_command = dataset_row.get("perf_command", "")
    (run_dir / "perf_command.txt").write_text(perf_command)

    # 4. Save raw outputs to separate files (NO TRUNCATION)
    baseline_raw = result.get("baseline_raw", "")
    if baseline_raw:
        (run_dir / "baseline_raw.txt").write_text(baseline_raw)
        logger.info(f"  baseline_raw.txt: {len(baseline_raw)} bytes")

    human_raw = result.get("human_raw", "")
    if human_raw:
        (run_dir / "human_raw.txt").write_text(human_raw)
        logger.info(f"  human_raw.txt: {len(human_raw)} bytes")

    agent_raw = result.get("agent_raw", "")
    if agent_raw:
        (run_dir / "agent_raw.txt").write_text(agent_raw)
        logger.info(f"  agent_raw.txt: {len(agent_raw)} bytes")

    # 5. Save agent patch (full patch, no truncation)
    if agent_patch and agent_patch.get("patch_content"):
        patch_content = agent_patch["patch_content"]
        (run_dir / "agent_patch.diff").write_text(patch_content)
        logger.info(f"  agent_patch.diff: {len(patch_content)} bytes")

    # 6. Save human-readable summary
    with open(run_dir / "summary.txt", "w") as f:
        f.write(f"Benchmark Results for {repo} commit {full_commit}\n")
        f.write(f"(short: {commit_short})\n")
        f.write(f"=" * 60 + "\n\n")
        f.write(f"PR: {dataset_row.get('pr_url', 'N/A')}\n")
        f.write(f"Subject: {dataset_row.get('commit_subject', 'N/A')}\n")
        f.write(f"Model: {dataset_row.get('models', 'N/A')}\n")
        f.write(f"GPU Config: {result.get('gpu_config', 'N/A')}\n")
        f.write(f"Mode: {result.get('benchmark_mode', 'N/A')}\n")
        f.write(f"Status: {result.get('status', 'N/A')}\n\n")

        f.write("BASELINE METRICS:\n")
        for k, v in result.get("baseline_metrics", {}).items():
            f.write(f"  {k}: {v}\n")

        f.write("\nHUMAN METRICS:\n")
        for k, v in result.get("human_metrics", {}).items():
            f.write(f"  {k}: {v}\n")

        f.write("\nAGENT METRICS:\n")
        agent_metrics = result.get("agent_metrics")
        if agent_metrics:
            for k, v in agent_metrics.items():
                f.write(f"  {k}: {v}\n")
        else:
            f.write("  N/A (no agent patch or benchmark failed)\n")

        f.write("\nIMPROVEMENTS:\n")
        f.write(f"  Human vs Baseline: {result.get('human_improvement', {})}\n")
        f.write(f"  Agent vs Baseline: {result.get('agent_improvement', 'N/A')}\n")
        f.write(f"  Agent vs Human: {result.get('agent_vs_human', 'N/A')}\n")

        if result.get("error"):
            f.write(f"\nERROR: {result['error']}\n")

        f.write(f"\nDuration: {result.get('duration_s', 0):.1f}s\n")

        # Add file sizes for raw outputs
        f.write("\nRAW OUTPUT FILES:\n")
        f.write(f"  baseline_raw.txt: {len(baseline_raw)} bytes\n")
        f.write(f"  human_raw.txt: {len(human_raw)} bytes\n")
        f.write(f"  agent_raw.txt: {len(agent_raw) if agent_raw else 0} bytes\n")

    logger.info(f"Summary saved to: {run_dir / 'summary.txt'}")

    # 7. Also save full result as complete_result.json for debugging
    # (includes raw outputs - may be large but useful for debugging)
    (run_dir / "complete_result.json").write_text(
        json.dumps(result, indent=2, default=str)
    )

    return run_dir


def main():
    parser = argparse.ArgumentParser(
        description="Run benchmark for a single commit",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Run SGLang benchmark (sequential, 1 GPU)
    python -m src.benchmark.run_single_commit 09deb20d --repo sglang

    # Run PARALLEL benchmark (3 GPUs, ~3x faster)
    python -m src.benchmark.run_single_commit 09deb20d --repo sglang --parallel

    # Run vLLM benchmark with specific GPU config
    python -m src.benchmark.run_single_commit abc12345 --repo vllm --gpu H100:2

    # Human-only mode (no baseline/agent comparison)
    python -m src.benchmark.run_single_commit 09deb20d --repo sglang --human-only

    # Dry run (just show what would be done)
    python -m src.benchmark.run_single_commit 09deb20d --repo sglang --dry-run
        """
    )
    parser.add_argument("commit", help="Commit hash (full or short)")
    parser.add_argument("--repo", choices=["sglang", "vllm"], default="sglang", help="Repository")
    parser.add_argument("--gpu", default="H100:1", help="GPU config (H100:1, H100:2, H100:4, H100:8)")
    parser.add_argument("--human-only", action="store_true", help="Only benchmark human commit")
    parser.add_argument("--baseline-only", action="store_true", help="Only benchmark baseline commit")
    parser.add_argument("--agent-only", action="store_true", help="Only benchmark agent commit (requires agent patch)")
    parser.add_argument("--parallel", action="store_true", help="Run 3-way benchmark in parallel (3 GPUs, ~3x faster)")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be done without running")
    parser.add_argument("--perf-command", type=str, help="Override perf_command from dataset")

    args = parser.parse_args()

    print(f"\n{'=' * 60}")
    print(f"  Single Commit Benchmark Runner")
    print(f"{'=' * 60}\n")

    # Step 1: Find in dataset
    print(f"[1/4] Looking up commit {args.commit[:8]} in {args.repo} dataset...")
    dataset_row = find_dataset_row(args.commit, args.repo)
    if not dataset_row:
        print(f"ERROR: Commit not found in local parquet dataset")
        sys.exit(1)

    print(f"  Found: {dataset_row.get('commit_subject', 'N/A')[:60]}")
    print(f"  PR: {dataset_row.get('pr_url', 'N/A')}")
    print(f"  Model: {dataset_row.get('models', 'N/A')}")

    # Step 2: Find agent patch
    print(f"\n[2/4] Looking for agent patch...")
    agent_patch = find_agent_patch(args.commit, args.repo)
    if agent_patch:
        print(f"  Found patch: {agent_patch['patch_path']}")
        print(f"  Patch size: {len(agent_patch['patch_content'])} bytes")
    else:
        print(f"  No agent patch found (will skip agent benchmark)")

    # Step 3: Show plan
    print(f"\n[3/4] Benchmark plan:")
    print(f"  Commit: {args.commit}")
    print(f"  Repo: {args.repo}")
    print(f"  GPU: {args.gpu}")
    if args.human_only:
        print(f"  Mode: human-only")
    elif args.baseline_only:
        print(f"  Mode: baseline-only")
    elif args.agent_only:
        print(f"  Mode: agent-only")
    elif args.parallel:
        print(f"  Mode: 3-way PARALLEL (3 GPUs, ~3x faster)")
    else:
        print(f"  Mode: 3-way sequential (1 GPU)")
    print(f"  Command: {dataset_row.get('perf_command', 'N/A')[:80]}...")

    if args.dry_run:
        print(f"\n[DRY RUN] Would run benchmark with above settings")
        sys.exit(0)

    # Step 4: Run benchmark
    print(f"\n[4/4] Running benchmark on Modal...")
    result = run_benchmark(
        commit=args.commit,
        repo=args.repo,
        dataset_row=dataset_row,
        agent_patch=agent_patch,
        gpu_config=args.gpu,
        human_only=args.human_only,
        baseline_only=args.baseline_only,
        agent_only=args.agent_only,
        parallel=args.parallel,
    )

    # Save results
    result_dir = save_results(
        commit=args.commit,
        repo=args.repo,
        dataset_row=dataset_row,
        agent_patch=agent_patch,
        result=result,
    )

    # Print summary
    print(f"\n{'=' * 60}")
    print(f"  RESULTS")
    print(f"{'=' * 60}")
    print(f"Status: {result.get('status', 'unknown')}")

    if result.get("error"):
        print(f"Error: {result['error']}")

    print(f"\nBaseline metrics: {result.get('baseline_metrics', {})}")
    print(f"Human metrics: {result.get('human_metrics', {})}")
    print(f"Agent metrics: {result.get('agent_metrics', 'N/A')}")

    if result.get("human_improvement"):
        print(f"\nHuman improvement: {result['human_improvement']}")
    if result.get("agent_improvement"):
        print(f"Agent improvement: {result['agent_improvement']}")

    print(f"\nResults saved to: {result_dir}/")
    print(f"  metrics.json      - Parsed benchmark metrics")
    print(f"  baseline_raw.txt  - Full baseline output")
    print(f"  human_raw.txt     - Full human output")
    print(f"  agent_raw.txt     - Full agent output (if applicable)")
    print(f"  agent_patch.diff  - Agent patch (if applicable)")
    print(f"  summary.txt       - Human-readable summary")


if __name__ == "__main__":
    main()
