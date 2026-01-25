#!/usr/bin/env python3
"""
SGLang 3-Way Benchmark Runner using Pre-built Docker Images

This script runs 3-way benchmarks using pre-built Docker images from ayushnangia16.
Each image has the correct dependencies for its commit era.

Usage:
    python run_prebuilt_benchmarks.py --commit 148254d4 --dry-run
    python run_prebuilt_benchmarks.py --all --agents claude_code,codex
"""

import argparse
import json
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

# Paths
RESULTS_DIR = Path("/root/OmniPerf-Bench/omniperf_results_3way_sglang/prebuilt_benchmark_results")
COMMIT_MAPPING_FILE = Path("/root/OmniPerf-Bench/src/benchmark/fixes/sglang_commit_mapping.json")
AGENT_BASE_DIR = Path("/root/OmniPerf-Bench/perf-agents-bench/state/runs/sglang")

# Docker image template
DOCKER_IMAGE_TEMPLATE = "ayushnangia16/nvidia-sglang-docker:{full_commit}"

# Agent configurations
AGENT_CONFIGS = {
    "claude_code": AGENT_BASE_DIR / "claude_code/default/2025-12-23_06-28-44",
    "codex": AGENT_BASE_DIR / "codex/gpt-5/389be848",
    "trae_gpt5": AGENT_BASE_DIR / "trae/gpt-5/2025-11-14_21-05-32",
    "trae_sonnet45": AGENT_BASE_DIR / "trae/claude-sonnet-45/2025-11-28_15-26-15",
}

# Commits with pre-built images
PREBUILT_COMMITS = [
    "187b85b7",
    "6b231325",
    "6cb00c63",
    "148254d4",
    "2bd18e2d",
    "880221bd",
    "b1e5a33a",
    "da47621c",
    "dd1012fc",
    "ddcf9fe3",
    "df7f61ee",
    "e3ec6bf4",
]


def load_commit_mapping() -> dict:
    """Load commit mapping from JSON file."""
    with open(COMMIT_MAPPING_FILE) as f:
        data = json.load(f)

    mapping = {}
    for commit in data["commits"]:
        short = commit["human_commit"][:8]
        mapping[short] = {
            "human_commit": commit["human_commit"],
            "base_commit": commit["base_commit"],
            "pr_number": commit.get("pr_number"),
            "subject": commit.get("subject", "Unknown")
        }
    return mapping


def find_agent_patch(commit_short: str, agent: str) -> Path | None:
    """Find the patch file for a commit and agent variant."""
    agent_path = AGENT_CONFIGS.get(agent)
    if not agent_path or not agent_path.exists():
        return None

    for folder in agent_path.iterdir():
        if folder.is_dir() and commit_short in folder.name:
            patch_file = folder / "model_patch.diff"
            if patch_file.exists():
                return patch_file
    return None


def run_benchmark_in_container(
    image: str,
    variant: str,
    patch_path: str | None,
    num_prompts: int = 100,
    timeout: int = 600
) -> dict:
    """
    Run a benchmark inside a Docker container.

    Uses bench_serving with a server started inside the container.
    """

    # Build the benchmark script
    patch_cmd = ""
    if patch_path:
        docker_patch = patch_path.replace("/root/OmniPerf-Bench", "/workspace")
        patch_cmd = f"""
echo "Applying {variant} patch..."
cd /sgl-workspace/sglang
git apply {docker_patch} 2>&1 || echo "Patch apply had issues"
"""

    script = f'''#!/bin/bash
set -e

export PYTHONPATH=/sgl-workspace/sglang/python:$PYTHONPATH

{patch_cmd}

# Start server in background
echo "Starting server..."
python3 -m sglang.launch_server \\
    --model-path TinyLlama/TinyLlama-1.1B-Chat-v1.0 \\
    --port 30000 \\
    --host 0.0.0.0 \\
    --trust-remote-code \\
    --mem-fraction-static 0.8 \\
    > /tmp/server.log 2>&1 &

SERVER_PID=$!

# Wait for server to be ready
echo "Waiting for server..."
for i in $(seq 1 120); do
    if curl -s http://localhost:30000/health > /dev/null 2>&1; then
        echo "Server ready!"
        break
    fi
    sleep 2
done

# Check if server is running
if ! kill -0 $SERVER_PID 2>/dev/null; then
    echo "SERVER FAILED TO START"
    cat /tmp/server.log
    exit 1
fi

# Run benchmark
echo "Running benchmark..."
python3 -m sglang.bench_serving \\
    --backend sglang \\
    --base-url http://localhost:30000 \\
    --model TinyLlama/TinyLlama-1.1B-Chat-v1.0 \\
    --dataset-name random \\
    --random-input-len 512 \\
    --random-output-len 128 \\
    --num-prompts {num_prompts} \\
    --output-file /tmp/benchmark_result.json \\
    2>&1

# Output results
echo ""
echo "=== BENCHMARK RESULTS ==="
cat /tmp/benchmark_result.json

# Cleanup
kill $SERVER_PID 2>/dev/null || true
'''

    # Run Docker container
    cmd = [
        "docker", "run", "--rm", "--gpus", "all",
        "-v", "/root/OmniPerf-Bench:/workspace",
        "-e", f"HF_TOKEN={Path('/root/.cache/huggingface/token').read_text().strip() if Path('/root/.cache/huggingface/token').exists() else ''}",
        image,
        "bash", "-c", script
    ]

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        output = result.stdout + result.stderr

        # Extract benchmark results
        try:
            json_start = output.find("=== BENCHMARK RESULTS ===")
            if json_start != -1:
                json_text = output[json_start + len("=== BENCHMARK RESULTS ==="):].strip()
                # Find the JSON object
                brace_start = json_text.find("{")
                brace_end = json_text.rfind("}") + 1
                if brace_start != -1 and brace_end > brace_start:
                    results = json.loads(json_text[brace_start:brace_end])
                    return {
                        "status": "success",
                        "variant": variant,
                        "results": results,
                        "raw_output": output[-2000:]
                    }
        except json.JSONDecodeError:
            pass

        return {
            "status": "failed",
            "variant": variant,
            "error": "Could not parse results",
            "raw_output": output[-2000:]
        }

    except subprocess.TimeoutExpired:
        return {
            "status": "timeout",
            "variant": variant,
            "error": f"Timed out after {timeout}s"
        }
    except Exception as e:
        return {
            "status": "error",
            "variant": variant,
            "error": str(e)
        }


def run_3way_benchmark(
    commit_short: str,
    agents: list[str],
    dry_run: bool = False,
    num_prompts: int = 100
) -> dict:
    """Run 3-way benchmark for a commit using pre-built images."""

    mapping = load_commit_mapping()

    if commit_short not in mapping:
        print(f"ERROR: Commit {commit_short} not found in mapping")
        return {"error": f"Commit {commit_short} not found"}

    info = mapping[commit_short]
    full_commit = info["human_commit"]
    base_commit = info["base_commit"]
    subject = info.get("subject", "Unknown")
    pr_number = info.get("pr_number", "?")

    print(f"\n{'='*60}")
    print(f"Commit: {commit_short}")
    print(f"Subject: {subject[:50]}...")
    print(f"PR: #{pr_number}")
    print(f"{'='*60}")

    # Check if image exists
    image = DOCKER_IMAGE_TEMPLATE.format(full_commit=full_commit)

    # Find agent patches
    agent_patches = {}
    for agent in agents:
        patch = find_agent_patch(commit_short, agent)
        if patch:
            agent_patches[agent] = str(patch)
            print(f"Found {agent} patch")
        else:
            print(f"WARNING: No patch for {agent}")

    if dry_run:
        print(f"\n[DRY RUN] Would benchmark with image: {image}")
        print(f"[DRY RUN] Variants: baseline, human, {', '.join(agent_patches.keys())}")
        return {"status": "dry_run", "commit": commit_short}

    results = {
        "commit": commit_short,
        "full_commit": full_commit,
        "base_commit": base_commit,
        "subject": subject,
        "pr_number": pr_number,
        "timestamp": datetime.now().isoformat(),
        "image": image,
        "benchmarks": []
    }

    # 1. Human benchmark (code as-is in image)
    print(f"\n[1/N] Running HUMAN benchmark...")
    human_result = run_benchmark_in_container(image, "human", None, num_prompts)
    results["benchmarks"].append(human_result)

    if human_result.get("status") == "success":
        print(f"  HUMAN: {human_result['results'].get('output_throughput', 'N/A')} tokens/s")
    else:
        print(f"  HUMAN: FAILED - {human_result.get('error', 'Unknown error')}")

    # 2. Agent benchmarks
    for i, (agent, patch_path) in enumerate(agent_patches.items(), start=2):
        print(f"\n[{i}/N] Running {agent.upper()} benchmark...")
        agent_result = run_benchmark_in_container(image, agent, patch_path, num_prompts)
        results["benchmarks"].append(agent_result)

        if agent_result.get("status") == "success":
            print(f"  {agent.upper()}: {agent_result['results'].get('output_throughput', 'N/A')} tokens/s")
        else:
            print(f"  {agent.upper()}: FAILED - {agent_result.get('error', 'Unknown error')}")

    # Save results
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    output_file = RESULTS_DIR / f"{commit_short}_prebuilt.json"
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {output_file}")

    return results


def main():
    parser = argparse.ArgumentParser(description="SGLang 3-Way Benchmark (Pre-built Images)")
    parser.add_argument("--commit", type=str, help="Specific commit (8-char)")
    parser.add_argument("--all", action="store_true", help="Benchmark all pre-built commits")
    parser.add_argument("--agents", type=str, default="claude_code,codex,trae_gpt5",
                        help="Comma-separated agents")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be done")
    parser.add_argument("--num-prompts", type=int, default=100, help="Number of prompts")
    parser.add_argument("--list", action="store_true", help="List available commits")

    args = parser.parse_args()

    if args.list:
        print("\nPre-built commits available:")
        mapping = load_commit_mapping()
        for commit in PREBUILT_COMMITS:
            info = mapping.get(commit, {})
            subject = info.get("subject", "Unknown")[:40]
            pr = info.get("pr_number", "?")
            print(f"  {commit}: PR#{pr} {subject}")
        return

    agents = [a.strip() for a in args.agents.split(",")]

    commits_to_run = []
    if args.all:
        commits_to_run = PREBUILT_COMMITS
    elif args.commit:
        commits_to_run = [args.commit]
    else:
        parser.print_help()
        return

    all_results = []
    for commit in commits_to_run:
        try:
            result = run_3way_benchmark(commit, agents, args.dry_run, args.num_prompts)
            all_results.append(result)
        except Exception as e:
            print(f"ERROR processing {commit}: {e}")
            all_results.append({"commit": commit, "error": str(e)})

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    for result in all_results:
        commit = result.get("commit", "?")
        if "error" in result:
            print(f"\n{commit}: ERROR - {result['error']}")
        elif result.get("status") == "dry_run":
            print(f"\n{commit}: DRY RUN")
        else:
            print(f"\n{commit}:")
            for bench in result.get("benchmarks", []):
                variant = bench.get("variant", "?")
                if bench.get("status") == "success":
                    throughput = bench.get("results", {}).get("output_throughput", "N/A")
                    print(f"  {variant}: {throughput} tokens/s")
                else:
                    print(f"  {variant}: FAILED")


if __name__ == "__main__":
    main()
