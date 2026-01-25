#!/usr/bin/env python3
"""
SGLang 3-Way Benchmark Runner using Python Overlay Approach

This script runs 3-way benchmarks (baseline, human, agent) for SGLang commits
by using a working base Docker image and overlaying Python code via PYTHONPATH.

The approach:
1. Use sglang-rebuilt-2a754e57 Docker image (has all dependencies for June 2024 era)
2. Clone SGLang repo inside container
3. Fetch specific commits via git fetch --depth=1
4. Use PYTHONPATH to overlay the Python code
5. Run bench_latency.py for each variant

Usage:
    python run_3way_overlay.py --commit 2a754e57 --dry-run
    python run_3way_overlay.py --commit 148254d4 --agents claude_code,codex
    python run_3way_overlay.py --all
"""

import argparse
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path

# Paths
RESULTS_DIR = Path("/root/OmniPerf-Bench/omniperf_results_3way_sglang/overlay_benchmark_results")
COMMIT_MAPPING_FILE = Path("/root/OmniPerf-Bench/src/benchmark/fixes/sglang_commit_mapping.json")
AGENT_BASE_DIR = Path("/root/OmniPerf-Bench/perf-agents-bench/state/runs/sglang")

# Docker image to use (has SGLang 0.1.17 era dependencies)
DOCKER_IMAGE = "sglang-rebuilt-2a754e57"

# Agent configurations
AGENT_CONFIGS = {
    "claude_code": {
        "path": AGENT_BASE_DIR / "claude_code/default/2025-12-23_06-28-44",
        "prefix": "sglang_"
    },
    "codex": {
        "path": AGENT_BASE_DIR / "codex/gpt-5/389be848",
        "prefix": "sglang_core-"
    },
    "trae_gpt5": {
        "path": AGENT_BASE_DIR / "trae/gpt-5/2025-11-14_21-05-32",
        "prefix": "sglang_"
    },
    "trae_sonnet45": {
        "path": AGENT_BASE_DIR / "trae/claude-sonnet-45/2025-11-28_15-26-15",
        "prefix": "sglang_"
    }
}

# All 17 commits to benchmark (14 with Docker images + 3 without)
# Note: Some may fail due to dependency/code structure incompatibility
ALL_BENCHMARK_COMMITS = [
    # 14 commits with Docker images
    "187b85b7",  # [PD] Optimize custom mem pool usage
    "6b231325",  # [PD Perf] replace Queue to FastQueue
    "6cb00c63",  # [PD] Optimize time out logic
    "148254d4",  # Improve moe reduce sum kernel
    "2bd18e2d",  # Memory pool: Minor optimize
    "2a754e57",  # 2x perf for large prefill (VERIFIED WORKING)
    "880221bd",  # Revert transfer batch
    "b1e5a33a",  # Eliminate stream sync for LoRA
    "c087ddd6",  # Refine pre_reorder_triton_kernel
    "da47621c",  # Minor speedup topk postprocessing
    "dd1012fc",  # [PD] Fix potential perf spike
    "ddcf9fe3",  # Optimize triton attention mask
    "df7f61ee",  # Speed up rebalancing
    "e3ec6bf4",  # Minor speed up block_quant_dequant
    # 3 commits without Docker images (may still work with overlay)
    "4418f599",  # Fix FA3 DeepSeek prefill regression
    "2a413829",  # Add triton version config key
    "5e023301",  # [perf] dsv3 bmm fallback to bf16
]

# Commits verified to work with the rebuilt image (June-July 2024 era, PR < 1000)
JUNE_2024_COMMITS = [
    "09deb20d",  # PR#420: Optimize the memory usage of logits processor
    "1bf1cf19",  # PR#375: Reduce overhead when fork(1)
    "2a754e57",  # PR#579: 2x performance improvement for large prefill (VERIFIED)
    "564a898a",  # PR#619: Optimize mem indices management
    "6a2941f4",  # PR#625: Improve tensor parallel performance
    "6f560c76",  # PR#117: Improve the control of streaming
    "9216b106",  # PR#394: Improve performance when running with full parallel
    "ac971ff6",  # PR#658: perf: reduce ttft and itl with stream_interval 1
    "bb3a3b66",  # PR#137: Support Faster JSON decoding for llava
    "e822e590",  # PR#364: Optimize radix tree matching
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
    config = AGENT_CONFIGS.get(agent)
    if not config or not config["path"].exists():
        return None

    # Search for matching folder
    for folder in config["path"].iterdir():
        if folder.is_dir():
            # Check if commit is in folder name
            if commit_short in folder.name:
                patch_file = folder / "model_patch.diff"
                if patch_file.exists():
                    return patch_file
            # For codex, check prompt.json
            elif agent == "codex" and folder.name.startswith("sglang_core-"):
                prompt_file = folder / "prompt.json"
                if prompt_file.exists():
                    try:
                        with open(prompt_file) as f:
                            pdata = json.load(f)
                            human = pdata.get("commits", {}).get("human", "")
                            if human.startswith(commit_short):
                                patch_file = folder / "model_patch.diff"
                                if patch_file.exists():
                                    return patch_file
                    except:
                        pass
    return None


def generate_docker_script(
    human_commit: str,
    base_commit: str,
    agent_patches: dict,
    bench_args: dict
) -> str:
    """Generate the shell script to run inside Docker."""

    batch_size = bench_args.get("batch_size", 4)
    input_len = bench_args.get("input_len", 512)
    output_len = bench_args.get("output_len", 128)
    model = bench_args.get("model", "TinyLlama/TinyLlama-1.1B-Chat-v1.0")

    # Build agent patch commands - convert host path to Docker path
    agent_benchmarks = ""
    for agent, patch_path in agent_patches.items():
        docker_patch_path = patch_path.replace("/root/OmniPerf-Bench", "/workspace")
        agent_benchmarks += f'''
# {agent.upper()} benchmark
echo ""
echo "=== {agent.upper()}: base + {agent} patch ==="
git checkout $BASE_COMMIT 2>/dev/null
git checkout . 2>/dev/null
if [ -f "{docker_patch_path}" ]; then
    git apply "{docker_patch_path}" 2>&1
    echo "At: $(git log --oneline -1) + {agent} patch"
    run_bench "{agent}" >> /tmp/results.json
    echo "," >> /tmp/results.json
else
    echo "Patch not found: {docker_patch_path}"
fi
'''

    script = f'''#!/bin/bash
echo "=== 3-WAY BENCHMARK ==="
echo "Human: {human_commit[:8]}"
echo "Base: {base_commit[:8]}"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"

HUMAN_COMMIT="{human_commit}"
BASE_COMMIT="{base_commit}"

# Clone and setup
cd /tmp
rm -rf sglang-overlay
git clone --single-branch --branch main https://github.com/sgl-project/sglang.git sglang-overlay 2>&1 | tail -2
cd sglang-overlay

# Fetch commits
git fetch origin $BASE_COMMIT --depth=1 2>/dev/null
git fetch origin $HUMAN_COMMIT --depth=1 2>/dev/null

# Verify commits
git cat-file -t $BASE_COMMIT 2>/dev/null || {{ echo "ERROR: Base commit not found"; exit 1; }}
git cat-file -t $HUMAN_COMMIT 2>/dev/null || {{ echo "ERROR: Human commit not found"; exit 1; }}

export PYTHONPATH=/tmp/sglang-overlay/python:$PYTHONPATH

# Initialize results file
echo "[" > /tmp/results.json

run_bench() {{
    local name=$1

    # Capture benchmark output
    local output=$(python3 -m sglang.bench_latency \\
        --model-path {model} \\
        --batch-size {batch_size} \\
        --input-len {input_len} \\
        --output-len {output_len} \\
        --trust-remote-code 2>&1)

    # Extract metrics - handle both "token/s" formats
    local total_throughput=$(echo "$output" | grep "^Total." | tail -1 | awk '{{print $(NF-1)}}')
    local decode_throughput=$(echo "$output" | grep "Decode.*avg throughput" | tail -1 | awk '{{print $(NF-1)}}')
    local prefill_throughput=$(echo "$output" | grep "^Prefill." | tail -1 | awk '{{print $(NF-1)}}')

    # Output JSON record
    echo "{{"
    echo "  \\"variant\\": \\"$name\\","
    echo "  \\"commit\\": \\"$(git log --oneline -1 | cut -d\\  -f1)\\","
    echo "  \\"total_throughput_tokens_per_sec\\": ${{total_throughput:-0}},"
    echo "  \\"decode_throughput_tokens_per_sec\\": ${{decode_throughput:-0}},"
    echo "  \\"prefill_throughput_tokens_per_sec\\": ${{prefill_throughput:-0}},"
    echo "  \\"batch_size\\": {batch_size},"
    echo "  \\"input_len\\": {input_len},"
    echo "  \\"output_len\\": {output_len}"
    echo "}}"
}}

# BASELINE benchmark
echo ""
echo "=== BASELINE: base commit ==="
git checkout $BASE_COMMIT 2>/dev/null
echo "At: $(git log --oneline -1)"
run_bench "baseline" >> /tmp/results.json
echo "," >> /tmp/results.json

git checkout . 2>/dev/null

# HUMAN benchmark
echo ""
echo "=== HUMAN: human commit ==="
git checkout $HUMAN_COMMIT 2>/dev/null
echo "At: $(git log --oneline -1)"
run_bench "human" >> /tmp/results.json
{agent_benchmarks}

# Close JSON array (remove trailing comma and close)
sed -i '$ s/,$//' /tmp/results.json
echo "]" >> /tmp/results.json

# Output results
echo ""
echo "=== RESULTS JSON ==="
cat /tmp/results.json

echo ""
echo "=== BENCHMARK COMPLETE ==="
'''
    return script


def run_3way_benchmark(
    commit_short: str,
    agents: list[str],
    bench_args: dict,
    dry_run: bool = False
) -> dict:
    """Run 3-way benchmark for a commit."""

    mapping = load_commit_mapping()

    if commit_short not in mapping:
        print(f"ERROR: Commit {commit_short} not found in mapping")
        return {"error": f"Commit {commit_short} not found"}

    info = mapping[commit_short]
    human_commit = info["human_commit"]
    base_commit = info["base_commit"]
    subject = info.get("subject", "Unknown")
    pr_number = info.get("pr_number", "?")

    print(f"\n{'='*60}")
    print(f"Commit: {commit_short}")
    print(f"Subject: {subject[:50]}...")
    print(f"PR: #{pr_number}")
    print(f"Base: {base_commit[:12]}")
    print(f"Human: {human_commit[:12]}")
    print(f"{'='*60}")

    # Find agent patches
    agent_patches = {}
    for agent in agents:
        patch = find_agent_patch(commit_short, agent)
        if patch:
            agent_patches[agent] = str(patch)
            print(f"Found {agent} patch: {patch}")
        else:
            print(f"WARNING: No patch found for {agent}")

    if dry_run:
        print("\n[DRY RUN] Would run benchmarks with:")
        print(f"  - Docker image: {DOCKER_IMAGE}")
        print(f"  - Variants: baseline, human, {', '.join(agent_patches.keys())}")
        print(f"  - Bench args: {bench_args}")
        return {"status": "dry_run", "commit": commit_short}

    # Generate and run Docker command
    script = generate_docker_script(human_commit, base_commit, agent_patches, bench_args)

    # Save script for debugging
    script_file = RESULTS_DIR / f"script_{commit_short}.sh"
    script_file.parent.mkdir(parents=True, exist_ok=True)
    script_file.write_text(script)

    # Run Docker
    hf_token = ""
    token_file = Path("/root/.cache/huggingface/token")
    if token_file.exists():
        hf_token = token_file.read_text().strip()

    cmd = [
        "docker", "run", "--rm", "--gpus", "all",
        "-v", "/root/OmniPerf-Bench:/workspace",
        "-e", f"HF_TOKEN={hf_token}",
        DOCKER_IMAGE,
        "bash", "-c", script
    ]

    print("\nRunning benchmark...")
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)

    # Parse results from output
    output = result.stdout + result.stderr

    # Extract JSON results
    results = []
    try:
        json_start = output.find("=== RESULTS JSON ===")
        json_end = output.find("=== BENCHMARK COMPLETE ===")
        if json_start != -1 and json_end != -1:
            json_text = output[json_start + len("=== RESULTS JSON ==="):json_end].strip()
            # Clean up JSON - fix missing commas between objects
            json_text = json_text.replace("}\n{", "},\n{")
            if json_text.startswith("["):
                results = json.loads(json_text)
    except Exception as e:
        print(f"Failed to parse results JSON: {e}")
        # Try to extract individual results from raw output
        import re
        for match in re.finditer(r'\{[^}]+\}', output):
            try:
                obj = json.loads(match.group())
                if "variant" in obj:
                    results.append(obj)
            except:
                pass

    # Build final result
    final_result = {
        "commit": commit_short,
        "human_commit": human_commit,
        "base_commit": base_commit,
        "subject": subject,
        "pr_number": pr_number,
        "timestamp": datetime.now().isoformat(),
        "bench_args": bench_args,
        "benchmarks": results,
        "raw_output": output[-5000:] if len(output) > 5000 else output
    }

    # Save results
    output_file = RESULTS_DIR / f"{commit_short}_3way.json"
    with open(output_file, 'w') as f:
        json.dump(final_result, f, indent=2)
    print(f"\nResults saved to: {output_file}")

    return final_result


def main():
    parser = argparse.ArgumentParser(description="SGLang 3-Way Benchmark Runner (Overlay)")
    parser.add_argument("--commit", type=str, help="Specific commit to benchmark (8-char hash)")
    parser.add_argument("--all", action="store_true", help="Benchmark all 17 commits (most will fail)")
    parser.add_argument("--june2024", action="store_true", help="Benchmark only June 2024 era commits (10 commits, compatible with rebuilt image)")
    parser.add_argument("--agents", type=str, default="claude_code",
                        help="Comma-separated list of agent variants")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be done")
    parser.add_argument("--batch-size", type=int, default=4, help="Benchmark batch size")
    parser.add_argument("--input-len", type=int, default=512, help="Input sequence length")
    parser.add_argument("--output-len", type=int, default=128, help="Output sequence length")
    parser.add_argument("--model", type=str, default="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
                        help="Model to benchmark")
    parser.add_argument("--list-commits", action="store_true", help="List compatible commits")
    parser.add_argument("--check-patches", action="store_true", help="Check patch availability")

    args = parser.parse_args()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    agents = [a.strip() for a in args.agents.split(",")]
    bench_args = {
        "batch_size": args.batch_size,
        "input_len": args.input_len,
        "output_len": args.output_len,
        "model": args.model
    }

    if args.list_commits:
        print("\nAll 17 commits to benchmark:")
        mapping = load_commit_mapping()
        for i, commit in enumerate(ALL_BENCHMARK_COMMITS, 1):
            info = mapping.get(commit, {})
            subject = info.get("subject", "Unknown")[:45]
            pr = info.get("pr_number", "?")
            print(f"  {i:2}. {commit}: PR#{pr:<5} {subject}")
        return

    if args.check_patches:
        print("\nPatch availability for all 17 commits:")
        mapping = load_commit_mapping()
        print(f"{'#':<3} {'Commit':<10} " + " ".join(f"{a:<12}" for a in AGENT_CONFIGS.keys()))
        print("-" * 75)
        for i, commit in enumerate(ALL_BENCHMARK_COMMITS, 1):
            patches = []
            for agent in AGENT_CONFIGS.keys():
                patch = find_agent_patch(commit, agent)
                patches.append("✓" if patch else "✗")
            print(f"{i:<3} {commit:<10} " + " ".join(f"{p:<12}" for p in patches))
        return

    commits_to_run = []
    if args.all:
        commits_to_run = ALL_BENCHMARK_COMMITS
    elif args.june2024:
        commits_to_run = JUNE_2024_COMMITS
    elif args.commit:
        commits_to_run = [args.commit]
    else:
        parser.print_help()
        return

    all_results = []
    for commit in commits_to_run:
        try:
            result = run_3way_benchmark(commit, agents, bench_args, args.dry_run)
            all_results.append(result)
        except Exception as e:
            print(f"ERROR processing {commit}: {e}")
            all_results.append({"commit": commit, "error": str(e)})

    # Print summary
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
            print(f"\n{commit}: {result.get('subject', '')[:40]}")
            for bench in result.get("benchmarks", []):
                variant = bench.get("variant", "?")
                throughput = bench.get("total_throughput_tokens_per_sec", 0)
                print(f"  {variant}: {throughput:.2f} tokens/s")


if __name__ == "__main__":
    main()
