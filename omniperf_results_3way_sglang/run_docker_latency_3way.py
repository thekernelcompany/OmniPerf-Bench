#!/usr/bin/env python3
"""
Docker-based 3-Way Latency Benchmark for SGLang

Uses bench_latency.py or bench_one_batch.py inside Docker containers
with git overlay approach for proper baseline/human/agent comparison.

Usage:
    python run_docker_latency_3way.py --commit 148254d4 --dry-run
    python run_docker_latency_3way.py --commit 148254d4
    python run_docker_latency_3way.py --all
"""

import argparse
import json
import subprocess
import time
import re
from datetime import datetime
from pathlib import Path

# Configuration
RESULTS_DIR = Path("/root/OmniPerf-Bench/omniperf_results_3way_sglang/docker_benchmark_results")
COMMIT_MAPPING_FILE = Path("/root/OmniPerf-Bench/src/benchmark/fixes/sglang_commit_mapping.json")
AGENT_BASE_DIR = Path("/root/OmniPerf-Bench/perf-agents-bench/state/runs/sglang")
HF_CACHE = Path("/root/.cache/huggingface")
SGLANG_REPO = "https://github.com/sgl-project/sglang.git"

# Docker images available (priority order)
DOCKER_IMAGES = {
    "148254d4": "shikhar481/sglang-images:v05-improved-148254d4db8b",
    "187b85b7": "shikhar481/sglang-images:v05-improved-187b85b7f384",
    "dd1012fc": "shikhar481/sglang-images:v04x-fixed-dd1012fcbe2a",
    "2a754e57": "sglang-rebuilt-2a754e57",
    "c087ddd6": "shikhar481/sglang-images:c087ddd6",
}

# Agent configurations
AGENT_CONFIGS = {
    "claude_code": AGENT_BASE_DIR / "claude_code/default/2025-12-23_06-28-44",
    "codex": AGENT_BASE_DIR / "codex/gpt-5/389be848",
    "trae_gpt5": AGENT_BASE_DIR / "trae/gpt-5/2025-11-14_21-05-32",
    "trae_sonnet45": AGENT_BASE_DIR / "trae/claude-sonnet-45/2025-11-28_15-26-15",
}

# Target commits
TARGET_COMMITS = [
    "187b85b7", "6b231325", "6cb00c63", "148254d4", "2bd18e2d", "2a754e57",
    "880221bd", "b1e5a33a", "da47621c", "dd1012fc", "ddcf9fe3", "df7f61ee", "e3ec6bf4"
]


def load_commit_mapping():
    """Load commit mapping from JSON."""
    with open(COMMIT_MAPPING_FILE) as f:
        data = json.load(f)
    mapping = {}
    for i, commit in enumerate(data["commits"], 1):
        short = commit["human_commit"][:8]
        mapping[short] = {
            "human_commit": commit["human_commit"],
            "base_commit": commit["base_commit"],
            "pr_number": commit.get("pr_number"),
            "subject": commit.get("subject", "Unknown"),
            "codex_index": i,
        }
    return mapping


def get_docker_image(commit_short: str) -> str | None:
    """Get Docker image for a commit."""
    # Check our known working images
    if commit_short in DOCKER_IMAGES:
        return DOCKER_IMAGES[commit_short]

    # Try to find in other repos
    mapping = load_commit_mapping()
    if commit_short not in mapping:
        return None

    full_commit = mapping[commit_short]["human_commit"]
    short12 = full_commit[:12]

    # Check various image patterns
    candidates = [
        f"shikhar481/sglang-images:v05-improved-{short12}",
        f"shikhar481/sglang-images:v04x-fixed-{short12}",
        f"shikhar481/sglang-images:{commit_short}",
        f"ayushnangia16/nvidia-sglang-docker:{full_commit}",
    ]

    for img in candidates:
        result = subprocess.run(
            ["docker", "images", "-q", img],
            capture_output=True, text=True, timeout=10
        )
        if result.stdout.strip():
            return img

    return None


def find_agent_patch(commit_short: str, agent: str) -> Path | None:
    """Find agent patch file."""
    agent_path = AGENT_CONFIGS.get(agent)
    if not agent_path or not agent_path.exists():
        return None

    mapping = load_commit_mapping()

    if agent == "codex":
        idx = mapping.get(commit_short, {}).get("codex_index")
        if idx:
            patch_file = agent_path / f"sglang_core-{idx:04d}" / "model_patch.diff"
            if patch_file.exists():
                return patch_file
        return None

    for folder in agent_path.iterdir():
        if folder.is_dir() and commit_short in folder.name:
            patch_file = folder / "model_patch.diff"
            if patch_file.exists():
                return patch_file
    return None


def parse_latency_output(output: str) -> dict:
    """Parse bench_latency/bench_one_batch output."""
    metrics = {}

    # Parse prefill throughput
    matches = re.findall(r'Prefill\.\s*latency:.*?throughput:\s*([\d.]+)\s*token', output)
    if matches:
        metrics["prefill_throughput"] = float(matches[-1])

    # Parse decode throughput
    matches = re.findall(r'Decode\..*?avg throughput:\s*([\d.]+)\s*token', output)
    if matches:
        metrics["decode_throughput"] = float(matches[-1])

    # Parse total throughput
    matches = re.findall(r'Total\.\s*latency:.*?throughput:\s*([\d.]+)\s*token', output)
    if matches:
        metrics["total_throughput"] = float(matches[-1])

    return metrics


def run_docker_3way_benchmark(
    commit_short: str,
    docker_image: str,
    agents: list[str],
    model: str = "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    batch_size: int = 4,
    input_len: int = 1024,
    output_len: int = 256,
    timeout: int = 1800,
    dry_run: bool = False,
) -> dict:
    """Run 3-way benchmark inside Docker using git overlay."""

    mapping = load_commit_mapping()
    if commit_short not in mapping:
        return {"error": f"Commit {commit_short} not found"}

    info = mapping[commit_short]
    human_commit = info["human_commit"]
    base_commit = info["base_commit"]
    subject = info.get("subject", "Unknown")
    pr_number = info.get("pr_number", "?")

    print(f"\n{'='*60}")
    print(f"3-WAY BENCHMARK: {commit_short}")
    print(f"{'='*60}")
    print(f"  Subject: {subject[:50]}...")
    print(f"  PR: #{pr_number}")
    print(f"  Human: {human_commit[:12]}")
    print(f"  Base: {base_commit[:12]}")
    print(f"  Docker: {docker_image}")

    # Find agent patches
    agent_patches = {}
    for agent in agents:
        patch = find_agent_patch(commit_short, agent)
        if patch:
            agent_patches[agent] = patch
            print(f"  {agent} patch: Found")
        else:
            print(f"  {agent} patch: Not found")

    if dry_run:
        print("\n[DRY RUN] Would benchmark:")
        print(f"  - baseline: {base_commit[:12]}")
        print(f"  - human: {human_commit[:12]}")
        for agent in agent_patches:
            print(f"  - {agent}: {base_commit[:12]} + patch")
        return {"status": "dry_run"}

    results = {
        "commit": commit_short,
        "human_commit": human_commit,
        "base_commit": base_commit,
        "subject": subject,
        "pr_number": pr_number,
        "timestamp": datetime.now().isoformat(),
        "bench_args": {
            "batch_size": batch_size,
            "input_len": input_len,
            "output_len": output_len,
            "model": model,
        },
        "benchmarks": [],
    }

    # Build the variants to benchmark
    variants = [
        ("baseline", base_commit, None),
        ("human", human_commit, None),
    ]
    for agent, patch_path in agent_patches.items():
        variants.append((agent, base_commit, patch_path))

    # Create the comprehensive Docker command that does all benchmarks
    hf_token = ""
    token_file = HF_CACHE / "token"
    if token_file.exists():
        hf_token = token_file.read_text().strip()

    docker_script = f'''
set -e
export GLOO_SOCKET_IFNAME=lo

echo "=== 3-WAY BENCHMARK ==="
echo "Human: {human_commit[:8]}"
echo "Base: {base_commit[:8]}"

# Check GPU
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | head -1 | xargs -I{{}} echo "GPU: {{}}"

# Clone sglang for overlay
git clone --depth 1 {SGLANG_REPO} /sglang-overlay 2>/dev/null || true
cd /sglang-overlay

# Ensure we have proper Python path
export PYTHONPATH="/sglang-overlay/python:$PYTHONPATH"

# Function to run benchmark
run_bench() {{
    local variant=$1
    local commit=$2
    echo ""
    echo "=== $variant: $commit ==="

    git checkout . 2>/dev/null || true
    git clean -fd 2>/dev/null || true
    git fetch origin $commit --depth 1 2>/dev/null || true
    git checkout $commit 2>/dev/null || git checkout FETCH_HEAD 2>/dev/null || echo "Checkout failed, using current"

    # Show current commit
    echo "At: $(git log --oneline -1 2>/dev/null || echo 'unknown')"

    # Apply patch if provided
    if [ -n "$3" ] && [ -f "$3" ]; then
        echo "Applying patch: $3"
        git apply "$3" 2>/dev/null || patch -p1 < "$3" 2>/dev/null || echo "Patch failed"
    fi

    # Detect benchmark script
    if [ -f "python/sglang/bench_one_batch.py" ]; then
        BENCH_CMD="python3 -m sglang.bench_one_batch"
    elif [ -f "python/sglang/bench_latency.py" ]; then
        BENCH_CMD="python3 -m sglang.bench_latency --disable-flashinfer"
    else
        BENCH_CMD="python3 -m sglang.bench_one_batch"
    fi

    # Run benchmark
    $BENCH_CMD \\
        --model-path {model} \\
        --batch-size {batch_size} \\
        --input-len {input_len} \\
        --output-len {output_len} \\
        --load-format dummy \\
        --trust-remote-code \\
        --mem-fraction-static 0.8 \\
        2>&1 || echo "BENCHMARK_ERROR"
}}

# Track results
RESULTS_JSON="["

'''

    # Add benchmark commands for each variant
    for i, (variant, commit, patch_path) in enumerate(variants):
        patch_mount = ""
        patch_arg = ""
        if patch_path:
            patch_mount = f"-v {patch_path}:/patches/{variant}.diff:ro"
            patch_arg = f"/patches/{variant}.diff"

        docker_script += f'''
echo ""
echo "=== {variant.upper()}: {'' if not patch_path else 'base + ' + variant + ' patch'} ==="
git checkout . 2>/dev/null || true
git clean -fd 2>/dev/null || true
git fetch origin {commit} --depth 1 2>/dev/null || true
git checkout {commit} 2>/dev/null || git checkout FETCH_HEAD 2>/dev/null || echo "Checkout failed"
echo "At: $(git log --oneline -1 2>/dev/null || echo 'unknown'){' + ' + variant + ' patch' if patch_path else ''}"
'''

        if patch_path:
            docker_script += f'''
if [ -f "/patches/{variant}.diff" ]; then
    git apply "/patches/{variant}.diff" 2>/dev/null || patch -p1 < "/patches/{variant}.diff" 2>/dev/null || echo "Patch apply issue"
fi
'''

        docker_script += f'''
# Detect and run benchmark
if [ -f "python/sglang/bench_one_batch.py" ]; then
    BENCH_CMD="python3 -m sglang.bench_one_batch"
elif [ -f "python/sglang/bench_latency.py" ]; then
    BENCH_CMD="python3 -m sglang.bench_latency --disable-flashinfer"
else
    BENCH_CMD="python3 -m sglang.bench_one_batch"
fi

OUTPUT_FILE="/tmp/bench_{variant}.txt"
$BENCH_CMD \\
    --model-path {model} \\
    --batch-size {batch_size} \\
    --input-len {input_len} \\
    --output-len {output_len} \\
    --load-format dummy \\
    --trust-remote-code \\
    --mem-fraction-static 0.8 \\
    2>&1 | tee $OUTPUT_FILE

# Extract metrics
TOTAL=$(grep "^Total\\." $OUTPUT_FILE | tail -1 | grep -oP 'throughput:\\s*\\K[\\d.]+' || echo "0")
DECODE=$(grep "Decode.*avg throughput" $OUTPUT_FILE | tail -1 | grep -oP 'avg throughput:\\s*\\K[\\d.]+' || echo "0")
PREFILL=$(grep "^Prefill\\." $OUTPUT_FILE | tail -1 | grep -oP 'throughput:\\s*\\K[\\d.]+' || echo "0")

# Output JSON
echo '{{"variant": "{variant}", "commit": "{commit[:9]}", "total_throughput_tokens_per_sec": '$TOTAL', "decode_throughput_tokens_per_sec": '$DECODE', "prefill_throughput_tokens_per_sec": '$PREFILL', "batch_size": {batch_size}, "input_len": {input_len}, "output_len": {output_len}}}'
'''

    docker_script += '''
echo ""
echo "=== BENCHMARK COMPLETE ==="
'''

    # Build Docker command with patch mounts
    docker_cmd = [
        "docker", "run", "--rm",
        "--gpus", "all",
        "-e", f"HF_TOKEN={hf_token}",
        "-e", f"HUGGING_FACE_HUB_TOKEN={hf_token}",
        "-v", f"{HF_CACHE}:/root/.cache/huggingface",
        "--shm-size=16g",
    ]

    # Add patch mounts
    for agent, patch_path in agent_patches.items():
        docker_cmd.extend(["-v", f"{patch_path}:/patches/{agent}.diff:ro"])

    docker_cmd.extend([
        "--entrypoint", "bash",
        docker_image,
        "-c", docker_script
    ])

    print(f"\n[RUNNING] Docker 3-way benchmark...")
    start_time = time.time()

    try:
        result = subprocess.run(
            docker_cmd,
            capture_output=True,
            text=True,
            timeout=timeout
        )

        output = result.stdout + result.stderr
        duration = time.time() - start_time

        results["raw_output"] = output[-10000:]
        results["duration_s"] = duration

        # Parse individual benchmark results from output
        for variant, commit, _ in variants:
            variant_data = {
                "variant": variant,
                "commit": commit[:9],
            }

            # Find the JSON line for this variant
            json_pattern = rf'\{{"variant": "{variant}"[^}}]+\}}'
            match = re.search(json_pattern, output)
            if match:
                try:
                    parsed = json.loads(match.group())
                    variant_data.update(parsed)
                except:
                    pass

            results["benchmarks"].append(variant_data)

        # Print results
        print(f"\n{'='*60}")
        print("RESULTS")
        print(f"{'='*60}")
        for bench in results["benchmarks"]:
            variant = bench.get("variant", "?")
            total = bench.get("total_throughput_tokens_per_sec", 0)
            print(f"  {variant}: {total:.2f} tok/s")

    except subprocess.TimeoutExpired:
        results["error"] = f"Timeout after {timeout}s"
        print(f"ERROR: Timeout")
    except Exception as e:
        results["error"] = str(e)
        print(f"ERROR: {e}")

    # Save results
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    output_file = RESULTS_DIR / f"{commit_short}_3way.json"
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved: {output_file}")

    return results


def main():
    parser = argparse.ArgumentParser(description="Docker 3-Way Latency Benchmark")
    parser.add_argument("--commit", type=str, help="Specific commit (8-char)")
    parser.add_argument("--all", action="store_true", help="Benchmark all available commits")
    parser.add_argument("--agents", type=str, default="claude_code,codex,trae_gpt5,trae_sonnet45",
                        help="Comma-separated agents")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be done")
    parser.add_argument("--list", action="store_true", help="List commits with Docker images")
    parser.add_argument("--timeout", type=int, default=1800, help="Timeout in seconds")
    args = parser.parse_args()

    if args.list:
        print("\nCommits with Docker images:")
        for commit in TARGET_COMMITS:
            image = get_docker_image(commit)
            status = "✓" if image else "✗"
            print(f"  {status} {commit}: {image or 'NO IMAGE'}")
        return

    agents = [a.strip() for a in args.agents.split(",")]

    if args.all:
        commits_to_run = []
        for commit in TARGET_COMMITS:
            image = get_docker_image(commit)
            if image:
                commits_to_run.append((commit, image))

        if not commits_to_run:
            print("ERROR: No commits have Docker images available")
            return

        print(f"\nWill benchmark {len(commits_to_run)} commits with Docker images")
    elif args.commit:
        image = get_docker_image(args.commit)
        if not image:
            print(f"ERROR: No Docker image found for {args.commit}")
            print("\nAvailable images:")
            for c, i in DOCKER_IMAGES.items():
                print(f"  {c}: {i}")
            return
        commits_to_run = [(args.commit, image)]
    else:
        parser.print_help()
        return

    all_results = []
    for commit, image in commits_to_run:
        result = run_docker_3way_benchmark(
            commit, image, agents,
            dry_run=args.dry_run,
            timeout=args.timeout
        )
        all_results.append(result)

    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
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
                total = bench.get("total_throughput_tokens_per_sec", 0)
                print(f"  {variant}: {total:.2f} tok/s")


if __name__ == "__main__":
    main()
