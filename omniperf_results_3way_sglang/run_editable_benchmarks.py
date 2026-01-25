#!/usr/bin/env python3
"""
SGLang 3-Way Benchmark Runner - Editable Install Approach

Uses 'pip install -e "python"' for each commit as per official SGLang install guide.
This properly resolves dependencies for each SGLang version.

Usage:
    python run_editable_benchmarks.py --commit 148254d4 --dry-run
    python run_editable_benchmarks.py --commit 148254d4
    python run_editable_benchmarks.py --all
"""

import argparse
import json
import os
import subprocess
import sys
import time
import re
from datetime import datetime
from pathlib import Path

# Paths
SGLANG_REPO = Path("/root/OmniPerf-Bench/omniperf_results_3way_sglang/sglang-editable")
RESULTS_DIR = Path("/root/OmniPerf-Bench/omniperf_results_3way_sglang/editable_benchmark_results")
COMMIT_MAPPING_FILE = Path("/root/OmniPerf-Bench/src/benchmark/fixes/sglang_commit_mapping.json")
AGENT_BASE_DIR = Path("/root/OmniPerf-Bench/perf-agents-bench/state/runs/sglang")

# Agent configurations
AGENT_CONFIGS = {
    "claude_code": AGENT_BASE_DIR / "claude_code/default/2025-12-23_06-28-44",
    "codex": AGENT_BASE_DIR / "codex/gpt-5/389be848",
    "trae_gpt5": AGENT_BASE_DIR / "trae/gpt-5/2025-11-14_21-05-32",
    "trae_sonnet45": AGENT_BASE_DIR / "trae/claude-sonnet-45/2025-11-28_15-26-15",
}

# Target commits (13 total, 2a754e57 already done)
TARGET_COMMITS = [
    "187b85b7",  # [PD] Optimize custom mem pool usage
    "6b231325",  # [PD Perf] replace Queue to FastQueue
    "6cb00c63",  # [PD] Optimize time out logic
    "148254d4",  # Improve moe reduce sum kernel
    "2bd18e2d",  # Memory pool: Minor optimize
    # "2a754e57",  # DONE - 2x perf for large prefill
    "880221bd",  # Revert transfer batch
    "b1e5a33a",  # Eliminate stream sync for LoRA
    "c087ddd6",  # Refine pre_reorder_triton_kernel
    "da47621c",  # Minor speedup topk postprocessing
    "dd1012fc",  # [PD] Fix potential perf spike
    "ddcf9fe3",  # Optimize triton attention mask
    "df7f61ee",  # Speed up rebalancing
    "e3ec6bf4",  # Minor speed up block_quant_dequant
]


def run_cmd(cmd, cwd=None, timeout=1800, capture=True, env=None):
    """Run a command and return output."""
    try:
        full_env = os.environ.copy()
        if env:
            full_env.update(env)
        result = subprocess.run(
            cmd,
            shell=True,
            cwd=cwd,
            capture_output=capture,
            text=True,
            timeout=timeout,
            executable="/bin/bash",
            env=full_env
        )
        return result.returncode, result.stdout + result.stderr
    except subprocess.TimeoutExpired:
        return -1, "TIMEOUT"
    except Exception as e:
        return -1, str(e)


def load_commit_mapping():
    """Load commit mapping from JSON file."""
    with open(COMMIT_MAPPING_FILE) as f:
        data = json.load(f)

    mapping = {}
    for i, commit in enumerate(data["commits"], start=1):
        short = commit["human_commit"][:8]
        mapping[short] = {
            "human_commit": commit["human_commit"],
            "base_commit": commit["base_commit"],
            "pr_number": commit.get("pr_number"),
            "subject": commit.get("subject", "Unknown"),
            "index": i  # 1-indexed for codex
        }
    return mapping


def get_codex_index(commit_short: str) -> int | None:
    """Get the codex index for a commit (1-indexed)."""
    mapping = load_commit_mapping()
    info = mapping.get(commit_short)
    return info["index"] if info else None


def find_agent_patch(commit_short: str, agent: str) -> Path | None:
    """Find the patch file for a commit and agent variant."""
    agent_path = AGENT_CONFIGS.get(agent)
    if not agent_path or not agent_path.exists():
        return None

    # For codex, use index-based mapping (sglang_core-NNNN)
    if agent == "codex":
        idx = get_codex_index(commit_short)
        if idx:
            folder_name = f"sglang_core-{idx:04d}"
            patch_file = agent_path / folder_name / "model_patch.diff"
            if patch_file.exists():
                return patch_file
        return None

    # For other agents, look for commit hash in folder name
    for folder in agent_path.iterdir():
        if folder.is_dir() and commit_short in folder.name:
            patch_file = folder / "model_patch.diff"
            if patch_file.exists():
                return patch_file
    return None


def setup_repo():
    """Clone or reset the SGLang repository."""
    if SGLANG_REPO.exists():
        print(f"  Resetting existing repo: {SGLANG_REPO}")
        run_cmd("git reset --hard HEAD", cwd=SGLANG_REPO)
        run_cmd("git clean -fdx", cwd=SGLANG_REPO)
        run_cmd("git fetch origin", cwd=SGLANG_REPO)
    else:
        print(f"  Cloning SGLang to: {SGLANG_REPO}")
        SGLANG_REPO.parent.mkdir(parents=True, exist_ok=True)
        ret, out = run_cmd(
            f"git clone https://github.com/sgl-project/sglang.git {SGLANG_REPO}",
            timeout=600
        )
        if ret != 0:
            print(f"  ERROR cloning: {out[:500]}")
            return False
    return True


def install_sglang_editable(commit_hash: str) -> bool:
    """Checkout and install SGLang as editable package with dependencies."""
    print(f"  Checking out {commit_hash[:12]}...")

    # Reset any changes
    run_cmd("git reset --hard HEAD", cwd=SGLANG_REPO)
    run_cmd("git clean -fd", cwd=SGLANG_REPO)

    # Checkout commit
    ret, out = run_cmd(f"git checkout {commit_hash}", cwd=SGLANG_REPO)
    if ret != 0:
        print(f"  ERROR checking out: {out[:300]}")
        return False

    # Install as editable with runtime dependencies
    print(f"  Installing SGLang editable with deps...")

    # First uninstall existing sglang
    run_cmd("pip uninstall -y sglang 2>/dev/null", timeout=60)

    python_dir = SGLANG_REPO / "python"

    # Install transformers and other commonly needed deps first
    print(f"    Installing base dependencies...")
    ret, out = run_cmd(
        "uv pip install --system transformers accelerate sentencepiece protobuf pillow "
        "torch vllm flashinfer-python triton",
        timeout=600
    )

    # Install sglang editable with runtime_common extras
    print(f"    Installing sglang[runtime_common]...")
    ret, out = run_cmd(f"uv pip install --system -e '{python_dir}[runtime_common]'", timeout=900)

    if ret != 0:
        # Try regular pip as fallback
        print(f"    uv failed, trying pip...")
        ret, out = run_cmd(f"pip install -e '{python_dir}[runtime_common]'", timeout=900)

    if ret != 0:
        print(f"    Install error: {out[-500:]}")
        # Try basic install without extras
        print(f"    Trying basic install without extras...")
        ret, out = run_cmd(f"uv pip install --system -e '{python_dir}'", timeout=600)

    if ret != 0:
        print(f"  Final install error: {out[-500:]}")
        return False

    # Verify installation
    ret, out = run_cmd("python -c 'import sglang; print(f\"SGLang installed\")'")
    if ret == 0:
        print(f"    {out.strip()}")
    else:
        print(f"    Verification warning: {out[:200]}")

    return True


def detect_benchmark_script() -> str:
    """Detect which benchmark script is available and works."""
    # Check bench_one_batch first (newer naming)
    if (SGLANG_REPO / "python/sglang/bench_one_batch.py").exists():
        # Make sure it doesn't just redirect
        content = (SGLANG_REPO / "python/sglang/bench_one_batch.py").read_text()[:200]
        if "raise ValueError" not in content and "renamed" not in content.lower():
            return "bench_one_batch"

    # Check bench_latency (older naming, June 2024 era)
    if (SGLANG_REPO / "python/sglang/bench_latency.py").exists():
        content = (SGLANG_REPO / "python/sglang/bench_latency.py").read_text()[:200]
        if "raise ValueError" not in content and "renamed" not in content.lower():
            return "bench_latency"

    # Last resort: try both imports
    ret, out = run_cmd("python -c 'from sglang import bench_one_batch; print(\"ok\")'")
    if ret == 0 and "ok" in out:
        return "bench_one_batch"

    ret, out = run_cmd("python -c 'from sglang import bench_latency; print(\"ok\")'")
    if ret == 0 and "ok" in out:
        return "bench_latency"

    return "unknown"


def run_benchmark(
    commit_short: str,
    variant: str,
    patch_path: Path = None,
    model: str = "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    batch_size: int = 4,
    input_len: int = 512,
    output_len: int = 64
) -> dict:
    """Run benchmark using installed SGLang."""

    result = {
        "commit": commit_short,
        "variant": variant,
        "timestamp": datetime.now().isoformat(),
        "model": model,
        "batch_size": batch_size,
        "input_len": input_len,
        "output_len": output_len,
    }

    # Apply patch if provided (to the checked-out repo)
    if patch_path:
        print(f"    Applying patch: {patch_path.name}")
        ret, out = run_cmd(f"git apply {patch_path}", cwd=SGLANG_REPO)
        if ret != 0:
            # Try with --reject
            ret2, out2 = run_cmd(f"git apply --reject {patch_path}", cwd=SGLANG_REPO)
            result["patch_status"] = "partial" if ret2 == 0 else "failed"
            result["patch_error"] = out[:500]
        else:
            result["patch_status"] = "applied"

    # Detect benchmark script
    bench_script = detect_benchmark_script()
    result["bench_script"] = bench_script

    # Get HF token
    hf_token = ""
    hf_token_file = Path("/root/.cache/huggingface/token")
    if hf_token_file.exists():
        hf_token = hf_token_file.read_text().strip()

    env_vars = {
        "HF_TOKEN": hf_token,
        "GLOO_SOCKET_IFNAME": "lo",
    }
    env_str = " ".join(f"{k}={v}" for k, v in env_vars.items() if v)

    if bench_script == "bench_latency":
        benchmark_cmd = f"""
{env_str} python -m sglang.bench_latency \\
    --model-path {model} \\
    --batch-size {batch_size} \\
    --input-len {input_len} \\
    --output-len {output_len} \\
    --load-format dummy \\
    --trust-remote-code \\
    --mem-fraction-static 0.8 \\
    --disable-flashinfer \\
    2>&1
"""
    elif bench_script == "bench_one_batch":
        benchmark_cmd = f"""
{env_str} python -m sglang.bench_one_batch \\
    --model-path {model} \\
    --batch-size {batch_size} \\
    --input-len {input_len} \\
    --output-len {output_len} \\
    --load-format dummy \\
    --trust-remote-code \\
    --mem-fraction-static 0.8 \\
    2>&1
"""
    else:
        result["status"] = "failed"
        result["error"] = f"Unknown benchmark script: {bench_script}"
        return result

    print(f"    Running benchmark ({bench_script})...")
    start_time = time.time()
    ret, output = run_cmd(benchmark_cmd, timeout=600)
    duration = time.time() - start_time

    result["duration_s"] = duration
    result["return_code"] = ret
    result["raw_output"] = output[-5000:] if output else ""

    # Parse benchmark results
    if ret == 0 and output:
        lines = output.split('\n')
        prefill_throughputs = []
        decode_throughputs = []
        total_throughputs = []

        for line in lines:
            # Parse prefill throughput
            if line.startswith("Prefill.") and "throughput" in line:
                match = re.search(r'throughput:\s*([\d.]+)\s*token', line)
                if match:
                    prefill_throughputs.append(float(match.group(1)))

            # Parse decode avg throughput
            if "Decode." in line and "avg throughput" in line:
                match = re.search(r'avg throughput:\s*([\d.]+)\s*token', line)
                if match:
                    decode_throughputs.append(float(match.group(1)))
            elif "Decode." in line and "median throughput" in line:
                match = re.search(r'median throughput:\s*([\d.]+)\s*token', line)
                if match:
                    decode_throughputs.append(float(match.group(1)))

            # Parse total throughput
            if line.startswith("Total.") and "throughput" in line:
                match = re.search(r'throughput:\s*([\d.]+)\s*token', line)
                if match:
                    total_throughputs.append(float(match.group(1)))

            # Generic output throughput
            if "output throughput" in line.lower():
                try:
                    parts = line.split(':')
                    if len(parts) >= 2:
                        result["output_throughput"] = float(parts[-1].strip().split()[0])
                except:
                    pass

        # Take the last (warmed up) values
        if prefill_throughputs:
            result["prefill_throughput"] = prefill_throughputs[-1]
        if decode_throughputs:
            result["decode_throughput"] = decode_throughputs[-1]
        if total_throughputs:
            result["total_throughput"] = total_throughputs[-1]

        if any(k in result for k in ["prefill_throughput", "decode_throughput", "total_throughput", "output_throughput"]):
            result["status"] = "success"
        else:
            result["status"] = "no_metrics"
    else:
        result["status"] = "failed"
        if "ImportError" in output or "ModuleNotFoundError" in output:
            result["error"] = "Import error"
        elif "CUDA" in output and "error" in output.lower():
            result["error"] = "CUDA error"
        elif "TIMEOUT" in output:
            result["error"] = "Timeout"
        else:
            result["error"] = "Unknown error"

    # Reset git state after patch
    if patch_path:
        run_cmd("git checkout .", cwd=SGLANG_REPO)
        run_cmd("git clean -fd", cwd=SGLANG_REPO)

    return result


def run_3way_benchmark(
    commit_short: str,
    agents: list[str],
    dry_run: bool = False
) -> dict:
    """Run 3-way benchmark (baseline, human, agents) for a commit."""

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
    print(f"Human: {human_commit[:12]}")
    print(f"Base: {base_commit[:12]}")
    print(f"{'='*60}")

    # Find agent patches
    agent_patches = {}
    for agent in agents:
        patch = find_agent_patch(commit_short, agent)
        if patch:
            agent_patches[agent] = patch
            print(f"Found {agent} patch: {patch.name}")
        else:
            print(f"No patch for {agent}")

    if dry_run:
        print(f"\n[DRY RUN] Would benchmark:")
        print(f"  - baseline: checkout {base_commit[:12]}, install editable")
        print(f"  - human: checkout {human_commit[:12]}, install editable")
        for agent in agent_patches:
            print(f"  - {agent}: checkout {base_commit[:12]} + patch, install editable")
        return {"status": "dry_run", "commit": commit_short}

    results = {
        "commit": commit_short,
        "human_commit": human_commit,
        "base_commit": base_commit,
        "subject": subject,
        "pr_number": pr_number,
        "timestamp": datetime.now().isoformat(),
        "benchmarks": []
    }

    # Setup repo
    print(f"\n[SETUP] Preparing repository...")
    if not setup_repo():
        results["error"] = "Failed to setup repo"
        return results

    # 1. Baseline benchmark (parent commit)
    print(f"\n[1/N] BASELINE benchmark...")
    if not install_sglang_editable(base_commit):
        results["benchmarks"].append({"variant": "baseline", "status": "failed", "error": "Install failed"})
    else:
        baseline_result = run_benchmark(commit_short, "baseline")
        results["benchmarks"].append(baseline_result)
        print_result(baseline_result)

    # 2. Human benchmark (optimization commit)
    print(f"\n[2/N] HUMAN benchmark...")
    if not install_sglang_editable(human_commit):
        results["benchmarks"].append({"variant": "human", "status": "failed", "error": "Install failed"})
    else:
        human_result = run_benchmark(commit_short, "human")
        results["benchmarks"].append(human_result)
        print_result(human_result)

    # 3. Agent benchmarks (base + patch)
    for i, (agent, patch_path) in enumerate(agent_patches.items(), start=3):
        print(f"\n[{i}/N] {agent.upper()} benchmark...")
        if not install_sglang_editable(base_commit):
            results["benchmarks"].append({"variant": agent, "status": "failed", "error": "Install failed"})
        else:
            agent_result = run_benchmark(commit_short, agent, patch_path=patch_path)
            results["benchmarks"].append(agent_result)
            print_result(agent_result)

    # Save results
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    output_file = RESULTS_DIR / f"{commit_short}_3way.json"
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {output_file}")

    return results


def print_result(result: dict):
    """Print a single benchmark result."""
    variant = result.get("variant", "?")
    status = result.get("status", "?")

    if status == "success":
        total = result.get("total_throughput")
        prefill = result.get("prefill_throughput")
        decode = result.get("decode_throughput")
        output = result.get("output_throughput")

        if total:
            print(f"    {variant}: {total:.2f} tok/s (total)")
        elif output:
            print(f"    {variant}: {output:.2f} tok/s (output)")
        elif prefill or decode:
            metrics = []
            if prefill:
                metrics.append(f"prefill={prefill:.0f}")
            if decode:
                metrics.append(f"decode={decode:.0f}")
            print(f"    {variant}: {', '.join(metrics)} tok/s")
        else:
            print(f"    {variant}: SUCCESS (no throughput metric)")
    elif status == "no_metrics":
        print(f"    {variant}: Completed but no metrics parsed")
    else:
        error = result.get("error", "Unknown")
        print(f"    {variant}: FAILED - {error}")


def main():
    parser = argparse.ArgumentParser(description="SGLang 3-Way Benchmark (Editable Install)")
    parser.add_argument("--commit", type=str, help="Specific commit (8-char)")
    parser.add_argument("--all", action="store_true", help="Benchmark all target commits")
    parser.add_argument("--agents", type=str, default="claude_code,codex,trae_gpt5,trae_sonnet45",
                        help="Comma-separated agents")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be done")
    parser.add_argument("--list", action="store_true", help="List target commits")

    args = parser.parse_args()

    if args.list:
        print("\nTarget commits for benchmarking:")
        mapping = load_commit_mapping()
        for commit in TARGET_COMMITS:
            info = mapping.get(commit, {})
            subject = info.get("subject", "Unknown")[:40]
            pr = info.get("pr_number", "?")
            print(f"  {commit}: PR#{pr} {subject}")
        return

    agents = [a.strip() for a in args.agents.split(",")]

    commits_to_run = []
    if args.all:
        commits_to_run = TARGET_COMMITS
    elif args.commit:
        commits_to_run = [args.commit]
    else:
        parser.print_help()
        return

    all_results = []
    for commit in commits_to_run:
        try:
            result = run_3way_benchmark(commit, agents, args.dry_run)
            all_results.append(result)
        except Exception as e:
            print(f"ERROR processing {commit}: {e}")
            import traceback
            traceback.print_exc()
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
                print_result(bench)


if __name__ == "__main__":
    main()
