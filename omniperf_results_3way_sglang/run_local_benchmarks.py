#!/usr/bin/env python3
"""
SGLang 3-Way Benchmark Runner - Local Git Checkout + UV Approach

This script:
1. Checks out specific SGLang commits
2. Creates venvs with proper dependencies using uv
3. Runs bench_one_batch.py for latency measurements
4. Applies agent patches for comparison

Usage:
    python run_local_benchmarks.py --commit 148254d4 --dry-run
    python run_local_benchmarks.py --commit 148254d4 --variant human
    python run_local_benchmarks.py --all --agents claude_code,codex
"""

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

# Paths
SGLANG_REPO = Path("/root/OmniPerf-Bench/omniperf_results_3way_sglang/sglang-local")
RESULTS_DIR = Path("/root/OmniPerf-Bench/omniperf_results_3way_sglang/local_benchmark_results")
COMMIT_MAPPING_FILE = Path("/root/OmniPerf-Bench/src/benchmark/fixes/sglang_commit_mapping.json")
AGENT_BASE_DIR = Path("/root/OmniPerf-Bench/perf-agents-bench/state/runs/sglang")
VENV_BASE = Path("/tmp/sglang-venvs")

# Agent configurations
AGENT_CONFIGS = {
    "claude_code": AGENT_BASE_DIR / "claude_code/default/2025-12-23_06-28-44",
    "codex": AGENT_BASE_DIR / "codex/gpt-5/389be848",
    "trae_gpt5": AGENT_BASE_DIR / "trae/gpt-5/2025-11-14_21-05-32",
    "trae_sonnet45": AGENT_BASE_DIR / "trae/claude-sonnet-45/2025-11-28_15-26-15",
}

# Target commits (from the plan)
TARGET_COMMITS = [
    "187b85b7",  # [PD] Optimize custom mem pool usage
    "6b231325",  # [PD Perf] replace Queue to FastQueue
    "6cb00c63",  # [PD] Optimize time out logic
    "148254d4",  # Improve moe reduce sum kernel
    "2bd18e2d",  # Memory pool: Minor optimize
    "2a754e57",  # 2x perf for large prefill
    "880221bd",  # Revert transfer batch
    "b1e5a33a",  # Eliminate stream sync for LoRA
    "da47621c",  # Minor speedup topk postprocessing
    "dd1012fc",  # [PD] Fix potential perf spike
    "ddcf9fe3",  # Optimize triton attention mask
    "df7f61ee",  # Speed up rebalancing
    "e3ec6bf4",  # Minor speed up block_quant_dequant
]


def run_cmd(cmd, cwd=None, timeout=600, capture=True):
    """Run a command and return output."""
    try:
        result = subprocess.run(
            cmd,
            shell=True,
            cwd=cwd,
            capture_output=capture,
            text=True,
            timeout=timeout,
            executable="/bin/bash"  # Use bash for proper shell features
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
    for commit in data["commits"]:
        short = commit["human_commit"][:8]
        mapping[short] = {
            "human_commit": commit["human_commit"],
            "base_commit": commit["base_commit"],
            "pr_number": commit.get("pr_number"),
            "subject": commit.get("subject", "Unknown")
        }
    return mapping


def get_codex_index(commit_short: str) -> int | None:
    """Get the codex index for a commit (1-indexed)."""
    with open(COMMIT_MAPPING_FILE) as f:
        data = json.load(f)
    for i, commit in enumerate(data["commits"], start=1):
        if commit["human_commit"][:8] == commit_short:
            return i
    return None


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


def setup_venv(commit_short: str, force_recreate: bool = False) -> Path:
    """Create a venv for a specific commit with proper dependencies."""
    venv_dir = VENV_BASE / f"sglang-{commit_short}"
    python_path = venv_dir / "bin" / "python"

    if venv_dir.exists() and not force_recreate:
        print(f"  Using existing venv: {venv_dir}")
        return venv_dir

    print(f"  Creating venv: {venv_dir}")
    VENV_BASE.mkdir(parents=True, exist_ok=True)

    # Remove old venv if force recreate
    if venv_dir.exists():
        run_cmd(f"rm -rf {venv_dir}")

    # Create venv with uv
    ret, out = run_cmd(f"uv venv {venv_dir}")
    if ret != 0:
        print(f"  ERROR creating venv: {out}")
        return None

    # Use uv pip with VIRTUAL_ENV set
    pip_cmd = f"VIRTUAL_ENV={venv_dir} uv pip install"

    # Install PyTorch with CUDA
    print("  Installing PyTorch...")
    ret, out = run_cmd(f"{pip_cmd} torch==2.5.1 --index-url https://download.pytorch.org/whl/cu121", timeout=600)
    if ret != 0:
        print(f"  PyTorch install warning: {out[-500:]}")

    # Install core dependencies (including tqdm which was missing)
    print("  Installing core dependencies...")
    ret, out = run_cmd(f"{pip_cmd} numpy triton setuptools wheel packaging tqdm", timeout=180)

    # Install vllm
    print("  Installing vllm...")
    ret, out = run_cmd(f"{pip_cmd} vllm==0.6.4.post1", timeout=900)
    if ret != 0:
        # Try without version constraint
        print(f"  vllm specific version failed, trying latest...")
        ret, out = run_cmd(f"{pip_cmd} vllm", timeout=900)

    # Install flashinfer
    print("  Installing flashinfer...")
    ret, out = run_cmd(f"{pip_cmd} flashinfer-python -i https://flashinfer.ai/whl/cu121/torch2.5/", timeout=300)

    # Install sgl-kernel
    print("  Installing sgl-kernel...")
    ret, out = run_cmd(f"{pip_cmd} sgl-kernel", timeout=300)

    # Install additional deps
    print("  Installing additional dependencies...")
    ret, out = run_cmd(
        f"{pip_cmd} transformers accelerate sentencepiece protobuf pillow "
        "requests aiohttp psutil uvicorn fastapi pydantic uvloop einops msgpack "
        "tiktoken interegular outlines lark diskcache orjson filelock compressed-tensors "
        "ipython rpyc zmq hf_transfer huggingface_hub",
        timeout=600
    )

    # Verify installation
    print("  Verifying installation...")
    ret, out = run_cmd(f"{python_path} -c 'import torch; print(f\"PyTorch: {{torch.__version__}}\")'")
    if ret == 0:
        print(f"    {out.strip()}")
    else:
        print(f"    PyTorch verification failed: {out[:200]}")

    return venv_dir


def detect_benchmark_script(repo_path: Path) -> str:
    """Detect which benchmark script is available."""
    if (repo_path / "python/sglang/bench_one_batch.py").exists():
        return "bench_one_batch"
    elif (repo_path / "python/sglang/bench_latency.py").exists():
        return "bench_latency"
    else:
        return "unknown"


def run_benchmark(
    commit_short: str,
    variant: str,
    venv_dir: Path,
    patch_path: Path = None,
    model: str = "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    batch_size: int = 4,
    input_len: int = 512,
    output_len: int = 64
) -> dict:
    """Run benchmark (auto-detects bench_latency.py or bench_one_batch.py)."""

    result = {
        "commit": commit_short,
        "variant": variant,
        "timestamp": datetime.now().isoformat(),
        "model": model,
        "batch_size": batch_size,
        "input_len": input_len,
        "output_len": output_len,
    }

    # Apply patch if provided
    if patch_path:
        print(f"  Applying patch: {patch_path}")
        ret, out = run_cmd(f"git apply {patch_path}", cwd=SGLANG_REPO)
        if ret != 0:
            # Try with --reject to see what fails
            ret2, out2 = run_cmd(f"git apply --reject {patch_path}", cwd=SGLANG_REPO)
            result["patch_status"] = "partial" if ret2 == 0 else "failed"
            result["patch_error"] = out[:500]
        else:
            result["patch_status"] = "applied"

    # Detect which benchmark script to use
    bench_script = detect_benchmark_script(SGLANG_REPO)
    result["bench_script"] = bench_script

    # Build the benchmark command using venv's python directly
    python_path = venv_dir / "bin" / "python"
    hf_token = ""
    hf_token_file = Path("/root/.cache/huggingface/token")
    if hf_token_file.exists():
        hf_token = hf_token_file.read_text().strip()

    env_vars = {
        "PYTHONPATH": f"{SGLANG_REPO}/python",
        "HF_TOKEN": hf_token,
    }
    env_str = " ".join(f"{k}={v}" for k, v in env_vars.items() if v)

    if bench_script == "bench_latency":
        # Older sglang (June 2024 era) - uses bench_latency.py
        benchmark_cmd = f"""
cd {SGLANG_REPO}
{env_str} {python_path} -m sglang.bench_latency \
    --model-path {model} \
    --batch-size {batch_size} \
    --input-len {input_len} \
    --output-len {output_len} \
    --load-format dummy \
    --trust-remote-code \
    --mem-fraction-static 0.8 \
    --disable-flashinfer \
    2>&1
"""
    else:
        # Newer sglang - uses bench_one_batch.py
        benchmark_cmd = f"""
cd {SGLANG_REPO}
{env_str} {python_path} -m sglang.bench_one_batch \
    --model-path {model} \
    --batch-size {batch_size} \
    --input-len {input_len} \
    --output-len {output_len} \
    --load-format dummy \
    --trust-remote-code \
    --mem-fraction-static 0.8 \
    2>&1
"""

    print(f"  Running benchmark ({bench_script})...")
    start_time = time.time()
    ret, output = run_cmd(benchmark_cmd, timeout=600)
    duration = time.time() - start_time

    result["duration_s"] = duration
    result["return_code"] = ret
    result["raw_output"] = output[-5000:] if output else ""

    # Parse benchmark results from output
    import re
    if ret == 0 and output:
        lines = output.split('\n')

        # For bench_latency output format:
        # "Total. latency:  0.667 s, throughput:   3452.79 token/s"
        # "Prefill. latency: 0.01888 s, throughput: 108488.91 token/s"
        # "Decode.  avg latency: 0.01013 s, avg throughput:    394.81 token/s"
        prefill_throughputs = []
        decode_throughputs = []
        total_throughputs = []

        for line in lines:
            # Parse prefill throughput (last value is after warmup)
            if line.startswith("Prefill.") and "throughput" in line:
                match = re.search(r'throughput:\s*([\d.]+)\s*token', line)
                if match:
                    prefill_throughputs.append(float(match.group(1)))

            # Parse decode avg throughput
            if "Decode." in line and "avg throughput" in line:
                match = re.search(r'avg throughput:\s*([\d.]+)\s*token', line)
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

        # Check if we got any metrics
        if any(k in result for k in ["prefill_throughput", "decode_throughput", "total_throughput", "output_throughput"]):
            result["status"] = "success"
        else:
            result["status"] = "no_metrics"
    else:
        result["status"] = "failed"
        # Try to extract error
        if "ImportError" in output or "ModuleNotFoundError" in output:
            result["error"] = "Import error"
        elif "CUDA" in output and "error" in output.lower():
            result["error"] = "CUDA error"
        elif "TIMEOUT" in output:
            result["error"] = "Timeout"
        else:
            result["error"] = "Unknown error"

    # Reset git state after applying patch
    if patch_path:
        run_cmd("git checkout .", cwd=SGLANG_REPO)
        run_cmd("git clean -fd", cwd=SGLANG_REPO)

    return result


def run_3way_benchmark(
    commit_short: str,
    agents: list[str],
    dry_run: bool = False,
    force_venv: bool = False
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
            print(f"Found {agent} patch: {patch}")
        else:
            print(f"No patch for {agent}")

    if dry_run:
        print(f"\n[DRY RUN] Would benchmark:")
        print(f"  - baseline: checkout {base_commit[:12]}")
        print(f"  - human: checkout {human_commit[:12]}")
        for agent in agent_patches:
            print(f"  - {agent}: checkout {base_commit[:12]} + patch")
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

    # Setup venv
    print(f"\n[SETUP] Creating environment...")
    venv_dir = setup_venv(commit_short, force_venv)
    if not venv_dir:
        results["error"] = "Failed to create venv"
        return results

    # 1. Baseline benchmark (parent commit)
    print(f"\n[1] BASELINE benchmark...")
    ret, out = run_cmd(f"git checkout {base_commit}", cwd=SGLANG_REPO)
    if ret != 0:
        print(f"  ERROR checking out base: {out[:200]}")
    baseline_result = run_benchmark(commit_short, "baseline", venv_dir)
    results["benchmarks"].append(baseline_result)
    print_result(baseline_result)

    # 2. Human benchmark (optimization commit)
    print(f"\n[2] HUMAN benchmark...")
    ret, out = run_cmd(f"git checkout {human_commit}", cwd=SGLANG_REPO)
    if ret != 0:
        print(f"  ERROR checking out human: {out[:200]}")
    human_result = run_benchmark(commit_short, "human", venv_dir)
    results["benchmarks"].append(human_result)
    print_result(human_result)

    # 3. Agent benchmarks (base + patch)
    for i, (agent, patch_path) in enumerate(agent_patches.items(), start=3):
        print(f"\n[{i}] {agent.upper()} benchmark...")
        ret, out = run_cmd(f"git checkout {base_commit}", cwd=SGLANG_REPO)
        agent_result = run_benchmark(commit_short, agent, venv_dir, patch_path=patch_path)
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
            print(f"  {variant}: {total:.2f} tok/s (total)")
        elif output:
            print(f"  {variant}: {output:.2f} tok/s (output)")
        elif prefill or decode:
            metrics = []
            if prefill:
                metrics.append(f"prefill={prefill:.0f}")
            if decode:
                metrics.append(f"decode={decode:.0f}")
            print(f"  {variant}: {', '.join(metrics)} tok/s")
        else:
            print(f"  {variant}: SUCCESS (no throughput metric)")
    elif status == "no_metrics":
        print(f"  {variant}: Completed but no metrics parsed")
    else:
        error = result.get("error", "Unknown")
        print(f"  {variant}: FAILED - {error}")


def main():
    parser = argparse.ArgumentParser(description="SGLang 3-Way Benchmark (Local)")
    parser.add_argument("--commit", type=str, help="Specific commit (8-char)")
    parser.add_argument("--all", action="store_true", help="Benchmark all target commits")
    parser.add_argument("--agents", type=str, default="claude_code,codex,trae_gpt5,trae_sonnet45",
                        help="Comma-separated agents")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be done")
    parser.add_argument("--force-venv", action="store_true", help="Recreate venv")
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
            result = run_3way_benchmark(commit, agents, args.dry_run, args.force_venv)
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
                print_result(bench)


if __name__ == "__main__":
    main()
