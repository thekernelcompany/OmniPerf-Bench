#!/usr/bin/env python3
"""
SGLang 3-Way Benchmark Runner - Installed SGLang Approach

For newer commits (Oct 2024+), we use pip-installed sglang and apply patches
to the installed files directly.

Usage:
    python run_installed_benchmarks.py --commit 148254d4 --dry-run
    python run_installed_benchmarks.py --commit 148254d4 --variant human
    python run_installed_benchmarks.py --all --agents claude_code,codex,trae_gpt5,trae_sonnet45
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
import re
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


def run_cmd(cmd, cwd=None, timeout=600, capture=True, env=None):
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


def setup_venv_with_sglang(commit_short: str, force_recreate: bool = False) -> Path | None:
    """Create a venv with sglang installed via pip."""
    venv_dir = VENV_BASE / f"sglang-installed-{commit_short}"
    python_path = venv_dir / "bin" / "python"

    if venv_dir.exists() and not force_recreate:
        # Verify sglang is installed
        ret, out = run_cmd(f"{python_path} -c 'import sglang; print(sglang.__version__)'")
        if ret == 0:
            print(f"  Using existing venv with sglang: {venv_dir}")
            return venv_dir
        print(f"  Existing venv broken, recreating...")

    print(f"  Creating venv with pip-installed sglang: {venv_dir}")
    VENV_BASE.mkdir(parents=True, exist_ok=True)

    # Remove old venv if exists
    if venv_dir.exists():
        shutil.rmtree(venv_dir)

    # Create venv with uv
    ret, out = run_cmd(f"uv venv {venv_dir}")
    if ret != 0:
        print(f"  ERROR creating venv: {out}")
        return None

    pip_cmd = f"VIRTUAL_ENV={venv_dir} uv pip install"

    # Install PyTorch with CUDA
    print("  Installing PyTorch...")
    ret, out = run_cmd(f"{pip_cmd} torch==2.5.1 --index-url https://download.pytorch.org/whl/cu121", timeout=600)

    # Install sglang with all dependencies
    print("  Installing sglang[all]...")
    ret, out = run_cmd(f"{pip_cmd} 'sglang[all]'", timeout=900)
    if ret != 0:
        print(f"  sglang[all] failed, trying sglang...")
        ret, out = run_cmd(f"{pip_cmd} sglang", timeout=600)

    # Install flashinfer
    print("  Installing flashinfer...")
    ret, out = run_cmd(f"{pip_cmd} flashinfer-python -i https://flashinfer.ai/whl/cu121/torch2.5/", timeout=300)

    # Install sgl-kernel
    print("  Installing sgl-kernel...")
    ret, out = run_cmd(f"{pip_cmd} sgl-kernel", timeout=300)

    # Verify installation
    ret, out = run_cmd(f"{python_path} -c 'import sglang; print(f\"sglang: {{sglang.__version__}}\")'")
    if ret == 0:
        print(f"    {out.strip()}")
    else:
        print(f"    sglang verification failed: {out[:200]}")
        return None

    return venv_dir


def get_sglang_install_dir(venv_dir: Path) -> Path | None:
    """Get the sglang installation directory in the venv."""
    python_path = venv_dir / "bin" / "python"
    ret, out = run_cmd(f"{python_path} -c 'import sglang; import os; print(os.path.dirname(sglang.__file__))'")
    if ret == 0:
        return Path(out.strip())
    return None


def apply_patch_to_installed(patch_path: Path, sglang_dir: Path) -> tuple[bool, str]:
    """Apply a patch to the installed sglang files."""
    # Read the patch and transform paths
    with open(patch_path) as f:
        patch_content = f.read()

    # Transform paths from python/sglang/... to just sglang/...
    # The patch might have paths like:
    # a/python/sglang/srt/... -> needs to become a/srt/...
    transformed = patch_content.replace("a/python/sglang/", "a/")
    transformed = transformed.replace("b/python/sglang/", "b/")

    # Also handle paths without python/ prefix
    # a/sglang/srt/... -> a/srt/...
    transformed = transformed.replace("a/sglang/", "a/")
    transformed = transformed.replace("b/sglang/", "b/")

    # Write transformed patch to temp file
    temp_patch = Path("/tmp/transformed_patch.diff")
    with open(temp_patch, 'w') as f:
        f.write(transformed)

    # Try to apply patch
    ret, out = run_cmd(f"git apply --check {temp_patch}", cwd=sglang_dir)
    if ret != 0:
        # Try with -p1 to strip first component
        ret, out = run_cmd(f"git apply --check -p1 {temp_patch}", cwd=sglang_dir)

    if ret != 0:
        return False, f"Patch check failed: {out[:500]}"

    # Actually apply
    ret, out = run_cmd(f"git apply {temp_patch}", cwd=sglang_dir)
    if ret != 0:
        ret, out = run_cmd(f"git apply -p1 {temp_patch}", cwd=sglang_dir)

    if ret != 0:
        return False, f"Patch apply failed: {out[:500]}"

    return True, "Applied successfully"


def backup_sglang_dir(sglang_dir: Path) -> Path:
    """Backup the sglang installation directory."""
    backup_dir = sglang_dir.parent / f"{sglang_dir.name}_backup"
    if backup_dir.exists():
        shutil.rmtree(backup_dir)
    shutil.copytree(sglang_dir, backup_dir)
    return backup_dir


def restore_sglang_dir(backup_dir: Path, sglang_dir: Path):
    """Restore sglang from backup."""
    if backup_dir.exists():
        shutil.rmtree(sglang_dir)
        shutil.copytree(backup_dir, sglang_dir)
        shutil.rmtree(backup_dir)


def run_benchmark(
    venv_dir: Path,
    variant: str,
    commit_short: str,
    model: str = "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    batch_size: int = 4,
    input_len: int = 512,
    output_len: int = 64
) -> dict:
    """Run benchmark using installed sglang."""

    result = {
        "commit": commit_short,
        "variant": variant,
        "timestamp": datetime.now().isoformat(),
        "model": model,
        "batch_size": batch_size,
        "input_len": input_len,
        "output_len": output_len,
    }

    python_path = venv_dir / "bin" / "python"

    # Build benchmark command
    benchmark_cmd = f"""
GLOO_SOCKET_IFNAME=lo {python_path} -m sglang.bench_one_batch \
    --model-path {model} \
    --batch-size {batch_size} \
    --input-len {input_len} \
    --output-len {output_len} \
    --load-format dummy \
    --trust-remote-code \
    --mem-fraction-static 0.8 \
    2>&1
"""

    print(f"  Running benchmark ({variant})...")
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
            if line.startswith("Prefill.") and "throughput" in line:
                match = re.search(r'throughput:\s*([\d.]+)\s*token', line)
                if match:
                    prefill_throughputs.append(float(match.group(1)))

            if "Decode." in line and "avg throughput" in line:
                match = re.search(r'avg throughput:\s*([\d.]+)\s*token', line)
                if match:
                    decode_throughputs.append(float(match.group(1)))

            if line.startswith("Total.") and "throughput" in line:
                match = re.search(r'throughput:\s*([\d.]+)\s*token', line)
                if match:
                    total_throughputs.append(float(match.group(1)))

        if prefill_throughputs:
            result["prefill_throughput"] = prefill_throughputs[-1]
        if decode_throughputs:
            result["decode_throughput"] = decode_throughputs[-1]
        if total_throughputs:
            result["total_throughput"] = total_throughputs[-1]

        if any(k in result for k in ["prefill_throughput", "decode_throughput", "total_throughput"]):
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
            result["error"] = output[-500:] if output else "Unknown error"

    return result


def run_3way_benchmark(
    commit_short: str,
    agents: list[str],
    dry_run: bool = False,
    force_venv: bool = False
) -> dict:
    """Run 3-way benchmark for a commit using installed sglang."""

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
        print(f"  - baseline: installed sglang")
        print(f"  - human: installed sglang (same as baseline for installed approach)")
        for agent in agent_patches:
            print(f"  - {agent}: installed sglang + patch")
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

    # Setup venv with installed sglang
    print(f"\n[SETUP] Creating environment with installed sglang...")
    venv_dir = setup_venv_with_sglang(commit_short, force_venv)
    if not venv_dir:
        results["error"] = "Failed to create venv"
        return results

    sglang_dir = get_sglang_install_dir(venv_dir)
    if not sglang_dir:
        results["error"] = "Could not find sglang installation"
        return results

    print(f"  SGLang installed at: {sglang_dir}")

    # Backup sglang directory for restoration after patches
    backup_dir = backup_sglang_dir(sglang_dir)

    try:
        # 1. Baseline benchmark (installed sglang without modifications)
        print(f"\n[1] BASELINE benchmark...")
        baseline_result = run_benchmark(venv_dir, "baseline", commit_short)
        results["benchmarks"].append(baseline_result)
        print_result(baseline_result)

        # 2. Human benchmark (for installed sglang, this is same as baseline)
        # Note: With installed sglang, we can't easily switch to human commit
        # The human optimization is already in the installed version
        print(f"\n[2] HUMAN benchmark (same as baseline for installed approach)...")
        human_result = run_benchmark(venv_dir, "human", commit_short)
        results["benchmarks"].append(human_result)
        print_result(human_result)

        # 3. Agent benchmarks (apply patches to installed sglang)
        for i, (agent, patch_path) in enumerate(agent_patches.items(), start=3):
            print(f"\n[{i}] {agent.upper()} benchmark...")

            # Restore from backup first
            restore_sglang_dir(backup_dir, sglang_dir)
            backup_dir = backup_sglang_dir(sglang_dir)  # Re-create backup

            # Apply patch
            success, msg = apply_patch_to_installed(patch_path, sglang_dir)
            agent_result = {"commit": commit_short, "variant": agent}

            if success:
                agent_result["patch_status"] = "applied"
                agent_result = run_benchmark(venv_dir, agent, commit_short)
            else:
                agent_result["patch_status"] = "failed"
                agent_result["patch_error"] = msg
                agent_result["status"] = "failed"
                agent_result["error"] = f"Patch failed: {msg}"

            results["benchmarks"].append(agent_result)
            print_result(agent_result)

    finally:
        # Restore sglang to original state
        if backup_dir.exists():
            restore_sglang_dir(backup_dir, sglang_dir)

    # Save results
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    output_file = RESULTS_DIR / f"{commit_short}_installed_3way.json"
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

        if total:
            print(f"  {variant}: {total:.2f} tok/s (total)")
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
        error = result.get("error", "Unknown")[:100]
        print(f"  {variant}: FAILED - {error}")


def main():
    parser = argparse.ArgumentParser(description="SGLang 3-Way Benchmark (Installed)")
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
            import traceback
            print(f"ERROR processing {commit}: {e}")
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
