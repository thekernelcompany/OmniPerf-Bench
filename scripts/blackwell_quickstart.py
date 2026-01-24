#!/usr/bin/env python3
"""
SGLang Benchmark Quick Start for Blackwell GPU

This script helps you get started with SGLang benchmarks on Blackwell (SM100).
Run this first to verify your setup before running full benchmarks.

Usage:
    python scripts/blackwell_quickstart.py
"""

import subprocess
import sys
import json
from pathlib import Path

# ANSI colors for output
GREEN = "\033[92m"
RED = "\033[91m"
YELLOW = "\033[93m"
BLUE = "\033[94m"
RESET = "\033[0m"

def run_cmd(cmd, timeout=60):
    """Run command and return (success, output)"""
    try:
        result = subprocess.run(
            cmd, shell=True, capture_output=True, text=True, timeout=timeout
        )
        return result.returncode == 0, result.stdout + result.stderr
    except subprocess.TimeoutExpired:
        return False, "Command timed out"
    except Exception as e:
        return False, str(e)

def print_status(message, success=None):
    """Print status message with color"""
    if success is True:
        print(f"{GREEN}✓{RESET} {message}")
    elif success is False:
        print(f"{RED}✗{RESET} {message}")
    else:
        print(f"{BLUE}→{RESET} {message}")

def main():
    print(f"\n{BLUE}{'='*60}{RESET}")
    print(f"{BLUE}SGLang Benchmark Quick Start for Blackwell{RESET}")
    print(f"{BLUE}{'='*60}{RESET}\n")

    # Load working commits
    status_file = Path(__file__).parent.parent / "docs" / "sglang_benchmark_status.json"
    if status_file.exists():
        with open(status_file) as f:
            status_data = json.load(f)
        working_commits = status_data.get("working_commits", {}).get("commits", [])
    else:
        working_commits = []
        print(f"{YELLOW}Warning: Could not load status file{RESET}")

    # Step 1: Check GPU
    print(f"\n{YELLOW}Step 1: Checking GPU{RESET}")
    success, output = run_cmd("nvidia-smi --query-gpu=name,compute_cap --format=csv,noheader")
    if success:
        print_status(f"GPU detected: {output.strip()}", True)
        if "100" in output or "blackwell" in output.lower():
            print_status("Blackwell architecture confirmed", True)
        else:
            print_status("Note: This may not be a Blackwell GPU", None)
    else:
        print_status("Failed to detect GPU", False)
        print(f"   {output}")

    # Step 2: Check CUDA version
    print(f"\n{YELLOW}Step 2: Checking CUDA{RESET}")
    success, output = run_cmd("nvcc --version 2>/dev/null | grep release")
    if success:
        print_status(f"CUDA: {output.strip()}", True)
    else:
        success, output = run_cmd("nvidia-smi --query-gpu=driver_version --format=csv,noheader")
        if success:
            print_status(f"Driver version: {output.strip()}", True)
        else:
            print_status("Could not determine CUDA version", False)

    # Step 3: Check Docker
    print(f"\n{YELLOW}Step 3: Checking Docker{RESET}")
    success, output = run_cmd("docker --version")
    if success:
        print_status(f"Docker: {output.strip()}", True)
    else:
        print_status("Docker not found", False)
        print("   Install with: sudo apt-get install docker.io")

    # Step 4: Check NVIDIA Container Toolkit
    success, output = run_cmd("docker run --rm --gpus all nvidia/cuda:12.6.0-base-ubuntu22.04 nvidia-smi -L 2>&1", timeout=120)
    if success and "GPU" in output:
        print_status("NVIDIA Container Toolkit working", True)
    else:
        print_status("NVIDIA Container Toolkit may not be configured", False)
        print("   Install with: sudo apt-get install nvidia-container-toolkit && sudo systemctl restart docker")

    # Step 5: Test sgl-kernel import
    print(f"\n{YELLOW}Step 4: Testing sgl-kernel (PyPI){RESET}")
    test_cmd = '''docker run --rm --gpus all python:3.11 bash -c "
pip install -q sgl-kernel torch --index-url https://download.pytorch.org/whl/cu126 2>/dev/null
python -c 'import sgl_kernel; print(f\"sgl_kernel version: {sgl_kernel.__version__}\")'
"'''
    success, output = run_cmd(test_cmd, timeout=300)
    if success and "sgl_kernel version" in output:
        print_status(f"sgl-kernel working: {output.strip().split()[-1]}", True)
    else:
        print_status("sgl-kernel test failed", False)
        print(f"   Output: {output[:200]}")

    # Step 6: Check working Docker images
    print(f"\n{YELLOW}Step 5: Checking Docker Images{RESET}")
    if working_commits:
        test_commit = working_commits[0]
        image = f"ayushnangia16/nvidia-sglang-docker:{test_commit['hash']}"
        success, output = run_cmd(f"docker manifest inspect {image} 2>/dev/null")
        if success:
            print_status(f"Working image available: {test_commit['short']}", True)
        else:
            print_status(f"Image not found locally, will need to pull", None)
        print_status(f"Total working commits: {len(working_commits)}", None)

    # Summary
    print(f"\n{BLUE}{'='*60}{RESET}")
    print(f"{BLUE}Summary{RESET}")
    print(f"{BLUE}{'='*60}{RESET}")

    print(f"""
Working commits available: {len(working_commits)}
Documentation: docs/SGLANG_BENCHMARK_SESSION_NOTES.md
Status data: docs/sglang_benchmark_status.json

{YELLOW}Next steps:{RESET}
1. Pull a working image:
   docker pull ayushnangia16/nvidia-sglang-docker:187b85b7f38496653948a2aba546d53c09ada0f3

2. Run a test benchmark:
   python scripts/runners/hero_sglang_benchmark.py \\
       --commit 187b85b7 \\
       --benchmark-type serving

3. Batch process all working commits:
   python scripts/run_working_commits.py

{YELLOW}Key files to read:{RESET}
- docs/SGLANG_BENCHMARK_SESSION_NOTES.md (full documentation)
- docs/sglang_benchmark_status.json (structured data)
- src/benchmark/fixes/sglang_commit_mapping.json (commit mappings)
""")

    # List working commits
    print(f"\n{YELLOW}Working commits (17 total):{RESET}")
    for i, commit in enumerate(working_commits[:10], 1):
        print(f"  {i}. {commit['short']} - {commit['subject'][:50]}...")
    if len(working_commits) > 10:
        print(f"  ... and {len(working_commits) - 10} more (see docs/sglang_benchmark_status.json)")

if __name__ == "__main__":
    main()
