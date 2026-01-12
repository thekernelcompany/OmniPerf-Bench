#!/usr/bin/env python3
"""
SGLang Docker Image Builder

Builds Docker images for specific SGLang commits and pushes to DockerHub.
These images are used by the Modal benchmark runner for 3-way benchmarks.

Usage:
    # Build image for single commit
    python tools/build_sglang_images.py --commit abc123

    # Build images for all runnable commits
    python tools/build_sglang_images.py --all

    # Dry run (show what would be built)
    python tools/build_sglang_images.py --all --dry-run

    # Build with specific GPU (for multi-GPU machines)
    CUDA_VISIBLE_DEVICES=0 python tools/build_sglang_images.py --commit abc123
"""

import argparse
import json
import os
import re
import subprocess
import sys
import tempfile
import time
import urllib.request
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

# Constants
SGLANG_REPO_URL = "https://github.com/sgl-project/sglang.git"
DOCKER_REPO = "shikhar481/sglang-images"
CLAUDE_CODE_PATCHES_DIR = Path("perf-agents-bench/state/runs/sglang/claude_code")
WORK_DIR = Path("/tmp/sglang_docker_build")

# Default torch version fallback
DEFAULT_TORCH_VERSION = "2.5.1"


def get_torch_version_for_commit(repo_path: Path, commit: str) -> str:
    """
    Auto-detect torch version from commit's sgl-kernel/pyproject.toml.
    Falls back to python/pyproject.toml if sgl-kernel doesn't specify.

    Args:
        repo_path: Path to the SGLang repository
        commit: Commit hash to check

    Returns:
        Torch version string (e.g., "2.5.1")
    """
    # Try sgl-kernel/pyproject.toml first (build requirements)
    result = subprocess.run(
        ["git", "show", f"{commit}:sgl-kernel/pyproject.toml"],
        cwd=repo_path, capture_output=True, text=True, timeout=30
    )
    if result.returncode == 0:
        content = result.stdout
        # Look for torch version in build-system requires
        for line in content.split('\n'):
            if 'torch' in line.lower():
                # Match patterns like: torch==2.5.1, torch>=2.5.1, "torch==2.5.1"
                match = re.search(r'torch[>=<]+([0-9.]+)', line)
                if match:
                    version = match.group(1)
                    print(f"  Detected torch=={version} from sgl-kernel/pyproject.toml")
                    return version

    # Fall back to python/pyproject.toml (SGLang's runtime requirements)
    result = subprocess.run(
        ["git", "show", f"{commit}:python/pyproject.toml"],
        cwd=repo_path, capture_output=True, text=True, timeout=30
    )
    if result.returncode == 0:
        content = result.stdout
        for line in content.split('\n'):
            # Match "torch==X.X.X" in dependencies
            if '"torch==' in line:
                match = re.search(r'torch==([0-9.]+)', line)
                if match:
                    version = match.group(1)
                    print(f"  Detected torch=={version} from python/pyproject.toml")
                    return version

    print(f"  Using default torch=={DEFAULT_TORCH_VERSION}")
    return DEFAULT_TORCH_VERSION


def get_flashinfer_index_for_torch(torch_version: str) -> str:
    """
    Get the flashinfer wheel index URL for a given torch version.

    Args:
        torch_version: Torch version string (e.g., "2.5.1")

    Returns:
        Flashinfer wheel index URL
    """
    # Extract major.minor version
    parts = torch_version.split('.')
    major_minor = f"{parts[0]}.{parts[1]}" if len(parts) >= 2 else "2.5"

    # Map to flashinfer index
    # Note: flashinfer may not have wheels for all torch versions
    return f"https://flashinfer.ai/whl/cu124/torch{major_minor}/"

# Per-commit configuration for torch versions
COMMIT_CONFIG = {
    "d1112d85": {"torch": "2.5.1", "torch_minor": "2.5"},
    "48efec7b": {"torch": "2.5.1", "torch_minor": "2.5"},
    "93470a14": {"torch": "2.5.1", "torch_minor": "2.5"},
    "db452760": {"torch": "2.5.1", "torch_minor": "2.5"},
    "9c088829": {"torch": "2.6.0", "torch_minor": "2.6"},
    "005aad32": {"torch": "2.6.0", "torch_minor": "2.6"},
}

# Build configuration
# CRITICAL: sgl-kernel must be built FROM SOURCE with submodules to get deep_gemm
BENCHMARK_DOCKERFILE = '''
# SGLang Docker image with sgl-kernel built from source
# CRITICAL: Uses git submodules for deep_gemm module

ARG CUDA_VERSION=12.4.0
FROM nvidia/cuda:${{CUDA_VERSION}}-cudnn-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \\
    CUDA_HOME=/usr/local/cuda \\
    PATH="${{PATH}}:/usr/local/cuda/bin" \\
    LD_LIBRARY_PATH="${{LD_LIBRARY_PATH}}:/usr/local/cuda/lib64"

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \\
    python3.11 python3.11-dev python3.11-venv python3-pip \\
    git curl wget build-essential cmake ninja-build \\
    libopenmpi-dev libnuma-dev \\
    && ln -sf /usr/bin/python3.11 /usr/bin/python3 \\
    && ln -sf /usr/bin/python3.11 /usr/bin/python \\
    && rm -rf /var/lib/apt/lists/*

RUN python3 -m pip install --upgrade pip setuptools wheel

# Clone SGLang at specific commit WITH SUBMODULES
ARG COMMIT_HASH
WORKDIR /opt
RUN git clone --recursive {sglang_repo} sglang \\
    && cd sglang \\
    && git checkout ${{COMMIT_HASH}} \\
    && git submodule update --init --recursive

# Install PyTorch (version from commit's pyproject.toml)
ARG TORCH_VERSION=2.5.1
RUN pip install torch==${{TORCH_VERSION}} --index-url https://download.pytorch.org/whl/cu124

# Build sgl-kernel FROM SOURCE (not from PyPI!)
# This creates the deep_gemm module from 3rdparty/deepgemm submodule
WORKDIR /opt/sglang/sgl-kernel
RUN pip install scikit-build-core ninja cmake packaging
RUN pip install -e . --no-build-isolation -v 2>&1 | tee /tmp/sgl_kernel_build.log || (tail -100 /tmp/sgl_kernel_build.log && exit 1)

# Skip import verification during build (no GPU available)
# sgl_kernel and deep_gemm will work at runtime with GPU access
RUN ls -la /opt/sglang/sgl-kernel/python/sgl_kernel/ && echo "sgl_kernel directory exists"
RUN ls -la /usr/local/lib/python3.11/dist-packages/ | grep -E "sgl|deep" || echo "checking installed packages..."
RUN pip list | grep -E "sgl|deep" || true

# Install flashinfer
ARG TORCH_MINOR=2.5
RUN pip install flashinfer-python -i https://flashinfer.ai/whl/cu124/torch${{TORCH_MINOR}}/ || \\
    pip install flashinfer-python || true

# Install SGLang dependencies and package
# Note: Do NOT install vllm as it will upgrade torch and break sgl-kernel ABI
WORKDIR /opt/sglang
RUN pip install transformers huggingface_hub tokenizers accelerate "numpy<2.0" \\
    requests aiohttp triton packaging datasets pandas tqdm xgrammar || true
RUN pip install -e "python[srt]" --no-deps 2>/dev/null || pip install -e "python" --no-deps || pip install -e "python"

# Verify file structure (actual import requires GPU at runtime)
RUN ls -la /opt/sglang/python/sglang/ | head -20 && echo "SGLang installed"
RUN pip show sglang || pip list | grep -i sglang || echo "sglang installation info"

WORKDIR /workspace
CMD ["python", "-c", "import sglang; print('ready')"]
'''


def check_image_exists(commit: str) -> bool:
    """Check if Docker image already exists on DockerHub."""
    tag = commit[:40] if len(commit) >= 40 else commit
    url = f"https://hub.docker.com/v2/repositories/{DOCKER_REPO}/tags/{tag}"

    try:
        req = urllib.request.Request(url, method='HEAD')
        urllib.request.urlopen(req, timeout=10)
        return True
    except urllib.error.HTTPError as e:
        if e.code == 404:
            return False
        # Other errors might be rate limiting, assume image doesn't exist
        return False
    except Exception:
        return False


def get_existing_images() -> Set[str]:
    """Get set of commits that already have Docker images."""
    existing = set()

    try:
        url = f"https://hub.docker.com/v2/repositories/{DOCKER_REPO}/tags?page_size=100"
        with urllib.request.urlopen(url, timeout=30) as response:
            data = json.loads(response.read())
            for tag in data.get('results', []):
                name = tag.get('name', '')
                if len(name) >= 8:  # Commit hashes are at least 8 chars
                    existing.add(name[:8])  # Store short hash for matching
    except Exception as e:
        print(f"Warning: Could not fetch existing images: {e}")

    return existing


def get_runnable_commits() -> List[Dict]:
    """Get list of commits that need Docker images."""
    try:
        from datasets import load_dataset
        ds = load_dataset("Ayushnangia/omniperf_v1", split="sglang")
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return []

    # Find commits with Claude Code patches
    patch_commits = set()
    if CLAUDE_CODE_PATCHES_DIR.exists():
        for run_dir in CLAUDE_CODE_PATCHES_DIR.glob("*/*/sglang_*"):
            parts = run_dir.name.split("_")
            if len(parts) >= 3:
                patch_commits.add(parts[-1])

    # Find commits with perf_command AND patch
    runnable = []
    for item in ds:
        commit = item['commit_hash'][:8]
        if item.get('perf_command') and commit in patch_commits:
            runnable.append({
                'short_commit': commit,
                'full_commit': item['commit_hash'],
                'subject': item.get('commit_subject', 'N/A'),
                'perf_command': item.get('perf_command', ''),
                'models': item.get('models', []),
            })

    return runnable


def get_parent_commit(commit: str, repo_path: Path) -> Optional[str]:
    """Get parent commit hash."""
    result = subprocess.run(
        ["git", "rev-parse", f"{commit}^"],
        cwd=repo_path,
        capture_output=True, text=True, timeout=30
    )
    if result.returncode == 0:
        return result.stdout.strip()
    return None


def build_docker_image(
    commit: str,
    work_dir: Path,
    push: bool = True,
    verbose: bool = True
) -> Tuple[bool, str]:
    """
    Build Docker image for a specific SGLang commit.

    Args:
        commit: Full commit hash
        work_dir: Working directory for build
        push: Whether to push to DockerHub
        verbose: Print detailed output

    Returns:
        Tuple of (success, message)
    """
    short_commit = commit[:8]
    full_commit = commit[:40] if len(commit) >= 40 else commit

    if verbose:
        print(f"\n{'='*60}")
        print(f"Building SGLang Docker image for {short_commit}")
        print(f"{'='*60}")

    # Get torch version from config or auto-detect
    config = COMMIT_CONFIG.get(short_commit, {})
    torch_version = config.get("torch", DEFAULT_TORCH_VERSION)
    torch_minor = config.get("torch_minor", ".".join(torch_version.split(".")[:2]))

    if verbose:
        print(f"  Torch version: {torch_version}")
        print(f"  Torch minor: {torch_minor}")

    # Create work directory
    work_dir.mkdir(parents=True, exist_ok=True)
    repo_dir = work_dir / "sglang"

    try:
        # Clone or update repo
        if not repo_dir.exists():
            if verbose:
                print(f"Cloning SGLang repository...")
            result = subprocess.run(
                ["git", "clone", "--depth", "100", SGLANG_REPO_URL, str(repo_dir)],
                capture_output=True, text=True, timeout=300
            )
            if result.returncode != 0:
                return False, f"Git clone failed: {result.stderr[:500]}"

        # Fetch and checkout commit
        if verbose:
            print(f"Checking out commit {short_commit}...")

        subprocess.run(
            ["git", "fetch", "origin", commit],
            cwd=repo_dir, capture_output=True, timeout=120
        )

        result = subprocess.run(
            ["git", "checkout", commit],
            cwd=repo_dir, capture_output=True, text=True, timeout=60
        )
        if result.returncode != 0:
            return False, f"Git checkout failed: {result.stderr[:500]}"

        # Write Dockerfile with correct substitutions
        dockerfile_path = repo_dir / "Dockerfile.benchmark"
        dockerfile_content = BENCHMARK_DOCKERFILE.format(
            sglang_repo=SGLANG_REPO_URL
        )

        dockerfile_path.write_text(dockerfile_content)

        # Build Docker image
        image_tag = f"{DOCKER_REPO}:{full_commit}"

        if verbose:
            print(f"Building Docker image: {image_tag}")

        build_cmd = [
            "docker", "build",
            "-f", str(dockerfile_path),
            "-t", image_tag,
            "--build-arg", f"COMMIT_HASH={full_commit}",
            "--build-arg", f"TORCH_VERSION={torch_version}",
            "--build-arg", f"TORCH_MINOR={torch_minor}",
            str(repo_dir)
        ]

        result = subprocess.run(
            build_cmd,
            capture_output=not verbose,
            text=True,
            timeout=7200  # 2 hour timeout for build (sgl-kernel takes time)
        )

        if result.returncode != 0:
            error_msg = result.stderr if hasattr(result, 'stderr') else "Build failed"
            return False, f"Docker build failed: {error_msg[:500]}"

        if verbose:
            print(f"Docker image built successfully: {image_tag}")

        # Push to DockerHub
        if push:
            if verbose:
                print(f"Pushing image to DockerHub...")

            result = subprocess.run(
                ["docker", "push", image_tag],
                capture_output=not verbose,
                text=True,
                timeout=1800  # 30 min timeout for push
            )

            if result.returncode != 0:
                error_msg = result.stderr if hasattr(result, 'stderr') else "Push failed"
                return False, f"Docker push failed: {error_msg[:500]}"

            if verbose:
                print(f"Image pushed successfully: {image_tag}")

        return True, f"Successfully built {short_commit}"

    except subprocess.TimeoutExpired as e:
        return False, f"Timeout during build: {str(e)}"
    except Exception as e:
        return False, f"Build exception: {str(e)}"


def main():
    parser = argparse.ArgumentParser(description="Build SGLang Docker images")
    parser.add_argument("--commit", type=str, help="Build image for specific commit")
    parser.add_argument("--all", action="store_true", help="Build images for all runnable commits")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be built")
    parser.add_argument("--no-push", action="store_true", help="Don't push to DockerHub")
    parser.add_argument("--skip-existing", action="store_true", default=True, help="Skip commits with existing images")
    parser.add_argument("--verbose", action="store_true", default=True, help="Verbose output")
    parser.add_argument("--work-dir", type=str, default=str(WORK_DIR), help="Working directory")
    args = parser.parse_args()

    if not args.commit and not args.all:
        parser.print_help()
        print("\nError: Must specify --commit or --all")
        sys.exit(1)

    work_dir = Path(args.work_dir)

    # Get commits to build
    if args.commit:
        commits = [{"short_commit": args.commit[:8], "full_commit": args.commit, "subject": "Manual build"}]
    else:
        commits = get_runnable_commits()
        print(f"Found {len(commits)} runnable commits")

    if not commits:
        print("No commits to build")
        sys.exit(0)

    # Check existing images
    existing = get_existing_images() if args.skip_existing else set()
    if existing:
        print(f"Found {len(existing)} existing images on DockerHub")

    # Filter commits
    to_build = []
    skipped = 0
    for c in commits:
        if args.skip_existing and c['short_commit'] in existing:
            skipped += 1
            continue
        to_build.append(c)

    print(f"Commits to build: {len(to_build)} (skipped {skipped} existing)")

    if args.dry_run:
        print("\n=== DRY RUN - Would build: ===")
        for i, c in enumerate(to_build, 1):
            print(f"  {i}. {c['short_commit']} - {c['subject'][:50]}...")
        sys.exit(0)

    # Build images
    success_count = 0
    error_count = 0

    for i, c in enumerate(to_build, 1):
        print(f"\n[{i}/{len(to_build)}] Building {c['short_commit']}: {c['subject'][:40]}...")

        success, msg = build_docker_image(
            commit=c['full_commit'],
            work_dir=work_dir,
            push=not args.no_push,
            verbose=args.verbose
        )

        if success:
            success_count += 1
            print(f"  SUCCESS: {msg}")
        else:
            error_count += 1
            print(f"  ERROR: {msg}")

    # Summary
    print(f"\n{'='*60}")
    print("BUILD SUMMARY")
    print(f"{'='*60}")
    print(f"  Total: {len(to_build)}")
    print(f"  Success: {success_count}")
    print(f"  Errors: {error_count}")
    print(f"  Skipped (existing): {skipped}")


if __name__ == "__main__":
    main()
