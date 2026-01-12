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

# Build configuration
# Using a simpler Dockerfile for benchmarking (not the full production Dockerfile)
BENCHMARK_DOCKERFILE = '''
# Simplified SGLang Docker image for benchmarking
# Based on NVIDIA's CUDA image with SGLang installed from source

ARG CUDA_VERSION=12.4.0
FROM nvidia/cuda:${CUDA_VERSION}-cudnn-devel-ubuntu22.04

ARG TARGETARCH=amd64
ARG COMMIT_HASH=main

ENV DEBIAN_FRONTEND=noninteractive \\
    CUDA_HOME=/usr/local/cuda \\
    PATH="${PATH}:/usr/local/cuda/bin" \\
    LD_LIBRARY_PATH="${LD_LIBRARY_PATH}:/usr/local/cuda/lib64"

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \\
    python3.11 python3.11-dev python3.11-venv python3-pip \\
    git curl wget build-essential cmake ninja-build \\
    libopenmpi-dev libnuma-dev \\
    && ln -sf /usr/bin/python3.11 /usr/bin/python3 \\
    && ln -sf /usr/bin/python3.11 /usr/bin/python \\
    && rm -rf /var/lib/apt/lists/* \\
    && apt-get clean

# Install pip and basic Python packages
RUN python3 -m pip install --upgrade pip setuptools wheel

# Install PyTorch with CUDA support
RUN pip install torch==2.4.0 --index-url https://download.pytorch.org/whl/cu124

# Clone SGLang at specific commit
WORKDIR /opt
RUN git clone ${SGLANG_REPO_URL} sglang \\
    && cd sglang \\
    && git checkout ${COMMIT_HASH}

# Install SGLang dependencies first
RUN pip install transformers>=4.40.0 huggingface_hub>=0.23.0 tokenizers>=0.19.0 \\
    accelerate>=0.30.0 numpy<2.0 requests aiohttp \\
    triton>=3.0.0 packaging ninja

# Install flashinfer (required for SGLang)
RUN pip install flashinfer-python -i https://flashinfer.ai/whl/cu124/torch2.4/

# Install sgl-kernel from PyPI (contains deep_gemm and other CUDA kernels)
# This is CRITICAL - without it, SGLang server crashes with "No module named 'deep_gemm'"
RUN pip install sgl-kernel

# Install SGLang from source
WORKDIR /opt/sglang
RUN pip install -e "python[all]" || pip install -e "python"

# Install benchmark dependencies
RUN pip install datasets pandas tqdm pybase64 Pillow

# Verify installation (including sgl-kernel)
RUN python -c "import sglang; print(f'SGLang version: {sglang.__version__}')"
RUN python -c "import sgl_kernel; import deep_gemm; print('sgl-kernel and deep_gemm OK')"

# Set working directory for benchmarks
WORKDIR /workspace

# Copy benchmark scripts from SGLang repo
RUN cp -r /opt/sglang/python/sglang/bench_* /workspace/ 2>/dev/null || true
RUN cp -r /opt/sglang/benchmark* /workspace/ 2>/dev/null || true
RUN cp -r /opt/sglang/benchmarks /workspace/ 2>/dev/null || true

# Default command
CMD ["python", "-c", "import sglang; print(f'SGLang {sglang.__version__} ready')"]
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

        # Write Dockerfile
        dockerfile_path = repo_dir / "Dockerfile.benchmark"
        dockerfile_content = BENCHMARK_DOCKERFILE.replace(
            "ARG COMMIT_HASH=main",
            f"ARG COMMIT_HASH={full_commit}"
        ).replace(
            "${SGLANG_REPO_URL}",
            SGLANG_REPO_URL
        ).replace(
            "git checkout ${COMMIT_HASH}",
            f"git checkout {full_commit}"
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
            str(repo_dir)
        ]

        result = subprocess.run(
            build_cmd,
            capture_output=not verbose,
            text=True,
            timeout=3600  # 1 hour timeout for build
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
