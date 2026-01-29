#!/usr/bin/env python3
"""
Build baseline Docker images for vLLM parent commits.

This script builds Docker images with vLLM installed from source at the parent commit.
These pre-built images can then be used for fast baseline benchmarks (no build time).

Usage:
    python3 build_baseline_images.py --commit PARENT_HASH  # Build single commit
    python3 build_baseline_images.py --all                 # Build all parent commits
    python3 build_baseline_images.py --dry-run             # Show what would be built
"""

import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

# Configuration - Compute project root dynamically
ROOT_DIR = Path(__file__).resolve().parent.parent.parent  # scripts/docker/ -> OmniPerf-Bench/

BASELINE_MAPPING_FILE = ROOT_DIR / "baseline_benchmark_mapping.json"
BASELINE_IMAGE_PREFIX = "ayushnangia16/vllm-baseline"  # Or your Docker Hub username
HUMAN_IMAGE_PREFIX = "ayushnangia16/nvidia-vllm-docker"
FIXED_IMAGE_PREFIX = "shikhar481/vllm_fixed_human_images"

# Commits that need fixed images
FIXED_IMAGE_COMMITS = {
    "015069b0", "22dd9c27", "67da5720", "d55e446d", "e493e485",
    "35fad35a", "3092375e", "93e5f3c5", "9d72daf4", "b10e5198",
    "ad8d696a", "b6d10354",
}


def get_human_image(human_commit_short: str, human_commit_full: str) -> str:
    """Get the appropriate human Docker image for a commit."""
    if human_commit_short in FIXED_IMAGE_COMMITS:
        return f"{FIXED_IMAGE_PREFIX}:{human_commit_full}"
    return f"{HUMAN_IMAGE_PREFIX}:{human_commit_full}"


def load_baseline_mapping():
    """Load the baseline mapping file."""
    with open(BASELINE_MAPPING_FILE) as f:
        return json.load(f)


def get_unique_parent_commits(mapping):
    """Get unique parent commits from the mapping."""
    parent_commits = {}
    for entry in mapping:
        parent = entry['parent_commit']
        if parent not in parent_commits:
            parent_commits[parent] = entry
    return parent_commits


def create_dockerfile(parent_commit: str, human_image: str) -> str:
    """Create a Dockerfile for building baseline vLLM at parent commit."""
    return f'''# Baseline vLLM image for parent commit {parent_commit[:8]}
# Built from human image {human_image}

FROM {human_image}

# Install CUDA toolkit for building
RUN apt-get update -qq && \\
    apt-get install -y -qq cuda-toolkit-12-4 && \\
    rm -rf /var/lib/apt/lists/*

# Clone vLLM at parent commit
WORKDIR /opt
RUN git clone --depth 1 https://github.com/vllm-project/vllm.git vllm_baseline && \\
    cd vllm_baseline && \\
    git fetch --depth 1 origin {parent_commit} && \\
    git checkout {parent_commit}

# Install build dependencies
RUN pip install uv -q && \\
    uv pip install setuptools wheel packaging ninja cmake --system

# Uninstall human vLLM
RUN uv pip uninstall vllm --system || true

# Build vLLM from source
WORKDIR /opt/vllm_baseline
ENV TORCH_CUDA_ARCH_LIST="9.0"
ENV MAX_JOBS=4
RUN pip install -e . --no-build-isolation

# Fix aimv2 config registration conflict
RUN if grep -q 'AutoConfig.register("aimv2"' /opt/vllm_baseline/vllm/transformers_utils/configs/ovis2.py 2>/dev/null; then \\
        sed -i 's/AutoConfig.register("aimv2", AIMv2Config)/AutoConfig.register("aimv2", AIMv2Config, exist_ok=True)/' /opt/vllm_baseline/vllm/transformers_utils/configs/ovis2.py; \\
    fi

# Verify installation
RUN python3 -c "import vllm; print(f'vLLM version: {{vllm.__version__}}')"

# Install benchmark dependencies
RUN uv pip install aiohttp pandas datasets -q --system

WORKDIR /
'''


def build_baseline_image(parent_commit: str, human_commit_short: str, human_commit_full: str,
                         push: bool = True, timeout: int = 3600) -> bool:
    """Build a baseline Docker image for a parent commit."""

    human_image = get_human_image(human_commit_short, human_commit_full)
    baseline_tag = f"{BASELINE_IMAGE_PREFIX}:{parent_commit}"

    print(f"\n{'='*60}")
    print(f"Building baseline image for parent commit: {parent_commit[:8]}")
    print(f"Using human image: {human_image}")
    print(f"Target tag: {baseline_tag}")
    print(f"{'='*60}")

    # Check if image already exists
    result = subprocess.run(
        ['docker', 'manifest', 'inspect', baseline_tag],
        capture_output=True, timeout=30
    )
    if result.returncode == 0:
        print(f"Image {baseline_tag} already exists. Skipping.")
        return True

    # Create Dockerfile
    dockerfile_content = create_dockerfile(parent_commit, human_image)

    with tempfile.TemporaryDirectory() as tmpdir:
        dockerfile_path = Path(tmpdir) / "Dockerfile"
        dockerfile_path.write_text(dockerfile_content)

        # Build image
        print("Building Docker image...")
        try:
            result = subprocess.run(
                ['docker', 'build', '-t', baseline_tag, '-f', str(dockerfile_path), tmpdir],
                capture_output=True, text=True, timeout=timeout
            )

            if result.returncode != 0:
                print(f"BUILD FAILED for {parent_commit[:8]}")
                print(f"STDERR: {result.stderr[-3000:]}")
                return False

            print(f"BUILD SUCCESS for {parent_commit[:8]}")

            # Push to Docker Hub
            if push:
                print(f"Pushing to Docker Hub: {baseline_tag}")
                push_result = subprocess.run(
                    ['docker', 'push', baseline_tag],
                    capture_output=True, text=True, timeout=600
                )
                if push_result.returncode != 0:
                    print(f"PUSH FAILED: {push_result.stderr}")
                    return False
                print(f"PUSH SUCCESS")

            return True

        except subprocess.TimeoutExpired:
            print(f"BUILD TIMEOUT for {parent_commit[:8]}")
            return False
        except Exception as e:
            print(f"BUILD ERROR for {parent_commit[:8]}: {e}")
            return False


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Build baseline Docker images')
    parser.add_argument('--commit', type=str, help='Build specific parent commit')
    parser.add_argument('--all', action='store_true', help='Build all parent commits')
    parser.add_argument('--dry-run', action='store_true', help='Show what would be built')
    parser.add_argument('--no-push', action='store_true', help='Do not push to Docker Hub')
    args = parser.parse_args()

    # Load mapping
    mapping = load_baseline_mapping()
    parent_commits = get_unique_parent_commits(mapping)

    print(f"Found {len(parent_commits)} unique parent commits")

    if args.dry_run:
        print("\nParent commits to build:")
        for parent, entry in parent_commits.items():
            human_short = entry['human_commit_short']
            human_full = entry['human_commit_full']
            human_image = get_human_image(human_short, human_full)
            print(f"  {parent[:8]} (from human {human_short}) <- {human_image}")
        return

    if args.commit:
        # Build single commit
        matching = [p for p in parent_commits if p.startswith(args.commit)]
        if not matching:
            print(f"No parent commit found matching {args.commit}")
            return 1
        parent = matching[0]
        entry = parent_commits[parent]
        success = build_baseline_image(
            parent, entry['human_commit_short'], entry['human_commit_full'],
            push=not args.no_push
        )
        return 0 if success else 1

    elif args.all:
        # Build all commits
        results = []
        for parent, entry in parent_commits.items():
            success = build_baseline_image(
                parent, entry['human_commit_short'], entry['human_commit_full'],
                push=not args.no_push
            )
            results.append((parent[:8], success))

        print("\n" + "="*60)
        print("BUILD SUMMARY")
        print("="*60)
        success_count = sum(1 for _, s in results if s)
        print(f"Success: {success_count}/{len(results)}")
        for parent, success in results:
            status = "SUCCESS" if success else "FAILED"
            print(f"  {parent}: {status}")

        return 0 if all(s for _, s in results) else 1

    else:
        parser.print_help()
        return 1


if __name__ == '__main__':
    sys.exit(main() or 0)
