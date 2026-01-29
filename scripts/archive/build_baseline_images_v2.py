#!/usr/bin/env python3
"""
Build baseline Docker images for vLLM parent commits (V2 - Optimized).

This script builds Docker images with vLLM installed from source at the parent commit.
Uses MAX_JOBS=40 and NVCC_THREADS=2 for faster compilation on multi-core machines.

Supports both v1 and v2 mapping files and the combined baseline_build_list.json.

Usage:
    python3 build_baseline_images_v2.py --commit PARENT_HASH  # Build single commit
    python3 build_baseline_images_v2.py --all                 # Build all missing commits
    python3 build_baseline_images_v2.py --dry-run             # Show what would be built
    python3 build_baseline_images_v2.py --from-list           # Build from baseline_build_list.json
"""

import json
import os
import subprocess
import sys
import tempfile
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Configuration
SCRIPT_DIR = Path(__file__).parent
BASELINE_MAPPING_V1 = SCRIPT_DIR / "baseline_benchmark_mapping.json"
BASELINE_MAPPING_V2 = SCRIPT_DIR / "baseline_benchmark_mapping_v2.json"
BASELINE_BUILD_LIST = SCRIPT_DIR / "baseline_build_list.json"

# Docker Hub repositories
BASELINE_IMAGE_PREFIX = "shikhar481/vllm_fixed_human_images"  # Push baselines here
HUMAN_IMAGE_PREFIX = "ayushnangia16/nvidia-vllm-docker"
FIXED_IMAGE_PREFIX = "shikhar481/vllm_fixed_human_images"

# Build configuration - OPTIMIZED
# Reduced further for OOM stability
MAX_JOBS = 56  # -j 56 for stability
NVCC_THREADS = 2
TORCH_CUDA_ARCH_LIST = "9.0"  # H100
BUILD_TIMEOUT = 5400  # 90 minutes

# Commits that need fixed images (from previous analysis)
FIXED_IMAGE_COMMITS = {
    "015069b0", "22dd9c27", "67da5720", "d55e446d", "e493e485",  # aimv2 fix
    "35fad35a", "3092375e", "93e5f3c5", "9d72daf4", "b10e5198",  # V1 engine fix
    "ad8d696a", "b6d10354",  # NumPy fix
}

# Parent commits that need older CUDA toolkit (old vLLM versions with CUDA API mismatch)
# These fail with "undefined symbol: cuPointerGetAttribute" when using cuda-toolkit-12-4
# Will use cuda-toolkit-11-8 instead
USE_OLD_CUDA_COMMITS = {
    "51c31bc10ca7",  # vLLM v0.4.0 - CUDA API mismatch
}

# Commit-specific retry configurations for previously failed builds
# Each config can override: max_jobs, pre_build_patch (shell commands to run before pip install)
RETRY_CONFIGS = {
    "9bde5ba12709": {
        "max_jobs": 32,
        # Fix Python version constraint: <=3.12 doesn't accept 3.12.11
        "pre_build_patch": """
echo "=== Patching Python version constraint ===" && \\
sed -i 's/<=3.12/<3.13/g' pyproject.toml 2>/dev/null || true && \\
sed -i 's/<=3.12/<3.13/g' setup.py 2>/dev/null || true && \\
grep -i "python" pyproject.toml | head -3 || true && \\
echo "=== Python constraint patched ==="
"""
    },
    "bc8a8ce5ec37": {
        "max_jobs": 32,  # Was OOM during parallel sglang build
    },
    "7206ce4ce112": {
        "max_jobs": 24,  # CUDA stubs library path fix required
        "pre_build_patch": """
echo "=== Adding CUDA stubs to library path ===" && \\
export LIBRARY_PATH="/usr/local/cuda/lib64/stubs:$LIBRARY_PATH" && \\
export LD_LIBRARY_PATH="/usr/local/cuda/lib64/stubs:/usr/local/cuda/lib64:$LD_LIBRARY_PATH" && \\
echo "LIBRARY_PATH=$LIBRARY_PATH"
"""
    },
    "a869baca73eb": {
        "max_jobs": 24,  # cmake exit 255, has many CUDA kernels
    },
    "f728ab8e3578": {
        "max_jobs": 24,  # cmake exit 255 at step 304/314
    },
}

# Existing baseline images on Docker Hub (from analysis + newly built)
EXISTING_BASELINES = {
    "a4d577b37944", "95baec828f3e", "2b04c209ee98", "f508e03e7f2d",
    "1d35662e6dc1", "f721096d48a7", "51f8aa90ad40", "6dd55af6c9dd",
    "beebf4742af8", "5c04bb8b863b", "70363bccfac1", "388596c91437",
    "0fca3cdcf265", "084a01fd3544", "bd43973522ea", "51e971d39e12",
    "dd2a6a82e3f4", "95a178f86120", "6a11fdfbb8d6", "64172a976c8d",
    "ebce310b7433", "b0e96aaebbfb", "fbefc8a78d22", "f1c852014603",
    "0032903a5bb7", "36fb68f94792", "54600709b6d4",
    # Newly built and pushed
    "3cdfe1f38b2c", "005ae9be6c22", "067c34a15594",
    "10904e6d7550", "1da8f0e1ddda", "25373b6c6cc2",
    "270a5da495d2", "2f385183f354", "20478c4d3abc",
    "3014c920dae5", "333681408fea", "3a1e6481586e",
    "3cd91dc9555e", "526078a96c52", "5b8a1fde8422", "5fc5ce0fe45f",
    # Permanently failed (don't retry - fundamental incompatibility)
    "0e74d797ce86", "2a0309a646b1",
    "51c31bc10ca7",  # vLLM v0.4.0 - CUDA 11/12 API mismatch, cannot build with base image
    # Fixed with CUDA stubs library path:
    "7206ce4ce112",  # Needed LIBRARY_PATH="/usr/local/cuda/lib64/stubs:$LIBRARY_PATH"
    # Successfully built after retry with commit-specific fixes:
    "9bde5ba12709", "bc8a8ce5ec37", "a869baca73eb", "f728ab8e3578",
    # Additional user-requested builds:
    "029c71de11bc",
}


def get_human_image(human_commit_short: str, human_commit_full: str) -> str:
    """Get the appropriate human Docker image for a commit."""
    if human_commit_short in FIXED_IMAGE_COMMITS:
        return f"{FIXED_IMAGE_PREFIX}:{human_commit_full}"
    return f"{HUMAN_IMAGE_PREFIX}:{human_commit_full}"


def load_build_list() -> List[Dict]:
    """Load the combined build list."""
    if BASELINE_BUILD_LIST.exists():
        with open(BASELINE_BUILD_LIST) as f:
            return json.load(f)
    return []


def load_all_mappings() -> Dict[str, Dict]:
    """Load and combine all mapping files."""
    all_parents = {}

    # Load v1 mapping
    if BASELINE_MAPPING_V1.exists():
        with open(BASELINE_MAPPING_V1) as f:
            for entry in json.load(f):
                parent = entry['parent_commit']
                if parent[:12] not in all_parents:
                    all_parents[parent[:12]] = {
                        'parent_full': parent,
                        'human_short': entry['human_commit_short'],
                        'human_full': entry['human_commit_full'],
                        'model': entry.get('model', 'unknown'),
                        'source': 'v1'
                    }

    # Load v2 mapping
    if BASELINE_MAPPING_V2.exists():
        with open(BASELINE_MAPPING_V2) as f:
            for entry in json.load(f):
                parent = entry.get('parent_commit_full', entry.get('parent_commit', ''))
                if parent and parent[:12] not in all_parents:
                    all_parents[parent[:12]] = {
                        'parent_full': parent,
                        'human_short': entry['human_commit_short'],
                        'human_full': entry['human_commit_full'],
                        'model': entry.get('model', 'unknown'),
                        'source': 'v2'
                    }

    return all_parents


def create_dockerfile(parent_commit: str, human_image: str, use_base_cuda: bool = False,
                      custom_max_jobs: int = None, pre_build_patch: str = None) -> str:
    """Create a Dockerfile for building baseline vLLM at parent commit.

    Args:
        parent_commit: The vLLM parent commit hash
        human_image: The base Docker image to build from
        use_base_cuda: If True, use cuda-toolkit-11-8 instead of 12-4
        custom_max_jobs: Override MAX_JOBS for this specific build
        pre_build_patch: Shell commands to run after git clone but before pip install
    """
    # Use custom max_jobs if provided, otherwise use global default
    build_max_jobs = custom_max_jobs if custom_max_jobs is not None else MAX_JOBS

    # Choose CUDA install based on commit compatibility
    if use_base_cuda:
        cuda_install = """# Use older CUDA toolkit for old vLLM versions (CUDA API compatibility)
RUN apt-get update -qq && \\
    apt-get install -y -qq git cuda-toolkit-11-8 && \\
    rm -rf /var/lib/apt/lists/* && \\
    rm -f /usr/local/cuda && \\
    ln -s /usr/local/cuda-11.8 /usr/local/cuda"""
    else:
        cuda_install = """# Install CUDA toolkit and git for building
RUN apt-get update -qq && \\
    apt-get install -y -qq git cuda-toolkit-12-4 && \\
    rm -rf /var/lib/apt/lists/* && \\
    rm -f /usr/local/cuda && \\
    ln -s /usr/local/cuda-12.4 /usr/local/cuda"""

    # Build pre_build_patch section if provided
    pre_build_section = ""
    if pre_build_patch:
        pre_build_section = f"""
# Commit-specific pre-build patch
RUN {pre_build_patch.strip()}
"""

    return f'''# Baseline vLLM image for parent commit {parent_commit[:12]}
# Built from human image {human_image}
# Build settings: MAX_JOBS={build_max_jobs}, NVCC_THREADS={NVCC_THREADS}
# Using base CUDA: {use_base_cuda}

FROM {human_image}

{cuda_install}

# Clone vLLM at parent commit
WORKDIR /opt
RUN git clone --depth 1 https://github.com/vllm-project/vllm.git vllm_baseline && \\
    cd vllm_baseline && \\
    git fetch --depth 1 origin {parent_commit} && \\
    git checkout {parent_commit}

# Install build dependencies
RUN pip install uv -q && \\
    uv pip install setuptools setuptools_scm wheel packaging ninja cmake 'numpy<2' --system && \\
    pip install git+https://github.com/NICTA/pyairports.git -q

# Uninstall human vLLM
RUN uv pip uninstall vllm --system || true

# Build vLLM from source with optimized settings
WORKDIR /opt/vllm_baseline
ENV TORCH_CUDA_ARCH_LIST="{TORCH_CUDA_ARCH_LIST}"
ENV MAX_JOBS={build_max_jobs}
ENV NVCC_THREADS={NVCC_THREADS}
ENV CMAKE_BUILD_PARALLEL_LEVEL={build_max_jobs}
ENV VLLM_MAX_JOBS={build_max_jobs}
{pre_build_section}
# Force parallel jobs - comprehensive patching for all vLLM versions
# 1. Disable the nvcc_threads division that halves MAX_JOBS
# 2. Replace any hardcoded -j values in CMake
# 3. Force num_jobs to our value
RUN echo "=== Pre-build: Patching parallel job settings ===" && \\
    echo "Target: -j {build_max_jobs} (MAX_JOBS={build_max_jobs}, will be divided by NVCC_THREADS={NVCC_THREADS})" && \\
    grep -n "num_jobs // nvcc_threads" setup.py && echo "Found division logic" || echo "No division logic found" && \\
    sed -i 's/num_jobs = max(1, num_jobs \/\/ nvcc_threads)/num_jobs = num_jobs  # PATCHED: skip nvcc division/g' setup.py 2>/dev/null || true && \\
    sed -i 's/num_jobs \/\/ nvcc_threads/num_jobs/g' setup.py 2>/dev/null || true && \\
    sed -i -E 's/-j[= ]+[0-9]+/-j {build_max_jobs}/g' CMakeLists.txt 2>/dev/null || true && \\
    sed -i -E 's/-j[= ]+[0-9]+/-j {build_max_jobs}/g' setup.py 2>/dev/null || true && \\
    grep -n "\-j" CMakeLists.txt setup.py 2>/dev/null | head -5 || true && \\
    echo "=== Patching complete ==="

RUN pip install -e . --no-build-isolation && echo "BUILD_SUCCESS"

# Fix aimv2 config registration conflict (for newer vLLM versions)
RUN for f in /opt/vllm_baseline/vllm/transformers_utils/configs/ovis*.py; do \\
        if [ -f "$f" ] && grep -q 'AutoConfig.register("aimv2"' "$f"; then \\
            sed -i 's/AutoConfig.register("aimv2", AIMv2Config)/AutoConfig.register("aimv2", AIMv2Config, exist_ok=True)/g' "$f"; \\
        fi; \\
    done || true

# Download fa_utils.py if needed (for V1 engine)
RUN if [ -d "/opt/vllm_baseline/vllm/vllm_flash_attn" ] && [ ! -f "/opt/vllm_baseline/vllm/vllm_flash_attn/fa_utils.py" ]; then \\
        curl -s "https://raw.githubusercontent.com/vllm-project/vllm/{parent_commit}/vllm/vllm_flash_attn/fa_utils.py" \\
            -o /opt/vllm_baseline/vllm/vllm_flash_attn/fa_utils.py 2>/dev/null || true; \\
    fi

# Install pyairports from GitHub (PyPI version is a placeholder, not the real package)
RUN pip install git+https://github.com/NICTA/pyairports.git -q

# Verify installation
RUN python3 -c "import vllm; print(f'vLLM version: {{vllm.__version__}}')"

# Install benchmark dependencies
RUN uv pip install aiohttp pandas datasets -q --system

WORKDIR /
LABEL baseline.parent_commit="{parent_commit}"
LABEL baseline.build_date="{datetime.now().isoformat()}"
'''


def check_image_exists(image_tag: str) -> bool:
    """Check if a Docker image exists on Docker Hub."""
    try:
        result = subprocess.run(
            ['sudo', 'docker', 'manifest', 'inspect', image_tag],
            capture_output=True, timeout=30
        )
        return result.returncode == 0
    except Exception:
        return False


def build_baseline_image(parent_commit: str, human_commit_short: str, human_commit_full: str,
                         push: bool = True, timeout: int = BUILD_TIMEOUT) -> Tuple[bool, str]:
    """Build a baseline Docker image for a parent commit."""

    human_image = get_human_image(human_commit_short, human_commit_full)
    baseline_tag = f"{BASELINE_IMAGE_PREFIX}:baseline-{parent_commit[:12]}"

    # Check for commit-specific retry configuration
    retry_config = RETRY_CONFIGS.get(parent_commit[:12], {})
    custom_max_jobs = retry_config.get("max_jobs")
    pre_build_patch = retry_config.get("pre_build_patch")
    effective_max_jobs = custom_max_jobs if custom_max_jobs else MAX_JOBS

    print(f"\n{'='*70}")
    print(f"[{datetime.now().strftime('%H:%M:%S')}] Building baseline: {parent_commit[:12]}")
    print(f"  Human commit: {human_commit_short}")
    print(f"  Human image: {human_image}")
    print(f"  Target tag: {baseline_tag}")
    print(f"  Settings: MAX_JOBS={effective_max_jobs}, NVCC_THREADS={NVCC_THREADS}")
    if retry_config:
        print(f"  [RETRY] Using commit-specific config: max_jobs={custom_max_jobs}, has_patch={bool(pre_build_patch)}")
    print(f"{'='*70}")

    # Check if baseline already exists
    if check_image_exists(baseline_tag):
        print(f"  [SKIP] Image already exists on Docker Hub")
        return True, "already_exists"

    # Check if human image exists
    if not check_image_exists(human_image):
        print(f"  [ERROR] Human image not found: {human_image}")
        return False, "human_image_missing"

    # Create Dockerfile (check if commit needs older CUDA)
    use_old_cuda = parent_commit[:12] in USE_OLD_CUDA_COMMITS
    if use_old_cuda:
        print(f"  [INFO] Using cuda-toolkit-11-8 for this commit (old vLLM version)")
    dockerfile_content = create_dockerfile(
        parent_commit, human_image, use_old_cuda,
        custom_max_jobs=custom_max_jobs, pre_build_patch=pre_build_patch
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        dockerfile_path = Path(tmpdir) / "Dockerfile"
        dockerfile_path.write_text(dockerfile_content)

        # Build image
        start_time = time.time()
        print(f"  [BUILD] Starting Docker build...")

        try:
            result = subprocess.run(
                ['sudo', 'docker', 'build',
                 '-t', baseline_tag,
                 '-f', str(dockerfile_path),
                 tmpdir],
                capture_output=True, text=True, timeout=timeout
            )

            duration = time.time() - start_time

            if result.returncode != 0:
                print(f"  [FAILED] Build failed after {duration:.0f}s")
                # Save error log
                error_log = SCRIPT_DIR / f"build_errors/{parent_commit[:12]}_error.log"
                error_log.parent.mkdir(exist_ok=True)
                error_log.write_text(f"STDOUT:\n{result.stdout[-5000:]}\n\nSTDERR:\n{result.stderr[-5000:]}")
                print(f"  Error log: {error_log}")
                return False, "build_failed"

            print(f"  [SUCCESS] Build completed in {duration:.0f}s")

            # Push to Docker Hub
            if push:
                print(f"  [PUSH] Pushing to Docker Hub...")
                push_result = subprocess.run(
                    ['sudo', 'docker', 'push', baseline_tag],
                    capture_output=True, text=True, timeout=600
                )
                if push_result.returncode != 0:
                    print(f"  [ERROR] Push failed: {push_result.stderr[-500:]}")
                    return False, "push_failed"
                print(f"  [PUSHED] {baseline_tag}")

            return True, "success"

        except subprocess.TimeoutExpired:
            print(f"  [TIMEOUT] Build exceeded {timeout}s")
            return False, "timeout"
        except Exception as e:
            print(f"  [ERROR] {e}")
            return False, str(e)


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Build baseline Docker images (v2 - optimized)')
    parser.add_argument('--commit', type=str, help='Build specific parent commit')
    parser.add_argument('--all', action='store_true', help='Build all missing parent commits')
    parser.add_argument('--from-list', action='store_true', help='Build from baseline_build_list.json')
    parser.add_argument('--dry-run', action='store_true', help='Show what would be built')
    parser.add_argument('--no-push', action='store_true', help='Do not push to Docker Hub')
    parser.add_argument('--skip-existing', action='store_true', default=True,
                        help='Skip commits that already have baselines (default: True)')
    args = parser.parse_args()

    print("="*70)
    print("BASELINE IMAGE BUILDER v2")
    print(f"  MAX_JOBS: {MAX_JOBS}")
    print(f"  NVCC_THREADS: {NVCC_THREADS}")
    print(f"  TORCH_CUDA_ARCH_LIST: {TORCH_CUDA_ARCH_LIST}")
    print(f"  Target repository: {BASELINE_IMAGE_PREFIX}")
    print("="*70)

    # Load all parent commits
    all_parents = load_all_mappings()
    print(f"\nLoaded {len(all_parents)} unique parent commits from mappings")

    # Filter to only missing baselines
    if args.skip_existing:
        missing = {k: v for k, v in all_parents.items() if k not in EXISTING_BASELINES}
        print(f"Already have baselines: {len(EXISTING_BASELINES)}")
        print(f"Missing baselines: {len(missing)}")
    else:
        missing = all_parents

    if args.dry_run:
        print("\n--- DRY RUN: Would build these baselines ---")
        for i, (parent, info) in enumerate(sorted(missing.items()), 1):
            human_img = get_human_image(info['human_short'], info['human_full'])
            print(f"{i:2}. {parent} <- {info['human_short']} ({info['source']}) | {info['model'][:30]}")
        print(f"\nTotal: {len(missing)} baseline images to build")
        return 0

    if args.commit:
        # Build single commit
        matching = [k for k in all_parents if k.startswith(args.commit) or
                    all_parents[k]['parent_full'].startswith(args.commit)]
        if not matching:
            print(f"No parent commit found matching: {args.commit}")
            return 1

        parent_short = matching[0]
        info = all_parents[parent_short]
        success, status = build_baseline_image(
            info['parent_full'], info['human_short'], info['human_full'],
            push=not args.no_push
        )
        return 0 if success else 1

    elif args.all or args.from_list:
        # Build all missing commits
        results = []
        total = len(missing)

        for i, (parent_short, info) in enumerate(sorted(missing.items()), 1):
            print(f"\n[{i}/{total}] Processing {parent_short}...")
            success, status = build_baseline_image(
                info['parent_full'], info['human_short'], info['human_full'],
                push=not args.no_push
            )
            results.append({
                'parent': parent_short,
                'human': info['human_short'],
                'success': success,
                'status': status
            })

            # Save progress
            progress_file = SCRIPT_DIR / "build_progress.json"
            with open(progress_file, 'w') as f:
                json.dump(results, f, indent=2)

        # Print summary
        print("\n" + "="*70)
        print("BUILD SUMMARY")
        print("="*70)

        success_count = sum(1 for r in results if r['success'])
        print(f"Success: {success_count}/{len(results)}")

        for r in results:
            status_icon = "OK" if r['success'] else "FAIL"
            print(f"  [{status_icon}] {r['parent']} <- {r['human']}: {r['status']}")

        return 0 if success_count == len(results) else 1

    else:
        parser.print_help()
        return 1


if __name__ == '__main__':
    sys.exit(main() or 0)
