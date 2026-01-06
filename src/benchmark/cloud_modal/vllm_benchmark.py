"""
Modal-based benchmark runner for OmniPerf.

This module provides cloud GPU execution for benchmarks that require:
- Multi-GPU instances (models > 80GB VRAM)
- Specific GPU configurations (H100:4, H100:8)
- Clean, isolated environments

Usage:
    # Deploy the Modal app
    modal deploy src/eval/modal_benchmark.py

    # Or run directly
    modal run src/eval/modal_benchmark.py::run_benchmark_single_gpu --wheel-url "..." --perf-command "..."
"""

import modal
import subprocess
import time
import re
import os
import sys
import signal
import traceback
from typing import Dict, Optional, Any, Tuple
from pathlib import Path

# Modal app configuration
app = modal.App("omniperf-benchmark")

# Base image with CUDA, Python, and benchmark dependencies
base_image = (
    modal.Image.from_registry("nvidia/cuda:12.4.0-devel-ubuntu22.04", add_python="3.11")
    .apt_install(
        "git", "curl", "wget", "patch",
        # Build dependencies for vLLM compilation
        "build-essential", "cmake", "ninja-build",
        "libnuma-dev", "libucx-dev",
        # Port management tools for cleanup
        "psmisc",  # provides fuser for killing processes on ports
        "net-tools",  # provides netstat for port diagnostics
        "lsof",  # for process/port diagnostics
    )
    .pip_install(
        "torch==2.4.0",
        "transformers>=4.40.0",
        "huggingface_hub>=0.23.0",
        "tokenizers>=0.19.0",
        "accelerate>=0.30.0",
        "numpy<2.0",
        "requests",
        "aiohttp",
        # Benchmark dependencies
        "datasets",
        "pandas",
        "tqdm",
        # Build dependencies
        "ninja",
        "packaging",
        "setuptools>=61",
        "setuptools-scm>=8",
        "wheel",
    )
    .run_commands([
        # Install uv for faster, more reliable package management
        "curl -LsSf https://astral.sh/uv/install.sh | sh",
        "ln -sf /root/.local/bin/uv /usr/local/bin/uv",
        # Clean any broken uv config and ensure fresh state (force rebuild: v2)
        "rm -rf /root/.config/uv",
        # Install cmake 3.28+ (required by vLLM, Ubuntu 22.04 has 3.22 which is too old)
        "apt-get remove -y cmake || true",
        "curl -LO https://github.com/Kitware/CMake/releases/download/v3.28.3/cmake-3.28.3-linux-x86_64.tar.gz",
        "tar -xzf cmake-3.28.3-linux-x86_64.tar.gz -C /usr/local --strip-components=1",
        "rm cmake-3.28.3-linux-x86_64.tar.gz",
        # Clone vLLM repo for benchmark scripts
        "git clone --depth 1 https://github.com/vllm-project/vllm.git /opt/vllm-benchmarks",
    ])
    .env({
        # Cache bust - change to force image rebuild
        "MODAL_CACHE_BUST": "20260102_v21_inplace_build_with_lock",
        "HF_HOME": "/root/.cache/huggingface",
        "TRANSFORMERS_CACHE": "/root/.cache/huggingface",
        # CUDA environment for compilation
        "CUDA_HOME": "/usr/local/cuda",
        "PATH": "/usr/local/cuda/bin:$PATH",
        "LD_LIBRARY_PATH": "/usr/local/cuda/lib64:$LD_LIBRARY_PATH",
        # Skip wheel filename version check (vLLM wheels have mismatched versions)
        "UV_SKIP_WHEEL_FILENAME_CHECK": "1",
        # Fix for vLLM 0.6.x port binding issue (socket duplication during fork)
        # See: https://github.com/vllm-project/vllm/issues/8791
        "VLLM_WORKER_MULTIPROC_METHOD": "spawn",
    })
)

# Volume for caching models (v2 - no inode limit)
model_cache = modal.Volume.from_name("omniperf-model-cache-v2", create_if_missing=True, version=2)

# Volume for caching vLLM builds (v2 - no inode limit for parallel builds)
build_cache = modal.Volume.from_name("omniperf-build-cache-v2", create_if_missing=True, version=2)

# GPU configurations for different model sizes
GPU_CONFIGS = {
    "H100:1": {"gpu": "H100", "count": 1, "timeout": 3600},
    "H100:2": {"gpu": "H100", "count": 2, "timeout": 5400},
    "H100:4": {"gpu": "H100", "count": 4, "timeout": 7200},
    "H100:8": {"gpu": "H100", "count": 8, "timeout": 14400},
}

# CPU configuration for build-only tasks (no GPU needed for compilation)
# 24 parallel nvcc jobs can spike to 8GB+ each during complex kernel compilation
# 24 jobs × 8GB = 192GB peak + 64GB headroom = 256GB total
CPU_BUILD_CONFIG = {
    "cpu": 24,         # 24 CPUs for parallel compilation
    "memory": 262144,  # 256GB RAM - extra headroom for nvcc memory spikes
    "timeout": 5400,   # 90 min
}

# CRITICAL: Target only H100 (SM 9.0) instead of building for all architectures
# This reduces build time by ~60-80% (from 30-40min to ~10-15min)
# Without this, vLLM builds for SM 7.0, 7.5, 8.0, 8.6, 8.9, 9.0 (6 architectures!)
CUDA_BUILD_TARGETS = "9.0"  # H100 = SM 9.0


@app.function(
    image=base_image,
    cpu=CPU_BUILD_CONFIG["cpu"],
    memory=CPU_BUILD_CONFIG["memory"],
    timeout=CPU_BUILD_CONFIG["timeout"],
    volumes={"/cache": build_cache},
)
def build_vllm_cpu_only(commit_hash: str, force_build_dir: bool = False) -> Dict[str, Any]:
    """
    Pre-build vLLM wheel on CPU-only instance (no GPU needed).

    This function builds a WHEEL FILE that can be installed in any container.
    The wheel is cached in the Modal volume for fast reuse.

    vLLM CUDA kernel compilation is CPU-bound and does NOT require a GPU.
    The nvcc compiler cross-compiles for target GPU architectures.

    Args:
        commit_hash: vLLM commit to build
        force_build_dir: If True, always build from source even if wheel exists.
                        This ensures the build directory exists (needed for agent patches).

    Returns:
        Dict with keys:
        - success: bool
        - cache_hit: bool (True if wheel already existed)
        - version: str (vLLM version if successful)
        - wheel_path: str (path to cached wheel file)
        - error: str (error message if failed)
    """
    import os
    import subprocess
    import shutil
    import glob as glob_mod

    result = {
        "success": False,
        "cache_hit": False,
        "version": None,
        "wheel_path": None,
        "error": None,
        "commit": commit_hash,
    }

    cache_dir = "/cache"
    wheel_dir = f"{cache_dir}/wheels"
    build_path = f"{cache_dir}/build_{commit_hash}"
    marker_file = f"{build_path}/.build_complete"
    os.makedirs(wheel_dir, exist_ok=True)

    # Check if wheel already exists
    wheel_pattern = f"{wheel_dir}/vllm-*+g{commit_hash[:8]}*.whl"
    existing_wheels = glob_mod.glob(wheel_pattern)

    # If force_build_dir is True, we need to check if build dir also exists
    if existing_wheels and not force_build_dir:
        wheel_path = existing_wheels[0]
        # Extract version from wheel filename
        wheel_name = os.path.basename(wheel_path)
        # vllm-0.10.1.dev296+geefbf4a68.cu124-cp38-abi3-manylinux1_x86_64.whl
        version_match = wheel_name.split("-")[1] if "-" in wheel_name else "unknown"
        print(f"[CPU BUILD] Wheel cache HIT: {wheel_name}")
        result["success"] = True
        result["cache_hit"] = True
        result["version"] = version_match
        result["wheel_path"] = wheel_path
        return result

    # If force_build_dir, check if build directory already exists (even if wheel exists)
    if force_build_dir and os.path.exists(marker_file):
        if existing_wheels:
            wheel_path = existing_wheels[0]
            wheel_name = os.path.basename(wheel_path)
            version_match = wheel_name.split("-")[1] if "-" in wheel_name else "unknown"
            print(f"[CPU BUILD] Build dir + wheel cache HIT: {wheel_name}")
            result["success"] = True
            result["cache_hit"] = True
            result["version"] = version_match
            result["wheel_path"] = wheel_path
            return result

    print(f"[CPU BUILD] Wheel cache MISS - Building vLLM {commit_hash[:8]} wheel...")
    print(f"[CPU BUILD] Using {CPU_BUILD_CONFIG['cpu']} CPUs, {CPU_BUILD_CONFIG['memory']}MB RAM")

    # Clean up any partial build
    if os.path.exists(build_path):
        shutil.rmtree(build_path)
    os.makedirs(build_path, exist_ok=True)

    try:
        # Clone vLLM repository
        print(f"[CPU BUILD] Cloning vLLM repository...")
        clone_result = subprocess.run(
            ["git", "clone", "https://github.com/vllm-project/vllm.git", build_path],
            capture_output=True, text=True, timeout=600,
        )
        if clone_result.returncode != 0:
            result["error"] = f"Git clone failed: {clone_result.stderr[:500]}"
            return result

        # Checkout the commit
        print(f"[CPU BUILD] Checking out {commit_hash[:8]}...")
        checkout_result = subprocess.run(
            ["git", "checkout", commit_hash],
            cwd=build_path, capture_output=True, text=True, timeout=60,
        )
        if checkout_result.returncode != 0:
            result["error"] = f"Git checkout failed: {checkout_result.stderr[:500]}"
            shutil.rmtree(build_path, ignore_errors=True)
            return result

        # Normalize pyproject.toml for setuptools compatibility
        pyproject_path = os.path.join(build_path, "pyproject.toml")
        if os.path.exists(pyproject_path):
            import re
            with open(pyproject_path, 'r') as f:
                content = f.read()
            original = content
            target_license = 'license = { text = "Apache-2.0" }'
            content = re.sub(r'^license\s*=\s*"[^"]*"$', target_license, content, flags=re.MULTILINE)
            content = re.sub(r'license\s*=\s*\{\s*["\']?file["\']?\s*[=:]\s*["\'][^"\']*["\']\s*\}',
                           target_license, content, flags=re.IGNORECASE)
            content = re.sub(r'^license-files\s*=\s*\[.*\]\s*$', '', content, flags=re.MULTILINE)
            if content != original:
                with open(pyproject_path, 'w') as f:
                    f.write(content)
                print(f"[CPU BUILD] Normalized pyproject.toml for setuptools compatibility")

        # Install build dependencies
        print(f"[CPU BUILD] Installing build dependencies...")
        subprocess.run(
            ["uv", "pip", "install", "--system", "-r", "requirements/build.txt"],
            cwd=build_path, capture_output=True, text=True, timeout=300,
        )
        # Also install wheel and build tools
        subprocess.run(
            ["uv", "pip", "install", "--system", "wheel", "build"],
            capture_output=True, text=True, timeout=60,
        )

        # Build wheel with parallel compilation
        # OPTIMIZED: Target only H100 architecture to reduce build time by ~60-80%
        print(f"[CPU BUILD] Building vLLM wheel (optimized for H100, ~15-25 minutes)...")
        env = os.environ.copy()
        env["MAX_JOBS"] = str(CPU_BUILD_CONFIG["cpu"])  # Use all available CPUs for parallelism
        env["VLLM_INSTALL_PUNICA_KERNELS"] = "0"  # Skip optional kernels
        # CRITICAL: Set CUDA architecture to only build for H100 (SM 9.0)
        # Must set BOTH env vars - TORCH_CUDA_ARCH_LIST for PyTorch/vLLM, CMAKE_CUDA_ARCHITECTURES for CMake
        env["TORCH_CUDA_ARCH_LIST"] = CUDA_BUILD_TARGETS
        env["CMAKE_CUDA_ARCHITECTURES"] = "90"  # CMake format: 90 = SM 9.0 (H100)
        env["VLLM_TARGET_DEVICE"] = "cuda"  # Ensure CUDA build
        # CRITICAL: Prevent CUDA runtime from detecting devices on CPU-only instance
        env["CUDA_VISIBLE_DEVICES"] = ""

        # Build wheel using pip wheel (creates .whl file)
        # NOTE: Do NOT use capture_output=True - let output stream to stdout for Modal visibility
        build_result = subprocess.run(
            ["python", "-m", "pip", "wheel", ".", "--no-deps", "--no-build-isolation", "-w", wheel_dir, "-v"],
            cwd=build_path,
            timeout=7200,  # 2 hour timeout
            env=env,
        )

        if build_result.returncode != 0:
            result["error"] = f"Wheel build failed with return code {build_result.returncode}"
            shutil.rmtree(build_path, ignore_errors=True)
            return result

        # Find the built wheel
        new_wheels = glob_mod.glob(f"{wheel_dir}/vllm-*.whl")
        if not new_wheels:
            result["error"] = "No wheel file produced"
            return result

        # Find the wheel we just built (most recent)
        wheel_path = max(new_wheels, key=os.path.getctime)
        wheel_name = os.path.basename(wheel_path)

        # Extract version from wheel filename
        version = wheel_name.split("-")[1] if "-" in wheel_name else "unknown"

        # IMPORTANT: Keep build directory for incremental compilation!
        # For agent patches with C/CUDA changes, we need the source + build artifacts
        # in the build directory for ninja to do incremental compilation.
        # The wheel is also cached in /cache/wheels/ for fast wheel-only installs.
        # DO NOT DELETE: shutil.rmtree(build_path, ignore_errors=True)

        # Create marker file to indicate successful build
        marker_file = f"{build_path}/.build_complete"
        with open(marker_file, "w") as f:
            f.write(f"wheel={wheel_name}\nversion={version}\n")

        # CRITICAL: Commit the volume to persist the wheel to Modal storage
        # Without this, the wheel is lost when the container exits!
        print(f"[CPU BUILD] Committing wheel to Modal volume...")
        build_cache.commit()

        print(f"[CPU BUILD] SUCCESS: Built wheel {wheel_name}")
        result["success"] = True
        result["cache_hit"] = False
        result["version"] = version
        result["wheel_path"] = wheel_path
        return result

    except subprocess.TimeoutExpired:
        result["error"] = "Build timed out (exceeded 2 hours)"
        shutil.rmtree(build_path, ignore_errors=True)
        return result
    except Exception as e:
        result["error"] = f"Build error: {str(e)}"
        shutil.rmtree(build_path, ignore_errors=True)
        return result


@app.function(
    image=base_image,
    cpu=CPU_BUILD_CONFIG["cpu"],
    memory=CPU_BUILD_CONFIG["memory"],
    timeout=CPU_BUILD_CONFIG["timeout"],
    volumes={"/cache": build_cache},
)
def build_agent_patch_cpu(base_commit: str, patch_content: str) -> Dict[str, Any]:
    """
    Apply agent patch and build wheel on CPU-only instance (no GPU needed).

    IN-PLACE BUILD APPROACH (v21):
    Instead of copying the build directory (which breaks CMake caches), we:
    1. Acquire a file lock on the base build directory
    2. Apply the patch directly to the base build
    3. Run incremental build (Ninja recompiles ONLY changed files - very fast!)
    4. Save the wheel
    5. Revert the patch with git checkout
    6. Release the lock

    This avoids all CMake cache path issues and enables TRUE incremental builds.

    Args:
        base_commit: The base vLLM commit hash (must have been built previously)
        patch_content: Unified diff patch from agent

    Returns:
        Dict with keys:
        - success: bool
        - wheel_path: str (path to patched wheel in volume)
        - patch_hash: str (short hash identifying this patch)
        - error: str (error message if failed)
    """
    import os
    import subprocess
    import shutil
    import hashlib
    import glob as glob_mod
    import fcntl
    import time

    # Create a short hash of the patch content for unique identification
    patch_hash = hashlib.sha256(patch_content.encode()).hexdigest()[:12]

    result = {
        "success": False,
        "wheel_path": None,
        "patch_hash": patch_hash,
        "base_commit": base_commit,
        "error": None,
    }

    cache_dir = "/cache"
    wheel_dir = f"{cache_dir}/wheels"
    build_path = f"{cache_dir}/build_{base_commit}"
    lock_file = f"{cache_dir}/build_{base_commit}.lock"
    os.makedirs(wheel_dir, exist_ok=True)

    # Check if patched wheel already exists (before acquiring lock)
    patched_wheel_pattern = f"{wheel_dir}/vllm-*+g{base_commit[:8]}_patch{patch_hash}*.whl"
    existing_wheels = glob_mod.glob(patched_wheel_pattern)
    if existing_wheels:
        wheel_path = existing_wheels[0]
        print(f"[AGENT BUILD] Patched wheel cache HIT: {os.path.basename(wheel_path)}")
        result["success"] = True
        result["wheel_path"] = wheel_path
        return result

    print(f"[AGENT BUILD] Building patched wheel for {base_commit[:8]} + patch {patch_hash}...")

    # Step 1: Ensure base build exists
    marker_file = f"{build_path}/.build_complete"
    if not os.path.exists(marker_file):
        print(f"[AGENT BUILD] Base build not found at {build_path}")
        result["error"] = f"Base build for {base_commit[:8]} not found. Call build_vllm_cpu_only first."
        return result

    # Step 2: Acquire exclusive lock on the build directory
    # This prevents concurrent builds from corrupting each other
    print(f"[AGENT BUILD] Acquiring lock on build directory...")
    lock_fd = None
    try:
        lock_fd = open(lock_file, 'w')
        fcntl.flock(lock_fd.fileno(), fcntl.LOCK_EX)  # Blocking exclusive lock
        print(f"[AGENT BUILD] Lock acquired")
    except Exception as e:
        result["error"] = f"Failed to acquire lock: {e}"
        if lock_fd:
            lock_fd.close()
        return result

    try:
        # Step 3: Apply the agent patch IN-PLACE (no copy!)
        print(f"[AGENT BUILD] Applying agent patch in-place...")
        patch_file = f"{build_path}/agent.patch"
        with open(patch_file, 'w') as f:
            f.write(patch_content)

        # Try git apply first
        apply_result = subprocess.run(
            ["git", "apply", "--verbose", "agent.patch"],
            cwd=build_path,
            capture_output=True,
            text=True,
            timeout=60,
        )
        if apply_result.returncode != 0:
            # Fallback to patch command
            apply_result = subprocess.run(
                ["patch", "-p1", "-i", "agent.patch"],
                cwd=build_path,
                capture_output=True,
                text=True,
                timeout=60,
            )
            if apply_result.returncode != 0:
                result["error"] = f"Patch apply failed: {apply_result.stderr}\nstdout: {apply_result.stdout}"
                # Clean up patch file
                os.remove(patch_file) if os.path.exists(patch_file) else None
                return result

        print(f"[AGENT BUILD] Patch applied successfully")

        # Normalize pyproject.toml for setuptools compatibility
        pyproject_path = os.path.join(build_path, "pyproject.toml")
        pyproject_backup = None
        if os.path.exists(pyproject_path):
            import re
            with open(pyproject_path, 'r') as f:
                pyproject_backup = f.read()  # Save original for revert
            content = pyproject_backup
            target_license = 'license = { text = "Apache-2.0" }'
            content = re.sub(r'^license\s*=\s*"[^"]*"$', target_license, content, flags=re.MULTILINE)
            content = re.sub(r'license\s*=\s*\{\s*["\']?file["\']?\s*[=:]\s*["\'][^"\']*["\']\s*\}',
                           target_license, content, flags=re.IGNORECASE)
            content = re.sub(r'^license-files\s*=\s*\[.*\]\s*$', '', content, flags=re.MULTILINE)
            if content != pyproject_backup:
                with open(pyproject_path, 'w') as f:
                    f.write(content)

        # Step 4: Run TRUE incremental build
        # Since we're in the SAME directory, CMake caches are valid!
        # Ninja will only recompile files that the patch actually changed
        print(f"[AGENT BUILD] Running TRUE incremental build (only patched files recompile)...")

        env = os.environ.copy()
        env["MAX_JOBS"] = str(CPU_BUILD_CONFIG["cpu"])
        env["VLLM_INSTALL_PUNICA_KERNELS"] = "0"
        env["TORCH_CUDA_ARCH_LIST"] = CUDA_BUILD_TARGETS
        env["CMAKE_CUDA_ARCHITECTURES"] = "90"
        env["VLLM_TARGET_DEVICE"] = "cuda"
        env["CUDA_VISIBLE_DEVICES"] = ""  # No GPU on CPU instance

        # Build wheel
        tmp_wheel_dir = "/tmp/patch_wheels"
        os.makedirs(tmp_wheel_dir, exist_ok=True)

        build_start = time.time()
        build_result = subprocess.run(
            ["python", "-m", "pip", "wheel", ".", "--no-deps", "--no-build-isolation", "-w", tmp_wheel_dir, "-v"],
            cwd=build_path,
            timeout=5400,  # 90 min timeout
            env=env,
        )
        build_time = time.time() - build_start
        print(f"[AGENT BUILD] Build completed in {build_time:.1f}s")

        if build_result.returncode != 0:
            result["error"] = f"Incremental wheel build failed with return code {build_result.returncode}"
            # Still need to revert before returning
        else:
            # Step 5: Find and save the wheel
            built_wheels = glob_mod.glob(f"{tmp_wheel_dir}/vllm-*.whl")
            if built_wheels:
                src_wheel = built_wheels[0]
                wheel_name = os.path.basename(src_wheel)
                # Add patch hash to wheel name for identification
                base_name = wheel_name.replace(".whl", "")
                patched_wheel_name = f"{base_name}_patch{patch_hash}.whl"
                dst_wheel = f"{wheel_dir}/{patched_wheel_name}"
                shutil.copy2(src_wheel, dst_wheel)
                print(f"[AGENT BUILD] Saved patched wheel: {patched_wheel_name}")
                result["success"] = True
                result["wheel_path"] = dst_wheel
            else:
                result["error"] = "No wheel file produced by build"

    finally:
        # Step 6: ALWAYS revert the patch to restore clean state
        print(f"[AGENT BUILD] Reverting patch to restore clean build state...")
        try:
            # Remove the patch file
            patch_file = f"{build_path}/agent.patch"
            if os.path.exists(patch_file):
                os.remove(patch_file)

            # Revert all changes with git checkout
            revert_result = subprocess.run(
                ["git", "checkout", "."],
                cwd=build_path,
                capture_output=True,
                text=True,
                timeout=60,
            )
            if revert_result.returncode != 0:
                print(f"[AGENT BUILD] WARNING: git checkout failed: {revert_result.stderr}")
                # Try harder - git reset
                subprocess.run(["git", "reset", "--hard", "HEAD"], cwd=build_path, timeout=60)
            else:
                print(f"[AGENT BUILD] Clean state restored")
        except Exception as e:
            print(f"[AGENT BUILD] WARNING: Failed to revert patch: {e}")

        # Step 7: Release the lock
        if lock_fd:
            fcntl.flock(lock_fd.fileno(), fcntl.LOCK_UN)
            lock_fd.close()
            print(f"[AGENT BUILD] Lock released")

    return result


def ensure_agent_patch_built(base_commit: str, patch_content: str) -> Dict[str, Any]:
    """
    Ensure agent patch is built on CPU instance before GPU allocation.

    This is the external entry point for building agent patches.
    Should be called from run_3way_modal_benchmark() before allocating GPU.

    Args:
        base_commit: Base vLLM commit hash
        patch_content: Unified diff patch from agent

    Returns:
        Dict with success status and wheel_path or error
    """
    import hashlib

    patch_hash = hashlib.sha256(patch_content.encode()).hexdigest()[:12]
    print(f"[ENSURE AGENT BUILD] Checking/building agent patch {patch_hash} on base {base_commit[:8]}...")

    try:
        # First ensure base commit is built
        base_result = ensure_vllm_build_cached(base_commit)
        if not base_result.get("success"):
            return {
                "success": False,
                "error": f"Base build failed: {base_result.get('error', 'unknown')}",
                "patch_hash": patch_hash,
            }

        # Now build the patched version
        fn = modal.Function.from_name("omniperf-benchmark", "build_agent_patch_cpu")
        with modal.enable_output():
            result = fn.remote(base_commit, patch_content)

        if result["success"]:
            print(f"[ENSURE AGENT BUILD] Patched wheel ready: {result.get('wheel_path', 'unknown')}")
        else:
            print(f"[ENSURE AGENT BUILD] FAILED: {result.get('error', 'unknown')}")

        return result

    except Exception as e:
        tb = traceback.format_exc()
        print(f"[ENSURE AGENT BUILD] Exception: {str(e)}")
        print(f"[ENSURE AGENT BUILD] Traceback:\n{tb}")
        return {
            "success": False,
            "error": str(e),
            "traceback": tb,
            "patch_hash": patch_hash,
        }


# Pre-built Docker images from ayushnangia16/nvidia-vllm-docker
# These images have vLLM pre-installed at the specific commit, eliminating build time
DOCKER_IMAGE_REPO = "ayushnangia16/nvidia-vllm-docker"
PREBUILT_COMMITS = {
    "015069b01741e9ecb9e604c7fe87fbdfc306ebe5",
    "0d243f2a54fbd1c56da8a571f0899c30b6aba5d9",
    "0ec82edda59aaf5cf3b07aadf4ecce1aa1131add",
    "19d98e0c7db96713f0e2201649159431177a56e2",
    "21d93c140d0a97af5f0c59e660cf04bd417fd424",
    "22d33baca2c0c639cfd45c48e99803e56c3efa74",
    "22dd9c2730dc1124b9d0ac15fff223d0b8d9020b",
    "25ebed2f8ca6d747d63f2be9ede023c561851ac8",
    "296f927f2493908984707354e3cc5d7b2e41650b",
    "299ebb62b269ce167eb1c71b5e39a1dc1f65ce1c",
    "2deb029d115dadd012ce5ea70487a207cb025493",
    "30172b4947c52890b808c6da3a6c7580f55cbb74",
    "3092375e274e9e003961e600e10a6192d33ceaa0",
    "310aca88c984983189a57f1b72e3b1dde89fb92f",
    "3127e975fb9417d10513e25b80820870f594c627",
    "319ad7f1d386699e94f629341c9988a926821f24",
    "3476ed0809ec91a3457da0cb90543133a4f4b519",
    "35fad35a485eac9195c510731ba4a9d297dfd963",
    "379da6dcb5f5d062d0452b2fc23291e5113dcf04",
    "3b61cb450d899dc423feb264c297d4d18d701678",
    "4c822298981a8f7521492075ff72659985fc4c3f",
    "4fb56914c5f27ef062e10d44a0f79c6ceab382f9",
    "526de822d501c792b051c864ba873a836d78d5bf",
    "58eee5f2e05b74eb2cb1a3bbda9c04df4805e4cc",
    "5e5c8e091eacc16672a0a8265eb5cb0ece85d24b",
    "61b8cea3b42feab021d506e9143551de18f9165c",
    "660470e5a36b8e52083615ad7c85e9b4fd4c72ce",
    "67da5720d4ed2aa1f615ec812031f4f3753b3f62",
    "6a417b8600d4d1e57698a91b71a38446e8fc5c45",
    "6ce01f30667bbae33f112152e07a3b66b841078f",
    "6d0734c562e759fdb7076d762222b3881e62ab1f",
    "6d646d08a2e0e73e83e313a5ae470c1f9e4f200e",
    "6dd94dbe94c1820a1e224cba65efcf0befa97995",
    "6e36f4fa6ce64619b9ea94c88a157f5783a63a65",
    "70b808fe1a63322bc6bf5f46a91981a8f6b8af00",
    "7661e92ef85e552936195ae4b803e292b9a96776",
    "7c01f706418d593b3cf23d2ec9110dca7151c539",
    "80aa7e91fcd547a7a1396f71b9bdce18e5c92245",
    "81ede99ca44a5b3518932a07ea4a76a719e7416e",
    "83450458339b07765b0e72a822e5fe93eeaf5258",
    "886936837ca89e5645bc1f71cc0e1492b65b1590",
    "89a84b0bb7b30706a02836234a94493ea8f780bf",
    "8a4e5c5f3c1d39e924e48a87c9cc6cf382aa3532",
    "8aa1485fcff7be3e42300c0615ee0f3f3cbce9a8",
    "8bc68e198c4c90ddc2e54fa76eb81c2c714bb1cd",
    "8c1e77fb585c4f42783a3d88c1efc7c9e15fd89f",
    "8d75fe48ca5f46b7af0f5201d8500b9604eed769",
    "9323a3153b20d4a2ca7ac04a2784609d6ce656e0",
    "93e5f3c5fb4a4bbd49610efb96aad30df95fca66",
    "98f47f2a4032f8c395268de80858c64ffcfc60fa",
    "99abb8b650c66664cdc84d815b7f306f33bd9881",
    "9a3b88328f7e434cac35b90ee463de6689f9a833",
    "9badee53decb3d432dc805336abfb0eb81dfb48f",
    "9d72daf4ced05a5fec1ad8ea2914a39296f402da",
    "9ed82e7074a18e25680ab106fc846364ad97bc00",
    "9f1710f1ace3535920c0bb6d4cc329c36289080e",
    "a0dce9383ab7de0015060fb9fedadeb7d8ffdfb9",
    "a32237665df876fcb51196dc209e8aff9fd89d29",
    "a377f0bd5e1fa0ca069e3dbf28f4de5af64d0bb1",
    "ab7165f2c7ea358df969d68a0fb0ce9bb184a083",
    "ac45c44d98e77f30e47b8fb69134f4635183070d",
    "ad932a221d2a4c1e6355021bb9e9c47f7a179e51",
    "aea94362c9bdd08ed2b346701bdc09d278e85f66",
    "b10e51989551cd80dd74079429ccf91f0807bd92",
    "b2e0ad3b598ed0e022cdbd678a20821d411873c2",
    "b55ed6ef8ab0dce7fb0f79ff292dafdb4d22610c",
    "b690e34824fd5a5c4054a0c0468ebfb6aa1dd215",
    "b6d103542c654fb63013a1e45a586d654ae36a2a",
    "b9986454fe8ba80e2a109d069397b6b59aae658b",
    "baeded25699f9f4851843306f27f685c4d4ee7c5",
    "bc7c4d206bbfb56b06d218b6c2971e8ca191db36",
    "bd6028d6b0bbc0c569ece0535067081c5e8bdc14",
    "bd852f2a8b9e9129de69fa7349906a9115538d5a",
    "bfdb1ba5c3fb14387c69acb1f5067102d8028e56",
    "c0569dbc82b5e945a77878190114d1b68027828b",
    "ca7a2d5f28eac9621474563cdda0e08596222755",
    "ccf02fcbaebb1a5b59dfc6c7cb64aa7cc489f04c",
    "ce6bf3a2cff4860c5661cac2280e0a28bedb6440",
    "cf2f084d56a1293cb08da2393984cdc7685ac019",
    "d55e446d1320d0f5f22bc3584f81f18d7924f166",
    "d7740ea4dcee4ab75d7d6eef723f33cae957b288",
    "dae68969774e41b93b01cd31171ca033a92b574a",
    "dcc6cfb991cd76369aad96e04424f29c8fecdbd8",
    "e02ac5561748306186aaeaad6dad4c89484a2b45",
    "e206b5433109d298e53451015465b2bf8f03ef0a",
    "e3580537a41a46b0f3cd750b86b633c1857a8c90",
    "e493e48524e9e78ab33eafec6461b3940e361189",
    "e7523c2e031bc96740723ab63833d1cf94229ab4",
    "e7b204268132cb775c139574c1ff4ad7e15c8f66",
    "eb6d3c264d0cd8e44dec16bca7947fbe96415ce9",
    "ed25054577f7abca2aee32a5290200c4a1aed561",
    "eefbf4a68b7b0a5b8364a59647906be1b7f043e2",
    "f092153fbe349a9a1742940e3703bfcff6aa0a6d",
    "f26c4aeecba481ce1445be7a998b0b97460a13bb",
    "fa63e710c7fbaae3a445f669d3b5ba6b9a4ef412",
    "fb0acb6c72874e98617cabee4ff4851569374fc9",
    "fc542144c4477ffec1d3de6fa43e54f8fb5351e8",
    "fc7b8d1eefcbe837a56b7c080509417fe5167e6c",
    "fd4ea8ef5c17a8b991107402a414f6ed355d854d",
    "fe66b34728e5d383e3d19aefc544eeee808c99fb",
}


def has_prebuilt_image(commit_hash: str) -> bool:
    """Check if a pre-built Docker image exists for a commit."""
    # Normalize to full 40-char hash or check prefix match
    for prebuilt in PREBUILT_COMMITS:
        if prebuilt.startswith(commit_hash) or commit_hash.startswith(prebuilt):
            return True
    return commit_hash in PREBUILT_COMMITS


def get_prebuilt_commit(commit_hash: str) -> Optional[str]:
    """Get the full pre-built commit hash matching the given prefix."""
    if commit_hash in PREBUILT_COMMITS:
        return commit_hash
    for prebuilt in PREBUILT_COMMITS:
        if prebuilt.startswith(commit_hash) or commit_hash.startswith(prebuilt):
            return prebuilt
    return None


def get_prebuilt_image(commit_hash: str) -> Optional[modal.Image]:
    """Get a Modal image for a pre-built vLLM commit.

    Returns None if no pre-built image exists for this commit.
    The returned image has vLLM already installed, eliminating build time.

    NOTE: The pre-built Docker images have python3 but not python symlink.
    We use setup_dockerfile_commands to create the symlink BEFORE Modal's setup runs.
    """
    full_commit = get_prebuilt_commit(commit_hash)
    if not full_commit:
        return None

    # Use setup_dockerfile_commands to run BEFORE Modal's pip setup
    # This creates the python symlink that Modal's builder needs
    # CRITICAL: Clear ENTRYPOINT and CMD to prevent Modal from injecting
    # keepalive commands (sleep 172800) that get passed as arguments to scripts
    return (
        modal.Image.from_registry(
            f"{DOCKER_IMAGE_REPO}:{full_commit}",
            setup_dockerfile_commands=[
                # Clear ENTRYPOINT and CMD from base image to allow Modal control
                "ENTRYPOINT []",
                "CMD []",
                # Create python symlink BEFORE Modal tries to use it
                "RUN ln -sf /usr/bin/python3 /usr/bin/python",
                # CRITICAL FIX: vLLM is installed at /workspace/vllm in Docker images
                # but /workspace is NOT in PYTHONPATH by default, so imports fail
                "ENV PYTHONPATH=/workspace:/workspace/vllm:$PYTHONPATH",
            ]
        )
        .run_commands([
            # Clone vLLM repo for benchmark scripts (after Modal setup)
            "git clone --depth 1 https://github.com/vllm-project/vllm.git /opt/vllm-benchmarks || true",
        ])
        .env({
            "MODAL_CACHE_BUST": "20260102_v21_inplace_build_with_lock",
            "HF_HOME": "/root/.cache/huggingface",
            "TRANSFORMERS_CACHE": "/root/.cache/huggingface",
            # Also set via .env() for runtime (belt and suspenders)
            "PYTHONPATH": "/workspace:/workspace/vllm",
        })
    )


# Model size to GPU mapping
LARGE_MODEL_GPU_MAP = {
    "deepseek-ai/DeepSeek-V3": "H100:8",
    "deepseek-ai/DeepSeek-V3-0324": "H100:8",
    "deepseek-ai/DeepSeek-V2": "H100:8",
    "deepseek-ai/DeepSeek-V2-Lite": "H100:1",
    "nvidia/Nemotron-4-340B": "H100:8",
    "meta-llama/Llama-4-Scout-17B-16E": "H100:2",
    "meta-llama/Meta-Llama-3-70B": "H100:4",
    "meta-llama/Llama-3.1-70B": "H100:4",
    "meta-llama/Llama-3-70B": "H100:4",
    "70B": "H100:4",  # Catch-all for 70B models
    "70b": "H100:4",
}

# Models that are too large or unstable to benchmark reliably
# These will be skipped entirely to avoid wasting compute on container crashes/OOM
BLOCKED_MODELS = {
    # DeepSeek V3/R1 - requires 8xH100 with special configuration, often OOM
    "deepseek-ai/DeepSeek-V3",
    "deepseek-ai/DeepSeek-V3-0324",
    "deepseek-ai/DeepSeek-R1",
    "deepseek-ai/DeepSeek-R1-Distill-Llama-70B",
    # Nemotron 340B - too large even for 8xH100
    "nvidia/Nemotron-4-340B",
    "nvidia/Nemotron-4-340B-Instruct",
    # Extremely large MoE models
    "databricks/dbrx",
    "databricks/dbrx-instruct",
    # Llama 4 Scout - specialized MoE architecture, needs special handling
    "meta-llama/Llama-4-Scout-17B-16E-Instruct",
}


def is_model_blocked(model: str) -> bool:
    """Check if a model is blocked (too large/unstable to benchmark)."""
    model_lower = model.lower()
    for blocked in BLOCKED_MODELS:
        if blocked.lower() in model_lower or model_lower in blocked.lower():
            return True
    # Also block any model explicitly mentioning DeepSeek-V3 or R1
    if any(x in model_lower for x in ["deepseek-v3", "deepseek-r1", "nemotron-340b"]):
        return True
    return False


def get_vllm_bench_cli() -> Optional[str]:
    """Get the working vllm bench CLI command prefix.

    Returns:
        CLI prefix string (e.g., "vllm bench" or "python -m vllm.entrypoints.cli.main bench")
        or None if not available.
    """
    def is_help_output(stdout: str, stderr: str) -> bool:
        """Check if output looks like valid help output."""
        combined = (stdout + stderr).lower()
        # Look for common help indicators
        return any(indicator in combined for indicator in [
            "usage:", "usage ", "--help", "--model", "benchmark",
            "latency", "options:", "arguments:", "positional arguments"
        ])

    # Method 1: Try direct CLI (preferred)
    try:
        result = subprocess.run(
            ["vllm", "bench", "latency", "--help"],
            capture_output=True,
            text=True,
            timeout=30,
        )
        if result.returncode == 0 and is_help_output(result.stdout, result.stderr):
            print("[DEBUG] vllm bench CLI available (direct)")
            return "vllm bench"
        else:
            print(f"[DEBUG] vllm bench direct: rc={result.returncode}, out[:100]={result.stdout[:100]}")
    except Exception as e:
        print(f"[DEBUG] vllm bench CLI not in PATH: {e}")

    # Method 2: Try via Python module with .main (newer vLLM >= 0.6.x)
    try:
        result = subprocess.run(
            ["python", "-m", "vllm.entrypoints.cli.main", "bench", "latency", "--help"],
            capture_output=True,
            text=True,
            timeout=30,
        )
        if result.returncode == 0 and is_help_output(result.stdout, result.stderr):
            print("[DEBUG] vllm bench available via python -m vllm.entrypoints.cli.main")
            return "python -m vllm.entrypoints.cli.main bench"
        else:
            print(f"[DEBUG] cli.main bench: rc={result.returncode}, err[:100]={result.stderr[:100]}")
    except Exception as e:
        print(f"[DEBUG] vllm bench not available via python -m .main: {e}")

    # Method 3: Try via Python module without .main (older vLLM)
    try:
        result = subprocess.run(
            ["python", "-m", "vllm.entrypoints.cli", "bench", "latency", "--help"],
            capture_output=True,
            text=True,
            timeout=30,
        )
        if result.returncode == 0 and is_help_output(result.stdout, result.stderr):
            print("[DEBUG] vllm bench available via python -m vllm.entrypoints.cli")
            return "python -m vllm.entrypoints.cli bench"
        else:
            print(f"[DEBUG] cli bench: rc={result.returncode}")
    except Exception as e:
        print(f"[DEBUG] vllm bench not available via python -m: {e}")

    # Method 4: Check if vllm CLI exists at all (even if bench subcommand fails)
    # Some vLLM versions have the CLI but different command structure
    try:
        result = subprocess.run(
            ["vllm", "--help"],
            capture_output=True,
            text=True,
            timeout=30,
        )
        if result.returncode == 0 and "bench" in result.stdout.lower():
            # CLI exists and has bench command, try running it directly
            print("[DEBUG] vllm CLI exists with bench command, assuming direct CLI works")
            return "vllm bench"
    except Exception:
        pass

    print("[DEBUG] vllm bench CLI NOT available - will use benchmark scripts")
    return None


def check_vllm_bench_available() -> bool:
    """Check if 'vllm bench' CLI is available (vLLM >= 0.7)."""
    return get_vllm_bench_cli() is not None


def checkout_vllm_for_benchmarks(commit: str, checkout_dir: str = "/opt/vllm-commit") -> bool:
    """Checkout vLLM at a specific commit for benchmark scripts.

    The benchmark scripts (benchmark_serving.py, etc.) are NOT installed with the wheel.
    They're standalone scripts in the repo. For older vLLM versions without 'vllm bench' CLI,
    we need to checkout the same commit to get compatible benchmark scripts.

    Args:
        commit: The commit hash to checkout
        checkout_dir: Directory to clone/checkout to

    Returns:
        True if successful, False otherwise
    """
    import shutil

    try:
        # Check if we already have the right commit
        if os.path.exists(f"{checkout_dir}/.git"):
            result = subprocess.run(
                ["git", "rev-parse", "HEAD"],
                cwd=checkout_dir,
                capture_output=True,
                text=True,
                timeout=10,
            )
            if result.returncode == 0 and result.stdout.strip().startswith(commit[:12]):
                print(f"[CHECKOUT] Already at commit {commit[:8]}")
                return True

        # Clean and clone fresh
        if os.path.exists(checkout_dir):
            shutil.rmtree(checkout_dir)

        print(f"[CHECKOUT] Cloning vLLM at commit {commit[:8]} for benchmark scripts...")

        # Clone with --filter to minimize download, then checkout specific commit
        result = subprocess.run(
            ["git", "clone", "--filter=blob:none", "https://github.com/vllm-project/vllm.git", checkout_dir],
            capture_output=True,
            text=True,
            timeout=300,
        )
        if result.returncode != 0:
            print(f"[CHECKOUT] Clone failed: {result.stderr}")
            return False

        # Fetch and checkout the specific commit
        subprocess.run(
            ["git", "fetch", "origin", commit],
            cwd=checkout_dir,
            capture_output=True,
            timeout=120,
        )

        result = subprocess.run(
            ["git", "checkout", commit],
            cwd=checkout_dir,
            capture_output=True,
            text=True,
            timeout=60,
        )
        if result.returncode != 0:
            print(f"[CHECKOUT] Checkout failed: {result.stderr}")
            return False

        print(f"[CHECKOUT] Successfully checked out vLLM at {commit[:8]}")
        return True

    except Exception as e:
        print(f"[CHECKOUT] Error: {e}")
        return False


def translate_benchmark_command(perf_command: str, commit: Optional[str] = None) -> str:
    """Translate benchmark command to work with installed vLLM version.

    Strategy:
    1. Check if 'vllm bench' CLI is available (vLLM >= 0.7)
    2. If yes, translate to 'vllm bench' commands (using correct CLI prefix)
    3. If no, use benchmark scripts from /opt/vllm-commit/ (commit-specific checkout)

    Args:
        perf_command: The original benchmark command
        commit: Optional commit hash to checkout for benchmark scripts (for older vLLM)

    This ensures we use benchmark scripts compatible with the installed vLLM version.
    """
    cmd = perf_command

    # Get the working CLI prefix (e.g., "vllm bench" or "python -m vllm.entrypoints.cli bench")
    cli_prefix = get_vllm_bench_cli()

    if cli_prefix:
        # Translate to vllm bench CLI for newer vLLM versions
        if "benchmark_serving.py" in cmd:
            cmd = re.sub(
                r'python3?\s+(?:\./)?(?:benchmarks/)?benchmark_serving\.py',
                f'{cli_prefix} serve',
                cmd
            )
            print(f"[TRANSLATE] benchmark_serving.py -> {cli_prefix} serve")

        if "benchmark_throughput.py" in cmd:
            cmd = re.sub(
                r'python3?\s+(?:\./)?(?:benchmarks/)?benchmark_throughput\.py',
                f'{cli_prefix} throughput',
                cmd
            )
            print(f"[TRANSLATE] benchmark_throughput.py -> {cli_prefix} throughput")

        if "benchmark_latency.py" in cmd:
            cmd = re.sub(
                r'python3?\s+(?:\./)?(?:benchmarks/)?benchmark_latency\.py',
                f'{cli_prefix} latency',
                cmd
            )
            print(f"[TRANSLATE] benchmark_latency.py -> {cli_prefix} latency")
    else:
        # For older vLLM versions, use commit-specific benchmark scripts
        # Checkout vLLM at the specific commit if provided
        benchmark_base = "/opt/vllm-commit/benchmarks"
        if commit:
            checkout_vllm_for_benchmarks(commit)
        else:
            # Fall back to /opt/vllm-benchmarks (latest) if no commit specified
            benchmark_base = "/opt/vllm-benchmarks/benchmarks"
            print(f"[TRANSLATE] Warning: No commit specified, using latest benchmark scripts")

        if "benchmark_serving.py" in cmd:
            cmd = re.sub(
                r'python3?\s+(?:\./)?(?:benchmarks/)?benchmark_serving\.py',
                f'python {benchmark_base}/benchmark_serving.py',
                cmd
            )
            print(f"[TRANSLATE] benchmark_serving.py -> {benchmark_base}/benchmark_serving.py")

        if "benchmark_throughput.py" in cmd:
            cmd = re.sub(
                r'python3?\s+(?:\./)?(?:benchmarks/)?benchmark_throughput\.py',
                f'python {benchmark_base}/benchmark_throughput.py',
                cmd
            )
            print(f"[TRANSLATE] benchmark_throughput.py -> {benchmark_base}/benchmark_throughput.py")

        if "benchmark_latency.py" in cmd:
            cmd = re.sub(
                r'python3?\s+(?:\./)?(?:benchmarks/)?benchmark_latency\.py',
                f'python {benchmark_base}/benchmark_latency.py',
                cmd
            )
            print(f"[TRANSLATE] benchmark_latency.py -> {benchmark_base}/benchmark_latency.py")

    return cmd


def is_serving_benchmark(perf_command: str) -> bool:
    """Determine if benchmark requires a running server or runs standalone.

    Serving benchmarks (need server): benchmark_serving.py
    Standalone benchmarks (no server): benchmark_throughput.py, benchmark_latency.py
    """
    standalone_patterns = [
        "benchmark_throughput",
        "benchmark_latency",
    ]

    serving_patterns = [
        "benchmark_serving",
    ]

    cmd_lower = perf_command.lower()

    # Check standalone patterns first (more specific)
    for pattern in standalone_patterns:
        if pattern in cmd_lower:
            return False

    # Check serving patterns
    for pattern in serving_patterns:
        if pattern in cmd_lower:
            return True

    # Default to serving if unclear
    return True


def get_gpu_config(model: str, perf_command: str) -> str:
    """Determine GPU requirements based on model and command."""
    # Check for explicit tensor parallelism in command
    tp_match = re.search(r'(?:-tp|--tensor-parallel-size)\s+(\d+)', perf_command)
    if tp_match:
        tp_size = int(tp_match.group(1))
        if tp_size >= 8:
            return "H100:8"
        elif tp_size >= 4:
            return "H100:4"
        elif tp_size >= 2:
            return "H100:2"

    # Check known large models
    for pattern, config in LARGE_MODEL_GPU_MAP.items():
        if pattern.lower() in model.lower():
            return config

    return "H100:1"


def check_vllm_preinstalled() -> Tuple[bool, Optional[str]]:
    """Check if vLLM is already installed in the container (from pre-built Docker image).

    Returns:
        (is_installed, version) tuple
    """
    try:
        result = subprocess.run(
            ["python", "-c", "import vllm; print(vllm.__version__)"],
            capture_output=True,
            text=True,
            timeout=30,
            cwd="/tmp",  # Avoid local vllm directory
        )
        if result.returncode == 0:
            version = result.stdout.strip()
            return True, version
        return False, None
    except Exception:
        return False, None


def vllm_version_supports_serving(version: str) -> bool:
    """Check if vLLM version supports serving benchmarks without port binding issues.

    vLLM < 0.7.0 has a bug where uvicorn fails to bind ports correctly in multiprocess mode.
    The fix (pre-created socket) was added in commit 578087e (Feb 2025) for vLLM 0.7.x.

    Args:
        version: vLLM version string (e.g., "0.6.3.post2.dev398", "0.7.1.dev57")

    Returns:
        True if serving benchmarks are supported, False otherwise
    """
    import re
    # Extract major.minor version
    match = re.match(r'(\d+)\.(\d+)', version)
    if not match:
        print(f"[VERSION] Could not parse vLLM version: {version}, assuming serving NOT supported")
        return False

    major = int(match.group(1))
    minor = int(match.group(2))

    # vLLM >= 0.7.0 supports serving benchmarks
    if major > 0 or (major == 0 and minor >= 7):
        print(f"[VERSION] vLLM {version} (>= 0.7.0) supports serving benchmarks")
        return True
    else:
        print(f"[VERSION] vLLM {version} (< 0.7.0) has port binding bug - use standalone benchmark only")
        return False


def install_wheel(wheel_url: str) -> Tuple[bool, str]:
    """Install a vLLM wheel from URL."""
    import os
    try:
        # Set env to skip wheel filename version check (vLLM wheels have mismatched versions)
        env = os.environ.copy()
        env["UV_SKIP_WHEEL_FILENAME_CHECK"] = "1"

        # Debug: Print env var status
        print(f"[DEBUG] UV_SKIP_WHEEL_FILENAME_CHECK in env: {env.get('UV_SKIP_WHEEL_FILENAME_CHECK')}")
        print(f"[DEBUG] UV_SKIP_WHEEL_FILENAME_CHECK in os.environ: {os.environ.get('UV_SKIP_WHEEL_FILENAME_CHECK')}")

        # Uninstall existing vLLM
        subprocess.run(
            ["uv", "pip", "uninstall", "--system", "vllm"],
            capture_output=True,
            timeout=60,
            env=env,
        )

        # Install new wheel
        # --no-cache forces re-evaluation; UV_SKIP_WHEEL_FILENAME_CHECK skips version check
        print(f"[DEBUG] Running: uv pip install --system --no-cache {wheel_url}")
        result = subprocess.run(
            ["uv", "pip", "install", "--system", "--no-cache", wheel_url],
            capture_output=True,
            text=True,
            timeout=600,
            env=env,
        )

        print(f"[DEBUG] uv install return code: {result.returncode}")
        if result.stdout:
            print(f"[DEBUG] uv stdout: {result.stdout[:500]}")
        if result.stderr:
            print(f"[DEBUG] uv stderr: {result.stderr[:500]}")

        if result.returncode != 0:
            return False, f"Failed to install wheel: {result.stderr}"

        # Verify installation
        version_result = subprocess.run(
            ["python", "-c", "import vllm; print(vllm.__version__)"],
            capture_output=True,
            text=True,
        )

        return True, version_result.stdout.strip()
    except Exception as e:
        return False, str(e)


# Wheel URL pattern for vLLM S3 bucket
VLLM_WHEEL_URL_PATTERN = "https://vllm-wheels.s3.us-west-2.amazonaws.com/{commit}/vllm-1.0.0.dev-cp38-abi3-manylinux1_x86_64.whl"


def extract_commit_from_wheel_url(wheel_url: str) -> Optional[str]:
    """Extract commit hash from wheel URL.

    Example:
        https://vllm-wheels.s3.us-west-2.amazonaws.com/abc123def/vllm-1.0.0.dev-...
        -> abc123def
    """
    import re
    # Match commit hash between slashes: /commit_hash/
    match = re.search(r'/([a-f0-9]{7,40})/', wheel_url)
    if match:
        return match.group(1)
    return None


def check_wheel_url_exists(commit: str) -> bool:
    """Check if a wheel exists for a given commit hash."""
    import urllib.request
    wheel_url = VLLM_WHEEL_URL_PATTERN.format(commit=commit)
    try:
        req = urllib.request.Request(wheel_url, method='HEAD')
        with urllib.request.urlopen(req, timeout=10) as response:
            return response.status == 200
    except:
        return False


def find_nearest_wheel_commit(target_commit: str, repo_path: str = "/tmp/vllm-checkout") -> Optional[str]:
    """
    Find the nearest ancestor commit that has an available wheel.

    Walks back through git history to find a commit with a pre-built wheel.
    Returns the commit hash or None if no wheel found within 50 ancestors.
    """
    import shutil

    # Clone repo if needed
    if not os.path.exists(repo_path):
        print(f"Cloning vLLM repo for commit traversal...")
        result = subprocess.run(
            ["git", "clone", "--filter=blob:none", "https://github.com/vllm-project/vllm.git", repo_path],
            capture_output=True, text=True, timeout=300
        )
        if result.returncode != 0:
            print(f"Failed to clone: {result.stderr}")
            return None

    # Fetch and checkout target commit
    subprocess.run(["git", "fetch", "--all"], cwd=repo_path, capture_output=True, timeout=120)

    # Walk ancestors looking for available wheel
    current = target_commit
    for i in range(50):  # Max 50 ancestors
        if check_wheel_url_exists(current):
            print(f"Found wheel at ancestor {i}: {current[:12]}")
            return current

        # Get parent commit
        result = subprocess.run(
            ["git", "rev-parse", f"{current}^"],
            cwd=repo_path, capture_output=True, text=True, timeout=10
        )
        if result.returncode != 0:
            break
        current = result.stdout.strip()

    print(f"No wheel found within 50 ancestors of {target_commit[:12]}")
    return None


def install_vllm_with_python_overlay(
    base_wheel_commit: str,
    target_commit: str,
    repo_path: str = "/tmp/vllm-overlay"
) -> Tuple[bool, str]:
    """
    Install vLLM by combining a base wheel with Python files from target commit.

    This enables running Python-only commits that don't have pre-built wheels.
    The C/CUDA extensions come from the base wheel, Python files from target commit.

    Args:
        base_wheel_commit: Commit hash with available wheel (provides C/CUDA binaries)
        target_commit: Commit hash to run (provides Python files)
        repo_path: Directory for git operations

    Returns:
        (success, version_or_error) tuple
    """
    import shutil
    import site

    try:
        # Step 1: Install base wheel
        print(f"Installing base wheel from {base_wheel_commit[:12]}...")
        wheel_url = VLLM_WHEEL_URL_PATTERN.format(commit=base_wheel_commit)
        success, msg = install_wheel(wheel_url)
        if not success:
            return False, f"Failed to install base wheel: {msg}"

        # Step 2: Clone/update repo and checkout target commit
        print(f"Checking out target commit {target_commit[:12]}...")
        if os.path.exists(repo_path):
            shutil.rmtree(repo_path)

        result = subprocess.run(
            ["git", "clone", "--filter=blob:none", "https://github.com/vllm-project/vllm.git", repo_path],
            capture_output=True, text=True, timeout=300
        )
        if result.returncode != 0:
            return False, f"Git clone failed: {result.stderr}"

        result = subprocess.run(
            ["git", "checkout", target_commit],
            cwd=repo_path, capture_output=True, text=True, timeout=60
        )
        if result.returncode != 0:
            return False, f"Git checkout failed: {result.stderr}"

        # Step 3: Find vLLM in site-packages
        vllm_site_path = None
        for sp in site.getsitepackages():
            candidate = Path(sp) / "vllm"
            if candidate.exists():
                vllm_site_path = candidate
                break

        if not vllm_site_path:
            return False, "Could not find vLLM in site-packages"

        print(f"Overlaying Python files to {vllm_site_path}...")

        # Step 4: Copy Python files from checkout to site-packages
        # Only copy .py files, preserve .so and other compiled files
        source_vllm = Path(repo_path) / "vllm"
        copied_count = 0

        for src_file in source_vllm.rglob("*.py"):
            rel_path = src_file.relative_to(source_vllm)
            dst_file = vllm_site_path / rel_path

            # Create parent directories if needed
            dst_file.parent.mkdir(parents=True, exist_ok=True)

            # Copy the Python file
            shutil.copy2(src_file, dst_file)
            copied_count += 1

        print(f"Overlaid {copied_count} Python files from {target_commit[:12]} onto base {base_wheel_commit[:12]}")

        # Step 5: Verify vLLM still imports
        result = subprocess.run(
            ["python", "-c", "import vllm; print(vllm.__version__)"],
            capture_output=True, text=True, cwd="/tmp"
        )

        if result.returncode != 0:
            return False, f"vLLM import failed after overlay: {result.stderr}"

        version = result.stdout.strip()
        return True, f"{version} (overlay: {target_commit[:8]} on {base_wheel_commit[:8]})"

    except Exception as e:
        return False, f"Overlay failed: {str(e)}"


def install_vllm_for_commit(commit: str) -> Tuple[bool, str, str]:
    """
    Install vLLM for a specific commit, using wheel if available or overlay if not.

    Returns:
        (success, version_or_error, method) tuple
        method is one of: "wheel", "overlay", "failed"
    """
    # Try exact wheel first
    if check_wheel_url_exists(commit):
        wheel_url = VLLM_WHEEL_URL_PATTERN.format(commit=commit)
        success, msg = install_wheel(wheel_url)
        if success:
            return True, msg, "wheel"
        print(f"Wheel install failed despite existing: {msg}")

    # Find nearest ancestor with wheel and use overlay
    print(f"No wheel for {commit[:12]}, searching for ancestor...")
    base_commit = find_nearest_wheel_commit(commit)

    if base_commit:
        success, msg = install_vllm_with_python_overlay(base_commit, commit)
        if success:
            return True, msg, "overlay"
        return False, msg, "failed"

    return False, f"No wheel available and no ancestor wheel found for {commit[:12]}", "failed"


def build_vllm_from_source(
    base_commit: str,
    patch_content: Optional[str] = None,
    build_dir: str = "/tmp/vllm-build"
) -> Tuple[bool, str]:
    """
    Build vLLM from source at a specific commit, optionally with a patch applied.

    This is used for agent benchmarks that include C/CUDA changes requiring recompilation.

    Args:
        base_commit: The commit hash to checkout before applying patch
        patch_content: Optional unified diff patch to apply
        build_dir: Directory to clone and build in

    Returns:
        (success, message) tuple
    """
    import shutil

    try:
        # Clean up any existing build directory
        if os.path.exists(build_dir):
            shutil.rmtree(build_dir)

        # Uninstall existing vLLM
        print("Uninstalling existing vLLM...")
        subprocess.run(
            ["uv", "pip", "uninstall", "--system", "vllm"],
            capture_output=True,
            timeout=60,
        )

        # Clone vLLM repository (full clone needed for checkout)
        print(f"Cloning vLLM repository...")
        result = subprocess.run(
            ["git", "clone", "https://github.com/vllm-project/vllm.git", build_dir],
            capture_output=True,
            text=True,
            timeout=600,
        )
        if result.returncode != 0:
            return False, f"Git clone failed: {result.stderr}"

        # Checkout the base commit
        print(f"Checking out commit {base_commit}...")
        result = subprocess.run(
            ["git", "checkout", base_commit],
            cwd=build_dir,
            capture_output=True,
            text=True,
            timeout=60,
        )
        if result.returncode != 0:
            return False, f"Git checkout failed: {result.stderr}"

        # Apply patch if provided
        if patch_content:
            print("Applying agent patch...")
            patch_file = f"{build_dir}/agent.patch"
            with open(patch_file, 'w') as f:
                f.write(patch_content)

            # Try git apply first (handles more cases)
            result = subprocess.run(
                ["git", "apply", "--verbose", "agent.patch"],
                cwd=build_dir,
                capture_output=True,
                text=True,
                timeout=60,
            )
            if result.returncode != 0:
                # Try with patch command as fallback
                result = subprocess.run(
                    ["patch", "-p1", "-i", "agent.patch"],
                    cwd=build_dir,
                    capture_output=True,
                    text=True,
                    timeout=60,
                )
                if result.returncode != 0:
                    return False, f"Patch apply failed: {result.stderr}\nstdout: {result.stdout}"

            print(f"Patch applied successfully")

        # Install build dependencies
        print("Installing vLLM build dependencies...")
        result = subprocess.run(
            ["uv", "pip", "install", "--system", "-r", "requirements/build.txt"],
            cwd=build_dir,
            capture_output=True,
            text=True,
            timeout=300,
        )
        # Don't fail if requirements file doesn't exist (older vLLM versions)

        # Build and install vLLM
        # OPTIMIZED: Target only H100 architecture to reduce build time by ~60-80%
        print("Building vLLM from source (optimized for H100, ~10-15 minutes)...")
        env = os.environ.copy()
        env["MAX_JOBS"] = "16"  # Increased from 4 for faster builds
        env["VLLM_INSTALL_PUNICA_KERNELS"] = "0"  # Skip optional kernels to speed up
        # CRITICAL: Set CUDA architecture to only build for H100 (SM 9.0)
        env["TORCH_CUDA_ARCH_LIST"] = CUDA_BUILD_TARGETS
        env["CMAKE_CUDA_ARCHITECTURES"] = "90"  # CMake format: 90 = SM 9.0 (H100)
        env["VLLM_TARGET_DEVICE"] = "cuda"

        result = subprocess.run(
            ["uv", "pip", "install", "--system", "-e", ".", "--no-build-isolation"],
            cwd=build_dir,
            capture_output=True,
            text=True,
            timeout=2400,  # 40 min timeout (reduced due to optimizations)
            env=env,
        )

        if result.returncode != 0:
            return False, f"Build failed: {result.stderr[-2000:]}"

        # Verify installation
        version_result = subprocess.run(
            ["python", "-c", "import vllm; print(vllm.__version__)"],
            capture_output=True,
            text=True,
            cwd="/tmp",  # Run from /tmp to avoid local vllm/ directory
        )

        if version_result.returncode != 0:
            return False, f"vLLM import failed after build: {version_result.stderr}"

        version = version_result.stdout.strip()
        print(f"vLLM {version} built and installed successfully")
        return True, version

    except subprocess.TimeoutExpired:
        return False, "Build timed out (exceeded 1 hour)"
    except Exception as e:
        return False, f"Build error: {str(e)}"


def normalize_pyproject_license(build_path: str) -> None:
    """
    Normalize pyproject.toml license field to be compatible with all setuptools versions.

    Different setuptools versions accept different license formats:
    - Very old: license = { file = "LICENSE" }
    - Middle: license = { text = "Apache-2.0" } (PEP 621 compliant)
    - Very new: license = "Apache-2.0" (SPDX identifier, PEP 639)

    The { text = "..." } format is universally accepted across all setuptools versions.
    This function converts ANY license format to { text = "Apache-2.0" }.
    """
    import re

    pyproject_path = os.path.join(build_path, "pyproject.toml")
    if not os.path.exists(pyproject_path):
        return

    with open(pyproject_path, 'r') as f:
        content = f.read()

    original = content

    # Universal target format that works with all setuptools versions
    target_license = 'license = { text = "Apache-2.0" }'

    # Pattern 1: SPDX string format - license = "Apache-2.0"
    # This is the NEW format that some middle-version setuptools reject
    spdx_pattern = r'^license\s*=\s*"[^"]*"$'
    content = re.sub(spdx_pattern, target_license, content, flags=re.MULTILINE)

    # Pattern 2: Old table format - license = { file = "LICENSE" } or license = {"file"= "LICENSE"}
    old_file_pattern = r'license\s*=\s*\{\s*["\']?file["\']?\s*[=:]\s*["\'][^"\']*["\']\s*\}'
    content = re.sub(old_file_pattern, target_license, content, flags=re.IGNORECASE)

    # Pattern 3: Table with text key but different value - license = { text = "Apache License 2.0" }
    old_text_pattern = r'license\s*=\s*\{\s*["\']?text["\']?\s*[=:]\s*["\'][^"\']*["\']\s*\}'
    content = re.sub(old_text_pattern, target_license, content, flags=re.IGNORECASE)

    # Remove license-files if present - it's not allowed with { text = ... } format
    # and causes "project must not contain {'license-files'}" error
    license_files_pattern = r'^license-files\s*=\s*\[.*\]\s*$'
    content = re.sub(license_files_pattern, '', content, flags=re.MULTILINE)

    # Clean up any resulting double blank lines
    content = re.sub(r'\n\n\n+', '\n\n', content)

    if content != original:
        print(f"Normalized pyproject.toml license field for setuptools compatibility")
        with open(pyproject_path, 'w') as f:
            f.write(content)


def install_from_cached_wheel(commit_hash: str, cache_dir: str = "/cache") -> Tuple[bool, str]:
    """
    Install vLLM from a cached wheel file.

    Args:
        commit_hash: The commit hash to look for
        cache_dir: Root directory for build cache (Modal volume mount point)

    Returns:
        (success, version_or_error) tuple
    """
    import glob as glob_mod

    wheel_dir = f"{cache_dir}/wheels"
    wheel_pattern = f"{wheel_dir}/vllm-*+g{commit_hash[:8]}*.whl"
    existing_wheels = glob_mod.glob(wheel_pattern)

    if not existing_wheels:
        return False, f"No cached wheel found for {commit_hash[:8]}"

    wheel_path = existing_wheels[0]
    wheel_name = os.path.basename(wheel_path)
    print(f"[WHEEL INSTALL] Found cached wheel: {wheel_name}")

    # Install the wheel
    install_result = subprocess.run(
        ["uv", "pip", "install", "--system", "--force-reinstall", wheel_path],
        capture_output=True, text=True, timeout=300,
    )

    if install_result.returncode != 0:
        return False, f"Wheel install failed: {install_result.stderr[-500:]}"

    # Verify installation
    version_result = subprocess.run(
        ["python", "-c", "import vllm; print(vllm.__version__)"],
        capture_output=True, text=True, cwd="/tmp",
    )

    if version_result.returncode != 0:
        return False, f"vLLM import failed after wheel install: {version_result.stderr[:500]}"

    version = version_result.stdout.strip()
    print(f"[WHEEL INSTALL] SUCCESS: vLLM {version} installed from cached wheel")
    return True, version


def get_or_create_cached_build(base_commit: str, cache_dir: str = "/cache") -> Tuple[bool, str]:
    """
    Check if a cached vLLM wheel exists for the commit, if not create one.

    This function now uses WHEEL-based caching for portability across containers.
    Wheels are built once on CPU and can be installed instantly on GPU.

    Args:
        base_commit: The commit hash to build/cache
        cache_dir: Root directory for build cache (Modal volume mount point)

    Returns:
        (cache_hit, build_path_or_error) - whether cache existed, path to build directory or error
    """
    import shutil
    import glob as glob_mod

    # FIRST: Try to install from cached wheel (fast path)
    wheel_dir = f"{cache_dir}/wheels"
    wheel_pattern = f"{wheel_dir}/vllm-*+g{base_commit[:8]}*.whl"
    existing_wheels = glob_mod.glob(wheel_pattern)

    if existing_wheels:
        wheel_path = existing_wheels[0]
        wheel_name = os.path.basename(wheel_path)
        print(f"[CACHE] Wheel cache HIT: {wheel_name}")

        # Check if already installed
        version_result = subprocess.run(
            ["python", "-c", "import vllm; print(vllm.__version__)"],
            capture_output=True, text=True, cwd="/tmp",
        )
        if version_result.returncode == 0 and base_commit[:8] in version_result.stdout:
            print(f"[CACHE] vLLM {version_result.stdout.strip()} already installed")
            # Return a dummy build path - we don't need it for wheel installs
            return True, f"{cache_dir}/build_{base_commit}"

        # Install the cached wheel
        print(f"[CACHE] Installing cached wheel...")
        install_result = subprocess.run(
            ["uv", "pip", "install", "--system", "--force-reinstall", wheel_path],
            capture_output=True, text=True, timeout=300,
        )

        if install_result.returncode == 0:
            # Verify
            verify = subprocess.run(
                ["python", "-c", "import vllm; print(vllm.__version__)"],
                capture_output=True, text=True, cwd="/tmp",
            )
            if verify.returncode == 0:
                print(f"[CACHE] vLLM {verify.stdout.strip()} installed from cached wheel")
                return True, f"{cache_dir}/build_{base_commit}"
            else:
                print(f"[CACHE] Warning: Wheel install succeeded but import fails: {verify.stderr[:200]}")
        else:
            print(f"[CACHE] Warning: Wheel install failed: {install_result.stderr[-300:]}")

    # FALLBACK: Check for legacy editable install cache
    build_path = f"{cache_dir}/{base_commit}"
    marker_file = f"{build_path}/.build_complete"

    if os.path.exists(marker_file):
        cmake_exists = os.path.exists(f"{build_path}/CMakeLists.txt")
        build_dir_exists = os.path.exists(f"{build_path}/build")

        if cmake_exists and build_dir_exists:
            print(f"[CACHE] Legacy editable cache found for {base_commit[:8]}, re-registering...")
            # Try to re-register the editable install
            env = os.environ.copy()
            env["TORCH_CUDA_ARCH_LIST"] = CUDA_BUILD_TARGETS
            env["CMAKE_CUDA_ARCHITECTURES"] = "90"  # CMake format: 90 = SM 9.0 (H100)
            env["VLLM_TARGET_DEVICE"] = "cuda"

            reinstall_result = subprocess.run(
                ["uv", "pip", "install", "--system", "-e", ".", "--no-build-isolation"],
                cwd=build_path, capture_output=True, text=True,
                timeout=1800,  # 30 min for potential rebuild
                env=env,
            )
            if reinstall_result.returncode == 0:
                verify = subprocess.run(
                    ["python", "-c", "import vllm; print(vllm.__version__)"],
                    capture_output=True, text=True, cwd="/tmp",
                )
                if verify.returncode == 0:
                    print(f"[CACHE] vLLM {verify.stdout.strip()} re-registered from legacy cache")
                    return True, build_path
        else:
            print(f"[CACHE] Legacy cache invalid, removing...")
            shutil.rmtree(build_path, ignore_errors=True)

    print(f"[CACHE] Cache MISS: Building vLLM wheel at {base_commit[:8]} (this is a one-time cost)...")

    # Clean up any partial build
    if os.path.exists(build_path):
        shutil.rmtree(build_path)

    os.makedirs(build_path, exist_ok=True)
    os.makedirs(wheel_dir, exist_ok=True)

    try:
        # Clone vLLM repository
        print(f"[CACHE] Cloning vLLM repository...")
        result = subprocess.run(
            ["git", "clone", "https://github.com/vllm-project/vllm.git", build_path],
            capture_output=True,
            text=True,
            timeout=600,
        )
        if result.returncode != 0:
            return False, f"Git clone failed: {result.stderr}"

        # Checkout the base commit
        print(f"[CACHE] Checking out commit {base_commit}...")
        result = subprocess.run(
            ["git", "checkout", base_commit],
            cwd=build_path,
            capture_output=True,
            text=True,
            timeout=60,
        )
        if result.returncode != 0:
            return False, f"Git checkout failed: {result.stderr}"

        # Normalize pyproject.toml license field for setuptools compatibility
        normalize_pyproject_license(build_path)

        # Install build dependencies
        print("[CACHE] Installing build dependencies...")
        subprocess.run(
            ["uv", "pip", "install", "--system", "-r", "requirements/build.txt"],
            cwd=build_path,
            capture_output=True,
            text=True,
            timeout=300,
        )
        subprocess.run(
            ["uv", "pip", "install", "--system", "wheel", "build"],
            capture_output=True, text=True, timeout=60,
        )

        # Build wheel (cached for instant reuse)
        # OPTIMIZED: Target only H100 architecture to reduce build time by ~60-80%
        print("[CACHE] Building vLLM wheel (optimized for H100, ~15-25 minutes)...")
        env = os.environ.copy()
        env["MAX_JOBS"] = "16"  # Use available CPUs
        env["VLLM_INSTALL_PUNICA_KERNELS"] = "0"
        # CRITICAL: Set CUDA architecture to only build for H100 (SM 9.0)
        # Must set BOTH env vars - TORCH_CUDA_ARCH_LIST for PyTorch/vLLM, CMAKE_CUDA_ARCHITECTURES for CMake
        env["TORCH_CUDA_ARCH_LIST"] = CUDA_BUILD_TARGETS
        env["CMAKE_CUDA_ARCHITECTURES"] = "90"  # CMake format: 90 = SM 9.0 (H100)
        env["VLLM_TARGET_DEVICE"] = "cuda"

        result = subprocess.run(
            ["python", "-m", "pip", "wheel", ".", "--no-deps", "--no-build-isolation", "-w", wheel_dir],
            cwd=build_path,
            capture_output=True,
            text=True,
            timeout=7200,  # 2 hour timeout
            env=env,
        )

        if result.returncode != 0:
            # Clean up failed build
            shutil.rmtree(build_path, ignore_errors=True)
            return False, f"Wheel build failed: {result.stderr[-2000:]}"

        # Find and install the built wheel
        new_wheels = glob_mod.glob(f"{wheel_dir}/vllm-*+g{base_commit[:8]}*.whl")
        if not new_wheels:
            # Try broader pattern
            new_wheels = glob_mod.glob(f"{wheel_dir}/vllm-*.whl")

        if not new_wheels:
            shutil.rmtree(build_path, ignore_errors=True)
            return False, "No wheel file produced"

        wheel_path = max(new_wheels, key=os.path.getctime)
        wheel_name = os.path.basename(wheel_path)
        print(f"[CACHE] Built wheel: {wheel_name}")

        # Install the wheel
        install_result = subprocess.run(
            ["uv", "pip", "install", "--system", "--force-reinstall", wheel_path],
            capture_output=True, text=True, timeout=300,
        )

        if install_result.returncode != 0:
            return False, f"Wheel install failed: {install_result.stderr[-500:]}"

        # Verify installation
        version_result = subprocess.run(
            ["python", "-c", "import vllm; print(vllm.__version__)"],
            capture_output=True,
            text=True,
            cwd="/tmp",
        )

        if version_result.returncode != 0:
            return False, f"vLLM import failed: {version_result.stderr}"

        # IMPORTANT: Keep build directory for incremental compilation!
        # The wheel is cached in /cache/wheels/, but for agent patches with C/CUDA changes,
        # we need the source + build artifacts in the build directory for ninja to do
        # incremental compilation (only recompile changed files).
        # DO NOT DELETE: shutil.rmtree(build_path, ignore_errors=True)

        # Create marker file to indicate successful build
        marker_file = f"{build_path}/.build_complete"
        with open(marker_file, "w") as f:
            f.write(f"wheel={wheel_name}\nversion={version_result.stdout.strip()}\n")

        print(f"[CACHE] Cache created: vLLM {version_result.stdout.strip()} wheel + build dir (for incremental)")
        return False, f"{cache_dir}/build_{base_commit}"  # cache_hit=False, we just created it

    except subprocess.TimeoutExpired:
        shutil.rmtree(build_path, ignore_errors=True)
        return False, "Cache build timed out"
    except Exception as e:
        shutil.rmtree(build_path, ignore_errors=True)
        return False, f"Cache build error: {str(e)}"


def build_vllm_incremental(
    base_commit: str,
    patch_content: str,
    cache_dir: str = "/cache"
) -> Tuple[bool, str]:
    """
    Apply patch to cached build and run incremental compilation.

    This leverages ninja's incremental build capability - only changed .cu/.cpp
    files are recompiled, making it much faster than a full build.

    Expected time:
    - Cache miss (first time): 30-60 minutes (full build, cached)
    - Cache hit + patch: 3-10 minutes (incremental recompile)

    Args:
        base_commit: The base commit hash (must have cached build)
        patch_content: Unified diff patch to apply
        cache_dir: Root directory for build cache

    Returns:
        (success, version_or_error) tuple
    """
    import shutil

    build_path = f"{cache_dir}/{base_commit}"

    # Step 1: Ensure cache exists
    cache_hit, result = get_or_create_cached_build(base_commit, cache_dir)
    if isinstance(result, str) and result.startswith(("Git", "Full build", "Cache build", "vLLM import")):
        # Error message, not path
        return False, result

    build_path = result if result.startswith("/") else build_path

    # Step 2: Reset any previous patches (clean state)
    print("Resetting cached build to clean state...")
    result = subprocess.run(
        ["git", "reset", "--hard", "HEAD"],
        cwd=build_path,
        capture_output=True,
        text=True,
        timeout=60,
    )
    if result.returncode != 0:
        return False, f"Git reset failed: {result.stderr}"

    # Clean only specific directories, PRESERVE build artifacts and marker
    # -fd removes untracked files and directories, but we exclude:
    # - .build_complete (our cache marker)
    # - build/ (cmake build directory with .o files)
    # - *.egg-info (Python package info)
    result = subprocess.run(
        ["git", "clean", "-fd", "-e", ".build_complete", "-e", "build/", "-e", "*.egg-info"],
        cwd=build_path,
        capture_output=True,
        text=True,
        timeout=60,
    )

    # Step 3: Apply the agent patch
    print("Applying agent patch to cached build...")
    patch_file = f"{build_path}/agent.patch"
    with open(patch_file, 'w') as f:
        f.write(patch_content)

    # Try git apply
    result = subprocess.run(
        ["git", "apply", "--verbose", "agent.patch"],
        cwd=build_path,
        capture_output=True,
        text=True,
        timeout=60,
    )
    if result.returncode != 0:
        # Fallback to patch command
        result = subprocess.run(
            ["patch", "-p1", "-i", "agent.patch"],
            cwd=build_path,
            capture_output=True,
            text=True,
            timeout=60,
        )
        if result.returncode != 0:
            return False, f"Patch apply failed: {result.stderr}\nstdout: {result.stdout}"

    print("Patch applied successfully")

    # Normalize pyproject.toml license field for setuptools compatibility
    normalize_pyproject_license(build_path)

    # Step 4: Check if patch modifies C/CUDA files
    # NOTE: We DO NOT clear cmake cache - ninja handles incremental builds correctly.
    # Clearing cmake cache would force a FULL rebuild, defeating the purpose of caching.
    cuda_extensions = ('.cu', '.cpp', '.c', '.h', '.cuh', '.hpp')
    has_cuda_changes = any(ext in patch_content.lower() for ext in cuda_extensions)
    if has_cuda_changes:
        print("Patch modifies C/CUDA files - ninja will recompile only changed files")
    else:
        print("Python-only patch - no CUDA recompilation needed")

    # NOTE: We DO NOT uninstall vLLM before rebuild.
    # pip/uv can handle in-place updates with editable installs (-e .).
    # Uninstalling first is wasteful and adds extra time.

    # Step 5: Run incremental build
    # Ninja will detect only changed files and recompile just those
    # OPTIMIZED: Target only H100 architecture to reduce build time by ~60-80%
    print("Running incremental build (ninja will only recompile changed files)...")

    env = os.environ.copy()
    env["MAX_JOBS"] = "16"  # Increased from 8 for faster builds
    env["VLLM_INSTALL_PUNICA_KERNELS"] = "0"
    # CRITICAL: Set CUDA architecture to only build for H100 (SM 9.0)
    # Must set BOTH env vars - TORCH_CUDA_ARCH_LIST for PyTorch/vLLM, CMAKE_CUDA_ARCHITECTURES for CMake
    env["TORCH_CUDA_ARCH_LIST"] = CUDA_BUILD_TARGETS
    env["CMAKE_CUDA_ARCHITECTURES"] = "90"  # CMake format: 90 = SM 9.0 (H100)
    env["VLLM_TARGET_DEVICE"] = "cuda"

    # For cache hit: incremental should be <5 min with optimizations
    # For cache miss: this runs AFTER get_or_create_cached_build() already did full build,
    #   so we're just reinstalling from the already-built source (should be fast)
    build_timeout = 2400 if not cache_hit else 900  # Reduced due to optimizations
    print(f"Running build with {build_timeout}s timeout (cache_hit={cache_hit})...")

    result = subprocess.run(
        ["uv", "pip", "install", "--system", "-e", ".", "--no-build-isolation"],
        cwd=build_path,
        capture_output=True,
        text=True,
        timeout=build_timeout,
        env=env,
    )

    if result.returncode != 0:
        return False, f"Incremental build failed: {result.stderr[-2000:]}"

    # Step 6: Verify installation
    version_result = subprocess.run(
        ["python", "-c", "import vllm; print(vllm.__version__)"],
        capture_output=True,
        text=True,
        cwd="/tmp",
    )

    if version_result.returncode != 0:
        return False, f"vLLM import failed after incremental build: {version_result.stderr}"

    version = version_result.stdout.strip()
    print(f"Incremental build successful: vLLM {version}")
    return True, version


def start_server(model: str, port: int = 29000, tensor_parallel: int = 1, extra_args: list = None, log_file: str = None) -> subprocess.Popen:
    """Start vLLM server as background process.

    Args:
        model: Model name/path
        port: Server port
        tensor_parallel: Number of GPUs for tensor parallelism
        extra_args: Additional server arguments
        log_file: Optional path to capture server logs (for debugging failures)

    Returns:
        subprocess.Popen object
    """
    print(f"  [start_server] Preparing to start server on port {port}...")

    # Diagnostic: check what's on the port before cleanup
    try:
        port_check = subprocess.run(["lsof", "-i", f":{port}"], capture_output=True, text=True, timeout=5)
        if port_check.stdout.strip():
            print(f"  [start_server] WARNING: Port {port} already in use before cleanup:")
            print(f"    {port_check.stdout.strip()[:200]}")
    except Exception as e:
        print(f"  [start_server] lsof check failed: {e}")

    # Kill any existing vLLM servers (more aggressive - use SIGKILL)
    try:
        subprocess.run(["pkill", "-9", "-f", "vllm.entrypoints"], capture_output=True, timeout=5)
        time.sleep(2)  # Give time for ports to be released
    except:
        pass

    # Kill any Ray workers that might be holding resources
    try:
        subprocess.run(["pkill", "-9", "-f", "ray::"], capture_output=True, timeout=5)
        time.sleep(1)
    except:
        pass

    # Kill any process on the target port (using fuser with SIGKILL)
    try:
        subprocess.run(["fuser", "-k", "-9", f"{port}/tcp"], capture_output=True, timeout=5)
        time.sleep(2)  # Longer wait after killing port processes
    except:
        pass

    # Double-check the port is free after cleanup
    try:
        port_check = subprocess.run(["lsof", "-i", f":{port}"], capture_output=True, text=True, timeout=5)
        if port_check.stdout.strip():
            print(f"  [start_server] WARNING: Port {port} STILL in use after cleanup:")
            print(f"    {port_check.stdout.strip()[:200]}")
            # Last resort: wait longer
            time.sleep(5)
    except:
        pass

    print(f"  [start_server] Port cleanup complete, starting server...")

    cmd = [
        "python", "-m", "vllm.entrypoints.openai.api_server",
        "--model", model,
        "--host", "0.0.0.0",  # Bind to all interfaces
        "--port", str(port),
        "--tensor-parallel-size", str(tensor_parallel),
        "--disable-frontend-multiprocessing",  # May help with port binding issues
    ]

    if extra_args:
        cmd.extend(extra_args)

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = ",".join(str(i) for i in range(tensor_parallel))
    # Enable unbuffered output for real-time logging
    env["PYTHONUNBUFFERED"] = "1"
    # Fix for vLLM 0.6.x port binding issue (socket duplication during fork)
    # See: https://github.com/vllm-project/vllm/issues/8791
    env["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
    # CRITICAL: Set MASTER_PORT to avoid NCCL binding to same port as HTTP server
    # Use completely different port range (35xxx) that doesn't collide with any vLLM internal ports
    # HTTP ports are now 18001-18003, NCCL should be far from both HTTP and 29xxx internal range
    nccl_port = 35000 + (port % 1000)  # e.g., 18001 -> 35001, 18002 -> 35002
    env["MASTER_PORT"] = str(nccl_port)
    env["MASTER_ADDR"] = "127.0.0.1"
    print(f"  [start_server] Setting MASTER_PORT={nccl_port}, MASTER_ADDR=127.0.0.1 (HTTP port={port})")

    # Capture logs to file if path provided, otherwise inherit (stream to Modal)
    if log_file:
        log_fh = open(log_file, 'w')
        process = subprocess.Popen(
            cmd,
            stdout=log_fh,
            stderr=subprocess.STDOUT,  # Combine stderr into stdout
            env=env,
            preexec_fn=os.setsid,
        )
        process._log_file = log_file  # Store for later retrieval
        process._log_fh = log_fh
    else:
        process = subprocess.Popen(
            cmd,
            stdout=None,  # Inherit parent stdout - streams to Modal logs
            stderr=None,  # Inherit parent stderr - streams to Modal logs
            env=env,
            preexec_fn=os.setsid,
        )

    return process


def get_server_logs(process: subprocess.Popen, max_lines: int = 200) -> str:
    """Get captured server logs if available."""
    if hasattr(process, '_log_file') and process._log_file:
        try:
            # Close the file handle first
            if hasattr(process, '_log_fh'):
                process._log_fh.close()
            with open(process._log_file, 'r') as f:
                lines = f.readlines()
                return ''.join(lines[-max_lines:])
        except:
            pass
    return ""


def wait_for_server(port: int = 29000, timeout: int = 1200) -> bool:
    """Wait for server to be ready.

    Default timeout increased to 1200s (20 min) to handle larger models
    that need time to load weights and compile CUDA kernels.
    """
    import urllib.request

    start = time.time()
    while time.time() - start < timeout:
        try:
            with urllib.request.urlopen(f"http://localhost:{port}/health", timeout=5) as response:
                if response.status == 200:
                    return True
        except:
            pass
        time.sleep(5)

    return False


def stop_server(process: subprocess.Popen, port: int = 29000):
    """Stop server process and cleanup, ensuring port is released."""
    print(f"  [stop_server] Stopping server process and cleaning up port {port}...")
    if process:
        try:
            # Kill entire process group with SIGTERM first
            os.killpg(os.getpgid(process.pid), signal.SIGTERM)
            process.wait(timeout=10)
        except:
            try:
                # Force kill if SIGTERM didn't work
                os.killpg(os.getpgid(process.pid), signal.SIGKILL)
                process.wait(timeout=5)
            except:
                pass

    # Additional cleanup to ensure port is released
    try:
        subprocess.run(["pkill", "-9", "-f", "vllm.entrypoints"], capture_output=True, timeout=5)
    except:
        pass

    try:
        subprocess.run(["fuser", "-k", "-9", f"{port}/tcp"], capture_output=True, timeout=5)
    except:
        pass

    # Wait for OS to release the port
    time.sleep(3)
    print(f"  [stop_server] Cleanup complete.")


def parse_metrics(output: str) -> Dict[str, float]:
    """Parse benchmark output for metrics.

    Handles multiple output formats:
    - Old benchmark scripts (benchmark_serving.py, benchmark_throughput.py)
    - New vLLM CLI (vllm bench serve, vllm bench throughput)
    - JSON output from some benchmarks
    """
    metrics = {}

    # Standard patterns from benchmark_serving.py and vllm bench serve
    patterns = {
        # Throughput patterns
        r"Request throughput:\s*([\d.]+)\s*requests/s": "request_throughput",
        r"Output token throughput:\s*([\d.]+)\s*tokens/s": "output_throughput",
        r"Total Token throughput:\s*([\d.]+)\s*tokens/s": "total_throughput",
        r"Input token throughput:\s*([\d.]+)\s*tokens/s": "input_throughput",
        r"Throughput:\s*([\d.]+)\s*requests/s": "request_throughput",
        r"Throughput:\s*[\d.]+\s*requests/s,\s*([\d.]+)\s*tokens/s": "total_throughput",

        # New vLLM CLI throughput format (vllm bench throughput)
        r"Throughput:\s*([\d.]+)\s*requests/s,\s*[\d.]+\s*total": "request_throughput",
        r"requests/s,\s*([\d.]+)\s*total\s*tokens/s": "total_throughput",
        r"total\s*tokens/s,\s*([\d.]+)\s*output\s*tokens/s": "output_throughput",

        # TTFT patterns
        r"Mean TTFT \(ms\):\s*([\d.]+)": "ttft_mean",
        r"Median TTFT \(ms\):\s*([\d.]+)": "ttft_median",
        r"P50 TTFT \(ms\):\s*([\d.]+)": "ttft_median",
        r"P90 TTFT \(ms\):\s*([\d.]+)": "ttft_p90",
        r"P99 TTFT \(ms\):\s*([\d.]+)": "ttft_p99",

        # TPOT patterns
        r"Mean TPOT \(ms\):\s*([\d.]+)": "tpot_mean",
        r"Median TPOT \(ms\):\s*([\d.]+)": "tpot_median",
        r"P90 TPOT \(ms\):\s*([\d.]+)": "tpot_p90",
        r"P99 TPOT \(ms\):\s*([\d.]+)": "tpot_p99",

        # ITL patterns
        r"Mean ITL \(ms\):\s*([\d.]+)": "itl_mean",
        r"Median ITL \(ms\):\s*([\d.]+)": "itl_median",
        r"P90 ITL \(ms\):\s*([\d.]+)": "itl_p90",
        r"P99 ITL \(ms\):\s*([\d.]+)": "itl_p99",

        # E2E Latency patterns
        r"Mean E2E Latency \(ms\):\s*([\d.]+)": "e2e_latency_mean",
        r"Median E2E Latency \(ms\):\s*([\d.]+)": "e2e_latency_median",
        r"P90 E2E Latency \(ms\):\s*([\d.]+)": "e2e_latency_p90",
        r"P99 E2E Latency \(ms\):\s*([\d.]+)": "e2e_latency_p99",

        # General latency patterns
        r"Avg latency:\s*([\d.]+)\s*seconds": "latency_avg_s",
        r"Avg latency:\s*([\d.]+)\s*ms": "latency_avg",

        # Generic throughput catch-all
        r"throughput[=:]\s*([\d.]+)": "throughput",
    }

    for pattern, key in patterns.items():
        match = re.search(pattern, output, re.IGNORECASE)
        if match:
            value = float(match.group(1))
            # Convert seconds to ms for latency if needed
            if key == "latency_avg_s":
                metrics["latency_avg"] = value * 1000
            else:
                metrics[key] = value

    # Try to parse JSON output (some benchmarks output JSON)
    try:
        import json
        # Look for JSON block in output
        json_match = re.search(r'\{[^{}]*"throughput"[^{}]*\}', output, re.DOTALL)
        if json_match:
            data = json.loads(json_match.group())
            if "throughput" in data and "throughput" not in metrics:
                metrics["throughput"] = float(data["throughput"])
            if "latency" in data and "latency_avg" not in metrics:
                lat = float(data["latency"])
                metrics["latency_avg"] = lat * 1000 if lat < 100 else lat
    except:
        pass

    # Also try to find metrics in table format (new vLLM CLI)
    # Format: "| Metric | Value |" tables
    table_patterns = {
        r"\|\s*Request throughput\s*\|\s*([\d.]+)\s*req/s": "request_throughput",
        r"\|\s*Output throughput\s*\|\s*([\d.]+)\s*tok/s": "output_throughput",
        r"\|\s*Total throughput\s*\|\s*([\d.]+)\s*tok/s": "total_throughput",
        r"\|\s*TTFT\s*\|\s*([\d.]+)\s*ms": "ttft_mean",
        r"\|\s*TPOT\s*\|\s*([\d.]+)\s*ms": "tpot_mean",
        r"\|\s*ITL\s*\|\s*([\d.]+)\s*ms": "itl_mean",
    }

    for pattern, key in table_patterns.items():
        if key not in metrics:
            match = re.search(pattern, output, re.IGNORECASE)
            if match:
                metrics[key] = float(match.group(1))

    return metrics


def run_simple_benchmark(model: str, port: int = 29000, num_requests: int = 50) -> Dict[str, float]:
    """Run a simple benchmark against the vLLM server."""
    import urllib.request
    import json

    url = f"http://127.0.0.1:{port}/v1/completions"
    metrics = {}

    latencies = []
    total_tokens = 0

    for i in range(num_requests):
        data = json.dumps({
            "model": model,
            "prompt": "Write a short story about a robot learning to paint:",
            "max_tokens": 100,
            "temperature": 0.7
        }).encode()

        req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})

        start = time.time()
        try:
            with urllib.request.urlopen(req, timeout=60) as response:
                result = json.loads(response.read().decode())
                latency = time.time() - start
                latencies.append(latency)
                if "usage" in result:
                    total_tokens += result["usage"].get("completion_tokens", 0)
        except Exception as e:
            print(f"Request {i} failed: {e}")
            continue

    if latencies:
        total_time = sum(latencies)
        metrics["request_throughput"] = len(latencies) / total_time
        metrics["latency_avg"] = sum(latencies) / len(latencies)
        metrics["latency_p50"] = sorted(latencies)[len(latencies) // 2]
        metrics["latency_p99"] = sorted(latencies)[int(len(latencies) * 0.99)]
        if total_tokens > 0:
            metrics["output_throughput"] = total_tokens / total_time
        metrics["successful_requests"] = len(latencies)
        metrics["total_requests"] = num_requests

    return metrics


def prepare_serving_command(perf_command: str, model: str, port: int = 8000) -> str:
    """Prepare benchmark client command - returns None to use built-in benchmark."""
    # We now use run_simple_benchmark instead of external commands
    return None


@app.function(
    image=base_image,
    gpu="H100",
    timeout=3600,
    secrets=[modal.Secret.from_name("huggingface-secret")],
    volumes={"/root/.cache/huggingface": model_cache},
)
def run_benchmark_single_gpu(
    wheel_url: str,
    perf_command: str,
    model: str,
    benchmark_type: str = "serving",
    num_prompts: int = 100,
) -> Dict[str, Any]:
    """Run benchmark on single H100 GPU.

    Args:
        wheel_url: URL to vLLM wheel
        perf_command: Benchmark command to run
        model: Model name/path
        benchmark_type: One of "serving", "throughput", "latency"
        num_prompts: Number of prompts for serving benchmark

    Returns:
        Dict with metrics, status, and timing info
    """
    result = {
        "status": "error",
        "metrics": {},
        "vllm_version": None,
        "error": None,
        "duration_s": 0,
    }

    start_time = time.time()

    try:
        # Install wheel
        success, version = install_wheel(wheel_url)
        if not success:
            result["error"] = f"Wheel installation failed: {version}"
            return result

        result["vllm_version"] = version

        if benchmark_type == "serving":
            # Start server
            server = start_server(model, port=29000, tensor_parallel=1)

            try:
                if not wait_for_server(port=29000, timeout=3600):
                    result["error"] = "Server failed to start within timeout"
                    return result

                # Run built-in benchmark
                print(f"Server ready, running benchmark against {model}...")
                benchmark_metrics = run_simple_benchmark(model, port=29000, num_requests=50)

                result["metrics"] = benchmark_metrics
                result["raw_output"] = f"Benchmark completed: {benchmark_metrics}"

                if result["metrics"]:
                    result["status"] = "success"
                else:
                    result["error"] = "No metrics from benchmark"

            finally:
                stop_server(server)

        else:
            # Direct benchmark (throughput, latency)
            bench_result = subprocess.run(
                perf_command,
                shell=True,
                capture_output=True,
                text=True,
                timeout=3600,
            )

            output = bench_result.stdout + bench_result.stderr
            result["metrics"] = parse_metrics(output)
            result["raw_output"] = output

            if result["metrics"]:
                result["status"] = "success"
            else:
                result["error"] = "No metrics parsed from output"

    except subprocess.TimeoutExpired:
        result["error"] = "Benchmark timed out"
    except Exception as e:
        result["error"] = str(e)
    finally:
        result["duration_s"] = time.time() - start_time
        model_cache.commit()

    return result


@app.function(
    image=base_image,
    gpu="H100:4",
    timeout=7200,
    secrets=[modal.Secret.from_name("huggingface-secret")],
    volumes={"/root/.cache/huggingface": model_cache},
)
def run_benchmark_4gpu(
    wheel_url: str,
    perf_command: str,
    model: str,
    benchmark_type: str = "serving",
) -> Dict[str, Any]:
    """Run benchmark on 4x H100 GPUs for large models (70B-340B)."""
    result = {
        "status": "error",
        "metrics": {},
        "vllm_version": None,
        "error": None,
        "duration_s": 0,
        "gpu_config": "H100:4",
    }

    start_time = time.time()

    try:
        success, version = install_wheel(wheel_url)
        if not success:
            result["error"] = f"Wheel installation failed: {version}"
            return result

        result["vllm_version"] = version

        if benchmark_type == "serving":
            server = start_server(model, port=29000, tensor_parallel=4)

            try:
                if not wait_for_server(port=29000, timeout=900):
                    result["error"] = "Server failed to start within timeout"
                    return result

                # Run built-in benchmark
                print(f"Server ready, running benchmark against {model} with 4x H100...")
                benchmark_metrics = run_simple_benchmark(model, port=29000, num_requests=50)

                result["metrics"] = benchmark_metrics
                result["raw_output"] = f"Benchmark completed: {benchmark_metrics}"

                if result["metrics"]:
                    result["status"] = "success"
                else:
                    result["error"] = "No metrics from benchmark"

            finally:
                stop_server(server)
        else:
            # Update command with tensor parallelism
            cmd = re.sub(r'--tensor-parallel-size\s+\d+', '--tensor-parallel-size 4', perf_command)
            if '--tensor-parallel-size' not in cmd:
                cmd += ' --tensor-parallel-size 4'

            bench_result = subprocess.run(
                cmd,
                shell=True,
                capture_output=True,
                text=True,
                timeout=7200,
            )

            output = bench_result.stdout + bench_result.stderr
            result["metrics"] = parse_metrics(output)
            result["raw_output"] = output

            if result["metrics"]:
                result["status"] = "success"
            else:
                result["error"] = "No metrics parsed from output"

    except subprocess.TimeoutExpired:
        result["error"] = "Benchmark timed out"
    except Exception as e:
        result["error"] = str(e)
    finally:
        result["duration_s"] = time.time() - start_time
        model_cache.commit()

    return result


@app.function(
    image=base_image,
    gpu="H100:8",
    timeout=14400,
    secrets=[modal.Secret.from_name("huggingface-secret")],
    volumes={"/root/.cache/huggingface": model_cache},
)
def run_benchmark_8gpu(
    wheel_url: str,
    perf_command: str,
    model: str,
    benchmark_type: str = "serving",
) -> Dict[str, Any]:
    """Run benchmark on 8x H100 GPUs for massive models (340B-671B)."""
    result = {
        "status": "error",
        "metrics": {},
        "vllm_version": None,
        "error": None,
        "duration_s": 0,
        "gpu_config": "H100:8",
    }

    start_time = time.time()

    try:
        success, version = install_wheel(wheel_url)
        if not success:
            result["error"] = f"Wheel installation failed: {version}"
            return result

        result["vllm_version"] = version

        if benchmark_type == "serving":
            server = start_server(model, port=29000, tensor_parallel=8)

            try:
                if not wait_for_server(port=29000, timeout=3600):
                    result["error"] = "Server failed to start within timeout (check Modal dashboard logs for details)"
                    return result

                # Run built-in benchmark
                print(f"Server ready, running benchmark against {model} with 8x H100...")
                benchmark_metrics = run_simple_benchmark(model, port=29000, num_requests=50)

                result["metrics"] = benchmark_metrics
                result["raw_output"] = f"Benchmark completed: {benchmark_metrics}"

                if result["metrics"]:
                    result["status"] = "success"
                else:
                    result["error"] = "No metrics from benchmark"

            finally:
                stop_server(server)
        else:
            cmd = re.sub(r'--tensor-parallel-size\s+\d+', '--tensor-parallel-size 8', perf_command)
            if '--tensor-parallel-size' not in cmd:
                cmd += ' --tensor-parallel-size 8'

            bench_result = subprocess.run(
                cmd,
                shell=True,
                capture_output=True,
                text=True,
                timeout=14400,
            )

            output = bench_result.stdout + bench_result.stderr
            result["metrics"] = parse_metrics(output)
            result["raw_output"] = output

            if result["metrics"]:
                result["status"] = "success"
            else:
                result["error"] = "No metrics parsed from output"

    except subprocess.TimeoutExpired:
        result["error"] = "Benchmark timed out"
    except Exception as e:
        result["error"] = str(e)
    finally:
        result["duration_s"] = time.time() - start_time
        model_cache.commit()

    return result


# =====================================================================
# 3-WAY BENCHMARK FUNCTIONS
# Run baseline vs human vs agent comparison with actual perf_command
# =====================================================================

def run_perf_command(perf_command: str, model: str, port: int = 8000, commit: Optional[str] = None) -> Tuple[str, Dict[str, float]]:
    """
    Run the actual benchmark command from the dataset.

    Uses Python benchmark scripts (benchmark_serving.py, etc.) for compatibility
    with all vLLM versions.

    Args:
        perf_command: The benchmark command to run
        model: Model name
        port: Server port for serving benchmarks
        commit: Optional commit hash to use commit-specific benchmark scripts
    """
    # Translate to appropriate benchmark command (vllm bench CLI or commit-specific scripts)
    cmd = translate_benchmark_command(perf_command, commit=commit)

    # Determine if this is a vllm CLI command or a python script
    is_vllm_cli = cmd.strip().startswith("vllm ")

    # For serving benchmarks, need to add host/port
    is_serving_cmd = "benchmark_serving" in cmd or "bench serve" in cmd
    if is_serving_cmd:
        # Remove server-only args that don't belong in benchmark client
        cmd = re.sub(r'--dtype\s+\S+', '', cmd)
        cmd = re.sub(r'--tensor-parallel-size\s+\d+', '', cmd)
        cmd = re.sub(r'-tp\s+\d+', '', cmd)
        cmd = re.sub(r'--trust-remote-code', '', cmd)
        cmd = re.sub(r'--max-model-len\s+\d+', '', cmd)

        # Remove args that don't exist in old vLLM versions (0.5.x, early 0.6.x)
        # These cause "unrecognized arguments" errors with old benchmark_serving.py
        cmd = re.sub(r'--enable-prefix-caching', '', cmd)
        cmd = re.sub(r'--use-v2-block-manager', '', cmd)
        cmd = re.sub(r'--kv-cache-dtype\s+\S+', '', cmd)
        cmd = re.sub(r'--enable-chunked-prefill', '', cmd)
        cmd = re.sub(r'--speculative-model\s+\S+', '', cmd)
        cmd = re.sub(r'--num-speculative-tokens\s+\d+', '', cmd)
        cmd = re.sub(r'--gpu-memory-utilization\s+[\d.]+', '', cmd)
        cmd = re.sub(r'--enforce-eager', '', cmd)
        # --backend was added in later vLLM versions, old benchmark_serving.py doesn't have it
        cmd = re.sub(r'--backend\s+\S+', '', cmd)

        # Add host/port
        if '--host' not in cmd and '--base-url' not in cmd:
            cmd += ' --host 127.0.0.1'
        if '--port' not in cmd and '--base-url' not in cmd:
            cmd += f' --port {port}'

        # Add dataset if not specified (required for serving benchmarks)
        if '--dataset-name' not in cmd and '--dataset-path' not in cmd:
            cmd += ' --dataset-name random --random-input-len 512 --random-output-len 128'

        # Add num-prompts if not specified
        if '--num-prompts' not in cmd:
            cmd += ' --num-prompts 100'

    # Clean up multiple spaces
    cmd = re.sub(r'\s+', ' ', cmd).strip()

    print(f"Running benchmark: {cmd[:200]}...")

    # Determine working directory
    # vllm CLI: run from /tmp to avoid importing local vllm/ directory
    # python scripts: run from /tmp, scripts are called with absolute paths
    work_dir = "/tmp"

    try:
        result = subprocess.run(
            cmd,
            shell=True,
            capture_output=True,
            text=True,
            timeout=1800,  # 30 min timeout for benchmark
            cwd=work_dir,
        )

        output = result.stdout + "\n" + result.stderr
        print(f"Benchmark stdout (last 500 chars): {result.stdout[-500:]}")
        print(f"Benchmark stderr (last 500 chars): {result.stderr[-500:]}")
        metrics = parse_metrics(output)

        return output, metrics

    except subprocess.TimeoutExpired:
        return "Benchmark timed out", {}
    except Exception as e:
        return f"Error: {str(e)}", {}


def run_standalone_benchmark(perf_command: str, model: str, tensor_parallel: int = 1, commit: Optional[str] = None) -> Tuple[str, Dict[str, float]]:
    """
    Run a standalone benchmark (throughput/latency) that doesn't need a server.

    These benchmarks load and run the model directly.

    Args:
        perf_command: The benchmark command to run
        model: Model name
        tensor_parallel: Number of GPUs for tensor parallelism
        commit: Optional commit hash to use commit-specific benchmark scripts
    """
    # Translate to appropriate benchmark command (vllm bench CLI or commit-specific scripts)
    cmd = translate_benchmark_command(perf_command, commit=commit)

    # Determine if this is a vllm CLI command or a python script
    is_vllm_cli = cmd.strip().startswith("vllm ")

    # Ensure tensor parallel is set correctly
    if tensor_parallel > 1:
        if '--tensor-parallel-size' not in cmd and '-tp' not in cmd:
            cmd += f' --tensor-parallel-size {tensor_parallel}'

    # Clean up multiple spaces
    cmd = re.sub(r'\s+', ' ', cmd).strip()

    print(f"Running standalone benchmark: {cmd[:200]}...")

    # Set CUDA devices
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = ",".join(str(i) for i in range(tensor_parallel))

    # Always run from /tmp to avoid importing local vllm/ directory
    # Benchmark scripts are called with absolute paths now
    work_dir = "/tmp"

    try:
        result = subprocess.run(
            cmd,
            shell=True,
            capture_output=True,
            text=True,
            timeout=3600,  # 1 hour timeout for standalone benchmarks
            cwd=work_dir,
            env=env,
        )

        output = result.stdout + "\n" + result.stderr
        print(f"Benchmark stdout (last 500 chars): {result.stdout[-500:]}")
        print(f"Benchmark stderr (last 500 chars): {result.stderr[-500:]}")

        metrics = parse_metrics(output)
        return output, metrics

    except subprocess.TimeoutExpired:
        return "Benchmark timed out", {}
    except Exception as e:
        return f"Error: {str(e)}", {}


def filter_patch_python_only(patch_content: str) -> Tuple[str, list, list]:
    """
    Filter a unified diff patch to only include Python file changes.

    Returns:
        (filtered_patch, python_files, skipped_files) tuple
    """
    c_extensions = ('.c', '.cpp', '.cu', '.cuh', '.h', '.hpp', '.cc', '.hh')

    # Split patch into individual file diffs
    # Each diff starts with "diff --git"
    diffs = re.split(r'(?=^diff --git )', patch_content, flags=re.MULTILINE)

    python_diffs = []
    python_files = []
    skipped_files = []

    for diff in diffs:
        if not diff.strip():
            continue

        # Extract file path from "diff --git a/path b/path" line
        match = re.search(r'^diff --git a/(\S+) b/(\S+)', diff, re.MULTILINE)
        if not match:
            continue

        file_path = match.group(2)  # Use b/ path (destination)

        # Check if this is a Python file in vllm/
        if file_path.endswith('.py') and file_path.startswith('vllm/'):
            python_diffs.append(diff)
            python_files.append(file_path)
        elif file_path.endswith(c_extensions):
            skipped_files.append(file_path)
        elif not file_path.startswith('vllm/'):
            # Skip non-vllm files (tests, benchmarks, etc.)
            skipped_files.append(file_path)
        else:
            # Other files in vllm/ that aren't Python or C/CUDA
            skipped_files.append(file_path)

    filtered_patch = ''.join(python_diffs)
    return filtered_patch, python_files, skipped_files


def apply_patch_to_vllm(patch_content: str) -> Tuple[bool, str]:
    """
    Apply a patch to the installed vLLM package.
    Only applies Python file changes, skips C/CUDA extensions.

    Returns:
        (success, message) tuple
    """
    import site
    import tempfile

    # Filter patch to only Python files in vllm/
    filtered_patch, python_files, skipped_files = filter_patch_python_only(patch_content)

    if skipped_files:
        print(f"Skipping non-Python files: {skipped_files}")

    if not filtered_patch.strip():
        if skipped_files:
            return False, f"Patch contains only non-Python files (skipped: {skipped_files})"
        return False, "Patch is empty or contains no vllm/ Python changes"

    print(f"Applying patch to Python files: {python_files}")

    # Find vLLM in site-packages
    vllm_path = None
    for sp in site.getsitepackages():
        candidate = Path(sp) / "vllm"
        if candidate.exists():
            vllm_path = Path(sp)
            break

    if not vllm_path:
        return False, "Could not find vLLM in site-packages"

    # Write patch to temp file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.patch', delete=False) as f:
        f.write(filtered_patch)
        patch_file = f.name

    try:
        # Rewrite paths: a/vllm/... -> site-packages/vllm/...
        modified_patch = filtered_patch

        # Find all vllm/ paths in the patch and make them absolute
        for rel_path in python_files:
            # Strip leading vllm/ since we'll add it back with full path
            inner_path = rel_path[5:] if rel_path.startswith('vllm/') else rel_path
            old_a = f"a/vllm/{inner_path}"
            old_b = f"b/vllm/{inner_path}"
            new_path = str(vllm_path / "vllm" / inner_path)
            modified_patch = modified_patch.replace(old_a, new_path)
            modified_patch = modified_patch.replace(old_b, new_path)

        # Write modified patch
        with open(patch_file, 'w') as f:
            f.write(modified_patch)

        # Apply with patch command
        result = subprocess.run(
            ["patch", "-p0", "--forward", "--ignore-whitespace", "-i", patch_file],
            capture_output=True,
            text=True,
            cwd="/",
        )

        if result.returncode == 0:
            msg = f"Patch applied successfully to {vllm_path} ({len(python_files)} Python files)"
            if skipped_files:
                msg += f", skipped {len(skipped_files)} non-Python files"
            return True, msg
        else:
            # Try with --dry-run first to see what would happen
            return False, f"Patch failed: {result.stderr}\nstdout: {result.stdout}"

    finally:
        try:
            os.unlink(patch_file)
        except:
            pass


def compute_improvement(baseline: Dict[str, float], other: Dict[str, float]) -> Dict[str, float]:
    """Compute percentage improvement for each metric."""
    improvement = {}

    # Throughput metrics (higher is better)
    for key in ["request_throughput", "output_throughput", "total_throughput", "throughput"]:
        b = baseline.get(key)
        o = other.get(key)
        if b and o and b > 0:
            improvement[key] = ((o - b) / b) * 100

    # Latency metrics (lower is better, so invert)
    for key in ["ttft_mean", "ttft_median", "ttft_p99", "tpot_mean", "itl_mean", "latency_avg"]:
        b = baseline.get(key)
        o = other.get(key)
        if b and o and b > 0:
            improvement[key] = ((b - o) / b) * 100

    return improvement


@app.function(
    image=base_image,
    gpu="H100:4",
    timeout=10800,  # 3 hours for 3 benchmarks
    secrets=[modal.Secret.from_name("huggingface-secret")],
    volumes={"/root/.cache/huggingface": model_cache},
)
def run_3way_benchmark_4gpu(
    baseline_wheel_url: str,
    human_wheel_url: str,
    agent_patch: Optional[str],
    perf_command: str,
    model: str,
) -> Dict[str, Any]:
    """
    Run 3-way benchmark comparison on 4x H100 GPUs.

    Handles both:
    - Serving benchmarks: Starts vLLM server, runs benchmark client
    - Standalone benchmarks: Runs benchmark directly (throughput/latency)

    1. Baseline wheel → metrics
    2. Human wheel → metrics
    3. Baseline wheel + agent patch → metrics (if Python-only)
    """
    result = {
        "status": "error",
        "gpu_config": "H100:4",
        "baseline_metrics": {},
        "human_metrics": {},
        "agent_metrics": None,
        "human_improvement": {},
        "agent_improvement": None,
        "agent_vs_human": None,
        "error": None,
        "duration_s": 0,
        "perf_command": perf_command,
    }

    start_time = time.time()
    tensor_parallel = 4

    # Extract commits from wheel URLs for benchmark script compatibility
    baseline_commit = extract_commit_from_wheel_url(baseline_wheel_url)
    human_commit = extract_commit_from_wheel_url(human_wheel_url)
    print(f"Extracted commits - baseline: {baseline_commit[:8] if baseline_commit else 'None'}, human: {human_commit[:8] if human_commit else 'None'}")

    # Determine if this is a serving or standalone benchmark
    needs_server = is_serving_benchmark(perf_command)
    result["benchmark_mode"] = "serving" if needs_server else "standalone"
    print(f"Benchmark mode: {result['benchmark_mode']}")

    def run_benchmark_phase(phase_name: str, commit_for_scripts: Optional[str] = None) -> Tuple[str, Dict[str, float]]:
        """Run benchmark (serving or standalone) for a phase."""
        if needs_server:
            print(f"  Starting server for {phase_name} benchmark...")
            server = start_server(model, port=29000, tensor_parallel=tensor_parallel)
            try:
                if not wait_for_server(port=29000, timeout=3600):
                    raise RuntimeError(f"{phase_name} server failed to start")
                print(f"  Running {phase_name} serving benchmark...")
                return run_perf_command(perf_command, model, port=29000, commit=commit_for_scripts)
            finally:
                stop_server(server)
        else:
            print(f"  Running {phase_name} standalone benchmark...")
            return run_standalone_benchmark(perf_command, model, tensor_parallel, commit=commit_for_scripts)

    try:
        # ========== 1. BASELINE BENCHMARK ==========
        print(f"[1/3] Installing baseline wheel...")
        success, version = install_wheel(baseline_wheel_url)
        if not success:
            result["error"] = f"Baseline wheel install failed: {version}"
            return result

        result["baseline_version"] = version
        print(f"Baseline vLLM: {version}")

        try:
            baseline_output, baseline_metrics = run_benchmark_phase("BASELINE", baseline_commit)
            result["baseline_metrics"] = baseline_metrics
            result["baseline_raw"] = baseline_output

            if not baseline_metrics:
                result["error"] = "Baseline benchmark produced no metrics"
                result["status"] = "baseline_failed"
                return result

            print(f"Baseline metrics: {baseline_metrics}")
        except RuntimeError as e:
            result["error"] = str(e)
            result["status"] = "baseline_failed"
            return result

        # ========== 2. HUMAN BENCHMARK ==========
        print(f"[2/3] Installing human wheel...")
        success, version = install_wheel(human_wheel_url)
        if not success:
            result["error"] = f"Human wheel install failed: {version}"
            result["status"] = "human_wheel_failed"
            return result

        result["human_version"] = version
        print(f"Human vLLM: {version}")

        try:
            human_output, human_metrics = run_benchmark_phase("HUMAN", human_commit)
            result["human_metrics"] = human_metrics
            result["human_raw"] = human_output

            if not human_metrics:
                result["error"] = "Human benchmark produced no metrics"
                result["status"] = "human_failed"
                return result

            print(f"Human metrics: {human_metrics}")
            result["human_improvement"] = compute_improvement(baseline_metrics, human_metrics)
            print(f"Human improvement: {result['human_improvement']}")
        except RuntimeError as e:
            result["error"] = str(e)
            result["status"] = "human_failed"
            return result

        # ========== 3. AGENT BENCHMARK ==========
        if agent_patch:
            print(f"[3/3] Setting up AGENT benchmark...")

            # Reinstall baseline wheel
            success, version = install_wheel(baseline_wheel_url)
            if not success:
                print(f"Warning: Could not reinstall baseline for agent test")
                result["agent_metrics"] = None
                result["agent_error"] = "Could not reinstall baseline wheel"
            else:
                # Apply patch
                patch_success, patch_msg = apply_patch_to_vllm(agent_patch)

                if not patch_success:
                    print(f"Warning: {patch_msg}")
                    result["agent_metrics"] = None
                    result["agent_error"] = patch_msg
                else:
                    print(f"Patch applied: {patch_msg}")

                    try:
                        agent_output, agent_metrics = run_benchmark_phase("AGENT", baseline_commit)
                        result["agent_metrics"] = agent_metrics
                        result["agent_raw"] = agent_output

                        if agent_metrics:
                            result["agent_improvement"] = compute_improvement(baseline_metrics, agent_metrics)
                            result["agent_vs_human"] = compute_improvement(human_metrics, agent_metrics)
                            print(f"Agent metrics: {agent_metrics}")
                            print(f"Agent improvement: {result['agent_improvement']}")
                        else:
                            error_snippet = agent_output[-2000:] if agent_output else "no output"
                            result["agent_error"] = f"Agent benchmark produced no metrics. Output tail: {error_snippet}"
                            print(f"Agent benchmark failed - output tail:\n{error_snippet}")
                    except RuntimeError as e:
                        result["agent_error"] = str(e)
        else:
            print(f"[3/3] No agent patch provided, skipping")
            result["agent_metrics"] = None

        result["status"] = "success"

    except Exception as e:
        result["error"] = str(e)
        result["status"] = "error"
    finally:
        result["duration_s"] = time.time() - start_time
        model_cache.commit()

    return result


@app.function(
    image=base_image,
    gpu="H100:2",
    timeout=10800,
    secrets=[modal.Secret.from_name("huggingface-secret")],
    volumes={"/root/.cache/huggingface": model_cache},
)
def run_3way_benchmark_2gpu(
    baseline_wheel_url: str,
    human_wheel_url: str,
    agent_patch: Optional[str],
    perf_command: str,
    model: str,
) -> Dict[str, Any]:
    """Run 3-way benchmark on 2x H100 GPUs.

    Handles both:
    - Serving benchmarks: Starts vLLM server, runs benchmark client
    - Standalone benchmarks: Runs benchmark directly (throughput/latency)
    """
    result = {
        "status": "error",
        "gpu_config": "H100:2",
        "baseline_metrics": {},
        "human_metrics": {},
        "agent_metrics": None,
        "human_improvement": {},
        "agent_improvement": None,
        "agent_vs_human": None,
        "error": None,
        "duration_s": 0,
        "perf_command": perf_command,
    }

    start_time = time.time()
    tensor_parallel = 2

    # Extract commits from wheel URLs for benchmark script compatibility
    baseline_commit = extract_commit_from_wheel_url(baseline_wheel_url)
    human_commit = extract_commit_from_wheel_url(human_wheel_url)
    print(f"Extracted commits - baseline: {baseline_commit[:8] if baseline_commit else 'None'}, human: {human_commit[:8] if human_commit else 'None'}")

    # Determine if this is a serving or standalone benchmark
    needs_server = is_serving_benchmark(perf_command)
    result["benchmark_mode"] = "serving" if needs_server else "standalone"
    print(f"Benchmark mode: {result['benchmark_mode']}")

    def run_benchmark_phase(phase_name: str, commit_for_scripts: Optional[str] = None) -> Tuple[str, Dict[str, float]]:
        """Run benchmark (serving or standalone) for a phase."""
        if needs_server:
            print(f"  Starting server for {phase_name} benchmark...")
            server = start_server(model, port=29000, tensor_parallel=tensor_parallel)
            try:
                if not wait_for_server(port=29000, timeout=3600):
                    raise RuntimeError(f"{phase_name} server failed to start")
                print(f"  Running {phase_name} serving benchmark...")
                return run_perf_command(perf_command, model, port=29000, commit=commit_for_scripts)
            finally:
                stop_server(server)
        else:
            print(f"  Running {phase_name} standalone benchmark...")
            return run_standalone_benchmark(perf_command, model, tensor_parallel, commit=commit_for_scripts)

    try:
        # BASELINE
        print(f"[1/3] Installing baseline wheel...")
        success, version = install_wheel(baseline_wheel_url)
        if not success:
            result["error"] = f"Baseline wheel install failed: {version}"
            return result

        result["baseline_version"] = version
        try:
            baseline_output, baseline_metrics = run_benchmark_phase("BASELINE", baseline_commit)
            result["baseline_metrics"] = baseline_metrics

            if not baseline_metrics:
                result["error"] = "Baseline benchmark produced no metrics"
                result["status"] = "baseline_failed"
                return result
        except RuntimeError as e:
            result["error"] = str(e)
            result["status"] = "baseline_failed"
            return result

        # HUMAN
        print(f"[2/3] Installing human wheel...")
        success, version = install_wheel(human_wheel_url)
        if not success:
            result["error"] = f"Human wheel install failed: {version}"
            return result

        result["human_version"] = version
        try:
            human_output, human_metrics = run_benchmark_phase("HUMAN", human_commit)
            result["human_metrics"] = human_metrics

            if not human_metrics:
                result["error"] = "Human benchmark produced no metrics"
                result["status"] = "human_failed"
                return result

            result["human_improvement"] = compute_improvement(baseline_metrics, human_metrics)
        except RuntimeError as e:
            result["error"] = str(e)
            result["status"] = "human_failed"
            return result

        # AGENT
        if agent_patch:
            success, version = install_wheel(baseline_wheel_url)
            if success:
                patch_success, patch_msg = apply_patch_to_vllm(agent_patch)
                if patch_success:
                    try:
                        agent_output, agent_metrics = run_benchmark_phase("AGENT", baseline_commit)
                        result["agent_metrics"] = agent_metrics
                        if agent_metrics:
                            result["agent_improvement"] = compute_improvement(baseline_metrics, agent_metrics)
                            result["agent_vs_human"] = compute_improvement(human_metrics, agent_metrics)
                    except RuntimeError as e:
                        result["agent_error"] = str(e)
                else:
                    result["agent_error"] = patch_msg

        result["status"] = "success"

    except Exception as e:
        result["error"] = str(e)
    finally:
        result["duration_s"] = time.time() - start_time
        model_cache.commit()

    return result


@app.function(
    image=base_image,
    gpu="H100:8",
    timeout=14400,  # 4 hours
    secrets=[modal.Secret.from_name("huggingface-secret")],
    volumes={"/root/.cache/huggingface": model_cache},
)
def run_3way_benchmark_8gpu(
    baseline_wheel_url: str,
    human_wheel_url: str,
    agent_patch: Optional[str],
    perf_command: str,
    model: str,
) -> Dict[str, Any]:
    """Run 3-way benchmark on 8x H100 GPUs.

    Handles both:
    - Serving benchmarks: Starts vLLM server, runs benchmark client
    - Standalone benchmarks: Runs benchmark directly (throughput/latency)
    """
    result = {
        "status": "error",
        "gpu_config": "H100:8",
        "baseline_metrics": {},
        "human_metrics": {},
        "agent_metrics": None,
        "human_improvement": {},
        "agent_improvement": None,
        "agent_vs_human": None,
        "error": None,
        "duration_s": 0,
        "perf_command": perf_command,
    }

    start_time = time.time()
    tensor_parallel = 8

    # Extract commits from wheel URLs for benchmark script compatibility
    baseline_commit = extract_commit_from_wheel_url(baseline_wheel_url)
    human_commit = extract_commit_from_wheel_url(human_wheel_url)
    print(f"Extracted commits - baseline: {baseline_commit[:8] if baseline_commit else 'None'}, human: {human_commit[:8] if human_commit else 'None'}")

    # Determine if this is a serving or standalone benchmark
    needs_server = is_serving_benchmark(perf_command)
    result["benchmark_mode"] = "serving" if needs_server else "standalone"
    print(f"Benchmark mode: {result['benchmark_mode']}")

    def run_benchmark_phase(phase_name: str, commit_for_scripts: Optional[str] = None) -> Tuple[str, Dict[str, float]]:
        """Run benchmark (serving or standalone) for a phase."""
        if needs_server:
            print(f"  Starting server for {phase_name} benchmark...")
            server = start_server(model, port=29000, tensor_parallel=tensor_parallel)
            try:
                if not wait_for_server(port=29000, timeout=3600):
                    raise RuntimeError(f"{phase_name} server failed to start")
                print(f"  Running {phase_name} serving benchmark...")
                return run_perf_command(perf_command, model, port=29000, commit=commit_for_scripts)
            finally:
                stop_server(server)
        else:
            print(f"  Running {phase_name} standalone benchmark...")
            return run_standalone_benchmark(perf_command, model, tensor_parallel, commit=commit_for_scripts)

    try:
        # BASELINE
        print(f"[1/3] Installing baseline wheel...")
        success, version = install_wheel(baseline_wheel_url)
        if not success:
            result["error"] = f"Baseline wheel install failed: {version}"
            return result

        result["baseline_version"] = version
        try:
            baseline_output, baseline_metrics = run_benchmark_phase("BASELINE", baseline_commit)
            result["baseline_metrics"] = baseline_metrics

            if not baseline_metrics:
                result["error"] = "Baseline benchmark produced no metrics"
                result["status"] = "baseline_failed"
                return result
        except RuntimeError as e:
            result["error"] = str(e)
            result["status"] = "baseline_failed"
            return result

        # HUMAN
        print(f"[2/3] Installing human wheel...")
        success, version = install_wheel(human_wheel_url)
        if not success:
            result["error"] = f"Human wheel install failed: {version}"
            return result

        result["human_version"] = version
        try:
            human_output, human_metrics = run_benchmark_phase("HUMAN", human_commit)
            result["human_metrics"] = human_metrics

            if not human_metrics:
                result["error"] = "Human benchmark produced no metrics"
                result["status"] = "human_failed"
                return result

            result["human_improvement"] = compute_improvement(baseline_metrics, human_metrics)
        except RuntimeError as e:
            result["error"] = str(e)
            result["status"] = "human_failed"
            return result

        # AGENT
        if agent_patch:
            success, version = install_wheel(baseline_wheel_url)
            if success:
                patch_success, patch_msg = apply_patch_to_vllm(agent_patch)
                if patch_success:
                    try:
                        agent_output, agent_metrics = run_benchmark_phase("AGENT", baseline_commit)
                        result["agent_metrics"] = agent_metrics
                        if agent_metrics:
                            result["agent_improvement"] = compute_improvement(baseline_metrics, agent_metrics)
                            result["agent_vs_human"] = compute_improvement(human_metrics, agent_metrics)
                    except RuntimeError as e:
                        result["agent_error"] = str(e)
                else:
                    result["agent_error"] = patch_msg

        result["status"] = "success"

    except Exception as e:
        result["error"] = str(e)
    finally:
        result["duration_s"] = time.time() - start_time
        model_cache.commit()

    return result


@app.function(
    image=base_image,
    gpu="H100",  # Single H100
    timeout=14400,  # 4 hours (increased for build-from-source)
    secrets=[modal.Secret.from_name("huggingface-secret")],
    volumes={
        "/root/.cache/huggingface": model_cache,
        "/cache": build_cache,  # Build cache for incremental C/CUDA compilation
    },
)
def run_3way_benchmark_1gpu(
    baseline_wheel_url: str,
    human_wheel_url: str,
    agent_patch: Optional[str],
    perf_command: str,
    model: str,
    base_commit: Optional[str] = None,
    human_commit: Optional[str] = None,
    skip_cuda_agent_build: bool = True,  # Skip slow C/CUDA builds by default
    agent_wheel_path: Optional[str] = None,  # Pre-built agent wheel from CPU (NEW)
) -> Dict[str, Any]:
    """Run 3-way benchmark on single H100 GPU.

    NEW ARCHITECTURE (v2): NO COMPILATION ON GPU
    - All wheels are pre-built on CPU and passed via Modal volume
    - This function ONLY installs wheels and runs benchmarks
    - Much faster and cheaper than compiling on H100

    Handles both:
    - Serving benchmarks: Starts vLLM server, runs benchmark client
    - Standalone benchmarks: Runs benchmark directly (throughput/latency)

    For commits without pre-built wheels:
    - Uses Python overlay: Install nearest ancestor wheel + overlay Python files from git checkout
    - Only works for Python-only commits (C/CUDA changes require full build)

    For agent benchmarks:
    - If agent_wheel_path provided: Install pre-built wheel from volume (fast!)
    - If only agent_patch provided: Apply Python overlay (Python-only patches)
    """
    result = {
        "status": "error",
        "gpu_config": "H100:1",
        "baseline_metrics": {},
        "human_metrics": {},
        "agent_metrics": None,
        "human_improvement": {},
        "agent_improvement": None,
        "agent_vs_human": None,
        "error": None,
        "server_logs": None,
        "duration_s": 0,
        "perf_command": perf_command,
    }

    start_time = time.time()
    tensor_parallel = 1
    server_log_file = "/tmp/vllm_server.log"

    # CRITICAL: Clean up any leftover processes at container start
    # Modal reuses containers, so previous crashed runs might leave zombie processes
    print("[INIT] Aggressive cleanup at function start...")
    try:
        subprocess.run(["pkill", "-9", "-f", "vllm"], capture_output=True, timeout=5)
        subprocess.run(["pkill", "-9", "-f", "uvicorn"], capture_output=True, timeout=5)
        subprocess.run(["pkill", "-9", "-f", "ray::"], capture_output=True, timeout=5)
        # Clean all phase ports (18001-18003 for HTTP, 29xxx and 30xxx for vLLM internal)
        for port in [8000, 18001, 18002, 18003, 29000, 29001, 29002, 29003, 29500, 30010, 30020, 30030]:
            subprocess.run(["fuser", "-k", "-9", f"{port}/tcp"], capture_output=True, timeout=5)
        time.sleep(3)
        # Check what's listening on phase ports
        for port in [18001, 18002, 18003]:
            port_check = subprocess.run(["lsof", "-i", f":{port}"], capture_output=True, text=True, timeout=5)
            if port_check.stdout.strip():
                print(f"[INIT] WARNING: Port {port} still in use after cleanup: {port_check.stdout.strip()[:100]}")
            else:
                print(f"[INIT] Port {port} is clear")
    except Exception as e:
        print(f"[INIT] Cleanup warning: {e}")

    # Determine if this is a serving or standalone benchmark
    needs_server = is_serving_benchmark(perf_command)
    result["benchmark_mode"] = "serving" if needs_server else "standalone"
    print(f"Benchmark mode: {result['benchmark_mode']}")

    # Check if patch contains C/CUDA changes (determines benchmark order)
    has_c_cuda_changes = False
    if agent_patch:
        c_extensions = ('.c', '.cpp', '.cu', '.cuh', '.h', '.hpp', '.cc', '.hh')
        patch_files = re.findall(r'^diff --git a/(\S+) b/\S+', agent_patch, re.MULTILINE)
        has_c_cuda_changes = any(f.endswith(c_extensions) for f in patch_files)
        if has_c_cuda_changes:
            print(f"C/CUDA patch detected - using optimized order: Baseline → Agent → Human")
            result["patch_type"] = "c_cuda"
        else:
            print(f"Python-only patch detected - using standard order: Baseline → Human → Agent")
            result["patch_type"] = "python_only"

    # Track current commit for benchmark script checkout
    current_benchmark_commit = base_commit  # Start with baseline commit

    # Use unique ports per phase to avoid port conflicts from incomplete cleanup
    # vLLM 0.6.x has a bug where internal processes bind to same ports as HTTP server
    # Using ports far from 29xxx range to avoid collision with vLLM internal ports
    # See: https://github.com/vllm-project/vllm/issues/8791
    PHASE_PORTS = {
        "BASELINE": 18001,
        "HUMAN": 18002,
        "AGENT": 18003,
    }

    def run_benchmark_phase(phase_name: str, commit_for_scripts: Optional[str] = None) -> Tuple[str, Dict[str, float]]:
        """Helper to run benchmark (serving or standalone).

        Args:
            phase_name: Name of the benchmark phase (BASELINE, HUMAN, AGENT)
            commit_for_scripts: Commit hash to use for benchmark scripts (for older vLLM)
        """
        script_commit = commit_for_scripts or current_benchmark_commit
        # Use phase-specific port to avoid port conflicts
        port = PHASE_PORTS.get(phase_name, 29000)
        if needs_server:
            print(f"  Starting server for {phase_name} benchmark on port {port}...")
            server = start_server(model, port=port, tensor_parallel=tensor_parallel, log_file=server_log_file)
            try:
                if not wait_for_server(port=port, timeout=600):
                    server_logs = get_server_logs(server)
                    raise RuntimeError(f"{phase_name} server failed to start. Logs: {server_logs[-2000:] if server_logs else 'none'}")
                print(f"  Running {phase_name} serving benchmark...")
                return run_perf_command(perf_command, model, port=port, commit=script_commit)
            finally:
                stop_server(server, port=port)
        else:
            print(f"  Running {phase_name} standalone benchmark...")
            return run_standalone_benchmark(perf_command, model, tensor_parallel, commit=script_commit)

    try:
        # ========== BASELINE (always first) ==========
        print(f"[1/3] Installing baseline vLLM...")

        # Try wheel URL first if provided
        baseline_success = False
        if baseline_wheel_url:
            success, version = install_wheel(baseline_wheel_url)
            if success:
                baseline_success = True
                result["baseline_version"] = version
                result["baseline_install_method"] = "wheel"
                print(f"Baseline vLLM: {version} (from wheel)")

        # If wheel failed and we have a commit, try overlay approach
        if not baseline_success and base_commit:
            print(f"  Wheel not available, trying Python overlay for {base_commit[:12]}...")
            success, version, method = install_vllm_for_commit(base_commit)
            if success:
                baseline_success = True
                result["baseline_version"] = version
                result["baseline_install_method"] = method
                print(f"Baseline vLLM: {version} (via {method})")
            else:
                result["error"] = f"Baseline install failed: {version}"
                return result

        if not baseline_success:
            result["error"] = f"Baseline wheel install failed and no commit provided for overlay"
            return result

        # Check for vLLM 0.6.0-0.6.5 which have a known port binding bug in serving mode
        # See: https://github.com/vllm-project/vllm/issues/8791
        # The bug causes internal worker processes to bind to the same port as HTTP server
        # This cannot be fixed with port changes or environment variables
        version = result.get('baseline_version', '')
        if needs_server and version:
            # Parse version string like "0.6.3.post2.dev398+g4a18fd14"
            # Note: re is already imported at module level
            import re as re_mod
            version_match = re_mod.search(r'0\.6\.([0-5])\.', version)
            if version_match:
                minor = int(version_match.group(1))
                print(f"[VERSION CHECK] vLLM {version} has known port binding bug in serving mode")
                print(f"[VERSION CHECK] Serving benchmarks are not supported for vLLM 0.6.0-0.6.5")
                result["error"] = f"vLLM {version} has known port binding bug (issue #8791) - serving benchmarks not supported"
                result["status"] = "version_bug"
                return result
            else:
                print(f"[VERSION CHECK] vLLM {version} - proceeding with serving benchmark")

        baseline_output, baseline_metrics = run_benchmark_phase("BASELINE", base_commit)
        result["baseline_metrics"] = baseline_metrics
        result["baseline_raw"] = baseline_output

        if not baseline_metrics:
            result["error"] = "Baseline benchmark produced no metrics"
            result["status"] = "baseline_failed"
            return result

        print(f"Baseline metrics: {baseline_metrics}")

        # ========== C/CUDA PATH: Baseline → Agent → Human ==========
        # NEW: Use pre-built wheel from CPU if available (no GPU compilation!)
        if has_c_cuda_changes and agent_patch and base_commit:
            # AGENT benchmark - use pre-built wheel from volume
            if agent_wheel_path:
                print(f"[2/3] Installing AGENT from pre-built wheel (no GPU compilation!)...")
                print(f"  Wheel path: {agent_wheel_path}")

                # Install the pre-built wheel from volume
                install_result = subprocess.run(
                    ["uv", "pip", "install", "--system", "--force-reinstall", agent_wheel_path],
                    capture_output=True, text=True, timeout=300,
                )

                if install_result.returncode != 0:
                    print(f"Warning: Agent wheel install failed: {install_result.stderr[-500:]}")
                    result["agent_error"] = f"Agent wheel install failed: {install_result.stderr[-500:]}"
                else:
                    # Verify installation
                    version_result = subprocess.run(
                        ["python", "-c", "import vllm; print(vllm.__version__)"],
                        capture_output=True, text=True, cwd="/tmp",
                    )
                    if version_result.returncode == 0:
                        result["agent_version"] = version_result.stdout.strip()
                        print(f"Agent vLLM: {result['agent_version']} (from pre-built wheel)")

                        try:
                            agent_output, agent_metrics = run_benchmark_phase("AGENT", base_commit)
                            result["agent_metrics"] = agent_metrics
                            result["agent_raw"] = agent_output

                            if agent_metrics:
                                result["agent_improvement"] = compute_improvement(baseline_metrics, agent_metrics)
                                print(f"Agent metrics: {agent_metrics}")
                                print(f"Agent improvement vs baseline: {result['agent_improvement']}")
                            else:
                                error_snippet = agent_output[-2000:] if agent_output else "no output"
                                result["agent_error"] = f"Agent benchmark produced no metrics. Output tail: {error_snippet}"
                                print(f"Agent benchmark failed - output tail:\n{error_snippet}")
                        except RuntimeError as e:
                            result["agent_error"] = str(e)
                    else:
                        result["agent_error"] = f"vLLM import failed after wheel install: {version_result.stderr}"
            else:
                # FALLBACK: No pre-built wheel, try incremental build on GPU (slow!)
                print(f"[2/3] WARNING: No pre-built wheel, falling back to GPU build (slow!)...")
                build_success, build_msg = build_vllm_incremental(base_commit, agent_patch, cache_dir="/cache")

                if not build_success:
                    print(f"Warning: Incremental build failed: {build_msg}")
                    result["agent_error"] = f"Incremental build failed: {build_msg}"
                else:
                    print(f"Incremental build successful: {build_msg}")
                    result["agent_version"] = build_msg

                    try:
                        agent_output, agent_metrics = run_benchmark_phase("AGENT", base_commit)
                        result["agent_metrics"] = agent_metrics
                        result["agent_raw"] = agent_output

                        if agent_metrics:
                            result["agent_improvement"] = compute_improvement(baseline_metrics, agent_metrics)
                            print(f"Agent metrics: {agent_metrics}")
                            print(f"Agent improvement vs baseline: {result['agent_improvement']}")
                        else:
                            error_snippet = agent_output[-2000:] if agent_output else "no output"
                            result["agent_error"] = f"Agent benchmark produced no metrics. Output tail: {error_snippet}"
                            print(f"Agent benchmark failed - output tail:\n{error_snippet}")
                    except RuntimeError as e:
                        result["agent_error"] = str(e)

            # HUMAN benchmark (wheel or overlay install)
            print(f"[3/3] Installing human vLLM...")
            human_success = False
            if human_wheel_url:
                success, version = install_wheel(human_wheel_url)
                if success:
                    human_success = True
                    result["human_version"] = version
                    result["human_install_method"] = "wheel"
                    print(f"Human vLLM: {version} (from wheel)")

            if not human_success and human_commit:
                print(f"  Wheel not available, trying Python overlay for {human_commit[:12]}...")
                success, version, method = install_vllm_for_commit(human_commit)
                if success:
                    human_success = True
                    result["human_version"] = version
                    result["human_install_method"] = method
                    print(f"Human vLLM: {version} (via {method})")
                else:
                    result["error"] = f"Human install failed: {version}"
                    return result

            if not human_success:
                result["error"] = f"Human wheel install failed and no commit provided for overlay"
                return result

            try:
                human_output, human_metrics = run_benchmark_phase("HUMAN", human_commit)
                result["human_metrics"] = human_metrics
                result["human_raw"] = human_output

                if human_metrics:
                    result["human_improvement"] = compute_improvement(baseline_metrics, human_metrics)
                    print(f"Human metrics: {human_metrics}")
                    print(f"Human improvement: {result['human_improvement']}")

                    # Compute agent vs human if both succeeded
                    if result.get("agent_metrics"):
                        result["agent_vs_human"] = compute_improvement(human_metrics, result["agent_metrics"])
                        print(f"Agent vs Human: {result['agent_vs_human']}")
                else:
                    result["error"] = "Human benchmark produced no metrics"
                    result["status"] = "human_failed"
                    return result
            except RuntimeError as e:
                result["error"] = str(e)
                result["status"] = "human_failed"
                return result

        # ========== PYTHON-ONLY PATH: Baseline → Human → Agent ==========
        else:
            # HUMAN benchmark (wheel or overlay install)
            print(f"[2/3] Installing human vLLM...")
            human_success = False
            if human_wheel_url:
                success, version = install_wheel(human_wheel_url)
                if success:
                    human_success = True
                    result["human_version"] = version
                    result["human_install_method"] = "wheel"
                    print(f"Human vLLM: {version} (from wheel)")

            if not human_success and human_commit:
                print(f"  Wheel not available, trying Python overlay for {human_commit[:12]}...")
                success, version, method = install_vllm_for_commit(human_commit)
                if success:
                    human_success = True
                    result["human_version"] = version
                    result["human_install_method"] = method
                    print(f"Human vLLM: {version} (via {method})")
                else:
                    result["error"] = f"Human install failed: {version}"
                    return result

            if not human_success:
                result["error"] = f"Human wheel install failed and no commit provided for overlay"
                return result

            try:
                human_output, human_metrics = run_benchmark_phase("HUMAN", human_commit)
                result["human_metrics"] = human_metrics
                result["human_raw"] = human_output

                if not human_metrics:
                    result["error"] = "Human benchmark produced no metrics"
                    result["status"] = "human_failed"
                    return result

                result["human_improvement"] = compute_improvement(baseline_metrics, human_metrics)
                print(f"Human metrics: {human_metrics}")
                print(f"Human improvement: {result['human_improvement']}")
            except RuntimeError as e:
                result["error"] = str(e)
                result["status"] = "human_failed"
                return result

            # AGENT benchmark (baseline + patch)
            if agent_patch:
                if has_c_cuda_changes and not base_commit:
                    result["agent_error"] = "Patch contains C/CUDA changes but no base_commit provided"
                    print(f"Warning: {result['agent_error']}")
                # PRIORITY 1: Use pre-built agent wheel if available (most reliable!)
                elif agent_wheel_path:
                    print(f"[3/3] Installing AGENT from pre-built wheel...")
                    print(f"  Wheel path: {agent_wheel_path}")

                    install_result = subprocess.run(
                        ["uv", "pip", "install", "--system", "--force-reinstall", agent_wheel_path],
                        capture_output=True, text=True, timeout=300,
                    )

                    if install_result.returncode != 0:
                        print(f"Warning: Agent wheel install failed: {install_result.stderr[-500:]}")
                        result["agent_error"] = f"Agent wheel install failed: {install_result.stderr[-500:]}"
                    else:
                        # Verify installation
                        version_result = subprocess.run(
                            ["python", "-c", "import vllm; print(vllm.__version__)"],
                            capture_output=True, text=True, cwd="/tmp",
                        )
                        if version_result.returncode == 0:
                            result["agent_version"] = version_result.stdout.strip()
                            result["agent_install_method"] = "wheel"
                            print(f"Agent vLLM: {result['agent_version']} (from pre-built wheel)")

                            try:
                                agent_output, agent_metrics = run_benchmark_phase("AGENT", base_commit)
                                result["agent_metrics"] = agent_metrics
                                result["agent_raw"] = agent_output

                                if agent_metrics:
                                    result["agent_improvement"] = compute_improvement(baseline_metrics, agent_metrics)
                                    result["agent_vs_human"] = compute_improvement(human_metrics, agent_metrics)
                                    print(f"Agent metrics: {agent_metrics}")
                                    print(f"Agent improvement: {result['agent_improvement']}")
                                else:
                                    # Include actual output in error for debugging
                                    error_snippet = agent_output[-2000:] if agent_output else "no output"
                                    result["agent_error"] = f"Agent benchmark produced no metrics. Output tail: {error_snippet}"
                                    print(f"Agent benchmark failed - output tail:\n{error_snippet}")
                            except RuntimeError as e:
                                result["agent_error"] = str(e)
                        else:
                            result["agent_error"] = f"vLLM import failed after wheel install: {version_result.stderr}"
                # PRIORITY 2: Fall back to Python overlay (reinstall baseline + apply patch)
                else:
                    print(f"[3/3] Reinstalling baseline for agent patch (Python overlay fallback)...")
                    # Try wheel first, then overlay
                    agent_baseline_success = False
                    if baseline_wheel_url:
                        success, version = install_wheel(baseline_wheel_url)
                        if success:
                            agent_baseline_success = True

                    if not agent_baseline_success and base_commit:
                        success, version, method = install_vllm_for_commit(base_commit)
                        if success:
                            agent_baseline_success = True

                    if not agent_baseline_success:
                        result["agent_error"] = "Could not reinstall baseline for agent"
                    else:
                        patch_success, patch_msg = apply_patch_to_vllm(agent_patch)
                        if not patch_success:
                            result["agent_error"] = patch_msg
                            print(f"Warning: {patch_msg}")
                        else:
                            print(f"Patch applied: {patch_msg}")
                            result["agent_install_method"] = "python_overlay"
                            try:
                                # Agent uses baseline commit scripts (it's baseline + patch)
                                agent_output, agent_metrics = run_benchmark_phase("AGENT", base_commit)
                                result["agent_metrics"] = agent_metrics
                                result["agent_raw"] = agent_output

                                if agent_metrics:
                                    result["agent_improvement"] = compute_improvement(baseline_metrics, agent_metrics)
                                    result["agent_vs_human"] = compute_improvement(human_metrics, agent_metrics)
                                    print(f"Agent metrics: {agent_metrics}")
                                    print(f"Agent improvement: {result['agent_improvement']}")
                                else:
                                    error_snippet = agent_output[-2000:] if agent_output else "no output"
                                    result["agent_error"] = f"Agent benchmark produced no metrics. Output tail: {error_snippet}"
                                    print(f"Agent benchmark failed - output tail:\n{error_snippet}")
                            except RuntimeError as e:
                                result["agent_error"] = str(e)
            else:
                print(f"[3/3] No agent patch provided, skipping")

        result["status"] = "success"

    except Exception as e:
        result["error"] = str(e)
    finally:
        result["duration_s"] = time.time() - start_time
        # Commit both volumes
        model_cache.commit()
        build_cache.commit()

    return result


def ensure_vllm_build_cached(commit_hash: str, force_build_dir: bool = False) -> Dict[str, Any]:
    """
    Ensure vLLM is built and cached for the given commit using CPU-only instance.

    This function should be called BEFORE benchmarking commits that don't have
    pre-built wheels. It runs the compilation on a cheap CPU-only instance
    instead of wasting expensive GPU time.

    The build is cached in a Modal volume, so subsequent calls for the same
    commit will return immediately (cache hit).

    Args:
        commit_hash: Full or short vLLM commit hash to build
        force_build_dir: If True, ensure build directory exists (not just wheel).
                        Required for agent patch builds which need to modify source.

    Returns:
        Dict with keys:
        - success: bool
        - cache_hit: bool
        - version: str (vLLM version if successful)
        - error: str (error message if failed)

    Example:
        # Before running benchmark
        build_result = ensure_vllm_build_cached("88faa466d788e25082c02dc9688931d7976361f9")
        if not build_result["success"]:
            print(f"Build failed: {build_result['error']}")
            return

        # Now run benchmark (build cache will be used)
        benchmark_result = run_3way_modal_benchmark(...)
    """
    print(f"[ENSURE BUILD] Checking/building vLLM {commit_hash[:8]} on CPU-only instance...")

    try:
        # Call the CPU-only build function on Modal
        # Use enable_output() to stream build logs to local terminal
        fn = modal.Function.from_name("omniperf-benchmark", "build_vllm_cpu_only")
        with modal.enable_output():
            result = fn.remote(commit_hash, force_build_dir=force_build_dir)

        if result["success"]:
            if result["cache_hit"]:
                print(f"[ENSURE BUILD] Cache HIT: vLLM {result.get('version', 'unknown')} already cached")
            else:
                print(f"[ENSURE BUILD] Built and cached: vLLM {result.get('version', 'unknown')}")
        else:
            print(f"[ENSURE BUILD] FAILED: {result.get('error', 'Unknown error')}")

        return result

    except Exception as e:
        tb = traceback.format_exc()
        print(f"[ENSURE BUILD] Exception: {str(e)}")
        print(f"[ENSURE BUILD] Traceback:\n{tb}")
        return {
            "success": False,
            "cache_hit": False,
            "version": None,
            "error": str(e),
            "traceback": tb,
            "commit": commit_hash,
        }


def run_3way_modal_benchmark(
    baseline_wheel_url: str,
    human_wheel_url: str,
    agent_patch: Optional[str],
    perf_command: str,
    model: str,
    gpu_config: str = None,
    base_commit: Optional[str] = None,
    human_commit: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Run 3-way benchmark on Modal with automatic GPU selection.

    This is the main entry point for calling from native_benchmark_runner.py

    NEW ARCHITECTURE (v2):
    1. ALL builds happen on CPU-only instances (~$0.20/hr)
    2. GPU instance ONLY installs pre-built wheels and runs benchmarks (~$4/hr)
    3. All wheels cached in Modal volume for reuse

    Args:
        baseline_wheel_url: URL to baseline vLLM wheel (can be empty if base_commit provided)
        human_wheel_url: URL to human (post-commit) vLLM wheel (can be empty if human_commit provided)
        agent_patch: Unified diff patch from agent
        perf_command: Benchmark command to run
        model: Model name/path
        gpu_config: GPU configuration (H100:1, H100:2, H100:4, H100:8)
        base_commit: Base commit hash for Python overlay when baseline wheel unavailable
        human_commit: Human commit hash for Python overlay when human wheel unavailable
    """
    import re as local_re

    if gpu_config is None:
        gpu_config = get_gpu_config(model, perf_command)

    # Early exit for blocked models (too large/unstable to benchmark reliably)
    if is_model_blocked(model):
        print(f"[BLOCKED] Model {model} is blocked (too large/unstable), skipping benchmark")
        return {
            "status": "blocked_model",
            "error": f"Model {model} is blocked: too large or unstable for reliable benchmarking",
            "model": model,
            "gpu_config": gpu_config,
        }

    print(f"Running 3-way benchmark on Modal with {gpu_config}...")

    # =====================================================================
    # PHASE 0: VALIDATE WHEEL URLs (don't trust URLs that don't actually exist)
    # =====================================================================
    # The caller may provide S3 wheel URLs that don't exist (old commits without pre-built wheels).
    # We need to verify they exist before assuming they're valid.
    if baseline_wheel_url and base_commit:
        if not check_wheel_url_exists(base_commit):
            print(f"[URL CHECK] Baseline wheel URL does not exist for {base_commit[:8]}, will build from source")
            baseline_wheel_url = None

    if human_wheel_url and human_commit:
        if not check_wheel_url_exists(human_commit):
            print(f"[URL CHECK] Human wheel URL does not exist for {human_commit[:8]}, will build from source")
            human_wheel_url = None

    # =====================================================================
    # PHASE 1: CPU PRE-BUILDS (all compilation on cheap CPU instances)
    # =====================================================================
    # Track wheel sources for GPU function
    wheel_sources = {
        "baseline": {"url": baseline_wheel_url, "volume_path": None, "source": "url" if baseline_wheel_url else None},
        "human": {"url": human_wheel_url, "volume_path": None, "source": "url" if human_wheel_url else None},
        "agent": {"url": None, "volume_path": None, "source": None},
    }

    # 1a. Build baseline on CPU if:
    #     - No S3 wheel available (need wheel from volume), OR
    #     - We have an agent patch (need build directory for incremental compilation)
    need_base_build_for_agent = agent_patch and base_commit
    need_base_build_for_wheel = base_commit and not baseline_wheel_url

    if need_base_build_for_agent or need_base_build_for_wheel:
        reason = "for agent patch (need build dir)" if need_base_build_for_agent else "no S3 wheel"
        print(f"[CPU PRE-BUILD] Building baseline {base_commit[:8]} on CPU ({reason})...")
        # When building for agent patch, we need the build directory, not just the wheel
        baseline_result = ensure_vllm_build_cached(base_commit, force_build_dir=need_base_build_for_agent)
        if baseline_result.get("success"):
            # Only update wheel source if we don't have S3 wheel
            if not baseline_wheel_url:
                wheel_sources["baseline"]["volume_path"] = baseline_result.get("wheel_path")
                wheel_sources["baseline"]["source"] = "volume"
            print(f"[CPU PRE-BUILD] Baseline build ready (wheel: {wheel_sources['baseline']['source']})")
        else:
            print(f"[CPU PRE-BUILD] WARNING: Baseline build failed: {baseline_result.get('error')}")

    # 1b. Build human wheel if no S3 wheel available
    if human_commit and not human_wheel_url:
        print(f"[CPU PRE-BUILD] Building human {human_commit[:8]} on CPU...")
        human_result = ensure_vllm_build_cached(human_commit)
        if human_result.get("success"):
            wheel_sources["human"]["volume_path"] = human_result.get("wheel_path")
            wheel_sources["human"]["source"] = "volume"
            print(f"[CPU PRE-BUILD] Human wheel ready in volume")
        else:
            print(f"[CPU PRE-BUILD] WARNING: Human build failed: {human_result.get('error')}")

    # 1c. Build agent patched wheel (ALWAYS on CPU - this is the key change)
    agent_wheel_path = None
    if agent_patch and base_commit:
        print(f"[CPU PRE-BUILD] Building agent patch on base {base_commit[:8]}...")
        agent_result = ensure_agent_patch_built(base_commit, agent_patch)
        if agent_result.get("success"):
            agent_wheel_path = agent_result.get("wheel_path")
            wheel_sources["agent"]["volume_path"] = agent_wheel_path
            wheel_sources["agent"]["source"] = "volume"
            print(f"[CPU PRE-BUILD] Agent patched wheel ready: {agent_wheel_path}")
        else:
            print(f"[CPU PRE-BUILD] WARNING: Agent patch build failed: {agent_result.get('error')}")
            # Continue anyway - GPU function will report the error

    print(f"[CPU PRE-BUILD] Wheel sources: baseline={wheel_sources['baseline']['source']}, "
          f"human={wheel_sources['human']['source']}, agent={wheel_sources['agent']['source']}")

    # =====================================================================
    # PHASE 2: Check for Docker fallback (old commits without wheels)
    # =====================================================================
    needs_docker_fallback = False

    if base_commit and human_commit:
        baseline_wheel_ok = bool(baseline_wheel_url) or wheel_sources["baseline"]["source"] == "volume"
        human_wheel_ok = bool(human_wheel_url) or wheel_sources["human"]["source"] == "volume"

        if not baseline_wheel_ok or not human_wheel_ok:
            if has_prebuilt_image(human_commit):
                print(f"Using Docker image fallback for {human_commit[:8]}")
                needs_docker_fallback = True

    if needs_docker_fallback and human_commit and base_commit:
        return run_3way_benchmark_docker_fallback(
            human_commit=human_commit,
            base_commit=base_commit,
            agent_patch=agent_patch,
            perf_command=perf_command,
            model=model,
            gpu_config=gpu_config,
        )

    # =====================================================================
    # PHASE 3: GPU BENCHMARK (no compilation - just install wheels and run)
    # =====================================================================
    if gpu_config == "H100:8":
        fn = modal.Function.from_name("omniperf-benchmark", "run_3way_benchmark_8gpu")
    elif gpu_config == "H100:4":
        fn = modal.Function.from_name("omniperf-benchmark", "run_3way_benchmark_4gpu")
    elif gpu_config == "H100:2":
        fn = modal.Function.from_name("omniperf-benchmark", "run_3way_benchmark_2gpu")
    elif gpu_config == "H100:1" or gpu_config == "H100":
        fn = modal.Function.from_name("omniperf-benchmark", "run_3way_benchmark_1gpu")
    else:
        print(f"Warning: Unknown GPU config '{gpu_config}', using H100:1")
        fn = modal.Function.from_name("omniperf-benchmark", "run_3way_benchmark_1gpu")

    # Pass both URL and volume path - GPU function will use whichever is available
    if gpu_config in ("H100:1", "H100") or gpu_config is None:
        return fn.remote(
            baseline_wheel_url=baseline_wheel_url,
            human_wheel_url=human_wheel_url,
            agent_patch=agent_patch,  # Still passed for Python-only patches
            perf_command=perf_command,
            model=model,
            base_commit=base_commit,
            human_commit=human_commit,
            # New params for pre-built wheels from volume
            agent_wheel_path=agent_wheel_path,
        )
    else:
        return fn.remote(
            baseline_wheel_url=baseline_wheel_url,
            human_wheel_url=human_wheel_url,
            agent_patch=agent_patch,
            perf_command=perf_command,
            model=model,
        )


def run_3way_modal_benchmark_prebuilt(
    baseline_commit: str,
    human_commit: str,
    agent_patch: Optional[str],
    perf_command: str,
    model: str,
    gpu_config: str = None,
) -> Dict[str, Any]:
    """
    Run 3-way benchmark using pre-built Docker images for baseline and human.

    This function provides ~70% faster benchmark times by using pre-built Docker
    images from ayushnangia16/nvidia-vllm-docker instead of downloading wheels.

    Prerequisites:
    - Both baseline_commit and human_commit must have pre-built images
    - The Modal app must be deployed with this function available

    Args:
        baseline_commit: Full commit hash for baseline (must have pre-built image)
        human_commit: Full commit hash for human/perf commit (must have pre-built image)
        agent_patch: Unified diff patch from agent (built from source)
        perf_command: Benchmark command to run
        model: Model name/path
        gpu_config: GPU configuration (H100:1, H100:2, H100:4, H100:8)
    """
    # Verify pre-built images exist
    baseline_full = get_prebuilt_commit(baseline_commit)
    human_full = get_prebuilt_commit(human_commit)

    if not baseline_full:
        raise ValueError(f"No pre-built image for baseline commit: {baseline_commit}")
    if not human_full:
        raise ValueError(f"No pre-built image for human commit: {human_commit}")

    if gpu_config is None:
        gpu_config = get_gpu_config(model, perf_command)

    print(f"Running 3-way benchmark with pre-built images on Modal ({gpu_config})...")
    print(f"  Baseline image: {DOCKER_IMAGE_REPO}:{baseline_full[:12]}")
    print(f"  Human image: {DOCKER_IMAGE_REPO}:{human_full[:12]}")
    print(f"  Agent: {'patch' if agent_patch else 'none'}")

    # Create sandbox with pre-built baseline image for initial test
    # Then use Modal's container recycling for different commits
    baseline_image = get_prebuilt_image(baseline_full)
    human_image = get_prebuilt_image(human_full)

    result = {
        "status": "error",
        "gpu_config": gpu_config,
        "baseline_metrics": {},
        "human_metrics": {},
        "agent_metrics": None,
        "human_improvement": {},
        "agent_improvement": None,
        "agent_vs_human": None,
        "error": None,
        "prebuilt_mode": True,
        "baseline_image": f"{DOCKER_IMAGE_REPO}:{baseline_full}",
        "human_image": f"{DOCKER_IMAGE_REPO}:{human_full}",
        # Raw outputs for debugging and analysis
        "baseline_raw": "",
        "human_raw": "",
        "agent_raw": None,
    }

    # For pre-built images, we need to run separate sandboxes for baseline and human
    # because each has different vLLM versions installed
    try:
        # Run baseline benchmark in sandbox with pre-built image
        print("[1/3] Running BASELINE benchmark with pre-built image...")
        with modal.enable_output():
            sb_baseline = modal.Sandbox.create(
                image=baseline_image,
                gpu="H100",
                timeout=1800,
                secrets=[modal.Secret.from_name("huggingface-secret")],
            )
            # Verify vLLM is installed and run benchmark
            baseline_result = sb_baseline.exec(
                "python", "-c",
                f'''
import vllm
import json
import time
import subprocess

print(f"vLLM version: {{vllm.__version__}}")

# Run benchmark command
result = subprocess.run(
    {repr(perf_command)},
    shell=True,
    capture_output=True,
    text=True,
    timeout=1200,
)
print("BENCHMARK_OUTPUT_START")
print(result.stdout)
print(result.stderr)
print("BENCHMARK_OUTPUT_END")
'''
            )
            baseline_output = baseline_result.stdout.read()
            if "BENCHMARK_OUTPUT_START" in baseline_output:
                benchmark_text = baseline_output.split("BENCHMARK_OUTPUT_START")[1].split("BENCHMARK_OUTPUT_END")[0]
                result["baseline_metrics"] = parse_metrics(benchmark_text)
                # Save full raw output (no truncation - saved to separate file by caller)
                result["baseline_raw"] = benchmark_text
            sb_baseline.terminate()

        if not result["baseline_metrics"]:
            result["error"] = "Baseline benchmark produced no metrics"
            return result

        print(f"  Baseline metrics: {result['baseline_metrics']}")

        # Run human benchmark in sandbox with pre-built image
        print("[2/3] Running HUMAN benchmark with pre-built image...")
        with modal.enable_output():
            sb_human = modal.Sandbox.create(
                image=human_image,
                gpu="H100",
                timeout=1800,
                secrets=[modal.Secret.from_name("huggingface-secret")],
            )
            human_result = sb_human.exec(
                "python", "-c",
                f'''
import vllm
import json
import time
import subprocess

print(f"vLLM version: {{vllm.__version__}}")

result = subprocess.run(
    {repr(perf_command)},
    shell=True,
    capture_output=True,
    text=True,
    timeout=1200,
)
print("BENCHMARK_OUTPUT_START")
print(result.stdout)
print(result.stderr)
print("BENCHMARK_OUTPUT_END")
'''
            )
            human_output = human_result.stdout.read()
            if "BENCHMARK_OUTPUT_START" in human_output:
                benchmark_text = human_output.split("BENCHMARK_OUTPUT_START")[1].split("BENCHMARK_OUTPUT_END")[0]
                result["human_metrics"] = parse_metrics(benchmark_text)
                # Save full raw output (no truncation - saved to separate file by caller)
                result["human_raw"] = benchmark_text
            sb_human.terminate()

        if not result["human_metrics"]:
            result["error"] = "Human benchmark produced no metrics"
            return result

        result["human_improvement"] = compute_improvement(result["baseline_metrics"], result["human_metrics"])
        print(f"  Human metrics: {result['human_metrics']}")
        print(f"  Human improvement: {result['human_improvement']}")

        # For agent, use the existing build-from-source approach
        if agent_patch:
            print("[3/3] Building and running AGENT patch (incremental build)...")
            # Fall back to existing function for agent (needs build from source)
            fn = modal.Function.from_name("omniperf-benchmark", "run_3way_benchmark_1gpu")
            agent_result = fn.remote(
                baseline_wheel_url="",  # Not used when building from source
                human_wheel_url="",
                agent_patch=agent_patch,
                perf_command=perf_command,
                model=model,
                base_commit=baseline_commit,
            )
            if agent_result.get("agent_metrics"):
                result["agent_metrics"] = agent_result["agent_metrics"]
                result["agent_improvement"] = compute_improvement(result["baseline_metrics"], result["agent_metrics"])
                result["agent_vs_human"] = compute_improvement(result["human_metrics"], result["agent_metrics"])
            # Also capture agent raw output if available
            if agent_result.get("agent_raw"):
                result["agent_raw"] = agent_result["agent_raw"]

        result["status"] = "success"

    except Exception as e:
        tb = traceback.format_exc()
        result["error"] = f"Pre-built benchmark failed: {str(e)}"
        result["traceback"] = tb
        print(f"[PRE-BUILT BENCHMARK] Exception: {str(e)}")
        print(f"[PRE-BUILT BENCHMARK] Traceback:\n{tb}")

    return result


# =====================================================================
# Parallel 3-Way Benchmark (3 sandboxes, 3 GPUs, ~3x faster)
# =====================================================================

def _run_vllm_phase_sandbox(
    phase: str,
    commit: str,
    perf_command: str,
    model: str,
    gpu_config: str = "H100:1",
    agent_patch: Optional[str] = None,
    base_commit: Optional[str] = None,
) -> Dict[str, Any]:
    """Run a single vLLM benchmark phase in its own sandbox."""

    result = {
        "phase": phase,
        "metrics": {},
        "status": "error",
        "error": None,
        "duration_s": 0,
        "raw_output": "",  # Full raw output from benchmark
        "commit": commit,
    }

    start_time = time.time()

    try:
        # Get the pre-built image for this commit
        full_commit = get_prebuilt_commit(commit)
        if not full_commit:
            result["error"] = f"No pre-built image for {commit[:8]}"
            return result

        image = get_prebuilt_image(full_commit)
        if not image:
            result["error"] = f"Failed to get image for {commit[:8]}"
            return result

        print(f"[{phase.upper()}] Creating sandbox with H100...")

        # Create sandbox
        sandbox_app = modal.App.lookup("omniperf-benchmark", create_if_missing=True)

        sb = modal.Sandbox.create(
            app=sandbox_app,
            image=image,
            gpu="H100",
            timeout=7200,
            secrets=[modal.Secret.from_name("huggingface-secret")],
            volumes={"/root/.cache/huggingface": model_cache},
        )

        # For agent phase, we need to apply patch
        if phase == "agent" and agent_patch:
            # Clone vLLM repo and apply patch
            patch_script = f'''
import subprocess
import sys
import os

# Clone vLLM repo
subprocess.run(["git", "clone", "--depth", "100", "https://github.com/vllm-project/vllm.git", "/tmp/vllm_agent"], capture_output=True)
subprocess.run(["git", "fetch", "origin", "{base_commit}"], cwd="/tmp/vllm_agent", capture_output=True)
subprocess.run(["git", "checkout", "{base_commit}"], cwd="/tmp/vllm_agent", capture_output=True)

# Write patch
patch_content = {repr(agent_patch)}
with open("/tmp/agent.patch", "w") as f:
    f.write(patch_content)

# Apply patch
result = subprocess.run(["git", "apply", "/tmp/agent.patch"], cwd="/tmp/vllm_agent", capture_output=True, text=True)
if result.returncode != 0:
    result = subprocess.run(["patch", "-p1", "-i", "/tmp/agent.patch"], cwd="/tmp/vllm_agent", capture_output=True, text=True)

# Overlay Python files
import shutil
from pathlib import Path
vllm_install = subprocess.run(["python", "-c", "import vllm; print(vllm.__path__[0])"], capture_output=True, text=True)
target_dir = Path(vllm_install.stdout.strip())
source_dir = Path("/tmp/vllm_agent/vllm")

if source_dir.exists() and target_dir.exists():
    for py_file in source_dir.rglob("*.py"):
        rel_path = py_file.relative_to(source_dir)
        dest_file = target_dir / rel_path
        if dest_file.exists():
            shutil.copy2(py_file, dest_file)
        else:
            dest_file.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(py_file, dest_file)
    print("PATCH_APPLIED_SUCCESS")
else:
    print("PATCH_APPLIED_FAILED")
'''
            patch_proc = sb.exec("python", "-c", patch_script)
            patch_output = ""
            for line in patch_proc.stdout:
                patch_output += line
                print(f"[{phase.upper()}] {line.rstrip()}")
            patch_proc.wait()

            if "PATCH_APPLIED_FAILED" in patch_output:
                result["error"] = "Failed to apply agent patch"
                sb.terminate()
                return result

        # Run benchmark
        benchmark_script = f'''
import vllm
import json
import time
import subprocess
import re

print(f"vLLM version: {{vllm.__version__}}")

# Run benchmark command
result = subprocess.run(
    {repr(perf_command)},
    shell=True,
    capture_output=True,
    text=True,
    timeout=3600,
)

output = result.stdout + result.stderr
print("BENCHMARK_OUTPUT_START")
print(output)
print("BENCHMARK_OUTPUT_END")

# Parse metrics
metrics = {{}}
patterns = {{
    "request_throughput": r"Request throughput[:\\s]+([\\d.]+)",
    "output_throughput": r"Output token throughput[:\\s]+([\\d.]+)",
    "ttft_mean": r"Mean TTFT[:\\s]+([\\d.]+)",
    "tpot_mean": r"Mean TPOT[:\\s]+([\\d.]+)",
    "itl_mean": r"Mean ITL[:\\s]+([\\d.]+)",
}}

for name, pattern in patterns.items():
    match = re.search(pattern, output, re.IGNORECASE)
    if match:
        metrics[name] = float(match.group(1))

print("METRICS_JSON_START")
print(json.dumps(metrics))
print("METRICS_JSON_END")
'''

        print(f"[{phase.upper()}] Running benchmark...")
        bench_proc = sb.exec("python", "-c", benchmark_script)

        stdout_lines = []
        for line in bench_proc.stdout:
            stdout_lines.append(line)
            print(f"[{phase.upper()}] {line.rstrip()}")

        bench_proc.wait()

        # Save full raw output IMMEDIATELY before any other processing
        full_output = "\n".join(stdout_lines)
        result["raw_output"] = full_output

        sb.terminate()

        # Also extract just the benchmark output (between markers)
        import re as re_mod
        bench_match = re_mod.search(r'BENCHMARK_OUTPUT_START\s*(.*?)\s*BENCHMARK_OUTPUT_END', full_output, re_mod.DOTALL)
        if bench_match:
            result["benchmark_raw"] = bench_match.group(1).strip()

        # Parse metrics from output
        metrics_match = re_mod.search(r'METRICS_JSON_START\s*(\{.*?\})\s*METRICS_JSON_END', full_output, re_mod.DOTALL)
        if metrics_match:
            result["metrics"] = json.loads(metrics_match.group(1))
            result["status"] = "success"
        else:
            result["error"] = "No metrics found in output"

    except Exception as e:
        result["error"] = f"Sandbox error: {str(e)}"
        # Try to capture any partial output that was collected
        if 'stdout_lines' in locals() and stdout_lines:
            result["raw_output"] = "\n".join(stdout_lines)

    result["duration_s"] = time.time() - start_time
    return result


def run_3way_modal_benchmark_prebuilt_parallel(
    baseline_commit: str,
    human_commit: str,
    agent_patch: Optional[str],
    perf_command: str,
    model: str,
    gpu_config: str = "H100:1",
) -> Dict[str, Any]:
    """
    Run 3-way benchmark in PARALLEL using 3 separate Modal sandboxes.

    Each benchmark (baseline, human, agent) runs on its own GPU simultaneously.
    This is ~3x faster but uses 3x GPU hours.

    Args:
        baseline_commit: Full commit hash for baseline (must have pre-built image)
        human_commit: Full commit hash for human/perf commit (must have pre-built image)
        agent_patch: Unified diff patch from agent (optional)
        perf_command: Benchmark command to run
        model: Model name/path
        gpu_config: GPU configuration (H100:1, etc.)

    Returns:
        Dict with baseline_metrics, human_metrics, agent_metrics, status, error
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed

    result = {
        "status": "error",
        "gpu_config": gpu_config,
        "baseline_metrics": {},
        "human_metrics": {},
        "agent_metrics": None,
        "human_improvement": {},
        "agent_improvement": None,
        "agent_vs_human": None,
        "error": None,
        "duration_s": 0,
        "perf_command": perf_command,
        "install_method": "prebuilt_parallel",
        "benchmark_mode": "parallel_3way",
        # Raw outputs for debugging and analysis
        "baseline_raw": "",
        "human_raw": "",
        "agent_raw": None,
    }

    start_time = time.time()

    # Verify pre-built images exist
    baseline_full = get_prebuilt_commit(baseline_commit)
    human_full = get_prebuilt_commit(human_commit)

    if not baseline_full:
        result["error"] = f"No pre-built image for baseline: {baseline_commit[:8]}"
        return result
    if not human_full:
        result["error"] = f"No pre-built image for human: {human_commit[:8]}"
        return result

    print(f"=== PARALLEL 3-WAY BENCHMARK (vLLM) ===")
    print(f"Baseline: {baseline_commit[:8]}, Human: {human_commit[:8]}")
    print(f"Agent patch: {'Yes' if agent_patch else 'No'}")
    print(f"GPU config: {gpu_config} x 3 sandboxes")

    # Prepare phases to run
    phases = [
        ("baseline", baseline_commit, None, None),
        ("human", human_commit, None, None),
    ]

    if agent_patch:
        phases.append(("agent", baseline_commit, agent_patch, baseline_commit))

    print(f"Running {len(phases)} phases in parallel: {[p[0] for p in phases]}")

    # Run phases in parallel
    phase_results = {}
    with ThreadPoolExecutor(max_workers=3) as executor:
        futures = {}
        for phase, commit, patch, base in phases:
            future = executor.submit(
                _run_vllm_phase_sandbox,
                phase=phase,
                commit=commit,
                perf_command=perf_command,
                model=model,
                gpu_config=gpu_config,
                agent_patch=patch,
                base_commit=base,
            )
            futures[future] = phase

        for future in as_completed(futures):
            phase = futures[future]
            try:
                phase_result = future.result()
                phase_results[phase] = phase_result
                print(f"[{phase.upper()}] Completed: {phase_result.get('status')}")
            except Exception as e:
                phase_results[phase] = {"status": "error", "error": str(e), "metrics": {}}
                print(f"[{phase.upper()}] Failed: {e}")

    # Combine results - metrics AND raw outputs
    if "baseline" in phase_results:
        result["baseline_metrics"] = phase_results["baseline"].get("metrics", {})
        result["baseline_raw"] = phase_results["baseline"].get("raw_output", "")

    if "human" in phase_results:
        result["human_metrics"] = phase_results["human"].get("metrics", {})
        result["human_raw"] = phase_results["human"].get("raw_output", "")
        if phase_results["human"].get("status") == "success":
            result["status"] = "success"

    if "agent" in phase_results:
        result["agent_metrics"] = phase_results["agent"].get("metrics", {})
        result["agent_raw"] = phase_results["agent"].get("raw_output", "")

    # Calculate improvements
    if result["baseline_metrics"] and result["human_metrics"]:
        result["human_improvement"] = compute_improvement(result["baseline_metrics"], result["human_metrics"])

    if result["baseline_metrics"] and result.get("agent_metrics"):
        result["agent_improvement"] = compute_improvement(result["baseline_metrics"], result["agent_metrics"])

    if result.get("human_metrics") and result.get("agent_metrics"):
        result["agent_vs_human"] = compute_improvement(result["human_metrics"], result["agent_metrics"])

    result["duration_s"] = time.time() - start_time
    result["phase_durations"] = {p: r.get("duration_s", 0) for p, r in phase_results.items()}

    print(f"\n=== PARALLEL BENCHMARK COMPLETE ===")
    print(f"Total duration: {result['duration_s']:.1f}s")
    print(f"Phase durations: {result['phase_durations']}")

    return result


def run_3way_benchmark_docker_fallback(
    human_commit: str,
    base_commit: str,
    agent_patch: Optional[str],
    perf_command: str,
    model: str,
    gpu_config: str = "H100:1",
) -> Dict[str, Any]:
    """
    Run 3-way benchmark using Docker image fallback.

    This is used when S3 wheels are not available for old commits.
    Uses the HUMAN commit's Docker image and overlays Python files for BASELINE.

    Strategy:
    1. Start a Modal Sandbox with HUMAN's Docker image (vLLM pre-installed)
    2. For BASELINE: git checkout parent, overlay Python files, run benchmark
    3. For HUMAN: restore original files (or just use as-is), run benchmark
    4. For AGENT: overlay parent's Python files + apply patch, run benchmark

    Args:
        human_commit: Human commit hash (must have Docker image available)
        base_commit: Baseline/parent commit hash
        agent_patch: Optional unified diff patch from agent
        perf_command: Benchmark command to run
        model: Model name/path
        gpu_config: GPU configuration (only H100:1 supported for now)
    """
    result = {
        "status": "error",
        "gpu_config": gpu_config,
        "baseline_metrics": {},
        "human_metrics": {},
        "agent_metrics": None,
        "human_improvement": {},
        "agent_improvement": None,
        "agent_vs_human": None,
        "error": None,
        "duration_s": 0,
        "perf_command": perf_command,
        "install_method": "docker_fallback",
    }

    start_time = time.time()

    # Verify Docker image exists
    if not has_prebuilt_image(human_commit):
        result["error"] = f"No Docker image available for human commit {human_commit}"
        return result

    docker_image = get_prebuilt_image(human_commit)
    if docker_image is None:
        result["error"] = f"Failed to create Modal image for {human_commit}"
        return result

    full_human_commit = get_prebuilt_commit(human_commit)
    print(f"Using Docker image fallback: {DOCKER_IMAGE_REPO}:{full_human_commit[:12]}")
    print(f"  Human commit: {human_commit}")
    print(f"  Base commit: {base_commit}")

    # Prepare the benchmark script that runs inside the container
    # This script handles all 3 benchmarks with Python overlay
    benchmark_script = f'''
import subprocess
import sys
import os
import json
import time
import re
import shutil
import site
from pathlib import Path

# Verify HF_TOKEN is available (required for gated models like Llama)
hf_token = os.environ.get("HF_TOKEN")
if hf_token:
    print(f"  HF_TOKEN found ({{len(hf_token)}} chars), will be used for model downloads")
else:
    print("  WARNING: HF_TOKEN not found in environment! Gated models will fail.")
    print(f"  Available env vars: {{[k for k in os.environ.keys() if 'HF' in k or 'HUGGING' in k]}}")

# Fix known dependency issues before importing vLLM
def fix_dependency_issues():
    """Fix known dependency compatibility issues in Docker images.

    Main issue: lmformatenforcer < 0.10.0 requires transformers API that was
    removed in transformers 4.41+. The old lmformatenforcer tries to import
    LogitsWarper which no longer exists.

    Fix: Upgrade lmformatenforcer to >= 0.10.0 which handles both old and new
    transformers APIs.
    """
    print("Checking for dependency compatibility issues...")

    # Test if vLLM can be imported (longer timeout for CUDA initialization)
    test_result = subprocess.run(
        ["python", "-c", "import vllm"],
        capture_output=True, text=True, timeout=180
    )

    if test_result.returncode == 0:
        print("  vLLM imports successfully, no fixes needed")
        return True

    error_output = test_result.stderr + test_result.stdout

    # Check for LogitsWarper import error (lmformatenforcer + transformers incompatibility)
    if "LogitsWarper" in error_output or "lmformatenforcer" in error_output.lower():
        print("  Detected lmformatenforcer/transformers incompatibility")
        print("  Upgrading lmformatenforcer to fix LogitsWarper import error...")

        # Try uv first for faster, more reliable dependency resolution
        # Fall back to pip if uv not available
        try:
            fix_result = subprocess.run(
                ["uv", "pip", "install", "--quiet", "lmformatenforcer>=0.10.0"],
                capture_output=True, text=True, timeout=300
            )
            if fix_result.returncode != 0:
                raise Exception("uv failed")
        except Exception:
            # uv not available or failed, use pip
            fix_result = subprocess.run(
                ["pip", "install", "--quiet", "lmformatenforcer>=0.10.0"],
                capture_output=True, text=True, timeout=300
            )

        if fix_result.returncode == 0:
            print("  lmformatenforcer upgraded successfully")

            # Verify fix worked (longer timeout for CUDA initialization)
            verify_result = subprocess.run(
                ["python", "-c", "import vllm; print(f'vLLM {{vllm.__version__}} imported successfully')"],
                capture_output=True, text=True, timeout=180
            )

            if verify_result.returncode == 0:
                print(f"  {{verify_result.stdout.strip()}}")
                return True
            else:
                print(f"  Fix applied but vLLM still fails: {{verify_result.stderr[:500]}}")
                return False
        else:
            print(f"  Failed to upgrade lmformatenforcer: {{fix_result.stderr[:500]}}")
            return False

    # Unknown error
    print(f"  Unknown vLLM import error: {{error_output[:500]}}")
    return False

# Apply dependency fixes
fix_dependency_issues()

# Configuration
HUMAN_COMMIT = "{full_human_commit}"
BASE_COMMIT = "{base_commit}"
PERF_COMMAND = {repr(perf_command)}
MODEL = {repr(model)}
AGENT_PATCH = {repr(agent_patch) if agent_patch else "None"}

def parse_metrics(output):
    """Parse benchmark output for metrics."""
    metrics = {{}}
    patterns = {{
        r"Request throughput:\\s*([\\d.]+)\\s*requests/s": "request_throughput",
        r"Output token throughput:\\s*([\\d.]+)\\s*tokens/s": "output_throughput",
        r"Total Token throughput:\\s*([\\d.]+)\\s*tokens/s": "total_throughput",
        r"Mean TTFT \\(ms\\):\\s*([\\d.]+)": "ttft_mean",
        r"Median TTFT \\(ms\\):\\s*([\\d.]+)": "ttft_median",
        r"P99 TTFT \\(ms\\):\\s*([\\d.]+)": "ttft_p99",
        r"Mean TPOT \\(ms\\):\\s*([\\d.]+)": "tpot_mean",
        r"Mean ITL \\(ms\\):\\s*([\\d.]+)": "itl_mean",
        r"Avg latency:\\s*([\\d.]+)\\s*seconds": "latency_avg_s",
        r"throughput[=:]\\s*([\\d.]+)": "throughput",
    }}
    for pattern, key in patterns.items():
        match = re.search(pattern, output, re.IGNORECASE)
        if match:
            value = float(match.group(1))
            if key == "latency_avg_s":
                metrics["latency_avg"] = value * 1000
            else:
                metrics[key] = value
    return metrics

def find_vllm_site_packages():
    """Find vLLM installation directory."""
    for sp in site.getsitepackages():
        candidate = Path(sp) / "vllm"
        if candidate.exists():
            return candidate
    # Try common locations
    for path in ["/usr/local/lib/python3.12/dist-packages/vllm",
                 "/usr/local/lib/python3.11/dist-packages/vllm"]:
        if os.path.exists(path):
            return Path(path)
    return None

def overlay_python_files(repo_path, vllm_site_path):
    """Copy Python files from repo checkout to installed vLLM."""
    source_vllm = Path(repo_path) / "vllm"
    if not source_vllm.exists():
        return 0

    copied = 0
    for src_file in source_vllm.rglob("*.py"):
        rel_path = src_file.relative_to(source_vllm)
        dst_file = vllm_site_path / rel_path
        dst_file.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src_file, dst_file)
        copied += 1
    return copied

def clone_and_checkout(commit, repo_path="/tmp/vllm-checkout"):
    """Clone vLLM repo and checkout specific commit.

    Uses shallow clone + targeted fetch strategy:
    1. git clone --depth 1 (fast, ~5s)
    2. git fetch --depth 1 origin <commit> (fetches just that commit)
    3. git checkout <commit>

    This works for any commit regardless of age, unlike treeless/blobless clones
    which fail on old commits not in advertised refs.
    """
    if os.path.exists(repo_path):
        shutil.rmtree(repo_path)

    print(f"  Cloning vLLM repo (shallow clone)...")
    result = subprocess.run(
        ["git", "clone", "--depth", "1", "https://github.com/vllm-project/vllm.git", repo_path],
        capture_output=True, text=True, timeout=120
    )
    if result.returncode != 0:
        return False, f"Clone failed: {{result.stderr}}"

    print(f"  Fetching commit {{commit[:8]}}...")
    result = subprocess.run(
        ["git", "fetch", "--depth", "1", "origin", commit],
        cwd=repo_path, capture_output=True, text=True, timeout=120
    )
    if result.returncode != 0:
        return False, f"Fetch failed: {{result.stderr}}"

    # Checkout the specific commit
    print(f"  Checking out {{commit[:8]}}...")
    result = subprocess.run(
        ["git", "checkout", commit],
        cwd=repo_path, capture_output=True, text=True, timeout=60
    )
    if result.returncode != 0:
        return False, f"Checkout failed: {{result.stderr}}"

    return True, repo_path

def is_serving_benchmark(cmd):
    """Check if benchmark needs a server."""
    standalone = ["benchmark_throughput", "benchmark_latency"]
    for p in standalone:
        if p in cmd.lower():
            return False
    return "benchmark_serving" in cmd.lower()

def vllm_version_supports_serving(version):
    """Check if vLLM version supports serving benchmarks (>= 0.7.0)."""
    import re as _re
    match = _re.match(r'(\\d+)\\.(\\d+)', version)
    if not match:
        return False
    major = int(match.group(1))
    minor = int(match.group(2))
    return major > 0 or (major == 0 and minor >= 7)

def translate_benchmark_command(cmd, repo_path="/tmp/vllm-checkout"):
    """Translate benchmark command for this vLLM version.

    Uses benchmark scripts from the checked-out repo to ensure version compatibility.
    Falls back to vllm CLI if available.
    """
    # Check if vllm bench CLI exists (newer versions)
    try:
        result = subprocess.run(["vllm", "bench", "--help"], capture_output=True, timeout=10)
        has_cli = result.returncode == 0
    except:
        has_cli = False

    # Use benchmark scripts from the repo checkout (same version as code being tested)
    # This ensures benchmark script compatibility with the vLLM version
    benchmark_base = f"{{repo_path}}/benchmarks"

    if has_cli:
        if "benchmark_serving.py" in cmd:
            cmd = re.sub(r"python3?\\s+(?:\\./)?(?:benchmarks/)?benchmark_serving\\.py", "vllm bench serve", cmd)
        if "benchmark_throughput.py" in cmd:
            cmd = re.sub(r"python3?\\s+(?:\\./)?(?:benchmarks/)?benchmark_throughput\\.py", "vllm bench throughput", cmd)
        if "benchmark_latency.py" in cmd:
            cmd = re.sub(r"python3?\\s+(?:\\./)?(?:benchmarks/)?benchmark_latency\\.py", "vllm bench latency", cmd)
    else:
        # Use benchmark scripts from repo checkout (version-compatible)
        if "benchmark_serving.py" in cmd:
            cmd = re.sub(r"python3?\\s+(?:\\./)?(?:benchmarks/)?benchmark_serving\\.py", f"python {{benchmark_base}}/benchmark_serving.py", cmd)
        if "benchmark_throughput.py" in cmd:
            cmd = re.sub(r"python3?\\s+(?:\\./)?(?:benchmarks/)?benchmark_throughput\\.py", f"python {{benchmark_base}}/benchmark_throughput.py", cmd)
        if "benchmark_latency.py" in cmd:
            cmd = re.sub(r"python3?\\s+(?:\\./)?(?:benchmarks/)?benchmark_latency\\.py", f"python {{benchmark_base}}/benchmark_latency.py", cmd)

    return cmd

def run_serving_benchmark(cmd, model, repo_path="/tmp/vllm-checkout", port=29000):
    """Run a serving benchmark (starts server, runs client, stops server)."""
    import signal

    # Extract server-relevant parameters from the benchmark command
    # These are needed for proper model loading
    server_cmd = [
        "python", "-m", "vllm.entrypoints.openai.api_server",
        "--model", model,
        "--host", "0.0.0.0",  # Bind to all interfaces
        "--port", str(port),
        "--disable-frontend-multiprocessing",  # Fix for vLLM 0.6.x port binding bug
    ]

    # Extract dtype (critical for memory management)
    dtype_match = re.search(r"--dtype\\s+(\\S+)", cmd)
    if dtype_match:
        server_cmd.extend(["--dtype", dtype_match.group(1)])

    # Extract tensor parallel size (critical for multi-GPU)
    tp_match = re.search(r"(?:--tensor-parallel-size|-tp)\\s+(\\d+)", cmd)
    if tp_match:
        server_cmd.extend(["--tensor-parallel-size", tp_match.group(1)])

    # Extract trust-remote-code (needed for custom models)
    if "--trust-remote-code" in cmd:
        server_cmd.append("--trust-remote-code")

    # Extract max-model-len if specified
    max_len_match = re.search(r"--max-model-len\\s+(\\d+)", cmd)
    if max_len_match:
        server_cmd.extend(["--max-model-len", max_len_match.group(1)])

    # Extract gpu-memory-utilization if specified
    gpu_mem_match = re.search(r"--gpu-memory-utilization\\s+([\\d.]+)", cmd)
    if gpu_mem_match:
        server_cmd.extend(["--gpu-memory-utilization", gpu_mem_match.group(1)])

    # Extract enforce-eager if specified
    if "--enforce-eager" in cmd:
        server_cmd.append("--enforce-eager")

    # Kill any existing vLLM servers to avoid port conflicts (aggressive cleanup)
    print(f"  [run_serving_benchmark] Cleaning up ports (HTTP={{port}}, NCCL=29001,29500)...")

    # Kill vLLM and Ray processes
    try:
        subprocess.run(["pkill", "-9", "-f", "vllm.entrypoints"], capture_output=True, timeout=5)
    except:
        pass
    try:
        subprocess.run(["pkill", "-9", "-f", "ray::"], capture_output=True, timeout=5)
    except:
        pass
    try:
        subprocess.run(["pkill", "-9", "-f", "torch.distributed"], capture_output=True, timeout=5)
    except:
        pass

    time.sleep(2)

    # Clean up HTTP port
    try:
        subprocess.run(["fuser", "-k", "-9", f"{{port}}/tcp"], capture_output=True, timeout=5)
    except:
        pass

    # Clean up NCCL/torch.distributed ports (29001 is the common culprit)
    for nccl_port in [29001, 29500, 29502, 12355]:
        try:
            subprocess.run(["fuser", "-k", "-9", f"{{nccl_port}}/tcp"], capture_output=True, timeout=3)
        except:
            pass

    # Extra wait to ensure sockets are fully released
    time.sleep(3)

    # Start server
    print(f"  Starting vLLM server with cmd: {{' '.join(server_cmd[:8])}}...")

    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    # Fix for vLLM 0.6.x port binding issue (socket duplication during fork)
    env["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
    # CRITICAL: Set MASTER_PORT to avoid NCCL binding to same port as HTTP server
    nccl_port = 35000 + (port % 1000)  # e.g., 18001 -> 35001
    env["MASTER_PORT"] = str(nccl_port)
    env["MASTER_ADDR"] = "127.0.0.1"
    print(f"  Setting MASTER_PORT={{nccl_port}}, MASTER_ADDR=127.0.0.1 (HTTP port={{port}})")

    server = subprocess.Popen(
        server_cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        env=env,
        preexec_fn=os.setsid,
    )

    # Wait for server to be ready
    import urllib.request
    start = time.time()
    ready = False
    while time.time() - start < 600:
        try:
            with urllib.request.urlopen(f"http://localhost:{{port}}/health", timeout=5) as r:
                if r.status == 200:
                    ready = True
                    break
        except:
            pass
        time.sleep(5)

    if not ready:
        # Capture server output for debugging before killing
        server_output = ""
        try:
            # Try to read any available output
            import select
            if server.stdout and select.select([server.stdout], [], [], 0)[0]:
                server_output = server.stdout.read(10000).decode('utf-8', errors='ignore')
        except:
            pass
        os.killpg(os.getpgid(server.pid), signal.SIGTERM)
        error_msg = f"Server failed to start. Server output: {{server_output[-2000:]}}" if server_output else "Server failed to start (no output captured)"
        print(f"  ERROR: {{error_msg[:500]}}")
        return error_msg, {{}}

    print("  Server ready, running benchmark...")

    # Prepare benchmark command (use scripts from repo checkout)
    bench_cmd = translate_benchmark_command(cmd, repo_path=repo_path)

    # Remove server-only args
    bench_cmd = re.sub(r"--dtype\\s+\\S+", "", bench_cmd)
    bench_cmd = re.sub(r"--tensor-parallel-size\\s+\\d+", "", bench_cmd)
    bench_cmd = re.sub(r"-tp\\s+\\d+", "", bench_cmd)
    bench_cmd = re.sub(r"--trust-remote-code", "", bench_cmd)

    # Remove args that don't exist in old vLLM versions (0.5.x, early 0.6.x)
    # These cause "unrecognized arguments" errors with old benchmark_serving.py
    bench_cmd = re.sub(r"--enable-prefix-caching", "", bench_cmd)
    bench_cmd = re.sub(r"--use-v2-block-manager", "", bench_cmd)
    bench_cmd = re.sub(r"--kv-cache-dtype\\s+\\S+", "", bench_cmd)
    bench_cmd = re.sub(r"--enable-chunked-prefill", "", bench_cmd)
    bench_cmd = re.sub(r"--speculative-model\\s+\\S+", "", bench_cmd)
    bench_cmd = re.sub(r"--num-speculative-tokens\\s+\\d+", "", bench_cmd)
    bench_cmd = re.sub(r"--max-model-len\\s+\\d+", "", bench_cmd)
    bench_cmd = re.sub(r"--gpu-memory-utilization\\s+[\\d.]+", "", bench_cmd)
    bench_cmd = re.sub(r"--enforce-eager", "", bench_cmd)
    # --backend was added in later vLLM versions, old benchmark_serving.py doesn't have it
    bench_cmd = re.sub(r"--backend\\s+\\S+", "", bench_cmd)

    # Add host/port
    if "--host" not in bench_cmd and "--base-url" not in bench_cmd:
        bench_cmd += " --host 127.0.0.1"
    if "--port" not in bench_cmd and "--base-url" not in bench_cmd:
        bench_cmd += f" --port {{port}}"

    # Remove old --dataset <file> argument (deprecated, requires local file that doesn't exist in sandbox)
    # This allows the code below to add --dataset-name random instead
    bench_cmd = re.sub(r"--dataset\\s+\\S+\\.json", "", bench_cmd)
    bench_cmd = re.sub(r"--dataset\\s+ShareGPT\\S*", "", bench_cmd)

    # Add dataset if needed
    if "--dataset-name" not in bench_cmd and "--dataset-path" not in bench_cmd and "--dataset " not in bench_cmd:
        bench_cmd += " --dataset-name random --random-input-len 512 --random-output-len 128"

    if "--num-prompts" not in bench_cmd:
        bench_cmd += " --num-prompts 100"

    bench_cmd = re.sub(r"\\s+", " ", bench_cmd).strip()

    try:
        result = subprocess.run(
            bench_cmd, shell=True, capture_output=True, text=True, timeout=1800, cwd="/tmp"
        )
        output = result.stdout + "\\n" + result.stderr
    except Exception as e:
        output = f"Error: {{str(e)}}"

    # Stop server
    try:
        os.killpg(os.getpgid(server.pid), signal.SIGTERM)
        server.wait(timeout=10)
    except:
        try:
            os.killpg(os.getpgid(server.pid), signal.SIGKILL)
        except:
            pass

    return output, parse_metrics(output)

def run_standalone_benchmark(cmd, model, repo_path="/tmp/vllm-checkout"):
    """Run a standalone benchmark (throughput/latency)."""
    bench_cmd = translate_benchmark_command(cmd, repo_path=repo_path)
    bench_cmd = re.sub(r"\\s+", " ", bench_cmd).strip()

    print(f"  Running: {{bench_cmd[:100]}}...")

    try:
        result = subprocess.run(
            bench_cmd, shell=True, capture_output=True, text=True, timeout=3600, cwd="/tmp"
        )
        output = result.stdout + "\\n" + result.stderr
    except Exception as e:
        output = f"Error: {{str(e)}}"

    return output, parse_metrics(output)

def run_benchmark(cmd, model, repo_path="/tmp/vllm-checkout", port=29000):
    """Run benchmark (auto-detect serving vs standalone)."""
    if is_serving_benchmark(cmd):
        return run_serving_benchmark(cmd, model, repo_path=repo_path, port=port)
    else:
        return run_standalone_benchmark(cmd, model, repo_path=repo_path)

# Main execution
results = {{
    "baseline_metrics": {{}},
    "human_metrics": {{}},
    "agent_metrics": None,
    "status": "error",
}}

try:
    vllm_path = find_vllm_site_packages()
    if not vllm_path:
        raise RuntimeError("Could not find vLLM installation")

    print(f"vLLM installed at: {{vllm_path}}")

    # Clone repo for Python file overlay
    print("Cloning vLLM repo for Python overlay...")
    success, repo_path = clone_and_checkout(BASE_COMMIT)
    if not success:
        raise RuntimeError(f"Failed to checkout base commit: {{repo_path}}")

    # ========== BASELINE BENCHMARK ==========
    print("\\n[1/3] Running BASELINE benchmark...")
    print(f"  Overlaying Python files from {{BASE_COMMIT[:8]}}...")

    copied = overlay_python_files(repo_path, vllm_path)
    print(f"  Overlaid {{copied}} Python files")

    # Verify import works
    result = subprocess.run(["python", "-c", "import vllm; print(vllm.__version__)"], capture_output=True, text=True, cwd="/tmp")
    if result.returncode != 0:
        raise RuntimeError(f"vLLM import failed after overlay: {{result.stderr}}")
    print(f"  vLLM version: {{result.stdout.strip()}}")

    # Use unique ports per phase to avoid port conflicts (vLLM 0.6.x bug)
    baseline_output, baseline_metrics = run_benchmark(PERF_COMMAND, MODEL, repo_path=repo_path, port=29001)
    results["baseline_metrics"] = baseline_metrics
    results["baseline_raw"] = baseline_output

    if not baseline_metrics:
        raise RuntimeError("Baseline benchmark produced no metrics")
    print(f"  Baseline metrics: {{baseline_metrics}}")

    # ========== HUMAN BENCHMARK ==========
    print("\\n[2/3] Running HUMAN benchmark...")
    print(f"  Restoring Python files from {{HUMAN_COMMIT[:8]}}...")

    # Checkout human commit and overlay
    success, human_repo_path = clone_and_checkout(HUMAN_COMMIT, "/tmp/vllm-human")
    if not success:
        raise RuntimeError(f"Failed to checkout human commit: {{human_repo_path}}")

    copied = overlay_python_files(human_repo_path, vllm_path)
    print(f"  Overlaid {{copied}} Python files")

    human_output, human_metrics = run_benchmark(PERF_COMMAND, MODEL, repo_path=human_repo_path, port=29002)
    results["human_metrics"] = human_metrics
    results["human_raw"] = human_output

    if not human_metrics:
        raise RuntimeError("Human benchmark produced no metrics")
    print(f"  Human metrics: {{human_metrics}}")

    # ========== AGENT BENCHMARK ==========
    if AGENT_PATCH:
        print("\\n[3/3] Running AGENT benchmark...")
        print(f"  Overlaying Python files from {{BASE_COMMIT[:8]}} + patch...")

        # Restore baseline
        success, agent_repo_path = clone_and_checkout(BASE_COMMIT, "/tmp/vllm-agent")
        if not success:
            raise RuntimeError(f"Failed to checkout base commit for agent: {{agent_repo_path}}")

        # Apply patch
        patch_file = "/tmp/agent.patch"
        with open(patch_file, "w") as f:
            f.write(AGENT_PATCH)

        result = subprocess.run(
            ["git", "apply", "--verbose", "agent.patch"],
            cwd=agent_repo_path + "/",
            capture_output=True, text=True,
            input=AGENT_PATCH
        )

        # Try alternative if git apply fails
        if result.returncode != 0:
            result = subprocess.run(
                ["patch", "-p1", "-i", patch_file],
                cwd=agent_repo_path,
                capture_output=True, text=True
            )

        if result.returncode != 0:
            results["agent_error"] = f"Patch apply failed: {{result.stderr}}"
            print(f"  Warning: {{results['agent_error']}}")
        else:
            copied = overlay_python_files(agent_repo_path, vllm_path)
            print(f"  Overlaid {{copied}} Python files (with patch)")

            agent_output, agent_metrics = run_benchmark(PERF_COMMAND, MODEL, repo_path=agent_repo_path, port=29003)
            results["agent_metrics"] = agent_metrics
            results["agent_raw"] = agent_output

            if agent_metrics:
                print(f"  Agent metrics: {{agent_metrics}}")
            else:
                error_snippet = agent_output[-2000:] if agent_output else "no output"
                results["agent_error"] = f"Agent benchmark produced no metrics. Output tail: {{error_snippet}}"
                print(f"  Agent benchmark failed - output tail:\\n{{error_snippet}}")
    else:
        print("\\n[3/3] No agent patch provided, skipping")

    results["status"] = "success"

except Exception as e:
    results["error"] = str(e)
    print(f"ERROR: {{e}}")

# Output results as JSON
print("\\n=== BENCHMARK_RESULTS_JSON ===")
print(json.dumps(results))
print("=== END_BENCHMARK_RESULTS_JSON ===")
'''

    try:
        # Determine GPU count from config
        gpu_count = 1
        if ":" in gpu_config:
            gpu_count = int(gpu_config.split(":")[1])
        gpu_type = gpu_config.split(":")[0] if ":" in gpu_config else gpu_config

        print(f"Creating Modal Sandbox with {gpu_type} x {gpu_count}...")

        # Get App reference for Sandbox (required when running outside Modal container)
        sandbox_app = modal.App.lookup("omniperf-benchmark", create_if_missing=True)

        # Create sandbox with the Docker image (verbose=True for detailed logging)
        sb = modal.Sandbox.create(
            image=docker_image,
            gpu=f"{gpu_type}:{gpu_count}" if gpu_count > 1 else gpu_type,
            timeout=10800,  # 3 hours
            secrets=[modal.Secret.from_name("huggingface-secret")],
            app=sandbox_app,
            verbose=True,  # Enable detailed sandbox logging
        )

        print("Running 3-way benchmark in sandbox...")

        # Write the benchmark script to a file in the sandbox using Modal's file API
        script_path = "/tmp/benchmark_3way.py"

        print(f"Writing benchmark script to sandbox ({len(benchmark_script)} chars)...")
        f = sb.open(script_path, "w")
        f.write(benchmark_script)
        f.close()

        print(f"Benchmark script written to {script_path}")

        # Execute the benchmark script from the file
        # Use -u for unbuffered stdout to get real-time output
        exec_result = sb.exec("python", "-u", script_path)

        # Wait for process to complete and collect output
        stdout_lines = []
        for line in exec_result.stdout:
            stdout_lines.append(line)
            print(f"[SANDBOX] {line.rstrip()}")  # Real-time output for debugging

        # Wait for process to fully complete
        exec_result.wait()
        return_code = exec_result.returncode

        # Collect stderr
        stderr_lines = []
        for line in exec_result.stderr:
            stderr_lines.append(line)

        stdout = "".join(stdout_lines)
        stderr = "".join(stderr_lines)

        print(f"\nSandbox process completed with return code: {return_code}")
        print(f"Sandbox stdout (last 1000 chars): {stdout[-1000:]}")
        if stderr:
            print(f"Sandbox stderr (last 500 chars): {stderr[-500:]}")

        # Parse results from JSON output
        import json as json_module  # Ensure json is available in this scope
        if "=== BENCHMARK_RESULTS_JSON ===" in stdout:
            json_str = stdout.split("=== BENCHMARK_RESULTS_JSON ===")[1].split("=== END_BENCHMARK_RESULTS_JSON ===")[0].strip()
            benchmark_results = json_module.loads(json_str)

            result["baseline_metrics"] = benchmark_results.get("baseline_metrics", {})
            result["human_metrics"] = benchmark_results.get("human_metrics", {})
            result["agent_metrics"] = benchmark_results.get("agent_metrics")
            result["baseline_raw"] = benchmark_results.get("baseline_raw", "")
            result["human_raw"] = benchmark_results.get("human_raw", "")
            result["agent_raw"] = benchmark_results.get("agent_raw", "")

            if benchmark_results.get("error"):
                result["error"] = benchmark_results["error"]

            if benchmark_results.get("agent_error"):
                result["agent_error"] = benchmark_results["agent_error"]

            # Compute improvements
            if result["baseline_metrics"] and result["human_metrics"]:
                result["human_improvement"] = compute_improvement(
                    result["baseline_metrics"], result["human_metrics"]
                )

                if result["agent_metrics"]:
                    result["agent_improvement"] = compute_improvement(
                        result["baseline_metrics"], result["agent_metrics"]
                    )
                    result["agent_vs_human"] = compute_improvement(
                        result["human_metrics"], result["agent_metrics"]
                    )

                result["status"] = "success"
        else:
            result["error"] = f"No JSON results in output. Stdout: {stdout[-2000:]}"

        # Terminate sandbox
        sb.terminate()

    except Exception as e:
        result["error"] = f"Docker fallback failed: {str(e)}"
        import traceback
        print(f"Exception traceback: {traceback.format_exc()}")

    finally:
        result["duration_s"] = time.time() - start_time

    return result


# Helper function for local orchestration
def run_modal_benchmark(
    wheel_url: str,
    perf_command: str,
    model: str,
    benchmark_type: str = "serving",
    gpu_config: str = None,
) -> Dict[str, Any]:
    """
    Run a benchmark on Modal with automatic GPU selection.

    This function should be called from the local runner to dispatch
    benchmarks to Modal cloud GPUs.

    Args:
        wheel_url: URL to vLLM wheel on wheels.vllm.ai
        perf_command: The benchmark command to execute
        model: Model name (used for GPU selection)
        benchmark_type: "serving", "throughput", or "latency"
        gpu_config: Override GPU config (e.g., "H100:4")

    Returns:
        Benchmark results dict with metrics and status
    """
    # Determine GPU configuration
    if gpu_config is None:
        gpu_config = get_gpu_config(model, perf_command)

    # Use Function.lookup to find deployed functions
    # This is required when calling from outside the Modal app context
    if gpu_config == "H100:8":
        fn = modal.Function.from_name("omniperf-benchmark", "run_benchmark_8gpu")
        return fn.remote(
            wheel_url=wheel_url,
            perf_command=perf_command,
            model=model,
            benchmark_type=benchmark_type,
        )
    elif gpu_config == "H100:4":
        fn = modal.Function.from_name("omniperf-benchmark", "run_benchmark_4gpu")
        return fn.remote(
            wheel_url=wheel_url,
            perf_command=perf_command,
            model=model,
            benchmark_type=benchmark_type,
        )
    else:
        fn = modal.Function.from_name("omniperf-benchmark", "run_benchmark_single_gpu")
        return fn.remote(
            wheel_url=wheel_url,
            perf_command=perf_command,
            model=model,
            benchmark_type=benchmark_type,
        )


@app.local_entrypoint()
def main(
    wheel_url: str = "",
    perf_command: str = "",
    model: str = "",
    benchmark_type: str = "serving",
):
    """CLI entrypoint for testing Modal deployment."""
    if not all([wheel_url, perf_command, model]):
        print("Usage: modal run modal_benchmark.py --wheel-url URL --perf-command CMD --model MODEL")
        return

    gpu_config = get_gpu_config(model, perf_command)
    print(f"Selected GPU config: {gpu_config}")

    result = run_modal_benchmark(
        wheel_url=wheel_url,
        perf_command=perf_command,
        model=model,
        benchmark_type=benchmark_type,
        gpu_config=gpu_config,
    )

    print(f"Status: {result['status']}")
    print(f"Metrics: {result.get('metrics', {})}")
    if result.get('error'):
        print(f"Error: {result['error']}")
    if result.get('raw_output'):
        print(f"Raw output:\n{result['raw_output']}")
