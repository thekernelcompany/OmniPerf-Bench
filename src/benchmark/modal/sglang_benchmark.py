"""
Modal-based benchmark runner for SGLang OmniPerf.

This module provides cloud GPU execution for SGLang benchmarks using Docker images.
Each phase (human, baseline, agent) uses its own pre-built Docker image from DockerHub.

Docker images: ayushnangia16/sglang-docker:{commit_hash}
- Human phase: Uses human commit image (has optimization)
- Baseline phase: Uses base commit image (pre-optimization)
- Agent phase: Uses base commit image + applies agent patch

Usage:
    # Deploy the Modal app
    modal deploy src/benchmark/modal/sglang_benchmark.py

    # Run a 3-way benchmark
    modal run src/benchmark/modal/sglang_benchmark.py::run_3way_benchmark_docker \\
        --human-commit "abc123" --base-commit "def456" \\
        --perf-command "..." --model "..."
"""

import modal
import subprocess
import time
import re
import os
import sys
import json
import urllib.request
import urllib.error
from typing import Dict, Optional, Any, Tuple, List
from pathlib import Path
from functools import lru_cache

# Modal app configuration
app = modal.App("sglang-benchmark")

# Docker image repository for SGLang commits
SGLANG_DOCKER_REPO = "ayushnangia16/sglang-docker" # do not change
SGLANG_REPO_URL = "https://github.com/sgl-project/sglang.git"

# Volume for caching models
model_cache = modal.Volume.from_name("sglang-model-cache", create_if_missing=True)

# Volume for caching SGLang wheel builds (similar to vLLM)
build_cache = modal.Volume.from_name("sglang-build-cache", create_if_missing=True)

# Volume for storing benchmark results
results_volume = modal.Volume.from_name("sglang-benchmark-results", create_if_missing=True)

# GPU configurations
GPU_CONFIGS = {
    "H100:1": {"gpu": "H100", "count": 1, "timeout": 21600},
    "H100:2": {"gpu": "H100", "count": 2, "timeout": 21600},
    "H100:4": {"gpu": "H100", "count": 4, "timeout": 21600},
    "H100:8": {"gpu": "H100", "count": 8, "timeout": 21600},
}

# CPU configuration for wheel building (no GPU needed for CUDA compilation)
CPU_BUILD_CONFIG = {
    "cpu": 16,          # 16 CPUs for parallel compilation
    "memory": 131072,   # 128GB RAM
    "timeout": 7200,    # 2 hour timeout (SGLang + flashinfer can take a while)
}

# Target CUDA architecture for H100
CUDA_BUILD_TARGETS = "9.0"  # H100 = SM 9.0

# Base image for wheel building (CPU-only, with CUDA toolkit for nvcc)
sglang_build_image = (
    modal.Image.from_registry("nvidia/cuda:12.4.0-devel-ubuntu22.04", add_python="3.11")
    .apt_install(
        "git", "curl", "wget", "patch",
        "build-essential", "cmake", "ninja-build",
        "libnuma-dev",
    )
    .pip_install(
        "torch==2.4.0",
        "numpy<2.0",
        "packaging",
        "setuptools>=61",
        "setuptools-scm>=8",
        "wheel",
        "build",
        "ninja",
    )
    .run_commands([
        # Install uv for faster package management
        "curl -LsSf https://astral.sh/uv/install.sh | sh",
        "ln -sf /root/.local/bin/uv /usr/local/bin/uv",
        # Install cmake 3.28+ (required for modern CUDA builds)
        "apt-get remove -y cmake || true",
        "curl -LO https://github.com/Kitware/CMake/releases/download/v3.28.3/cmake-3.28.3-linux-x86_64.tar.gz",
        "tar -xzf cmake-3.28.3-linux-x86_64.tar.gz -C /usr/local --strip-components=1",
        "rm cmake-3.28.3-linux-x86_64.tar.gz",
    ])
    .env({
        "CUDA_HOME": "/usr/local/cuda",
        "PATH": "/usr/local/cuda/bin:$PATH",
        "LD_LIBRARY_PATH": "/usr/local/cuda/lib64:$LD_LIBRARY_PATH",
    })
)

# Default ports for SGLang servers
BASELINE_PORT = 30001
HUMAN_PORT = 30002
AGENT_PORT = 30003


# =====================================================================
# Docker Image Management
# =====================================================================

def has_prebuilt_image(commit: str) -> bool:
    """Check if a Docker image exists for this commit on DockerHub."""
    # Normalize commit hash (check both short and full)
    for tag in [commit[:40], commit[:12], commit[:8]]:
        url = f"https://hub.docker.com/v2/repositories/{SGLANG_DOCKER_REPO}/tags/{tag}"
        try:
            req = urllib.request.Request(url, method='HEAD')
            urllib.request.urlopen(req, timeout=10)
            return True
        except urllib.error.HTTPError as e:
            if e.code == 404:
                continue
        except Exception:
            continue
    return False


def get_prebuilt_commit(commit: str) -> Optional[str]:
    """Get the full commit hash that has a Docker image."""
    # Try different tag formats
    for tag_len in [40, 12, 8]:
        tag = commit[:tag_len]
        url = f"https://hub.docker.com/v2/repositories/{SGLANG_DOCKER_REPO}/tags/{tag}"
        try:
            req = urllib.request.Request(url, method='HEAD')
            urllib.request.urlopen(req, timeout=10)
            return tag
        except:
            continue
    return None


def get_prebuilt_image(commit: str) -> Optional[modal.Image]:
    """Get Modal Image from pre-built Docker image."""
    tag = get_prebuilt_commit(commit)
    if tag is None:
        return None

    docker_image = f"{SGLANG_DOCKER_REPO}:{tag}"

    try:
        image = (
            modal.Image.from_registry(docker_image)
            .apt_install("git", "curl", "patch", "psmisc", "net-tools")
            .pip_install("datasets", "pandas", "tqdm", "aiohttp", "requests")
            .env({
                "HF_HOME": "/root/.cache/huggingface",
                "TRANSFORMERS_CACHE": "/root/.cache/huggingface",
            })
        )
        return image
    except Exception as e:
        print(f"Failed to create image from {docker_image}: {e}")
        return None


# =====================================================================
# Wheel Build System (similar to vLLM)
# =====================================================================

@app.function(
    image=sglang_build_image,
    cpu=CPU_BUILD_CONFIG["cpu"],
    memory=CPU_BUILD_CONFIG["memory"],
    timeout=CPU_BUILD_CONFIG["timeout"],
    volumes={"/cache": build_cache},
)
def build_sglang_cpu_only(commit_hash: str, force_rebuild: bool = False) -> Dict[str, Any]:
    """
    Build SGLang wheel on CPU-only instance (no GPU needed for CUDA compilation).

    Similar to vLLM's build approach - nvcc cross-compiles for target GPU architecture.
    Wheels are cached in Modal volume for fast reuse.

    Args:
        commit_hash: SGLang commit to build
        force_rebuild: If True, rebuild even if wheel exists in cache

    Returns:
        Dict with success, wheel_path, version, cache_hit, error
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

    # Check if wheel already exists in cache
    wheel_pattern = f"{wheel_dir}/sglang-*{commit_hash[:8]}*.whl"
    existing_wheels = glob_mod.glob(wheel_pattern)

    # Also check for wheels without commit in name (setuptools-scm style)
    if not existing_wheels:
        all_wheels = glob_mod.glob(f"{wheel_dir}/sglang-*.whl")
        # Look for marker file to match commit
        for wheel in all_wheels:
            wheel_marker = wheel + f".{commit_hash[:8]}"
            if os.path.exists(wheel_marker):
                existing_wheels = [wheel]
                break

    if existing_wheels and not force_rebuild:
        wheel_path = existing_wheels[0]
        wheel_name = os.path.basename(wheel_path)
        version = wheel_name.split("-")[1] if "-" in wheel_name else "unknown"
        print(f"[SGLANG BUILD] Wheel cache HIT: {wheel_name}")
        result["success"] = True
        result["cache_hit"] = True
        result["version"] = version
        result["wheel_path"] = wheel_path
        return result

    print(f"[SGLANG BUILD] Wheel cache MISS - Building SGLang {commit_hash[:8]}...")
    print(f"[SGLANG BUILD] Using {CPU_BUILD_CONFIG['cpu']} CPUs, {CPU_BUILD_CONFIG['memory']}MB RAM")

    # Clean up any partial build
    if os.path.exists(build_path):
        shutil.rmtree(build_path)
    os.makedirs(build_path, exist_ok=True)

    try:
        # Clone SGLang repository
        print(f"[SGLANG BUILD] Cloning SGLang repository...")
        clone_result = subprocess.run(
            ["git", "clone", "--depth", "100", SGLANG_REPO_URL, build_path],
            capture_output=True, text=True, timeout=600,
        )
        if clone_result.returncode != 0:
            result["error"] = f"Git clone failed: {clone_result.stderr[:500]}"
            return result

        # Fetch and checkout the commit
        print(f"[SGLANG BUILD] Checking out {commit_hash[:8]}...")
        subprocess.run(
            ["git", "fetch", "origin", commit_hash],
            cwd=build_path, capture_output=True, timeout=120,
        )
        checkout_result = subprocess.run(
            ["git", "checkout", commit_hash],
            cwd=build_path, capture_output=True, text=True, timeout=60,
        )
        if checkout_result.returncode != 0:
            result["error"] = f"Git checkout failed: {checkout_result.stderr[:500]}"
            shutil.rmtree(build_path, ignore_errors=True)
            return result

        # Build wheel
        print(f"[SGLANG BUILD] Building SGLang wheel (target: H100/SM9.0)...")
        env = os.environ.copy()
        env["MAX_JOBS"] = str(CPU_BUILD_CONFIG["cpu"])
        env["TORCH_CUDA_ARCH_LIST"] = CUDA_BUILD_TARGETS
        env["CMAKE_CUDA_ARCHITECTURES"] = "90"
        env["CUDA_VISIBLE_DEVICES"] = ""  # Prevent CUDA runtime from detecting devices

        # SGLang's python package is in the python/ subdirectory
        python_dir = os.path.join(build_path, "python")

        build_result = subprocess.run(
            ["python", "-m", "pip", "wheel", ".", "--no-deps", "--no-build-isolation", "-w", wheel_dir, "-v"],
            cwd=python_dir,
            timeout=7200,
            env=env,
        )

        if build_result.returncode != 0:
            result["error"] = f"Wheel build failed with return code {build_result.returncode}"
            return result

        # Find the built wheel
        new_wheels = glob_mod.glob(f"{wheel_dir}/sglang-*.whl")
        if not new_wheels:
            result["error"] = "No wheel file produced"
            return result

        wheel_path = max(new_wheels, key=os.path.getctime)
        wheel_name = os.path.basename(wheel_path)
        version = wheel_name.split("-")[1] if "-" in wheel_name else "unknown"

        # CRITICAL: Rename wheel to include commit hash
        # This prevents overwriting when building different commits with same version
        # E.g., sglang-0.4.7-py3-none-any.whl -> sglang-0.4.7+021f76e4-py3-none-any.whl
        commit_short = commit_hash[:8]
        if f"+{commit_short}" not in wheel_name:
            new_wheel_name = wheel_name.replace(
                f"-{version}-",
                f"-{version}+{commit_short}-"
            )
            new_wheel_path = os.path.join(wheel_dir, new_wheel_name)
            os.rename(wheel_path, new_wheel_path)
            wheel_path = new_wheel_path
            wheel_name = new_wheel_name
            print(f"[SGLANG BUILD] Renamed wheel to include commit hash: {wheel_name}")

        # Create marker file to associate wheel with commit (backup method)
        with open(f"{wheel_path}.{commit_hash[:8]}", "w") as f:
            f.write(commit_hash)

        # Keep build directory for incremental builds
        with open(marker_file, "w") as f:
            f.write(f"wheel={wheel_name}\nversion={version}\ncommit={commit_hash}\n")

        # Commit volume to persist
        print(f"[SGLANG BUILD] Committing wheel to Modal volume...")
        build_cache.commit()

        print(f"[SGLANG BUILD] SUCCESS: Built wheel {wheel_name}")
        result["success"] = True
        result["cache_hit"] = False
        result["version"] = version
        result["wheel_path"] = wheel_path
        return result

    except subprocess.TimeoutExpired:
        result["error"] = "Build timed out (exceeded 2 hours)"
        return result
    except Exception as e:
        result["error"] = f"Build error: {str(e)}"
        return result


@app.function(
    image=sglang_build_image,
    cpu=CPU_BUILD_CONFIG["cpu"],
    memory=CPU_BUILD_CONFIG["memory"],
    timeout=CPU_BUILD_CONFIG["timeout"],
    volumes={"/cache": build_cache},
)
def build_sglang_agent_patch(base_commit: str, patch_content: str) -> Dict[str, Any]:
    """
    Apply agent patch and build SGLang wheel on CPU.

    Uses incremental build if base commit was previously built.

    Args:
        base_commit: The base SGLang commit hash
        patch_content: Unified diff patch from agent

    Returns:
        Dict with success, wheel_path, patch_hash, error
    """
    import os
    import subprocess
    import shutil
    import hashlib
    import glob as glob_mod
    import fcntl

    patch_hash = hashlib.sha256(patch_content.encode()).hexdigest()[:12]
    result = {
        "success": False,
        "wheel_path": None,
        "patch_hash": patch_hash,
        "error": None,
    }

    cache_dir = "/cache"
    wheel_dir = f"{cache_dir}/wheels"
    build_path = f"{cache_dir}/build_{base_commit}"
    patched_wheel_pattern = f"{wheel_dir}/sglang-*+patch{patch_hash}*.whl"

    # Check if patched wheel already exists
    existing = glob_mod.glob(patched_wheel_pattern)
    if existing:
        print(f"[SGLANG PATCH] Patched wheel cache HIT")
        result["success"] = True
        result["wheel_path"] = existing[0]
        return result

    # Check if base build exists
    if not os.path.exists(f"{build_path}/.build_complete"):
        result["error"] = f"Base build not found for {base_commit[:8]}. Run build_sglang_cpu_only first."
        return result

    lock_file = f"{build_path}/.build_lock"
    os.makedirs(wheel_dir, exist_ok=True)

    try:
        # Acquire lock on build directory
        lock_fd = open(lock_file, "w")
        fcntl.flock(lock_fd, fcntl.LOCK_EX)
        print(f"[SGLANG PATCH] Acquired lock on build directory")

        python_dir = os.path.join(build_path, "python")

        # Write and apply patch
        patch_file = f"/tmp/agent_{patch_hash}.patch"
        with open(patch_file, "w") as f:
            f.write(patch_content)

        print(f"[SGLANG PATCH] Applying patch ({len(patch_content)} bytes)...")
        apply_result = subprocess.run(
            ["git", "apply", "--verbose", patch_file],
            cwd=build_path, capture_output=True, text=True,
        )

        if apply_result.returncode != 0:
            # Try 3-way merge
            apply_result = subprocess.run(
                ["git", "apply", "--3way", patch_file],
                cwd=build_path, capture_output=True, text=True,
            )

        if apply_result.returncode != 0:
            result["error"] = f"Patch apply failed: {apply_result.stderr[:500]}"
            return result

        # Build patched wheel
        print(f"[SGLANG PATCH] Building patched wheel (incremental)...")
        env = os.environ.copy()
        env["MAX_JOBS"] = str(CPU_BUILD_CONFIG["cpu"])
        env["TORCH_CUDA_ARCH_LIST"] = CUDA_BUILD_TARGETS
        env["CMAKE_CUDA_ARCHITECTURES"] = "90"
        env["CUDA_VISIBLE_DEVICES"] = ""
        env["SGLANG_LOCAL_VERSION"] = f"patch{patch_hash}"

        build_result = subprocess.run(
            ["python", "-m", "pip", "wheel", ".", "--no-deps", "--no-build-isolation", "-w", wheel_dir, "-v"],
            cwd=python_dir,
            timeout=3600,  # Shorter timeout for incremental build
            env=env,
        )

        # Revert patch before checking result
        print(f"[SGLANG PATCH] Reverting patch...")
        subprocess.run(["git", "checkout", "."], cwd=build_path, capture_output=True)

        if build_result.returncode != 0:
            result["error"] = f"Patched build failed with return code {build_result.returncode}"
            return result

        # Find the patched wheel
        new_wheels = glob_mod.glob(f"{wheel_dir}/sglang-*patch{patch_hash}*.whl")
        if not new_wheels:
            # Fallback: find most recent wheel
            new_wheels = glob_mod.glob(f"{wheel_dir}/sglang-*.whl")

        if not new_wheels:
            result["error"] = "No patched wheel file produced"
            return result

        wheel_path = max(new_wheels, key=os.path.getctime)

        # Commit volume
        print(f"[SGLANG PATCH] Committing patched wheel...")
        build_cache.commit()

        print(f"[SGLANG PATCH] SUCCESS: Built patched wheel")
        result["success"] = True
        result["wheel_path"] = wheel_path
        return result

    except Exception as e:
        result["error"] = f"Patch build error: {str(e)}"
        # Try to revert patch
        subprocess.run(["git", "checkout", "."], cwd=build_path, capture_output=True)
        return result
    finally:
        try:
            fcntl.flock(lock_fd, fcntl.LOCK_UN)
            lock_fd.close()
        except:
            pass


def check_wheel_exists(commit: str) -> bool:
    """Check if a wheel exists in cache for this commit."""
    # This would require calling the Modal function, so we'll do a simple check
    # In practice, the build function handles cache checking
    return False


# =====================================================================
# Wheel-based 3-Way Benchmark
# =====================================================================

# Runtime image for wheel-based benchmarks (GPU instance)
# NOTE: Install all SGLang dependencies since we use --no-deps when installing the wheel
# Use torch 2.5.1 to match sgl-kernel's compiled ABI (PyPI sgl-kernel is built with torch 2.5+)
# Uses uv for fast package installation
sglang_runtime_image = (
    modal.Image.from_registry("nvidia/cuda:12.4.0-runtime-ubuntu22.04", add_python="3.11")
    .apt_install("git", "curl", "psmisc", "net-tools", "lsof", "libnuma-dev", "libnuma1")
    .run_commands([
        # Install uv first for fast package management
        "curl -LsSf https://astral.sh/uv/install.sh | sh",
        "ln -sf /root/.local/bin/uv /usr/local/bin/uv",
        # Install all dependencies with uv (much faster than pip)
        "uv pip install --system "
        "torch==2.5.1 "
        "'transformers>=4.40.0' "
        "'huggingface_hub>=0.23.0' "
        "'tokenizers>=0.19.0' "
        "'accelerate>=0.30.0' "
        "'numpy<2.0' "
        "requests aiohttp fastapi uvicorn httpx 'pydantic>=2.0' starlette "
        "datasets pandas tqdm packaging filelock regex safetensors sentencepiece "
        "ipython 'triton>=3.0.0' interegular lark 'outlines>=0.0.34' xgrammar "
        "pillow einops zmq pyzmq msgspec orjson prometheus-client psutil pynvml "
        "setproctitle rpyc cloudpickle 'ray>=2.9' 'vllm>=0.5.0'",
        # Install flashinfer (SGLang's attention backend) - version compatible with torch 2.5
        "uv pip install --system flashinfer -i https://flashinfer.ai/whl/cu124/torch2.5/ || true",
        # Install sgl_kernel (SGLang's CUDA kernels) - try from PyPI
        # Note: May have ABI issues if compiled with different torch version
        # SGLang will fall back to flashinfer if sgl_kernel import fails
        "uv pip install --system sgl-kernel || true",
    ])
    .env({
        "HF_HOME": "/root/.cache/huggingface",
        "TRANSFORMERS_CACHE": "/root/.cache/huggingface",
    })
)


def _extract_tp_from_command(command: str) -> int:
    """Extract tensor parallel size from benchmark command."""
    tp_patterns = [
        r'--tp[=\s]+(\d+)',
        r'--tp-size[=\s]+(\d+)',
        r'--tensor-parallel-size[=\s]+(\d+)',
        r'-tp[=\s]+(\d+)',
    ]
    for pattern in tp_patterns:
        match = re.search(pattern, command)
        if match:
            return int(match.group(1))
    return 1


def _extract_server_args(command: str) -> List[str]:
    """Extract server-relevant args from perf_command."""
    args = []
    if "--trust-remote-code" in command:
        args.append("--trust-remote-code")
    dtype_match = re.search(r'--dtype[=\s]+(\S+)', command)
    if dtype_match:
        args.extend(["--dtype", dtype_match.group(1)])
    quant_match = re.search(r'--quantization[=\s]+(\S+)', command)
    if quant_match:
        args.extend(["--quantization", quant_match.group(1)])
    if "--disable-radix-cache" in command:
        args.append("--disable-radix-cache")
    return args


def _run_3way_benchmark_wheel_impl(
    baseline_commit: str,
    human_commit: str,
    agent_patch: Optional[str],
    perf_command: str,
    model: str,
    gpu_count: int = 1,
) -> Dict[str, Any]:
    """
    Implementation for wheel-based 3-way benchmark.

    This approach:
    1. Installs baseline wheel → runs benchmark
    2. Installs human wheel → runs benchmark
    3. Installs agent wheel (patched) → runs benchmark

    Much more reliable than Docker image + Python overlay because
    wheels include ALL code (Python + compiled extensions).
    """
    import subprocess
    import time
    import re
    import os
    import signal
    import glob as glob_mod

    result = {
        "status": "error",
        "baseline_metrics": {},
        "human_metrics": {},
        "agent_metrics": None,
        "human_improvement": {},
        "agent_improvement": None,
        "error": None,
        "install_method": "wheel",
        "gpu_count": gpu_count,
    }

    wheel_dir = "/cache/wheels"
    start_time = time.time()

    # Extract TP size from command (default to gpu_count if not specified)
    tp_size = _extract_tp_from_command(perf_command)
    if tp_size == 1 and gpu_count > 1:
        tp_size = gpu_count
    extra_server_args = _extract_server_args(perf_command)

    print(f"GPU count: {gpu_count}, TP size: {tp_size}")

    def find_wheel(commit: str) -> Optional[str]:
        """Find wheel for a commit in cache."""
        # Try direct match
        pattern = f"{wheel_dir}/sglang-*{commit[:8]}*.whl"
        matches = glob_mod.glob(pattern)
        if matches:
            return matches[0]
        # Try marker file match
        for wheel in glob_mod.glob(f"{wheel_dir}/sglang-*.whl"):
            marker = f"{wheel}.{commit[:8]}"
            if os.path.exists(marker):
                return wheel
        return None

    def install_wheel(wheel_path: str) -> Tuple[bool, str]:
        """Install SGLang wheel."""
        print(f"Installing wheel: {os.path.basename(wheel_path)}")
        install_result = subprocess.run(
            ["uv", "pip", "install", "--system", "--force-reinstall", "--no-deps", wheel_path],
            capture_output=True, text=True, timeout=300,
        )
        if install_result.returncode != 0:
            return False, install_result.stderr[:500]
        # Get version
        ver_result = subprocess.run(
            ["python", "-c", "import sglang; print(getattr(sglang, '__version__', 'unknown'))"],
            capture_output=True, text=True,
        )
        return True, ver_result.stdout.strip()

    def start_server(port: int, model_path: str, tp: int = 1, extra_args: List[str] = None) -> Tuple[Optional[subprocess.Popen], bool, str]:
        """Start SGLang server with TP support."""
        cmd = [
            "python", "-m", "sglang.launch_server",
            "--model-path", model_path,
            "--port", str(port),
            "--host", "0.0.0.0",
            "--tp-size", str(tp),
            "--log-level", "info",
        ]
        if extra_args:
            cmd.extend(extra_args)

        print(f"Starting server on port {port} with TP={tp}...")
        print(f"  Command: {' '.join(cmd)}")
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, preexec_fn=os.setsid)

        # Wait for server ready
        logs = []
        start = time.time()
        while time.time() - start < 900:  # 15 min timeout for large models
            if proc.poll() is not None:
                remaining = proc.stdout.read()
                logs.append(remaining)
                return proc, False, "".join(logs)
            try:
                import requests
                resp = requests.get(f"http://localhost:{port}/health", timeout=2)
                if resp.status_code == 200:
                    print(f"Server ready in {time.time() - start:.1f}s")
                    return proc, True, "".join(logs)
            except:
                pass
            time.sleep(3)
        return proc, False, "".join(logs)

    def stop_server(proc):
        """Stop SGLang server."""
        if proc:
            try:
                os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
                proc.wait(timeout=10)
            except:
                try:
                    proc.kill()
                except:
                    pass

    def run_benchmark(port: int, command: str) -> Tuple[str, Dict]:
        """Run benchmark and parse results."""
        cmd = command
        if "--port" in cmd:
            cmd = re.sub(r'--port\s+\d+', f'--port {port}', cmd)
        else:
            cmd += f" --port {port}"
        if "--host" not in cmd and "--base-url" not in cmd:
            cmd += " --host 127.0.0.1"
        if "--num-prompt" not in cmd.lower():
            cmd += " --num-prompts 100"
        if "--warmup" not in cmd.lower():
            cmd += " --warmup-requests 2"
        if "--flush-cache" not in cmd:
            cmd += " --flush-cache"

        print(f"Running: {cmd[:100]}...")
        bench_result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=900)
        output = bench_result.stdout + "\n" + bench_result.stderr

        # Parse metrics
        metrics = {}
        patterns = {
            "request_throughput": r"Request throughput \(req/s\):\s+([\d.]+)",
            "output_throughput": r"Output token throughput \(tok/s\):\s+([\d.]+)",
            "input_throughput": r"Input token throughput \(tok/s\):\s+([\d.]+)",
            "ttft_mean": r"Mean TTFT \(ms\):\s+([\d.]+)",
            "ttft_median": r"Median TTFT \(ms\):\s+([\d.]+)",
            "ttft_p99": r"P99 TTFT \(ms\):\s+([\d.]+)",
            "tpot_mean": r"Mean TPOT \(ms\):\s+([\d.]+)",
            "tpot_median": r"Median TPOT \(ms\):\s+([\d.]+)",
            "itl_mean": r"Mean ITL \(ms\):\s+([\d.]+)",
            "e2e_latency_mean": r"Mean E2E Latency \(ms\):\s+([\d.]+)",
        }
        for name, pattern in patterns.items():
            match = re.search(pattern, output)
            if match:
                metrics[name] = float(match.group(1))

        return output, metrics

    try:
        # Find wheels
        baseline_wheel = find_wheel(baseline_commit)
        human_wheel = find_wheel(human_commit)

        if not baseline_wheel:
            result["error"] = f"Baseline wheel not found for {baseline_commit[:8]}. Build it first."
            return result
        if not human_wheel:
            result["error"] = f"Human wheel not found for {human_commit[:8]}. Build it first."
            return result

        # 1. BASELINE
        print("=" * 60)
        print("[1/3] BASELINE benchmark")
        print("=" * 60)
        success, version = install_wheel(baseline_wheel)
        if not success:
            result["error"] = f"Baseline wheel install failed: {version}"
            return result
        print(f"Installed baseline: {version}")

        proc, ready, logs = start_server(BASELINE_PORT, model, tp_size, extra_server_args)
        if not ready:
            result["error"] = f"Baseline server failed: {logs[-500:]}"
            stop_server(proc)
            return result

        output, metrics = run_benchmark(BASELINE_PORT, perf_command)
        stop_server(proc)
        result["baseline_metrics"] = metrics
        print(f"Baseline metrics: {metrics}")

        # 2. HUMAN
        print("=" * 60)
        print("[2/3] HUMAN benchmark")
        print("=" * 60)
        success, version = install_wheel(human_wheel)
        if not success:
            result["error"] = f"Human wheel install failed: {version}"
            return result
        print(f"Installed human: {version}")

        proc, ready, logs = start_server(HUMAN_PORT, model, tp_size, extra_server_args)
        if not ready:
            result["error"] = f"Human server failed: {logs[-500:]}"
            stop_server(proc)
            return result

        output, metrics = run_benchmark(HUMAN_PORT, perf_command)
        stop_server(proc)
        result["human_metrics"] = metrics
        print(f"Human metrics: {metrics}")

        # 3. AGENT (if patch provided)
        if agent_patch:
            print("=" * 60)
            print("[3/3] AGENT benchmark")
            print("=" * 60)
            # Find agent wheel (built from patch)
            import hashlib
            patch_hash = hashlib.sha256(agent_patch.encode()).hexdigest()[:12]
            agent_wheels = glob_mod.glob(f"{wheel_dir}/sglang-*patch{patch_hash}*.whl")
            if not agent_wheels:
                result["agent_error"] = f"Agent wheel not found for patch {patch_hash}"
                print(f"WARNING: Agent wheel not found")
            else:
                agent_wheel = agent_wheels[0]
                success, version = install_wheel(agent_wheel)
                if not success:
                    result["agent_error"] = f"Agent wheel install failed"
                else:
                    print(f"Installed agent: {version}")
                    proc, ready, logs = start_server(AGENT_PORT, model, tp_size, extra_server_args)
                    if not ready:
                        result["agent_error"] = f"Agent server failed"
                        stop_server(proc)
                    else:
                        output, metrics = run_benchmark(AGENT_PORT, perf_command)
                        stop_server(proc)
                        result["agent_metrics"] = metrics
                        print(f"Agent metrics: {metrics}")

        result["status"] = "success"
        result["duration_s"] = time.time() - start_time

        # Calculate improvements
        if result["baseline_metrics"] and result["human_metrics"]:
            for k in result["baseline_metrics"]:
                if k in result["human_metrics"] and result["baseline_metrics"][k] > 0:
                    base_v = result["baseline_metrics"][k]
                    human_v = result["human_metrics"][k]
                    if "throughput" in k:
                        result["human_improvement"][k] = (human_v - base_v) / base_v * 100
                    else:
                        result["human_improvement"][k] = (base_v - human_v) / base_v * 100

        if result["baseline_metrics"] and result.get("agent_metrics"):
            agent_imp = {}
            for k in result["baseline_metrics"]:
                if k in result["agent_metrics"] and result["baseline_metrics"][k] > 0:
                    base_v = result["baseline_metrics"][k]
                    agent_v = result["agent_metrics"][k]
                    if "throughput" in k:
                        agent_imp[k] = (agent_v - base_v) / base_v * 100
                    else:
                        agent_imp[k] = (base_v - agent_v) / base_v * 100
            result["agent_improvement"] = agent_imp

        return result

    except Exception as e:
        result["error"] = str(e)
        result["duration_s"] = time.time() - start_time
        return result


# =====================================================================
# Multi-GPU Wheel Benchmark Functions
# =====================================================================

@app.function(
    image=sglang_runtime_image,
    gpu="H100",
    timeout=7200,
    volumes={
        "/root/.cache/huggingface": model_cache,
        "/cache": build_cache,
    },
    secrets=[modal.Secret.from_name("huggingface-secret")],
)
def run_3way_benchmark_wheel(
    baseline_commit: str,
    human_commit: str,
    agent_patch: Optional[str],
    perf_command: str,
    model: str,
) -> Dict[str, Any]:
    """Run 3-way benchmark using wheel installation (1 GPU)."""
    return _run_3way_benchmark_wheel_impl(
        baseline_commit, human_commit, agent_patch, perf_command, model, gpu_count=1
    )


@app.function(
    image=sglang_runtime_image,
    gpu="H100:2",
    timeout=7200,
    volumes={
        "/root/.cache/huggingface": model_cache,
        "/cache": build_cache,
    },
    secrets=[modal.Secret.from_name("huggingface-secret")],
)
def run_3way_benchmark_wheel_2gpu(
    baseline_commit: str,
    human_commit: str,
    agent_patch: Optional[str],
    perf_command: str,
    model: str,
) -> Dict[str, Any]:
    """Run 3-way benchmark using wheel installation (2 GPUs)."""
    return _run_3way_benchmark_wheel_impl(
        baseline_commit, human_commit, agent_patch, perf_command, model, gpu_count=2
    )


@app.function(
    image=sglang_runtime_image,
    gpu="H100:4",
    timeout=10800,  # 3 hours for larger models
    volumes={
        "/root/.cache/huggingface": model_cache,
        "/cache": build_cache,
    },
    secrets=[modal.Secret.from_name("huggingface-secret")],
)
def run_3way_benchmark_wheel_4gpu(
    baseline_commit: str,
    human_commit: str,
    agent_patch: Optional[str],
    perf_command: str,
    model: str,
) -> Dict[str, Any]:
    """Run 3-way benchmark using wheel installation (4 GPUs)."""
    return _run_3way_benchmark_wheel_impl(
        baseline_commit, human_commit, agent_patch, perf_command, model, gpu_count=4
    )


@app.function(
    image=sglang_runtime_image,
    gpu="H100:8",
    timeout=14400,  # 4 hours for largest models
    volumes={
        "/root/.cache/huggingface": model_cache,
        "/cache": build_cache,
    },
    secrets=[modal.Secret.from_name("huggingface-secret")],
)
def run_3way_benchmark_wheel_8gpu(
    baseline_commit: str,
    human_commit: str,
    agent_patch: Optional[str],
    perf_command: str,
    model: str,
) -> Dict[str, Any]:
    """Run 3-way benchmark using wheel installation (8 GPUs)."""
    return _run_3way_benchmark_wheel_impl(
        baseline_commit, human_commit, agent_patch, perf_command, model, gpu_count=8
    )


# =====================================================================
# Benchmark Execution Helpers
# =====================================================================

def parse_sglang_benchmark_output(output: str) -> Dict[str, float]:
    """
    Parse SGLang benchmark output to extract metrics.

    SGLang outputs similar format to vLLM:
        Mean TTFT (ms):                          xxx
        Median TTFT (ms):                        xxx
        Mean TPOT (ms):                          xxx
        Mean ITL (ms):                           xxx
    """
    metrics = {}

    patterns = {
        "ttft_mean": r"Mean TTFT \(ms\):\s+([\d.]+)",
        "ttft_median": r"Median TTFT \(ms\):\s+([\d.]+)",
        "ttft_p99": r"P99 TTFT \(ms\):\s+([\d.]+)",
        "tpot_mean": r"Mean TPOT \(ms\):\s+([\d.]+)",
        "tpot_median": r"Median TPOT \(ms\):\s+([\d.]+)",
        "tpot_p99": r"P99 TPOT \(ms\):\s+([\d.]+)",
        "itl_mean": r"Mean ITL \(ms\):\s+([\d.]+)",
        "itl_median": r"Median ITL \(ms\):\s+([\d.]+)",
        "itl_p99": r"P99 ITL \(ms\):\s+([\d.]+)",
        "throughput": r"Request throughput \(req/s\):\s+([\d.]+)",
        "output_throughput": r"Output token throughput \(tok/s\):\s+([\d.]+)",
    }

    for metric_name, pattern in patterns.items():
        match = re.search(pattern, output)
        if match:
            metrics[metric_name] = float(match.group(1))

    return metrics


def calculate_improvement(baseline: Dict, optimized: Dict) -> Dict[str, float]:
    """Calculate percentage improvement between baseline and optimized metrics."""
    improvement = {}
    for key in baseline:
        if key in optimized and baseline[key] > 0:
            baseline_val = baseline[key]
            optimized_val = optimized[key]
            # For latency metrics, lower is better (positive improvement = reduction)
            # For throughput metrics, higher is better (positive improvement = increase)
            if "throughput" in key:
                improvement[key] = (optimized_val - baseline_val) / baseline_val * 100
            else:
                improvement[key] = (baseline_val - optimized_val) / baseline_val * 100
    return improvement


# =====================================================================
# Docker-based 3-Way Benchmark (Primary Approach)
# =====================================================================

def run_3way_benchmark_docker(
    human_commit: str,
    base_commit: Optional[str],
    agent_patch: Optional[str],
    perf_command: str,
    model: str,
    gpu_config: str = "H100:1",
) -> Dict[str, Any]:
    """
    Run benchmark using Docker images.

    MODES:
    1. Human-only (base_commit=None): Only benchmark human commit using Docker image as-is
       - Most reliable mode since Docker image has all correct dependencies
       - baseline_metrics and agent_metrics will be empty

    2. 3-way (base_commit provided): Attempt baseline/human/agent comparison
       - WARNING: Baseline/agent will likely fail due to ABI mismatch
       - Docker image has compiled extensions for human commit
       - Overlaying Python from baseline/agent creates incompatibility

    Args:
        human_commit: Human commit hash (must have Docker image available)
        base_commit: Baseline/parent commit hash (None for human-only mode)
        agent_patch: Optional unified diff patch from agent
        perf_command: Benchmark command to run
        model: Model name/path
        gpu_config: GPU configuration (e.g., "H100:1")

    Returns:
        Dict with baseline_metrics, human_metrics, agent_metrics, status, error
    """
    # Enable Modal output for detailed logging
    modal.enable_output()

    # Determine benchmark mode
    human_only_mode = base_commit is None

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
        "install_method": "docker",
        "baseline_raw": "",
        "human_raw": "",
        "agent_raw": "",
        "benchmark_mode": "human_only" if human_only_mode else "3way",
    }

    start_time = time.time()

    # Check if Docker image exists for human commit
    if not has_prebuilt_image(human_commit):
        result["error"] = f"No Docker image available for human commit {human_commit[:8]}"
        result["duration_s"] = time.time() - start_time
        return result

    full_human_commit = get_prebuilt_commit(human_commit)
    docker_image = f"{SGLANG_DOCKER_REPO}:{full_human_commit}"

    print(f"Using Docker image: {docker_image}")
    print(f"  Human commit: {human_commit[:8]}")
    print(f"  Mode: {'human-only' if human_only_mode else '3-way'}")
    if not human_only_mode:
        print(f"  Base commit: {base_commit[:8]}")

    # Parse GPU config
    gpu_cfg = GPU_CONFIGS.get(gpu_config, GPU_CONFIGS["H100:1"])

    # Create the benchmark script that runs inside the container
    if human_only_mode:
        benchmark_script = _create_human_only_benchmark_script(
            human_commit=human_commit,
            perf_command=perf_command,
            model=model,
        )
    else:
        benchmark_script = _create_benchmark_script(
            base_commit=base_commit,
            human_commit=human_commit,
            agent_patch=agent_patch,
            perf_command=perf_command,
            model=model,
        )

    try:
        # Create Modal Sandbox with the Docker image
        print(f"Creating Modal Sandbox with {gpu_cfg['gpu']} x {gpu_cfg['count']}...")

        # Note: SGLang Docker images don't need ENTRYPOINT/CMD clearing like vLLM
        # In fact, clearing them causes the container to crash immediately
        # setup_dockerfile_commands fixes Python symlink, typing_extensions, and missing libnuma
        image = modal.Image.from_registry(
            docker_image,
            force_build=True,
            setup_dockerfile_commands=[
                # Fix missing libnuma (required by sgl_kernel on newer SGLang versions)
                "RUN apt-get update && apt-get install -y libnuma1 libnuma-dev && rm -rf /var/lib/apt/lists/*",
                "RUN ln -sf $(which python3) /usr/local/bin/python || true",
                "RUN ln -sf $(which pip3) /usr/local/bin/pip || true",
                "RUN pip3 install 'typing_extensions>=4.10.0' --upgrade -q || true",
            ],
        )

        # Get App reference for Sandbox (required when running outside Modal container)
        sandbox_app = modal.App.lookup("sglang-benchmark", create_if_missing=True)

        sandbox = modal.Sandbox.create(
            app=sandbox_app,
            image=image,
            gpu=f"{gpu_cfg['gpu']}:{gpu_cfg['count']}" if gpu_cfg['count'] > 1 else gpu_cfg['gpu'],
            timeout=gpu_cfg['timeout'],
            volumes={"/root/.cache/huggingface": model_cache},
            secrets=[modal.Secret.from_name("huggingface-secret")],
            verbose=True,  # Enable detailed sandbox logging
        )

        print("Running 3-way benchmark in sandbox...")

        # Write benchmark script to sandbox using Modal's file API
        script_path = "/tmp/benchmark_3way.py"
        print(f"Writing benchmark script to sandbox ({len(benchmark_script)} chars)...")
        f = sandbox.open(script_path, "w")
        f.write(benchmark_script)
        f.close()

        # Verify script was written
        verify_proc = sandbox.exec("ls", "-la", script_path)
        for line in verify_proc.stdout:
            print(f"[VERIFY] {line.rstrip()}")
        verify_proc.wait()

        # Run benchmark script with unbuffered output
        # Use python3 instead of python for consistency
        print("Executing benchmark script...")
        proc = sandbox.exec("python3", "-u", script_path)

        # Collect output - interleave stdout and stderr
        stdout_lines = []
        stderr_lines = []

        for line in proc.stdout:
            stdout_lines.append(line)
            print(f"[SANDBOX] {line.rstrip()}")

        for line in proc.stderr:
            stderr_lines.append(line)
            print(f"[STDERR] {line.rstrip()}")

        proc.wait()
        return_code = proc.returncode
        print(f"Benchmark script completed with return code: {return_code}")

        # Print summary
        print(f"Stdout lines: {len(stdout_lines)}")
        print(f"Stderr lines: {len(stderr_lines)}")
        if stderr_lines:
            print(f"Stderr (last 1000 chars): {''.join(stderr_lines)[-1000:]}")

        sandbox.terminate()

        # Join lines, stripping trailing newlines to avoid double newlines
        full_output = "\n".join(line.rstrip('\n') for line in stdout_lines)

        # Parse results from output
        result = _parse_benchmark_results(full_output, result)

        # If no results found but we have stderr, include it in error
        if result.get("error") == "No benchmark results found in output" and stderr_lines:
            result["error"] = f"Script error: {''.join(stderr_lines)[-500:]}"

    except modal.exception.SandboxTimeoutError:
        result["error"] = f"Sandbox timeout after {gpu_cfg['timeout']}s"
    except Exception as e:
        result["error"] = f"Sandbox error: {str(e)}"

    result["duration_s"] = time.time() - start_time
    return result


def _create_human_only_benchmark_script(
    human_commit: str,
    perf_command: str,
    model: str,
) -> str:
    """
    Create a simple benchmark script for human-only mode.

    This script runs ONLY the human benchmark using the Docker image as-is.
    No Python overlay, no git operations, no baseline/agent phases.

    This is the most reliable approach since it uses exactly what the Docker image
    was built with, avoiding any ABI mismatches with compiled extensions.
    """

    script = f'''
import subprocess
import sys
import os
import json
import time
import re

# Configuration
HUMAN_COMMIT = "{human_commit}"
PERF_COMMAND = """{perf_command}"""
MODEL = "{model}"

HUMAN_PORT = 30002

# Results storage
results = {{
    "baseline_metrics": {{}},
    "human_metrics": {{}},
    "agent_metrics": None,
    "status": "error",
    "error": None,
    "baseline_raw": "",
    "human_raw": "",
    "agent_raw": "",
    "benchmark_mode": "human_only",
    "baseline_skip_reason": "Human-only mode - baseline requires separate Docker image",
    "agent_skip_reason": "Human-only mode - agent requires separate Docker image",
}}

def parse_benchmark_output(output: str):
    """Parse benchmark output to extract metrics."""
    metrics = {{}}

    patterns = {{
        "request_throughput": r"Request throughput \\(req/s\\):\\s+([\\d.]+)",
        "output_throughput": r"Output token throughput \\(tok/s\\):\\s+([\\d.]+)",
        "input_throughput": r"Input token throughput \\(tok/s\\):\\s+([\\d.]+)",
        "total_throughput": r"Total token throughput \\(tok/s\\):\\s+([\\d.]+)",
        "ttft_mean": r"Mean TTFT \\(ms\\):\\s+([\\d.]+)",
        "ttft_median": r"Median TTFT \\(ms\\):\\s+([\\d.]+)",
        "ttft_p99": r"P99 TTFT \\(ms\\):\\s+([\\d.]+)",
        "tpot_mean": r"Mean TPOT \\(ms\\):\\s+([\\d.]+)",
        "tpot_median": r"Median TPOT \\(ms\\):\\s+([\\d.]+)",
        "tpot_p99": r"P99 TPOT \\(ms\\):\\s+([\\d.]+)",
        "itl_mean": r"Mean ITL \\(ms\\):\\s+([\\d.]+)",
        "itl_median": r"Median ITL \\(ms\\):\\s+([\\d.]+)",
        "itl_p95": r"P95 ITL \\(ms\\):\\s+([\\d.]+)",
        "itl_p99": r"P99 ITL \\(ms\\):\\s+([\\d.]+)",
        "e2e_latency_mean": r"Mean E2E Latency \\(ms\\):\\s+([\\d.]+)",
        "e2e_latency_median": r"Median E2E Latency \\(ms\\):\\s+([\\d.]+)",
        "e2e_latency_p99": r"P99 E2E Latency \\(ms\\):\\s+([\\d.]+)",
        "total_input_tokens": r"Total input tokens:\\s+([\\d.]+)",
        "total_output_tokens": r"Total generated tokens:\\s+([\\d.]+)",
    }}

    for name, pattern in patterns.items():
        match = re.search(pattern, output, re.IGNORECASE)
        if match:
            metrics[name] = float(match.group(1))

    return metrics

def extract_tp_from_command(command: str) -> int:
    """Extract tensor parallel size from benchmark command."""
    tp_patterns = [
        r'--tp[=\\s]+(\\d+)',
        r'--tp-size[=\\s]+(\\d+)',
        r'--tensor-parallel-size[=\\s]+(\\d+)',
        r'-tp[=\\s]+(\\d+)',
    ]
    for pattern in tp_patterns:
        match = re.search(pattern, command)
        if match:
            return int(match.group(1))
    return 1

def extract_server_args_from_command(command: str) -> list:
    """Extract server-relevant args from perf_command."""
    args = []
    if "--trust-remote-code" in command:
        args.extend(["--trust-remote-code"])
    dtype_match = re.search(r'--dtype[=\\s]+(\\S+)', command)
    if dtype_match:
        args.extend(["--dtype", dtype_match.group(1)])
    quant_match = re.search(r'--quantization[=\\s]+(\\S+)', command)
    if quant_match:
        args.extend(["--quantization", quant_match.group(1)])
    if "--disable-radix-cache" in command:
        args.extend(["--disable-radix-cache"])
    return args

def needs_server(command: str) -> bool:
    """Check if benchmark command requires a running server."""
    server_free_patterns = ["bench_one_batch", "bench_offline_throughput", "bench_offline"]
    cmd_lower = command.lower()
    return not any(p in cmd_lower for p in server_free_patterns)

def prepare_benchmark_command(command: str, port: int) -> str:
    """Prepare benchmark command with SGLang best practices."""
    cmd = command
    if cmd.startswith("python "):
        cmd = "python3 " + cmd[7:]
    if "--port" in cmd:
        cmd = re.sub(r'--port[=\\s]+\\d+', f'--port {{port}}', cmd)
    else:
        cmd += f" --port {{port}}"
    if "--host" not in cmd and "--base-url" not in cmd:
        cmd += " --host 127.0.0.1"
    if "--warmup" not in cmd.lower():
        cmd += " --warmup-requests 2"
    if "--flush-cache" not in cmd:
        cmd += " --flush-cache"
    if "--num-prompt" not in cmd.lower():
        cmd += " --num-prompts 100"
    return cmd

def start_server(model: str, port: int, tp_size: int = 1, extra_args: list = None, timeout: int = 600):
    """Start SGLang server and wait for ready."""
    import select

    cmd = [
        "python3", "-m", "sglang.launch_server",
        "--model-path", model,
        "--port", str(port),
        "--tp-size", str(tp_size),
        "--host", "0.0.0.0",
        "--log-level", "info",
    ]
    if extra_args:
        cmd.extend(extra_args)

    print(f"Starting SGLang server on port {{port}} with TP={{tp_size}}...")
    print(f"  Command: {{' '.join(cmd)}}")

    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, preexec_fn=os.setsid)

    start = time.time()
    ready = False
    server_logs = []

    while time.time() - start < timeout:
        if proc.poll() is not None:
            remaining = proc.stdout.read()
            server_logs.append(remaining)
            print(f"Server process died. Exit code: {{proc.returncode}}")
            break

        try:
            ready_fds, _, _ = select.select([proc.stdout], [], [], 1.0)
            if ready_fds:
                line = proc.stdout.readline()
                server_logs.append(line)
                if any(x in line for x in ["Application startup complete", "Uvicorn running", "Server is ready"]):
                    print(f"Server ready! ({{time.time() - start:.1f}}s)")
                    ready = True
                    break
        except:
            pass

        try:
            import requests
            resp = requests.get(f"http://localhost:{{port}}/health", timeout=2)
            if resp.status_code == 200:
                print(f"Server health check passed! ({{time.time() - start:.1f}}s)")
                ready = True
                break
        except:
            pass

        elapsed = time.time() - start
        if int(elapsed) % 30 == 0 and int(elapsed) > 0:
            print(f"  Still waiting for server... ({{elapsed:.0f}}s)")

    return proc, ready, "".join(server_logs)

def stop_server(proc, port):
    """Stop server and clean up."""
    if proc is None:
        return
    try:
        import signal
        os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
        proc.wait(timeout=10)
    except:
        try:
            proc.kill()
            proc.wait(timeout=5)
        except:
            pass
    subprocess.run(["fuser", "-k", f"{{port}}/tcp"], capture_output=True, timeout=5)

def run_benchmark(port: int, command: str):
    """Run benchmark command against server."""
    cmd = prepare_benchmark_command(command, port)
    print(f"Running benchmark command:")
    print(f"  {{cmd}}")
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=900)
    output = result.stdout + "\\n" + result.stderr
    print(f"Benchmark output (last 1000 chars): ...{{output[-1000:]}}")
    return output

def run_benchmark_direct(command: str):
    """Run benchmark command directly (no server needed)."""
    cmd = command
    if cmd.startswith("python "):
        cmd = "python3 " + cmd[7:]
    print(f"Running direct benchmark: {{cmd[:100]}}...")
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=1800)
    output = result.stdout + "\\n" + result.stderr
    print(f"Direct benchmark output (last 500 chars): ...{{output[-500:]}}")
    return output

# Main benchmark flow
try:
    print("=" * 60)
    print("SGLang Human-Only Benchmark")
    print("=" * 60)

    # Verify SGLang is importable
    import sglang
    print(f"SGLang module loaded: {{sglang}}")

    # Extract settings from command
    tp_size = extract_tp_from_command(PERF_COMMAND)
    print(f"Tensor parallelism: {{tp_size}}")

    extra_server_args = extract_server_args_from_command(PERF_COMMAND)
    if extra_server_args:
        print(f"Extra server args: {{extra_server_args}}")

    use_server = needs_server(PERF_COMMAND)
    print(f"Benchmark type: {{'server-based' if use_server else 'direct (no server)'}}")

    # ==================== HUMAN BENCHMARK ====================
    print("\\n" + "=" * 60)
    print("HUMAN BENCHMARK (Docker image as-is)")
    print("=" * 60)

    if use_server:
        server, ready, server_logs = start_server(MODEL, HUMAN_PORT, tp_size, extra_server_args)
        if not ready:
            results["human_server_logs"] = server_logs
            raise Exception(f"Server failed to start. Logs: {{server_logs[-1000:]}}")

        human_output = run_benchmark(HUMAN_PORT, PERF_COMMAND)
        stop_server(server, HUMAN_PORT)
    else:
        human_output = run_benchmark_direct(PERF_COMMAND)

    results["human_raw"] = human_output
    results["human_metrics"] = parse_benchmark_output(human_output)
    print(f"Human metrics: {{results['human_metrics']}}")

    if not results["human_metrics"]:
        print("WARNING: No metrics extracted from human. Check output format.")

    # ==================== SUCCESS ====================
    results["status"] = "success"

except Exception as e:
    results["error"] = str(e)
    results["status"] = "error"

# Sanitize raw output fields
def sanitize_for_json(text):
    """Remove ANSI escape codes and control characters from text."""
    if not isinstance(text, str):
        return text
    import re as re_clean
    esc = chr(27)
    cleaned = re_clean.sub(esc + r'\\[[0-9;]*[a-zA-Z]', '', text)
    cleaned = re_clean.sub(esc + r'\\][^' + chr(7) + r']*' + chr(7), '', cleaned)
    cleaned = re_clean.sub(esc + r'[\\[\\]()#;?0-9]*[0-9A-Za-z]', '', cleaned)
    cleaned = cleaned.replace(chr(13), '')
    cleaned = ''.join(c for c in cleaned if ord(c) == 10 or ord(c) == 9 or ord(c) >= 32)
    # No truncation - keep full output
    return cleaned

for key in ["baseline_raw", "human_raw", "agent_raw"]:
    if key in results and results[key]:
        results[key] = sanitize_for_json(results[key])

# Output results as JSON
print("\\n=== BENCHMARK_RESULTS_JSON ===")
print(json.dumps(results))
print("=== END_BENCHMARK_RESULTS_JSON ===")
'''
    return script


def _create_benchmark_script(
    base_commit: str,
    human_commit: str,
    agent_patch: Optional[str],
    perf_command: str,
    model: str,
) -> str:
    """Create the Python script that runs inside the Modal sandbox for 3-way benchmark."""

    # Escape the agent patch for embedding in script
    escaped_patch = repr(agent_patch) if agent_patch else "None"

    script = f'''
import subprocess
import sys
import os
import json
import time
import re
import shutil
from pathlib import Path

# Configuration
BASE_COMMIT = "{base_commit}"
HUMAN_COMMIT = "{human_commit}"
AGENT_PATCH = {escaped_patch}
PERF_COMMAND = """{perf_command}"""
MODEL = "{model}"

SGLANG_REPO_URL = "https://github.com/sgl-project/sglang.git"
BASELINE_PORT = 30001
HUMAN_PORT = 30002
AGENT_PORT = 30003

# Results storage
results = {{
    "baseline_metrics": {{}},
    "human_metrics": {{}},
    "agent_metrics": None,
    "status": "error",
    "error": None,
    "baseline_raw": "",
    "human_raw": "",
    "agent_raw": "",
}}

def find_sglang_path():
    """Find where SGLang is installed.

    SGLang Docker images often use editable installs where __file__ is None.
    We need multiple fallback strategies to find the actual path.
    """
    # Strategy 1: Try __file__ attribute
    result = subprocess.run(
        ["python3", "-c", "import sglang; print(sglang.__file__ or 'None')"],
        capture_output=True, text=True
    )
    sglang_file = result.stdout.strip() if result.returncode == 0 else "None"
    print(f"  sglang.__file__ = {{sglang_file}}")

    if sglang_file != "None" and sglang_file.startswith("/"):
        sglang_dir = str(Path(sglang_file).parent)
        print(f"  Found via __file__: {{sglang_dir}}")
        return sglang_dir

    # Strategy 2: Use __path__ attribute (works for packages)
    result = subprocess.run(
        ["python3", "-c", "import sglang; print(sglang.__path__[0] if hasattr(sglang, '__path__') else 'None')"],
        capture_output=True, text=True
    )
    if result.returncode == 0:
        sglang_path = result.stdout.strip()
        if sglang_path != "None" and sglang_path.startswith("/"):
            # For editable installs, __path__ might be the repo root, not the python dir
            # Check if we need to add python/sglang
            init_file = os.path.join(sglang_path, "__init__.py")
            nested_init = os.path.join(sglang_path, "python", "sglang", "__init__.py")
            if os.path.isfile(init_file):
                print(f"  Found via __path__ (direct): {{sglang_path}}")
                return sglang_path
            elif os.path.isfile(nested_init):
                nested_path = os.path.join(sglang_path, "python", "sglang")
                print(f"  Found via __path__ (editable): {{nested_path}}")
                return nested_path
            else:
                print(f"  __path__ returned {{sglang_path}} but no __init__.py found")

    # Strategy 3: Use importlib to find spec location
    result = subprocess.run(
        ["python3", "-c", "import importlib.util; spec = importlib.util.find_spec('sglang'); print(spec.submodule_search_locations[0] if spec and spec.submodule_search_locations else 'None')"],
        capture_output=True, text=True
    )
    if result.returncode == 0:
        sglang_path = result.stdout.strip()
        if sglang_path != "None" and sglang_path.startswith("/"):
            print(f"  Found via importlib: {{sglang_path}}")
            return sglang_path

    # Strategy 4: Check known Docker image paths
    known_paths = [
        "/sgl-workspace/sglang/python/sglang",
        "/usr/local/lib/python3.10/dist-packages/sglang",
        "/opt/sglang/python/sglang",
    ]
    for path in known_paths:
        if os.path.exists(path) and os.path.isfile(os.path.join(path, "__init__.py")):
            print(f"  Found via known path: {{path}}")
            return path

    print("  ERROR: Could not find sglang installation path")
    return None

def clone_sglang_source(target_dir: str, commit: str):
    """Clone SGLang repo at specific commit."""
    if os.path.exists(target_dir):
        shutil.rmtree(target_dir)

    subprocess.run(
        ["git", "clone", "--depth", "100", SGLANG_REPO_URL, target_dir],
        capture_output=True, timeout=300
    )

    subprocess.run(
        ["git", "fetch", "origin", commit],
        cwd=target_dir, capture_output=True, timeout=120
    )

    result = subprocess.run(
        ["git", "checkout", commit],
        cwd=target_dir, capture_output=True, text=True
    )

    return result.returncode == 0

def overlay_python_files(source_dir: str, target_dir: str, create_new: bool = True):
    """Overlay Python files from source to target SGLang installation.

    Args:
        source_dir: Git repo directory containing python/sglang
        target_dir: Target sglang package directory
        create_new: If True, create files that don't exist in target (for full replacement)

    Returns:
        Tuple[bool, list]: (success, list of newly created file paths for cleanup)
    """
    source_python = Path(source_dir) / "python" / "sglang"
    target_python = Path(target_dir)

    print(f"  Overlay: {{source_python}} -> {{target_python}}")

    if not source_python.exists():
        print(f"  ERROR: Source path does not exist: {{source_python}}")
        # List parent directory to debug
        parent = Path(source_dir)
        if parent.exists():
            print(f"  Contents of {{parent}}: {{list(parent.iterdir())[:10]}}")
        return False, []

    if not target_python.exists():
        print(f"  ERROR: Target path does not exist: {{target_python}}")
        return False, []

    # Count source files
    source_files = list(source_python.rglob("*.py"))
    print(f"  Source has {{len(source_files)}} Python files")

    copied = 0
    created = 0
    skipped = 0
    created_files = []  # Track newly created files for cleanup
    for py_file in source_files:
        rel_path = py_file.relative_to(source_python)
        dest_file = target_python / rel_path

        if dest_file.exists():
            shutil.copy2(py_file, dest_file)
            copied += 1
        elif create_new:
            # Create directory structure if needed
            dest_file.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(py_file, dest_file)
            created += 1
            created_files.append(str(dest_file))  # Track for cleanup
        else:
            skipped += 1

    print(f"  Overlaid {{copied}} Python files, created {{created}} new ({{skipped}} skipped)")
    return copied > 0 or created > 0, created_files

def apply_patch(repo_dir: str, patch_content: str):
    """Apply patch to repository.

    The patch should be applied to the repo root (e.g., /tmp/sglang_base)
    since patch paths are like 'python/sglang/srt/...'
    """
    patch_file = "/tmp/agent.patch"
    with open(patch_file, 'w') as f:
        f.write(patch_content)

    print(f"  Applying patch in {{repo_dir}}...")

    # First try git apply (works best for git repos)
    result = subprocess.run(
        ["git", "apply", "--verbose", patch_file],
        cwd=repo_dir, capture_output=True, text=True
    )

    if result.returncode == 0:
        print(f"  git apply succeeded")
        return True

    print(f"  git apply failed: {{result.stderr[:200]}}")

    # Try git apply with 3way merge
    result = subprocess.run(
        ["git", "apply", "--3way", patch_file],
        cwd=repo_dir, capture_output=True, text=True
    )

    if result.returncode == 0:
        print(f"  git apply --3way succeeded")
        return True

    print(f"  git apply --3way failed: {{result.stderr[:200]}}")

    # Fallback: use patch command with -p1 to strip a/ b/ prefixes
    result = subprocess.run(
        ["patch", "-p1", "-i", patch_file],
        cwd=repo_dir, capture_output=True, text=True
    )

    if result.returncode == 0:
        print(f"  patch -p1 succeeded")
        return True

    print(f"  patch -p1 failed: {{result.stderr[:200]}}")
    return False

def parse_benchmark_output(output: str):
    """Parse benchmark output to extract metrics.

    Following SGLang benchmark output format:
    - Request throughput (req/s):              8.09
    - Input token throughput (tok/s):          2716.38
    - Mean TTFT (ms):                          509.10
    """
    metrics = {{}}

    # Throughput patterns - format: "Request throughput (req/s):              8.09"
    throughput_patterns = {{
        "request_throughput": r"Request throughput \\(req/s\\):\\s+([\\d.]+)",
        "output_throughput": r"Output token throughput \\(tok/s\\):\\s+([\\d.]+)",
        "input_throughput": r"Input token throughput \\(tok/s\\):\\s+([\\d.]+)",
        "total_throughput": r"Total token throughput \\(tok/s\\):\\s+([\\d.]+)",
    }}

    # TTFT (Time To First Token) - format: "Mean TTFT (ms):                          509.10"
    ttft_patterns = {{
        "ttft_mean": r"Mean TTFT \\(ms\\):\\s+([\\d.]+)",
        "ttft_median": r"Median TTFT \\(ms\\):\\s+([\\d.]+)",
        "ttft_p99": r"P99 TTFT \\(ms\\):\\s+([\\d.]+)",
    }}

    # TPOT (Time Per Output Token) - lower is better
    tpot_patterns = {{
        "tpot_mean": r"Mean TPOT \\(ms\\):\\s+([\\d.]+)",
        "tpot_median": r"Median TPOT \\(ms\\):\\s+([\\d.]+)",
        "tpot_p99": r"P99 TPOT \\(ms\\):\\s+([\\d.]+)",
    }}

    # ITL (Inter-Token Latency) - lower is better
    itl_patterns = {{
        "itl_mean": r"Mean ITL \\(ms\\):\\s+([\\d.]+)",
        "itl_median": r"Median ITL \\(ms\\):\\s+([\\d.]+)",
        "itl_p95": r"P95 ITL \\(ms\\):\\s+([\\d.]+)",
        "itl_p99": r"P99 ITL \\(ms\\):\\s+([\\d.]+)",
    }}

    # E2E latency patterns
    e2e_patterns = {{
        "e2e_latency_mean": r"Mean E2E Latency \\(ms\\):\\s+([\\d.]+)",
        "e2e_latency_median": r"Median E2E Latency \\(ms\\):\\s+([\\d.]+)",
        "e2e_latency_p99": r"P99 E2E Latency \\(ms\\):\\s+([\\d.]+)",
    }}

    # Token count patterns
    token_patterns = {{
        "total_input_tokens": r"Total input tokens:\\s+([\\d.]+)",
        "total_output_tokens": r"Total generated tokens:\\s+([\\d.]+)",
    }}

    # Offline/batch benchmark patterns (bench_one_batch, bench_offline_throughput)
    offline_patterns = {{
        "batch_latency": r"Latency[:\\s]+([\\d.]+)\\s*ms",
        "prefill_throughput": r"Prefill throughput[:\\s]+([\\d.]+)",
        "decode_throughput": r"Decode throughput[:\\s]+([\\d.]+)",
        "total_latency_s": r"Total latency[:\\s]+([\\d.]+)\\s*s",
    }}

    # Combine all patterns
    all_patterns = {{}}
    all_patterns.update(throughput_patterns)
    all_patterns.update(ttft_patterns)
    all_patterns.update(tpot_patterns)
    all_patterns.update(itl_patterns)
    all_patterns.update(e2e_patterns)
    all_patterns.update(token_patterns)
    all_patterns.update(offline_patterns)

    for name, pattern in all_patterns.items():
        match = re.search(pattern, output, re.IGNORECASE)
        if match:
            metrics[name] = float(match.group(1))

    return metrics

def extract_tp_from_command(command: str) -> int:
    """Extract tensor parallel size from benchmark command."""
    # Try various TP flag patterns
    tp_patterns = [
        r'--tp[=\\s]+(\\d+)',
        r'--tp-size[=\\s]+(\\d+)',
        r'--tensor-parallel-size[=\\s]+(\\d+)',
        r'-tp[=\\s]+(\\d+)',
    ]
    for pattern in tp_patterns:
        match = re.search(pattern, command)
        if match:
            return int(match.group(1))
    return 1

def extract_server_args_from_command(command: str) -> list:
    """Extract server-relevant args from perf_command to pass to server."""
    args = []

    # Trust remote code
    if "--trust-remote-code" in command:
        args.extend(["--trust-remote-code"])

    # Dtype
    dtype_match = re.search(r'--dtype[=\\s]+(\\S+)', command)
    if dtype_match:
        args.extend(["--dtype", dtype_match.group(1)])

    # Quantization
    quant_match = re.search(r'--quantization[=\\s]+(\\S+)', command)
    if quant_match:
        args.extend(["--quantization", quant_match.group(1)])

    # Context length
    ctx_match = re.search(r'--context-length[=\\s]+(\\d+)', command)
    if ctx_match:
        args.extend(["--context-length", ctx_match.group(1)])

    # Disable radix cache (important for VLM)
    if "--disable-radix-cache" in command:
        args.extend(["--disable-radix-cache"])

    return args

def needs_server(command: str) -> bool:
    """Check if benchmark command requires a running server.

    Server-free benchmarks (per SGLang docs):
    - bench_one_batch: single static batch without server
    - bench_offline_throughput: offline throughput measurement
    """
    server_free_patterns = [
        "bench_one_batch",
        "bench_offline_throughput",
        "bench_offline",
    ]
    cmd_lower = command.lower()
    return not any(p in cmd_lower for p in server_free_patterns)

def run_benchmark_direct(command: str):
    """Run benchmark command directly (no server needed).

    For bench_one_batch and bench_offline_throughput.
    """
    # Ensure we use python3 (Docker container may not have 'python' symlink)
    cmd = command
    if cmd.startswith("python "):
        cmd = "python3 " + cmd[7:]

    print(f"Running direct benchmark: {{cmd[:100]}}...")

    # For direct benchmarks, run as-is
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=1800)
    output = result.stdout + "\\n" + result.stderr
    print(f"Direct benchmark output (last 500 chars): ...{{output[-500:]}}")
    return output

def prepare_benchmark_command(command: str, port: int) -> str:
    """Prepare benchmark command with proper SGLang best practices.

    Follows https://docs.sglang.io/developer_guide/bench_serving.html
    - Adds warmup requests if not present
    - Adds flush-cache for consistent results
    - Ensures proper host/port configuration
    - Adds default --num-prompts if not specified
    """
    cmd = command

    # Ensure we use python3 (Docker container may not have 'python' symlink)
    if cmd.startswith("python "):
        cmd = "python3 " + cmd[7:]

    # Update port in command
    if "--port" in cmd:
        cmd = re.sub(r'--port[=\\s]+\\d+', f'--port {{port}}', cmd)
    else:
        cmd += f" --port {{port}}"

    # Add host if not present
    if "--host" not in cmd and "--base-url" not in cmd:
        cmd += " --host 127.0.0.1"

    # Add warmup if not present (SGLang default is 1, but explicit is better)
    if "--warmup" not in cmd.lower():
        cmd += " --warmup-requests 2"

    # Add flush-cache for consistent benchmark results
    if "--flush-cache" not in cmd:
        cmd += " --flush-cache"

    # Add default --num-prompts if not specified (required for bench_serving)
    if "--num-prompt" not in cmd.lower():
        cmd += " --num-prompts 100"

    return cmd

def start_server(model: str, port: int, tp_size: int = 1, extra_args: list = None, timeout: int = 600):
    """Start SGLang server and wait for ready.

    Args:
        model: Model path/name
        port: Server port
        tp_size: Tensor parallel size
        extra_args: Additional server arguments (dtype, trust-remote-code, etc.)
        timeout: Max wait time for server ready (increased to 600s for large models)
    """
    import select

    cmd = [
        "python3", "-m", "sglang.launch_server",
        "--model-path", model,
        "--port", str(port),
        "--tp-size", str(tp_size),
        "--host", "0.0.0.0",
        "--log-level", "info",
    ]

    # Add extra args from command
    if extra_args:
        cmd.extend(extra_args)

    print(f"Starting SGLang server on port {{port}} with TP={{tp_size}}...")
    print(f"  Command: {{' '.join(cmd)}}")

    proc = subprocess.Popen(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, preexec_fn=os.setsid
    )

    start = time.time()
    ready = False
    server_logs = []

    while time.time() - start < timeout:
        if proc.poll() is not None:
            # Process died
            remaining = proc.stdout.read()
            server_logs.append(remaining)
            print(f"Server process died. Exit code: {{proc.returncode}}")
            print(f"Server logs: {{''.join(server_logs)[-2000:]}}")
            break

        try:
            ready_fds, _, _ = select.select([proc.stdout], [], [], 1.0)
            if ready_fds:
                line = proc.stdout.readline()
                server_logs.append(line)
                # SGLang server ready indicators
                if any(x in line for x in ["Application startup complete", "Uvicorn running", "Server is ready"]):
                    print(f"Server ready! ({{time.time() - start:.1f}}s)")
                    ready = True
                    break
        except:
            pass

        # Health check via /health endpoint
        try:
            import requests
            resp = requests.get(f"http://localhost:{{port}}/health", timeout=2)
            if resp.status_code == 200:
                print(f"Server health check passed! ({{time.time() - start:.1f}}s)")
                ready = True
                break
        except:
            pass

        # Progress indicator every 30s
        elapsed = time.time() - start
        if int(elapsed) % 30 == 0 and int(elapsed) > 0:
            print(f"  Still waiting for server... ({{elapsed:.0f}}s)")

    return proc, ready, "".join(server_logs)

def stop_server(proc, port):
    """Stop server and clean up."""
    if proc is None:
        return
    try:
        import signal
        os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
        proc.wait(timeout=10)
    except:
        try:
            proc.kill()
            proc.wait(timeout=5)
        except:
            pass
    subprocess.run(["fuser", "-k", f"{{port}}/tcp"], capture_output=True, timeout=5)

def run_benchmark(port: int, command: str):
    """Run benchmark command against server with SGLang best practices."""
    # Prepare command with warmup, flush-cache, proper port/host
    cmd = prepare_benchmark_command(command, port)

    print(f"Running benchmark command:")
    print(f"  {{cmd}}")

    result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=900)
    output = result.stdout + "\\n" + result.stderr
    print(f"Benchmark output (last 1000 chars): ...{{output[-1000:]}}")
    return output

# Main benchmark flow
try:
    print("=" * 60)
    print("SGLang 3-Way Benchmark (Best Practices)")
    print("=" * 60)

    sglang_path = find_sglang_path()
    if not sglang_path:
        raise Exception("SGLang not found in container")

    print(f"SGLang installed at: {{sglang_path}}")

    # Verify SGLang is importable
    import sglang
    print(f"SGLang module loaded: {{sglang}}")

    # Extract tensor parallelism from command
    tp_size = extract_tp_from_command(PERF_COMMAND)
    print(f"Tensor parallelism: {{tp_size}}")

    # Extract additional server args from command
    extra_server_args = extract_server_args_from_command(PERF_COMMAND)
    if extra_server_args:
        print(f"Extra server args: {{extra_server_args}}")

    # Check if benchmark needs a server
    use_server = needs_server(PERF_COMMAND)
    print(f"Benchmark type: {{'server-based' if use_server else 'direct (no server)'}}")

    # Clone source repos for overlay
    base_repo = "/tmp/sglang_base"
    human_repo = "/tmp/sglang_human"

    # ==================== BASELINE ====================
    print("\\n" + "=" * 60)
    print("[1/3] BASELINE benchmark")
    print("=" * 60)

    # Clone and overlay baseline Python files
    print(f"Cloning baseline commit {{BASE_COMMIT[:8]}}...")
    if not clone_sglang_source(base_repo, BASE_COMMIT):
        raise Exception(f"Failed to clone baseline {{BASE_COMMIT[:8]}}")

    # Save original files for restoration
    original_files = {{}}
    base_python = Path(base_repo) / "python" / "sglang"
    for py_file in base_python.rglob("*.py"):
        rel_path = py_file.relative_to(base_python)
        target = Path(sglang_path) / rel_path
        if target.exists():
            original_files[str(rel_path)] = target.read_text()

    print(f"Saved {{len(original_files)}} original files for restoration")

    # Overlay baseline files - track created files for cleanup
    _, baseline_created_files = overlay_python_files(base_repo, sglang_path)
    print(f"  Created {{len(baseline_created_files)}} new files (will be cleaned up)")

    baseline_failed = False
    if use_server:
        # Start server with baseline code
        server, ready, server_logs = start_server(MODEL, BASELINE_PORT, tp_size, extra_server_args)
        if not ready:
            results["baseline_server_logs"] = server_logs
            print(f"WARNING: Baseline server failed to start (likely incompatible code structure)")
            print(f"  This happens when baseline commit has different module structure than Docker image")
            print(f"  Skipping baseline, will compare human vs agent only")
            baseline_failed = True
            results["baseline_skip_reason"] = "Server failed - incompatible overlay"
        else:
            # Run benchmark against server
            baseline_output = run_benchmark(BASELINE_PORT, PERF_COMMAND)
            stop_server(server, BASELINE_PORT)
            results["baseline_raw"] = baseline_output
            results["baseline_metrics"] = parse_benchmark_output(baseline_output)
            print(f"Baseline metrics: {{results['baseline_metrics']}}")
    else:
        # Run benchmark directly (no server)
        try:
            baseline_output = run_benchmark_direct(PERF_COMMAND)
            results["baseline_raw"] = baseline_output
            results["baseline_metrics"] = parse_benchmark_output(baseline_output)
            print(f"Baseline metrics: {{results['baseline_metrics']}}")
        except Exception as e:
            print(f"WARNING: Baseline benchmark failed: {{e}}")
            baseline_failed = True
            results["baseline_skip_reason"] = str(e)

    if not baseline_failed and not results.get("baseline_metrics"):
        print("WARNING: No metrics extracted from baseline. Check output format.")

    # ==================== HUMAN ====================
    print("\\n" + "=" * 60)
    print("[2/3] HUMAN benchmark")
    print("=" * 60)

    # Restore original files (human commit = Docker image default)
    for rel_path, content in original_files.items():
        target = Path(sglang_path) / rel_path
        target.write_text(content)
    print(f"Restored {{len(original_files)}} files to human commit state")

    # CRITICAL: Delete files that were created during baseline overlay
    # These don't exist in the original Docker image and can break imports
    deleted_count = 0
    for created_file in baseline_created_files:
        try:
            Path(created_file).unlink()
            deleted_count += 1
        except Exception as e:
            print(f"  Warning: Could not delete {{created_file}}: {{e}}")
    print(f"Deleted {{deleted_count}} files created during baseline overlay")

    if use_server:
        # Start server with human code (original Docker image)
        server, ready, server_logs = start_server(MODEL, HUMAN_PORT, tp_size, extra_server_args)
        if not ready:
            results["human_server_logs"] = server_logs
            raise Exception(f"Human server failed to start. Logs: {{server_logs[-1000:]}}")

        # Run benchmark against server
        human_output = run_benchmark(HUMAN_PORT, PERF_COMMAND)
        stop_server(server, HUMAN_PORT)
    else:
        # Run benchmark directly (no server)
        human_output = run_benchmark_direct(PERF_COMMAND)

    results["human_raw"] = human_output
    results["human_metrics"] = parse_benchmark_output(human_output)
    print(f"Human metrics: {{results['human_metrics']}}")

    if not results["human_metrics"]:
        print("WARNING: No metrics extracted from human. Check output format.")

    # ==================== AGENT (if patch provided) ====================
    if AGENT_PATCH:
        print("\\n" + "=" * 60)
        print("[3/3] AGENT benchmark")
        print("=" * 60)

        # Overlay baseline files again (agent starts from baseline)
        _, _ = overlay_python_files(base_repo, sglang_path)

        # Apply agent patch to the BASE REPO ROOT (not sglang subdir!)
        # Patch paths are like 'python/sglang/srt/...' so they need repo root
        print(f"Applying agent patch ({{len(AGENT_PATCH)}} bytes)...")
        patch_success = apply_patch(base_repo, AGENT_PATCH)
        if patch_success:
            # Re-overlay to copy patched files to sglang installation
            print("Re-overlaying patched files to installation...")
            _, _ = overlay_python_files(base_repo, sglang_path)
        else:
            print("WARNING: Patch apply failed - agent benchmark may match baseline")

        if use_server:
            # Start server with agent code
            server, ready, server_logs = start_server(MODEL, AGENT_PORT, tp_size, extra_server_args)
            if not ready:
                results["agent_server_logs"] = server_logs
                results["agent_error"] = f"Agent server failed to start"
                print(f"Agent server failed. Logs: {{server_logs[-1000:]}}")
            else:
                # Run benchmark against server
                agent_output = run_benchmark(AGENT_PORT, PERF_COMMAND)
                stop_server(server, AGENT_PORT)
                results["agent_raw"] = agent_output
                results["agent_metrics"] = parse_benchmark_output(agent_output)
                print(f"Agent metrics: {{results['agent_metrics']}}")
        else:
            # Run benchmark directly (no server)
            agent_output = run_benchmark_direct(PERF_COMMAND)
            results["agent_raw"] = agent_output
            results["agent_metrics"] = parse_benchmark_output(agent_output)
            print(f"Agent metrics: {{results['agent_metrics']}}")

    # ==================== SUCCESS ====================
    results["status"] = "success"

    # Calculate improvements
    if results["baseline_metrics"] and results["human_metrics"]:
        human_imp = {{}}
        for k in results["baseline_metrics"]:
            if k in results["human_metrics"] and results["baseline_metrics"][k] > 0:
                base_v = results["baseline_metrics"][k]
                human_v = results["human_metrics"][k]
                if "throughput" in k:
                    human_imp[k] = (human_v - base_v) / base_v * 100
                else:
                    human_imp[k] = (base_v - human_v) / base_v * 100
        results["human_improvement"] = human_imp

    if results["baseline_metrics"] and results.get("agent_metrics"):
        agent_imp = {{}}
        for k in results["baseline_metrics"]:
            if k in results["agent_metrics"] and results["baseline_metrics"][k] > 0:
                base_v = results["baseline_metrics"][k]
                agent_v = results["agent_metrics"][k]
                if "throughput" in k:
                    agent_imp[k] = (agent_v - base_v) / base_v * 100
                else:
                    agent_imp[k] = (base_v - agent_v) / base_v * 100
        results["agent_improvement"] = agent_imp

    if results.get("human_metrics") and results.get("agent_metrics"):
        agent_vs_human = {{}}
        for k in results["human_metrics"]:
            if k in results["agent_metrics"] and results["human_metrics"][k] > 0:
                human_v = results["human_metrics"][k]
                agent_v = results["agent_metrics"][k]
                if "throughput" in k:
                    agent_vs_human[k] = (agent_v - human_v) / human_v * 100
                else:
                    agent_vs_human[k] = (human_v - agent_v) / human_v * 100
        results["agent_vs_human"] = agent_vs_human

except Exception as e:
    results["error"] = str(e)
    results["status"] = "error"

# Sanitize raw output fields to remove control characters that break JSON
def sanitize_for_json(text):
    """Remove ANSI escape codes and control characters from text."""
    if not isinstance(text, str):
        return text
    import re as re_clean
    # Build ESC character (chr(27) = 0x1b)
    esc = chr(27)
    # Remove ANSI CSI sequences: ESC [ params letter
    cleaned = re_clean.sub(esc + r'\\[[0-9;]*[a-zA-Z]', '', text)
    # Remove ANSI OSC sequences: ESC ] ... BEL
    cleaned = re_clean.sub(esc + r'\\][^' + chr(7) + r']*' + chr(7), '', cleaned)
    # Remove other ANSI sequences
    cleaned = re_clean.sub(esc + r'[\\[\\]()#;?0-9]*[0-9A-Za-z]', '', cleaned)
    # Remove carriage returns (tqdm progress bar uses these)
    cleaned = cleaned.replace(chr(13), '')  # CR
    # Remove remaining control characters (keep newline chr(10), tab chr(9), space chr(32)+)
    cleaned = ''.join(c for c in cleaned if ord(c) == 10 or ord(c) == 9 or ord(c) >= 32)
    # No truncation - keep full output
    return cleaned

# Clean raw fields before JSON serialization
for key in ["baseline_raw", "human_raw", "agent_raw"]:
    if key in results and results[key]:
        results[key] = sanitize_for_json(results[key])

# Output results as JSON for parsing
print("\\n=== BENCHMARK_RESULTS_JSON ===")
print(json.dumps(results))
print("=== END_BENCHMARK_RESULTS_JSON ===")
'''
    return script


def _sanitize_json_string(json_str: str) -> str:
    """Remove control characters that break JSON parsing."""
    # Remove ANSI escape codes
    esc = chr(27)  # ESC character
    # ANSI CSI sequences: ESC [ ... letter
    json_str = re.sub(esc + r'\[[0-9;]*[a-zA-Z]', '', json_str)
    # ANSI OSC sequences: ESC ] ... BEL
    json_str = re.sub(esc + r'\][^' + chr(7) + ']*' + chr(7), '', json_str)
    # Other ESC sequences
    json_str = re.sub(esc + r'[\[\]()#;?0-9]*[0-9A-Za-z]', '', json_str)
    # Remove carriage returns
    json_str = json_str.replace(chr(13), '')
    # Remove other control characters (keep \t \n and printable chars)
    json_str = ''.join(c for c in json_str if ord(c) == 10 or ord(c) == 9 or ord(c) >= 32)
    return json_str


def _parse_benchmark_results(output: str, result: Dict) -> Dict:
    """Parse benchmark results from sandbox output."""
    # Extract JSON results - use greedy match bounded by markers
    # The JSON is a single object on one line, so match everything between markers
    json_match = re.search(
        r'=== BENCHMARK_RESULTS_JSON ===\s*(\{.*\})\s*=== END_BENCHMARK_RESULTS_JSON ===',
        output,
        re.DOTALL
    )

    if json_match:
        json_str = json_match.group(1).strip()

        # First try: parse as-is
        try:
            parsed = json.loads(json_str)
            result.update(parsed)
            return result
        except json.JSONDecodeError:
            pass

        # Second try: sanitize control characters
        try:
            clean_json = _sanitize_json_string(json_str)
            parsed = json.loads(clean_json)
            result.update(parsed)
            print(f"Parsed JSON after sanitizing control characters")
            return result
        except json.JSONDecodeError:
            pass

        # Third try: extract key fields using regex (fallback)
        # This extracts the essential metrics even if raw fields have issues
        try:
            # Extract human_metrics
            human_metrics_match = re.search(r'"human_metrics":\s*(\{[^}]+\})', json_str)
            if human_metrics_match:
                human_metrics = json.loads(human_metrics_match.group(1))
                result["human_metrics"] = human_metrics

            # Extract baseline_metrics
            baseline_metrics_match = re.search(r'"baseline_metrics":\s*(\{[^}]*\})', json_str)
            if baseline_metrics_match:
                baseline_metrics = json.loads(baseline_metrics_match.group(1))
                result["baseline_metrics"] = baseline_metrics

            # Extract agent_metrics (might be null)
            agent_metrics_match = re.search(r'"agent_metrics":\s*(\{[^}]+\}|null)', json_str)
            if agent_metrics_match:
                agent_val = agent_metrics_match.group(1)
                result["agent_metrics"] = None if agent_val == "null" else json.loads(agent_val)

            # Extract status
            status_match = re.search(r'"status":\s*"([^"]+)"', json_str)
            if status_match:
                result["status"] = status_match.group(1)

            # Extract error
            error_match = re.search(r'"error":\s*("(?:[^"\\]|\\.)*"|null)', json_str)
            if error_match:
                error_val = error_match.group(1)
                result["error"] = None if error_val == "null" else json.loads(error_val)

            print(f"Extracted key fields via regex fallback")
            return result
        except Exception as e:
            print(f"Regex fallback failed: {e}")

        # Final fallback: report error
        result["error"] = f"Failed to parse benchmark results JSON"
        print(f"JSON string (first 500 chars): {json_str[:500]}")
    else:
        result["error"] = "No benchmark results found in output"
        # Check if markers exist at all
        if "BENCHMARK_RESULTS_JSON" in output:
            result["error"] = "Found JSON markers but couldn't extract content"
            print(f"Output contains markers but regex failed. Output length: {len(output)}")

    return result


# =====================================================================
# Parallel 3-Way Benchmark (3 sandboxes, 3 GPUs, ~3x faster)
# =====================================================================

def _create_single_phase_script(
    phase: str,
    commit: str,
    perf_command: str,
    model: str,
    agent_patch: Optional[str] = None,
    base_commit: Optional[str] = None,
    human_patch: Optional[str] = None,
) -> str:
    """Create a benchmark script for a single phase (baseline, human, or agent).

    Uses SEPARATE DOCKER IMAGES approach (cleaner and more reliable):
    - HUMAN: Human commit Docker image (has optimization)
    - BASELINE: Base commit Docker image (pre-optimization, no patching!)
    - AGENT: Base commit Docker image + agent_patch
    """

    port = {"baseline": 30001, "human": 30002, "agent": 30003}.get(phase, 30001)
    escaped_agent_patch = repr(agent_patch) if agent_patch else "None"
    # Note: human_patch no longer used - each phase uses its own Docker image

    script = f'''
import subprocess
import sys
import os
import json
import time
import re
import shutil
from pathlib import Path

# Set environment variables for HuggingFace cache
os.environ["HF_HOME"] = "/root/.cache/huggingface"
os.environ["TRANSFORMERS_CACHE"] = "/root/.cache/huggingface"

# Fix typing_extensions version for pydantic compatibility (Sentinel added in 4.10.0)
# This needs to happen BEFORE importing any pydantic-related modules
print("Upgrading typing_extensions for pydantic compatibility...")
subprocess.run(["pip3", "install", "typing_extensions>=4.10.0", "--upgrade", "-q"], check=False)

# Configuration
PHASE = "{phase}"
COMMIT = "{commit}"
BASE_COMMIT = "{base_commit or ''}"
AGENT_PATCH = {escaped_agent_patch}
PERF_COMMAND = """{perf_command}"""
MODEL = "{model}"
PORT = {port}

SGLANG_REPO_URL = "https://github.com/sgl-project/sglang.git"

# Results storage
results = {{
    "phase": PHASE,
    "metrics": {{}},
    "status": "error",
    "error": None,
    "raw_output": "",
}}

def find_sglang_path():
    """Find where SGLang is installed."""
    result = subprocess.run(
        ["python3", "-c", "import sglang; print(sglang.__path__[0] if hasattr(sglang, '__path__') else 'None')"],
        capture_output=True, text=True
    )
    if result.returncode == 0:
        sglang_path = result.stdout.strip()
        if sglang_path != "None" and sglang_path.startswith("/"):
            return sglang_path

    # Fallback to known paths
    known_paths = [
        "/sgl-workspace/sglang/python/sglang",
        "/usr/local/lib/python3.10/dist-packages/sglang",
    ]
    for path in known_paths:
        if os.path.exists(path) and os.path.isfile(os.path.join(path, "__init__.py")):
            return path
    return None

def clone_and_checkout(target_dir: str, commit: str):
    """Clone SGLang repo at specific commit."""
    if os.path.exists(target_dir):
        shutil.rmtree(target_dir)
    subprocess.run(["git", "clone", "--depth", "100", SGLANG_REPO_URL, target_dir], capture_output=True, timeout=300)
    subprocess.run(["git", "fetch", "origin", commit], cwd=target_dir, capture_output=True, timeout=120)
    result = subprocess.run(["git", "checkout", commit], cwd=target_dir, capture_output=True, text=True)
    return result.returncode == 0

def overlay_python_files(source_dir: str, target_dir: str):
    """Overlay Python files from source to target."""
    source_python = Path(source_dir) / "python" / "sglang"
    target_python = Path(target_dir)

    if not source_python.exists() or not target_python.exists():
        return False

    copied = 0
    for py_file in source_python.rglob("*.py"):
        rel_path = py_file.relative_to(source_python)
        dest_file = target_python / rel_path
        if dest_file.exists():
            shutil.copy2(py_file, dest_file)
            copied += 1
        else:
            dest_file.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(py_file, dest_file)
            copied += 1

    print(f"  Overlaid {{copied}} Python files")
    return copied > 0

def apply_patch(repo_dir: str, patch_content: str):
    """Apply patch to repository."""
    patch_file = "/tmp/agent.patch"
    with open(patch_file, 'w') as f:
        f.write(patch_content)

    result = subprocess.run(["git", "apply", "--verbose", patch_file], cwd=repo_dir, capture_output=True, text=True)
    if result.returncode == 0:
        return True

    result = subprocess.run(["patch", "-p1", "-i", patch_file], cwd=repo_dir, capture_output=True, text=True)
    return result.returncode == 0

def parse_benchmark_output(output: str):
    """Parse benchmark output to extract metrics."""
    metrics = {{}}
    patterns = {{
        "request_throughput": r"Request throughput \\(req/s\\):\\s+([\\d.]+)",
        "output_throughput": r"Output token throughput \\(tok/s\\):\\s+([\\d.]+)",
        "input_throughput": r"Input token throughput \\(tok/s\\):\\s+([\\d.]+)",
        "ttft_mean": r"Mean TTFT \\(ms\\):\\s+([\\d.]+)",
        "ttft_median": r"Median TTFT \\(ms\\):\\s+([\\d.]+)",
        "ttft_p99": r"P99 TTFT \\(ms\\):\\s+([\\d.]+)",
        "tpot_mean": r"Mean TPOT \\(ms\\):\\s+([\\d.]+)",
        "tpot_median": r"Median TPOT \\(ms\\):\\s+([\\d.]+)",
        "tpot_p99": r"P99 TPOT \\(ms\\):\\s+([\\d.]+)",
        "itl_mean": r"Mean ITL \\(ms\\):\\s+([\\d.]+)",
        "itl_median": r"Median ITL \\(ms\\):\\s+([\\d.]+)",
        "e2e_latency_mean": r"Mean E2E Latency \\(ms\\):\\s+([\\d.]+)",
    }}

    for name, pattern in patterns.items():
        match = re.search(pattern, output, re.IGNORECASE)
        if match:
            metrics[name] = float(match.group(1))

    return metrics

def needs_server(cmd: str) -> bool:
    """Check if benchmark needs a server."""
    return "bench_serving" in cmd or "--base-url" in cmd or "--host" in cmd

def start_server(model: str, port: int):
    """Start SGLang server."""
    cmd = ["python3", "-m", "sglang.launch_server", "--model-path", model, "--port", str(port), "--host", "127.0.0.1"]
    print(f"Server command: {{' '.join(cmd)}}")
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = "0"
    # Combine stdout/stderr for easier debugging
    return subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, env=env)

def wait_for_server(port: int, timeout: int = 600, proc=None):
    """Wait for server to be ready, capturing logs for debugging."""
    import socket
    import select
    start = time.time()
    server_logs = []

    while time.time() - start < timeout:
        # Check if server process died
        if proc and proc.poll() is not None:
            remaining = proc.stdout.read() if proc.stdout else ""
            server_logs.append(remaining)
            print(f"Server process died! Exit code: {{proc.returncode}}")
            print(f"Server logs:\\n{{''.join(server_logs)}}")
            return False

        # Try to read server output (non-blocking)
        if proc and proc.stdout:
            try:
                ready, _, _ = select.select([proc.stdout], [], [], 0.1)
                if ready:
                    line = proc.stdout.readline()
                    if line:
                        server_logs.append(line)
                        # Print key lines for debugging
                        if any(x in line.lower() for x in ["error", "exception", "failed", "cuda", "memory", "ready"]):
                            print(f"[SERVER] {{line.rstrip()}}")
                        if "application startup complete" in line.lower() or "uvicorn running" in line.lower():
                            print(f"Server ready after {{time.time() - start:.1f}}s")
                            return True
            except Exception as e:
                pass

        # Also try socket connection
        try:
            sock = socket.create_connection(("127.0.0.1", port), timeout=2)
            sock.close()
            print(f"Server responding on port {{port}} after {{time.time() - start:.1f}}s")
            time.sleep(2)  # Give it a moment to stabilize
            return True
        except:
            pass

        time.sleep(3)

    # Timeout - print what we captured
    print(f"Server startup timeout after {{timeout}}s")
    if server_logs:
        print(f"Last server logs:\\n{{''.join(server_logs[-50:])}}")  # Last 50 lines
    return False

def stop_server(proc):
    """Stop server process."""
    proc.terminate()
    try:
        proc.wait(timeout=10)
    except:
        proc.kill()

def run_benchmark(port: int, perf_command: str):
    """Run the benchmark command."""
    cmd = perf_command
    if needs_server(cmd):
        cmd = re.sub(r'--base-url\\s+\\S+', f'--base-url http://127.0.0.1:{{port}}', cmd)
        cmd = re.sub(r'--host\\s+\\S+', f'--host 127.0.0.1', cmd)
        # Replace existing --port or add it if not present
        if re.search(r'--port\\s+\\d+', cmd):
            cmd = re.sub(r'--port\\s+\\d+', f'--port {{port}}', cmd)
        else:
            cmd = f'{{cmd}} --port {{port}}'

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = "0"

    result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=3600, env=env)
    output = result.stdout + result.stderr
    return output, parse_benchmark_output(output)

# Main execution
print(f"=" * 60)
print(f"[{{PHASE.upper()}}] Starting benchmark")
print(f"=" * 60)

try:
    sglang_path = find_sglang_path()
    print(f"SGLang path: {{sglang_path}}")

    # =========================================================================
    # SEPARATE DOCKER IMAGE APPROACH (cleaner than reverse-patch)
    # =========================================================================
    # - HUMAN: Human commit Docker image (has the optimization)
    # - BASELINE: Base commit Docker image (pre-optimization, no patching!)
    # - AGENT: Base commit Docker image + AGENT_PATCH
    # =========================================================================

    if PHASE == "human":
        # Human: Use Docker image as-is (has the optimization)
        print("[HUMAN] Using human commit Docker image (no changes needed)")

    elif PHASE == "baseline":
        # Baseline: Use base commit Docker image directly (no patching needed!)
        print("[BASELINE] Using base commit Docker image (no changes needed)")

    elif PHASE == "agent":
        # Agent: Apply agent patch to base commit Docker image
        repo_root = str(Path(sglang_path).parent.parent) if sglang_path else None

        if AGENT_PATCH and repo_root:
            print("[AGENT] Applying agent patch to base commit Docker image...")
            print(f"[AGENT] Repo root: {{repo_root}}")
            agent_patch_file = "/tmp/agent.patch"
            with open(agent_patch_file, "w") as f:
                f.write(AGENT_PATCH)

            # Agent patch might have different path structure, try -p1 first
            result = subprocess.run(
                ["patch", "-p1", "-d", repo_root, "-i", agent_patch_file],
                capture_output=True, text=True
            )
            if result.returncode != 0:
                print(f"[AGENT] Warning: Agent patch with -p1 failed, trying -p0...")
                # Try -p0 (no strip)
                result = subprocess.run(
                    ["patch", "-p0", "-d", repo_root, "-i", agent_patch_file],
                    capture_output=True, text=True
                )
                if result.returncode != 0:
                    print(f"[AGENT] Agent patch with -p0 also failed: {{result.stderr}}")
                else:
                    print("[AGENT] Successfully applied agent optimization with -p0")
            else:
                print("[AGENT] Successfully applied agent optimization")
        elif not AGENT_PATCH:
            print("[AGENT] No agent patch provided, using base Docker image as-is")

    # Run benchmark
    if needs_server(PERF_COMMAND):
        print(f"[{{PHASE.upper()}}] Starting server on port {{PORT}}...")
        server = start_server(MODEL, PORT)
        try:
            if not wait_for_server(PORT, timeout=600, proc=server):
                raise Exception("Server failed to start")
            print(f"[{{PHASE.upper()}}] Running benchmark...")
            output, metrics = run_benchmark(PORT, PERF_COMMAND)
        finally:
            stop_server(server)
    else:
        print(f"[{{PHASE.upper()}}] Running standalone benchmark...")
        output, metrics = run_benchmark(PORT, PERF_COMMAND)

    results["raw_output"] = output  # Full output, no truncation
    results["metrics"] = metrics
    results["status"] = "success"
    print(f"[{{PHASE.upper()}}] Metrics: {{metrics}}")

except Exception as e:
    import traceback
    results["error"] = str(e)
    results["status"] = "error"
    print(f"[{{PHASE.upper()}}] Error: {{e}}")
    traceback.print_exc()

# Save results to Modal volume
try:
    from datetime import datetime
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = Path(f"/results/{{COMMIT[:8]}}/{{timestamp}}")
    results_dir.mkdir(parents=True, exist_ok=True)

    # Save phase result
    result_file = results_dir / f"{{PHASE}}_result.json"
    with open(result_file, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"[{{PHASE.upper()}}] Saved results to Modal volume: {{result_file}}")
except Exception as save_err:
    print(f"[{{PHASE.upper()}}] Warning: Could not save to volume: {{save_err}}")

# Output results
print("\\n=== PHASE_RESULTS_JSON ===")
print(json.dumps(results))
print("=== END_PHASE_RESULTS_JSON ===")
'''
    return script


def _run_single_phase_sandbox(
    phase: str,
    commit: str,
    docker_image: str,
    perf_command: str,
    model: str,
    gpu_config: str,
    agent_patch: Optional[str] = None,
    base_commit: Optional[str] = None,
    human_patch: Optional[str] = None,
) -> Dict[str, Any]:
    """Run a single benchmark phase in its own sandbox.

    Uses SEPARATE DOCKER IMAGES approach (each phase uses its own image):
    - HUMAN: Human commit Docker image (has optimization)
    - BASELINE: Base commit Docker image (no patching needed!)
    - AGENT: Base commit Docker image + agent_patch
    """

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
    gpu_cfg = GPU_CONFIGS.get(gpu_config, GPU_CONFIGS["H100:1"])

    try:
        # Create the benchmark script
        script = _create_single_phase_script(
            phase=phase,
            commit=commit,
            perf_command=perf_command,
            model=model,
            agent_patch=agent_patch,
            base_commit=base_commit,
            human_patch=human_patch,
        )

        # Create Modal image
        # setup_dockerfile_commands runs before the image is validated
        # This fixes Python symlink, typing_extensions, and missing libnuma
        print(f"[{phase.upper()}] Creating Modal image from {docker_image}...", flush=True)
        image = modal.Image.from_registry(
            docker_image,
            force_build=True,
            setup_dockerfile_commands=[
                # Fix missing libnuma (required by sgl_kernel on newer SGLang versions)
                "RUN apt-get update && apt-get install -y libnuma1 libnuma-dev && rm -rf /var/lib/apt/lists/*",
                "RUN ln -sf $(which python3) /usr/local/bin/python || true",
                "RUN ln -sf $(which pip3) /usr/local/bin/pip || true",
                "RUN pip3 install 'typing_extensions>=4.10.0' --upgrade -q || true",
            ],
        )
        print(f"[{phase.upper()}] Modal image created", flush=True)

        # Get App reference
        print(f"[{phase.upper()}] Looking up Modal app 'sglang-benchmark'...", flush=True)
        sandbox_app = modal.App.lookup("sglang-benchmark", create_if_missing=True)
        print(f"[{phase.upper()}] Modal app ready", flush=True)

        print(f"[{phase.upper()}] Creating sandbox with {gpu_cfg['gpu']}...", flush=True)
        sandbox = modal.Sandbox.create(
            app=sandbox_app,
            image=image,
            gpu=f"{gpu_cfg['gpu']}:{gpu_cfg['count']}" if gpu_cfg['count'] > 1 else gpu_cfg['gpu'],
            timeout=gpu_cfg['timeout'],
            volumes={
                "/root/.cache/huggingface": model_cache,
                "/results": results_volume,
            },
            secrets=[modal.Secret.from_name("huggingface-secret")],
        )
        print(f"[{phase.upper()}] Sandbox created successfully", flush=True)

        # Write and run script
        script_path = f"/tmp/benchmark_{phase}.py"
        f = sandbox.open(script_path, "w")
        f.write(script)
        f.close()

        proc = sandbox.exec("python3", "-u", script_path)

        stdout_lines = []
        for line in proc.stdout:
            stdout_lines.append(line)
            print(f"[{phase.upper()}] {line.rstrip()}")

        proc.wait()

        # Save full raw output IMMEDIATELY before any other processing
        full_output = "\n".join(line.rstrip('\n') for line in stdout_lines)
        result["raw_output"] = full_output

        # Commit results volume to persist benchmark results
        try:
            results_volume.commit()
            print(f"[{phase.upper()}] Results volume committed", flush=True)
        except Exception as commit_err:
            print(f"[{phase.upper()}] Warning: Could not commit volume: {commit_err}", flush=True)

        sandbox.terminate()

        # Parse results
        json_match = re.search(
            r'=== PHASE_RESULTS_JSON ===\s*(\{.*\})\s*=== END_PHASE_RESULTS_JSON ===',
            full_output,
            re.DOTALL
        )

        if json_match:
            parsed = json.loads(json_match.group(1).strip())
            result["metrics"] = parsed.get("metrics", {})
            result["status"] = parsed.get("status", "error")
            if parsed.get("error"):
                result["error"] = parsed["error"]
        else:
            result["error"] = "No results found in output"

    except Exception as e:
        result["error"] = f"Sandbox error: {str(e)}"
        # Try to capture any partial output that was collected
        if 'stdout_lines' in locals() and stdout_lines:
            result["raw_output"] = "\n".join(line.rstrip('\n') for line in stdout_lines)

    result["duration_s"] = time.time() - start_time
    return result


def run_3way_benchmark_docker_parallel(
    human_commit: str,
    base_commit: Optional[str],
    agent_patch: Optional[str],
    perf_command: str,
    model: str,
    gpu_config: str = "H100:1",
    human_patch: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Run 3-way benchmark in PARALLEL using 3 separate Modal sandboxes.

    Each benchmark (baseline, human, agent) runs on its own GPU simultaneously.
    This is ~3x faster but uses 3x GPU hours.

    Uses SEPARATE DOCKER IMAGES approach (cleaner and more reliable):
    - HUMAN: Human commit Docker image (has optimization)
    - BASELINE: Base commit Docker image (pre-optimization, no patching!)
    - AGENT: Base commit Docker image + agent_patch

    Args:
        human_commit: Human commit hash (must have Docker image available)
        base_commit: Baseline/parent commit hash (must have Docker image available)
        agent_patch: Optional unified diff patch from agent
        perf_command: Benchmark command to run
        model: Model name/path
        gpu_config: GPU configuration (e.g., "H100:1")
        human_patch: Deprecated, no longer used (kept for API compatibility)

    Returns:
        Dict with baseline_metrics, human_metrics, agent_metrics, status, error
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed
    import sys

    print("[MODAL] Enabling Modal output...", flush=True)
    modal.enable_output()
    print("[MODAL] Modal output enabled", flush=True)

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
        "install_method": "docker_parallel",
        "benchmark_mode": "parallel_3way",
        # Raw outputs from each phase
        "baseline_raw": "",
        "human_raw": "",
        "agent_raw": "",
    }

    start_time = time.time()

    # Check Docker images for BOTH human and base commits
    # Each phase uses its own Docker image (no more reverse-patch approach)
    print(f"[MODAL] Checking Docker image for human commit {human_commit[:8]}...", flush=True)
    if not has_prebuilt_image(human_commit):
        result["error"] = f"No Docker image for human commit {human_commit[:8]}"
        result["duration_s"] = time.time() - start_time
        return result

    full_human_commit = get_prebuilt_commit(human_commit)
    human_docker_image = f"{SGLANG_DOCKER_REPO}:{full_human_commit}"
    print(f"[MODAL] Human Docker image: {human_docker_image}", flush=True)

    # Check baseline Docker image if base_commit provided
    base_docker_image = None
    if base_commit:
        print(f"[MODAL] Checking Docker image for base commit {base_commit[:8]}...", flush=True)
        if not has_prebuilt_image(base_commit):
            result["error"] = f"No Docker image for base commit {base_commit[:8]}"
            result["duration_s"] = time.time() - start_time
            return result
        full_base_commit = get_prebuilt_commit(base_commit)
        base_docker_image = f"{SGLANG_DOCKER_REPO}:{full_base_commit}"
        print(f"[MODAL] Base Docker image: {base_docker_image}", flush=True)

    print(f"=== PARALLEL 3-WAY BENCHMARK ===")
    print(f"Human Docker: {human_docker_image}")
    print(f"Base Docker: {base_docker_image or 'N/A'}")
    print(f"Human: {human_commit[:8]}, Base: {base_commit[:8] if base_commit else 'N/A'}")
    print(f"Agent patch: {'Yes' if agent_patch else 'No'}")
    print(f"GPU config: {gpu_config} x 3 sandboxes")

    # Prepare phases to run - each with its own Docker image
    # Format: (phase_name, commit, docker_image, agent_patch)
    phases = []

    # HUMAN phase uses human commit Docker image
    phases.append(("human", human_commit, human_docker_image, None))

    # BASELINE phase uses base commit Docker image (no patching needed!)
    if base_commit and base_docker_image:
        phases.append(("baseline", base_commit, base_docker_image, None))

    # AGENT phase uses base commit Docker image + agent patch
    if agent_patch and base_commit and base_docker_image:
        phases.append(("agent", base_commit, base_docker_image, agent_patch))

    print(f"Running {len(phases)} phases in parallel: {[p[0] for p in phases]}")

    # Run phases in parallel
    phase_results = {}
    with ThreadPoolExecutor(max_workers=3) as executor:
        futures = {}
        for phase, commit, docker_image, patch in phases:
            future = executor.submit(
                _run_single_phase_sandbox,
                phase=phase,
                commit=commit,
                docker_image=docker_image,
                perf_command=perf_command,
                model=model,
                gpu_config=gpu_config,
                agent_patch=patch,
                base_commit=None,  # No longer needed - using separate Docker images
                human_patch=None,  # No longer needed - no reverse-patch
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
    errors = []

    if "human" in phase_results:
        result["human_metrics"] = phase_results["human"].get("metrics", {})
        result["human_raw"] = phase_results["human"].get("raw_output", "")
        if phase_results["human"].get("status") == "success":
            result["status"] = "success"
        elif phase_results["human"].get("error"):
            errors.append(f"HUMAN: {phase_results['human']['error']}")

    if "baseline" in phase_results:
        result["baseline_metrics"] = phase_results["baseline"].get("metrics", {})
        result["baseline_raw"] = phase_results["baseline"].get("raw_output", "")
        if phase_results["baseline"].get("error"):
            errors.append(f"BASELINE: {phase_results['baseline']['error']}")

    if "agent" in phase_results:
        result["agent_metrics"] = phase_results["agent"].get("metrics", {})
        result["agent_raw"] = phase_results["agent"].get("raw_output", "")
        if phase_results["agent"].get("error"):
            errors.append(f"AGENT: {phase_results['agent']['error']}")

    # Capture all errors
    if errors:
        result["error"] = "; ".join(errors)

    # Calculate improvements
    if result["baseline_metrics"] and result["human_metrics"]:
        for k in result["baseline_metrics"]:
            if k in result["human_metrics"] and result["baseline_metrics"][k] > 0:
                base_v = result["baseline_metrics"][k]
                human_v = result["human_metrics"][k]
                if "throughput" in k:
                    result["human_improvement"][k] = (human_v - base_v) / base_v * 100
                else:
                    result["human_improvement"][k] = (base_v - human_v) / base_v * 100

    if result["baseline_metrics"] and result.get("agent_metrics"):
        agent_imp = {}
        for k in result["baseline_metrics"]:
            if k in result["agent_metrics"] and result["baseline_metrics"][k] > 0:
                base_v = result["baseline_metrics"][k]
                agent_v = result["agent_metrics"][k]
                if "throughput" in k:
                    agent_imp[k] = (agent_v - base_v) / base_v * 100
                else:
                    agent_imp[k] = (base_v - agent_v) / base_v * 100
        result["agent_improvement"] = agent_imp

    if result.get("human_metrics") and result.get("agent_metrics"):
        agent_vs_human = {}
        for k in result["human_metrics"]:
            if k in result["agent_metrics"] and result["human_metrics"][k] > 0:
                human_v = result["human_metrics"][k]
                agent_v = result["agent_metrics"][k]
                if "throughput" in k:
                    agent_vs_human[k] = (agent_v - human_v) / human_v * 100
                else:
                    agent_vs_human[k] = (human_v - agent_v) / human_v * 100
        result["agent_vs_human"] = agent_vs_human

    result["duration_s"] = time.time() - start_time
    result["phase_durations"] = {p: r.get("duration_s", 0) for p, r in phase_results.items()}

    print(f"\n=== PARALLEL BENCHMARK COMPLETE ===")
    print(f"Total duration: {result['duration_s']:.1f}s")
    print(f"Phase durations: {result['phase_durations']}")

    return result


# =====================================================================
# Local Runner Interface
# =====================================================================

def ensure_wheel_built(commit_hash: str, force_rebuild: bool = False) -> Dict[str, Any]:
    """
    Ensure a wheel is built for the given commit.

    Calls the Modal function to build wheel (will use cache if available).

    Args:
        commit_hash: SGLang commit to build
        force_rebuild: If True, rebuild even if cached

    Returns:
        Build result dict with success, wheel_path, cache_hit, error
    """
    import traceback

    print(f"[ENSURE BUILD] Checking/building SGLang {commit_hash[:8]} on CPU-only instance...")

    try:
        # Call the CPU-only build function on Modal
        fn = modal.Function.from_name("sglang-benchmark", "build_sglang_cpu_only")
        with modal.enable_output():
            result = fn.remote(commit_hash, force_rebuild=force_rebuild)

        if result["success"]:
            if result["cache_hit"]:
                print(f"[ENSURE BUILD] Cache HIT: SGLang {result.get('version', 'unknown')} already cached")
            else:
                print(f"[ENSURE BUILD] Built and cached: SGLang {result.get('version', 'unknown')}")
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
            "error": f"Build invocation failed: {str(e)}",
            "commit": commit_hash,
        }


def ensure_agent_wheel_built(base_commit: str, patch_content: str) -> Dict[str, Any]:
    """
    Ensure a patched wheel is built for the agent.

    First ensures the base commit is built, then applies patch and builds.

    Args:
        base_commit: Base commit hash (must be built first)
        patch_content: Unified diff patch from agent

    Returns:
        Build result dict with success, wheel_path, patch_hash, error
    """
    import traceback
    import hashlib

    patch_hash = hashlib.sha256(patch_content.encode()).hexdigest()[:12]
    print(f"[ENSURE AGENT BUILD] Building patched wheel for base {base_commit[:8]}, patch {patch_hash}...")

    try:
        # First ensure base commit is built (required for incremental build)
        base_result = ensure_wheel_built(base_commit)
        if not base_result["success"]:
            return {
                "success": False,
                "error": f"Base build failed: {base_result.get('error', 'unknown')}",
                "patch_hash": patch_hash,
            }

        # Now build the patched version
        fn = modal.Function.from_name("sglang-benchmark", "build_sglang_agent_patch")
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
            "error": f"Agent build invocation failed: {str(e)}",
            "patch_hash": patch_hash,
        }


def run_3way_modal_benchmark(
    base_commit: str,
    human_commit: str,
    agent_patch: Optional[str],
    perf_command: str,
    model: str,
    gpu_config: str = "H100:1",
    use_wheel: bool = True,
) -> Dict[str, Any]:
    """
    Run 3-way benchmark on Modal.

    This is the main entry point for running benchmarks from local code.

    Two approaches available:
    1. Wheel-based (use_wheel=True, default): Build wheels on CPU, run benchmark on GPU
       - More reliable: wheels include ALL code (Python + compiled extensions)
       - Handles structural changes between commits
       - Cached in Modal volume for fast subsequent runs

    2. Docker-based (use_wheel=False): Pre-built Docker images with Python overlay
       - Faster first run (no wheel build needed)
       - May fail if module structure differs between commits

    Args:
        base_commit: Baseline (parent) commit hash
        human_commit: Human commit hash
        agent_patch: Optional unified diff patch from agent
        perf_command: Benchmark command to run
        model: Model name/path
        gpu_config: GPU configuration (e.g., "H100:1", "H100:4")
        use_wheel: If True, use wheel-based approach (default)
    """
    print(f"Running SGLang 3-way benchmark on Modal with {gpu_config}...")
    print(f"  Base commit: {base_commit[:8]}")
    print(f"  Human commit: {human_commit[:8]}")
    print(f"  Agent patch: {'yes (' + str(len(agent_patch)) + ' bytes)' if agent_patch else 'no'}")
    print(f"  Model: {model}")
    print(f"  Command: {perf_command[:80]}...")
    print(f"  Method: {'wheel' if use_wheel else 'docker'}")

    if use_wheel:
        # ===================================================================
        # WHEEL-BASED APPROACH (recommended)
        # ===================================================================
        # Phase 1: Build wheels (CPU-only, cached in Modal volume)
        print("\n" + "=" * 60)
        print("PHASE 1: Building wheels (CPU-only)")
        print("=" * 60)

        # Build baseline wheel first (required for agent patch)
        print(f"\n[1/3] Building baseline wheel ({base_commit[:8]})...")
        baseline_result = ensure_wheel_built(base_commit)
        if not baseline_result["success"]:
            return {
                "status": "error",
                "error": f"Baseline wheel build failed: {baseline_result.get('error', 'unknown')}",
                "baseline_metrics": {},
                "human_metrics": {},
                "agent_metrics": None,
            }

        # Build human wheel
        print(f"\n[2/3] Building human wheel ({human_commit[:8]})...")
        human_result = ensure_wheel_built(human_commit)
        if not human_result["success"]:
            return {
                "status": "error",
                "error": f"Human wheel build failed: {human_result.get('error', 'unknown')}",
                "baseline_metrics": {},
                "human_metrics": {},
                "agent_metrics": None,
            }

        # Build agent wheel (if patch provided)
        agent_wheel_path = None
        if agent_patch:
            print(f"\n[3/3] Building agent wheel (patch on {base_commit[:8]})...")
            agent_result = ensure_agent_wheel_built(base_commit, agent_patch)
            if not agent_result["success"]:
                print(f"WARNING: Agent wheel build failed: {agent_result.get('error', 'unknown')}")
                print("Will continue with baseline vs human comparison only")
            else:
                agent_wheel_path = agent_result.get("wheel_path")

        # Phase 2: Run benchmark (GPU)
        print("\n" + "=" * 60)
        print("PHASE 2: Running benchmark (GPU)")
        print("=" * 60)

        # Select GPU function based on config
        if gpu_config == "H100:8":
            fn_name = "run_3way_benchmark_wheel_8gpu"
        elif gpu_config == "H100:4":
            fn_name = "run_3way_benchmark_wheel_4gpu"
        elif gpu_config == "H100:2":
            fn_name = "run_3way_benchmark_wheel_2gpu"
        else:
            fn_name = "run_3way_benchmark_wheel"

        print(f"Using function: {fn_name}")
        fn = modal.Function.from_name("sglang-benchmark", fn_name)

        try:
            with modal.enable_output():
                result = fn.remote(
                    baseline_commit=base_commit,
                    human_commit=human_commit,
                    agent_patch=agent_patch,
                    perf_command=perf_command,
                    model=model,
                )
            return result
        except Exception as e:
            return {
                "status": "error",
                "error": f"Benchmark execution failed: {str(e)}",
                "baseline_metrics": {},
                "human_metrics": {},
                "agent_metrics": None,
            }

    else:
        # ===================================================================
        # DOCKER-BASED APPROACH (fallback)
        # ===================================================================
        # Check for Docker image
        if not has_prebuilt_image(human_commit):
            return {
                "status": "error",
                "error": f"No Docker image available for human commit {human_commit[:8]}. "
                         f"Please build images using: python tools/build_sglang_images.py --commit {human_commit}",
                "baseline_metrics": {},
                "human_metrics": {},
                "agent_metrics": None,
            }

        return run_3way_benchmark_docker(
            human_commit=human_commit,
            base_commit=base_commit,
            agent_patch=agent_patch,
            perf_command=perf_command,
            model=model,
            gpu_config=gpu_config,
        )


# =====================================================================
# Modal Function (for remote execution)
# =====================================================================

@app.function(
    image=modal.Image.debian_slim().pip_install("requests"),
    timeout=300,
)
def check_image_exists(commit: str) -> bool:
    """Check if Docker image exists for commit (Modal function for remote checks)."""
    return has_prebuilt_image(commit)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run SGLang benchmark on Modal")
    parser.add_argument("--base-commit", required=True, help="Base commit hash")
    parser.add_argument("--human-commit", required=True, help="Human commit hash")
    parser.add_argument("--agent-patch", help="Path to agent patch file")
    parser.add_argument("--perf-command", required=True, help="Performance benchmark command")
    parser.add_argument("--model", required=True, help="Model name")
    parser.add_argument("--gpu-config", default="H100:1", help="GPU configuration")

    args = parser.parse_args()

    agent_patch = None
    if args.agent_patch:
        with open(args.agent_patch, 'r') as f:
            agent_patch = f.read()

    result = run_3way_modal_benchmark(
        base_commit=args.base_commit,
        human_commit=args.human_commit,
        agent_patch=agent_patch,
        perf_command=args.perf_command,
        model=args.model,
        gpu_config=args.gpu_config,
    )

    print("\n" + "=" * 60)
    print("BENCHMARK RESULTS")
    print("=" * 60)
    print(f"Status: {result['status']}")
    if result.get('error'):
        print(f"Error: {result['error']}")
    print(f"Baseline metrics: {result.get('baseline_metrics', {})}")
    print(f"Human metrics: {result.get('human_metrics', {})}")
    print(f"Agent metrics: {result.get('agent_metrics')}")
    if result.get('human_improvement'):
        print(f"Human improvement: {result['human_improvement']}")
    if result.get('agent_improvement'):
        print(f"Agent improvement: {result['agent_improvement']}")
