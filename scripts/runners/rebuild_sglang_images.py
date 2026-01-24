#!/usr/bin/env python3
"""
Rebuild SGLang Docker images with dependency fixes.

This script rebuilds broken Docker images with proper dependency pinning
to resolve the compatibility issues found during testing.

Key fixes:
- transformers==4.44.2 (avoids torchao conflict)
- outlines==0.0.46 (has outlines.fsm module)
- Missing packages: pyairports, orjson, sentencepiece
- libnuma-dev for sgl_kernel
"""

import subprocess
import json
import time
import os
import re
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime

# Configuration
DOCKER_REPO = "ayushnangia16/nvidia-sglang-docker"
WORK_DIR = Path("/tmp/sglang_rebuild")
RESULTS_DIR = Path("/home/ubuntu/OmniPerf-Bench/omniperf_results_3way_sglang/rebuilt_images")

# Fixed Dockerfile for SGLang 0.3.x (uses torch 2.4.0 + flashinfer 0.1.6)
FIXED_DOCKERFILE_03X = '''
# SGLang Docker image for v0.3.x with dependency fixes
# Uses PyTorch 2.4.0 + flashinfer 0.1.6

FROM nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04

ARG COMMIT_HASH=main
ARG SGLANG_VERSION=unknown

ENV DEBIAN_FRONTEND=noninteractive \\
    CUDA_HOME=/usr/local/cuda \\
    PATH="${PATH}:/usr/local/cuda/bin" \\
    LD_LIBRARY_PATH="${LD_LIBRARY_PATH}:/usr/local/cuda/lib64" \\
    MAX_JOBS=32 \\
    NVCC_THREADS=2

# Install system dependencies (including libnuma for sgl_kernel)
RUN apt-get update && apt-get install -y --no-install-recommends \\
    python3.11 python3.11-dev python3.11-venv python3-pip \\
    git curl wget build-essential cmake ninja-build \\
    libopenmpi-dev libnuma-dev \\
    && ln -sf /usr/bin/python3.11 /usr/bin/python3 \\
    && ln -sf /usr/bin/python3.11 /usr/bin/python \\
    && rm -rf /var/lib/apt/lists/* \\
    && apt-get clean

# Upgrade pip
RUN python3 -m pip install --upgrade pip setuptools wheel

# Install PyTorch 2.4.0 with CUDA 12.4 support (SGLang 0.3.x compatible)
RUN pip install torch==2.4.0 torchvision --index-url https://download.pytorch.org/whl/cu124

# Pin transformers to 4.44.2 (has AutoProcessor, works with older torchao)
RUN pip install "transformers==4.44.2"

# Install other dependencies
RUN pip install "huggingface_hub>=0.23.0" "accelerate>=0.30.0" "numpy<2.0" requests aiohttp packaging ninja

# Install triton
RUN pip install "triton>=3.0.0"

# Install flashinfer 0.1.6 (SGLang 0.3.x API)
RUN pip install "flashinfer==0.1.6" -i https://flashinfer.ai/whl/cu124/torch2.4/ || \\
    pip install flashinfer -i https://flashinfer.ai/whl/cu124/torch2.4/ || \\
    echo "flashinfer install failed"

# Install missing packages
RUN pip install orjson sentencepiece "outlines==0.0.44" pybase64 Pillow pandas datasets tqdm \\
    uvicorn httptools uvloop fastapi starlette python-multipart pydantic \\
    pyzmq msgpack psutil aiofiles python-dotenv

# Install real pyairports (the PyPI version is a placeholder)
COPY pyairports /usr/local/lib/python3.11/dist-packages/pyairports

# Clone SGLang at specific commit
WORKDIR /opt
RUN git clone https://github.com/sgl-project/sglang.git sglang \\
    && cd sglang \\
    && git checkout ${COMMIT_HASH}

# Install SGLang from source
WORKDIR /opt/sglang
RUN pip install -e "python[all]" || pip install -e "python"

# Re-pin packages AFTER SGLang installation (it may upgrade them)
RUN pip install "flashinfer==0.1.6" --force-reinstall -i https://flashinfer.ai/whl/cu124/torch2.4/ && \\
    pip install "transformers>=4.46.0,<4.50.0" --upgrade && \\
    pip install "numpy<2.0" --force-reinstall && \\
    pip install "outlines>=0.0.46,<0.1.0" || pip install "outlines==0.0.44" --no-deps

# Verify flashinfer 0.1.x API works
RUN python3 -c "from flashinfer.decode import _grouped_size_compiled_for_decode_kernels; print('flashinfer 0.1.x OK')"

# Verify installation
RUN python3 -c "import sglang; print(f'SGLang version: {sglang.__version__}')"

# Verify critical imports work (non-fatal)
RUN python3 -c "from transformers import AutoProcessor; print('AutoProcessor OK')" || echo "AutoProcessor import skipped"
RUN python3 -c "import orjson; print('orjson OK')" || echo "orjson skipped"
RUN python3 -c "import sentencepiece; print('sentencepiece OK')" || echo "sentencepiece skipped"

WORKDIR /workspace
RUN cp -r /opt/sglang/python/sglang/bench_* /workspace/ 2>/dev/null || true
RUN cp -r /opt/sglang/benchmark* /workspace/ 2>/dev/null || true
RUN echo "${SGLANG_VERSION}" > /workspace/.sglang_version
RUN echo "${COMMIT_HASH}" > /workspace/.commit_hash

CMD ["python3", "-c", "import sglang; print(f'SGLang {sglang.__version__} ready')"]
'''

# Fixed Dockerfile for SGLang 0.4.x+ (lets SGLang install its own torch, pins compressed_tensors)
FIXED_DOCKERFILE_04X = '''
# SGLang Docker image for v0.4.x+ with dependency fixes
# Lets SGLang install its own torch version (2.10.0), fixes compressed_tensors

FROM nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04

ARG COMMIT_HASH=main
ARG SGLANG_VERSION=unknown

ENV DEBIAN_FRONTEND=noninteractive \\
    CUDA_HOME=/usr/local/cuda \\
    PATH="${PATH}:/usr/local/cuda/bin" \\
    LD_LIBRARY_PATH="${LD_LIBRARY_PATH}:/usr/local/cuda/lib64" \\
    MAX_JOBS=32 \\
    NVCC_THREADS=2

# Install system dependencies (including libnuma for sgl_kernel)
RUN apt-get update && apt-get install -y --no-install-recommends \\
    python3.11 python3.11-dev python3.11-venv python3-pip \\
    git curl wget build-essential cmake ninja-build \\
    libopenmpi-dev libnuma-dev \\
    && ln -sf /usr/bin/python3.11 /usr/bin/python3 \\
    && ln -sf /usr/bin/python3.11 /usr/bin/python \\
    && rm -rf /var/lib/apt/lists/* \\
    && apt-get clean

# Upgrade pip
RUN python3 -m pip install --upgrade pip setuptools wheel

# Pre-install some dependencies (don't pre-install torch - let SGLang choose)
RUN pip install requests aiohttp packaging ninja

# Install missing packages (before SGLang to ensure they're available)
RUN pip install orjson sentencepiece pybase64 Pillow pandas datasets tqdm \\
    uvicorn httptools uvloop fastapi starlette python-multipart pydantic \\
    pyzmq msgpack psutil aiofiles python-dotenv

# Install real pyairports (the PyPI version is a placeholder)
COPY pyairports /usr/local/lib/python3.11/dist-packages/pyairports

# Clone SGLang at specific commit
WORKDIR /opt
RUN git clone https://github.com/sgl-project/sglang.git sglang \\
    && cd sglang \\
    && git checkout ${COMMIT_HASH}

# Install SGLang from source (lets it install its own torch 2.10.0)
WORKDIR /opt/sglang
RUN pip install -e "python[all]" || pip install -e "python"

# CRITICAL FIX: Pin compressed_tensors<0.13.0 to avoid masking_utils import error
# v0.13.0 requires transformers.masking_utils which doesn't exist yet
RUN pip install "compressed-tensors<0.13.0" --force-reinstall

# Pin other packages after SGLang install
RUN pip install "numpy<2.0" --force-reinstall && \\
    pip install "outlines>=0.0.46,<0.1.0" || pip install "outlines==0.0.44" --no-deps

# Verify compressed_tensors version
RUN python3 -c "import compressed_tensors; print(f'compressed_tensors: {compressed_tensors.__version__}')"

# Verify installation
RUN python3 -c "import sglang; print(f'SGLang version: {sglang.__version__}')"

# Verify critical imports work (non-fatal)
RUN python3 -c "import orjson; print('orjson OK')" || echo "orjson skipped"
RUN python3 -c "import sentencepiece; print('sentencepiece OK')" || echo "sentencepiece skipped"

WORKDIR /workspace
RUN cp -r /opt/sglang/python/sglang/bench_* /workspace/ 2>/dev/null || true
RUN cp -r /opt/sglang/benchmark* /workspace/ 2>/dev/null || true
RUN echo "${SGLANG_VERSION}" > /workspace/.sglang_version
RUN echo "${COMMIT_HASH}" > /workspace/.commit_hash

CMD ["python3", "-c", "import sglang; print(f'SGLang {sglang.__version__} ready')"]
'''

# Alias for backward compatibility
FIXED_DOCKERFILE = FIXED_DOCKERFILE_03X

# Alternative Dockerfile for older SGLang versions (0.1.x - 0.2.x)
LEGACY_DOCKERFILE = '''
# SGLang Docker image for older versions (0.1.x - 0.2.x)
# Uses older dependency versions compatible with these releases

ARG CUDA_VERSION=12.1.0
FROM nvidia/cuda:${CUDA_VERSION}-cudnn8-devel-ubuntu22.04

ARG COMMIT_HASH=main

ENV DEBIAN_FRONTEND=noninteractive \\
    CUDA_HOME=/usr/local/cuda \\
    PATH="${PATH}:/usr/local/cuda/bin"

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \\
    python3.10 python3.10-dev python3.10-venv python3-pip \\
    git curl wget build-essential cmake \\
    libopenmpi-dev libnuma-dev \\
    && ln -sf /usr/bin/python3.10 /usr/bin/python3 \\
    && ln -sf /usr/bin/python3.10 /usr/bin/python \\
    && rm -rf /var/lib/apt/lists/*

RUN python3 -m pip install --upgrade pip setuptools wheel

# Older PyTorch for legacy versions
RUN pip install torch==2.1.0 --index-url https://download.pytorch.org/whl/cu121

# Older transformers that works with legacy SGLang
RUN pip install "transformers==4.36.0" "tokenizers>=0.15.0"

RUN pip install \\
    huggingface_hub>=0.20.0 \\
    accelerate>=0.25.0 \\
    numpy<2.0 \\
    requests aiohttp packaging ninja triton

# Legacy flashinfer
RUN pip install flashinfer -i https://flashinfer.ai/whl/cu121/torch2.1/ || true

# Missing packages
RUN pip install pyairports orjson sentencepiece pybase64 Pillow pandas datasets tqdm

# Clone and install SGLang
WORKDIR /opt
RUN git clone https://github.com/sgl-project/sglang.git sglang \\
    && cd sglang && git checkout ${COMMIT_HASH}

WORKDIR /opt/sglang
RUN pip install -e "python[all]" || pip install -e "python" || pip install -e .

RUN python3 -c "import sglang; print(f'SGLang: {sglang.__version__}')"

WORKDIR /workspace
CMD ["python3", "-c", "import sglang; print(f'SGLang {sglang.__version__} ready')"]
'''


def get_broken_images():
    """Get list of broken images from test results."""
    results_file = Path("/home/ubuntu/OmniPerf-Bench/omniperf_results_3way_sglang/all_images_benchmark/all_images_results.json")

    if not results_file.exists():
        print("No test results found. Run run_all_sglang_images.py first.")
        return []

    with open(results_file) as f:
        data = json.load(f)

    broken = []
    for r in data['results']:
        if r['status'] != 'success':
            # Extract commit hash from image name
            image = r['image']
            tag = image.split(':')[-1]

            # Get short commit (first 8 chars of tag, removing suffixes)
            commit = tag.split('-')[0][:8]

            broken.append({
                'image': image,
                'tag': tag,
                'commit': commit,
                'version': r.get('sglang_version', 'unknown'),
                'error': r.get('error', 'Unknown'),
            })

    return broken


def get_commit_for_rebuild(tag: str) -> str:
    """Extract full commit hash from image tag."""
    # Tags can be:
    # - Full 40-char hash: abc123...
    # - Short with suffix: abc123-src, abc123-v3
    base = tag.split('-')[0]
    return base


def determine_dockerfile_type(version: str) -> str:
    """Determine which Dockerfile to use based on SGLang version."""
    if not version or version == 'unknown':
        return 'fixed_03x'  # Default to 0.3.x which is most common

    # Parse version
    try:
        parts = version.replace('post', '.').split('.')
        major = int(parts[0])
        minor = int(parts[1]) if len(parts) > 1 else 0

        # Legacy versions (0.1.x, 0.2.x)
        if major == 0 and minor <= 2:
            return 'legacy'
        # SGLang 0.4.x+ uses torch 2.10.0, needs compressed_tensors fix
        elif major == 0 and minor >= 4:
            return 'fixed_04x'
        # SGLang 0.3.x uses torch 2.4.0 + flashinfer 0.1.6
        else:
            return 'fixed_03x'
    except:
        return 'fixed_03x'


def resolve_commit(commit: str, repo_dir: Path) -> str:
    """Resolve short commit hash to full hash."""
    # First try direct rev-parse
    proc = subprocess.run(
        ["git", "rev-parse", commit],
        cwd=repo_dir, capture_output=True, text=True, timeout=30
    )
    if proc.returncode == 0:
        return proc.stdout.strip()

    # Try to find matching commit with git log
    proc = subprocess.run(
        ["git", "log", "--all", "--oneline", f"--grep={commit[:8]}"],
        cwd=repo_dir, capture_output=True, text=True, timeout=30
    )
    if proc.returncode == 0 and proc.stdout.strip():
        first_line = proc.stdout.strip().split('\n')[0]
        return first_line.split()[0]

    # Try broader search
    proc = subprocess.run(
        ["git", "log", "--all", "--oneline", "-500"],
        cwd=repo_dir, capture_output=True, text=True, timeout=60
    )
    if proc.returncode == 0:
        for line in proc.stdout.split('\n'):
            if line.startswith(commit[:7]):
                return line.split()[0]

    return commit


def build_fixed_image(
    original_image: str,
    commit: str,
    version: str,
    gpu_id: int = 0,
    push: bool = True
) -> dict:
    """Build a fixed Docker image for a broken SGLang commit."""

    result = {
        'original_image': original_image,
        'commit': commit,
        'version': version,
        'status': 'error',
        'new_tag': None,
        'error': None,
        'duration_s': 0,
    }

    start_time = time.time()

    # Create work directory (use gpu_id for parallel isolation)
    work_dir = WORK_DIR / f"worker_{gpu_id}"
    work_dir.mkdir(parents=True, exist_ok=True)

    try:
        # Clone SGLang repo (full clone to find commits)
        repo_dir = work_dir / "sglang"
        if not repo_dir.exists():
            print(f"  Cloning SGLang repo (full)...")
            proc = subprocess.run(
                ["git", "clone", "https://github.com/sgl-project/sglang.git", str(repo_dir)],
                capture_output=True, text=True, timeout=600
            )
            if proc.returncode != 0:
                result['error'] = f"Clone failed: {proc.stderr[:200]}"
                return result
        else:
            # Fetch latest
            subprocess.run(["git", "fetch", "--all"], cwd=repo_dir, capture_output=True, timeout=120)

        # Resolve commit hash
        print(f"  Resolving commit {commit[:8]}...")
        full_commit = resolve_commit(commit, repo_dir)
        print(f"  Resolved to: {full_commit[:12]}")

        # Determine new tag
        new_tag = f"fixed-{full_commit[:12]}"
        result['new_tag'] = new_tag

        # Checkout commit
        proc = subprocess.run(
            ["git", "checkout", full_commit],
            cwd=repo_dir, capture_output=True, text=True, timeout=60
        )
        if proc.returncode != 0:
            result['error'] = f"Checkout failed: {proc.stderr[:200]}"
            return result

        # Copy pyairports module to build context
        pyairports_src = Path("/tmp/pyairports_real")
        pyairports_dst = repo_dir / "pyairports"
        if pyairports_src.exists():
            import shutil
            if pyairports_dst.exists():
                shutil.rmtree(pyairports_dst)
            shutil.copytree(pyairports_src, pyairports_dst)
            print(f"  Copied pyairports module to build context")
        else:
            print(f"  Warning: pyairports source not found at {pyairports_src}")

        # Write Dockerfile based on SGLang version
        dockerfile_type = determine_dockerfile_type(version)
        if dockerfile_type == 'legacy':
            dockerfile_content = LEGACY_DOCKERFILE
        elif dockerfile_type == 'fixed_04x':
            dockerfile_content = FIXED_DOCKERFILE_04X
        else:
            dockerfile_content = FIXED_DOCKERFILE_03X

        print(f"  Using Dockerfile: {dockerfile_type}")

        # Write Dockerfile to repo_dir (build context)
        dockerfile_path = repo_dir / "Dockerfile.fixed"
        dockerfile_path.write_text(dockerfile_content)

        # Build Docker image
        image_name = f"{DOCKER_REPO}:{new_tag}"
        print(f"  Building {image_name}...")

        build_cmd = [
            "docker", "build",
            "--no-cache",  # Force fresh build to ensure CUDA 12.4.1 is used
            "-f", str(dockerfile_path),
            "-t", image_name,
            "--build-arg", f"COMMIT_HASH={full_commit}",
            "--build-arg", f"SGLANG_VERSION={version}",
            str(repo_dir)
        ]

        proc = subprocess.run(
            build_cmd,
            capture_output=True, text=True, timeout=3600
        )

        if proc.returncode != 0:
            # Extract meaningful error from output
            error_lines = [l for l in proc.stdout.split('\n') + proc.stderr.split('\n') if 'error' in l.lower()]
            result['error'] = f"Build failed: {error_lines[-1] if error_lines else proc.stderr[-500:]}"
            return result

        print(f"  Build successful!")

        # Test the image
        print(f"  Testing image...")
        test_cmd = [
            "docker", "run", "--rm",
            "--gpus", f"device={gpu_id}",
            image_name,
            "python3", "-c", "import sglang; print(f'OK: {sglang.__version__}')"
        ]

        proc = subprocess.run(test_cmd, capture_output=True, text=True, timeout=120)
        if proc.returncode != 0:
            result['error'] = f"Test failed: {proc.stderr[:200]}"
            return result

        print(f"  Test passed: {proc.stdout.strip()}")

        # Push to DockerHub
        if push:
            print(f"  Pushing to DockerHub...")
            proc = subprocess.run(
                ["docker", "push", image_name],
                capture_output=True, text=True, timeout=1800
            )
            if proc.returncode != 0:
                result['error'] = f"Push failed: {proc.stderr[:200]}"
                return result
            print(f"  Pushed successfully!")

        result['status'] = 'success'

        # Run benchmark on the successfully built image
        print(f"  Running benchmark...")
        benchmark_result = run_benchmark_on_image(image_name, gpu_id)
        result['benchmark'] = benchmark_result
        if benchmark_result.get('status') == 'success':
            print(f"  Benchmark passed: {benchmark_result.get('metrics', {}).get('output_throughput', 0):.0f} tok/s")
        else:
            print(f"  Benchmark failed: {benchmark_result.get('error', 'Unknown')[:50]}")

    except subprocess.TimeoutExpired:
        result['error'] = "Timeout"
    except Exception as e:
        result['error'] = str(e)

    result['duration_s'] = time.time() - start_time
    return result


def run_benchmark_on_image(image: str, gpu_id: int) -> dict:
    """Run benchmark on a Docker image."""
    result = {
        'image': image,
        'status': 'error',
        'metrics': {},
        'error': None,
    }

    docker_cmd = '''
    timeout 180 python3 -m sglang.launch_server \\
        --model-path TinyLlama/TinyLlama-1.1B-Chat-v1.0 \\
        --port 30000 \\
        --host 0.0.0.0 \\
        --tp-size 1 \\
        --mem-fraction-static 0.8 \\
        2>&1 &
    SERVER_PID=$!

    # Wait for server
    for i in $(seq 1 120); do
        if curl -s http://localhost:30000/health 2>/dev/null | grep -qiE "ok|healthy|true"; then
            echo "SERVER_READY"
            break
        fi
        if curl -s http://localhost:30000/get_model_info 2>/dev/null | grep -qiE "model"; then
            echo "SERVER_READY"
            break
        fi
        if ! kill -0 $SERVER_PID 2>/dev/null; then
            echo "SERVER_CRASHED"
            exit 1
        fi
        sleep 1
    done

    # Run benchmark
    python3 -m sglang.bench_serving \\
        --backend sglang \\
        --host 127.0.0.1 \\
        --port 30000 \\
        --num-prompts 10 \\
        --random-input-len 128 \\
        --random-output-len 32 \\
        2>&1 || echo "BENCHMARK_FAILED"

    echo "BENCHMARK_DONE"
    kill $SERVER_PID 2>/dev/null || true
    '''

    try:
        hf_token = os.environ.get('HF_TOKEN', '')
        proc = subprocess.run(
            [
                'docker', 'run', '--rm',
                '--gpus', f'device={gpu_id}',
                '-e', f'HF_TOKEN={hf_token}',
                '-v', '/home/ubuntu/.cache/huggingface:/root/.cache/huggingface',
                '--shm-size=16g',
                image,
                'bash', '-c', docker_cmd
            ],
            capture_output=True, text=True, timeout=300
        )

        output = proc.stdout + proc.stderr

        if "BENCHMARK_DONE" in output and "BENCHMARK_FAILED" not in output:
            result['status'] = 'success'
            # Parse metrics
            import re
            patterns = {
                'request_throughput': r'Request throughput \(req/s\):\s+([\d.]+)',
                'output_throughput': r'Output token throughput \(tok/s\):\s+([\d.]+)',
            }
            for key, pattern in patterns.items():
                match = re.search(pattern, output)
                if match:
                    result['metrics'][key] = float(match.group(1))
        elif "SERVER_CRASHED" in output:
            result['error'] = "Server crashed"
        else:
            result['error'] = "Benchmark failed"

    except subprocess.TimeoutExpired:
        result['error'] = "Timeout"
    except Exception as e:
        result['error'] = str(e)

    return result


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Rebuild broken SGLang Docker images with fixes")
    parser.add_argument("--commit", help="Rebuild specific commit")
    parser.add_argument("--all", action="store_true", help="Rebuild all broken images")
    parser.add_argument("--max", type=int, default=10, help="Max images to rebuild")
    parser.add_argument("--offset", type=int, default=0, help="Skip first N images (for parallel runs)")
    parser.add_argument("--no-push", action="store_true", help="Don't push to DockerHub")
    parser.add_argument("--gpu", type=int, default=0, help="GPU to use for testing")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be rebuilt")
    args = parser.parse_args()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    if args.commit:
        # Rebuild single commit
        to_rebuild = [{
            'commit': args.commit,
            'image': f"manual:{args.commit}",
            'version': 'unknown',
            'error': 'manual rebuild'
        }]
    elif args.all:
        # Get all broken images
        to_rebuild = get_broken_images()
        print(f"Found {len(to_rebuild)} broken images")

        # Prioritize by error type - fix the ones that are most likely fixable
        fixable_errors = ['AutoProcessor', 'pyairports', 'orjson', 'sentencepiece', 'outlines']

        def is_likely_fixable(item):
            error = item.get('error', '')
            return any(e in error for e in fixable_errors)

        # Sort: likely fixable first
        to_rebuild.sort(key=lambda x: (not is_likely_fixable(x), x.get('error', '')))

        # Apply offset and limit for parallel runs
        to_rebuild = to_rebuild[args.offset:args.offset + args.max]
    else:
        parser.print_help()
        return

    if args.dry_run:
        print(f"\n=== DRY RUN - Would rebuild {len(to_rebuild)} images ===")
        for i, item in enumerate(to_rebuild, 1):
            print(f"  {i}. {item['commit'][:8]} | v{item.get('version', '?')} | {item.get('error', 'N/A')[:40]}")
        return

    print(f"\n{'='*60}")
    print(f"REBUILDING {len(to_rebuild)} IMAGES WITH FIXES")
    print(f"{'='*60}")

    results = []
    success = 0
    failed = 0

    for i, item in enumerate(to_rebuild, 1):
        commit = item['commit']
        version = item.get('version', 'unknown')

        print(f"\n[{i}/{len(to_rebuild)}] Rebuilding {commit[:8]} (v{version})")
        print(f"  Original error: {item.get('error', 'N/A')[:50]}")

        result = build_fixed_image(
            original_image=item['image'],
            commit=commit,
            version=version,
            gpu_id=args.gpu,
            push=not args.no_push
        )

        results.append(result)

        if result['status'] == 'success':
            success += 1
            print(f"  ✓ SUCCESS: {DOCKER_REPO}:{result['new_tag']}")
        else:
            failed += 1
            print(f"  ✗ FAILED: {result['error']}")

        # Save intermediate results (include offset in filename for parallel runs)
        results_file = f"rebuild_results_offset{args.offset}.json" if args.offset > 0 else "rebuild_results.json"
        with open(RESULTS_DIR / results_file, 'w') as f:
            json.dump({
                'timestamp': datetime.now().isoformat(),
                'total': len(to_rebuild),
                'completed': i,
                'success': success,
                'failed': failed,
                'results': results
            }, f, indent=2)

    # Final summary
    print(f"\n{'='*60}")
    print("REBUILD SUMMARY")
    print(f"{'='*60}")
    print(f"Total: {len(to_rebuild)}")
    print(f"Success: {success}")
    print(f"Failed: {failed}")

    if success > 0:
        print(f"\nSuccessfully rebuilt images:")
        for r in results:
            if r['status'] == 'success':
                print(f"  - {DOCKER_REPO}:{r['new_tag']}")


if __name__ == "__main__":
    main()
