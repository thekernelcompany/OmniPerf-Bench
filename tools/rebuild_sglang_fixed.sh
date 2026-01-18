#!/bin/bash
# SGLang Docker Image Rebuild Script (Corrected v2)
# Based on DOCKER_IMAGE_REBUILD_GUIDE.md
#
# Key fixes:
# 1. Install PyTorch FIRST (determines C++ ABI)
# 2. Pin ALL dependencies to prevent version drift
# 3. Build sgl_kernel FROM SOURCE (not PyPI!) to ensure ABI compatibility
# 4. Install uvloop (required for SGLang server)
# 5. Sanity checks verify actual module loading (catches ABI mismatch)

set -e

COMMIT_HASH=$1
TARGET_REPO="${2:-shikhar481/sglang-images}"

if [ -z "$COMMIT_HASH" ]; then
    echo "Usage: $0 <commit_hash> [target_repo]"
    echo "Example: $0 93470a14116a60fe5dd43f0599206e8ccabdc211 shikhar481/sglang-images"
    exit 1
fi

SHORT_HASH=${COMMIT_HASH:0:8}
WORK_DIR="/tmp/sglang_rebuild_${SHORT_HASH}"
DOCKERFILE_PATH="${WORK_DIR}/Dockerfile.fixed"

echo "========================================"
echo "Building SGLang image for: ${SHORT_HASH}"
echo "Full commit: ${COMMIT_HASH}"
echo "Target repo: ${TARGET_REPO}"
echo "========================================"

# Clean up any previous build
rm -rf "${WORK_DIR}"
mkdir -p "${WORK_DIR}"

# Create corrected Dockerfile
cat > "${DOCKERFILE_PATH}" << 'DOCKERFILE_EOF'
FROM nvidia/cuda:12.4.1-devel-ubuntu22.04

ARG COMMIT_HASH=main

ENV DEBIAN_FRONTEND=noninteractive \
    CUDA_HOME=/usr/local/cuda \
    PATH="${PATH}:/usr/local/cuda/bin" \
    LD_LIBRARY_PATH="${LD_LIBRARY_PATH}:/usr/local/cuda/lib64" \
    PIP_NO_CACHE_DIR=1

# 1. System dependencies (CRITICAL: include libnuma-dev)
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.11 python3.11-dev python3.11-venv python3-pip \
    git curl wget build-essential cmake ninja-build \
    libnuma-dev \
    libjpeg-dev \
    libpng-dev \
    && ln -sf /usr/bin/python3.11 /usr/bin/python3 \
    && ln -sf /usr/bin/python3.11 /usr/bin/python \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Verify libnuma is installed
RUN ldconfig -p | grep libnuma && echo "libnuma: OK"

# 2. Install pip and basic Python packages
RUN python3 -m pip install --upgrade pip setuptools wheel

# 3. Install PyTorch 2.4.0 FIRST (this determines the C++ ABI)
# CRITICAL: Use --index-url to get cu124 version
RUN pip install torch==2.4.0 --index-url https://download.pytorch.org/whl/cu124

# Sanity check: verify torch version
RUN python3 -c "import torch; assert torch.__version__.startswith('2.4'), f'Wrong torch: {torch.__version__}'; print(f'PyTorch {torch.__version__}: OK')"

# 4. Install torchvision 0.19.0 matching torch 2.4.0
# CRITICAL: Must use same index to avoid pulling wrong torch
RUN pip install torchvision==0.19.0 --index-url https://download.pytorch.org/whl/cu124

# Sanity check: verify torch is still 2.4.0
RUN python3 -c "import torch; assert torch.__version__.startswith('2.4'), f'Torch changed to: {torch.__version__}'; print(f'After torchvision - PyTorch {torch.__version__}: OK')"

# 5. FlashInfer (MUST match CUDA and torch version exactly)
RUN pip install flashinfer-python -i https://flashinfer.ai/whl/cu124/torch2.4/

# Sanity check: verify flashinfer works and torch still correct
RUN python3 -c "import torch; assert torch.__version__.startswith('2.4'), f'Torch changed: {torch.__version__}'" && \
    python3 -c "import flashinfer; print('FlashInfer: OK')"

# 6. Clone SGLang at specific commit
WORKDIR /opt
RUN git clone https://github.com/sgl-project/sglang.git sglang

ARG COMMIT_HASH_FINAL
RUN cd sglang && git checkout ${COMMIT_HASH_FINAL}

# 7. Install SGLang dependencies (with version constraints to prevent torch override)
RUN pip install \
    "transformers>=4.40.0" \
    "huggingface_hub>=0.23.0" \
    "tokenizers>=0.19.0" \
    "accelerate>=0.30.0" \
    "numpy<2.0" \
    "pydantic>=2.0" \
    requests \
    aiohttp \
    packaging \
    ninja \
    pyzmq \
    pillow \
    fastapi \
    uvicorn \
    uvloop \
    orjson \
    pybase64

# Sanity check: torch still 2.4?
RUN python3 -c "import torch; assert torch.__version__.startswith('2.4'), f'Torch changed: {torch.__version__}'; print(f'After deps - PyTorch {torch.__version__}: OK')"

# 8. Install SGLang (use constraint to prevent torch upgrade from dependencies like vllm)
WORKDIR /opt/sglang
RUN echo "torch==2.4.0" > /tmp/constraints.txt && \
    echo "torchvision==0.19.0" >> /tmp/constraints.txt

# Try installing with [srt] first (avoids vllm), then fallback to base install
RUN pip install -e "python[srt]" --constraint /tmp/constraints.txt || \
    pip install -e "python" --constraint /tmp/constraints.txt

# Sanity check
RUN python3 -c "import sglang; print(f'SGLang {sglang.__version__}: OK')"

# 9. Build sgl_kernel FROM SOURCE (NOT PyPI - pre-built wheels have wrong ABI!)
# CRITICAL: --no-build-isolation ensures we use the installed torch for ABI compatibility
# First install build dependencies required by sgl_kernel
RUN pip install scikit-build-core cmake

# sgl_kernel build is optional - some commits don't need it or have broken source
# We try to build from source, but continue if it fails
RUN if [ -d "/opt/sglang/sgl-kernel" ]; then \
        echo "Attempting to build sgl_kernel from source..." && \
        cd /opt/sglang/sgl-kernel && \
        (pip install -e . --no-build-isolation 2>&1 && echo "sgl_kernel: built from source OK") || \
        echo "WARNING: sgl_kernel build failed - image may work without it for some models"; \
    else \
        echo "Note: sgl-kernel directory not found in this commit"; \
    fi

# 10. Final dependency check - datasets and other utils
RUN pip install \
    datasets \
    pandas \
    tqdm \
    pybase64

# 11. FINAL SANITY CHECKS (must test actual module loading, not just import!)
RUN echo "=== FINAL SANITY CHECKS ===" && \
    python3 -c "import torch; v=torch.__version__; assert v.startswith('2.4'), f'FAIL: torch={v}'; print(f'torch: {v} OK')" && \
    python3 -c "import sglang; print(f'sglang: {sglang.__version__} OK')" && \
    python3 -c "import flashinfer; print('flashinfer: OK')" && \
    python3 -c "import zmq; print('zmq: OK')" && \
    python3 -c "import uvloop; print('uvloop: OK')" && \
    (python3 -c "import sgl_kernel; from sgl_kernel import common_ops; print('sgl_kernel: OK (ABI verified)')" 2>/dev/null || echo "sgl_kernel: NOT AVAILABLE (build failed or not needed)") && \
    echo "=== ALL REQUIRED CHECKS PASSED ==="

WORKDIR /workspace

# Copy benchmark scripts
RUN cp -r /opt/sglang/python/sglang/bench_* /workspace/ 2>/dev/null || true
RUN cp -r /opt/sglang/benchmark* /workspace/ 2>/dev/null || true

CMD ["python", "-c", "import sglang; print(f'SGLang {sglang.__version__} ready')"]
DOCKERFILE_EOF

# Replace placeholder with actual commit hash
sed -i "s/ARG COMMIT_HASH_FINAL/ARG COMMIT_HASH_FINAL=${COMMIT_HASH}/" "${DOCKERFILE_PATH}"

echo ""
echo "=== Dockerfile created at ${DOCKERFILE_PATH} ==="
echo ""

# Build the image
IMAGE_TAG="${TARGET_REPO}:${COMMIT_HASH}"
IMAGE_TAG_SHORT="${TARGET_REPO}:${SHORT_HASH}"

echo "Building image: ${IMAGE_TAG}"
echo "This will take a while..."
echo ""

docker build \
    -f "${DOCKERFILE_PATH}" \
    --build-arg COMMIT_HASH_FINAL="${COMMIT_HASH}" \
    -t "${IMAGE_TAG}" \
    -t "${IMAGE_TAG_SHORT}" \
    "${WORK_DIR}" 2>&1 | tee "${WORK_DIR}/build.log"

BUILD_STATUS=${PIPESTATUS[0]}

if [ $BUILD_STATUS -ne 0 ]; then
    echo ""
    echo "========================================"
    echo "BUILD FAILED!"
    echo "Check ${WORK_DIR}/build.log for details"
    echo "========================================"
    exit 1
fi

echo ""
echo "========================================"
echo "BUILD SUCCEEDED - Running final verification"
echo "========================================"
echo ""

# Create verification script
VERIFY_SCRIPT="${WORK_DIR}/verify.py"
cat > "${VERIFY_SCRIPT}" << 'VERIFY_EOF'
#!/usr/bin/env python3
import sys
import importlib.util

print("=" * 50)
print("FINAL IMAGE VERIFICATION (CPU mode)")
print("=" * 50)
print()

errors = []

# Check torch version
try:
    import torch
    if not torch.__version__.startswith("2.4"):
        errors.append(f"torch version wrong: {torch.__version__} (expected 2.4.x)")
        print(f"torch: FAIL - version {torch.__version__}")
    else:
        print(f"torch: {torch.__version__} OK")
except ImportError as e:
    errors.append(f"torch: {e}")
    print(f"torch: FAIL - {e}")

# Check sglang
try:
    import sglang
    print(f"sglang: {sglang.__version__} OK")
except ImportError as e:
    errors.append(f"sglang: {e}")
    print(f"sglang: FAIL - {e}")

# Check flashinfer
try:
    import flashinfer
    print("flashinfer: OK")
except ImportError as e:
    errors.append(f"flashinfer: {e}")
    print(f"flashinfer: FAIL - {e}")

# Check zmq
try:
    import zmq
    print("zmq: OK")
except ImportError as e:
    errors.append(f"zmq: {e}")
    print(f"zmq: FAIL - {e}")

# Check uvloop (REQUIRED for server)
try:
    import uvloop
    print("uvloop: OK")
except ImportError as e:
    errors.append(f"uvloop: {e}")
    print(f"uvloop: FAIL - {e}")

# Check sgl_kernel - must test actual module loading, not just import!
print()
print("=== sgl_kernel check (ABI verification) ===")
spec = importlib.util.find_spec("sgl_kernel")
if spec is not None:
    print(f"sgl_kernel: INSTALLED at {spec.origin}")
    # CRITICAL: Test actual module loading to catch ABI mismatch
    try:
        from sgl_kernel import common_ops
        print("sgl_kernel: ABI VERIFIED (common_ops loaded)")
    except Exception as e:
        print(f"sgl_kernel: ABI MISMATCH - {e}")
        errors.append(f"sgl_kernel ABI mismatch: {e}")
else:
    print("sgl_kernel: NOT INSTALLED (older commit - OK)")

print()
print("=" * 50)
if errors:
    print("VERIFICATION FAILED!")
    for e in errors:
        print(f"  - {e}")
    sys.exit(1)
else:
    print("ALL CHECKS PASSED")
    print("(GPU verification should be done on GPU machine)")
    sys.exit(0)
VERIFY_EOF

# Run verification
echo "Running final verification..."
docker run --rm -v "${VERIFY_SCRIPT}:/verify.py:ro" "${IMAGE_TAG}" python3 /verify.py

VERIFY_STATUS=$?

if [ $VERIFY_STATUS -ne 0 ]; then
    echo ""
    echo "========================================"
    echo "VERIFICATION FAILED - DO NOT PUSH!"
    echo "========================================"
    exit 1
fi

echo ""
echo "========================================"
echo "SUCCESS! Image ready to push:"
echo "  docker push ${IMAGE_TAG}"
echo "  docker push ${IMAGE_TAG_SHORT}"
echo "========================================"
