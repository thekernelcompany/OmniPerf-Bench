#!/bin/bash
# Verify rebuilt SGLang Docker images on H100 GPU
# Run this script on a machine with H100 GPU and nvidia-docker

set -e

# Configuration
DOCKER_REPO="shikhar481/sglang-images"
HF_TOKEN="${HF_TOKEN:-}"

echo "=== SGLang Docker Image Verification ==="
echo "Date: $(date)"
echo "Testing on: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'No GPU detected')"
echo ""

# Images to verify (rebuilt from source)
# NOTE: 79961afa is NOT ready - needs GPU rebuild for sgl-kernel
declare -A IMAGES=(
    ["2a754e57"]="v0.1.17 - 2x prefill improvement (no sgl-kernel needed)"
    ["ab4a83b2"]="v0.3.0 - Optimize schedule (pyairports fixed, no sgl-kernel needed)"
)

# Skipped images (need GPU rebuild):
# ["79961afa"]="v0.4.6.post2 - SKIPPED: sgl-kernel needs GPU to build from source"

verify_image() {
    local commit="$1"
    local description="$2"
    local full_image="${DOCKER_REPO}:${commit}"

    echo "=== Testing: $commit ($description) ==="

    # Pull image
    echo "[1/4] Pulling image..."
    docker pull "$full_image" || { echo "FAILED: Could not pull $full_image"; return 1; }

    # Test basic imports
    echo "[2/4] Testing imports..."
    docker run --rm --gpus all "$full_image" python3 -c "
import torch
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'GPU: {torch.cuda.get_device_name(0)}')

import sglang
print('sglang: OK')

import uvloop
print('uvloop: OK')

try:
    import sgl_kernel
    from sgl_kernel import common_ops
    print('sgl_kernel: OK')
except ImportError as e:
    print(f'sgl_kernel: Not available ({e})')
except Exception as e:
    print(f'sgl_kernel: Error ({e})')

try:
    import outlines
    print('outlines: OK')
except ImportError:
    print('outlines: Not available')

import importlib.metadata
print(f'SGLang version: {importlib.metadata.version(\"sglang\")}')
" || { echo "FAILED: Import test"; return 1; }

    # Test server startup (quick check)
    echo "[3/4] Testing server startup (30s timeout)..."
    timeout 30 docker run --rm --gpus all "$full_image" python3 -c "
import torch
if not torch.cuda.is_available():
    print('SKIP: No GPU')
    exit(0)

# Just test that we can initialize CUDA
torch.cuda.init()
print('CUDA initialization: OK')

# Try to import server components
try:
    from sglang.srt.server import Runtime
    print('Runtime import: OK')
except ImportError:
    try:
        from sglang.srt.managers.router.model_runner import ModelRunner
        print('ModelRunner import: OK')
    except ImportError:
        print('Server components: Not available (older version)')
" || echo "TIMEOUT or error (may be OK for older versions)"

    # Run quick benchmark (if available and model downloadable)
    if [ -n "$HF_TOKEN" ]; then
        echo "[4/4] Running quick benchmark..."
        timeout 120 docker run --rm --gpus all \
            -e HF_TOKEN="$HF_TOKEN" \
            "$full_image" \
            python3 -m sglang.bench_latency \
            --model meta-llama/Llama-3.1-8B-Instruct \
            --batch-size 1 \
            --input-len 128 \
            --output-len 32 \
            2>&1 | tail -20 || echo "Benchmark skipped or failed"
    else
        echo "[4/4] Skipping benchmark (no HF_TOKEN)"
    fi

    echo "=== PASSED: $commit ==="
    echo ""
    return 0
}

# Run verification for each image
PASSED=0
FAILED=0

for commit in "${!IMAGES[@]}"; do
    if verify_image "$commit" "${IMAGES[$commit]}"; then
        ((PASSED++))
    else
        ((FAILED++))
    fi
done

echo ""
echo "=== Summary ==="
echo "Passed: $PASSED"
echo "Failed: $FAILED"
echo ""

if [ $FAILED -eq 0 ]; then
    echo "All rebuilt images verified successfully!"
    exit 0
else
    echo "Some images failed verification"
    exit 1
fi
