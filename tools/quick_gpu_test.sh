#!/bin/bash
# Quick GPU test for rebuilt SGLang images
# Run this on H100 after pulling images

set -e

echo "=== Quick GPU Test for Rebuilt SGLang Images ==="
echo "Date: $(date)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'NOT DETECTED')"
echo ""

test_image() {
    local tag=$1
    local version=$2
    echo "--- Testing $tag ($version) ---"

    docker run --rm --gpus all shikhar481/sglang-images:$tag python3 -c "
import torch
assert torch.cuda.is_available(), 'CUDA not available'
print(f'  GPU: {torch.cuda.get_device_name(0)}')
import sglang
print('  sglang: OK')
import outlines
print('  outlines: OK')
import uvloop
print('  uvloop: OK')
print('  STATUS: PASS')
" 2>&1 || echo "  STATUS: FAIL"
    echo ""
}

echo "Pulling images..."
docker pull shikhar481/sglang-images:2a754e57 >/dev/null 2>&1 || true
docker pull shikhar481/sglang-images:ab4a83b2 >/dev/null 2>&1 || true

test_image "2a754e57" "v0.1.17"
test_image "ab4a83b2" "v0.3.0"

echo "=== Done ==="
