#!/bin/bash
# Push rebuilt SGLang images to Docker Hub
# NOTE: Only pushing images that are verified working

set -e

echo "Pushing rebuilt SGLang images to Docker Hub..."
echo ""

# Image 1: 2a754e57 (v0.1.17) - READY
echo "Pushing 2a754e57 (v0.1.17 - no sgl-kernel needed)..."
docker push shikhar481/sglang-images:2a754e57
docker push shikhar481/sglang-images:2a754e57b052e249ed4f8572cb6f0069ba6a495e

# Image 2: ab4a83b2 (v0.3.0) - READY
echo "Pushing ab4a83b2 (v0.3.0 - pyairports fixed)..."
docker push shikhar481/sglang-images:ab4a83b2
docker push shikhar481/sglang-images:ab4a83b25909aa98330b838a224e4fe5c943e483

echo ""
echo "Done! 2 images pushed."
echo ""
echo "SKIPPED: 79961afa (v0.4.6.post2)"
echo "  Reason: sgl-kernel needs to be built on GPU machine"
echo "  See REBUILD_FROM_SOURCE.md for instructions"
