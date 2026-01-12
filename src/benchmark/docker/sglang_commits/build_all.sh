#!/bin/bash
# Build all 6 SGLang Docker images (3 human commits + 3 parent commits)
# Usage: ./build_all.sh [--push] [--commit COMMIT]

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DOCKER_REPO="shikhar481/sglang-images"

# Commit mappings: human -> parent, dockerfile
declare -A COMMITS=(
    # Human commit d1112d85 and its parent
    ["d1112d8548eb13c842900b3a8d622345f9737759"]="Dockerfile.d1112d85"
    ["48efec7b052354865aa2f0605a5bf778721f3cbb"]="Dockerfile.d1112d85"
    # Human commit 93470a14 and its parent
    ["93470a14116a60fe5dd43f0599206e8ccabdc211"]="Dockerfile.93470a14"
    ["db452760e5b2378efd06b1ceb9385d2eeb6d217c"]="Dockerfile.93470a14"
    # Human commit 9c088829 and its parent
    ["9c088829ee2a28263f36d0814fde448c6090b5bc"]="Dockerfile.9c088829"
    ["005aad32ad45ce27d73fd39aa1f7e9ba5d8ebb8f"]="Dockerfile.9c088829"
)

# Parse arguments
PUSH=false
SINGLE_COMMIT=""
while [[ $# -gt 0 ]]; do
    case $1 in
        --push)
            PUSH=true
            shift
            ;;
        --commit)
            SINGLE_COMMIT="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            echo "Usage: $0 [--push] [--commit COMMIT_HASH]"
            exit 1
            ;;
    esac
done

build_image() {
    local commit=$1
    local dockerfile=$2
    local short_commit=${commit:0:8}

    echo ""
    echo "============================================================"
    echo "Building image for commit: $short_commit"
    echo "Using Dockerfile: $dockerfile"
    echo "============================================================"

    local image_tag="${DOCKER_REPO}:${commit}"

    # Build the image
    docker build \
        -f "${SCRIPT_DIR}/${dockerfile}" \
        --build-arg COMMIT_HASH="${commit}" \
        -t "${image_tag}" \
        "${SCRIPT_DIR}"

    echo "Built: ${image_tag}"

    # Push if requested
    if [ "$PUSH" = true ]; then
        echo "Pushing to DockerHub..."
        docker push "${image_tag}"
        echo "Pushed: ${image_tag}"
    fi
}

# Build images
if [ -n "$SINGLE_COMMIT" ]; then
    # Build single commit
    if [ -z "${COMMITS[$SINGLE_COMMIT]}" ]; then
        echo "Error: Unknown commit $SINGLE_COMMIT"
        echo "Known commits:"
        for c in "${!COMMITS[@]}"; do
            echo "  - $c"
        done
        exit 1
    fi
    build_image "$SINGLE_COMMIT" "${COMMITS[$SINGLE_COMMIT]}"
else
    # Build all commits
    echo "Building all 6 images..."
    for commit in "${!COMMITS[@]}"; do
        build_image "$commit" "${COMMITS[$commit]}"
    done
fi

echo ""
echo "============================================================"
echo "BUILD COMPLETE"
echo "============================================================"
if [ "$PUSH" = true ]; then
    echo "Images have been pushed to DockerHub"
else
    echo "Images built locally. Use --push to push to DockerHub"
fi
