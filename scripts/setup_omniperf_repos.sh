#!/bin/bash
#
# Setup script for OmniPerf Benchmark Runner
#
# This script clones/updates the vLLM and SGLang repositories
# required for running benchmarks.
#
# Usage:
#   ./setup_omniperf_repos.sh [--shallow]
#
# Options:
#   --shallow   Clone with shallow history (faster, less disk)
#

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"

# Repository URLs
VLLM_REPO_URL="https://github.com/vllm-project/vllm.git"
SGLANG_REPO_URL="https://github.com/sgl-project/sglang.git"

# Target directories
VLLM_DIR="$REPO_ROOT/repos/vllm"
SGLANG_DIR="$REPO_ROOT/repos/sglang"

# Parse arguments
SHALLOW=""
while [[ $# -gt 0 ]]; do
    case $1 in
        --shallow)
            SHALLOW="--depth 1000"  # Enough history for most commits
            shift
            ;;
        --help|-h)
            head -15 "$0" | tail -10
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

echo "=============================================="
echo "OmniPerf Repository Setup"
echo "=============================================="
echo "Repository root: $REPO_ROOT"
echo ""

# Create repos directory
mkdir -p "$REPO_ROOT/repos"

# Clone or update vLLM
echo "Setting up vLLM repository..."
if [[ -d "$VLLM_DIR/.git" ]]; then
    echo "  Updating existing clone..."
    cd "$VLLM_DIR"
    git fetch --all --prune
    git remote prune origin
else
    echo "  Cloning from $VLLM_REPO_URL..."
    git clone $SHALLOW "$VLLM_REPO_URL" "$VLLM_DIR"
fi
echo "  OK: vLLM ready at $VLLM_DIR"
echo ""

# Clone or update SGLang
echo "Setting up SGLang repository..."
if [[ -d "$SGLANG_DIR/.git" ]]; then
    echo "  Updating existing clone..."
    cd "$SGLANG_DIR"
    git fetch --all --prune
    git remote prune origin
else
    echo "  Cloning from $SGLANG_REPO_URL..."
    git clone $SHALLOW "$SGLANG_REPO_URL" "$SGLANG_DIR"
fi
echo "  OK: SGLang ready at $SGLANG_DIR"
echo ""

# Create symlinks for convenience
echo "Creating convenience symlinks..."
ln -sf "$VLLM_DIR" "$REPO_ROOT/vllm-repo" 2>/dev/null || true
ln -sf "$SGLANG_DIR" "$REPO_ROOT/sglang-repo" 2>/dev/null || true

echo ""
echo "=============================================="
echo "Setup Complete!"
echo "=============================================="
echo ""
echo "Repositories cloned to:"
echo "  vLLM:   $VLLM_DIR"
echo "  SGLang: $SGLANG_DIR"
echo ""
echo "To run benchmarks:"
echo "  export VLLM_REPO=$VLLM_DIR"
echo "  export SGLANG_REPO=$SGLANG_DIR"
echo "  ./run_omniperf_benchmark.sh --split vllm"
echo ""
