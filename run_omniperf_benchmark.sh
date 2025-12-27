#!/bin/bash
#
# OmniPerf Benchmark Runner for Claude Code Patches
#
# This script runs benchmarks from the Ayushnangia/omniperf_v1 HuggingFace dataset
# against Claude Code-generated patches.
#
# Usage:
#   ./run_omniperf_benchmark.sh [OPTIONS]
#
# Options:
#   --split SPLIT       Dataset split: vllm or sglang (default: vllm)
#   --limit N           Limit to first N instances
#   --commits HASH...   Only benchmark specific commits
#   --timeout SECS      Timeout per test (default: 600)
#   --verbose           Enable verbose logging
#   --help              Show this help message
#

set -e

# Default paths - adjust these for your environment
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VLLM_REPO="${VLLM_REPO:-$SCRIPT_DIR/../vllm}"
SGLANG_REPO="${SGLANG_REPO:-$SCRIPT_DIR/../sglang}"
STATE_ROOT="${STATE_ROOT:-$SCRIPT_DIR/perf-agents-bench/state}"
OUTPUT_DIR="${OUTPUT_DIR:-$SCRIPT_DIR/omniperf_results}"

# Default options
SPLIT="vllm"
LIMIT=""
COMMITS=""
TIMEOUT=600
VERBOSE=""

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --split)
            SPLIT="$2"
            shift 2
            ;;
        --limit)
            LIMIT="--limit $2"
            shift 2
            ;;
        --commits)
            shift
            COMMITS="--commits"
            while [[ $# -gt 0 && ! "$1" =~ ^-- ]]; do
                COMMITS="$COMMITS $1"
                shift
            done
            ;;
        --timeout)
            TIMEOUT="$2"
            shift 2
            ;;
        --verbose|-v)
            VERBOSE="-v"
            shift
            ;;
        --help|-h)
            head -30 "$0" | tail -25
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Verify repositories exist
if [[ ! -d "$VLLM_REPO/.git" ]]; then
    echo "Error: vLLM repository not found at $VLLM_REPO"
    echo "Set VLLM_REPO environment variable to the correct path"
    exit 1
fi

if [[ ! -d "$SGLANG_REPO/.git" ]]; then
    echo "Error: SGLang repository not found at $SGLANG_REPO"
    echo "Set SGLANG_REPO environment variable to the correct path"
    exit 1
fi

if [[ ! -d "$STATE_ROOT/runs" ]]; then
    echo "Error: State directory not found at $STATE_ROOT"
    echo "Set STATE_ROOT environment variable to the correct path"
    exit 1
fi

echo "=============================================="
echo "OmniPerf Benchmark Runner"
echo "=============================================="
echo "Configuration:"
echo "  vLLM repo:      $VLLM_REPO"
echo "  SGLang repo:    $SGLANG_REPO"
echo "  State root:     $STATE_ROOT"
echo "  Output dir:     $OUTPUT_DIR"
echo "  Split:          $SPLIT"
echo "  Timeout:        ${TIMEOUT}s"
echo "=============================================="
echo ""

# Ensure required packages are installed
if ! python3 -c "import datasets" 2>/dev/null; then
    echo "Installing required packages..."
    pip install datasets huggingface_hub --quiet
fi

# Run the benchmark
cd "$SCRIPT_DIR"
python3 -m src.eval.omniperf_benchmark_runner \
    --vllm-repo "$VLLM_REPO" \
    --sglang-repo "$SGLANG_REPO" \
    --state-root "$STATE_ROOT" \
    --output-dir "$OUTPUT_DIR" \
    --split "$SPLIT" \
    --timeout "$TIMEOUT" \
    $LIMIT \
    $COMMITS \
    $VERBOSE

echo ""
echo "Results saved to: $OUTPUT_DIR"
