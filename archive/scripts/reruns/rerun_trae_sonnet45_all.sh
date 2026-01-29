#!/bin/bash
#
# Rerun ALL unsuccessful TRAE Sonnet 4.5 commits (vLLM + SGLang)
#
# This script runs both vLLM and SGLang reruns sequentially.
# Total: 142 commits (91 vLLM + 51 SGLang)
#

set -e

SCRIPT_DIR="$(dirname "$0")"

echo "========================================="
echo "TRAE Sonnet 4.5 Complete Rerun"
echo "========================================="
echo "Total commits to rerun: 142"
echo "  - vLLM: 91 commits"
echo "  - SGLang: 51 commits"
echo ""
echo "Expected total duration: ~24-30 hours"
echo ""
echo "Press Ctrl+C to cancel, or wait 5 seconds to start..."
sleep 5

# Run vLLM rerun
echo ""
echo "========================================="
echo "Starting vLLM rerun..."
echo "========================================="
"$SCRIPT_DIR/rerun_trae_sonnet45_vllm.sh"

# Run SGLang rerun
echo ""
echo "========================================="
echo "Starting SGLang rerun..."
echo "========================================="
"$SCRIPT_DIR/rerun_trae_sonnet45_sglang.sh"

echo ""
echo "========================================="
echo "All reruns completed!"
echo "========================================="
