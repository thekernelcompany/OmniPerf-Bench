#!/bin/bash
#
# Rerun unsuccessful TRAE Sonnet 4.5 commits on vLLM
#
# This script runs the TRAE agent with Claude Sonnet 4.5 on 91 previously
# unsuccessful vLLM commits. The TRAE tool_results bug has been fixed, so
# many of these should now succeed.
#

set -e

cd "$(dirname "$0")/perf-agents-bench"

echo "========================================="
echo "TRAE Sonnet 4.5 vLLM Rerun"
echo "========================================="
echo "Total commits to rerun: 91"
echo "Expected duration: ~15-20 hours (at ~10-15 min/commit)"
echo ""
echo "The run will resume automatically if interrupted."
echo "Results will be saved to: state/runs/vllm_sonnet45_rerun/"
echo ""
echo "Press Ctrl+C to cancel, or wait 5 seconds to start..."
sleep 5

.venv/bin/python -m bench.cli prepare tasks/vllm.yaml \
  --from-plan state/plan_trae_sonnet45_vllm_rerun.json \
  --bench-cfg bench.yaml \
  --max-workers 1 \
  --resume

echo ""
echo "========================================="
echo "Rerun completed!"
echo "========================================="
echo ""
echo "To view results:"
echo "  LATEST=\$(ls -t state/runs | head -n1)"
echo "  .venv/bin/python -m bench.cli report state/runs/\$LATEST"
