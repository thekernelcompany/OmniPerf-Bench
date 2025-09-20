#!/usr/bin/env bash
set -euo pipefail

# Wrapper to run the agent → overlay build → Modal GPU tests pipeline
# Always uses the perf-agents-bench venv's Python so dependencies (typer, etc.) are present

ROOT_DIR=$(cd "$(dirname "$0")/.." && pwd)
BENCH_DIR="$ROOT_DIR/perf-agents-bench"
PY="$BENCH_DIR/.venv/bin/python"

if [ ! -x "$PY" ]; then
  echo "Bench venv not found at $PY"
  echo "Run setup first: bash OmniPerf-Bench/tools/setup_opb.sh"
  exit 1
fi

if [ $# -lt 1 ]; then
  echo "Usage: bash OmniPerf-Bench/tools/run_agent_pipeline.sh <PERF_COMMIT_SHA>"
  exit 1
fi

SHA="$1"

"$PY" "$ROOT_DIR/tools/agent_to_modal.py" "$SHA"


