#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/ISO-Bench"

echo "=== Running Codex vLLM bench harness ==="
echo "Working directory: $(pwd)"
echo "Timestamp: $(date)"

# Verify plan exists
PLAN_FILE="state/plan_codex_full.json"
if [ ! -f "$PLAN_FILE" ]; then
    echo "ERROR: Plan file not found: $PLAN_FILE"
    echo "Please run setup_codex_vllm_runs.sh first"
    exit 1
fi

# Activate virtual environment
if [ ! -f "../bench-env/bin/activate" ]; then
    echo "ERROR: bench-env not found at ../bench-env"
    exit 1
fi

source ../bench-env/bin/activate

# Set environment variables
export CODEX_CLI="${CODEX_CLI:-codex}"
export CODEX_PROFILE="${CODEX_PROFILE:-kernel-bot}"
export HOME="$SCRIPT_DIR/ISO-Bench/.codex_home"

echo "Codex CLI: $CODEX_CLI"
echo "Codex Profile: $CODEX_PROFILE"
echo "Codex Home: $HOME"

# Verify Codex CLI is accessible
if ! command -v "$CODEX_CLI" &> /dev/null; then
    echo "ERROR: Codex CLI not found: $CODEX_CLI"
    exit 1
fi

# Check bench configuration
BENCH_CFG="bench_codex.yaml"
if [ ! -f "$BENCH_CFG" ]; then
    echo "ERROR: Bench config not found: $BENCH_CFG"
    exit 1
fi

# Check task file
TASK_FILE="tasks/vllm.yaml"
if [ ! -f "$TASK_FILE" ]; then
    echo "ERROR: Task file not found: $TASK_FILE"
    exit 1
fi

echo ""
echo "Starting bench harness..."
echo "This will process all commits from the plan file."
echo "Logs will be written to: state/runs/<run_id>/"
echo ""
echo "Press Ctrl+C to stop (will save progress)"
echo ""

# Run the bench harness
python -m bench.cli prepare \
    "$TASK_FILE" \
    --from-plan "$PLAN_FILE" \
    --bench-cfg "$BENCH_CFG" \
    --max-workers 1 \
    --resume

echo ""
echo "=== Bench harness completed ==="
echo "Check results in: state/runs/"



