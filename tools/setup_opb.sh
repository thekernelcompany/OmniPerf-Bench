#!/usr/bin/env bash
set -euo pipefail

# Simple setup for OmniPerf-Bench automation
# - Checks Docker, uv, modal
# - Provisions perf-agents-bench venv (Python 3.12) and installs deps
# - Creates Modal volumes and uploads generator scripts

ROOT_DIR=$(cd "$(dirname "$0")/.." && pwd)
BENCH_DIR="$ROOT_DIR/perf-agents-bench"
VENVPY="$BENCH_DIR/.venv/bin/python"

echo "[1/6] Checking dependencies..."
command -v docker >/dev/null 2>&1 || { echo "Docker not found"; exit 1; }
command -v uv >/dev/null 2>&1 || { echo "uv not found (pipx install uv)"; exit 1; }
command -v modal >/dev/null 2>&1 || { echo "modal not found (pip install modal-client)"; exit 1; }

echo "[2/6] Ensuring Docker daemon is running..."
docker info >/dev/null 2>&1 || { echo "Docker daemon not reachable"; exit 1; }

echo "[3/6] Provisioning bench venv (3.12) and dependencies..."
mkdir -p "$BENCH_DIR"
cd "$BENCH_DIR"
if [ ! -x "$VENVPY" ]; then
  uv venv --python 3.12 .venv
  uv pip install -r requirements.txt -p "$VENVPY"
fi
# Ensure top-level requirements are available in the same venv (for Typer, etc.)
uv pip install -r "$ROOT_DIR/requirements.txt" -p "$VENVPY"

echo "[4/6] Ensuring perf-agents-bench .env is present..."
ENV_FILE="$BENCH_DIR/.env"
ENV_EXAMPLE="$BENCH_DIR/.env.example"
if [ ! -f "$ENV_FILE" ]; then
  if [ -f "$ENV_EXAMPLE" ]; then
    cp "$ENV_EXAMPLE" "$ENV_FILE"
  else
    cat > "$ENV_FILE" <<EOF
# perf-agents-bench environment
LLM_MODEL=${LLM_MODEL:-gpt-4o-mini}
LLM_API_KEY=
# Optional: LLM_BASE_URL=https://api.openai.com/v1
EOF
  fi
fi

# If user exported OPENAI_API_KEY or LLM_API_KEY, inject into .env if empty
if ! grep -q '^LLM_API_KEY=' "$ENV_FILE"; then
  echo "LLM_API_KEY=" >> "$ENV_FILE"
fi
CURRENT_KEY=$(grep '^LLM_API_KEY=' "$ENV_FILE" | head -n1 | cut -d'=' -f2-)
if [ -z "$CURRENT_KEY" ]; then
  if [ -n "${LLM_API_KEY:-}" ]; then
    sed -i '' "s/^LLM_API_KEY=.*/LLM_API_KEY=${LLM_API_KEY}/" "$ENV_FILE"
  elif [ -n "${OPENAI_API_KEY:-}" ]; then
    sed -i '' "s/^LLM_API_KEY=.*/LLM_API_KEY=${OPENAI_API_KEY}/" "$ENV_FILE"
  fi
fi

echo "[5/7] Creating Modal volumes (idempotent)..."
set +e
modal volume create opb-generators >/dev/null 2>&1
modal volume create opb-results >/dev/null 2>&1
set -e

echo "[6/7] Uploading generator scripts to Modal volume..."
# Default to sibling repo directory: ../test-generation-scripts/working_test_generators
PARENT_ROOT=$(cd "$ROOT_DIR/.." && pwd)
GEN_DIR_DEFAULT="$PARENT_ROOT/test-generation-scripts/working_test_generators"
GEN_DIR="${1:-$GEN_DIR_DEFAULT}"
if [ ! -d "$GEN_DIR" ]; then
  # Fallback to generated_test_generators_v4 if working_test_generators is absent
  ALT_DIR="$PARENT_ROOT/test-generation-scripts/generated_test_generators_v4"
  if [ -d "$ALT_DIR" ]; then
    echo "Working generators not found, falling back to: $ALT_DIR"
    GEN_DIR="$ALT_DIR"
  else
    echo "Generator directory not found: $GEN_DIR"
    echo "Try: bash OmniPerf-Bench/tools/setup_opb.sh /Users/fortuna/Desktop/IO-bench/test-generation-scripts/working_test_generators"
    exit 1
  fi
fi
set +e
modal volume put opb-generators "$GEN_DIR"
RC=$?
set -e
if [ $RC -ne 0 ]; then
  echo "Generators already present in Modal volume 'opb-generators'; skipping upload."
fi

echo "[7/7] Setup complete. Next steps:"
cat <<EOF

Quickstart:
  1) Authenticate Modal (once): modal token new
  2) Run the pipeline (uses bench venv):
       $BENCH_DIR/.venv/bin/python $ROOT_DIR/OmniPerf-Bench/tools/agent_to_modal.py <PERF_COMMIT_SHA>

Make sure your perf-agents-bench/.env contains LLM_MODEL and LLM_API_KEY.
If you export OPENAI_API_KEY before running setup, it will be injected automatically.

Optional config (env vars):
  OPB_BASE_IMAGE_REPO            Docker repo for base/overlay (default: ayushnangia16/nvidia-vllm-docker)
  OPB_MODAL_GENERATORS_VOLUME    Modal volume for generators (default: opb-generators)
  OPB_MODAL_RESULTS_VOLUME       Modal volume for results (default: opb-results)

EOF


