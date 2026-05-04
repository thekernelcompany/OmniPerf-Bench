#!/usr/bin/env bash
# OpenHands GPT-5 fanout via OpenRouter — exact mirror of the Sonnet-4.5 setup.
#
# Settings are deliberately MAXED (matches the sonnet45 fanout that landed on HF):
#   iterations          = 120
#   max_budget_per_task = $1000   (effectively no cap; OH stops on iteration cap)
#   max_input_tokens    = 200000
#   max_output_tokens   = 64000
#   time_budget_minutes = 120     (per task wall-clock)
#
# Pricing for openai/gpt-5 on OpenRouter: $1.25/M input, $10/M output.
# Sonnet-4.5 fanout consumed roughly $1100-1200 across all 54 tasks.
# Expected GPT-5 fanout total: ~$500-700 (≈50% cheaper per-token than Sonnet).
#
# Usage:
#   ./launch_openhands_gpt5_openrouter.sh smoke         # 1-task dry run (vllm_core-0095)
#   ./launch_openhands_gpt5_openrouter.sh vllm          # 39 vllm tasks
#   ./launch_openhands_gpt5_openrouter.sh sglang        # 15 sglang tasks
#
# Resume after partial completion:
#   ./launch_openhands_gpt5_openrouter.sh vllm --resume
#
set -euo pipefail

cd "$(dirname "$0")"

# === LLM env (OpenRouter) ===
# OPENROUTER key — passed via env, NEVER committed.
if [[ -z "${OPENROUTER_API_KEY:-}" ]]; then
    echo "ERROR: OPENROUTER_API_KEY not set in env." >&2
    echo "  export OPENROUTER_API_KEY='sk-or-v1-...'" >&2
    exit 2
fi

export LLM_MODEL="openai/gpt-5"
export LLM_API_KEY="${OPENROUTER_API_KEY}"
export LLM_BASE_URL="https://openrouter.ai/api/v1"

# OpenHands also reads OPENAI_API_KEY in some paths (custom_llm_provider="openai" path)
export OPENAI_API_KEY="${OPENROUTER_API_KEY}"
export OPENAI_BASE_URL="https://openrouter.ai/api/v1"

# Use the same TOML scaffolding the sonnet45 run used; LLM_MODEL/LLM_BASE_URL
# env vars override the TOML values in prepare.py.
export OPENHANDS_CONFIG_FILE="$(pwd)/config/main_openrouter_gpt5.toml"

CLI=".venv/bin/python -m bench.cli"
PY=".venv/bin/python"

# Sanity: the OH 0.62.0 env must exist
if [[ ! -x "../bench-env-oh062/bin/python" ]]; then
    echo "ERROR: ../bench-env-oh062/bin/python missing — install OpenHands 0.62.0 first." >&2
    exit 3
fi

mode="${1:-smoke}"
shift || true

case "$mode" in
    smoke)
        # Single-task dry run — pick vllm_core-0095, same task sonnet45 dry-runs used
        cat > tmp_openhands_gpt5_or_smoke_plan.json <<'JSON'
{
  "repo": "https://github.com/vllm-project/vllm.git",
  "task_id": "vllm_core",
  "items": [
    {
      "item_id": "vllm_core-0095-openhands-gpt5-or-smoke",
      "human": "fe66b34728e5d383e3d19aefc544eeee808c99fb",
      "pre": "",
      "pre_parent_index": 1
    }
  ]
}
JSON
        $PY -m bench.cli prepare tasks/vllm.yaml \
            --from-plan tmp_openhands_gpt5_or_smoke_plan.json \
            --bench-cfg tmp_openhands_gpt5_or_vllm_bench.yaml \
            --max-workers 1 --resume "$@"
        ;;
    vllm)
        # Full 39-task vllm fanout — uses canonical plan_iso.json
        # 4 parallel workers: ~1 GB per worker observed (heavier than estimated);
        # 6 workers exhausted RAM (8 GB → 2 GB available in 4 min). 4 is the safe ceiling.
        $PY -m bench.cli prepare tasks/vllm.yaml \
            --from-plan state/runs/oh54_hf_configs/plans/plan_iso.json \
            --bench-cfg tmp_openhands_gpt5_or_vllm_bench.yaml \
            --max-workers 4 --resume "$@"
        ;;
    sglang)
        # Full 15-task sglang fanout — 4 parallel workers (same memory budget as vllm)
        $PY -m bench.cli prepare tasks/sglang.yaml \
            --from-plan state/runs/oh54_hf_configs/plans/sglang_plan_iso.json \
            --bench-cfg tmp_openhands_gpt5_or_sglang_bench.yaml \
            --max-workers 4 --resume "$@"
        ;;
    *)
        echo "Unknown mode: $mode (use smoke|vllm|sglang)" >&2
        exit 1
        ;;
esac
