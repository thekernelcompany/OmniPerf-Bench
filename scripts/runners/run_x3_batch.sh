#!/usr/bin/env bash
# X3 throughput re-bench driver (2026-05-05).
# Strict scope: 5 unique commits, 6 OH cells + 5 baselines = 11 GPU runs.
# 4c822298 is re-extracted from existing data, no GPU.
set -uo pipefail
cd "$(dirname "$0")/../.."

ROOT="$(pwd)"
MAPPING="$ROOT/data/mappings/vllm_oh_mapping_x3.json"
RESULTS_BASE="$ROOT/archive/results/2026-05-x3"
LOG_DIR="$ROOT/archive/results/2026-05-x3/logs"
mkdir -p "$RESULTS_BASE/baseline" \
         "$RESULTS_BASE/openhands_gpt5" \
         "$RESULTS_BASE/openhands_sonnet45" \
         "$LOG_DIR"

run_cell() {
  local kind="$1"  # baseline | agent_gpt5 | agent_sonnet45
  local commit="$2"
  local rdir="$3"
  local extra_flags="$4"
  local log_file="$LOG_DIR/${kind}__${commit}.log"
  echo "=== [$(date +%H:%M:%S)] $kind / $commit ==="
  python3 scripts/runners/run_vllm_native.py \
      --commits "$commit" \
      --mapping-file "$MAPPING" \
      --results-dir "$rdir" \
      --no-iso-override \
      --agent-name openhands_gpt5 \
      $extra_flags 2>&1 | tee "$log_file"
  local out="$rdir/${commit}_agent_result.json"
  if [[ -f "$out" ]]; then
    python3 -c "
import json
d = json.load(open('$out'))
m = d.get('metrics', {})
print(f'  -> status={d.get(\"status\")} req/s={m.get(\"throughput_req_s\")} total_tok/s={m.get(\"throughput_total_tok_s\")} output_tok/s={m.get(\"throughput_output_tok_s\")}')
"
  fi
}

case "${1:-canary}" in
  canary)
    # Smallest model, fastest run. Baseline only (no patch). ~2 min wall.
    echo "### CANARY: 98f47f2a baseline (opt-125m) ###"
    run_cell "baseline" "98f47f2a" "$RESULTS_BASE/baseline" "--skip-patch"
    ;;
  baselines)
    # 5 baseline runs, one per unique commit.
    for c in 98f47f2a 6dd94dbe 3b61cb45 8c1e77fb fa63e710; do
      run_cell "baseline" "$c" "$RESULTS_BASE/baseline" "--skip-patch"
    done
    ;;
  baseline_4c822298)
    run_cell "baseline" "4c822298" "$RESULTS_BASE/baseline" "--skip-patch"
    ;;
  agents)
    # GPT-5 cells: 3b61cb45, 8c1e77fb, 98f47f2a, fa63e710
    for c in 98f47f2a 3b61cb45 8c1e77fb fa63e710; do
      run_cell "agent_gpt5" "$c" "$RESULTS_BASE/openhands_gpt5" \
               "--patches-dir $ROOT/ISO-Bench/state/runs/vllm/openhands_gpt5/flat"
    done
    # Sonnet cells: fa63e710, 6dd94dbe
    for c in 6dd94dbe fa63e710; do
      run_cell "agent_sonnet45" "$c" "$RESULTS_BASE/openhands_sonnet45" \
               "--patches-dir $ROOT/ISO-Bench/state/runs/vllm/openhands_sonnet45/flat"
    done
    ;;
  agents_4c822298)
    run_cell "agent_gpt5"     "4c822298" "$RESULTS_BASE/openhands_gpt5" \
             "--patches-dir $ROOT/ISO-Bench/state/runs/vllm/openhands_gpt5/flat"
    run_cell "agent_sonnet45" "4c822298" "$RESULTS_BASE/openhands_sonnet45" \
             "--patches-dir $ROOT/ISO-Bench/state/runs/vllm/openhands_sonnet45/flat"
    ;;
  all)
    "$0" baselines
    "$0" agents
    ;;
  *)
    echo "usage: $0 [canary|baselines|agents|all]"
    exit 1
    ;;
esac

echo "=== [$(date +%H:%M:%S)] done ==="
