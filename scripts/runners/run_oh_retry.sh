#!/usr/bin/env bash
# Retry failed fanout commits with udocker F1 (fakechroot/LD_PRELOAD) mode.
# Uses 4-way parallelism (less disk contention than 8-way).
#
# Usage: scripts/runners/run_oh_retry.sh

set -uo pipefail
cd "$(dirname "$0")/../.."

LOG_DIR="logs/oh_retry2/vllm"
mkdir -p "$LOG_DIR"
RES_DIR="archive/results/2026-05/iso_bench_results_3way_openhands_sonnet45/results"

# Find all errored commits EXCEPT NO_IMAGE (separate path)
mapfile -t COMMITS < <(python3 - <<EOF
import json, glob
out = []
for f in sorted(glob.glob("$RES_DIR/*_agent_result.json")):
    r = json.load(open(f))
    if r.get('status') != 'success':
        e = r.get('error', '')
        if 'No image' in e: continue   # unbuildable, separate native build
        out.append(f.split('/')[-1].split('_')[0])
for c in out:
    print(c)
EOF
)

NCOMMITS="${#COMMITS[@]}"
NGPU=8
echo "Retrying $NCOMMITS commits across $NGPU GPUs (F1 mode, less disk contention)"
echo "Commits: ${COMMITS[*]}"
echo

# Round-robin to 8 workers
declare -a CHUNK_0 CHUNK_1 CHUNK_2 CHUNK_3 CHUNK_4 CHUNK_5 CHUNK_6 CHUNK_7
for i in "${!COMMITS[@]}"; do
    g=$((i % NGPU))
    eval "CHUNK_${g}+=(${COMMITS[$i]})"
done

PIDS=()
for g in $(seq 0 $((NGPU-1))); do
    eval "chunk=(\"\${CHUNK_${g}[@]}\")"
    if [ "${#chunk[@]}" -eq 0 ]; then continue; fi
    echo "GPU $g  (${#chunk[@]} commits): ${chunk[*]}"
    # Delete old failed result files so runner re-runs
    for c in "${chunk[@]}"; do rm -f "$RES_DIR/${c}_agent_result.json"; done
    CUDA_VISIBLE_DEVICES=$g \
    nohup python3 -u scripts/runners/run_3way_benchmarks.py \
        --agent-type openhands_sonnet45 \
        --agent-only \
        --commits "${chunk[@]}" \
        --timeout 1800 \
        > "$LOG_DIR/worker_$g.log" 2>&1 &
    PIDS+=($!)
done

echo
echo "Workers: ${PIDS[*]}"
printf '%s\n' "${PIDS[@]}" > "$LOG_DIR/worker_pids.txt"
