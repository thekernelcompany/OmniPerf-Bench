#!/usr/bin/env bash
# 8-way GPU fanout for openhands_sonnet45 hard-metrics benchmarks.
#
# Splits commits across 8 background workers (one GPU each), pinned via
# CUDA_VISIBLE_DEVICES. Each worker drains its chunk sequentially.
#
# Usage:
#   scripts/runners/run_oh_fanout.sh [vllm|sglang]
#
# Reads commits from the prepared flat dir's run_summary.json files.
# Per-worker logs go to logs/oh_fanout/<repo>/worker_$i.log.

set -uo pipefail
cd "$(dirname "$0")/../.."

REPO="${1:-vllm}"
case "$REPO" in
    vllm)   AGENT_TYPE="openhands_sonnet45"        ; FLAT="ISO-Bench/state/runs/vllm/openhands_sonnet45/flat" ;;
    sglang) AGENT_TYPE="openhands_sonnet45_sglang" ; FLAT="ISO-Bench/state/runs/sglang/openhands_sonnet45/flat" ;;
    *) echo "usage: $0 [vllm|sglang]"; exit 2 ;;
esac

LOG_DIR="logs/oh_fanout/$REPO"
mkdir -p "$LOG_DIR"

# Collect human commits (8-char) for tasks with non-empty patches.
mapfile -t COMMITS < <(python3 - <<EOF
import json
from pathlib import Path
flat = Path("$FLAT")
out = []
for td in sorted(flat.iterdir()):
    rs = td / "run_summary.json"
    patch = td / "model_patch.diff"
    if not rs.exists() or not patch.exists() or patch.stat().st_size == 0:
        continue
    j = json.loads(rs.read_text())
    out.append(j["commits"]["human"][:8])
for c in out:
    print(c)
EOF
)

NCOMMITS="${#COMMITS[@]}"
if [ "$NCOMMITS" -eq 0 ]; then
    echo "No commits to run."
    exit 1
fi

NGPU=8
echo "Fanning out $NCOMMITS $REPO commits across $NGPU GPUs (agent_type=$AGENT_TYPE)"
echo "Logs: $LOG_DIR/worker_{0..$((NGPU-1))}.log"
echo

# Round-robin assignment: commit i goes to GPU i % 8. Balanced even when
# NCOMMITS is not divisible by 8 (38 -> 5,5,5,5,5,5,4,4).
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
    CUDA_VISIBLE_DEVICES=$g \
    nohup python3 -u scripts/runners/run_3way_benchmarks.py \
        --agent-type "$AGENT_TYPE" \
        --agent-only \
        --commits "${chunk[@]}" \
        --timeout 1800 \
        > "$LOG_DIR/worker_$g.log" 2>&1 &
    PIDS+=($!)
done

echo
echo "Workers launched: ${PIDS[*]}"
echo "Tail any worker:  tail -F $LOG_DIR/worker_*.log"
echo "Wait for all:     wait ${PIDS[*]}"
echo "Kill all:         kill ${PIDS[*]}"

# Persist worker pid set so a follow-up script can wait on them.
printf '%s\n' "${PIDS[@]}" > "$LOG_DIR/worker_pids.txt"
