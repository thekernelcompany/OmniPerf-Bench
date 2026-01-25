#!/bin/bash
set -e

RESULTS_DIR="/root/OmniPerf-Bench/omniperf_results_3way_sglang/overlay_benchmark_results"
LOG_FILE="$RESULTS_DIR/benchmark_run_$(date +%Y%m%d_%H%M%S).log"

echo "Starting benchmark run at $(date)" | tee "$LOG_FILE"
echo "Results will be saved to: $RESULTS_DIR" | tee -a "$LOG_FILE"

# All 17 commits
COMMITS=(
    "187b85b7"
    "6b231325"
    "6cb00c63"
    "148254d4"
    "2bd18e2d"
    "2a754e57"
    "880221bd"
    "b1e5a33a"
    "c087ddd6"
    "da47621c"
    "dd1012fc"
    "ddcf9fe3"
    "df7f61ee"
    "e3ec6bf4"
    "4418f599"
    "2a413829"
    "5e023301"
)

AGENTS="claude_code,codex,trae_gpt5,trae_sonnet45"
TOTAL=${#COMMITS[@]}
SUCCESS=0
FAILED=0

for i in "${!COMMITS[@]}"; do
    commit="${COMMITS[$i]}"
    num=$((i + 1))
    echo "" | tee -a "$LOG_FILE"
    echo "========================================" | tee -a "$LOG_FILE"
    echo "[$num/$TOTAL] Processing commit: $commit" | tee -a "$LOG_FILE"
    echo "Started at: $(date)" | tee -a "$LOG_FILE"
    echo "========================================" | tee -a "$LOG_FILE"
    
    if python3 run_3way_overlay.py \
        --commit "$commit" \
        --agents "$AGENTS" \
        --batch-size 4 \
        --input-len 1024 \
        --output-len 256 2>&1 | tee -a "$LOG_FILE"; then
        echo "[$num/$TOTAL] SUCCESS: $commit" | tee -a "$LOG_FILE"
        SUCCESS=$((SUCCESS + 1))
    else
        echo "[$num/$TOTAL] FAILED: $commit" | tee -a "$LOG_FILE"
        FAILED=$((FAILED + 1))
    fi
done

echo "" | tee -a "$LOG_FILE"
echo "========================================" | tee -a "$LOG_FILE"
echo "BENCHMARK RUN COMPLETE" | tee -a "$LOG_FILE"
echo "Total: $TOTAL, Success: $SUCCESS, Failed: $FAILED" | tee -a "$LOG_FILE"
echo "Finished at: $(date)" | tee -a "$LOG_FILE"
echo "========================================" | tee -a "$LOG_FILE"
