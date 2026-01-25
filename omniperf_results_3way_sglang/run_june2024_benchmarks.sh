#!/bin/bash
set -e

RESULTS_DIR="/root/OmniPerf-Bench/omniperf_results_3way_sglang/overlay_benchmark_results"
LOG_FILE="$RESULTS_DIR/june2024_benchmark_$(date +%Y%m%d_%H%M%S).log"

echo "Starting June 2024 benchmark run at $(date)" | tee "$LOG_FILE"

# June 2024 era commits (PR < 1000)
COMMITS=(
    "09deb20d"
    "1bf1cf19"
    "2a754e57"
    "564a898a"
    "6a2941f4"
    "6f560c76"
    "9216b106"
    "ac971ff6"
    "bb3a3b66"
    "e822e590"
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
    echo "[$num/$TOTAL] Processing: $commit" | tee -a "$LOG_FILE"
    echo "Started: $(date)" | tee -a "$LOG_FILE"
    echo "========================================" | tee -a "$LOG_FILE"
    
    if python3 run_3way_overlay.py \
        --commit "$commit" \
        --agents "$AGENTS" \
        --batch-size 4 \
        --input-len 1024 \
        --output-len 256 2>&1 | tee -a "$LOG_FILE"; then
        
        # Check if we got actual results
        result_file="$RESULTS_DIR/${commit}_3way.json"
        if [ -f "$result_file" ]; then
            baseline=$(python3 -c "import json; data=json.load(open('$result_file')); print([b.get('total_throughput_tokens_per_sec',0) for b in data.get('benchmarks',[]) if b.get('variant')=='baseline'][0] if data.get('benchmarks') else 0)" 2>/dev/null || echo "0")
            if [ "$baseline" != "0" ] && [ "$baseline" != "0.0" ]; then
                echo "[$num/$TOTAL] SUCCESS: $commit (baseline: $baseline tokens/s)" | tee -a "$LOG_FILE"
                SUCCESS=$((SUCCESS + 1))
            else
                echo "[$num/$TOTAL] FAILED: $commit (benchmark returned 0)" | tee -a "$LOG_FILE"
                FAILED=$((FAILED + 1))
            fi
        else
            echo "[$num/$TOTAL] FAILED: $commit (no result file)" | tee -a "$LOG_FILE"
            FAILED=$((FAILED + 1))
        fi
    else
        echo "[$num/$TOTAL] FAILED: $commit (script error)" | tee -a "$LOG_FILE"
        FAILED=$((FAILED + 1))
    fi
done

echo "" | tee -a "$LOG_FILE"
echo "========================================" | tee -a "$LOG_FILE"
echo "BENCHMARK RUN COMPLETE" | tee -a "$LOG_FILE"
echo "Total: $TOTAL, Success: $SUCCESS, Failed: $FAILED" | tee -a "$LOG_FILE"
echo "Finished: $(date)" | tee -a "$LOG_FILE"
echo "========================================" | tee -a "$LOG_FILE"
