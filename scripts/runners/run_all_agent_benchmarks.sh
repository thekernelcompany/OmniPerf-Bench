#!/bin/bash
# Run all agent benchmarks sequentially
# Usage: ./run_all_agent_benchmarks.sh

set -e
cd /root/OmniPerf-Bench

echo "=== Starting Agent Benchmark Suite ==="
echo "Start time: $(date)"
echo ""

# Codex GPT-5
echo "=========================================="
echo "[1/3] Running Codex GPT-5 benchmarks (93 commits)"
echo "=========================================="
python scripts/runners/run_3way_benchmarks.py --agent-type codex_gpt5 --agent-only --timeout 900
echo "Codex GPT-5 completed at $(date)"
echo ""

# TRAE GPT-5
echo "=========================================="
echo "[2/3] Running TRAE GPT-5 benchmarks (93 commits)"
echo "=========================================="
python scripts/runners/run_3way_benchmarks.py --agent-type trae_gpt5 --agent-only --timeout 900
echo "TRAE GPT-5 completed at $(date)"
echo ""

# TRAE Sonnet 4.5
echo "=========================================="
echo "[3/3] Running TRAE Sonnet 4.5 benchmarks (93 commits)"
echo "=========================================="
python scripts/runners/run_3way_benchmarks.py --agent-type trae_sonnet45 --agent-only --timeout 900
echo "TRAE Sonnet 4.5 completed at $(date)"
echo ""

echo "=== All Agent Benchmarks Completed ==="
echo "End time: $(date)"

# Count results
echo ""
echo "=== Results Summary ==="
for agent in codex trae_gpt5 trae_sonnet45; do
    dir="/root/OmniPerf-Bench/omniperf_results_3way_${agent}/results"
    if [ -d "$dir" ]; then
        total=$(ls "$dir"/*_agent_result.json 2>/dev/null | wc -l)
        success=$(grep -l '"status": "success"' "$dir"/*_agent_result.json 2>/dev/null | wc -l)
        echo "$agent: $success/$total successful"
    fi
done
