#!/bin/bash
set -x

cd /root/OmniPerf-Bench

# GPT-5 commits
GPT5_COMMITS="58eee5f2 a3223766"

# Sonnet 4.5 commits
SONNET_COMMITS="d7740ea4 e206b543 e7b20426 fa63e710 fc7b8d1e fe66b347"

run_benchmark() {
    local agent_type=$1
    local commit=$2

    echo ""
    echo "========================================"
    echo "$(date): Running $agent_type commit: $commit"
    echo "========================================"

    # Clean Docker first
    docker system prune -af > /dev/null 2>&1

    # Run with 30 minute timeout
    python3 -u scripts/runners/run_3way_benchmarks.py \
        --agent-type $agent_type \
        --agent-only \
        --commits $commit \
        --timeout 1800

    # Show result
    result_file="omniperf_results_3way_${agent_type}/results/${commit}_agent_result.json"
    if [ -f "$result_file" ]; then
        status=$(python3 -c "import json; print(json.load(open('$result_file')).get('status'))")
        echo "Result: $status"
    fi
}

echo "=== Starting sequential benchmarks ==="
echo "Start time: $(date)"

# Run GPT-5 commits
for commit in $GPT5_COMMITS; do
    run_benchmark "trae_gpt5_0123" "$commit"
done

# Run Sonnet 4.5 commits
for commit in $SONNET_COMMITS; do
    run_benchmark "trae_sonnet45_0123" "$commit"
done

echo ""
echo "=== All benchmarks complete ==="
echo "End time: $(date)"

# Final summary
echo ""
echo "=== Final Status ==="
echo "GPT-5 results:"
for f in omniperf_results_3way_trae_gpt5_0123/results/*_agent_result.json; do
    status=$(python3 -c "import json; print(json.load(open('$f')).get('status'))" 2>/dev/null)
    echo "  $(basename $f .json | sed 's/_agent_result//'): $status"
done

echo ""
echo "Sonnet 4.5 results:"
for f in omniperf_results_3way_trae_sonnet45_0123/results/*_agent_result.json; do
    status=$(python3 -c "import json; print(json.load(open('$f')).get('status'))" 2>/dev/null)
    echo "  $(basename $f .json | sed 's/_agent_result//'): $status"
done
