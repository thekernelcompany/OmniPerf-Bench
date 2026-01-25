#!/bin/bash
echo "=== 3-WAY BENCHMARK ==="
echo "Human: 187b85b7"
echo "Base: ceba0ce4"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"

HUMAN_COMMIT="187b85b7f38496653948a2aba546d53c09ada0f3"
BASE_COMMIT="ceba0ce4f661722198f6568a54ba20cf06b7e033"

# Clone and setup
cd /tmp
rm -rf sglang-overlay
git clone --single-branch --branch main https://github.com/sgl-project/sglang.git sglang-overlay 2>&1 | tail -2
cd sglang-overlay

# Fetch commits
git fetch origin $BASE_COMMIT --depth=1 2>/dev/null
git fetch origin $HUMAN_COMMIT --depth=1 2>/dev/null

# Verify commits
git cat-file -t $BASE_COMMIT 2>/dev/null || { echo "ERROR: Base commit not found"; exit 1; }
git cat-file -t $HUMAN_COMMIT 2>/dev/null || { echo "ERROR: Human commit not found"; exit 1; }

export PYTHONPATH=/tmp/sglang-overlay/python:$PYTHONPATH

# Initialize results file
echo "[" > /tmp/results.json

run_bench() {
    local name=$1

    # Capture benchmark output
    local output=$(python3 -m sglang.bench_latency \
        --model-path TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
        --batch-size 4 \
        --input-len 1024 \
        --output-len 256 \
        --trust-remote-code 2>&1)

    # Extract metrics - handle both "token/s" formats
    local total_throughput=$(echo "$output" | grep "^Total." | tail -1 | awk '{print $(NF-1)}')
    local decode_throughput=$(echo "$output" | grep "Decode.*avg throughput" | tail -1 | awk '{print $(NF-1)}')
    local prefill_throughput=$(echo "$output" | grep "^Prefill." | tail -1 | awk '{print $(NF-1)}')

    # Output JSON record
    echo "{"
    echo "  \"variant\": \"$name\","
    echo "  \"commit\": \"$(git log --oneline -1 | cut -d\  -f1)\","
    echo "  \"total_throughput_tokens_per_sec\": ${total_throughput:-0},"
    echo "  \"decode_throughput_tokens_per_sec\": ${decode_throughput:-0},"
    echo "  \"prefill_throughput_tokens_per_sec\": ${prefill_throughput:-0},"
    echo "  \"batch_size\": 4,"
    echo "  \"input_len\": 1024,"
    echo "  \"output_len\": 256"
    echo "}"
}

# BASELINE benchmark
echo ""
echo "=== BASELINE: base commit ==="
git checkout $BASE_COMMIT 2>/dev/null
echo "At: $(git log --oneline -1)"
run_bench "baseline" >> /tmp/results.json
echo "," >> /tmp/results.json

git checkout . 2>/dev/null

# HUMAN benchmark
echo ""
echo "=== HUMAN: human commit ==="
git checkout $HUMAN_COMMIT 2>/dev/null
echo "At: $(git log --oneline -1)"
run_bench "human" >> /tmp/results.json

# CLAUDE_CODE benchmark
echo ""
echo "=== CLAUDE_CODE: base + claude_code patch ==="
git checkout $BASE_COMMIT 2>/dev/null
git checkout . 2>/dev/null
if [ -f "/workspace/perf-agents-bench/state/runs/sglang/claude_code/default/2025-12-23_06-28-44/sglang_005_187b85b7/model_patch.diff" ]; then
    git apply "/workspace/perf-agents-bench/state/runs/sglang/claude_code/default/2025-12-23_06-28-44/sglang_005_187b85b7/model_patch.diff" 2>&1
    echo "At: $(git log --oneline -1) + claude_code patch"
    run_bench "claude_code" >> /tmp/results.json
    echo "," >> /tmp/results.json
else
    echo "Patch not found: /workspace/perf-agents-bench/state/runs/sglang/claude_code/default/2025-12-23_06-28-44/sglang_005_187b85b7/model_patch.diff"
fi

# CODEX benchmark
echo ""
echo "=== CODEX: base + codex patch ==="
git checkout $BASE_COMMIT 2>/dev/null
git checkout . 2>/dev/null
if [ -f "/workspace/perf-agents-bench/state/runs/sglang/codex/gpt-5/389be848/sglang_core-0006/model_patch.diff" ]; then
    git apply "/workspace/perf-agents-bench/state/runs/sglang/codex/gpt-5/389be848/sglang_core-0006/model_patch.diff" 2>&1
    echo "At: $(git log --oneline -1) + codex patch"
    run_bench "codex" >> /tmp/results.json
    echo "," >> /tmp/results.json
else
    echo "Patch not found: /workspace/perf-agents-bench/state/runs/sglang/codex/gpt-5/389be848/sglang_core-0006/model_patch.diff"
fi

# TRAE_GPT5 benchmark
echo ""
echo "=== TRAE_GPT5: base + trae_gpt5 patch ==="
git checkout $BASE_COMMIT 2>/dev/null
git checkout . 2>/dev/null
if [ -f "/workspace/perf-agents-bench/state/runs/sglang/trae/gpt-5/2025-11-14_21-05-32/sglang_005_187b85b7/model_patch.diff" ]; then
    git apply "/workspace/perf-agents-bench/state/runs/sglang/trae/gpt-5/2025-11-14_21-05-32/sglang_005_187b85b7/model_patch.diff" 2>&1
    echo "At: $(git log --oneline -1) + trae_gpt5 patch"
    run_bench "trae_gpt5" >> /tmp/results.json
    echo "," >> /tmp/results.json
else
    echo "Patch not found: /workspace/perf-agents-bench/state/runs/sglang/trae/gpt-5/2025-11-14_21-05-32/sglang_005_187b85b7/model_patch.diff"
fi

# TRAE_SONNET45 benchmark
echo ""
echo "=== TRAE_SONNET45: base + trae_sonnet45 patch ==="
git checkout $BASE_COMMIT 2>/dev/null
git checkout . 2>/dev/null
if [ -f "/workspace/perf-agents-bench/state/runs/sglang/trae/claude-sonnet-45/2025-11-28_15-26-15/sglang_005_187b85b7/model_patch.diff" ]; then
    git apply "/workspace/perf-agents-bench/state/runs/sglang/trae/claude-sonnet-45/2025-11-28_15-26-15/sglang_005_187b85b7/model_patch.diff" 2>&1
    echo "At: $(git log --oneline -1) + trae_sonnet45 patch"
    run_bench "trae_sonnet45" >> /tmp/results.json
    echo "," >> /tmp/results.json
else
    echo "Patch not found: /workspace/perf-agents-bench/state/runs/sglang/trae/claude-sonnet-45/2025-11-28_15-26-15/sglang_005_187b85b7/model_patch.diff"
fi


# Close JSON array (remove trailing comma and close)
sed -i '$ s/,$//' /tmp/results.json
echo "]" >> /tmp/results.json

# Output results
echo ""
echo "=== RESULTS JSON ==="
cat /tmp/results.json

echo ""
echo "=== BENCHMARK COMPLETE ==="
