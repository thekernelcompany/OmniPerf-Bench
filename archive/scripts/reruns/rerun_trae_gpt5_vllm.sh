#!/bin/bash
set -e

echo "=========================================="
echo "TRAE GPT-5 vLLM Rerun"
echo "=========================================="
echo "Date: $(date)"
echo "Commits: 53"
echo ""

cd /home/ubuntu/OmniPerf-Bench/perf-agents-bench

# Verify prerequisites
echo "Checking prerequisites..."
if [ -z "$OPENAI_API_KEY" ]; then
    echo "ERROR: OPENAI_API_KEY not set"
    echo "Please set it with: export OPENAI_API_KEY='your_key'"
    exit 1
fi

if [ ! -f "state/plan_trae_gpt5_vllm_rerun.json" ]; then
    echo "ERROR: Plan file not found: state/plan_trae_gpt5_vllm_rerun.json"
    echo "Run: python3 ../scripts/reruns/create_gpt5_rerun_plans.py"
    exit 1
fi

echo "✓ OPENAI_API_KEY is set"
echo "✓ Plan file exists"
echo ""

# Clean worktrees
echo "Cleaning stale worktrees..."
rm -rf .work/worktrees/*
git -C ../vllm worktree prune

echo ""
echo "Starting vLLM rerun..."
echo "Expected duration: ~6-8 hours (53 commits @ 7-10 min each)"
echo ""
echo "Monitor progress with:"
echo "  tail -f pipeline_run_*.log | grep -E '(Starting task|status determined|token usage)'"
echo ""

../bench-env/bin/python -m bench.cli prepare \
    tasks/vllm.yaml \
    --from-plan state/plan_trae_gpt5_vllm_rerun.json \
    --bench-cfg bench.yaml \
    --max-workers 1 \
    --resume

echo ""
echo "=========================================="
echo "vLLM Rerun Complete!"
echo "=========================================="
echo "Completed at: $(date)"
