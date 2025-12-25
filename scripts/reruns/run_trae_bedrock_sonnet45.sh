#!/bin/bash
set -e

echo "=========================================="
echo "TRAE Agent Pipeline - Bedrock Sonnet 4.5"
echo "=========================================="
echo ""

# Navigate to perf-agents-bench directory
cd /home/ubuntu/OmniPerf-Bench/perf-agents-bench

# Activate bench environment
source /home/ubuntu/OmniPerf-Bench/bench-env/bin/activate

# Set environment variables
export TRAE_PYTHON=/home/ubuntu/OmniPerf-Bench/bench-env/bin/python
export TRAE_CONFIG=/home/ubuntu/OmniPerf-Bench/third-party/trae-agent/trae_config.yaml
export AWS_REGION=us-east-1

# Verify configuration
echo "Configuration:"
echo "  Python: $TRAE_PYTHON"
echo "  TRAE Config: $TRAE_CONFIG"
echo "  AWS Region: $AWS_REGION"
echo "  Model: Bedrock Sonnet 4.5 (anthropic.claude-sonnet-4-5-20250929-v1:0)"
echo ""

# Check AWS credentials
echo "Checking AWS credentials..."
aws sts get-caller-identity --query 'Account' --output text
echo ""

# Run the pipeline
echo "Starting TRAE agent pipeline..."
echo "Plan: state/plan_bedrock_sonnet45.json (99 commits)"
echo "Output: state/runs/<run_id>/vllm_bedrock_sonnet45-<item_id>/"
echo ""

$TRAE_PYTHON -m bench.cli prepare \
    tasks/vllm.yaml \
    --from-plan state/plan_bedrock_sonnet45.json \
    --bench-cfg bench.yaml \
    --max-workers 1 \
    --resume

echo ""
echo "Pipeline completed!"
echo "Check logs in: state/runs/<latest_run_id>/"
