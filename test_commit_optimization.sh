#!/bin/bash
"""
Test script for the commit optimization pipeline.

This script demonstrates how to run the optimization pipeline on the 0ec82edd commit.
"""

set -e

# Configuration
COMMIT_JSON="tmp_single_commit/0ec82edda59aaf5cf3b07aadf4ecce1aa1131add.json"
TEST_SCRIPT="misc/experiments/generated_test_generators_v4/0ec82edd_test_case_generator.py"
REPO_PATH="/path/to/vllm"  # User needs to set this

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}Commit Optimization Pipeline Test${NC}"
echo "=================================="
echo

# Check if files exist
echo "Checking prerequisites..."

if [ ! -f "$COMMIT_JSON" ]; then
    echo -e "${RED}ERROR: Commit JSON not found: $COMMIT_JSON${NC}"
    exit 1
fi

if [ ! -f "$TEST_SCRIPT" ]; then
    echo -e "${RED}ERROR: Test script not found: $TEST_SCRIPT${NC}"
    exit 1
fi

if [ "$REPO_PATH" = "/path/to/vllm" ]; then
    echo -e "${YELLOW}WARNING: Please set REPO_PATH to your vLLM repository${NC}"
    echo "Example: export REPO_PATH=/home/user/vllm"
    exit 1
fi

if [ ! -d "$REPO_PATH" ]; then
    echo -e "${RED}ERROR: Repository not found: $REPO_PATH${NC}"
    exit 1
fi

# Check environment variables
if [ -z "$OPENAI_API_KEY" ] && [ -z "$ANTHROPIC_API_KEY" ]; then
    echo -e "${RED}ERROR: OpenHands requires OPENAI_API_KEY or ANTHROPIC_API_KEY${NC}"
    echo "Set one of these environment variables to run OpenHands"
    exit 1
fi

echo -e "${GREEN}✓ All prerequisites satisfied${NC}"
echo

# Check if OpenHands is installed
echo "Checking OpenHands installation..."
if python3 -c "import openhands" 2>/dev/null; then
    echo -e "${GREEN}✓ OpenHands is installed${NC}"
else
    echo -e "${YELLOW}WARNING: OpenHands not found, installing...${NC}"
    uv pip install openhands-ai
fi

echo

# Run the pipeline
echo "Running commit optimization pipeline..."
echo "This may take 10-30 minutes depending on optimization complexity"
echo

python3 run_commit_optimization.py \
    --commit-json "$COMMIT_JSON" \
    --test-script "$TEST_SCRIPT" \
    --repo-path "$REPO_PATH" \
    --work-dir ".commit_opt_work" \
    --cleanup

echo
echo -e "${GREEN}Pipeline test completed!${NC}"
