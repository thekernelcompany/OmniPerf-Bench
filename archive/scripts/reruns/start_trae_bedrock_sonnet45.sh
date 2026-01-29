#!/bin/bash
set -e

TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="/home/ubuntu/OmniPerf-Bench/trae_bedrock_sonnet45_${TIMESTAMP}.log"
SESSION_NAME="trae_bedrock_sonnet45"

echo "=========================================="
echo "Starting TRAE Agent Pipeline"
echo "Model: AWS Bedrock Sonnet 4.5"
echo "Commits: 99 vLLM commits"
echo "=========================================="
echo ""

# Check if tmux session already exists
if tmux has-session -t $SESSION_NAME 2>/dev/null; then
    echo "❌ Session '$SESSION_NAME' already exists!"
    echo "To attach: tmux attach -t $SESSION_NAME"
    echo "To kill: tmux kill-session -t $SESSION_NAME"
    exit 1
fi

# Create tmux session and run pipeline
echo "Creating tmux session: $SESSION_NAME"
echo "Log file: $LOG_FILE"
echo ""

tmux new-session -d -s $SESSION_NAME "cd /home/ubuntu/OmniPerf-Bench && /home/ubuntu/OmniPerf-Bench/run_trae_bedrock_sonnet45.sh 2>&1 | tee $LOG_FILE"

echo "✓ Pipeline started in background!"
echo ""
echo "Monitor progress:"
echo "  tmux attach -t $SESSION_NAME    # Attach to session (Ctrl+B then D to detach)"
echo "  tail -f $LOG_FILE                # Watch logs"
echo "  cd perf-agents-bench && ls -lrt state/runs/  # Check run directories"
echo ""
echo "Check status:"
echo "  grep -c 'Task status determined as: success' $LOG_FILE"
echo "  grep -c 'Task status determined as: error' $LOG_FILE"
echo ""
echo "Expected duration: ~25 hours (15 min/commit average)"
echo ""
