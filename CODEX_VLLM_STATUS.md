# Codex vLLM Bench Run Status

## Setup Complete ✓

- **Plan created**: `perf-agents-bench/state/plan_codex_full.json` with 99 commits
- **Codex CLI**: Configured with `kernel-bot` profile
- **Tmux session**: `codex_vllm` running in background
- **Log file**: `perf-agents-bench/codex_vllm_run_*.log`

## Current Status

The bench harness is running Codex CLI (`codex exec`) on all 99 vLLM commits in the plan.

### Monitor Progress

```bash
# Attach to tmux session
tmux attach -t codex_vllm

# View latest log
cd perf-agents-bench
tail -f codex_vllm_run_*.log

# Check progress
grep -c "Task status determined as: success" codex_vllm_run_*.log
grep -c "Task status determined as: error" codex_vllm_run_*.log
```

### Configuration

- **Agent**: Codex CLI with `kernel-bot` profile
- **Max workers**: 1 (sequential processing)
- **Time budget**: 120 minutes per commit
- **Max steps**: 120 per commit
- **Resume**: Enabled (skips completed commits)

### Expected Duration

- **Per commit**: ~15 minutes average
- **Total (99 commits)**: ~25 hours
- **Output location**: `perf-agents-bench/state/runs/<run_id>/<item_id>/`

### Important Notes

- The session runs in tmux, so it will continue even if you disconnect
- Use `tmux attach -t codex_vllm` to reconnect
- Detach with `Ctrl+B` then `D`
- Logs are written to timestamped files in `perf-agents-bench/`



