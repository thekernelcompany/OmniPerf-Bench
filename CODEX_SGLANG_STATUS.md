# Codex SGLang Bench Run Status

## Setup Complete ✓

- **Plan created**: `perf-agents-bench/state/plan_sglang_codex_full.json` with 80 commits
- **Codex CLI**: Configured with `kernel-bot` profile
- **Tmux session**: `codex_sglang` running in background
- **Log file**: `perf-agents-bench/codex_sglang_run_*.log`

## Current Status

The bench harness is running Codex CLI (`codex exec`) on all 80 SGLang commits in the plan.

### Monitor Progress

```bash
# Attach to tmux session
tmux attach -t codex_sglang

# View latest log
cd perf-agents-bench
tail -f codex_sglang_run_*.log

# Check progress
grep -c "Task status determined as: success" codex_sglang_run_*.log
grep -c "Task status determined as: error" codex_sglang_run_*.log
```

### Configuration

- **Agent**: Codex CLI with `kernel-bot` profile
- **Max workers**: 1 (sequential processing)
- **Time budget**: 120 minutes per commit
- **Max steps**: 120 per commit
- **Resume**: Enabled (skips completed commits)
- **Task**: `tasks/sglang.yaml`
- **Repository**: `/home/ubuntu/OmniPerf-Bench/sglang`

### Expected Duration

- **Per commit**: ~3 minutes average (based on vLLM results)
- **Total (80 commits)**: ~4 hours
- **Output location**: `perf-agents-bench/state/runs/<run_id>/<item_id>/`

### Important Notes

- The session runs in tmux, so it will continue even if you disconnect
- Use `tmux attach -t codex_sglang` to reconnect
- Detach with `Ctrl+B` then `D`
- Logs are written to timestamped files in `perf-agents-bench/`
- All 80 SGLang commits from `misc/experiments/sglang_commit_extractions_with_apis/` are included

