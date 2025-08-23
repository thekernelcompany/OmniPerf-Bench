# perf-agents-bench (uv-only)

Minimal, local-first workflow to plan commit pairs, run OpenHands headless to generate agent branches, and summarize results.

## Requirements
- uv (for OpenHands): `curl -LsSf https://astral.sh/uv/install.sh | sh`
- Docker (Stage B later)

## One-time setup
```bash
# Put your model creds in perf-agents-bench/.env
cat > perf-agents-bench/.env <<'ENV'
LLM_MODEL=anthropic/claude-3-7-sonnet-20250219
LLM_API_KEY=sk-...
# Optional for provider-backed repo ops
# GITHUB_TOKEN=ghp_...
ENV
```

## Run on vLLM (Stage A only)
```bash
# 1) Plan (edit tasks/vllm.yaml if you want different targets/constraints)
echo "f092153fbe349a9a1742940e3703bfcff6aa0a6d parent=1" > perf-agents-bench/.work/vllm_commits.txt
PYTHONPATH=perf-agents-bench python3 -m bench.cli plan \
  perf-agents-bench/tasks/vllm.yaml \
  --commits perf-agents-bench/.work/vllm_commits.txt \
  --out state/plan.json

# 2) Prepare (headless; loads .env automatically)
PYTHONPATH=perf-agents-bench python3 -m bench.cli prepare \
  perf-agents-bench/tasks/vllm.yaml \
  --from-plan state/plan.json \
  --bench-cfg perf-agents-bench/bench.yaml \
  --max-workers 1 --resume

# 3) Report
LATEST=$(ls -t perf-agents-bench/state/runs | head -n1)
PYTHONPATH=perf-agents-bench python3 -m bench.cli report perf-agents-bench/state/runs/$LATEST
```

## How it works
- Plan writes `state/plan.json` with (human, pre) pairs
- Prepare uses uvx to run OpenHands headless: `python -m openhands.core.main -d <worktree> -f task.txt`
- Journals/logs in `perf-agents-bench/state/runs/<run_id>/<item_id>/`

## Headless notes
- Docs: headless and CLI references: https://docs.all-hands.dev/usage/how-to/headless-mode, https://docs.all-hands.dev/usage/how-to/cli-mode
- Iterations configurable in `perf-agents-bench/bench.yaml` under `agents.openhands.args.iterations`.