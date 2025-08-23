# perf-agents-bench (uv + venv)

Minimal, local-first workflow to plan commit pairs, run OpenHands headless to generate agent branches, and summarize results.

## Requirements
- Python 3.12+
- uv (for OpenHands): `curl -LsSf https://astral.sh/uv/install.sh | sh`
- Docker (Stage B later)

## One-time setup
```bash
# Copy and fill env
cp perf-agents-bench/.env.example perf-agents-bench/.env
# Edit perf-agents-bench/.env with your creds (LLM_MODEL, LLM_API_KEY, optional GITHUB_TOKEN)

# Create local venv and install CLI deps (once)
cd perf-agents-bench
# Create venv with uv and install deps into it
uv venv .venv
uv pip install -r requirements.txt -p .venv/bin/python
cd -
```

## Run on vLLM (Stage A only)
```bash
# 1) Plan (edit tasks/vllm.yaml if you want different targets/constraints)
echo "f092153fbe349a9a1742940e3703bfcff6aa0a6d parent=1" > perf-agents-bench/.work/vllm_commits.txt
cd perf-agents-bench
.venv/bin/python -m bench.cli plan \
  tasks/vllm.yaml \
  --commits .work/vllm_commits.txt \
  --out ./state/plan.json

# 2) Prepare (headless; loads .env automatically)
.venv/bin/python -m bench.cli prepare \
  tasks/vllm.yaml \
  --from-plan ./state/plan.json \
  --bench-cfg bench.yaml \
  --max-workers 1 --resume

# 3) Report
LATEST=$(ls -t state/runs | head -n1)
.venv/bin/python -m bench.cli report state/runs/$LATEST
cd -
```

## How it works
- Plan writes `state/plan.json` with (human, pre) pairs
- Prepare uses uvx to run OpenHands headless: `python -m openhands.core.main -d <worktree> -f task.txt`
- Journals/logs in `perf-agents-bench/state/runs/<run_id>/<item_id>/`

## Headless notes
- Docs: headless and CLI references: https://docs.all-hands.dev/usage/how-to/headless-mode, https://docs.all-hands.dev/usage/how-to/cli-mode
- Iterations configurable in `perf-agents-bench/bench.yaml` under `agents.openhands.args.iterations`.

## Troubleshooting
- Doctor: `cd perf-agents-bench && .venv/bin/python -m bench.cli doctor --bench-cfg bench.yaml`
- Logs: `perf-agents-bench/state/runs/<run_id>/<item_id>/openhands_stderr.txt`