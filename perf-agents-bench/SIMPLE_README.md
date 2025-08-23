# Simple Performance Benchmark

The absolute simplest way to compare performance across commits.

## What it does

1. Takes one or more human commits (and pre-commits via parent index or explicit SHA)
2. Plans the work items, prepares agent branches with OpenHands (host-only)
3. Summarizes results from journals

## Quick Start (no Docker)

```bash
# 1. Scaffold example files
PYTHONPATH=perf-agents-bench python3 -m bench.cli init --out perf-agents-bench

# 2. Add commits to perf-agents-bench/.work/commits.txt
# Format: <40hex-human> [<40hex-pre>|parent=1]

# 3. Plan
PYTHONPATH=perf-agents-bench python3 -m bench.cli plan \
  perf-agents-bench/tasks/example.yaml \
  --commits perf-agents-bench/.work/commits.txt \
  --out state/plan.json

# 4. Prepare (runs OpenHands locally; resumable)
# If OpenHands is not installed yet, you can smoke-test with a no-op:
# export OPENHANDS_CLI=/bin/true
PYTHONPATH=perf-agents-bench python3 -m bench.cli prepare \
  perf-agents-bench/tasks/example.yaml \
  --from-plan state/plan.json \
  --max-workers 2 --resume

# 5. Report (prints JSON summary)
PYTHONPATH=perf-agents-bench python3 -m bench.cli report state/runs/<run_id>
```

## Minimal requirement in your project

Nothing special for Stage A. Your repo just needs to be a git repository that OpenHands can modify under the specified `target_files`.

## Later (optional, Docker)

You can build and run containerized evaluations:
```bash
# Build canonical images for baseline/human (and agent optionally)
bench build perf-agents-bench/tasks/example.yaml

# Smoke test inside containers
PYTHONPATH=perf-agents-bench python3 -m bench.cli smoke \
  perf-agents-bench/tasks/example.yaml \
  --bench-cfg perf-agents-bench/bench.yaml \
  --human-only \
  --cmd "python -c 'print(\"OK\")'"
```

## What happens

```
Planning commit pairs -> state/plan.json
Preparing items with OpenHands -> state/runs/<run_id>/<item_id>/{prompt,journal,diff_targets}.json
Summarizing journals -> JSON report
```