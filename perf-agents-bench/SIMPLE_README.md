# Simple Performance Benchmark

The absolute simplest way to compare performance between two git commits.

## What it does

1. Takes two git commit hashes (baseline vs optimized)
2. Runs your `benchmark.py` on both commits  
3. Compares the results

## Quick Start

```bash
# 1. Install 
cd perf-agents-bench
pip install -e .

# 2. Use literal YAML (no env exports)
cat > /path/to/perf-agents-bench/tasks/simple.yaml <<'YAML'
id: simple
name: Simple benchmark
repo:
  url: "https://github.com/your/repo.git"
  human_commit: "<40hex>"
runner:
  platform: "linux/amd64"   # recommended on Apple Silicon
env_build:
  allowed_strategies: [dockerfile, requirements]
testpack:
  entrypoint: "../vlm-bench-generic"
metrics: []
scoring:
  primary: "functional_match"
  tie_breaker: "functional_match"
YAML

# 3. Smoke run (build & run containers only)
PYTHONPATH=/path/to/perf-agents-bench python3 -m bench.cli smoke \
  /path/to/perf-agents-bench/tasks/simple.yaml \
  --bench-cfg /path/to/perf-agents-bench/bench.yaml \
  --human-only \
  --cmd "python -c 'print(\"OK\")'"
```

## What you need in your project

Just a `benchmark.py` that:
- Measures your performance 
- Outputs JSON with `--output filename.json`

Example:
```python
#!/usr/bin/env python3
import json, sys

# Your performance test here
result = {"throughput": 123.45}

if "--output" in sys.argv:
    with open(sys.argv[sys.argv.index("--output") + 1], 'w') as f:
        json.dump(result, f)
```

## What happens

```
Running benchmark: Simple performance benchmark
Baseline commit: def789ab
Human commit: abc12345

=== Running baseline (def789ab) ===
✓ baseline completed

=== Running human (abc12345) ===  
✓ human completed

=== Running agent optimization ===
(Agent optimization skipped - would run OpenHands here)
✓ agent completed

=== Results ===
Baseline: 85.2
Human:    127.8  
Agent:    85.2

Report saved: ./results/simple_benchmark/report.json
```

That's it! No containers, no complex config, just git + your benchmark script.

## Demo

Run the included demo from any git repo:
```bash
python /path/to/perf-agents-bench/demo.py
```