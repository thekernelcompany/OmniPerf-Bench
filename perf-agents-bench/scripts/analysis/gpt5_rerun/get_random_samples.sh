#!/bin/bash
cd /home/ubuntu/OmniPerf-Bench/perf-agents-bench/state/runs/vllm/trae/gpt-5-2025-08-07/2025-12-26_15-06-42

# Get successful commits
find . -name "journal.json" -exec grep -l '"status": "success"' {} \; | \
  xargs -I {} dirname {} | \
  shuf | \
  head -5 | \
  xargs -I {} basename {}
