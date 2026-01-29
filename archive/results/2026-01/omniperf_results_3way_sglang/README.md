# SGLang 3-Way Benchmark Results

This directory contains benchmark results for SGLang performance optimization commits.

## Structure

```
omniperf_results_3way_sglang/
├── README.md
├── sglang/
│   └── {commit_hash}/
│       └── benchmark_result.json
└── hero_run_logs/
    └── hero_sglang_{timestamp}.log
```

## Result Schema

Each `benchmark_result.json` contains:

```json
{
  "status": "success|error",
  "gpu_config": "H100:1|H100:2|H100:4|H100:8",
  "benchmark_mode": "human_only|3way",
  "human_metrics": {
    "request_throughput": 7.9,
    "output_throughput": 1679.58,
    "ttft_mean": 524.09,
    "tpot_mean": 24.18,
    "itl_mean": 9.61,
    "e2e_latency_mean": 2556.87
  },
  "baseline_metrics": {},
  "agent_metrics": null,
  "commit": "132dad87",
  "full_commit": "132dad874d2e44592d03a112e4b7d63b153e8346",
  "parent_commit": "60fdad7cf343333e956a3889c12956396a1516bf",
  "model": "meta-llama/Llama-3.1-8B-Instruct",
  "subject": "Commit message subject",
  "perf_command": "python -m sglang.bench_serving ...",
  "duration_s": 495.9,
  "has_agent_patch": true
}
```

## Notes

- `baseline_metrics` and `agent_metrics` are empty due to SGLang's multi-package ABI requirements
- See `docs/SGLANG_BENCHMARK_SETUP.md` for full documentation
- Results are also published to HuggingFace: `Inferencebench/claude-code-sglang-benchmarks`
