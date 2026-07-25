# W1 add-on — Per-task rollout variance (pass-at-k, generated)

Source: `Inferencebench/iso-bench-pass-at-k-results` — independent agent rollouts per task, each benchmarked once (captures agent stochasticity + benchmark noise). CV = std/mean across rollouts of the same task.

## sglang × claude_code (10 tasks, 76 rollouts)

- **TTFT mean (ms)**: tasks with ≥3 rollouts: 10; median CV 9.2%, p90 CV 19.4%
- **Output throughput (tok/s)**: tasks with ≥3 rollouts: 10; median CV 0.7%, p90 CV 1.0%

## sglang × codex_cli (10 tasks, 74 rollouts)

- **TTFT mean (ms)**: tasks with ≥3 rollouts: 10; median CV 7.1%, p90 CV 11.3%
- **Output throughput (tok/s)**: tasks with ≥3 rollouts: 10; median CV 0.4%, p90 CV 0.7%

## vllm × claude_code (30 tasks, 239 rollouts)

- **TTFT mean (ms)**: tasks with ≥3 rollouts: 23; median CV 1.1%, p90 CV 7.3%
- **Output throughput (tok/s)**: tasks with ≥3 rollouts: 23; median CV 0.8%, p90 CV 6.5%
- **Latency avg (ms)**: tasks with ≥3 rollouts: 5; median CV 0.2%, p90 CV 2.6%

## vllm × codex_cli (29 tasks, 229 rollouts)

- **TTFT mean (ms)**: tasks with ≥3 rollouts: 22; median CV 0.9%, p90 CV 5.6%
- **Output throughput (tok/s)**: tasks with ≥3 rollouts: 22; median CV 0.7%, p90 CV 8.2%
- **Latency avg (ms)**: tasks with ≥3 rollouts: 5; median CV 0.2%, p90 CV 0.7%

Interpretation: rollout-to-rollout CV bounds how much of a single-rollout Beats/Similar/Worse classification (±5% threshold) could flip under resampling; report alongside the Wilson CIs.
