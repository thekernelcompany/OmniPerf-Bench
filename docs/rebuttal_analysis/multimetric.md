# yX4G — Multi-metric agent-vs-human (serving subset, generated)

Median % delta vs human patch (positive = agent better). Coverage varies per metric because each task runs the PR's own benchmark command.

## vLLM (legacy agents, HF benchmark table, 39-task set)

| Agent | TTFT mean | TPOT mean | ITL mean | Throughput | Latency avg |
|---|---|---|---|---|---|
| claude_code | -2.4 (n=23) | -1.9 (n=22) | -2.3 (n=22) | +0.2 (n=11) | -0.5 (n=6) |
| codex | -3.6 (n=11) | -7.2 (n=11) | -7.5 (n=11) | -0.1 (n=12) | -26.9 (n=5) |
| trae (both models) | -2.4 (n=18) | -4.0 (n=18) | -7.7 (n=18) | +0.1 (n=21) | -27.0 (n=10) |

## SGLang (isolated 3-way files, 15-task set)

| Agent | TTFT mean | ITL mean | Output throughput |
|---|---|---|---|
| claude_code | -2.9 (n=6) | +5.1 (n=6) | -0.2 (n=6) |
| codex | -3.2 (n=3) | +4.4 (n=3) | -0.5 (n=3) |
| trae_gpt5 | +11.2 (n=7) | +1.5 (n=7) | +2.4 (n=7) |
| trae_sonnet | +45.1 (n=4) | -7.0 (n=4) | +24.9 (n=4) |

_SGLang benchmark_serving emits ITL but no TPOT; OpenHands SGLang runs were benchmarked separately (throughput extractor) and are not in the isolated 3-way files._
