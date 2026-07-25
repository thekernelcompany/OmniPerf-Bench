# W4 — Absolute improvement over unoptimized baseline (generated)

Caveats: (1) metric family per task = canonical `primary_metric` (majority across tools; falls back to whatever family has data); (2) vLLM OpenHands agent values come from the X3 re-bench context joined against the earlier parquet baseline — cross-context, treat as indicative only; legacy-agent values share the baseline's measurement context; (3) all numbers are single benchmark runs.

## vllm — human patch vs baseline (coverage 28/39 tasks)

- human improvement over baseline: min -31.9%, median 1.5%, max 99.8%
- tasks with human improvement > 5%: 11/28

| Agent | n | median agent-vs-baseline % | agent>+5% | agent<-5% |
|---|---|---|---|---|
| claude_code | 24 | 3.2 | 11 | 5 |
| openhands_sonnet45 | 23 | 44.8 | 18 | 5 |
| trae_sonnet | 16 | -0.0 | 6 | 6 |
| openhands_gpt5 | 21 | 48.2 | 14 | 6 |
| codex | 16 | -1.6 | 6 | 7 |
| trae_gpt5 | 12 | 1.0 | 5 | 3 |

## sglang — human patch vs baseline (coverage 14/15 tasks)

- human improvement over baseline: min -31.5%, median -11.9%, max 24.5%
- tasks with human improvement > 5%: 2/14

| Agent | n | median agent-vs-baseline % | agent>+5% | agent<-5% |
|---|---|---|---|---|
| claude_code | 6 | -14.0 | 0 | 4 |
| openhands_sonnet45 | 11 | -33.3 | 2 | 9 |
| trae_sonnet | 4 | -0.7 | 1 | 1 |
| openhands_gpt5 | 11 | -16.1 | 1 | 8 |
| codex | 3 | -12.1 | 0 | 2 |
| trae_gpt5 | 7 | 1.3 | 2 | 2 |

## Per-commit detail

| Project | Commit | Metric | Human % | claude_code | openhands_sonnet45 | trae_sonnet | openhands_gpt5 | codex | trae_gpt5 |
|---|---|---|---|---|---|---|---|---|---|
| vllm | fc542144 | ttft | -7.6 | -7.9 | -934.4 | -1764.3 | -920.1 | -1792.7 | — |
| vllm | fa63e710 | throughput | -0.2 | -0.2 | -10.9 | -0.0 | -10.3 | -0.1 | -0.1 |
| vllm | fa63e710 | latency | 0.0 | 0.0 | — | -0.1 | — | -0.1 | -0.1 |
| vllm | 6dd94dbe | throughput | 25.0 | — | — | — | — | — | — |
| vllm | 6dd94dbe | latency | 24.2 | — | — | — | — | — | — |
| vllm | 310aca88 | throughput | 99.8 | 100.2 | — | — | — | — | — |
| vllm | 310aca88 | latency | -1.2 | 0.4 | — | — | — | — | — |
| vllm | b55ed6ef | ttft | 9.9 | 7.8 | 51.0 | 49.3 | 73.7 | 48.5 | 48.8 |
| vllm | 3b61cb45 | throughput | 0.0 | — | 15.0 | -25.0 | -13.0 | -25.2 | -22.0 |
| vllm | 3b61cb45 | latency | -0.9 | — | — | -43.5 | — | -43.9 | -44.0 |
| vllm | 98f47f2a | throughput | 0.0 | 5.3 | 10.6 | 21.1 | — | 21.1 | 21.1 |
| vllm | 98f47f2a | latency | -1.3 | -2.3 | — | 17.4 | — | 16.0 | 16.6 |
| vllm | 8c1e77fb | throughput | 0.9 | — | 12.3 | -27.5 | -14.4 | -27.3 | -27.4 |
| vllm | 8c1e77fb | latency | 1.2 | — | — | -45.4 | — | -45.7 | -45.5 |
| vllm | 2deb029d | throughput | 0.3 | 0.7 | -15.7 | — | — | — | — |
| vllm | 2deb029d | latency | 29.3 | 1.8 | — | — | — | — | — |
| vllm | 3476ed08 | latency | -3.7 | -8.8 | — | — | — | — | — |
| vllm | b690e348 | ttft | 75.8 | 74.5 | 99.3 | — | 99.1 | — | — |
| vllm | 58eee5f2 | ttft | 3.2 | 0.3 | 36.1 | 27.8 | 58.1 | 28.2 | 24.2 |
| vllm | a3223766 | ttft | 6.2 | 13.9 | 54.6 | — | 48.2 | — | — |
| vllm | e7b20426 | ttft | 67.2 | — | — | 66.0 | — | 65.3 | — |
| vllm | e7b20426 | throughput | -10.0 | — | — | -20.0 | — | -21.5 | — |
| vllm | 015069b0 | ttft | -30.8 | -28.7 | -150.3 | — | -143.6 | — | — |
| vllm | 015069b0 | throughput | -0.0 | -0.0 | 43.3 | — | 43.9 | — | — |
| vllm | bc7c4d20 | ttft | -3.5 | -0.8 | 87.2 | — | 85.5 | — | — |
| vllm | 299ebb62 | ttft | 12.1 | 10.8 | 33.9 | -729.3 | 5.1 | -719.7 | — |
| vllm | 296f927f | ttft | 3.4 | -0.1 | 76.1 | — | 76.3 | — | -2.7 |
| vllm | 296f927f | throughput | 0.5 | -0.1 | 105.8 | — | 56.5 | — | 21.8 |
| vllm | 22d33bac | ttft | -31.9 | -9.2 | 44.8 | 1.0 | 47.0 | 0.3 | 2.1 |
| vllm | 22d33bac | throughput | 92.8 | — | 10.7 | 51.5 | 11.8 | 51.5 | 52.9 |
| vllm | 99abb8b6 | ttft | 92.7 | 91.6 | 96.0 | 91.6 | 95.8 | 91.6 | 91.5 |
| vllm | 99abb8b6 | throughput | -0.0 | — | 11.5 | 7.1 | -14.2 | 6.8 | 6.5 |
| vllm | 99abb8b6 | latency | -0.3 | — | — | — | — | — | — |
| vllm | fe66b347 | ttft | 8.1 | 5.6 | 92.9 | — | 94.5 | — | — |
| vllm | 70b808fe | ttft | 1.8 | 2.7 | 27.3 | — | -0.4 | — | -1035.0 |
| vllm | 9f1710f1 | ttft | -1.2 | -0.8 | -3606.4 | -57.2 | -5729.8 | -54.1 | — |
| vllm | 9f1710f1 | throughput | -0.1 | -1.1 | 23.5 | 4593.7 | -11.8 | 4630.3 | — |
| vllm | 9badee53 | ttft | 94.2 | 94.0 | 94.6 | 94.5 | 95.9 | 94.2 | 94.1 |
| vllm | 9badee53 | throughput | -67.7 | — | -57.2 | -10.4 | -60.0 | -17.5 | -12.4 |
| vllm | e206b543 | ttft | 0.4 | -16.2 | 47.8 | -2.3 | 40.3 | -3.1 | — |
| vllm | e206b543 | throughput | -0.4 | — | -7.4 | -0.4 | -26.7 | -0.6 | — |
| vllm | 6a417b86 | ttft | 34.1 | 37.7 | 79.7 | — | 80.8 | — | — |
| vllm | 4c822298 | throughput | 50.0 | 50.3 | — | 0.0 | — | -50.0 | -0.2 |
| vllm | 4c822298 | latency | -0.2 | -0.0 | — | -28.0 | — | -27.1 | -26.5 |
| vllm | 30172b49 | ttft | 1.1 | 3.7 | 55.8 | 46.3 | 70.4 | 47.5 | — |
| sglang | 021f76e4 | throughput | -18.8 | — | -31.4 | — | -62.9 | — | — |
| sglang | 132dad87 | throughput | -20.2 | -17.5 | -38.6 | — | -17.2 | — | — |
| sglang | 187b85b7 | throughput | -19.0 | -19.8 | — | — | -16.1 | — | — |
| sglang | 1acca3a2 | throughput | 23.0 | — | — | — | 4.8 | — | — |
| sglang | 205d5cb4 | throughput | 0.3 | — | — | — | — | — | 2.9 |
| sglang | 2ed68d7a | throughput | 2.7 | — | 8.5 | — | 55.2 | — | — |
| sglang | 31589e17 | throughput | 0.3 | -2.6 | -98.1 | — | — | — | 11.5 |
| sglang | 6b231325 | throughput | 1.9 | 2.6 | -14.6 | — | -14.3 | 1.4 | 1.3 |
| sglang | 73b13e69 | throughput | 24.5 | — | -8.0 | — | 3.3 | — | 17.2 |
| sglang | a191a0e4 | throughput | -11.3 | — | 19.4 | — | -17.2 | — | 0.5 |
| sglang | da47621c | throughput | -21.4 | — | -33.3 | -1.0 | -16.2 | — | -21.0 |
| sglang | dd1012fc | throughput | -12.6 | -10.7 | -42.7 | 8.3 | -10.4 | -12.1 | — |
| sglang | df7f61ee | throughput | -15.5 | -17.3 | -46.6 | -17.7 | — | -17.3 | — |
| sglang | e3ec6bf4 | throughput | -31.5 | — | -37.8 | -0.4 | -16.2 | — | -29.9 |
