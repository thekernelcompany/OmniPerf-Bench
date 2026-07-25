# kNyS Q3 — Trajectory component analysis (generated)

TRAE from local curated trajectories (per-step timestamps + tool names); OpenHands from the HF rebuttal snapshots (event logs, latest timestamp per task). Claude Code and Codex CLI have no step logs — see the coarse table below. All values are per-config medians unless noted.

| Repo | Config | n | steps | edits | bash/run | TTF-edit (s) | duration (s) | finished % |
|---|---|---|---|---|---|---|---|---|
| vllm | trae_sonnet | 39 | 44 | 17 | 24 | 8.4 | 502.1 | 0 |
| vllm | trae_gpt5 | 39 | 42 | 19 | 17 | 80.3 | 1143.1 | 36 |
| vllm | openhands_sonnet45 | 36 | 109 | 13 | 25 | 72.3 | 387.8 | 94 |
| vllm | openhands_gpt5 | 39 | 79 | 9 | 19 | 324.1 | 836.1 | 100 |
| sglang | openhands_sonnet45 | 5 | 114 | 15 | 25 | 74.7 | 474.2 | 80 |
| sglang | openhands_gpt5 | 15 | 93 | 10 | 23 | 341.5 | 1004.0 | 100 |

## Coarse per-run metrics (all six configs, vLLM)

| Config | n | patch generated % | median duration (s) | median LOC added | median files changed |
|---|---|---|---|---|---|
| claude_code | 39 | 100 | 207 | 20 | 2 |
| codex | 39 | 100 | 346 | 28 | 2 |
| trae_sonnet | 39 | 100 | 506 | 20 | 2 |
| trae_gpt5 | 39 | 97 | 1209 | 18 | 2 |
| openhands_sonnet45 | 39 | 97 | 411 | 44 | 2 |
| openhands_gpt5 | 39 | 97 | 870 | 32 | 2 |

_"finished %" semantics differ per harness: TRAE = trajectory `success` flag; OpenHands = explicit `finish` action emitted. Not directly comparable across harnesses._
