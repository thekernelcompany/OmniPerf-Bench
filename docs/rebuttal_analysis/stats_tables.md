# W1 / Q4b — Statistical tables (generated)

## True Success & Hard Success with 95% CIs

Canonical = post-X3 data (reproduces submitted Table 5 for 11/12 cells; vLLM OpenHands-Sonnet-4.5 canonical Q1=22 vs published 17 — published-count CIs shown in brackets for that cell).

| Project | Agent | n | True Succ % | Wilson 95% | Bootstrap 95% | Hard Succ % | Wilson 95% | Gap % |
|---|---|---|---|---|---|---|---|---|
| vllm | Claude Code | 39 | 46.2 | 31.6–61.4 | 30.8–61.5 | 56.4 | 41.0–70.7 | 10.3 |
| vllm | OpenHands (Sonnet-4.5) | 39 | 56.4 [pub 43.6, W 29.3–59.0] | 41.0–70.7 | 41.0–71.8 | 66.7 | 51.0–79.4 | 10.3 |
| vllm | TRAE (Sonnet) | 39 | 28.2 | 16.5–43.8 | 15.4–43.6 | 33.3 | 20.6–49.0 | 5.1 |
| vllm | OpenHands (GPT-5) | 39 | 28.2 | 16.5–43.8 | 15.4–43.6 | 43.6 | 29.3–59.0 | 15.4 |
| vllm | Codex CLI | 39 | 20.5 | 10.8–35.5 | 7.7–33.3 | 33.3 | 20.6–49.0 | 12.8 |
| vllm | TRAE (GPT-5) | 39 | 17.9 | 9.0–32.7 | 7.7–30.8 | 20.5 | 10.8–35.5 | 2.6 |
| sglang | Claude Code | 15 | 26.7 | 10.9–52.0 | 6.7–53.3 | 46.7 | 24.8–69.9 | 20.0 |
| sglang | OpenHands (Sonnet-4.5) | 15 | 13.3 | 3.7–37.9 | 0.0–33.3 | 13.3 | 3.7–37.9 | 0.0 |
| sglang | TRAE (Sonnet) | 15 | 80.0 | 54.8–93.0 | 60.0–100.0 | 80.0 | 54.8–93.0 | 0.0 |
| sglang | OpenHands (GPT-5) | 15 | 33.3 | 15.2–58.3 | 13.3–60.0 | 40.0 | 19.8–64.3 | 6.7 |
| sglang | Codex CLI | 15 | 80.0 | 54.8–93.0 | 60.0–100.0 | 80.0 | 54.8–93.0 | 0.0 |
| sglang | TRAE (GPT-5) | 15 | 86.7 | 62.1–96.3 | 66.7–100.0 | 86.7 | 62.1–96.3 | 0.0 |

## Pairwise McNemar (exact, True Success indicator, paired by task)

| Project | A | B | A-only | B-only | p (exact) |
|---|---|---|---|---|---|
| vllm | Claude Code | OpenHands (Sonnet-4.5) | 8 | 12 | 0.503 |
| vllm | Claude Code | TRAE (Sonnet) | 13 | 6 | 0.167 |
| vllm | Claude Code | OpenHands (GPT-5) | 10 | 3 | 0.092 |
| vllm | Claude Code | Codex CLI | 13 | 3 | **0.021** |
| vllm | Claude Code | TRAE (GPT-5) | 13 | 2 | **0.007** |
| vllm | OpenHands (Sonnet-4.5) | TRAE (Sonnet) | 15 | 4 | **0.019** |
| vllm | OpenHands (Sonnet-4.5) | OpenHands (GPT-5) | 12 | 1 | **0.003** |
| vllm | OpenHands (Sonnet-4.5) | Codex CLI | 15 | 1 | **0.001** |
| vllm | OpenHands (Sonnet-4.5) | TRAE (GPT-5) | 18 | 3 | **0.001** |
| vllm | TRAE (Sonnet) | OpenHands (GPT-5) | 8 | 8 | 1.000 |
| vllm | TRAE (Sonnet) | Codex CLI | 5 | 2 | 0.453 |
| vllm | TRAE (Sonnet) | TRAE (GPT-5) | 5 | 1 | 0.219 |
| vllm | OpenHands (GPT-5) | Codex CLI | 8 | 5 | 0.581 |
| vllm | OpenHands (GPT-5) | TRAE (GPT-5) | 8 | 4 | 0.388 |
| vllm | Codex CLI | TRAE (GPT-5) | 3 | 2 | 1.000 |
| sglang | Claude Code | OpenHands (Sonnet-4.5) | 4 | 2 | 0.688 |
| sglang | Claude Code | TRAE (Sonnet) | 0 | 8 | **0.008** |
| sglang | Claude Code | OpenHands (GPT-5) | 3 | 4 | 1.000 |
| sglang | Claude Code | Codex CLI | 0 | 8 | **0.008** |
| sglang | Claude Code | TRAE (GPT-5) | 0 | 9 | **0.004** |
| sglang | OpenHands (Sonnet-4.5) | TRAE (Sonnet) | 1 | 11 | **0.006** |
| sglang | OpenHands (Sonnet-4.5) | OpenHands (GPT-5) | 1 | 4 | 0.375 |
| sglang | OpenHands (Sonnet-4.5) | Codex CLI | 1 | 11 | **0.006** |
| sglang | OpenHands (Sonnet-4.5) | TRAE (GPT-5) | 1 | 12 | **0.003** |
| sglang | TRAE (Sonnet) | OpenHands (GPT-5) | 8 | 1 | **0.039** |
| sglang | TRAE (Sonnet) | Codex CLI | 0 | 0 | 1.000 |
| sglang | TRAE (Sonnet) | TRAE (GPT-5) | 1 | 2 | 1.000 |
| sglang | OpenHands (GPT-5) | Codex CLI | 1 | 8 | **0.039** |
| sglang | OpenHands (GPT-5) | TRAE (GPT-5) | 1 | 9 | **0.021** |
| sglang | Codex CLI | TRAE (GPT-5) | 1 | 2 | 1.000 |

## Same-only sensitivity (Related target NOT counted as correct)

| Project | Agent | True Succ (Same∨Related) % | True Succ (Same-only) % | Δ (pp) |
|---|---|---|---|---|
| vllm | Claude Code | 46.2 | 20.5 | -25.6 |
| vllm | OpenHands (Sonnet-4.5) | 56.4 | 7.7 | -48.7 |
| vllm | TRAE (Sonnet) | 28.2 | 7.7 | -20.5 |
| vllm | OpenHands (GPT-5) | 28.2 | 2.6 | -25.6 |
| vllm | Codex CLI | 20.5 | 5.1 | -15.4 |
| vllm | TRAE (GPT-5) | 17.9 | 10.3 | -7.7 |
| sglang | Claude Code | 26.7 | 0.0 | -26.7 |
| sglang | OpenHands (Sonnet-4.5) | 13.3 | 0.0 | -13.3 |
| sglang | TRAE (Sonnet) | 80.0 | 40.0 | -40.0 |
| sglang | OpenHands (GPT-5) | 33.3 | 13.3 | -20.0 |
| sglang | Codex CLI | 80.0 | 26.7 | -53.3 |
| sglang | TRAE (GPT-5) | 86.7 | 46.7 | -40.0 |
