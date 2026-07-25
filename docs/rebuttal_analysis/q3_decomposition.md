# W3 — Q3 (Lucky Win) decomposition (generated)

All Q3 cases from canonical quadrants: 26 (22 vLLM, 4 SGLang).
Correctness = GSM8K strict-match, agent patch vs unoptimized baseline (broken if drop > 5% pp). Coverage: legacy 4-agent vLLM cases only; OpenHands and SGLang cases lack lm-eval runs (marked NO_EVAL_DATA).

| Project | Commit | Agent | Hard | Target | Approach | Base acc | Agent acc | Correctness |
|---|---|---|---|---|---|---|---|---|
| sglang | 132dad87 | Claude Code | similar | different_target | valid_alternative | — | — | NO_EVAL_DATA |
| sglang | 31589e17 | Claude Code | similar | different_target | ineffective | — | — | NO_EVAL_DATA |
| sglang | dd1012fc | Claude Code | similar | different_target | valid_alternative | — | — | NO_EVAL_DATA |
| sglang | 132dad87 | OpenHands (GPT-5) | similar | different_target | valid_alternative | — | — | NO_EVAL_DATA |
| vllm | 22d33bac | Claude Code | beats | different_target | valid_alternative | 0.70 | 0.70 | PRESERVED |
| vllm | 2deb029d | Claude Code | similar | different_target | ineffective | 0.74 | 0.74 | PRESERVED |
| vllm | e3580537 | Claude Code | beats | different_target | ineffective | 0.74 | 0.74 | PRESERVED |
| vllm | fa63e710 | Claude Code | similar | different_target | ineffective | 0.51 | 0.51 | PRESERVED |
| vllm | 19d98e0c | Codex CLI | beats | different_target | ineffective | 0.77 | 0.77 | PRESERVED |
| vllm | 22d33bac | Codex CLI | beats | different_target | ineffective | 0.70 | 0.70 | PRESERVED |
| vllm | 58eee5f2 | Codex CLI | beats | different_target | ineffective | 0.72 | 0.72 | PRESERVED |
| vllm | 6a417b86 | Codex CLI | beats | different_target | ineffective | 0.71 | 0.71 | PRESERVED |
| vllm | 9badee53 | Codex CLI | similar | different_target | ineffective | 0.28 | 0.28 | PRESERVED |
| vllm | 22d33bac | OpenHands (GPT-5) | beats | different_target | valid_alternative | 0.70 | — | NO_EVAL_DATA |
| vllm | 30172b49 | OpenHands (GPT-5) | beats | different_target | ineffective | 0.71 | — | NO_EVAL_DATA |
| vllm | 58eee5f2 | OpenHands (GPT-5) | beats | different_target | valid_alternative | 0.72 | — | NO_EVAL_DATA |
| vllm | 6a417b86 | OpenHands (GPT-5) | beats | no_optimization | ineffective | 0.71 | — | NO_EVAL_DATA |
| vllm | b690e348 | OpenHands (GPT-5) | beats | different_target | ineffective | 0.39 | — | NO_EVAL_DATA |
| vllm | e206b543 | OpenHands (GPT-5) | beats | no_optimization | ineffective | 0.71 | — | NO_EVAL_DATA |
| vllm | 22d33bac | OpenHands (Sonnet-4.5) | beats | different_target | valid_alternative | 0.70 | — | NO_EVAL_DATA |
| vllm | 58eee5f2 | OpenHands (Sonnet-4.5) | beats | different_target | ineffective | 0.72 | — | NO_EVAL_DATA |
| vllm | 6a417b86 | OpenHands (Sonnet-4.5) | beats | different_target | ineffective | 0.71 | — | NO_EVAL_DATA |
| vllm | 9badee53 | OpenHands (Sonnet-4.5) | beats | different_target | ineffective | 0.28 | — | NO_EVAL_DATA |
| vllm | 22d33bac | TRAE (GPT-5) | beats | different_target | valid_alternative | 0.70 | 0.70 | PRESERVED |
| vllm | 22d33bac | TRAE (Sonnet) | beats | different_target | ineffective | 0.70 | 0.70 | PRESERVED |
| vllm | fe66b347 | TRAE (Sonnet) | beats | different_target | ineffective | 0.32 | 0.00 | BROKEN |

## Cross-tab (evaluated cases)

| Approach | PRESERVED (emergent win) | BROKEN (hacking the win) | NO_EVAL_DATA |
|---|---|---|---|
| ineffective | 9 | 1 | 8 |
| valid_alternative | 2 | 0 | 6 |

**Summary:** of 12 evaluated Q3 cases, 11 preserved correctness (emergent wins) and 1 broke it (hacking the win). 14 cases lack eval data (OpenHands vLLM + all SGLang).
