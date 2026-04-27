We are using the repo-root OpenRouter key from `.env`:

`[REDACTED-OPENROUTER-KEY]`  
`sha256[:12]=[REDACTED]`

I verified this from the live process env before OpenHands exited. `OPENROUTER_API_KEY` and `LLM_API_KEY` were both set to that same key for `bench.cli`, `openhands.core.main`, and `action_execution_server`.

The other key in `ISO-Bench/.env` is not the one used now:

`[REDACTED-OPENROUTER-KEY]`  
`sha256[:12]=[REDACTED]`

That second key was the stale/bad one that produced the OpenRouter `401 User not found` earlier.

The OpenHands run has now completed and produced real logs/artifacts here:

[run directory](/home/raven/coding-mess/kernel-corp/OmniPerf-Bench/ISO-Bench/state/runs/vllm/openhands/openrouter-anthropic-claude-sonnet-4-5/2026-04-27_19-58-00/vllm_core-0095-openhands-sonnet45-20260427)

Key artifacts:
- [openhands_stderr.txt](/home/raven/coding-mess/kernel-corp/OmniPerf-Bench/ISO-Bench/state/runs/vllm/openhands/openrouter-anthropic-claude-sonnet-4-5/2026-04-27_19-58-00/vllm_core-0095-openhands-sonnet45-20260427/openhands_stderr.txt): 5,780 lines, 684,730 bytes
- [trajectory.json](/home/raven/coding-mess/kernel-corp/OmniPerf-Bench/ISO-Bench/state/runs/vllm/openhands/openrouter-anthropic-claude-sonnet-4-5/2026-04-27_19-58-00/vllm_core-0095-openhands-sonnet45-20260427/trajectory.json): 79 events, 584,642 bytes
- [model_patch.diff](/home/raven/coding-mess/kernel-corp/OmniPerf-Bench/ISO-Bench/state/runs/vllm/openhands/openrouter-anthropic-claude-sonnet-4-5/2026-04-27_19-58-00/vllm_core-0095-openhands-sonnet45-20260427/model_patch.diff): 165 lines

This run did not hit the 2-hour budget. It finished successfully in about 296.5 seconds with return code `0`. No OpenHands processes are still running.
