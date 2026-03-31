# Codex CLI Pass@K Status

Last updated: 2026-03-28T18:14:30Z

This is the Codex-specific pass@k ledger. It lives at `/home/ubuntu/OmniPerf-Bench/codex_cli_pass_at_k_status.md`.

## Scope

- Agent: `codex_cli`
- Model: `gpt-5`
- Subscription path used for live runs: `/usr/local/bin/codex -m gpt-5`
- HF repo: `Inferencebench/pass-at-k-samples`
- Authoritative task set: `39 vllm + 15 sglang = 54 tasks`
- pass@8 sample budget: `432 total samples`
- Success criterion for rerun purposes: non-empty `model_patch.diff`

## Canonical Locations

- Repo root: `/home/ubuntu/OmniPerf-Bench`
- vLLM worktree root: `/home/ubuntu/OmniPerf-Bench-remote/ISO-Bench`
- SGLang worktree root: `/home/ubuntu/OmniPerf-Bench-remote-sglang/ISO-Bench`
- HF cache root: `/home/ubuntu/.cache/huggingface/hub/datasets--Inferencebench--pass-at-k-samples/snapshots`
- vLLM strict completed file: `/home/ubuntu/OmniPerf-Bench-remote/ISO-Bench/state/completed_codex_cli_gpt5_vllm_iso_bench_39_strict.json`
- vLLM local runs: `/home/ubuntu/OmniPerf-Bench-remote/ISO-Bench/state/runs/vllm/codex_cli/gpt-5`
- vLLM upload queue: `/home/ubuntu/OmniPerf-Bench-remote/ISO-Bench/state/pass_at_k_upload_queue/inferencebench__pass-at-k-samples/data`
- SGLang strict completed file: `/home/ubuntu/OmniPerf-Bench-remote-sglang/ISO-Bench/state/completed_codex_cli_gpt5_sglang_iso_bench_15_strict.json`
- SGLang local runs: `/home/ubuntu/OmniPerf-Bench-remote-sglang/ISO-Bench/state/runs/sglan/codex_cli/gpt-5`
- SGLang upload queue: `/home/ubuntu/OmniPerf-Bench-remote-sglang/ISO-Bench/state/pass_at_k_upload_queue/inferencebench__pass-at-k-samples/data`
- Strict HF/local audit map: `/home/ubuntu/OmniPerf-Bench/docs/codex_cli_hf_strict_task_map_2026-03-28.json`

## Current Summary

| Family | Target | Strictly resolved on HF | Strictly resolved local-only | Strictly unresolved |
|---|---:|---:|---:|---:|---:|---:|
| vllm | 312 | 61 | 0 | 251 |
| sglang | 120 | 16 | 2 | 102 |

- The authoritative strict rerun scope is `353` samples total.
- `vllm` unresolved reasons from the strict map: `156 usage_limit`, `76 wrapper_not_logged_in`, `19 auth_401`.
- `sglang` unresolved reasons from the strict map: `72 usage_limit`, `30 wrapper_not_logged_in`.
- The stale `sglang_summary.unresolved=102` value in the JSON summary is only correct when the two local-only patch samples are counted as resolved; row-level counts are the source of truth.
- Strict completion files used for relaunch:
  - `vllm`: `/home/ubuntu/OmniPerf-Bench-remote/ISO-Bench/state/completed_codex_cli_gpt5_vllm_iso_bench_39_strict.json` (`61` resolved)
  - `sglang`: `/home/ubuntu/OmniPerf-Bench-remote-sglang/ISO-Bench/state/completed_codex_cli_gpt5_sglang_iso_bench_15_strict.json` (`18` resolved)

## Live Progress

| Family | Total target | Strictly resolved before relaunch | Active now | Remaining to run |
|---|---:|---:|---:|---:|---:|
| vllm | 312 | 61 | 0 | 251 |
| sglang | 120 | 18 | 0 | 102 |
| total | 432 | 79 | 0 | 353 |

- No codex rerun is active at this snapshot.
- The next relaunch must use only the strict completion files above.

## Latest Relaunch Attempt

- At `2026-03-28T17:36:03Z` a serialized strict rerun was launched from `/home/ubuntu/OmniPerf-Bench/logs/codex_pass_at_k/serialized_20260328.log`.
- Scheduling policy for that launch:
  - `vllm` first, then `sglang`
  - `--max-parallel-items 1`
  - strict completion files only
  - inline HF push+verify only
- The first checked newly pushed shard from that attempt was:
  - `data/vllm_core__codex_cli__gpt-5__vllm_core-0015__s5__1774719368.parquet`
  - actual contents on HF: `status='error'`, `model_patch_len=0`, `changed_files_count=0`
  - exact stderr tail: `ERROR: You've hit your usage limit ... try again at 8:45 PM.`
- A later shard from the same attempt was also inspected:
  - `data/vllm_core__codex_cli__gpt-5__vllm_core-0017__s5__1774719405.parquet`
  - same result: `status='error'`, `model_patch_len=0`, `changed_files_count=0`, usage-limit stderr
- Conclusion:
  - the internal fixes are holding
  - the current external blocker is still Codex usage exhaustion
  - continuing the rerun right now would only create more empty-patch error rows
- Action taken:
  - stopped the serialized rerun attempt
  - deleted the entire bad 9-shard aborted batch from HF with the local `hf` CLI
  - HF delete commit: `ff9d8d9e63f434e658f1504ebe9c0d2cf263f345`

## Cross-Model Smoke Tests

- Purpose:
  - determine whether the current blocker is specific to `gpt-5` or applies more broadly to Codex under the current shell subscription
- Temporary smoke repo:
  - `/tmp/codex-smoke-Bt8jAM`
- `gpt-5.4` smoke test:
  - command path used: `HOME=/home/ubuntu/OmniPerf-Bench-remote/ISO-Bench/.codex_home /usr/local/bin/codex -m gpt-5.4 exec ...`
  - result: failed immediately
  - exact error: `ERROR: You've hit your usage limit ... try again at 8:45 PM.`
- `gpt-5.3-codex-spark` smoke test:
  - command path used: `HOME=/home/ubuntu/OmniPerf-Bench-remote/ISO-Bench/.codex_home /usr/local/bin/codex -m gpt-5.3-codex-spark exec ...`
  - result: rejected before execution
  - exact error: `The 'gpt-5.3-codex-spark' model is not supported when using Codex with a ChatGPT account.`
- Implication:
  - the current blocker is not limited to `gpt-5`
  - at least `gpt-5.4` is also currently unusable for live Codex exec from this environment
  - `gpt-5.3-codex-spark` cannot be used as a fallback with the current ChatGPT-account auth path

## Auth Correction

- Root cause correction:
  - the benchmark runner homes had been authenticated as `shikhar2807.ace@gmail.com`
  - the intended current-shell account is `mlthings.me@gmail.com`
- Shared file-based Codex auth was explicitly enabled by creating:
  - `/home/ubuntu/.codex/config.toml`
  - with `cli_auth_credentials_store = "file"`
- Device login was then completed into shared `HOME=/home/ubuntu`, which created:
  - `/home/ubuntu/.codex/auth.json`
- Decoded shared auth now resolves to:
  - email: `mlthings.me@gmail.com`
  - name: `Ayush Nangia`
  - account_id: `05b09c30-d4ab-46e2-b8ea-ad9524c91810`
- The runner homes were then synced to this auth:
  - `/home/ubuntu/OmniPerf-Bench-remote/ISO-Bench/.codex_home/.codex/auth.json`
  - `/home/ubuntu/OmniPerf-Bench-remote-sglang/ISO-Bench/.codex_home/.codex/auth.json`
- Both runner homes now report `Logged in using ChatGPT` on the correct account.

## Post-Fix Smoke Test

- Re-ran a non-interactive Codex exec smoke test after syncing `mlthings` auth:
  - model: `gpt-5.4`
  - workdir: `/tmp/codex-smoke-Bt8jAM`
  - command path: `HOME=/home/ubuntu/OmniPerf-Bench-remote/ISO-Bench/.codex_home /usr/local/bin/codex -m gpt-5.4 exec ...`
- Result:
  - success
  - real file edit applied
  - verified `/tmp/codex-smoke-Bt8jAM/smoke.txt` changed from `before` to `after`
- Implication:
  - the earlier non-interactive failures were consistent with the wrong-account auth path
  - the current-shell `mlthings` auth is now usable for benchmark-style Codex exec

## Pre-Skip Audit

- `vllm` pre-skips are valid under the strict patch rule.
  - Source file: `/home/ubuntu/OmniPerf-Bench-remote/ISO-Bench/state/completed_codex_cli_gpt5_vllm_iso_bench_39_patch_only.json`
  - Count: `41`
  - HF artifact audit for those `41` keys:
    - `41/41` have non-empty `model_patch.diff`
    - `41/41` have `journal`
    - `41/41` have `run_summary`
    - `41/41` have `stderr`
    - `41/41` have `stdout`
    - `41/41` have `prompt`
    - `41/41` have `task`
    - `0/41` have `trajectory`
- `sglang` pre-skips are not valid under the strict patch rule.
  - Current source file in use: `/home/ubuntu/OmniPerf-Bench-remote-sglang/ISO-Bench/state/completed_codex_cli_gpt5_sglang_iso_bench_15_anywhere.json`
  - Count: `32`
  - HF artifact audit for those `32` keys:
    - `0/32` have non-empty `model_patch.diff`
    - `32/32` have `journal`
    - `32/32` have `run_summary`
    - `32/32` have `stderr`
    - `32/32` have `prompt`
    - `32/32` have `task`
    - `0/32` have `trajectory`
- Implication:
  - `vllm` pre-skipped samples can be treated as patch-bearing completions.
  - `sglang` pre-skipped samples should currently be interpreted as log-evidenced rows, not patch-bearing completions.
  - Any `sglang` progress/accounting that treats those `32` as complete under the strict patch rule is overstated.

## Live Runs

- Active codex vLLM sample: `none`
- Active codex SGLang sample: `none`
- Shared-home auth check: `HOME=/home/ubuntu /usr/local/bin/codex login status => Not logged in`
- Wrapper auth check now uses isolated persistent Codex homes and passes in both worktrees:
  - `/home/ubuntu/OmniPerf-Bench-remote/ISO-Bench/tools/codex_gpt5_high_wrapper.sh login status => Logged in using ChatGPT`
  - `/home/ubuntu/OmniPerf-Bench-remote-sglang/ISO-Bench/tools/codex_gpt5_high_wrapper.sh login status => Logged in using ChatGPT`
- Authenticated per-worktree Codex homes:
  - `/home/ubuntu/OmniPerf-Bench-remote/ISO-Bench/.codex_home`
  - `/home/ubuntu/OmniPerf-Bench-remote-sglang/ISO-Bench/.codex_home`

## Push Status

- Hardened both Codex wrappers to force shared `HOME=/home/ubuntu` and fail fast if `codex login status` is not authenticated.
- Permanent auth fix: both wrappers now run Codex from their own persistent isolated `.codex_home`, and they opportunistically sync credentials from the current shell's shared `/home/ubuntu/.codex` when available.
- This removes shared `~/.codex` volatility as the runtime dependency while still inheriting the current shell subscription when shared auth is present.
- Fixed the SGLang bench config to use the SGLang wrapper path instead of the vLLM wrapper path.
- Fixed the SGLang runner upload path to match vLLM: it now uploads inline instead of returning `queued:...`.
- Stopped the queue watchdog because it could race with vLLM inline pushes and is no longer needed once both runners use inline upload.
- Direct shared-home smoke test passed after the self-heal fix: `codex_cli` changed a test file from `before` to exact content `after`.
- Deleted known auth-broken HF shards from the 2026-03-28 rerun wave after confirming they were `401 Unauthorized` rows with no `model_patch.diff`.
- Deleted the later strict-rerun usage-limit shards from the aborted `2026-03-28T17:36Z` launch after confirming they were empty-patch error rows.
- Deleted the stale local queued auth-failure shard for `vllm_core-0033/s3` instead of pushing it.
- vLLM upload queue files: `0`
- SGLang upload queue files: `0`
- Latest known local auth-failure shard before cleanup: `vllm_core-0033/s3`
- Latest known local auth-failure stderr before cleanup: `codex wrapper error: shared HOME=/home/ubuntu is not logged in`
- Current state: queues are empty, both runners are configured for inline push+verify, and the hard blocker is current Codex usage-limit exhaustion rather than pipeline correctness.

## vLLM Historical Error Cause

- `191`: `ERROR: You've hit your usage limit. Upgrade to Pro (https://chatgpt.com/explore/pro), visit https://chatgpt.com/codex/settings/usage to purchase more credits or try again at 7:23 PM.`
- `64`: `ERROR: You've hit your usage limit. Upgrade to Pro (https://chatgpt.com/explore/pro), visit https://chatgpt.com/codex/settings/usage to purchase more credits or try again at 7:45 PM.`
- `12`: `ERROR: You've hit your usage limit. Upgrade to Pro (https://chatgpt.com/explore/pro), visit https://chatgpt.com/codex/settings/usage to purchase more credits or try again at 2:27 PM.`
- `1`: `ERROR: unexpected status 401 Unauthorized: Missing bearer or basic authentication in header, url: https://api.openai.com/v1/responses, cf-ray: 9e36fb5d5a63a124-IAD, request id: req_8d7e991ce8b34d9ab710c14f38da6641`
- `1`: `ERROR: unexpected status 401 Unauthorized: Missing bearer or basic authentication in header, url: https://api.openai.com/v1/responses, cf-ray: 9e36fbd13874c9a7-IAD, request id: req_37a4cf62b0e449e1b5cc9c2d0aa68862`
- `1`: `ERROR: unexpected status 401 Unauthorized: Missing bearer or basic authentication in header, url: https://api.openai.com/v1/responses, cf-ray: 9e3701886870f828-IAD, request id: req_bfe77b7144764c388ddad2a4997b84a0`
- `1`: `ERROR: unexpected status 401 Unauthorized: Missing bearer or basic authentication in header, url: https://api.openai.com/v1/responses, cf-ray: 9e3701fc3e675187-IAD, request id: req_08992e33db6c473897760954042a2ae7`

## vLLM Patch-Only Rerun Policy

- `41` samples are treated as complete because HF has a non-empty `model_patch.diff`.
- `271` samples are being rerun because the latest HF row for each has no patch.
- The live runner is using `--resume --completed-identities-json state/completed_codex_cli_gpt5_vllm_iso_bench_39_patch_only.json --skip-hf-resume-fetch` so only patch-bearing samples are skipped.
- The runner is expected to push each completed sample immediately, verify the exact HF shard, and only then nuke local state.
- As of 2026-03-28, HF is rate-limiting dataset commits (`128/hour`), so the runner now hard-stops if any local queue backlog exists.
