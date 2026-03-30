# Codex CLI Pass@K Status

Last updated: 2026-03-30T01:10:00Z

This is the Codex-specific pass@k ledger. It lives at `/home/ubuntu/OmniPerf-Bench/codex_cli_pass_at_k_status.md`.

## Scope

- Agent: `codex_cli`
- Model: `gpt-5`
- Subscription path used for live runs: `/usr/local/bin/codex -m gpt-5`
- HF repo: `Inferencebench/pass-at-k-samples`
- Authoritative task set: `39 vllm + 15 sglang = 54 tasks`
- pass@8 sample budget: `432 total samples`
- Success criterion for rerun purposes: non-empty `model_patch.diff`

## 2026-03-29 Reconciliation Addendum

- The March 28 notes mixed historical HF/local reconciliation with current local filesystem state.
- The corrected current view is:
  - strict-completed in ledgers: `217`
  - strict-completed currently present as local run dirs: `165`
  - strict-completed but not present as local run dirs on this machine: `52`
- Critical distinction:
  - `217` means strict-completed by ledger/HF reconciliation.
  - It does **not** mean `217` are instantly benchmarkable from current local disk.
  - The instantly benchmarkable count from current local run dirs is `165 = 137 vllm + 28 sglang`.
- HF shard-level verification was checked directly against the dataset tree, not via `load_dataset` alone.
- Result of raw HF shard verification:
  - `vllm`: `160/160` strict-completed keys are present on HF
  - `sglang`: `57/57` strict-completed keys are present on HF
  - combined: `217/217` strict-completed keys are present on HF
- Current overlap by source:
  - `165` strict-completed keys exist both on HF and as local run dirs
  - `52` strict-completed keys are HF-only from the perspective of this machine right now
- Current strict remaining count:
  - total budget: `432`
  - strict-completed: `217`
  - strict-unresolved remaining: `215`
  - breakdown: `152 vllm`, `63 sglang`
- Important local-artifact caveat:
  - local run-dir presence is weaker than local strict artifact completeness
  - among the `165` local strict-present keys, only a very small subset currently has a clearly strong local patch-bearing bundle on disk
  - do not interpret `165` as `165` locally complete patch-bearing bundles
- Canonical files for this correction:
  - `/home/ubuntu/OmniPerf-Bench/docs/codex_cli_reconciliation_2026-03-29.md`
  - `/home/ubuntu/OmniPerf-Bench/docs/codex_cli_strict_completed_with_local_runs_vllm_2026-03-29.txt`
  - `/home/ubuntu/OmniPerf-Bench/docs/codex_cli_strict_completed_with_local_runs_sglang_2026-03-29.txt`

## 2026-03-29 Final HF Reconciliation

- The strict ledgers are no longer the best answer to "what remains to run".
- Direct HF verification of the latest shard per in-plan `(item_id, sample_idx)` is now the authoritative completion view.
- Final in-plan HF state after the last targeted rerun:
  - total in-plan sample keys on HF: `432/432`
  - latest in-plan HF rows with non-empty patch: `432/432`
  - `vllm`: `312/312` latest rows are patch-bearing
  - `sglang`: `120/120` latest rows are patch-bearing
- Final operational conclusion:
  - actual remaining patch-bearing rerun scope: `0`
  - the earlier "`~4` remaining" recollection was stale
  - the earlier strict-ledger-derived "`215` remaining" count was also stale for live HF completion status
- The only real unfinished in-plan latest row before final reconciliation was:
  - `vllm_core-0095 / s7`
- That sample was rerun and completed successfully at:
  - run id: `vllm/codex_cli/gpt-5/2026-03-29_22-56-44_s7`
  - pushed shard: `data/vllm_core__codex_cli__gpt-5__vllm_core-0095__s7__1774825155.parquet`
  - verified HF row: `status='success'`, non-empty patch
- Important residual caveat:
  - many HF success rows still have incomplete auxiliary fields, especially missing `trajectory_json`
  - this does **not** block patch-bearing pass@k completion, but it means "strict artifact completeness" and "HF patch completion" are different notions
- Explicitly stale sections below:
  - any section in this file claiming `353` remaining or `79` resolved
  - any section implying both Codex reruns are still live
  - any section treating strict ledgers alone as the current rerun source of truth

## 2026-03-30 Shard Audit And Targeted Backfill

- A follow-up shard-level audit was run for the 8 disputed in-plan sample keys:
  - `sglang_core-0071/s3`
  - `sglang_core-0071/s4`
  - `sglang_core-0071/s5`
  - `sglang_core-0071/s6`
  - `sglang_core-0071/s7`
  - `vllm_core-0055/s5`
  - `vllm_core-0078/s3`
  - `vllm_core-0081/s0`
- Critical correction from that audit:
  - those 8 keys were not "missing all evidence"
  - they each had HF shard history
  - but there was no successful patch-bearing `codex_cli/gpt-5` row for any of them at the time of audit
- What existed before rerun:
  - newer simple-name shards for several keys belonged to `claude_code/sonnet`
  - older Codex-specific shards for those same keys were `status='error'` with empty `model_patch`
  - some local Codex run dirs also existed, but where present they had empty `model_patch.diff`
- Targeted rerun policy used:
  - explicit completed-identity files were built from Codex-specific successful shard history for just the affected items
  - this avoided replaying already-complete samples while still rerunning the 8 genuine Codex gaps
- SGLang targeted rerun:
  - item: `sglang_core-0071`
  - skipped as already complete: `s0`, `s1`, `s2`
  - rerun and push-verified:
    - `s3` -> `data/sglang_core__codex_cli__gpt-5__sglang_core-0071__s3__1774831322.parquet`
    - `s4` -> `data/sglang_core__codex_cli__gpt-5__sglang_core-0071__s4__1774831494.parquet`
    - `s5` -> `data/sglang_core__codex_cli__gpt-5__sglang_core-0071__s5__1774831587.parquet`
    - `s6` -> `data/sglang_core__codex_cli__gpt-5__sglang_core-0071__s6__1774831690.parquet`
    - `s7` -> `data/sglang_core__codex_cli__gpt-5__sglang_core-0071__s7__1774831828.parquet`
  - final log: `/home/ubuntu/OmniPerf-Bench/logs/codex_pass_at_k/sglang_missing8_resume_20260330.log`
- vLLM targeted rerun:
  - items: `vllm_core-0055`, `vllm_core-0078`, `vllm_core-0081`
  - rerun and push-verified:
    - `vllm_core-0055/s5` -> `data/vllm_core__codex_cli__gpt-5__vllm_core-0055__s5__1774832105.parquet`
    - `vllm_core-0078/s3` -> `data/vllm_core__codex_cli__gpt-5__vllm_core-0078__s3__1774832253.parquet`
    - `vllm_core-0081/s0` -> `data/vllm_core__codex_cli__gpt-5__vllm_core-0081__s0__1774832836.parquet`
  - final log: `/home/ubuntu/OmniPerf-Bench/logs/codex_pass_at_k/vllm_missing8_resume_20260330.log`
- Post-rerun operational conclusion:
  - the 8 disputed in-plan Codex gaps are now closed on HF
  - the in-plan `432/432` patch-bearing completion view still holds after direct recheck

## 2026-03-30 pass_at_k Mirror Update

- The mirror repo at `/home/ubuntu/everything_analysis_data` was also updated.
- Root-cause fix:
  - `/home/ubuntu/everything_analysis_data/pass_at_k/extract_codex_cli_gpt5.py` had been deduping by latest shard per `(item_id, sample_index)` before filtering to `codex_cli/gpt-5`
  - that was wrong in a mixed-agent HF history because newer `claude_code` shards could mask older successful Codex shards
  - the extractor was corrected to pick the latest successful patch-bearing shard within `codex_cli/gpt-5`
- After rerunning the corrected extractor:
  - output root: `/home/ubuntu/everything_analysis_data/pass_at_k/codex_cli/gpt-5`
  - summary:
    - `hf_successful_unique_samples`: `544`
    - `local_only_successful_unique_samples`: `2`
    - `total_written_sample_dirs`: `546`
- The 8 newly rerun samples were verified to exist in the mirror with materialized artifact bundles:
  - `/home/ubuntu/everything_analysis_data/pass_at_k/codex_cli/gpt-5/sglang/sglang_core-0071/s3`
  - `/home/ubuntu/everything_analysis_data/pass_at_k/codex_cli/gpt-5/sglang/sglang_core-0071/s4`
  - `/home/ubuntu/everything_analysis_data/pass_at_k/codex_cli/gpt-5/sglang/sglang_core-0071/s5`
  - `/home/ubuntu/everything_analysis_data/pass_at_k/codex_cli/gpt-5/sglang/sglang_core-0071/s6`
  - `/home/ubuntu/everything_analysis_data/pass_at_k/codex_cli/gpt-5/sglang/sglang_core-0071/s7`
  - `/home/ubuntu/everything_analysis_data/pass_at_k/codex_cli/gpt-5/vllm/vllm_core-0055/s5`
  - `/home/ubuntu/everything_analysis_data/pass_at_k/codex_cli/gpt-5/vllm/vllm_core-0078/s3`
  - `/home/ubuntu/everything_analysis_data/pass_at_k/codex_cli/gpt-5/vllm/vllm_core-0081/s0`
- Important caveat:
  - this mirror is still **not** strict-plan-only
  - it now includes the strict set, but it also includes additional non-strict successful Codex samples because the extractor's scope is "all successful `codex_cli/gpt-5` rows recoverable from HF plus local-only successes"

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

| Family | Target | Latest HF rows present | Latest HF rows with non-empty patch | Actual remaining patch-bearing reruns |
|---|---:|---:|---:|---:|
| vllm | 312 | 312 | 312 | 0 |
| sglang | 120 | 120 | 120 | 0 |
| total | 432 | 432 | 432 | 0 |

- This summary is based on direct enumeration of the current HF dataset tree and inspection of the latest row per in-plan sample key.
- It supersedes the older strict-ledger-only counts in this file.

## Live Progress

| Family | Total target | Completed on latest HF patch criterion | Active now | Remaining to run |
|---|---:|---:|---:|---:|
| vllm | 312 | 312 | 0 | 0 |
| sglang | 120 | 120 | 0 | 0 |
| total | 432 | 432 | 0 | 0 |

- No Codex rerun is active at this snapshot.
- No additional rerun is currently warranted for the patch-bearing pass@k objective.

## Latest Relaunch Attempt

- The final targeted relaunch was the single-sample rerun of `vllm_core-0095/s7`.
- Context:
  - direct HF verification showed that all other in-plan samples already had patch-bearing latest rows
  - the previous `vllm_core-0095/s7` attempt had died mid-prepare and never pushed
- The first naive resume attempt on `vllm_core-0095` was aborted because the runner undercounted already-complete HF samples and started to rerun `s0`.
- Safe relaunch used:
  - `--resume`
  - `--completed-identities-json state/completed_codex_cli_gpt5_vllm_core-0095_s0_s6_20260329.json`
  - `--skip-hf-resume-fetch`
  - `--items vllm_core-0095`
  - `--max-parallel-items 1`
- Result:
  - `2026-03-29 22:56:44 UTC`: targeted rerun started
  - `2026-03-29 22:59:23 UTC`: push verified and local state nuked
  - final log: `/home/ubuntu/OmniPerf-Bench/logs/codex_pass_at_k/vllm_resume_single_0095_s7_strict_20260329.log`

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
- Current rerun state:
  - no active Codex pass@k runner
  - no active `bench.cli prepare` for Codex pass@k
  - no pending remaining sample under the patch-bearing completion rule
- Contention caveat:
  - other agent families can still target the same HF dataset
  - that remains an operational risk for future reruns, but there is no current Codex rerun to contend with
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
- Current state:
  - queues are empty
  - both runners are configured for inline push+verify
  - the last real remaining Codex sample has now been pushed successfully
  - the remaining issue is ledger/resume staleness, not missing HF patch-bearing samples

## vLLM Historical Error Cause

- `191`: `ERROR: You've hit your usage limit. Upgrade to Pro (https://chatgpt.com/explore/pro), visit https://chatgpt.com/codex/settings/usage to purchase more credits or try again at 7:23 PM.`
- `64`: `ERROR: You've hit your usage limit. Upgrade to Pro (https://chatgpt.com/explore/pro), visit https://chatgpt.com/codex/settings/usage to purchase more credits or try again at 7:45 PM.`
- `12`: `ERROR: You've hit your usage limit. Upgrade to Pro (https://chatgpt.com/explore/pro), visit https://chatgpt.com/codex/settings/usage to purchase more credits or try again at 2:27 PM.`
- `1`: `ERROR: unexpected status 401 Unauthorized: Missing bearer or basic authentication in header, url: https://api.openai.com/v1/responses, cf-ray: 9e36fb5d5a63a124-IAD, request id: req_8d7e991ce8b34d9ab710c14f38da6641`
- `1`: `ERROR: unexpected status 401 Unauthorized: Missing bearer or basic authentication in header, url: https://api.openai.com/v1/responses, cf-ray: 9e36fbd13874c9a7-IAD, request id: req_37a4cf62b0e449e1b5cc9c2d0aa68862`
- `1`: `ERROR: unexpected status 401 Unauthorized: Missing bearer or basic authentication in header, url: https://api.openai.com/v1/responses, cf-ray: 9e3701886870f828-IAD, request id: req_bfe77b7144764c388ddad2a4997b84a0`
- `1`: `ERROR: unexpected status 401 Unauthorized: Missing bearer or basic authentication in header, url: https://api.openai.com/v1/responses, cf-ray: 9e3701fc3e675187-IAD, request id: req_08992e33db6c473897760954042a2ae7`

## vLLM Patch-Only Rerun Policy

- Historical note:
  - this section described an earlier patch-only rerun policy when only `41` vLLM samples were being treated as complete
  - it is no longer current
- Current vLLM patch-bearing completion state:
  - latest HF patch-bearing rows for in-plan vLLM samples: `312/312`
  - actual remaining vLLM rerun scope under the patch-bearing rule: `0`
