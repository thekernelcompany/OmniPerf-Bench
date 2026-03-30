# Claude Code Sonnet 4.5 — pass@k Collection Status

**Last updated:** 2026-03-30T05:10Z
**Agent:** claude_code | **Model:** sonnet (Sonnet 4.5 via Max subscription)
**HF repo:** `Inferencebench/pass-at-k-samples`
**Target:** 54 items (39 vllm + 15 sglang) x 8 samples = 432 total

---

## COLLECTION COMPLETE

| Metric | vllm | sglang | Total |
|--------|------|--------|-------|
| Plan items | 39 | 15 | 54 |
| Target samples | 312 | 120 | 432 |
| Valid on HF | 312 | 120 | **432** |
| **Tasks fully done (8/8 on HF)** | **39** | **15** | **54 / 54** |

All 54 tasks at 8/8 samples. Every sample has a non-empty `model_patch` (`status=success`).

---

## Timeline

| Date | Event |
|------|-------|
| 2026-03-28 11:51Z | sglang collection started (15 tasks, 120 samples) |
| 2026-03-28 13:43Z | sglang collection complete — 15/15 tasks at 8/8 |
| 2026-03-28 14:31Z | vllm collection started (39 tasks, 312 samples) |
| 2026-03-29 09:38Z | sglang 3 stranded samples pushed; vllm 2 stranded samples pushed |
| 2026-03-29 14:27Z | Runner died mid-prepare on vllm_core-0057/s5 (cause unknown) |
| 2026-03-29 17:19Z | Runner restarted — 94 remaining at that point |
| 2026-03-29 22:19Z | Runner died mid-prepare on vllm_core-0082/s2 (cause unknown) |
| 2026-03-29 23:01Z | Runner restarted after reconciliation — 46 remaining |
| 2026-03-30 04:51Z | **vllm collection complete** — 39/39 tasks at 8/8, 0 failures |
| 2026-03-30 05:05Z | All data extracted to `everything_analysis_data/pass_at_k/claude_code/sonnet/` |

Final runner session (started 23:01Z Mar 29, finished 04:51Z Mar 30): 46 run, 266 skipped, **0 failed**.

---

## Data Locations

### HuggingFace
- **Repo:** `Inferencebench/pass-at-k-samples`
- **Naming:** `data/{task_id}_{sample}_{epoch}.parquet` (e.g. `data/vllm_core-0082_s3_1774826285.parquet`)
- **Columns:** `item_id`, `sample_index`, `status`, `model_patch`, `agent_stdout`, `journal_json`, `run_summary_json`, `prompt_json`, `task_text`, `diff_targets_json`, `agent_stderr`, `trajectory_json`, `agent_name`, `model_name`, `human_commit`, `pre_commit`, `duration_s`, `patch_size_loc`, `changed_files_count`, `collected_at`
- **432 valid parquets** (plus ~99 stale duplicates/error shards superseded by good versions)

### Local — everything_analysis_data
- **Path:** `/home/ubuntu/everything_analysis_data/pass_at_k/claude_code/sonnet/`
- **Structure:** `{vllm,sglang}/{task_id}/s{0-7}/{model_patch.diff, journal.json, run_summary.json, prompt.json, task.txt, diff_targets.json, agent_stdout.txt, metadata.json}`
- **458 sample dirs** (432 from plan + 26 from older-scope tasks not in current plan)
- **0 empty patches** — verified all `model_patch.diff` files are non-empty
- Extracted via `pass_at_k/extract_from_hf.py` which deduplicates by taking the latest successful parquet per (task, sample)

### Local — ISO-Bench state
- Local run dirs are nuked after each successful push to HF (by design, to prevent agent peeking)
- Only `prompt.json` and `task.txt` remain for some older runs

---

## Completed Tasks (54/54)

### vllm (39/39) — ALL COMPLETE
vllm_core-0000, 0003, 0005, 0008, 0009, 0011, 0013, 0015, 0017, 0018,
0021, 0022, 0025, 0027, 0029, 0033, 0034, 0035, 0041, 0045, 0049, 0050,
0051, 0053, 0055, 0056, 0057, 0059, 0063, 0064, 0067, 0078, 0081, 0082,
0085, 0091, 0093, 0094, 0095

### sglang (15/15) — ALL COMPLETE
sglang_core-0000, 0003, 0005, 0006, 0008, 0017, 0019, 0027, 0033, 0047, 0059, 0065, 0068, 0070, 0071

---

## Runner Configuration

```bash
cd /home/ubuntu/OmniPerf-Bench/ISO-Bench
source .venv/bin/activate
python scripts/run_pass_at_k.py \
    --task tasks/vllm.yaml \
    --plan state/plan_iso.json \
    --bench-cfg bench-claude.yaml \
    --n 8 \
    --hf-repo Inferencebench/pass-at-k-samples \
    --resume \
    --max-parallel-items 1
```

### Agent Config (bench-claude.yaml)
- Agent: `claude_code` (CLI: `claude`)
- Model: `sonnet` (Sonnet 4.5)
- Time budget: 120 minutes per sample
- Container: 2 CPUs, 4 GB memory, no GPU, no network

## Log Files

| Log | Path |
|-----|------|
| vllm (final run) | `ISO-Bench/logs/claude_pass_at_k/vllm_20260330_225424.log` |
| vllm (died 22:19 Mar 29) | `ISO-Bench/logs/claude_pass_at_k/vllm_20260329_173000.log` |
| vllm (died 14:27 Mar 29) | `ISO-Bench/logs/claude_pass_at_k/vllm_20260329_093920.log` |
| vllm (initial session) | `ISO-Bench/logs/claude_pass_at_k/vllm_20260328_143136.log` |
| sglang (session) | `ISO-Bench/logs/claude_pass_at_k/sglang_20260328_115154.log` |

## HF Quality Notes

- **432/432 plan samples** have `status=success` and non-empty `model_patch`
- `agent_stdout` present for all samples (Claude Code transcript)
- `agent_stderr` and `trajectory_json` are empty by design (Claude Code agent doesn't produce these)
- ~99 stale/duplicate parquets on HF from older runs — superseded by good versions, harmless
- 14 extra task IDs on HF not in current plan (10 vllm + 4 sglang from older benchmark scope) — 43 files, harmless
