# Claude Code Sonnet 4.5 — pass@k Collection Status

**Last updated:** 2026-03-29T02:21Z
**Agent:** claude_code | **Model:** sonnet (Sonnet 4.5 via Max subscription)
**HF repo:** `Inferencebench/pass-at-k-samples`
**Target:** 54 items (39 vllm + 15 sglang) x 8 samples = 432 total

---

## Overall Progress

| Metric | vllm | sglang | Total |
|--------|------|--------|-------|
| Plan items | 39 | 15 | 54 |
| Target samples | 312 | 120 | 432 |
| Valid on HF (pre-run) | 21 | 39 | 60 |
| Completed this run | 151 | 81 | 232 |
| Pushed to HF this run | 146 | 80 | 226 |
| **Genuinely stranded** | **2** | **1** | **3** |
| Agent errors | 0 | 0 | 0 |
| Remaining (samples) | ~145 | **0** | ~145 |
| **Tasks fully done (8/8 on HF)** | **20** | **14** | **34 / 54** |

**Agent success rate: 232/232 (100%)**
**Push success rate: 226/232 (97%)**
**Quality audit (22:55Z): 254 unique valid samples on HF — ALL have model_patch + full artifacts. Zero issues.**

### SGLANG COMPLETE! (21:42Z)
All 15 sglang items done (81 run, 39 skipped, 1 failed = 120 total). 80 pushed to HF.
Only `sglang_core-0003/s1` stranded (HF 429 pre-Pro). All other 14 items have 8/8 on HF.
**Sglang runner has exited. Only vllm runner remains (187 samples left).**

**Stranded (3):** `vllm_core-0009/s6` (429), `vllm_core-0022/s2` (412), `sglang_core-0003/s1` (429). All have valid data locally.
**Quality audit (17:35Z): 154 unique valid samples on HF — ALL have model_patch + journal + stdout + prompt + task + diff_targets + run_summary. Zero issues.**
**Genuinely stranded (2):** `vllm_core-0009/s6` + `sglang_core-0003/s1`. Both have valid local data, need manual push or re-run.
**Stale dirs (3):** Old `vllm_core-0005` s1/s2/s3 — data already on HF, can be deleted.
**Quality check (15:01Z):** All shards on HF inspected — 100% valid patches, full artifacts.

### Tasks fully completed (8/8 samples on HF)
| # | Task | Source |
|---|------|--------|
| 1 | vllm_core-0000 | pre-existing |
| 2 | vllm_core-0003 | this run |
| 3 | vllm_core-0005 | this run (s1-s3 re-run after restart) |
| 4 | vllm_core-0008 | this run |
| 5 | sglang_core-0000 | this run + pre-existing |
| 6 | sglang_core-0005 | this run |
| 7 | sglang_core-0017 | pre-existing |
| 8 | sglang_core-0059 | pre-existing |
| 9 | sglang_core-0065 | pre-existing |
| 10 | sglang_core-0068 | pre-existing |

## Active Processes

| Process | Window | Status |
|---------|--------|--------|
| vllm pass@k runner | tmux:2 (`claude_vllm`) | Running — `vllm_core-0050/s1` (125/267) — new item |
| sglang pass@k runner | tmux:3 (`claude_sglang`) | **FINISHED** at 21:42Z |
| claude -p agents | — | 2 active |

---

## ACTION REQUIRED: Manual Push for Stranded Samples

These completed successfully but HF push failed (429 rate limit). **Local state preserved. DO NOT delete until pushed.**

| # | Sample | Item | Duration | Patch LOC | Local path |
|---|--------|------|----------|-----------|------------|
| 1 | **vllm_core-0005/s1** | vllm_core-0005 | 236s | 8 | `state/runs/vllm/claude_code/sonnet/2026-03-28_12-47-19_s1/vllm_core-0005/` |
| 2 | **vllm_core-0005/s2** | vllm_core-0005 | 377s | 31 | `state/runs/vllm/claude_code/sonnet/2026-03-28_12-51-17_s2/vllm_core-0005/` |
| 3 | **vllm_core-0005/s3** | vllm_core-0005 | 344s | 32 | `state/runs/vllm/claude_code/sonnet/2026-03-28_12-57-36_s3/vllm_core-0005/` |
| 4 | **sglang_core-0003/s1** | sglang_core-0003 | 397s | 91 | `state/runs/sglan/claude_code/sonnet/2026-03-28_12-48-37_s1/sglang_core-0003/` |

---

## Confirmed on HF (this run) — 19 shards

| # | Sample | Duration | HF shard | Nuked |
|---|--------|----------|----------|-------|
| 1 | vllm_core-0003/s0 | 467s | `vllm_core-0003_s0_1774698610.parquet` | Yes |
| 2 | vllm_core-0003/s1 | 831s | `vllm_core-0003_s1_1774699446.parquet` | Yes |
| 3 | vllm_core-0003/s2 | 440s | `vllm_core-0003_s2_1774699888.parquet` | Yes |
| 4 | vllm_core-0003/s3 | 308s | `vllm_core-0003_s3_1774700199.parquet` | Yes |
| 5 | vllm_core-0003/s4 | 503s | `vllm_core-0003_s4_1774700707.parquet` | Yes |
| 6 | vllm_core-0003/s5 | 291s | `vllm_core-0003_s5_1774700999.parquet` | Yes |
| 7 | vllm_core-0003/s6 | 324s | `vllm_core-0003_s6_1774701325.parquet` | Yes |
| 8 | vllm_core-0003/s7 | 473s | `vllm_core-0003_s7_1774701800.parquet` | Yes |
| 9 | vllm_core-0005/s0 | 236s | `vllm_core-0005_s0_1774702037.parquet` | Yes |
| 10 | vllm_core-0005/s4 | 454s | `vllm_core-0005_s4_*.parquet` | Yes |
| 11 | vllm_core-0005/s5 | 277s | `vllm_core-0005_s5_*.parquet` | Yes |
| 12 | sglang_core-0000/s2 | 565s | `sglang_core-0000_s2_1774699546.parquet` | Yes |
| 12 | sglang_core-0000/s3 | 325s | `sglang_core-0000_s3_1774699875.parquet` | Yes |
| 13 | sglang_core-0000/s4 | 336s | `sglang_core-0000_s4_1774700214.parquet` | Yes |
| 14 | sglang_core-0000/s5 | 592s | `sglang_core-0000_s5_1774700808.parquet` | Yes |
| 15 | sglang_core-0000/s6 | 446s | `sglang_core-0000_s6_1774701256.parquet` | Yes |
| 16 | sglang_core-0000/s7 | 487s | `sglang_core-0000_s7_1774701744.parquet` | Yes |
| 17 | sglang_core-0003/s0 | 369s | `sglang_core-0003_s0_1774702116.parquet` | Yes |
| 18 | sglang_core-0003/s2 | ~540s | `sglang_core-0003_s2_*.parquet` | Yes |
| 19 | sglang_core-0003/s3 | 364s | `sglang_core-0003_s3_*.parquet` | Yes |
| 20 | sglang_core-0003/s4 | 262s | `sglang_core-0003_s4_*.parquet` | Yes |
| 21 | sglang_core-0003/s5 | 353s | `sglang_core-0003_s5_*.parquet` | Yes |

## Items Completed (8/8 done)

| Item | Source | All on HF? |
|------|--------|------------|
| vllm_core-0000 | pre-existing | Yes (8/8) |
| vllm_core-0003 | this run | Yes (8/8) |
| vllm_core-0005 | this run | Yes (8/8) — s1-s3 re-run after restart, now all on HF |
| vllm_core-0008 | this run | Yes (8/8) |
| vllm_core-0009 | this run (s0-s5,s7 on HF; s6 stranded) | 7/8 on HF, 1 stranded |
| vllm_core-0011 | pre-existing (s0-s6) + this run (s7) | Yes (8/8) |
| vllm_core-0013 | pre-existing (s1,s3-s7) + this run (s0,s2) | Yes (8/8) |
| vllm_core-0015 | this run | Yes (8/8) |
| vllm_core-0017 | this run | Yes (8/8) |
| vllm_core-0018 | this run | Yes (8/8) |
| vllm_core-0021 | this run | Yes (8/8) |
| vllm_core-0022 | this run | Yes (8/8) |
| vllm_core-0025 | this run | Yes (8/8) |
| vllm_core-0027 | this run | Yes (8/8) |
| vllm_core-0029 | this run | Yes (8/8) |
| vllm_core-0033 | this run | Yes (8/8) |
| vllm_core-0034 | this run | Yes (8/8) |
| vllm_core-0035 | this run | Yes (8/8) |
| vllm_core-0041 | this run | Yes (8/8) |
| vllm_core-0045 | this run | Yes (8/8) |
| vllm_core-0049 | this run | Yes (8/8) |
| sglang_core-0006 | this run | Yes (8/8) |
| sglang_core-0008 | this run | Yes (8/8) |
| sglang_core-0019 | this run (s0-s1,s3-s7) + pre-existing (s2) | Yes (8/8) |
| sglang_core-0027 | this run (s1-s7) + pre-existing (s0) | Yes (8/8) |
| sglang_core-0033 | this run | Yes (8/8) |
| sglang_core-0047 | this run | Yes (8/8) |
| sglang_core-0070 | this run (s3-s7) + pre-existing (s0-s2) | Yes (8/8) |
| sglang_core-0071 | this run | Yes (8/8) |
| sglang_core-0000 | this run + pre-existing | Yes (8/8) |
| sglang_core-0003 | this run (s0,s2-s7 on HF; s1 stranded) | 7/8 on HF, 1 stranded |
| sglang_core-0005 | this run | Yes (8/8) |
| sglang_core-0017 | pre-existing | Yes (8/8) |
| sglang_core-0059 | pre-existing | Yes (8/8) |
| sglang_core-0065 | pre-existing | Yes (8/8) |
| sglang_core-0068 | pre-existing | Yes (8/8) |

## HF Rate Limit Issue

**Problem:** HF free tier = 128 commits/hour per repo. Three runners share the repo.
**Status: RESOLVED.** HF Pro purchased ~13:11Z. **Zero push failures since.** 10 consecutive pushes confirmed. The 4 stranded samples are from before the Pro upgrade and still need manual push.

### Currently In Progress

| Sample | Started |
|--------|---------|
| vllm_core-0050/s1 | 02:08Z (~13min, longer sample) |
| ~~sglang~~ | **DONE** (21:42Z) |

### New stranded sample
| Sample | Local path | Reason |
|--------|-----------|--------|
| **vllm_core-0009/s6** | `state/runs/vllm/.../2026-03-28_15-01-49_s6/vllm_core-0009/` | HF 429 at 15:04Z (sporadic, despite Pro) |
**Stale local dirs** (Mar 26, already on HF): 7 dirs under `vllm/.../2026-03-26_*/` — safe to ignore.

## Log Files

| Log | Path |
|-----|------|
| vllm | `ISO-Bench/logs/claude_pass_at_k/vllm_20260328_113911.log` |
| sglang | `ISO-Bench/logs/claude_pass_at_k/sglang_20260328_115154.log` |

## Bugs Fixed This Session

1. **Garbage push prevention:** gates on `status=success` + non-empty patch before pushing
2. **Content-aware resume:** validates parquet content, not just filenames
3. **Agent/model filtering:** no cross-agent contamination in resume
4. **Model name fix:** hardcoded `model: "sonnet"` in bench-claude.yaml
5. **Push verification fix:** `list_repo_files()` instead of broken `list_repo_tree().rfilename`

## Incidents

### 14:15-14:31Z — vllm runner crashed
- **What:** vllm `run_pass_at_k` process and tmux window vanished at ~14:15Z. No error in log — clean exit after `vllm_core-0009/s3 Running prepare...`
- **Cause:** Likely `set -euo pipefail` in bash wrapper or tmux window accidentally closed.
- **Impact:** `vllm_core-0009/s3` left as orphaned incomplete run (prompt+task only). Cleaned up.
- **Recovery:** Restarted at 14:31Z in tmux window `claude_vllm` with `--resume`. Re-validates HF shards then picks up from `vllm_core-0009/s3`.
- **Log:** New vllm log file created at restart: `ISO-Bench/logs/claude_pass_at_k/vllm_20260328_143100.log`
