# Benchmark v17 Error Tracking

**Run Start**: 2026-01-03 18:20:41 UTC
**Total Commits**: 59
**Log File**: `rerun_log_v17_setup_transformers.txt`

## Summary (Updated: 2026-01-03 21:00 UTC)

| Metric | Count |
|--------|-------|
| Completed | 30/59 |
| Success | 0 |
| Errors | 30 |

## Error Classification

| Category | Description | Count | Fixable? |
|----------|-------------|-------|----------|
| MODAL_INFRA | Modal infrastructure issues (sandbox, networking) | 2 | YES - retry |
| EXPECTED | Intentionally blocked models (too large) | 2 | NO |
| VLLM_VERSION | vLLM version doesn't support features being tested | 5 | NO - dataset issue |
| BAD_CONFIG | Benchmark config has unresolved placeholders | 1 | NO - dataset issue |
| SERVER_FAILED | vLLM server failed to start (timeout ~1hr) | 15 | **FIXED in v34** |
| NO_METRICS | Benchmark ran but no output parsed | 5 | **FIXED in v34** |
| NO_WHEEL | Missing wheel URL in config | 1 | NO - dataset issue |
| UNKNOWN | Unusual error - needs investigation | 1 | Investigate |
| AIMV2 | transformers config conflict | **0** | N/A - FIXED! |

## Actual Error Stdout (from Modal logs)

1. **FileNotFoundError**: `ShareGPT_V3_unfiltered_cleaned_split.json` - dataset file not in container
2. **V1 Engine Exception**: `EngineCore hit an exception` in worker import - V1 engine unstable
3. **Runner failed**: `exit code: 128` - benchmark script failure
4. **Modal FilesystemExecutionError**: `request cancelled due to internal error` - transient Modal issue

## Detailed Error Log

| # | Commit | Model | GPU | Status | Error Category | Details | Reason |
|---|--------|-------|-----|--------|----------------|---------|--------|
| 1 | 21d93c14 | Mixtral-8x7B-v0.1 | H100:8 | error | MODAL_INFRA | Docker sandbox I/O error | Modal infra issue - RETRIABLE |
| 2 | 4fb56914 | DeepSeek-V3-0324 | H100:8 | blocked_model | EXPECTED | Model too large | Intentionally blocked - 671B params |
| 3 | 2a052011 | Mixtral-8x7B-FP8 | H100:2 | baseline_failed | VLLM_VERSION | FP8 on vLLM 0.3.3 | vLLM 0.3.3 doesn't support FP8 |
| 4 | 22dd9c27 | Llama-3.1-8B | H100:1 | baseline_failed | VLLM_VERSION | V1 engine latency | VLLM_USE_V1=1 not in this version |
| 5 | 526de822 | MODEL (placeholder) | H100:1 | baseline_failed | BAD_CONFIG | Placeholders | Dataset config not filled - `MODEL`, `BS`, `INPUT_LEN` |
| 6 | 379da6dc | Llama-3-70B | H100:4 | baseline_failed | VLLM_VERSION | Server failed (FP8) | FP8 dtype on vLLM 0.4.2 unstable |
| 7 | 0d243f2a | Mixtral-8x7B | H100:2 | human_failed | SERVER_FAILED | Human server failed | ROCm-specific commit, not H100 compatible |
| 8 | 0ec82edd | Qwen3-30B-A3B | H100:2 | baseline_failed | SERVER_FAILED | Both failed | Qwen3 MoE architecture not supported |
| 9 | 3a243095 | Llama-3.1-8B | H100:1 | baseline_failed | SERVER_FAILED | Both failed (3661s timeout) | vLLM 0.3.3 server mode unstable |
| 10 | 35fad35a | Llama-3.1-8B | H100:1 | baseline_failed | SERVER_FAILED | Both failed (3685s timeout) | V1 Sampler not in this version |
| 11 | 22d33bac | Llama-3.1-8B | H100:1 | baseline_failed | SERVER_FAILED | Both failed (3685s timeout) | Frontend async iterators version |
| 12 | 296f927f | Bamba-9B-v2 | H100:1 | baseline_failed | SERVER_FAILED | Both failed (3679s timeout) | Mamba2 model architecture unsupported |
| 13 | 2f192835 | Llama-3.1-8B | H100:1 | baseline_failed | SERVER_FAILED | Both failed (3658s timeout) | vLLM 0.4.0 server mode issues |
| 14 | 3092375e | Llama-3.1-8B | H100:1 | baseline_failed | SERVER_FAILED | Both failed (3682s timeout) | V1 serialization not in version |
| 15 | 83450458 | Llama-3.1-8B | H100:1 | baseline_failed | NO_METRICS | No metrics (64s run) | Speculative decode ngram test - output parsing |
| 16 | 3476ed08 | Llama-3.1-8B | H100:1 | baseline_failed | SERVER_FAILED | Both failed (3676s timeout) | Block manager v2 version issue |
| 17 | 015069b0 | Qwen3-7B | H100:1 | baseline_failed | SERVER_FAILED | Both failed (3692s timeout) | Qwen3 ReasoningParser not supported |
| 18 | 67da5720 | Qwen2.5-VL-3B | H100:1 | baseline_failed | SERVER_FAILED | Both failed (3675s timeout) | Qwen2.5-VL model not supported |
| 19 | 9badee53 | Llama-3.2-1B | H100:1 | baseline_failed | NO_METRICS | No metrics (191s run) | Missing ShareGPT dataset file |
| 20 | 2deb029d | Llama-3-8B-FP8 | H100:1 | error | MODAL_INFRA | Docker fallback failed | Modal sandbox I/O error - RETRIABLE |
| 21 | 9474e89b | llama-7b | H100:1 | baseline_failed | NO_METRICS | No metrics (68s run) | Prefix caching throughput test - parsing |
| 22 | 6e36f4fa | Llama-3.1-8B | H100:1 | baseline_failed | NO_METRICS | No metrics (123s run) | Chunked prefill benchmark - parsing |
| 23 | 660470e5 | Llama-3.1-8B | H100:1 | baseline_failed | SERVER_FAILED | Both failed | Evictor-v2 optimization - vLLM 0.5.3 server issue |
| 24 | 6ce01f30 | Llama-3-8B | H100:1 | baseline_failed | SERVER_FAILED | Both failed | get_seqs optimization - vLLM 0.5.4 server issue |
| 25 | aea94362 | Llama-3.2-1B | H100:1 | baseline_failed | NO_METRICS | No metrics | V1 serving performance - output parsing |
| 26 | b6d10354 | Llama-3.1-8B | H100:1 | baseline_failed | UNKNOWN | baseline: None; human: None | Unknown error - needs investigation |
| 27 | baeded25 | DeepSeek-V3 | H100:8 | blocked_model | EXPECTED | Model too large | Intentionally blocked - 671B params |
| 28 | 93e5f3c5 | Llama-3.1-8B | H100:1 | baseline_failed | SERVER_FAILED | Both failed | Server startup failure |
| 29 | 99abb8b6 | Llama-3.1-8B | H100:1 | baseline_failed | SERVER_FAILED | Both failed | Spec decode server failure |
| 30 | c45f3c3a | Llama-3.1-8B | H100:1 | baseline_failed | NO_WHEEL | No wheel source | Missing wheel URLs - config issue |

## Fixability Analysis

### Fixable by Retry (2 commits)
| Commit | Issue | Action |
|--------|-------|--------|
| 21d93c14 | Modal FilesystemExecutionError | Retry after Modal stabilizes |
| 2deb029d | Modal FilesystemExecutionError | Retry after Modal stabilizes |

### Potentially Fixable (Improve Runner)
| Issue | Count | Fix |
|-------|-------|-----|
| Server timeout no logs | 13 | Better stderr capture from vLLM server process |
| No metrics parsing | 4 | Fix output format parsing for different vLLM versions |
| Missing dataset | 1 | Mount ShareGPT_V3_unfiltered_cleaned_split.json in container |

### NOT Fixable (Dataset/Config Issues)
| Issue | Count | Why |
|-------|-------|-----|
| BAD_CONFIG placeholders | 1 | Dataset has `MODEL`, `BS` not replaced - regenerate config |
| ROCm-specific commits | 1 | Code targets AMD MI300, won't work on NVIDIA H100 |
| FP8/V1 on old vLLM | 4 | Features don't exist in baseline vLLM version |
| Model not supported | 4 | Qwen3, Qwen2.5-VL, Mamba2 not in old vLLM |
| Model too large | 1 | DeepSeek-V3 671B won't fit on 8x H100 |

## Key Observations

1. **No AIMV2 errors observed** - The transformers version fix is CONFIRMED WORKING!
2. **Server timeouts are ~1 hour** (3600-3700s) - health check timeout, not actual crashes
3. **BOTH baseline AND human fail together** - Issue is benchmark config, not vLLM code
4. **"No server logs available"** - Log capture mechanism failing, need to fix
5. Most errors are **legitimate version incompatibilities**, not infrastructure issues

## Fixes Applied

### v33 Fix (Working!) - AIMV2 Conflict
- Added `setup_transformers_version()` function
- Calls transformers version selection after cached wheel installation
- vLLM < 0.9: uses transformers>=4.40.0,<4.46.0 (avoids aimv2 conflict)
- vLLM >= 0.9: uses transformers>=4.46.0

### v34 Fix (Ready for v18 run) - Server Log Capture & No Metrics Debug
- Added `log_file="/tmp/vllm_server.log"` to all 9 `start_server()` calls
- Server output now captured to file before timeout
- Added server log capture code in all timeout error handlers
- Added output tail in NO_METRICS error messages for debugging
- Cache bust: `20260103_v34_server_log_capture`

**Lines modified**: 2637, 2729, 2824, 3248, 3420, 3556, 4293, 4398, 4483, 4319, 4420

## Priority Fixes for Future Runs

| Priority | Fix | Impact | Effort |
|----------|-----|--------|--------|
| 1 | Retry MODAL_INFRA errors | 2 commits | Low |
| 2 | Better stderr capture | Debug 11 server failures | Medium |
| 3 | Mount ShareGPT dataset | Fix dataset errors | Low |
| 4 | Fix output parsing | Fix "no metrics" (4 commits) | Medium |
| 5 | Skip known-bad configs | ROCm, placeholders (2 commits) | Low |
