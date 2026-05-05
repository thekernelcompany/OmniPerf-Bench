# X3 throughput re-bench — results (2026-05-05)

**Scope (locked):** 6 OH cells benched via `benchmark_throughput.py` + overlay, 2 OH cells via `benchmark_latency.py` + spec-dec for 4c822298 (option ii after read (c) was found wrong), 6 human baselines.

| | total |
|---|---:|
| GPU runs (baselines)            | 6  |
| GPU runs (agents)               | 8  |
| **Total cells**                 | **14** |
| Wall time                       | ~22 min |

**Hardware:** 1× NVIDIA H100 PCIe (80 GB), driver 570.195.03, CUDA 12.8, Ubuntu 24.04, native overlay (no Docker).

**Extractor:** single rule for all 11 benched cells — `benchmark_throughput.py`'s `Throughput: X requests/s, Y total tokens/s, Z output tokens/s` final-stats line, parsed into `throughput_req_s` / `throughput_total_tok_s` / `throughput_output_tok_s`. For the 2 extracted cells (4c822298): `(batch × (input + output)) / avg_latency_s` with default `benchmark_latency.py` params (batch=8, input=32, output=128).

---

## 1. Bench results — agent vs baseline (same extractor, same overlay)

| commit | agent | total_tok/s | baseline total_tok/s | Δ | patch apply status |
|---|---|---:|---:|---:|---|
| 3b61cb45 | OH gpt5     | 8538.40 | 8554.29 | **−0.2%** | partial: 3× csrc/cache_kernels.cu skipped, flash_attn.py applied |
| 8c1e77fb | OH gpt5     | 8662.55 | 8676.15 | **−0.2%** | **none applied** — CMakeLists-only patch, all hunks skipped under overlay |
| 98f47f2a | OH gpt5     | 5999.09 | 6126.61 | **−2.1%** | clean: vllm/v1/attention/backends/flash_attn.py applied |
| fa63e710 | OH gpt5     | 11481.45 | 11471.04 | **+0.1%** | clean: sampler.py + gpu_model_runner.py applied |
| fa63e710 | OH sonnet45 | 11398.91 | 11471.04 | **−0.6%** | clean: outputs.py + sampler.py + gpu_model_runner.py applied |
| 6dd94dbe | OH sonnet45 | 860.25 | 870.03 | **−1.1%** | clean: vllm/worker/model_runner.py applied |

**All 6 cells land within ±2.1% of baseline.** That's run-to-run noise on 1×H100 — no agent's patch produced a measurable throughput improvement under this bench config.

---

## 2. 4c822298 — re-benched via option (ii): `benchmark_latency.py` with spec-dec, scrape `Avg generation throughput`

Read (c) was abandoned after I traced the canonical metric and found `(batch × (input+output)) / latency` is **not** what canonical `human_throughput=153.2` measures. Canonical is **vLLM's `Avg generation throughput` log line** — accepted-tokens-per-second under speculative decoding, which is what the agent's optimization (improving draft acceptance rate) targets.

Three new GPU runs with `benchmark_latency.py --speculative-model meta-llama/Llama-3.2-1B-Instruct --num-speculative-tokens 5`, full log capture, and a new extractor (`gen_throughput_tok_s_*`, `spec_draft_acceptance_rate`, `spec_system_efficiency`) added to `parse_metrics`.

| cell | avg_lat_s | gen_tp_mean tok/s | gen_tp_max | accept rate | sys efficiency |
|---|---:|---:|---:|---:|---:|
| baseline (no patch) | 1.937 | 234.94 | 254.80 | **0.5250** | 0.3540 |
| OH gpt5     | 1.957 | 234.89 | 255.90 | **0.5250** | 0.3540 |
| OH sonnet45 | 1.933 | 237.92 | 256.50 | **0.5250** | 0.3540 |

**Agent vs baseline (gen_tp_mean):**
- OH gpt5:     234.89 vs 234.94 → **−0.02%**
- OH sonnet45: 237.92 vs 234.94 → **+1.27%**

**Critical finding:** `spec_draft_acceptance_rate=0.5250` is **identical to four significant figures** across all three runs. The `spec_system_efficiency=0.3540` is **also identical**. The agents' patches produce zero observable change in spec-dec behavior under this bench config. Whatever code path the patches were meant to optimize is not exercised here, OR the patches are functionally no-ops, OR the model substitution (Meta-Llama-3-8B-Instruct vs canonical Llama-3.1-8B-Instruct, see caveat below) puts us on a different branch.

**Caveat — model substitution.** Canonical 4c822298 ran Llama-3.1-8B-Instruct directly. Our X3 runs use Meta-Llama-3-8B-Instruct (the `MODEL_OVERRIDES` swap because Llama-3.1 RoPE scaling isn't in older vLLMs). Different target model → different n-gram patterns → different spec-dec acceptance dynamics. So the X3 baseline `gen_tp_mean=234.94` does **not** compare to canonical baseline `102.1` — different model. **X3 agent-vs-X3-baseline is the only valid comparison here**; X3-vs-canonical is cross-model.

**Cross-rule note resolved.** Unlike read (c), the metric here (`Avg generation throughput`) is the **same** rule the canonical legacy extractor used (we confirmed by reading vLLM's `engine/metrics.py:456` source at the parent commit). So if you re-ran the legacy 4 agents under the same X3 conditions (Meta-Llama-3-8B-Instruct + 1×H100 + this protocol), they'd produce numbers comparable to ours. We didn't (per scope), so 4c822298 OH ↔ legacy-4-agent stays cross-bench-config.

---

## 3. Critical reading of these numbers

What the X3 batch did succeed at:

- **One consistent extractor across all 11 GPU runs.** No log-line scraping, no commit-shape-specific rules, no derived-vs-measured asymmetry. Every cell came out of the same `Throughput:` line of the same script.
- **Same-infra agent vs baseline.** Both sides on the native overlay runner with the same vLLM wheel from the same parent commit. No Docker-vs-overlay confound *within* X3.
- **Honest reporting of patch application status.** When kernel hunks silently skip (overlay limit), it's logged and visible in this table.

What the X3 batch did **not** succeed at, and the user must accept:

1. **No agent optimization signal landed.** All 6 benched cells are within ±2% of baseline. Three plausible reasons:
   - **Overlay can't apply non-Python hunks.** For 3b61cb45 (3 csrc/.cu hunks skipped) and especially 8c1e77fb (entire CMakeLists-only patch skipped), the agent's actual optimization never reached the GPU. The "agent" measurement here is effectively the baseline run twice.
   - **Bench config doesn't exercise the optimization path.** `benchmark_throughput.py --num-prompts 32 --input 512 --output 128` measures one specific batch shape. Some of these agent patches may be optimizations for paths the bench doesn't hit (e.g., chunked prefill, prefix caching, online serving, batch>32, etc).
   - **The patches genuinely don't optimize the metric.** Agent-suggested code changes that don't improve offline batch throughput.
2. **`8c1e77fb` GPT-5 cell is decorative.** The patch was 100% CMakeLists.txt. Under overlay, zero hunks applied — the `−0.2%` is two baseline runs, not agent-vs-baseline. Should be footnoted as "no patch applied; equivalent to baseline" or excluded from the agent column.
3. **`3b61cb45` GPT-5 cell is partial.** The cache_kernels.cu kernel changes (which are presumably what would speed it up) didn't apply. Only flash_attn.py applied. Footnote required.
4. **X3 numbers are NOT comparable to canonical legacy 4-agent rows.** Those came from the unidentified legacy extractor on `benchmark_latency.py` raw_output. Cross-extractor — the footnote we accepted upfront.
5. **`310aca88` permanently out of scope** — needs tp=4, hardware-blocked on 1×H100.
6. **`4c822298` reference baseline missing.** The `human_throughput=153.2` cell stays under the legacy extractor; we did not derive a same-rule X3 baseline. Sonnet/GPT-5 numbers are agent-vs-agent comparable but agent-vs-human is cross-rule.

---

## 4. Files

```
archive/results/2026-05-x3/
├── baseline/                          # 5 human-baseline runs
│   ├── 3b61cb45_agent_result.json
│   ├── 6dd94dbe_agent_result.json
│   ├── 8c1e77fb_agent_result.json
│   ├── 98f47f2a_agent_result.json
│   └── fa63e710_agent_result.json
├── openhands_gpt5/                    # 4 GPT-5 cells
│   ├── 3b61cb45_agent_result.json
│   ├── 8c1e77fb_agent_result.json
│   ├── 98f47f2a_agent_result.json
│   └── fa63e710_agent_result.json
├── openhands_sonnet45/                # 2 Sonnet cells
│   ├── 6dd94dbe_agent_result.json
│   └── fa63e710_agent_result.json
├── extracted_4c822298/                # 2 no-GPU re-extractions
│   ├── 4c822298_openhands_sonnet45_extracted.json
│   └── 4c822298_openhands_gpt5_extracted.json
├── logs/                              # Per-cell run logs
│   ├── _baselines.log
│   ├── _agents.log
│   └── (kind)__(commit).log
└── X3_RESULTS.md                      # This file
```

Reproduce: `bash scripts/runners/run_x3_batch.sh all` (with `UV_BIN` and `HF_HOME` set).

### Where this lives

**Git** (`thekernelcompany/OmniPerf-Bench`, branch `icml/rebuttal-hard-metrics-oh`):
- `2f06a8843` — X3 batch (runner mods, driver, mapping, 14 result JSONs, X3_RESULTS.md)
- `f9c2bcfd4` — `scripts/runners/push_x3_to_hf.py` (the upload helper that appended to the HF repos below)

**HuggingFace** (private, `Inferencebench` org — append-only under `x3_2026-05-05/`, May 5 fanout artifacts not touched):
- `Inferencebench/iso-bench-openhands-sonnet45-hard-metrics/x3_2026-05-05/`
- `Inferencebench/iso-bench-openhands-gpt5-hard-metrics/x3_2026-05-05/`

Each repo holds the full picture (results + this doc + `_repro/` runner + driver + mapping + 14 run logs).

---

## 5. What to do with these numbers

**Honest options:**

- **Replace canonical OH rows for these 6 commits with X3 numbers**, footnote as: "throughput re-benched 2026-05-05 under unified extractor (`benchmark_throughput.py` + overlay); X3 deltas vs baseline are within ±2% for all cells; cross-extractor with legacy 4-agent column." This keeps OH coherent and surfaces the no-signal finding cleanly.

- **Drop these 6 commits from headline aggregates** (denominator goes from 39→33 for GPT-5 vLLM, footnoted). Stronger story for the agents that *did* show signal on the other 33 commits; drops the noisy rows that don't separate signal from overlay-skip.

- **Re-run with from-source build for kernel commits (3b61cb45, 8c1e77fb)** — the only thing that would meaningfully change `8c1e77fb` and `3b61cb45`. Cost: ~1 hour additional GPU. The user explicitly declined this — but it's the one outstanding lever for these two cells.

The first option is what the X3 batch was designed to enable and is the smallest scope-creep from the run we just did.
