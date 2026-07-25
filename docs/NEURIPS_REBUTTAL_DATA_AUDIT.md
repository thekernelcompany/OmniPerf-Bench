# Rebuttal Data Audit — ISO-Bench (submission 2841)

**Date:** 2026-07-25
**Reviewers:** utqG (W1–W4), kNyS (Q1–Q4), yX4G
**Purpose:** fact-check every "analysis-only" rebuttal item against what actually exists in the repo before committing to anything in the response.

Paths below abbreviate `third-party/everything_analysis_data/` as `EAD/`.

---

## TL;DR

- **utqG W4 (baseline deltas) is the trap.** Clean 3-way baseline/human/agent data exists **only for SGLang (15/15)**. vLLM has real serving baselines for **19 of 39 commits**, the canonical vLLM aggregate has **no baseline field**, and all OpenHands cells have **no baseline at all**. §4.3's sentence ("against both the unoptimized baseline and the human solution") overstates coverage as written — do not restate it at full strength.
- **kNyS Q3 (trajectory analysis): only TRAE has full trajectories.** OpenHands main sets saved none (5 smoke runs only); Claude Code / Codex have no step logs. Cross-harness time-to-first-edit cannot be promised.
- **`performance_areas` does not exist** as a dataset field (nor does `difficulty`). The composition table must be built from the fields that do exist (see yX4G below).
- **E.5–E.7 are single-task anecdotes**, not full benchmark runs on open models. That fallback sentence for kNyS Q1 must not be written as-is. A no-GPU middle path exists (soft-metrics over the existing `trae_opensource` sweep).
- Everything else in the analysis-only list is **confirmed**, and W3 (Lucky Win decomposition) is even more turnkey than assumed — the cross-tab files already exist.

---

## Item-by-item verification

### utqG W1 — Bootstrap CIs / significance on Tables 3 & 4 — CONFIRMED (rates only)

Tables 3/4 are success **rates** (True Success = Q1; Hard Success = Q1+Q3) over N=39 vLLM / 15 SGLang tasks, single rollout, no variance reported anywhere in the main tables. Variance appears only in Appendix G (Table 9), for 2 of 6 agents on a 30-task subset.

- Per-task classifications exist for all 6 agent configs in `EAD/hard_metrics/{vllm,sglang}/hard_metrics.json` and `EAD/soft_metrics/{vllm,sglang}/soft_metrics.json` → Wilson/bootstrap CIs on rates + McNemar paired tests between agents are pure scripting.
- **Caveat (a):** every benchmark measurement is a **single run** — zero repeat/std fields anywhere in `EAD/benchmark_results/`. CIs must bootstrap across the task axis; never imply within-task replication.
- **Caveat (b):** for speedup *magnitudes* the effective paired n collapses: non-null `primary_pct` per vLLM cell is 19–40 (of 99 rows / 39 tasks), paired human+agent throughput as low as 14. Stick to rates.
- **Preempt:** Table 9 pass@1 = 50.0% vs Table 3's 46.2% is a 30-task-subset artifact — say so explicitly.

### utqG W2 — Level 2 validation — FALLBACK IS THE ONLY HONEST OPTION

Exactly **one** multi-GPU datapoint exists in the entire tree: commit `310aca88` run at TP2 when the spec says TP4, explicitly flagged not-comparable in `docs/HARD_METRICS_OH_GPT5_FINAL.md`. There is no Level-2 suite to extend.

- Reposition Level 2 explicitly as a curated data release (the paper already says "all experiments are on Level 1").
- `EAD/paper/camera_ready_edits.tex` contains **five unapplied edits** that reconcile the 66/40 vs 39/15 task-count inconsistency and demote TensorRT-LLM/FlashInfer to an extension pool. Apply them; cite them in the rebuttal as already-drafted camera-ready fixes.

### utqG W3 — Q3 "Lucky Win" decomposition — CONFIRMED, turnkey (strongest item)

The cross-tab is essentially pre-built:

- `third-party/vllm-lm-eval/consolidated/q3_commit_vllm.jsonl` — the 12 Lucky-Win (commit, agent) pairs with hard/target/approach labels.
- `q3_agent_eval_summary.jsonl` + `q2_accuracy_comparison.jsonl` — GSM8K lm-eval accuracies with MATCH/REGRESSION status; join on (commit, agent).
- The headline `fe66b347` 32%→0% accuracy collapse is already in there.

**Scope caveats for the text:** correctness coverage is vLLM-only and the 4-agent legacy set (no OpenHands, no SGLang lm-eval); correctness = GSM8K exact-match. Half a day including prose.

### utqG W4 — Absolute-vs-baseline deltas — PARTIALLY EXISTS (the critical check)

§4.3 claims hard metrics were measured "against both the unoptimized baseline and the human solution." The data half-supports this:

| Slice | Baseline status |
|---|---|
| SGLang (15 tasks) | **Complete.** `EAD/benchmark_results/sglang/*_isolated.json` has `variants.{baseline, human, <agent>}` for all 6 configs, identical hardware, full serving metrics. |
| vLLM (39 tasks) | **19 commits** with usable serving baselines (`*_baseline_result.json`); 42 baseline files empty/failed; canonical aggregate has **no baseline field**; human numbers largely HF-imported, not freshly benchmarked. |
| OpenHands cells | **No baseline anywhere.** Agent-only runs; human reference = merged HF number. |

**Framing:** present the complete SGLang table + the 19-commit vLLM subset now. The one cheap [RUN] item worth compute: re-run the ~20 missing vLLM baselines (1×H100, single runs each) — far cheaper than Level 2 or new models.

### yX4G — Dataset composition table — FEASIBLE, but not from the fields named

`performance_areas` and `difficulty` **do not exist** anywhere (repo-wide grep). What `EAD/dataset/{vllm,sglang}.jsonl` (39+15 records) does have per task:

`files_changed`, `stats` (lines/files/hunks), `models`, `has_serving/latency/throughput`, `uses_lm_eval`, `lm_eval_commands`, `affected_paths`, `llm_reason` / `llm_api_reason` (GPT-5-mini judge rationale), `hardware`, `perf_command`.

A composition table from those is derivable today. Optional: performance-area categories — partial start in `archive/misc/results/reviews/vllm_classification_review.csv` (model-based/kernel-based/misc, 33 commits); extending to 54 is a quick pass but is **new labeling** — do it or don't claim it.

### yX4G / kNyS Q4a — Judge disagreement + boundary examples — BLOCKED ON SUPABASE EXPORT

Raw per-task H1/H2 labels are **not in the repo**. `EAD/scripts/compute_agreement.py` fetches them live from Supabase (needs `NEXT_PUBLIC_SUPABASE_URL` + key, expected at `soft-metrics-review/.env.local`, absent locally). Only aggregate κ made it into the paper (Table 8: mean κ 0.836 targeting / 0.880 approach).

- **Do the export first** — the only analysis-only item with an infrastructure dependency; if credentials are lost, the item silently becomes impossible.
- Volunteer human–human κ (computed in the script, omitted from the paper) to preempt the "only 2 annotators" follow-up.

### kNyS Q4b — Same-only sensitivity — TRIVIAL, but look before promising

One predicate change in `EAD/scripts/generate_quadrant_findings.py` (line 88: `correct_target = same or related`).

**Heads-up:** across the 234 vLLM soft-metric records, `related_target` = 122 vs `same_target` = 63 — Related is the *majority* correct-target label, so Same-only True Success will drop sharply. Run it, see the numbers, then decide whether to present as a sensitivity bound ("True Success under the strictest definition is X–Y%") rather than a table that eats the headline.

### kNyS Q2 — Scaffold decoupling — CONFIRMED, nothing to run

Table 7 is only the agent-config table; the scaffold argument is textual in §5.5 off Table 4. Stronger reframing available from existing data: a **2×2 factorial** — TRAE and OpenHands each ran with both Sonnet-4.5 and GPT-5 — plus Claude Code (Sonnet-only) and Codex (GPT-5-only) as scaffold-locked points. Present scaffold × model as crossed factors.

### kNyS Q3 — Trajectory component analysis — ORIGINAL CLAIM WRONG; TRAE-only

Verified state of step-level trajectories:

| Harness | Trajectories | What's computable |
|---|---|---|
| TRAE (GPT-5 + Sonnet) | **39/39 vLLM each**, per-step ISO timestamps, `str_replace_based_edit_tool` vs `bash` tool names, per-step token usage | time-to-first-edit, edit counts, tool mix, steps, tokens — all clean |
| OpenHands (both models, main sets) | **None** (only ~5 smoke-run trajectories); `journal.metrics.time_to_first_edit_s` null in 0/39 | `duration_s`, patch stats only |
| Claude Code | No step logs; heuristic `time_to_first_edit_s` in journal (unverifiable) | duration, patch stats |
| Codex CLI | No step logs; TTFE null | duration, patch stats |

**Honest offering:** full component table for TRAE × 2 models; coarse table (duration_s, patch LOC, files changed, patch-generation rate — all in `journal.json`/`run_summary.json`) for all six configs; explicit statement that step logging wasn't enabled for the other harnesses. Do **not** promise cross-harness time-to-first-edit.

### kNyS Q1 — New model families — EXPENSIVE; fallback wording must change

E.5–E.7 are **three qualitative single-task case studies** (MiniMax-M2.1, GPT-OSS-120B, GLM-4.7), all on the same vLLM scheduler task — *not* full benchmark runs. Do not write the "E.5–E.7 were full runs" fallback; a reviewer reading the appendix will catch it.

**No-GPU middle path:** `EAD/analysis/opensource_summary.json` shows an existing `trae_opensource` sweep — 6 open models (GLM-4.7, GLM-4.7-Flash, Kimi-K2-Thinking, MiniMax-M2.1, GPT-OSS-120B, Qwen3-Coder), 66 runs with saved patches/journals (many errored). Run the existing soft-metrics judge over those saved patches → patch-generation rates + bottleneck-targeting distributions for open models. LLM-judge cost only; turns anecdotes into a quantitative, clearly-labeled soft-metrics-only table. Full hard-metrics on new frontier families stays a camera-ready commitment.

### yX4G — Composite throughput/TPOT — PARTIALLY FREE, known boundaries

Serving-mode raw JSONs capture the full `benchmark_serving.py` field set (TTFT/TPOT/ITL mean/median/p99 + request/output/total throughput): 202 vLLM result files + all 15 SGLang tasks. But:

- ~30 vLLM tasks are latency-mode only (`avg/p10–p99 latency` scalars); some are throughput-only; 417 vLLM result files have empty metrics (crashed/intermediate).
- SGLang records ITL but has **no TPOT field**.

A multi-metric table for the serving subset is free; a uniform composite across all 54 tasks is **not** derivable from logs and must not be promised. Frame heterogeneity as inherent to using each PR's original benchmark command (defensible; the paper only claims TTFT+throughput).

---

## Execution order

1. **Tonight-cheap, do first:**
   - Q3 cross-tab (join `q3_commit_vllm.jsonl` × `q3_agent_eval_summary.jsonl`).
   - Same-only quadrant recompute (inspect result before promising).
   - Wilson/bootstrap CIs + McNemar on Tables 3/4.
   - **Supabase export of H1/H2 labels** (credential risk — don't defer).
2. **Day-scale:**
   - Composition table from dataset JSONL fields.
   - TRAE trajectory component table + coarse all-agent table.
   - Serving-subset multi-metric (TTFT/TPOT/ITL/throughput) table.
   - Scaffold 2×2 factorial rewrite of §5.5.
   - Apply `camera_ready_edits.tex`.
3. **Only [RUN] item worth compute:** ~20 missing vLLM baselines (1×H100, single runs) to complete W4.
   Level 2 → data-release repositioning. New models → soft-metrics-only open-model table + camera-ready commitment.

---

## Cross-cutting warnings

- **HF-import artifacts:** some vLLM rows have agent and human numbers both HF-imported with identical values (0% delta artifacts). The 2026-05-05 audit + X3 re-bench cleaned exactly this — build every rebuttal table from the **post-X3 canonical** `hard_metrics.json`, not pre-audit copies. Reviewers may later diff against the released data.
- **Provenance asymmetry:** OpenHands (both models) was added for the paper on top of the 4-agent legacy analysis set; lm-eval correctness and trajectories don't cover it. Keep per-table coverage statements exact.
- **Single-run measurements everywhere** — never phrase CIs as measurement-level.
- `camera_ready_edits.tex` is drafted but **unapplied** — the compiled PDF still contains the 66/40 vs 39/15 inconsistency a reviewer can flag.

---

## Key artifact index

| Item | Path |
|---|---|
| Hard classifications per (commit, agent) | `EAD/hard_metrics/{vllm,sglang}/hard_metrics.json` |
| Soft dims (targeting + approach) per (commit, agent) | `EAD/soft_metrics/{vllm,sglang}/soft_metrics.json` |
| Quadrant generator (Same-only edit @ line 88) | `EAD/scripts/generate_quadrant_findings.py` |
| Quadrant findings (rendered) | `EAD/analysis/{vllm,sglang}/commit_quadrant_findings.md` |
| Lucky-Win pairs + correctness | `third-party/vllm-lm-eval/consolidated/q{1,2,3}_commit_vllm.jsonl`, `q3_agent_eval_summary.jsonl`, `q2_accuracy_comparison.jsonl` |
| SGLang 3-way benchmark data | `EAD/benchmark_results/sglang/<commit>_isolated.json` |
| vLLM baselines (19 usable) | `EAD/benchmark_results/vllm/{tool}/<commit>_baseline_result.json` |
| Dataset schema (composition table source) | `EAD/dataset/{vllm,sglang}.jsonl` + `EAD/dataset/README.md` |
| Human-agreement script (Supabase-backed) | `EAD/scripts/compute_agreement.py` |
| TRAE trajectories (curated) | `EAD/runs/vllm/{trae_gpt5,trae_sonnet}/<hash>/trajectory.json` |
| Coarse per-run metrics (all agents) | per-task `journal.json` / `run_summary.json` under `EAD/runs/` and `ISO-Bench/state/runs/` |
| Open-model sweep summary | `EAD/analysis/opensource_summary.json` |
| Camera-ready fixes (unapplied) | `EAD/paper/camera_ready_edits.tex` |
| Multi-GPU caveat (310aca88, TP2-vs-TP4) | `docs/HARD_METRICS_OH_GPT5_FINAL.md` |
