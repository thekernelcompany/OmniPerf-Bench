# Rebuttal Data Audit — ISO-Bench (submission 2841, NeurIPS 2026 ED track)

**Date:** 2026-07-25
**Reviewers:** utqG (W1–W5, rating 2), kNyS (Q1–Q4, rating 3), yX4G (rating 4)
**Purpose:** fact-check every rebuttal item against what actually exists before committing to anything in the response.
**Inputs:** repo artifacts + submitted PDF (`2841_ISO_Bench_Can_Coding_Agen.pdf`, 32 pp) + draft responses (`iso-bench-rebuttal-drafts.md.pdf`) + HuggingFace inventory (both orgs, authenticated).

Paths below abbreviate `third-party/everything_analysis_data/` as `EAD/`.

---

## TL;DR

- **utqG W4 (baseline deltas): mostly recoverable.** SGLang 15/15 3-way locally; vLLM baselines cover **30/39** after merging HF `claude-code-vllm-benchmarks` (29) with local raw files (+`89a84b0b`); human measurements **39/39** on HF. Remaining [RUN]: 9 vLLM baselines. §4.3's sentence is still ahead of the canonical table (no baseline field) — merge before citing it.
- **kNyS Q3 (trajectory analysis): feasible for 4/6 configs.** TRAE × 2 locally; OpenHands × 2 on HF (GPT-5 54/54; Sonnet-4.5 36/39 vLLM + 5/15 SGLang). Codex CLI and Claude Code have no step logs anywhere — Appendix H's blanket claim still needs softening for Codex.
- **utqG W1: real rollout-variance data exists** — HF pass-at-k results: ~8 benchmarked rollouts/task, full serving metrics, Claude Code + Codex on both repos. Appendix G undersells its own data.
- **W3 count mismatch:** draft says 19 vLLM Q3 cases; Table 5 says 22 (+4 SGLang); lm-eval correctness covers only the 12 legacy-agent cases. OpenHands/SGLang Q3 correctness = small [RUN] or scope honestly.
- **`performance_areas` does not exist** as a stored field (only in Figure 7's illustration; nor does `difficulty`). Composition table must use dataset JSONL fields.
- **E.5–E.7 are single-task anecdotes**, not full benchmark runs. Middle path: soft-metrics over the existing 6-model `trae_opensource` sweep (66 runs, no GPU needed).

---

## Item-by-item verification

### utqG W1 — Bootstrap CIs / significance on Tables 3 & 4 — CONFIRMED (rates), plus real rollout data

Tables 3/4 are success **rates** (True Success = Q1; Hard Success = Q1+Q3) over N=39 vLLM / 15 SGLang tasks, single rollout, no variance reported in the main tables; the checklist (Q7) itself concedes error bars exist only for judge reliability + Appendix G, so this is a straight gap-fill.

- Per-task classifications exist for all 6 agent configs in `EAD/hard_metrics/{vllm,sglang}/hard_metrics.json` and `EAD/soft_metrics/{vllm,sglang}/soft_metrics.json` → Wilson/bootstrap CIs on rates are pure scripting.
- **Test choice:** agents share the same 39/15 tasks — use paired **McNemar** for cross-agent comparisons (not Fisher's exact as drafted); keep Fisher for unpaired contrasts (e.g., vLLM vs SGLang within an agent).
- **Rollout variance:** `Inferencebench/iso-bench-pass-at-k-results` (public) holds **618 all-success benchmark rows with full serving metrics** (TTFT/TPOT/ITL mean/median/p99 + throughputs) at up to **8 rollouts per task**: Claude Code 30 vLLM + 10 SGLang tasks, Codex 29 vLLM + 10 SGLang. Report per-task rollout distributions and rollout-level significance for those 2 agents × 2 repos instead of leaning on Appendix G's pass@2 alone. The sample pool (`pass-at-k-samples`, 49 vLLM/19 SGLang tasks × up to 8 seeds) is larger than the benchmarked subset.
- **Caveat (a):** the main result set is single-run per measurement — zero repeat/std fields in `EAD/benchmark_results/`. Main-table CIs must bootstrap across the task axis.
- **Caveat (b):** for speedup *magnitudes* the effective paired n collapses: non-null `primary_pct` per vLLM cell is 19–40 (of 99 rows / 39 tasks), paired human+agent throughput as low as 14. Stick to rates for the main tables.
- **Preempt:** Table 9 pass@1 = 50.0% vs Table 3's 46.2% is a 30-task-subset artifact — say so explicitly.

### utqG W2 — Level 2 validation — FALLBACK IS THE ONLY HONEST OPTION

Exactly **one** multi-GPU datapoint exists in the entire tree: commit `310aca88` run at TP2 when the spec says TP4, explicitly flagged not-comparable in `docs/HARD_METRICS_OH_GPT5_FINAL.md`. There is no Level-2 suite to extend.

- Reposition Level 2 explicitly as a curated data release — defensible, since `ISO-Bench/ISO-Bench` on HF genuinely contains the Level-2 parquets and the paper already says "all experiments are on Level 1".
- `EAD/paper/camera_ready_edits.tex` contains **five unapplied edits** that reconcile the 66/40 vs 39/15 task-count inconsistency and demote TensorRT-LLM/FlashInfer to an extension pool. Apply them; cite them in the rebuttal as already-drafted camera-ready fixes.

### utqG W3 — Q3 "Lucky Win" decomposition — CONFIRMED for the legacy 4-agent set; OpenHands/SGLang cases lack correctness data

**Count reconciliation (three different numbers in play):**

| Source | vLLM Q3 | SGLang Q3 |
|---|---|---|
| Draft response ("19 Q3 cases total … on vLLM") | 19 | "a handful" |
| Paper Table 5 (authoritative) | **22** (CC 4, OH-S45 4, TRAE-S 2, OH-GPT5 6, Codex 5, TRAE-GPT5 1) | **4** (CC 3, OH-GPT5 1) |
| `q3_commit_vllm.jsonl` (correctness cross-tab) | **12** = exactly the 4 legacy agents (4+2+5+1) | — |

The draft's "19" matches nothing — fix before posting. The consolidated lm-eval files cover exactly the legacy-agent quadrant membership (q1=44, q2=82, q3=12 all match Table 5 sums for CC/TRAE-S/Codex/TRAE-GPT5), confirming **no correctness results exist for the 10 OpenHands vLLM Q3 cases or the 4 SGLang Q3 cases**.

The cross-tab for the covered cases is pre-built:

- `third-party/vllm-lm-eval/consolidated/q3_commit_vllm.jsonl` — 12 Lucky-Win (commit, agent) pairs with hard/target/approach labels.
- `q3_agent_eval_summary.jsonl` + `q2_accuracy_comparison.jsonl` — GSM8K lm-eval accuracies with MATCH/REGRESSION status; join on (commit, agent).
- The headline `fe66b347` 32%→0% accuracy collapse is already in there.

**To present the decomposition over ALL Q3 cases** (which is what the draft promises): needs ~10 GSM8K lm-eval runs for the OpenHands vLLM cases (small, 1×H100) and an SGLang lm-eval flow that currently does not exist. Otherwise scope the table to the 12 covered cases and say so explicitly. Note §3.3.4's claim of validating "all Hard Success cases" is itself only artifact-backed for the legacy agents.

### utqG W4 — Absolute-vs-baseline deltas — 30/39 vLLM + 15/15 SGLang recoverable

§4.3 claims hard metrics were measured "against both the unoptimized baseline and the human solution." Verified coverage:

| Slice | Baseline status |
|---|---|
| SGLang (15 tasks) | **Complete.** `EAD/benchmark_results/sglang/*_isolated.json` has `variants.{baseline, human, <agent>}` for all 6 configs, identical hardware, full serving metrics. (The HF `claude-code-sglang-benchmarks` baseline columns are empty — the local isolated files are the source of truth.) |
| vLLM (39 tasks) | **30 of 39 covered:** `Inferencebench/claude-code-vllm-benchmarks` parquet (99-commit pool) has `baseline_{ttft,tpot,itl}_{mean,median,p99}` + `baseline_throughput` + `baseline_latency_avg` for **29 of the final 39**; local raw files add `89a84b0b` → union 30/39. Truly missing: `19d98e0c, 660470e5, 6e36f4fa, 9474e89b, 9ed82e70, ad8d696a, d7740ea4, e3580537, fc7b8d1e`. Canonical aggregate still has no baseline field — needs a merge script. |
| Human reference (vLLM) | **39/39 measured** in the same HF parquet (`human_*` columns) — the "HF-imported" numbers are real measurements with full metric families, just imported rather than re-run. Provenance note stands; "not benchmarked" does not. |
| OpenHands cells | No OpenHands-specific baseline runs, but baseline is patch-independent — the per-commit baselines above serve all six configs. X3 audit added 13 fresh baseline re-measurements (`x3_2026-05-05/baseline/` in both OH hard-metrics HF repos). |

**Framing:** absolute-vs-baseline deltas are reportable for 15/15 SGLang + 30/39 vLLM. The [RUN] item shrinks to **9 vLLM baseline runs** (1×H100, single runs each) — trivial compared to Level 2 or new models; land it during the window and W4 is fully closed.

**Draft's "minimum/median human speedup — you have this from the PR performance claims":** PR claims are unstructured timeline text in `EAD/reference_data/{vllm,sglang}/human_commits.jsonl` (99+15 records) — extracting a min/median needs a parsing pass over 54 PRs (feasible, small, but it is work, not a lookup). Alternatively compute measured human-vs-baseline from the coverage above (15/15 SGLang + 30/39 vLLM).

### yX4G — Dataset composition table — FEASIBLE, but not from the fields named

`performance_areas` and `difficulty` **do not exist** anywhere — repo-wide grep (src, data, configs, scripts, tools, bench, state, EAD): zero hits for `performance_areas` / `performance_score` / `is_performance_related`. The field appears only in Figure 7's illustration; the Stage-2 raw judge outputs were not persisted. Also an internal paper inconsistency to fix at camera-ready: Figure 7 labels the Stage-2 judge `gemini-3-flash-preview`, Appendix C.1 text says GPT-5-mini.

What `EAD/dataset/{vllm,sglang}.jsonl` (39+15 records) does have per task:

`files_changed`, `stats` (lines/files/hunks), `models`, `has_serving/latency/throughput`, `uses_lm_eval`, `lm_eval_commands`, `affected_paths`, `llm_reason` / `llm_api_reason` (judge rationale), `hardware`, `perf_command`.

A composition table from those is derivable today. Optional: performance-area categories — partial start in `archive/misc/results/reviews/vllm_classification_review.csv` (model-based/kernel-based/misc, 33 commits); extending to 54 is a quick pass but is **new labeling** — do it or don't claim it.

### yX4G / kNyS Q4a — Judge disagreement + boundary examples — BLOCKED ON SUPABASE EXPORT

Raw per-task H1/H2 labels are **not in the repo**. `EAD/scripts/compute_agreement.py` fetches them live from Supabase (needs `NEXT_PUBLIC_SUPABASE_URL` + key, expected at `soft-metrics-review/.env.local`, absent locally). Only aggregate κ made it into the paper (Table 8: mean κ 0.836 targeting / 0.880 approach).

- **Do the export first** — the only analysis-only item with an infrastructure dependency; if credentials are lost, the item silently becomes impossible.
- The draft's sentence "none of the disagreements flip a Q1 outcome to Q3" **cannot be verified without the export**. If the export shows flips, follow the draft's own instruction: report honestly and quantify the Table 3 effect.
- The boundary examples (multi-module Related case, partial-overlap case, annotator-disagreement case) also require the per-item labels.
- Volunteer human–human κ (computed in the script, omitted from the paper) to preempt the "only 2 annotators" follow-up.

### kNyS Q4b — Same-only sensitivity — TRIVIAL, but look before promising

One predicate change in `EAD/scripts/generate_quadrant_findings.py` (line 88: `correct_target = same or related`).

**Heads-up:** across the 234 vLLM soft-metric records, `related_target` = 122 vs `same_target` = 63 — Related is the *majority* correct-target label, so Same-only True Success will drop sharply. Run it, see the numbers, then decide whether to present as a sensitivity bound ("True Success under the strictest definition is X–Y%") rather than a table that eats the headline.

### kNyS Q2 — Scaffold decoupling — CONFIRMED, nothing to run

Table 7 is only the agent-config table; the scaffold argument is textual in §5.5 off Table 4. The draft's grid numbers check out against Table 4 (Sonnet scaffolds 46.2/43.6/28.2 vLLM; OpenHands model swap 43.6→28.2) — safe to post as-is. Stronger reframing available from existing data: a **2×2 factorial** — TRAE and OpenHands each ran with both Sonnet-4.5 and GPT-5 — plus Claude Code (Sonnet-only) and Codex (GPT-5-only) as scaffold-locked points. Present scaffold × model as crossed factors.

### kNyS Q3 — Trajectory component analysis — FEASIBLE FOR 4 OF 6 CONFIGS

Verified state of step-level trajectories:

| Harness | Trajectories | What's computable |
|---|---|---|
| TRAE (GPT-5 + Sonnet) | **39/39 vLLM each** locally (+ `Inferencebench/trae-{sonnet45,gpt5}-trajectories` on HF), per-step ISO timestamps, tool names, token usage | time-to-first-edit, edit counts, tool mix, steps, tokens — all clean |
| OpenHands (GPT-5) | **54/54 on HF** (`Inferencebench/iso-bench-openhands-gpt5-rebuttal`, public): 39 vLLM + 15 SGLang event-log trajectories with per-event timestamps, action types, token metrics | full component analysis |
| OpenHands (Sonnet-4.5) | **36/39 vLLM + 5/15 SGLang on HF** (`…-sonnet45-rebuttal`, private; + 54 stderr logs). Missing vLLM: `vllm_core-0000/0003/0005` | near-full component analysis |
| Claude Code | No step logs anywhere (HF `Claude_code_dump` is May-2025 profile traces, not step logs); heuristic `time_to_first_edit_s` in journal | duration, patch stats |
| Codex CLI | No step logs anywhere; TTFE null | duration, patch stats |

**Offering:** cross-scaffold trajectory-feature analysis on shared tasks for **TRAE × 2 models and OpenHands × 2 models** (4 of 6 configs — this covers the exact TRAE-vs-OpenHands scaffold contrast the reviewer asks about), plus coarse table (duration_s, patch LOC, files changed, patch-generation rate) for all six. Still do **not** claim "full trajectories for the three open-source harnesses" — Codex CLI has none (D.3 itself concedes Codex is hard to instrument), so Appendix H's blanket claim remains falsifiable as written. Pull the OH trajectories from HF before scripting (they are not in the local tree).

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
   - Wilson/bootstrap CIs + McNemar on Tables 3/4; pull pass-at-k rollout distributions from HF.
   - **Supabase export of H1/H2 labels** (credential risk — don't defer).
2. **Day-scale:**
   - Download OH trajectories from HF → 4-config trajectory component table + coarse all-agent table.
   - Baseline merge script (HF `claude-code-vllm-benchmarks` parquet → canonical `hard_metrics.json`).
   - Composition table from dataset JSONL fields.
   - Serving-subset multi-metric (TTFT/TPOT/ITL/throughput) table.
   - Scaffold 2×2 factorial rewrite of §5.5.
   - PR-claims human-speedup extraction (or use measured human-vs-baseline).
   - Apply `camera_ready_edits.tex`.
3. **[RUN] items, cheap → expensive:** 9 missing vLLM baselines (closes W4) → ~10 OpenHands lm-evals (closes W3 fully) → Level-2 TP2 human-reference subset (only if nodes materialize) → new model families (camera-ready commitment otherwise).

---

## Posting notes

The draft's own notes are sound: post utqG first (rating 2 anchors the AC discussion); lead with substance; disclose the Same-only sensitivity proactively rather than letting a reviewer compute it; resolve every bracketed [fill in] with real numbers; don't cite TRT-LLM/FlashInfer to utqG as a scale fix. One addition: **don't repeat Appendix H's "full trajectories for the open-source harnesses" phrasing anywhere** — the Codex part is the one claim a reviewer can falsify by asking for the artifact.

---

## Cross-cutting warnings

- **HF-import artifacts:** some vLLM rows have agent and human numbers both HF-imported with identical values (0% delta artifacts). The 2026-05-05 audit + X3 re-bench cleaned exactly this — build every rebuttal table from the **post-X3 canonical** `hard_metrics.json`, not pre-audit copies. Reviewers may later diff against the released data.
- **Provenance asymmetry:** OpenHands (both models) was added for the paper on top of the 4-agent legacy analysis set; lm-eval correctness doesn't cover it and its trajectories live on HF, not locally. Keep per-table coverage statements exact.
- **Single-run measurements in the main result set** — never phrase main-table CIs as measurement-level. (The pass-at-k HF datasets are the exception: ~8 rollouts/task for Claude Code + Codex.)
- `camera_ready_edits.tex` is drafted but **unapplied** — the compiled PDF still contains the 66/40 vs 39/15 inconsistency a reviewer can flag.

---

## HuggingFace inventory (checked 2026-07-25, authenticated)

Orgs: `Inferencebench` (25 datasets) and `ISO-Bench` (1). Rebuttal-relevant:

| HF dataset | Vis | What it adds |
|---|---|---|
| `Inferencebench/iso-bench-openhands-gpt5-rebuttal` | public | **54/54 OpenHands GPT-5 trajectories** (39 vLLM + 15 SGLang), event logs with timestamps/actions/tokens |
| `Inferencebench/iso-bench-openhands-sonnet45-rebuttal` | private | **36/39 vLLM + 5/15 SGLang OH Sonnet-4.5 trajectories** + 54 stderr logs (missing vllm_core-0000/0003/0005) |
| `Inferencebench/claude-code-vllm-benchmarks` | private | 99-commit × 4-agent benchmark table, 76 cols: `baseline_*` for **29/39 final tasks**, `human_*` for **39/39**, full TTFT/TPOT/ITL/throughput families |
| `Inferencebench/claude-code-sglang-benchmarks` | private | 41-commit table; baseline cols empty (SGLang truth = local `*_isolated.json`) |
| `Inferencebench/iso-bench-pass-at-k-results` | public | **618 benchmark rows, all success**: Claude Code (30 vLLM + 10 SGLang tasks) + Codex (29 + 10), up to 8 rollouts/task, full serving metrics |
| `Inferencebench/pass-at-k-samples` | public | rollout sample pool: 49 vLLM + 19 SGLang tasks × up to 8 seeds |
| `Inferencebench/iso-bench-openhands-{gpt5,sonnet45}-hard-metrics` | private | mirrors of local EAD dirs + `x3_2026-05-05/baseline/` (13 fresh baseline re-measurements each) |
| `Inferencebench/trae-{sonnet45,gpt5}-trajectories` | private | TRAE trajectory mirrors |
| `Inferencebench/iso-bench-trtllm-flashinfer-perf-commits` | private | the 47+52 extension-pool commits (kNyS "limited codebase coverage" response can cite a real artifact) |
| `ISO-Bench/ISO-Bench` | public | released dataset: `data/level{1,2}/{vllm,sglang}/train.parquet` — Level 2 genuinely is released |
| `Inferencebench/Claude_code_dump` | private | May-2025 profile traces; **not** step logs — no help for Claude Code trajectories |

Access: local token at `~/.cache/huggingface/token` reads all of the above.

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
| vLLM baselines (local 19; union with HF = 30/39) | `EAD/benchmark_results/vllm/{tool}/<commit>_baseline_result.json` + HF `claude-code-vllm-benchmarks` parquet |
| Dataset schema (composition table source) | `EAD/dataset/{vllm,sglang}.jsonl` + `EAD/dataset/README.md` |
| Human-agreement script (Supabase-backed) | `EAD/scripts/compute_agreement.py` |
| TRAE trajectories (curated) | `EAD/runs/vllm/{trae_gpt5,trae_sonnet}/<hash>/trajectory.json` |
| Coarse per-run metrics (all agents) | per-task `journal.json` / `run_summary.json` under `EAD/runs/` and `ISO-Bench/state/runs/` |
| Open-model sweep summary | `EAD/analysis/opensource_summary.json` |
| Camera-ready fixes (unapplied) | `EAD/paper/camera_ready_edits.tex` |
| Multi-GPU caveat (310aca88, TP2-vs-TP4) | `docs/HARD_METRICS_OH_GPT5_FINAL.md` |
