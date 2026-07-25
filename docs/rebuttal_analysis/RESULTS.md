# Rebuttal Analysis — Results (generated 2026-07-25)

All analysis-only items from `docs/NEURIPS_REBUTTAL_DATA_AUDIT.md` executed.
Scripts in `scripts/rebuttal_analysis/`; every table regenerable from canonical data.
Run order: `build_master.py` first (verification gate), then the rest in any order.

| Analysis | Output | Status |
|---|---|---|
| Master quadrant table + Table 5 verification | `master.json` | ✅ 11/12 cells exact (see finding 1) |
| W1: Wilson/bootstrap CIs + McNemar | `stats_tables.md` | ✅ |
| W1 add-on: pass-at-k rollout variance | `pass_at_k_variance.md` | ✅ |
| W3: Q3 Lucky-Win decomposition | `q3_decomposition.md` | ✅ |
| Q4b: Same-only sensitivity | `stats_tables.md` (§3) | ✅ |
| W4: absolute-vs-baseline deltas | `baseline_deltas.md` | ✅ with caveats (finding 4) |
| yX4G: dataset composition | `composition.md` | ✅ |
| yX4G: multi-metric agent-vs-human | `multimetric.md` | ✅ |
| kNyS Q3: trajectory components (4 configs) | `trajectory_components.md` | ✅ |
| kNyS Q2: scaffold × model grid | below | ✅ |
| yX4G/Q4a: judge disagreement (Supabase) | — | ❌ BLOCKED: no credentials anywhere in repo/env |

---

## Key findings (read before drafting the responses)

### 1. Canonical data no longer reproduces one published cell — decide before posting

The master rebuild reproduces the submitted Table 5 exactly for 11/12 cells. **vLLM
OpenHands (Sonnet-4.5) is now Q1=22/Q2=10 (True Success 56.4%) vs published 17/15
(43.6%)** — the 2026-05-05/06 X3 re-bench revised five OH-S45 hard classifications
after the submission froze. The as-submitted per-task state is not recoverable from
submodule history (nearby snapshots give 9 or 26 beats/similar, never the published 21).
Every table below uses canonical data; `stats_tables.md` shows the published-count CI in
brackets for the affected cell. **The team must decide: rebut on published numbers
(consistent with the PDF) or disclose the correction (favorable to OH-S45 — it becomes
the top vLLM agent). Do not silently mix the two.**

### 2. W1 — which claims survive statistics

- Wilson 95% CIs: vLLM cells ±13–16pp wide; SGLang cells ±20–25pp wide (n=15).
- McNemar (exact, paired): on vLLM, Claude Code and OH-S45 significantly beat Codex and
  TRAE-GPT5 (p=0.001–0.021); Sonnet-scaffold vs GPT-5-scaffold ordering largely
  significant. On SGLang, TRAE (both) and Codex significantly beat Claude Code and
  OH-S45 (p=0.003–0.008) — **the headline "rankings invert across codebases" claim
  survives significance testing in both directions.** Claude-Code-vs-OH-S45 and
  most same-model-class pairs are not significant — soften any ordering claims there.
- Pass-at-k rollout CVs (per task, ~8 rollouts): output-throughput CV ≈ 0.4–0.8%
  (well inside the 5% threshold), but **TTFT CV is 7–9% median / up to 19% p90 on
  SGLang** — single-rollout Beats/Similar/Worse on TTFT-classified SGLang tasks can
  flip under resampling; say so and lean on the CIs.

### 3. W3 — Q3 decomposition is a strong result

Of 26 canonical Q3 cases (22 vLLM + 4 SGLang), 12 have lm-eval coverage (legacy agents):
**11/12 preserved GSM8K accuracy (emergent wins); exactly 1 broke it** (`fe66b347`
TRAE-Sonnet, 0.32→0.00 — the paper's Bamba case). This directly supports the planned
reframing: most Lucky Wins are correctness-preserving optimizations at a different
location, not reward hacking. 14 cases (OpenHands vLLM + all SGLang) lack eval data —
scope the table or run ~10 vLLM lm-evals.

### 4. W4 — baseline deltas exist BUT do not tell the story the draft wants

Coverage achieved: vLLM 28/39 tasks with measured baseline (HF parquet + local),
SGLang 14/15. However the **measured** human-vs-baseline improvements are small:
vLLM min −31.9% / median +1.5% / max +99.8% (only 11/28 tasks show >5% human gain);
SGLang median is **negative** (−11.9%, only 2/14 > 5%) — the short isolated benchmark
runs (~50 requests) do not reproduce the PR-claimed speedups for most tasks.
**Do NOT write the draft's planned sentence quantifying "non-trivial verified speedup
by construction" from measured data — it would undercut the benchmark.** Options:
extract PR-*claimed* speedups from the timeline text (separate pass), scope the W4
response to reporting agent-vs-baseline alongside human-vs-baseline without the
inclusion-criterion claim, or explain the measurement-context gap explicitly.
Also: vLLM OpenHands agent-vs-baseline values are cross-context (X3 re-bench vs
earlier parquet baselines) — the +45/+48% OH medians in `baseline_deltas.md` are
provenance artifacts, not real OH superiority; treat as indicative only.

### 5. Q4b — Same-only sensitivity is brutal; disclose with framing

True Success under Same-only drops 8–53pp everywhere (e.g. vLLM Claude Code 46.2→20.5;
SGLang Codex 80.0→26.7; canonical OH-S45 vLLM 56.4→7.7). `related_target` is the
majority correct-target label. Present as a strictest-definition bound and defend
Related-as-correct (same module = right bottleneck neighborhood), citing the κ=0.836
human agreement on the targeting dimension.

### 6. kNyS Q3 — trajectory components quantify real scaffold differences

4-config table (medians, vLLM): TRAE-Sonnet 44 steps / TTF-edit 8s / 502s;
TRAE-GPT5 42 / 80s / 1143s; OH-S45 109 events / 72s / 388s; OH-GPT5 79 / 324s / 836s.
OpenHands terminates cleanly (94–100% explicit finish) vs TRAE-GPT5 36%.
Coarse 6-config table adds Claude Code (207s median, fastest) and Codex (346s).
GPT-5 configs are consistently ~2–4× slower to first edit and overall than Sonnet
configs under the same scaffold — a concrete mechanism for the scaffold/model story.

### 7. kNyS Q2 — scaffold × model grid (True Success %, canonical)

| Scaffold | Sonnet-4.5 vLLM | GPT-5 vLLM | Sonnet-4.5 SGLang | GPT-5 SGLang |
|---|---|---|---|---|
| TRAE | 28.2 | 17.9 | 80.0 | 86.7 |
| OpenHands | 56.4 (pub. 43.6) | 28.2 | 13.3 | 33.3 |
| Claude Code (locked) | 46.2 | — | 26.7 | — |
| Codex CLI (locked) | — | 20.5 | — | 80.0 |

Holding model fixed, scaffold moves vLLM True Success by up to ~28pp; holding scaffold
fixed, model moves it by up to ~28pp (OpenHands) — scaffold and model effects are the
same order of magnitude, and the TRAE row shows the scaffold effect *reverses sign*
across codebases.

### 8. yX4G — composition and multi-metric tables ready

Composition (`composition.md`): median 2 files / ~52–56 edited lines per task; vLLM:
30 serving / 7 latency / 2 throughput, 22 distinct models, top areas `vllm/v1`,
`vllm/model_executor`, `vllm/core`; SGLang: 14 serving / 1 latency, concentrated in
`python/sglang`. No performance_areas field — bottleneck categories would be new labeling.
Multi-metric (`multimetric.md`): agents are modestly worse than human on TPOT/ITL too
(e.g. Codex −7.2% TPOT median), throughput ≈ parity; SGLang has ITL but no TPOT.

### 9. Supabase export — hard blocked

No Supabase URL/key anywhere in the repo or environment (`soft-metrics-review/.env.local`
absent). The judge-disagreement analysis, boundary examples, and the "no disagreement
flips Q1→Q3" claim cannot be produced until whoever ran the review app supplies
credentials. This is the only unexecuted analysis item.

---

## Not run (out of scope for analysis-only)

- 9 missing vLLM baselines, ~10 OpenHands lm-evals, Level-2 TP2 subset, new model
  families — GPU/[RUN] items.
- Soft-metrics judge over the `trae_opensource` sweep — needs `OPENROUTER_API_KEY`
  spend approval (LLM cost, no GPU).
- `camera_ready_edits.tex` application — paper edit, not analysis.
