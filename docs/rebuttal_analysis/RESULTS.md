# Rebuttal Analysis — Results (generated 2026-07-25)

This document collects everything we ran for the rebuttal, in plain language, with
the technical details and the caveats spelled out. Every number here comes from a
script in `scripts/rebuttal_analysis/` and can be regenerated. Run
`build_master.py` first — it rebuilds the paper's quadrant table from the raw data
and refuses to continue if the rebuild stops matching the paper. The other scripts
can run in any order after that.

**Terms used throughout:**

- **Quadrants:** every (task, agent) pair lands in Q1–Q4. Q1 = right target and good
  performance ("True Success"). Q2 = right target, bad performance. Q3 = good
  performance but wrong target ("Lucky Win"). Q4 = wrong target and bad performance.
  "Hard Success" = Q1 + Q3 (what performance numbers alone would call success).
- **Canonical vs published:** "published" means the numbers in the submitted PDF.
  "Canonical" means the numbers you get today from the current data files, which were
  partially re-benchmarked *after* submission (the "X3 re-bench", 2026-05-05/06).
  These two disagree in exactly one cell — see Finding 1.
- **Legacy agents:** the four configs that existed before OpenHands was added —
  Claude Code, Codex CLI, TRAE (Sonnet), TRAE (GPT-5). Some data (like correctness
  tests) only exists for these four.

---

## Status: what is done vs what is still to do

| Reviewer item | What the review asks | Done (artifact) | Still to do |
|---|---|---|---|
| utqG W1 — scale / statistics | Confidence intervals + significance tests on Tables 3/4 | ✅ `stats_tables.md`, `pass_at_k_variance.md` | Write the response text; decide canonical-vs-published for one cell (Finding 1) |
| utqG W2 — Level 2 unevaluated | Validate Level 2 or reposition it | ✅ Confirmed Level 2 was never run; fallback framing ready | Optional GPU run: a few Level-2 TP2 tasks with the human patch; apply `camera_ready_edits.tex` |
| utqG W3 — two kinds of Lucky Win | Split Q3 into "broke correctness" vs "legitimate alternative" | ✅ `q3_decomposition.md` — 11 of 12 testable cases kept correctness | GPU run (~10 tasks) to cover the OpenHands cases; SGLang cannot be covered (no correctness setup exists) |
| utqG W4 — baseline-relative numbers | Show absolute gains over the unoptimized code, not just vs the human patch | ✅ `baseline_deltas.md` — 28/39 vLLM + 14/15 SGLang tasks covered | GPU run: 9 missing vLLM baselines; **do not** quote the measured medians as proof of "non-trivial speedups by construction" (Finding 4) |
| utqG W5 — contamination | Mitigation + durability plan | ✅ Nothing to compute; the draft's argument holds | Response text only |
| yX4G — metric completeness | Report TPOT/ITL/throughput alongside TTFT | ✅ `multimetric.md` | Response text only |
| yX4G — judge reliability | Show the judge-vs-human disagreements and boundary cases | ❌ **Blocked** | Needs the Supabase export — the raw human labels are not in the repo and no credentials exist locally |
| yX4G — representativeness | Describe what kinds of tasks the dataset contains | ✅ `composition.md` | Optional: hand-label bottleneck categories (~54 commits) |
| kNyS — limited scale | Statistical rigor at n=39/15 | ✅ Same as W1 | Response text only |
| kNyS — codebase coverage | Why only vLLM/SGLang | ✅ Nothing to compute; the TRT-LLM/FlashInfer task pool exists on HuggingFace and can be cited | Response text only |
| kNyS Q1 — model diversity | Results for more model families | ❌ Not run | Cheapest path: run the LLM judge over the 66 existing open-model runs (API cost only, no GPU). Full benchmark runs on new families = expensive, or promise for camera-ready. Also fix the draft's wrong claim that appendices E.5–E.7 were full runs |
| kNyS Q2 — scaffold vs model | Separate the effect of the scaffold from the model | ✅ 2×2 grid (Finding 7) + significance tests | Rewrite of paper §5.5 |
| kNyS Q3 — scaffold mechanisms | Which scaffold behaviors drive the gap | ✅ `trajectory_components.md` — step-level analysis for 4 of 6 configs | Response text; never claim step logs exist for Codex or Claude Code |
| kNyS Q4 — judge boundary cases | Worked examples + a stricter-definition check | ✅ Strict-definition check done (Finding 5) | Worked examples blocked on the same Supabase export |

**Two decisions must be made before posting anything:**
1. Which numbers to stand on for the one cell where current data disagrees with the
   PDF (Finding 1). Pick one; never mix.
2. How to frame W4, since the measured baseline numbers do not support the draft's
   planned claim (Finding 4).

---

## Finding 1 — The current data no longer matches one cell of the published paper

We rebuilt the paper's quadrant table (Table 5) from the raw soft-metric and
hard-metric files. For 11 of the 12 (project × agent) cells, the rebuild matches the
PDF exactly. The exception: **vLLM OpenHands (Sonnet-4.5)**.

- Published: Q1=17, Q2=15 → True Success 43.6%.
- Current data: Q1=22, Q2=10 → True Success 56.4%.

What happened: after the paper was submitted, the team re-ran a batch of benchmarks
(the X3 re-bench, 2026-05-05/06) and five of this agent's results changed from
"worse than human" to "beats/similar". The data files were updated in place, so the
exact per-task state the paper was computed from no longer exists — we checked the
git history of the data submodule and the nearby snapshots contain 9 or 26
"beats/similar" rows for this agent, never the 21 the paper implies.

**Caveats and consequences:**
- Every table in this analysis uses the *current* (canonical) data. Where it matters,
  the published number is shown next to it.
- Under current data, OpenHands (Sonnet-4.5) becomes the *best* vLLM agent, ahead of
  Claude Code. That changes the paper's story slightly (in a direction favorable to
  the benchmark — the correction is upward).
- The rebuttal must either stick to the published numbers everywhere, or openly say
  "we re-benchmarked and this cell improved". Mixing the two silently is the one
  thing a careful reviewer could catch and would look bad.

## Finding 2 — What survives statistical testing (utqG W1, kNyS "scale")

We added three kinds of statistics (`stats_tables.md`, `pass_at_k_variance.md`):

**Confidence intervals on the success rates.** Wilson 95% intervals plus a
bootstrap check (10,000 resamples over tasks). On vLLM (39 tasks) the intervals are
roughly ±13–16 percentage points; on SGLang (15 tasks) roughly ±20–25 points. So
point estimates like "80.0%" on SGLang really mean "somewhere between ~55% and ~93%".
Any sentence in the paper that ranks agents on SGLang by a few points is not
supported; sentences about large gaps are.

**Paired significance tests.** Because every agent ran the same tasks, we used
McNemar's exact test, which only looks at tasks where two agents disagree (one
succeeded, the other failed). Results:
- On vLLM: Claude Code and OpenHands (Sonnet-4.5) are significantly better than
  Codex CLI and TRAE (GPT-5) (p between 0.001 and 0.021). The general pattern
  "Sonnet-based configs beat GPT-5-based configs on vLLM" holds up.
- On SGLang: TRAE (both models) and Codex CLI are significantly better than Claude
  Code and OpenHands (Sonnet-4.5) (p between 0.003 and 0.008).
- Together these mean the paper's headline claim — the ranking *inverts* between
  codebases — survives testing in both directions.
- Not significant: Claude Code vs OpenHands (Sonnet-4.5) on vLLM, TRAE vs Codex on
  SGLang, and most pairs that share a model. Ordering claims between those specific
  pairs should be softened to "comparable".

**How noisy is a single measurement?** The paper classifies each patch from one
benchmark run. Using the pass-at-k dataset on HuggingFace (about 8 independent
rollouts per task for Claude Code and Codex, each benchmarked), we measured how much
the same task's numbers vary run to run (coefficient of variation = standard
deviation / mean):
- Throughput: 0.4–0.8% typical variation. The paper's ±5% classification threshold
  is comfortably wider than the noise. Throughput-based classifications are stable.
- TTFT: 7–9% typical variation on SGLang, up to 19% for the noisiest tasks. This is
  *larger* than the ±5% threshold — so a single-run Beats/Similar/Worse label on a
  TTFT-classified SGLang task can flip if you re-run it.

**Caveats:** these rollout numbers mix two sources of variation (the agent producing
a different patch each rollout, and benchmark noise) — they are an upper bound on
benchmark noise, not a clean measurement of it. Also, all the *main* tables in the
paper remain single-run; the confidence intervals are across tasks, not across
repeated measurements. Don't phrase them as if the benchmarks were repeated.

## Finding 3 — Most "Lucky Wins" did not cheat (utqG W3)

The reviewer asked: when an agent gets a speedup without touching the intended
bottleneck (Q3), did it break the model to get it, or did it find a legitimate
optimization somewhere else?

We joined every Q3 case with its GSM8K correctness result (LM Evaluation Harness:
run the model with the agent's patch, check accuracy against the unpatched code).
There are 26 Q3 cases in total (22 vLLM + 4 SGLang). Correctness tests exist for 12
of them — the legacy-agent vLLM cases. Result (`q3_decomposition.md`):

- **11 of 12 kept accuracy unchanged.** These are real optimizations at a different
  location — "emergent wins", not cheating.
- **Exactly 1 broke the model:** commit `fe66b347`, TRAE (Sonnet), accuracy 0.32 → 0.00.
  This is the same Bamba case the paper already uses as its example.

This is the strongest single rebuttal deliverable: it shows the soft-metric
framework catches the one genuinely dangerous case, while most Lucky Wins are
legitimate work the current framing undersells.

**Caveats:** (1) the 14 untested cases are all OpenHands cases plus all SGLang cases —
covering the 10 OpenHands vLLM ones needs about 10 cheap single-GPU lm-eval runs;
SGLang has no correctness harness at all, so those 4 can't be covered; say so rather
than hiding it. (2) "Correctness" here means GSM8K exact-match only — one benchmark,
one signal. (3) The draft response says "19 Q3 cases on vLLM"; the real count is 22
(the draft's number matches nothing — fix it).

## Finding 4 — The baseline numbers exist but do NOT say what the draft wants (utqG W4)

The reviewer asked for gains measured against the *unoptimized* code (the commit
before the human's fix), not just against the human patch. We assembled this from
the HuggingFace benchmark table (which stores baseline measurements per commit) plus
local files: **28 of 39 vLLM tasks and 14 of 15 SGLang tasks now have a measured
baseline** (`baseline_deltas.md`).

The problem is what the measurements show. The *human* patch, measured against the
baseline on our harness:
- vLLM: median gain **+1.5%** (range −31.9% to +99.8%); only 11 of 28 tasks show a
  gain above 5%.
- SGLang: median gain **−11.9%** (negative!); only 2 of 14 tasks show a gain above 5%.

In plain terms: on our benchmark configuration, the majority of human reference
patches do not reproduce the speedups their pull requests claimed. The likely
reasons: the SGLang isolated runs are very short (~50 requests, ~1 second of
benchmarking — heavily noise- and warmup-dominated), and the benchmark command
sometimes exercises a different configuration than the one the PR author measured.

**Consequences for the rebuttal:**
- The draft planned to write "manual curation required non-trivial verified speedup,
  here is the minimum/median human speedup." **Written from measured data, that
  sentence is false and would hand the reviewer a weapon.** Do not write it.
- Honest options: (a) extract the *claimed* speedups from the PR text (they exist as
  unstructured text in `reference_data/*/human_commits.jsonl` — a separate parsing
  pass); (b) present agent-vs-baseline next to human-vs-baseline without any
  inclusion-criterion claim; (c) openly discuss the measurement-context gap.
- Additional caveat inside the table: the vLLM OpenHands agent-vs-baseline medians
  (+45%/+48%) are an artifact of mixing measurement eras — OpenHands agent numbers
  come from the May 2026 re-bench, the baselines from an earlier benchmarking
  campaign, possibly different hardware/software state. They are not evidence that
  OpenHands is dramatically better than baseline. The legacy agents' numbers share
  the baseline's era and are comparable.
- Remaining gap: 9 vLLM tasks have no baseline at all
  (`19d98e0c, 660470e5, 6e36f4fa, 9474e89b, 9ed82e70, ad8d696a, d7740ea4, e3580537,
  fc7b8d1e`). One single-GPU session closes this.

## Finding 5 — The strict-definition check cuts success rates hard (kNyS Q4)

The reviewer suggested checking what happens if only "Same target" (the agent edited
the exact locations the human did) counts as correct targeting, instead of Same OR
Related (same module). We re-ran the quadrant assignment under that stricter rule
(`stats_tables.md`, section 3):

- True Success drops by 8 to 53 percentage points in every cell. Examples: vLLM
  Claude Code 46.2% → 20.5%; SGLang Codex 80.0% → 26.7%; SGLang TRAE (GPT-5)
  86.7% → 46.7%; vLLM OpenHands (Sonnet-4.5, canonical) 56.4% → 7.7%.
- The reason: "Related target" is actually the *most common* correct-targeting label
  (122 of 234 vLLM records, vs 63 "Same").

**How to use this:** disclose it proactively (the reviewer can compute it
themselves), but frame it as a sensitivity bound, and defend counting Related as
correct: "same module" genuinely means the agent found the right bottleneck
neighborhood, and the human annotators agreed with the judge's targeting labels at
κ = 0.836. Presented without that framing, this table would eat the paper's
headline numbers.

## Finding 6 — Step-level data shows how the scaffolds actually differ (kNyS Q3)

We computed per-run behavioral metrics from the step-level logs — number of steps,
number of edit actions, number of shell/run actions, time from start to first file
edit, total duration, and how the run ended (`trajectory_components.md`). Step logs
exist for 4 of the 6 configs: TRAE (both models, stored locally) and OpenHands (both
models — these had to be fetched from the HuggingFace rebuttal datasets; they are
not in the local tree). Medians on vLLM:

| Config | steps | edits | first edit after | total duration | ended cleanly |
|---|---|---|---|---|---|
| TRAE (Sonnet) | 44 | 17 | 8 s | 502 s | 0%* |
| TRAE (GPT-5) | 42 | 19 | 80 s | 1143 s | 36%* |
| OpenHands (Sonnet-4.5) | 109 | 13 | 72 s | 388 s | 94% |
| OpenHands (GPT-5) | 79 | 9 | 324 s | 836 s | 100% |

\* "ended cleanly" is measured differently per harness (TRAE: its `success` flag;
OpenHands: an explicit `finish` action), so compare within a harness, not across.

Patterns worth using: GPT-5 configs take 2–4× longer to make their first edit and to
finish than Sonnet configs *under the same scaffold* — a concrete, quantified
mechanism behind the model-vs-scaffold discussion. OpenHands runs many more, smaller
steps and almost always terminates deliberately.

A coarser table (duration, patch size, patch-generation rate — available for all six
configs from the run summaries) adds: Claude Code is the fastest overall (median
207 s), Codex 346 s; patch-generation rate is 97–100% everywhere.

**Caveats:** Codex CLI and Claude Code have *no step logs anywhere* — only final
patches and durations. The paper's Appendix H sentence claiming "full trajectories
for the open-source harnesses" is wrong for Codex; don't repeat it. OpenHands
(Sonnet-4.5) is missing 3 of 39 vLLM trajectories and 10 of 15 SGLang ones.

## Finding 7 — Scaffold and model matter about equally (kNyS Q2)

True Success %, arranged as scaffold × model (canonical data):

| Scaffold | Sonnet-4.5, vLLM | GPT-5, vLLM | Sonnet-4.5, SGLang | GPT-5, SGLang |
|---|---|---|---|---|
| TRAE | 28.2 | 17.9 | 80.0 | 86.7 |
| OpenHands | 56.4 (published: 43.6) | 28.2 | 13.3 | 33.3 |
| Claude Code (Sonnet only) | 46.2 | — | 26.7 | — |
| Codex CLI (GPT-5 only) | — | 20.5 | — | 80.0 |

Holding the model fixed, switching scaffold moves vLLM True Success by up to ~28
points. Holding the scaffold fixed (OpenHands), switching model also moves it by
~28 points. So scaffold and model effects are the same order of magnitude — and the
TRAE row flips direction across codebases (worst-tier on vLLM, best-tier on SGLang),
which is the cleanest evidence that single-codebase evaluations mislead.

## Finding 8 — Dataset composition and extra metrics (yX4G)

**Composition** (`composition.md`, built from the dataset files): typical task
touches 2 files (max 19) and ~52–56 edited lines. vLLM: 30 serving-mode / 7
latency-mode / 2 throughput-mode benchmarks, 22 distinct models benchmarked, edits
concentrated in `vllm/v1`, `vllm/model_executor`, `vllm/core`. SGLang: 14 serving /
1 latency, almost all edits in `python/sglang`.
*Caveat:* the draft response references a `performance_areas` field from the
filtering pipeline — that field exists only in the paper's Figure 7 illustration,
not in any stored data. A bottleneck-category column would require fresh labeling
(a 33-commit partial start exists in `archive/misc/results/reviews/`).

**Extra metrics** (`multimetric.md`): the raw serving benchmarks record TTFT, TPOT,
ITL, and three throughput measures. Median agent-vs-human across them: agents are
slightly worse on TPOT and ITL too (e.g. Codex −7.2% TPOT), roughly at parity on
throughput — consistent with the headline story, so reporting them helps rather
than hurts.
*Caveats:* coverage differs per metric because every task runs its own PR's
benchmark command (~30 vLLM tasks emit only a latency scalar); SGLang's benchmark
emits ITL but no TPOT. A single uniform "composite metric" across all 54 tasks is
not derivable from the logs — don't promise one.

## Finding 9 — The judge-disagreement item is blocked on missing credentials

The raw per-task labels from the two human annotators live in a Supabase database.
The script that computes agreement (`EAD/scripts/compute_agreement.py`) fetches them
live and needs a Supabase URL + key that exist nowhere in the repo or environment
(the expected `soft-metrics-review/.env.local` is absent). Until someone supplies
credentials, three promised items cannot be produced: the disagreement error
analysis, the worked boundary examples, and the claim "no disagreement flips a Q1
outcome to Q3" (which must be *verified*, not asserted — if flips exist, report them
and quantify the effect). This is the only analysis item that could not be executed.

---

## Not run (needs GPUs, API spend, or is a paper edit)

- **GPU:** 9 missing vLLM baselines (closes W4); ~10 OpenHands lm-evals (closes W3);
  a Level-2 TP2 human-reference subset (optional, for W2); new model families
  (kNyS Q1, expensive).
- **API spend:** LLM-judge pass over the 66 existing open-model runs
  (`trae_opensource` sweep) — the cheapest real answer to kNyS Q1.
- **Paper edits:** apply `camera_ready_edits.tex` (five drafted fixes, including the
  66/40-vs-39/15 task-count inconsistency); fix Figure 7 naming the Stage-2 judge
  `gemini-3-flash-preview` while Appendix C.1 says GPT-5-mini.
