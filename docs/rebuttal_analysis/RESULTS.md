# NeurIPS Rebuttal — Answers built on the ICML rebuttal (updated 2026-07-25)

The same paper was reviewed at ICML (submission 28104: UJ1j accept-5 → fully
resolved; dpfc reject-2; Rdwi weak-reject-3; meta-reject on scale / novelty /
closed-source scaffolds — see `ICML-rebuttal.txt`). Most of what the NeurIPS
reviewers ask was already asked at ICML, we already wrote answers, and — most
importantly — **the fixes we promised at ICML are already inside the NeurIPS
submission**. So the default response pattern is:

1. **Point at the paper** (the asset often already exists in an appendix).
2. **Reuse the ICML rebuttal text** that resolved or partially resolved the same
   concern (adapted, without mentioning ICML).
3. **Attach the new numbers** from `docs/rebuttal_analysis/` only as supplements.

## What the NeurIPS submission already contains because of ICML

These were promised or delivered in the ICML rebuttal and are now IN the paper —
each one directly answers a NeurIPS ask:

| Added after ICML | In the paper now | Answers (NeurIPS) |
|---|---|---|
| Judge prompt + 8-run stability (±2.5%) + Cohen's κ vs 2 humans (0.836 / 0.880) | Appendix F, Figures 19–20, Table 8 | yX4G judge reliability, kNyS Q4 |
| Pass@1 vs pass@2 rollout variance | Appendix G, Table 9 | utqG W1, kNyS scale |
| OpenHands runs — **the exact thing ICML-Rdwi's follow-up requested** ("run open source scaffolds like OpenHands") | Both models, all 54 tasks, Tables 3–5 | kNyS Q2/Q3, ICML reproducibility meta-concern |
| Open-model failure case studies (GPT-OSS-120B, MiniMax-M2.1, GLM-4.7) | Appendix E.5–E.7 | kNyS Q1 |
| Trajectory worked examples (same Bamba commit under two scaffolds) | Appendix E.8–E.9 | kNyS Q3 |
| TRT-LLM / FlashInfer filtering runs (47 + 52 curated candidates) | Appendix C, Table 6 | kNyS codebase coverage |
| Level 2 multi-GPU split released (52 tasks) | §3.2, HF dataset | utqG W2 |

Frame all of this as "the paper already contains X" — reviewers score what's in
the PDF, and it's all in the PDF.

---

## Item-by-item: question → answer → source

### utqG W1 + kNyS "limited scale" — error bars / significance

**The question:** point estimates on 39/15 tasks with no variance.

**The answer (3 parts):**
- Reuse ICML-UJ1j's resolved response: judge stability (8 runs, ±2.5%) and
  rollout variance (Appendix G: std ≤3.3%, ranking preserved) are already in the
  paper. Reuse the ICML size justification (Rdwi): strict inclusion criteria —
  every task must be a reproducible, Docker-buildable optimization commit — plus
  the cost argument (2h agent budget + H100 re-eval + correctness + judge ≈
  thousands of dollars at 54 tasks).
- New supplement: Wilson + bootstrap 95% CIs for every cell and 30 paired exact
  McNemar tests (`stats_tables.md`). The headline claim — rankings invert across
  codebases — is significant in both directions (Sonnet scaffolds > GPT-5
  scaffolds on vLLM, p ≤ 0.021; TRAE/Codex > Claude Code/OH-S45 on SGLang,
  p ≤ 0.008). Within-class orderings are not significant — soften those.
- New supplement: per-task rollout CVs from the pass@k data (~8 rollouts/task):
  throughput CV < 1% (threshold is 5%), TTFT CV 7–9% median on SGLang
  (`pass_at_k_variance.md`).

### utqG W2 — Level 2 not evaluated

**The question:** Level 2 is released but never run.

**The answer:** reuse the ICML L1/scope response almost verbatim — "the same
pipeline applies to multi-GPU and heterogeneous hardware without modification;
we restricted evaluation to Level 1 due to time and budget constraints" — and
note Level 2 already went through the same three-stage filtering + manual
curation as Level 1 (only agent rollouts and re-execution are missing). Since
ICML, Level 2 is now actually released on HF. Offer the repositioning
(curated data release) if the reviewer prefers; optional [RUN] a few TP2 tasks
with the human patch if GPUs land.

### utqG W3 — two kinds of Lucky Win

**The question:** does a Q3 speedup mean the agent broke the model, or found a
legitimate optimization elsewhere?

**The answer:** the paper already contains the anchor case — **Bamba, Appendix
E.3 / Figure 12** (TRAE-Sonnet matches human speedup, accuracy 32%→0%, caught by
soft metrics + LM Eval Harness, §5.6). Use it as-is. The new table
(`q3_decomposition.md`) generalizes it: across the Q3 cases with correctness
runs, **11 of 12 preserved accuracy** (valid alternative-location optimizations)
and exactly 1 broke it — the E.3 Bamba case. One sentence of framing: the
framework catches the harmful case, and most Lucky Wins are legitimate work,
which we will say in §5.2.

### utqG W4 — measured against baseline?

**The question is simply: did you compare against the unoptimized baseline, or
only against the human patch?**

**The answer: yes, we did — it's in the paper.** §4.3: hard metrics are executed
"against both the unoptimized baseline and the human solution." §3.3.4:
functional correctness is measured for the baseline and the agent patch. The
paper *reports* the human-relative classification because agent-vs-human is the
benchmark's scoring axis; the baseline measurements exist. Supplement: attach
the per-task absolute-improvement-over-baseline table (28/39 vLLM + 14/15
SGLang, `baseline_deltas.md`) so readers can see when "Similar" means matching
a large gain vs a small one.

*Internal note (one line):* when quoting how big the human speedups were, cite
the PR-claimed numbers from the PR discussions (that's what curation verified);
our short re-benchmark runs measure smaller deltas and aren't the right source
for that specific sentence.

### utqG W5 — contamination

**The answer:** reuse the ICML-dpfc contamination paragraph nearly verbatim — it
worked: "if contamination were a major factor we would expect high execution
success; instead agents frequently identify the correct bottleneck but fail to
implement it (the dominant Q2 outcome)"; same tradeoff as SWE-Perf /
SWE-fficiency / GSO; add the live-refresh (dated splits) commitment.

### yX4G — TTFT vs throughput / fuller metric profile

**The answer:** both are tracked for every task (§3.3.1, Eqs. 1–2); which one
classifies a task follows the PR's own benchmark command — a deliberate design
choice (comparability with the human author's claimed improvement). Supplement:
`multimetric.md` adds TPOT/ITL medians — agents are slightly worse there too,
consistent with the headline, so it strengthens rather than changes the story.
Don't promise a uniform composite metric across all 54 tasks (~30 vLLM tasks
emit only a latency scalar; SGLang has no TPOT field).

### yX4G — judge reliability

**Mostly already answered — in the paper — because ICML asked the same thing
twice (UJ1j, dpfc/Rdwi):**
- Appendix F: full judge prompt (Figure 19), 8-run stability with 95% CIs
  (Figure 20, max ±2.5%), LLM-human κ vs two independent annotators (Table 8:
  0.836 / 0.880). This exact material flipped ICML-UJ1j to "fully resolved."
- For the "alternative but sound solutions" worry, reuse the ICML-dpfc follow-up
  paragraph: the judge is not grading from scratch — it's a **constrained
  reference comparison** (sees human patch + agent patch + task, GSO-style), the
  taxonomy has an explicit Valid-alternative category, and that category is well
  populated in practice (Figures 5–6).
- The only piece that needs new work is the worked disagreement/boundary
  examples — blocked on the Supabase export of the raw H1/H2 labels (no
  credentials in the repo; whoever ran the review app has them).

### yX4G — representativeness / dataset composition

**The question:** characterize the dataset instead of asserting validity.

**The answer:** reuse the ICML strict-inclusion-criteria text for why the set is
what it is, then attach the composition table (`composition.md`): median 2 files
and ~52–56 edited lines per task; vLLM 30 serving / 7 latency / 2 throughput
across 22 models, edits concentrated in `vllm/v1`, `vllm/model_executor`,
`vllm/core`; SGLang 14 serving / 1 latency in `python/sglang`. State explicitly
(as the draft already does) that the benchmark deliberately targets isolated,
measurable optimizations — a scope decision, not a coverage claim.

### kNyS — codebase coverage

**The answer:** reuse ICML-Rdwi Q1 verbatim: the collection strategy is not
vLLM/SGLang-specific. It's now stronger than at ICML: the pipeline has actually
been run end-to-end on TensorRT-LLM and FlashInfer (Appendix C, Table 6 — 47 and
52 curated candidates), held out only because TRT-LLM's engine-build flow
differs. Commit to releasing them as an extension split. Keep the ICML framing:
depth on the two highest-adoption serving stacks is the intended contribution.

### kNyS Q1 — model diversity

**The answer:** reuse the ICML line — "current open-source models, including
GPT-OSS-120B, MiniMax-M2.1, and GLM-4.7, fail on these tasks" — now backed by
Appendix E.5–E.7 case studies (describe them as case studies, not full benchmark
runs). Cheapest upgrade: run the soft-metrics judge over the 66 existing
`trae_opensource` runs (API cost only) → a quantitative patch-rate + targeting
table for 6 open models. Frontier-family hard metrics = [RUN, expensive] or
camera-ready commitment.

### kNyS Q2 — scaffold vs model decoupling

**This is the paper's own thesis — showcase the paper's tables.** §5.5 + Table 4:
same model (Sonnet 4.5) under three scaffolds spans 28.2–46.2% True Success on
vLLM and the ordering inverts on SGLang; same scaffold (OpenHands), swapping the
model, moves vLLM 43.6→28.2. Figures 5–6 show the *mechanism* the ICML rebuttal
described qualitatively: Claude Code explores alternative approaches while TRAE
stays close to the reference, and which strategy wins flips between codebases.
Supplement with the 2×2 grid (scaffold × model — TRAE and OpenHands both ran
both models) and the McNemar significance from W1. Restructure §5.5 around the
grid so it reads as designed, not incidental.

### kNyS Q3 — which scaffold components drive the gap

**This is the request ICML-Rdwi made in their follow-up ("quantitative
treatments... run OpenHands"), and the NeurIPS submission + our new table are
exactly that.** Point to E.8–E.9 first: the same Bamba commit under OpenHands
(clean 296s run, valid 165-line patch, Alternative approach) vs TRAE-GPT5
(tool-call emission failure, empty patch) — the paper's worked example of
scaffold-driven divergence. Then attach the aggregate table
(`trajectory_components.md`, step logs for TRAE ×2 + OpenHands ×2): OpenHands
runs ~2× more steps and terminates deliberately (94–100% explicit finish);
GPT-5 configs take 2–4× longer to first edit than Sonnet under the same
scaffold. For Claude Code, keep the paper's existing framing (inputs/outputs
only, Appendix H) — the open-scaffold analysis brackets the mechanisms.

### kNyS Q4 — judge boundary cases + stricter rule

**The question has two parts.** (a) Where is the Same/Related/Different boundary
and can we see examples? (b) What happens if Related doesn't count as correct?

**(a)** Reuse the ICML-dpfc follow-up: categories are author-calibrated for this
benchmark; the judge does a constrained reference comparison, not open-ended
grading. Worked examples need the Supabase export (same blocker as yX4G).
**(b)** Done — we re-ran the quadrant assignment counting only Same as correct
(`stats_tables.md` §3): True Success drops in every cell (e.g. vLLM Claude Code
46.2→20.5; SGLang Codex 80.0→26.7) because Related ("same module") is the most
common correct-targeting label. Present it proactively as the strict bound, and
defend Related-as-correct: same module = the right bottleneck neighborhood, and
the targeting dimension is exactly where LLM-human agreement is strong
(κ = 0.836, Table 8). Disclose it ourselves rather than letting the reviewer
compute it.

---

## Before posting: one data decision

Rebuilding Table 5 from today's data reproduces the PDF for 11/12 cells; vLLM
OpenHands (Sonnet-4.5) now computes to True Success 56.4% vs the published 43.6%
(five results changed in the post-submission X3 re-bench). Decide once: stand on
the published numbers everywhere, or disclose the (upward) correction. All new
tables mark the affected cell either way.

## Still to do

- Assemble the OpenReview responses from the sections above (post utqG first).
- Supabase export → disagreement/boundary examples (only remaining analysis blocker).
- Optional [RUN]: 9 vLLM baselines, ~10 OpenHands lm-evals, TP2 subset,
  open-model soft-metrics pass.
- Apply `camera_ready_edits.tex`.
