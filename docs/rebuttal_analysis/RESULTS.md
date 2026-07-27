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

**The answer (final): the ICML "Q4: Rollouts and variance" answer, used as-is.**
- Inline the Appendix G pass@1/pass@2 table exactly as ICML posted it (Rdwi marked
  that answer *addressed*), including ICML's closing line that pass@k was limited to
  2 agents on 30 vLLM tasks by GPU budget. One correction to ICML's own text: it said
  "std of 3-4%" while the table shows 3.3 and 2.2, so quote "at most 3.3 points".
- Add the subset note ICML lacked: pass@1 = 50.0% here vs Table 3's 46.2% because
  these rates are on the 30-task subset.
- Answer the reviewer's resolution example in the same terms, without statistics: at
  n=15 one SGLang task moves the rate 6.7 points, which is larger than the rollout
  variation, so the task-set size is the binding constraint. Ranking claims are
  softened where two agents differ by less than that.
- Reuse the ICML size justification (Rdwi): strict inclusion criteria (reproducible,
  Docker-buildable optimization commit) plus the cost argument (2h agent budget +
  H100 re-eval + correctness + judge ~ thousands of dollars at 54 tasks).
- **CIs come from the rollout table itself, not from the headline tables.** utqG asked
  for CIs, so we compute them from the Appendix G spread: SE = sd/sqrt(3), Student's t
  with df=2, giving Claude Code 38.5-54.9 and Codex CLI 20.0-31.0. They do not
  overlap, which is a quantitative version of ICML's "relative ranking is preserved".
  State plainly that these cover run-to-run variability, not task sampling.
  Wilson per-cell intervals and the 30 McNemar pairs stay in `stats_tables.md` for
  internal use only; both were entangled with the OH-S45 vLLM cell divergence. ICML-UJ1j's judge-stability answer (8 runs, ±2.5%, κ) belongs in the
  yX4G response, since it is variance of the labeler, not of the benchmark results.
- Note on the "pass@2" label: it is the authors' shorthand for the rate across
  repeated rollouts, not best-of-k sampling. The text above Table 9 says "to quantify
  variance", and 15/14/13 successes over 30 tasks reproduce 46.7% ± 3.3% exactly.
  Leave it as-is in the rebuttal; clarify the Table 9 caption at camera-ready.

**Held back deliberately** (`pass_at_k_variance.md`, internal): the same campaign
actually has up to 8 isolated rollouts per task, 618 benchmarked patches, and
measured on the metric that classifies each task, 82% of task-agent cells have
every rollout inside the ±5% band (throughput-classified 1.0% median; the tail is
TTFT-classified vLLM tasks). It is *not* in the response because it is
cross-campaign (agent patches only, no human reference in that context, 12 of 40
tasks ran a modified command), a reviewer cannot verify it, and citing 8 rollouts
invites "then report pass@8". Bring it out only if a reviewer challenges whether
the ±5% band sits above the measurement noise floor. Also: an earlier CV-only
summary said "TTFT CV 7–9% on SGLang" — that measured TTFT on SGLang tasks, which
are classified on throughput; SGLang cells are in fact the stable ones.
Optional [RUN] if the team ever wants real pass@k: one human-patch benchmark per
task in that context (~40 runs, 1×H100) unlocks Hard Success pass@k up to k=8;
True Success pass@k additionally needs a judge pass over the 618 patches (API only).

### utqG W2 — Level 2 not evaluated

**The question:** Level 2 is released but never run.

**The answer:** reuse the ICML L1/scope response almost verbatim — "the same
pipeline applies to multi-GPU and heterogeneous hardware without modification;
we restricted evaluation to Level 1 due to time and budget constraints" — and
make the key point that Level 2 is not a lesser collection: it is the *same*
pipeline split by hardware requirement, and it received the same three-stage
filtering and the same four-purpose manual curation (verified in §3.2 and
Appendix C.2 — the 66 vLLM / 40 SGLang curated counts are L1 ∪ L2). Only execution
is missing. Since ICML, Level 2 is actually released on HF, so "released" is
defensible. Offer the repositioning (curated data release), and volunteer the
count reconciliation ourselves — Table 6's 66/40 vs §3.2's 39/15 is the L1∪L2 vs
L1 distinction, already drafted as EDIT 2 in `camera_ready_edits.tex`.

**TP2 demo — back in, hedged.** utqG's W2 asks specifically for "confirmation the
code runs or that agents can operate under the benchmark", which nothing in the
existing artifacts answers. The OpenHands re-bench ran on 2xH100, so a small Level-2
TP2 subset (environment build + human patch + one agent) is landable; the response
offers it with an explicit "if compute allows / we do not want to promise runs we
cannot land". The one existing multi-GPU datapoint (`310aca88`, TP2 against a TP4
spec) is flagged not-comparable in `docs/HARD_METRICS_OH_GPT5_FINAL.md` and is not
cited.

**Internal, camera-ready:** the *released* Level-1 parquet sets `hardware` to "H100"
for all 54 tasks, so the released data is self-consistent with the paper's
single-GPU claim. The local provenance copy (`EAD/dataset/*.jsonl`) disagrees for 15
of them — vLLM `310aca88` (H100-TP4), two L4, one AMD-MI300X, one AWS-Neuron, and 10
of 15 SGLang tasks (TP8, TP16-DP16, PD/DP/EP) — i.e. the field means "hardware we
evaluated on" in the release and "hardware the PR author used" locally. Worth
reconciling before the data release so the two copies do not tell different stories.

Also checked, because W2 invites it: on the released data the L1/L2 boundary is
directionally clean but not crisp — 70-72% of Level 2 is multi-GPU, heterogeneous
hardware, or an oversized model, versus 23-27% of Level 1. That is why the response
describes the split as strict vs relaxed *inclusion criteria* (the ICML framing)
rather than as "Level 2 = multi-GPU", which a reviewer checking the parquet would
falsify.

### utqG W3 — two kinds of Lucky Win

**The question (verbatim, and read it carefully):** the reviewer *grants* that the
benchmark correctly catches "hacking the win" and cites Bamba E.3 approvingly. Their
concern is the other half — "a genuinely valid alternative optimization may be
labeled Lucky Win purely for editing a different location than the human, which
seems to penalize correct work" — and they want to see how Q3 splits between the two.
So this is **not** a request for a reward-hacking count. Answering it with "11 of 12
Q3 cases preserved accuracy" answers a question they did not ask.

**The answer (decided): Bamba E.3 + the ICML Cohen's κ table. Nothing else.**
1. **Bamba E.3 as-is** for the kind they already credit (Aman: use the appendix case,
   no hedging).
2. **The κ table verbatim from the ICML rebuttal** — LLM vs H1 0.881 / H2 0.791
   (targeting), 0.949 / 0.810 (approach), means 0.836 / 0.880, plus the ±2.5%
   eight-run stability. This is what answers their actual worry: "is the
   different-location label just the judge being harsh?" No — two human experts read
   these cases the same way. Pair it with the taxonomy point (Table 2 has an explicit
   *Valid alternative — different but sound* category, and the judge labels approach
   independently of targeting), so a sound alternative is recorded as sound.
3. One commitment sentence: report the Implementation Approach breakdown of Q3
   alongside the Q3 rate, and say in §5.2 that Valid-alternative Q3 is legitimate
   work a single "Lucky Win" label undersells.

**Deliberately not in the response** (available in `q3_decomposition.md` if wanted):
the actual Q3 split — 8 Valid alternative / 18 Ineffective over 26 cases, clustered
on 4 commits — and the generous-direction sensitivity (crediting those 8: vLLM CC
46.2→48.7, OH-S45 43.6→46.2, OH-GPT5 28.2→33.3, TRAE-GPT5 17.9→20.5; SGLang CC
26.7→40.0, OH-GPT5 33.3→40.0; ≤5.1pp vLLM / ≤13.3pp SGLang, top tier unchanged).
Held back to keep W3 short and to avoid putting an alternative scoring of the
headline tables in writing. Note the reviewer did literally ask to "see how Q3
splits", so if they push, the 8/18 line is the answer and needs no new runs.

**Also left out on purpose:** the "11 of 12 preserved accuracy" framing from the
earlier draft — that answers a reward-hacking question this reviewer never asked and
9 of those 11 are judged *Ineffective* ("misses bottleneck"), so it would have
claimed legitimacy our own labels deny. And no promise to extend correctness runs:
§3.3.4/§5 already claim validation of *all* Hard Success cases, so raising coverage
invites a comparison with the 12 artifact-backed cases. Fix that wording at
camera-ready instead.

### utqG W4 — measured against baseline?

**The question is simply: did you compare against the unoptimized baseline, or
only against the human patch?**

**The answer: yes, we did — it's in the paper.** §4.3: hard metrics are executed
"against both the unoptimized baseline and the human solution." §3.3.4:
functional correctness is measured for the baseline and the agent patch. Then
*defend the reporting choice* rather than just conceding it: "faster than
unoptimized code" is a weak bar an agent clears by touching anything on a hot path
— that is the Q3 behaviour the benchmark exists to expose — while "matches the fix
the maintainers merged" is the capability under test. Close with the curation
criterion (Appendix C.2 required a documented performance claim in the PR), which
rules out certifying "Similar" against a human patch that achieved nothing.

**Do NOT attach the per-task baseline table** (`baseline_deltas.md`), which the
earlier draft promised. Measured human-vs-baseline is median **+1.5% on vLLM and
−11.9% on SGLang** (only 11/28 and 2/14 tasks above +5%) — the short isolated runs
do not reproduce the PR-claimed speedups, so publishing that table next to the
"non-trivial verified speedup by construction" sentence would refute our own
inclusion criterion in the same response. The response instead offers baseline /
human / agent measurements in the **released per-task data**, which is honest,
useful, and does not put a self-contradicting median in the rebuttal text. If anyone
insists on quantifying human speedups, use the PR-*claimed* numbers from
`reference_data/*/human_commits.jsonl` (a text-extraction pass), never our measured
medians. Coverage if it ever matters: 28/39 vLLM + 14/15 SGLang have a baseline;
vLLM OpenHands baseline deltas are cross-context artifacts.

*Internal note (one line):* when quoting how big the human speedups were, cite
the PR-claimed numbers from the PR discussions (that's what curation verified);
our short re-benchmark runs measure smaller deltas and aren't the right source
for that specific sentence.

### utqG W5 — contamination

**The answer:** reuse the ICML-dpfc contamination paragraph nearly verbatim — it
worked: "if contamination were a major factor we would expect high execution
success; instead agents frequently identify the correct bottleneck but fail to
implement it (the dominant Q2 outcome)"; and keep the sentence the earlier draft
dropped — the concern is not specific to us, since SWE-Perf, SWE-fficiency and GSO
are all built from public repositories and face the same realism/contamination
tradeoff. Paper backing: Appendix H already states both mitigations (Q1–Q4 not at
ceiling; Q2/Q3 at non-trivial rates).

**On difficulty decay,** the live-refresh commitment (dated, versioned splits after
SWE-bench-Goes-Live) needs one supporting fact or it reads aspirational: the
collection pipeline is automated end to end (§3.2) — mining, LLM filtering,
benchmark-command extraction — so a refresh is running it against a later cutoff,
not rebuilding it. ICML-Rdwi independently praised exactly this ("collection schema
seems quite automatic and could scale"), so it is a claim a reviewer has already
found credible.

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

## Internal only — 14 "failed" cells that measured fine in the pass@k campaign

Not for any reviewer response. 17 (task, agent) cells have no agent measurement in
the canonical table (`primary_metric = agent_failed`) and are therefore scored as
Worse — i.e. as failures in Tables 3–5. 14 of them ran the *identical* benchmark
command in the pass@k campaign and produced 7–8 successfully benchmarked patches
there (`pass_at_k_variance.md` §4; 11 are Claude Code cells). Either the single
canonical rollout failed and that failure is not reproducible, or the aggregate is
missing a measurement that was never persisted — for the vLLM cells `failure_reason`
is null and no `*_agent_result.json` exists under `benchmark_results/vllm/claude_code/`
while the same commits have agent results for other tools. Both readings mean the
published rates for those agents are conservative. Resolve before the data release,
since a reader with the released artifacts can reconstruct this.

## Still to do

- Assemble the OpenReview responses from the sections above (post utqG first).
- Supabase export → disagreement/boundary examples (only remaining analysis blocker).
- Optional [RUN]: 9 vLLM baselines, ~10 OpenHands lm-evals, TP2 subset,
  open-model soft-metrics pass.
- Optional [RUN], highest value per GPU-hour for W1: ~40 human-patch benchmarks in
  the pass@k measurement context → turns Appendix G's pass@2-on-30-tasks into
  Hard Success pass@k up to k=8 on 40 tasks × 2 agents from rollouts already run.
- Apply `camera_ready_edits.tex`.
