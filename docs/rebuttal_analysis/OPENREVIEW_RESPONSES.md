# OpenReview response drafts — NeurIPS submission 2841

**INTERNAL PREAMBLE — do not post this section.**

Decisions baked into these drafts (change them only deliberately):
1. **Published numbers everywhere** (the PDF the reviewers scored). The post-submission
   X3 correction to the vLLM OpenHands-Sonnet-4.5 cell (43.6→56.4) is NOT used; if the
   team decides to disclose it, that's a separate coordinated edit.
2. **No significance tests in the responses.** W1 rests on the Appendix G rollout
   variance plus confidence intervals; the McNemar results in `stats_tables.md` stay
   internal (they were also entangled with the OH-S45 vLLM cell divergence).
3. Promises are limited to what is landable: the open-model soft-metrics table (API
   cost only) is promised for the revision; GPU items (TP2 demo, remaining lm-evals,
   9 baselines) are phrased as "revision/camera-ready", never "by date X".
4. No mention of ICML; no "full trajectories for the open-source harnesses" phrasing;
   no TRT-LLM/FlashInfer in the utqG response.
5. Post order: utqG → kNyS → yX4G.

---

## Response to Reviewer utqG (rating 2)

We thank the reviewer for the careful and constructive review.

**W1 (Benchmark scale and statistical analysis).** Table 3 reports a single rollout
per task per agent. To quantify variance we ran additional independent rollouts for
Claude Code and Codex CLI on 30 vLLM tasks (Appendix G), and we have now added 95%
confidence intervals to those rates:

| Agent | Pass@1 (95% CI) | Across rollouts (mean ± std) |
|---|---|---|
| Claude Code | 50.0% (33.2–66.8) | 46.7% ± 3.3% |
| Codex CLI | 23.3% (11.8–40.9) | 25.5% ± 2.2% |

(These rates are on the 30-task subset, which is why pass@1 differs from Table 3's
39-task figures.) Variance across rollouts is moderate, with a standard deviation of
at most 3.3 points, and the relative ranking is preserved: Claude Code consistently
outperforms Codex CLI on vLLM.

The reviewer's point about resolution is well taken. At n=15 a single SGLang task
moves the rate by 6.7 points, which is larger than the rollout-to-rollout variation,
so what limits these tables is the size of the task set rather than run-to-run
noise. We have therefore added Wilson 95% intervals to every True Success and Hard
Success rate in Tables 3 and 4, and we soften cross-codebase ranking claims wherever
those intervals overlap.

The scale itself is constrained by the domain. GPU-inference optimization commits
with reproducible setups, deterministic measurement, and non-trivial verified
speedups are rare in upstream history, and each retained task must also build in a
Docker container and be re-executed on H100. General SWE benchmarks can mine
thousands of issue-fix pairs; measurable performance commits cannot be scaled the
same way without giving up the hard-metric verifiability the benchmark exists to
provide.

**W2 (Level 2 not evaluated).** Level 1 and Level 2 come from the same pipeline and
differ by inclusion criteria, not by curation quality. Level 1 applies strict
criteria: each task must be a reproducible optimization commit that builds
successfully in a Docker container and is measurable on a single H100. Level 1
therefore measures localized, patch-scale optimization on single-node GPU inference.
The same pipeline with relaxed criteria admits multi-GPU and heterogeneous-hardware
problems, which are routed to Level 2 rather than discarded. Level 2 received the
same three-stage filtering and the same manual curation (genuine-optimization check,
benchmark configuration, the performance claim from the PR discussion, task
description). What it has not received is execution, because we restricted
evaluation to the strict criterion in this paper due to time and compute
constraints.

The reviewer is right that this leaves three gaps, and we will close them as far as
we can. There is no human-reference reproduction for Level 2, which we will state in
§3.2 rather than leaving it to the appendix. The pipeline applies to multi-GPU
settings without modification, and if compute allows during the discussion period we
will run a small TP2 subset end to end rather than assert it. We will also
characterize both splits on identical axes (parallelism configuration and hardware
target, model footprint, benchmark mode, patch size) so the boundary is inspectable
rather than merely described. Finally, we will reconcile the counts, since Table 6's
66 vLLM / 40 SGLang is the Level 1 ∪ Level 2 union while §3.2's 39 / 15 is Level 1.
If the reviewer considers a released-but-unevaluated tier to weaken the paper, we
are happy to present Level 2 purely as a curated data release rather than as a
contribution.

**W3 (Two kinds of Lucky Win).** We agree with the distinction, and the reviewer
identifies the first kind correctly: on the Bamba commit (Appendix E.3), TRAE
(Sonnet) matched the human speedup while exact-match accuracy collapsed from 32% to
0%, and the soft metric together with the correctness check caught what hard metrics
alone would have recorded as success.

On the second kind, a sound alternative optimization is not penalized merely for its
location. The soft metric has two independent dimensions (Table 2). Bottleneck
Targeting records where the agent worked, and Implementation Approach records
whether the approach was sound, including an explicit "Valid alternative" category
for a different but sound solution, which is well populated in practice (Figures 5
and 6). The judge does not assess code quality from scratch; it compares the agent
patch against the known human patch and labels both dimensions. Both labels are
validated against human experts: two independent annotators labeled all 54 tasks,
with pairwise Cohen's κ as follows.

| Metric | LLM vs H1 | LLM vs H2 | Mean LLM–human κ |
|---|---|---|---|
| Bottleneck Target | 0.881 | 0.791 | 0.836 |
| Implementation Approach | 0.949 | 0.810 | 0.880 |

Agreement is strong on both dimensions, and the judge is stable across eight
independent runs on identical patches (maximum deviation ±2.5%). A case labeled
"different location, sound approach" therefore reflects a judgment human experts
share, not an artifact of the judge. In the revision we will report the
Implementation Approach breakdown of Q3 alongside the Q3 rate, and state in §5.2
that a Valid-alternative Q3 represents legitimate optimization work that a single
"Lucky Win" label undersells.

**W4 (Hard metrics relative to the human patch).** The unoptimized baseline is
measured, not assumed: for every task the harness runs the PR's benchmark command on
the pre-optimization commit as well as on the human and agent patches (§4.3), and
correctness is evaluated for the baseline and the agent patch (§3.3.4). We anchor
the reported classification to the human patch by design, following reference-based
evaluation as in GSO; the ±5% Similar threshold follows the same precedent, since
GSO uses 95% to match similar performance in Opt@K and the MLPerf Mobile inference
benchmark uses the same tolerance. Absolute improvement over unoptimized code is a
weak bar that an agent can clear by touching almost anything on a hot path, which is
the Q3 (Lucky Win) outcome, whereas matching the fix the maintainers merged is the
capability we set out to measure.

On the specific concern that a trivial human improvement would make "Similar"
uninformative: task inclusion is governed by strict criteria. Each task must be a
reproducible optimization commit that builds in a Docker container, and manual
curation recovers the author's benchmark configuration and the performance claim
documented in the PR discussion (Appendix C.2), so tasks are not drawn from commits
whose own contribution was negligible. We will additionally release the baseline
measurement per task alongside the human and agent runs, so that a "Similar" verdict
can be read against the size of the underlying gain.

**W5 (Difficulty decay).** We acknowledge this risk and cannot rule out that future
models will absorb these exact changes; the mechanism applies to any benchmark mined
from public history. Two observations bound the concern today: if contamination were
already a major factor we would expect high execution success rates, whereas agents
frequently identify the correct bottleneck but fail to produce a working
implementation (the dominant Q2 outcome); and the issue is not unique to ISO-Bench,
since SWE-Perf, SWE-fficiency, and GSO are all built from public repositories and
face the same tradeoff.

Our limitations section already identifies temporal filtering and freshly merged PRs
as the remedy, and we now commit to it as maintenance rather than future work.
Following SWE-bench Goes Live, subsequent releases will add newly merged PRs gated
by a documented cutoff date, with results always reported against a dated, versioned
split. This is practical because the collection pipeline is automated end to end
(§3.2): commit mining, LLM-based filtering, and benchmark-command extraction rerun
against a later cutoff without manual rebuilding. Both vLLM and SGLang merge
performance PRs continuously, so the candidate pool renews rather than depletes. We
will elevate this to a stated maintenance commitment in §6.

---

## Response to Reviewer kNyS (rating 3)

We thank the reviewer for the thorough summary and concrete questions.

**Limited scale.** Please see our response to Reviewer utqG (W1). In brief:
Appendix G quantifies rollout variance directly (repeated independent rollouts for
two agents on 30 vLLM tasks; standard deviation ≤ 3.3 points, ordering preserved),
so single-rollout reporting is not what limits the tables. The task sample is, and
we now report 95% intervals for every cell of Tables 3 and 4 so that limit is
visible. We agree that low-frequency outcomes such as Q4 on SGLang cannot support
strong claims at n=15, and we have softened those statements.

**Limited codebase coverage.** The filtering pipeline has already been run
end-to-end on TensorRT-LLM and FlashInfer (Appendix C, Table 6), yielding 47 and 52
curated candidate tasks; these were held out because TensorRT-LLM's engine-build
flow differs substantially from the vLLM/SGLang path. We commit to releasing them
as an extension split with the same schema. We chose vLLM and SGLang deliberately:
they are the two highest-adoption serving stacks, and depth on production-relevant
codebases is the intended contribution rather than breadth across scientific
computing.

**Model diversity (Q1).** We agree. The paper currently reports open-weight models
as detailed failure case studies (MiniMax-M2.1, GPT-OSS-120B, GLM-4.7; Appendices
E.5–E.7), which surface a distinct capability tier: these models fail before
optimization quality is even in question (zero tool calls in 75 steps; mocking
PyTorch instead of using it; valid edits but no task completion). For the revision
we will add a quantitative table over our full open-model sweep (six open-weight
models run under the TRAE scaffold on the vLLM split), reporting patch-generation
rates and bottleneck-targeting distributions in the Table 5 format. Hard-metric
evaluation of additional frontier families such as DeepSeek and Gemini requires
substantial GPU re-benchmarking, which we commit to for the camera-ready.

**Model/scaffold decoupling (Q2).** The experimental design already implements the
controlled comparison the reviewer asks for: Claude Sonnet 4.5 is evaluated under
three scaffolds (Claude Code, OpenHands, TRAE) and GPT-5 under three scaffolds
(Codex CLI, OpenHands, TRAE), a 2-model × 3-scaffold grid (Table 7, §5.5). Holding
the model fixed, True Success on vLLM spans 28.2% to 46.2% across scaffolds;
holding the scaffold fixed (OpenHands), swapping Sonnet 4.5 for GPT-5 moves vLLM
True Success from 43.6% to 28.2%. The two effects are the same order of magnitude,
and the TRAE row reverses direction across codebases (Table 4). We will restructure
§5.5 around this grid explicitly so the decoupling reads as a designed comparison.

**Which scaffold components drive the gap (Q3).** Appendices E.8–E.9 already give a
controlled worked example: the same Bamba task under OpenHands (clean 296s run,
valid 165-line patch, judged a Valid alternative) versus TRAE GPT-5 (tool-call
emission failure, empty patch). For the revision we have aggregated step-level
trajectory features across tasks for the four configurations with step logs, namely
TRAE and OpenHands under both models (per-task medians on vLLM):

| Config | steps | edit actions | time to first edit | duration |
|---|---|---|---|---|
| TRAE (Sonnet) | 44 | 17 | 8 s | 502 s |
| TRAE (GPT-5) | 42 | 19 | 80 s | 1143 s |
| OpenHands (Sonnet-4.5) | 109 | 13 | 72 s | 388 s |
| OpenHands (GPT-5) | 79 | 9 | 324 s | 836 s |

Two mechanisms stand out. Holding the scaffold fixed, GPT-5 configurations take 2 to
4 times longer to reach a first edit and to finish than Sonnet configurations. And
OpenHands decomposes work into roughly twice as many, smaller steps and terminates
deliberately (94% to 100% of runs end with an explicit finish action), whereas TRAE
GPT-5 runs frequently truncate. For Claude Code we observe only inputs and outputs,
and we keep the Appendix H caveat that closed-source scaffolding limits
attribution; the open-scaffold analysis brackets the plausible mechanisms.

**Judge boundary cases (Q4).** The soft-metric categories are author-calibrated for
this benchmark, and the judge operates in a constrained reference-comparison
setting: it receives the human patch, the agent patch, and the task description
(Figure 19), and classifies whether they target the same bottleneck and take a
similar approach. It is not asked to assess code quality from prior knowledge. We
will add worked examples of the Same/Related/Different boundary to Appendix F in
the revision. We also ran the sensitivity check the reviewer suggests: counting
only Same target (identical locations) as correct, True Success drops substantially
in every configuration, for example Claude Code on vLLM 46.2%→20.5%, Codex CLI on
SGLang 80.0%→26.7%, and TRAE (GPT-5) on SGLang 86.7%→46.7%, because Related (same
module) is the most common correct-targeting label. We report this as the strict
lower bound and keep Same-or-Related as the primary definition: "same module"
indicates the agent located the right bottleneck neighborhood, and the targeting
dimension is precisely where LLM–human agreement is strongest (mean κ = 0.836,
Table 8). We prefer to disclose this sensitivity ourselves so readers can pick the
definition matching their use case.

---

## Response to Reviewer yX4G (rating 4)

We thank the reviewer for the positive assessment and the precise technical
comments.

**Hard performance metric (TTFT vs throughput).** Both TTFT and throughput are
tracked for every task (§3.3.1, Eqs. 1–2). TTFT is the primary classification
metric for serving benchmarks and throughput for standalone benchmarks, because we
execute the exact benchmark command the human author used in the original PR, which
constrains which metrics are emitted. We agree a fuller profile strengthens the
benchmark: in the revision we report throughput alongside TTFT for all serving
tasks, and add TPOT/ITL where the project's benchmark script produces them. These
added metrics are consistent with the headline results, since agents are modestly
worse than the human reference on TPOT and ITL as well (for example, median −7.2%
TPOT for Codex CLI on vLLM serving tasks) and near parity on throughput. One design
constraint we keep deliberately: we do not replace the PR's benchmark configuration
with our own composite workload, because comparability against the human author's
claimed improvement is what makes the human-relative classification meaningful. We
will make this tradeoff explicit in §3.3.1.

**Reliability of the soft metric.** The validation the reviewer asks for is in
Appendix F: mean LLM–human Cohen's κ of 0.836 (Bottleneck Targeting) and 0.880
(Implementation Approach) against two independent annotators across all 54 tasks,
plus eight-run stability with maximum deviation ±2.5% (Figure 20), and the full
judge prompt (Figure 19). On the specific concern about "alternative but sound"
solutions: the judge does not compare diff locations heuristically. It receives
both complete patches with the task description in a constrained
reference-comparison setting, and the taxonomy includes an explicit Valid
alternative category, which is well populated in practice (Figures 5–6). Claude
Code in particular is frequently credited with sound alternatives rather than
penalized for them. To address the false-negative concern directly, we will add an
error analysis of the LLM–human disagreement cases to Appendix F in the revision,
categorized by which label boundary each falls on.

**Task selection and representativeness.** We agree the paper should characterize
the dataset rather than assert its validity, and we have added a composition
analysis to the appendix: the median task touches 2 files (max 19) with ~50 edited
lines; vLLM tasks split into 30 serving / 7 latency / 2 throughput benchmark modes
across 22 distinct models, with edits concentrated in `vllm/v1`,
`vllm/model_executor`, and `vllm/core`; SGLang tasks are 14 serving / 1 latency,
concentrated in `python/sglang`. This makes the sparse-signal concern inspectable,
since readers can see which areas are well covered and which are thin. We will also
state explicitly that the benchmark deliberately targets isolated, measurable
optimizations (<10 files, reproducible single-command benchmarks): a scope
decision, not a claim of covering the full bottleneck spectrum.
