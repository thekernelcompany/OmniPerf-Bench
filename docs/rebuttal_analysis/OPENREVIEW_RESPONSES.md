# OpenReview response drafts — NeurIPS submission 2841

**INTERNAL PREAMBLE — do not post this section.**

Decisions baked into these drafts (change them only deliberately):
1. **Published numbers everywhere** (the PDF the reviewers scored). The post-submission
   X3 correction to the vLLM OpenHands-Sonnet-4.5 cell (43.6→56.4) is NOT used; if the
   team decides to disclose it, that's a separate coordinated edit.
2. McNemar p-values are quoted only for agent pairs whose cells are identical between
   published and current data (i.e., not involving the OH-S45 vLLM cell).
3. Promises are limited to what is landable: the open-model soft-metrics table (API
   cost only) is promised for the revision; GPU items (TP2 demo, remaining lm-evals,
   9 baselines) are phrased as "revision/camera-ready", never "by date X".
4. No mention of ICML; no "full trajectories for the open-source harnesses" phrasing;
   no TRT-LLM/FlashInfer in the utqG response.
5. Post order: utqG → kNyS → yX4G.

---

## Response to Reviewer utqG (rating 2)

We thank the reviewer for the careful and constructive review. We address each
concern with concrete additions to the paper.

**W1 (Benchmark scale and statistical analysis).** We agree that point estimates
alone are insufficient at this scale. We have added Wilson 95% confidence intervals
for all True Success and Hard Success rates in Tables 3 and 4, and pairwise exact
McNemar tests (paired on shared tasks) for cross-agent comparisons:

| Project | Agent | True Success | 95% CI |
|---|---|---|---|
| vLLM | Claude Code | 46.2% | 31.6–61.4 |
| vLLM | OpenHands (Sonnet-4.5) | 43.6% | 29.3–59.0 |
| vLLM | TRAE (Sonnet) | 28.2% | 16.5–43.8 |
| vLLM | OpenHands (GPT-5) | 28.2% | 16.5–43.8 |
| vLLM | Codex CLI | 20.5% | 10.8–35.5 |
| vLLM | TRAE (GPT-5) | 17.9% | 9.0–32.7 |
| SGLang | TRAE (GPT-5) | 86.7% | 62.1–96.3 |
| SGLang | TRAE (Sonnet) | 80.0% | 54.8–93.0 |
| SGLang | Codex CLI | 80.0% | 54.8–93.0 |
| SGLang | OpenHands (GPT-5) | 33.3% | 15.2–58.3 |
| SGLang | Claude Code | 26.7% | 10.9–52.0 |
| SGLang | OpenHands (Sonnet-4.5) | 13.3% | 3.7–37.9 |

The paper's central cross-codebase claim survives significance testing in both
directions: on vLLM, Claude Code significantly outperforms Codex CLI (p=0.021) and
TRAE GPT-5 (p=0.007); on SGLang the ordering inverts and TRAE (Sonnet), Codex CLI,
and TRAE (GPT-5) each significantly outperform Claude Code (p=0.008, 0.008, 0.004).
Pairs that are not significant (e.g., agents sharing the same model class) will be
described as comparable, and we soften cross-codebase ranking prose wherever the
n=15 SGLang intervals overlap.

On measurement noise specifically: beyond the rollout variance already in Appendix G
(pass@1 vs pass@2, std ≤3.3%, ranking preserved), we analyzed per-task variability
across ~8 independent rollouts (Claude Code and Codex, 30 vLLM + 10 SGLang tasks,
each rollout benchmarked). Throughput varies by under 1% per task — well inside the
±5% classification threshold — while TTFT varies by 7–9% (median) on SGLang. We will
report this and note that TTFT-classified SGLang tasks carry higher single-run
uncertainty, which the confidence intervals above absorb.

Finally, on scale: as discussed in Appendix H, the constraint is the domain. GPU
inference optimization commits with reproducible setups, deterministic measurement,
and author-verified speedups are rare in upstream history; each retained task also
requires a Docker-buildable snapshot and H100 re-execution. General SWE benchmarks
can mine thousands of issue-fix pairs; measurable performance commits cannot be
scaled the same way without giving up exactly the hard-metric verifiability this
benchmark exists to provide.

**W2 (Level 2 not evaluated).** This is fair, and we will reposition Level 2
precisely. Every Level 2 task passed the same three-stage filtering and manual
curation as Level 1 — including reproduction of the benchmark configuration from
the PR discussion and verification that the commit is a genuine optimization. What
Level 2 lacks is agent rollouts and human-reference re-execution, because TP4/TP8
runs on A100/H100/H200 were beyond our compute budget. We will (a) state this
distinction explicitly in §3.2 rather than the appendix, and (b) frame Level 2 as a
curated data release accompanying the benchmark, with multi-GPU evaluation as
future work. If compute allows during the discussion period we will additionally
demonstrate the harness end-to-end on a small TP2 subset with the human reference
patches, but we do not want to overpromise runs we cannot land in the window.

**W3 (Two kinds of Lucky Win).** We think this is the most valuable suggestion in
the review, and the data to answer it already exists in our pipeline. Q3 cases
carry both an Implementation Approach label (Table 2) and a functional-correctness
result (§3.3.4, LM Evaluation Harness). Crossing them on vLLM separates exactly the
two cases the reviewer describes. Of the 12 Q3 cases with completed
functional-correctness evaluations:

| Outcome | Count |
|---|---|
| Correctness-preserving optimization at a different location ("emergent win") | 11 (2 judged Valid alternative) |
| Correctness-breaking speedup ("hacking the win") | 1 |

The single correctness-breaking case is the Bamba example already detailed in
Appendix E.3 (accuracy 32%→0%). We will add this decomposition as a table in the
revision, extend the correctness runs to the remaining Q3 cases, and adjust §5.2
accordingly: correctness-preserving Q3 cases with a Valid-alternative label
represent legitimate optimization work that our current framing undersells, and we
will say so.

**W4 (Hard metrics relative to the human patch).** To clarify: our evaluation
protocol already measures every patch against both the unoptimized baseline and the
human solution. §4.3 executes the benchmark commands against both; §3.3.4 measures
functional correctness for the baseline and the agent patch. The paper reports the
human-relative classification because agent-vs-human is the benchmark's scoring
axis, but the baseline measurements exist, and we have added a per-task appendix
table reporting absolute improvement over baseline for both the human patch and
each agent patch, so readers can see when "Similar" means matching a large gain
versus a modest one. We also note that manual curation (Appendix C.2) required an
author-verified performance claim in the source PR for task inclusion, which
excludes the degenerate case of certifying "Similar" against a near-zero human gain.

**W5 (Contamination and difficulty decay).** We agree and discuss this in
Appendix H. Two observations mitigate the concern for current results: execution
success is far from ceiling even though every reference patch is public, and the
dominant failure mode is Q2 — agents find the correct target but fail to implement
a working fix — which is inconsistent with verbatim recall of reference patches.
For durability we commit to a live-refresh protocol following SWE-bench-Goes-Live:
future releases add freshly merged PRs gated by a documented cutoff date, and
results are always reported against a dated split. We will elevate this from future
work to a stated maintenance commitment in §6.

Given the added statistical analysis and the Q3 decomposition, we hope the reviewer
will reconsider their assessment.

---

## Response to Reviewer kNyS (rating 3)

We thank the reviewer for the thorough summary and concrete questions.

**Limited scale.** Please see our response to Reviewer utqG (W1): we have added
Wilson 95% confidence intervals to all headline tables and pairwise exact McNemar
tests. The central cross-codebase inversion is significant in both directions
(Claude Code > Codex/TRAE-GPT5 on vLLM, p ≤ 0.021; TRAE/Codex > Claude Code on
SGLang, p ≤ 0.008). We agree that low-frequency outcomes (e.g., Q4 on SGLang)
cannot support strong claims at n=15 and we have softened those statements.

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
evaluation of additional frontier families (e.g., DeepSeek, Gemini) requires
substantial GPU re-benchmarking, which we commit to for the camera-ready.

**Model/scaffold decoupling (Q2).** The experimental design already implements the
controlled comparison the reviewer asks for: Claude Sonnet 4.5 is evaluated under
three scaffolds (Claude Code, OpenHands, TRAE) and GPT-5 under three scaffolds
(Codex CLI, OpenHands, TRAE) — a 2-model × 3-scaffold grid (Table 7, §5.5).
Holding the model fixed, True Success on vLLM spans 28.2%–46.2% across scaffolds;
holding the scaffold fixed (OpenHands), swapping Sonnet 4.5 for GPT-5 moves vLLM
True Success from 43.6% to 28.2%. The two effects are the same order of magnitude,
and the TRAE row reverses direction across codebases (Table 4). We will restructure
§5.5 around this grid explicitly so the decoupling reads as a designed comparison.

**Which scaffold components drive the gap (Q3).** Appendices E.8–E.9 already give a
controlled worked example: the same Bamba task under OpenHands (clean 296s run,
valid 165-line patch, judged a Valid alternative) versus TRAE GPT-5 (tool-call
emission failure, empty patch). For the revision we have aggregated step-level
trajectory features across tasks for the four configurations with step logs — TRAE
and OpenHands, each under both models (per-task medians on vLLM):

| Config | steps | edit actions | time to first edit | duration |
|---|---|---|---|---|
| TRAE (Sonnet) | 44 | 17 | 8 s | 502 s |
| TRAE (GPT-5) | 42 | 19 | 80 s | 1143 s |
| OpenHands (Sonnet-4.5) | 109 | 13 | 72 s | 388 s |
| OpenHands (GPT-5) | 79 | 9 | 324 s | 836 s |

Two mechanisms stand out: (i) holding the scaffold fixed, GPT-5 configurations take
2–4× longer to reach a first edit and to finish than Sonnet configurations; (ii)
OpenHands decomposes work into roughly twice as many, smaller steps and terminates
deliberately (94–100% of runs end with an explicit finish action), whereas TRAE
GPT-5 runs frequently truncate. For Claude Code we observe only inputs and outputs,
and we keep the Appendix H caveat that closed-source scaffolding limits
attribution; the open-scaffold analysis brackets the plausible mechanisms.

**Judge boundary cases (Q4).** The soft-metric categories are author-calibrated for
this benchmark, and the judge operates in a constrained reference-comparison
setting: it receives the human patch, the agent patch, and the task description
(Figure 19), and classifies whether they target the same bottleneck and take a
similar approach — it is not asked to assess code quality from prior knowledge. We
will add worked examples of the Same/Related/Different boundary to Appendix F in
the revision. We also ran the sensitivity check the reviewer suggests: counting
only Same target (identical locations) as correct, True Success drops substantially
in every configuration — e.g., Claude Code on vLLM 46.2%→20.5%; Codex CLI on SGLang
80.0%→26.7%; TRAE (GPT-5) on SGLang 86.7%→46.7% — because Related (same module) is
the most common correct-targeting label. We report this as the strict lower bound
and keep Same-or-Related as the primary definition: "same module" indicates the
agent located the right bottleneck neighborhood, and the targeting dimension is
precisely where LLM–human agreement is strongest (mean κ = 0.836, Table 8). We
prefer to disclose this sensitivity ourselves so readers can pick the definition
matching their use case.

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
added metrics are consistent with the headline results — agents are modestly worse
than the human reference on TPOT and ITL as well (e.g., median −7.2% TPOT for Codex
CLI on vLLM serving tasks) and near parity on throughput. One design constraint we
keep deliberately: we do not replace the PR's benchmark configuration with our own
composite workload, because comparability against the human author's claimed
improvement is what makes the human-relative classification meaningful. We will
make this tradeoff explicit in §3.3.1.

**Reliability of the soft metric.** The validation the reviewer asks for is in
Appendix F: mean LLM–human Cohen's κ of 0.836 (Bottleneck Targeting) and 0.880
(Implementation Approach) against two independent annotators across all 54 tasks,
plus eight-run stability with maximum deviation ±2.5% (Figure 20), and the full
judge prompt (Figure 19). On the specific concern about "alternative but sound"
solutions: the judge does not compare diff locations heuristically. It receives
both complete patches with the task description in a constrained
reference-comparison setting, and the taxonomy includes an explicit Valid
alternative category, which is well populated in practice (Figures 5–6) — Claude
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
concentrated in `python/sglang`. This makes the sparse-signal concern inspectable —
readers can see which areas are well covered and which are thin. We will also state
explicitly that the benchmark deliberately targets isolated, measurable
optimizations (<10 files, reproducible single-command benchmarks): a scope
decision, not a claim of covering the full bottleneck spectrum.
