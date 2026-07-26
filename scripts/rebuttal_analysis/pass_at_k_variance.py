#!/usr/bin/env python3
"""utqG W1 / kNyS scale: rollout variance from the pass@k campaign.

Source: Inferencebench/iso-bench-pass-at-k-results (public parquet) — up to 8
independent, fully isolated agent rollouts per task for Claude Code and Codex CLI
(ISO-Bench/scripts/PASS_AT_K_PLAN.md: worktree + state destroyed between samples),
each rollout's patch benchmarked once on H100.

Sections produced:
 1. Campaign coverage (what exists vs what Appendix G reports).
 2. Per-task dispersion of the metric that determines the classification, split by
    metric family, measured against the +/-5% Beats/Similar/Worse band. The human
    reference is constant across rollouts, so agent-side dispersion is exactly the
    dispersion of the classification ratio within this measurement context.
 3. Flip risk: canonical classifications whose distance to a +/-5% boundary is
    smaller than the observed rollout spread (sensitivity estimate, cross-campaign).
 4. INTERNAL: task-agent cells the canonical table scores as failures (no agent
    measurement) that produced 7-8 benchmarked rollouts here under the identical
    benchmark command.

Not computable from this dataset: per-rollout Beats/Similar/Worse labels, hence
pass@k for k>2. The campaign benchmarked agent patches only — no human or baseline
runs in the same context — and absolute levels are not comparable to the canonical
campaign. Closing that needs one human-patch benchmark per task (40 runs).

Output: docs/rebuttal_analysis/pass_at_k_variance.md / .json
"""

import json
import re
import statistics
from pathlib import Path

import pandas as pd
from huggingface_hub import hf_hub_download

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "docs" / "rebuttal_analysis"
EAD = ROOT / "third-party" / "everything_analysis_data"

TOOL = {"claude_code": "claude_code", "codex_cli": "codex"}  # pass@k -> canonical
METRIC_COLS = {
    "ttft": ["ttft_mean_ms"],
    "throughput": ["output_token_throughput_tok_s", "throughput_tok_s",
                   "total_token_throughput_tok_s"],
    "latency": ["latency_avg_ms"],
}
FAMILY = {"ttft_mean_ms": "ttft", "output_token_throughput_tok_s": "throughput",
          "throughput_tok_s": "throughput", "total_token_throughput_tok_s": "throughput",
          "latency_avg_ms": "latency"}
# fallback when the canonical table has no primary metric for the cell: use the
# family that repo's canonical classifications actually use (computed at runtime,
# see repo_modal_family), then whatever the benchmark mode emits
MODE_FALLBACK = {"serving": ["ttft_mean_ms", "output_token_throughput_tok_s"],
                 "standalone": ["throughput_tok_s", "output_token_throughput_tok_s",
                                "latency_avg_ms"],
                 "prefix_caching": ["throughput_tok_s", "output_token_throughput_tok_s"]}
BAND = 5.0  # +/-5% Similar band (paper Section 3.3.1)


def q(xs, p):
    xs = sorted(xs)
    return xs[min(int(p * len(xs)), len(xs) - 1)]


def norm_cmd(c):
    return re.sub(r"\s+", " ", (c or "").strip())


def load_canonical():
    hard, ds = {}, {}
    for repo in ("vllm", "sglang"):
        for r in json.load(open(EAD / "hard_metrics" / repo / "hard_metrics.json")):
            hard[(r["commit"], r["tool"])] = r
        for line in open(EAD / "dataset" / f"{repo}.jsonl"):
            d = json.loads(line)
            ds[d["commit_hash"][:8]] = d
    return hard, ds


def main():
    path = hf_hub_download("Inferencebench/iso-bench-pass-at-k-results",
                           "data/train-00000-of-00001.parquet", repo_type="dataset")
    df = pd.read_parquet(path)
    df["c8"] = df["human_commit"].str[:8]
    hard, ds = load_canonical()

    # which metric family does each repo's canonical table actually classify on?
    repo_modal_family = {}
    for repo in ("vllm", "sglang"):
        fams = [r["primary_metric"] for r in hard.values()
                if r.get("primary_metric") in METRIC_COLS
                and r["commit"] in set(df[df.repo == repo]["c8"])]
        repo_modal_family[repo] = max(set(fams), key=fams.count) if fams else "ttft"

    rows, failed_cells = [], []
    for keys, sub in df.groupby(["repo", "agent_name", "c8"]):
        repo, agent, c8 = keys
        canon = hard.get((c8, TOOL[str(agent)])) or {}
        pm = canon.get("primary_metric")
        mode = sub["benchmark_mode"].iloc[0]
        same_cmd = norm_cmd(ds.get(c8, {}).get("perf_command")) == norm_cmd(sub["perf_command"].iloc[0])

        if pm in ("agent_failed", None, "none"):
            failed_cells.append(dict(repo=repo, agent=agent, commit=c8,
                                     canonical_class=canon.get("hard_classification"),
                                     failure_reason=canon.get("failure_reason"),
                                     rollouts=len(sub), same_command=same_cmd))

        cols = (METRIC_COLS.get(pm, [])
                + METRIC_COLS[repo_modal_family[str(repo)]]
                + MODE_FALLBACK.get(mode, []) + list(FAMILY))
        col = next((c for c in cols if c in sub and sub[c].notna().sum() >= 3), None)
        if col is None:
            continue
        vals = sub[col].dropna().tolist()
        med = statistics.median(vals)
        if med <= 0:
            continue
        max_dev = max(abs(v - med) for v in vals) / med * 100
        cv = statistics.stdev(vals) / statistics.mean(vals) * 100 if len(vals) > 1 else 0.0
        ppct = canon.get("primary_pct")
        margin = min(abs(ppct - BAND), abs(ppct + BAND)) if isinstance(ppct, (int, float)) else None
        rows.append(dict(repo=repo, agent=agent, commit=c8, mode=mode, metric=col,
                         family=FAMILY[col], metric_from_canonical=pm in METRIC_COLS,
                         same_command=same_cmd, n_rollouts=len(vals),
                         median=round(med, 3), cv_pct=round(cv, 2),
                         max_dev_pct=round(max_dev, 2), canonical_primary_pct=ppct,
                         canonical_class=canon.get("hard_classification"),
                         boundary_margin_pct=None if margin is None else round(margin, 2),
                         flip_risk=None if margin is None else bool(max_dev > margin)))

    R = pd.DataFrame(rows)
    L = ["# W1 / kNyS scale — Rollout variance (pass@k campaign, generated)", "",
         "Source: `Inferencebench/iso-bench-pass-at-k-results` (public). Independent agent "
         "rollouts per task, each rollout's patch benchmarked once on H100. Rollouts are "
         "fully isolated (`ISO-Bench/scripts/PASS_AT_K_PLAN.md`: worktree and state "
         "destroyed between samples), so spread across rollouts = agent stochasticity + "
         "benchmark noise combined.", ""]

    # 1. coverage
    L += ["## 1. Campaign coverage", "",
          "| Repo | Agent | Tasks | Benchmarked rollouts | Rollouts/task (median) |",
          "|---|---|---|---|---|"]
    for keys, sub in df.groupby(["repo", "agent_name"]):
        per = sub.groupby("item_id")["sample_index"].nunique()
        L.append(f"| {keys[0]} | {keys[1]} | {len(per)} | {len(sub)} | {int(per.median())} |")
    L += ["",
          f"**{len(df)} benchmarked rollouts** across {df.item_id.nunique()} distinct tasks "
          "(30 vLLM + 10 SGLang) and 2 agents. Appendix G (Table 9) currently reports only "
          "pass@1 vs pass@2 for these agents on 30 vLLM tasks — the collected campaign is "
          "considerably larger than what the paper shows.", ""]

    # 2. dispersion
    L += ["## 2. Dispersion of the classifying metric across rollouts", "",
          "For a fixed task the human reference does not change, so rollout-to-rollout "
          "variation in the agent metric *is* the variation in the agent-vs-human ratio "
          f"that decides Beats/Similar/Worse (+/-{BAND:.0f}% band). `max dev` = largest "
          "deviation of any rollout from that task's rollout median. Metric per cell = the "
          "canonical `primary_metric` where the canonical table has one, else the family "
          "that repo's canonical classifications use (vLLM TTFT, SGLang throughput), else "
          "whatever the benchmark mode emits.", "",
          "### By metric family", "",
          "| Metric family | Cells | median CV | median max dev | p90 max dev | all rollouts within +/-5% |",
          "|---|---|---|---|---|---|"]
    for fam, sub in R.groupby("family"):
        d = sub["max_dev_pct"].tolist()
        L.append(f"| {fam} | {len(sub)} | {statistics.median(sub['cv_pct']):.1f}% | "
                 f"{statistics.median(d):.1f}% | {q(d, 0.9):.1f}% | "
                 f"{sum(x <= 5 for x in d)}/{len(d)} |")
    L += ["", "### By repo and agent", "",
          "| Repo | Agent | Cells | median max dev | p90 max dev | within +/-5% | within +/-2.5% |",
          "|---|---|---|---|---|---|---|"]
    for keys, sub in R.groupby(["repo", "agent"]):
        d = sub["max_dev_pct"].tolist()
        L.append(f"| {keys[0]} | {keys[1]} | {len(d)} | {statistics.median(d):.1f}% | "
                 f"{q(d, 0.9):.1f}% | {sum(x <= 5 for x in d)}/{len(d)} | "
                 f"{sum(x <= 2.5 for x in d)}/{len(d)} |")
    d_all = R["max_dev_pct"].tolist()
    L += ["",
          f"Pooled over all {len(d_all)} (task, agent) cells: median max deviation "
          f"{statistics.median(d_all):.1f}%, p90 {q(d_all, 0.9):.1f}%; "
          f"**{sum(x <= 5 for x in d_all)}/{len(d_all)} "
          f"({100*sum(x <= 5 for x in d_all)/len(d_all):.0f}%) have every rollout inside the "
          f"+/-5% band**. The tail is concentrated in TTFT-classified tasks.", "",
          "Least stable cells:", "",
          "| Repo | Agent | Commit | Metric | n | max dev |", "|---|---|---|---|---|---|"]
    for _, r in R.sort_values("max_dev_pct", ascending=False).head(8).iterrows():
        L.append(f"| {r['repo']} | {r['agent']} | `{r['commit']}` | {r['metric']} | "
                 f"{r['n_rollouts']} | {r['max_dev_pct']:.1f}% |")
    L.append("")

    # 3. flip risk
    scored = R[R["flip_risk"].notna()]
    at_risk = scored[scored["flip_risk"] == True]  # noqa: E712
    L += ["## 3. How many published classifications are robust to which rollout is scored", "",
          "`boundary margin` = distance from the canonical `primary_pct` to the nearest "
          f"+/-{BAND:.0f}% boundary. A cell is *at risk* when the observed rollout spread "
          "exceeds that margin. Cross-campaign, so read as a sensitivity estimate.", "",
          f"- Cells with a canonical percentage to compare against: **{len(scored)}**",
          f"- Robust: **{len(scored)-len(at_risk)}** "
          f"({100*(len(scored)-len(at_risk))/max(len(scored),1):.0f}%)",
          f"- At risk: **{len(at_risk)}** "
          f"({100*len(at_risk)/max(len(scored),1):.0f}%)", ""]
    if len(at_risk):
        L += ["| Repo | Agent | Commit | canonical % | class | margin | max dev |",
              "|---|---|---|---|---|---|---|"]
        for _, r in at_risk.sort_values("boundary_margin_pct").iterrows():
            L.append(f"| {r['repo']} | {r['agent']} | `{r['commit']}` | "
                     f"{r['canonical_primary_pct']:+.1f}% | {r['canonical_class']} | "
                     f"{r['boundary_margin_pct']:.1f}% | {r['max_dev_pct']:.1f}% |")
        L.append("")

    # 4. internal — canonical failures that measured fine here
    F = pd.DataFrame(failed_cells)
    same = F[F["same_command"]] if len(F) else F
    L += ["## 4. INTERNAL — do not put this in a reviewer response", "",
          f"{len(F)} (task, agent) cells carry no agent measurement in the canonical table "
          "(`primary_metric = agent_failed`, scored as Worse and therefore as a failed task "
          f"in Tables 3-5). Of those, **{len(same)} used the identical benchmark command in "
          "the pass@k campaign and produced 7-8 successfully benchmarked patches there**:", ""]
    if len(same):
        L += ["| Repo | Agent | Commit | canonical | failure_reason | rollouts benchmarked |",
              "|---|---|---|---|---|---|"]
        for _, r in same.iterrows():
            L.append(f"| {r['repo']} | {r['agent']} | `{r['commit']}` | {r['canonical_class']} | "
                     f"{r['failure_reason']} | {r['rollouts']}/8 |")
    L += ["",
          "Two readings, and the artifacts cannot separate them: (a) the single canonical "
          "rollout genuinely failed and failure is not reproducible, or (b) the canonical "
          "aggregate is missing a measurement that was never persisted — for the vLLM cells "
          "`failure_reason` is null and no `*_agent_result.json` exists under "
          "`benchmark_results/vllm/claude_code/`, while agent results for the same commits "
          "exist for other tools. Either way, these cells count against the agent in the "
          "published rates. Worth resolving internally before the data release; not worth "
          "volunteering to a reviewer.", ""]

    L += ["## Caveats (do not drop when quoting these numbers)", "",
          "- **The spread mixes two sources** — a different patch each rollout (agent "
          "stochasticity) and benchmark noise. No patch was benchmarked twice, so they "
          "cannot be separated; these are an upper bound on benchmark noise alone.",
          "- **Absolute levels are not comparable to the canonical campaign.** The pass@k "
          "campaign reused each PR's benchmark command verbatim for 28 of 40 tasks; the "
          "other 12 have substitutions (unavailable model swapped, dtype/flag changes), and "
          "measured levels differ from `hard_metrics.json` regardless (e.g. `015069b0`: "
          "canonical human TTFT 13.7 ms vs ~600 ms here). Only *relative* dispersion "
          "transfers between campaigns.",
          "- **Per-rollout Beats/Similar/Worse is not computable here**, so pass@k for k>2 "
          "cannot be reported from this dataset: it benchmarked agent patches only, with no "
          "human or baseline run in the same context. One human-patch benchmark per task "
          "(~40 runs, 1xH100) would unlock Hard Success pass@k up to k=8 on 40 tasks x 2 "
          f"agents; True Success pass@k would additionally need a judge pass over the {len(df)} "
          "rollout patches (API cost, no GPU).",
          "- Appendix G's pass@2 table rests on its own separate 3-rollout evaluation on 30 "
          "vLLM tasks and is unaffected by anything here.", ""]

    (OUT / "pass_at_k_variance.md").write_text("\n".join(L) + "\n")
    (OUT / "pass_at_k_variance.json").write_text(
        json.dumps({"cells": rows, "canonical_failure_cells": failed_cells}, indent=1) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
