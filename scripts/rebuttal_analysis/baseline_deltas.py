#!/usr/bin/env python3
"""utqG W4: absolute improvement over the unoptimized baseline for the human
patch and each agent patch.

vLLM: HF Inferencebench/claude-code-vllm-benchmarks parquet (baseline_*,
human_*, agent_* per commit x legacy agent, plus precomputed *_improvement_*
columns) filtered to the final 39 tasks; OpenHands agent values joined from
canonical hard_metrics.json against the same baseline.
SGLang: local EAD/benchmark_results/sglang/<commit>_isolated.json 3-way
variants (baseline/human/agent) for the 4 legacy agents; OpenHands agent
throughput from canonical hard_metrics vs the isolated baseline.

Output: docs/rebuttal_analysis/baseline_deltas.md / .json
"""

import json
import statistics
from pathlib import Path

import pandas as pd
from huggingface_hub import hf_hub_download

ROOT = Path(__file__).resolve().parents[2]
EAD = ROOT / "third-party" / "everything_analysis_data"
OUT = ROOT / "docs" / "rebuttal_analysis"

LEGACY_PARQUET_AGENT = {"claude_code": ["claude_code", "claude-code"],
                        "codex": ["codex"], "trae_gpt5": ["trae"],
                        "trae_sonnet": ["trae"]}
SGL_VARIANT = {"claude_code": "claude_code", "codex": "codex",
               "trae_gpt5": "trae_gpt5", "trae_sonnet": "trae_sonnet45"}
AGENTS = ["claude_code", "openhands_sonnet45", "trae_sonnet",
          "openhands_gpt5", "codex", "trae_gpt5"]


def pct_impr(base, val, lower_is_better):
    if base in (None, 0) or val is None or pd.isna(base) or pd.isna(val):
        return None
    return (base - val) / base * 100 if lower_is_better else (val - base) / base * 100


def main():
    final39 = [json.loads(l)["commit_hash"][:8]
               for l in open(EAD / "dataset" / "vllm.jsonl")]
    hard_v = json.load(open(EAD / "hard_metrics" / "vllm" / "hard_metrics.json"))
    hard_s = json.load(open(EAD / "hard_metrics" / "sglang" / "hard_metrics.json"))

    p = hf_hub_download("Inferencebench/claude-code-vllm-benchmarks",
                        "data/train-00000-of-00001.parquet", repo_type="dataset")
    df = pd.read_parquet(p)
    df["short"] = df["commit_hash"].str[:8]

    rows = []

    # ---------------- vLLM ----------------
    for c in final39:
        sub = df[df["short"] == c]
        if sub.empty:
            continue
        # Baseline + human are per-commit (identical across agent rows); take
        # the first row with a non-null baseline, else first row.
        base_rows = sub[sub[[col for col in sub.columns
                             if col.startswith("baseline_")
                             and col != "baseline_raw"]].notna().any(axis=1)]
        ref = (base_rows.iloc[0] if not base_rows.empty else sub.iloc[0])
        for metric, bcol, hcol, lower in [
                ("ttft", "baseline_ttft_mean", "human_ttft_mean", True),
                ("throughput", "baseline_throughput", "human_throughput", False),
                ("latency", "baseline_latency_avg", "human_latency_avg", True)]:
            h_impr = pct_impr(ref.get(bcol), ref.get(hcol), lower)
            if h_impr is None:
                continue
            row = {"project": "vllm", "commit": c, "metric": metric,
                   "human_improvement_pct": h_impr, "agents": {}}
            # Legacy agents from parquet rows
            for agent in ("claude_code", "codex", "trae_gpt5", "trae_sonnet"):
                names = LEGACY_PARQUET_AGENT[agent]
                arow = sub[sub["agent_name"].isin(names)]
                if agent.startswith("trae"):
                    model_key = "gpt" if agent == "trae_gpt5" else "sonnet"
                    arow = arow[arow["agent_model"].astype(str).str.lower()
                                .str.contains(model_key, na=False)]
                if not arow.empty:
                    acol = {"ttft": "agent_ttft_mean",
                            "throughput": "agent_throughput",
                            "latency": "agent_latency_avg"}[metric]
                    a_impr = pct_impr(ref.get(bcol), arow.iloc[0].get(acol), lower)
                    if a_impr is not None:
                        row["agents"][agent] = a_impr
            # OpenHands from canonical hard_metrics vs the same baseline
            for agent in ("openhands_gpt5", "openhands_sonnet45"):
                h = next((x for x in hard_v
                          if x.get("commit") == c and x.get("tool") == agent), None)
                if h:
                    aval = h.get("agent_ttft") if metric == "ttft" else (
                        h.get("agent_throughput") if metric == "throughput" else None)
                    a_impr = pct_impr(ref.get(bcol), aval, metric == "ttft")
                    if a_impr is not None:
                        row["agents"][agent] = a_impr
            rows.append(row)

    # ---------------- SGLang ----------------
    for f in sorted((EAD / "benchmark_results" / "sglang").glob("*_isolated.json")):
        data = json.load(open(f))
        variants = data.get("variants") or {}
        base = ((variants.get("baseline") or {}).get("metrics") or {})
        human = ((variants.get("human") or {}).get("metrics") or {})
        b_tp = base.get("output_token_throughput")
        h_impr = pct_impr(b_tp, human.get("output_token_throughput"), False)
        if h_impr is None:
            continue
        c = f.name.split("_")[0][:8]
        row = {"project": "sglang", "commit": c, "metric": "throughput",
               "human_improvement_pct": h_impr, "agents": {}}
        for agent, vname in SGL_VARIANT.items():
            m = ((variants.get(vname) or {}).get("metrics") or {})
            a_impr = pct_impr(b_tp, m.get("output_token_throughput"), False)
            if a_impr is not None:
                row["agents"][agent] = a_impr
        for agent in ("openhands_gpt5", "openhands_sonnet45"):
            h = next((x for x in hard_s
                      if x.get("commit") == c and x.get("tool") == agent), None)
            if h and h.get("agent_throughput"):
                a_impr = pct_impr(b_tp, h["agent_throughput"], False)
                if a_impr is not None:
                    row["agents"][agent] = a_impr
        rows.append(row)

    # Canonical primary metric per commit (majority across tools)
    primary = {}
    for h in hard_v + hard_s:
        pm = h.get("primary_metric")
        if pm in ("ttft", "throughput", "latency"):
            primary.setdefault(h["commit"], []).append(pm)
    primary = {c: max(set(v), key=v.count) for c, v in primary.items()}

    # ---------------- Summaries ----------------
    lines = ["# W4 — Absolute improvement over unoptimized baseline (generated)",
             "",
             "Caveats: (1) metric family per task = canonical `primary_metric` "
             "(majority across tools; falls back to whatever family has data); "
             "(2) vLLM OpenHands agent values come from the X3 re-bench context "
             "joined against the earlier parquet baseline — cross-context, treat "
             "as indicative only; legacy-agent values share the baseline's "
             "measurement context; (3) all numbers are single benchmark runs.", ""]
    for project in ("vllm", "sglang"):
        # one row per commit: canonical primary metric, else prefer ttft
        best = {}
        for r in rows:
            if r["project"] != project:
                continue
            pm = primary.get(r["commit"])
            pref = 0 if r["metric"] == pm else \
                {"ttft": 1, "throughput": 2, "latency": 3}[r["metric"]]
            if r["commit"] not in best or pref < best[r["commit"]][0]:
                best[r["commit"]] = (pref, r)
        chosen = [v[1] for v in best.values()]
        h_vals = [r["human_improvement_pct"] for r in chosen]
        n_total = 39 if project == "vllm" else 15
        lines += [f"## {project} — human patch vs baseline "
                  f"(coverage {len(chosen)}/{n_total} tasks)", ""]
        if h_vals:
            lines += [f"- human improvement over baseline: min {min(h_vals):.1f}%, "
                      f"median {statistics.median(h_vals):.1f}%, max {max(h_vals):.1f}%",
                      f"- tasks with human improvement > 5%: "
                      f"{sum(1 for v in h_vals if v > 5)}/{len(h_vals)}", ""]
        lines += ["| Agent | n | median agent-vs-baseline % | agent>+5% | agent<-5% |",
                  "|---|---|---|---|---|"]
        for agent in AGENTS:
            a_vals = [r["agents"][agent] for r in chosen if agent in r["agents"]]
            if not a_vals:
                lines.append(f"| {agent} | 0 | — | — | — |")
                continue
            lines.append(f"| {agent} | {len(a_vals)} | {statistics.median(a_vals):.1f} | "
                         f"{sum(1 for v in a_vals if v > 5)} | "
                         f"{sum(1 for v in a_vals if v < -5)} |")
        lines.append("")

    lines += ["## Per-commit detail", "",
              "| Project | Commit | Metric | Human % | " +
              " | ".join(AGENTS) + " |",
              "|---|---|---|---|" + "---|" * len(AGENTS)]
    for r in rows:
        cells = [f"{r['agents'][a]:.1f}" if a in r["agents"] else "—" for a in AGENTS]
        lines.append(f"| {r['project']} | {r['commit']} | {r['metric']} | "
                     f"{r['human_improvement_pct']:.1f} | " + " | ".join(cells) + " |")

    (OUT / "baseline_deltas.json").write_text(json.dumps(rows, indent=1))
    (OUT / "baseline_deltas.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines[:40]))
    print(f"... ({len(rows)} rows total; full detail in baseline_deltas.md)")


if __name__ == "__main__":
    main()
