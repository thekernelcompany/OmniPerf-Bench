#!/usr/bin/env python3
"""yX4G: multi-metric (TTFT/TPOT/ITL/throughput) agent-vs-human coverage and
medians for the serving subset.

vLLM: HF claude-code-vllm-benchmarks parquet (per-metric agent_vs_human_*
precomputed for the legacy agents; positive = agent better).
SGLang: local *_isolated.json variants for all measured configs.

Output: docs/rebuttal_analysis/multimetric.md
"""

import json
import statistics
from pathlib import Path

import pandas as pd
from huggingface_hub import hf_hub_download

ROOT = Path(__file__).resolve().parents[2]
EAD = ROOT / "third-party" / "everything_analysis_data"
OUT = ROOT / "docs" / "rebuttal_analysis"

SGL_VARIANT = {"claude_code": "claude_code", "codex": "codex",
               "trae_gpt5": "trae_gpt5", "trae_sonnet": "trae_sonnet45"}


def med(vals):
    vals = [v for v in vals if v is not None and not pd.isna(v)]
    return (statistics.median(vals), len(vals)) if vals else (None, 0)


def fmt(pair):
    v, n = pair
    return f"{v:+.1f} (n={n})" if v is not None else "— (0)"


def main():
    final39 = {json.loads(l)["commit_hash"][:8] for l in open(EAD / "dataset" / "vllm.jsonl")}
    p = hf_hub_download("Inferencebench/claude-code-vllm-benchmarks",
                        "data/train-00000-of-00001.parquet", repo_type="dataset")
    df = pd.read_parquet(p)
    df["short"] = df["commit_hash"].str[:8]
    df = df[df["short"].isin(final39)]

    lines = ["# yX4G — Multi-metric agent-vs-human (serving subset, generated)", "",
             "Median % delta vs human patch (positive = agent better). Coverage varies "
             "per metric because each task runs the PR's own benchmark command.", "",
             "## vLLM (legacy agents, HF benchmark table, 39-task set)", "",
             "| Agent | TTFT mean | TPOT mean | ITL mean | Throughput | Latency avg |",
             "|---|---|---|---|---|---|"]
    def delta(sub, hcol, acol, lower_is_better):
        vals = []
        for _, r in sub.iterrows():
            h, a = r.get(hcol), r.get(acol)
            if h and a and not pd.isna(h) and not pd.isna(a):
                vals.append((h - a) / h * 100 if lower_is_better else (a - h) / h * 100)
        return med(vals)

    for agent_sel, label in [(("claude_code", "claude-code"), "claude_code"),
                             (("codex",), "codex"),
                             (("trae",), "trae (both models)")]:
        sub = df[df["agent_name"].isin(list(agent_sel))]
        lines.append(f"| {label} | "
                     f"{fmt(delta(sub, 'human_ttft_mean', 'agent_ttft_mean', True))} | "
                     f"{fmt(delta(sub, 'human_tpot_mean', 'agent_tpot_mean', True))} | "
                     f"{fmt(delta(sub, 'human_itl_mean', 'agent_itl_mean', True))} | "
                     f"{fmt(delta(sub, 'human_throughput', 'agent_throughput', False))} | "
                     f"{fmt(delta(sub, 'human_latency_avg', 'agent_latency_avg', True))} |")

    lines += ["", "## SGLang (isolated 3-way files, 15-task set)", "",
              "| Agent | TTFT mean | ITL mean | Output throughput |",
              "|---|---|---|---|"]
    per_agent = {a: {"ttft": [], "itl": [], "tp": []} for a in SGL_VARIANT}
    for f in sorted((EAD / "benchmark_results" / "sglang").glob("*_isolated.json")):
        variants = (json.load(open(f)).get("variants") or {})
        h = (variants.get("human") or {}).get("metrics") or {}
        for agent, vname in SGL_VARIANT.items():
            a = (variants.get(vname) or {}).get("metrics") or {}
            if h.get("mean_ttft_ms") and a.get("mean_ttft_ms"):
                per_agent[agent]["ttft"].append(
                    (h["mean_ttft_ms"] - a["mean_ttft_ms"]) / h["mean_ttft_ms"] * 100)
            if h.get("mean_itl_ms") and a.get("mean_itl_ms"):
                per_agent[agent]["itl"].append(
                    (h["mean_itl_ms"] - a["mean_itl_ms"]) / h["mean_itl_ms"] * 100)
            if h.get("output_token_throughput") and a.get("output_token_throughput"):
                per_agent[agent]["tp"].append(
                    (a["output_token_throughput"] - h["output_token_throughput"])
                    / h["output_token_throughput"] * 100)
    for agent, d in per_agent.items():
        lines.append(f"| {agent} | {fmt(med(d['ttft']))} | {fmt(med(d['itl']))} | "
                     f"{fmt(med(d['tp']))} |")
    lines += ["", "_SGLang benchmark_serving emits ITL but no TPOT; OpenHands SGLang runs "
              "were benchmarked separately (throughput extractor) and are not in the "
              "isolated 3-way files._"]

    (OUT / "multimetric.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
