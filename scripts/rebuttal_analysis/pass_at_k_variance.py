#!/usr/bin/env python3
"""utqG W1 add-on: per-task rollout variance from HF pass-at-k results.

Source: Inferencebench/iso-bench-pass-at-k-results (public parquet).
Output: docs/rebuttal_analysis/pass_at_k_variance.md
"""

import statistics
from pathlib import Path

import pandas as pd
from huggingface_hub import hf_hub_download

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "docs" / "rebuttal_analysis"

METRICS = [
    ("ttft_mean_ms", "TTFT mean (ms)", True),
    ("output_token_throughput_tok_s", "Output throughput (tok/s)", False),
    ("latency_avg_ms", "Latency avg (ms)", True),
]


def main():
    p = hf_hub_download("Inferencebench/iso-bench-pass-at-k-results",
                        "data/train-00000-of-00001.parquet", repo_type="dataset")
    df = pd.read_parquet(p)
    lines = ["# W1 add-on — Per-task rollout variance (pass-at-k, generated)", "",
             "Source: `Inferencebench/iso-bench-pass-at-k-results` — independent agent "
             "rollouts per task, each benchmarked once (captures agent stochasticity + "
             "benchmark noise). CV = std/mean across rollouts of the same task.", ""]
    for (repo, agent), sub in df.groupby(["repo", "agent_name"]):
        lines += [f"## {repo} × {agent} "
                  f"({sub['item_id'].nunique()} tasks, {len(sub)} rollouts)", ""]
        for col, label, _ in METRICS:
            cvs = []
            for _, task in sub.groupby("item_id"):
                vals = task[col].dropna()
                if len(vals) >= 3 and vals.mean() > 0:
                    cvs.append(vals.std() / vals.mean())
            if cvs:
                lines.append(f"- **{label}**: tasks with ≥3 rollouts: {len(cvs)}; "
                             f"median CV {100*statistics.median(cvs):.1f}%, "
                             f"p90 CV {100*sorted(cvs)[int(0.9*len(cvs))]:.1f}%")
        lines.append("")
    lines += ["Interpretation: rollout-to-rollout CV bounds how much of a "
              "single-rollout Beats/Similar/Worse classification (±5% threshold) "
              "could flip under resampling; report alongside the Wilson CIs."]
    (OUT / "pass_at_k_variance.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
