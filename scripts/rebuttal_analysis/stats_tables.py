#!/usr/bin/env python3
"""utqG W1 + kNyS Q4b: CIs, pairwise McNemar, Same-only sensitivity.

Reads docs/rebuttal_analysis/master.json (built by build_master.py).
Outputs docs/rebuttal_analysis/stats_tables.md and .json.

Notes
- Canonical (post-X3) data is primary. For vLLM OpenHands (Sonnet-4.5) the
  canonical quadrants differ from the submitted Table 5 (Q1 22 vs 17); CIs
  for the published counts are shown alongside for that cell.
- Wilson 95% CI (closed form) + percentile bootstrap over the task axis
  (10k resamples). McNemar is exact (binomial on discordant pairs).
"""

import json
import math
import random
from pathlib import Path

from scipy.stats import binomtest

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "docs" / "rebuttal_analysis"

AGENTS = [
    "claude_code", "openhands_sonnet45", "trae_sonnet",
    "openhands_gpt5", "codex", "trae_gpt5",
]
LABEL = {
    "claude_code": "Claude Code", "openhands_sonnet45": "OpenHands (Sonnet-4.5)",
    "trae_sonnet": "TRAE (Sonnet)", "openhands_gpt5": "OpenHands (GPT-5)",
    "codex": "Codex CLI", "trae_gpt5": "TRAE (GPT-5)",
}
# Published Table 5 counts (Q1,Q2,Q3,Q4) for the published-CI columns.
PUBLISHED = {
    ("vllm", "claude_code"): (18, 15, 4, 2),
    ("vllm", "openhands_sonnet45"): (17, 15, 4, 3),
    ("vllm", "trae_sonnet"): (11, 20, 2, 6),
    ("vllm", "openhands_gpt5"): (11, 16, 6, 6),
    ("vllm", "codex"): (8, 20, 5, 6),
    ("vllm", "trae_gpt5"): (7, 27, 1, 4),
    ("sglang", "claude_code"): (4, 8, 3, 0),
    ("sglang", "openhands_sonnet45"): (2, 12, 0, 1),
    ("sglang", "trae_sonnet"): (12, 3, 0, 0),
    ("sglang", "openhands_gpt5"): (5, 8, 1, 1),
    ("sglang", "codex"): (12, 3, 0, 0),
    ("sglang", "trae_gpt5"): (13, 2, 0, 0),
}


def wilson(k, n, z=1.959964):
    if n == 0:
        return (float("nan"),) * 2
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return max(0.0, c - h), min(1.0, c + h)


def boot_ci(indicators, n_boot=10000, seed=0):
    rng = random.Random(seed)
    n = len(indicators)
    stats = sorted(
        sum(indicators[rng.randrange(n)] for _ in range(n)) / n
        for _ in range(n_boot)
    )
    return stats[int(0.025 * n_boot)], stats[int(0.975 * n_boot) - 1]


def mcnemar_exact(a_wins_b_loses, b_wins_a_loses):
    n_disc = a_wins_b_loses + b_wins_a_loses
    if n_disc == 0:
        return 1.0
    return binomtest(a_wins_b_loses, n_disc, 0.5).pvalue


def main():
    master = json.load(open(OUT / "master.json"))
    out = {"cells": [], "mcnemar": [], "same_only": []}
    lines = ["# W1 / Q4b — Statistical tables (generated)", ""]

    # ---- Per-cell CIs -------------------------------------------------
    lines += ["## True Success & Hard Success with 95% CIs",
              "",
              "Canonical = post-X3 data (reproduces submitted Table 5 for 11/12 cells; "
              "vLLM OpenHands-Sonnet-4.5 canonical Q1=22 vs published 17 — published-count "
              "CIs shown in brackets for that cell).",
              "",
              "| Project | Agent | n | True Succ % | Wilson 95% | Bootstrap 95% | Hard Succ % | Wilson 95% | Gap % |",
              "|---|---|---|---|---|---|---|---|---|"]
    for project in ("vllm", "sglang"):
        for agent in AGENTS:
            sub = [r for r in master if r["project"] == project and r["agent"] == agent]
            n = len(sub)
            ts = [1 if r["quadrant"] == "Q1" else 0 for r in sub]
            hs = [1 if r["quadrant"] in ("Q1", "Q3") else 0 for r in sub]
            k_ts, k_hs = sum(ts), sum(hs)
            wl, wu = wilson(k_ts, n)
            bl, bu = boot_ci(ts)
            hwl, hwu = wilson(k_hs, n)
            row = {
                "project": project, "agent": agent, "n": n,
                "true_success": k_ts / n, "ts_wilson": [wl, wu], "ts_boot": [bl, bu],
                "hard_success": k_hs / n, "hs_wilson": [hwl, hwu],
                "gap": (k_hs - k_ts) / n,
            }
            pub = PUBLISHED[(project, agent)]
            pub_ts, pub_hs = pub[0], pub[0] + pub[2]
            note = ""
            if (pub_ts, pub_hs) != (k_ts, k_hs):
                pwl, pwu = wilson(pub_ts, n)
                row["published_ts"] = pub_ts / n
                row["published_ts_wilson"] = [pwl, pwu]
                note = f" [pub {100*pub_ts/n:.1f}, W {100*pwl:.1f}–{100*pwu:.1f}]"
            out["cells"].append(row)
            lines.append(
                f"| {project} | {LABEL[agent]} | {n} | {100*k_ts/n:.1f}{note} | "
                f"{100*wl:.1f}–{100*wu:.1f} | {100*bl:.1f}–{100*bu:.1f} | "
                f"{100*k_hs/n:.1f} | {100*hwl:.1f}–{100*hwu:.1f} | {100*(k_hs-k_ts)/n:.1f} |")

    # ---- Pairwise McNemar on True Success (canonical, paired by task) --
    lines += ["", "## Pairwise McNemar (exact, True Success indicator, paired by task)",
              "",
              "| Project | A | B | A-only | B-only | p (exact) |",
              "|---|---|---|---|---|---|"]
    for project in ("vllm", "sglang"):
        per_agent = {}
        for agent in AGENTS:
            per_agent[agent] = {
                r["commit"]: 1 if r["quadrant"] == "Q1" else 0
                for r in master if r["project"] == project and r["agent"] == agent
            }
        for i, a in enumerate(AGENTS):
            for b in AGENTS[i + 1:]:
                commits = sorted(set(per_agent[a]) & set(per_agent[b]))
                a_only = sum(1 for c in commits if per_agent[a][c] and not per_agent[b][c])
                b_only = sum(1 for c in commits if per_agent[b][c] and not per_agent[a][c])
                p = mcnemar_exact(a_only, b_only)
                out["mcnemar"].append({"project": project, "a": a, "b": b,
                                       "a_only": a_only, "b_only": b_only, "p": p})
                mark = "**" if p < 0.05 else ""
                lines.append(f"| {project} | {LABEL[a]} | {LABEL[b]} | {a_only} | {b_only} | {mark}{p:.3f}{mark} |")

    # ---- Same-only sensitivity (kNyS Q4b) ------------------------------
    lines += ["", "## Same-only sensitivity (Related target NOT counted as correct)",
              "",
              "| Project | Agent | True Succ (Same∨Related) % | True Succ (Same-only) % | Δ (pp) |",
              "|---|---|---|---|---|"]
    for project in ("vllm", "sglang"):
        for agent in AGENTS:
            sub = [r for r in master if r["project"] == project and r["agent"] == agent]
            n = len(sub)
            base = sum(1 for r in sub if r["quadrant"] == "Q1")
            strict = sum(1 for r in sub
                         if r["target"] == "same_target" and r["hard"] in ("beats", "similar"))
            out["same_only"].append({"project": project, "agent": agent,
                                     "base": base / n, "strict": strict / n})
            lines.append(f"| {project} | {LABEL[agent]} | {100*base/n:.1f} | "
                         f"{100*strict/n:.1f} | {100*(strict-base)/n:+.1f} |")

    (OUT / "stats_tables.json").write_text(json.dumps(out, indent=1))
    (OUT / "stats_tables.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
