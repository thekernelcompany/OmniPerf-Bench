#!/usr/bin/env python3
"""Build the master per-(project, commit, agent) table for rebuttal analysis.

Reproduces the paper's quadrant assignments (Table 5) from canonical
soft_metrics.json + hard_metrics.json using the same logic as
EAD/scripts/generate_quadrant_findings.py, then verifies against the
published Table 3/4/5 numbers before anything downstream consumes it.

Output: docs/rebuttal_analysis/master.json
"""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
EAD = ROOT / "third-party" / "everything_analysis_data"
OUT_DIR = ROOT / "docs" / "rebuttal_analysis"

TOOL_ALIASES = {
    "codex": ["codex", "codex_cli"],
    "trae_sonnet": ["trae_sonnet", "trae_sonnet45"],
}

AGENT_ORDER = [
    "claude_code", "openhands_sonnet45", "trae_sonnet",
    "openhands_gpt5", "codex", "trae_gpt5",
]

# Published numbers (submitted PDF) used as the verification gate.
EXPECTED_Q = {
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


def hard_class(commit, tool, hard_metrics):
    for try_tool in TOOL_ALIASES.get(tool, [tool]):
        for h in hard_metrics:
            if h.get("commit") == commit and h.get("tool") == try_tool:
                return (
                    h.get("hard_classification", "NO_DATA"),
                    h.get("primary_metric"),
                    h.get("primary_pct"),
                    h.get("throughput_pct"),
                    h.get("ttft_pct"),
                    h.get("benchmark_mode"),
                )
    for h in hard_metrics:
        if h.get("commit") == commit and "tool" not in h:
            return (
                h.get("hard_classification", "NO_DATA"),
                h.get("primary_metric"), h.get("primary_pct"),
                h.get("throughput_pct"), h.get("ttft_pct"),
                h.get("benchmark_mode"),
            )
    return ("NO_DATA", None, None, None, None, None)


def quadrant(hard, target):
    correct = target in ("same_target", "related_target")
    success = hard in ("beats", "similar")
    if correct and success:
        return "Q1"
    if correct and not success:
        return "Q2"
    if not correct and success:
        return "Q3"
    return "Q4"


def main():
    rows = []
    for project in ("vllm", "sglang"):
        soft = json.load(open(EAD / "soft_metrics" / project / "soft_metrics.json"))
        hard = json.load(open(EAD / "hard_metrics" / project / "hard_metrics.json"))
        for e in soft:
            tool, commit = e["tool"], e["commit"]
            pq = e.get("patch_quality") or {}
            target = (pq.get("bottleneck_target") or {}).get("category", "unknown")
            approach = (pq.get("approach_comparison") or {}).get("category", "unknown")
            hc, pmetric, ppct, tp_pct, ttft_pct, mode = hard_class(commit, tool, hard)
            rows.append({
                "project": project, "commit": commit, "agent": tool,
                "hard": hc, "target": target, "approach": approach,
                "quadrant": quadrant(hc, target),
                "primary_metric": pmetric, "primary_pct": ppct,
                "throughput_pct": tp_pct, "ttft_pct": ttft_pct,
                "benchmark_mode": mode,
            })

    # Verification gate against the published Table 5.
    failures = []
    for (project, agent), expected in EXPECTED_Q.items():
        sub = [r for r in rows if r["project"] == project and r["agent"] == agent]
        got = tuple(sum(1 for r in sub if r["quadrant"] == q)
                    for q in ("Q1", "Q2", "Q3", "Q4"))
        n = len(sub)
        status = "OK" if got == expected else "MISMATCH"
        if got != expected:
            failures.append((project, agent, expected, got))
        print(f"{project:6s} {agent:20s} n={n:2d} Q1-Q4={got} expected={expected} {status}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUT_DIR / "master.json", "w") as f:
        json.dump(rows, f, indent=1)
    print(f"\nwrote {OUT_DIR/'master.json'} ({len(rows)} rows)")

    if failures:
        print(f"\nVERIFICATION FAILED for {len(failures)} cells", file=sys.stderr)
        sys.exit(1)
    print("VERIFICATION PASSED: master table reproduces published Table 5 exactly")


if __name__ == "__main__":
    main()
