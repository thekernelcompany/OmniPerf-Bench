#!/usr/bin/env python3
"""utqG W3: decompose Q3 (Lucky Win) cases into correctness-breaking
("hacking the win") vs correctness-preserving ("emergent win").

Sources:
- docs/rebuttal_analysis/master.json      — all Q3 cases (canonical, 6 agents)
- third-party/vllm-lm-eval/consolidated/  — GSM8K lm-eval results (legacy 4 agents, vLLM)

Output: docs/rebuttal_analysis/q3_decomposition.md / .json
"""

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CONS = ROOT / "third-party" / "vllm-lm-eval" / "consolidated"
OUT = ROOT / "docs" / "rebuttal_analysis"

LABEL = {
    "claude_code": "Claude Code", "openhands_sonnet45": "OpenHands (Sonnet-4.5)",
    "trae_sonnet": "TRAE (Sonnet)", "openhands_gpt5": "OpenHands (GPT-5)",
    "codex": "Codex CLI", "trae_gpt5": "TRAE (GPT-5)",
}
REGRESSION_PP = 0.05  # strict-match accuracy drop beyond this = broken


def load_jsonl(path):
    recs = []
    bad = 0
    for line in open(path):
        line = line.strip()
        if not line:
            continue
        try:
            recs.append(json.loads(line))
        except json.JSONDecodeError:
            bad += 1
    if bad:
        print(f"  [warn] {path.name}: skipped {bad} malformed line(s)")
    return recs


def _find_strict(obj):
    if isinstance(obj, dict):
        for k, v in obj.items():
            if "strict" in k and isinstance(v, (int, float)):
                return v
            found = _find_strict(v)
            if found is not None:
                return found
    return None


def strict(rec):
    for key in ("results", "metrics"):
        found = _find_strict(rec.get(key) or {})
        if found is not None:
            return found
    return None


def main():
    master = json.load(open(OUT / "master.json"))
    q3 = [r for r in master if r["quadrant"] == "Q3"]

    # Latest successful agent eval per (commit, agent)
    agent_eval = {}
    for rec in load_jsonl(CONS / "q3_agent_eval_summary.jsonl"):
        if rec.get("status") != "success":
            continue
        key = (rec["commit"], rec["agent"])
        if key not in agent_eval or rec.get("timestamp", "") > agent_eval[key].get("timestamp", ""):
            agent_eval[key] = rec

    # Latest successful baseline eval per commit
    base_eval = {}
    for rec in load_jsonl(CONS / "all_commits_summary.jsonl"):
        if rec.get("status") != "success" or rec.get("image_type") != "baseline":
            continue
        c = rec["commit"]
        if c not in base_eval or rec.get("timestamp", "") > base_eval[c].get("timestamp", ""):
            base_eval[c] = rec

    # Fallback: q2_accuracy_comparison carries per-commit baseline_strict
    base_strict_fallback = {}
    for rec in load_jsonl(CONS / "q2_accuracy_comparison.jsonl"):
        if rec.get("baseline_strict") is not None:
            base_strict_fallback.setdefault(rec["commit"], rec["baseline_strict"])

    # Alias: master uses 'codex'; lm-eval files may use 'codex' or 'codex_cli'
    def eval_for(commit, agent):
        for a in {"codex": ["codex", "codex_cli"],
                  "trae_sonnet": ["trae_sonnet", "trae_sonnet45"]}.get(agent, [agent]):
            if (commit, a) in agent_eval:
                return agent_eval[(commit, a)]
        return None

    rows = []
    for r in sorted(q3, key=lambda x: (x["project"], x["agent"], x["commit"])):
        ev = eval_for(r["commit"], r["agent"]) if r["project"] == "vllm" else None
        base = base_eval.get(r["commit"]) if r["project"] == "vllm" else None
        a_acc = strict(ev) if ev else None
        b_acc = strict(base) if base else None
        if b_acc is None and r["project"] == "vllm":
            b_acc = base_strict_fallback.get(r["commit"])
        if a_acc is not None and b_acc is not None:
            verdict = "BROKEN" if a_acc < b_acc - REGRESSION_PP else "PRESERVED"
        else:
            verdict = "NO_EVAL_DATA"
        rows.append({
            "project": r["project"], "commit": r["commit"], "agent": r["agent"],
            "hard": r["hard"], "target": r["target"], "approach": r["approach"],
            "baseline_acc": b_acc, "agent_acc": a_acc, "correctness": verdict,
        })

    # Cross-tab: approach × correctness
    lines = ["# W3 — Q3 (Lucky Win) decomposition (generated)", "",
             f"All Q3 cases from canonical quadrants: {len(rows)} "
             f"({sum(1 for r in rows if r['project']=='vllm')} vLLM, "
             f"{sum(1 for r in rows if r['project']=='sglang')} SGLang).",
             "Correctness = GSM8K strict-match, agent patch vs unoptimized baseline "
             f"(broken if drop > {REGRESSION_PP:.0%} pp). Coverage: legacy 4-agent vLLM cases only; "
             "OpenHands and SGLang cases lack lm-eval runs (marked NO_EVAL_DATA).", "",
             "| Project | Commit | Agent | Hard | Target | Approach | Base acc | Agent acc | Correctness |",
             "|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        b = f"{r['baseline_acc']:.2f}" if r["baseline_acc"] is not None else "—"
        a = f"{r['agent_acc']:.2f}" if r["agent_acc"] is not None else "—"
        lines.append(f"| {r['project']} | {r['commit']} | {LABEL[r['agent']]} | {r['hard']} | "
                     f"{r['target']} | {r['approach']} | {b} | {a} | {r['correctness']} |")

    lines += ["", "## Cross-tab (evaluated cases)", "",
              "| Approach | PRESERVED (emergent win) | BROKEN (hacking the win) | NO_EVAL_DATA |",
              "|---|---|---|---|"]
    approaches = sorted({r["approach"] for r in rows})
    for ap in approaches:
        sub = [r for r in rows if r["approach"] == ap]
        p = sum(1 for r in sub if r["correctness"] == "PRESERVED")
        b = sum(1 for r in sub if r["correctness"] == "BROKEN")
        n = sum(1 for r in sub if r["correctness"] == "NO_EVAL_DATA")
        lines.append(f"| {ap} | {p} | {b} | {n} |")

    ev_rows = [r for r in rows if r["correctness"] != "NO_EVAL_DATA"]
    pres = sum(1 for r in ev_rows if r["correctness"] == "PRESERVED")
    lines += ["", f"**Summary:** of {len(ev_rows)} evaluated Q3 cases, {pres} preserved correctness "
              f"(emergent wins) and {len(ev_rows)-pres} broke it (hacking the win). "
              f"{sum(1 for r in rows if r['correctness']=='NO_EVAL_DATA')} cases lack eval data "
              "(OpenHands vLLM + all SGLang)."]

    (OUT / "q3_decomposition.json").write_text(json.dumps(rows, indent=1))
    (OUT / "q3_decomposition.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
