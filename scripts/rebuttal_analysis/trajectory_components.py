#!/usr/bin/env python3
"""kNyS Q3: trajectory component analysis for TRAE (x2 models, local) and
OpenHands (x2 models, HF rebuttal snapshots in the session scratchpad), plus
a coarse table for all six configs from run_summary/journal files.

Output: docs/rebuttal_analysis/trajectory_components.md / .json
"""

import json
import statistics
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
EAD = ROOT / "third-party" / "everything_analysis_data"
SCRATCH = Path("/tmp/claude-1000/-home-raven-coding-mess-kernel-corp-OmniPerf-Bench/"
               "446c5209-13e4-4a4c-8fd2-9eb4b2c26fbd/scratchpad/oh_trajectories")
OUT = ROOT / "docs" / "rebuttal_analysis"


def ts(s):
    if not s:
        return None
    try:
        return datetime.fromisoformat(s.replace("Z", "+00:00"))
    except ValueError:
        return None


def med(vals):
    vals = [v for v in vals if v is not None]
    return statistics.median(vals) if vals else None


def fmt(v, nd=1):
    return f"{v:.{nd}f}" if v is not None else "—"


# ---------------- TRAE ----------------
def trae_metrics(path):
    t = json.load(open(path))
    steps = t.get("agent_steps") or []
    if not steps:
        return None
    start = ts(t.get("start_time"))
    edit_calls = bash_calls = 0
    first_edit = None
    for s in steps:
        for c in (s.get("tool_calls") or []):
            name = c.get("name", "")
            if "edit" in name:
                edit_calls += 1
                if first_edit is None:
                    first_edit = ts(s.get("timestamp"))
            elif name == "bash":
                bash_calls += 1
    ttfe = (first_edit - start).total_seconds() if (first_edit and start) else None
    return {"steps": len(steps), "edit_calls": edit_calls, "bash_calls": bash_calls,
            "time_to_first_edit_s": ttfe,
            "duration_s": t.get("execution_time"),
            "finished": bool(t.get("success"))}


# ---------------- OpenHands ----------------
def oh_metrics(path):
    events = json.load(open(path))
    if not isinstance(events, list) or not events:
        return None
    t0 = ts(events[0].get("timestamp"))
    counts = {}
    first_edit = None
    finished = False
    tokens = None
    for e in events:
        a = e.get("action")
        if a:
            counts[a] = counts.get(a, 0) + 1
            if a == "edit" and first_edit is None:
                first_edit = ts(e.get("timestamp"))
            if a == "finish":
                finished = True
        lm = e.get("llm_metrics") or {}
        acc = lm.get("accumulated_token_usage") or {}
        if acc.get("completion_tokens"):
            tokens = acc
    tN = ts(events[-1].get("timestamp"))
    return {"steps": len(events),
            "edit_calls": counts.get("edit", 0),
            "bash_calls": counts.get("run", 0),
            "read_calls": counts.get("read", 0),
            "time_to_first_edit_s": (first_edit - t0).total_seconds() if (first_edit and t0) else None,
            "duration_s": (tN - t0).total_seconds() if (t0 and tN) else None,
            "finished": finished,
            "completion_tokens": (tokens or {}).get("completion_tokens")}


def collect_oh(tag):
    """Latest-timestamp trajectory per (repo, task)."""
    base = SCRATCH / tag / "runs"
    latest = {}
    for p in base.glob("*/*/*/trajectory.json"):
        repo, stamp, task = p.parts[-4], p.parts[-3], p.parts[-2]
        key = (repo, task)
        if key not in latest or stamp > latest[key][0]:
            latest[key] = (stamp, p)
    return latest


def main():
    results = {}

    for agent, d in [("trae_gpt5", EAD / "runs" / "vllm" / "trae_gpt5"),
                     ("trae_sonnet", EAD / "runs" / "vllm" / "trae_sonnet")]:
        ms = []
        for p in sorted(d.glob("*/trajectory.json")):
            m = trae_metrics(p)
            if m:
                ms.append(m)
        results[("vllm", agent)] = ms

    for tag, agent in [("gpt5", "openhands_gpt5"), ("sonnet45", "openhands_sonnet45")]:
        per_repo = {}
        for (repo, task), (_, p) in collect_oh(tag).items():
            m = oh_metrics(p)
            if m:
                per_repo.setdefault(repo, []).append(m)
        for repo, ms in per_repo.items():
            results[(repo, agent)] = ms

    lines = ["# kNyS Q3 — Trajectory component analysis (generated)", "",
             "TRAE from local curated trajectories (per-step timestamps + tool names); "
             "OpenHands from the HF rebuttal snapshots (event logs, latest timestamp per task). "
             "Claude Code and Codex CLI have no step logs — see the coarse table below. "
             "All values are per-config medians unless noted.", "",
             "| Repo | Config | n | steps | edits | bash/run | TTF-edit (s) | duration (s) | finished % |",
             "|---|---|---|---|---|---|---|---|---|"]
    order = [("vllm", "trae_sonnet"), ("vllm", "trae_gpt5"),
             ("vllm", "openhands_sonnet45"), ("vllm", "openhands_gpt5"),
             ("sglang", "openhands_sonnet45"), ("sglang", "openhands_gpt5")]
    for key in order:
        ms = results.get(key, [])
        if not ms:
            continue
        repo, agent = key
        fin = 100 * sum(1 for m in ms if m.get("finished")) / len(ms)
        lines.append(
            f"| {repo} | {agent} | {len(ms)} | {fmt(med([m['steps'] for m in ms]), 0)} | "
            f"{fmt(med([m['edit_calls'] for m in ms]), 0)} | "
            f"{fmt(med([m['bash_calls'] for m in ms]), 0)} | "
            f"{fmt(med([m['time_to_first_edit_s'] for m in ms]))} | "
            f"{fmt(med([m['duration_s'] for m in ms]))} | {fin:.0f} |")

    # ---------------- Coarse table (all six configs) ----------------
    def run_summaries(d):
        out = []
        for p in d.glob("*/run_summary.json"):
            try:
                r = json.load(open(p))
            except json.JSONDecodeError:
                continue
            if not (r.get("agent") or {}).get("duration_s"):
                jp = p.parent / "journal.json"
                if jp.exists():
                    try:
                        j = json.load(open(jp))
                        for blk in j.values():
                            if isinstance(blk, dict) and blk.get("duration_s"):
                                r.setdefault("agent", {})["duration_s"] = blk["duration_s"]
                                break
                    except json.JSONDecodeError:
                        pass
            out.append(r)
        return out

    lines += ["", "## Coarse per-run metrics (all six configs, vLLM)", "",
              "| Config | n | patch generated % | median duration (s) | median LOC added | median files changed |",
              "|---|---|---|---|---|---|"]
    coarse_dirs = {
        "claude_code": EAD / "runs" / "vllm" / "claude_code",
        "codex": EAD / "runs" / "vllm" / "codex_cli",
        "trae_sonnet": EAD / "runs" / "vllm" / "trae_sonnet",
        "trae_gpt5": EAD / "runs" / "vllm" / "trae_gpt5",
        "openhands_sonnet45": ROOT / "ISO-Bench" / "state" / "runs" / "vllm" / "openhands_sonnet45" / "flat",
        "openhands_gpt5": ROOT / "ISO-Bench" / "state" / "runs" / "vllm" / "openhands_gpt5" / "flat",
    }
    for agent, d in coarse_dirs.items():
        rs = run_summaries(d)
        if not rs:
            lines.append(f"| {agent} | 0 | — | — | — | — |")
            continue
        ag = [r.get("agent") or {} for r in rs]
        patched = 100 * sum(1 for a in ag if a.get("patch_generated")) / len(ag)
        dur = med([a.get("duration_s") for a in ag])
        loc = med([(a.get("patch_stats") or {}).get("lines_added") for a in ag])
        fc = med([(a.get("patch_stats") or {}).get("files_changed") for a in ag])
        lines.append(f"| {agent} | {len(rs)} | {patched:.0f} | {fmt(dur, 0)} | "
                     f"{fmt(loc, 0)} | {fmt(fc, 0)} |")
    lines += ["", "_\"finished %\" semantics differ per harness: TRAE = trajectory "
              "`success` flag; OpenHands = explicit `finish` action emitted. "
              "Not directly comparable across harnesses._"]

    serial = {f"{k[0]}/{k[1]}": v for k, v in results.items()}
    (OUT / "trajectory_components.json").write_text(json.dumps(serial, indent=1))
    (OUT / "trajectory_components.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
