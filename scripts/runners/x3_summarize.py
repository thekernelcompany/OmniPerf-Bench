#!/usr/bin/env python3
"""Build a single X3 summary table after the bench batch completes."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
RDIR = ROOT / "archive/results/2026-05-x3"

ROWS = []  # (commit, kind, agent, model, status, req_s, total_tok_s, output_tok_s, lat_s, source)


def add_from_bench(path: Path, kind: str):
    if not path.exists():
        return
    d = json.loads(path.read_text())
    m = d.get("metrics") or {}
    # For benchmark_throughput.py runs: prefer total_tok/s. For benchmark_latency.py
    # spec-dec runs: prefer gen_throughput_tok_s_mean.
    tot = m.get("throughput_total_tok_s") or m.get("gen_throughput_tok_s_mean")
    out = m.get("throughput_output_tok_s") or m.get("gen_throughput_tok_s_max")
    req = m.get("throughput_req_s")
    lat = m.get("avg_latency_s")
    ROWS.append((
        d.get("human_commit", path.stem.split("_")[0]),
        kind,
        d.get("model"),
        d.get("status"),
        req,
        tot,
        out,
        lat,
        f"{path.relative_to(RDIR)}",
    ))


def add_from_extract(path: Path, kind: str):
    d = json.loads(path.read_text())
    m = d.get("metrics") or {}
    ROWS.append((
        d.get("human_commit"),
        kind,
        d.get("model"),
        "success_extracted",
        None,
        m.get("derived_total_throughput_tok_s"),
        None,
        m.get("avg_latency_s"),
        d.get("extraction_rule"),
    ))


# bench results (6 commits each: baseline / OH-gpt5 / OH-sonnet45 if present)
for commit in ["3b61cb45", "8c1e77fb", "98f47f2a", "fa63e710", "6dd94dbe", "4c822298"]:
    for kind, sub in [("baseline", "baseline"),
                      ("openhands_gpt5", "openhands_gpt5"),
                      ("openhands_sonnet45", "openhands_sonnet45")]:
        add_from_bench(RDIR / sub / f"{commit}_agent_result.json", kind)

print(f"\n=== X3 results — {len(ROWS)} rows ===")
print(f"{'commit':>9s}  {'kind':>20s}  {'status':>20s}  {'req/s':>8s}  {'tot tok/s':>10s}  {'out tok/s':>10s}  {'lat_s':>8s}")
print("-" * 110)
for r in ROWS:
    commit, kind, model, status, req, tot, out, lat, src = r
    print(f"{commit:>9s}  {kind:>20s}  {(status or '-'):>20s}  "
          f"{(f'{req:.2f}' if req else '-'):>8s}  "
          f"{(f'{tot:.2f}' if tot else '-'):>10s}  "
          f"{(f'{out:.2f}' if out else '-'):>10s}  "
          f"{(f'{lat:.4f}' if lat else '-'):>8s}")
print()

# Per-commit comparison
print("\n=== Per-commit: agent vs baseline (X3 unified extractor) ===")
by_commit = {}
for r in ROWS:
    by_commit.setdefault(r[0], {})[r[1]] = r
for commit in ["3b61cb45", "8c1e77fb", "98f47f2a", "fa63e710", "6dd94dbe", "4c822298"]:
    if commit not in by_commit: continue
    c = by_commit[commit]
    base = c.get("baseline")
    base_tp = base[5] if base else None
    print(f"\n{commit}: baseline_total={base_tp}")
    for k in ("openhands_gpt5", "openhands_sonnet45"):
        if k not in c: continue
        agent_tp = c[k][5]
        if agent_tp and base_tp:
            delta = (agent_tp - base_tp) / base_tp * 100
            print(f"  {k}: {agent_tp} tok/s  ({delta:+.1f}% vs baseline)")
        else:
            print(f"  {k}: {agent_tp} tok/s  (no baseline comparison)")
