#!/usr/bin/env python3
"""yX4G: dataset composition table from EAD/dataset/{vllm,sglang}.jsonl.

Output: docs/rebuttal_analysis/composition.md
"""

import json
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
EAD = ROOT / "third-party" / "everything_analysis_data"
OUT = ROOT / "docs" / "rebuttal_analysis"


def dist(vals):
    vals = sorted(vals)
    return (f"min {vals[0]}, p25 {vals[len(vals)//4]}, median {statistics.median(vals)}, "
            f"p75 {vals[3*len(vals)//4]}, max {vals[-1]}")


def main():
    lines = ["# yX4G — Dataset composition (generated from dataset JSONL fields)", ""]
    for project in ("vllm", "sglang"):
        recs = [json.loads(l) for l in open(EAD / "dataset" / f"{project}.jsonl")]
        n = len(recs)
        lines += [f"## {project} ({n} tasks)", ""]

        files_touched = [len(r.get("files_changed") or []) for r in recs]
        lines.append(f"- **Files touched per task:** {dist(files_touched)}")

        edited = [(r.get("stats") or {}).get("num_edited_lines", 0) for r in recs]
        non_test = [(r.get("stats") or {}).get("num_non_test_edited_lines", 0) for r in recs]
        hunks = [(r.get("stats") or {}).get("num_hunks", 0) for r in recs]
        lines.append(f"- **Edited lines per task:** {dist(edited)}")
        lines.append(f"- **Non-test edited lines per task:** {dist(non_test)}")
        lines.append(f"- **Hunks per task:** {dist(hunks)}")

        serving = sum(1 for r in recs if r.get("has_serving"))
        latency = sum(1 for r in recs if r.get("has_latency"))
        throughput = sum(1 for r in recs if r.get("has_throughput"))
        lm = sum(1 for r in recs if r.get("uses_lm_eval"))
        lines += [f"- **Benchmark mode:** serving {serving}, latency {latency}, "
                  f"throughput {throughput} (non-exclusive flags)",
                  f"- **Tasks with lm-eval correctness commands:** {lm}/{n}"]

        models = {}
        for r in recs:
            for m in (r.get("models") or []):
                models[m] = models.get(m, 0) + 1
        top = sorted(models.items(), key=lambda kv: -kv[1])[:8]
        lines.append(f"- **Distinct benchmark models:** {len(models)}; top: "
                     + ", ".join(f"{m} ({c})" for m, c in top))

        areas = {}
        for r in recs:
            for f in (r.get("files_changed") or []):
                top_dir = "/".join(f.split("/")[:2]) if "/" in f else f
                areas[top_dir] = areas.get(top_dir, 0) + 1
        top_areas = sorted(areas.items(), key=lambda kv: -kv[1])[:10]
        lines.append(f"- **Most-touched code areas (top-2-level dirs):** "
                     + ", ".join(f"`{a}` ({c})" for a, c in top_areas))
        lines.append("")

    lines += ["_Note: `performance_areas` / `difficulty` are not stored fields; "
              "bottleneck-category labels would require a fresh labeling pass "
              "(partial start: archive/misc/results/reviews/vllm_classification_review.csv)._"]
    (OUT / "composition.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
