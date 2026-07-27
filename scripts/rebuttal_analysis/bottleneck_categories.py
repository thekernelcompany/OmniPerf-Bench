#!/usr/bin/env python3
"""yX4G task-selection/representativeness: bottleneck-family distribution of Level 1.

The pipeline never stored a bottleneck-area label (no `performance_areas` field
exists anywhere), so this is a fresh labeling pass over the 54 Level-1 commits,
assigned from the commit subject and the touched paths. One primary family per task.

Output: docs/rebuttal_analysis/bottleneck_categories.md
"""

import json
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
EAD = ROOT / "third-party" / "everything_analysis_data"
OUT = ROOT / "docs" / "rebuttal_analysis"

# commit -> bottleneck family (assigned from commit subject + affected paths)
LABELS = {
    # ---- vLLM (39) ----
    "299ebb62": "Sampling and logits", "30172b49": "Sampling and logits",
    "35fad35a": "Sampling and logits", "99abb8b6": "Sampling and logits",
    "a3223766": "Sampling and logits", "d7740ea4": "Sampling and logits",
    "2deb029d": "KV cache and block management", "3476ed08": "KV cache and block management",
    "660470e5": "KV cache and block management", "9474e89b": "KV cache and block management",
    "6e36f4fa": "Scheduling and batching", "ad8d696a": "Scheduling and batching",
    "e3580537": "Scheduling and batching", "fa63e710": "Scheduling and batching",
    "3b61cb45": "Attention kernels and backends", "8c1e77fb": "Attention kernels and backends",
    "98f47f2a": "Attention kernels and backends", "9f1710f1": "Attention kernels and backends",
    "bc7c4d20": "Attention kernels and backends",
    "296f927f": "Host-device memory traffic", "310aca88": "Host-device memory traffic",
    "70b808fe": "Host-device memory traffic", "b55ed6ef": "Host-device memory traffic",
    "b690e348": "Host-device memory traffic", "fe66b347": "Host-device memory traffic",
    "19d98e0c": "MoE and expert parallelism", "e7b20426": "MoE and expert parallelism",
    "015069b0": "Tokenization and frontend", "22d33bac": "Tokenization and frontend",
    "58eee5f2": "Tokenization and frontend",
    "e206b543": "Structured output", "fc542144": "Structured output",
    "4c822298": "Speculative decoding",
    "6a417b86": "General CPU overhead", "6dd94dbe": "General CPU overhead",
    "89a84b0b": "General CPU overhead", "9badee53": "General CPU overhead",
    "9ed82e70": "General CPU overhead", "fc7b8d1e": "General CPU overhead",
    # ---- SGLang (15) ----
    "132dad87": "Prefill-decode disaggregation", "187b85b7": "Prefill-decode disaggregation",
    "2ed68d7a": "Prefill-decode disaggregation", "6b231325": "Prefill-decode disaggregation",
    "dd1012fc": "Prefill-decode disaggregation",
    "31589e17": "Scheduling and batching", "73b13e69": "Scheduling and batching",
    "a191a0e4": "Scheduling and batching",
    "1acca3a2": "Attention kernels and backends", "205d5cb4": "Attention kernels and backends",
    "c087ddd6": "MoE and expert parallelism", "df7f61ee": "MoE and expert parallelism",
    "da47621c": "Sampling and logits",
    "e3ec6bf4": "Quantization",
    "021f76e4": "LoRA",
}


def main():
    per_repo, missing, subjects = {}, [], {}
    for repo in ("vllm", "sglang"):
        recs = [json.loads(l) for l in open(EAD / "dataset" / f"{repo}.jsonl")]
        counts = Counter()
        for r in recs:
            c8 = r["commit_hash"][:8]
            subjects[c8] = r["commit_subject"]
            fam = LABELS.get(c8)
            if fam is None:
                missing.append((repo, c8, r["commit_subject"][:60]))
                continue
            counts[fam] += 1
        per_repo[repo] = (counts, len(recs))

    if missing:
        print("UNLABELED:", missing)

    fams = sorted({f for c, _ in per_repo.values() for f in c},
                  key=lambda f: -(per_repo["vllm"][0][f] + per_repo["sglang"][0][f]))
    L = ["# yX4G — Bottleneck-family distribution of Level 1 (generated)", "",
         "Fresh labeling pass over the 54 Level-1 commits: the collection pipeline never "
         "stored a bottleneck-area field, so each task was assigned one primary family from "
         "its commit subject and touched paths. Labels live in "
         "`scripts/rebuttal_analysis/bottleneck_categories.py` and are auditable per commit.", "",
         "| Bottleneck family | vLLM | SGLang | Total |", "|---|---|---|---|"]
    for f in fams:
        v, s = per_repo["vllm"][0][f], per_repo["sglang"][0][f]
        L.append(f"| {f} | {v or ''} | {s or ''} | {v+s} |")
    L += [f"| **Total** | **{per_repo['vllm'][1]}** | **{per_repo['sglang'][1]}** | "
          f"**{per_repo['vllm'][1]+per_repo['sglang'][1]}** |", "",
          f"Families represented: {len(fams)}. vLLM spans "
          f"{len([f for f in fams if per_repo['vllm'][0][f]])}, SGLang "
          f"{len([f for f in fams if per_repo['sglang'][0][f]])}.", "",
          "Per-task labels:", ""]
    for repo in ("vllm", "sglang"):
        L.append(f"### {repo}")
        L.append("")
        for c8, fam in sorted(LABELS.items(), key=lambda kv: (kv[1], kv[0])):
            if c8 in subjects and any(json.loads(l)["commit_hash"][:8] == c8
                                      for l in open(EAD / "dataset" / f"{repo}.jsonl")):
                L.append(f"- `{c8}` **{fam}** — {subjects[c8][:88]}")
        L.append("")
    (OUT / "bottleneck_categories.md").write_text("\n".join(L) + "\n")
    print("\n".join(L[:24]))


if __name__ == "__main__":
    main()
