#!/usr/bin/env python3
"""Re-extract throughput-equivalent numbers for 4c822298 (Sonnet + GPT-5)
from existing OH raw_output. No GPU.

Per X3 design read (c): 4c822298 is speculative-decoding; benchmark_throughput.py
doesn't fit cleanly. Don't re-bench. Use the existing benchmark_latency.py raw
output and compute (batch * (input + output)) / avg_latency_s with explicit
defaults from vllm's benchmark_latency.py argparse for that commit's vllm era.
This is *one* defensible derivation rule, applied symmetrically. Not the same
as the unidentified rule that produced canonical 102.1, but at least
internally consistent and documented.
"""
import json, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
SUBM = ROOT / "third-party/everything_analysis_data"
OUT_DIR = ROOT / "archive/results/2026-05-x3/extracted_4c822298"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# benchmark_latency.py defaults at the parent commit era for 4c822298:
#   --batch-size 8 (default), --input-len 32 (default), --output-len 128 (default)
# perf_command was: ... --speculative-model meta-llama/Llama-3.2-1B-Instruct --num-speculative-tokens 5
# (no batch/input/output overrides, so defaults apply)
PARAMS = {"batch": 8, "input": 32, "output": 128}


def derive(lat_s):
    return (PARAMS["batch"] * (PARAMS["input"] + PARAMS["output"])) / lat_s


def extract(label, src_path):
    if not src_path.exists():
        print(f"  {label}: MISSING source file at {src_path}")
        return None
    d = json.loads(src_path.read_text())
    lat = (d.get("metrics") or {}).get("avg_latency_s")
    if lat is None:
        print(f"  {label}: no avg_latency_s in metrics")
        return None
    derived_total = derive(lat)
    out = {
        "human_commit": "4c822298",
        "human_commit_full": "4c822298981a8f7521492075ff72659985fc4c3f",
        "parent_commit": "c8d70e2437feecdb3762ce17298df33439ae1bd1",
        "model": d.get("model"),
        "perf_command_original": d.get("perf_command"),
        "extraction_rule": "derived_from_existing_latency: (batch * (input + output)) / avg_latency_s",
        "extraction_params": PARAMS,
        "metrics": {
            "avg_latency_s": lat,
            "derived_total_throughput_tok_s": round(derived_total, 2),
        },
        "source_file": str(src_path.relative_to(ROOT)),
        "caveat": "speculative decoding commit; benchmark_throughput.py was not run (read (c) of X3). Number is methodologically distinct from the canonical extractor's 102.1 — comparison cross-rule.",
    }
    out_file = OUT_DIR / f"4c822298_{label}_extracted.json"
    out_file.write_text(json.dumps(out, indent=2))
    print(f"  {label}: lat={lat:.4f}s -> derived_total_tok_s={derived_total:.2f}  ->  {out_file.relative_to(ROOT)}")
    return out


# OH Sonnet 4.5
extract("openhands_sonnet45", SUBM / "iso-bench-openhands-sonnet45-hard-metrics/results_vllm/results/4c822298_agent_result.json")
# OH GPT-5
extract("openhands_gpt5", SUBM / "iso-bench-openhands-gpt5-hard-metrics/results_vllm/results/4c822298_agent_result.json")
