#!/usr/bin/env python3
"""Analyze TRAE GPT-5 rerun results."""

import json
from pathlib import Path
from collections import defaultdict

def analyze_run(run_dir):
    """Analyze a single run directory."""
    results = {
        "total": 0,
        "success": 0,
        "error": 0,
        "patch_generated": 0,
        "no_patch": 0,
    }

    for item_dir in run_dir.glob("*_gpt5_rerun_*"):
        if not item_dir.is_dir():
            continue

        journal_file = item_dir / "journal.json"
        if not journal_file.exists():
            continue

        results["total"] += 1

        try:
            journal = json.loads(journal_file.read_text())
            status = journal.get("status", "unknown")

            if status == "success":
                results["success"] += 1
            else:
                results["error"] += 1

            if (item_dir / "model_patch.diff").exists():
                results["patch_generated"] += 1
            else:
                results["no_patch"] += 1
        except Exception as e:
            print(f"Error reading {journal_file}: {e}")

    return results

print("="*70)
print("TRAE GPT-5 Rerun Results Analysis")
print("="*70)

# Find latest runs
state_dir = Path("/home/ubuntu/OmniPerf-Bench/perf-agents-bench/state/runs")

vllm_runs = sorted((state_dir / "vllm/trae").glob("gpt-5*/20*"))
sglang_runs = sorted((state_dir / "sglang/trae").glob("gpt-5*/20*"))

vllm_results = None
sglang_results = None

if vllm_runs:
    latest_vllm = vllm_runs[-1]
    print(f"\nvLLM Run: {latest_vllm.name}")
    print(f"Path: {latest_vllm}")
    vllm_results = analyze_run(latest_vllm)

    print("\nvLLM Results:")
    print(f"  Total: {vllm_results['total']}")
    if vllm_results['total'] > 0:
        success_pct = vllm_results['success']/vllm_results['total']*100
        print(f"  Success: {vllm_results['success']} ({success_pct:.1f}%)")
        print(f"  Error: {vllm_results['error']}")
        print(f"  Patches: {vllm_results['patch_generated']}")
    else:
        print("  No results found yet")
else:
    print("\nNo vLLM GPT-5 rerun results found yet")

if sglang_runs:
    latest_sglang = sglang_runs[-1]
    print(f"\nSGLang Run: {latest_sglang.name}")
    print(f"Path: {latest_sglang}")
    sglang_results = analyze_run(latest_sglang)

    print("\nSGLang Results:")
    print(f"  Total: {sglang_results['total']}")
    if sglang_results['total'] > 0:
        success_pct = sglang_results['success']/sglang_results['total']*100
        print(f"  Success: {sglang_results['success']} ({success_pct:.1f}%)")
        print(f"  Error: {sglang_results['error']}")
        print(f"  Patches: {sglang_results['patch_generated']}")
    else:
        print("  No results found yet")
else:
    print("\nNo SGLang GPT-5 rerun results found yet")

# Calculate combined improvement
if vllm_results and sglang_results:
    if vllm_results['total'] > 0 and sglang_results['total'] > 0:
        original_vllm_total = 434  # From eval_results_v2
        original_vllm_success = 55  # Tests passed
        original_success_rate = original_vllm_success / original_vllm_total * 100

        rerun_success = vllm_results['success'] + sglang_results['success']
        rerun_total = vllm_results['total'] + sglang_results['total']

        combined_success = original_vllm_success + rerun_success
        combined_total = original_vllm_total
        new_success_rate = combined_success / combined_total * 100
        improvement = new_success_rate - original_success_rate

        print(f"\n{'='*70}")
        print("Combined Analysis")
        print(f"{'='*70}")
        print(f"Original success rate: {original_success_rate:.1f}% ({original_vllm_success}/{original_vllm_total})")
        print(f"Rerun recovered: {rerun_success}/{rerun_total} ({rerun_success/rerun_total*100:.1f}%)")
        print(f"New success rate: {new_success_rate:.1f}% ({combined_success}/{combined_total})")
        print(f"Improvement: +{improvement:.1f} percentage points")
        print(f"{'='*70}")
