#!/usr/bin/env python3
"""
Monitor the current SGLang TRAE Sonnet 4.5 rerun in real-time.
"""
import json
from pathlib import Path
from datetime import datetime
from collections import Counter

def monitor_current_run():
    # Find the most recent run
    run_dirs = list(Path("/home/ubuntu/OmniPerf-Bench/perf-agents-bench/state/runs").glob("sglan/trae/*/2025-12-24_*"))

    if not run_dirs:
        print("❌ No run directories found!")
        return

    latest_run = sorted(run_dirs, key=lambda x: x.name)[-1]

    print("=" * 70)
    print(f"🔍 SGLang Rerun Sanity Check")
    print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)
    print(f"\n📁 Run Directory: {latest_run.name}\n")

    # Find all commit directories
    commit_dirs = [d for d in latest_run.iterdir() if d.is_dir() and d.name.startswith("sglang_sonnet45_rerun_")]

    print(f"📊 Total commits processed: {len(commit_dirs)}/51")
    print(f"📈 Progress: {len(commit_dirs)/51*100:.1f}%\n")

    if not commit_dirs:
        print("⚠️  No commits processed yet. Waiting for first results...")
        return

    # Analyze results
    results = []
    for commit_dir in commit_dirs:
        journal_file = commit_dir / "journal.json"
        if journal_file.exists():
            try:
                with open(journal_file) as f:
                    data = json.load(f)

                status = data.get("status", "unknown")

                # Check patch
                patch_file = commit_dir / "model_patch.diff"
                patch_size = patch_file.stat().st_size if patch_file.exists() else 0
                has_real_patch = patch_size > 0

                # Check trajectory
                traj_file = commit_dir / "trajectory.json"
                steps = 0
                trae_success = False
                trae_error = None

                if traj_file.exists():
                    with open(traj_file) as f:
                        traj = json.load(f)
                    steps = len(traj.get("llm_interactions", []))
                    trae_success = traj.get("success", False)
                    trae_error = traj.get("error", None)

                results.append({
                    "commit": commit_dir.name.replace("sglang_sonnet45_rerun_", ""),
                    "status": status,
                    "has_patch": has_real_patch,
                    "patch_size": patch_size,
                    "steps": steps,
                    "trae_success": trae_success,
                    "trae_error": trae_error
                })
            except Exception as e:
                print(f"⚠️  Error reading {commit_dir.name}: {e}")

    if not results:
        print("⚠️  No valid results found yet.")
        return

    # Calculate statistics
    success_count = sum(1 for r in results if r["status"] == "success")
    error_count = sum(1 for r in results if r["status"] == "error")
    with_patch = sum(1 for r in results if r["has_patch"])

    print(f"✅ Success: {success_count} ({success_count/len(results)*100:.1f}%)")
    print(f"❌ Error: {error_count} ({error_count/len(results)*100:.1f}%)")
    print(f"📄 Patches generated: {with_patch} ({with_patch/len(results)*100:.1f}%)\n")

    # Check for credential errors
    cred_errors = sum(1 for r in results if r["trae_error"] and "token" in str(r["trae_error"]).lower())
    if cred_errors > 0:
        print(f"🚨 WARNING: {cred_errors} credential errors detected!\n")

    # Check average steps
    avg_steps = sum(r["steps"] for r in results) / len(results) if results else 0
    print(f"📊 Average TRAE steps: {avg_steps:.1f}")

    # Show recent results (last 5)
    print(f"\n📋 Latest 5 Results:")
    for r in sorted(results, key=lambda x: x["commit"])[-5:]:
        status_icon = "✅" if r["status"] == "success" else "❌"
        patch_icon = "📄" if r["has_patch"] else "  "
        steps_str = f"({r['steps']} steps)" if r['steps'] > 0 else "(0 steps)"
        print(f"  {status_icon} {patch_icon} {r['commit']} - {r['status']} {steps_str}")

    # Check for anomalies
    print(f"\n🔍 Anomaly Detection:")

    # Zero-step failures
    zero_step_failures = sum(1 for r in results if r["status"] == "error" and r["steps"] == 0)
    if zero_step_failures > 0:
        print(f"  ⚠️  {zero_step_failures} commits failed with 0 steps (credential/startup issues?)")
    else:
        print(f"  ✅ No zero-step failures")

    # Success without patches
    success_no_patch = sum(1 for r in results if r["status"] == "success" and not r["has_patch"])
    if success_no_patch > 0:
        print(f"  ⚠️  {success_no_patch} successes without patches (suspicious!)")
    else:
        print(f"  ✅ All successes have patches")

    # Very short trajectories for successes
    short_success = sum(1 for r in results if r["status"] == "success" and r["steps"] < 3)
    if short_success > 0:
        print(f"  ⚠️  {short_success} successes with <3 steps (might be false positives)")
    else:
        print(f"  ✅ All successes have reasonable trajectories")

    print(f"\n{'=' * 70}")

    # Overall health
    if zero_step_failures == 0 and success_no_patch == 0 and cred_errors == 0:
        print("✅ Overall Status: HEALTHY - Everything looks good!")
    elif zero_step_failures > 0 or cred_errors > 0:
        print("🚨 Overall Status: ISSUES DETECTED - May need intervention")
    else:
        print("⚠️  Overall Status: MINOR ISSUES - Monitoring recommended")

    print(f"{'=' * 70}\n")

    # Time estimate
    if len(results) > 0:
        remaining = 51 - len(results)
        # Check most recent commit's timestamp
        latest_commit_dir = sorted(commit_dirs, key=lambda x: x.stat().st_mtime)[-1]
        print(f"⏱️  Commits remaining: {remaining}")
        print(f"⏱️  ETA: ~{remaining * 6} minutes (~{remaining * 6 / 60:.1f} hours)")

if __name__ == "__main__":
    monitor_current_run()
