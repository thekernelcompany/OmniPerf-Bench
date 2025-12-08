#!/usr/bin/env python3
"""
Minimal demo: plan, prepare, report using PYTHONPATH (no install).
Run from repo root.
"""
import os
import subprocess
from pathlib import Path

def main():
    root = Path(__file__).resolve().parents[1]
    bench_dir = root / "perf-agents-bench"

    # Ensure commits file
    work = bench_dir / ".work"
    work.mkdir(parents=True, exist_ok=True)
    commits = work / "vllm_commits.txt"
    if not commits.exists():
        commits.write_text("f092153fbe349a9a1742940e3703bfcff6aa0a6d parent=1\n")

    env = os.environ.copy()
    env["PYTHONPATH"] = str(bench_dir)

    # Plan
    subprocess.run([
        "python3", "-m", "bench.cli", "plan",
        str(bench_dir / "tasks/vllm.yaml"),
        "--commits", str(commits),
        "--out", str(root / "state/plan.json"),
    ], check=True, cwd=root, env=env)

    # Prepare
    subprocess.run([
        "python3", "-m", "bench.cli", "prepare",
        str(bench_dir / "tasks/vllm.yaml"),
        "--from-plan", str(root / "state/plan.json"),
        "--bench-cfg", str(bench_dir / "bench.yaml"),
        "--max-workers", "1", "--resume",
    ], check=True, cwd=root, env=env)

    # Report
    latest = sorted((bench_dir / "state/runs").iterdir(), key=lambda p: p.stat().st_mtime, reverse=True)[0]
    subprocess.run([
        "python3", "-m", "bench.cli", "report", str(latest)
    ], check=True, cwd=root, env=env)

if __name__ == "__main__":
    main()