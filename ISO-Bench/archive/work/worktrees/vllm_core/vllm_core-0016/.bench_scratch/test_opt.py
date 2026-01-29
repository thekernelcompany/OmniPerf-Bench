#!/usr/bin/env python3
import os
import sys
import time
import hashlib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

EXCLUDES = {'.git', '.bench_scratch', '.venv', 'build', 'dist', '.mypy_cache', '.ruff_cache', '__pycache__'}
EXTS = {'.py', '.pyi', '.cu', '.cuh', '.c', '.h', '.hpp', '.cc', '.md', '.rst', '.yaml', '.yml'}


def should_skip(path: Path) -> bool:
    parts = set(path.parts)
    return any(p in EXCLUDES for p in parts)


def file_iter():
    for dirpath, dirnames, filenames in os.walk(ROOT):
        # prune excluded directories in-place for faster traversal
        dirnames[:] = [d for d in dirnames if d not in EXCLUDES]
        for fn in filenames:
            p = Path(dirpath) / fn
            if p.suffix in EXTS and not should_skip(p):
                yield p


def time_scan_and_hash():
    t0 = time.perf_counter()
    total_bytes = 0
    h = hashlib.sha256()
    count = 0
    for p in file_iter():
        try:
            with open(p, 'rb') as f:
                data = f.read()
            total_bytes += len(data)
            h.update(data)
            count += 1
        except Exception:
            pass
    elapsed = time.perf_counter() - t0
    return elapsed, count, total_bytes, h.hexdigest()[:16]


def main():
    runs = int(os.environ.get('BENCH_RUNS', '2'))
    timings = []
    for _ in range(runs):
        elapsed, count, total_bytes, digest = time_scan_and_hash()
        timings.append(elapsed)
        print(f"scan: {elapsed:.4f}s files={count} bytes={total_bytes} digest={digest}")
    if timings:
        avg = sum(timings) / len(timings)
        print(f"avg_scan: {avg:.4f}s")
    # Heuristic: count mypy hooks configured to manual stage as a proxy for faster pre-commit
    try:
        cfg_lines = (ROOT / '.pre-commit-config.yaml').read_text().splitlines()
        total_mypy = 0
        manual_mypy = 0
        for i, line in enumerate(cfg_lines):
            if line.strip().startswith('- id: mypy-'):
                total_mypy += 1
                window = cfg_lines[i:i+12]
                if any(('stages:' in w and 'manual' in w) for w in window):
                    manual_mypy += 1
        print(f"mypy_hooks_total: {total_mypy}")
        print(f"mypy_hooks_manual: {manual_mypy}")
    except Exception:
        pass


if __name__ == '__main__':
    sys.exit(main())
