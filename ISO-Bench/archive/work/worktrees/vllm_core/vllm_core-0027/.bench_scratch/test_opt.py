import time
import random
import importlib.util
import sys
from pathlib import Path

# Load LRUEvictor directly from file to avoid importing full vllm package
REPO_ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = REPO_ROOT / 'vllm' / 'core' / 'evictor_v2.py'
spec = importlib.util.spec_from_file_location('evictor_v2', MODULE_PATH)
evictor_v2 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(evictor_v2)  # type: ignore
LRUEvictor = evictor_v2.LRUEvictor

# Micro-benchmark for LRUEvictor
# We construct an evictor with many blocks and repeatedly call evict/update

def build_evictor(n):
    ev = LRUEvictor()
    now = time.time()
    # Create groups of equal last_accessed times to exercise tie-breaking
    for i in range(n):
        # cluster timestamps so several have the same value
        ts = now - (i // 16)
        hashed = random.randint(1, 4096)
        ev.add(i, content_hash=i ^ 0x9E3779B97F4A7C15, num_hashed_tokens=hashed, last_accessed=ts)
    return ev


def bench_evict(evictor, rounds):
    t0 = time.perf_counter()
    # We perform evictions from a copied evictor to avoid rebuilding each time
    for _ in range(rounds):
        # make a shallow copy of the dict state by rebuilding a new evictor
        e = LRUEvictor()
        for bid, meta in evictor.free_table.items():
            e.add(bid, meta.content_hash, meta.num_hashed_tokens, meta.last_accessed)
        # evict K items
        k = 256
        for _ in range(k):
            e.evict()
    t1 = time.perf_counter()
    return t1 - t0


def bench_update(evictor, ops):
    ids = list(evictor.free_table.keys())
    t0 = time.perf_counter()
    for i in range(ops):
        bid = ids[i % len(ids)]
        evictor.update(bid, time.time() + i)
    t1 = time.perf_counter()
    return t1 - t0


if __name__ == "__main__":
    random.seed(0)
    N = 8192
    base = build_evictor(N)
    t_evict = bench_evict(base, rounds=8)
    t_update = bench_update(base, ops=20000)
    print({
        "items": N,
        "evict_rounds": 8,
        "evict_time_s": round(t_evict, 6),
        "update_ops": 20000,
        "update_time_s": round(t_update, 6),
    })
