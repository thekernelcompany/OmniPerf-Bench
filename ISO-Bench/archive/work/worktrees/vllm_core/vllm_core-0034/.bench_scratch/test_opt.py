"""
Micro-benchmark to approximate the effect of avoiding list concatenations
with '+' and using extend on a preallocated list instead.

This does not import vllm to keep the benchmark environment-lightweight.
Run as: python .bench_scratch/test_opt.py
"""
import time
import random


def build_lists(n_groups=5, base=200, jitter=50):
    rng = random.Random(0)
    return [list(range(base + rng.randint(-jitter, jitter))) for _ in range(n_groups)]


def concat_plus(groups):
    # Simulate: a + b + c + d + e
    out = []
    for g in groups:
        out = out + g
    return out


def concat_extend(groups):
    # Simulate: extend into a single list
    out = []
    for g in groups:
        out.extend(g)
    return out


def bench(fn, groups, iters=2000):
    t0 = time.perf_counter()
    s = 0
    for _ in range(iters):
        s += len(fn(groups))
    dt = time.perf_counter() - t0
    return dt, s


if __name__ == "__main__":
    groups = build_lists()
    t_plus, s1 = bench(concat_plus, groups)
    t_ext, s2 = bench(concat_extend, groups)
    assert s1 == s2
    print(f"plus:   {t_plus:.6f}s")
    print(f"extend: {t_ext:.6f}s")
    print(f"speedup: {t_plus/t_ext:.2f}x")
