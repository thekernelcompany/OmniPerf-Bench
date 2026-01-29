import time
import sys

try:
    import torch
except Exception as e:
    print("PyTorch not available:", e)
    sys.exit(0)

# Simple microbench to simulate overhead reduction from slicing vs no-slicing
# and zeros vs empty allocations.

def bench_slicing(iters: int = 20000, n: int = 4096, d: int = 128):
    x = torch.empty((n, d))
    t0 = time.perf_counter()
    for _ in range(iters):
        y = x  # no slicing
    t_no_slice = time.perf_counter() - t0

    t0 = time.perf_counter()
    for _ in range(iters):
        y = x[:n]  # slicing view
    t_slice = time.perf_counter() - t0
    return t_no_slice, t_slice


def bench_alloc(iters: int = 2000, n: int = 4096, d: int = 128):
    t0 = time.perf_counter()
    for _ in range(iters):
        _ = torch.empty((n, d))
    t_empty = time.perf_counter() - t0

    t0 = time.perf_counter()
    for _ in range(iters):
        _ = torch.zeros((n, d))
    t_zeros = time.perf_counter() - t0
    return t_empty, t_zeros


def main():
    print("Running micro-benchmarks (simulating optimization effects)...")
    t_no_slice, t_slice = bench_slicing()
    print(f"Slicing: no-slice={t_no_slice:.6f}s, slice={t_slice:.6f}s, speedup={(t_slice/max(t_no_slice,1e-9)):.2f}x over slice")
    t_empty, t_zeros = bench_alloc()
    print(f"Allocation: empty={t_empty:.6f}s, zeros={t_zeros:.6f}s, speedup={(t_zeros/max(t_empty,1e-9)):.2f}x over empty")

if __name__ == "__main__":
    main()
