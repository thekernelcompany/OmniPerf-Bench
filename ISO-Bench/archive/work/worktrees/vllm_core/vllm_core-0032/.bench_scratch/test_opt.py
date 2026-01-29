import os
import time
import torch

# Microbench to approximate improvements from:
# 1) Using torch.empty (pinned) instead of torch.zeros (pinned)
# 2) Using non_blocking=True for HtoD copies when src is pinned
# 3) Using torch.empty_like instead of torch.zeros_like

def bench_alloc_pinned_empty_vs_zeros(iters: int = 2000, shape=(2048, 1)):
    pin = torch.cuda.is_available()
    # Warmup
    for _ in range(100):
        x = torch.empty(shape, dtype=torch.long, device='cpu', pin_memory=pin)
        del x
        x = torch.zeros(shape, dtype=torch.long, device='cpu', pin_memory=pin)
        del x

    t0 = time.perf_counter()
    for _ in range(iters):
        x = torch.empty(shape, dtype=torch.long, device='cpu', pin_memory=pin)
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        del x
    t1 = time.perf_counter()

    for _ in range(iters):
        x = torch.zeros(shape, dtype=torch.long, device='cpu', pin_memory=pin)
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        del x
    t2 = time.perf_counter()

    return (t1 - t0), (t2 - t1)


def bench_non_blocking_copy(iters: int = 2000, shape=(2048, 1)):
    if not torch.cuda.is_available():
        return None
    # Pinned source
    src = torch.empty(shape, dtype=torch.long, device='cpu', pin_memory=True)
    dst = None
    torch.cuda.synchronize()

    # Warmup
    for _ in range(100):
        dst = src.cuda(non_blocking=True)
        del dst
        dst = src.cuda(non_blocking=False)
        del dst
    torch.cuda.synchronize()

    t0 = time.perf_counter()
    for _ in range(iters):
        dst = src.cuda(non_blocking=True)
        del dst
    torch.cuda.synchronize()
    t1 = time.perf_counter()

    for _ in range(iters):
        dst = src.cuda(non_blocking=False)
        del dst
    torch.cuda.synchronize()
    t2 = time.perf_counter()

    return (t1 - t0), (t2 - t1)


def bench_empty_like_vs_zeros_like(iters: int = 2000, shape=(1024, 4096)):
    base = torch.empty(shape, dtype=torch.float16, device='cuda' if torch.cuda.is_available() else 'cpu')

    # Warmup
    for _ in range(100):
        x = torch.empty_like(base)
        del x
        x = torch.zeros_like(base)
        del x

    t0 = time.perf_counter()
    for _ in range(iters):
        x = torch.empty_like(base)
        del x
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    t1 = time.perf_counter()

    for _ in range(iters):
        x = torch.zeros_like(base)
        del x
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    t2 = time.perf_counter()

    return (t1 - t0), (t2 - t1)


if __name__ == '__main__':
    iters = int(os.environ.get('OPT_BENCH_ITERS', '2000'))

    t_empty_pinned, t_zeros_pinned = bench_alloc_pinned_empty_vs_zeros(iters)
    print(f"Pinned alloc: empty={t_empty_pinned:.4f}s zeros={t_zeros_pinned:.4f}s speedup={(t_zeros_pinned/max(t_empty_pinned,1e-9)):.2f}x")

    nb = bench_non_blocking_copy(iters)
    if nb is None:
        print("non_blocking copy: CUDA not available; skipped")
    else:
        t_nb_true, t_nb_false = nb
        print(f"HtoD copy (pinned src): non_blocking=True={t_nb_true:.4f}s False={t_nb_false:.4f}s speedup={(t_nb_false/max(t_nb_true,1e-9)):.2f}x")

    t_empty_like, t_zeros_like = bench_empty_like_vs_zeros_like(iters)
    print(f"*_like alloc: empty_like={t_empty_like:.4f}s zeros_like={t_zeros_like:.4f}s speedup={(t_zeros_like/max(t_empty_like,1e-9)):.2f}x")
