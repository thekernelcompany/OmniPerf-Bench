import time
import torch


def bench_alloc(shape=(8192, 8192), iters=50, device=None, dtype=torch.float16):
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    torch.manual_seed(0)
    # Warmup
    for _ in range(5):
        _ = torch.empty(shape, device=device, dtype=dtype)
        _ = torch.zeros(shape, device=device, dtype=dtype)
    if torch.cuda.is_available():
        torch.cuda.synchronize()

    t0 = time.perf_counter()
    for _ in range(iters):
        _ = torch.zeros(shape, device=device, dtype=dtype)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t1 = time.perf_counter()

    t2 = time.perf_counter()
    for _ in range(iters):
        _ = torch.empty(shape, device=device, dtype=dtype)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t3 = time.perf_counter()

    return (t1 - t0) / iters, (t3 - t2) / iters, device, dtype


def bench_layernorm_like(shape=(4096, 8192), iters=50, device=None, dtype=torch.float16):
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    x = torch.randn(shape, device=device, dtype=dtype)
    w = torch.randn(shape[-1], device=device, dtype=dtype)
    eps = 1e-6
    # Warmup
    for _ in range(5):
        y = x * w
        _ = y.norm(dim=-1)
    if torch.cuda.is_available():
        torch.cuda.synchronize()

    t0 = time.perf_counter()
    for _ in range(iters):
        var = (x.to(torch.float32) ** 2).mean(dim=-1, keepdim=True)
        inv = torch.rsqrt(var + eps)
        y = (x * inv) * w
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t1 = time.perf_counter()
    return (t1 - t0) / iters


if __name__ == '__main__':
    zeros_t, empty_t, device, dtype = bench_alloc()
    ln_t = bench_layernorm_like()
    print(f"device={device} dtype={dtype}")
    print(f"alloc_avg_ms zeros={zeros_t*1e3:.3f} ms empty={empty_t*1e3:.3f} ms")
    print(f"layernorm_like_avg_ms={ln_t*1e3:.3f} ms")
