import os
import time
import inspect
import importlib.util
import torch

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODULE_PATH = os.path.join(REPO_ROOT, 'vllm', 'model_executor', 'layers', 'quantization', 'compressed_tensors', 'triton_scaled_mm.py')

spec = importlib.util.spec_from_file_location('triton_scaled_mm_mod', MODULE_PATH)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
triton_scaled_mm = mod.triton_scaled_mm


SUPPORTS_HEUR = 'use_heuristic' in inspect.signature(triton_scaled_mm).parameters


def run_once(use_heuristic: bool, M: int, N: int, K: int, dtype=torch.int8, out_dtype=torch.float16):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if device != 'cuda':
        print('CUDA not available; skipping kernel run.')
        return None

    a = torch.randint(-128, 127, (M, K), dtype=dtype, device=device)
    b = torch.randint(-128, 127, (K, N), dtype=dtype, device=device)
    # scale factors as scalars (fast path)
    scale_a = torch.tensor([[1.0]], dtype=torch.float32, device=device)
    scale_b = torch.tensor([[1.0]], dtype=torch.float32, device=device)
    bias = None

    # warmup
    if SUPPORTS_HEUR:
        _ = triton_scaled_mm(a, b, scale_a, scale_b, out_dtype, bias=bias, use_heuristic=use_heuristic)
    else:
        _ = triton_scaled_mm(a, b, scale_a, scale_b, out_dtype, bias=bias)
    torch.cuda.synchronize()

    iters = 10
    start = time.perf_counter()
    for _ in range(iters):
        if SUPPORTS_HEUR:
            _ = triton_scaled_mm(a, b, scale_a, scale_b, out_dtype, bias=bias, use_heuristic=use_heuristic)
        else:
            _ = triton_scaled_mm(a, b, scale_a, scale_b, out_dtype, bias=bias)
    torch.cuda.synchronize()
    end = time.perf_counter()
    return (end - start) / iters


def main():
    M = int(os.environ.get('M', 1024))
    N = int(os.environ.get('N', 2048))
    K = int(os.environ.get('K', 1024))

    print(f'Benchmarking triton_scaled_mm with shapes M={M}, K={K}, N={N}')
    t_base = run_once(False, M, N, K)
    t_opt = run_once(True, M, N, K) if SUPPORTS_HEUR else t_base

    if t_base is None or t_opt is None:
        print('Benchmark skipped (no CUDA).')
        return

    print(f'Baseline (no heuristic): {t_base*1e3:.3f} ms/iter')
    if SUPPORTS_HEUR:
        print(f'Optimized (heuristic):  {t_opt*1e3:.3f} ms/iter')
        speedup = t_base / t_opt if t_opt > 0 else float("inf")
        print(f'Speedup: {speedup:.2f}x')
    else:
        print('Optimized (heuristic):  N/A (pre-change baseline)')


if __name__ == '__main__':
    main()
