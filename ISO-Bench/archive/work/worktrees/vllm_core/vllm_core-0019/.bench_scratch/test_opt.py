import time
import torch
from torch.nn import Parameter

# Import repo functions for context where safe
from vllm.model_executor.layers.quantization.fp8 import all_close_1d as new_all_close_1d


def old_all_close_1d(x: torch.Tensor) -> bool:
    assert len(x.shape) == 1
    return all(torch.allclose(x[0], x[i]) for i in range(x.shape[0]))


def slow_requantize(weight: torch.Tensor, weight_scale: torch.Tensor, logical_widths):
    # Simulate the old per-partition requantization using repeated .max()
    start = 0
    for idx, logical_width in enumerate(logical_widths):
        end = start + logical_width
        fake_qweight = weight[start:end, :].to(torch.float16)
        dq_weight = fake_qweight * weight_scale[idx]
        # old: recompute max each time
        max_w_scale = weight_scale.max()
        fq = (dq_weight / max_w_scale).clamp(
            min=torch.finfo(torch.float8_e4m3fn).min,
            max=torch.finfo(torch.float8_e4m3fn).max,
        ).to(torch.float8_e4m3fn)
        weight[start:end, :] = fq
        start = end
    return weight


def fast_requantize(weight: torch.Tensor, weight_scale: torch.Tensor, logical_widths):
    # Simulate the optimized requantization using cached max_w_scale
    max_w_scale = weight_scale.max()
    start = 0
    for idx, logical_width in enumerate(logical_widths):
        end = start + logical_width
        fake_qweight = weight[start:end, :].to(torch.float16)
        dq_weight = fake_qweight * weight_scale[idx]
        fq = (dq_weight / max_w_scale).clamp(
            min=torch.finfo(torch.float8_e4m3fn).min,
            max=torch.finfo(torch.float8_e4m3fn).max,
        ).to(torch.float8_e4m3fn)
        weight[start:end, :] = fq
        start = end
    return weight


def bench_fn(fn, *args, iters=200):
    # Warmup
    for _ in range(20):
        fn(*args)
    t0 = time.perf_counter()
    for _ in range(iters):
        fn(*args)
    return time.perf_counter() - t0


def main():
    torch.manual_seed(0)

    # 1) Vectorized all_close_1d
    x = torch.randn(4096, dtype=torch.float32)
    x[:] = x[0]  # make them equal to avoid numerical issues
    t_old = bench_fn(old_all_close_1d, x)
    t_new = bench_fn(new_all_close_1d, x)
    print(f"all_close_1d: old={t_old:.6f}s new={t_new:.6f}s speedup={t_old/max(t_new,1e-12):.2f}x")

    # 2) Requantization: repeated .max vs cached max
    out_parts = [512, 512]
    out_size = sum(out_parts)
    in_size = 1024
    weight = torch.empty(out_size, in_size, dtype=torch.float8_e4m3fn)
    weight_scale = torch.rand(len(out_parts), dtype=torch.float32)

    t_slow = bench_fn(slow_requantize, weight.clone(), weight_scale, out_parts)
    t_fast = bench_fn(fast_requantize, weight.clone(), weight_scale, out_parts)
    print(f"requantize loop: old={t_slow:.6f}s new={t_fast:.6f}s speedup={t_slow/max(t_fast,1e-12):.2f}x")

    # 3) zeros vs empty allocation for 1-element scale tensor (mimics scaled_fp8_quant)
    iters = 200000
    def alloc_zero():
        return torch.zeros(1, device='cpu', dtype=torch.float32)
    def alloc_empty():
        return torch.empty(1, device='cpu', dtype=torch.float32)

    t0 = time.perf_counter();
    for _ in range(iters):
        s = alloc_zero()
    t_zeros = time.perf_counter() - t0

    t0 = time.perf_counter();
    for _ in range(iters):
        s = alloc_empty()
    t_empty = time.perf_counter() - t0

    print(f"alloc scale: zeros={t_zeros:.6f}s empty={t_empty:.6f}s speedup={t_zeros/max(t_empty,1e-12):.2f}x")


if __name__ == "__main__":
    main()
