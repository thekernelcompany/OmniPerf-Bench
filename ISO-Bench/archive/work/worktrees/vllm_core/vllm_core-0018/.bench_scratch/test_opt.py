import time
import torch
from torch import Generator

from vllm.v1.sample.ops.topk_topp_sampler import apply_top_k_top_p, random_sample


def bench(run_name: str, device: str = "cpu", batch_size: int = 256, vocab_size: int = 32768, k_val: int = 50, iters: int = 10, seed: int = 33):
    print(f"run={run_name} device={device} B={batch_size} V={vocab_size} k={k_val} iters={iters}")
    torch.manual_seed(seed)
    gen = Generator(device=device).manual_seed(seed)
    logits = torch.rand((batch_size, vocab_size), device=device, generator=gen)
    k = torch.full((batch_size,), k_val, dtype=torch.long, device=device)
    p = None
    # warmup
    for _ in range(2):
        masked = apply_top_k_top_p(logits.clone(), k, p)
        probs = masked.softmax(dim=-1, dtype=torch.float32)
        _ = random_sample(probs, generators={})
    torch.cuda.synchronize() if device.startswith("cuda") else None
    t0 = time.perf_counter()
    for _ in range(iters):
        masked = apply_top_k_top_p(logits.clone(), k, p)
        probs = masked.softmax(dim=-1, dtype=torch.float32)
        _ = random_sample(probs, generators={})
    torch.cuda.synchronize() if device.startswith("cuda") else None
    t1 = time.perf_counter()
    dt = (t1 - t0) / iters
    print(f"avg_time_s={dt:.6f}")


if __name__ == "__main__":
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    bench("topk_only", device=dev)
