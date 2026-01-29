import time
def bench_prob_lp_methods(device="cpu", n=512, v=8192, iters=30):
    logits = torch.randn(n, v, device=device)
    # warmup
    for _ in range(3):
        _ = torch.softmax(logits, dim=-1, dtype=torch.float)
        _ = torch.log_softmax(logits, dim=-1, dtype=torch.float)
        _ = torch.log_softmax(logits, dim=-1, dtype=torch.float)
        _ = torch.exp(_)
    t0 = time.perf_counter()
    for _ in range(iters):
        _ = torch.softmax(logits, dim=-1, dtype=torch.float)
        _ = torch.log_softmax(logits, dim=-1, dtype=torch.float)
    torch.cuda.synchronize() if device != "cpu" else None
    t1 = time.perf_counter()
    t_softmax_pair = t1 - t0
    t0 = time.perf_counter()
    for _ in range(iters):
        lp = torch.log_softmax(logits, dim=-1, dtype=torch.float)
        _ = torch.exp(lp)
    torch.cuda.synchronize() if device != "cpu" else None
    t2 = time.perf_counter()
    t_logsoftmax_exp = t2 - t0
    return t_softmax_pair, t_logsoftmax_exp

import torch

from vllm.model_executor.layers import sampler as sampler_mod


def bench_get_ranks(device="cpu", n=512, v=8192, iters=30):
    x = torch.randn(n, v, device=device)
    # simulate logprobs
    x = torch.log_softmax(x, dim=-1)
    indices = torch.randint(0, v, (n,), device=device)
    # warmup
    for _ in range(5):
        sampler_mod._get_ranks(x, indices.tolist())
    t0 = time.perf_counter()
    for _ in range(iters):
        r = sampler_mod._get_ranks(x, indices.tolist())
    torch.cuda.synchronize() if device != "cpu" else None
    t1 = time.perf_counter()
    return t1 - t0


def bench_apply_top_k_top_p(device="cpu", n=512, v=8192, iters=30):
    logits = torch.randn(n, v, device=device)
    p = torch.rand(n, device=device) * 0.9  # p in [0,0.9]
    k = torch.randint(1, max(2, v // 4), (n,), device=device)
    # warmup
    for _ in range(3):
        sampler_mod._apply_top_k_top_p(logits.clone(), p, k)
    t0 = time.perf_counter()
    for _ in range(iters):
        sampler_mod._apply_top_k_top_p(logits.clone(), p, k)
    torch.cuda.synchronize() if device != "cpu" else None
    t1 = time.perf_counter()
    return t1 - t0


def bench_apply_penalties(device="cpu", n=256, v=4096, t_prompt=16, t_output=32, iters=10):
    logits = torch.randn(n, v, device=device)
    prompt_tokens = torch.randint(0, v, (n, t_prompt), device=device, dtype=torch.long)
    output_tokens = torch.randint(0, v, (n, t_output), device=device, dtype=torch.long)
    presence_penalties = torch.rand(n, device=device)
    frequency_penalties = torch.rand(n, device=device)
    repetition_penalties = 0.5 + torch.rand(n, device=device)  # >0
    # warmup
    for _ in range(3):
        sampler_mod._apply_penalties(logits.clone(), prompt_tokens, output_tokens,
                                     presence_penalties.clone(), frequency_penalties.clone(),
                                     repetition_penalties.clone())
    t0 = time.perf_counter()
    for _ in range(iters):
        sampler_mod._apply_penalties(logits.clone(), prompt_tokens, output_tokens,
                                     presence_penalties.clone(), frequency_penalties.clone(), repetition_penalties.clone())
    torch.cuda.synchronize() if device != "cpu" else None
    t1 = time.perf_counter()
    return t1 - t0


def main():
    device = "cpu"
    print(f"device={device}")
    t1 = bench_get_ranks(device=device)
    print(f"get_ranks: {t1:.4f}s")
    t2 = bench_apply_top_k_top_p(device=device)
    print(f"apply_top_k_top_p: {t2:.4f}s")
    t3 = bench_apply_penalties(device=device)
    print(f"apply_penalties: {t3:.4f}s")
    t_old, t_new = bench_prob_lp_methods(device=device)
    print(f"softmax+log_softmax: {t_old:.4f}s vs log_softmax+exp: {t_new:.4f}s")


if __name__ == "__main__":
    main()
