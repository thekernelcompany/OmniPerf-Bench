import time
import torch

# Use CPU to avoid GPU-specific dependencies in this timing harness
DEVICE = torch.device('cpu')
DTYPE = torch.float32

def build_cu_seqlens(seqlens_list):
    # Returns a 1D tensor starting with 0 of cumulative seqlens in int32
    cu = [0]
    s = 0
    for l in seqlens_list:
        s += int(l)
        cu.append(s)
    return torch.tensor(cu, dtype=torch.int32, device=DEVICE)


def time_attention(attn_cls, name, *, projection_size=128, num_heads=8, batch=1, groups=64, iters=50, pass_opt_params=False):
    torch.manual_seed(0)
    embed_dim = projection_size
    # Construct seqlens: groups entries of length 32 tokens (total tokens S)
    seqlen_per = 32
    seqlens_list = [seqlen_per] * groups
    S = seqlen_per * groups
    cu_seqlens = build_cu_seqlens(seqlens_list)

    # x shape [S, B, C]
    x = torch.randn(S, batch, embed_dim, device=DEVICE, dtype=DTYPE)

    # rotary_pos_emb is optional in both classes; pass None to avoid extra compute
    rotary = None

    # Instantiate attention
    try:
        attn = attn_cls(embed_dim=embed_dim, num_heads=num_heads, projection_size=projection_size)
    except TypeError:
        # Some variant may use different arg names (e.g., embed_dim vs hidden_size)
        attn = attn_cls(embed_dim, num_heads, projection_size)
    attn.to(DEVICE)
    attn.eval()

    # Precompute optional params
    max_seqlen = max(seqlens_list)
    # seqlens as python list for xformers path (if used)

    # Warmup
    with torch.no_grad():
        try:
            if pass_opt_params:
                out = attn(x, cu_seqlens, rotary, max_seqlen=max_seqlen, seqlens=seqlens_list)
            else:
                out = attn(x, cu_seqlens, rotary)
        except TypeError:
            # Fallback in case signature differs
            out = attn(x, cu_seqlens, rotary)

    # Timing
    t0 = time.perf_counter()
    with torch.no_grad():
        for _ in range(iters):
            try:
                if pass_opt_params:
                    out = attn(x, cu_seqlens, rotary, max_seqlen=max_seqlen, seqlens=seqlens_list)
                else:
                    out = attn(x, cu_seqlens, rotary)
            except TypeError:
                out = attn(x, cu_seqlens, rotary)
    t1 = time.perf_counter()

    ms = (t1 - t0) * 1000.0 / iters
    print(f"{name} pass_opt_params={pass_opt_params}: {ms:.3f} ms/iter (S={S}, groups={groups})")


def micro_bench_fallback(iters=200, groups=1024):
    # Simulate overhead of repeatedly computing seqlens and max from cu_seqlens
    seqlen_per = 8
    seqlens_list = [seqlen_per] * groups
    cu = build_cu_seqlens(seqlens_list)

    # Baseline: compute inside loop
    t0 = time.perf_counter()
    for _ in range(iters):
        # Compute seqlens and max every iteration (simulating old code)
        seqlens = (cu[1:] - cu[:-1]).tolist()
        max_seqlen = (cu[1:] - cu[:-1]).max().item()
        # do some dummy work with values to prevent optimization
        _ = max_seqlen + len(seqlens)
    t1 = time.perf_counter()
    base_ms = (t1 - t0) * 1000.0 / iters

    # Optimized: precompute once
    seqlens_once = (cu[1:] - cu[:-1]).tolist()
    max_once = max(seqlens_once)
    t2 = time.perf_counter()
    for _ in range(iters):
        # reuse
        _ = max_once + len(seqlens_once)
    t3 = time.perf_counter()
    opt_ms = (t3 - t2) * 1000.0 / iters

    print(f"MicroBench compute_in_loop={base_ms:.4f} ms/iter; precomputed={opt_ms:.4f} ms/iter")


def main():
    try:
        from vllm.model_executor.models.qwen2_vl import Qwen2VisionAttention
        from vllm.model_executor.models.qwen2_5_vl import Qwen2_5_VisionAttention
        # Baseline (without passing optional params)
        time_attention(Qwen2VisionAttention, "Qwen2VisionAttention", pass_opt_params=False)
        time_attention(Qwen2_5_VisionAttention, "Qwen2_5_VisionAttention", pass_opt_params=False)
        # Optimized path (attempt to pass optional params if supported)
        time_attention(Qwen2VisionAttention, "Qwen2VisionAttention", pass_opt_params=True)
        time_attention(Qwen2_5_VisionAttention, "Qwen2_5_VisionAttention", pass_opt_params=True)
    except Exception as e:
        print("Import of vllm models failed; running micro-benchmark fallback:", e)
        micro_bench_fallback()


if __name__ == "__main__":
    main()
