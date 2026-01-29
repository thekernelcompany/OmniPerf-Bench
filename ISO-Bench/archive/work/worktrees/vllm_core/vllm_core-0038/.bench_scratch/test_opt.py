import time
import torch

from vllm.attention.backends.torch_sdpa import (
    TorchSDPABackendImpl,
    TorchSDPAMetadata,
)


def run_once(num_tokens=2048, num_heads=16, head_size=128, num_kv_heads=None, dtype=torch.float32, alibi_slopes=None, sliding_window=None):
    if num_kv_heads is None:
        num_kv_heads = num_heads
    hidden_q = num_heads * head_size
    hidden_kv = num_kv_heads * head_size

    # Create random inputs
    query = torch.randn(num_tokens, hidden_q, dtype=dtype)
    key = torch.randn(num_tokens, hidden_kv, dtype=dtype)
    value = torch.randn(num_tokens, hidden_kv, dtype=dtype)

    # Build metadata for a single prompt sequence
    seq_lens = [num_tokens]
    slot_mapping = torch.zeros(num_tokens, dtype=torch.int32)
    meta = TorchSDPAMetadata(
        # AttentionMetadata
        num_prefills=1,
        num_prefill_tokens=num_tokens,
        num_decode_tokens=0,
        slot_mapping=slot_mapping,
        # PagedAttentionMetadata
        seq_lens_tensor=None,
        max_decode_seq_len=0,
        block_tables=None,
        # TorchSDPAMetadata
        is_prompt=True,
        seq_lens=seq_lens,
    )

    backend = TorchSDPABackendImpl(
        num_heads=num_heads,
        head_size=head_size,
        scale=1.0 / (head_size ** 0.5),
        num_kv_heads=num_kv_heads,
        alibi_slopes=alibi_slopes,
        sliding_window=sliding_window,
        kv_cache_dtype="auto",
        blocksparse_params=None,
    )

    out = backend.forward(query, key, value, kv_cache=None, attn_metadata=meta)
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    return out


def time_run(iters=5, masked=False):
    # Warmup
    if masked:
        alibi = [1.0 for _ in range(16)]
        run_once(alibi_slopes=alibi)
    else:
        run_once()
    t0 = time.time()
    for _ in range(iters):
        if masked:
            alibi = [1.0 for _ in range(16)]
            run_once(alibi_slopes=alibi)
        else:
            run_once()
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    t1 = time.time()
    return (t1 - t0) / iters


if __name__ == "__main__":
    avg = time_run(iters=3, masked=False)
    print(f"Average time per run (no mask): {avg:.6f} s")
    avg2 = time_run(iters=3, masked=True)
    print(f"Average time per run (alibi mask): {avg2:.6f} s")
