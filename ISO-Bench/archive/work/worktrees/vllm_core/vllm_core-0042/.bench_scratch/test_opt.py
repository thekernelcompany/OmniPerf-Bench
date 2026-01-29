import os
import time
import random
import statistics

# Ensure working directory is repo root when invoked
from vllm.distributed.kv_transfer.kv_connector.v1.p2p.p2p_nccl_connector import ReqMeta


def bench_make_meta(iters: int = 300, blocks: int = 64, block_size: int = 128):
    times = []
    for _ in range(iters):
        # simulate variable token and block counts per request
        n_tok = random.randint(block_size // 2, block_size * 2)
        n_blk = max(1, (n_tok + block_size - 1) // block_size)
        token_ids = list(range(n_tok))
        block_ids = list(range(n_blk))
        t0 = time.perf_counter()
        _ = ReqMeta.make_meta("req", token_ids, block_ids, block_size)
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1e6)
    return {
        "mean_us": statistics.mean(times),
        "p50_us": statistics.median(times),
        "p95_us": sorted(times)[int(0.95 * len(times)) - 1],
        "iters": iters,
    }


def main():
    print("Benchmark: ReqMeta.make_meta")
    res = bench_make_meta()
    print(res)


if __name__ == "__main__":
    main()
