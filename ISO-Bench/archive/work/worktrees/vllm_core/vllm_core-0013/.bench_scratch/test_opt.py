import time
import torch
import numpy as np

# Benchmark InputBatch initialization to stress tensor allocations
from vllm.v1.worker.gpu_input_batch import InputBatch


def bench_input_batch(iters: int = 10,
                      max_num_reqs: int = 256,
                      max_model_len: int = 2048,
                      max_num_blocks_per_req: int = 256):
    times = []
    for _ in range(iters):
        t0 = time.perf_counter()
        b = InputBatch(
            max_num_reqs=max_num_reqs,
            max_model_len=max_model_len,
            max_num_blocks_per_req=max_num_blocks_per_req,
            device=torch.device("cpu"),
            pin_memory=False,
            vocab_size=32000,
        )
        # touch a couple of arrays to ensure they are realized
        _ = b.token_ids_cpu_tensor.shape, b.temperature.shape
        t1 = time.perf_counter()
        times.append(t1 - t0)
        # cleanup
        del b
    return np.array(times)


if __name__ == "__main__":
    times = bench_input_batch()
    print({
        "iters": len(times),
        "mean_s": float(times.mean()),
        "min_s": float(times.min()),
        "max_s": float(times.max()),
    })
