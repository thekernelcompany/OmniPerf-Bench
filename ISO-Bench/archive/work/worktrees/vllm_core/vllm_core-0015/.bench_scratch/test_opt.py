
import time
import torch

def bench_alloc(shape, iters=2000):
    # Measure allocation-only cost (no computation), CPU tensors
    t0 = time.time()
    for _ in range(iters):
        _ = torch.zeros(shape)
    t1 = time.time()
    for _ in range(iters):
        _ = torch.empty(shape)
    t2 = time.time()
    return (t1 - t0), (t2 - t1)

if __name__ == '__main__':
    shape = (1024, 256)
    iters = 3000
    zeros_t, empty_t = bench_alloc(shape, iters)
    print(f'shape={shape}, iters={iters}')
    print(f'zeros_alloc_time={zeros_t:.6f}s')
    print(f'empty_alloc_time={empty_t:.6f}s')
