import random
import time
from vllm.core.evictor import LRUEvictor


def bench_lru(n_blocks: int = 20000, update_ratio: float = 0.5, seed: int = 42):
    random.seed(seed)
    ev = LRUEvictor()

    t0 = time.perf_counter()
    for i in range(n_blocks):
        # stagger last_accessed; num_hashed_tokens in small range
        ev.add(i, content_hash=i * 13 + 7, num_hashed_tokens=random.randint(1, 8), last_accessed=float(i))
    t1 = time.perf_counter()

    # updates
    k = int(n_blocks * update_ratio)
    ids = random.sample(range(n_blocks), k)
    for idx, bid in enumerate(ids):
        ev.update(bid, last_accessed=float(n_blocks + idx))
    t2 = time.perf_counter()

    # evict all
    for _ in range(n_blocks):
        ev.evict()
    t3 = time.perf_counter()

    print({
        'n_blocks': n_blocks,
        'add_s': round(t1 - t0, 6),
        'update_s': round(t2 - t1, 6),
        'evict_s': round(t3 - t2, 6),
        'total_s': round(t3 - t0, 6),
    })


if __name__ == '__main__':
    bench_lru()
