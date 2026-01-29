import time
from dataclasses import dataclass

import numpy as np
import torch

from vllm.v1.serial_utils import MsgpackDecoder, MsgpackEncoder


@dataclass
class MyType:
    tensor1: torch.Tensor
    a_string: str
    list_of_tensors: list[torch.Tensor]
    numpy_array: np.ndarray
    small_f_contig_tensor: torch.Tensor
    large_f_contig_tensor: torch.Tensor
    small_non_contig_tensor: torch.Tensor
    large_non_contig_tensor: torch.Tensor


def make_obj():
    return MyType(
        tensor1=torch.randint(low=0, high=100, size=(1024,), dtype=torch.int32),
        a_string="hello",
        list_of_tensors=[
            torch.rand((1, 10), dtype=torch.float32),
            torch.rand((3, 5, 4000), dtype=torch.float64),
            torch.tensor(1984),
        ],
        numpy_array=np.arange(512),
        small_f_contig_tensor=torch.rand(5, 4).t(),
        large_f_contig_tensor=torch.rand(1024, 4).t(),
        small_non_contig_tensor=torch.rand(2, 4)[:, 1:3],
        large_non_contig_tensor=torch.rand(1024, 512)[:, 10:20],
    )


def bench(n_iters: int = 1000):
    obj = make_obj()
    enc = MsgpackEncoder(size_threshold=256)
    dec = MsgpackDecoder(MyType)

    # Warmup
    for _ in range(10):
        bufs = enc.encode(obj)
        _ = dec.decode(bufs)

    t0 = time.perf_counter()
    for _ in range(n_iters):
        bufs = enc.encode(obj)
        _ = dec.decode(bufs)
    t1 = time.perf_counter()

    print(f"encode+decode loops: {n_iters}, time: {t1 - t0:.6f}s, per: {(t1 - t0)/n_iters*1000:.3f} ms")


if __name__ == "__main__":
    bench(1000)
