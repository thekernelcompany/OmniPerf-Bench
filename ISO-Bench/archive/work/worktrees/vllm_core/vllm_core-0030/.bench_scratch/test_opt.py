import time
import os
import sys
import types

# Lightweight package stub to avoid heavy vllm.__init__ side effects
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
VLLM_DIR = os.path.join(REPO_ROOT, 'vllm')
if 'vllm' not in sys.modules:
    vllm_pkg = types.ModuleType('vllm')
    vllm_pkg.__path__ = [VLLM_DIR]
    sys.modules['vllm'] = vllm_pkg

import sys
import types

# Stub minimal triton runtime cache to allow importing vllm without full deps
if 'triton.runtime.cache' not in sys.modules:
    triton_mod = types.ModuleType('triton')
    triton_runtime = types.ModuleType('triton.runtime')
    triton_runtime_cache = types.ModuleType('triton.runtime.cache')

    def _default_cache_dir():
        return '/tmp'

    class _FileCacheManager:
        def __init__(self, *args, **kwargs):
            pass

    triton_runtime_cache.default_cache_dir = _default_cache_dir
    triton_runtime_cache.FileCacheManager = _FileCacheManager

    triton_runtime.cache = triton_runtime_cache
    triton_mod.runtime = triton_runtime

    sys.modules['triton'] = triton_mod
    sys.modules['triton.runtime'] = triton_runtime
    sys.modules['triton.runtime.cache'] = triton_runtime_cache

from typing import List, Optional

import torch  # noqa: F401

from vllm.core.block.prefix_caching_block import PrefixCachingBlockAllocator
from vllm.core.block.interfaces import Block
from vllm.core.block_manager_v1 import BlockSpaceManagerV1
from vllm.sequence import Sequence, SequenceGroup, SequenceStatus


def bench_prefix_caching_allocator(
    block_size: int = 16,
    num_blocks: int = 256,
    num_sequences: int = 64,
    common_prefix_blocks: int = 8,
) -> float:
    allocator = PrefixCachingBlockAllocator(num_blocks=num_blocks,
                                            block_size=block_size)

    # Common token IDs for shared prefix
    common_token_ids: List[int] = list(range(block_size * common_prefix_blocks))

    start = time.time()

    # Allocate blocks for multiple sequences with common prefixes
    for _ in range(num_sequences):
        prev_block: Optional[Block] = None
        for block_idx in range(common_prefix_blocks):
            start_idx = block_idx * block_size
            end_idx = start_idx + block_size
            token_ids = common_token_ids[start_idx:end_idx]

            prev_block = allocator.allocate_immutable_block(
                prev_block=prev_block, token_ids=token_ids)

    # Some implementations require explicit marking of computed blocks; handle gracefully
    try:
        allocator.mark_blocks_as_computed([])  # type: ignore[attr-defined]
    except Exception:
        pass

    end = time.time()
    return end - start


def make_sequence(seq_id: int, block_size: int, num_blocks: int) -> Sequence:
    # Minimal inputs to construct a Sequence
    prompt_token_ids = list(range(block_size * num_blocks))
    inputs = {"prompt_token_ids": prompt_token_ids, "prompt": None}
    return Sequence(seq_id=seq_id, inputs=inputs, block_size=block_size)


def bench_block_manager_mark_computed(
    block_size: int = 16,
    nseqs: int = 128,
    prompt_blocks: int = 16,
    num_gpu_blocks: int = 4096,
    num_cpu_blocks: int = 0,
) -> float:
    # Construct sequences and group
    seqs = [make_sequence(i, block_size, prompt_blocks) for i in range(nseqs)]
    for s in seqs:
        s.status = SequenceStatus.WAITING

    sg = SequenceGroup(request_id="req-0",
                       seqs=seqs,
                       arrival_time=time.time())

    # Allocate with BlockSpaceManagerV1
    mgr = BlockSpaceManagerV1(block_size=block_size,
                              num_gpu_blocks=num_gpu_blocks,
                              num_cpu_blocks=num_cpu_blocks,
                              enable_caching=True)
    mgr.allocate(sg)

    # Time the mark_blocks_as_computed which is the optimized path
    start = time.time()
    mgr.mark_blocks_as_computed(sg)
    end = time.time()
    return end - start


if __name__ == "__main__":
    # Example 1: Prefix caching allocator
    d1 = bench_prefix_caching_allocator()
    print(f"PrefixCachingBlockAllocator duration: {d1:.6f}s")

    # Example 2: BlockSpaceManagerV1 mark computed
    d2 = bench_block_manager_mark_computed()
    print(f"BlockSpaceManagerV1.mark_blocks_as_computed duration: {d2:.6f}s")
