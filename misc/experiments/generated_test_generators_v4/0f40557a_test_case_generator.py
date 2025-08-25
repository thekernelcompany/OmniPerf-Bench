# tests/test_copy_and_sampling.py
# PyTest-compatible test module that covers:
#  - cache_ops.copy_blocks GPU kernel behavior (functional correctness)
#  - CacheEngine.copy wiring to cache_ops.copy_blocks (argument shapes)
#  - sample._sample_from_generation_tokens beam-search indexing behavior
#
# Notes:
#  - GPU tests are skipped when CUDA is not available.
#  - The cache kernel test uses small tensor sizes to keep runtime low.
#  - The CacheEngine copy wiring test uses monkeypatch to avoid invoking the GPU kernel.
#
# Run with: pytest -q tests/test_copy_and_sampling.py

import random
from typing import Dict, List, Tuple

import pytest
import torch

# Import the project's modules. These imports assume that the package root
# is on PYTHONPATH. Adjust imports if your test runner config differs.
from cacheflow import cache_ops
from cacheflow.worker.cache_engine import CacheEngine
from cacheflow.models.sample import _sample_from_generation_tokens
from cacheflow.sampling_params import SamplingParams


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required for kernel tests")
def test_copy_blocks_kernel_small_end_to_end():
    """
    Functional test for cache_ops.copy_blocks:
    - Build small per-layer key/value cache tensors on GPU.
    - Define a block mapping that copies a small set of source blocks to destination blocks.
    - Call cache_ops.copy_blocks and compare results with a reference CPU-side copy.
    """
    dtype = torch.float32
    num_layers = 2
    # Small shapes for quick tests
    num_heads = 1
    head_size = 8  # must be consistent with x computation below
    block_size = 2
    num_blocks = 4  # per-layer blocks

    # x is computed the same way as in code (16 / element_size)
    x = 16 // torch.tensor([], dtype=dtype).element_size()

    # key_cache shape: (num_blocks, num_heads, head_size // x, block_size, x)
    key_cache_shape = (num_blocks, num_heads, head_size // x, block_size, x)
    # value_cache shape: (num_blocks, num_heads, head_size, block_size)
    value_cache_shape = (num_blocks, num_heads, head_size, block_size)

    # Create per-layer caches and their clones for reference comparison.
    key_caches: List[torch.Tensor] = []
    cloned_key_caches: List[torch.Tensor] = []
    value_caches: List[torch.Tensor] = []
    cloned_value_caches: List[torch.Tensor] = []

    for _ in range(num_layers):
        kc = torch.randn(size=key_cache_shape, dtype=dtype, device='cuda')
        vc = torch.randn(size=value_cache_shape, dtype=dtype, device='cuda')
        key_caches.append(kc)
        value_caches.append(vc)
        cloned_key_caches.append(kc.clone())
        cloned_value_caches.append(vc.clone())

    # Create a mapping: randomly map some source blocks to destination blocks.
    # Ensure no overlap of source and dest to keep reference simple.
    num_mappings = 2
    all_blocks = list(range(num_blocks))
    src_blocks = random.sample(all_blocks, num_mappings)
    remaining = list(set(all_blocks) - set(src_blocks))
    dst_blocks = random.sample(remaining, num_mappings)
    block_mapping: Dict[int, List[int]] = {s: [d] for s, d in zip(src_blocks, dst_blocks)}

    # Execute kernel (this code exercises the new API that accepts lists of tensors).
    cache_ops.copy_blocks(key_caches, value_caches, block_mapping)

    # Reference implementation: for each (src -> dst) pair, assert dst == src after copy.
    # Since we cloned the caches before the kernel call, we simulate what the kernel
    # should have done on the clones; then compare.
    for src, dsts in block_mapping.items():
        for dst in dsts:
            for k_clone in cloned_key_caches:
                # In the reference, copying a block sets destination block to the source block.
                k_clone[dst] = k_clone[src]
            for v_clone in cloned_value_caches:
                v_clone[dst] = v_clone[src]

    # Compare kernel results with the reference clones
    for actual, expected in zip(key_caches, cloned_key_caches):
        assert torch.allclose(actual, expected), "key_cache mismatch after copy_blocks"
    for actual, expected in zip(value_caches, cloned_value_caches):
        assert torch.allclose(actual, expected), "value_cache mismatch after copy_blocks"


def test_cache_engine_copy_calls_cache_ops(monkeypatch):
    """
    Unit test to assert CacheEngine.copy constructs the expected arguments and
    delegates to cache_ops.copy_blocks with lists of per-layer tensors.

    This test does not require GPU; we monkeypatch cache_ops.copy_blocks to avoid
    requiring CUDA allocations. We create a dummy CacheEngine-like object by
    instantiating CacheEngine but monkeypatch its allocation functions to use
    CPU tensors so instantiation won't try to allocate CUDA memory.
    """
    called = {}

    def fake_copy_blocks(key_caches, value_caches, mapping):
        # Record that copy_blocks was called and capture argument shapes/types.
        called['key_caches_len'] = len(key_caches)
        called['value_caches_len'] = len(value_caches)
        # capture first tensor shapes in a simple serializable form
        called['first_key_shape'] = tuple(key_caches[0].shape)
        called['first_value_shape'] = tuple(value_caches[0].shape)
        called['mapping'] = mapping
        # Return None (original function returns void)
        return None

    # Monkeypatch the cache_ops.copy_blocks symbol used by CacheEngine
    monkeypatch.setattr(cache_ops, 'copy_blocks', fake_copy_blocks)

    # To avoid CUDA allocations in CacheEngine.__init__, we monkeypatch its
    # allocate_gpu_cache and allocate_cpu_cache to allocate small CPU tensors.
    from cacheflow.worker import cache_engine as ce_mod

    def fake_allocate_gpu_cache(self):
        # Return small CPU tensors (the code under test only packages them into lists)
        dummy_key = torch.zeros((2, 1, 1, 1, 1), dtype=torch.float32)
        dummy_value = torch.zeros((2, 1, 1, 1), dtype=torch.float32)
        return [(dummy_key, dummy_value) for _ in range(self.num_layers)]

    def fake_allocate_cpu_cache(self):
        dummy_key = torch.zeros((2, 1, 1, 1, 1), dtype=torch.float32)
        dummy_value = torch.zeros((2, 1, 1, 1), dtype=torch.float32)
        return [(dummy_key, dummy_value) for _ in range(self.num_layers)]

    monkeypatch.setattr(ce_mod.CacheEngine, 'allocate_gpu_cache', fake_allocate_gpu_cache)
    monkeypatch.setattr(ce_mod.CacheEngine, 'allocate_cpu_cache', fake_allocate_cpu_cache)

    # Now construct CacheEngine with small, valid parameters.
    # head_size must be multiple of 16 in constructor; choose 16.
    engine = ce_mod.CacheEngine(
        worker_id=0,
        num_layers=3,
        num_heads=1,
        head_size=16,
        block_size=2,
        num_gpu_blocks=2,
        num_cpu_blocks=2,
        dtype=torch.float32,
    )

    # Prepare a mapping and call copy().
    mapping = {0: [1, 2], 1: [3]}
    engine.copy(mapping)

    # Verify that our fake_copy_blocks was invoked and captured the args.
    assert 'key_caches_len' in called, "copy_blocks was not called by CacheEngine.copy"
    assert called['key_caches_len'] == engine.num_layers
    assert called['value_caches_len'] == engine.num_layers
    assert called['mapping'] == mapping
    # Shapes are those created by fake_allocate_* functions
    assert called['first_key_shape'] == (2, 1, 1, 1, 1)
    assert called['first_value_shape'] == (2, 1, 1, 1)


def test_sample_from_generation_tokens_beam_search_indexing():
    """
    Test the _sample_from_generation_tokens beam-search branch where the code
    flattens logprobs and selects topk from the flattened tensor, and then
    uses Python list arithmetic to compute seq_idx and token_ids.

    We craft a small logprobs tensor so that the topk indices are predictable:
    - vocab_size = 3, seq_len = 2 -> flattened size = 6
    - Place two very large values at flattened indices 5 and 2 to force selection order [5,2]
    Expectation: seq_idx = [5 // 3 = 1, 2 // 3 = 0] -> beam_seq_ids order [seq_ids[1], seq_ids[0]]
                 token_ids = [5 % 3 = 2, 2 % 3 = 2]
    """
    # two sequences in the group
    seq_ids = [10, 11]
    vocab_size = 3

    # Create logprobs shaped (2, 3) with large values at flattened indices 5 and 2.
    # Row-major flattening order: row0 indices 0..2, row1 indices 3..5.
    # Put e.g. 100 at idx 2 and 200 at idx 5.
    row0 = [0.0, 0.0, 100.0]  # flattened indices 0,1,2
    row1 = [0.0, 0.0, 200.0]  # flattened indices 3,4,5
    probs = torch.softmax(torch.tensor([row0, row1]), dim=-1)
    logprobs = torch.log(probs)

    # sampling_params: enable beam search
    sampling_params = SamplingParams.from_dict({
        'n': len(seq_ids),
        'temperature': 0.0,  # not used in beam-search branch (use_beam_search True takes precedence)
        'top_p': 1.0,
        'use_beam_search': True,
        'stop_token_ids': set(),
        'max_num_steps': 1,
        'num_logprobs': 0,
    })

    # seq_logprobs (cumulative logprobs across sequences in the group)
    seq_logprobs = [0.0 for _ in seq_ids]

    parent_seq_ids, next_token_ids = _sample_from_generation_tokens(
        seq_ids, probs, logprobs, seq_logprobs, sampling_params)

    # Validate sizes
    assert len(parent_seq_ids) == len(seq_ids)
    assert len(next_token_ids) == len(seq_ids)

    # For our chosen values, flatten topk should pick indices [5, 2] -> seq_idx [1, 0] -> beam_seq_ids [11, 10]
    # The algorithm ensures each seq_id appears in the final beam_outputs exactly once; mapping specifics:
    # parent_seq_ids is a list aligned to seq_ids order (returned by function). Check token selection validity.
    # Since there may be nondeterminism in ordering for ties etc., we assert that produced token ids are in {2}
    assert all(tid == 2 for tid in next_token_ids), f"Expected token ids to be 2 but got {next_token_ids}"

    # parent_seq_ids should be a permutation of the original seq_ids (they point to parent sequence for each output)
    assert set(parent_seq_ids) == set(seq_ids), "parent_seq_ids should be the same set as input seq_ids"


# End of test module