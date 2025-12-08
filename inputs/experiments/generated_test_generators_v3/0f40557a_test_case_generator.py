# tests/kernels/test_copy_and_sample.py
#
# Tests for:
#  - cacheflow.cache_ops.copy_blocks (new vectorized copy kernel)
#  - cacheflow.worker.cache_engine.CacheEngine.copy wrapper behavior
#  - cacheflow.models.sample._sample_from_generation_tokens beam-search / greedy branches
#
# These tests require a CUDA-capable environment for kernel tests. The sampling tests
# do not require CUDA.

import random
import torch
import pytest

from cacheflow import cache_ops
from cacheflow.worker.cache_engine import CacheEngine
from cacheflow.models.sample import _sample_from_generation_tokens

# Helper lightweight sampling params object to mimic the interface expected by the
# sampler functions used in the codebase.
class _SamplingParams:
    def __init__(self, use_beam_search=False, temperature=1.0, n=1, num_logprobs=0):
        self.use_beam_search = use_beam_search
        self.temperature = temperature
        self.n = n
        self.num_logprobs = num_logprobs

# Mark GPU tests to skip if CUDA isn't available (these test the CUDA kernel).
requires_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for kernel tests")


@requires_cuda
def test_copy_blocks_kernel_copies_blocks_consistently():
    """
    Functional test for cache_ops.copy_blocks:
    - Create a small multi-layer key/value cache (random content)
    - Define a mapping src -> [dst]
    - Call copy_blocks and check the result matches a CPU-side reference behavior:
      for each mapping, dst block should equal src block for every layer.
    """
    torch.manual_seed(42)
    random.seed(42)

    # Small dimensions to keep the test fast while exercising the kernel.
    num_layers = 2
    num_blocks = 6
    num_heads = 2
    head_size = 8
    block_size = 3
    dtype = torch.float32

    # Compute x (packing factor) consistent with kernel code: x = 16 / element_size
    x = 16 // torch.tensor([], dtype=dtype).element_size()
    assert head_size % x == 0, "head_size must be divisible by packing factor x for the key layout."

    # Build key_caches and value_caches as lists of CUDA tensors (one tensor per layer)
    key_cache_shape = (num_blocks, num_heads, head_size // x, block_size, x)
    value_cache_shape = (num_blocks, num_heads, head_size, block_size)

    key_caches = []
    value_caches = []
    for _ in range(num_layers):
        key = torch.randn(size=key_cache_shape, dtype=dtype, device='cuda')
        val = torch.randn(size=value_cache_shape, dtype=dtype, device='cuda')
        key_caches.append(key)
        value_caches.append(val)

    # Clone for reference computation on CPU (we'll keep GPU clones and compute expected by copying clones)
    cloned_key_caches = [k.clone() for k in key_caches]
    cloned_value_caches = [v.clone() for v in value_caches]

    # Build a mapping: pick num_mappings unique src blocks and dests (no overlap to keep it simple)
    num_mappings = min(3, num_blocks // 2)
    src_blocks = random.sample(range(num_blocks), num_mappings)
    remaining = list(set(range(num_blocks)) - set(src_blocks))
    dst_blocks = random.sample(remaining, num_mappings)
    block_mapping = {s: [d] for s, d in zip(src_blocks, dst_blocks)}

    # Call the kernel under test.
    # The kernel is asynchronous; we'll synchronize after to ensure completion.
    cache_ops.copy_blocks(key_caches, value_caches, block_mapping)
    torch.cuda.synchronize()

    # Compute expected state from cloned copies by performing the equivalent copy logic
    for src, dsts in block_mapping.items():
        for dst in dsts:
            for cloned_key in cloned_key_caches:
                # Equivalent of cloned_key[dst] = cloned_key[src]
                cloned_key[dst] = cloned_key[src]
            for cloned_val in cloned_value_caches:
                cloned_val[dst] = cloned_val[src]

    # Compare actual caches against the expected cloned caches
    for actual, expected in zip(key_caches, cloned_key_caches):
        assert torch.allclose(actual, expected), "Key cache mismatch after copy_blocks"
    for actual, expected in zip(value_caches, cloned_value_caches):
        assert torch.allclose(actual, expected), "Value cache mismatch after copy_blocks"


@requires_cuda
def test_cache_engine_copy_invokes_copy_blocks_with_lists(monkeypatch):
    """
    Regression test for CacheEngine.copy wrapper:
    - Ensure that CacheEngine.copy constructs lists of key/value caches and calls cache_ops.copy_blocks
      with the provided mapping, reflecting the API change.
    - Use monkeypatch to capture the call arguments rather than invoking the real kernel again.
    """
    called = {}

    def fake_copy_blocks(key_caches_arg, value_caches_arg, src_to_dsts_arg):
        # Record call and validate minimal invariants.
        called['key_caches'] = key_caches_arg
        called['value_caches'] = value_caches_arg
        called['mapping'] = src_to_dsts_arg
        # Do not mutate tensors; this is only to capture the call.

    monkeypatch.setattr(cache_ops, "copy_blocks", fake_copy_blocks)

    # Instantiate the CacheEngine with small dimensions (allocations will be on GPU).
    # Use head_size multiple of 16 to satisfy constructor validation; pick a multiple consistent with x.
    worker = CacheEngine(
        worker_id=0,
        num_layers=3,
        num_heads=2,
        head_size=16,
        block_size=4,
        num_gpu_blocks=8,
        num_cpu_blocks=0,
        dtype=torch.float32,
    )

    # Prepare a sample mapping
    mapping = {0: [1, 2], 3: [4]}

    # Call the wrapper under test.
    worker.copy(mapping)

    # Validate that the patched copy_blocks was invoked and received lists of tensors.
    assert 'mapping' in called, "CacheEngine.copy did not call cache_ops.copy_blocks"
    assert called['mapping'] == mapping, "CacheEngine.copy forwarded an incorrect mapping"
    assert isinstance(called['key_caches'], list) and isinstance(called['value_caches'], list)
    assert len(called['key_caches']) == worker.num_layers
    assert len(called['value_caches']) == worker.num_layers
    # Each element should be a tensor on CUDA
    for t in called['key_caches'] + called['value_caches']:
        assert isinstance(t, torch.Tensor) and t.device.type == 'cuda'


def test_sample_from_generation_tokens_beam_search_and_greedy():
    """
    Unit tests for _sample_from_generation_tokens covering:
    - Beam search branch (use_beam_search=True)
    - Greedy branch (temperature == 0.0 and single seq_id)
    - Negative test: greedy branch assertion when multiple seq_ids
    """
    torch.manual_seed(123)

    # Beam search case:
    # Prepare seq_ids and logprobs such that the flattened topk picks predictable (seq_idx, token)
    seq_ids = [10, 11, 12]  # arbitrary sequence IDs; function uses them only to build outputs
    beam_width = len(seq_ids)
    vocab_size = 7

    # Create logprobs so that for each i, the position i * vocab_size + i is the largest.
    # We'll make all other entries very small.
    logprobs = torch.full((beam_width, vocab_size), -1000.0)
    for i in range(beam_width):
        token_choice = i % vocab_size
        logprobs[i, token_choice] = float(100 + i)  # distinct high scores for deterministic topk

    # probs is not used for beam search branch except for shapes; pass zeros
    probs = torch.zeros_like(logprobs)

    # seq_logprobs are the cumulative logprob for each sequence; pick zeros for simplicity
    seq_logprobs = [0.0 for _ in seq_ids]

    sampling_params = _SamplingParams(use_beam_search=True)

    parent_ids, next_tokens = _sample_from_generation_tokens(
        seq_ids=seq_ids,
        probs=probs,
        logprobs=logprobs,
        seq_logprobs=seq_logprobs,
        sampling_params=sampling_params,
    )

    # For the constructed logprobs the token chosen for each seq should be i % vocab_size
    expected_tokens = [i % vocab_size for i in range(beam_width)]
    assert parent_ids == seq_ids, "Beam-search parent ids should equal input seq_ids in this canonical case"
    assert next_tokens == expected_tokens, f"Expected tokens {expected_tokens} but got {next_tokens}"

    # Greedy case (temperature == 0.0), must have a single seq_id (assertion enforced)
    single_seq_ids = [42]
    vocab_size = 5
    probs_single = torch.tensor([0.1, 0.2, 0.5, 0.05, 0.15])
    # reshape to [1, vocab_size] to match expected shape
    probs_single = probs_single.unsqueeze(0)
    logprobs_single = torch.log(probs_single)

    sampling_params_greedy = _SamplingParams(use_beam_search=False, temperature=0.0)
    parent_ids_greedy, next_tokens_greedy = _sample_from_generation_tokens(
        seq_ids=single_seq_ids,
        probs=probs_single,
        logprobs=logprobs_single,
        seq_logprobs=[0.0],
        sampling_params=sampling_params_greedy,
    )

    # Greedy should pick argmax token (which is 2 in our probs)
    assert parent_ids_greedy == single_seq_ids
    assert next_tokens_greedy == [int(torch.argmax(probs_single[0]).item())]

    # Negative test: greedy with multiple seq_ids should raise an assertion error
    with pytest.raises(AssertionError):
        _sample_from_generation_tokens(
            seq_ids=[1, 2],  # >1 length triggers the assertion
            probs=torch.rand((2, 5)),
            logprobs=torch.rand((2, 5)),
            seq_logprobs=[0.0, 0.0],
            sampling_params=sampling_params_greedy,
        )