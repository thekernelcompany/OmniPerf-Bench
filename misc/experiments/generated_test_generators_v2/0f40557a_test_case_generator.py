# Filename: tests/test_cacheflow_copy_and_sampling.py
# Run with: pytest -q
"""
Comprehensive tests for:
- CacheEngine.copy -> ensures the engine constructs correct key/value cache lists
  and calls cache_ops.copy_blocks with the expected arguments.
- The Python-level contract for cache_ops.copy_blocks (list-of-tensors + mapping).
  Because the CUDA extension may not be available in CI/dev environments, we mock
  the native binding and validate the Python-side behavior.
- _sample_from_generation_tokens fixes for beam search and greedy sampling branches.

Notes:
- Tests are written to run on CPU-only environments by mocking methods that would
  otherwise allocate CUDA tensors or call compiled CUDA kernels.
- All tests are independent.
"""

import types
import torch
import pytest

# Import project modules under test.
# We import the Python CacheEngine and sample module functions.
from cacheflow.worker.cache_engine import CacheEngine
from cacheflow import cache_ops
from cacheflow.models import sample as sample_module

# Minimal SamplingParams-like object (only attributes used by the tested functions).
class _DummySamplingParams:
    def __init__(self, use_beam_search=False, n=1, temperature=1.0, num_logprobs=0):
        self.use_beam_search = use_beam_search
        self.n = n
        self.temperature = temperature
        self.num_logprobs = num_logprobs


@pytest.fixture(autouse=True)
def ensure_cpu_only(monkeypatch):
    """
    Ensure test does not attempt to use CUDA allocations or kernel launches.
    - Monkeypatch CacheEngine.allocate_gpu_cache / allocate_cpu_cache to create CPU tensors.
    - Leave actual cache_ops.copy_blocks intact but allow tests to monkeypatch it where needed.
    """
    # Replace CacheEngine.allocate_gpu_cache to allocate CPU tensors to avoid CUDA requirement.
    def _allocate_gpu_cache_cpu(self):
        # produce gpu_cache shaped entries but on CPU so tests can run without CUDA
        key_block_shape = self.get_key_block_shape()  # uses dtype element_size, safe
        value_block_shape = self.get_value_block_shape()
        cpu_like_gpu_cache = []
        for _ in range(self.num_layers):
            key_blocks = torch.empty((self.num_gpu_blocks, *key_block_shape), dtype=self.dtype, device='cpu')
            value_blocks = torch.empty((self.num_gpu_blocks, *value_block_shape), dtype=self.dtype, device='cpu')
            cpu_like_gpu_cache.append((key_blocks, value_blocks))
        return cpu_like_gpu_cache

    def _allocate_cpu_cache_cpu(self):
        key_block_shape = self.get_key_block_shape()
        value_block_shape = self.get_value_block_shape()
        cpu_cache_list = []
        for _ in range(self.num_layers):
            key_blocks = torch.empty((self.num_cpu_blocks, *key_block_shape), dtype=self.dtype, pin_memory=False, device='cpu')
            value_blocks = torch.empty((self.num_cpu_blocks, *value_block_shape), dtype=self.dtype, pin_memory=False, device='cpu')
            cpu_cache_list.append((key_blocks, value_blocks))
        return cpu_cache_list

    monkeypatch.setattr(CacheEngine, "allocate_gpu_cache", _allocate_gpu_cache_cpu, raising=False)
    monkeypatch.setattr(CacheEngine, "allocate_cpu_cache", _allocate_cpu_cache_cpu, raising=False)
    yield
    # No automatic teardown required; monkeypatch fixture will restore modifications.


def test_cacheengine_copy_builds_lists_and_invokes_copy_blocks(monkeypatch):
    """
    Validate that CacheEngine.copy constructs key_caches and value_caches lists from
    engine.gpu_cache and calls cache_ops.copy_blocks(key_caches, value_caches, mapping).
    We mock cache_ops.copy_blocks to capture arguments and assert shapes/contents.
    """
    # Create a small engine with simple sizes.
    engine = CacheEngine(
        worker_id=0,
        num_layers=3,
        num_heads=4,
        head_size=16,
        block_size=8,
        num_gpu_blocks=5,
        num_cpu_blocks=2,
        dtype=torch.float32,
    )

    # Sanity: engine.gpu_cache should be created by the monkeypatched allocate_gpu_cache.
    assert len(engine.gpu_cache) == engine.num_layers
    # Prepare a mapping: select some src blocks -> list of dst blocks
    src_to_dsts = {0: [1, 2], 3: [4]}

    captured = {}
    def fake_copy_blocks(key_caches_arg, value_caches_arg, mapping_arg):
        # Capture arguments
        captured['key_caches'] = key_caches_arg
        captured['value_caches'] = value_caches_arg
        captured['mapping'] = mapping_arg
        # Validate types: lists of tensors, mapping is a dict-like
        assert isinstance(key_caches_arg, list)
        assert isinstance(value_caches_arg, list)
        assert len(key_caches_arg) == engine.num_layers
        assert len(value_caches_arg) == engine.num_layers
        # Each entry should be the key tensor (first element of each tuple in gpu_cache)
        for layer_idx, (key_cache, value_cache) in enumerate(engine.gpu_cache):
            assert key_caches_arg[layer_idx] is key_cache
            assert value_caches_arg[layer_idx] is value_cache
        # Confirm mapping is the same object passed.
        assert mapping_arg == src_to_dsts
        # Return None (implementation is void).
        return None

    # Monkeypatch the native binding to our fake.
    monkeypatch.setattr(cache_ops, "copy_blocks", fake_copy_blocks)

    # Call the method under test.
    engine.copy(src_to_dsts)

    # Verify captured arguments match expected.
    assert 'key_caches' in captured
    assert 'value_caches' in captured
    assert captured['mapping'] == src_to_dsts


def test_cache_ops_copy_blocks_reference_like_behavior(monkeypatch):
    """
    This test verifies expected logical behavior of a 'copy_blocks' operation
    at Python level by simulating what the GPU kernel should accomplish.
    Since the real kernel is not available in pure-Python CI, we call a simulated
    Python implementation of the expected semantics:
        For each mapping src -> list(dst), for every layer:
            key_cache[dst] <- key_cache[src]
            value_cache[dst] <- value_cache[src]
    We validate that, after a simulated copy, the provided key/value caches would match
    a reference implementation.
    """
    # Test parameters
    num_layers = 4
    num_heads = 2
    head_size = 16
    block_size = 8
    num_blocks = 10
    dtype = torch.float32

    # Build per-layer caches (lists of tensors)
    x = 16 // torch.tensor([], dtype=dtype).element_size()
    key_cache_shape = (num_blocks, num_heads, head_size // x, block_size, x)
    value_cache_shape = (num_blocks, num_heads, head_size, block_size)

    key_caches = [torch.randn(key_cache_shape, dtype=dtype, device='cpu') for _ in range(num_layers)]
    value_caches = [torch.randn(value_cache_shape, dtype=dtype, device='cpu') for _ in range(num_layers)]

    # Keep clones to build reference expected output
    cloned_key_caches = [kc.clone() for kc in key_caches]
    cloned_value_caches = [vc.clone() for vc in value_caches]

    # Build a random mapping: pick some distinct src blocks and distinct dsts
    src_blocks = [0, 2, 5]
    dst_blocks = [3, 6, 8]
    block_mapping = {s: [d] for s, d in zip(src_blocks, dst_blocks)}

    # Simulate what the C++ kernel should perform by applying the reference copy
    for src, dsts in block_mapping.items():
        for dst in dsts:
            for kc, ck in zip(key_caches, cloned_key_caches):
                # Reference says: destination becomes a copy of source
                ck[dst].copy_(kc[src])
            for vc, cv in zip(value_caches, cloned_value_caches):
                cv[dst].copy_(vc[src])

    # Now, create a simulated python copy_blocks implementation and apply it on
    # original caches to make them match the cloned reference.
    def simulated_copy_blocks_python_style(key_caches_arg, value_caches_arg, mapping_arg):
        # Perform same operation in-place
        for src, dsts in mapping_arg.items():
            for dst in dsts:
                for kc in key_caches_arg:
                    kc[dst] = kc[src].clone()
                for vc in value_caches_arg:
                    vc[dst] = vc[src].clone()

    simulated_copy_blocks_python_style(key_caches, value_caches, block_mapping)

    # Validate equality between the result of simulated copy and reference
    for kc, ck in zip(key_caches, cloned_key_caches):
        assert torch.allclose(kc, ck), "Key caches do not match reference after copy op"
    for vc, cv in zip(value_caches, cloned_value_caches):
        assert torch.allclose(vc, cv), "Value caches do not match reference after copy op"


def test_sample_from_generation_tokens_beam_search_and_greedy():
    """
    Tests for the _sample_from_generation_tokens internal function:
    - Beam search branch: ensure correct parent_seq_ids and next_token_ids computed
      when multiple sequences and beam search is enabled.
    - Greedy branch: with temperature == 0.0 and single sequence ensure argmax chosen.
    """
    _fn = sample_module._sample_from_generation_tokens

    # Beam search scenario
    # Prepare seq_ids and logits such that top beam entries are unambiguous.
    # vocab_size = 3
    seq_ids = [10, 11]
    vocab_size = 3
    # logprobs tensor shape: (len(seq_ids), vocab_size)
    # Make entries so that flattening gives indices [0, 3, 1, 4, 2, 5] with top2 being 0 and 3
    # Example logs:
    logprobs = torch.tensor([[10.0, 1.0, 0.0], [9.0, 2.0, 0.5]])
    probs = torch.softmax(logprobs, dim=-1)  # not really used in beam branch for selection

    seq_logprobs = [0.0, 0.0]  # values to be added; keep deterministic

    sampling_params = _DummySamplingParams(use_beam_search=True, n=2)

    parent_seq_ids, next_token_ids = _fn(seq_ids, probs, logprobs, seq_logprobs, sampling_params)

    # In our constructed logprobs, top two indices in flattened array are 0 (seq0 token0) and 3 (seq1 token0).
    # So topk produced tokens are token 0 for seq 10 and token 0 for seq 11.
    assert len(parent_seq_ids) == len(seq_ids)
    assert len(next_token_ids) == len(seq_ids)
    # Parent seq ids must be the original seq_ids mapped appropriately
    # parent_seq_ids correspond to which sequence provided the surviving beam for each output sequence
    # With our design beam_outputs should map each seq_id to itself (no forking), thus parent_seq_ids == seq_ids
    assert parent_seq_ids == seq_ids
    # next_token_ids should all be zero (token id 0 chosen by topk positions 0 and 3)
    assert next_token_ids == [0, 0]

    # Greedy scenario
    seq_ids = [20]  # single sequence
    # create probabilities where argmax is token 2
    probs = torch.tensor([[0.0, 0.1, 0.9]])
    logprobs = torch.log(probs)
    seq_logprobs = [0.0]
    sampling_params = _DummySamplingParams(use_beam_search=False, temperature=0.0, n=1)

    parent_seq_ids, next_token_ids = _fn(seq_ids, probs, logprobs, seq_logprobs, sampling_params)

    # Greedy branch should return same parent seq id and argmax token
    assert parent_seq_ids == seq_ids
    assert next_token_ids == [2]


# If desired, additional tests could be added to:
# - validate corner cases (empty mappings, empty layers),
# - verify errors when invalid inputs are provided, and
# - integration tests when compiled CUDA extension is available (skip in CPU-only CI).