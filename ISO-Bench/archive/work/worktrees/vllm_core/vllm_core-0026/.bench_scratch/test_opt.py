#!/usr/bin/env python3
import os
# Force KV cache layout early to avoid config probing
os.environ.setdefault('VLLM_KV_CACHE_LAYOUT', 'HND')
import sys
import time
import types

# Ensure repo root is on sys.path
REPO_ROOT = os.path.abspath(os.path.dirname(__file__) + '/..')
sys.path.insert(0, REPO_ROOT)

# Mock flashinfer to avoid requiring the actual package
class _DummyWrapper:
    def __init__(self, *args, **kwargs):
        pass
    def plan(self, *args, **kwargs):
        pass
    def run(self, *args, **kwargs):
        pass

flashinfer_mod = types.ModuleType('flashinfer')
flashinfer_mod.BatchDecodeWithPagedKVCacheWrapper = _DummyWrapper
flashinfer_mod.BatchPrefillWithPagedKVCacheWrapper = _DummyWrapper
flashinfer_mod.MultiLevelCascadeAttentionWrapper = _DummyWrapper
sys.modules['flashinfer'] = flashinfer_mod

flashinfer_decode_mod = types.ModuleType('flashinfer.decode')
flashinfer_decode_mod.trtllm_batch_decode_with_kv_cache = lambda *a, **k: None
sys.modules['flashinfer.decode'] = flashinfer_decode_mod

import torch

# Import helpers to create CommonAttentionMetadata only (safe)
from tests.v1.attention.utils import (
    BatchSpec,
    create_common_attn_metadata,
)

from vllm.v1.kv_cache_interface import FullAttentionSpec
from vllm.v1.attention.backends.flashinfer import FlashInferBackend
# Avoid config/device probing by forcing a KV cache layout
from vllm.v1.attention.backends.utils import set_kv_cache_layout as _set_kv_layout
_set_kv_layout('HND')
# Patch get_per_layer_parameters to avoid dependency on model layers
import vllm.v1.attention.backends.flashinfer as _fi_mod
from vllm.v1.attention.backends.flashinfer import PerLayerParameters as _PLP

def _mock_get_per_layer_parameters(_vllm_config, _impl_cls):
    # Use standard scale for head_size=128
    return {"mock_layer": _PLP(window_left=-1, logits_soft_cap=0.0, sm_scale=1.0 / (128 ** 0.5))}

_fi_mod.get_per_layer_parameters = _mock_get_per_layer_parameters


# Minimal stand-in configs to avoid heavy vllm.model config machinery
class _DummyModelConfig:
    def __init__(self, dtype: torch.dtype, num_q_heads: int, num_kv_heads: int, head_size: int):
        self.dtype = dtype
        self._num_q_heads = num_q_heads
        self._num_kv_heads = num_kv_heads
        self._head_size = head_size
    def get_num_attention_heads(self, _):
        return self._num_q_heads
    def get_num_kv_heads(self, _):
        return self._num_kv_heads
    def get_head_size(self):
        return self._head_size

class _DummyCacheConfig:
    def __init__(self, block_size: int, cache_dtype: str = 'auto'):
        self.block_size = block_size
        self.cache_dtype = cache_dtype

class _DummyParallelConfig:
    pass

class _DummyVllmConfig:
    def __init__(self, model_config, cache_config, parallel_config):
        self.model_config = model_config
        self.cache_config = cache_config
        self.parallel_config = parallel_config


def bench_builder_runs(device=torch.device('cpu'), loops=2000):
    # Small realistic batch spec
    batch_spec = BatchSpec(seq_lens=[128, 192, 64, 96], query_lens=[8, 8, 4, 4])

    # Dummy, light-weight config avoiding any model inspection
    model_config = _DummyModelConfig(dtype=torch.float16,
                                     num_q_heads=8,
                                     num_kv_heads=8,
                                     head_size=128)
    cache_config = _DummyCacheConfig(block_size=16, cache_dtype='auto')
    parallel_config = _DummyParallelConfig()
    vllm_config = _DummyVllmConfig(model_config, cache_config, parallel_config)

    kv_cache_spec = FullAttentionSpec(block_size=cache_config.block_size,
                                      num_kv_heads=model_config._num_kv_heads,
                                      head_size=model_config._head_size,
                                      dtype=model_config.dtype,
                                      use_mla=False,
                                      sliding_window=None)

    common_attn_metadata = create_common_attn_metadata(
        batch_spec, cache_config.block_size, device
    )

    builder_cls = FlashInferBackend.get_builder_cls()
    builder = builder_cls(kv_cache_spec, vllm_config, device)

    # Warmup
    for _ in range(50):
        builder.build(common_prefix_len=0, common_attn_metadata=common_attn_metadata)

    t0 = time.perf_counter()
    for _ in range(loops):
        builder.build(common_prefix_len=0, common_attn_metadata=common_attn_metadata)
    t1 = time.perf_counter()

    # Also test cascade path
    block_size = cache_config.block_size
    assert block_size > 0
    for _ in range(10):
        builder.build(common_prefix_len=block_size, common_attn_metadata=common_attn_metadata)

    t2 = time.perf_counter()
    for _ in range(loops):
        builder.build(common_prefix_len=block_size, common_attn_metadata=common_attn_metadata)
    t3 = time.perf_counter()

    print(f'FlashInferMetadataBuilder.build decode-only: {t1 - t0:.6f}s for {loops} loops')
    print(f'FlashInferMetadataBuilder.build cascade:    {t3 - t2:.6f}s for {loops} loops')


def main():
    device = torch.device('cpu')
    bench_builder_runs(device=device, loops=1200)

if __name__ == '__main__':
    main()
