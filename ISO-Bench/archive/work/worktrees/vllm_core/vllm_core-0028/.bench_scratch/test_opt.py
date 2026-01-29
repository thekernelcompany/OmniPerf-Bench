import time
import torch

# Patch Transformers to allow duplicate registrations to avoid import-time failures in this environment
try:
    from transformers.models.auto.configuration_auto import CONFIG_MAPPING
    _orig_register = CONFIG_MAPPING.register
    def _safe_register(key, value, exist_ok: bool = False):
        try:
            return _orig_register(key, value, exist_ok=True)
        except Exception:
            # Ignore duplicates or other registration issues
            return None
    CONFIG_MAPPING.register = _safe_register  # type: ignore[attr-defined]
except Exception:
    pass
# Stub out missing optional dependencies to avoid heavy imports during benchmarking
import sys, types
if 'gguf' not in sys.modules:
    sys.modules['gguf'] = types.ModuleType('gguf')


# Benchmark for Qwen2_5_VisionRotaryEmbedding in vllm.model_executor.models.qwen2_5_vl
from vllm.model_executor.models.qwen2_5_vl import (
    Qwen2_5_VisionRotaryEmbedding,
    Qwen2_5_VisionTransformer,
)

def bench_rotary_embed(dim: int = 64, seqlens=(512, 1024, 2048, 4096)):
    emb = Qwen2_5_VisionRotaryEmbedding(dim)

    times = []
    for s in seqlens:
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t0 = time.time()
        out = emb(int(s))
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t1 = time.time()
        _ = out[:1]
        times.append((s, t1 - t0))
    return times


def make_min_vision_config():
    class Cfg:
        patch_size = 14
        temporal_patch_size = 1
        in_channels = 3
        depth = 1
        hidden_size = 64
        num_heads = 4
        out_hidden_size = 64
        spatial_merge_size = 2
        window_size = 4
        fullatt_block_indexes = []
        intermediate_size = 128
        hidden_act = "silu"
    return Cfg()


def bench_rotpos_and_window():
    cfg = make_min_vision_config()
    vt = Qwen2_5_VisionTransformer(cfg)

    grid_thw = torch.tensor([
        [1, 8, 8],
        [1, 8, 8],
        [1, 8, 8],
        [1, 8, 8],
    ], dtype=torch.int32)

    # Warmup
    _ = vt.rot_pos_emb(grid_thw)
    _ = vt.get_window_index(grid_thw)

    # Measure repeat usage (cache should help)
    t0 = time.time()
    for _ in range(50):
        _ = vt.rot_pos_emb(grid_thw)
    t1 = time.time()

    t2 = time.time()
    for _ in range(50):
        _ = vt.get_window_index(grid_thw)
    t3 = time.time()

    print(f"rot_pos_emb_repeat_50_ms={(t1-t0)*1000:.3f}")
    print(f"get_window_index_repeat_50_ms={(t3-t2)*1000:.3f}")


def main():
    times = bench_rotary_embed()
    for s, dt in times:
        print(f"seqlen={s}: {dt:.6f}s")
    try:
        bench_rotpos_and_window()
    except Exception as e:
        print(f"bench_rotpos_and_window_skipped: {type(e).__name__}")

if __name__ == "__main__":
    main()
