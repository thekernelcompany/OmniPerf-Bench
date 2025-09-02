import os
import json
import torch
import random
import numpy as np
from typing import Dict, Any, Tuple, List

# Import the actual optimized modules from the commit
# Based on the diff, the optimized API is cacheflow.cache_ops.copy_blocks with signature:
#   copy_blocks(key_caches: List[Tensor], value_caches: List[Tensor], block_mapping: Dict[int, List[int]])
try:
    from cacheflow import cache_ops
except Exception as e:
    print(f"Failed to import cacheflow.cache_ops. Ensure the extension is built. Error: {e}")
    # Do not exit here; run_test will handle gracefully if not available


def _build_block_mapping(num_blocks: int, num_src: int, fanout: int, rng: random.Random) -> Dict[int, List[int]]:
    # Create a mapping: choose 'num_src' unique source blocks and for each,
    # choose 'fanout' unique destination blocks from the remaining pool.
    if num_src <= 0 or fanout <= 0:
        return {}
    num_src = min(num_src, num_blocks // 2)  # ensure we have room for distinct dests
    src_blocks = rng.sample(range(num_blocks), num_src)
    remaining = list(set(range(num_blocks)) - set(src_blocks))
    required_dests = num_src * fanout
    if required_dests > len(remaining):
        # Reduce fanout if necessary
        fanout = max(1, len(remaining) // num_src)
    rng.shuffle(remaining)
    block_mapping: Dict[int, List[int]] = {}
    idx = 0
    for s in src_blocks:
        dests = remaining[idx: idx + fanout]
        idx += fanout
        block_mapping[s] = dests
    return block_mapping


def setup() -> Dict[str, Any]:
    """Create realistic workload that exercises the optimized block copy kernel."""
    torch.manual_seed(42)
    np.random.seed(42)
    rng = random.Random(42)

    if not torch.cuda.is_available():
        raise RuntimeError("This test requires a CUDA-capable GPU to run the block copy kernel.")

    # Workload dimensions inspired by KV cache usage in beam search
    # Keep memory moderate but realistic to exercise the kernel meaningfully.
    num_layers = int(os.environ.get("CF_NUM_LAYERS", "8"))
    num_heads = int(os.environ.get("CF_NUM_HEADS", "16"))
    head_size = int(os.environ.get("CF_HEAD_SIZE", "64"))  # must be multiple of 16
    block_size = int(os.environ.get("CF_BLOCK_SIZE", "16"))
    num_blocks = int(os.environ.get("CF_NUM_BLOCKS", "512"))
    dtype = torch.half  # half precision to match typical KV cache dtype

    if head_size % 16 != 0:
        raise ValueError(f"head_size ({head_size}) must be a multiple of 16.")

    # Compute x as in CacheEngine.get_key_block_shape
    element_size = torch.tensor([], dtype=dtype).element_size()
    x = 16 // element_size  # for half: 8
    key_cache_shape = (num_blocks, num_heads, head_size // x, block_size, x)
    value_cache_shape = (num_blocks, num_heads, head_size, block_size)

    # Allocate KV caches for all layers on the GPU
    key_caches: List[torch.Tensor] = []
    value_caches: List[torch.Tensor] = []
    for _ in range(num_layers):
        key_cache = torch.randn(size=key_cache_shape, dtype=dtype, device="cuda")
        value_cache = torch.randn(size=value_cache_shape, dtype=dtype, device="cuda")
        key_caches.append(key_cache)
        value_caches.append(value_cache)

    # Build a mapping representative of beam expansion: one-to-many copies
    # Use ~1/8 of blocks as sources, with fanout 3 (i.e., copy each source block to 3 destinations)
    num_src = max(1, num_blocks // 8)
    fanout = 3
    block_mapping = _build_block_mapping(num_blocks, num_src, fanout, rng)

    return {
        "key_caches": key_caches,
        "value_caches": value_caches,
        "block_mapping": block_mapping,
        "dtype": dtype,
        "num_layers": num_layers,
        "num_heads": num_heads,
        "head_size": head_size,
        "block_size": block_size,
        "num_blocks": num_blocks,
        # Control whether experiment returns a digest (only for eqcheck/reference)
        "collect_result": False,
    }


def _compute_digest(key_caches: List[torch.Tensor], value_caches: List[torch.Tensor],
                    block_mapping: Dict[int, List[int]], max_blocks_to_check: int = 5) -> torch.Tensor:
    # Create a compact digest tensor that summarizes a few destination blocks across layers.
    # Deterministic selection: first up to max_blocks_to_check unique destination blocks in sorted order.
    dest_blocks: List[int] = []
    for dsts in block_mapping.values():
        for d in dsts:
            dest_blocks.append(d)
    if not dest_blocks:
        # If no mapping, return zeros to keep shape consistent.
        num_layers = len(key_caches)
        return torch.zeros((num_layers, 1, 2), dtype=torch.float32)

    unique_dests = sorted(set(dest_blocks))[:max_blocks_to_check]
    num_layers = len(key_caches)
    digest = torch.empty((num_layers, len(unique_dests), 2), dtype=torch.float32)  # [layers, blocks, (key_sum, value_sum)]

    # Summarize per selected block per layer via sums to keep digest tiny and robust
    for li in range(num_layers):
        for bi, b in enumerate(unique_dests):
            ks = key_caches[li][b].float().sum()
            vs = value_caches[li][b].float().sum()
            digest[li, bi, 0] = ks
            digest[li, bi, 1] = vs
    return digest.cpu()


def experiment(data: Dict[str, Any]) -> Any:
    """
    Execute the performance-critical code path being optimized:
    cacheflow.cache_ops.copy_blocks(key_caches, value_caches, block_mapping)
    """
    key_caches: List[torch.Tensor] = data["key_caches"]
    value_caches: List[torch.Tensor] = data["value_caches"]
    block_mapping: Dict[int, List[int]] = data["block_mapping"]

    # Invoke the optimized block copy kernel. This operates in-place.
    cache_ops.copy_blocks(key_caches, value_caches, block_mapping)

    # Optionally return a small digest for equivalence checking only.
    if data.get("collect_result", False):
        return _compute_digest(key_caches, value_caches, block_mapping)
    return None


def store_result(result: Any, filepath: str) -> None:
    """Store result for future equivalence checking."""
    # Result is a small CPU tensor digest or None.
    if result is None:
        # Store a placeholder and indicate no result
        torch.save({"has_result": False}, filepath)
        return
    if isinstance(result, torch.Tensor):
        torch.save({
            "has_result": True,
            "shape": tuple(result.shape),
            "dtype": str(result.dtype),
            "result": result.cpu(),
        }, filepath)
    else:
        # Fallback: JSON serialize basic types if needed
        try:
            with open(filepath, "w") as f:
                json.dump({"has_result": True, "result": result}, f)
        except Exception as e:
            raise RuntimeError(f"Unsupported result type for storage: {type(result)}; error: {e}")


def load_result(filepath: str) -> Any:
    """Load reference result for equivalence checking."""
    # Try torch.load first
    try:
        data = torch.load(filepath, map_location="cpu")
        if isinstance(data, dict) and data.get("has_result", False):
            return data["result"]
        return None
    except Exception:
        # Try JSON
        try:
            with open(filepath, "r") as f:
                data = json.load(f)
            if isinstance(data, dict) and data.get("has_result", False):
                return data.get("result", None)
            return None
        except Exception as e:
            raise RuntimeError(f"Failed to load reference result from {filepath}: {e}")


def check_equivalence(current_result: Any, reference_result: Any) -> None:
    """Check that current result is equivalent to reference."""
    if reference_result is None and current_result is None:
        return
    if not isinstance(current_result, torch.Tensor) or not isinstance(reference_result, torch.Tensor):
        raise AssertionError(f"Result types mismatch or unsupported: {type(current_result)} vs {type(reference_result)}")

    # Shape and dtype checks
    assert current_result.shape == reference_result.shape, f"Shape mismatch: {current_result.shape} vs {reference_result.shape}"
    # Numerical closeness with tolerances suitable for float32 sums
    torch.testing.assert_close(current_result, reference_result, rtol=1e-4, atol=1e-4)


def run_test(eqcheck: bool = False, reference: bool = False, prefix: str = '') -> float:
    """
    Run the performance test and return average execution time in milliseconds.

    Args:
        eqcheck: Whether to perform equivalence checking against stored reference.
        reference: Whether to store current result as reference.
        prefix: Prefix for reference files.

    Returns:
        Average execution time per iteration in milliseconds.
    """
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this test (block copy kernel runs on GPU).")

    if not hasattr(cache_ops, "copy_blocks"):
        raise RuntimeError("cacheflow.cache_ops.copy_blocks is not available. Ensure the extension is built for this commit.")

    # Setup data once
    data = setup()

    # Use CUDA events for precise GPU timing
    torch.cuda.synchronize()

    # Warmup to stabilize performance
    with torch.no_grad():
        for _ in range(5):
            experiment(data)
    torch.cuda.synchronize()

    # Measure performance using CUDA events
    num_iterations = 50
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    with torch.no_grad():
        start_event.record()
        for _ in range(num_iterations):
            experiment(data)
        end_event.record()

    torch.cuda.synchronize()
    elapsed_time_ms = start_event.elapsed_time(end_event)  # milliseconds total

    # Handle equivalence checking and reference storage
    if eqcheck or reference:
        # Compute digest without impacting measured timing
        data["collect_result"] = True
        result = experiment(data)
        data["collect_result"] = False

        ref_path = f'{prefix}_reference.pt'
        if reference:
            store_result(result, ref_path)
        elif eqcheck:
            reference_result = load_result(ref_path)
            check_equivalence(result, reference_result)

    return elapsed_time_ms / num_iterations

# The harness will be automatically appended here with argument parsing