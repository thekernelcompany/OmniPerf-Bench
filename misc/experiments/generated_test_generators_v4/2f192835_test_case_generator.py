import json
import time
from typing import Dict, Any, Tuple

import torch
import numpy as np

# Import the actual optimized modules from the commit
# Focus: vllm.core.block_manager_v1.BlockSpaceManagerV1._is_last_block_full
try:
    from vllm.core.block_manager_v1 import BlockSpaceManagerV1 as _RealBlockSpaceManagerV1
    USING_STUB = False
except Exception as e:
    print("Warning: Could not import vllm.core.block_manager_v1.BlockSpaceManagerV1.")
    print("Reason:", repr(e))
    print("Proceeding with a minimal stub to exercise the optimized code path.")
    USING_STUB = True

    class _RealBlockSpaceManagerV1:
        # Minimal stub that matches the optimized behavior of the commit
        def __init__(self, block_size: int, num_gpu_blocks: int, num_cpu_blocks: int,
                     watermark: float = 0.01, sliding_window: int = None,
                     enable_caching: bool = False) -> None:
            self.block_size = block_size

        def _is_last_block_full(self, seq) -> bool:
            # Commit-optimized path: use get_len instead of get_token_ids
            token_ids_len = seq.data.get_len()
            return token_ids_len > 0 and token_ids_len % seq.block_size == 0


def setup() -> Dict[str, Any]:
    """Create realistic workload that exercises the optimization"""
    torch.manual_seed(42)
    np.random.seed(42)

    # We simulate a Sequence-like object where:
    # - data.get_len() is O(1)
    # - data.get_token_ids() is intentionally expensive to construct
    # This mirrors the latency optimization: new code calls get_len() (fast),
    # old code called get_token_ids() (potentially slow).
    class FakeData:
        def __init__(self, n_tokens: int):
            self._len = n_tokens

        def get_len(self) -> int:
            return self._len

        def get_token_ids(self):
            # Intentionally expensive to mimic pre-commit behavior cost
            # (We won't call it in the optimized path.)
            return list(range(self._len))

    class FakeSequence:
        def __init__(self, block_size: int, n_tokens: int):
            self.block_size = block_size
            self.data = FakeData(n_tokens)

    # Choose a block size and token length to frequently hit the boundary condition.
    # token_len is a multiple of block_size so _is_last_block_full returns True.
    block_size = 64
    token_len = 4_096 * block_size  # large multiple to make get_token_ids() prohibitively slow if used
    seq = FakeSequence(block_size=block_size, n_tokens=token_len)

    # Instantiate the real BlockSpaceManagerV1 if available; otherwise, use the stub.
    manager = _RealBlockSpaceManagerV1(block_size=block_size,
                                       num_gpu_blocks=1_000,
                                       num_cpu_blocks=1_000,
                                       watermark=0.01,
                                       sliding_window=None,
                                       enable_caching=True)

    # Inner loops within experiment to amplify the hot check cost while keeping total runtime reasonable.
    inner_loops = 100_000

    return {
        'manager': manager,
        'seq': seq,
        'inner_loops': inner_loops,
        'using_stub': USING_STUB,
    }


def experiment(data: Dict[str, Any]) -> Any:
    """
    Execute the performance-critical code path being optimized.

    Returns:
        Integer count of times the last block is considered full.
    """
    manager = data['manager']
    seq = data['seq']
    inner_loops = data['inner_loops']

    # Tight loop over the optimized check (_is_last_block_full).
    # This directly targets the commit's change.
    count_full = 0
    for _ in range(inner_loops):
        if manager._is_last_block_full(seq):
            count_full += 1
    return count_full


def store_result(result: Any, filepath: str) -> None:
    """Store result for future equivalence checking"""
    payload = {
        'result': int(result),
        'meta': {
            'type': 'int',
        }
    }
    with open(filepath, 'w') as f:
        json.dump(payload, f)


def load_result(filepath: str) -> Any:
    """Load reference result for equivalence checking"""
    with open(filepath, 'r') as f:
        payload = json.load(f)
    return payload['result']


def check_equivalence(current_result: Any, reference_result: Any) -> None:
    """Check that current result is equivalent to reference"""
    assert isinstance(current_result, int) and isinstance(reference_result, int), \
        f"Type mismatch: {type(current_result)} vs {type(reference_result)}"
    assert current_result == reference_result, \
        f"Value mismatch: {current_result} vs {reference_result}"


def run_test(eqcheck: bool = False, reference: bool = False, prefix: str = '') -> float:
    """
    Run the performance test and return execution time.

    Args:
        eqcheck: Whether to perform equivalence checking
        reference: Whether to store result as reference
        prefix: Prefix for reference files

    Returns:
        Average execution time in milliseconds
    """
    data = setup()

    # CPU timing (this is a CPU-side latency optimization)
    num_iterations = 10

    # Warmup
    for _ in range(3):
        _ = experiment(data)

    start_time = time.time()
    for _ in range(num_iterations):
        result = experiment(data)
    end_time = time.time()
    elapsed_time = (end_time - start_time) * 1000.0  # milliseconds

    # Handle equivalence checking
    if eqcheck or reference:
        result = experiment(data)  # Fresh result
        ref_path = f'{prefix}_reference.json'
        if reference:
            store_result(result, ref_path)
        elif eqcheck:
            reference_result = load_result(ref_path)
            check_equivalence(result, reference_result)

    return elapsed_time / num_iterations

# The harness will be automatically appended here with argument parsing