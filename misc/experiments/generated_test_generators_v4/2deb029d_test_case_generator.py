import torch
import numpy as np
from typing import Dict, Any, Tuple, List

# Import the actual optimized modules from the commit
try:
    # Core allocator implementing prefix caching and the new touched/computed logic
    from vllm.core.block.prefix_caching_block import PrefixCachingBlockAllocator
except Exception as e:
    print(f"Required vLLM modules not available or import failed: {e}")
    raise

# Helper to create a chain of immutable blocks (replicates the test helper pattern)
def _create_immutable_chain(
    block_size: int,
    token_ids: List[int],
    allocator: PrefixCachingBlockAllocator,
):
    blocks = []
    num_blocks = (len(token_ids) + block_size - 1) // block_size
    prev_block = None
    for i in range(num_blocks):
        block_token_ids = token_ids[i * block_size:(i + 1) * block_size]
        prev_block = allocator.allocate_immutable_block(
            prev_block=prev_block, token_ids=block_token_ids
        )
        blocks.append(prev_block)
    return blocks


def setup() -> Dict[str, Any]:
    """Create realistic workload that exercises the optimization."""
    torch.manual_seed(42)
    np.random.seed(42)

    # Based on commit tests: block_size 16, and reuse the same prefix across a batch.
    block_size = 16
    # Choose a reasonably large number of common blocks to stress the touched->computed path.
    # Keep within allocator capacity while allowing room for management structures.
    common_blocks = 2048  # Length of common prefix (number of blocks)
    num_blocks = common_blocks + 64  # total physical blocks in allocator

    allocator = PrefixCachingBlockAllocator(num_blocks=num_blocks, block_size=block_size)

    # Create token ids that form a single long chain of full blocks
    common_token_ids = list(range(block_size * common_blocks))

    # Mimic allocating the same chain for a batch of 3 sequences (as in new test case).
    # The first allocation will populate _touched_blocks with the new block_ids.
    # Subsequent allocations will reuse the cached blocks and mark them computed per-block.
    last_blocks = None
    for _ in range(3):
        last_blocks = _create_immutable_chain(
            block_size=block_size,
            token_ids=common_token_ids,
            allocator=allocator,
        )

    assert last_blocks is not None
    block_ids = [b.block_id for b in last_blocks]
    # Sanity: ensure no None ids
    assert all(bid is not None for bid in block_ids)

    # At this point:
    # - allocator._touched_blocks contains the unique block_ids introduced by the first sequence
    # - Calling mark_blocks_as_computed([]) will mark them computed and clear touched set
    #
    # For repeated performance iterations, we will repopulate the touched set with the same block ids
    # to isolate the cost of the new path.

    return {
        'allocator': allocator,
        'block_ids': block_ids,
        'block_size': block_size,
        'common_blocks': common_blocks,
        'num_blocks': num_blocks,
    }


def experiment(data: Dict[str, Any]) -> Any:
    """
    Execute the performance-critical code path being optimized.

    This focuses on the new/changed code path:
      - PrefixCachingBlockAllocator.mark_blocks_as_computed
    We repopulate allocator._touched_blocks with a large set of block_ids,
    then call mark_blocks_as_computed([]) to exercise the marking path.
    """
    allocator: PrefixCachingBlockAllocator = data['allocator']
    block_ids: List[int] = data['block_ids']

    # Repopulate touched set to simulate "after schedule" state for a batch.
    # This mirrors the behavior where blocks promoted during scheduling are
    # recorded and later marked as computed en-masse.
    allocator._touched_blocks.update(block_ids)  # type: ignore[attr-defined]

    # Mark all touched blocks as computed (the operation added by this commit).
    allocator.mark_blocks_as_computed([])

    # Optionally, verify that these blocks are now considered computed by querying once.
    # This also returns a stable result for equivalence checking.
    computed_block_ids = allocator.get_computed_block_ids(
        prev_computed_block_ids=[],
        block_ids=block_ids,
        skip_last_block_id=False,
    )
    return torch.tensor(computed_block_ids, dtype=torch.int64)


def store_result(result: Any, filepath: str) -> None:
    """Store result for future equivalence checking."""
    if isinstance(result, torch.Tensor):
        torch.save({
            'type': 'tensor',
            'shape': tuple(result.shape),
            'dtype': str(result.dtype),
            'device': str(result.device),
            'result': result.cpu(),
        }, filepath)
    else:
        torch.save({'type': 'object', 'result': result}, filepath)


def load_result(filepath: str) -> Any:
    """Load reference result for equivalence checking."""
    data = torch.load(filepath, map_location='cpu')
    if data.get('type') == 'tensor':
        return data['result']
    return data['result']


def check_equivalence(current_result: Any, reference_result: Any) -> None:
    """Check that current result is equivalent to reference."""
    if isinstance(current_result, torch.Tensor) and isinstance(reference_result, torch.Tensor):
        assert current_result.shape == reference_result.shape, \
            f"Shape mismatch: {current_result.shape} vs {reference_result.shape}"
        assert current_result.dtype == reference_result.dtype, \
            f"Dtype mismatch: {current_result.dtype} vs {reference_result.dtype}"
        # Values must match exactly (these are integer block IDs).
        torch.testing.assert_close(current_result, reference_result, rtol=0, atol=0)
    else:
        assert current_result == reference_result, "Non-tensor results differ"


def run_test(eqcheck: bool = False, reference: bool = False, prefix: str = '') -> float:
    """
    Run the performance test and return average execution time in milliseconds.

    Args:
        eqcheck: Whether to perform equivalence checking.
        reference: Whether to store result as reference.
        prefix: Prefix for reference files.

    Returns:
        Average execution time in milliseconds.
    """
    data = setup()

    # CPU timing (allocator logic runs on CPU)
    import time
    num_iterations = 200  # iterations to achieve stable timing

    # Warmup
    for _ in range(5):
        _ = experiment(data)

    start_time = time.time()
    for _ in range(num_iterations):
        result = experiment(data)
    end_time = time.time()
    elapsed_time_ms = (end_time - start_time) * 1000.0

    # Equivalence handling on a fresh run
    if eqcheck or reference:
        result = experiment(data)
        if reference:
            store_result(result, f'{prefix}_reference.pt')
        elif eqcheck:
            reference_result = load_result(f'{prefix}_reference.pt')
            check_equivalence(result, reference_result)

    return elapsed_time_ms / num_iterations

# The harness will be automatically appended here with argument parsing