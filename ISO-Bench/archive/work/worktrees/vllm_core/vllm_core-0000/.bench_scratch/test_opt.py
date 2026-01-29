import time
from vllm.core.block.prefix_caching_block import PrefixCachingBlockAllocator

# Benchmark prefix caching block allocation with common prefixes
block_size = 16
num_blocks = 256
num_sequences = 8
common_prefix_blocks = 4

# Create allocator
allocator = PrefixCachingBlockAllocator(num_blocks=num_blocks, block_size=block_size)

# Common token IDs for shared prefix
common_token_ids = list(range(block_size * common_prefix_blocks))

# Warm-up
for _ in range(2):
    prev_block = None
    for block_idx in range(common_prefix_blocks):
        start_idx = block_idx * block_size
        end_idx = start_idx + block_size
        token_ids = common_token_ids[start_idx:end_idx]
        block = allocator.allocate_immutable_block(prev_block=prev_block, token_ids=token_ids)
        prev_block = block
    allocator.mark_blocks_as_computed([])

# Time the allocation and marking operation
start = time.time()

# Allocate blocks for multiple sequences with common prefixes
for seq_idx in range(num_sequences):
    prev_block = None
    for block_idx in range(common_prefix_blocks):
        start_idx = block_idx * block_size
        end_idx = start_idx + block_size
        token_ids = common_token_ids[start_idx:end_idx]

        block = allocator.allocate_immutable_block(
            prev_block=prev_block,
            token_ids=token_ids,
        )
        prev_block = block

# Mark blocks as computed (optimized operation)
allocator.mark_blocks_as_computed([])

duration = time.time() - start
print(f"Duration: {duration:.6f} seconds")
print(f"Cache hit rate: {allocator.get_prefix_cache_hit_rate():.3f}")
