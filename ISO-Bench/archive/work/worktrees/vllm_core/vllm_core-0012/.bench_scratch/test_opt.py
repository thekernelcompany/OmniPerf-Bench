import time
import torch  # ensure torch is available as some modules import it
from vllm.core.block_manager_v1 import BlockSpaceManagerV1
from vllm.sequence import Sequence, SequenceGroup
from vllm.sampling_params import SamplingParams

# Parameters similar to the provided example
block_size = 16
num_blocks = 256
num_sequences = 8
common_prefix_blocks = 4

# Shared prefix token IDs
common_token_ids = list(range(block_size * common_prefix_blocks))

# Create sequences with a common prefix
seqs = []
for i in range(num_sequences):
    seq = Sequence(
        seq_id=i,
        prompt=f"prompt-{i}",
        prompt_token_ids=common_token_ids.copy(),
        block_size=block_size,
    )
    seqs.append(seq)

# Create a sequence group and block space manager with caching enabled
sampling_params = SamplingParams()
seq_group = SequenceGroup(
    request_id="req-1",
    seqs=seqs,
    sampling_params=sampling_params,
    arrival_time=time.time(),
)

bsm = BlockSpaceManagerV1(
    block_size=block_size,
    num_gpu_blocks=num_blocks,
    num_cpu_blocks=num_blocks,
    enable_caching=True,
)

start = time.time()

# Allocate blocks for the group (prompt stage)
if bsm.can_allocate(seq_group).name in ("OK", "LATER"):
    bsm.allocate(seq_group)

# Mark blocks as computed (optimized code path)
bsm.mark_blocks_as_computed(seq_group)

end = time.time()

print(f"Duration: {end - start:.6f} seconds")
# Print number of common computed block ids as a simple correctness check
common_ids = bsm.get_common_computed_block_ids(seqs)
print(f"Common computed blocks: {len(common_ids)}")
