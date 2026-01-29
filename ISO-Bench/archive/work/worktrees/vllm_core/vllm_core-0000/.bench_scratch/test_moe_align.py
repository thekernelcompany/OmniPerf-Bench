import time
import torch
from vllm.model_executor.layers.fused_moe.moe_align_block_size import moe_align_block_size

# Parameters
num_tokens = 4096
num_experts = 64
topk = 2
block_size = 128

# Input data: shape [num_tokens, topk]
topk_ids = torch.randint(0, num_experts, (num_tokens, topk), dtype=torch.int32, device='cuda')

# Warmup
for _ in range(5):
    torch.cuda.synchronize()
    moe_align_block_size(topk_ids, block_size, num_experts)
    torch.cuda.synchronize()

# Benchmark
iters = 20
start = time.time()
for _ in range(iters):
    torch.cuda.synchronize()
    sorted_ids, expert_ids, num_tokens_post_pad = moe_align_block_size(topk_ids, block_size, num_experts)
    torch.cuda.synchronize()
end = time.time()

duration = (end - start) / iters
print(f"Avg Duration: {duration:.6f} s over {iters} iters")
print(f"Outputs: sorted_ids={sorted_ids.shape}, expert_ids={expert_ids.shape}, num_tokens_post_pad={int(num_tokens_post_pad)}")
