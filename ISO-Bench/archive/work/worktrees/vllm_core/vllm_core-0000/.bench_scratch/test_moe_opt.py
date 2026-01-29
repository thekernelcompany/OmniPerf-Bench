import torch
import time
from vllm.model_executor.layers.fused_moe.moe_align_block_size import moe_align_block_size

# Benchmark the MoE align block size operation
num_tokens = 4096
num_experts = 64
topk = 2
block_size = 128

# Create input data
# Use flat layout; kernel only relies on numel()
topk_ids = torch.randint(0, num_experts, (num_tokens * topk,), dtype=torch.int32, device="cuda")

# Warmup
for _ in range(5):
    torch.cuda.synchronize()
    moe_align_block_size(topk_ids, block_size, num_experts)
    torch.cuda.synchronize()

# Time the operation
torch.cuda.synchronize()
start = time.time()

sorted_ids, expert_ids, num_tokens_post_pad = moe_align_block_size(
    topk_ids, block_size, num_experts
)

torch.cuda.synchronize()
duration = time.time() - start

print(f"Duration: {duration:.6f} seconds")
print("sorted_ids:", sorted_ids.shape, sorted_ids.dtype)
print("expert_ids:", expert_ids.shape, expert_ids.dtype)
print("num_tokens_post_pad:", int(num_tokens_post_pad) if num_tokens_post_pad.numel() == 0 else num_tokens_post_pad.item())
