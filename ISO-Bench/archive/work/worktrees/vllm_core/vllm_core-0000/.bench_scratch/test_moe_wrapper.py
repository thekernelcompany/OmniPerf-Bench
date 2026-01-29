import torch
import time
import vllm._custom_ops as ops


def local_moe_align_block_size(topk_ids: torch.Tensor, block_size: int, num_experts: int):
    max_num_tokens_padded = topk_ids.numel() + num_experts * (block_size - 1)
    sorted_ids = torch.empty((max_num_tokens_padded,), dtype=torch.int32, device=topk_ids.device)
    max_num_m_blocks = (max_num_tokens_padded + block_size - 1) // block_size
    expert_ids = torch.empty((max_num_m_blocks,), dtype=torch.int32, device=topk_ids.device)
    num_tokens_post_pad = torch.empty((1,), dtype=torch.int32, device=topk_ids.device)
    ops.moe_align_block_size(topk_ids, num_experts, block_size, sorted_ids, expert_ids, num_tokens_post_pad)
    return sorted_ids, expert_ids, num_tokens_post_pad


# Benchmark
num_tokens = 4096
num_experts = 64
topk = 2
block_size = 128

# Create input data (flat)
topk_ids = torch.randint(0, num_experts, (num_tokens * topk,), dtype=torch.int32, device="cuda")

# Warmup
for _ in range(5):
    torch.cuda.synchronize()
    local_moe_align_block_size(topk_ids, block_size, num_experts)
    torch.cuda.synchronize()

# Time
torch.cuda.synchronize()
start = time.time()

sorted_ids, expert_ids, num_tokens_post_pad = local_moe_align_block_size(
    topk_ids, block_size, num_experts
)

torch.cuda.synchronize()
duration = time.time() - start

print(f"Duration: {duration:.6f} seconds")
print("sorted_ids:", sorted_ids.shape, sorted_ids.dtype)
print("expert_ids:", expert_ids.shape, expert_ids.dtype)
print("num_tokens_post_pad:", num_tokens_post_pad.item())
