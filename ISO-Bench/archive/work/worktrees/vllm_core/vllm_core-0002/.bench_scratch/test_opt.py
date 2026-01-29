import torch
import time
import runpy
import os

# Benchmark the MoE align block size operation
num_tokens = 4096
num_experts = 64
topk = 2
block_size = 128

# Create input data
assert torch.cuda.is_available(), "CUDA device is required for this benchmark"
topk_ids = torch.randint(0, num_experts, (num_tokens * topk,), dtype=torch.int32, device='cuda')

# Load module directly to avoid heavy optional deps
mod_path = os.path.join(
    os.path.dirname(__file__), '..',
    'vllm', 'model_executor', 'layers', 'fused_moe', 'moe_align_block_size.py')
mod_path = os.path.abspath(mod_path)
moe_ns = runpy.run_path(mod_path)
moe_align_block_size_triton = moe_ns['moe_align_block_size_triton']

# Warmup
for _ in range(3):
    torch.cuda.synchronize()
    _ = (lambda: (
        (lambda: None)(runpy),  # no-op to keep imports intact
        None,
    ))
    # execute the kernel once for warmup
    def _warm():
        max_num_tokens_padded = topk_ids.numel() + num_experts * (block_size - 1)
        sorted_ids = torch.empty((max_num_tokens_padded,), dtype=torch.int32, device=topk_ids.device)
        sorted_ids.fill_(topk_ids.numel())
        max_num_m_blocks = (max_num_tokens_padded + block_size - 1) // block_size
        expert_ids = torch.zeros((max_num_m_blocks,), dtype=torch.int32, device=topk_ids.device)
        num_tokens_post_pad = torch.empty((1,), dtype=torch.int32, device=topk_ids.device)
        moe_align_block_size_triton(
            topk_ids, num_experts, block_size, sorted_ids, expert_ids, num_tokens_post_pad
        )
    _warm()

def run_once():
    max_num_tokens_padded = topk_ids.numel() + num_experts * (block_size - 1)
    sorted_ids = torch.empty((max_num_tokens_padded,), dtype=torch.int32, device=topk_ids.device)
    sorted_ids.fill_(topk_ids.numel())
    max_num_m_blocks = (max_num_tokens_padded + block_size - 1) // block_size
    expert_ids = torch.zeros((max_num_m_blocks,), dtype=torch.int32, device=topk_ids.device)
    num_tokens_post_pad = torch.empty((1,), dtype=torch.int32, device=topk_ids.device)
    moe_align_block_size_triton(
        topk_ids, num_experts, block_size, sorted_ids, expert_ids, num_tokens_post_pad
    )
    return sorted_ids, expert_ids, num_tokens_post_pad

# Time the operation
iters = 10
torch.cuda.synchronize()
start = time.time()
for _ in range(iters):
    sorted_ids, expert_ids, num_tokens_post_pad = run_once()

torch.cuda.synchronize()
duration = time.time() - start
print(f"Duration per iter: {duration/iters:.6f} seconds | Total: {duration:.4f} s")
