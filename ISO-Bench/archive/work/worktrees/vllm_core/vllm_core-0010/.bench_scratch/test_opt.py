import time
import torch
import importlib.util
import os

# Load fused_moe module directly to avoid heavy vllm package imports
repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
module_path = os.path.join(repo_root, 'vllm', 'model_executor', 'layers', 'fused_moe', 'fused_moe.py')
spec = importlib.util.spec_from_file_location('fused_moe', module_path)
fused_moe = importlib.util.module_from_spec(spec)
spec.loader.exec_module(fused_moe)  # type: ignore
moe_align_block_size = fused_moe.moe_align_block_size

# Benchmark parameters
num_tokens = 4096
num_experts = 64
topk = 2
block_size = 128

# Prepare random top-k expert ids for each token (shape: [num_tokens, topk])
if not torch.cuda.is_available():
    print("CUDA is not available; skipping benchmark.")
    raise SystemExit(0)

device = 'cuda'

topk_ids = torch.randint(0, num_experts, (num_tokens, topk), dtype=torch.int32, device=device)

# Warm-up
for _ in range(5):
    sorted_ids, expert_ids, num_tokens_post_pad = moe_align_block_size(topk_ids, block_size, num_experts)

torch.cuda.synchronize()

# Timed runs
iters = 50
timings = []
for _ in range(iters):
    torch.cuda.synchronize()
    start = time.time()
    sorted_ids, expert_ids, num_tokens_post_pad = moe_align_block_size(topk_ids, block_size, num_experts)
    torch.cuda.synchronize()
    timings.append(time.time() - start)

print(f'Runs: {iters}')
print(f'Best: {min(timings):.6f}s, Median: {sorted(timings)[len(timings)//2]:.6f}s, Mean: {sum(timings)/len(timings):.6f}s')
