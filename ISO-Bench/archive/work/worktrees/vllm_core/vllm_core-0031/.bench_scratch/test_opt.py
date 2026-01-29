import time
import torch
import importlib.util
import os

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MOE_ALIGN_PATH = os.path.join(REPO_ROOT, 'vllm', 'model_executor', 'layers', 'fused_moe', 'moe_align_block_size.py')

spec = importlib.util.spec_from_file_location('moe_align_block_size_mod', MOE_ALIGN_PATH)
moe_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(moe_module)  # type: ignore
moe_align_block_size = moe_module.moe_align_block_size


def run_once(num_tokens=4096, num_experts=64, topk=2, block_size=128):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    topk_ids = torch.randint(0, num_experts, (num_tokens * topk,), dtype=torch.int32, device=device)

    # Warmup
    if device == 'cuda':
        torch.cuda.synchronize()
    for _ in range(3):
        sorted_ids, expert_ids, num_tokens_post_pad = moe_align_block_size(
            topk_ids, block_size, num_experts
        )
        if device == 'cuda':
            torch.cuda.synchronize()

    # Benchmark
    if device == 'cuda':
        torch.cuda.synchronize()
    start = time.time()
    sorted_ids, expert_ids, num_tokens_post_pad = moe_align_block_size(
        topk_ids, block_size, num_experts
    )
    if device == 'cuda':
        torch.cuda.synchronize()
    duration = time.time() - start
    return duration, (sorted_ids, expert_ids, num_tokens_post_pad)


if __name__ == '__main__':
    dur, _ = run_once()
    print(f"Duration: {dur:.6f} s")
