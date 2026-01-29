import time
import torch


def main():
    # Configuration similar to the provided example
    num_tokens = int(4096)
    num_experts = int(64)
    topk = int(2)
    block_size = int(128)

    device = torch.device("cuda")

    # Create input data
    topk_ids = torch.randint(0, num_experts, (num_tokens, topk),
                             dtype=torch.int32, device=device)

    # Import fused_moe directly by file path to avoid package-level imports
    import importlib.util
    import os
    repo_root = os.path.dirname(os.path.dirname(__file__))
    fm_path = os.path.join(repo_root, 'vllm', 'model_executor', 'layers',
                           'fused_moe', 'fused_moe.py')
    spec = importlib.util.spec_from_file_location("_fused_moe", fm_path)
    fm = importlib.util.module_from_spec(spec)
    assert spec and spec.loader is not None
    spec.loader.exec_module(fm)
    moe_align_block_size_fn = fm.moe_align_block_size
    # Warmup
    torch.cuda.synchronize()
    _ = moe_align_block_size_fn(topk_ids, block_size, num_experts)
    torch.cuda.synchronize()

    # Time the operation
    torch.cuda.synchronize()
    start = time.time()
    sorted_ids, expert_ids, num_tokens_post_pad = moe_align_block_size_fn(
        topk_ids, block_size, num_experts
    )
    torch.cuda.synchronize()
    duration = time.time() - start

    print(f"Duration: {duration:.6f} seconds")
    # Sanity checks to avoid dead code elimination by JIT
    print("Shapes:", sorted_ids.shape, expert_ids.shape, num_tokens_post_pad)


if __name__ == "__main__":
    main()
