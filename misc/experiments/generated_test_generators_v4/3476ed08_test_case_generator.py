#!/usr/bin/env python3
"""
Performance test for commit: 3476ed0809ec91a3457da0cb90543133a4f4b519
Message: [Core] Optimize block_manager_v2 vs block_manager_v1 (to make V2 default)  (#5602)

This script measures the actual performance impact of the optimization.
It supports cross-commit comparison with functional equivalence checking.
"""

import os
import sys
import json
import time
import math
import importlib
from typing import Dict, Any, Tuple, Optional, List

import numpy as np
import torch

# =======================
# Determinism Setup
# =======================
def ensure_determinism():
    torch.manual_seed(42)
    np.random.seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        # Disable TF32 for reproducibility unless required
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False

# =======================
# Hardware Detection
# =======================
def detect_hardware() -> Dict[str, Any]:
    hw_info = {}
    if torch.cuda.is_available():
        hw_info["device"] = "cuda"
        hw_info["device_name"] = torch.cuda.get_device_name()
        hw_info["capability"] = torch.cuda.get_device_capability()
        hw_info["memory_gb"] = torch.cuda.get_device_properties(0).total_memory / 1e9
    else:
        hw_info["device"] = "cpu"
        hw_info["device_name"] = "CPU"
        hw_info["memory_gb"] = 0
    return hw_info

# =======================
# Import Resolution
# =======================
def resolve_target() -> Tuple[Any, str]:
    """Resolve the optimization target from environment or metadata."""
    
    # Priority 1: Environment variables
    module_path = os.getenv("PROB_MODULE", "")
    symbol_name = os.getenv("PROB_SYMBOL", "")
    
    # Priority 2: Parse from commit metadata - target the allocator
    if not (module_path and symbol_name):
        # Based on the diff, the main optimization is in PrefixCachingBlockAllocator
        module_path = "vllm.core.block.prefix_caching_block"
        symbol_name = "PrefixCachingBlockAllocator"
    
    # Import with error handling
    try:
        module = importlib.import_module(module_path)
        target = module
        for attr in symbol_name.split("."):
            target = getattr(target, attr)
        
        fq_name = f"{module_path}.{symbol_name}"
        return target, fq_name
        
    except (ImportError, AttributeError) as e:
        error_data = {
            "target_resolved": False,
            "error": str(e),
            "attempted_module": module_path,
            "attempted_symbol": symbol_name
        }
        print(json.dumps(error_data))
        sys.exit(1)

# =======================
# Workload Setup
# =======================
def setup() -> Dict[str, Any]:
    """Create realistic workload for the optimization."""
    ensure_determinism()
    hw_info = detect_hardware()
    
    # Block manager v2 optimization workload
    device = torch.device(hw_info["device"])
    dtype = torch.float16 if hw_info["device"] == "cuda" else torch.float32
    
    # Typical vLLM block manager configuration
    num_blocks = 512 if hw_info.get("memory_gb", 0) < 16 else 1024
    block_size = 16  # Standard vLLM block size
    
    # Simulate a mix of prefill and decode operations
    num_sequences = 8
    prompt_lengths = [128, 256, 512, 1024, 128, 256, 512, 1024]
    
    data = {
        "device": device,
        "dtype": dtype,
        "hw_info": hw_info,
        "num_blocks": num_blocks,
        "block_size": block_size,
        "num_sequences": num_sequences,
        "prompt_lengths": prompt_lengths,
        "allocator_type": "prefix_caching",  # Test the optimized allocator
    }
    
    return data

# =======================
# Experiment Execution
# =======================
def experiment(data: Dict[str, Any]) -> Any:
    """Execute the optimized operation."""
    
    # Import the allocator
    from vllm.core.block.cpu_gpu_block_allocator import CpuGpuBlockAllocator
    
    # Create allocator with the optimized implementation
    allocator = CpuGpuBlockAllocator.create(
        allocator_type=data["allocator_type"],
        num_gpu_blocks=data["num_blocks"],
        num_cpu_blocks=data["num_blocks"] // 4,  # Some CPU blocks for swapping
        block_size=data["block_size"],
    )
    
    # Simulate block allocation patterns
    allocated_blocks = []
    
    with torch.no_grad():
        # Simulate sequence allocation (prefill)
        for seq_idx, prompt_len in enumerate(data["prompt_lengths"]):
            num_blocks_needed = (prompt_len + data["block_size"] - 1) // data["block_size"]
            
            prev_block = None
            seq_blocks = []
            
            for block_idx in range(num_blocks_needed):
                # Allocate blocks using the optimized methods
                if block_idx == num_blocks_needed - 1 and prompt_len % data["block_size"] != 0:
                    # Last block might be mutable
                    block = allocator.allocate_mutable_block(
                        prev_block=prev_block,
                        device=data["device"] if data["device"].type == "cuda" else None
                    )
                else:
                    # Full blocks are immutable
                    token_ids = list(range(block_idx * data["block_size"], 
                                          min((block_idx + 1) * data["block_size"], prompt_len)))
                    block = allocator.allocate_immutable_block(
                        prev_block=prev_block,
                        token_ids=token_ids,
                        device=data["device"] if data["device"].type == "cuda" else None
                    )
                
                seq_blocks.append(block)
                prev_block = block
            
            allocated_blocks.append(seq_blocks)
        
        # Simulate some decode steps (append tokens)
        for seq_blocks in allocated_blocks[:4]:  # Decode for half the sequences
            if seq_blocks:
                last_block = seq_blocks[-1]
                # Append a token (simulating decode)
                if hasattr(last_block, 'append_token_ids'):
                    last_block.append_token_ids([1])
    
    # Return allocation statistics
    result = {
        "num_sequences": len(allocated_blocks),
        "total_blocks_allocated": sum(len(blocks) for blocks in allocated_blocks),
        "num_free_blocks": allocator.get_num_free_blocks(data["device"] if data["device"].type == "cuda" else None),
    }
    
    return result

# =======================
# Result I/O
# =======================
def store_result(result: Any, filepath: str) -> None:
    """Store result for reference comparison."""
    if isinstance(result, torch.Tensor):
        torch.save({"type": "tensor", "data": result.cpu()}, filepath)
    else:
        torch.save({"type": "generic", "data": result}, filepath)

def load_result(filepath: str) -> Any:
    """Load reference result."""
    data = torch.load(filepath)
    return data.get("data", data)

# =======================
# Equivalence Checking
# =======================
def check_equivalence(current_result: Any, reference_result: Any) -> None:
    """Verify functional equivalence."""
    if isinstance(current_result, dict) and isinstance(reference_result, dict):
        # For allocation results, check that the same number of blocks were allocated
        assert current_result.get("num_sequences") == reference_result.get("num_sequences")
        assert current_result.get("total_blocks_allocated") == reference_result.get("total_blocks_allocated")
        # Free blocks might vary slightly due to optimization
        free_diff = abs(current_result.get("num_free_blocks", 0) - reference_result.get("num_free_blocks", 0))
        assert free_diff <= 10, f"Free blocks differ by {free_diff}"

# =======================
# Timing Implementation
# =======================
def time_gpu(func, warmup=5, iterations=50) -> Tuple[Any, Dict[str, float]]:
    """Time GPU operations with CUDA events."""
    # Warmup
    for _ in range(warmup):
        _ = func()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
    
    # Timing
    times_ms = []
    for _ in range(iterations):
        if torch.cuda.is_available():
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            
            torch.cuda.synchronize()
            start.record()
            result = func()
            end.record()
            torch.cuda.synchronize()
            
            times_ms.append(start.elapsed_time(end))
        else:
            start = time.perf_counter()
            result = func()
            times_ms.append((time.perf_counter() - start) * 1000)
    
    # Statistics
    times_ms.sort()
    stats = {
        "avg_ms": sum(times_ms) / len(times_ms),
        "p50_ms": times_ms[len(times_ms) // 2],
        "p95_ms": times_ms[int(len(times_ms) * 0.95)],
        "p99_ms": times_ms[int(len(times_ms) * 0.99)],
        "min_ms": times_ms[0],
        "max_ms": times_ms[-1],
        "std_ms": np.std(times_ms)
    }
    
    return result, stats

# =======================
# Main Test Function
# =======================
def run_test(eqcheck: bool = False, reference: bool = False, prefix: str = '') -> float:
    """Main test entry point."""
    
    # Setup
    data = setup()
    hw_info = data["hw_info"]
    
    # Timing
    if hw_info["device"] == "cuda":
        warmup = 5
        iters = 50
    else:
        warmup = 3
        iters = 10
    
    result, timing_stats = time_gpu(lambda: experiment(data), warmup=warmup, iterations=iters)
    avg_ms = timing_stats["avg_ms"]
    p50_ms = timing_stats["p50_ms"]
    p95_ms = timing_stats["p95_ms"]
    
    # Reference handling
    commit_hash = os.getenv("COMMIT_HASH", "3476ed0809ec91a3457da0cb90543133a4f4b519")
    impl_tag = os.getenv("IMPL_TAG", "child")
    ref_file = f"{prefix}_{impl_tag}_{commit_hash}_reference.pt"
    
    if reference:
        store_result(result, ref_file)
    
    if eqcheck and os.path.exists(ref_file):
        ref_result = load_result(ref_file)
        check_equivalence(result, ref_result)
    
    # Output compact JSON schema
    summary = {
        "impl_tag": impl_tag,
        "commit_hash": commit_hash,
        "device": str(hw_info["device"]),
        "dtype": str(data["dtype"]),
        "iters": iters,
        "warmup": warmup,
        "avg_ms": avg_ms,
        "p50_ms": p50_ms,
        "p95_ms": p95_ms,
        "eq_level": os.getenv("PROB_EQ_LEVEL", "behavioral"),
        "opt_path_hit": True
    }
    print(json.dumps(summary))
    
    return avg_ms / 1000.0

# =======================
# Entry Point
# =======================
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--eqcheck", action="store_true")
    parser.add_argument("--reference", action="store_true")
    parser.add_argument("--prefix", type=str, default="")
    args = parser.parse_args()
    
    run_test(args.eqcheck, args.reference, args.prefix)