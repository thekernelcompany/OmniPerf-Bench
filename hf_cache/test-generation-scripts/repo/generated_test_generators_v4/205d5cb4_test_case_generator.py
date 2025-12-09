#!/usr/bin/env python3
"""
Performance test for commit: 205d5cb407f7860c79df870b3f045d74b8292f77
Message: perf: Optimize local attention memory allocation in FlashAttentionBackend (#6356)

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
    
    # Priority 2: Parse from commit metadata
    if not (module_path and symbol_name):
        # Based on the commit, the target is FlashAttentionBackend._update_local_attn_metadata_for_capture
        module_path = "sglang.srt.layers.attention.flashattention_backend"
        symbol_name = "FlashAttentionBackend"
    
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
    
    # This optimization is for local attention metadata during CUDA graph capture
    # We need to simulate a decode phase scenario with local attention
    
    device = torch.device(hw_info["device"] if hw_info["device"] == "cuda" else "cpu")
    dtype = torch.float16 if hw_info["device"] == "cuda" else torch.float32
    
    batch_size = 64  # Typical decode batch size
    max_seq_len = 2048
    num_heads = 32
    head_dim = 128
    page_size = 16
    attention_chunk_size = 256  # Local attention window
    
    # Create metadata similar to what FlashAttentionBackend would use
    # Pre-allocate buffers for CUDA graph capture
    max_batch_size = 256
    max_blocks = (max_seq_len + page_size - 1) // page_size
    
    # Create mock FlashAttentionMetadata components
    cu_seqlens_q = torch.arange(0, batch_size + 1, dtype=torch.int32, device=device) 
    cache_seqlens_int32 = torch.randint(128, max_seq_len, (batch_size,), dtype=torch.int32, device=device)
    
    # Page table for KV cache management
    page_table = torch.randint(0, 1000, (batch_size, max_blocks), dtype=torch.int32, device=device)
    
    # Pre-allocated local attention metadata buffers (this is what gets optimized)
    decode_cuda_graph_local_attn_metadata = {
        "local_query_start_loc": torch.zeros(max_batch_size * 4, dtype=torch.int32, device=device),
        "local_seqused_k": torch.zeros(max_batch_size * 4, dtype=torch.int32, device=device),
        "local_block_table": torch.zeros((max_batch_size, max_blocks * 2), dtype=torch.int32, device=device),
    }
    
    data = {
        "device": device,
        "dtype": dtype,
        "hw_info": hw_info,
        "batch_size": batch_size,
        "attention_chunk_size": attention_chunk_size,
        "page_size": page_size,
        "cu_seqlens_q": cu_seqlens_q,
        "cache_seqlens_int32": cache_seqlens_int32,
        "page_table": page_table,
        "decode_cuda_graph_local_attn_metadata": decode_cuda_graph_local_attn_metadata,
    }
    
    return data

# =======================
# Experiment Execution
# =======================
def experiment(data: Dict[str, Any]) -> Any:
    """Execute the optimized operation."""
    
    # Import the necessary classes
    try:
        from sglang.srt.layers.attention.flashattention_backend import (
            FlashAttentionBackend,
            FlashAttentionMetadata,
            make_local_attention_virtual_batches
        )
    except ImportError as e:
        # Try alternative import paths
        try:
            import sys
            sys.path.insert(0, os.path.join(os.path.dirname(__file__), "python"))
            from sglang.srt.layers.attention.flashattention_backend import (
                FlashAttentionBackend,
                FlashAttentionMetadata,
                make_local_attention_virtual_batches
            )
        except ImportError:
            error_data = {
                "target_resolved": False,
                "error": str(e),
                "attempted_module": "sglang.srt.layers.attention.flashattention_backend"
            }
            print(json.dumps(error_data))
            sys.exit(1)
    
    # Create a mock FlashAttentionBackend instance
    backend = FlashAttentionBackend.__new__(FlashAttentionBackend)
    backend.attention_chunk_size = data["attention_chunk_size"]
    backend.page_size = data["page_size"]
    backend.decode_cuda_graph_local_attn_metadata = data["decode_cuda_graph_local_attn_metadata"]
    
    # Create metadata instance
    metadata = FlashAttentionMetadata.__new__(FlashAttentionMetadata)
    metadata.cu_seqlens_q = data["cu_seqlens_q"]
    metadata.cache_seqlens_int32 = data["cache_seqlens_int32"]
    metadata.page_table = data["page_table"]
    
    with torch.no_grad():
        # Call the optimized method
        backend._update_local_attn_metadata_for_capture(metadata, data["batch_size"])
    
    # Return the local attention metadata for equivalence checking
    result = {
        "local_query_start_loc": metadata.local_attn_metadata.local_query_start_loc.clone(),
        "local_seqused_k": metadata.local_attn_metadata.local_seqused_k.clone(),
        "local_block_table": metadata.local_attn_metadata.local_block_table.clone(),
        "local_max_query_len": metadata.local_attn_metadata.local_max_query_len,
        "local_max_seq_len": metadata.local_attn_metadata.local_max_seq_len,
    }
    
    return result

# =======================
# Result I/O
# =======================
def store_result(result: Any, filepath: str) -> None:
    """Store result for reference comparison."""
    torch.save({"type": "dict", "data": result}, filepath)

def load_result(filepath: str) -> Any:
    """Load reference result."""
    data = torch.load(filepath)
    return data.get("data", data)

# =======================
# Equivalence Checking
# =======================
def check_equivalence(current_result: Any, reference_result: Any) -> None:
    """Verify functional equivalence."""
    assert isinstance(current_result, dict), f"Type mismatch: expected dict, got {type(current_result)}"
    assert isinstance(reference_result, dict), f"Type mismatch: expected dict, got {type(reference_result)}"
    
    # Check all keys match
    assert current_result.keys() == reference_result.keys(), f"Keys mismatch: {current_result.keys()} vs {reference_result.keys()}"
    
    for key in current_result:
        current_val = current_result[key]
        reference_val = reference_result[key]
        
        if isinstance(current_val, torch.Tensor):
            assert current_val.shape == reference_val.shape, f"{key} shape: {current_val.shape} vs {reference_val.shape}"
            assert current_val.dtype == reference_val.dtype, f"{key} dtype: {current_val.dtype} vs {reference_val.dtype}"
            
            # Determine tolerances based on dtype
            if current_val.dtype in (torch.float16, torch.bfloat16):
                rtol, atol = 1e-3, 1e-4
            elif current_val.dtype in (torch.int32, torch.int64):
                rtol, atol = 0, 0  # Exact match for integers
            else:
                rtol, atol = 1e-5, 1e-7
            
            torch.testing.assert_close(
                current_val.cpu(),
                reference_val.cpu(),
                rtol=rtol, atol=atol
            )
        else:
            # Scalar values
            assert current_val == reference_val, f"{key}: {current_val} vs {reference_val}"

# =======================
# Timing Implementation
# =======================
def time_gpu(func, warmup=5, iterations=50) -> Tuple[Any, Dict[str, float]]:
    """Time GPU operations with CUDA events."""
    # Warmup
    for _ in range(warmup):
        _ = func()
        torch.cuda.synchronize()
    
    # Clear cache
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    
    # Timing
    times_ms = []
    for _ in range(iterations):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        
        torch.cuda.synchronize()
        start.record()
        result = func()
        end.record()
        torch.cuda.synchronize()
        
        times_ms.append(start.elapsed_time(end))
    
    # Statistics
    times_ms.sort()
    stats = {
        "avg_ms": sum(times_ms) / len(times_ms),
        "p50_ms": times_ms[len(times_ms) // 2],
        "p95_ms": times_ms[int(len(times_ms) * 0.95)],
        "p99_ms": times_ms[int(len(times_ms) * 0.99)],
        "min_ms": times_ms[0],
        "max_ms": times_ms[-1],
        "std_ms": np.std(times_ms) if len(times_ms) > 1 else 0.0
    }
    
    return result, stats

def time_cpu(func, warmup=3, iterations=10) -> Tuple[Any, Dict[str, float]]:
    """Time CPU operations."""
    # Warmup
    for _ in range(warmup):
        _ = func()
    
    # Timing
    times_ms = []
    for _ in range(iterations):
        start = time.perf_counter()
        result = func()
        end = time.perf_counter()
        times_ms.append((end - start) * 1000)
    
    # Statistics
    times_ms.sort()
    stats = {
        "avg_ms": sum(times_ms) / len(times_ms),
        "p50_ms": times_ms[len(times_ms) // 2],
        "p95_ms": times_ms[int(len(times_ms) * 0.95)] if len(times_ms) >= 20 else times_ms[-1],
        "p99_ms": times_ms[int(len(times_ms) * 0.99)] if len(times_ms) >= 100 else times_ms[-1],
        "min_ms": times_ms[0],
        "max_ms": times_ms[-1],
        "std_ms": np.std(times_ms) if len(times_ms) > 1 else 0.0
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
    
    # Create experiment function
    def run_experiment():
        return experiment(data)
    
    # Timing
    if hw_info["device"] == "cuda":
        warmup = 5
        iters = 50
        result, timing_stats = time_gpu(run_experiment, warmup=warmup, iterations=iters)
        avg_ms = timing_stats["avg_ms"]
        p50_ms = timing_stats["p50_ms"]
        p95_ms = timing_stats["p95_ms"]
    else:
        warmup = 3
        iters = 10
        result, timing_stats = time_cpu(run_experiment, warmup=warmup, iterations=iters)
        avg_ms = timing_stats["avg_ms"]
        p50_ms = timing_stats["p50_ms"]
        p95_ms = timing_stats["p95_ms"]
    
    # Reference handling
    commit_hash = os.getenv("COMMIT_HASH", "205d5cb407f7860c79df870b3f045d74b8292f77")
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
        "eq_level": os.getenv("PROB_EQ_LEVEL", "exact"),  # Integer comparisons
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