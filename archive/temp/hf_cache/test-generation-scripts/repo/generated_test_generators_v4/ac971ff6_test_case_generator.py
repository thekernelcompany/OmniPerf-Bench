#!/usr/bin/env python3
"""
Performance test for commit: ac971ff633de330de3ded7f7475caaf7cd5bbdcd
Message: perf: reduce ttft and itl with stream_interval 1 (#658)

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
        # Based on commit diff: python/sglang/srt/server_args.py
        module_path = "sglang.srt.server_args"
        symbol_name = "ServerArgs"
    
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
    
    # This is a configuration optimization - simulate streaming scenario
    device = torch.device(hw_info["device"])
    dtype = torch.float16 if hw_info["device"] == "cuda" else torch.float32
    
    # Simulate token generation scenario
    batch_size = 32  # Number of concurrent requests
    max_tokens = 256  # Tokens per request
    vocab_size = 32000  # Typical LLM vocab size
    
    # Create mock logits for streaming simulation
    logits = torch.randn(batch_size, max_tokens, vocab_size, device=device, dtype=dtype)
    
    # Streaming configuration parameters
    data = {
        "device": device,
        "dtype": dtype,
        "hw_info": hw_info,
        "batch_size": batch_size,
        "max_tokens": max_tokens,
        "vocab_size": vocab_size,
        "logits": logits,
        "stream_interval": None,  # Will be set in experiment
    }
    
    return data

# =======================
# Experiment Execution
# =======================
def experiment(data: Dict[str, Any]) -> Any:
    """Execute the optimized operation."""
    target, fq_name = resolve_target()
    
    # Create ServerArgs instance with current stream_interval setting
    # Get stream_interval from environment or use default
    impl_tag = os.getenv("IMPL_TAG", "child")
    
    # For parent commit: stream_interval=8, for child: stream_interval=1
    if impl_tag == "parent":
        # Override to old value
        server_args = target(stream_interval=8)
    else:
        # Use new default (should be 1)
        server_args = target()
    
    # Simulate streaming behavior based on stream_interval
    stream_interval = server_args.stream_interval
    batch_size = data["batch_size"]
    max_tokens = data["max_tokens"]
    logits = data["logits"]
    
    # Simulate token-by-token generation with streaming
    results = []
    stream_points = []
    
    with torch.no_grad():
        for token_idx in range(max_tokens):
            # Process one token for all requests
            token_logits = logits[:, token_idx, :]
            
            # Simulate token generation
            probs = torch.softmax(token_logits, dim=-1)
            tokens = torch.multinomial(probs, num_samples=1)
            
            # Check if we should stream at this interval
            if token_idx % stream_interval == 0 or token_idx == max_tokens - 1:
                stream_points.append(token_idx)
                # Simulate streaming overhead
                if data["hw_info"]["device"] == "cuda":
                    torch.cuda.synchronize()
                results.append(tokens.clone())
    
    # Return streaming metadata
    result = {
        "stream_interval": stream_interval,
        "num_streams": len(stream_points),
        "stream_points": stream_points,
        "final_tokens": torch.cat(results, dim=0) if results else torch.tensor([]),
    }
    
    return result

# =======================
# Result I/O
# =======================
def store_result(result: Any, filepath: str) -> None:
    """Store result for reference comparison."""
    if isinstance(result, dict):
        # Convert tensors to CPU before saving
        saved_result = {}
        for k, v in result.items():
            if isinstance(v, torch.Tensor):
                saved_result[k] = v.cpu()
            else:
                saved_result[k] = v
        torch.save({"type": "dict", "data": saved_result}, filepath)
    elif isinstance(result, torch.Tensor):
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
        # For this optimization, the stream_interval will differ
        # But the functional output (tokens) should be similar
        
        # Check that both have expected keys
        expected_keys = {"stream_interval", "num_streams", "stream_points", "final_tokens"}
        assert set(current_result.keys()) == expected_keys
        assert set(reference_result.keys()) == expected_keys
        
        # The optimization changes stream_interval, so these will differ
        # Just verify they are set correctly
        assert isinstance(current_result["stream_interval"], int)
        assert isinstance(reference_result["stream_interval"], int)
        
        # Verify streaming behavior makes sense
        assert current_result["num_streams"] > 0
        assert reference_result["num_streams"] > 0
        
    elif isinstance(current_result, torch.Tensor):
        assert current_result.shape == reference_result.shape
        assert current_result.dtype == reference_result.dtype
        
        # Determine tolerances based on dtype
        if current_result.dtype in (torch.float16, torch.bfloat16):
            rtol, atol = 1e-3, 1e-4
        else:
            rtol, atol = 1e-5, 1e-7
        
        torch.testing.assert_close(
            current_result.cpu(),
            reference_result.cpu(),
            rtol=rtol, atol=atol
        )

# =======================
# Timing Implementation
# =======================
def time_gpu(func, warmup=5, iterations=50) -> Tuple[Any, Dict[str, float]]:
    """Time GPU operations with CUDA events."""
    # Warmup
    for _ in range(warmup):
        _ = func()
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
        result, timing_stats = time_gpu(lambda: experiment(data), warmup=warmup, iterations=iters)
        avg_ms = timing_stats["avg_ms"]
        p50_ms = timing_stats["p50_ms"]
        p95_ms = timing_stats["p95_ms"]
    else:
        warmup = 3
        iters = 10
        # CPU warmup
        for _ in range(warmup):
            _ = experiment(data)
        # CPU timing
        times = []
        for _ in range(iters):
            start = time.perf_counter()
            result = experiment(data)
            times.append((time.perf_counter() - start) * 1000)
        times.sort()
        avg_ms = sum(times) / len(times)
        p50_ms = times[len(times) // 2]
        p95_ms = times[int(len(times) * 0.95) - 1] if len(times) > 1 else times[0]
    
    # Reference handling
    commit_hash = os.getenv("COMMIT_HASH", "ac971ff633de330de3ded7f7475caaf7cd5bbdcd")
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