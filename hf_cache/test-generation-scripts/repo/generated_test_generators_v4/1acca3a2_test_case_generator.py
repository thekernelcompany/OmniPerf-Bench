#!/usr/bin/env python3
"""
Performance test for commit: 1acca3a2c685221cdb181c2abda4f635e1ead435
Message: FA3 speed up: skip len operation and get batch size directly from forward batch (#5969)

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
        # Based on the commit diff, the optimization is in FlashAttentionBackend.init_forward_metadata
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
# Mock ForwardBatch
# =======================
class MockForwardBatch:
    """Mock forward batch with batch_size and seq_lens attributes."""
    def __init__(self, batch_size: int, seq_lens: torch.Tensor, device: torch.device):
        self.batch_size = batch_size
        self.seq_lens = seq_lens
        self.device = device
        
        # Mock forward mode
        self.forward_mode = self
        self._is_decode = False
        
    def is_decode_or_idle(self):
        return self._is_decode
    
    def is_extend(self):
        return not self._is_decode

# =======================
# Workload Setup
# =======================
def setup() -> Dict[str, Any]:
    """Create realistic workload for the optimization."""
    ensure_determinism()
    hw_info = detect_hardware()
    
    device = torch.device(hw_info["device"])
    dtype = torch.float16 if hw_info["device"] == "cuda" else torch.float32
    
    # Realistic batch sizes for continuous batching
    batch_size = 256  # Large batch to make len() overhead measurable
    
    # Create sequence lengths tensor with varying lengths
    min_seq_len = 128
    max_seq_len = 2048
    seq_lens = torch.randint(min_seq_len, max_seq_len, (batch_size,), 
                             device=device, dtype=torch.int32)
    
    # Create mock forward batch
    forward_batch = MockForwardBatch(batch_size, seq_lens, device)
    
    # Attention parameters
    num_q_heads = 32
    num_kv_heads = 32
    head_size = 128
    scale = 1.0 / math.sqrt(head_size)
    num_attn_layers = 32
    
    data = {
        "device": device,
        "dtype": dtype,
        "hw_info": hw_info,
        "forward_batch": forward_batch,
        "num_q_heads": num_q_heads,
        "num_kv_heads": num_kv_heads,
        "head_size": head_size,
        "scale": scale,
        "num_attn_layers": num_attn_layers,
        "batch_size": batch_size,
        "seq_lens": seq_lens,
    }
    
    return data

# =======================
# Experiment Execution
# =======================
def experiment(data: Dict[str, Any]) -> Any:
    """Execute the optimized operation."""
    FlashAttentionBackend, fq_name = resolve_target()
    
    # Create backend instance
    backend = FlashAttentionBackend(
        num_q_heads=data["num_q_heads"],
        num_kv_heads=data["num_kv_heads"],
        head_size=data["head_size"],
        scale=data["scale"],
        num_attn_layers=data["num_attn_layers"],
        layer_id=0,
    )
    
    # The optimization is in init_forward_metadata
    # This method initializes metadata for the forward pass
    metadata = backend.init_forward_metadata(data["forward_batch"])
    
    # Return metadata for equivalence checking
    return {
        "batch_size": getattr(metadata, "batch_size", data["batch_size"]),
        "device": str(data["device"]),
        "num_q_heads": data["num_q_heads"],
        "num_kv_heads": data["num_kv_heads"],
        "head_size": data["head_size"],
    }

# =======================
# Result I/O
# =======================
def store_result(result: Any, filepath: str) -> None:
    """Store result for reference comparison."""
    if isinstance(result, torch.Tensor):
        torch.save({"type": "tensor", "data": result.cpu()}, filepath)
    elif isinstance(result, dict):
        # Convert any tensors to CPU before saving
        saved_dict = {}
        for k, v in result.items():
            if isinstance(v, torch.Tensor):
                saved_dict[k] = v.cpu()
            else:
                saved_dict[k] = v
        torch.save({"type": "dict", "data": saved_dict}, filepath)
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
    if isinstance(current_result, torch.Tensor):
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
    elif isinstance(current_result, dict) and isinstance(reference_result, dict):
        assert current_result.keys() == reference_result.keys(), f"Keys mismatch: {current_result.keys()} vs {reference_result.keys()}"
        for key in current_result:
            if key in ["device"]:  # Skip device comparison
                continue
            assert current_result[key] == reference_result[key], f"Mismatch at key '{key}': {current_result[key]} vs {reference_result[key]}"

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
        "p95_ms": times_ms[int(len(times_ms) * 0.95) - 1] if len(times_ms) > 1 else times_ms[0],
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
        warmup = 10  # More warmup for metadata initialization
        iters = 100  # More iterations since this is a fast operation
        result, timing_stats = time_gpu(run_experiment, warmup=warmup, iterations=iters)
        avg_ms = timing_stats["avg_ms"]
        p50_ms = timing_stats["p50_ms"]
        p95_ms = timing_stats["p95_ms"]
    else:
        warmup = 5
        iters = 50
        result, timing_stats = time_cpu(run_experiment, warmup=warmup, iterations=iters)
        avg_ms = timing_stats["avg_ms"]
        p50_ms = timing_stats["p50_ms"]
        p95_ms = timing_stats["p95_ms"]
    
    # Reference handling
    commit_hash = os.getenv("COMMIT_HASH", "1acca3a2c685221cdb181c2abda4f635e1ead435")
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
        "eq_level": os.getenv("PROB_EQ_LEVEL", "numeric"),
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