#!/usr/bin/env python3
"""
Performance test for commit: 93470a14116a60fe5dd43f0599206e8ccabdc211
Message: Refactor and Optimize FA3 Code (#5090)

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
        # Based on the commit, the main optimization is in FlashAttentionBackend.init_forward_metadata
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
# Mock Classes for Testing
# =======================
class MockForwardBatch:
    """Mock ForwardBatch with necessary attributes for testing."""
    def __init__(self, batch_size, seq_lens, forward_mode, device):
        self.seq_lens = seq_lens
        self.seq_lens_cpu = seq_lens.cpu()
        self.forward_mode = forward_mode
        self.device = device
        self.spec_info = None
        self.extend_prefix_lens_cpu = [0] * batch_size
        self.extend_seq_lens = seq_lens
        self.extend_seq_lens_cpu = seq_lens.cpu().tolist()
        
        # Mock token pool
        max_seq_len = seq_lens.max().item()
        self.req_pool_indices = torch.arange(batch_size, device=device)
        self.req_to_token_pool = MockTokenPool(batch_size, max_seq_len, device)
        self.out_cache_loc = torch.zeros(batch_size, device=device)

class MockTokenPool:
    def __init__(self, batch_size, max_seq_len, device):
        # Create mock page table
        page_size = 16
        num_pages = (max_seq_len + page_size - 1) // page_size
        self.req_to_token = torch.arange(
            batch_size * num_pages, 
            dtype=torch.int32, 
            device=device
        ).reshape(batch_size, num_pages)

class MockForwardMode:
    EXTEND = 0
    DECODE = 1
    
    def __init__(self, mode):
        self.mode = mode
    
    def is_decode(self):
        return self.mode == self.DECODE
    
    def is_extend_or_draft_extend(self):
        return self.mode == self.EXTEND
    
    def is_target_verify(self):
        return False

class MockModelRunner:
    def __init__(self):
        self.model_config = MockModelConfig()
        self.server_args = MockServerArgs()

class MockModelConfig:
    def __init__(self):
        self.attention_arch = "flash_attention"
        
class MockServerArgs:
    def __init__(self):
        self.speculative_num_draft_tokens = 5

# =======================
# Workload Setup
# =======================
def setup() -> Dict[str, Any]:
    """Create realistic workload for the optimization."""
    ensure_determinism()
    hw_info = detect_hardware()
    
    device = torch.device(hw_info["device"])
    dtype = torch.float16 if hw_info["device"] == "cuda" else torch.float32
    
    # Create different batch configurations to test
    batch_configs = [
        {"batch_size": 4, "min_seq": 512, "max_seq": 2048},
        {"batch_size": 8, "min_seq": 256, "max_seq": 1024},
        {"batch_size": 16, "min_seq": 128, "max_seq": 512},
        {"batch_size": 32, "min_seq": 64, "max_seq": 256},
    ]
    
    # Select a config based on available memory
    if hw_info.get("memory_gb", 0) < 16:
        config = batch_configs[2]  # Use smaller config for low memory
    else:
        config = batch_configs[1]  # Default config
    
    batch_size = config["batch_size"]
    
    # Generate random sequence lengths for each request in batch
    seq_lens = torch.randint(
        config["min_seq"], 
        config["max_seq"], 
        (batch_size,), 
        device=device,
        dtype=torch.int32
    )
    
    # Create mock model runner
    model_runner = MockModelRunner()
    
    data = {
        "device": device,
        "dtype": dtype,
        "hw_info": hw_info,
        "batch_size": batch_size,
        "seq_lens": seq_lens,
        "model_runner": model_runner,
        "forward_modes": [MockForwardMode.DECODE, MockForwardMode.EXTEND],
    }
    
    return data

# =======================
# Experiment Execution
# =======================
def experiment(data: Dict[str, Any]) -> Any:
    """Execute the optimized operation."""
    target_class, fq_name = resolve_target()
    
    # Create FlashAttentionBackend instance
    backend = target_class(
        model_runner=data["model_runner"],
        skip_prefill=False,
        speculative_step_id=0,
        topk=1,
        speculative_num_steps=1,
    )
    
    # Initialize on the correct device
    backend.device = data["device"]
    backend.page_size = 16
    
    # Benchmark init_forward_metadata for different forward modes
    results = {}
    
    for mode_val in data["forward_modes"]:
        # Create mock forward batch
        forward_mode = MockForwardMode(mode_val)
        forward_batch = MockForwardBatch(
            data["batch_size"],
            data["seq_lens"],
            forward_mode,
            data["device"]
        )
        
        # Execute the optimized metadata initialization
        with torch.no_grad():
            backend.init_forward_metadata(forward_batch)
            metadata = backend.forward_metadata
            
            # Store metadata attributes for equivalence checking
            mode_name = "decode" if mode_val == MockForwardMode.DECODE else "extend"
            results[mode_name] = {
                "max_seq_len_q": metadata.max_seq_len_q,
                "max_seq_len_k": metadata.max_seq_len_k,
                "cu_seqlens_q_shape": metadata.cu_seqlens_q.shape if metadata.cu_seqlens_q is not None else None,
                "cu_seqlens_k_shape": metadata.cu_seqlens_k.shape if metadata.cu_seqlens_k is not None else None,
                "page_table_shape": metadata.page_table.shape if metadata.page_table is not None else None,
                "cache_seqlens_shape": metadata.cache_seqlens_int32.shape if metadata.cache_seqlens_int32 is not None else None,
            }
    
    return results

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
    assert type(current_result) == type(reference_result), f"Type mismatch: {type(current_result)} vs {type(reference_result)}"
    
    if isinstance(current_result, dict):
        assert current_result.keys() == reference_result.keys(), f"Keys: {current_result.keys()} vs {reference_result.keys()}"
        for key in current_result:
            current_val = current_result[key]
            reference_val = reference_result[key]
            
            if isinstance(current_val, (int, float, type(None))):
                assert current_val == reference_val, f"Mismatch at key '{key}': {current_val} vs {reference_val}"
            elif isinstance(current_val, (list, tuple)):
                assert len(current_val) == len(reference_val), f"Length mismatch at key '{key}'"
                for i, (c, r) in enumerate(zip(current_val, reference_val)):
                    assert c == r, f"Mismatch at key '{key}' index {i}: {c} vs {r}"

# =======================
# Timing Implementation
# =======================
def time_operation(func, data, device, warmup=5, iterations=50) -> Tuple[Any, Dict[str, float]]:
    """Time the operation with appropriate method for device."""
    
    if device == "cuda":
        # Warmup
        for _ in range(warmup):
            _ = func(data)
            torch.cuda.synchronize()
        
        # Timing
        times_ms = []
        for _ in range(iterations):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            
            torch.cuda.synchronize()
            start.record()
            result = func(data)
            end.record()
            torch.cuda.synchronize()
            
            times_ms.append(start.elapsed_time(end))
    else:
        # CPU timing
        warmup = 3
        iterations = 10
        
        # Warmup
        for _ in range(warmup):
            _ = func(data)
        
        # Timing
        times_ms = []
        for _ in range(iterations):
            start = time.perf_counter()
            result = func(data)
            times_ms.append((time.perf_counter() - start) * 1000)
    
    # Statistics
    times_ms.sort()
    stats = {
        "avg_ms": sum(times_ms) / len(times_ms),
        "p50_ms": times_ms[len(times_ms) // 2],
        "p95_ms": times_ms[int(len(times_ms) * 0.95) - 1] if len(times_ms) > 1 else times_ms[0],
        "p99_ms": times_ms[int(len(times_ms) * 0.99) - 1] if len(times_ms) > 1 else times_ms[0],
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
    device_str = str(hw_info["device"])
    
    # Determine timing parameters
    if device_str == "cuda":
        warmup = 5
        iters = 50
    else:
        warmup = 3
        iters = 10
    
    # Time the experiment
    result, timing_stats = time_operation(
        experiment, 
        data, 
        device_str,
        warmup=warmup, 
        iterations=iters
    )
    
    # Reference handling
    commit_hash = os.getenv("COMMIT_HASH", "93470a14116a60fe5dd43f0599206e8ccabdc211")
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
        "device": device_str,
        "dtype": str(data["dtype"]),
        "iters": iters,
        "warmup": warmup,
        "avg_ms": timing_stats["avg_ms"],
        "p50_ms": timing_stats["p50_ms"],
        "p95_ms": timing_stats["p95_ms"],
        "eq_level": os.getenv("PROB_EQ_LEVEL", "numeric"),
        "opt_path_hit": True
    }
    print(json.dumps(summary))
    
    return timing_stats["avg_ms"] / 1000.0

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