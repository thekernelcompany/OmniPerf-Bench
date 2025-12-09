#!/usr/bin/env python3
"""
Performance test for commit: 9c745d078e29e153a64300bd07636c7c9c1c42d5
Message: [Performance] Update xgrammar-related constrained decoding (#2056)

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
        # Focus on SamplingBatchInfo.update_regex_vocab_mask as it orchestrates the changes
        module_path = "sglang.srt.sampling.sampling_batch_info"
        symbol_name = "SamplingBatchInfo"
    
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
    
    device = torch.device(hw_info["device"])
    dtype = torch.float16 if hw_info["device"] == "cuda" else torch.float32
    
    # Create workload for constrained decoding mask operations
    batch_size = 32  # Multiple requests
    vocab_size = 32000  # Typical LLM vocab size
    
    # Create mock grammar objects for testing
    # We'll need to create a minimal mock since xgrammar may not be available
    class MockGrammar:
        def __init__(self, vocab_size):
            self.vocab_size = vocab_size
            
        def allocate_vocab_mask(self, vocab_size: int, batch_size: int, device) -> torch.Tensor:
            # New optimized allocation method
            return torch.zeros(batch_size, vocab_size, dtype=torch.bool, device=device)
        
        def fill_vocab_mask(self, vocab_mask: torch.Tensor, idx: int) -> None:
            # Fill with some pattern (simulate constrained tokens)
            vocab_mask[idx, :100] = True  # Block first 100 tokens
            vocab_mask[idx, 1000:1500] = True  # Block some middle tokens
        
        @staticmethod
        def apply_vocab_mask(logits: torch.Tensor, vocab_mask: torch.Tensor) -> None:
            logits.masked_fill_(vocab_mask, float("-inf"))
    
    # Create grammars list with some None entries (realistic scenario)
    grammars = []
    for i in range(batch_size):
        if i % 3 == 0:  # Some requests don't have constraints
            grammars.append(None)
        else:
            grammars.append(MockGrammar(vocab_size))
    
    # Create logits tensor for masking
    logits = torch.randn(batch_size, vocab_size, device=device, dtype=dtype)
    
    data = {
        "device": device,
        "dtype": dtype,
        "hw_info": hw_info,
        "batch_size": batch_size,
        "vocab_size": vocab_size,
        "grammars": grammars,
        "logits": logits.clone(),
        "temperatures": torch.ones(batch_size, device=device),
    }
    
    return data

# =======================
# Experiment Execution
# =======================
def experiment(data: Dict[str, Any]) -> Any:
    """Execute the optimized operation."""
    target, fq_name = resolve_target()
    
    # Create SamplingBatchInfo instance
    sampling_info = target(
        temperatures=data["temperatures"],
        top_ps=torch.ones(data["batch_size"], device=data["device"]),
        top_ks=torch.full((data["batch_size"],), 50, device=data["device"]),
        min_ps=torch.zeros(data["batch_size"], device=data["device"]),
        vocab_size=data["vocab_size"],
        device=data["device"],
    )
    
    # Set grammars
    sampling_info.grammars = data["grammars"]
    
    # Execute the optimized mask update operation
    with torch.no_grad():
        sampling_info.update_regex_vocab_mask()
        
        # Apply the mask if it was created
        if sampling_info.vocab_mask is not None and sampling_info.apply_mask is not None:
            logits_copy = data["logits"].clone()
            sampling_info.apply_mask(logits_copy, sampling_info.vocab_mask)
            result = {
                "vocab_mask": sampling_info.vocab_mask.cpu() if sampling_info.vocab_mask is not None else None,
                "masked_logits": logits_copy.cpu()
            }
        else:
            result = {
                "vocab_mask": None,
                "masked_logits": data["logits"].cpu()
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
    assert type(current_result) == type(reference_result), f"Type mismatch"
    
    if current_result["vocab_mask"] is None:
        assert reference_result["vocab_mask"] is None
    else:
        assert current_result["vocab_mask"].shape == reference_result["vocab_mask"].shape
        assert torch.equal(current_result["vocab_mask"], reference_result["vocab_mask"])
    
    # Check masked logits with tolerance
    rtol, atol = 1e-3, 1e-4
    torch.testing.assert_close(
        current_result["masked_logits"],
        reference_result["masked_logits"],
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
            _ = experiment(data)
            times.append((time.perf_counter() - start) * 1000)
        times.sort()
        avg_ms = sum(times) / len(times)
        p50_ms = times[len(times) // 2]
        p95_ms = times[int(len(times) * 0.95) - 1] if len(times) > 1 else times[0]
        # Produce a result for reference handling
        result = experiment(data)
    
    # Reference handling
    commit_hash = os.getenv("COMMIT_HASH", "9c745d078e29e153a64300bd07636c7c9c1c42d5")
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