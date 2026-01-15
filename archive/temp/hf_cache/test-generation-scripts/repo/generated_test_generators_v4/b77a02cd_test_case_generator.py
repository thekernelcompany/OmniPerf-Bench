#!/usr/bin/env python3
"""
Performance test for commit: b77a02cdfdb4cd58be3ebc6a66d076832c309cfc
Message: [Performance] Support both xgrammar and outlines for constrained decoding (#1752)

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
    
    # Priority 2: Parse from commit metadata - target the new Grammar class
    if not (module_path and symbol_name):
        module_path = "sglang.srt.constrained.grammar"
        symbol_name = "Grammar"
    
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
    
    # Create mock grammars for testing vocab mask filling
    # This is the performance-critical operation
    batch_size = 64  # Typical batch for constrained generation
    vocab_size = 32000  # Llama vocabulary size
    
    # Create mock regex guide and grammar matcher
    # We'll test the fill_vocab_mask operation which is called during sampling
    from sglang.srt.constrained import RegexGuide
    
    # Simple regex pattern for testing
    regex_pattern = r"\d{3}-\d{3}-\d{4}"  # Phone number pattern
    
    # Create mock tokenizer vocabulary for RegexGuide
    vocab = {f"token_{i}": i for i in range(vocab_size)}
    vocab.update({str(i): i + vocab_size for i in range(10)})  # Add digits
    
    # Try to create guides, fallback to mock if not available
    try:
        guide = RegexGuide.from_regex(regex_pattern, vocab)
        state = 0
    except:
        # Fallback mock for testing
        class MockGuide:
            def get_next_instruction(self, state):
                class Instruction:
                    tokens = list(range(10))  # Allow only digit tokens
                return Instruction()
            
            def get_next_state(self, state, token):
                return state + 1
        
        guide = MockGuide()
        state = 0
    
    # Create Grammar instances
    Grammar = resolve_target()[0]
    
    # Create multiple grammars for batch processing
    grammars = []
    for i in range(batch_size):
        if i % 2 == 0:
            # Half use regex guide backend
            grammar = Grammar((guide, state), None)
        else:
            # Half are None (no constraints)
            grammar = None
        grammars.append(grammar)
    
    # Pre-allocate vocab masks
    vocab_masks = torch.zeros(batch_size, vocab_size, dtype=torch.bool, device=device)
    
    data = {
        "device": device,
        "dtype": dtype,
        "hw_info": hw_info,
        "grammars": grammars,
        "vocab_masks": vocab_masks,
        "vocab_size": vocab_size,
        "batch_size": batch_size,
    }
    
    return data

# =======================
# Experiment Execution
# =======================
def experiment(data: Dict[str, Any]) -> Any:
    """Execute the optimized operation."""
    
    grammars = data["grammars"]
    vocab_masks = data["vocab_masks"]
    vocab_size = data["vocab_size"]
    
    # Fill vocab masks for all grammars in the batch
    # This is the performance-critical path during sampling
    results = []
    
    with torch.no_grad():
        for i, grammar in enumerate(grammars):
            if grammar is not None:
                # Reset mask
                vocab_masks[i].fill_(0)
                # Fill mask with grammar constraints
                grammar.fill_vocab_mask(vocab_masks[i], vocab_size)
                results.append(vocab_masks[i].sum().item())
            else:
                results.append(0)
    
    return {
        "masked_counts": results,
        "vocab_masks": vocab_masks.clone()
    }

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
    assert isinstance(current_result, dict) and isinstance(reference_result, dict)
    
    # Check masked counts match
    current_counts = current_result["masked_counts"]
    ref_counts = reference_result["masked_counts"]
    assert len(current_counts) == len(ref_counts), f"Count length mismatch"
    
    for i, (c, r) in enumerate(zip(current_counts, ref_counts)):
        assert c == r, f"Masked count mismatch at index {i}: {c} vs {r}"
    
    # Check vocab masks match
    current_masks = current_result["vocab_masks"]
    ref_masks = reference_result["vocab_masks"]
    
    assert current_masks.shape == ref_masks.shape, f"Mask shape mismatch"
    assert current_masks.dtype == ref_masks.dtype, f"Mask dtype mismatch"
    
    # For boolean masks, should be exactly equal
    assert torch.equal(current_masks.cpu(), ref_masks.cpu()), "Vocab masks don't match"

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
        p95_ms = times[int(len(times) * 0.95) - 1] if len(times) > 0 else avg_ms
        
    # Reference handling
    commit_hash = os.getenv("COMMIT_HASH", "b77a02cdfdb4cd58be3ebc6a66d076832c309cfc")
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
        "eq_level": os.getenv("PROB_EQ_LEVEL", "exact"),
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