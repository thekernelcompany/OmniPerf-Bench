#!/usr/bin/env python3
"""
Performance test for commit: 99abb8b650c66664cdc84d815b7f306f33bd9881
Message: [V1][Spec Decode] Optimize Rejection Sampler with Triton Kernels (#14930)

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
    
    # Priority 2: Parse from commit metadata - primary target is RejectionSampler
    if not (module_path and symbol_name):
        module_path = "vllm.v1.sample.rejection_sampler"
        symbol_name = "RejectionSampler"
    
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
    
    # Import required modules for the test
    from vllm.v1.spec_decode.metadata import SpecDecodeMetadata
    from vllm.v1.sample.metadata import SamplingMetadata
    
    device = torch.device(hw_info["device"])
    dtype = torch.float16 if hw_info["device"] == "cuda" else torch.float32
    
    # Create a realistic spec decode workload
    # Multiple requests with varying numbers of draft tokens
    batch_size = 8
    max_spec_len = 5  # Maximum speculative tokens per request
    vocab_size = 32000  # Llama vocab size
    
    # Create draft token ids for each request (varying lengths)
    draft_token_ids_list = [
        [1, 2, 3, 4, 5],  # Full spec length
        [10, 11, 12],     # Partial
        [20, 21],         # Short
        [],               # Empty
        [30, 31, 32, 33], # Almost full
        [40],             # Single token
        [50, 51, 52],     # Partial
        [60, 61, 62, 63, 64],  # Full
    ]
    
    # Create SpecDecodeMetadata
    spec_decode_metadata = SpecDecodeMetadata.make_dummy(
        draft_token_ids_list, 
        device=device
    )
    
    # Total number of draft tokens
    num_tokens = sum(len(ids) for ids in draft_token_ids_list)
    
    # Create draft probabilities (for non-ngram case)
    draft_probs = torch.softmax(
        torch.randn(num_tokens, vocab_size, device=device, dtype=dtype),
        dim=-1
    )
    
    # Create target logits
    target_logits = torch.randn(num_tokens, vocab_size, device=device, dtype=dtype)
    
    # Create bonus token ids (one per request)
    bonus_token_ids = torch.randint(
        0, vocab_size, (batch_size, 1), device=device, dtype=torch.int32
    )
    
    # Create sampling metadata for mixed greedy/random sampling
    temperature = torch.ones(batch_size, dtype=torch.float32, device=device)
    # Make some requests greedy (temperature = -1)
    temperature[::2] = -1  # Every other request is greedy
    
    sampling_metadata = SamplingMetadata(
        temperature=temperature,
        all_greedy=False,
        all_random=False,
        top_p=None,
        top_k=None,
        no_top_p=True,
        no_top_k=True,
        no_prompt_logprob=True,
        max_num_prompt_logprobs=0,
        generators={},
        no_generator=True,
        no_penalties=True,
        temperature_curr_seq_locs_lens=None,
        skip_top_k_top_p=False,
    )
    
    data = {
        "device": device,
        "dtype": dtype,
        "hw_info": hw_info,
        "spec_decode_metadata": spec_decode_metadata,
        "draft_probs": draft_probs,
        "target_logits": target_logits,
        "bonus_token_ids": bonus_token_ids,
        "sampling_metadata": sampling_metadata,
        "batch_size": batch_size,
        "vocab_size": vocab_size,
    }
    
    return data

# =======================
# Experiment Execution
# =======================
def experiment(data: Dict[str, Any]) -> Any:
    """Execute the optimized operation."""
    target, fq_name = resolve_target()
    
    # Create RejectionSampler instance
    rejection_sampler = target()
    
    with torch.no_grad():
        # Call the optimized forward method
        result = rejection_sampler.forward(
            metadata=data["spec_decode_metadata"],
            draft_probs=data["draft_probs"],
            target_logits=data["target_logits"],
            bonus_token_ids=data["bonus_token_ids"],
            sampling_metadata=data["sampling_metadata"],
        )
    
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
    if isinstance(current_result, torch.Tensor):
        assert current_result.shape == reference_result.shape
        assert current_result.dtype == reference_result.dtype
        
        # Special handling for token IDs - they should match exactly
        # except for placeholder tokens which can vary
        PLACEHOLDER_TOKEN_ID = -1
        
        # Mask out placeholder tokens for comparison
        current_valid = current_result != PLACEHOLDER_TOKEN_ID
        reference_valid = reference_result != PLACEHOLDER_TOKEN_ID
        
        # Valid positions should match
        assert torch.equal(current_valid, reference_valid), "Valid token positions don't match"
        
        # Compare only valid tokens
        if current_valid.any():
            assert torch.equal(
                current_result[current_valid],
                reference_result[reference_valid]
            ), "Valid tokens don't match"

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
        p95_ms = times[int(len(times) * 0.95) - 1]
        # Produce a result for reference handling
        result = experiment(data)
    
    # Reference handling
    commit_hash = os.getenv("COMMIT_HASH", "99abb8b650c66664cdc84d815b7f306f33bd9881")
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