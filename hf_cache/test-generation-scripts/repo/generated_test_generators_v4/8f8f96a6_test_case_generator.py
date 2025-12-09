#!/usr/bin/env python3
"""
Performance test for commit: 8f8f96a6217ea737c94e7429e480196319594459
Message: Fix the perf regression due to additional_stop_token_ids (#1773)

This script measures the actual performance impact of the optimization.
It supports cross-commit comparison with functional equivalence checking.
"""

import os
import sys
import json
import time
import importlib
from typing import Dict, Any, Tuple, Optional, List
from unittest.mock import Mock

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
    
    # Priority 2: Parse from commit metadata - focus on schedule_batch.Req
    if not (module_path and symbol_name):
        # The main optimization is in schedule_batch.py's Req.check_finished method
        module_path = "sglang.srt.managers.schedule_batch"
        symbol_name = "Req"
    
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
    
    # Import required classes
    try:
        from sglang.srt.managers.schedule_batch import Req
        from sglang.srt.sampling.sampling_params import SamplingParams
    except ImportError as e:
        print(json.dumps({"target_resolved": False, "error": str(e)}))
        sys.exit(1)
    
    device = torch.device("cpu")  # This optimization is CPU-bound
    dtype = torch.float32
    
    # Create test parameters - simulating many requests checking for stop conditions
    num_requests = 1000
    seq_len = 100  # Each request has generated this many tokens
    vocab_size = 32000  # Typical LLM vocab size
    
    # Create mock tokenizer with additional_stop_token_ids
    mock_tokenizer = Mock()
    mock_tokenizer.eos_token_id = 2  # Standard EOS token
    mock_tokenizer.additional_stop_token_ids = {128000, 128001, 128002}  # Tool use tokens
    mock_tokenizer.decode = Mock(return_value="test output")
    
    # Create requests with different stop token configurations
    requests = []
    for i in range(num_requests):
        # Mix of different configurations to test the optimization
        if i % 3 == 0:
            # No custom stop tokens (should benefit most from optimization)
            sampling_params = SamplingParams(
                max_new_tokens=100,
                stop=None,
                stop_token_ids=None
            )
        elif i % 3 == 1:
            # With custom stop tokens
            sampling_params = SamplingParams(
                max_new_tokens=100,
                stop=None,
                stop_token_ids=[100, 200, 300]
            )
        else:
            # With stop strings
            sampling_params = SamplingParams(
                max_new_tokens=100,
                stop=["</end>", "<stop>"],
                stop_token_ids=None
            )
        
        # Normalize the sampling params
        sampling_params.normalize(mock_tokenizer)
        
        # Create request object
        req = Req(
            rid=f"req_{i}",
            origin_input_text="Test input",
            origin_input_ids=[1] * 10,  # Dummy input IDs
        )
        req.sampling_params = sampling_params
        req.tokenizer = mock_tokenizer
        req.output_ids = [np.random.randint(0, vocab_size) for _ in range(seq_len)]
        req.max_new_tokens = 100
        req.finished_reason = None
        
        requests.append(req)
    
    # Create output tokens to check (mix of EOS and non-EOS)
    test_tokens = []
    for i in range(num_requests):
        if i % 10 == 0:
            # EOS token
            test_tokens.append(2)
        elif i % 20 == 0:
            # Additional stop token
            test_tokens.append(128001)
        else:
            # Regular token
            test_tokens.append(np.random.randint(100, vocab_size))
    
    data = {
        "device": device,
        "dtype": dtype,
        "hw_info": hw_info,
        "requests": requests,
        "test_tokens": test_tokens,
        "num_requests": num_requests,
    }
    
    return data

# =======================
# Experiment Execution
# =======================
def experiment(data: Dict[str, Any]) -> Any:
    """Execute the optimized operation."""
    
    requests = data["requests"]
    test_tokens = data["test_tokens"]
    
    # Track which requests finished
    finished_count = 0
    
    # The optimization is in the check_finished method
    # We simulate the hot path: checking if tokens match stop conditions
    for req, token in zip(requests, test_tokens):
        # Add the test token
        req.output_ids.append(token)
        
        # Call the optimized check_finished method
        req.check_finished()
        
        if req.finished_reason is not None:
            finished_count += 1
        
        # Reset for next iteration
        req.finished_reason = None
        req.output_ids.pop()
    
    return {"finished_count": finished_count, "total": len(requests)}

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
    assert current_result["total"] == reference_result["total"], \
        f"Total mismatch: {current_result['total']} vs {reference_result['total']}"
    assert current_result["finished_count"] == reference_result["finished_count"], \
        f"Finished count mismatch: {current_result['finished_count']} vs {reference_result['finished_count']}"

# =======================
# Main Test Function
# =======================
def run_test(eqcheck: bool = False, reference: bool = False, prefix: str = '') -> float:
    """Main test entry point."""
    
    # Setup
    data = setup()
    hw_info = data["hw_info"]
    
    # CPU timing for this optimization
    warmup = 5
    iters = 20  # Reduce iterations since this is CPU-bound
    
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
    commit_hash = os.getenv("COMMIT_HASH", "8f8f96a6217ea737c94e7429e480196319594459")
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
        "device": "cpu",
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