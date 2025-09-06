#!/usr/bin/env python3
"""
Performance test for commit: 319ad7f1d386699e94f629341c9988a926821f24
Message: [CI/Build][Misc] Add CI that benchmarks vllm performance on those PRs with `perf-benchmarks` label (#5073)

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
    
    # Priority 2: Parse from commit metadata - target the load_format functionality
    if not (module_path and symbol_name):
        # The commit adds load_format parameter to benchmark scripts
        # We'll test the vLLM LLM class which uses this parameter
        module_path = "vllm"
        symbol_name = "LLM"
    
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
    
    # Test the load_format parameter functionality added in this commit
    # We'll create a minimal LLM instance with dummy weights
    
    device = torch.device(hw_info["device"])
    dtype = torch.float16 if hw_info["device"] == "cuda" else torch.float32
    
    # Prepare test parameters for LLM initialization
    # The commit adds support for load_format parameter
    data = {
        "device": device,
        "dtype": dtype,
        "hw_info": hw_info,
        "model": "facebook/opt-125m",  # Small model for testing
        "load_format": "dummy",  # New parameter added in this commit
        "tensor_parallel_size": 1,
        "gpu_memory_utilization": 0.5,  # Use less memory for testing
        "max_model_len": 512,  # Limit context length
        "enforce_eager": True,  # Disable CUDA graphs for consistency
    }
    
    # Create dummy input for generation
    batch_size = 4
    input_len = 32
    output_len = 16
    
    # Generate random prompts
    dummy_prompts = ["Test prompt " * 4 for _ in range(batch_size)]
    
    data["prompts"] = dummy_prompts
    data["sampling_params"] = {
        "temperature": 0.0,
        "max_tokens": output_len,
        "ignore_eos": True,
    }
    
    return data

# =======================
# Experiment Execution
# =======================
def experiment(data: Dict[str, Any]) -> Any:
    """Execute the optimized operation."""
    target, fq_name = resolve_target()
    
    # Import SamplingParams as well
    from vllm import SamplingParams
    
    # Create LLM instance with the new load_format parameter
    llm_kwargs = {
        "model": data["model"],
        "load_format": data["load_format"],  # Key parameter added in commit
        "tensor_parallel_size": data["tensor_parallel_size"],
        "gpu_memory_utilization": data["gpu_memory_utilization"],
        "max_model_len": data["max_model_len"],
        "enforce_eager": data["enforce_eager"],
        "dtype": str(data["dtype"]).replace("torch.", ""),
    }
    
    # Filter out CPU-specific adjustments
    if data["hw_info"]["device"] == "cpu":
        llm_kwargs["device"] = "cpu"
        llm_kwargs.pop("gpu_memory_utilization", None)
    
    # Initialize LLM with load_format
    llm = target(**llm_kwargs)
    
    # Create sampling parameters
    sampling_params = SamplingParams(**data["sampling_params"])
    
    # Run generation (this tests that load_format works correctly)
    with torch.no_grad():
        outputs = llm.generate(data["prompts"], sampling_params, use_tqdm=False)
    
    # Extract generated tokens for validation
    result = {
        "num_outputs": len(outputs),
        "output_lengths": [len(output.outputs[0].token_ids) for output in outputs],
        "model_config": llm_kwargs,
    }
    
    # Clean up
    del llm
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
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
    # Check that both results have the same structure
    assert isinstance(current_result, dict), "Current result must be a dict"
    assert isinstance(reference_result, dict), "Reference result must be a dict"
    
    # Check number of outputs
    assert current_result["num_outputs"] == reference_result["num_outputs"], \
        f"Number of outputs mismatch: {current_result['num_outputs']} vs {reference_result['num_outputs']}"
    
    # Check output lengths (should be consistent with max_tokens)
    assert len(current_result["output_lengths"]) == len(reference_result["output_lengths"]), \
        "Output length array size mismatch"
    
    # The actual token counts might vary slightly due to EOS handling
    # but should be within reasonable bounds
    for i, (curr_len, ref_len) in enumerate(zip(current_result["output_lengths"], 
                                                  reference_result["output_lengths"])):
        assert abs(curr_len - ref_len) <= 2, \
            f"Output length mismatch at index {i}: {curr_len} vs {ref_len}"
    
    # Verify model configuration matches
    assert current_result["model_config"]["load_format"] == reference_result["model_config"]["load_format"], \
        "load_format parameter mismatch"

# =======================
# Timing Implementation
# =======================
def time_gpu(func, warmup=5, iterations=50) -> Tuple[Any, Dict[str, float]]:
    """Time GPU operations with CUDA events."""
    # Warmup
    for _ in range(warmup):
        _ = func()
        torch.cuda.synchronize()
    
    # Clear cache before timing
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
    
    # For this infrastructure change, we're testing the load_format functionality
    # Rather than timing the full generation, we'll time the model initialization
    # with the new load_format parameter
    
    def init_and_generate():
        """Initialize model and run a quick generation."""
        from vllm import LLM, SamplingParams
        
        llm_kwargs = {
            "model": data["model"],
            "load_format": data["load_format"],
            "tensor_parallel_size": data["tensor_parallel_size"],
            "max_model_len": data["max_model_len"],
            "enforce_eager": data["enforce_eager"],
            "dtype": str(data["dtype"]).replace("torch.", ""),
        }
        
        if hw_info["device"] == "cuda":
            llm_kwargs["gpu_memory_utilization"] = data["gpu_memory_utilization"]
        else:
            llm_kwargs["device"] = "cpu"
        
        llm = LLM(**llm_kwargs)
        sampling_params = SamplingParams(**data["sampling_params"])
        
        # Single generation to validate functionality
        outputs = llm.generate(data["prompts"][:1], sampling_params, use_tqdm=False)
        
        result = {
            "num_outputs": len(outputs),
            "output_lengths": [len(output.outputs[0].token_ids) for output in outputs],
            "model_config": llm_kwargs,
        }
        
        del llm
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        return result
    
    # Timing
    if hw_info["device"] == "cuda":
        # For GPU, we'll do fewer iterations since model init is expensive
        warmup = 1
        iters = 3
        
        # Warmup
        for _ in range(warmup):
            _ = init_and_generate()
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
        
        # Timing
        times = []
        for _ in range(iters):
            torch.cuda.synchronize()
            start = time.perf_counter()
            result = init_and_generate()
            torch.cuda.synchronize()
            times.append((time.perf_counter() - start) * 1000)
        
        times.sort()
        avg_ms = sum(times) / len(times)
        p50_ms = times[len(times) // 2]
        p95_ms = times[-1]  # With few iterations, use max as p95
    else:
        warmup = 1
        iters = 2
        
        # Warmup
        for _ in range(warmup):
            _ = init_and_generate()
        
        # Timing
        times = []
        for _ in range(iters):
            start = time.perf_counter()
            result = init_and_generate()
            times.append((time.perf_counter() - start) * 1000)
        
        times.sort()
        avg_ms = sum(times) / len(times)
        p50_ms = times[len(times) // 2]
        p95_ms = times[-1]
    
    # Reference handling
    commit_hash = os.getenv("COMMIT_HASH", "319ad7f1d386699e94f629341c9988a926821f24")
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
        "opt_path_hit": True  # load_format parameter is being used
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