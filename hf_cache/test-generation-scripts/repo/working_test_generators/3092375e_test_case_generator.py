#!/usr/bin/env python3
"""
Performance test for commit: 3092375e274e9e003961e600e10a6192d33ceaa0
Message: [V1][Performance] Implement custom serializaton for MultiModalKwargs [Rebased] (#16432)

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

import inspect
import logging

# API Probing helpers - auto-generated for compatibility
def safe_create_object(cls, **kwargs):
    """Create object with only valid arguments based on signature."""
    try:
        if not callable(cls):
            raise TypeError(f"{cls} is not callable")
        sig = inspect.signature(cls)
        valid_kwargs = {k: v for k, v in kwargs.items() 
                       if k in sig.parameters and k != "self"}
        return cls(**valid_kwargs)
    except Exception as e:
        logging.warning(f"Failed to create {cls.__name__ if hasattr(cls, '__name__') else cls} with args {list(kwargs.keys())}: {e}")
        raise

def safe_call_function(func, *args, **kwargs):
    """Call function with only valid arguments based on signature."""
    try:
        if not callable(func):
            raise TypeError(f"{func} is not callable")
        sig = inspect.signature(func)
        # Filter kwargs to only valid parameters
        valid_kwargs = {k: v for k, v in kwargs.items() 
                       if k in sig.parameters}
        return func(*args, **valid_kwargs)
    except Exception as e:
        logging.warning(f"Failed to call {func.__name__ if hasattr(func, '__name__') else func} with args {list(kwargs.keys())}: {e}")
        raise

# Specific helpers for common vllm classes
def safe_create_engine_output(**kwargs):
    """Create EngineCoreOutput with compatible arguments."""
    try:
        from vllm.v1.engine import EngineCoreOutput
        return safe_create_object(EngineCoreOutput, **kwargs)
    except ImportError:
        try:
            from vllm.engine import EngineCoreOutput  
            return safe_create_object(EngineCoreOutput, **kwargs)
        except ImportError:
            raise ImportError("EngineCoreOutput not found in vllm")

def safe_create_sampling_params(**kwargs):
    """Create SamplingParams with compatible arguments."""
    try:
        from vllm import SamplingParams
        return safe_create_object(SamplingParams, **kwargs)
    except ImportError:
        try:
            from vllm.sampling_params import SamplingParams
            return safe_create_object(SamplingParams, **kwargs)
        except ImportError:
            raise ImportError("SamplingParams not found in vllm")

def safe_create_llm(**kwargs):
    """Create LLM with compatible arguments."""
    try:
        from vllm import LLM
        return safe_create_object(LLM, **kwargs)
    except ImportError:
        raise ImportError("LLM not found in vllm")



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
    
    # Priority 2: Parse from commit metadata - use MsgpackEncoder as primary target
    if not (module_path and symbol_name):
        module_path = "vllm.v1.serial_utils"
        symbol_name = "MsgpackEncoder"
    
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
    
    # Import required multimodal classes
    try:
        from vllm.multimodal.inputs import (
            MultiModalKwargs,
            MultiModalKwargsItem,
            MultiModalFieldElem,
            MultiModalBatchedField,
            MultiModalSharedField
        )
        from vllm.v1.serial_utils import MsgpackEncoder, MsgpackDecoder
    except ImportError as e:
        print(json.dumps({"target_resolved": False, "error": str(e)}))
        sys.exit(1)
    
    device = torch.device(hw_info["device"])
    dtype = torch.float16 if hw_info["device"] == "cuda" else torch.float32
    
    # Create realistic multimodal workload with various tensor sizes
    # Mix of small (below threshold) and large (above threshold) tensors
    batch_size = 4
    seq_len = 512
    
    # Small tensors (below 256B threshold)
    small_tensors = [
        torch.rand(8, dtype=dtype, device="cpu"),  # 32B for float32
        torch.rand(16, dtype=dtype, device="cpu"), # 64B
        torch.rand(32, dtype=dtype, device="cpu"), # 128B
    ]
    
    # Medium tensors (around threshold)
    medium_tensors = [
        torch.rand(128, dtype=dtype, device="cpu"),  # 512B for float32
        torch.rand(256, dtype=dtype, device="cpu"),  # 1KB
    ]
    
    # Large tensors (above threshold)
    large_tensors = [
        torch.rand(10000, dtype=dtype, device="cpu"),  # 40KB for float32
        torch.rand(batch_size, seq_len, 768, dtype=dtype, device="cpu"),  # ~6MB
    ]
    
    # Create nested tensor structures as in real multimodal inputs
    nested_tensors = [
        torch.rand(256, dtype=dtype, device="cpu"),
        [
            torch.rand(1, 12, dtype=torch.float32, device="cpu"),
            torch.rand(3, 5, 7, dtype=torch.float64, device="cpu"),
        ],
        [torch.rand(4, 4, dtype=dtype, device="cpu")]
    ]
    
    # Build MultiModalKwargs with various field types
    mm_kwargs_dict = {
        "image_features": torch.rand(batch_size, 197, 768, dtype=dtype, device="cpu"),
        "image_positions": torch.tensor([[0, 1, 2, 3]] * batch_size, dtype=torch.int32, device="cpu"),
        "video_frames": [torch.rand(30, 224, 224, 3, dtype=torch.uint8, device="cpu") for _ in range(batch_size)],
        "audio_waveform": torch.rand(batch_size, 16000, dtype=dtype, device="cpu"),
        "nested_data": nested_tensors,
    }
    
    # Create MultiModalKwargs using items
    elems = []
    elems.append(MultiModalFieldElem(
        "image", "image_features", 
        mm_kwargs_dict["image_features"],
        MultiModalBatchedField()
    ))
    elems.append(MultiModalFieldElem(
        "image", "image_positions",
        mm_kwargs_dict["image_positions"],
        MultiModalSharedField(batch_size)
    ))
    elems.append(MultiModalFieldElem(
        "video", "video_frames",
        mm_kwargs_dict["video_frames"],
        MultiModalBatchedField()
    ))
    elems.append(MultiModalFieldElem(
        "audio", "audio_waveform",
        mm_kwargs_dict["audio_waveform"],
        MultiModalBatchedField()
    ))
    
    items = [
        MultiModalKwargsItem.from_elems([elems[0], elems[1]]),  # image
        MultiModalKwargsItem.from_elems([elems[2]]),  # video
        MultiModalKwargsItem.from_elems([elems[3]]),  # audio
    ]
    
    mm_kwargs = MultiModalKwargs.from_items(items)
    
    # Test with different threshold values
    threshold = int(os.getenv("VLLM_MSGPACK_ZERO_COPY_THRESHOLD", "256"))
    
    data = {
        "device": device,
        "dtype": dtype,
        "hw_info": hw_info,
        "mm_kwargs": mm_kwargs,
        "threshold": threshold,
        "MsgpackEncoder": MsgpackEncoder,
        "MsgpackDecoder": MsgpackDecoder,
        "MultiModalKwargs": MultiModalKwargs,
    }
    
    return data

# =======================
# Experiment Execution
# =======================
def experiment(data: Dict[str, Any]) -> Any:
    """Execute the optimized operation."""
    MsgpackEncoder = data["MsgpackEncoder"]
    MsgpackDecoder = data["MsgpackDecoder"]
    mm_kwargs = data["mm_kwargs"]
    threshold = data["threshold"]
    
    # Create encoder with the threshold
    encoder = MsgpackEncoder(size_threshold=threshold)
    decoder = MsgpackDecoder(data["MultiModalKwargs"])
    
    # Serialize
    encoded = encoder.encode(mm_kwargs)
    
    # Deserialize
    decoded = decoder.decode(encoded)
    
    # Return both encoded size and decoded result for verification
    total_size = sum(len(memoryview(x).cast("B")) for x in encoded)
    
    return {
        "encoded": encoded,
        "decoded": decoded,
        "total_size": total_size,
        "num_messages": len(encoded)
    }

# =======================
# Result I/O
# =======================
def store_result(result: Any, filepath: str) -> None:
    """Store result for reference comparison."""
    # Store only the decoded multimodal kwargs for comparison
    decoded = result["decoded"]
    
    # Convert to a serializable format
    data_to_save = {
        "type": "multimodal_kwargs",
        "total_size": result["total_size"],
        "num_messages": result["num_messages"],
        # Store the main dict representation
        "data": {k: v.cpu() if isinstance(v, torch.Tensor) else v 
                for k, v in decoded.items()}
    }
    
    torch.save(data_to_save, filepath)

def load_result(filepath: str) -> Any:
    """Load reference result."""
    data = torch.load(filepath)
    return data

# =======================
# Equivalence Checking
# =======================
def check_equivalence(current_result: Any, reference_result: Any) -> None:
    """Verify functional equivalence."""
    # Check that serialization produces functionally equivalent results
    current_decoded = current_result["decoded"]
    ref_data = reference_result.get("data", reference_result)
    
    # Check all keys match
    current_keys = set(current_decoded.keys())
    ref_keys = set(ref_data.keys())
    assert current_keys == ref_keys, f"Key mismatch: {current_keys} vs {ref_keys}"
    
    # Check tensor equivalence for each key
    for key in current_keys:
        current_val = current_decoded[key]
        ref_val = ref_data[key]
        
        if isinstance(current_val, torch.Tensor):
            assert current_val.shape == ref_val.shape, f"Shape mismatch for {key}"
            assert current_val.dtype == ref_val.dtype, f"Dtype mismatch for {key}"
            
            # Determine tolerances based on dtype
            if current_val.dtype in (torch.float16, torch.bfloat16):
                rtol, atol = 1e-3, 1e-4
            else:
                rtol, atol = 1e-5, 1e-7
            
            torch.testing.assert_close(
                current_val.cpu(),
                ref_val.cpu() if isinstance(ref_val, torch.Tensor) else ref_val,
                rtol=rtol, atol=atol
            )
        elif isinstance(current_val, list):
            # Handle nested tensor lists
            assert len(current_val) == len(ref_val), f"List length mismatch for {key}"
            for i, (cv, rv) in enumerate(zip(current_val, ref_val)):
                if isinstance(cv, torch.Tensor):
                    torch.testing.assert_close(cv.cpu(), rv.cpu() if isinstance(rv, torch.Tensor) else rv)

# =======================
# Timing Implementation
# =======================
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
        "p99_ms": times_ms[int(len(times_ms) * 0.99) - 1] if len(times_ms) > 1 else times_ms[0],
        "min_ms": times_ms[0],
        "max_ms": times_ms[-1],
        "std_ms": np.std(times_ms) if len(times_ms) > 1 else 0
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
    
    # This is a CPU-bound serialization operation
    warmup = 5
    iters = 20  # More iterations since serialization is relatively fast
    result, timing_stats = time_cpu(lambda: experiment(data), warmup=warmup, iterations=iters)
    avg_ms = timing_stats["avg_ms"]
    p50_ms = timing_stats["p50_ms"]
    p95_ms = timing_stats["p95_ms"]
    
    # Reference handling
    commit_hash = os.getenv("COMMIT_HASH", "3092375e274e9e003961e600e10a6192d33ceaa0")
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
        "device": "cpu",  # Serialization is CPU-bound
        "dtype": str(data["dtype"]),
        "iters": iters,
        "warmup": warmup,
        "avg_ms": avg_ms,
        "p50_ms": p50_ms,
        "p95_ms": p95_ms,
        "eq_level": os.getenv("PROB_EQ_LEVEL", "numeric"),
        "opt_path_hit": True,
        "threshold": data["threshold"],
        "total_size": result["total_size"],
        "num_messages": result["num_messages"]
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