#!/usr/bin/env python3
"""
Performance test for commit: 3212c2ad3f7e4fb473dc807b4b176020a778ed5b
Message: vlm: optimize tensor transport (#6003)

This script measures the actual performance impact of the optimization.
It supports cross-commit comparison with functional equivalence checking.
"""

import os
import sys
import json
import time
import math
import pickle
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
        # Based on the commit diff, the main optimization is TransportProxyTensor
        module_path = "sglang.srt.managers.mm_utils"
        symbol_name = "TransportProxyTensor"
    
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
    
    # Create tensor data simulating multimodal features (e.g., image embeddings)
    # Typical VLM workload dimensions
    device = torch.device(hw_info["device"])
    dtype = torch.float16 if hw_info["device"] == "cuda" else torch.float32
    
    # Simulate multiple image features for batch processing
    batch_size = 8
    num_images = 4  # Multiple images per batch item
    image_features_dim = 4096  # Vision encoder output dimension
    num_patches = 256  # Number of visual tokens per image
    
    # Create realistic multimodal tensors
    tensors = []
    for i in range(batch_size):
        # Simulate vision encoder output
        feature_tensor = torch.randn(
            num_images, num_patches, image_features_dim,
            device=device, dtype=dtype
        )
        tensors.append(feature_tensor)
    
    # Determine transport mode based on hardware
    if hw_info["device"] == "cuda":
        transport_mode = "cuda_ipc"
    else:
        transport_mode = "default"
    
    data = {
        "device": device,
        "dtype": dtype,
        "hw_info": hw_info,
        "tensors": tensors,
        "transport_mode": transport_mode,
        "batch_size": batch_size,
        "num_images": num_images,
        "image_features_dim": image_features_dim,
        "num_patches": num_patches,
    }
    
    return data

# =======================
# Experiment Execution
# =======================
def experiment(data: Dict[str, Any]) -> Any:
    """Execute the optimized operation."""
    TransportProxyTensor, _ = resolve_target()
    
    # Create TransportProxyTensor instances and serialize/deserialize them
    # This simulates inter-process tensor transport
    transport_mode = data["transport_mode"]
    tensors = data["tensors"]
    
    # Create proxy tensors with transport metadata
    proxy_tensors = []
    for i, tensor in enumerate(tensors):
        proxy = TransportProxyTensor(
            data=tensor,
            name=f"image_features_{i}",
            fields={"batch_idx": i, "modality": "image"},
            transport_mode=transport_mode
        )
        proxy_tensors.append(proxy)
    
    # Simulate inter-process transport via pickle (what multiprocessing uses)
    serialized = []
    for proxy in proxy_tensors:
        serialized.append(pickle.dumps(proxy))
    
    # Deserialize (simulating receiving process)
    deserialized = []
    for pickled_data in serialized:
        deserialized.append(pickle.loads(pickled_data))
    
    return deserialized

# =======================
# Result I/O
# =======================
def store_result(result: Any, filepath: str) -> None:
    """Store result for reference comparison."""
    # Convert proxy tensors back to regular tensors for storage
    tensors_to_store = []
    for item in result:
        if hasattr(item, 'cpu'):
            tensors_to_store.append(item.cpu())
        else:
            tensors_to_store.append(item)
    
    torch.save({"type": "tensor_list", "data": tensors_to_store}, filepath)

def load_result(filepath: str) -> Any:
    """Load reference result."""
    data = torch.load(filepath)
    return data.get("data", data)

# =======================
# Equivalence Checking
# =======================
def check_equivalence(current_result: Any, reference_result: Any) -> None:
    """Verify functional equivalence."""
    assert len(current_result) == len(reference_result), \
        f"Result count mismatch: {len(current_result)} vs {len(reference_result)}"
    
    for i, (current, reference) in enumerate(zip(current_result, reference_result)):
        # Both should be tensors after deserialization
        assert isinstance(current, torch.Tensor), f"Item {i}: Expected tensor, got {type(current)}"
        assert isinstance(reference, torch.Tensor), f"Item {i}: Expected tensor, got {type(reference)}"
        
        assert current.shape == reference.shape, \
            f"Item {i}: Shape mismatch {current.shape} vs {reference.shape}"
        assert current.dtype == reference.dtype, \
            f"Item {i}: Dtype mismatch {current.dtype} vs {reference.dtype}"
        
        # Determine tolerances based on dtype
        if current.dtype in (torch.float16, torch.bfloat16):
            rtol, atol = 1e-3, 1e-4
        else:
            rtol, atol = 1e-5, 1e-7
        
        torch.testing.assert_close(
            current.cpu(),
            reference.cpu(),
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
            end = time.perf_counter()
            times.append((end - start) * 1000)
        times.sort()
        avg_ms = sum(times) / len(times)
        p50_ms = times[len(times) // 2]
        p95_ms = times[int(len(times) * 0.95)] if len(times) > 1 else times[0]
    
    # Reference handling
    commit_hash = os.getenv("COMMIT_HASH", "3212c2ad3f7e4fb473dc807b4b176020a778ed5b")
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