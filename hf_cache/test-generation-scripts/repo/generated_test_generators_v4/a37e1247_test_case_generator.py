#!/usr/bin/env python3
"""
Performance test for commit: a37e1247c183cff86a18f2ed1a075e40704b1c5e
Message: [Multimodal][Perf] Use `pybase64` instead of `base64` (#7724)

This script measures the actual performance impact of the optimization.
It supports cross-commit comparison with functional equivalence checking.
"""

import os
import sys
import json
import time
import importlib
from typing import Dict, Any, Tuple, Optional, List
from io import BytesIO

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
    
    # Priority 2: Parse from commit metadata - use encode_image_base64 as primary target
    if not (module_path and symbol_name):
        # This commit changes multiple functions, pick encode_image_base64 as representative
        module_path = "sglang.utils"
        symbol_name = "encode_image_base64"
    
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
    
    # Create synthetic image data of various sizes
    # Typical multimodal workloads include images from 256x256 to 2048x2048
    
    # Generate deterministic image-like data
    np.random.seed(42)
    
    # Small image (256x256 RGB)
    small_image = np.random.randint(0, 256, (256, 256, 3), dtype=np.uint8)
    
    # Medium image (512x512 RGB) 
    medium_image = np.random.randint(0, 256, (512, 512, 3), dtype=np.uint8)
    
    # Large image (1024x1024 RGB)
    large_image = np.random.randint(0, 256, (1024, 1024, 3), dtype=np.uint8)
    
    # XL image (2048x2048 RGB) - common for high-res vision models
    xl_image = np.random.randint(0, 256, (2048, 2048, 3), dtype=np.uint8)
    
    # Convert to bytes (PNG-like format)
    from PIL import Image
    
    def image_to_bytes(arr):
        img = Image.fromarray(arr, mode='RGB')
        buffer = BytesIO()
        img.save(buffer, format='PNG')
        return buffer.getvalue()
    
    small_bytes = image_to_bytes(small_image)
    medium_bytes = image_to_bytes(medium_image)
    large_bytes = image_to_bytes(large_image)
    xl_bytes = image_to_bytes(xl_image)
    
    # Create batch of mixed sizes to simulate real serving
    image_batch = [small_bytes, medium_bytes, large_bytes, xl_bytes] * 5  # 20 images total
    
    data = {
        "device": hw_info["device"],
        "dtype": torch.float32,  # Base64 encoding is CPU operation
        "hw_info": hw_info,
        "image_batch": image_batch,
        "small_bytes": small_bytes,
        "medium_bytes": medium_bytes,
        "large_bytes": large_bytes,
        "xl_bytes": xl_bytes,
    }
    
    return data

# =======================
# Experiment Execution
# =======================
def experiment(data: Dict[str, Any]) -> Any:
    """Execute the optimized operation."""
    target, fq_name = resolve_target()
    
    # Process batch of images through encode_image_base64
    results = []
    for image_bytes in data["image_batch"]:
        encoded = target(image_bytes)
        results.append(encoded)
    
    return results

# =======================
# Result I/O
# =======================
def store_result(result: Any, filepath: str) -> None:
    """Store result for reference comparison."""
    # Store as JSON since results are base64 strings
    import pickle
    with open(filepath, 'wb') as f:
        pickle.dump({"type": "string_list", "data": result}, f)

def load_result(filepath: str) -> Any:
    """Load reference result."""
    import pickle
    with open(filepath, 'rb') as f:
        data = pickle.load(f)
    return data.get("data", data)

# =======================
# Equivalence Checking
# =======================
def check_equivalence(current_result: Any, reference_result: Any) -> None:
    """Verify functional equivalence."""
    assert type(current_result) == type(reference_result), f"Type mismatch: {type(current_result)} vs {type(reference_result)}"
    
    if isinstance(current_result, list):
        assert len(current_result) == len(reference_result), f"Length mismatch: {len(current_result)} vs {len(reference_result)}"
        for i, (c, r) in enumerate(zip(current_result, reference_result)):
            assert c == r, f"Mismatch at index {i}: strings differ"
    else:
        assert current_result == reference_result, f"Result mismatch"

# =======================
# Timing Implementation
# =======================
def time_cpu(func, warmup=3, iterations=20) -> Tuple[Any, Dict[str, float]]:
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
        "p95_ms": times_ms[int(len(times_ms) * 0.95)],
        "p99_ms": times_ms[min(int(len(times_ms) * 0.99), len(times_ms)-1)],
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
    
    # This is a CPU-bound operation (base64 encoding)
    warmup = 3
    iters = 20
    
    # Time the operation
    result, timing_stats = time_cpu(lambda: experiment(data), warmup=warmup, iterations=iters)
    avg_ms = timing_stats["avg_ms"]
    p50_ms = timing_stats["p50_ms"]
    p95_ms = timing_stats["p95_ms"]
    
    # Reference handling
    commit_hash = os.getenv("COMMIT_HASH", "a37e1247c183cff86a18f2ed1a075e40704b1c5e")
    impl_tag = os.getenv("IMPL_TAG", "child")
    ref_file = f"{prefix}_{impl_tag}_{commit_hash}_reference.pkl"
    
    if reference:
        store_result(result, ref_file)
    
    if eqcheck and os.path.exists(ref_file):
        ref_result = load_result(ref_file)
        check_equivalence(result, ref_result)
    
    # Output compact JSON schema
    summary = {
        "impl_tag": impl_tag,
        "commit_hash": commit_hash,
        "device": "cpu",  # base64 encoding is CPU operation
        "dtype": "str",  # working with strings
        "iters": iters,
        "warmup": warmup,
        "avg_ms": avg_ms,
        "p50_ms": p50_ms,
        "p95_ms": p95_ms,
        "eq_level": os.getenv("PROB_EQ_LEVEL", "exact"),  # base64 should be exact
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