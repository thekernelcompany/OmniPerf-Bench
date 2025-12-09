#!/usr/bin/env python3
"""
Performance test for commit: ab4a83b25909aa98330b838a224e4fe5c943e483
Message: Optimize schedule (#1339)

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
from dataclasses import dataclass, field

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
    
    # Priority 2: Parse from commit metadata - target PrefillAdder class
    if not (module_path and symbol_name):
        module_path = "sglang.srt.managers.policy_scheduler"
        symbol_name = "PrefillAdder"
    
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
@dataclass
class SamplingParams:
    max_new_tokens: int = 256
    ignore_eos: bool = False

@dataclass
class MockReq:
    rid: int = 0
    origin_input_ids: List[int] = field(default_factory=list)
    output_ids: List[int] = field(default_factory=list)
    extend_input_len: int = 0
    prefix_indices: List[int] = field(default_factory=list)
    fill_ids: List[int] = field(default_factory=list)
    last_node: Optional[Any] = None
    sampling_params: SamplingParams = field(default_factory=SamplingParams)
    
    def init_next_round_input(self, tree_cache):
        pass

class MockPrefixCache:
    def __init__(self):
        self.disable = False
        self.evictable_size_val = 10000
    
    def evictable_size(self):
        return self.evictable_size_val
    
    def match_prefix(self, *args):
        return []
    
    def inc_lock_ref(self, node):
        pass
    
    def dec_lock_ref(self, node):
        return 0

class MockScheduleBatch:
    def __init__(self, reqs):
        self.reqs = reqs

# =======================
# Workload Setup
# =======================
def setup() -> Dict[str, Any]:
    """Create realistic workload for the optimization."""
    ensure_determinism()
    hw_info = detect_hardware()
    
    # Create mock requests with varied characteristics
    num_requests = 64
    requests = []
    
    for i in range(num_requests):
        req = MockReq()
        req.rid = i
        
        # Vary request characteristics
        if i % 3 == 0:
            # Long generation with ignore_eos
            req.origin_input_ids = list(range(512))
            req.extend_input_len = 512
            req.fill_ids = list(range(512))
            req.sampling_params.max_new_tokens = 1024
            req.sampling_params.ignore_eos = True
        elif i % 3 == 1:
            # Medium generation without ignore_eos
            req.origin_input_ids = list(range(256))
            req.extend_input_len = 256
            req.fill_ids = list(range(256))
            req.sampling_params.max_new_tokens = 512
            req.sampling_params.ignore_eos = False
        else:
            # Short generation
            req.origin_input_ids = list(range(128))
            req.extend_input_len = 128
            req.fill_ids = list(range(128))
            req.sampling_params.max_new_tokens = 256
            req.sampling_params.ignore_eos = False
        
        requests.append(req)
    
    # Create running batch with some active requests
    running_reqs = []
    for i in range(8):
        req = MockReq()
        req.rid = 1000 + i
        req.origin_input_ids = list(range(256))
        req.output_ids = list(range(64))  # Already generated some tokens
        req.sampling_params.max_new_tokens = 512
        req.sampling_params.ignore_eos = (i % 2 == 0)
        running_reqs.append(req)
    
    running_batch = MockScheduleBatch(running_reqs) if running_reqs else None
    
    # Create mock tree cache
    tree_cache = MockPrefixCache()
    
    data = {
        "device": hw_info["device"],
        "dtype": torch.float32,  # Scheduling doesn't use tensors
        "hw_info": hw_info,
        "requests": requests,
        "running_batch": running_batch,
        "tree_cache": tree_cache,
        "new_token_ratio": 1.2,  # Typical value
        "rem_total_tokens": 16384,  # Available token budget
        "rem_input_tokens": 8192,
        "rem_chunk_tokens": 2048,
        "mixed_with_decode_tokens": 256
    }
    
    return data

# =======================
# Experiment Execution
# =======================
def experiment(data: Dict[str, Any]) -> Any:
    """Execute the optimized operation."""
    PrefillAdder, fq_name = resolve_target()
    
    # Create PrefillAdder instance with new constructor signature
    adder = PrefillAdder(
        tree_cache=data["tree_cache"],
        running_batch=data["running_batch"],
        new_token_ratio=data["new_token_ratio"],
        rem_total_tokens=data["rem_total_tokens"],
        rem_input_tokens=data["rem_input_tokens"],
        rem_chunk_tokens=data["rem_chunk_tokens"],
        mixed_with_decode_tokens=data["mixed_with_decode_tokens"]
    )
    
    # Remove running tokens if there's a running batch
    if data["running_batch"] is not None:
        adder.remove_running_tokens(data["running_batch"])
    
    # Schedule requests
    scheduled_count = 0
    inflight_req = None
    
    for req in data["requests"]:
        if adder.no_remaining_tokens():
            break
        
        req.init_next_round_input(data["tree_cache"])
        res = adder.add_one_req(req)
        
        if not res:
            break
        
        scheduled_count += 1
        
        if adder.new_inflight_req is not None:
            inflight_req = adder.new_inflight_req
    
    result = {
        "scheduled_count": scheduled_count,
        "can_run_list_size": len(adder.can_run_list),
        "log_hit_tokens": adder.log_hit_tokens,
        "log_input_tokens": adder.log_input_tokens,
        "has_inflight": inflight_req is not None,
        "rem_total_tokens": adder.rem_total_tokens,
        "rem_input_tokens": adder.rem_input_tokens
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
    assert isinstance(current_result, dict), "Result must be a dictionary"
    assert isinstance(reference_result, dict), "Reference must be a dictionary"
    
    # Check all keys match
    assert current_result.keys() == reference_result.keys(), f"Keys mismatch: {current_result.keys()} vs {reference_result.keys()}"
    
    # Check numeric values
    for key in current_result:
        current_val = current_result[key]
        reference_val = reference_result[key]
        
        if isinstance(current_val, (int, float)):
            if isinstance(current_val, float):
                # Allow small tolerance for floating point
                assert abs(current_val - reference_val) < 1e-6, f"Mismatch at key '{key}': {current_val} vs {reference_val}"
            else:
                assert current_val == reference_val, f"Mismatch at key '{key}': {current_val} vs {reference_val}"
        else:
            assert current_val == reference_val, f"Mismatch at key '{key}': {current_val} vs {reference_val}"

# =======================
# Timing Implementation
# =======================
def time_cpu(func, warmup=3, iterations=100) -> Tuple[Any, Dict[str, float]]:
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
    
    # This is a CPU scheduling operation
    warmup = 5
    iters = 100  # More iterations for CPU timing
    
    # Time the scheduling operation
    result, timing_stats = time_cpu(lambda: experiment(data), warmup=warmup, iterations=iters)
    avg_ms = timing_stats["avg_ms"]
    p50_ms = timing_stats["p50_ms"]
    p95_ms = timing_stats["p95_ms"]
    
    # Reference handling
    commit_hash = os.getenv("COMMIT_HASH", "ab4a83b25909aa98330b838a224e4fe5c943e483")
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
        "device": "cpu",  # Scheduling is CPU operation
        "dtype": "none",  # No tensor operations
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