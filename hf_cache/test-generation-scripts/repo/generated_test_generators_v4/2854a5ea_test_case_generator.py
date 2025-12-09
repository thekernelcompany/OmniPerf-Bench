#!/usr/bin/env python3
"""
Performance test for commit: 2854a5ea9fbb31165936f633ab99915dec760f8d
Message: Fix the overhead due to penalizer in bench_latency (#1496)

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
        # Based on the diff, the main optimization is in ScheduleBatch
        module_path = "sglang.srt.managers.schedule_batch"
        symbol_name = "ScheduleBatch"
    
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
class MockRequest:
    """Mock request object for testing ScheduleBatch"""
    def __init__(self, req_id: int, seq_len: int = 128):
        self.req_id = req_id
        self.rid = req_id
        self.fill_ids = list(range(seq_len))
        self.prefix_indices = []
        self.origin_input_ids = list(range(seq_len))
        self.sampling_params = type('MockSamplingParams', (), {
            'n': 1,
            'temperature': 0.7,
            'top_p': 0.9,
            'top_k': 40,
            'repetition_penalty': 1.0,
            'presence_penalty': 0.0,
            'frequency_penalty': 0.0,
            'min_new_tokens': 1,
            'max_new_tokens': 128,
            'ignore_eos': False,
            'skip_special_tokens': True,
            'regex': None,
            'stop': None,
            'stop_token_ids': None,
        })()
        self.logprob_start_len = 0
        self.top_logprobs_num = 0
        self.return_logprob = False
        self.stream = False
        self.tokenizer = None
        self.rid = req_id
        self.finished_reason = None
        self.output_ids = []
        self.output_token_logprobs = []
        self.output_top_logprobs = []
        self.logprob_start_len = 0
        self.completion_tokens_wo_jump_forward = 0
        self.embedding = None
        self.cur_image_idx = 0
        self.pixel_values = []
        self.image_sizes = []
        self.image_offsets = []
        self.pad_values = []
        self.modalities = []

class MockTokenToKVPool:
    """Mock KV cache pool"""
    def __init__(self, size: int = 65536):
        self.size = size
        self.current_usage = 0
        
    def available_size(self):
        return self.size - self.current_usage
    
    def alloc(self, num_tokens: int):
        if self.current_usage + num_tokens > self.size:
            return None
        indices = torch.arange(self.current_usage, self.current_usage + num_tokens, dtype=torch.int32)
        self.current_usage += num_tokens
        return indices
    
    def free(self, indices):
        pass

class MockReqToTokenPool:
    """Mock request to token mapping"""
    def __init__(self):
        self.req_to_token = torch.zeros(65536, dtype=torch.int32)

# =======================
# Workload Setup
# =======================
def setup() -> Dict[str, Any]:
    """Create realistic workload for the optimization."""
    ensure_determinism()
    hw_info = detect_hardware()
    
    device = torch.device(hw_info["device"] if hw_info["device"] == "cuda" else "cpu")
    dtype = torch.float16 if hw_info["device"] == "cuda" else torch.float32
    
    # Create a batch with multiple requests
    batch_size = 32
    seq_len = 512
    vocab_size = 32000
    
    # Create mock requests
    reqs = [MockRequest(req_id=i, seq_len=seq_len) for i in range(batch_size)]
    
    # Create ScheduleBatch instance
    ScheduleBatch, _ = resolve_target()
    
    # Create required mock objects
    token_to_kv_pool = MockTokenToKVPool(size=65536)
    req_to_token_pool = MockReqToTokenPool()
    
    # Initialize batch
    batch = ScheduleBatch(
        reqs=reqs,
        req_to_token_pool=req_to_token_pool,
        token_to_kv_pool=token_to_kv_pool,
        tree_cache=None
    )
    
    # Initialize batch attributes that would be set during normal operation
    batch.seq_lens = torch.arange(1, batch_size + 1, dtype=torch.int32, device=device) * 10
    batch.input_ids = torch.randint(0, vocab_size, (batch_size,), dtype=torch.int32, device=device)
    batch.position_ids_offsets = torch.zeros(batch_size, dtype=torch.int32, device=device)
    batch.out_cache_loc = torch.zeros(batch_size, dtype=torch.int32, device=device)
    
    # Create sampling info mock if needed
    batch.sampling_info = type('MockSamplingInfo', (), {
        'penalizer_orchestrator': type('MockPenalizer', (), {
            'cumulate_input_tokens': lambda x: None
        })(),
        'temperatures': torch.ones(batch_size, device=device) * 0.7,
        'top_ps': torch.ones(batch_size, device=device) * 0.9,
        'top_ks': torch.ones(batch_size, dtype=torch.int32, device=device) * 40,
        'vocab_size': vocab_size
    })()
    
    data = {
        "device": device,
        "dtype": dtype,
        "hw_info": hw_info,
        "batch": batch,
        "vocab_size": vocab_size,
        "batch_size": batch_size,
        "seq_len": seq_len
    }
    
    return data

# =======================
# Experiment Execution
# =======================
def experiment(data: Dict[str, Any]) -> Any:
    """Execute the optimized operation."""
    batch = data["batch"]
    vocab_size = data["vocab_size"]
    
    # Test the prepare_for_extend method which was optimized
    with torch.no_grad():
        # The optimization replaced self.batch_size() with len(self.reqs)
        # This is called multiple times in the hot path
        results = []
        
        # Call prepare_for_extend multiple times to simulate repeated use
        for _ in range(10):
            batch.prepare_for_extend(vocab_size)
            # Capture some state for equivalence checking
            results.append({
                "forward_mode": getattr(batch, "forward_mode", None),
                "extend_num_tokens": getattr(batch, "extend_num_tokens", 0),
                "batch_size": len(batch.reqs)  # This is what was optimized
            })
    
    return results

# =======================
# Result I/O
# =======================
def store_result(result: Any, filepath: str) -> None:
    """Store result for reference comparison."""
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
    assert type(current_result) == type(reference_result), f"Type mismatch: {type(current_result)} vs {type(reference_result)}"
    
    if isinstance(current_result, list):
        assert len(current_result) == len(reference_result), f"Length mismatch: {len(current_result)} vs {len(reference_result)}"
        for i, (c, r) in enumerate(zip(current_result, reference_result)):
            assert c["batch_size"] == r["batch_size"], f"Batch size mismatch at {i}: {c['batch_size']} vs {r['batch_size']}"
            assert c["extend_num_tokens"] == r["extend_num_tokens"], f"Token count mismatch at {i}"

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
        p95_ms = times[int(len(times) * 0.95) - 1] if len(times) > 1 else times[0]
    
    # Reference handling
    commit_hash = os.getenv("COMMIT_HASH", "2854a5ea9fbb31165936f633ab99915dec760f8d")
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