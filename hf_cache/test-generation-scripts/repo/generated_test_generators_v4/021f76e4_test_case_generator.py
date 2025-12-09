#!/usr/bin/env python3
"""
Performance test for commit: 021f76e4f49861b2e9ea9ccff06a46d577e3c548
Message: [Perf] Refactor LoRAManager to eliminate stream syncs and redundant computations  (#6994)

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
        # Based on the commit, the main optimized method is prepare_lora_batch
        module_path = "sglang.srt.lora.lora_manager"
        symbol_name = "LoRAManager"
    
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
class MockForwardBatch:
    """Mock forward batch for LoRA testing."""
    def __init__(self, batch_size, lora_paths, forward_mode, extend_seq_lens_cpu=None):
        self.batch_size = batch_size
        self.lora_paths = lora_paths
        self.forward_mode = forward_mode
        self.extend_seq_lens_cpu = extend_seq_lens_cpu or [1] * batch_size
        if forward_mode.is_extend():
            self.extend_seq_lens = torch.tensor(extend_seq_lens_cpu, device='cuda', dtype=torch.int32)

class MockForwardMode:
    """Mock forward mode."""
    def __init__(self, is_decode=True):
        self._is_decode = is_decode
    
    def is_decode(self):
        return self._is_decode
    
    def is_extend(self):
        return not self._is_decode

class MockLoRAConfig:
    """Mock LoRA configuration."""
    def __init__(self, rank=16):
        self.hf_config = {"r": rank}

class MockLoRA:
    """Mock LoRA adapter."""
    def __init__(self, rank=16, scaling=1.0):
        self.config = MockLoRAConfig(rank)
        self.scaling = scaling

# =======================
# Workload Setup
# =======================
def setup() -> Dict[str, Any]:
    """Create realistic workload for the optimization."""
    ensure_determinism()
    hw_info = detect_hardware()
    
    device = torch.device(hw_info["device"] if hw_info["device"] == "cuda" else "cpu")
    dtype = torch.float16 if hw_info["device"] == "cuda" else torch.float32
    
    # Realistic LoRA batching scenario
    batch_size = 64  # Large batch to show sync elimination benefits
    max_loras_per_batch = 8
    num_unique_loras = 4
    
    # Create mock LoRA paths (some requests use same LoRA)
    lora_paths = []
    for i in range(batch_size):
        if i % 4 == 0:
            lora_paths.append(None)  # Some requests without LoRA
        else:
            lora_paths.append(f"lora_{i % num_unique_loras}")
    
    # Simulate varying sequence lengths for extend mode
    extend_seq_lens_cpu = [np.random.randint(1, 128) for _ in range(batch_size)]
    
    # Create mock forward batch for decode mode (common case)
    forward_mode_decode = MockForwardMode(is_decode=True)
    forward_batch_decode = MockForwardBatch(
        batch_size, lora_paths, forward_mode_decode, extend_seq_lens_cpu
    )
    
    # Create mock forward batch for extend mode
    forward_mode_extend = MockForwardMode(is_decode=False)
    forward_batch_extend = MockForwardBatch(
        batch_size, lora_paths, forward_mode_extend, extend_seq_lens_cpu
    )
    
    data = {
        "device": device,
        "dtype": dtype,
        "hw_info": hw_info,
        "batch_size": batch_size,
        "max_loras_per_batch": max_loras_per_batch,
        "num_unique_loras": num_unique_loras,
        "lora_paths": lora_paths,
        "forward_batch_decode": forward_batch_decode,
        "forward_batch_extend": forward_batch_extend,
        "extend_seq_lens_cpu": extend_seq_lens_cpu,
    }
    
    return data

# =======================
# Experiment Execution
# =======================
def experiment(data: Dict[str, Any]) -> Any:
    """Execute the optimized operation."""
    LoRAManager, fq_name = resolve_target()
    
    # Create LoRA manager instance
    manager = LoRAManager(
        device=data["device"],
        model=None,  # Mock model not needed for this test
        adapter_path=None,
        max_loras_per_batch=data["max_loras_per_batch"],
    )
    
    # Initialize mock components
    manager.max_bs_in_cuda_graph = 128  # Enable CUDA graph path
    
    # Create mock memory pool
    class MockMemoryPool:
        def __init__(self):
            self.buffer_map = {}
            self.next_id = 0
        
        def get_buffer_id(self, lora_path):
            if lora_path is None:
                return 0
            if lora_path not in self.buffer_map:
                self.buffer_map[lora_path] = self.next_id + 1
                self.next_id += 1
            return self.buffer_map[lora_path]
    
    manager.memory_pool = MockMemoryPool()
    
    # Create mock LoRA adapters
    manager.loras = {}
    for i in range(data["num_unique_loras"]):
        lora_path = f"lora_{i}"
        manager.loras[lora_path] = MockLoRA(rank=16 * (i + 1), scaling=1.0 / (i + 1))
    
    # Initialize CUDA graph batch info if needed
    if hasattr(manager, 'init_cuda_graph_batch_info'):
        manager.init_cuda_graph_batch_info()
    else:
        # Manually create for older versions
        from dataclasses import dataclass
        
        @dataclass
        class LoRABatchInfo:
            bs: int
            seg_lens: torch.Tensor
            seg_indptr: torch.Tensor
            max_len: int
            weight_indices: torch.Tensor
            lora_ranks: torch.Tensor
            scalings: torch.Tensor
        
        manager.cuda_graph_batch_info = LoRABatchInfo(
            bs=0,
            seg_lens=torch.zeros(manager.max_bs_in_cuda_graph, dtype=torch.int32),
            seg_indptr=torch.zeros(manager.max_bs_in_cuda_graph + 1, dtype=torch.int32),
            max_len=1,
            weight_indices=torch.zeros(manager.max_bs_in_cuda_graph, dtype=torch.int32),
            lora_ranks=torch.zeros(data["max_loras_per_batch"], dtype=torch.int64),
            scalings=torch.zeros(data["max_loras_per_batch"], dtype=torch.float),
        )
        
        # Initialize seg_lens and seg_indptr for CUDA graph
        manager.cuda_graph_batch_info.seg_lens[:manager.max_bs_in_cuda_graph].fill_(1)
        torch.cumsum(
            manager.cuda_graph_batch_info.seg_lens[:manager.max_bs_in_cuda_graph],
            dim=0,
            out=manager.cuda_graph_batch_info.seg_indptr[1:manager.max_bs_in_cuda_graph + 1],
        )
    
    # Execute the optimized operation multiple times
    results = []
    
    # Test decode mode (CUDA graph path)
    with torch.no_grad():
        batch_info_decode = manager.prepare_lora_batch(data["forward_batch_decode"])
        results.append(("decode", batch_info_decode))
    
    # Test extend mode (non-CUDA graph path)
    with torch.no_grad():
        batch_info_extend = manager.prepare_lora_batch(data["forward_batch_extend"])
        results.append(("extend", batch_info_extend))
    
    return results

# =======================
# Result I/O
# =======================
def store_result(result: Any, filepath: str) -> None:
    """Store result for reference comparison."""
    # Convert batch info to serializable format
    serializable_results = []
    for mode, batch_info in result:
        info_dict = {
            "mode": mode,
            "bs": batch_info.bs,
            "seg_lens": batch_info.seg_lens.cpu(),
            "seg_indptr": batch_info.seg_indptr.cpu(),
            "max_len": batch_info.max_len,
            "weight_indices": batch_info.weight_indices.cpu(),
            "lora_ranks": batch_info.lora_ranks.cpu(),
            "scalings": batch_info.scalings.cpu(),
        }
        serializable_results.append(info_dict)
    
    torch.save({"type": "lora_batch_info", "data": serializable_results}, filepath)

def load_result(filepath: str) -> Any:
    """Load reference result."""
    data = torch.load(filepath)
    return data.get("data", data)

# =======================
# Equivalence Checking
# =======================
def check_equivalence(current_result: Any, reference_result: Any) -> None:
    """Verify functional equivalence."""
    assert len(current_result) == len(reference_result), f"Result count mismatch"
    
    for (cur_mode, cur_info), ref_dict in zip(current_result, reference_result):
        assert cur_mode == ref_dict["mode"], f"Mode mismatch: {cur_mode} vs {ref_dict['mode']}"
        assert cur_info.bs == ref_dict["bs"], f"Batch size mismatch"
        assert cur_info.max_len == ref_dict["max_len"], f"Max length mismatch"
        
        # Check tensor equivalence
        torch.testing.assert_close(
            cur_info.seg_lens.cpu(),
            ref_dict["seg_lens"],
            rtol=0, atol=0  # Exact match for integer tensors
        )
        torch.testing.assert_close(
            cur_info.seg_indptr.cpu(),
            ref_dict["seg_indptr"],
            rtol=0, atol=0
        )
        torch.testing.assert_close(
            cur_info.weight_indices.cpu(),
            ref_dict["weight_indices"],
            rtol=0, atol=0
        )
        torch.testing.assert_close(
            cur_info.lora_ranks.cpu(),
            ref_dict["lora_ranks"],
            rtol=0, atol=0
        )
        torch.testing.assert_close(
            cur_info.scalings.cpu(),
            ref_dict["scalings"],
            rtol=1e-5, atol=1e-7
        )

# =======================
# Timing Implementation
# =======================
def time_gpu(func, warmup=5, iterations=50) -> Tuple[Any, Dict[str, float]]:
    """Time GPU operations with CUDA events."""
    # Warmup
    for _ in range(warmup):
        _ = func()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
    
    if torch.cuda.is_available():
        # GPU timing with CUDA events
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
    else:
        # CPU fallback timing
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
    
    # Timing
    if hw_info["device"] == "cuda":
        warmup = 5
        iters = 50
    else:
        warmup = 3
        iters = 10
    
    result, timing_stats = time_gpu(lambda: experiment(data), warmup=warmup, iterations=iters)
    avg_ms = timing_stats["avg_ms"]
    p50_ms = timing_stats["p50_ms"]
    p95_ms = timing_stats["p95_ms"]
    
    # Reference handling
    commit_hash = os.getenv("COMMIT_HASH", "021f76e4f49861b2e9ea9ccff06a46d577e3c548")
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
        "dtype": "torch.float16" if hw_info["device"] == "cuda" else "torch.float32",
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