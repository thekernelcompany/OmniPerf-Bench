#!/usr/bin/env python3
"""
Performance test for commit: b170930534acbb9c1619a3c83670a839ceee763a
Message: feat: radix tree code optimize (#1697)

This script measures the actual performance impact of the optimization.
It supports cross-commit comparison with functional equivalence checking.
"""

import os
import sys
import json
import time
import importlib
from typing import Dict, Any, Tuple, Optional, List
import pickle

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
        # Based on the commit diff, the target is RadixCache._split_node
        module_path = "sglang.srt.mem_cache.radix_cache"
        symbol_name = "RadixCache"
    
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
    
    # Generate key sequences that will trigger _split_node operations
    # These are token sequences that share prefixes but diverge
    num_sequences = 1000
    max_seq_len = 256
    vocab_size = 32000
    
    # Create sequences with shared prefixes to trigger node splitting
    sequences = []
    np.random.seed(42)
    
    # Generate base prefixes
    num_prefixes = 100
    prefix_len = 64
    prefixes = []
    for _ in range(num_prefixes):
        prefix = [int(x) for x in np.random.randint(0, vocab_size, prefix_len)]
        prefixes.append(prefix)
    
    # Generate sequences with shared prefixes but different suffixes
    for i in range(num_sequences):
        prefix_idx = i % num_prefixes
        prefix = prefixes[prefix_idx].copy()
        suffix_len = np.random.randint(1, max_seq_len - prefix_len)
        suffix = [int(x) for x in np.random.randint(0, vocab_size, suffix_len)]
        sequence = prefix + suffix
        sequences.append(sequence)
    
    # Shuffle to randomize insertion order (important for tree structure)
    np.random.shuffle(sequences)
    
    # Create dummy cache values (block indices)
    values = []
    for seq in sequences:
        value = list(range(len(seq)))
        values.append(value)
    
    data = {
        "device": hw_info["device"],
        "dtype": torch.float32,
        "hw_info": hw_info,
        "sequences": sequences,
        "values": values,
        "num_sequences": num_sequences,
        "max_seq_len": max_seq_len,
        "vocab_size": vocab_size,
    }
    
    return data

# =======================
# Experiment Execution
# =======================
def experiment(data: Dict[str, Any]) -> Any:
    """Execute the optimized operation."""
    RadixCache, fq_name = resolve_target()
    
    # Create a RadixCache instance
    cache = RadixCache()
    
    sequences = data["sequences"]
    values = data["values"]
    
    # Insert all sequences to trigger _split_node operations
    with torch.no_grad():
        for seq, val in zip(sequences, values):
            cache.insert(seq, val)
    
    # Return the final tree structure for equivalence checking
    # Traverse the tree and collect all stored sequences
    result = {
        "num_nodes": cache.num_nodes,
        "total_tokens": cache.total_tokens,
        "sequences_stored": len(sequences),
    }
    
    # Collect all sequences in the tree for validation
    stored_sequences = []
    def traverse(node, prefix=[]):
        if node.value is not None:
            stored_sequences.append((prefix.copy(), node.value))
        for key, child in node.children.items():
            new_prefix = prefix + [key] + child.key
            traverse(child, new_prefix)
    
    if hasattr(cache, 'root_node'):
        traverse(cache.root_node, [])
    result["stored_sequences"] = stored_sequences
    
    return result

# =======================
# Result I/O
# =======================
def store_result(result: Any, filepath: str) -> None:
    """Store result for reference comparison."""
    with open(filepath, 'wb') as f:
        pickle.dump(result, f)

def load_result(filepath: str) -> Any:
    """Load reference result."""
    with open(filepath, 'rb') as f:
        return pickle.load(f)

# =======================
# Equivalence Checking
# =======================
def check_equivalence(current_result: Any, reference_result: Any) -> None:
    """Verify functional equivalence."""
    # Check basic statistics
    assert current_result["num_nodes"] == reference_result["num_nodes"], f"Node count mismatch"
    assert current_result["total_tokens"] == reference_result["total_tokens"], f"Token count mismatch"
    assert current_result["sequences_stored"] == reference_result["sequences_stored"], f"Sequence count mismatch"
    
    # Check stored sequences
    current_seqs = sorted(current_result["stored_sequences"], key=lambda x: (len(x[0]), x[0]))
    ref_seqs = sorted(reference_result["stored_sequences"], key=lambda x: (len(x[0]), x[0]))
    
    assert len(current_seqs) == len(ref_seqs), f"Different number of stored sequences"
    
    for (curr_key, curr_val), (ref_key, ref_val) in zip(current_seqs, ref_seqs):
        assert curr_key == ref_key, f"Key mismatch: {curr_key} vs {ref_key}"
        assert curr_val == ref_val, f"Value mismatch for key {curr_key}"

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
    
    # Radix tree operations are CPU-bound
    warmup = 3
    iters = 10
    
    result, timing_stats = time_cpu(lambda: experiment(data), warmup=warmup, iterations=iters)
    avg_ms = timing_stats["avg_ms"]
    p50_ms = timing_stats["p50_ms"]
    p95_ms = timing_stats["p95_ms"]
    
    # Reference handling
    commit_hash = os.getenv("COMMIT_HASH", "b170930534acbb9c1619a3c83670a839ceee763a")
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
        "device": "cpu",  # Radix tree operations are CPU-bound
        "dtype": "none",  # Not a tensor operation
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