#!/usr/bin/env python3
"""
Performance test for commit: e822e5900b98d89d19e0a293d9ad384f4df2945a
Message: Optimize radix tree matching (#364)

This script measures the actual performance impact of the optimization.
It supports cross-commit comparison with functional equivalence checking.
"""

import os
import sys
import json
import time
import importlib
from typing import Dict, Any, Tuple, Optional, List
import random

import numpy as np
import torch

# =======================
# Determinism Setup
# =======================
def ensure_determinism():
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)

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
        # Based on the diff, the optimized class is RadixCache
        module_path = "sglang.srt.managers.router.radix_cache"
        symbol_name = "RadixCache"
    
    # Import with error handling
    try:
        module = importlib.import_module(module_path)
        target = getattr(module, symbol_name)
        
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
    """Create realistic workload for the radix cache optimization."""
    ensure_determinism()
    hw_info = detect_hardware()
    
    # Create realistic token sequences for a radix tree cache
    # Simulate common prefixes with branching (typical LLM cache pattern)
    vocab_size = 32000  # Common LLM vocab size
    num_sequences = 1000  # Number of cached sequences
    min_seq_len = 10
    max_seq_len = 200
    common_prefix_len = 20  # Sequences often share prefixes
    
    # Generate sequences with shared prefixes to stress the tree structure
    sequences = []
    
    # Create base prefixes that will be shared
    num_prefixes = 50
    base_prefixes = []
    for _ in range(num_prefixes):
        prefix_len = random.randint(5, common_prefix_len)
        prefix = [random.randint(0, vocab_size-1) for _ in range(prefix_len)]
        base_prefixes.append(prefix)
    
    # Generate sequences by extending prefixes
    for _ in range(num_sequences):
        # Pick a random prefix to extend
        if random.random() < 0.7:  # 70% chance to use existing prefix
            prefix = random.choice(base_prefixes).copy()
        else:
            prefix = []
        
        # Extend to full sequence
        total_len = random.randint(min_seq_len, max_seq_len)
        extension_len = max(0, total_len - len(prefix))
        extension = [random.randint(0, vocab_size-1) for _ in range(extension_len)]
        sequence = prefix + extension
        sequences.append(sequence)
    
    # Create lookup queries - mix of hits and misses
    num_queries = 500
    queries = []
    
    # 60% exact matches (cache hits)
    for _ in range(int(num_queries * 0.6)):
        queries.append(random.choice(sequences).copy())
    
    # 30% prefix matches (partial hits)
    for _ in range(int(num_queries * 0.3)):
        seq = random.choice(sequences)
        prefix_len = random.randint(1, min(len(seq)-1, 50))
        queries.append(seq[:prefix_len])
    
    # 10% new sequences (cache misses)
    for _ in range(int(num_queries * 0.1)):
        new_len = random.randint(min_seq_len, max_seq_len)
        new_seq = [random.randint(0, vocab_size-1) for _ in range(new_len)]
        queries.append(new_seq)
    
    # Shuffle queries for realistic access pattern
    random.shuffle(queries)
    
    data = {
        "device": hw_info["device"],
        "dtype": torch.float32,  # Not used for this CPU operation
        "hw_info": hw_info,
        "sequences": sequences,
        "queries": queries,
        "num_sequences": num_sequences,
        "num_queries": num_queries,
        "vocab_size": vocab_size,
    }
    
    return data

# =======================
# Experiment Execution
# =======================
def experiment(data: Dict[str, Any]) -> Any:
    """Execute the optimized radix cache operations."""
    RadixCache, fq_name = resolve_target()
    
    # Create cache instance
    cache = RadixCache()
    cache.reset()
    
    sequences = data["sequences"]
    queries = data["queries"]
    
    # Phase 1: Build the tree by inserting all sequences
    insert_results = []
    for seq in sequences:
        # Each token gets a dummy value (simulating cache entries)
        values = list(range(len(seq)))
        cache.insert(seq, values)
        insert_results.append(len(seq))
    
    # Phase 2: Perform lookups (the optimized operation)
    lookup_results = []
    for query in queries:
        result = cache.match_prefix(query)
        lookup_results.append(len(result))
    
    # Return both phases for validation
    return {
        "insert_results": insert_results,
        "lookup_results": lookup_results,
        "total_size": cache.total_size(),
        "evictable_size": cache.evictable_size(),
    }

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
    assert isinstance(current_result, dict), "Result should be a dictionary"
    assert isinstance(reference_result, dict), "Reference should be a dictionary"
    
    # Check all keys present
    assert current_result.keys() == reference_result.keys(), f"Keys mismatch"
    
    # Check insert results
    assert current_result["insert_results"] == reference_result["insert_results"], \
        "Insert results mismatch"
    
    # Check lookup results
    assert current_result["lookup_results"] == reference_result["lookup_results"], \
        "Lookup results mismatch"
    
    # Check cache statistics
    assert current_result["total_size"] == reference_result["total_size"], \
        f"Total size mismatch: {current_result['total_size']} vs {reference_result['total_size']}"
    
    assert current_result["evictable_size"] == reference_result["evictable_size"], \
        f"Evictable size mismatch: {current_result['evictable_size']} vs {reference_result['evictable_size']}"

# =======================
# Timing Implementation
# =======================
def time_cpu_operation(func, warmup=3, iterations=10) -> Tuple[Any, Dict[str, float]]:
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
    
    # This is a CPU-only operation (radix tree is not GPU-accelerated)
    warmup = 3
    iters = 10
    
    # Time the experiment
    result, timing_stats = time_cpu_operation(lambda: experiment(data), warmup=warmup, iterations=iters)
    
    avg_ms = timing_stats["avg_ms"]
    p50_ms = timing_stats["p50_ms"]
    p95_ms = timing_stats["p95_ms"]
    
    # Reference handling
    commit_hash = os.getenv("COMMIT_HASH", "e822e5900b98d89d19e0a293d9ad384f4df2945a")
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
        "device": "cpu",  # Radix tree operations are CPU-based
        "dtype": "none",  # Not applicable for this integer-based tree operation
        "iters": iters,
        "warmup": warmup,
        "avg_ms": avg_ms,
        "p50_ms": p50_ms,
        "p95_ms": p95_ms,
        "eq_level": os.getenv("PROB_EQ_LEVEL", "exact"),  # Tree operations are deterministic
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