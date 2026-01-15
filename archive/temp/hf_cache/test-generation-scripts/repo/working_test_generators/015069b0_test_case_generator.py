#!/usr/bin/env python3
"""
Performance test for commit: 015069b01741e9ecb9e604c7fe87fbdfc306ebe5
Message: [Misc] Optimize the Qwen3_ReasoningParser extract_reasoning_content (#17515)

This script measures the actual performance impact of the optimization.
It supports cross-commit comparison with functional equivalence checking.
"""

import os
import sys
import json
import time
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
    
    # Priority 2: Parse from commit metadata - targeting extract_reasoning_content
    if not (module_path and symbol_name):
        module_path = "vllm.reasoning.qwen3_reasoning_parser"
        symbol_name = "Qwen3ReasoningParser"
    
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
# Mock Dependencies
# =======================
class MockChatCompletionRequest:
    """Mock request object for testing."""
    def __init__(self):
        self.model = "test-model"
        self.messages = []

class MockTokenizer:
    """Mock tokenizer for the parser."""
    def __init__(self):
        self.model_max_length = 32768
    
    def encode(self, text, add_special_tokens=False):
        # Simple mock encoding - return list of integers based on text length
        return list(range(len(text)))
    
    def decode(self, token_ids, skip_special_tokens=False):
        # Simple mock decoding
        return "decoded_text"

# =======================
# Workload Setup
# =======================
def setup() -> Dict[str, Any]:
    """Create realistic workload for the optimization."""
    ensure_determinism()
    hw_info = detect_hardware()
    
    # Create test cases with varying patterns and sizes
    test_cases = []
    
    # Case 1: Small reasoning content
    test_cases.append({
        "input": "<think>This is a short reasoning.</think>This is the response.",
        "name": "small_reasoning"
    })
    
    # Case 2: Large reasoning content (more realistic for LLM outputs)
    large_reasoning = "Let me think step by step. " * 100  # ~2KB of reasoning
    test_cases.append({
        "input": f"<think>{large_reasoning}</think>Based on my analysis, the answer is 42.",
        "name": "large_reasoning"
    })
    
    # Case 3: No reasoning tags
    test_cases.append({
        "input": "This is a direct response without any reasoning tags.",
        "name": "no_reasoning"
    })
    
    # Case 4: Multiple occurrences (edge case)
    test_cases.append({
        "input": "<think>First thought</think>Response<think>Second thought</think>Final",
        "name": "multiple_tags"
    })
    
    # Case 5: Very large content (stress test)
    huge_reasoning = "Complex reasoning involving multiple steps. " * 500  # ~10KB
    huge_response = "The detailed answer is as follows. " * 200  # ~4KB
    test_cases.append({
        "input": f"<think>{huge_reasoning}</think>{huge_response}",
        "name": "huge_content"
    })
    
    # Case 6: Missing end tag (edge case)
    test_cases.append({
        "input": "<think>Incomplete reasoning without end tag",
        "name": "missing_end_tag"
    })
    
    # Case 7: Empty reasoning
    test_cases.append({
        "input": "<think></think>Just the response",
        "name": "empty_reasoning"
    })
    
    # Generate more realistic test cases
    for i in range(20):
        # Varying sizes of reasoning and response
        reasoning_size = np.random.randint(10, 500)
        response_size = np.random.randint(10, 200)
        reasoning = " ".join([f"Step {j}: reasoning detail." for j in range(reasoning_size)])
        response = " ".join([f"Response part {j}." for j in range(response_size)])
        
        if i % 3 == 0:
            # With reasoning tags
            test_cases.append({
                "input": f"<think>{reasoning}</think>{response}",
                "name": f"synthetic_{i}"
            })
        elif i % 3 == 1:
            # Without tags
            test_cases.append({
                "input": f"{reasoning} {response}",
                "name": f"synthetic_no_tags_{i}"
            })
        else:
            # Edge case - only start tag
            test_cases.append({
                "input": f"<think>{reasoning} {response}",
                "name": f"synthetic_incomplete_{i}"
            })
    
    device = torch.device("cpu")  # This is a CPU optimization
    dtype = torch.float32
    
    data = {
        "device": device,
        "dtype": dtype,
        "hw_info": hw_info,
        "test_cases": test_cases,
        "mock_request": MockChatCompletionRequest(),
        "mock_tokenizer": MockTokenizer()
    }
    
    return data

# =======================
# Experiment Execution
# =======================
def experiment(data: Dict[str, Any]) -> Any:
    """Execute the optimized operation."""
    target_class, fq_name = resolve_target()
    
    # Instantiate the parser with mock tokenizer
    parser = target_class(model_tokenizer=data["mock_tokenizer"])
    
    # Run extract_reasoning_content on all test cases
    results = []
    for test_case in data["test_cases"]:
        reasoning, content = parser.extract_reasoning_content(
            test_case["input"],
            data["mock_request"]
        )
        results.append({
            "name": test_case["name"],
            "reasoning": reasoning,
            "content": content
        })
    
    return results

# =======================
# Result I/O
# =======================
def store_result(result: Any, filepath: str) -> None:
    """Store result for reference comparison."""
    # Convert to JSON-serializable format
    import pickle
    with open(filepath, 'wb') as f:
        pickle.dump(result, f)

def load_result(filepath: str) -> Any:
    """Load reference result."""
    import pickle
    with open(filepath, 'rb') as f:
        return pickle.load(f)

# =======================
# Equivalence Checking
# =======================
def check_equivalence(current_result: Any, reference_result: Any) -> None:
    """Verify functional equivalence."""
    assert isinstance(current_result, list) and isinstance(reference_result, list)
    assert len(current_result) == len(reference_result), f"Result count mismatch: {len(current_result)} vs {len(reference_result)}"
    
    for i, (curr, ref) in enumerate(zip(current_result, reference_result)):
        assert curr["name"] == ref["name"], f"Test case name mismatch at {i}: {curr['name']} vs {ref['name']}"
        assert curr["reasoning"] == ref["reasoning"], f"Reasoning mismatch at {i} ({curr['name']})"
        assert curr["content"] == ref["content"], f"Content mismatch at {i} ({curr['name']})"

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
    
    # Timing - this is a CPU optimization
    warmup = 5
    iters = 20  # More iterations for CPU string operations to get stable measurements
    
    result, timing_stats = time_cpu(lambda: experiment(data), warmup=warmup, iterations=iters)
    avg_ms = timing_stats["avg_ms"]
    p50_ms = timing_stats["p50_ms"]
    p95_ms = timing_stats["p95_ms"]
    
    # Reference handling
    commit_hash = os.getenv("COMMIT_HASH", "015069b01741e9ecb9e604c7fe87fbdfc306ebe5")
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
        "device": "cpu",
        "dtype": "str",  # String operations
        "iters": iters,
        "warmup": warmup,
        "avg_ms": avg_ms,
        "p50_ms": p50_ms,
        "p95_ms": p95_ms,
        "eq_level": os.getenv("PROB_EQ_LEVEL", "exact"),  # String comparison is exact
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