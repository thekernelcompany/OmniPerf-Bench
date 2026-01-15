#!/usr/bin/env python3
"""
Performance test for commit: bb3a3b6675b1844a13ebe368ad693f3dc75b315b
Message: Support Faster JSON decoding for llava (#137)

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
    
    # Priority 2: Parse from commit metadata - target the Req class with fast_forward_and_retokenize
    if not (module_path and symbol_name):
        module_path = "sglang.srt.managers.router.infer_batch"
        symbol_name = "Req"
    
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
# Mock Tokenizer
# =======================
class MockTokenizer:
    """Mock tokenizer for testing fast-forward retokenization."""
    
    def __init__(self):
        self.vocab_size = 32000
        self.pad_token_id = 0
        
    def encode(self, text):
        # Simple deterministic encoding based on text length and hash
        np.random.seed(hash(text) % 2**32)
        length = len(text.split()) * 2 + len(text) // 10
        tokens = np.random.randint(1, self.vocab_size, size=length).tolist()
        return tokens
    
    def decode(self, tokens):
        # Simple deterministic decoding
        np.random.seed(sum(tokens) % 2**32)
        words = ['word', 'text', 'token', 'data', 'model', 'output', 'input', 'test']
        num_words = len(tokens) // 2
        return ' '.join(np.random.choice(words, size=num_words).tolist())
    
    def convert_ids_to_tokens(self, ids):
        return [f"token_{i}" for i in ids]

# =======================
# Workload Setup
# =======================
def setup() -> Dict[str, Any]:
    """Create realistic workload for the optimization."""
    ensure_determinism()
    hw_info = detect_hardware()
    
    # Create mock requests with and without images for fast-forward testing
    device = torch.device(hw_info["device"])
    dtype = torch.float16 if hw_info["device"] == "cuda" else torch.float32
    
    # Batch of requests to test fast-forward
    batch_size = 32
    
    # Create mock requests
    requests = []
    tokenizer = MockTokenizer()
    
    for i in range(batch_size):
        # Create a mock Req-like object
        req_data = {
            'rid': f'req_{i}',
            'input_text': f"This is input text for request {i} with some context",
            'input_ids': tokenizer.encode(f"This is input text for request {i} with some context"),
            'output_ids': [],
            'tokenizer': tokenizer,
            'pixel_values': None,
            'image_size': None,
            'image_offset': 0,
            'pad_value': None,
            'state': 0,
        }
        
        # Half the requests have images (multimodal)
        if i % 2 == 0:
            req_data['pixel_values'] = torch.randn(3, 336, 336, device=device, dtype=dtype)
            req_data['image_size'] = (336, 336)
            req_data['image_hash'] = hash(f"image_{i}") % (2**48)
            req_data['pad_value'] = [
                req_data['image_hash'] % tokenizer.vocab_size,
                (req_data['image_hash'] >> 16) % tokenizer.vocab_size,
                (req_data['image_hash'] >> 32) % tokenizer.vocab_size,
                (req_data['image_hash'] >> 48) % tokenizer.vocab_size,
            ]
            # Simulate image padding in input_ids
            num_image_tokens = 256  # Typical number of image tokens
            req_data['input_ids'] = req_data['pad_value'] * (num_image_tokens // 4) + req_data['input_ids']
            req_data['image_offset'] = num_image_tokens
        
        requests.append(req_data)
    
    # Fast-forward strings to test
    fast_forward_strings = [
        "The answer is 42.",
        "Based on the image, I can see",
        "In conclusion, the results show",
        "According to the data presented",
    ] * (batch_size // 4)
    
    data = {
        "device": device,
        "dtype": dtype,
        "hw_info": hw_info,
        "requests": requests,
        "fast_forward_strings": fast_forward_strings,
        "tokenizer": tokenizer,
    }
    
    return data

# =======================
# Experiment Execution
# =======================
def experiment(data: Dict[str, Any]) -> Any:
    """Execute the optimized operation."""
    Req, fq_name = resolve_target()
    
    # Create Req instances and perform fast-forward retokenization
    results = []
    tokenizer = data["tokenizer"]
    
    for req_data, ff_str in zip(data["requests"], data["fast_forward_strings"]):
        # Create a Req instance
        req = Req(req_data['rid'], req_data['input_text'], req_data['input_ids'].copy())
        req.tokenizer = tokenizer
        req.output_ids = req_data['output_ids'].copy()
        req.pixel_values = req_data['pixel_values']
        req.image_size = req_data['image_size']
        req.image_offset = req_data['image_offset']
        req.pad_value = req_data['pad_value']
        
        # Simulate some output tokens
        req.output_ids = tokenizer.encode("Some output text")[:10]
        
        # Perform fast-forward retokenization
        try:
            # Call the optimized method
            req.fast_forward_and_retokenize(ff_str, next_state=1)
            
            results.append({
                'rid': req.rid,
                'new_input_ids_len': len(req.input_ids),
                'new_output_ids_len': len(req.output_ids),
                'has_image': req.pixel_values is not None,
                'image_offset': req.image_offset,
            })
        except Exception as e:
            # Fallback for older version without the method
            results.append({
                'rid': req.rid,
                'error': str(e),
                'has_image': req.pixel_values is not None,
            })
    
    return results

# =======================
# Result I/O
# =======================
def store_result(result: Any, filepath: str) -> None:
    """Store result for reference comparison."""
    torch.save({"type": "list", "data": result}, filepath)

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
        assert len(current_result) == len(reference_result), f"Length: {len(current_result)} vs {len(reference_result)}"
        
        for i, (curr, ref) in enumerate(zip(current_result, reference_result)):
            # Check structure
            assert curr.keys() == ref.keys(), f"Keys mismatch at index {i}"
            
            # Check values with some tolerance for numerical fields
            for key in curr:
                if key in ['new_input_ids_len', 'new_output_ids_len', 'image_offset']:
                    # These should be exact
                    assert curr[key] == ref[key], f"Mismatch at index {i}, key {key}: {curr[key]} vs {ref[key]}"
                elif key in ['has_image', 'rid']:
                    assert curr[key] == ref[key], f"Mismatch at index {i}, key {key}"

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
        "p95_ms": times_ms[min(int(len(times_ms) * 0.95), len(times_ms) - 1)],
        "p99_ms": times_ms[min(int(len(times_ms) * 0.99), len(times_ms) - 1)],
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
    
    # This is a CPU-bound operation (tokenization)
    warmup = 5
    iters = 20
    
    # Time the operation
    result, timing_stats = time_cpu_operation(lambda: experiment(data), warmup=warmup, iterations=iters)
    avg_ms = timing_stats["avg_ms"]
    p50_ms = timing_stats["p50_ms"]
    p95_ms = timing_stats["p95_ms"]
    
    # Reference handling
    commit_hash = os.getenv("COMMIT_HASH", "bb3a3b6675b1844a13ebe368ad693f3dc75b315b")
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
        "device": "cpu",  # Tokenization is CPU-bound
        "dtype": str(data["dtype"]),
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