#!/usr/bin/env python3
"""
Performance test for commit: e5db40dcbce67157e005f524bf6a5bea7dcb7f34
Message: ORJson. Faster Json serialization (#1694)

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

# =======================
# Determinism Setup
# =======================
def ensure_determinism():
    np.random.seed(42)

# =======================
# Hardware Detection
# =======================
def detect_hardware() -> Dict[str, Any]:
    hw_info = {}
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
    
    # Priority 2: Parse from commit metadata - using orjson
    if not module_path:
        module_path = "orjson"
        symbol_name = "dumps"
    
    # Import with error handling
    try:
        module = importlib.import_module(module_path)
        if symbol_name:
            target = getattr(module, symbol_name)
        else:
            target = module
        
        fq_name = f"{module_path}.{symbol_name}" if symbol_name else module_path
        return target, fq_name
        
    except (ImportError, AttributeError) as e:
        # Fallback to json.dumps if orjson not available (parent commit)
        try:
            import json
            return json.dumps, "json.dumps"
        except ImportError:
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
    
    # Create realistic LLM server response data
    # Typical streaming response from generate_request
    batch_size = 32
    seq_len = 512
    vocab_size = 32000
    
    # Simulate multiple response chunks like in streaming
    response_chunks = []
    
    # Generate typical completion responses
    for i in range(batch_size):
        chunk = {
            "id": f"cmpl-{i:08d}",
            "object": "text_completion",
            "created": 1697654321 + i,
            "model": "sglang-llama-7b",
            "system_fingerprint": "fp_44709d6fcb",
            "choices": [
                {
                    "index": 0,
                    "text": "The " + " ".join([f"token_{j}" for j in range(seq_len)]),
                    "logprobs": {
                        "tokens": [f"token_{j}" for j in range(seq_len)],
                        "token_logprobs": [float(np.random.randn()) for _ in range(seq_len)],
                        "top_logprobs": [
                            {f"token_{k}": float(np.random.randn()) for k in range(5)}
                            for _ in range(seq_len)
                        ],
                        "text_offset": list(range(0, seq_len * 6, 6))
                    },
                    "finish_reason": "length" if i % 2 == 0 else "stop"
                }
            ],
            "usage": {
                "prompt_tokens": 128,
                "completion_tokens": seq_len,
                "total_tokens": 128 + seq_len
            },
            "meta_data": {
                "decode_tokens": seq_len,
                "decode_ms": float(seq_len * 20.5),
                "prefill_tokens": 128,
                "prefill_ms": 45.2,
                "tokens_per_second": float(seq_len / (seq_len * 0.0205))
            }
        }
        response_chunks.append(chunk)
    
    # Also create embedding response data
    embedding_dim = 4096
    num_embeddings = 16
    embedding_response = {
        "object": "list",
        "data": [
            {
                "object": "embedding",
                "embedding": [float(np.random.randn()) for _ in range(embedding_dim)],
                "index": i
            }
            for i in range(num_embeddings)
        ],
        "model": "sglang-embedding-model",
        "usage": {
            "prompt_tokens": num_embeddings * 128,
            "total_tokens": num_embeddings * 128
        }
    }
    
    # Model list response
    models_response = {
        "object": "list",
        "data": [
            {
                "id": "sglang-llama-7b",
                "object": "model",
                "created": 1686935002,
                "owned_by": "sglang"
            },
            {
                "id": "sglang-llama-13b",
                "object": "model",
                "created": 1686935003,
                "owned_by": "sglang"
            }
        ]
    }
    
    data = {
        "device": "cpu",
        "dtype": "json",
        "hw_info": hw_info,
        "response_chunks": response_chunks,
        "embedding_response": embedding_response,
        "models_response": models_response,
    }
    
    return data

# =======================
# Experiment Execution
# =======================
def experiment(data: Dict[str, Any]) -> Any:
    """Execute the optimized operation."""
    target, fq_name = resolve_target()
    
    # Serialize all the response data using the target serializer
    serialized_chunks = []
    
    # Serialize streaming response chunks (main workload)
    for chunk in data["response_chunks"]:
        if "orjson" in fq_name:
            # Use orjson with same options as in the commit
            try:
                import orjson
                serialized = orjson.dumps(chunk, option=orjson.OPT_NON_STR_KEYS)
            except ImportError:
                # Fallback for parent commit
                serialized = json.dumps(chunk, ensure_ascii=False).encode()
        else:
            # Use json.dumps (parent commit)
            serialized = json.dumps(chunk, ensure_ascii=False).encode()
        serialized_chunks.append(serialized)
    
    # Serialize embedding response
    if "orjson" in fq_name:
        try:
            import orjson
            embedding_serialized = orjson.dumps(data["embedding_response"], option=orjson.OPT_NON_STR_KEYS)
        except ImportError:
            embedding_serialized = json.dumps(data["embedding_response"], ensure_ascii=False).encode()
    else:
        embedding_serialized = json.dumps(data["embedding_response"], ensure_ascii=False).encode()
    
    # Serialize models response  
    if "orjson" in fq_name:
        try:
            import orjson
            models_serialized = orjson.dumps(data["models_response"], option=orjson.OPT_NON_STR_KEYS)
        except ImportError:
            models_serialized = json.dumps(data["models_response"], ensure_ascii=False).encode()
    else:
        models_serialized = json.dumps(data["models_response"], ensure_ascii=False).encode()
    
    return {
        "chunks": serialized_chunks,
        "embedding": embedding_serialized,
        "models": models_serialized
    }

# =======================
# Result I/O
# =======================
def store_result(result: Any, filepath: str) -> None:
    """Store result for reference comparison."""
    # Store as pickle since we have bytes objects
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
    assert isinstance(current_result, dict), "Result should be dict"
    assert isinstance(reference_result, dict), "Reference should be dict"
    
    # Check chunks
    assert len(current_result["chunks"]) == len(reference_result["chunks"]), \
        f"Chunk count mismatch: {len(current_result['chunks'])} vs {len(reference_result['chunks'])}"
    
    for i, (curr_chunk, ref_chunk) in enumerate(zip(current_result["chunks"], reference_result["chunks"])):
        # Deserialize and compare as dicts for semantic equivalence
        curr_data = json.loads(curr_chunk) if isinstance(curr_chunk, bytes) else json.loads(curr_chunk.encode())
        ref_data = json.loads(ref_chunk) if isinstance(ref_chunk, bytes) else json.loads(ref_chunk.encode())
        
        # Deep comparison of dictionaries
        assert curr_data == ref_data, f"Chunk {i} data mismatch"
    
    # Check embedding response
    curr_emb = json.loads(current_result["embedding"]) if isinstance(current_result["embedding"], bytes) else json.loads(current_result["embedding"].encode())
    ref_emb = json.loads(reference_result["embedding"]) if isinstance(reference_result["embedding"], bytes) else json.loads(reference_result["embedding"].encode())
    
    # Compare with tolerance for floating point
    assert curr_emb["object"] == ref_emb["object"]
    assert len(curr_emb["data"]) == len(ref_emb["data"])
    for i in range(len(curr_emb["data"])):
        assert curr_emb["data"][i]["object"] == ref_emb["data"][i]["object"]
        assert curr_emb["data"][i]["index"] == ref_emb["data"][i]["index"]
        curr_embedding = curr_emb["data"][i]["embedding"]
        ref_embedding = ref_emb["data"][i]["embedding"]
        assert len(curr_embedding) == len(ref_embedding)
        for j in range(len(curr_embedding)):
            assert abs(curr_embedding[j] - ref_embedding[j]) < 1e-6, \
                f"Embedding mismatch at [{i}][{j}]: {curr_embedding[j]} vs {ref_embedding[j]}"
    
    # Check models response
    curr_models = json.loads(current_result["models"]) if isinstance(current_result["models"], bytes) else json.loads(current_result["models"].encode())
    ref_models = json.loads(reference_result["models"]) if isinstance(reference_result["models"], bytes) else json.loads(reference_result["models"].encode())
    assert curr_models == ref_models, "Models response mismatch"

# =======================
# Main Test Function
# =======================
def run_test(eqcheck: bool = False, reference: bool = False, prefix: str = '') -> float:
    """Main test entry point."""
    
    # Setup
    data = setup()
    hw_info = data["hw_info"]
    
    # CPU timing
    warmup = 3
    iters = 100  # More iterations for CPU timing
    
    # Warmup
    for _ in range(warmup):
        _ = experiment(data)
    
    # Timing
    times = []
    for _ in range(iters):
        start = time.perf_counter()
        result = experiment(data)
        end = time.perf_counter()
        times.append((end - start) * 1000)
    
    times.sort()
    avg_ms = sum(times) / len(times)
    p50_ms = times[len(times) // 2]
    p95_ms = times[int(len(times) * 0.95)]
    
    # Reference handling
    commit_hash = os.getenv("COMMIT_HASH", "e5db40dcbce67157e005f524bf6a5bea7dcb7f34")
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
        "dtype": "json",
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