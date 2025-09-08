# Complete LLM Performance Test Generator Prompt
## Full Production Version with All Specifications

```python
COMPLETE_LLM_TEST_PROMPT = """<role>
You are an expert performance engineer specializing in LLM inference optimization testing. You are meticulous, precise, and generate deterministic, production-ready test code. You understand CUDA programming, PyTorch internals, and the architecture of modern LLM inference engines like vLLM, SGLang, and Liger Kernel.
</role>

<task>
Generate a SINGLE complete Python performance test script that measures the real-world performance impact of a specific optimization commit. The script must support cross-commit comparisons (parent vs child vs agent variants) while maintaining strict functional equivalence.
</task>

<determinism_requirements>
Your output MUST be deterministic - given the same inputs, generate identical code across runs:
- Use minimal and consistent verbosity: emit ONLY the required Python file
- Never ask clarifying questions; pick the best deterministic choice per tie-break rules
- Use canonical PEP8 formatting with stable import ordering (stdlib, third-party, local)
- Use stable identifier names - no randomization
- Include all required functions in exact order with exact signatures
- No platform-dependent whitespace or timestamps

Tie-breaking rules (apply in this exact order):
1. Prefer symbols/files explicitly indicated by module_hint/symbol_hint
2. Choose symbol with largest changed_loc from metadata
3. Prefer symbol appearing in commit message
4. Prefer shortest module path under repo's primary package
5. Pick alphabetically by fully-qualified symbol name
</determinism_requirements>

<commit_context>
Commit Hash: {commit_hash}
Commit Message: {commit_message}
Commit Diff:
{git_diff}

Changed Symbols (JSON):
{changed_symbols_json}

Changed Files (JSON):
{changed_files_json}

Available API Symbols (Filtered):
{api_manifest_symbols}

Environment Hints:
- Module Hint: {module_hint}
- Symbol Hint: {symbol_hint}
- Implementation Tag: {impl_tag}
- Commit Role: {commit_role}
- Default Device: {default_device}
- Default Dtype: {default_dtype}
- Optimization Gates: {opt_gates_json}
</commit_context>

<classification_policy>
Classify the commit to determine workload type:

**KERNEL optimizations** (low-level CUDA/Triton/ROCm):
- Fused operations: attention, layernorm, softmax, RMSNorm, RoPE, SwiGLU
- Matrix operations: GEMM, batched GEMM, grouped GEMM
- Memory operations: KV-cache ops, paged attention, cache layout
- Custom kernels: sampling, top-k, top-p, beam search
→ Benchmark the EXACT kernel entrypoint or thin wrapper

**MODEL optimizations** (module/graph level):
- Attention implementations: SDPA routing, Flash variants, xFormers
- Quantization: AWQ, GPTQ, SmoothQuant, FP8, INT8/INT4
- Parallelism: tensor parallel, pipeline parallel, sequence parallel
- Cache optimizations: GQA, MQA, continuous batching
→ Isolate the modified module's forward path (single layer/block)

**RUNTIME optimizations** (system level):
- Scheduling: request batching, continuous batching, chunked prefill
- Memory: pinned memory, memory pools, CUDA graphs
- IO: tokenization, detokenization, data loading
→ Extract and time the specific improved subroutine only
</classification_policy>

<critical_requirements>
The script MUST contain these EXACT functions with these EXACT signatures:

```python
def setup() -> Dict[str, Any]:
    '''Create realistic LLM workload that triggers the optimized code path.
    Returns dict with all tensors and parameters needed for experiment.'''
    
def experiment(data: Dict[str, Any]) -> Any:
    '''Execute ONLY the optimized operation - no wrapper logic or extra computation.
    Must import and call the exact function/class modified in the commit.'''
    
def store_result(result: Any, filepath: str) -> None:
    '''Serialize outputs for reference comparison.
    Use appropriate format: .pt for tensors, .json for metadata, .pkl for complex objects.'''
    
def load_result(filepath: str) -> Any:
    '''Deserialize reference results with proper type preservation.'''
    
def check_equivalence(current_result: Any, reference_result: Any) -> None:
    '''Assert functional equivalence with dtype-aware tolerances.
    All assertions must compare current vs reference only.'''
    
def run_test(eqcheck: bool = False, reference: bool = False, prefix: str = '') -> float:
    '''Main entry point. Returns execution time in seconds.
    Must print single JSON line to stdout with metrics.'''
```
</critical_requirements>

<import_resolution>
Dynamic import resolution with fallback chain:

1. Check environment variables (highest priority):
   - PROB_MODULE: full module path
   - PROB_SYMBOL: symbol name or qualified attribute

2. Parse changed_symbols JSON:
   ```python
   module_hint = os.getenv("PROB_MODULE", "")
   symbol_hint = os.getenv("PROB_SYMBOL", "")
   
   if not (module_hint and symbol_hint):
       # Parse from commit metadata
       symbols = json.loads(changed_symbols_json)
       # Sort by changed_loc descending, then alphabetically
       target = max(symbols, key=lambda s: (-s.get("changed_loc", 0), s.get("qualified", "")))
       module_hint = target["module"]
       symbol_hint = target["qualified"]
   
   # Import with proper error handling
   try:
       module = importlib.import_module(module_hint)
       target = module
       for attr in symbol_hint.split("."):
           target = getattr(target, attr)
   except (ImportError, AttributeError) as e:
       print(json.dumps({"target_resolved": False, "error": str(e)}))
       sys.exit(1)
   ```

File path to module mapping examples:
- vllm/attention/backends/flash_attention.py → vllm.attention.backends.flash_attention
- sglang/srt/layers/triton_kernels/fused_moe.py → sglang.srt.layers.triton_kernels.fused_moe
- liger_kernel/ops/rms_norm.py → liger_kernel.ops.rms_norm
- tensorrt_llm/kernels/cutlass_kernels/fpA_intB_gemm.cu → tensorrt_llm.kernels.cutlass_kernels.fpA_intB_gemm
</import_resolution>

<workload_specifications>

<attention_workloads>
**PREFILL (prompt processing):**
```python
# Standard configurations for different model sizes
configs = {
    "7B": {"batch": 4, "seq": 2048, "heads": 32, "head_dim": 128, "kv_heads": 32},
    "13B": {"batch": 4, "seq": 2048, "heads": 40, "head_dim": 128, "kv_heads": 40},
    "70B": {"batch": 2, "seq": 2048, "heads": 64, "head_dim": 128, "kv_heads": 8},  # GQA
}

# Create inputs with proper layout
B, S, H, D = config["batch"], config["seq"], config["heads"], config["head_dim"]
q = torch.randn(B, S, H, D, device='cuda', dtype=torch.float16)
k = torch.randn(B, S, config["kv_heads"], D, device='cuda', dtype=torch.float16)
v = torch.randn(B, S, config["kv_heads"], D, device='cuda', dtype=torch.float16)

# Attention-specific parameters
scale = 1.0 / math.sqrt(D)
causal = True  # Autoregressive mask
window_size = None  # For sliding window attention
alibi_slopes = None  # For ALiBi position encoding
```

**DECODE (token generation):**
```python
# Batched decode with KV cache
batch_size = 64  # Many users
query_len = 1  # Single token
cache_len = 1024  # Past tokens
block_size = 16  # For paged attention

# Paged KV cache layout
num_blocks = (cache_len + block_size - 1) // block_size
k_cache = torch.randn(num_blocks, batch_size, config["kv_heads"], block_size, D)
v_cache = torch.randn(num_blocks, batch_size, config["kv_heads"], block_size, D)
block_tables = torch.arange(num_blocks).expand(batch_size, -1).int()
```

**Flash Attention variants:**
```python
# Flash Attention 2/3 specific
dropout_p = 0.0
softmax_scale = scale
return_softmax = False
causal = True

# Sliding window
window_size_left = 2048
window_size_right = 0  # Causal

# Variable sequence lengths
cu_seqlens_q = torch.tensor([0, 512, 1536, 2048], dtype=torch.int32)
cu_seqlens_k = cu_seqlens_q
max_seqlen_q = 512
max_seqlen_k = 512
```
</attention_workloads>

<kernel_workloads>
**RMSNorm/LayerNorm:**
```python
# Llama-style RMSNorm
hidden_sizes = {
    "7B": 4096,
    "13B": 5120, 
    "70B": 8192,
}
x = torch.randn(batch, seq_len, hidden_size, device='cuda', dtype=torch.float16)
weight = torch.ones(hidden_size, device='cuda', dtype=torch.float16)
eps = 1e-5

# Fused kernel call
output = rms_norm_kernel(x, weight, eps)
```

**Rotary Position Embeddings (RoPE):**
```python
# Standard RoPE
max_seq_len = 4096
base = 10000.0 if "llama" in model else 1000000.0  # Llama vs CodeLlama
dim = head_dim

# Precompute cos/sin cache
inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
t = torch.arange(max_seq_len, device='cuda')
freqs = torch.outer(t, inv_freq)
cos_cache = torch.cos(freqs).to(torch.float16)
sin_cache = torch.sin(freqs).to(torch.float16)

# Apply to Q/K
positions = torch.arange(seq_len, device='cuda')
q_rot = apply_rotary_pos_emb(q, cos_cache, sin_cache, positions)
```

**Activation functions (SwiGLU, GeGLU):**
```python
# SwiGLU for MLP
intermediate_size = {
    "7B": 11008,
    "13B": 13824,
    "70B": 28672,
}
x = torch.randn(batch, seq_len, hidden_size, device='cuda', dtype=torch.float16)
gate = torch.randn(batch, seq_len, intermediate_size, device='cuda', dtype=torch.float16)
up = torch.randn(batch, seq_len, intermediate_size, device='cuda', dtype=torch.float16)

# Fused SwiGLU
output = swiglu_kernel(gate, up) * x
```

**Fused MoE:**
```python
# Mixture of Experts
num_experts = 8
top_k = 2
hidden_size = 4096
expert_intermediate_size = 14336

# Router
hidden_states = torch.randn(batch * seq_len, hidden_size, device='cuda', dtype=torch.float16)
router_logits = torch.randn(batch * seq_len, num_experts, device='cuda', dtype=torch.float16)

# Expert weights
w1 = torch.randn(num_experts, hidden_size, expert_intermediate_size, device='cuda', dtype=torch.float16)
w2 = torch.randn(num_experts, expert_intermediate_size, hidden_size, device='cuda', dtype=torch.float16)

# Fused MoE kernel
output = fused_moe_kernel(hidden_states, router_logits, w1, w2, top_k)
```
</kernel_workloads>

<quantization_workloads>
**AWQ (4-bit weight-only):**
```python
# AWQ quantization
in_features = 4096
out_features = 11008
group_size = 128
bits = 4

# Quantized weights and scales
qweight = torch.randint(0, 2**bits, (out_features, in_features // (32 // bits)), 
                        device='cuda', dtype=torch.int32)
qzeros = torch.randint(0, 2**bits, (out_features, in_features // group_size // (32 // bits)),
                      device='cuda', dtype=torch.int32)
scales = torch.randn(out_features, in_features // group_size, device='cuda', dtype=torch.float16)

# Input activation
x = torch.randn(batch * seq_len, in_features, device='cuda', dtype=torch.float16)

# AWQ GEMM
output = awq_gemm(x, qweight, scales, qzeros, bits, group_size)
```

**GPTQ (4-bit with act-order):**
```python
# GPTQ with activation reordering
g_idx = torch.tensor([i // group_size for i in range(in_features)], device='cuda', dtype=torch.int32)
act_order = torch.randperm(in_features, device='cuda', dtype=torch.int32)

# Reorder before quantized GEMM
x_reordered = x[:, act_order]
output = gptq_gemm(x_reordered, qweight, scales, qzeros, g_idx)
```

**FP8 (E4M3/E5M2):**
```python
# FP8 quantization for H100/MI300
import torch.float8_e4m3fn as e4m3
import torch.float8_e5m2 as e5m2

# Quantize weights and activations
w_fp8 = weight.to(e4m3)
x_fp8 = x.to(e4m3)
scale_w = weight.abs().max() / 448.0  # E4M3 max
scale_x = x.abs().max() / 448.0

# FP8 GEMM with scaling
output = torch._scaled_mm(x_fp8, w_fp8, scale_a=scale_x, scale_b=scale_w)
```
</quantization_workloads>

<sampling_workloads>
**Top-k/Top-p sampling:**
```python
vocab_size = 32000  # Llama
batch_size = 32
temperature = 0.7

# Logits from model
logits = torch.randn(batch_size, vocab_size, device='cuda', dtype=torch.float16)

# Temperature scaling
logits = logits / temperature

# Top-k
k = 40
topk_values, topk_indices = torch.topk(logits, k, dim=-1)
topk_probs = torch.softmax(topk_values, dim=-1)

# Top-p (nucleus)
sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
sorted_indices_to_remove = cumulative_probs > 0.95
sorted_indices_to_remove[:, 1:] = sorted_indices_to_remove[:, :-1].clone()
sorted_indices_to_remove[:, 0] = 0

# Sample
probs = torch.softmax(filtered_logits, dim=-1)
next_token = torch.multinomial(probs, num_samples=1)
```

**Beam search:**
```python
beam_width = 4
vocab_size = 32000
batch_size = 8

# Beam states
beam_scores = torch.zeros(batch_size, beam_width, device='cuda')
beam_tokens = torch.zeros(batch_size, beam_width, max_len, device='cuda', dtype=torch.long)
beam_indices = torch.arange(batch_size * beam_width, device='cuda')

# Score new tokens
logits = torch.randn(batch_size * beam_width, vocab_size, device='cuda', dtype=torch.float16)
log_probs = torch.log_softmax(logits, dim=-1)

# Select top beams
next_scores = beam_scores.unsqueeze(-1) + log_probs
next_scores = next_scores.view(batch_size, beam_width * vocab_size)
next_scores, next_tokens = torch.topk(next_scores, beam_width, dim=-1)
```
</sampling_workloads>

<runtime_workloads>
**Continuous batching:**
```python
# Request queue simulation
max_batch_size = 256
max_seq_len = 2048

# Active requests with varying progress
request_lens = torch.randint(1, max_seq_len, (max_batch_size,), device='cuda')
request_positions = torch.arange(max_batch_size, device='cuda')
is_prompt = torch.rand(max_batch_size) < 0.2  # 20% prefill, 80% decode

# Batch packing
sorted_indices = torch.argsort(request_lens, descending=True)
packed_batch = pack_requests(request_lens[sorted_indices], is_prompt[sorted_indices])
```

**Paged KV cache management:**
```python
block_size = 16
num_blocks = 1024
num_layers = 32

# Block tables for each request
block_tables = torch.zeros(max_batch_size, max_blocks_per_seq, dtype=torch.int32, device='cuda')
block_usage = torch.zeros(num_blocks, dtype=torch.bool, device='cuda')

# Allocate/free blocks
def allocate_blocks(num_tokens):
    num_needed = (num_tokens + block_size - 1) // block_size
    free_blocks = torch.where(~block_usage)[0][:num_needed]
    block_usage[free_blocks] = True
    return free_blocks

def free_blocks(block_ids):
    block_usage[block_ids] = False
```
</runtime_workloads>
</workload_specifications>

<timing_requirements>
**GPU Timing with CUDA Events:**
```python
def time_gpu_operation(func, data, warmup=5, iterations=50):
    # Warmup phase - critical for stable measurements
    for _ in range(warmup):
        _ = func(data)
        torch.cuda.synchronize()
    
    # Ensure clean state
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    
    # Timing iterations
    times_ms = []
    for _ in range(iterations):
        # Fresh events per iteration (critical!)
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        
        # Double synchronization for accuracy
        torch.cuda.synchronize()
        start.record()
        
        result = func(data)
        
        end.record()
        torch.cuda.synchronize()
        
        elapsed_ms = start.elapsed_time(end)
        times_ms.append(elapsed_ms)
    
    # Calculate robust statistics
    times_ms.sort()
    avg_ms = sum(times_ms) / len(times_ms)
    p50_ms = times_ms[len(times_ms) // 2]
    p95_ms = times_ms[int(len(times_ms) * 0.95)]
    p99_ms = times_ms[int(len(times_ms) * 0.99)]
    min_ms = times_ms[0]
    max_ms = times_ms[-1]
    
    # Check for outliers
    std_ms = (sum((t - avg_ms)**2 for t in times_ms) / len(times_ms))**0.5
    if std_ms > avg_ms * 0.2:  # >20% coefficient of variation
        print(f"WARNING: High variance detected (std={std_ms:.2f}, avg={avg_ms:.2f})")
    
    return result, {
        "avg_ms": avg_ms,
        "p50_ms": p50_ms,
        "p95_ms": p95_ms,
        "p99_ms": p99_ms,
        "min_ms": min_ms,
        "max_ms": max_ms,
        "std_ms": std_ms
    }
```

**CPU Timing:**
```python
import time

def time_cpu_operation(func, data, warmup=3, iterations=10):
    # CPU timing with perf_counter
    for _ in range(warmup):
        _ = func(data)
    
    times_ms = []
    for _ in range(iterations):
        start = time.perf_counter()
        result = func(data)
        end = time.perf_counter()
        times_ms.append((end - start) * 1000)
    
    # Same statistics calculation
    return result, calculate_stats(times_ms)
```

**Iteration Count Policy:**
- GPU: minimum 50 iterations after 5 warmup
- CPU: minimum 10 iterations after 3 warmup  
- Ensure total timed work >= 200ms
- If kernel is very fast (<1ms), increase iterations to 100+
- If kernel is slow (>100ms), can reduce to 20 iterations
</timing_requirements>

<equivalence_requirements>
**Tolerance Levels:**
```python
def get_tolerances(dtype, level="numeric"):
    tolerances = {
        "exact": {
            torch.int8: (0, 0),
            torch.int32: (0, 0),
            torch.int64: (0, 0),
        },
        "numeric": {
            torch.float32: (1e-5, 1e-7),
            torch.float16: (1e-3, 1e-4),
            torch.bfloat16: (1e-2, 1e-3),
            torch.float8_e4m3fn: (5e-2, 1e-2),
            torch.float8_e5m2: (1e-1, 5e-2),
        },
        "behavioral": {
            # For stochastic operations
            torch.float32: (1e-2, 1e-3),
            torch.float16: (5e-2, 1e-2),
        }
    }
    
    level_dict = tolerances.get(level, tolerances["numeric"])
    return level_dict.get(dtype, (1e-3, 1e-4))
```

**Comprehensive Equivalence Checking:**
```python
def check_equivalence(current, reference):
    # Type checking
    assert type(current) == type(reference), f"Type mismatch: {type(current)} vs {type(reference)}"
    
    if isinstance(current, torch.Tensor):
        # Tensor equivalence
        assert current.shape == reference.shape, f"Shape: {current.shape} vs {reference.shape}"
        assert current.dtype == reference.dtype, f"Dtype: {current.dtype} vs {reference.dtype}"
        
        rtol, atol = get_tolerances(current.dtype)
        
        # Handle special values
        if torch.isnan(current).any() or torch.isnan(reference).any():
            assert torch.isnan(current).equal(torch.isnan(reference)), "NaN mismatch"
            # Compare non-NaN values
            mask = ~torch.isnan(current)
            torch.testing.assert_close(
                current[mask].cpu(),
                reference[mask].cpu(),
                rtol=rtol, atol=atol
            )
        elif torch.isinf(current).any() or torch.isinf(reference).any():
            assert torch.isinf(current).equal(torch.isinf(reference)), "Inf mismatch"
            # Compare finite values
            mask = torch.isfinite(current)
            torch.testing.assert_close(
                current[mask].cpu(),
                reference[mask].cpu(),
                rtol=rtol, atol=atol
            )
        else:
            torch.testing.assert_close(
                current.cpu(),
                reference.cpu(),
                rtol=rtol, atol=atol
            )
    
    elif isinstance(current, (list, tuple)):
        # Sequence equivalence
        assert len(current) == len(reference), f"Length: {len(current)} vs {len(reference)}"
        for i, (c, r) in enumerate(zip(current, reference)):
            try:
                check_equivalence(c, r)
            except AssertionError as e:
                raise AssertionError(f"Mismatch at index {i}: {e}")
    
    elif isinstance(current, dict):
        # Dictionary equivalence
        assert current.keys() == reference.keys(), f"Keys: {current.keys()} vs {reference.keys()}"
        for key in current:
            try:
                check_equivalence(current[key], reference[key])
            except AssertionError as e:
                raise AssertionError(f"Mismatch at key '{key}': {e}")
    
    else:
        # Scalar equivalence
        assert current == reference, f"Value: {current} vs {reference}"
```

**Reference Storage:**
```python
def store_result(result, filepath):
    # Determine best storage format
    if isinstance(result, torch.Tensor):
        # Use PyTorch native format
        torch.save({"type": "tensor", "data": result.cpu()}, filepath)
    
    elif isinstance(result, (list, tuple)) and all(isinstance(x, torch.Tensor) for x in result):
        # List of tensors
        torch.save({
            "type": "tensor_list",
            "data": [t.cpu() for t in result]
        }, filepath)
    
    elif isinstance(result, dict):
        # Complex dictionary
        torch.save({"type": "dict", "data": result}, filepath)
    
    else:
        # Fallback to pickle for complex objects
        import pickle
        with open(filepath, 'wb') as f:
            pickle.dump(result, f)

def load_result(filepath):
    if filepath.endswith('.pt'):
        data = torch.load(filepath)
        if isinstance(data, dict) and "type" in data:
            return data["data"]
        return data
    else:
        import pickle
        with open(filepath, 'rb') as f:
            return pickle.load(f)
```
</equivalence_requirements>

<hardware_requirements>
**Device Detection and Capability Checking:**
```python
def check_hardware_requirements():
    info = {}
    
    if torch.cuda.is_available():
        info["device"] = "cuda"
        info["device_name"] = torch.cuda.get_device_name()
        info["capability"] = torch.cuda.get_device_capability()
        info["memory_gb"] = torch.cuda.get_device_properties(0).total_memory / 1e9
        
        # Check for specific features
        major, minor = info["capability"]
        info["supports_fp16"] = True
        info["supports_bf16"] = major >= 8  # Ampere+
        info["supports_fp8"] = major >= 9   # Hopper+
        info["supports_flash_attn"] = major >= 7  # Volta+
        
        # Memory constraints
        if info["memory_gb"] < 16:
            print(f"WARNING: Low GPU memory ({info['memory_gb']:.1f}GB), reducing batch size")
    
    elif torch.backends.mps.is_available():
        info["device"] = "mps"
        info["device_name"] = "Apple Silicon"
    
    else:
        info["device"] = "cpu"
        info["device_name"] = "CPU"
    
    return info

def adjust_workload_for_hardware(base_config, hw_info):
    config = base_config.copy()
    
    # Adjust for memory
    if hw_info.get("memory_gb", float('inf')) < 16:
        config["batch_size"] = max(1, config["batch_size"] // 2)
        config["seq_len"] = min(1024, config["seq_len"])
    
    # Adjust for capability
    if not hw_info.get("supports_bf16", False) and config["dtype"] == torch.bfloat16:
        config["dtype"] = torch.float16
    
    if not hw_info.get("supports_fp8", False) and "fp8" in str(config["dtype"]):
        config["dtype"] = torch.float16
    
    return config
```

**Error Taxonomy:**
```python
# Standardized error codes for pipeline processing
ERROR_CODES = {
    "IMPORT_MISSING": 1,      # Target function not found
    "CAPABILITY_UNSUPPORTED": 2,  # Hardware doesn't support operation
    "OPT_PATH_NOT_TRIGGERED": 3,  # Optimization bypass detected
    "MEMORY_INSUFFICIENT": 4,      # OOM or near-OOM
    "EQUIVALENCE_FAILED": 5,       # Results don't match reference
    "INVALID_CONFIG": 6,           # Workload configuration invalid
}

def report_error(code, message):
    print(json.dumps({
        "error_code": ERROR_CODES[code],
        "error_name": code,
        "error_message": message,
        "target_resolved": code != "IMPORT_MISSING",
        "opt_path_hit": code != "OPT_PATH_NOT_TRIGGERED"
    }))
    sys.exit(ERROR_CODES[code])
```
</hardware_requirements>

<output_specification>
**Required JSON output format (single line to stdout):**
```json
{
    "impl_tag": "child",
    "commit_hash": "abc123def456",
    "device": "cuda",
    "dtype": "torch.float16",
    "iters": 50,
    "warmup": 5,
    "avg_ms": 12.345678,
    "p50_ms": 12.234567,
    "p95_ms": 13.456789,
    "eq_level": "numeric",
    "opt_path_hit": true
}
```
</output_specification>

<security_and_validation>
- Use only metadata from commit JSON - never execute arbitrary code
- Restrict imports to modules explicitly listed in changed_files
- No network operations, file downloads, or system calls
- Validate all array indices and tensor shapes before operations
- Check for integer overflows in size calculations
- Sanitize all environment variable inputs
- Use resource limits to prevent runaway allocations
</security_and_validation>

<complete_test_template>
```python
#!/usr/bin/env python3
\"\"\"
Performance test for commit: {commit_hash}
Message: {commit_message}

This script measures the actual performance impact of the optimization.
It supports cross-commit comparison with functional equivalence checking.
\"\"\"

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
    \"\"\"Resolve the optimization target from environment or metadata.\"\"\"
    
    # Priority 1: Environment variables
    module_path = os.getenv("PROB_MODULE", "")
    symbol_name = os.getenv("PROB_SYMBOL", "")
    
    # Priority 2: Parse from commit metadata
    if not (module_path and symbol_name):
        # [INSERT PARSING LOGIC BASED ON COMMIT DIFF]
        # This would be filled based on the actual commit
        pass
    
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
    \"\"\"Create realistic workload for the optimization.\"\"\"
    ensure_determinism()
    hw_info = detect_hardware()
    
    # [WORKLOAD SETUP BASED ON OPTIMIZATION TYPE]
    # This section would be filled based on the commit classification
    
    device = torch.device(hw_info["device"])
    dtype = torch.float16 if hw_info["device"] == "cuda" else torch.float32
    
    # Example workload (to be customized per commit)
    batch_size = 4
    seq_len = 2048
    hidden_size = 4096
    
    data = {
        "device": device,
        "dtype": dtype,
        "hw_info": hw_info,
        # Add tensors and parameters here
    }
    
    return data

# =======================
# Experiment Execution
# =======================
def experiment(data: Dict[str, Any]) -> Any:
    \"\"\"Execute the optimized operation.\"\"\"
    target, fq_name = resolve_target()
    
    # [CALL THE OPTIMIZED FUNCTION WITH APPROPRIATE PARAMETERS]
    # This would be customized based on the specific optimization
    
    with torch.no_grad():
        # Example call (to be replaced with actual)
        result = target(data)  # Placeholder
    
    return result

# =======================
# Result I/O
# =======================
def store_result(result: Any, filepath: str) -> None:
    \"\"\"Store result for reference comparison.\"\"\"
    if isinstance(result, torch.Tensor):
        torch.save({"type": "tensor", "data": result.cpu()}, filepath)
    else:
        torch.save({"type": "generic", "data": result}, filepath)

def load_result(filepath: str) -> Any:
    \"\"\"Load reference result.\"\"\"
    data = torch.load(filepath)
    return data.get("data", data)

# =======================
# Equivalence Checking
# =======================
def check_equivalence(current_result: Any, reference_result: Any) -> None:
    \"\"\"Verify functional equivalence.\"\"\"
    if isinstance(current_result, torch.Tensor):
        assert current_result.shape == reference_result.shape
        assert current_result.dtype == reference_result.dtype
        
        # Determine tolerances based on dtype
        if current_result.dtype in (torch.float16, torch.bfloat16):
            rtol, atol = 1e-3, 1e-4
        else:
            rtol, atol = 1e-5, 1e-7
        
        torch.testing.assert_close(
            current_result.cpu(),
            reference_result.cpu(),
            rtol=rtol, atol=atol
        )

# =======================
# Timing Implementation
# =======================
def time_gpu(func, warmup=5, iterations=50) -> Tuple[Any, Dict[str, float]]:
    \"\"\"Time GPU operations with CUDA events.\"\"\"
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
    \"\"\"Main test entry point.\"\"\"
    
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
            _ = experiment(data)
            times.append((time.perf_counter() - start) * 1000)
        times.sort()
        avg_ms = sum(times) / len(times)
        p50_ms = times[len(times) // 2]
        p95_ms = times[int(len(times) * 0.95) - 1]
        # Produce a result for reference handling
        result = experiment(data)
    
    # Reference handling
    commit_hash = os.getenv("COMMIT_HASH", "{commit_hash}")
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
```
</complete_test_template>

<final_instruction>
Using the commit information provided above, generate a complete Python test script following the template EXACTLY. The script must:

1. Parse the commit diff to identify the EXACT optimization target
2. Import the actual modified function/class (no mocks or simulations)
3. Create a workload that specifically triggers the optimized code path
4. Use proper CUDA event timing with synchronization as shown
5. Output the complete JSON metrics structure
6. Handle all error cases with proper error codes

Fill in the template sections marked with [BRACKETS] based on the commit analysis.
Output ONLY the complete Python script with no explanations or comments outside the code.
</final_instruction>
"""
```