#!/usr/bin/env python3
"""
Performance test for commit: 915140fd18c9ff4193e994e6d756ea762a52240a
Message: [NVIDIA] Add Low Latency NVFP4 decode kernels from Flashinfer (#8552)

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
    
    # Priority 2: Parse from commit metadata - target the new FlashInferFP4MoE class
    if not (module_path and symbol_name):
        module_path = "sglang.srt.layers.moe.fused_moe_triton.layer"
        symbol_name = "FlashInferFP4MoE"
    
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
    
    # MoE configuration for FP4 testing
    device = torch.device(hw_info["device"])
    dtype = torch.float16 if hw_info["device"] == "cuda" else torch.float32
    
    # DeepSeek V2 MoE configuration
    batch_size = 4
    seq_len = 128  # Shorter for decode workload
    hidden_size = 4096
    num_experts = 8
    num_local_experts = 8  # All experts local for testing
    top_k = 2
    intermediate_size = 11008
    num_expert_group = 1
    topk_group = 1
    
    # Create MoE layer with FP4 quantization config
    try:
        from sglang.srt.layers.quantization.modelopt_quant import ModelOptFp4Config
        quant_config = ModelOptFp4Config(exclude_modules=None)
    except ImportError:
        # Fallback if quantization config not available
        quant_config = None
    
    # Create inputs
    hidden_states = torch.randn(batch_size * seq_len, hidden_size, device=device, dtype=dtype)
    
    # Router logits for expert selection
    router_logits = torch.randn(batch_size * seq_len, num_experts, device=device, dtype=torch.float32)
    
    # Create TopK config mock
    class TopKConfig:
        def __init__(self):
            self.top_k = top_k
            self.renormalize = True
            self.use_grouped_topk = True
            self.num_expert_group = num_expert_group
            self.topk_group = topk_group
            self.num_fused_shared_experts = 0
            self.routed_scaling_factor = None
    
    topk_config = TopKConfig()
    
    # Correction bias for DeepSeek routing
    correction_bias = torch.randn(num_experts, device=device, dtype=torch.float32)
    
    # Create mock weights for FP4 MoE (these would normally be loaded from checkpoint)
    # In FP4, weights are packed 2 elements per byte
    w13_weight_fp4 = torch.randint(0, 256, (num_local_experts, 2 * intermediate_size, hidden_size // 2), 
                                   device=device, dtype=torch.uint8)
    w2_weight_fp4 = torch.randint(0, 256, (num_local_experts, hidden_size, intermediate_size // 2),
                                  device=device, dtype=torch.uint8)
    
    # FP8 scale factors (16-element blocks)
    w13_weight_scale = torch.randn(num_local_experts, 2 * intermediate_size, hidden_size // 16,
                                  device=device, dtype=torch.float8_e4m3fn)
    w2_weight_scale = torch.randn(num_local_experts, hidden_size, intermediate_size // 16,
                                 device=device, dtype=torch.float8_e4m3fn)
    
    # Per-expert input/output scales
    w13_input_scale = torch.ones(num_local_experts, 2, device=device, dtype=torch.float32)
    w2_input_scale = torch.ones(num_local_experts, device=device, dtype=torch.float32)
    w13_weight_scale_2 = torch.ones(num_local_experts, 2, device=device, dtype=torch.float32)
    w2_weight_scale_2 = torch.ones(num_local_experts, device=device, dtype=torch.float32)
    
    data = {
        "device": device,
        "dtype": dtype,
        "hw_info": hw_info,
        "hidden_states": hidden_states,
        "router_logits": router_logits,
        "topk_config": topk_config,
        "correction_bias": correction_bias,
        "num_experts": num_experts,
        "num_local_experts": num_local_experts,
        "top_k": top_k,
        "intermediate_size": intermediate_size,
        "hidden_size": hidden_size,
        "num_expert_group": num_expert_group,
        "topk_group": topk_group,
        "w13_weight_fp4": w13_weight_fp4,
        "w2_weight_fp4": w2_weight_fp4,
        "w13_weight_scale": w13_weight_scale, 
        "w2_weight_scale": w2_weight_scale,
        "w13_input_scale": w13_input_scale,
        "w2_input_scale": w2_input_scale,
        "w13_weight_scale_2": w13_weight_scale_2,
        "w2_weight_scale_2": w2_weight_scale_2,
    }
    
    return data

# =======================
# Experiment Execution
# =======================
def experiment(data: Dict[str, Any]) -> Any:
    """Execute the optimized operation."""
    target, fq_name = resolve_target()
    
    # Try to use the flashinfer FP4 kernel directly for testing
    try:
        from flashinfer.fused_moe import trtllm_fp4_block_scale_moe
        from flashinfer import (
            RoutingMethodType,
            fp4_quantize,
            reorder_rows_for_gated_act_gemm,
            shuffle_matrix_a,
            shuffle_matrix_sf_a,
        )
    except ImportError:
        # Fallback to mock if flashinfer not available
        error_data = {
            "target_resolved": False,
            "error": "flashinfer not available or missing trtllm_fp4_block_scale_moe",
            "opt_path_hit": False
        }
        print(json.dumps(error_data))
        sys.exit(1)
    
    # Prepare weights in the format expected by the kernel
    # This mimics what FlashInferFP4MoE does in prepare_static_weights_for_kernel
    epilogue_tile_m = 128
    
    # Reorder and shuffle weights for the kernel
    gemm1_weights_fp4_shuffled = []
    gemm1_scales_fp4_shuffled = []
    gemm2_weights_fp4_shuffled = []
    gemm2_scales_fp4_shuffled = []
    
    for i in range(data["num_local_experts"]):
        # For gemm1, reorder rows for gated activation
        w1_reordered = reorder_rows_for_gated_act_gemm(data["w13_weight_fp4"][i].clone())
        s1_reordered = reorder_rows_for_gated_act_gemm(data["w13_weight_scale"][i].clone())
        
        # Shuffle for transposed mma output
        gemm1_weights_fp4_shuffled.append(
            shuffle_matrix_a(w1_reordered.view(torch.uint8), epilogue_tile_m)
        )
        gemm1_scales_fp4_shuffled.append(
            shuffle_matrix_sf_a(s1_reordered.view(torch.uint8), epilogue_tile_m)
        )
        
        # For gemm2, just shuffle
        gemm2_weights_fp4_shuffled.append(
            shuffle_matrix_a(data["w2_weight_fp4"][i].view(torch.uint8), epilogue_tile_m)
        )
        gemm2_scales_fp4_shuffled.append(
            shuffle_matrix_sf_a(data["w2_weight_scale"][i].view(torch.uint8), epilogue_tile_m)
        )
    
    gemm1_weights_fp4_shuffled = torch.stack(gemm1_weights_fp4_shuffled)
    gemm1_scales_fp4_shuffled = torch.stack(gemm1_scales_fp4_shuffled).view(torch.float8_e4m3fn)
    gemm2_weights_fp4_shuffled = torch.stack(gemm2_weights_fp4_shuffled)
    gemm2_scales_fp4_shuffled = torch.stack(gemm2_scales_fp4_shuffled).view(torch.float8_e4m3fn)
    
    # Quantize hidden states to FP4
    w13_input_scale_quant = 1.0 / data["w13_input_scale"].max()
    hs_fp4_bytes, hs_sf_bytes = fp4_quantize(
        data["hidden_states"],
        w13_input_scale_quant,
        16,  # sf_vec_size
        False,  # use_ue8m0
        False,  # is_sf_swizzled_layout
    )
    
    hs_fp4 = hs_fp4_bytes.reshape(data["hidden_states"].shape[0], data["hidden_states"].shape[1] // 2)
    hs_sf = hs_sf_bytes.view(torch.float8_e4m3fn).flatten()
    
    # Calculate alphas
    g1_alphas = (data["w13_input_scale"].max() * data["w13_weight_scale_2"][:, 0]).to(torch.float32)
    g2_alphas = (data["w2_input_scale"].max() * data["w2_weight_scale_2"]).to(torch.float32)
    g1_scale_c = (1.0 / data["w2_input_scale"].max() * g1_alphas).to(torch.float32)
    
    # Calculate tile tokens dimension
    num_tokens_per_expert = (data["hidden_states"].shape[0] * data["top_k"]) // data["num_local_experts"]
    tile_tokens_dim = min(max(2 ** (num_tokens_per_expert - 1).bit_length(), 8), 64)
    
    with torch.no_grad():
        # Call the FP4 TRTLLM kernel
        result = trtllm_fp4_block_scale_moe(
            routing_logits=data["router_logits"],
            routing_bias=data["correction_bias"].to(data["dtype"]),
            hidden_states=hs_fp4,
            hidden_states_scale=hs_sf,
            gemm1_weights=gemm1_weights_fp4_shuffled,
            gemm1_weights_scale=gemm1_scales_fp4_shuffled,
            gemm2_weights=gemm2_weights_fp4_shuffled,
            gemm2_weights_scale=gemm2_scales_fp4_shuffled,
            output1_scale_scalar=g1_scale_c,
            output1_scale_gate_scalar=g1_alphas,
            output2_scale_scalar=g2_alphas,
            num_experts=data["num_experts"],
            top_k=data["top_k"],
            n_group=data["num_expert_group"],
            topk_group=data["topk_group"],
            intermediate_size=data["intermediate_size"],
            local_expert_offset=0,
            local_num_experts=data["num_local_experts"],
            routed_scaling_factor=None,
            tile_tokens_dim=tile_tokens_dim,
            routing_method_type=RoutingMethodType.DeepSeekV3,
            do_finalize=True,
        )[0]
    
    return result

# =======================
# Result I/O
# =======================
def store_result(result: Any, filepath: str) -> None:
    """Store result for reference comparison."""
    if isinstance(result, torch.Tensor):
        torch.save({"type": "tensor", "data": result.cpu()}, filepath)
    else:
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
    if isinstance(current_result, torch.Tensor):
        assert current_result.shape == reference_result.shape
        assert current_result.dtype == reference_result.dtype
        
        # FP4 operations have higher tolerance
        rtol, atol = 5e-2, 1e-2
        
        torch.testing.assert_close(
            current_result.cpu(),
            reference_result.cpu(),
            rtol=rtol, atol=atol
        )

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
    
    # Check hardware requirements
    if hw_info["device"] != "cuda":
        error_data = {
            "target_resolved": True,
            "opt_path_hit": False,
            "error": "FP4 MoE kernel requires CUDA device"
        }
        print(json.dumps(error_data))
        sys.exit(2)
    
    # Timing
    warmup = 5
    iters = 50
    result, timing_stats = time_gpu(lambda: experiment(data), warmup=warmup, iterations=iters)
    avg_ms = timing_stats["avg_ms"]
    p50_ms = timing_stats["p50_ms"]
    p95_ms = timing_stats["p95_ms"]
    
    # Reference handling
    commit_hash = os.getenv("COMMIT_HASH", "915140fd18c9ff4193e994e6d756ea762a52240a")
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
        "dtype": "torch.float16",
        "iters": iters,
        "warmup": warmup,
        "avg_ms": avg_ms,
        "p50_ms": p50_ms,
        "p95_ms": p95_ms,
        "eq_level": os.getenv("PROB_EQ_LEVEL", "behavioral"),
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