import os
import json
import time
import importlib
import types
from typing import Dict, Any, Tuple, Callable, Optional

import numpy as np
import torch

# -------------------------------
# Error taxonomy (deterministic)
# -------------------------------
E_IMPORT_MISSING = "IMPORT_MISSING_SYMBOL"
E_OPT_PATH_NOT_TRIGGERED = "OPT_PATH_NOT_TRIGGERED"
E_CAPABILITY = "CAPABILITY_UNSUPPORTED"
E_EQFAIL = "EQUIVALENCE_FAILED"

# -------------------------------
# Determinism & policy helpers
# -------------------------------
def ensure_determinism() -> None:
    torch.manual_seed(1234)
    np.random.seed(1234)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(1234)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    # Be strict by default unless commit requires TF32
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

def pick_device() -> torch.device:
    want = os.getenv("PROB_DEVICE", "auto").lower()
    if want == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    if want == "mps" and getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    if want == "cpu":
        return torch.device("cpu")
    # auto
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def pick_dtype() -> torch.dtype:
    key = os.getenv("PROB_FORCE_DTYPE", "auto").lower()
    map_ = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16, "auto": torch.float32, "": torch.float32}
    return map_.get(key, torch.float32)

def parse_opt_gates() -> Dict[str, Any]:
    raw = os.getenv("PROB_OPT_GATES", '{}')
    if not raw:
        return {}
    try:
        return json.loads(raw)
    except Exception:
        gates = {}
        for kv in raw.split(","):
            if "=" in kv:
                k, v = kv.split("=", 1)
                gates[k.strip()] = v.strip()
        return gates

# -------------------------------
# Diff-aware import resolution
# -------------------------------
def _infer_candidates_from_diff() -> Tuple[Optional[str], Optional[str], list]:
    """
    Deterministically infer (module, symbol) from the provided diff and commit message.
    Returns: (module_path or None, symbol_name or None, candidate_list_for_errors)
    """
    module_hint = os.getenv("PROB_MODULE", "").strip() or "".strip()
    symbol_hint = os.getenv("PROB_SYMBOL", "").strip() or "".strip()
    candidates = []

    # Highest priority: explicit hints
    if module_hint and symbol_hint:
        return module_hint, symbol_hint, [(module_hint, symbol_hint)]

    # Parse inputs (static text replacement already done upstream)
    diff_text = '''diff --git a/vllm/_custom_ops.py b/vllm/_custom_ops.py
index 462ba8a75..cae682216 100644
--- a/vllm/_custom_ops.py
+++ b/vllm/_custom_ops.py
@@ -179,7 +179,7 @@ def gptq_marlin_24_gemm(a: torch.Tensor, b_q_weight: torch.Tensor,
 
 # cutlass
 def cutlass_scaled_mm_dq(a: torch.Tensor, b: torch.Tensor,
-                         a_scales: torch.Tensor, b_scales: torch.Tensor,
+                         scale_a: torch.Tensor, scale_b: torch.Tensor,
                          out_dtype: Type[torch.dtype]) -> torch.Tensor:
     assert (b.shape[0] % 16 == 0 and b.shape[1] % 16 == 0)
     assert (out_dtype is torch.bfloat16 or out_dtype is torch.float16)
@@ -188,7 +188,7 @@ def cutlass_scaled_mm_dq(a: torch.Tensor, b: torch.Tensor,
     n = b.shape[1]
     out = torch.empty((m, n), dtype=out_dtype, device=a.device)
 
-    vllm_ops.cutlass_scaled_mm_dq(out, a, b, a_scales, b_scales)
+    vllm_ops.cutlass_scaled_mm_dq(out, a, b, scale_a, scale_b)
 
     return out
diff --git a/vllm/model_executor/layers/quantization/fp8.py b/vllm/model_executor/layers/quantization/fp8.py
index bf3a59e3d..136a64623 100644
--- a/vllm/model_executor/layers/quantization/fp8.py
+++ b/vllm/model_executor/layers/quantization/fp8.py
@@ -17,6 +17,24 @@ ACTIVATION_SCHEMES = ["static", "dynamic"]
 logger = init_logger(__name__)
 
 
+def cutlass_fp8_supported() -> bool:
+    capability = torch.cuda.get_device_capability()
+    capability = capability[0] * 10 + capability[1]
+    version = torch.version.cuda
+    version = version[0] * 10 + version[1]
+
+    # CUTLASS FP8 kernels need at least
+    #   CUDA 12.0 on SM90 systems (Hopper)
+    #   CUDA 12.4 on SM89 systems (Lovelace)
+    gpu_is_supported = False
+    if capability >= 900:
+        gpu_is_supported = version > 120
+    elif capability >= 890:
+        gpu_is_supported = version > 124
+
+    return gpu_is_supported
+
+
 class Fp8Config(QuantizationConfig):
     """Config class for FP8."""
 
@@ -92,6 +110,7 @@ class Fp8LinearMethod(LinearMethodBase):
 
     def __init__(self, quant_config: Fp8Config):
         self.quant_config = quant_config
+        self.cutlass_fp8_supported = cutlass_fp8_supported()
 
     def _create_scale_param(
         self,
@@ -233,25 +252,40 @@ class Fp8LinearMethod(LinearMethodBase):
               layer: torch.nn.Module,
               x: torch.Tensor,
               bias: Optional[torch.Tensor] = None) -> torch.Tensor:
+
         # ops.scaled_fp8_quant supports both dynamic and static quant.
         #   If dynamic, layer.act_scale is None and x_scale computed from x.
         #   If static,  layer.act_scale is scalar and x_scale set to act_scale.
-        qinput, x_scale = ops.scaled_fp8_quant(x,
-                                               layer.act_scale,
-                                               batch_dim_padding=17)
-
-        # Fused GEMM_DQ -- note we padded the input above because
-        # torch._scaled_mm is more performant for matrices with
-        # batch dimension > 16. Note that this could change
-        # in the future.
-        output, _ = torch._scaled_mm(
-            qinput,
-            layer.weight,
-            out_dtype=x.dtype,
-            scale_a=x_scale,
-            scale_b=layer.weight_scale,
-            bias=bias,
-        )
+
+        if bias is None and self.cutlass_fp8_supported:
+            qinput, x_scale = ops.scaled_fp8_quant(x, layer.act_scale)
+
+            # Fused GEMM_DQ
+            output = ops.cutlass_scaled_mm_dq(
+                qinput,
+                layer.weight,
+                out_dtype=x.dtype,
+                scale_a=x_scale,
+                scale_b=layer.weight_scale,
+            )
+
+        else:
+            qinput, x_scale = ops.scaled_fp8_quant(x,
+                                                   layer.act_scale,
+                                                   batch_dim_padding=17)
+
+            # Fused GEMM_DQ -- note we padded the input above because
+            # torch._scaled_mm is more performant for matrices with
+            # batch dimension > 16. Note that this could change
+            # in the future.
+            output, _ = torch._scaled_mm(
+                qinput,
+                layer.weight,
+                out_dtype=x.dtype,
+                scale_a=x_scale,
+                scale_b=layer.weight_scale,
+                bias=bias,
+            )
 
         return torch.narrow(output, 0, 0, x.shape[0])'''
    commit_msg = """[Kernel] Switch fp8 layers to use the CUTLASS kernels (#5183)

Switching from torch._scaled_mm to vLLM's cutlass fp8 kernels when supported as we are seeing 5-15% improvement in e2e performance on neuralmagic/Meta-Llama-3-8B-Instruct-FP8

see https://docs.google.com/spreadsheets/d/1GiAnmzyGHgZ6zL_LDSTm35Bdrt4A8AaFEurDlISYYA4/ for some quick e2e benchmarks and #5144 for comparisons across different GEMM sizes."""
    changed_files_json = """["vllm/_custom_ops.py", "vllm/model_executor/layers/quantization/fp8.py"]"""
    changed_symbols_json = """[]"""

    # 1) Prefer items listed in changed_symbols_json if present.
    try:
        import json as _json
        syms = _json.loads(changed_symbols_json) if changed_symbols_json.strip() else []
    except Exception:
        syms = []
    if syms:
        syms_sorted = sorted(syms, key=lambda s: (-int(s.get("changed_loc", 0)), s.get("qualified", "")))
        top = syms_sorted[0]
        mod = top.get("module", "")
        sym = top.get("qualified", "") or top.get("name", "")
        if mod and sym:
            candidates.append((mod, sym))

    # 2) Fallback: scan diff headers and known tokens
    for line in diff_text.splitlines():
        if line.startswith("+++ ") or line.startswith("--- "):
            path = line.split("\t")[0].split()[-1]
            if path.endswith(".py") and "/tests/" not in path:
                mod = path.replace("/", ".").rstrip(".py").rstrip(".")
                candidates.append((mod, None))
        elif line.startswith("+class ") or line.startswith("+def "):
            token = line.split()[0][1:]
            name = line.split()[1].split("(")[0].strip()
            if candidates and candidates[-1][1] is None:
                mod = candidates[-1][0]
                candidates[-1] = (mod, name)
            else:
                candidates.append((None, name))

    # Token-based deterministic hints for this commit
    if "cutlass_scaled_mm_dq" in diff_text:
        candidates.append(("vllm._custom_ops", "cutlass_scaled_mm_dq"))
    if "class Fp8LinearMethod" in diff_text:
        candidates.append(("vllm.model_executor.layers.quantization.fp8", "Fp8LinearMethod.apply"))

    # 3) Boost candidates mentioned in commit message
    boosted = []
    for mod, sym in candidates:
        score = 0
        if mod and "fp8" in mod:
            score += 1
        if mod and "cutlass" in mod:
            score += 1
        if "CUTLASS" in commit_msg or "cutlass" in commit_msg.lower():
            score += 1
        if sym and "cutlass_scaled_mm_dq" in (sym or ""):
            score += 2
        if sym and "Fp8LinearMethod.apply" in (sym or ""):
            score += 3  # Favor method with larger LOC change
        boosted.append((score, mod, sym))
    boosted.sort(key=lambda t: (-t[0], t[1] or "", t[2] or ""))
    if boosted:
        _, mod, sym = boosted[0]
        return mod, sym, [(m, s) for _, m, s in boosted]

    return None, None, candidates

def resolve_target() -> Tuple[Callable, Dict[str, Any], str]:
    """
    Returns (callable_or_bound_method, call_kwargs, fq_name_string).
    Must resolve to the EXACT code path described by the commit diff, honoring env hints first.
    """
    mod_hint = os.getenv("PROB_MODULE", "").strip()
    sym_hint = os.getenv("PROB_SYMBOL", "").strip()

    if mod_hint and sym_hint:
        mod_path, sym_name = mod_hint, sym_hint
        candidates = [(mod_path, sym_name)]
    else:
        mod_path, sym_name, candidates = _infer_candidates_from_diff()

    if not mod_path or not sym_name:
        raise ImportError(f"{E_IMPORT_MISSING}: Unable to infer target. Candidates={candidates}")

    m = importlib.import_module(mod_path)
    target = m
    for part in sym_name.split("."):
        if not hasattr(target, part):
            raise ImportError(f"{E_IMPORT_MISSING}: {mod_path}.{sym_name} not found; nearest candidates={candidates}")
        target = getattr(target, part)

    fq = f"{mod_path}.{sym_name}"
    return target, {}, fq

# -------------------------------
# Setup: workload reflecting commit
# -------------------------------
def _cap_by_memory(nelms: int, bytes_per: int, frac: float = 0.7) -> int:
    try:
        if torch.cuda.is_available():
            free, total = torch.cuda.mem_get_info()
            cap = int((total * frac) // max(bytes_per, 1))
            return min(nelms, cap)
    except Exception:
        pass
    return nelms

def _require_cuda_for_fp8() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError(f"{E_CAPABILITY}: CUDA device is required for FP8 CUTLASS kernels.")

def setup() -> Dict[str, Any]:
    """Create realistic workload that exercises the optimization."""
    ensure_determinism()
    device = pick_device()
    _require_cuda_for_fp8()
    dtype_pref = pick_dtype()
    # For FP8 path, out dtype must be fp16 or bf16; choose fp16 by default if not provided.
    out_dtype = dtype_pref if dtype_pref in (torch.float16, torch.bfloat16) else torch.float16

    # GEMM sizes (ensure multiples of 16 for B matrix dims as required by kernel)
    M = 2048  # batch (first dim)
    K = 4096  # shared dim
    N = 4096  # output dim

    # Memory cap heuristic (no-op but keeps determinism hooks)
    _ = _cap_by_memory(M * K + K * N + M * N, bytes_per=2 if out_dtype in (torch.float16, torch.bfloat16) else 4)

    # Pre-allocate inputs on device
    x = torch.randn((M, K), dtype=out_dtype, device=device)
    w_full = torch.randn((K, N), dtype=out_dtype, device=device)

    # Import ops dynamically
    try:
        ops_mod = importlib.import_module("vllm._custom_ops")
    except Exception as e:
        raise ImportError(f"{E_IMPORT_MISSING}: failed to import vllm._custom_ops: {e}")

    # Quantize weight once (dynamic per-tensor scale)
    qweight, w_scale = ops_mod.scaled_fp8_quant(w_full, scale=None)

    # Store gates/environment flags
    opt_gates = parse_opt_gates()
    for k_env, v_env in opt_gates.items():
        os.environ[str(k_env)] = str(v_env)

    data = {
        "device": device,
        "dtype": out_dtype,
        "M": M, "K": K, "N": N,
        "x": x,
        "qweight": qweight,
        "w_scale": w_scale,
        "opt_gates": opt_gates,
        "ops_mod": ops_mod,
        # Lazy-initialized for Fp8LinearMethod.apply path
        "method": None,
        "layer": None,
        "opt_path_hit": False,
    }
    return data

# -------------------------------
# Experiment: EXACT optimized path
# -------------------------------
def experiment(data: Dict[str, Any]) -> Any:
    """
    Execute ONLY the performance-critical code path being optimized.
    """
    target, call_kwargs, fqname = resolve_target()

    device = data["device"]
    x = data["x"]
    qweight = data["qweight"]
    w_scale = data["w_scale"]
    ops_mod = data["ops_mod"]
    out_dtype = data["dtype"]

    with torch.no_grad():
        # Branch based on resolved target
        if fqname.endswith("vllm._custom_ops.cutlass_scaled_mm_dq") or fqname.endswith("._custom_ops.cutlass_scaled_mm_dq"):
            # Quantize activation dynamically each call (matches Fp8LinearMethod path)
            qinput, x_scale = ops_mod.scaled_fp8_quant(x, scale=None)
            # Call CUTLASS fused GEMM_DQ
            result = target(qinput,
                            qweight,
                            out_dtype=out_dtype,
                            scale_a=x_scale,
                            scale_b=w_scale,
                            **call_kwargs)
            data["opt_path_hit"] = True
            return result

        if fqname.endswith("vllm.model_executor.layers.quantization.fp8.Fp8LinearMethod.apply") or fqname.endswith(".quantization.fp8.Fp8LinearMethod.apply"):
            # Lazily construct method and minimal layer with required attributes
            if data["method"] is None or data["layer"] is None:
                mod = importlib.import_module("vllm.model_executor.layers.quantization.fp8")
                Fp8Config = getattr(mod, "Fp8Config")
                Fp8LinearMethod = getattr(mod, "Fp8LinearMethod")
                method = Fp8LinearMethod(Fp8Config())
                # Minimal layer-like object
                layer = types.SimpleNamespace()
                layer.weight = qweight
                layer.weight_scale = w_scale
                layer.act_scale = None  # dynamic activation scaling
                data["method"] = method
                data["layer"] = layer
            method = data["method"]
            layer = data["layer"]
            # Bias must be None to trigger CUTLASS path inside apply
            result = target(method, layer, x, None)
            # We mark opt path as hit if method indicates support; apply handles fallback internally otherwise.
            data["opt_path_hit"] = bool(getattr(method, "cutlass_fp8_supported", False))
            return result

        # Fallback: this should not be reached for this commit; raise deterministically.
        raise ImportError(f"{E_IMPORT_MISSING}: Resolved unexpected target {fqname}")

# -------------------------------
# Result I/O for equivalence
# -------------------------------
def store_result(result: Any, filepath: str) -> None:
    """Store result for future equivalence checking."""
    if isinstance(result, torch.Tensor):
        payload = {
            "type": "torch_tensor",
            "shape": tuple(result.shape),
            "dtype": str(result.dtype),
            "device": "cpu",
            "sample": result.flatten()[:4096].detach().cpu(),
        }
        torch.save(payload, filepath)
    else:
        torch.save({"type": "generic", "value": result}, filepath)

def load_result(filepath: str) -> Any:
    return torch.load(filepath)

# -------------------------------
# Equivalence with dtype-aware tolerances
# -------------------------------
def _eq_tolerances(dtype: torch.dtype, level: str) -> Tuple[float, float]:
    if level == "exact":
        return (0.0, 0.0)
    if dtype in (torch.float16, torch.bfloat16):
        return (1e-3, 1e-4)
    if dtype == torch.float32:
        return (1e-5, 1e-7)
    return (1e-5, 1e-7)

def check_equivalence(current_result: Any, reference_payload: Any) -> None:
    level = os.getenv("PROB_EQ_LEVEL", "numeric").lower()
    if isinstance(current_result, torch.Tensor) and reference_payload.get("type") == "torch_tensor":
        ref_sample = reference_payload["sample"]
        assert tuple(current_result.shape) == tuple(reference_payload["shape"]), \
            f"Shape mismatch: {tuple(current_result.shape)} vs {tuple(reference_payload['shape'])}"
        assert str(current_result.dtype) == reference_payload["dtype"], \
            f"Dtype mismatch: {current_result.dtype} vs {reference_payload['dtype']}"
        rtol, atol = _eq_tolerances(current_result.dtype, level)
        torch.testing.assert_close(
            current_result.flatten()[: ref_sample.numel()].cpu(),
            ref_sample,
            rtol=rtol,
            atol=atol,
            msg=f"{E_EQFAIL}: deviation beyond tolerances (level={level})"
        )
    else:
        assert current_result == reference_payload.get("value"), f"{E_EQFAIL}: non-tensor results not equal"

# -------------------------------
# Timing utilities
# -------------------------------
def _time_gpu(run: Callable, iters: int) -> Tuple[float, float, float]:
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    times = []
    for _ in range(iters):
        start.record()
        _ = run()
        end.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end))  # ms
    times.sort()
    avg = sum(times) / len(times)
    p50 = times[len(times) // 2]
    p95 = times[int(len(times) * 0.95) - 1]
    return avg, p50, p95

def _time_cpu(run: Callable, iters: int) -> Tuple[float, float, float]:
    times = []
    for _ in range(iters):
        t0 = time.perf_counter()
        _ = run()
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1000.0)
    times.sort()
    avg = sum(times) / len(times)
    p50 = times[len(times) // 2]
    p95 = times[int(len(times) * 0.95) - 1]
    return avg, p50, p95

# -------------------------------
# Main entry: run_test
# -------------------------------
def run_test(eqcheck: bool = False, reference: bool = False, prefix: str = '') -> float:
    """
    Run the performance test and return average execution time in milliseconds.

    Args:
        eqcheck: Compare current result vs stored reference.
        reference: Store current result as reference.
        prefix: Prefix for reference filenames.

    Returns:
        Average execution time (ms).
    """
    data = setup()
    impl_tag = os.getenv("PROB_IMPL_TAG", "parent")
    commit_hash = os.getenv("PROB_COMMIT_HASH", "8d75fe48ca5f46b7af0f5201d8500b9604eed769")

    # Warmup
    warmup = 5 if torch.cuda.is_available() else 3
    for _ in range(warmup):
        _ = experiment(data)

    # Timing iterations
    iters = 50 if torch.cuda.is_available() else 10
    if torch.cuda.is_available():
        avg_ms, p50_ms, p95_ms = _time_gpu(lambda: experiment(data), iters)
    else:
        avg_ms, p50_ms, p95_ms = _time_cpu(lambda: experiment(data), iters)

    # Equivalence/reference I/O
    result = experiment(data)
    ref_path = f"{prefix}_{impl_tag}_{commit_hash}_reference.pt"
    if reference:
        store_result(result, ref_path)
    if eqcheck:
        reference_payload = load_result(ref_path)
        check_equivalence(result, reference_payload)

    summary = {
        "impl_tag": impl_tag,
        "commit_hash": commit_hash,
        "device": str(data["device"]),
        "dtype": str(data["dtype"]),
        "iters": iters,
        "warmup": warmup,
        "avg_ms": round(avg_ms, 6),
        "p50_ms": round(p50_ms, 6),
        "p95_ms": round(p95_ms, 6),
        "eq_level": os.getenv("PROB_EQ_LEVEL", "numeric"),
        "opt_path_hit": bool(data.get("opt_path_hit", False)),
    }
    print(json.dumps(summary, sort_keys=True))
    return avg_ms

# End of script