<research_benchmark_prompt version="1.0">

  <about>
    You are GPT-5 acting as a meticulous performance engineer in a research benchmark setting (not production). 
    Your task is to generate a SINGLE Python script (prob_script) that measures the real-world performance impact of a specific commit to an LLM inference engine, while supporting cross-commit (child vs parent vs agent-variant) comparisons and strict functional equivalence.
    The output MUST be deterministic: given the same inputs, you MUST generate identical code and formatting across runs.
  </about>

  <inputs>
    <!-- These values are supplied by my pipeline. Use them deterministically. Do not improvise new fields. -->
    <commit>
      <hash>{commit_hash}</hash>
      <message><![CDATA[{commit_message}]]></message>
      <diff><![CDATA[{git_diff_patch_or_unified_diff}]]></diff>
      <!-- Optional structured metadata (if available). Prefer these over guessing; otherwise infer from diff. -->
      <changed_symbols><![CDATA[{json_changed_symbols_array_or_empty}]]></changed_symbols>
      <changed_files><![CDATA[{json_changed_files_array_or_empty}]]></changed_files>
    </commit>

    <env>
      <!-- Hints; may be empty. If provided, OBEY strictly for imports/targets. -->
      <module_hint>{optional_python_module_path_or_empty}</module_hint>
      <symbol_hint>{optional_symbol_or_qualified_attr_or_empty}</symbol_hint>
      <impl_tag>{parent|child|agent}</impl_tag>
      <commit_role>{baseline|optimized|agent_variant}</commit_role>

      <!-- Default runtime knobs (can be overridden at execution via env vars). Use to shape defaults deterministically. -->
      <default_device>{cuda|cpu|auto}</default_device>
      <default_dtype>{fp32|fp16|bf16|auto}</default_dtype>
      <eq_level_default>{numeric|exact|behavioral}</eq_level_default>
      <opt_gates><![CDATA[{json_kv_for_env_flags_or_empty}]]></opt_gates>
    </env>

    <repo_context>
      <root_path>{repo_root_or_placeholder}</root_path>
      <ecosystem>{pytorch|triton|tensorrt-llm|vllm|transformers|mlx|rocm|other}</ecosystem>
      <target_domain>LLM_inference</target_domain>
      <!-- If provided: e.g., "CUDA-only", "ROCm supported". -->
      <hardware_scope>{optional_text}</hardware_scope>
    </repo_context>
  </inputs>

  <determinism>
    <!-- Make your own generation deterministic: -->
    <rules>
      - Use minimal and consistent verbosity in the final output: ONLY emit the required Python file, nothing else.
      - Never ask clarifying questions or present alternatives; pick the best deterministic choice per tie-break rules.
      - Resolve ambiguities via the specified deterministic tie-breakers (below) in the exact stated order.
      - Use canonical code formatting: PEP8-compatible, stable import ordering (stdlib, third-party, local), stable helper order.
      - Use stable identifier names: do not randomize variable or function names.
      - Include all the requested functions in the exact order with the exact signatures.
      - Do not include platform-dependent whitespace or timestamps.
    </rules>
    <tie_breakers>
      <!-- Apply strictly in this order whenever ambiguity arises (e.g., multiple symbol candidates). -->
      1) Prefer symbols/files explicitly indicated by <module_hint>/<symbol_hint>.
      2) Otherwise, choose the symbol with the largest changed_loc from metadata.
      3) If still tied, prefer the symbol appearing in commit message.
      4) If still tied, prefer the path under the repo's primary package (e.g., shortest module path).
      5) If still tied, pick alphabetically by fully-qualified symbol name.
    </tie_breakers>
  </determinism>

  <agentic_control>
    <!-- We want rigorous but cost-aware behavior in a research pipeline. -->
    <reasoning_effort>minimal</reasoning_effort>
    <verbosity>low</verbosity>
    <context_gathering>
      Goal: extract only what is needed from the provided metadata and commit message to generate the script.
      Method:
        - Use changed_symbols and changed_files metadata to identify target functions.
        - Parse commit message for optimization details and gating flags.
        - Infer data types and computational patterns from the commit category.
      Early stop:
        - As soon as the exact performance-critical call(s) and required arguments are identified.
      Tool budget:
        - No external tools; no web, no extra files. Use only the provided metadata.
    </context_gathering>
    <persistence>
      - Do not hand back partially; produce the final script in one shot.
      - If uncertain, apply the tie-breakers and proceed. Document assumptions in code comments succinctly.
    </persistence>
  </agentic_control>

  <classification>
    <!-- Infer commit category from metadata to set up realistic workloads and equivalence criteria. -->
    <categories>
      - kernel: low-level CUDA/ROCm/Triton/TensorCore fused ops (e.g., attention, layernorm, softmax, matmul, rope, kv-cache ops, paged attention, sampling).
      - model: graph-level / module-level changes (e.g., attention API swaps, SDPA routing, cache layout, quantization, tensor parallel sharding).
      - misc: runtime/batching/scheduler/IO (e.g., request queuing, contiguous vs paged KV, pinned memory, CUDA graph capture); test the core optimized subroutine only.
    </categories>
    <policy>
      - kernel → benchmark the EXACT kernel entrypoint or the thin wrapper that maps directly to it.
      - model → isolate the modified module forward path (single-step inference or single attention block), not full text generation.
      - misc → extract the specific improved routine (e.g., batch pack/unpack) and time that; avoid queue/network/IO overhead.
    </policy>
  </classification>

  <script_requirements>
    <!-- Keep these EXACT function names / signatures and core behavior. -->
    <functions>
      setup()
      experiment(data)
      store_result(result, filepath)
      load_result(filepath)
      check_equivalence(current_result, reference_result)
      run_test(eqcheck: bool = False, reference: bool = False, prefix: str = '') -> float
    </functions>
    <imports_policy>
      - Import the actual modified/optimized target function/class based on the metadata (or hints).
      - Support dynamic import roots via env vars at runtime: PROB_MODULE, PROB_SYMBOL.
      - Use importlib and attribute walking for qualified symbols (e.g., Class.method).
      - Provide crisp ImportError messages with symbol candidates if resolution fails.
    </imports_policy>
    <workload_policy>
      - Create inputs that trigger the optimized path: correct shapes, strides, masks, dtype, causal flags, rope params, kv layout, etc.
      - Scale inputs to match the commit's intended use (e.g., prompt-len vs decode-len). Avoid degenerate sizes that skip work.
      - Random seeds: set NumPy + PyTorch (CPU/GPU); set cuDNN deterministic; disable TF32 by default unless optimization requires it.
      - Use ≤ ~70% of available device memory heuristically; cap sizes deterministically (stable formula).
    </workload_policy>
    <timing_policy>
      - GPU: CUDA events + synchronize; CPU: time.perf_counter.
      - Warmup iterations (GPU: 5, CPU: 3). Timing iterations (GPU: ≥50, CPU: ≥10) unless extremely slow; ensure ≥200ms total timed work.
      - Pre-allocate tensors/buffers in setup(); no allocations inside the timed loop.
      - Report avg ms and percentiles; return avg ms.
    </timing_policy>
    <equivalence_policy>
      - Levels: exact (bitwise or integer equality), numeric (dtype-aware tolerances), behavioral (invariants for stochastic paths).
      - Default numeric tolerances: fp32 rtol=1e-5/atol=1e-7; fp16/bf16 rtol=1e-3/atol=1e-4. Tighten/loosen only if the optimization specifies.
      - For logits/attention: check shapes/dtypes, and assert_close on tensors (or masked regions).
      - For index outputs (e.g., top-k indices), require exact equality.
      - For sampling kernels, fix seeds and compare downstream statistics + a fixed small sample.
      - Store references per implementation: {prefix}_{impl_tag}_{commit_hash}_reference.pt.
    </equivalence_policy>
    <reporting_policy>
      - Print a single JSON line summary with: impl_tag, commit_hash, device, dtype, iters, warmup, avg_ms, p50_ms, p95_ms, eq_level, opt_path_hit(bool).
      - No other prints except explicit error messages.
    </reporting_policy>
    <hardware_policy>
      - Detect device: prefer CUDA if available unless default_device overrides; otherwise CPU.
      - Probe capability needs if the optimization implies (e.g., SM version for Tensor Cores).
      - If unsupported, raise CAPABILITY_UNSUPPORTED with guidance.
    </hardware_policy>
    <security_policy>
      - Use metadata safely; do not execute arbitrary code.
      - Restrict imports to hinted/inferred modules only.
      - No networking or file downloads.
    </security_policy>
  </script_requirements>

  <output_format>
    - Output EXACTLY one Markdown code block containing the Python script. No prose before/after.
    - The script’s top docstring must embed {commit_hash} and {commit_message}.
    - Use stable section ordering: imports → dynamic resolution helpers → setup → experiment → IO → equivalence → timing → run_test.
  </output_format>

  <quality_checklist>
    - [ ] Uses PROB_MODULE/PROB_SYMBOL if provided; otherwise resolves from metadata per tie-breakers.
    - [ ] Targets the exact optimized call signature and gates.
    - [ ] Workload triggers the optimized path (e.g., sdpa with causal mask or kv-cache layout required).
    - [ ] Deterministic seeds and cuDNN settings applied.
    - [ ] Warmup excluded from timing; allocations outside loop.
    - [ ] Reference files named with {prefix}_{impl_tag}_{commit_hash}_reference.pt.
    - [ ] Only one JSON summary line printed.
    - [ ] No extra text outside the code block.
  </quality_checklist>

  <scoring_rubric>
    - Import correctness (30%): exact symbol and kwargs per metadata; robust fallback with clear error taxonomy.
    - Workload fidelity (30%): shapes/dtypes/flags match optimization; triggers hot path.
    - Timing rigor (20%): CUDA events / perf_counter correct; stable iteration counts; preallocations.
    - Equivalence depth (15%): dtype-aware tolerances; appropriate invariants.
    - Output hygiene (5%): single file, deterministic, no extraneous text.
  </scoring_rubric>

  <final_instruction>
    Produce the Python script now. Do not include any explanation or commentary. Only emit the code block. 
    The code MUST be complete and runnable as-is, with placeholders replaced by concrete imports and calls derived from the provided metadata and commit message.
  </final_instruction>

  <required_python_file>
    <![CDATA[
```python
#!/usr/bin/env python3
"""
Performance test script for commit: {commit_hash}
{commit_message}

This script measures the actual performance of the optimization described in the commit.
It supports running against different checkouts (child/parent/agent) and storing/loading
references for cross-commit equivalence checks in a research benchmark pipeline.
"""

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
    want = os.getenv("PROB_DEVICE", "{default_device}").lower()
    if want == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    if want == "cpu":
        return torch.device("cpu")
    # auto: prefer CUDA, fallback to CPU
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def pick_dtype() -> torch.dtype:
    key = os.getenv("PROB_FORCE_DTYPE", "{default_dtype}").lower()
    map_ = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16, "auto": torch.float32, "": torch.float32}
    return map_.get(key, torch.float32)

def parse_opt_gates() -> Dict[str, Any]:
    raw = os.getenv("PROB_OPT_GATES", '{opt_gates}')
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
    Deterministically infer (module, symbol) from the provided metadata and commit message.
    Returns: (module_path or None, symbol_name or None, candidate_list_for_errors)
    """
    module_hint = os.getenv("PROB_MODULE", "").strip() or "{optional_python_module_path_or_empty}".strip()
    symbol_hint = os.getenv("PROB_SYMBOL", "").strip() or "{optional_symbol_or_qualified_attr_or_empty}".strip()
    candidates = []

    # Highest priority: explicit hints
    if module_hint and symbol_hint:
        return module_hint, symbol_hint, [(module_hint, symbol_hint)]

    # Parse metadata provided by the pipeline
    commit_msg = """{commit_message}"""
    changed_files_json = """{json_changed_files_array_or_empty}"""
    changed_symbols_json = """{json_changed_symbols_array_or_empty}"""

    # 1) Prefer items listed in changed_symbols_json if present.
    try:
        import json as _json
        syms = _json.loads(changed_symbols_json) if changed_symbols_json.strip() else []
    except Exception:
        syms = []
    if syms:
        # Choose symbol with largest reported 'changed_loc' or first in order
        syms_sorted = sorted(syms, key=lambda s: (-int(s.get("changed_loc", 0)), s.get("qualified", "")))
        top = syms_sorted[0]
        mod = top.get("module", "")
        sym = top.get("qualified", "") or top.get("name", "")
        if mod and sym:
            candidates.append((mod, sym))

    # 2) Fallback: use changed_files_json to identify modules
    try:
        files = _json.loads(changed_files_json) if changed_files_json.strip() else []
    except Exception:
        files = []
    for file_path in files:
        if isinstance(file_path, str) and file_path.endswith(".py") and "/tests/" not in file_path:
            mod = file_path.replace("/", ".").rstrip(".py").rstrip(".")
            candidates.append((mod, None))

    # 3) Boost candidates mentioned in commit message
    boosted = []
    for mod, sym in candidates:
        score = 0
        if mod and mod in commit_msg:
            score += 1
        if sym and sym in commit_msg:
            score += 1
        boosted.append((score, mod, sym))
    boosted.sort(key=lambda t: (-t[0], t[1] or "", t[2] or ""))
    if boosted:
        _, mod, sym = boosted[0]
        return mod, sym, [(m, s) for _, m, s in boosted]

    return None, None, candidates

def resolve_target() -> Tuple[Callable, Dict[str, Any], str]:
    """
    Returns (callable_or_bound_method, call_kwargs, fq_name_string).
    Must resolve to the EXACT code path described by the commit metadata, honoring env hints first.
    """
    mod_hint = os.getenv("PROB_MODULE", "").strip()
    sym_hint = os.getenv("PROB_SYMBOL", "").strip()

    if mod_hint and sym_hint:
        mod_path, sym_name = mod_hint, sym_hint
        candidates = [(mod_path, sym_name)]
    else:
        mod_path, sym_name, candidates = _infer_candidates_from_diff()

    # Deterministic resolution with tie-breakers baked in above
    if not mod_path or not sym_name:
        raise ImportError(f"{E_IMPORT_MISSING}: Unable to infer target. Candidates={candidates}")

    m = importlib.import_module(mod_path)
    target = m
    for part in sym_name.split("."):
        if not hasattr(target, part):
            raise ImportError(f"{E_IMPORT_MISSING}: {mod_path}.{sym_name} not found; nearest candidates={candidates}")
        target = getattr(target, part)

    fq = f"{mod_path}.{sym_name}"
    # DEFAULT: no extra kwargs; adjust in experiment() if optimization requires
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

def setup() -> Dict[str, Any]:
    """Create realistic workload that exercises the optimization."""
    ensure_determinism()
    device = pick_device()
    dtype = pick_dtype()

    # Heuristic defaults; will be refined per category inferred from metadata
    # LLM inference defaults: matmul/attention-heavy shapes
    B = 8          # batch
    H = 32         # heads
    D = 128        # head_dim
    T_q = 128      # query length
    T_kv = 2048    # key/value length (prefill-style)
    hidden = H * D

    # Memory cap (deterministic function of defaults and device)
    bytes_per = 2 if dtype in (torch.float16, torch.bfloat16) else 4
    _ = _cap_by_memory(B * T_q * hidden, bytes_per)

    # Create representative tensors; avoid degenerate cases
    q = torch.randn((B, H, T_q, D), dtype=dtype, device=device)
    k = torch.randn((B, H, T_kv, D), dtype=dtype, device=device)
    v = torch.randn((B, H, T_kv, D), dtype=dtype, device=device)

    # Example masks for causal attention; adjusted in experiment if needed
    causal = True
    attn_mask = None  # Prefer kernel-native causal paths; set explicit masks only if required by optimization

    # Apply opt gates/environment flags
    opt_gates = parse_opt_gates()
    for k_env, v_env in opt_gates.items():
        os.environ[str(k_env)] = str(v_env)

    return {
        "device": device,
        "dtype": dtype,
        "B": B, "H": H, "D": D, "T_q": T_q, "T_kv": T_kv,
        "q": q, "k": k, "v": v,
        "causal": causal,
        "attn_mask": attn_mask,
        "opt_gates": opt_gates,
    }

# -------------------------------
# Experiment: EXACT optimized path
# -------------------------------
def experiment(data: Dict[str, Any]) -> Any:
    """
    Execute ONLY the performance-critical code path being optimized.
    Replace the call site and kwargs to match the commit diff EXACTLY.
    """
    target, call_kwargs, fqname = resolve_target()

    # Category-sensitive calling convention:
    # - kernel (e.g., fused attention/layernorm/rope): call target(q, k, v, mask/causal/scale/rope as per optimization)
    # - model  (e.g., module.forward): call target.forward(...) with tensors matching shapes from setup()
    # - misc   (e.g., pack/unpack kv-cache): call target with appropriate strides/layout flags

    with torch.no_grad():
        # !!!! REPLACE BELOW WITH THE EXACT CALL PATTERN FOR THE OPTIMIZATION !!!!
        # Examples (commented to preserve determinism and guidance):
        # result = target(data["q"], data["k"], data["v"], attn_mask=data["attn_mask"], is_causal=data["causal"], **call_kwargs)
        # result = target(data["q"], data["k"], data["v"], scale=1.0 / (data["D"] ** 0.5), **call_kwargs)
        # result = target.forward(data["q"], data["k"], data["v"], **call_kwargs)
        # result = target(data["q"])  # e.g., layernorm or rope
        #
        # Placeholder fallback (non-trivial compute to keep timing meaningful):
        q = data["q"]
        k = data["k"]
        v = data["v"]
        # scaled dot-product attention reference-ish path to avoid no-op:
        scores = torch.matmul(q, k.transpose(-1, -2)) / (data["D"] ** 0.5)
        if data["causal"]:
            # causal mask: disallow attention to future tokens
            Tq, Tkv = scores.shape[-2], scores.shape[-1]
            mask = torch.triu(torch.ones((Tq, Tkv), device=scores.device, dtype=torch.bool), diagonal=1)
            scores = scores.masked_fill(mask, float("-inf"))
        probs = torch.softmax(scores, dim=-1)
        result = torch.matmul(probs, v)
        # !!!! END REPLACE REGION !!!!

    return result

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
            "device": "cpu",  # store on CPU for portability
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
    level = os.getenv("PROB_EQ_LEVEL", "{eq_level_default}").lower()
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
    impl_tag = os.getenv("PROB_IMPL_TAG", "{parent|child|agent}")
    commit_hash = os.getenv("PROB_COMMIT_HASH", "{commit_hash}")

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

    # Summary JSON (single line)
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
        "eq_level": os.getenv("PROB_EQ_LEVEL", "{eq_level_default}"),
        "opt_path_hit": True  # Set to False if you detect fallback in experiment()
    }
    print(json.dumps(summary, sort_keys=True))
    return avg_ms

# End of script
</required_python_file>

</research_benchmark_prompt>