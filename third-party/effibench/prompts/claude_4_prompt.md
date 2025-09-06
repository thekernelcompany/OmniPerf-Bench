You are an expert performance engineer specializing in LLM inference optimization testing. Your task is to analyze a commit and generate a Python script that measures its exact performance impact. Follow these instructions carefully to create the script:

1. Review the following commit information:

<commit_hash>{{COMMIT_HASH}}</commit_hash>
<commit_message>{{COMMIT_MESSAGE}}</commit_message>
<git_diff>
{{GIT_DIFF}}
</git_diff>
<changed_symbols_json>
{{CHANGED_SYMBOLS_JSON}}
</changed_symbols_json>
<changed_files_json>
{{CHANGED_FILES_JSON}}
</changed_files_json>

2. Your script MUST include these exact functions with NO modifications to signatures:

```python
def setup() -> Dict[str, Any]:
    '''Create realistic LLM workload that exercises the optimization'''
    
def experiment(data: Dict[str, Any]) -> Any:
    '''Execute ONLY the optimized code path - no wrapper logic'''
    
def store_result(result: Any, filepath: str) -> None:
    '''Serialize results for reference comparison'''
    
def load_result(filepath: str) -> Any:
    '''Deserialize reference results'''
    
def check_equivalence(current_result: Any, reference_result: Any) -> None:
    '''Assert outputs match within tolerances'''
    
def run_test(eqcheck: bool = False, reference: bool = False, prefix: str = '') -> float:
    '''Main entry - returns execution time in seconds'''
```

3. Follow these target resolution rules:
   a. Parse the changed_symbols JSON to identify the exact function/class modified
   b. Check environment variables first: PROB_MODULE and PROB_SYMBOL
   c. Import using this mapping pattern:

   Example mappings:
   - File: vllm/attention/backends/flash_attention.py
     Symbol: FlashAttentionImpl.forward
     Import: from vllm.attention.backends.flash_attention import FlashAttentionImpl
     Call: FlashAttentionImpl().forward(...)

   - File: sglang/srt/layers/attention/triton_ops/prefill_attention.py  
     Symbol: prefill_attention_triton
     Import: from sglang.srt.layers.attention.triton_ops.prefill_attention import prefill_attention_triton
     Call: prefill_attention_triton(...)

   d. If import fails, print JSON and exit:
      {"target_resolved": false, "error": "Cannot import module.symbol"}
      sys.exit(1)

4. Create appropriate workloads based on the optimization type:

   For attention operations (Flash, Paged, GQA):
   - Prefill: batch=4, seq_len=2048, heads=32, head_dim=128
   - Decode: batch=64, query_len=1, kv_cache_len=1024

   For kernels (RMSNorm, RoPE, SwiGLU):
   - hidden_size=4096 (7B) or 8192 (70B)
   - seq_len=512 (vary: 1, 128, 512, 2048)

   For quantization (AWQ, GPTQ):
   - weight_shape=(4096, 4096) or (4096, 11008)
   - group_size=128, bits=4 or 8

   Use torch.float16 for GPU, proper CUDA event timing with synchronization.

5. Implement proper CUDA event timing:
   a. Create fresh events per iteration
   b. Synchronize before and after recording
   c. Minimum 50 iterations after 5 warmup runs

6. Print EXACTLY ONE JSON line to stdout with these fields:
   - commit_hash, target, target_resolved, opt_path_hit
   - device, dtype, batch_size, seq_len
   - avg_ms, p50_ms, p95_ms, min_ms, max_ms

7. Generate a complete Python test script that:
   a. Imports the ACTUAL optimized function (no mocks)
   b. Creates workloads that trigger the optimization
   c. Uses proper CUDA event timing
   d. Outputs a single JSON line with metrics
   e. Exits with an error if import fails

Output ONLY the complete Python script with no additional explanation. Ensure that all required functions are included and that the script follows all the specified requirements.