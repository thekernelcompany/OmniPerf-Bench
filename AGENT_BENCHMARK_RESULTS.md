# Agent Benchmark Results - vLLM Performance Optimization

**Date**: January 13, 2026
**Branch**: `feature/vllm-modal-benchmarks`
**Hardware**: NVIDIA H100 80GB GPU (single GPU)
**Timeout**: 900s per benchmark

## Summary

Three AI coding agents were benchmarked on 93 vLLM performance optimization commits. Each agent was given the task of reproducing human-authored performance improvements by generating patches applied to a baseline vLLM version.

| Agent | Total Commits | Successful | Error | Timeout | Success Rate |
|-------|---------------|------------|-------|---------|--------------|
| **Codex GPT-5** | 93 | 12 | 80 | 1 | 12.9% |
| **TRAE GPT-5** | 93 | 22 | 71 | 0 | 23.7% |
| **TRAE Sonnet 4.5** | 93 | 13 | 80 | 0 | 14.0% |
| **Total** | 279 | 47 | 231 | 1 | 16.8% |

## Benchmark Methodology

### 3-Way Comparison
1. **Baseline**: Parent commit of human optimization (Docker image: `shikhar481/vllm_fixed_human_images:baseline-<parent>`)
2. **Human**: Human-authored optimized commit (Docker image: `ayushnangia16/nvidia-vllm-docker:<commit>`)
3. **Agent**: Agent-generated patch applied to baseline

### Metrics Collected
- `request_throughput_req_s`: Requests processed per second
- `output_token_throughput_tok_s`: Output tokens generated per second

### Benchmark Parameters
- 100 requests per benchmark
- Input tokens: 200 per request
- Output tokens: 64 per request
- Server startup timeout: 300s
- Benchmark timeout: 900s

## Results by Agent

### Codex GPT-5

**Agent Patches**: `perf-agents-bench/state/runs/vllm/codex/gpt-5/`
**Results Directory**: `omniperf_results_3way_codex/results/`

| Outcome | Count | Percentage |
|---------|-------|------------|
| Success | 12 | 12.9% |
| No metrics (patch applied but benchmark failed) | 66 | 71.0% |
| Server crashed | 14 | 15.1% |
| Timeout | 1 | 1.1% |

**Sample Successful Benchmarks:**
| Commit | Model | Throughput (req/s) | Token Rate (tok/s) |
|--------|-------|-------------------|-------------------|
| 9badee53 | Llama-3.2-1B-Instruct | 151.31 | 9683.57 |
| 58eee5f2 | Llama-3.1-8B-Instruct | 55.38 | 3544.06 |
| 8a4e5c5f | Llama-3.1-8B-Instruct | 54.98 | 3518.76 |
| 30172b49 | Llama-3.1-8B-Instruct | 53.17 | 3402.63 |
| 6d646d08 | Meta-Llama-3-8B | 41.35 | 2646.48 |

---

### TRAE GPT-5

**Agent Patches**: `perf-agents-bench/state/runs/vllm/trae/gpt-5/`
**Results Directory**: `omniperf_results_3way_trae_gpt5/results/`

| Outcome | Count | Percentage |
|---------|-------|------------|
| Success | 22 | 23.7% |
| No metrics (patch applied but benchmark failed) | 61 | 65.6% |
| Server crashed | 10 | 10.8% |
| Timeout | 0 | 0.0% |

**Sample Successful Benchmarks:**
| Commit | Model | Throughput (req/s) | Token Rate (tok/s) |
|--------|-------|-------------------|-------------------|
| 61b8cea3 | Llama-3.2-3B-Instruct | 76.27 | 4881.50 |
| 25ebed2f | Llama-3.1-8B-Instruct | 54.22 | 3470.05 |
| 6e36f4fa | Llama-3.1-8B-Instruct | 40.50 | 2592.00 |
| 660470e5 | Llama-3.1-8B-Instruct | 37.94 | 2428.36 |
| 296f927f | Bamba-9B | 17.93 | 1147.31 |

---

### TRAE Sonnet 4.5

**Agent Patches**: `perf-agents-bench/state/runs/vllm/trae/claude-sonnet-45/`
**Results Directory**: `omniperf_results_3way_trae_sonnet45/results/`

| Outcome | Count | Percentage |
|---------|-------|------------|
| Success | 13 | 14.0% |
| No metrics (patch applied but benchmark failed) | 71 | 76.3% |
| Server crashed | 9 | 9.7% |
| Timeout | 0 | 0.0% |

**Sample Successful Benchmarks:**
| Commit | Model | Throughput (req/s) | Token Rate (tok/s) |
|--------|-------|-------------------|-------------------|
| 89a84b0b | Qwen1.5-0.5B | 65.72 | 4205.76 |
| 99abb8b6 | Llama-3.1-8B-Instruct | 53.65 | 3433.70 |
| 93e5f3c5 | Llama-3.1-8B-Instruct | 51.35 | 3286.28 |
| 660470e5 | Llama-3.1-8B-Instruct | 37.90 | 2425.75 |
| bfdb1ba5 | Llama-2-7b-chat-hf | 36.02 | 2305.03 |

---

## Error Analysis

### Common Failure Modes

1. **No metrics in agent output (198 total, 71%)**
   - Patch applied successfully but server didn't produce benchmark metrics
   - Often due to incompatible changes or runtime errors

2. **Server crashed after applying patch (33 total, 12%)**
   - Patch caused immediate server failure on startup
   - Common causes: import errors, syntax errors, incompatible API changes

3. **Timeout (1 total, <1%)**
   - Benchmark exceeded 900s timeout
   - Usually due to extremely slow inference or hanging processes

### CUDA OOM Analysis

One commit (dae68969) failed with CUDA OOM on DeepSeek-R1 model:
- Used 79.02 GiB of 79.11 GiB available GPU memory
- Model genuinely too large for single H100 80GB
- Not fixable without multi-GPU or quantization

## File Structure

```
omniperf_results_3way_codex/
├── results/
│   ├── <commit>_agent_result.json  # 93 files
│   └── ...

omniperf_results_3way_trae_gpt5/
├── results/
│   ├── <commit>_agent_result.json  # 93 files
│   └── ...

omniperf_results_3way_trae_sonnet45/
├── results/
│   ├── <commit>_agent_result.json  # 93 files
│   └── ...
```

### Result JSON Schema

```json
{
  "human_commit": "8-char commit hash",
  "human_commit_full": "full 40-char commit hash",
  "parent_commit": "baseline commit hash",
  "model": "HuggingFace model name",
  "status": "success|error|timeout",
  "error": "error message or null",
  "duration_s": 123.45,
  "metrics": {
    "request_throughput_req_s": 50.0,
    "output_token_throughput_tok_s": 3200.0
  },
  "raw_output": "server logs...",
  "timestamp": "2026-01-13 12:00:00"
}
```

## Reproduction

### Running Agent Benchmarks

```bash
# Run Codex GPT-5 benchmarks
python scripts/runners/run_3way_benchmarks.py \
    --agent-type codex_gpt5 \
    --agent-only \
    --timeout 900

# Run TRAE GPT-5 benchmarks
python scripts/runners/run_3way_benchmarks.py \
    --agent-type trae_gpt5 \
    --agent-only \
    --timeout 900

# Run TRAE Sonnet 4.5 benchmarks
python scripts/runners/run_3way_benchmarks.py \
    --agent-type trae_sonnet45 \
    --agent-only \
    --timeout 900
```

### Agent Type Configuration

```python
AGENT_CONFIGS = {
    "codex_gpt5": "perf-agents-bench/state/runs/vllm/codex/gpt-5",
    "trae_gpt5": "perf-agents-bench/state/runs/vllm/trae/gpt-5",
    "trae_sonnet45": "perf-agents-bench/state/runs/vllm/trae/claude-sonnet-45",
}
```

## Key Observations

1. **TRAE GPT-5 performed best** with 23.7% success rate, nearly double the other agents
2. **Most failures were "no metrics"** - patches applied but didn't produce working benchmarks
3. **Server crashes were relatively rare** (10-15% depending on agent)
4. **Smaller models had higher success rates** (1B-8B parameters vs 70B+)
5. **Single H100 80GB is insufficient** for some large models (DeepSeek-R1, Llama-3-70B)

## Next Steps

- [ ] Export results to HuggingFace datasets
- [ ] Compare agent vs human performance on successful benchmarks
- [ ] Analyze patch quality differences between agents
- [ ] Generate detailed comparison reports
