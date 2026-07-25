# yX4G — Dataset composition (generated from dataset JSONL fields)

## vllm (39 tasks)

- **Files touched per task:** min 1, p25 1, median 2, p75 4, max 19
- **Edited lines per task:** min 2, p25 13, median 34, p75 117, max 1721
- **Non-test edited lines per task:** min 2, p25 7, median 32, p75 107, max 1634
- **Hunks per task:** min 1, p25 2, median 4, p75 12, max 107
- **Benchmark mode:** serving 30, latency 7, throughput 2 (non-exclusive flags)
- **Tasks with lm-eval correctness commands:** 5/39
- **Distinct benchmark models:** 22; top: meta-llama/Llama-3.1-8B-Instruct (20), facebook/opt-125m (2), neuralmagic/Meta-Llama-3-8B-Instruct-FP8 (2), ibm-ai-platform/Bamba-9B-v2 (2), meta-llama/Llama-3-8B (1), meta-llama/Meta-Llama-3-8B (1), meta-llama/Meta-Llama-3-70B (1), Qwen/Qwen1.5-0.5B (1)
- **Most-touched code areas (top-2-level dirs):** `vllm/v1` (29), `vllm/model_executor` (23), `vllm/core` (22), `tests/core` (12), `tests/v1` (6), `vllm/entrypoints` (5), `vllm/worker` (4), `vllm/sequence.py` (4), `tests/kernels` (4), `vllm/utils.py` (3)

## sglang (15 tasks)

- **Files touched per task:** min 1, p25 1, median 2, p75 2, max 3
- **Edited lines per task:** min 2, p25 24, median 47, p75 83, max 308
- **Non-test edited lines per task:** min 2, p25 24, median 47, p75 83, max 308
- **Hunks per task:** min 1, p25 2, median 3, p75 6, max 12
- **Benchmark mode:** serving 14, latency 1, throughput 0 (non-exclusive flags)
- **Tasks with lm-eval correctness commands:** 1/15
- **Distinct benchmark models:** 4; top: meta-llama/Llama-3.1-8B-Instruct (11), deepseek-ai/DeepSeek-V3 (2), meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8 (1), deepseek-ai/DeepSeek-V3-0324 (1)
- **Most-touched code areas (top-2-level dirs):** `python/sglang` (21), `scripts/ci_install_dependency.sh` (1), `test/srt` (1), `benchmark/kernels` (1), `docs/backend` (1)

_Note: `performance_areas` / `difficulty` are not stored fields; bottleneck-category labels would require a fresh labeling pass (partial start: archive/misc/results/reviews/vllm_classification_review.csv)._
