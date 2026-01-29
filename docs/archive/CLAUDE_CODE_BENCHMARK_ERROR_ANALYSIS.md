# Claude Code vLLM Benchmark Error Analysis

This document contains the raw error analysis from running 94 vLLM performance optimization benchmarks with Claude Code (claude-sonnet-4).

**Benchmark Date:** January 2, 2026
**Total Commits:** 94
**Successful:** 24 (25.5%)
**Failed:** 70 (74.5%)

---

## Summary by Error Category

| Category | Count | Description |
|----------|-------|-------------|
| SUCCESS | 24 | Full 3-way comparison completed |
| S3_WHEEL_404 | 24 | No pre-built wheel for old commits on S3 |
| BASELINE_NO_METRICS | 17 | Benchmark ran but produced no parseable output |
| BROKEN_PIPE | 10 | Container crashed immediately |
| SERVER_STARTUP_FAILED | 6 | vLLM server failed to initialize |
| VLLM_0.6.X_PORT_BUG | 5 | Known vLLM issue #8791 - cannot be fixed |
| OTHER | 5 | Various: blocked models, git timeout, missing command |
| WHEEL_INSTALL_FAILED | 2 | Wheel download/compilation error |
| NO_PERF_COMMAND | 1 | Missing benchmark command in dataset |

---

## Successful Commits (24)

| Commit | Model | vLLM Version |
|--------|-------|--------------|
| 299ebb62 | Qwen/Qwen2.5-1.5B-Instruct | 0.8.5.dev38 |
| 30172b49 | meta-llama/Llama-3.1-8B-Instruct | 0.10.0rc2.dev26 |
| 310aca88 | meta-llama/Meta-Llama-3-70B | 0.6.6.post2.dev145 |
| 3b61cb45 | meta-llama/Llama-3.1-8B-Instruct | 0.6.4.post2.dev277 |
| 4c822298 | meta-llama/Llama-3.1-8B-Instruct | - |
| 58eee5f2 | meta-llama/Llama-3.1-8B-Instruct | 0.6.2 |
| 61b8cea3 | meta-llama/Llama-3.2-3B-Instruct | 0.10.0rc3.dev11 |
| 6a417b86 | meta-llama/Llama-3.1-8B-Instruct | 0.7.3 |
| 6d0734c5 | mistralai/Mistral-7B-Instruct-v0.3 | 0.9.2rc2.dev356 |
| 6dd94dbe | meta-llama/Meta-Llama-3-8B | 0.6.6.post2.dev364 |
| 70b808fe | Qwen/Qwen2-VL-7B | 0.6.6.post1 |
| 8a4e5c5f | meta-llama/Llama-3.1-8B-Instruct | 0.10.0rc2.dev12 |
| 8c1e77fb | meta-llama/Llama-3.1-8B-Instruct | 0.6.4.post2.dev181 |
| 98f47f2a | unknown | 0.8.5.post1.dev8 |
| a3223766 | facebook/opt-125m | 0.10.0rc2.dev36 |
| b55ed6ef | meta-llama/Llama-3.1-8B-Instruct | 0.7.0.dev119 |
| b690e348 | ibm-ai-platform/Bamba-9B-v2 | 0.9.2.dev83 |
| bc7c4d20 | meta-llama/Llama-3.1-8B-Instruct | 0.7.0.dev71 |
| ce6bf3a2 | google/gemma-2b | - |
| ed250545 | meta-llama/Llama-3.1-8B-Instruct | 0.10.0rc2.dev38 |
| f26c4aee | meta-llama/Llama-3.1-8B-Instruct | 0.6.6.dev17 |
| fa63e710 | meta-llama/Meta-Llama-3-8B | 0.8.5.post1.dev23 |
| fc542144 | meta-llama/Llama-3.1-8B-Instruct | 0.8.4.dev144 |
| fe66b347 | ibm-ai-platform/Bamba-9B | 0.9.1 |

---

## Raw Error Data for Failed Commits

### 1. BROKEN PIPE ERRORS (10 commits)

Container crashed immediately before any work started. Duration: ~0.00003s each.

#### 015069b0
- **Status:** exception
- **Model:** Qwen/Qwen3-7B-Instruct
- **Perf Command:** `python benchmarks/benchmark_serving.py --model Qwen/Qwen3-7B-Instruct --dataset-name sharegpt --request-rate 1`
- **Duration:** 0.00003s
- **Error:**
```
[Errno 32] Broken pipe
```

#### 296f927f
- **Status:** exception
- **Model:** ibm-ai-platform/Bamba-9B
- **Perf Command:** `python benchmarks/benchmark_serving.py --model ibm-ai-platform/Bamba-9B --dtype float16 --num-prompts 300 --seed 0`
- **Duration:** 0.00004s
- **Error:**
```
[Errno 32] Broken pipe
```

#### 67da5720
- **Status:** exception
- **Model:** Qwen/Qwen2.5-7B-Instruct
- **Perf Command:** `python benchmarks/benchmark_serving.py --model Qwen/Qwen2.5-7B-Instruct --dtype float16 --num-prompts 300 --seed 0`
- **Duration:** 0.00003s
- **Error:**
```
[Errno 32] Broken pipe
```

#### 7661e92e
- **Status:** exception
- **Model:** nvidia/Nemotron-4-340B-Instruct
- **Perf Command:** `python benchmarks/benchmark_serving.py --model nvidia/Nemotron-4-340B-Instruct --dataset-name sharegpt --request-rate 1`
- **Duration:** 0.00003s
- **Error:**
```
[Errno 32] Broken pipe
```

#### 99abb8b6
- **Status:** exception
- **Model:** meta-llama/Llama-3.1-8B-Instruct
- **Perf Command:** `python benchmarks/benchmark_serving.py --model meta-llama/Llama-3.1-8B-Instruct --dataset-name sharegpt --num-prompts 1000`
- **Duration:** 0.00003s
- **Error:**
```
[Errno 32] Broken pipe
```

#### c0569dbc
- **Status:** exception
- **Model:** Qwen/Qwen3-30B-A3B-FP8
- **Perf Command:** `python benchmarks/benchmark_serving.py --model Qwen/Qwen3-30B-A3B-FP8 --dtype float16 --num-prompts 300 --seed 0`
- **Duration:** 0.00003s
- **Error:**
```
[Errno 32] Broken pipe
```

#### ca7a2d5f
- **Status:** exception
- **Model:** deepseek-ai/DeepSeek-Coder-V2-Lite-Instruct
- **Perf Command:** `python benchmarks/benchmark_serving.py --model deepseek-ai/DeepSeek-Coder-V2-Lite-Instruct --dtype float16 --num-prompts 300 --seed 0`
- **Duration:** 0.00003s
- **Error:**
```
[Errno 32] Broken pipe
```

#### ccf02fcb
- **Status:** exception
- **Model:** ibm-ai-platform/Bamba-9B
- **Perf Command:** `python benchmarks/benchmark_serving.py --model ibm-ai-platform/Bamba-9B --dtype float16 --num-prompts 300 --seed 0`
- **Duration:** 0.00003s
- **Error:**
```
[Errno 32] Broken pipe
```

#### dcc6cfb9
- **Status:** exception
- **Model:** Qwen/Qwen3-30B-A3B-FP8
- **Perf Command:** `python benchmarks/benchmark_serving.py --model Qwen/Qwen3-30B-A3B-FP8 --dtype float16 --num-prompts 300 --seed 0`
- **Duration:** 0.00003s
- **Error:**
```
[Errno 32] Broken pipe
```

#### e7b20426
- **Status:** exception
- **Model:** 01-ai/Yi-1.5-9B-Chat
- **Perf Command:** `python benchmarks/benchmark_serving.py --model 01-ai/Yi-1.5-9B-Chat --dtype float16 --num-prompts 300 --seed 0`
- **Duration:** 0.00003s
- **Error:**
```
[Errno 32] Broken pipe
```

---

### 2. S3 WHEEL 404 ERRORS (24 commits)

No pre-built wheel available on S3 for old commits.

#### 2a052011
- **Status:** error
- **Model:** nm-testing/Mixtral-8x7B-Instruct-v0.1-FP8
- **Perf Command:** `python benchmarks/benchmark_throughput.py --model nm-testing/Mixtral-8x7B-Instruct-v0.1-FP8`
- **Duration:** 1059.83s
- **Error:**
```
Baseline install failed: No wheel available and no ancestor wheel found for 36fb68f94792
```

#### 2f192835
- **Status:** error
- **Model:** meta-llama/Llama-3.1-8B-Instruct
- **Perf Command:** `python benchmarks/benchmark_serving.py --model meta-llama/Llama-3.1-8B-Instruct --backend vllm --num-prompts 100`
- **Duration:** 874.41s
- **Error:**
```
Baseline install failed: No wheel available and no ancestor wheel found for 95baec828f3e
```

#### 3476ed08
- **Status:** error
- **Model:** meta-llama/Llama-3.1-8B-Instruct
- **Perf Command:** `python benchmarks/benchmark_serving.py --model meta-llama/Llama-3.1-8B-Instruct --dtype float16 --num-prompts 300 --seed 0`
- **Duration:** 2412.49s
- **Error:**
```
Baseline install failed: No wheel available and no ancestor wheel found for 54600709b6d4
```

#### 3a243095
- **Status:** error
- **Model:** meta-llama/Llama-3.1-8B-Instruct
- **Perf Command:** `python benchmarks/benchmark_serving.py --model meta-llama/Llama-3.1-8B-Instruct --backend vllm --num-prompts 100`
- **Duration:** 845.24s
- **Error:**
```
Baseline install failed: No wheel available and no ancestor wheel found for 64172a976c8d
```

#### 660470e5
- **Status:** error
- **Model:** meta-llama/Llama-3.1-8B-Instruct
- **Perf Command:** `python benchmarks/benchmark_serving.py --model meta-llama/Llama-3.1-8B-Instruct --tensor-parallel-size 1 --enable-prefix-caching --use-v2-block-manager`
- **Duration:** 2860.73s
- **Error:**
```
Baseline install failed: No wheel available and no ancestor wheel found for 8d59dbb00044
```

#### 6ce01f30
- **Status:** error
- **Model:** meta-llama/Meta-Llama-3-8B
- **Perf Command:** `python benchmarks/benchmark_serving.py --model meta-llama/Meta-Llama-3-8B --backend vllm --num-prompts 100`
- **Duration:** 2418.75s
- **Error:**
```
Baseline install failed: No wheel available and no ancestor wheel found for 6a11fdfbb8d6
```

#### 6d646d08
- **Status:** error
- **Model:** meta-llama/Meta-Llama-3-8B
- **Perf Command:** `python benchmarks/benchmark_serving.py --model meta-llama/Meta-Llama-3-8B --dataset-name sharegpt --multi-step`
- **Duration:** 3837.96s
- **Error:**
```
Baseline install failed: No wheel available and no ancestor wheel found for 95a178f86120
```

#### 6e36f4fa
- **Status:** error
- **Model:** meta-llama/Llama-3.1-8B-Instruct
- **Perf Command:** `python benchmarks/benchmark_serving.py --model meta-llama/Llama-3.1-8B-Instruct --backend vllm --num-prompts 100`
- **Duration:** 3063.73s
- **Error:**
```
Baseline install failed: No wheel available and no ancestor wheel found for dd2a6a82e3f4
```

#### 7c01f706
- **Status:** error
- **Model:** meta-llama/Llama-3.1-8B-Instruct
- **Perf Command:** `python benchmarks/benchmark_serving.py --model meta-llama/Llama-3.1-8B-Instruct --dtype float16 --num-prompts 300 --seed 0`
- **Duration:** 2673.78s
- **Error:**
```
Baseline install failed: No wheel available and no ancestor wheel found for 51e971d39e12
```

#### 80aa7e91
- **Status:** error
- **Model:** meta-llama/Llama-3.1-8B-Instruct
- **Perf Command:** `python benchmarks/benchmark_serving.py --model meta-llama/Llama-3.1-8B-Instruct --dtype float16 --num-prompts 300 --seed 0`
- **Duration:** 2155.29s
- **Error:**
```
Baseline install failed: No wheel available and no ancestor wheel found for bd43973522ea
```

#### 89a84b0b
- **Status:** error
- **Model:** Qwen/Qwen1.5-0.5B
- **Perf Command:** `python benchmarks/benchmark_serving.py --model Qwen/Qwen1.5-0.5B --backend vllm --num-prompts 2048 --input-len 1024`
- **Duration:** 2147.82s
- **Error:**
```
Baseline install failed: No wheel available and no ancestor wheel found for 084a01fd3544
```

#### 8bc68e19
- **Status:** error
- **Model:** meta-llama/Llama-3.1-8B-Instruct
- **Perf Command:** `python benchmarks/benchmark_serving.py --model meta-llama/Llama-3.1-8B-Instruct --backend vllm --num-prompts 100`
- **Duration:** 870.97s
- **Error:**
```
Baseline install failed: No wheel available and no ancestor wheel found for 0fca3cdcf265
```

#### 9474e89b
- **Status:** error
- **Model:** huggyllama/llama-7b
- **Perf Command:** `python benchmarks/benchmark_throughput.py --model huggyllama/llama-7b --dataset-name sharegpt --num-prompts 2000`
- **Duration:** 32.60s
- **Error:**
```
Baseline install failed: No wheel available and no ancestor wheel found for 20478c4d3abc
```

#### 9ed82e70
- **Status:** error
- **Model:** meta-llama/Llama-3.1-8B-Instruct
- **Perf Command:** `python benchmarks/benchmark_serving.py --model meta-llama/Llama-3.1-8B-Instruct --backend vllm --num-prompts 100`
- **Duration:** 2275.37s
- **Error:**
```
Baseline install failed: No wheel available and no ancestor wheel found for 51f8aa90ad40
```

#### ad8d696a
- **Status:** error
- **Model:** meta-llama/Llama-3.1-8B-Instruct
- **Perf Command:** `python benchmarks/benchmark_serving.py --model meta-llama/Llama-3.1-8B-Instruct --backend vllm --num-prompts 100`
- **Duration:** 868.25s
- **Error:**
```
Baseline install failed: No wheel available and no ancestor wheel found for 3d925165f2b1
```

#### b6d10354
- **Status:** error
- **Model:** meta-llama/Llama-2-70b-hf
- **Perf Command:** `python benchmarks/benchmark_latency.py --model meta-llama/Llama-2-70b-hf --dtype float16 --tensor-parallel-size 1`
- **Duration:** 879.80s
- **Error:**
```
Baseline install failed: No wheel available and no ancestor wheel found for 51c31bc10ca7
```

#### bfdb1ba5
- **Status:** error
- **Model:** meta-llama/Llama-2-7b-chat-hf
- **Perf Command:** `python /home/ray/default/vllm_public/benchmarks/benchmark_latency.py --model meta-llama/Llama-2-7b-chat-hf --batch-size 1 --output-len 2 --input-len 1000 --num-iters 1`
- **Duration:** 893.63s
- **Error:**
```
Baseline install failed: No wheel available and no ancestor wheel found for cf2f084d56a1
```

#### c45f3c3a
- **Status:** error
- **Model:** facebook/opt-13b
- **Perf Command:** `python benchmark/benchmark_latency.py --model facebook/opt-13b`
- **Duration:** 117.65s
- **Error:**
```
Baseline install failed: No wheel available and no ancestor wheel found for 7a7929abe8e2
```

#### cf2f084d
- **Status:** error
- **Model:** meta-llama/Llama-3.1-8B-Instruct
- **Perf Command:** `python benchmarks/benchmark_serving.py --model meta-llama/Llama-3.1-8B-Instruct --backend vllm --num-prompts 100`
- **Duration:** 850.17s
- **Error:**
```
Baseline install failed: No wheel available and no ancestor wheel found for f721096d48a7
```

#### d4bc1a4d
- **Status:** error
- **Model:** facebook/opt-125m
- **Perf Command:** `python benchmarks/benchmark_serving.py --model facebook/opt-125m --num-prompts 100`
- **Duration:** 96.38s
- **Error:**
```
Baseline install failed: No wheel available and no ancestor wheel found for b56b6ca0d650
```

#### d7740ea4
- **Status:** error
- **Model:** meta-llama/Llama-3.1-8B-Instruct
- **Perf Command:** `python benchmarks/benchmark_throughput.py --model meta-llama/Llama-3.1-8B-Instruct --input-len 256 --output-len 256`
- **Duration:** 1069.49s
- **Error:**
```
Baseline install failed: No wheel available and no ancestor wheel found for cc466a32903d
```

#### e3580537
- **Status:** error
- **Model:** neuralmagic/Meta-Llama-3-8B-Instruct-FP8
- **Perf Command:** `python benchmarks/benchmark_serving.py --model neuralmagic/Meta-Llama-3-8B-Instruct-FP8 --enable-prefix-caching --enable-chunked-prefill --max-num-batched-tokens 2048`
- **Duration:** 2931.08s
- **Error:**
```
Baseline install failed: No wheel available and no ancestor wheel found for f508e03e7f2d
```

#### ec3b5ce9
- **Status:** error
- **Model:** meta-llama/Llama-3.1-8B-Instruct
- **Perf Command:** `python benchmarks/benchmark_serving.py --model meta-llama/Llama-3.1-8B-Instruct --num-prompts 100`
- **Duration:** 107.72s
- **Error:**
```
Baseline install failed: No wheel available and no ancestor wheel found for 6368e777a8ea
```

#### fc7b8d1e
- **Status:** error
- **Model:** meta-llama/Llama-3.1-8B-Instruct
- **Perf Command:** `python benchmarks/benchmark_serving.py --model meta-llama/Llama-3.1-8B-Instruct --backend vllm --num-prompts 100`
- **Duration:** 2677.22s
- **Error:**
```
Baseline install failed: No wheel available and no ancestor wheel found for 67abdbb42fdb
```

---

### 3. BASELINE NO METRICS (17 commits)

Benchmark ran but produced no parseable output.

#### 22d33bac
- **Status:** baseline_failed
- **Model:** meta-llama/Llama-3.1-8B-Instruct
- **vLLM Version:** 0.8.2.dev3+gb0e96aae
- **Perf Command:** `python benchmarks/benchmark_serving.py --model meta-llama/Llama-3.1-8B-Instruct --backend vllm --num-prompts 100`
- **Duration:** 3363.25s
- **Error:**
```
Baseline benchmark produced no metrics
```

#### 22dd9c27
- **Status:** baseline_failed
- **Model:** meta-llama/Llama-3.1-8B-Instruct
- **vLLM Version:** 0.9.2rc2.dev58+ga6d795d59
- **Perf Command:** `VLLM_ATTENTION_BACKEND=TRITON_ATTN_VLLM_V1 VLLM_USE_V1=1 python benchmarks/benchmark_latency.py --model meta-llama/Llama-3.1-8B-Instruct --input-len 16000 --output-len 4 --batch-size 1`
- **Duration:** 5317.95s
- **Error:**
```
Baseline benchmark produced no metrics
```

#### 2deb029d
- **Status:** baseline_failed
- **Model:** neuralmagic/Meta-Llama-3-8B-Instruct-FP8
- **Perf Command:** `python3 benchmarks/benchmark_prefix_caching.py --model neuralmagic/Meta-Llama-3-8B-Instruct-FP8 --output-len 200 --enable-prefix-caching [--use-v2-block-manager]`
- **Duration:** 118.38s
- **Error:**
```
(no error field)
```

#### 3092375e
- **Status:** baseline_failed
- **Model:** meta-llama/Llama-3.1-8B-Instruct
- **vLLM Version:** 0.8.5.dev42+g3cd91dc95
- **Perf Command:** `python benchmarks/benchmark_serving.py --model meta-llama/Llama-3.1-8B-Instruct --backend vllm --num-prompts 100`
- **Duration:** 2721.27s
- **Error:**
```
Baseline benchmark produced no metrics
```

#### 35fad35a
- **Status:** baseline_failed
- **Model:** meta-llama/Llama-3.1-8B-Instruct
- **vLLM Version:** 0.8.3.dev36+g733e7c9e
- **Perf Command:** `python benchmarks/benchmark_serving.py --model meta-llama/Llama-3.1-8B-Instruct --backend vllm`
- **Duration:** 2504.40s
- **Error:**
```
Baseline benchmark produced no metrics
```

#### 83450458
- **Status:** baseline_failed
- **Model:** meta-llama/Llama-3.1-8B-Instruct
- **vLLM Version:** 0.6.4.dev22+g5b8a1fde.d20241016
- **Perf Command:** `python benchmarks/benchmark_latency.py --model meta-llama/Llama-3.1-8B-Instruct --speculative-model '[ngram]' --num-speculative-tokens 5 --input-len 550 --output-len 150`
- **Duration:** 1181.77s
- **Error:**
```
Baseline benchmark produced no metrics
```

#### 8aa1485f
- **Status:** baseline_failed
- **Model:** meta-llama/Llama-4-Scout-17B-16E-Instruct
- **vLLM Version:** 0.10.1.dev149+g89ac266b2
- **Perf Command:** `python benchmarks/benchmark_serving.py --model meta-llama/Llama-4-Scout-17B-16E-Instruct --tensor-parallel-size 4 --max-model-len 16384`
- **Duration:** 6572.79s
- **Error:**
```
BASELINE server failed to start
```

#### 8d75fe48
- **Status:** error
- **Model:** neuralmagic/Meta-Llama-3-8B-Instruct-FP8
- **Perf Command:** `python benchmarks/benchmark_serving.py --model neuralmagic/Meta-Llama-3-8B-Instruct-FP8 --dataset-name sharegpt --dataset-path ./ShareGPT_V3_unfiltered_cleaned_split.json`
- **Duration:** 826.04s
- **Error:**
```
Baseline benchmark produced no metrics
```

#### 93e5f3c5
- **Status:** baseline_failed
- **Model:** meta-llama/Llama-3.1-8B-Instruct
- **vLLM Version:** 0.8.3rc2.dev173+g70363bccf
- **Perf Command:** `python benchmarks/benchmark_serving.py --model meta-llama/Llama-3.1-8B-Instruct --backend vllm --num-prompts 100`
- **Duration:** 4522.12s
- **Error:**
```
Baseline benchmark produced no metrics
```

#### 9badee53
- **Status:** baseline_failed
- **Model:** meta-llama/Llama-3.2-1B-Instruct
- **vLLM Version:** 0.7.4.dev208+gbeebf474
- **Perf Command:** `python benchmarks/benchmark_serving.py --model meta-llama/Llama-3.2-1B-Instruct --dataset-path ShareGPT_V3_unfiltered_cleaned_split.json`
- **Duration:** 1670.15s
- **Error:**
```
Baseline benchmark produced no metrics
```

#### 9d72daf4
- **Status:** baseline_failed
- **Model:** meta-llama/Llama-3.1-8B-Instruct
- **vLLM Version:** 0.8.3.dev20+g6dd55af6
- **Perf Command:** `python benchmarks/benchmark_serving.py --model meta-llama/Llama-3.1-8B-Instruct --backend vllm --num-prompts 100`
- **Duration:** 2533.91s
- **Error:**
```
Baseline benchmark produced no metrics
```

#### aea94362
- **Status:** baseline_failed
- **Model:** meta-llama/Llama-3.2-1B-Instruct
- **vLLM Version:** 0.6.6.post2.dev337+g7206ce4c
- **Perf Command:** `python benchmarks/benchmark_serving.py --backend vllm --model meta-llama/Llama-3.2-1B-Instruct --dataset-name sharegpt --num-prompts 6000 --request-rate inf --max-concurrency 400`
- **Duration:** 302.35s
- **Error:**
```
Baseline benchmark produced no metrics
```

#### b10e5198
- **Status:** baseline_failed
- **Model:** meta-llama/Llama-3.1-8B-Instruct
- **vLLM Version:** 0.8.3rc2.dev22+g9bde5ba1
- **Perf Command:** `python benchmarks/benchmark_serving.py --model meta-llama/Llama-3.1-8B-Instruct --backend vllm --num-prompts 100`
- **Duration:** 4818.58s
- **Error:**
```
Baseline benchmark produced no metrics
```

#### bd6028d6
- **Status:** baseline_failed
- **Model:** RedHatAI/Llama-4-Scout-17B-16E-Instruct-FP8-dynamic
- **vLLM Version:** 0.8.3rc2.dev163
- **Perf Command:** `python benchmarks/benchmark_latency.py --model RedHatAI/Llama-4-Scout-17B-16E-Instruct-FP8-dynamic --max-model-len 8000 --tensor-parallel-size 2 --input-len 1000 --output-len 1000`
- **Duration:** 2244.36s
- **Error:**
```
Baseline benchmark produced no metrics
```

#### dae68969
- **Status:** baseline_failed
- **Model:** deepseek-ai/DeepSeek-R1
- **vLLM Version:** 0.7.4.dev281
- **Perf Command:** `VLLM_USE_V1=1 VLLM_ATTENTION_BACKEND=FLASHMLA python benchmarks/benchmark_serving.py --model deepseek-ai/DeepSeek-R1 --tensor-parallel-size 8`
- **Duration:** 6726.26s
- **Error:**
```
BASELINE server failed to start
```

#### e206b543
- **Status:** baseline_failed
- **Model:** meta-llama/Llama-3.1-8B-Instruct
- **vLLM Version:** 0.7.4.dev103+g1d35662e
- **Perf Command:** `python benchmarks/benchmark_serving.py --model meta-llama/Llama-3.1-8B-Instruct --backend vllm --num-prompts 100 --guided-decoding-backend xgrammar`
- **Duration:** 3055.22s
- **Error:**
```
Baseline benchmark produced no metrics
```

#### fb0acb6c
- **Status:** baseline_failed
- **Model:** deepseek-ai/DeepSeek-R1
- **vLLM Version:** 0.7.4.dev351+g92b0ce2a
- **Perf Command:** `python benchmarks/benchmark_throughput.py --model deepseek-ai/DeepSeek-R1 --load-format dummy --trust-remote-code --input-len 6000 --output-len 1000 --num-prompts 50 --tensor-parallel-size 8`
- **Duration:** 3332.48s
- **Error:**
```
Baseline benchmark produced no metrics
```

---

### 4. VLLM 0.6.x PORT BINDING BUG (5 commits)

Known issue #8791 - serving benchmarks fail on vLLM 0.6.x versions due to socket duplication during fork.

#### 25ebed2f
- **Status:** version_bug
- **Model:** meta-llama/Llama-3.1-8B-Instruct
- **vLLM Version:** 0.6.4.post2.dev375+gd263bd9d
- **Perf Command:** `python benchmarks/benchmark_serving.py --model meta-llama/Llama-3.1-8B-Instruct --backend vllm --num-prompts 100`
- **Duration:** 27.69s
- **Error:**
```
vLLM 0.6.4.post2.dev375+gd263bd9d has known port binding bug (issue #8791) - serving benchmarks not supported
```

#### 88693683
- **Status:** version_bug
- **Model:** meta-llama/Meta-Llama-3-8B
- **vLLM Version:** 0.6.4.post2.dev368+g6d917d0e
- **Perf Command:** `python benchmarks/benchmark_serving.py --model meta-llama/Meta-Llama-3-8B --enable-prefix-caching`
- **Duration:** 26.96s
- **Error:**
```
vLLM 0.6.4.post2.dev368+g6d917d0e has known port binding bug (issue #8791) - serving benchmarks not supported
```

#### 9323a315
- **Status:** version_bug
- **Model:** meta-llama/Llama-3.2-3B-Instruct
- **vLLM Version:** 0.6.4.post2.dev218+g3257d449
- **Perf Command:** `python benchmarks/benchmark_serving.py --model meta-llama/Llama-3.2-3B-Instruct --guided-decoding-backend xgrammar`
- **Duration:** 26.24s
- **Error:**
```
vLLM 0.6.4.post2.dev218+g3257d449 has known port binding bug (issue #8791) - serving benchmarks not supported
```

#### b2e0ad3b
- **Status:** version_bug
- **Model:** meta-llama/Llama-3.1-8B-Instruct
- **vLLM Version:** 0.6.3.post2.dev398+g4a18fd14
- **Perf Command:** `python benchmarks/benchmark_serving.py --model meta-llama/Llama-3.1-8B-Instruct --dataset-name sharegpt --num-prompts 100`
- **Duration:** 35.60s
- **Error:**
```
vLLM 0.6.3.post2.dev398+g4a18fd14 has known port binding bug (issue #8791) - serving benchmarks not supported
```

#### f092153f
- **Status:** version_bug
- **Model:** meta-llama/Llama-3.1-8B-Instruct
- **vLLM Version:** 0.6.4.post2.dev330+g1da8f0e1
- **Perf Command:** `python benchmarks/benchmark_serving.py --model meta-llama/Llama-3.1-8B-Instruct --backend vllm --num-prompts 100`
- **Duration:** 33.36s
- **Error:**
```
vLLM 0.6.4.post2.dev330+g1da8f0e1 has known port binding bug (issue #8791) - serving benchmarks not supported
```

---

### 5. SERVER STARTUP FAILED (6 commits)

Various server startup failures including OOM, CUDA graph timeout, and transformers config conflicts.

#### 0d243f2a
- **Status:** error
- **Model:** mistralai/Mixtral-8x7B-Instruct-v0.1
- **vLLM Version:** 0.7.3.dev240+g88f6ba32
- **Perf Command:** `python benchmarks/benchmark_serving.py --model mistralai/Mixtral-8x7B-Instruct-v0.1`
- **Duration:** 2192.63s
- **Error:**
```
BASELINE server failed to start. Logs:
torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 896.00 MiB. GPU 0 has a total capacity of 79.18 GiB of which 812.56 MiB is free. Process 1 has 78.38 GiB memory in use.
```

#### 0ec82edd
- **Status:** error
- **Model:** Qwen/Qwen3-30B-A3B
- **vLLM Version:** 0.10.0rc2.dev19+g005ae9be6
- **Perf Command:** `vllm bench throughput --model Qwen/Qwen3-30B-A3B --load-format dummy --input-len 1000 --output-len 100`
- **Duration:** 3963.46s
- **Error:**
```
BASELINE server failed to start. Logs:
Capturing CUDA graph shapes: 90%|████████▊ | ... (timed out during CUDA graph capture)
```

#### 9a3b8832
- **Status:** error
- **Model:** Qwen/Qwen2.5-VL-3B-Instruct
- **vLLM Version:** 0.9.2.dev225+g3014c920d
- **Perf Command:** `python benchmarks/benchmark_serving.py --model Qwen/Qwen2.5-VL-3B-Instruct`
- **Duration:** 3382.99s
- **Error:**
```
BASELINE server failed to start. Logs:
ValueError: 'aimv2' is already used by a Transformers config, pick another name.
```

#### d55e446d
- **Status:** error
- **Model:** meta-llama/Meta-Llama-3-8B
- **Perf Command:** `python benchmarks/benchmark_serving.py --model meta-llama/Meta-Llama-3-8B --batch-size 2`
- **Duration:** 6243.78s
- **Error:**
```
BASELINE server failed to start. Logs:
ValueError: 'aimv2' is already used by a Transformers config, pick another name.
(from vllm/transformers_utils/configs/ovis.py)
```

#### e493e485
- **Status:** error
- **Model:** microsoft/phi-1_5
- **Perf Command:** `python benchmarks/benchmark_serving.py --model microsoft/phi-1_5 --backend vllm --num-prompts 100`
- **Duration:** 3236.92s
- **Error:**
```
BASELINE server failed to start. Logs:
ValueError: 'aimv2' is already used by a Transformers config, pick another name.
(from vllm/transformers_utils/configs/ovis.py)
```

#### e7523c2e
- **Status:** error
- **Model:** google/gemma-3-12b-it
- **Perf Command:** `python benchmarks/benchmark_serving.py --backend openai-chat --model google/gemma-3-12b-it --endpoint /v1/chat/completions --dataset-name hf --dataset-path lmarena-ai/VisionArena-Chat --hf-split train --num-prompts 1000`
- **Duration:** 5997.98s
- **Error:**
```
BASELINE server failed to start. Logs:
ValueError: 'aimv2' is already used by a Transformers config, pick another name.
(from vllm/transformers_utils/configs/ovis.py)
```

---

### 6. WHEEL INSTALL FAILED VIA S3 404 (2 commits)

S3 URL returned 404 but the system tried to download anyway.

#### 21d93c14
- **Status:** error
- **Model:** mistralai/Mixtral-8x7B-v0.1
- **Perf Command:** `python benchmarks/benchmark_throughput.py --model mistralai/Mixtral-8x7B-v0.1 --tensor-parallel-size 8`
- **Duration:** 67.88s
- **Error:**
```
Baseline wheel install failed: Failed to install wheel:
HTTP status client error (404 Not Found) for url
(https://vllm-wheels.s3.us-west-2.amazonaws.com/f1c8520146031a650404a6ab120ee11e91c10bed/vllm-1.0.0.dev-cp38-abi3-manylinux1_x86_64.whl)
```

#### 379da6dc
- **Status:** error
- **Model:** meta-llama/Meta-Llama-3-70B
- **Perf Command:** `python benchmarks/benchmark_serving.py --model meta-llama/Meta-Llama-3-70B --dtype float8 --input-len 1000 --output-len 50`
- **Duration:** 1175.22s
- **Error:**
```
Baseline wheel install failed: Failed to install wheel:
HTTP status client error (404 Not Found) for url
(https://vllm-wheels.s3.us-west-2.amazonaws.com/ebce310b7433e050086f52ca48571807df467f50/vllm-1.0.0.dev-cp38-abi3-manylinux1_x86_64.whl)
```

---

### 7. OTHER ERRORS (6 commits)

Various other errors including blocked models, missing commands, and git timeout.

#### 3127e975
- **Status:** no_perf_command
- **Model:** N/A
- **Perf Command:** None
- **Duration:** 0.00004s
- **Error:**
```
(no error field - missing perf command in dataset)
```

#### 4fb56914
- **Status:** error
- **Model:** deepseek-ai/DeepSeek-V3-0324
- **Perf Command:** `python benchmarks/benchmark_serving.py --model deepseek-ai/DeepSeek-V3-0324 --dataset-name sharegpt --dataset-path ShareGPT_V3_unfiltered_cleaned_split.json`
- **Duration:** 3837.01s
- **Error:**
```
(no error field - blocked model)
```

#### 526de822
- **Status:** error
- **Model:** Qwen/Qwen2-7B-Instruct
- **Perf Command:** `python benchmarks/benchmark_latency.py --dtype bfloat16 --enable-chunked-prefill False --load-format dummy --batch-size BS --num-iters-warmup 2 --num-iters 5 --input-len INPUT_LEN --output-len OUTPUT_LEN --model MODEL`
- **Duration:** 3620.29s
- **Error:**
```
(no error field - malformed perf command with placeholders)
```

#### ac45c44d
- **Status:** error
- **Model:** deepseek-ai/DeepSeek-V2
- **Perf Command:** `python benchmarks/benchmark_serving.py --model deepseek-ai/DeepSeek-V2`
- **Duration:** 3822.68s
- **Error:**
```
(no error field - blocked model)
```

#### baeded25
- **Status:** error
- **Model:** deepseek-ai/DeepSeek-V3
- **Perf Command:** `python benchmarks/benchmark_serving.py --model deepseek-ai/DeepSeek-V3 --dtype float16`
- **Duration:** 3722.23s
- **Error:**
```
(no error field - blocked model)
```

#### eefbf4a6
- **Status:** error
- **Model:** Qwen/Qwen3-30B-A3B-FP8
- **vLLM Version:** 0.10.1.dev295+g88faa466d
- **Perf Command:** `python benchmarks/benchmark_latency.py --model Qwen/Qwen3-30B-A3B-FP8 --batch-size 32 --input-len 512 --output-len 128`
- **Duration:** 3272.47s
- **Error:**
```
Command '['git', 'reset', '--hard', 'HEAD']' timed out after 60 seconds
```

---

## Key Insights

### Root Causes by Fixability

| Category | Count | Fixable? | Fix Strategy |
|----------|-------|----------|--------------|
| S3_WHEEL_404 | 24 | Yes | Build from source on CPU instance |
| BASELINE_NO_METRICS | 17 | Partially | Improve output parsing, increase timeouts |
| BROKEN_PIPE | 10 | Maybe | Investigate Modal container crashes |
| SERVER_STARTUP_FAILED | 6 | Partially | OOM needs multi-GPU, aimv2 needs transformers pin |
| VLLM_0.6.X_PORT_BUG | 5 | No | Known vLLM bug, cannot fix |
| OTHER | 5 | Varies | Case-by-case |
| WHEEL_INSTALL_FAILED | 2 | Yes | URL validation before download |
| NO_PERF_COMMAND | 1 | No | Dataset issue |

### Recommended Actions

1. **S3 Wheel 404 (24 commits)**: Enable CPU-based wheel building for commits without pre-built wheels
2. **Baseline No Metrics (17 commits)**:
   - Increase benchmark timeout
   - Improve metrics extraction regex
   - Add better error logging
3. **Broken Pipe (10 commits)**: Investigate Modal container stability, possibly add retries
4. **Server Startup (6 commits)**:
   - For OOM: Use multi-GPU configs
   - For aimv2 conflict: Pin transformers version or update vLLM
5. **vLLM 0.6.x Bug (5 commits)**: Mark as known unfixable in dataset
6. **Other (5 commits)**: Fix dataset entries with malformed commands

---

## Files Referenced

- Benchmark results: `omniperf_results_3way_claude_code/vllm/*/benchmark_result.json`
- Modal benchmark code: `src/eval/modal_benchmark.py`
- Hero benchmark runner: `hero_benchmark_runner.py`
