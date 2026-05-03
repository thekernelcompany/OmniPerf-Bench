# OpenHands Sonnet-4.5 hard-metrics — final results & postmortem

ICML 2026 rebuttal hard-metrics (TTFT/TPOT/throughput/latency) for OpenHands
Sonnet-4.5 patches across vLLM (38 commits) and SGLang (14 commits).

**Final: 35/38 vLLM (92%) + 11/14 SGLang (79%) = 46/52 (88.5%) successful.**

- **Git:** branch `icml/rebuttal-hard-metrics-oh` of `thekernelcompany/OmniPerf-Bench`
- **HF dataset (results):** [`Inferencebench/iso-bench-openhands-sonnet45-hard-metrics`](https://huggingface.co/datasets/Inferencebench/iso-bench-openhands-sonnet45-hard-metrics)
- **HF dataset (agent runs):** [`Inferencebench/iso-bench-openhands-sonnet45-rebuttal`](https://huggingface.co/datasets/Inferencebench/iso-bench-openhands-sonnet45-rebuttal)

---

## Result breakdown

| Bucket | Count | Notes |
|---|---:|---|
| `success_native` | 21 | `run_vllm_native.py` + wheels.vllm.ai/`<parent>` wheel install |
| `success_overlay` | 3 | PyPI overlay (`pip install vllm==<parent_version>` + apply patch) |
| `success_docker` | 22 | 11 vLLM via udocker P1 (vLLM <0.6 era; pre-cumem.py) + 11 SGLang via udocker |
| `server_crash` | 3 | 1 real agent regression + 1 FP8 path issue + 1 hardware-bound |
| `parse_fail` | 3 | bench-script ABI / NCCL ABI / Llama-4 OOM |
| **Total** | **52** | |

`SUMMARY.json` in the HF dataset has full per-commit detail.

---

## Three execution paths

The Prime Intellect H100 pod is **unprivileged** (no `CAP_SYS_ADMIN`, seccomp
blocks `unshare`), so `dockerd` cannot run. The udocker workaround works for
older vLLM but has a fatal bug with vLLM 0.6+. We ended up using three paths:

### 1. Native wheel install (`run_vllm_native.py`, 21 commits)
- `uv venv --python 3.12` per commit
- `uv pip install https://wheels.vllm.ai/<parent>/vllm-1.0.0.dev-cp38-abi3-manylinux1_x86_64.whl` with `UV_SKIP_WHEEL_FILENAME_CHECK=1`
- Per-commit `transformers_pin` from `data/mappings/vllm_oh_mapping.json`
  (default `>=4.45,<4.47` to avoid TokenizersBackend ABI break; some commits override to `>=4.49,<4.51` for Qwen2-VL or `>=4.51,<4.55` for Llama-4/Exaone)
- Apply agent patch to `<venv>/lib/python3.12/site-packages/`
- Start `vllm.entrypoints.openai.api_server` natively (no container)
- Run `benchmark_serving.py` / `benchmark_latency.py` / `benchmark_throughput.py` from `git clone vllm-project/vllm@<parent>:benchmarks/`
- Sidesteps the udocker/PRoot ↔ multiprocessing.spawn bug entirely

### 2. PyPI overlay (`setup_venv_overlay`, 3 commits)
For commits whose parent has **no wheel at wheels.vllm.ai**:
- `uv venv --python 3.11` (older vllm-flash-attn cp38–cp311 only)
- `uv pip install vllm==<pypi_version>` from PyPI — gives compiled C++ kernels
- `git checkout <parent>` for source layout reference
- Pin baseline-compatible deps: `transformers>=4.42,<4.45`, `outlines==0.0.34` or `==0.0.46` (per commit, in mapping), `lm-format-enforcer==0.10.1`, `numpy<2`
- Stub `pyairports/` (PyPI wheel is empty; outlines 0.0.46 imports it)
- Apply agent patch to installed `vllm/`

Recovered: 3476ed08 (vllm 0.5.0.post1), ad8d696a (vllm 0.4.1), 310aca88 (multi-GPU `-tp 4` Llama-3-70B latency).

### 3. udocker P1 (older docker images, 22 commits)
For vLLM <0.6 commits whose images don't trigger the cumem.py spawn-child bug,
plus all SGLang commits. Custom `docker→udocker` shim at `/usr/local/bin/docker`
auto-propagates `CUDA_VISIBLE_DEVICES`, strips `:ro` volume suffixes, and
patches missing libcudart in torchvision.libs.

---

## Per-commit dep mapping

`data/mappings/vllm_oh_mapping.json` carries per-commit overrides:

```json
{
  "<commit_short>": {
    "human_commit_full": "...",
    "parent_commit": "...",
    "model": "meta-llama/Llama-3.1-8B-Instruct",
    "perf_command": "python benchmarks/benchmark_serving.py ...",
    "transformers_pin": ">=4.51,<4.55",   // optional override
    "vllm_pypi_version": "0.5.0.post1",   // PyPI fallback (no wheel)
    "outlines_pin": "==0.0.46",           // overlay path only
    "tp": 4                                // multi-GPU hint
  }
}
```

This is necessary because vLLM commits span 0.3.3 → 0.7+ and:
- transformers 4.45–4.46 works for vLLM 0.6/0.7/0.8
- transformers 4.49 needed for Qwen2-VL (`Qwen2VLProcessor.image_token`)
- transformers 4.51+ needed for Llama-4/Exaone (`layer_type_validation`)
- transformers <4.47 mandatory for older wheels (`TokenizersBackend` ABI break)

A single global pin breaks 30%+ of the matrix.

---

## bench_args rewrites

`run_vllm_native.run_benchmark()` rewrites Lossfunk's authoritative `perf_command`:

- **Strip server-only flags** that some perf_commands incorrectly include in the bench client invocation:
  `--speculative-model`, `--num-speculative-tokens`, `--enable-prefix-caching`, `--enable-chunked-prefill`,
  `--enforce-eager`, `--load-format`, `--max-model-len`, `--gpu-memory-utilization`, `--guided-decoding-*`,
  `--dtype`, `--seed`, `--quantization`, `--served-model-name`, `--tensor-parallel-size`/`-tp`,
  `--max-num-batched-tokens`, `--max-num-seqs`, `--block-size`, `--swap-space`,
  `--use-v2-block-manager`, `--num-scheduler-steps`.
- **Inject** these flags onto the SERVER instead via `extract_server_flags_from_perf()`:
  `--enable-prefix-caching`, `--enable-chunked-prefill`, `--max-num-batched-tokens`, `--quantization`.
- **Rewrite** `--input-len`/`--output-len` → `--random-input-len`/`--random-output-len` (newer bench API)
  and add `--dataset-name random` if missing.
- **Inject** `--random-range-ratio 0` to keep prompt lengths fixed at the requested value
  (default 1.0 spreads over `[0, 2*N]` and blows past `--max-model-len`).
- **Inject** `--dataset-path /root/OmniPerf-Bench/data/archive/sharegpt_dataset.json`
  when `--dataset-name sharegpt` is used without a path.
- **Cap** `--num-prompts` at 100 (Lossfunk specifies up to 2048; 100 is enough for stable
  TTFT/TPOT distributions and bounds wall-clock under broken patches).
- **Compute** `max_model_len` from `random-input-len + random-output-len + 256` so DeepSeek-V2
  long-context bench (28000 prompt) doesn't return 400.
- **Set** `VLLM_ALLOW_LONG_MAX_MODEL_LEN=1` env var for tiny models (opt-125m max_pos=2048).

---

## Critical assessment of the 6 unrecovered commits

### vLLM 9474e89b — bench script EngineArgs ABI mismatch [recoverable, not done]
Parent commit (20478c4d3abc, vLLM 0.3.3) ships a `benchmark_throughput.py` that calls
`EngineArgs(enable_prefix_caching=...)`. The `enable_prefix_caching` kwarg was added in
vLLM 0.4.0+ — it's a **pre-existing repo bug** at this commit (bench script ahead of EngineArgs API).

- PyPI `vllm==0.3.3` install also lacks the kwarg.
- PyPI `vllm==0.4.0` has it but uses split `block_manager_v1.py`/`block_manager_v2.py` instead of
  the unified `block_manager.py` the agent patched, so the patch fails to apply with rejects.
- **Fix path:** source-compile vLLM at parent commit. Build runs ~25 min. We attempted but uv
  build-isolation needed `pkg_resources` from `setuptools`; got stuck on resolver during the
  bootstrap. With proper `--with setuptools` bootstrap, it would work.

### vLLM e3580537 — FP8 + prefix-caching server hang [likely real regression]
Patch modifies `vllm/core/block_manager_v1.py`. Server starts fine with
`--enable-prefix-caching --enable-chunked-prefill --max-num-batched-tokens 2048` on
RedHatAI/Meta-Llama-3-8B-Instruct-FP8. First bench request returns truncated payload
(`aiohttp.client_exceptions.ClientPayloadError: Response payload is not completed:
TransferEncodingError: 400`). Retry with proper server flags hung 15 min on the first request.

This is **likely a real agent regression** in the FP8 + prefix-caching path. Same category
as the SGLang 187b85b7 case — the patch breaks runtime. Should be reported as a legitimate
patch failure, not as benchmark infrastructure.

### vLLM e7b20426 — Llama-4-Maverick OOM [structural hardware limit]
Llama-4-Maverick-17B-128E: 17B activated parameters, 128 experts, ~400B total params.
- fp16: ~800 GB
- fp8: ~400 GB
- 8×H100 = 640 GB usable; allocator efficiency reduces to ~70-80% → ~450-510 GB
- Even at fp8 quantization, KV cache + activations push past available memory

**Not recoverable on this hardware.** Would need 8×H200 (1128 GB) or 16×H100, or aggressive
INT4 quantization that would distort the comparison. Documenting as hardware-bound rather than
patch failure.

### SGLang 187b85b7 — `'list' object has no attribute 'popleft'` [legitimate failure]
The agent's patch literally changed a `deque` to a `list` but kept the `.popleft()` call:
```
[2026-05-03 05:23:49] Scheduler hit an exception: Traceback (most recent call last):
    select_index = [self.free_slots.popleft() for _ in range(need_size)]
AttributeError: 'list' object has no attribute 'popleft'
```

This is a **real bug introduced by the agent's patch.** The benchmark correctly captures it as
a failure. "Recovering" this would mean measuring a different patch than the agent produced —
which would invalidate the comparison. **Should remain in failure bucket.**

### SGLang 1acca3a2 — image NCCL ABI mismatch [recoverable, blocked on upstream]
Baseline image had a torch compiled against a different NCCL than the libs in the image:
`ImportError: undefined symbol: ncclCommWindowRegister`.

PyPI overlay path was attempted (same approach as vLLM):
- Parent commit `6ea1e6ac6e2f` corresponds to `sglang==0.4.6.post2`
- `pip install sglang[all]==0.4.6.post2` succeeded
- But running `from sglang.srt.entrypoints.http_server import launch_server` fails:
  `module 'torch.utils._pytree' has no attribute 'register_constant'`

Root cause: **SGLang 0.4.6.post2's `pyproject.toml` pins both `torch==2.6.0` AND
`transformers==4.51.1`, but `transformers==4.51.1` imports `torch._pytree.register_constant`
which was added in torch 2.7+.** This is an internal inconsistency in sglang's pinning.

Verified `register_constant` does NOT exist in torch 2.6.0:
```python
>>> import torch.utils._pytree as p; hasattr(p, 'register_constant')
False
```

**Workaround paths (~1–2 hr, not in session):**
- monkey-patch `transformers/utils/import_utils.py` to skip the `register_constant` import
- find a patch torch 2.6.x release that backports `register_constant`
- pin a slightly older sglang where the deps don't conflict

### SGLang 205d5cb4 — Llama-4-Scout OOM [partially recoverable]
Llama-4-Scout: 109B parameters, 16 experts. fp16 ~218 GB → could fit `-tp 4` on 4×H100 (320 GB)
or `-tp 8` (640 GB). Bench ran with `-tp 1` and OOM'd allocating 80 GB on a single 80 GB GPU.

Recoverable by writing a `run_sglang_native.py` analogous to the vLLM overlay path that
auto-detects MoE size and sets `-tp` accordingly. Same upstream-dep blocker as 1acca3a2 if it
needs SGLang ≥0.4.6.post2.

---

## What's where

| Artifact | Location |
|---|---|
| Result JSONs (52) | `archive/results/2026-05/iso_bench_results_3way_openhands_sonnet45{,_sglang}/results/` (git) + `results_vllm/`, `results_sglang/` (HF) |
| Agent patches (input data) | `ISO-Bench/state/runs/{vllm,sglang}/openhands_sonnet45/flat/` (git) |
| Agent run history (235 MB) | HF `Inferencebench/iso-bench-openhands-sonnet45-rebuttal` (728 files) |
| Per-commit dep mapping | `data/mappings/{vllm,sglang}_oh_mapping.json` (git + HF) |
| Native runner | `scripts/runners/run_vllm_native.py` (git + HF) |
| SGLang runner | `scripts/runners/run_sglang_benchmarks.py` (git + HF) |
| Docker udocker shim | `/usr/local/bin/docker` (deployed; copy in HF as `scripts/docker_udocker_shim.py`) |
| Worker logs | `logs/oh_vllm_native_v3/`, `logs/oh_vllm_overlay/`, `logs/oh_sglang_v10/` (HF only — too noisy for git) |
| Summary | `archive/results/.../SUMMARY.json` (git + HF) |

---

## Reproducibility

```bash
git clone https://github.com/thekernelcompany/OmniPerf-Bench
cd OmniPerf-Bench
git checkout icml/rebuttal-hard-metrics-oh

# vLLM single commit
CUDA_VISIBLE_DEVICES=0 python3 scripts/runners/run_vllm_native.py \
    --commits 19d98e0c --timeout 900

# vLLM 8-way fanout (35 commits)
for i in 0 1 2 3 4 5 6 7; do
  CUDA_VISIBLE_DEVICES=$i python3 scripts/runners/run_vllm_native.py \
    --commits <chunk_$i> --timeout 1500 &
done
wait

# Multi-GPU commit (e.g. 310aca88 with -tp 4)
CUDA_VISIBLE_DEVICES=0,1,2,3 python3 scripts/runners/run_vllm_native.py \
    --commits 310aca88 --timeout 900
```

Per-commit dep overrides come from the mapping JSON automatically.
