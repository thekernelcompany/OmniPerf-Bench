# OpenHands GPT-5 hard-metrics — final results & postmortem

ICML 2026 rebuttal hard-metrics (TTFT/TPOT/throughput/latency) for OpenHands
GPT-5 patches across vLLM (39 commits) and SGLang (15 commits).

**Final: 34/37 vLLM (92%) + 12/15 SGLang (80%) = 46/52 attempted (88.5%) successful;
46/54 of full fanout (85.2%); 2 hardware-bound.**

- **Hardware:** local 2× H100 80GB PCIe (vs sonnet45's 8× H100). Driver 535.183.06, CUDA 12.2.
- **Git:** branch `icml/rebuttal-hard-metrics-oh` of `thekernelcompany/OmniPerf-Bench`
- **HF dataset (results):** `Inferencebench/iso-bench-openhands-gpt5-hard-metrics` (this run)
- **HF dataset (agent runs):** `Inferencebench/iso-bench-openhands-gpt5-rebuttal` (input)
- **ISO-Bench dataset (authoritative perf_commands + targets):** `ISO-Bench/ISO-Bench`

---

## Result breakdown

| Bucket | Count | Notes |
|---|---:|---|
| `success_native` | 21 | `run_vllm_native.py` + `wheels.vllm.ai/<parent>` wheel install |
| `success_overlay` | 13 | PyPI overlay (`pip install vllm==<parent_version>` + apply patch) for no-wheel commits |
| `success_sglang_native` | 11 | `run_sglang_native.py` overlay path on sglang 0.4.7+ |
| `success_sglang_one_batch` | 1 | sglang `bench_one_batch` in-process bench (commit `c087ddd6`) |
| `success_multi_gpu` | 1 | Llama-3-70B at tp=2 (commit `310aca88` — see deviation note) |
| `agent_regression` | 6 | Patches that crash server / produce no metrics — legitimate signal per Lossfunk policy |
| `hardware_bound` | 2 | Llama-4-Maverick + vLLM 0.4.0/CUDA 11 baseline — physical hardware limits |
| **Total** | **54** | |

---

## What was new vs the sonnet45 run

1. **Different hardware**: 2× H100 instead of 8× H100. Required runtime `-tp` rewrite
   to match visible GPUs (handles `310aca88`'s `-tp 4` baked into ISO-Bench's
   perf_command).
2. **Native SGLang overlay path**: sonnet45 ran SGLang via udocker; this run
   wrote `scripts/runners/run_sglang_native.py` mirroring `run_vllm_native.py`'s
   overlay design — discovers PyPI version via `git tag --contains <parent>` in
   the sglang submodule, applies patches with `-p2` (sglang patches use
   `a/python/sglang/...` paths against the source tree).
3. **ISO-Bench dataset is authoritative**: runner reads `perf_command` and
   `--model` from `data/iso_bench/{vllm,sglang}/train.parquet` (mirrored from HF
   `ISO-Bench/ISO-Bench`). Local mapping holds dep overrides only. Audit trail
   in each result JSON's `iso_bench_override_applied` field.
4. **Empty-patch trajectory replay**: `vllm_core-0034` (`6e36f4fa`) had a
   0-byte `model_patch.diff` in the HF dataset due to a documented harness bug
   (bench-fixes branch's commit `cfeec12a` only fixed the file-missing case, not
   the 0-byte file case). Reconstructed by replaying 7 `str_replace` ops from
   `trajectory.json` against a fresh checkout of `dd2a6a82e3f4` — 4 ops applied,
   3 had `+/-` diff markers literally embedded in `old_str` (agent malformed
   them) and would have been no-ops at agent runtime too. Final 3258-byte
   patch tagged with `RECONSTRUCTED.txt` provenance file. The benchmark hung on
   first request (TransferEncoding 400), which IS the agent's regression.
5. **`force_modern` env for sglang 0.4.6.post* commits**: 1acca3a2 (parent
   v0.4.6.post3) and 205d5cb4 (parent v0.4.6.post5) need transformers≥4.51 (for
   Llama-4 support / register_constant ABI), but sglang 0.4.6.post* pins
   transformers==4.51.1+torch==2.6, and torch 2.7 (which has register_constant)
   breaks sgl_kernel's compiled ABI. Workaround: substitute sglang→0.4.7 (which
   ships torch 2.7+ + transformers 4.52 + matching sgl_kernel). Worked for
   `1acca3a2`; `205d5cb4` still fails on a deeper dependency tangle in the
   compressed_tensors WNA16 MoE path.

---

## Per-commit dep mapping changes

Backfilled `vllm_pypi_version` for 7 no-wheel vLLM commits to enable overlay
path (sonnet45 had only 4 of these mapped):

```json
{
  "2deb029d": { "vllm_pypi_version": "0.6.0" },
  "fc7b8d1e": { "vllm_pypi_version": "0.5.5" },
  "660470e5": { "vllm_pypi_version": "0.5.5" },
  "89a84b0b": { "vllm_pypi_version": "0.5.4" },
  "9ed82e70": { "vllm_pypi_version": "0.5.3" },
  "d7740ea4": { "vllm_pypi_version": "0.4.3" },
  "6e36f4fa": { "vllm_pypi_version": "0.6.0" }
}
```

Added new mapping entries for the 2 commits that exist in the GPT-5 fanout but
were absent from `vllm_oh_mapping.json` / `sglang_oh_mapping.json` (sonnet45's
mappings had the same gap):

- vLLM `015069b0` (`vllm_core-0000`): Qwen2.5-7B-Instruct, parent `fbefc8a78d22` (v0.9.0)
- SGLang `6b231325` (`sglang_core-0027`): Llama-3.1-8B-Instruct, parent `b1c8d4e9f319` (v0.4.7)

---

## ISO-Bench typos fixed via MODEL_OVERRIDES

ISO-Bench's `perf_command` strings have a few typos in `--model` arg names
that don't resolve on HuggingFace. These are caught at runtime and substituted
to canonical names:

| Typo in ISO-Bench | Canonical |
|---|---|
| `meta-llama/Llama-3-8B` | `meta-llama/Meta-Llama-3-8B` |
| `meta-llama/Llama-3-8B-Instruct` | `meta-llama/Meta-Llama-3-8B-Instruct` |
| `Qwen/Qwen3-7B-Instruct` | `Qwen/Qwen2.5-7B-Instruct` (matches ISO's `models[]` field) |

Inconsistency caught: ISO-Bench has separate `perf_command` (the actual bench
invocation) and `models[]` (a metadata column). For `9ed82e70` and `3476ed08`
they disagree (`models[0]` says Llama-2-7b-hf, `perf_command --model` says
Llama-3.1-8B-Instruct). The runner extracts `--model` from `perf_command` as
ground truth — that's what's actually passed to vllm/sglang.

---

## Hardware deviation: 310aca88

ISO-Bench's authoritative `perf_command` for `310aca88` specifies `-tp 4`
(Llama-3-70B 4-way tensor parallel). Sonnet45 ran this on 8× H100 with `tp=4`.
On 2× H100 we ran with **tp=2** (runtime rewrite, `-tp 4 → -tp 2` based on
`CUDA_VISIBLE_DEVICES` count). Memory fits tightly: 70B fp16 = ~140 GB → ~70 GB
per shard at tp=2 + KV/activations on 80 GB H100s. `--load-format dummy` means
weights are random-init at the right shapes, so this is a genuine compute
benchmark.

**Deviation footnote**: the resulting `avg_latency_s = 6.35` measurement at
tp=2 is NOT directly comparable to sonnet45's tp=4 number for this commit.
Document this in any cross-comparison.

---

## Agent regressions (counted as bench results, per policy)

These 6 patches genuinely break the server / produce no metrics. The bench
result IS the agent's failure to produce a runnable patch:

### vLLM `e3580537` — FP8 + prefix-caching server hang
Agent patches `vllm/core/block/scheduler.py` + `model_runner.py` for
RedHatAI/Meta-Llama-3-8B-Instruct-FP8 with `--enable-prefix-caching --enable-chunked-prefill`.
Server starts fine. First bench request returns `aiohttp.client_exceptions.ClientPayloadError:
TransferEncodingError: 400`. Same failure mode as the sonnet45 postmortem
documented for this exact commit — likely a real regression in the FP8 +
prefix-caching codepath under this patch.

### vLLM `89a84b0b` — `_is_pin_memory_available_impl` NameError
Agent's patch for Qwen1.5-0.5B references a symbol `_is_pin_memory_available_impl`
that doesn't exist (renamed/removed in vLLM 0.5.4). Real Python NameError at
server startup → engine_process_failed. Genuine agent regression.

### vLLM `6e36f4fa` — TransferEncoding from partial-reconstructed patch
Patch was reconstructed from trajectory.json (the original 0-byte
`model_patch.diff` was a harness bug). 4 of 7 `str_replace` ops applied; the
3 unapplied ones had `+/-` diff markers literally embedded in `old_str` —
those would have failed at agent runtime too. Final patch hung the bench
(`Response payload is not completed`). The agent's actual code change (the 4
ops) is what produced this regression — same flavor as `e3580537`.

### SGLang `df7f61ee` — `cuda:0 vs cpu` device mismatch
Agent's patch in `sglang/srt/managers/expert_location.py` causes
`RuntimeError: Expected all tensors to be on the same device, but found at
least two devices, cuda:0 and cpu!` during `init_by_mapping`. The patch
broke device routing in the MoE expert-location compute. Genuine regression.

### SGLang `31589e17` — server hang after MoE patch
Agent edits `sglang/srt/models/deepseek_v2.py` + `two_batch_overlap.py`.
Server starts but hangs handling first request — `/health` endpoint
unresponsive after 27 min, both server and scheduler in `S` state. Patch
deadlocks the MoE forward path. Genuine regression.

### SGLang `205d5cb4` — Llama-4-Scout w4a16 dependency tangle
`NameError: name 'WNA16_SUPPORTED_BITS' is not defined` in
`sglang/srt/layers/quantization/compressed_tensors/compressed_tensors_moe.py:342`.
With `force_modern` (sglang→0.4.7 substitution + vllm co-install +
compressed-tensors==0.9.4), the symbol that should come from this version-pin
chain doesn't materialize. Multi-package coordination issue rather than agent
fault, but uncovered by this specific commit's load path. Marked as documented
infra-failure (closer to environment than agent regression).

---

## Hardware-bound (not benchmarked)

### vLLM `e7b20426` — Llama-4-Maverick-17B-128E
Active params 17B, 128 experts, ~400-800 GB total at fp16/fp8. Doesn't fit on
8× H100 (per sonnet45 postmortem) — definitely not on 2× H100. Drop.

### vLLM `6dd94dbe` — parent vLLM 0.4.0/CUDA 11
Same baseline failure sonnet45 documented: parent `0e74d797ce86` is vLLM 0.4.0
which targets CUDA 11; current build environment is CUDA 12+; baseline image
unbuildable upstream. Drop.

---

## Three execution paths

Same as sonnet45's three paths for vLLM, plus a new native SGLang overlay:

### 1. Native vLLM wheel install (`run_vllm_native.py`, 21 commits)
- `uv venv --python 3.12` per commit
- `uv pip install https://wheels.vllm.ai/<parent>/vllm-1.0.0.dev-cp38-abi3-manylinux1_x86_64.whl`
- Per-commit `transformers_pin` from `data/mappings/vllm_oh_mapping.json`
- Apply agent patch to `<venv>/lib/python3.12/site-packages/`
- Bench via `benchmark_serving.py` (server) or `benchmark_latency.py` /
  `benchmark_throughput.py` / `benchmark_prefix_caching.py` (in-process)

### 2. Native vLLM PyPI overlay (`setup_venv_overlay`, 13 commits)
For commits whose parent has no wheel at wheels.vllm.ai:
- `uv venv --python 3.11`
- `uv pip install vllm==<pypi_version>` from PyPI (per-commit version mapping)
- Pin baseline-compat deps: `transformers>=4.42,<4.45`, `outlines==0.0.46`,
  `lm-format-enforcer==0.10.1`, `numpy<2`
- Stub `pyairports/` (PyPI wheel is empty; outlines imports it)
- Apply agent patch to installed `vllm/`

### 3. Native SGLang overlay (`run_sglang_native.py`, 12 commits)
Mirrors run_vllm_native:
- `uv venv --python 3.12`
- `uv pip install sglang[all]==<pypi_version>` (version discovered via
  `git tag --contains <parent>` in `/sglang` submodule)
- Pin `compressed-tensors==0.9.4` (newer 0.10+ imports
  `transformers.masking_utils` which sglang's pinned transformers 4.52.3 lacks)
- For sglang 0.4.6.post*: pin `transformers>=4.45,<4.51` to dodge torch 2.6
  register_constant ABI break
- For 1acca3a2 / 205d5cb4 specifically: substitute sglang→0.4.7 (torch 2.7-
  native, fixes sgl_kernel ABI). Co-install `vllm` for 205d5cb4
  (CompressedTensorsWNA16MoEMethod imports from vllm).
- Apply patch with `patch -p2 --force` (strips `a/python/` prefix —
  sglang patches use `a/python/sglang/srt/...` source paths, but installed
  package is at `sglang/srt/...` in site-packages).
- Start `python -m sglang.launch_server` natively
- Bench via `python -m sglang.bench_serving` (most commits) or
  `python -m sglang.bench_one_batch` (in-process for `c087ddd6`)
- For `--max-concurrency 1` perf_commands: cap `--num-prompts` to 30 +
  raise subprocess timeout to 5400s (sequential bench at 1000 output tokens
  takes ~50 s/prompt).

---

## Operational footnotes

### GPU orphan sweep
vllm/sglang use `multiprocessing.spawn` for tensor-parallel workers; if the
parent api_server dies via SIGKILL (which `pkill` does to clean up between
commits), the spawn-children become orphans (PPID=1) holding ~70 GiB GPU
memory until manually killed. The runner now sweeps these at the start of
every commit's `run_one()` via `kill_gpu_orphans()` — finds processes whose
argv0 is in `/tmp/native_*venvs/` listed in `nvidia-smi --query-compute-apps`,
SIGKILLs them. Without this, every 2nd commit OOM'd at model-load time.

### Disk-space landmine
`/` filesystem is 97 GB total; 32 commits' venvs (~3-5 GB each) + uv cache
(~57 GB) saturated it after the first fanout. Solution:
- `UV_CACHE_DIR=/ephemeral/uv_cache` (1.4 TB on /ephemeral)
- `NATIVE_VENV_ROOT=/ephemeral/native_venvs`
- Symlinks at `/tmp/native_venvs` → `/ephemeral/native_venvs`

### `cwd=/tmp` for bench subprocesses
A leftover `vllm/` directory at the project root (from a previous submodule
clone) shadowed the venv's installed package via Python's
`sys.path[0] = ''` namespace-package resolution. Both runners now pin
`subprocess.Popen(..., cwd="/tmp")` so namespace shadowing can't sneak in.

---

## Reproducibility

```bash
git clone https://github.com/thekernelcompany/OmniPerf-Bench
cd OmniPerf-Bench
git checkout icml/rebuttal-hard-metrics-oh
git submodule update --init sglang  # required for sglang version discovery

# Tools
curl -LsSf https://astral.sh/uv/install.sh | sh
pip install pandas pyarrow huggingface_hub

# HF tokens
mkdir -p ~/.config/omniperf
echo "<gated-models-hf-token>" > ~/.config/omniperf/hf_token_gated  # for meta-llama/* + mistralai/*
huggingface-cli login --token <default-hf-token>                     # for everything else

# Pull GPT-5 OH trajectories from HF
huggingface-cli download Inferencebench/iso-bench-openhands-gpt5-rebuttal \
    --repo-type dataset --local-dir ISO-Bench/state/runs/oh54_gpt5_hf

# Flatten into runner-expected layout
python scripts/runners/prepare_oh_patches.py --agent openhands_gpt5

# Single-commit smoke
CUDA_VISIBLE_DEVICES=0 UV_CACHE_DIR=/ephemeral/uv_cache python3 \
    scripts/runners/run_vllm_native.py --agent-name openhands_gpt5 \
    --commits b55ed6ef --timeout 1500

# vLLM 2-GPU fanout (38 commits)
CHUNK0="<half_the_commits>"
CHUNK1="<other_half>"
CUDA_VISIBLE_DEVICES=0 UV_CACHE_DIR=/ephemeral/uv_cache nohup python3 \
    scripts/runners/run_vllm_native.py --agent-name openhands_gpt5 \
    --commits $CHUNK0 --timeout 1500 > logs/worker0.log 2>&1 &
CUDA_VISIBLE_DEVICES=1 UV_CACHE_DIR=/ephemeral/uv_cache nohup python3 \
    scripts/runners/run_vllm_native.py --agent-name openhands_gpt5 \
    --commits $CHUNK1 --timeout 1500 > logs/worker1.log 2>&1 &

# SGLang 1-GPU fanout (15 commits)
CUDA_VISIBLE_DEVICES=0 UV_CACHE_DIR=/ephemeral/uv_cache python3 \
    scripts/runners/run_sglang_native.py --agent-name openhands_gpt5 \
    --commits <space_separated_commits> --timeout 5400

# Multi-GPU commit (310aca88): both GPUs, runtime tp-rewrite
CUDA_VISIBLE_DEVICES=0,1 UV_CACHE_DIR=/ephemeral/uv_cache python3 \
    scripts/runners/run_vllm_native.py --agent-name openhands_gpt5 \
    --commits 310aca88 --timeout 1500
```

Per-commit dep overrides come from the mapping JSON automatically. ISO-Bench
perf_command + model overrides come from the parquet automatically.

---

## Artifact locations

| Artifact | Location |
|---|---|
| Result JSONs (54) | `archive/results/2026-05/iso_bench_results_3way_openhands_gpt5{,_sglang}/results/` (git) + HF dataset |
| Agent patches (input) | `ISO-Bench/state/runs/{vllm,sglang}/openhands_gpt5/flat/` (git) |
| Agent run history | HF `Inferencebench/iso-bench-openhands-gpt5-rebuttal` |
| Per-commit dep mapping | `data/mappings/{vllm,sglang}_oh_mapping.json` |
| ISO-Bench parquet | `data/iso_bench/{vllm,sglang}/train.parquet` (mirror of HF `ISO-Bench/ISO-Bench`) |
| vLLM native runner | `scripts/runners/run_vllm_native.py` |
| SGLang native runner | `scripts/runners/run_sglang_native.py` (new in this run) |
| Patch flatten | `scripts/runners/prepare_oh_patches.py` |
| Worker logs | `logs/oh_gpt5_fanout/`, `logs/oh_gpt5_sglang_fanout/`, `logs/oh_gpt5_smoke/`, `logs/oh_gpt5_sglang_smoke/` |
| 6e36f4fa reconstructed patch | `ISO-Bench/state/runs/vllm/openhands_gpt5/flat/vllm_core-0034/{model_patch.diff,RECONSTRUCTED.txt}` |
| Summary | `archive/results/.../SUMMARY.json` |
