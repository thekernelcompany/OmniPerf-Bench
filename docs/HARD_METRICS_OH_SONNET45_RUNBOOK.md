# Hard-metrics runbook — OpenHands Sonnet-4.5 ICML rebuttal

**Audience:** whoever has the 8×H100 box in front of them.
**Goal:** run agent-only hard metrics (TTFT, throughput) for the 54-task OH Sonnet-4.5 fanout, join against the existing baseline+human numbers, write final results JSONs.
**Doc generated:** 2026-05-03. Treat anything older than a week here as suspect — Docker tags and HF dataset state change.

This runbook is intentionally critical. Where we have not verified something end-to-end, it says **UNVERIFIED**. Don't skip those checks just because the doc reads smooth.

---

## 0. TL;DR

| | |
|---|---|
| Tasks | 54 (39 vllm + 15 sglang) |
| Patches with content (will benchmark) | 52 |
| Empty patches (treat as 0-improvement) | 2 |
| Mode | `--agent-only` (baselines + humans already measured in `archive/results/2026-01/`) |
| Wall-clock estimate | **~4-5 hrs on 8×H100** if vLLM-only happy-path; +1-2 hrs if SGLang infra needs porting |
| Cost estimate | **$100-150** at $25-30/hr 8×H100 |
| Patch source | HF dataset `Inferencebench/iso-bench-openhands-sonnet45-rebuttal` |
| Runner (vLLM) | `scripts/runners/run_3way_benchmarks.py` with new `openhands_sonnet45` AGENT_CONFIGS entry |
| Runner (SGLang) | `scripts/archive/hero_sglang_benchmark.py` — **imports Modal, must be ported. Real blocker.** |

---

## 1. Pre-flight on the Prime Intellect box (do these before anything else)

### 1.1 System

```bash
# Check NVIDIA + Docker + driver
nvidia-smi                                    # 8 H100s, CUDA >= 12.4 ideally
docker info | grep -i runtime                 # nvidia runtime registered
docker run --gpus all --rm nvidia/cuda:12.4.0-base-ubuntu22.04 nvidia-smi
```

If `docker run --gpus` fails: install nvidia-container-toolkit, restart docker. This *must* work before anything else.

### 1.2 Credentials & tokens

You will need, on the box, in env or files:

- `HF_TOKEN` — must have approval for **gated models** referenced in the benchmark mappings (Llama-3.1, Llama-3.3, Mistral, Bamba). Without these the benchmark fanout dies on first model download. Note: `MODEL_OVERRIDES` in `run_3way_benchmarks.py` already swaps `Llama-3.1` → `Meta-Llama-3-8B-Instruct` for old-vLLM compat, so 3.1 access *may* not be strictly required — but verify.
- **Docker Hub login** for `shikhar481` and `ayushnangia16` orgs. The `vllm-baseline-images`, `vllm_fixed_human_images`, `sglang-images`, `nvidia-vllm-docker`, `sglang-docker`, `nvidia-sglang-docker` repos are private (404 from public API). `docker login` with appropriate creds. Verify with:
  ```bash
  docker pull shikhar481/vllm_fixed_human_images:baseline-fbefc8a78d22  # known to exist
  ```
- HuggingFace CLI logged in:
  ```bash
  huggingface-cli login --token $HF_TOKEN
  ```

### 1.3 Repo + dependencies

```bash
git clone --recursive git@github.com:thekernelcompany/OmniPerf-Bench.git
cd OmniPerf-Bench
git checkout icml/rebuttal-hard-metrics-oh    # this branch

# main env
curl -LsSf https://astral.sh/uv/install.sh | sh
uv venv && source .venv/bin/activate
uv sync

# bench-env (used by some runners)
python -m venv bench-env && source bench-env/bin/activate
pip install -r ISO-Bench/requirements.txt    # adjust if path differs
```

⚠️ **Heads-up about the remote**: this repo's `icml` remote URL contains an embedded GitHub PAT (`ghp_…`). If you `git remote -v` on a shared box that token is visible. **Rotate it after the rebuttal.** Don't push to `icml` from a shared box.

---

## 2. Inventory of what's already built vs needs work

### 2.1 OH patches

All 54 are on HF: `Inferencebench/iso-bench-openhands-sonnet45-rebuttal`.

- 728 files: plans, configs, harness, 54 stderr logs, 74 model_patch.diff (extras are retries — keep latest timestamp dir per task).
- 52 patches non-empty, 2 empty. The runner auto-skips zero-byte patches via `patch_file.stat().st_size > 0` so the 2 empties don't poison the fanout — they just get logged as 0-improvement.

### 2.2 Docker images on Docker Hub (verified 2026-05-03)

| Repo | Tags | Used for |
|---|---:|---|
| `shikhar481/vllm_fixed_human_images` | 92 | **Primary vLLM** — both `baseline-<12char>` and `human-<full>` tags |
| `shikhar481/vllm-baseline-images` | 6 | Supplementary vLLM baselines |
| `ayushnangia16/nvidia-vllm-docker` | 105 | Older vLLM humans (full-hash) |
| `shikhar481/sglang-images` | 252 | Primary SGLang (`v05-improved-<short>`, `v05-hf-<short>`) |
| `ayushnangia16/sglang-docker` | 144 | SGLang fallback (full-hash) |
| `ayushnangia16/nvidia-sglang-docker` | 78 | SGLang fallback |

### 2.3 Coverage cross-ref against our 54 tasks

| Group | Need | Pre-built | To build | Permanent fail |
|---|---:|---:|---:|---:|
| vLLM baselines (parents) | 39 | **35** | **3 buildable** | **1** (`0e74d797ce86` — vLLM 0.4.0/CUDA 11) |
| vLLM humans | 39 | 39 | 0 | 0 |
| SGLang humans | 15 | 15 | 0 | 0 |

vLLM parent commits to build (3): `a732900efc4e`, `d3ea50113c08`, `f67e9e9f221e`.
Drop or footnote: `0e74d797ce86`.

### 2.4 Existing baseline+human result JSONs (do NOT re-run)

`archive/results/2026-01/omniperf_results_3way_*` has prior 3-way runs for claude_code, codex_*, trae_*, sglang. Schema = same JSONL the new agent run will emit, so joining is simple.

---

## 3. Step-by-step runbook (the actual work)

Numbered so you can resume from any step.

### Step A — Pull the OH patches from HF (~5 min)

```bash
mkdir -p ISO-Bench/state/runs/oh54_hf
huggingface-cli download Inferencebench/iso-bench-openhands-sonnet45-rebuttal \
    --repo-type dataset \
    --local-dir ISO-Bench/state/runs/oh54_hf
```

### Step B — Flatten HF layout into runner-expected layout (~2 min)

The runner expects `AGENT_PATCHES_DIR / vllm_core-XXXX / {journal.json, model_patch.diff}`.
HF dataset has multiple timestamp dirs per task (retries). Pick the latest non-empty per task.

Write `scripts/runners/prepare_oh_patches.py` (TODO — not yet in repo):

```python
#!/usr/bin/env python3
"""Flatten HF OH dataset into the agent_patches dir layout run_3way_benchmarks expects.

Picks the latest non-empty model_patch.diff per task across timestamp dirs.
"""
import re, json, shutil
from pathlib import Path

SRC = Path("ISO-Bench/state/runs/oh54_hf/runs")
VLLM_DST  = Path("ISO-Bench/state/runs/vllm/openhands_sonnet45/flat")
SGLANG_DST = Path("ISO-Bench/state/runs/sglang/openhands_sonnet45/flat")

for repo, dst in [("vllm", VLLM_DST), ("sglang", SGLANG_DST)]:
    by_task = {}
    for patch in (SRC / repo).rglob("model_patch.diff"):
        # runs/{repo}/{ts}/{task}/model_patch.diff
        ts, task = patch.parts[-3], patch.parts[-2]
        size = patch.stat().st_size
        prev = by_task.get(task)
        if prev is None or ts > prev[0] or (ts == prev[0] and size > prev[1]):
            by_task[task] = (ts, size, patch)
    dst.mkdir(parents=True, exist_ok=True)
    for task, (ts, size, patch) in by_task.items():
        td = dst / task
        td.mkdir(exist_ok=True)
        shutil.copy(patch, td / "model_patch.diff")
        journal = patch.parent / "journal.json"
        if journal.exists():
            shutil.copy(journal, td / "journal.json")
    print(f"{repo}: wrote {len(by_task)} task dirs to {dst}")
```

Run: `python scripts/runners/prepare_oh_patches.py`.

**UNVERIFIED**: that the OH `journal.json` actually contains `journal.commits.human` (8-char hash). The runner reads exactly that key (`scripts/runners/run_3way_benchmarks.py:207`). If the OH journal schema differs, the adapter loops below silently produce zero patches. **Verify by `cat ISO-Bench/state/runs/vllm/openhands_sonnet45/flat/vllm_core-0000/journal.json | jq .commits.human`.** Fix the adapter to synthesize a fake journal from the task ID + plan-file commit if needed.

### Step C — Add `openhands_sonnet45` to runner config (~5 min)

Patch `scripts/runners/run_3way_benchmarks.py`:

```python
# in AGENT_CONFIGS, add:
    "openhands_sonnet45": "ISO-Bench/state/runs/vllm/openhands_sonnet45/flat",

# in AGENT_OUTPUT_DIRS, add:
    "openhands_sonnet45": ROOT_DIR / "archive/results/2026-05/iso_bench_results_3way_openhands_sonnet45",
```

### Step D — Build the 3 missing baselines (~1-2 hrs, can parallelize on 3 GPUs)

```bash
for parent in a732900efc4e d3ea50113c08 f67e9e9f221e; do
    python scripts/docker/build_baseline_images.py --commit $parent &
done
wait
python scripts/docker/push_baseline_images.py
```

**UNVERIFIED**: that `build_baseline_images.py` works on Prime Intellect's environment without modification — it was last touched on a different host. If it depends on a specific CUDA base image not on the Prime Intellect box, expect a couple hours of fixup. The 4 historically-permanent failures (vLLM 0.4.0 / CUDA 11) are unbuildable; if any of these 3 hits the same wall, we drop them and footnote.

### Step E — Dry-run on ONE vLLM commit (~30 min)

Pick a known-good baseline (e.g. parent `fbefc8a78d22` confirmed in registry), find a task whose parent is it, and run **only that one** through the runner:

```bash
python scripts/runners/run_3way_benchmarks.py \
    --agent-type openhands_sonnet45 \
    --agent-only \
    --commits <human_short_hash_for_that_task> \
    --timeout 1800
```

**Stop here and inspect the result JSON before fanout.** Confirm:
- Docker pulled the right baseline
- Patch applied cleanly to the vLLM source
- `benchmark_serving.py` (or whatever mode the mapping says) actually ran and produced TTFT/throughput numbers
- Output JSON written to `archive/results/2026-05/iso_bench_results_3way_openhands_sonnet45/agent_benchmark_results/`

Failures here are the single biggest risk to the whole plan. Spending 30 min validating beats spending 4 hrs on a broken fanout.

### Step F — vLLM fanout (~2.5 hrs on 8×H100)

```bash
# 8 parallel workers, each pinned to one GPU, draining a queue of 39 commits
mkdir -p logs
COMMITS=$(python -c "
import json
plan = json.load(open('ISO-Bench/state/runs/oh54_hf/plans/plan_iso.json'))
print(' '.join(it['human'][:8] for it in plan['items']))
")

# split into 8 roughly-equal chunks; one shell loop per GPU
i=0
for chunk in $(echo "$COMMITS" | tr ' ' '\n' | split -n l/8 -d - chunk_); do
    CUDA_VISIBLE_DEVICES=$i python scripts/runners/run_3way_benchmarks.py \
        --agent-type openhands_sonnet45 \
        --agent-only \
        --commits $(cat chunk_$(printf '%02d' $i)) \
        --timeout 1800 \
        > logs/worker_$i.log 2>&1 &
    i=$((i+1))
done
wait
```

**UNVERIFIED**: that `run_3way_benchmarks.py` respects `CUDA_VISIBLE_DEVICES` for the docker invocations. If it doesn't (it might bind all GPUs inside the container), parallelism breaks down to 1 worker. **Test with the dry-run in step E first** — watch `nvidia-smi` to confirm only one GPU goes hot.

If parallelism doesn't work via `CUDA_VISIBLE_DEVICES`: the runner's `docker run` line needs `--gpus device=$GPU_ID` instead of `--gpus all`. Patch and re-test.

### Step G — SGLang fanout — **biggest unknown**

`scripts/archive/hero_sglang_benchmark.py` is the closest thing to a non-Modal SGLang runner, but it imports `src.eval.sglang_modal_benchmark` for the actual benchmark execution. So today the SGLang path is Modal-only.

Three options, in order of preference:

1. **Port `sglang_modal_benchmark.py` to a local Docker variant** (~2-4 hrs of work). Strip the `@modal.function` decorators, replace the Modal volume mounts with local bind mounts, run `docker run --gpus device=$GPU_ID …` directly. The actual benchmark logic (vol setup, command exec, parse) is portable.
2. **Run SGLang on Modal** with whatever credit you have left, while vLLM runs on Prime Intellect. Mixed infra, but the SGLang fanout is small (15 tasks, ~5 GPU-hrs).
3. **Defer SGLang.** Report vLLM-only hard metrics in the rebuttal, footnote SGLang as "soft-metric only" or "deferred to camera-ready." Honest and fast.

**Decide which before kickoff** — option 1 changes the wall-clock estimate by 2-4 hrs; option 3 cuts scope by 28%.

### Step H — Aggregation (~15 min)

After fanout, results land in `archive/results/2026-05/iso_bench_results_3way_openhands_sonnet45/agent_benchmark_results/<commit>.jsonl`.
Join against existing baseline+human JSONs in `archive/results/2026-01/omniperf_results_3way_*/` to produce final TTFT/throughput delta tables.

Existing aggregation scripts: `archive/scripts/` has prior agents' aggregators — copy and adapt for the OH agent type.

---

## 4. Known issues & gotchas (read this before kickoff)

1. **`0e74d797ce86`** baseline is permanently unbuildable. Drop or footnote.
2. **2 OH patches are zero-byte.** Runner auto-skips. Report them as agent-failure / 0-improvement, not missing data.
3. **`MODEL_OVERRIDES` in the runner** silently swaps Llama-3.1 → Meta-Llama-3-8B for old-vLLM compat. The hard-metric numbers are therefore *not* on the model the PR's authors actually used in those cases. Worth acknowledging in the rebuttal.
4. **Modal token in `~/.modal.toml`** — irrelevant on Prime Intellect, but if your box is multi-tenant don't leave that file readable.
5. **`icml` git remote contains an embedded PAT.** Rotate after rebuttal.
6. **Empty `archive/results/2026-01/iso_bench_results_3way_claude_code/exports/`** — earlier expected this had the prior 3-way export bundle; it doesn't. The actual baseline+human result JSONs live one level up at `archive/results/2026-01/omniperf_results_3way_*/`.

---

## 5. Pre-kickoff sanity checks (the "I would not start without these" list)

Tick each before launching step F:

- [ ] `docker pull shikhar481/vllm_fixed_human_images:baseline-fbefc8a78d22` succeeds
- [ ] `huggingface-cli download Inferencebench/iso-bench-openhands-sonnet45-rebuttal` succeeds and produces 728 files
- [ ] `journal.json` from a downloaded OH task contains `commits.human` as a hex string
- [ ] Dry-run on one commit (step E) produces a non-error JSON with TTFT/throughput numbers
- [ ] `nvidia-smi` during dry-run shows only the intended GPU active (validates `CUDA_VISIBLE_DEVICES` works through Docker)
- [ ] You've decided vLLM-only vs vLLM+SGLang scope
- [ ] You've confirmed Llama-3.1/3.3 gated-model access OR confirmed `MODEL_OVERRIDES` covers every model in the mapping

---

## 6. What this doc does NOT solve (open work)

- Writing the actual `prepare_oh_patches.py` (sketch in Step B, not committed).
- Patching the runner's `AGENT_CONFIGS` (sketch in Step C, not committed).
- Porting the SGLang runner off Modal (Step G option 1).
- Any aggregation / final-table generation script.
- Verifying `build_baseline_images.py` works on Prime Intellect.

These were intentionally left out: writing them blind and committing them creates the illusion of done-ness. Better to write each on the box, against the actual filesystem state.

---

## 7. If you have to abort mid-run

Restart-safe state:
- Per-commit results write atomically. Re-running a single `--commits <hash>` overwrites the entry.
- Docker images are cached; re-runs after `docker pull` failure resume cleanly.
- HF download is `--resume-download` by default in modern `huggingface-cli`.

So `Ctrl-C` → fix → resume is safe at every step.
