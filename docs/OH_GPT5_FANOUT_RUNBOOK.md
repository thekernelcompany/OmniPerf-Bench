# OpenHands + GPT-5 fanout runbook (EC2-ready)

**Purpose:** produce 54 OpenHands trajectories on the ISO-Bench tasks (39 vllm + 15 sglang) using `openai/gpt-5` via OpenRouter, mirroring the earlier Sonnet-4.5 fanout. Output goes to `state/runs/vllm/openhands/openai-gpt-5/<ts>/` and `state/runs/sglang/openhands/openai-gpt-5/<ts>/`.

**Audience:** whoever has the EC2 box in front of them and just `git clone`d this repo.

**Last validated:** 2026-05-04 (smoke run on `vllm_core-0095` succeeded, $0.50, 47 min).

This is intentionally critical — risks and unverified bits are called out as **UNVERIFIED**. Don't skip them.

---

## 0. TL;DR

| | |
|---|---|
| Tasks | 39 vllm + 15 sglang = 54 |
| Model | `openai/gpt-5` via OpenRouter (`base_url=https://openrouter.ai/api/v1`) |
| OH version | 0.62.0 (last 0.x with `openhands.core.main` headless entrypoint) |
| Per-task settings | iterations=120, max_budget_per_task=$1000, max_input_tokens=200000, max_output_tokens=64000, time_budget=120min |
| Per-task **observed** cost (smoke) | **$0.50** in 33 iterations, agent finished early |
| Full-fanout cost estimate | **$30-150** (likely $50-90; depends on iteration depth) |
| Wall-clock estimate (8-way parallel on adequate EC2) | **~4-5 hr vllm + ~1.5-2 hr sglang** |
| OpenRouter pricing (current) | $1.25/M input, $10/M output, $0.125/M cached input |

---

## 1. EC2 instance sizing (the make-or-break choice)

Each OH worker peaks around **1 GB RSS** (OH agent + LocalRuntime + Playwright/chrome-headless instance + bash subprocesses). Confirmed on local 16 GB box: 6 workers OOM'd in 4 min, 4 workers were marginal.

| Workers | Min usable RAM | Wall-clock (vllm 39) | EC2 instance | $/hr (us-east-1, on-demand) |
|---:|---|---:|---|---:|
| 4 | 8 GB | ~8 hr | `c6i.xlarge` (4 vCPU, 8 GB) — borderline, no buffer | $0.17 |
| 8 | 16 GB | ~4 hr | `m6i.2xlarge` (8 vCPU, 32 GB) | $0.384 |
| 12 | 24 GB | ~2.7 hr | `m6i.4xlarge` (16 vCPU, 64 GB) | $0.768 |
| 16 | 32 GB | ~2 hr | `m6i.4xlarge` (16 vCPU, 64 GB) | $0.768 |

**Recommended: `m6i.2xlarge` with 8 workers.** 32 GB RAM gives ~2 GB/worker headroom, 8 vCPU is enough (workers are LLM-API-bound, not CPU-bound), $0.38/hr × 4-5 hr = $1.50-2 of EC2 spend on top of OpenRouter.

**Don't go cheaper than `m6i.2xlarge`** — 16 GB barely fits 8 workers and `c6i.*` is right at the edge for 4. Extra cost for the next size up is trivial vs your time.

**Don't go much bigger** — diminishing returns: above 16 workers you start hitting OpenRouter rate ceilings on GPT-5 (300 RPM is typical for new accounts). 8-12 workers is the sweet spot.

---

## 2. Pre-flight on the EC2 box

### 2.1 OS deps (Ubuntu 22.04+ assumed)

```bash
sudo apt-get update
sudo apt-get install -y git python3.12 python3.12-venv python3-pip docker.io tmux htop
sudo usermod -aG docker $USER && newgrp docker

# Playwright + chrome-headless deps (OpenHands agent uses these for browser tools)
sudo apt-get install -y libnss3 libnspr4 libatk1.0-0 libatk-bridge2.0-0 libcups2 \
    libdrm2 libxkbcommon0 libxcomposite1 libxdamage1 libxfixes3 libxrandr2 \
    libgbm1 libpango-1.0-0 libcairo2 libasound2t64
```

### 2.2 Repo + branch

```bash
git clone git@github.com:thekernelcompany/OmniPerf-Bench.git
cd OmniPerf-Bench
git checkout icml/rebuttal-gpt5-fanout    # this branch
git submodule update --init --recursive   # vllm/, sglang/ submodules
```

### 2.3 Two Python environments

The harness needs two separate venvs — keep them separate, do NOT merge:

```bash
# 1) ISO-Bench harness env (for bench.cli)
cd ISO-Bench
python3.12 -m venv .venv
.venv/bin/pip install --upgrade pip
.venv/bin/pip install -e .                # installs the bench module
cd ..

# 2) OpenHands 0.62 env (the agent runtime; bench.cli spawns this Python)
python3.12 -m venv bench-env-oh062
bench-env-oh062/bin/pip install --upgrade pip
bench-env-oh062/bin/pip install openhands-ai==0.62.0
bench-env-oh062/bin/python -m playwright install chromium  # required for browser tool
```

**UNVERIFIED**: the `pip install openhands-ai==0.62.0` line — version pinning may fail if 0.62 was yanked. Worst case, install latest 0.62.x (`pip install 'openhands-ai>=0.62,<0.63'`). The hard requirement is presence of `openhands.core.main` (removed in 1.0).

### 2.4 OpenRouter key + balance check

```bash
export OPENROUTER_API_KEY='sk-or-v1-...'   # paste your key

# Sanity check
curl -sS -H "Authorization: Bearer $OPENROUTER_API_KEY" \
    https://openrouter.ai/api/v1/credits | python3 -m json.tool
# Expect {"data":{"total_credits":N,"total_usage":M}} → available = N - M
```

**Top up to at least $200 before the full fanout.** Smoke alone costs $0.50; the full 54 tasks at smoke pace is ~$27, but worst-case (tasks running to the 120-iteration cap) can be 5×.

---

## 3. The configs (already in this branch — do not change)

| File | What it is |
|---|---|
| `ISO-Bench/config/main_openrouter_gpt5.toml` | LLM config — `model=openai/gpt-5`, `base_url=https://openrouter.ai/api/v1`, `max_input_tokens=200000`, `max_output_tokens=64000`, `custom_llm_provider=openai`. **Diff vs sonnet45's TOML is exactly 2 lines** (model + base_url) — everything else byte-identical to ground truth. |
| `ISO-Bench/tmp_openhands_gpt5_or_vllm_bench.yaml` | Bench config — `iterations=120`, `max_budget_per_task=1000.0`, `time_budget_minutes=120`. **Byte-identical to sonnet45's vllm bench yaml.** |
| `ISO-Bench/tmp_openhands_gpt5_or_sglang_bench.yaml` | Same for sglang track. |
| `ISO-Bench/launch_openhands_gpt5_openrouter.sh` | Driver script — modes `smoke` / `vllm` / `sglang`, OpenRouter env wiring, `--max-workers 4`. **Bump max-workers on EC2** (next section). |

The TOML's `model = "openai/gpt-5"` and `base_url = "https://openrouter.ai/api/v1"` are sufficient — at runtime the harness overrides via `LLM_MODEL` / `LLM_BASE_URL` / `LLM_API_KEY` env vars set by the launch script.

---

## 4. Bumping worker count for EC2

The script ships at `--max-workers 4` (safe for 16 GB local). On EC2 with 32+ GB, edit:

```bash
# Edit ISO-Bench/launch_openhands_gpt5_openrouter.sh
# Two lines, one for each fanout mode:
sed -i 's/--max-workers 4/--max-workers 8/g' ISO-Bench/launch_openhands_gpt5_openrouter.sh
```

Worker count guidance based on `free -h` available RAM:

| Available RAM | Max workers |
|---:|---:|
| ≥ 32 GB | 16 |
| 16-32 GB | 8-12 |
| 8-16 GB | 4-6 |
| < 8 GB | DON'T — instance too small |

---

## 5. Run order

### 5.1 Smoke (mandatory before fanout)

```bash
cd ISO-Bench
export OPENROUTER_API_KEY='sk-or-v1-...'
./launch_openhands_gpt5_openrouter.sh smoke 2>&1 | tee /tmp/smoke.log
```

Wait ~45 min. Look for:
- Final line `✓ Prepare completed: state/runs/vllm/openhands/openai-gpt-5/<ts>`
- `Task status determined as: success` in the log
- Patch at `state/runs/vllm/openhands/openai-gpt-5/<ts>/vllm_core-0095-openhands-gpt5-or-smoke/model_patch.diff`

If smoke fails: don't proceed to fanout. Diagnose first. Common causes:
- `OPENROUTER_API_KEY not set` → forgot to export
- `bench-env-oh062/bin/python missing` → venv not created
- Playwright "browser not installed" → run `bench-env-oh062/bin/python -m playwright install chromium`

### 5.2 Full vllm fanout (39 tasks, ~4-5 hr at 8-way)

Run in tmux so you can detach:

```bash
tmux new -s vllm
cd ISO-Bench
export OPENROUTER_API_KEY='sk-or-v1-...'
./launch_openhands_gpt5_openrouter.sh vllm 2>&1 | tee /tmp/vllm_fanout.log
# Ctrl-b d to detach; tmux attach -t vllm to reattach
```

### 5.3 SGLang fanout (15 tasks, ~1.5-2 hr at 8-way)

After vllm finishes, in a separate tmux:

```bash
tmux new -s sglang
cd ISO-Bench
export OPENROUTER_API_KEY='sk-or-v1-...'
./launch_openhands_gpt5_openrouter.sh sglang 2>&1 | tee /tmp/sglang_fanout.log
```

**Don't run vllm and sglang in parallel** unless your EC2 has 64+ GB RAM. Each fanout spawns N workers; running both = 2N concurrent OH instances.

---

## 6. Live monitoring (cost + RAM + progress)

In a third terminal:

```bash
# Cumulative OpenRouter spend across the active log
LOG=/tmp/vllm_fanout.log
watch -n 30 "
  grep -oE \"'cost': [0-9.]+\" $LOG | awk -F: '{s+=\$2} END {printf \"\$%.2f across %d calls\\n\", s, NR}'
  echo
  grep -c 'Task completed successfully' $LOG | awk '{print \"completed: \"\$1\"/39\"}'
  echo
  free -h | head -2
"
```

**Hard kill triggers:**
- OpenRouter spend > $150 → kill, top up, decide whether to resume
- RAM available < 1 GB → kill, drop --max-workers by 2, restart
- More than 3 tasks failing in a row → kill, diagnose

To kill cleanly:

```bash
# Find parent
pgrep -af 'bench.cli prepare'
kill -TERM <parent-pid>
sleep 5
# Mop up stragglers
pkill -KILL -f 'openhands.runtime.action_execution_server'
pkill -KILL -f 'bench.cli prepare'
```

---

## 7. Resume

The launch script passes `--resume`. If you kill mid-fanout, just relaunch the same mode — bench.cli skips already-complete tasks. State lives in `state/runs/<repo>/openhands/openai-gpt-5/<latest-ts>/`.

To **force-redo** a failed task, delete its run dir before resume:

```bash
rm -rf state/runs/vllm/openhands/openai-gpt-5/<ts>/vllm_core-0007-*
./launch_openhands_gpt5_openrouter.sh vllm
```

---

## 8. Output collection

Each task produces under `state/runs/<repo>/openhands/openai-gpt-5/<ts>/<task_id>/`:

| File | What |
|---|---|
| `model_patch.diff` | THE deliverable — agent's code change |
| `journal.json` | Status, commits, timing, target enforcement result |
| `run_summary.json` | Top-level metadata (status, model, timestamps) |
| `trajectory.json` | Full agent transcript (tool calls + LLM responses) |
| `prediction.jsonl` | Patch in eval-harness format |
| `task.txt` | Prompt the agent saw |
| `prompt.json` | Structured prompt metadata |
| `openhands_stdout.txt`, `openhands_stderr.txt` | Raw agent logs |

### 8.1 Push to HuggingFace (mirroring sonnet45)

The sonnet45 trajectories live at `Inferencebench/iso-bench-openhands-sonnet45-rebuttal`. For GPT-5, create the parallel dataset:

```bash
huggingface-cli login --token $HF_TOKEN

cd ISO-Bench/state/runs
huggingface-cli upload-large-folder \
    Inferencebench/iso-bench-openhands-gpt5-rebuttal \
    . \
    --repo-type dataset \
    --include "vllm/openhands/openai-gpt-5/*" "sglang/openhands/openai-gpt-5/*"
```

Also include the configs + plans + harness so the dataset is fully reproducible (mirror what sonnet45 dataset has):

```bash
# Bundle alongside trajectories — see sonnet45 layout for the file list
mkdir -p /tmp/hf_bundle/{configs,plans,harness}
cp ISO-Bench/config/main_openrouter_gpt5.toml /tmp/hf_bundle/configs/
cp ISO-Bench/tmp_openhands_gpt5_or_*.yaml /tmp/hf_bundle/configs/
cp ISO-Bench/state/runs/oh54_hf_configs/plans/plan_iso.json /tmp/hf_bundle/plans/
cp ISO-Bench/state/runs/oh54_hf_configs/plans/sglang_plan_iso.json /tmp/hf_bundle/plans/
cp ISO-Bench/launch_openhands_gpt5_openrouter.sh /tmp/hf_bundle/harness/
huggingface-cli upload Inferencebench/iso-bench-openhands-gpt5-rebuttal /tmp/hf_bundle . --repo-type dataset
```

---

## 9. Known issues & gotchas

1. **`trajectory.json` parse warning** — at end of each task, `eval.run_summary` logs `Error reading trajectory.json: 'list' object has no attribute 'get'`. This is a non-blocking schema mismatch in the post-hoc summarizer; the trajectory file itself is saved correctly.

2. **OOM under high worker count** — observed locally: 6 workers exhausted 16 GB in 4 min. Each worker peaks ~1 GB (OH + LocalRuntime + Playwright/chrome). Apply EC2 sizing rules in §1.

3. **`finish_reason='length'` is normal** — OH chunks responses; the full TOML `max_output_tokens=64000` is a per-call ceiling, but typical iterations only emit 30-4000 tokens. Don't be alarmed by `length` finishes.

4. **Task `vllm_core-0095` (Mamba2)** is a known-good smoke target — it succeeded in 33 iterations / $0.50 / 47 min on local. Use it as your smoke baseline.

5. **Cache-read pricing dominates** — GPT-5 cached input is $0.125/M (90% off the $1.25/M fresh rate). After the first iteration in a task, ~80%+ of input tokens are cache reads. This is why per-task cost lands so low.

6. **`max_budget_per_task = 1000.0`** is intentionally an effective no-op cap — sonnet45 did the same. Real budget enforcement happens at the OpenRouter account level. **Track the account balance, not the per-task cap.**

7. **Empty model_patch.diff is possible** — if agent finishes without editing target files, `model_patch.diff` will be 0 bytes. The downstream hard-metrics runner auto-skips zero-byte patches. Sonnet45 had 2/54 zero-byte; expect similar for GPT-5.

---

## 10. Pre-kickoff sanity checks (the "would not start without these" list)

- [ ] `bench-env-oh062/bin/python -c "import openhands; print(openhands.__version__)"` prints `0.62.x`
- [ ] `.venv/bin/python -m bench.cli --help` works (lists `prepare` command)
- [ ] `huggingface-cli download Inferencebench/iso-bench-openhands-sonnet45-rebuttal --include "plans/*" --repo-type dataset --local-dir ISO-Bench/state/runs/oh54_hf_configs` succeeded — plan files exist
- [ ] OpenRouter balance ≥ $200 (or you've decided to accept partial completion)
- [ ] Smoke run completed cleanly (`Task status determined as: success`)
- [ ] You've sized EC2 per §1 (don't skip — this is the OOM trap)
- [ ] You're running inside `tmux` so a dropped SSH doesn't kill the fanout

---

## 11. What this doc does NOT cover

- Hard-metrics evaluation of the produced patches — see `docs/HARD_METRICS_OH_SONNET45_RUNBOOK.md` (parallel doc, same workflow but for benchmarking the resulting patches on H100s).
- Comparison/aggregation against sonnet45 — that's a follow-up analysis once both fanouts have run.
- Soft-metrics LLM-as-Judge scoring — uses a separate analyzer in `ISO-Bench/bench/analysis/`.

---

## 12. If you have to abort and resume the next day

```bash
# Re-attach SSH to your instance, then:
tmux attach -t vllm 2>/dev/null || tmux new -s vllm
# If session was killed:
cd ISO-Bench
export OPENROUTER_API_KEY='sk-or-v1-...'
./launch_openhands_gpt5_openrouter.sh vllm     # --resume is in the script, picks up where it left off
```

Per-commit results write atomically. Re-running a single task overwrites only that task's dir.
