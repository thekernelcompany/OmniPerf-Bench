# pass@k Sample Collection Plan

## Goal

Collect 8 independent agent patches per task to enable unbiased pass@k
evaluation.  Each sample must be fully isolated — the agent running
sample N must have zero visibility into samples 0..N-1.

## Approach

Run the existing `bench.cli prepare` pipeline 8 times per plan item.
After **each** sample:

1. Collect every artifact (patch, logs, journal, trajectory, prompt, stderr)
2. Push the full artifact set as one row to a HuggingFace dataset
3. Confirm the push succeeded
4. Nuke all local state (worktree + state dir) — only then start the next sample

Nothing is batched.  Nothing sits on disk between samples.

## Isolation Guarantees

| Threat | Mitigation |
|--------|-----------|
| Agent reads previous patch from `state/runs/` | State dir nuked after verified HF push |
| Agent reads previous worktree | Worktree nuked after verified HF push |
| Agent reads git history of human commit | `detach_from_history=True` (existing, hardcoded in `prepare.py:126`) |
| Agent accesses staging/temp files | Temp plan uses unique `tempfile.mkstemp()` per sample, deleted after each sample |
| Agent reads cached state from `~/.claude` etc. | Agent cache dirs cleaned between samples (best-effort; true isolation requires containers) |
| Crash leaves stale state on disk | `--resume` queries HF for completed `(item_id, sample_index)` pairs and skips them; stale local dirs are nuked before the next sample starts |
| Push fails, state nuked prematurely | Nuke only fires after `push_single_row()` returns True AND verification confirms shard exists on HF; on failure, local state is preserved for manual recovery |
| Timeout leaves zombie processes | Process group kill (`SIGTERM` → `SIGKILL`) ensures agent + child processes are cleaned up |

## HuggingFace Dataset Schema

One row per sample.  Every artifact is stored — nothing is discarded.

**Structured columns** (for filtering/aggregation):

| Column | Type | Source |
|--------|------|--------|
| `item_id` | str | plan item identifier (e.g. `vllm_core-0042`) |
| `sample_index` | int | 0..7 |
| `run_id` | str | hierarchical run path |
| `collected_at` | str | ISO timestamp |
| `task_id` | str | from `journal.json` |
| `status` | str | `success`, `error`, `max_steps_exceeded` |
| `human_commit` | str | target optimization commit |
| `pre_commit` | str | baseline commit |
| `agent_name` | str | `trae`, `claude_code`, `codex`, `openhands` |
| `model_name` | str | `gpt-5`, `sonnet`, etc. |
| `duration_s` | float | agent wall-clock time |
| `time_to_first_edit_s` | float | seconds until first git commit |
| `commit_count` | int | number of agent commits |
| `patch_size_loc` | int | lines added + removed |
| `changed_files_count` | int | files touched |
| `violations_count` | int | files changed outside target set |

**Artifact columns** (full text, for ablation):

| Column | Source file |
|--------|-----------|
| `model_patch` | `model_patch.diff` |
| `journal_json` | `journal.json` |
| `prompt_json` | `prompt.json` |
| `task_text` | `task.txt` |
| `diff_targets_json` | `diff_targets.json` |
| `run_summary_json` | `run_summary.json` |
| `agent_stdout` | `{agent}_stdout.txt` |
| `agent_stderr` | `{agent}_stderr.txt` |
| `trajectory_json` | `trajectory.json` (TRAE/Codex only) |

## Usage

All commands run from `ISO-Bench/`.

```bash
# 1. Dry run — verify plan, no agents, no push
python scripts/run_pass_at_k.py \
    --task tasks/vllm.yaml \
    --plan state/plan.json \
    --bench-cfg bench.yaml \
    --n 8 --dry-run

# 2. Full run — 8 samples per item, push each to HF immediately
python scripts/run_pass_at_k.py \
    --task tasks/vllm.yaml \
    --plan state/plan.json \
    --bench-cfg bench.yaml \
    --n 8 \
    --hf-repo ISO-Bench/pass-at-k-samples

# 3. Resume after crash
python scripts/run_pass_at_k.py \
    --task tasks/vllm.yaml \
    --plan state/plan.json \
    --bench-cfg bench.yaml \
    --n 8 \
    --hf-repo ISO-Bench/pass-at-k-samples \
    --resume

# 4. Run a subset of items (for testing)
python scripts/run_pass_at_k.py \
    --task tasks/vllm.yaml \
    --plan state/plan.json \
    --bench-cfg bench.yaml \
    --n 2 \
    --hf-repo ISO-Bench/pass-at-k-samples \
    --items vllm_core-0000 vllm_core-0001
```

## Scale Estimate

| Agent | Per-sample size | 8 samples x 96 items | Approx. time per sample |
|-------|----------------|----------------------|------------------------|
| Claude Code | ~40 KB | ~30 MB | 2-10 min |
| Codex / Codex CLI | ~50-100 KB | ~40-75 MB | 5-30 min |
| OpenHands | ~30 KB | ~23 MB | 5-60 min |
| TRAE | ~1.8 MB | ~1.4 GB | 5-30 min |

Total HF dataset: ~6 GB worst case (TRAE-dominated, 4 agents x 96 items x 8 samples).

## What Comes Next (after collection)

Once all patches are on HuggingFace:

1. **Evaluate**: run each patch through the evaluation harness (`opt_at_k.py` /
   `run_tests_on_patches.py`) to get speedup numbers
2. **Define "pass" threshold**: decide what counts as a passing sample
   (e.g. `status == success`, or `speedup > 1.0`, or `speedup >= 1.2`)
3. **Compute pass@k**: group by `(item_id, agent_name, model_name)`, count
   successes `c` out of `n=8`, apply the unbiased estimator
   `pass@k = 1 - C(n-c, k) / C(n, k)`, average across items

These are separate scripts written after the data exists.
