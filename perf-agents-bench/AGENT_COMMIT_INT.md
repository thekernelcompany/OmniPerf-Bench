### GSO Agent Integration Guide (Inputs and Outputs)

This guide explains how external agents (e.g., OpenHands or any custom system) should consume GSO tasks and produce predictions that the GSO harness can evaluate. GSO is agent-agnostic: it does not embed or call agents; it only consumes their prediction files and grades them in containers.

Reference: GSO paper: [Benchmark overview and methodology](https://arxiv.org/pdf/2505.23671)

---

### What GSO Provides to Agents (Inputs)

Each task is a `GSOInstance` with the following fields (source: `src/gso/data/dataset.py`):

- **instance_id**: Unique task ID (`{owner}__{repo}-{commit7}`)
- **repo**: GitHub repository slug (`owner/repo`)
- **base_commit**: Commit SHA the agent’s patch must apply to
- **opt_commit**: Expert optimization commit SHA (for reference; not used to generate agent patches)
- **api**: Human-readable API target label
- **prob_script**: Single representative test script (cleaned) used as the problem specification
- **tests**: List of full performance tests used for grading
- **setup_commands**: Commands to prepare the VM/container environment
- **install_commands**: Repository installation commands (cleaned via `prepare_install_commands`)
- **created_at**: Timestamp for the ground-truth commit

Notes:
- During evaluation runs, the loader strips ground-truth fields (prefixed with `gt_`) so agents do not get the exact expert patch or message at eval time. See `src/gso/utils/io.py` (`load_gso_dataset`).
- You can derive the repo URL as `https://github.com/{repo}`.

Practical use by agents:
- Clone the repo, check out `base_commit`, run/install per `install_commands` locally to reproduce.
- Read `prob_script` and `tests` to understand the expected behavior and measured performance paths. Patches must pass equivalence checks encoded in those tests.

---

### What Agents Must Produce (Outputs)

Agents submit predictions as a JSON or JSONL file. Each prediction must be a dictionary with at least:

- **instance_id**: string (matches dataset instance)
- **model_patch**: string (unified diff to apply at repo root on `base_commit`)
- **model_name_or_path**: string (optional; used for logging/paths)

Schema references in the harness:
- Prediction loading: `src/gso/utils/io.py::load_gso_predictions`
- Grading entry: `src/gso/harness/run_evaluation.py::run_instances`
- Per-instance grading: `src/gso/harness/grading/grade.py::grade_instance`
- Metrics/reporting: `src/gso/harness/grading/metrics.py`, `src/gso/harness/grading/report.py`

Example JSONL line:

```json
{
  "instance_id": "pandas-dev__pandas-abc1234",
  "model_patch": "diff --git a/pandas/core/frame.py b/pandas/core/frame.py\nindex 123..456 789\n--- a/pandas/core/frame.py\n+++ b/pandas/core/frame.py\n@@ -100,6 +100,10 @@\n+    # fast-path for common case ...\n",
  "model_name_or_path": "openhands-1.0-gpt4o"
}
```

Constraints on `model_patch`:
- Must be a valid unified diff applying cleanly to a clean checkout of `base_commit`.
- Should not add/modify excluded files. The harness excludes files like `.venv/*`, `.git/*`, `__pycache__/*`, `*.egg-info/*`, and some data/log formats when applying patches (see `src/gso/harness/grading/evalscript.py`).
- Avoid changing the test suite or harness code; tests are the spec.

---

### How the Harness Evaluates Predictions

Overview (files):
- Docker env and container creation: `src/gso/harness/environment/docker_build.py`
- Patch apply + test execution script: `src/gso/harness/grading/evalscript.py`
- Orchestration and grading: `src/gso/harness/grading/grade.py`, `src/gso/harness/run_evaluation.py`
- Metrics/Reports: `src/gso/harness/grading/metrics.py`, `src/gso/harness/grading/report.py`

Evaluation flow per instance:
1. Build/launch container pinned to `base_commit` and repo install commands.
2. Apply `model_patch` using tolerant strategies (`git apply` variants, fallback to `patch`), excluding certain paths.
3. Run tests with a two-phase harness pattern:
   - Base (reference): store reference results
   - Patch (eqcheck): compare current vs reference results; measure runtime
   - Expert commit and optional main branch runs for reference timing
4. Parse logs; compute per-test and aggregated speedups (geometric/harmonic means).
5. Set success flags:
   - `opt_base`: patch improved vs base (≥ 1.2× by default)
   - `opt_commit`: patch reaches ≥ 95% of expert speedup
   - `opt_main`: patch reaches ≥ 95% of current main performance
6. Merge across attempts to compute opt@K.

Failure modes that invalidate an attempt:
- Patch fails to apply.
- Tests crash (ImportError/AttributeError/AssertionError) or time out.
- Equivalence checks (defined inside the generated tests) fail.

---

### What GSO Does Not Do

- GSO does not prompt or run OpenHands (or any agent). It is strictly agent-agnostic.
- Agents can use any internal prompting/strategy, as long as they produce the required predictions file.

---

### Tips for Successful Patches

- Preserve functional equivalence as encoded by the tests (`check_equivalence()` compares against stored reference results).
- Optimize hot paths that the tests actually exercise; avoid micro-optimizations that don’t affect measured code.
- Keep changes minimal and targeted to reduce patch-application conflicts and risk of breaking imports.
- Ensure determinism: avoid randomness or I/O that could destabilize timings or equivalence.

---

### Minimal Run Commands (Evaluator)

Build images (optional, pre-cache):
```bash
uv run src/gso/harness/prepare_images.py --push_to_registry false
```

Evaluate a predictions file:
```bash
uv run src/gso/harness/opt_at_k.py \
  --prediction_paths predictions.jsonl \
  --timeout 3600 \
  --run_id my_run \
  --k 10 \
  --model my_agent
```

---

### Pointers to Source Code

- Dataset model: `src/gso/data/dataset.py`
- Prediction loading: `src/gso/utils/io.py`
- Harness entrypoint: `src/gso/harness/opt_at_k.py`
- Grading core: `src/gso/harness/grading/grade.py`, `src/gso/harness/grading/evalscript.py`, `src/gso/harness/grading/metrics.py`, `src/gso/harness/grading/report.py`

---

### Task Prompt for the Agent to Solve a Task

I've uploaded a python code repository in the directory {workspace_dir_name}.  
Consider the following test script showing an example usage of the repository:

<test_script>
{SPECIFICATION_TEST}
</test_script>

Can you help me implement the necessary changes to the repository so that the runtime of the <test_script> is optimized?

Basic guidelines:
1. Your task is to make changes to non-test files in the /workspace directory to improve the performance of the <test_script>.
2. Make changes while ensuring the repository is functionally equivalent to the original.
3. Do not overoptimize for just the specific inputs in <test_script>. Make general performance improvements for the usage scenario shown.
4. You may need to rebuild the repo for your changes to take effect before testing. Some rebuilds may take time to run, so be patient with running them.

Follow these steps to improve performance:
1. As a first step, explore the repository structure.
2. Create a script in the /workspace directory (e.g., /workspace/test_opt.py) to reproduce and time the example, then execute it with python /workspace/<filename.py>.
3. Edit the source code of the repository to improve performance.
4. Rebuild and rerun your script to confirm that performance has improved.

If you are integrating OpenHands, configure it to emit one JSON/JSONL object per instance containing the fields above, with `model_patch` as the unified diff. The harness will take care of the rest.
