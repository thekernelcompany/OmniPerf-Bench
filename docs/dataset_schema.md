## OmniPerf-Bench Dataset Schema

References:
- SWE-Perf dataset card: [https://huggingface.co/datasets/SWE-Perf/SWE-Perf](https://huggingface.co/datasets/SWE-Perf/SWE-Perf)
- GSO dataset card: [https://huggingface.co/datasets/gso-bench/gso](https://huggingface.co/datasets/gso-bench/gso)

### Overview
This document defines a canonical, implementation-agnostic schema for OmniPerf-Bench instances. Export “views” (SWE-Perf view, GSO view) are produced from this canonical schema without coupling our core format to any one benchmark.

### Canonical schema (v1)
- Identity
  - repo (string, required): GitHub `owner/name`.
  - instance_id (string, required): Stable ID, prefer `owner__repo-<id>` where `<id>` is `PR-<n>` if resolvable, else `<commit7>`.
  - created_at (string, required): ISO8601 date-time (e.g., `2023-07-10T12:30:00Z`).
- Code change
  - base_commit (string, required): Commit hash before applying optimization.
  - head_commit (string, required): Optimized commit hash.
  - patch (string, required): Unified diff for non-test files only.
  <!-- - test_patch (string, required): Unified diff for test files only.
  - patch_functions (array[string], optional): Functions modified in non-test diffs. -->
  - test_functions (array[string], optional): Test specific to tasks.
- Tests and prompts
  - efficiency_test (array[string], required): Executable test scripts used to evaluate performance and correctness.
  <!-- - problem_statement_oracle (string|object, optional): Canonicalized description of the optimization problem.
  - problem_statement_realistic (string|object, optional): Realistic task description/prompt as seen by generators (sanitized). -->
- Timing/results
  - duration_changes (array[object], required): Per-test timing arrays; each object has keys `base` (array[number]) and `head` (array[number]).
  - human_performance (number, required): Summary performance metric (e.g., mean(base)/mean(head)).
- Environment
  <!-- - version (string, required): Reproducible environment signature, e.g., `python==3.9;arch=x86_64;image=latest;install_sha=abcd1234`. -->
  - setup_commands (array[string], optional): Environment setup commands.
  - install_commands (array[string], optional): Install/rebuild commands.
- Additional metadata
  <!-- - api (string, optional): API/endpoint impacted (if applicable). -->
  - gt_commit_message (string, optional): Human commit message.
  <!-- - notes (string, optional): Free-form notes. -->

Required minimum: repo, instance_id, created_at, base_commit, head_commit, patch, test_patch, efficiency_test, duration_changes, human_performance, version.

### JSON Schema (canonical)
```json
{
  "$schema": "https://json-schema.org/draft/2020-12/schema",
  "$id": "https://omni-perf-bench/schema/v1",
  "type": "object",
  "required": [
    "repo", "instance_id", "created_at",
    "base_commit", "head_commit", "patch", "test_patch",
    "efficiency_test", "duration_changes", "human_performance",
    "version"
  ],
  "properties": {
    "repo": { "type": "string" },
    "instance_id": { "type": "string" },
    "created_at": { "type": "string" },
    "base_commit": { "type": "string" },
    "head_commit": { "type": "string" },
    "patch": { "type": "string" },
    "test_patch": { "type": "string" },
    "patch_functions": {
      "type": "array",
      "items": { "type": "string" }
    },
    "test_functions": {
      "type": "array",
      "items": { "type": "string" }
    },
    "efficiency_test": {
      "type": "array",
      "items": { "type": "string" },
      "minItems": 1
    },
    "problem_statement_oracle": { },
    "problem_statement_realistic": { },
    "duration_changes": {
      "type": "array",
      "items": {
        "type": "object",
        "required": ["base", "head"],
        "properties": {
          "base": {
            "type": "array",
            "items": { "type": "number" },
            "minItems": 1
          },
          "head": {
            "type": "array",
            "items": { "type": "number" },
            "minItems": 1
          }
        },
        "additionalProperties": false
      },
      "minItems": 1
    },
    "human_performance": { "type": "number" },
    "version": { "type": "string" },
    "setup_commands": {
      "type": "array",
      "items": { "type": "string" }
    },
    "install_commands": {
      "type": "array",
      "items": { "type": "string" }
    },
    "api": { "type": "string" },
    "gt_commit_message": { "type": "string" },
    "notes": { "type": "string" }
  },
  "additionalProperties": false
}
```

### SWE-Perf view (constraints)
- Include only: repo, instance_id, patch, test_patch, base_commit, head_commit, created_at, version, duration_changes, efficiency_test, patch_functions, test_functions, problem_statement_oracle, problem_statement_realistic, human_performance.
- `instance_id`: prefer `owner__repo-PR-<n>` when PR can be resolved, else fallback to commit-based.
- `duration_changes`: preserve per-test arrays; if SWE-Perf requires a specific structure, adapt field names/shape accordingly.
- Field naming must match the SWE-Perf dataset card.

Reference: [SWE-Perf](https://huggingface.co/datasets/SWE-Perf/SWE-Perf)

### GSO view (constraints)
- Include: instance_id, repo, base_commit, opt_commit (from head_commit), api, prob_script (if applicable), tests (alias of efficiency_test), setup_commands, install_commands, hints_text/gt_commit_message, created_at, gt_diff (can be emitted as full diff if desired), arch, instance_image_tag as available from runtime metadata.
- Preserve GSO naming/format as described in the dataset card.

Reference: [GSO](https://huggingface.co/datasets/gso-bench/gso)

### Minimal canonical example
```json
{
  "repo": "vllm-project/vllm",
  "instance_id": "vllm-project__vllm-PR-4894",
  "created_at": "2023-04-07T17:45:07Z",
  "base_commit": "a490aafa3671da1b6b2be6cff4568913fcb1732c",
  "head_commit": "0f40557af6141ced118b81f2a04e651a0c6c9dbd",
  "patch": "diff --git a/cacheflow/models/sample.py ...",
  "test_patch": "diff --git a/tests/kernels/cache.py ...",
  "efficiency_test": ["<test_script_0>", "<test_script_1>"],
  "duration_changes": [
    { "base": [2.14, 2.10, 2.11], "head": [1.52, 1.54, 1.49] }
  ],
  "human_performance": 1.38,
  "version": "python==3.9;arch=x86_64;image=latest;install_sha=deadc0de"
}
```

### Implementation notes
- Diff splitting: treat files under `tests/`, names `test_*.py` or `*_test.py` as tests.
- Function extraction: prefer parser-based extraction; fallback to regex on hunk headers.
- Timing parsing: parse `Execution time: <sec>s` lines from stored results.
- Version: stable, reproducible string; include short hash of install commands.
