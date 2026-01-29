## OmniPerf-Bench dataset plan (concise)

Reference: [SWE-Perf on Hugging Face](https://huggingface.co/datasets/SWE-Perf/SWE-Perf)

### Objective
Define a canonical, implementation-agnostic dataset schema for OmniPerf-Bench and provide export views compatible with external consumers (e.g., SWE-Perf), without coupling our core format to any one benchmark.

### Canonical OmniPerf-Bench schema (guided by GSO and SWE-Perf)
- Identity
  - repo: string (owner/name)
  - instance_id: string (stable; prefer `owner__repo-<id>`; `<id>` can be PR-N or commit-7)
  - created_at: string (ISO8601)
- Code change
  - base_commit: string
  - head_commit: string
  - patch: string (non-test diffs)
  - test_patch: string (test-only diffs)
  - patch_functions: list[string]
  - test_functions: list[string]
- Tests and prompts
  - efficiency_test: list[string] (executable tests)
  - problem_statement_oracle: string or dict[list] (canonicalized summary)
  - problem_statement_realistic: string or dict[list] (sanitized prompt/messages)
- Timing/results
  - duration_changes: list[dict] (per test: { base: list[float], head: list[float] })
  - human_performance: float (e.g., mean(base)/mean(head))
- Environment
  - version: string (e.g., `python=={py_version};arch={arch};image={image_tag};install_sha={sha8}`)
  - setup_commands: list[string] (optional)
  - install_commands: list[string] (optional)
- Additional metadata
  - api: string (optional)
  - gt_commit_message: string (optional)
  - notes: string (optional)

Notes:
- Keep required fields minimal: identity, commits, patch/test_patch, efficiency_test, duration_changes, human_performance.
- Document field semantics in a local schema file and validate with JSON Schema.

### Export views (adapters)
- SWE-Perf view
  - Emits only fields required by SWE-Perf with expected names/format.
  - Enforces their `instance_id` preference when PR is known; falls back to commit-based.
- GSO view
  - Emits GSO-style fields for backward compatibility (e.g., `opt_commit`, `prob_script`).

### Implementation steps
1) Schema
- Add `docs/dataset_schema.md` describing canonical fields and a JSON Schema for validation.

2) Canonical builder
- Implement `src/collect/build_canonical.py` to assemble canonical records from executed problems (diff splitting, function extraction, timings, statements, version string).

3) Views/exporters
- Implement `src/collect/export_views.py` with `to_swe_perf_view(records)` and `to_gso_view(records)`; write JSONL/Parquet.

4) Validation
- Unit tests for required fields/types; golden tests for diff splitting and timing aggregation.

5) Docs
- Update README with schema overview and commands to build canonical data and export each view.

### Current OmniPerf-Bench structure (today)
This summarizes what we currently emit/store so the canonical schema and views can be implemented with minimal refactors.

- Dataset instances (built in `src/collect/build_dataset.py`)
  - instance_id: string (owner__repo-<commit7>)
  - repo: string
  - base_commit: string (if unset, derived as `<opt_commit>^`)
  - opt_commit: string (expert optimization commit)
  - api: string
  - prob_script: string (a composed performance test runner)
  - tests: list[string] (generated performance tests)
  - setup_commands: list[string]
  - install_commands: list[string]
  - created_at: string (commit date)
  - hints_text: string (commit subject/message)
  - gt_commit_message: string
  - gt_diff: string (full diff; includes tests and non-tests)
  - arch: string (e.g., x86_64)
  - instance_image_tag: string (e.g., latest)

- Execution results (in `Problem.results`)
  - Structure: dict[machine_id -> list[run]] where each run includes at least:
    - commit: string (quick hash)
    - test_id: int
    - test_file: string
    - base_result: string (stdout with timing lines: `Execution time: <sec>s`)
    - commit_result: string (optional)
    - target_result: string
  - Used by `src/collect/execute/evaluate.py` to parse per-test timing arrays and compute speedups.

- Test generation context (in `Problem.tests` of type `Tests`)
  - commit_hash: string
  - chat_messages: list[role/content dict]
  - samples: list[string] (test scripts)

- Commit metadata (in `PerformanceCommit`)
  - commit_hash, subject, message, date
  - files_changed: list[string]
  - functions_changed: list[string]
  - stats: dict[str, int] (e.g., num_non_test_edited_lines)
  - affected_paths: list[string]
  - apis: list[string]
  - diff_text: string
  - repo_path: Path | None
  - linked_pr: derived (best-effort PR number)
  - old_commit_hash: property (`<commit_hash>^`)

- Environment defaults (in `Problem`)
  - py_version: string (default 3.9)
  - setup_commands / install_commands: lists (populated in `model_post_init`)

Notes:
- We do not yet persist `duration_changes`, `human_performance`, `patch_functions`, or `test_functions` directly in instances; they are derivable from existing artifacts (results, diffs, parsers).
