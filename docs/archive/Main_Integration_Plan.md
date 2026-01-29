## OpenHands × Extraction Pipeline Integration Plan

### Goal
- **Unify** `ISO-Bench/` and `commit_to_dataset.py` into one coherent, optional-agent pipeline that: discovers commits, generates/validates efficiency tests, can run OpenHands, tracks cost/outcomes, and emits canonical dataset records.

### Reality check (current state)
- **commit_to_dataset.py**: clones repo, computes diff, LLM-generates test, runs timings (local/Docker), builds dataset record.
- **ISO-Bench/**: plans commit pairs, runs OpenHands headless (local/container), journals artifacts, summarizes results.
- **Gaps**: duplicated config/env handling, separate workspaces, no shared journals, dataset doesn’t call bench as a library.

### Key integration decisions
- **Single orchestrator**: `commit_to_dataset.py` remains entrypoint; it can call bench as a library for the optional OpenHands stage.
- **One config surface**: prefer `ISO-Bench/config/*.toml` plus project `.env`. Dataset reads these rather than duplicating flags.
- **One workspace policy**: reuse bench `RepoManager` worktrees for safety; never mutate user clones.
- **One cost model**: use bench `LLMConfigManager.get_cost_estimate` consistently; store estimate and journal evidence when available.
- **Modes**: `dry` (no API), `offline` (no OpenHands), `online` (OpenHands with iteration/budget caps).

### End‑to‑end flow (per commit)
1) Discover pair `(head, base)` from `extractions_dir`.
2) **Workspace**: use bench `RepoManager.ensure_base()` then `create_worktree(base, item_id)`.
3) **Efficiency test**: generate via current LLM path; quick local sanity run.
4) **Optional OpenHands**:
   - Build task via `create_opensource_task` with strict targets (diff-derived when unspecified).
   - Invoke `OpenHandsOrchestrator.optimize_repository(...)` with iteration/budget caps; collect `OptimizationResult` and journal paths.
5) **Timing runs**: execute generated test on `base`, `head`, and `main` (local; Docker if configured).
6) **Assemble record**: keep current fields; add agent metadata (branch/success/error/cost/iterations/journal_path).
7) **Persist** JSONL; optional HF push. Continue on failures with reason recorded.

### Interfaces and data contracts
- **Bench library calls**:
  - `OpenHandsOrchestrator(...).optimize_repository(repo_path, task_name, description, target_files, constraints, primary_metric, max_iterations, timeout_minutes)` → `OptimizationResult`.
  - `RepoManager(work_root, repo_url, repo_name).ensure_base()` and `.create_worktree(ref, item_id)`.
- **Journal ingestion**:
  - Read `state/runs/<run_id>/<item_id>/journal.json` and `openhands_stdout.txt` / `openhands_stderr.txt` when agent runs.
- **Dataset record extensions (optional fields)**:
  - `agent_branch`, `agent_success`, `agent_error`, `agent_cost_estimate`, `agent_iterations`, `agent_journal_path`.

### Configuration unification
- **Env**: single `.env` at `ISO-Bench/` root (LLM keys, base URLs, tokens).
- **TOML**: `config/main_<provider>.toml` as primary; dataset reads it (don’t duplicate knobs).
- **CLI flags from dataset**: `--mode=[dry|offline|online]`, `--iterations`, `--budget-usd`, `--timeout-min`, `--container=[on|off]`.

### Safety and resource policies
- **Workspace isolation**: always use `.work/` worktrees; avoid mutating user repo.
- **Network**: default off for timing; allow during OpenHands if configured.
- **Concurrency**: OpenHands sequential by default; timing runs can be parallelized modestly.
- **Guards**: strict timeouts and budget caps; clean failure handling.

### Error handling and fallbacks
- Missing OpenHands install/API keys → skip agent stage; record `agent_error`.
- Test generation fails → mark commit skipped with reason; continue.
- Worktree/git transient errors → retry once; then skip.

### Observability
- Surface bench journal path in records for traceability.
- Record cost estimates consistently; add measured usage later if available.
- Concise per-commit logs; aggregate end-of-run summary.

### Phased rollout
- **Phase 0**: Dataset loads bench config/provider; no OpenHands call yet.
- **Phase 1**: Dataset uses bench `RepoManager` for worktrees; timings unchanged.
- **Phase 2**: Add optional OpenHands call; store agent metadata; off by default.
- **Phase 3**: Add containerization toggles for both timings and OpenHands; validate on vLLM sample.
- **Phase 4**: Batch scale; tune concurrency/budget; HF push gating.

### Acceptance criteria
- `--mode=offline` produces records identical to current baseline.
- `--mode=online` (for ≥1 sample) yields clean agent outcome (success or timeout), stable journal path, and cost estimate within cap.
- No writes outside `.work/` and `data/`. HF push preserves nested fields (e.g., `efficiency_test`).

### Risks and mitigations
- **API cost runaway** → strict defaults, per-commit opt-in.
- **Environment drift** → containerized mode as parity baseline.
- **Schema creep** → only optional fields added; no breaking changes.
- **Long runtimes** → sensible timeouts, skip-on-failure, resumable batch.
