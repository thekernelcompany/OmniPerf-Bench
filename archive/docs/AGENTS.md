# Repository Guidelines

## Project Structure & Module Organization
- `src/harness/opt_at_k.py` orchestrates Opt@K scoring; keep harness helpers under `src/harness/`.
- `src/test_scripts/` holds LLM test-generation utilities and fixtures (JSON/CSV); treat outputs as read-only inputs.
- `src/collect/execute/skymgr.py` wraps SkyPilot cluster tasks; extend it instead of adding inline subprocess calls.
- `perf-agents-bench/bench/` contains the Typer CLI; anything mutating state writes to `perf-agents-bench/state/`.
- `tools/` exposes review scripts; keep generated datasets in `data/` or `misc/experiments/` but avoid committing large blobs.

## Build, Test, and Development Commands
- Install core deps via `uv pip install -r requirements.txt` (Python 3.12 keeps OpenHands happy).
- Provision the bench venv: `cd perf-agents-bench && uv venv --python 3.12 .venv && uv pip install -r requirements.txt -p .venv/bin/python`.
- Explore the commit pipeline with `uv run python commit_to_dataset.py --help` before wiring automation.
- Score prediction drops with `uv run python src/harness/opt_at_k.py --predictions-path reports/ --dataset-name gso --timeout 600 --k 5`.
- For QA, run `uv run python tools/manual_review.py <jsonl>` against newly generated tasks.

### Quickstart: One-command agent → Docker → Modal GPU tests

1. One-time setup:
   - Install Docker, `uv`, and `modal` CLI. Authenticate Modal: `modal token new`.
   - Run setup helper:
     ```bash
     bash OmniPerf-Bench/tools/setup_opb.sh
     ```

2. Full pipeline (give the perf commit SHA):
   ```bash
   bash OmniPerf-Bench/tools/run_agent_pipeline.sh b690e34824fd5a5c4054a0c0468ebfb6aa1dd215
   ```
   This will:
   - Run plan+prepare to let OpenHands edit the parent commit.
   - Build and push an overlay Docker image on top of your baseline image.
   - Deploy a Modal app, run tests on H100/A100/L40S, and download JSON results to `reports/<sha>/`.

Environment overrides (optional):
- `OPB_BASE_IMAGE_REPO` (default `ayushnangia16/nvidia-vllm-docker`)
- `OPB_MODAL_GENERATORS_VOLUME` (default `opb-generators`)
- `OPB_MODAL_RESULTS_VOLUME` (default `opb-results`)

## Coding Style & Naming Conventions
- Python-only code: 4-space indent, `snake_case` identifiers, `PascalCase` classes, constants in `UPPER_SNAKE_CASE`.
- Prefer type hints, docstrings, and small Typer command functions mirroring `bench/cli.py`.
- Load configuration through helpers like `_expand_env_vars`; avoid raw `os.environ[...]` in new code.
- Keep data artifacts out of source diffs; reference them via paths in docs or PRs.

## Testing Guidelines
- Run the safe integration check with `python perf-agents-bench/test_integration.py`; add `--real` only when budgeted.
- Execute unit suites with `uv run pytest perf-agents-bench` and `uv run pytest src/test_scripts` (stick to `test_*.py` files).
- When modifying evaluation logic, capture an Opt@K dry run and attach the resulting report path in review notes.

## Commit & Pull Request Guidelines
- Follow the existing format `Type: Title Case summary` (`Minor Update: Refresh harness docs`); keep subjects ≤72 chars.
- Keep commits focused and pair code with doc updates; stash large generated outputs in separate commits or gists.
- PRs must list executed commands, resulting artifacts under `perf-agents-bench/state/` or `reports/`, and call out new `.env` keys.

## Environment & Secrets
- Copy `perf-agents-bench/.env.example` to `.env`, populate locally, and never commit the file.
- Full harness runs need Docker with the daemon up before `bench.cli prepare`.
- Use throwaway worktrees for heavy dataset builds and prune them before pushing to avoid leaking cached runs.
