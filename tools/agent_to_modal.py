"""
Automated pipeline:

Input: perf commit SHA (human/perf commit)
Steps:
 1) Run perf-agents-bench plan+prepare to let OpenHands edit the parent commit.
 2) Locate the generated worktree with agent edits.
 3) Write an overlay Dockerfile into the worktree and build a thin image on top of the
    baseline perf image, replacing vLLM with the agent-built wheel.
 4) Push the overlay image to the configured registry.
 5) Generate and deploy a Modal app bound to that overlay image.
 6) Upload generator scripts to a Modal volume and run GPU-matrix tests (H100/A100/L40S).
 7) Download results to reports/<sha>/.

Usage:
  uv run python OmniPerf-Bench/tools/agent_to_modal.py run <PERF_COMMIT_SHA>
"""

from __future__ import annotations

import json  # noqa: F401 (used in generated app template)
import os
import shutil
import subprocess
import sys  # noqa: F401 (potential future use / CLI)
import time  # noqa: F401 (timestamps in logs / future use)
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import typer


app = typer.Typer(add_completion=False, no_args_is_help=True)


# Defaults/config
DEFAULT_REGISTRY_REPO = os.environ.get(
    "OPB_BASE_IMAGE_REPO", "ayushnangia16/nvidia-vllm-docker"
)
DEFAULT_GPUS = ["H100", "A100", "L40S"]
DEFAULT_GENERATORS_REL = "misc/experiments/generated_test_generators_v4"
MODAL_RESULTS_VOLUME = os.environ.get("OPB_MODAL_RESULTS_VOLUME", "opb-results")
MODAL_GENERATORS_VOLUME = os.environ.get("OPB_MODAL_GENERATORS_VOLUME", "opb-generators")
MODAL_SOURCE_VOLUME = os.environ.get("OPB_MODAL_SOURCE_VOLUME", "opb-source")


@dataclass
class BenchPaths:
    repo_root: Path
    bench_dir: Path
    state_dir: Path
    work_dir: Path
    reports_dir: Path


def _run(cmd: List[str], cwd: Optional[Path] = None, env: Optional[dict] = None) -> None:
    subprocess.run(cmd, cwd=str(cwd) if cwd else None, env=env, check=True)


def _run_capture(cmd: List[str], cwd: Optional[Path] = None) -> str:
    out = subprocess.run(cmd, cwd=str(cwd) if cwd else None, check=True, capture_output=True, text=True)
    return out.stdout.strip()


def discover_paths() -> BenchPaths:
    # This file lives at OmniPerf-Bench/tools/...
    tools_path = Path(__file__).resolve()
    repo_root = tools_path.parents[1]
    bench_dir = repo_root / "perf-agents-bench"
    state_dir = bench_dir / "state"
    work_dir = bench_dir / ".work"
    reports_dir = repo_root / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    work_dir.mkdir(parents=True, exist_ok=True)
    return BenchPaths(
        repo_root=repo_root,
        bench_dir=bench_dir,
        state_dir=state_dir,
        work_dir=work_dir,
        reports_dir=reports_dir,
    )


def ensure_bench_venv(bench_dir: Path) -> Path:
    venv_py = bench_dir / ".venv" / "bin" / "python"
    if not venv_py.exists():
        # Provision venv and install deps using uv
        _run(["uv", "venv", "--python", "3.12", ".venv"], cwd=bench_dir)
        _run([
            "uv",
            "pip",
            "install",
            "-r",
            "requirements.txt",
            "-p",
            str(venv_py),
        ], cwd=bench_dir)
    return venv_py


def bench_plan_prepare(
    bench_dir: Path,
    venv_py: Path,
    perf_commit_sha: str,
    tasks_path: str = "tasks/vllm.yaml",
) -> None:
    commits_file = bench_dir / ".work" / "vllm_commits.txt"
    commits_file.write_text(f"{perf_commit_sha} parent=1\n", encoding="utf-8")

    typer.echo("[Stage A] Planning commits (bench.cli plan)...")
    # plan
    _run([
        str(venv_py),
        "-m",
        "bench.cli",
        "plan",
        tasks_path,
        "--commits",
        str(commits_file),
        "--out",
        "state/plan.json",
    ], cwd=bench_dir)

    typer.echo("[Stage A] Preparing agent runs (bench.cli prepare)...")
    # prepare
    _run([
        str(venv_py),
        "-m",
        "bench.cli",
        "prepare",
        tasks_path,
        "--from-plan",
        "state/plan.json",
        "--bench-cfg",
        "bench.yaml",
        "--max-workers",
        "1",
        "--resume",
    ], cwd=bench_dir)


def find_latest_worktree(work_root: Path) -> Path:
    """Locate the newest git worktree created during prepare under .work/worktrees/**.

    The prepare step uses RepoManager.create_worktree which places worktrees at:
      <work_root>/worktrees/<repo_name>/<item_id>
    """
    worktrees_root = work_root / "worktrees"
    if not worktrees_root.exists():
        raise FileNotFoundError(f"No worktrees found at: {worktrees_root}")

    latest_path: Optional[Path] = None
    latest_mtime: float = -1.0
    for wt in worktrees_root.rglob("*"):
        if not wt.is_dir():
            continue
        git_entry = wt / ".git"
        if git_entry.exists():
            try:
                mtime = wt.stat().st_mtime
            except FileNotFoundError:
                continue
            if mtime > latest_mtime:
                latest_mtime = mtime
                latest_path = wt
    if latest_path is None:
        # Aid debugging by listing immediate children
        children = "\n".join(str(p) for p in worktrees_root.iterdir())
        raise RuntimeError(
            "Could not locate any git worktree under .work/worktrees. Found entries:\n" + children
        )
    return latest_path


OVERLAY_DOCKERFILE = """# syntax=docker/dockerfile:1.4
ARG BASE_TAG
ARG VLLM_VERSION=0.0.0+agent
FROM {base_repo}:${{BASE_TAG}} as build
# Ensure setuptools-scm doesn't need VCS metadata
ENV SETUPTOOLS_SCM_PRETEND_VERSION=${{VLLM_VERSION}}
WORKDIR /src
# Copy the edited vLLM source
COPY . /src
RUN python3 -m pip install --upgrade pip build

FROM {base_repo}:${{BASE_TAG}}
# Match runtime python for installing the wheel
RUN apt-get update && apt-get install -y --no-install-recommends python3 python3-pip \
    && rm -rf /var/lib/apt/lists/*
COPY --from=build /src/dist/*.whl /tmp/vllm.whl
RUN python3 -m pip install --no-deps --force-reinstall /tmp/vllm.whl && rm -f /tmp/vllm.whl

# Optional: provide a place for results and mounted generators
RUN mkdir -p /results && mkdir -p /opb-generators
"""


def write_overlay_dockerfile(worktree: Path, base_repo: str) -> Path:
    typer.echo("[Stage B] Writing overlay Dockerfile...")
    df_path = worktree / "Dockerfile.overlay"
    df_path.write_text(OVERLAY_DOCKERFILE.format(base_repo=base_repo), encoding="utf-8")
    return df_path


def docker_build_and_push(
    worktree: Path, base_repo: str, base_tag: str, overlay_repo: str, overlay_tag: str
) -> str:
    typer.echo("[Stage B] Building overlay image...")
    # Determine short SHA from worktree HEAD
    short_sha = _run_capture(["git", "rev-parse", "--short", "HEAD"], cwd=worktree)
    full_tag = f"{overlay_repo}:{overlay_tag or (base_tag + '-agent-' + short_sha)}"
    build_cmd = [
        "docker",
        "build",
        "-t",
        full_tag,
        "--build-arg",
        f"BASE_TAG={base_tag}",
        "-f",
        "Dockerfile.overlay",
        ".",
    ]
    _run(build_cmd, cwd=worktree)
    typer.echo(f"[Stage B] Pushing overlay image: {full_tag} ...")
    _run(["docker", "push", full_tag])
    return full_tag


def modal_cli_available() -> bool:
    return shutil.which("modal") is not None


def modal_prepare_volumes(generators_dir: Path) -> None:
    typer.echo("[Stage C] Ensuring Modal volumes and uploading generators (idempotent)...")
    # Create volumes (ignore if exist)
    try:
        _run(["modal", "volume", "create", MODAL_GENERATORS_VOLUME])
    except Exception:
        pass
    try:
        _run(["modal", "volume", "create", MODAL_RESULTS_VOLUME])
    except Exception:
        pass

    # Upload generators directory
    if not generators_dir.exists():
        raise FileNotFoundError(f"Generators dir not found: {generators_dir}")
    try:
        _run([
            "modal",
            "volume",
            "put",
            MODAL_GENERATORS_VOLUME,
            str(generators_dir),
        ])
    except Exception:
        # If files already exist in the volume, continue.
        pass


def modal_prepare_source_volume(worktree: Path, source_dir_name: str) -> None:
    typer.echo("[Stage C] Uploading source worktree to Modal volume (idempotent)...")
    try:
        _run(["modal", "volume", "create", MODAL_SOURCE_VOLUME])
    except Exception:
        pass
    try:
        _run(["modal", "volume", "put", MODAL_SOURCE_VOLUME, str(worktree), f"/{source_dir_name}"])
    except Exception:
        pass


def write_modal_app(app_py: Path, base_image: str, source_dir_name: str) -> None:
    # No user echo here; called within Stage C
    # First, write the Dockerfile
    dockerfile_path = app_py.parent / "Dockerfile.modal"
    dockerfile_path.write_text(f"""FROM {base_image}

# Clear ENTRYPOINT and CMD from base image to allow Modal control
ENTRYPOINT []
CMD []

# Ensure python3 is accessible as python
RUN ln -sf /usr/bin/python3 /usr/bin/python
""", encoding="utf-8")
    
    # Preserve existing app.py to avoid f-string templating issues
    if not app_py.exists():
        repo_app = Path(__file__).resolve().parent / "modal_runner" / "app.py"
        if repo_app.exists():
            app_py.write_text(repo_app.read_text(encoding="utf-8"), encoding="utf-8")


def write_modal_submit(submit_py: Path, commit_sha: str, script_rel: str) -> None:
    # No user echo here; called within Stage C
    content = f"""import modal

SCRIPT = "{script_rel}"
ARGS_H100 = ["--reference", "--json-out", f"/results/{commit_sha}-H100.json"]
ARGS_A100 = ["--reference", "--json-out", f"/results/{commit_sha}-A100.json"]
ARGS_L40S = ["--reference", "--json-out", f"/results/{commit_sha}-L40S.json"]

test_h100 = modal.Function.from_name("vllm-agent-tests-v5", "test_h100")
test_a100 = modal.Function.from_name("vllm-agent-tests-v5", "test_a100")
test_l40s = modal.Function.from_name("vllm-agent-tests-v5", "test_l40s")

print(test_h100.remote(SCRIPT, ARGS_H100))
print(test_a100.remote(SCRIPT, ARGS_A100))
print(test_l40s.remote(SCRIPT, ARGS_L40S))
"""
    submit_py.write_text(content, encoding="utf-8")


def modal_deploy_and_run(modal_dir: Path) -> None:
    typer.echo("[Stage C] Deploying Modal app...")
    _run(["modal", "deploy", "app.py"], cwd=modal_dir)
    typer.echo("[Stage C] Running GPU tests via Modal submitter...")
    _run(["modal", "run", "submit.py"], cwd=modal_dir)


def modal_pull_results(reports_dir: Path, commit_sha: str) -> Path:
    typer.echo("[Stage D] Downloading results from Modal volume...")
    target = reports_dir / commit_sha
    target.mkdir(parents=True, exist_ok=True)
    _run([
        "modal",
        "volume",
        "get",
        MODAL_RESULTS_VOLUME,
        str(target),
    ])
    return target


@app.command()
def run(
    commit_sha: str = typer.Argument(..., help="Perf/human commit SHA (baseline image tag)"),
    base_repo: str = typer.Option(DEFAULT_REGISTRY_REPO, help="Docker registry repo for baseline and overlay images"),
    overlay_tag: str = typer.Option("", help="Override overlay tag; default '<sha>-agent-<short>'"),
    gpu: List[str] = typer.Option(DEFAULT_GPUS, "--gpu", help="GPUs to run on (H100, A100, L40S)"),
    generator_script_rel: str = typer.Option(
        None,
        help="Path under generators volume to the script to execute; auto-derived from commit SHA if not specified",
    ),
    generators_dir: Optional[Path] = typer.Option(
        None,
        help="Local path to generator scripts directory; defaults to misc/experiments/generated_test_generators_v4",
    ),
) -> None:
    """Execute the full pipeline for a given perf commit SHA."""

    paths = discover_paths()

    # Ensure bench venv and run plan+prepare
    typer.echo("[Stage A] Starting plan+prepare for OpenHands edits...")
    venv_py = ensure_bench_venv(paths.bench_dir)
    bench_plan_prepare(paths.bench_dir, venv_py, commit_sha)

    # Locate latest worktree with agent edits
    typer.echo("[Stage A] Locating latest agent-edited worktree under .work/worktrees...")
    worktree = find_latest_worktree(paths.work_dir)
    typer.echo(f"[Stage A] Using worktree: {worktree}")

    # Modal-only mode: skip local overlay build; install agent source in Modal job
    typer.echo("[Stage B] Skipping local overlay build; will install source on Modal.")
    overlay_full_tag = f"{base_repo}:{commit_sha}"

    # Modal prep
    if not modal_cli_available():
        typer.echo("ERROR: modal CLI not found. Install Modal and run 'modal token new' to login.", err=True)
        raise typer.Exit(code=2)

    gen_dir = generators_dir or (paths.repo_root / DEFAULT_GENERATORS_REL).resolve()
    modal_prepare_volumes(gen_dir)
    
    # Compute source directory name before uploading
    source_dir_name = Path(str(worktree)).name
    modal_prepare_source_volume(worktree, source_dir_name)

    # Auto-derive generator script from commit SHA if not specified
    if generator_script_rel is None:
        short_sha = commit_sha[:8]
        generator_script_rel = f"generated_test_generators_v4/{short_sha}_test_case_generator.py"
        typer.echo(f"[Stage C] Auto-derived generator script: {generator_script_rel}")

    # Generate Modal app and submitter with this overlay image and script
    typer.echo("[Stage C] Generating Modal app and submitter...")
    modal_dir = paths.repo_root / "tools" / "modal_runner"
    modal_dir.mkdir(parents=True, exist_ok=True)
    write_modal_app(modal_dir / "app.py", overlay_full_tag, source_dir_name)
    write_modal_submit(modal_dir / "submit.py", commit_sha=commit_sha, script_rel=generator_script_rel)

    # Deploy and run
    modal_deploy_and_run(modal_dir)

    # Pull results
    out_dir = modal_pull_results(paths.reports_dir, commit_sha)
    typer.echo(f"[Done] Results downloaded to: {out_dir}")


if __name__ == "__main__":
    app()


