from __future__ import annotations
import json
import uuid
import subprocess
from pathlib import Path
from typing import Dict, Any

from .git_utils import clone_or_update, resolve_precommit, get_changed_files
from .testpack_api.loader import load_testpack
from .container.runtime import ContainerRuntime
from .env.build_plan import EnvBuilder
from .metrics.registry import run_metric
from .agents.openhands import OpenHandsAgent


def run_task(task: Dict[str, Any], bench_cfg: Dict[str, Any]):
    run_id = f"{task['id']}-{uuid.uuid4().hex[:8]}"
    
    work_root = Path(bench_cfg["paths"]["work_root"]).resolve()
    state_root = Path(bench_cfg["paths"]["state_root"]).resolve() 
    reports_root = Path(bench_cfg["paths"]["reports_root"]).resolve()
    
    run_dir = state_root / "runs" / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    # Clone and resolve commits
    repo_dir = work_root / f"{task['id']}-repo"
    work_root.mkdir(parents=True, exist_ok=True)
    clone_or_update(task["repo"]["url"], repo_dir)
    
    human = task["repo"]["human_commit"]
    pre = resolve_precommit(
        repo_dir, 
        human, 
        task["repo"].get("pre_commit"), 
        task["repo"].get("pre_parent_index")
    )

    # Container runtime; we'll build per-candidate images at the checked-out commit
    runtime = ContainerRuntime(bench_cfg["container"]["engine"])
    builder = EnvBuilder(runtime, Path(__file__).parent / "env" / "templates")

    # Load TestPack
    pack = load_testpack(task["testpack"]["entrypoint"])

    # Candidate runner helper
    def run_candidate(ref: str, tag: str):
        # Switch repo to ref
        subprocess.run(["git", "checkout", ref], cwd=repo_dir, check=True)

        cand_dir = run_dir / tag
        work_dir = cand_dir / "work"
        out_dir = cand_dir / "out"
        work_dir.mkdir(parents=True, exist_ok=True)
        out_dir.mkdir(parents=True, exist_ok=True)

        # Build container image for this commit so dependencies match
        if tag == "human":
            commit_sha = task["repo"]["human_commit"]
        elif tag == "baseline":
            commit_sha = pre
        else:
            commit_sha = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo_dir).decode().strip()
        short_sha = commit_sha[:12]
        image_tag = f"bench-{task['id']}-{tag}-{short_sha}"
        builder.build(repo_dir, task, image_tag=image_tag)

        # TestPack execution (host-level by default; TestPack can choose to call inside container)
        pack.prepare_fixtures(repo_dir, work_dir)
        pack.build(repo_dir, work_dir) 
        pack.run_candidate(repo_dir, work_dir, out_dir, tag)

        # Metrics collection
        metrics = []
        for spec in task.get("metrics", []):
            using = spec["using"]
            metrics.append(run_metric(using, spec, tag, str(out_dir)))
        
        (cand_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))

    # Run the three candidates
    run_candidate(pre, "baseline")
    run_candidate(human, "human")

    # Agent optimization
    subprocess.run(["git", "checkout", pre], cwd=repo_dir, check=True)
    
    if bench_cfg["agents"]["default"] == "openhands":
        agent_cfg = bench_cfg["agents"]["openhands"]
        agent = OpenHandsAgent(
            cli=agent_cfg["cli"],
            time_budget_minutes=agent_cfg["time_budget_minutes"]
        )
    else:
        raise NotImplementedError("Only 'openhands' agent configured in this scaffold.")
    
    agent_task = {
        "id": task["id"],
        "name": task["name"],
        "description": task.get("description", ""),
        "optimization_contract": task["optimization_contract"],
        "scoring": task["scoring"]
    }
    agent_branch = agent.optimize(repo_dir, agent_task)

    # Enforce target restrictions if requested
    if task["optimization_contract"].get("strict_targets"):
        _assert_only_targets_changed(repo_dir, pre, task["optimization_contract"]["target_files"])

    run_candidate(agent_branch, "agent")

    # Report generation
    report = _compare(task, run_dir)
    (run_dir / "report.json").write_text(json.dumps(report, indent=2))
    print(f"Done: {run_dir}")


def smoke_task(task: Dict[str, Any], bench_cfg: Dict[str, Any], cmd: str | None = None, include_agent: bool = True, use_human_for_all: bool = False):
    """Build and run per-commit containers (baseline, human, optional agent) and execute a simple command.
    Does not run TestPack or metrics. Intended as a quick sanity check.
    """
    work_root = Path(bench_cfg["paths"]["work_root"]).resolve()
    work_root.mkdir(parents=True, exist_ok=True)
    repo_dir = work_root / f"{task['id']}-repo"
    clone_or_update(task["repo"]["url"], repo_dir)

    human = task["repo"]["human_commit"]
    pre = resolve_precommit(
        repo_dir,
        human,
        task["repo"].get("pre_commit"),
        task["repo"].get("pre_parent_index"),
    )

    runtime = ContainerRuntime(bench_cfg["container"]["engine"])
    builder = EnvBuilder(runtime, Path(__file__).parent / "env" / "templates")

    def build_and_run(ref: str, tag: str):
        subprocess.run(["git", "checkout", ref], cwd=repo_dir, check=True)
        image_tag = f"bench-{task['id']}-{tag}"
        builder.build(repo_dir, task, image_tag=image_tag)
        smoke_cmd = cmd or "python -c 'import vllm; print(\"OK\")'"
        runtime.run(
            image=image_tag,
            repo_dir=repo_dir,
            mounts_ro={},
            cmd=smoke_cmd,
            cpus=str(bench_cfg["container"]["cpus"]),
            memory=str(bench_cfg["container"]["memory"]),
            gpus=str(bench_cfg["container"].get("gpus", "none")),
            network=str(bench_cfg["container"].get("network_policy", "off")),
        )

    # If requested, build all three from the human commit only
    if use_human_for_all:
        build_and_run(human, "baseline")
        build_and_run(human, "human")
        build_and_run(human, "agent")
        return

    # baseline and human (default behavior)
    build_and_run(pre, "baseline")
    build_and_run(human, "human")

    if include_agent:
        try:
            subprocess.run(["git", "checkout", pre], cwd=repo_dir, check=True)
            if bench_cfg["agents"]["default"] == "openhands":
                agent_cfg = bench_cfg["agents"]["openhands"]
                agent = OpenHandsAgent(
                    cli=agent_cfg["cli"],
                    time_budget_minutes=agent_cfg["time_budget_minutes"],
                    container_image=agent_cfg.get("container_image") or None,
                )
            else:
                raise NotImplementedError("Only 'openhands' agent configured in this scaffold.")

            agent_task = {
                "id": task["id"],
                "name": task["name"],
                "description": task.get("description", ""),
                "optimization_contract": task["optimization_contract"],
                "scoring": task["scoring"],
            }
            agent_branch = agent.optimize(repo_dir, agent_task)
            build_and_run(agent_branch, "agent")
        except Exception as e:
            print(f"Skipping agent smoke: {e}")


def _assert_only_targets_changed(repo_dir: Path, from_ref: str, targets: list[str]):
    changed = get_changed_files(repo_dir, from_ref, "HEAD")
    disallowed = [p for p in changed if p not in set(targets)]
    if disallowed:
        raise RuntimeError(f"Agent modified disallowed files: {disallowed}")


def _compare(task: Dict[str, Any], run_dir: Path) -> Dict[str, Any]:
    def read_metrics(tag: str):
        return json.loads((run_dir / tag / "metrics.json").read_text())
    
    primary = task["scoring"]["primary"]
    
    def get_value(metrics_list):
        return next(m["value"] for m in metrics_list if m["name"] == primary)
    
    baseline_metrics = read_metrics("baseline")
    human_metrics = read_metrics("human")
    agent_metrics = read_metrics("agent")
    
    return {
        "run_id": run_dir.name,
        "task_id": task["id"],
        "primary_metric": primary,
        "results": {
            "baseline": get_value(baseline_metrics),
            "human": get_value(human_metrics),
            "agent": get_value(agent_metrics)
        },
        "all_metrics": {
            "baseline": baseline_metrics,
            "human": human_metrics,
            "agent": agent_metrics
        }
    }