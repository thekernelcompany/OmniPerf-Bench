from __future__ import annotations
import subprocess
from pathlib import Path
from typing import Dict, Any, Tuple

from .git_utils import clone_or_update, resolve_precommit
from .container.runtime import ContainerRuntime
from .env.build_plan import EnvBuilder
from .agents.openhands import OpenHandsAgent


def _checkout(repo_dir: Path, ref: str) -> str:
    subprocess.run(["git", "checkout", ref], cwd=repo_dir, check=True)
    sha = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo_dir).decode().strip()
    return sha


def build_images(task: Dict[str, Any], bench_cfg: Dict[str, Any], include_agent: bool = False) -> Dict[str, str]:
    """Build per-candidate images for baseline and human (and optional agent).
    Returns a mapping of candidate->image_tag.
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

    tags: Dict[str, str] = {}

    # Baseline
    baseline_sha = _checkout(repo_dir, pre)
    baseline_short = baseline_sha[:12]
    baseline_tag = f"bench-{task['id']}-baseline-{baseline_short}"
    builder.build(repo_dir, task, image_tag=baseline_tag)
    tags["baseline"] = baseline_tag

    # Human
    human_sha = _checkout(repo_dir, human)
    human_short = human_sha[:12]
    human_tag = f"bench-{task['id']}-human-{human_short}"
    builder.build(repo_dir, task, image_tag=human_tag)
    tags["human"] = human_tag

    # Agent (optional)
    if include_agent:
        _checkout(repo_dir, pre)
        agent_cfg = bench_cfg["agents"]["openhands"]
        agent = OpenHandsAgent(cli=agent_cfg["cli"], time_budget_minutes=agent_cfg["time_budget_minutes"], container_image=agent_cfg.get("container_image") or None)
        agent_branch = agent.optimize(repo_dir, {
            "id": task["id"],
            "name": task["name"],
            "description": task.get("description", ""),
            "optimization_contract": task["optimization_contract"],
            "scoring": task["scoring"],
        })
        agent_sha = _checkout(repo_dir, agent_branch)
        agent_short = agent_sha[:12]
        agent_tag = f"bench-{task['id']}-agent-{agent_short}"
        builder.build(repo_dir, task, image_tag=agent_tag)
        tags["agent"] = agent_tag

    return tags

