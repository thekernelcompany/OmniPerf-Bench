from __future__ import annotations
import subprocess
from pathlib import Path
from typing import Optional


def run(cmd: list[str], cwd: Optional[Path] = None):
    subprocess.run(cmd, cwd=cwd, check=True)


def clone_or_update(url: str, dest: Path):
    if dest.exists():
        run(["git", "fetch", "--all"], cwd=dest)
    else:
        run(["git", "clone", url, str(dest)])


def checkout(repo_dir: Path, ref: str):
    run(["git", "checkout", "-B", ref, ref], cwd=repo_dir)


def resolve_precommit(
    repo_dir: Path, 
    human: str, 
    pre_commit: Optional[str], 
    pre_parent_index: Optional[int]
) -> str:
    # If explicit pre-commit is provided, use it
    if pre_commit:
        return pre_commit

    # Otherwise, default to the first parent of the human commit unless an index is provided
    out = subprocess.check_output(
        ["git", "rev-list", "--parents", "-n", "1", human], cwd=repo_dir
    ).decode().strip().split()

    parents = out[1:]
    if not parents:
        raise ValueError("human_commit has no parents; please provide repo.pre_commit explicitly.")

    index = pre_parent_index or 1
    if index < 1 or index > len(parents):
        raise ValueError(
            f"human_commit has {len(parents)} parents, pre_parent_index={index} invalid."
        )

    return parents[index - 1]


def get_changed_files(repo_dir: Path, from_ref: str, to_ref: str) -> list[str]:
    out = subprocess.check_output(
        ["git", "diff", "--name-only", from_ref, to_ref], cwd=repo_dir
    ).decode().splitlines()
    return out