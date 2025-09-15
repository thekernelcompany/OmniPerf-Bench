from __future__ import annotations
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Mapping


class TestPack(ABC):
    schema_version = "1.0.0"

    @abstractmethod
    def prepare_fixtures(self, repo_dir: Path, work_dir: Path) -> None:
        """Place immutable fixtures under work_dir/fixtures. Can use network ONLY if allowed by task."""
        pass

    @abstractmethod
    def build(self, repo_dir: Path, work_dir: Path) -> None:
        """Any build commands specific to the repo (wheels, cache)."""
        pass

    @abstractmethod
    def run_candidate(self, repo_dir: Path, work_dir: Path, out_dir: Path, candidate_tag: str) -> None:
        """
        Execute baseline|human|agent and produce artifacts (e.g., out/<tag>.json).
        Must not rely on network. Must be deterministic given the sandbox.
        """
        pass

    def discover_entrypoints(self) -> Mapping[str, str]:
        """Optional hints like {'profile_cmd': 'python tools/profile.py --input fixtures/val'}."""
        return {}