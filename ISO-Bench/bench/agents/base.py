from __future__ import annotations
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict


class Agent(ABC):
    @abstractmethod
    def optimize(self, repo_dir: Path, task: Dict[str, Any]) -> str:
        """
        Returns the git ref/branch for the agent result (e.g., 'agent/<task_id>').
        Must NOT modify files outside optimization_contract.target_files when strict_targets is true.
        """
        pass