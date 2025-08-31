from __future__ import annotations
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, Any, Optional


class BuildStrategy(ABC):
    def __init__(self, runtime, templates_dir: Path, params: Dict[str, Any], runner_cfg: Dict[str, Any]):
        self.runtime = runtime
        self.templates_dir = templates_dir
        self.params = params
        self.runner_cfg = runner_cfg

    @abstractmethod
    def matches(self, repo_dir: Path) -> bool:
        pass

    @abstractmethod
    def build(self, repo_dir: Path, task: Dict[str, Any], image_tag: Optional[str] = None) -> str:
        pass