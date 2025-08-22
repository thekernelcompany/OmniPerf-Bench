from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Type, Any


@dataclass
class BuildResult:
    image_tag: str
    strategy: str


class EnvBuilder:
    def __init__(self, runtime, templates_dir: Path):
        self.runtime = runtime
        self.templates_dir = templates_dir
        
        from .strategies.dockerfile import DockerfileStrategy
        from .strategies.requirements import RequirementsStrategy
        
        self.strategies: Dict[str, Type] = {
            "dockerfile": DockerfileStrategy,
            "requirements": RequirementsStrategy,
        }

    def build(self, repo_dir: Path, task: Dict[str, Any], image_tag: str | None = None) -> BuildResult:
        allowed = task["env_build"]["allowed_strategies"]
        params = task["env_build"].get("params", {})
        
        for name in allowed:
            if name not in self.strategies:
                continue
                
            cls = self.strategies[name]
            strat = cls(self.runtime, self.templates_dir, params, task["runner"])
            
            if strat.matches(repo_dir):
                built_tag = strat.build(repo_dir, task, image_tag=image_tag)
                return BuildResult(image_tag=built_tag, strategy=name)
        
        raise RuntimeError(
            "No env strategy matched the repo and no explicit params were provided. "
            f"Tried strategies: {allowed}"
        )