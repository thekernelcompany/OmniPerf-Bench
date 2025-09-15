from __future__ import annotations
from pathlib import Path
from typing import Dict, Any, Optional
from .base import BuildStrategy


class DockerfileStrategy(BuildStrategy):
    def matches(self, repo_dir: Path) -> bool:
        dockerfile_path = self.params.get("dockerfile_path")
        if dockerfile_path:
            candidates = [repo_dir / dockerfile_path]
        else:
            candidates = [repo_dir / "Dockerfile"]
        return any(c.exists() for c in candidates)

    def build(self, repo_dir: Path, task: Dict[str, Any], image_tag: Optional[str] = None) -> str:
        tag = image_tag or f"bench-{task['id']}"
        platform = task.get("runner", {}).get("platform") or None
        return self.runtime.build(repo_dir, tag, platform=platform)