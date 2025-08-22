from __future__ import annotations
import tempfile
from pathlib import Path
from typing import Dict, Any, Optional
from jinja2 import Template
from .base import BuildStrategy


class RequirementsStrategy(BuildStrategy):
    def matches(self, repo_dir: Path) -> bool:
        requirements_file = self.params.get("requirements_file")
        if requirements_file:
            candidates = [repo_dir / requirements_file]
        else:
            candidates = [
                repo_dir / "requirements.txt",
                repo_dir / "requirements-dev.txt", 
                repo_dir / "requirements/base.txt"
            ]
        return any(c.exists() for c in candidates)

    def build(self, repo_dir: Path, task: Dict[str, Any], image_tag: Optional[str] = None) -> str:
        requirements_file = self.params.get("requirements_file", "requirements.txt")
        python_version = self.runner_cfg.get("python_version", "3.10")
        requires_gpu = self.runner_cfg.get("requires_gpu", False)
        
        if requires_gpu:
            cuda_version = self.runner_cfg.get("cuda_version")
            if not cuda_version:
                raise ValueError("cuda_version required when requires_gpu=true")
            template_name = "base_cuda.j2"
        else:
            template_name = "base_cpu.j2"
            
        template_path = self.templates_dir / template_name
        if not template_path.exists():
            raise FileNotFoundError(f"Template not found: {template_path}")
            
        template = Template(template_path.read_text())
        dockerfile_content = template.render(
            python_version=python_version,
            requirements_file=requirements_file,
            cuda_version=self.runner_cfg.get("cuda_version", "12.1")
        )
        
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            (tmp_path / "Dockerfile").write_text(dockerfile_content)
            
            reqs_source = repo_dir / requirements_file
            if not reqs_source.exists():
                raise FileNotFoundError(f"Requirements file not found: {reqs_source}")
            (tmp_path / requirements_file).write_text(reqs_source.read_text())
            
            tag = image_tag or f"bench-{task['id']}"
            platform = task.get("runner", {}).get("platform") or None
            return self.runtime.build(tmp_path, tag, platform=platform)