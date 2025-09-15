from __future__ import annotations
import subprocess
import tempfile
import json
from pathlib import Path
from typing import Dict, Any
from .base import Agent


class OpenHandsAgent(Agent):
    def __init__(self, cli: str, time_budget_minutes: int, container_image: str | None = None):
        self.cli = cli
        self.time_budget_minutes = time_budget_minutes
        self.container_image = container_image

    def optimize(self, repo_dir: Path, task: Dict[str, Any]) -> str:
        targets = task["optimization_contract"]["target_files"]
        constraints = task["optimization_contract"].get("constraints", [])
        branch = f"agent/{task['id']}"

        prompt = {
            "task": task["name"],
            "description": task.get("description", ""),
            "constraints": constraints,
            "target_files": targets,
            "success": {
                "primary_metric": task["scoring"]["primary"],
                "rules": [
                    "Do not modify tests or metrics harness",
                    "Preserve external behavior; optimize internals only"
                ]
            }
        }
        
        with tempfile.NamedTemporaryFile("w", delete=False, suffix=".json") as f:
            json.dump(prompt, f, indent=2)
            prompt_file = f.name

        try:
            if self.container_image:
                # Run OpenHands inside its own container, mounting repo and prompt
                cmd = [
                    "docker", "run", "--rm",
                    "-v", f"{repo_dir}:/workspace:rw",
                    "-v", f"{prompt_file}:/prompt.json:ro",
                    "-w", "/workspace",
                    self.container_image,
                    "run",
                    "--repo", "/workspace",
                    "--prompt-file", "/prompt.json",
                    "--time", str(self.time_budget_minutes),
                    "--branch", branch,
                ]
            else:
                # Run local CLI on host
                cmd = [
                    self.cli, "run",
                    "--repo", str(repo_dir),
                    "--prompt-file", prompt_file,
                    "--time", str(self.time_budget_minutes),
                    "--branch", branch
                ]
            subprocess.run(cmd, check=True)
            return branch
        finally:
            Path(prompt_file).unlink(missing_ok=True)