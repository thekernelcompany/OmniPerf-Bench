from __future__ import annotations
import subprocess
from pathlib import Path
from typing import Dict, Optional


class ContainerRuntime:
    def __init__(self, engine: str):
        if engine not in ("docker", "podman"):
            raise ValueError("container.engine must be docker or podman")
        self.engine = engine

    def build(self, context_dir: Path, tag: str, platform: Optional[str] = None) -> str:
        args = [self.engine, "build", "-t", tag]
        if platform:
            args += ["--platform", platform]
        args += ["."]
        subprocess.run(args, cwd=context_dir, check=True)
        return tag

    def run(
        self,
        image: str,
        repo_dir: Path,
        mounts_ro: Dict[str, str],
        cmd: str,
        cpus: str,
        memory: str,
        gpus: str,
        network: str,
        env: Optional[Dict[str, str]] = None,
        return_output: bool = False,
    ) -> Optional[str]:
        args = [
            self.engine,
            "run",
            "--rm",
            "--cpus",
            cpus,
            "--memory",
            memory,
            "--security-opt=no-new-privileges",
            "--pids-limit",
            "4096",
            "--ulimit",
            "nofile=4096:4096",
            "--workdir",
            "/workspace",
        ]
        
        if network == "off":
            args += ["--network", "none"]
            
        if gpus and gpus != "none" and self.engine == "docker":
            args += ["--gpus", gpus]
            
        for host_path, container_path in mounts_ro.items():
            args += ["-v", f"{host_path}:{container_path}:ro"]
            
        args += ["-v", f"{repo_dir}:/workspace:rw"]
        
        if env:
            for k, v in env.items():
                args += ["-e", f"{k}={v}"]
                
        args += [image, "bash", "-lc", cmd]
        if return_output:
            out = subprocess.check_output(args)
            return out.decode()
        else:
            subprocess.run(args, check=True)
            return None