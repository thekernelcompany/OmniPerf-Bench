from __future__ import annotations
import typer
import yaml
import os
import re
from pathlib import Path
from typing import Dict, Any

from .pipeline import run_task
from .pipeline import smoke_task
# Ensure metrics registry is populated by importing builtins
from . import metrics as _metrics_autoload  # noqa: F401


app = typer.Typer(add_completion=False)


def _expand_env_vars(obj: Any) -> Any:
    """Recursively expand environment variables in config objects."""
    if isinstance(obj, str):
        # Handle ${VAR:-default} syntax
        def replacer(match):
            var_expr = match.group(1)
            if ":-" in var_expr:
                var_name, default = var_expr.split(":-", 1)
                return os.environ.get(var_name, default)
            else:
                return os.environ.get(var_expr, "")
        
        return re.sub(r'\$\{([^}]+)\}', replacer, obj)
    elif isinstance(obj, list):
        return [_expand_env_vars(item) for item in obj]
    elif isinstance(obj, dict):
        return {k: _expand_env_vars(v) for k, v in obj.items()}
    else:
        return obj


def _load_bench_cfg(path: Path) -> Dict[str, Any]:
    """Load and expand environment variables in bench config."""
    cfg = yaml.safe_load(path.read_text())
    return _expand_env_vars(cfg)


@app.command()
def run(task: str, bench_cfg: str = "bench.yaml"):
    """Run a performance benchmarking task."""
    task_path = Path(task)
    bench_cfg_path = Path(bench_cfg)
    
    if not task_path.exists():
        typer.echo(f"Task file not found: {task_path}")
        raise typer.Exit(1)
    
    if not bench_cfg_path.exists():
        typer.echo(f"Bench config not found: {bench_cfg_path}")
        raise typer.Exit(1)
    
    try:
        config = _load_bench_cfg(bench_cfg_path)
        run_task(task_path, config)
    except Exception as e:
        typer.echo(f"Error: {e}")
        raise typer.Exit(1)


@app.command()
def validate(task: str):
    """Validate a task configuration file."""
    task_path = Path(task)
    
    if not task_path.exists():
        typer.echo(f"Task file not found: {task_path}")
        raise typer.Exit(1)
    
    try:
        task_config = yaml.safe_load(task_path.read_text())
        expanded = _expand_env_vars(task_config)
        
        # Basic validation
        required_fields = ["id", "name", "repo", "env_build", "testpack", "metrics"]
        missing = [field for field in required_fields if field not in expanded]
        
        if missing:
            typer.echo(f"Missing required fields: {missing}")
            raise typer.Exit(1)
        
        # Repo validation
        repo_required = ["url", "human_commit"]
        repo_missing = [field for field in repo_required if field not in expanded["repo"]]
        if repo_missing:
            typer.echo(f"Missing required repo fields: {repo_missing}")
            raise typer.Exit(1)
        
        # Pre-commit validation: optional; if omitted, defaults to first parent of human
        # Keep a soft note to inform users of the defaulting behavior
        if not expanded["repo"].get("pre_commit") and not expanded["repo"].get("pre_parent_index"):
            typer.echo("Info: pre_commit not provided; will default to first parent of human_commit.")
        
        typer.echo(f"✓ Task configuration is valid: {task}")
    except yaml.YAMLError as e:
        typer.echo(f"YAML parse error: {e}")
        raise typer.Exit(1)
    except Exception as e:
        typer.echo(f"Validation error: {e}")
        raise typer.Exit(1)


@app.command()
def smoke(task: str, bench_cfg: str = "bench.yaml", cmd: str = "", human_only: bool = False):
    """Build and run baseline/human (and agent if configured) images and execute a simple command.
    Default smoke command tries to import vllm. Pass --cmd to override.
    """
    task_path = Path(task)
    bench_cfg_path = Path(bench_cfg)
    if not task_path.exists():
        typer.echo(f"Task file not found: {task_path}")
        raise typer.Exit(1)
    if not bench_cfg_path.exists():
        typer.echo(f"Bench config not found: {bench_cfg_path}")
        raise typer.Exit(1)
    config = _load_bench_cfg(bench_cfg_path)
    smoke_task(task_path, config, cmd or None, use_human_for_all=human_only)

if __name__ == "__main__":
    app()