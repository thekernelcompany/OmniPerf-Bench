from __future__ import annotations
import typer
import yaml
import json
import os
import re
import uuid
import shutil
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional

from .pipeline import run_task
from .pipeline import smoke_task
from .planner import MatrixPlanner
from .prepare import PrepareExecutor
from .report import summarize_stage_a, summarize_all_runs
from .build_cmd import build_images
# Ensure metrics registry is populated by importing builtins
from . import metrics as _metrics_autoload  # noqa: F401


def _extract_repo_name(repo_url: str) -> str:
    """Extract repository name from URL or path."""
    if not repo_url:
        return "unknown"
    # Handle GitHub URLs
    if "github.com" in repo_url:
        # https://github.com/vllm-project/vllm.git -> vllm
        parts = repo_url.rstrip("/").rstrip(".git").split("/")
        return parts[-1] if parts else "unknown"
    # Handle local paths
    return Path(repo_url).name or "unknown"


def _get_model_name(bench_cfg: Dict[str, Any], agent_name: str) -> str:
    """Extract model name from bench config or environment."""
    # Check environment first
    model_env = os.environ.get("LLM_MODEL")
    if model_env:
        return _sanitize_path_component(model_env)

    # Try to read from agent's config file
    agent_cfg = bench_cfg.get("agents", {}).get(agent_name, {})
    config_file = agent_cfg.get("config_file")

    if config_file:
        config_path = Path(config_file)
        # Expand environment variables in path
        config_path_str = os.path.expandvars(str(config_path))
        config_path = Path(config_path_str)

        if config_path.exists():
            try:
                cfg_data = yaml.safe_load(config_path.read_text())
                # Try common model config patterns
                if "models" in cfg_data:
                    for model_key, model_cfg in cfg_data["models"].items():
                        if "model" in model_cfg:
                            return _sanitize_path_component(model_cfg["model"])
                if "llm" in cfg_data and "model" in cfg_data["llm"]:
                    return _sanitize_path_component(cfg_data["llm"]["model"])
            except Exception:
                pass

    return "default"


def _sanitize_path_component(name: str) -> str:
    """Sanitize a string for use in directory path."""
    if not name:
        return "unknown"
    # Replace special characters with dashes
    sanitized = re.sub(r'[^\w\-]', '-', name.lower())
    # Remove consecutive dashes
    sanitized = re.sub(r'-+', '-', sanitized)
    # Remove leading/trailing dashes
    return sanitized.strip('-') or "unknown"


def _build_hierarchical_run_path(
    repo_name: str,
    agent_name: str,
    model_name: str,
    timestamp: Optional[str] = None
) -> str:
    """Build hierarchical run path: repo/agent/model/timestamp."""
    if timestamp is None:
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    return f"{repo_name}/{agent_name}/{model_name}/{timestamp}"


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


def _load_task_cfg(path: Path) -> Dict[str, Any]:
    """Load and expand environment variables in task config."""
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
        task_cfg = _load_task_cfg(task_path)
        run_task(task_cfg, config)
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
    task_cfg = _load_task_cfg(task_path)
    smoke_task(task_cfg, config, cmd or None, use_human_for_all=human_only)


@app.command()
def plan(task: str, commits: Optional[str] = typer.Option(None, help="Path to commits.txt or YAML with pairs"), out: str = typer.Option("state/plan.json", help="Path to write plan JSON")):
    """Resolve commit pairs (human/pre) into a plan file for bulk preparation."""
    task_path = Path(task)
    if not task_path.exists():
        typer.echo(f"Task file not found: {task_path}")
        raise typer.Exit(1)
    try:
        task_cfg = _load_task_cfg(task_path)
        planner = MatrixPlanner()
        plan = planner.build_plan(task_cfg, commits_path=Path(commits) if commits else None)
        out_path = Path(out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        if out_path.suffix == ".json":
            out_path.write_text(json.dumps(plan, indent=2))
        else:
            out_path.write_text(yaml.safe_dump(plan))
        typer.echo(f"✓ Wrote plan to {out_path}")
    except Exception as e:
        typer.echo(f"Planning error: {e}")
        raise typer.Exit(1)


@app.command()
def prepare(task: str, from_plan: str = typer.Option("state/plan.json", "--from-plan", "-p"), bench_cfg: str = "bench.yaml", max_workers: int = 4, resume: bool = True, run_id: Optional[str] = typer.Option(None, "--run-id", help="Resume a specific run ID (e.g., vllm/claude_code/default/2025-12-22_21-40-38)")):
    """Run OpenHands (host) for each plan item, enforce targets, and write journals."""
    task_p = Path(task)
    plan_p = Path(from_plan)
    bench_cfg_path = Path(bench_cfg)
    if not task_p.exists():
        typer.echo(f"Task file not found: {task_p}")
        raise typer.Exit(1)
    if not plan_p.exists():
        typer.echo(f"Plan file not found: {plan_p}")
        raise typer.Exit(1)
    if not bench_cfg_path.exists():
        typer.echo(f"Bench config not found: {bench_cfg_path}")
        raise typer.Exit(1)

    try:
        task_cfg = _load_task_cfg(task_p)
        cfg = _load_bench_cfg(bench_cfg_path)

        # Build hierarchical run path: repo/agent/model/timestamp
        repo_url = task_cfg.get("repo", {}).get("url", "")
        repo_name = _extract_repo_name(repo_url)
        agent_name = str(cfg.get("agents", {}).get("default", "unknown"))
        model_name = _get_model_name(cfg, agent_name)

        # Use provided run_id or generate new one
        if run_id:
            run_path = run_id
        else:
            run_path = _build_hierarchical_run_path(repo_name, agent_name, model_name)

        typer.echo(f"Run path: {run_path}")
        typer.echo(f"  Repo:   {repo_name}")
        typer.echo(f"  Agent:  {agent_name}")
        typer.echo(f"  Model:  {model_name}")

        executor = PrepareExecutor(cfg, run_id=run_path)
        executor.execute(task_cfg, plan_p, max_workers=max_workers, resume=resume)
        typer.echo(f"✓ Prepare completed: state/runs/{run_path}")
    except Exception as e:
        typer.echo(f"Prepare error: {e}")
        raise typer.Exit(1)


@app.command()
def init(out: str = typer.Option(".", help="Directory to write scaffolding into")):
    """Scaffold a minimal Stage A setup: example task and commits.txt."""
    out_dir = Path(out)
    tasks_dir = out_dir / "tasks"
    work_dir = out_dir / ".work"
    tasks_dir.mkdir(parents=True, exist_ok=True)
    work_dir.mkdir(parents=True, exist_ok=True)

    example_task = {
        "id": "sample_task",
        "name": "Sample optimization",
        "description": "Optimize selected files",
        "repo": {
            "url": "${REPO_URL}",
            "human_commit": "${HUMAN_COMMIT}",
            "pre_commit": "${PRE_COMMIT}",
        },
        "runner": {
            "requires_gpu": False,
            "python_version": None,
            "allow_network_during_prepare": True,
        },
        "env_build": {
            "allowed_strategies": ["dockerfile", "requirements"],
            "params": {"dockerfile_path": None, "requirements_file": None},
        },
        "optimization_contract": {
            "strict_targets": True,
            "target_files": ["src/module.py"],
            "constraints": ["No public API breakage"],
        },
        "testpack": {"entrypoint": "../vlm-bench-generic"},
        "metrics": [],
        "scoring": {"primary": "throughput", "tie_breaker": "functional"},
    }
    example_path = tasks_dir / "example.yaml"
    example_path.write_text(yaml.safe_dump(example_task))

    commits_file = work_dir / "commits.txt"
    if not commits_file.exists():
        commits_file.write_text("# <human_sha> [<pre_sha>|parent=1]\n")

    typer.echo(f"✓ Wrote {example_path} and {commits_file}")


@app.command()
def doctor(bench_cfg: str = "bench.yaml"):
    """Check environment prerequisites for Stage A and Stage B (optional)."""
    import subprocess
    ok = True
    try:
        subprocess.check_output(["git", "--version"])  # type: ignore[arg-type]
        typer.echo("✓ git found")
    except Exception as e:
        ok = False
        typer.echo(f"✗ git not found: {e}")

    try:
        cfg = _load_bench_cfg(Path(bench_cfg))
        cli = cfg["agents"]["openhands"]["cli"]
        cli_path = shutil.which(cli) or cli
        # Accept either an executable path or a successful --help invocation
        if Path(cli_path).exists() and os.access(cli_path, os.X_OK):
            typer.echo(f"✓ OpenHands CLI found: {cli_path}")
        else:
            res = subprocess.run([cli, "--help"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            if res.returncode == 0:
                typer.echo(f"✓ OpenHands CLI available: {cli}")
            else:
                ok = False
                typer.echo(f"✗ OpenHands CLI not available: {cli}")
    except Exception as e:
        ok = False
        typer.echo(f"✗ OpenHands CLI not found: {e}")

    try:
        subprocess.check_output(["docker", "--version"])  # type: ignore[arg-type]
        typer.echo("✓ docker found (for Stage B)")
    except Exception as e:
        typer.echo(f"! docker not found (needed for Stage B): {e}")

    raise typer.Exit(0 if ok else 1)


@app.command()
def report(run_dir: str = typer.Argument(..., help="Path to state/runs/<run_id>")):
    """Summarize Stage A journals into a compact JSON report printed to stdout."""
    try:
        out = summarize_stage_a(Path(run_dir))
        typer.echo(json.dumps(out, indent=2))
    except Exception as e:
        typer.echo(f"Report error: {e}")
        raise typer.Exit(1)


@app.command(name="report-all")
def report_all(state_root: str = typer.Option("./state", help="Path to state directory")):
    """Summarize all runs organized by repo/agent/model hierarchy."""
    try:
        out = summarize_all_runs(Path(state_root))
        typer.echo(json.dumps(out, indent=2))
    except Exception as e:
        typer.echo(f"Report error: {e}")
        raise typer.Exit(1)


@app.command()
def migrate(
    state_root: str = typer.Option("./state", help="Path to state directory"),
    dry_run: bool = typer.Option(True, "--dry-run/--execute", help="Preview changes without moving files"),
):
    """Migrate runs from flat to hierarchical structure (repo/agent/model/timestamp).

    Use --execute to actually perform the migration.
    """
    import sys
    # Import migration module
    migrate_script = Path(__file__).parent.parent / "migrate_runs.py"
    if not migrate_script.exists():
        typer.echo(f"Migration script not found: {migrate_script}")
        raise typer.Exit(1)

    # Import the migration module dynamically
    import importlib.util
    spec = importlib.util.spec_from_file_location("migrate_runs", migrate_script)
    if spec is None or spec.loader is None:
        typer.echo("Failed to load migration module")
        raise typer.Exit(1)

    migrate_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(migrate_module)

    # Run migration
    typer.echo(f"State root: {Path(state_root).resolve()}")
    typer.echo(f"Mode: {'DRY-RUN (preview only)' if dry_run else 'EXECUTE (will move files)'}")
    typer.echo()

    if not dry_run:
        if not typer.confirm("This will reorganize all run directories. Continue?"):
            typer.echo("Aborted.")
            raise typer.Exit(0)

    results = migrate_module.migrate_runs(Path(state_root).resolve(), dry_run=dry_run)
    migrate_module.print_summary(results)

    if dry_run:
        typer.echo("\nTo execute the migration, run with --execute flag:")
        typer.echo(f"  python -m bench.cli migrate --state-root {state_root} --execute")


@app.command()
def build(task: str, bench_cfg: str = "bench.yaml", include_agent: bool = False):
    """Docker-only: build baseline/human (and optional agent) images with canonical tags."""
    task_path = Path(task)
    bench_cfg_path = Path(bench_cfg)
    if not task_path.exists():
        typer.echo(f"Task file not found: {task_path}")
        raise typer.Exit(1)
    if not bench_cfg_path.exists():
        typer.echo(f"Bench config not found: {bench_cfg_path}")
        raise typer.Exit(1)
    try:
        cfg = _load_bench_cfg(bench_cfg_path)
        task_cfg = _load_task_cfg(task_path)
        tags = build_images(task_cfg, cfg, include_agent=include_agent)
        typer.echo(json.dumps(tags, indent=2))
    except Exception as e:
        typer.echo(f"Build error: {e}")
        raise typer.Exit(1)


@app.command()
def evaluate(
    repo_path: str = typer.Option(..., "--repo", "-r", help="Path to repository (vllm or sglang)"),
    run_ids: Optional[str] = typer.Option(None, "--run-ids", help="Comma-separated run IDs to evaluate (default: all)"),
    test_dataset: str = typer.Option("Inferencebench/test-generation-scripts", "--test-dataset", help="HuggingFace dataset ID for test scripts"),
    output_dir: str = typer.Option("eval_results", "--output-dir", "-o", help="Output directory for results"),
    timeout: int = typer.Option(600, "--timeout", "-t", help="Timeout per test in seconds"),
    download_only: bool = typer.Option(False, "--download-only", help="Only download and index test scripts"),
    report_only: bool = typer.Option(False, "--report-only", help="Only generate report from existing results"),
):
    """
    Evaluate agent patches against HuggingFace test scripts.

    Downloads test scripts from HuggingFace, matches them to agent runs by commit hash,
    runs tests on both baseline and patched code, and reports performance metrics.

    Example:
        python -m bench.cli evaluate --repo /path/to/vllm --output-dir eval_results/
    """
    import sys
    # Add src to path for eval module imports
    src_path = Path(__file__).resolve().parents[2] / "src"
    if str(src_path) not in sys.path:
        sys.path.insert(0, str(src_path))

    try:
        from eval.download_tests import download_and_index_tests, load_test_index
        from eval.run_tests_on_patches import TestRunner
        from eval.aggregate_results import aggregate_results, generate_report, print_summary
    except ImportError as e:
        typer.echo(f"Error importing eval module: {e}")
        typer.echo("Make sure src/eval/ exists and huggingface_hub is installed.")
        raise typer.Exit(1)

    output_path = Path(output_dir)
    repo = Path(repo_path)
    state_root = Path(__file__).resolve().parents[1] / "state"

    # Report only mode
    if report_only:
        if not output_path.exists():
            typer.echo(f"Output directory not found: {output_path}")
            raise typer.Exit(1)

        typer.echo("Generating report from existing results...")
        summaries = aggregate_results(output_path)
        print_summary(summaries)
        report = generate_report(summaries, output_path / "evaluation_report.json")
        typer.echo(f"\n✓ Report saved to {output_path / 'evaluation_report.json'}")
        raise typer.Exit(0)

    # Download and index test scripts
    typer.echo(f"Downloading test scripts from {test_dataset}...")
    try:
        test_index = download_and_index_tests()
        typer.echo(f"✓ Indexed {len(test_index)} test scripts")
    except Exception as e:
        typer.echo(f"Error downloading test scripts: {e}")
        raise typer.Exit(1)

    if download_only:
        typer.echo("Download complete (--download-only specified)")
        raise typer.Exit(0)

    # Validate repo path
    if not repo.exists():
        typer.echo(f"Repository not found: {repo}")
        raise typer.Exit(1)

    # Parse run IDs
    run_id_list = None
    if run_ids:
        run_id_list = [r.strip() for r in run_ids.split(",")]

    # Create test runner
    runner = TestRunner(
        repo_path=repo,
        state_root=state_root,
        output_dir=output_path,
        test_index=test_index,
        timeout=timeout,
    )

    # Discover runs
    runs = runner.discover_runs(run_id_list)
    if not runs:
        typer.echo("No runs found to evaluate")
        raise typer.Exit(1)

    typer.echo(f"Found {len(runs)} commits to evaluate")

    # Count how many have tests
    with_tests = sum(1 for r in runs if r.test_script_path)
    typer.echo(f"  - With matching tests: {with_tests}")
    typer.echo(f"  - Without tests: {len(runs) - with_tests}")

    # Run evaluation
    typer.echo("\nStarting evaluation...")
    results = runner.run_all(run_id_list)

    # Generate report
    typer.echo("\nGenerating report...")
    summaries = aggregate_results(output_path)
    print_summary(summaries)
    report = generate_report(summaries, output_path / "evaluation_report.json")

    typer.echo(f"\n✓ Evaluation complete. Results in {output_path}")
    typer.echo(f"✓ Report saved to {output_path / 'evaluation_report.json'}")


if __name__ == "__main__":
    app()