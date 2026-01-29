# Copyright (c) 2025
# SPDX-License-Identifier: MIT
"""
Command line entrypoint for the Codex agent.

Codex piggybacks on the TRAE agent runtime but adds hard guarantees that no
external web-search or document lookup tools are enabled. The CLI mirrors
`trae_agent.cli.run` so perf-agents-bench can invoke it just like TRAE.
"""

from __future__ import annotations

import asyncio
import os
import sys
import traceback
from pathlib import Path
from typing import Iterable

import click
from dotenv import load_dotenv
from rich.console import Console

PROJECT_ROOT = Path(__file__).resolve().parents[1]
THIRD_PARTY_TRAE = PROJECT_ROOT / "third-party" / "trae-agent"
if THIRD_PARTY_TRAE.exists():
    sys.path.insert(0, str(THIRD_PARTY_TRAE))

from trae_agent.agent import Agent
from trae_agent.utils.cli import CLIConsole, ConsoleFactory, ConsoleMode, ConsoleType
from trae_agent.utils.config import Config, TraeAgentConfig

_ = load_dotenv()

console = Console()


def _resolve_config_file(config_file: str) -> str:
    """Mirror TRAE's config resolution helper."""
    if config_file.endswith((".yaml", ".yml")):
        yaml_path = Path(config_file)
        json_path = Path(
            config_file.replace(".yaml", ".json").replace(".yml", ".json")
        )
        if yaml_path.exists():
            return str(yaml_path)
        if json_path.exists():
            console.print(
                f"[yellow]YAML config not found, using JSON config: {json_path}[/yellow]"
            )
            return str(json_path)
        console.print(
            "[red]Error: Config file not found. Please specify a valid config file in the command line option --config-file[/red]"
        )
        sys.exit(1)
    return config_file


def _ensure_no_search_capabilities(config: Config) -> None:
    """Abort early if the configuration would enable search/doc lookup."""
    trae_cfg: TraeAgentConfig | None = config.trae_agent
    if trae_cfg is None:
        console.print(
            "[red]Error: trae_agent configuration missing from Codex config file.[/red]"
        )
        sys.exit(1)

    disallowed_keywords: tuple[str, ...] = ("search", "lookup", "doc", "wiki")
    offending: list[str] = []
    for tool in trae_cfg.tools:
        lower = tool.lower()
        if any(keyword in lower for keyword in disallowed_keywords):
            offending.append(tool)

    if offending:
        console.print(
            "[red]Error: Codex agent does not permit web-search or document lookup tools.[/red]"
        )
        console.print(
            f"[yellow]Remove the following tools from codex_config.yaml: {', '.join(offending)}[/yellow]"
        )
        sys.exit(1)

    if trae_cfg.allow_mcp_servers:
        console.print(
            "[red]Error: Codex agent must not enable MCP servers (which could proxy search/doc access).[/red]"
        )
        console.print(
            f"[yellow]Remove allow_mcp_servers entries: {', '.join(trae_cfg.allow_mcp_servers)}[/yellow]"
        )
        sys.exit(1)

    if trae_cfg.mcp_servers_config:
        console.print(
            "[red]Error: MCP server configurations detected. Remove them to keep Codex offline.[/red]"
        )
        bad_servers: Iterable[str] = trae_cfg.mcp_servers_config.keys()
        console.print(f"[yellow]Offending MCP servers: {', '.join(bad_servers)}[/yellow]")
        sys.exit(1)


@click.group()
@click.version_option(version="0.1.0")
def cli() -> None:
    """Codex Agent CLI."""


@cli.command()
@click.argument("task", required=False)
@click.option("--file", "-f", "file_path", help="Path to a file containing the task.")
@click.option("--provider", "-p", help="LLM provider to use")
@click.option("--model", "-m", help="Specific model to use")
@click.option("--model-base-url", help="Base URL for the model API")
@click.option("--api-key", "-k", help="API key (or set via environment variable)")
@click.option("--max-steps", help="Maximum number of execution steps", type=int)
@click.option("--working-dir", "-w", help="Working directory for the agent")
@click.option("--must-patch", "-mp", is_flag=True, help="Whether to patch the code")
@click.option(
    "--config-file",
    help="Path to configuration file",
    default="codex_agent/codex_config.yaml",
    envvar="CODEX_CONFIG_FILE",
)
@click.option("--trajectory-file", "-t", help="Path to save trajectory file")
@click.option("--patch-path", "-pp", help="Path to patch file")
@click.option(
    "--console-type",
    "-ct",
    default="simple",
    type=click.Choice(["simple", "rich"], case_sensitive=False),
    help="Console type to use (simple or rich)",
)
@click.option(
    "--agent-type",
    "-at",
    type=click.Choice(["trae_agent"], case_sensitive=False),
    default="trae_agent",
    help="Underlying agent implementation to invoke (fixed to trae_agent).",
)
def run(
    task: str | None,
    file_path: str | None,
    patch_path: str,
    provider: str | None = None,
    model: str | None = None,
    model_base_url: str | None = None,
    api_key: str | None = None,
    max_steps: int | None = None,
    working_dir: str | None = None,
    must_patch: bool = False,
    config_file: str = "codex_agent/codex_config.yaml",
    trajectory_file: str | None = None,
    console_type: str | None = "simple",
    agent_type: str = "trae_agent",
) -> None:
    """Execute a task using the Codex agent."""

    config_file = _resolve_config_file(config_file)

    if file_path:
        if task:
            console.print(
                "[red]Error: Cannot use both a task string and the --file argument.[/red]"
            )
            sys.exit(1)
        try:
            task = Path(file_path).read_text()
        except FileNotFoundError:
            console.print(f"[red]Error: File not found: {file_path}[/red]")
            sys.exit(1)
    elif not task:
        console.print(
            "[red]Error: Must provide either a task string or use the --file argument.[/red]"
        )
        sys.exit(1)

    config = Config.create(config_file=config_file).resolve_config_values(
        provider=provider,
        model=model,
        model_base_url=model_base_url,
        api_key=api_key,
        max_steps=max_steps,
    )

    _ensure_no_search_capabilities(config)

    if console_type:
        selected_console_type = (
            ConsoleType.SIMPLE if console_type.lower() == "simple" else ConsoleType.RICH
        )
    else:
        selected_console_type = ConsoleFactory.get_recommended_console_type(
            ConsoleMode.RUN
        )

    cli_console: CLIConsole = ConsoleFactory.create_console(
        console_type=selected_console_type, mode=ConsoleMode.RUN
    )

    if selected_console_type == ConsoleType.RICH and hasattr(
        cli_console, "set_initial_task"
    ):
        cli_console.set_initial_task(task)

    agent = Agent(agent_type, config, trajectory_file, cli_console)

    if working_dir:
        try:
            os.chdir(working_dir)
            console.print(f"[blue]Changed working directory to: {working_dir}[/blue]")
        except Exception as exc:  # pragma: no cover - defensive
            console.print(f"[red]Error changing directory: {exc}[/red]")
            sys.exit(1)
    else:
        working_dir = os.getcwd()

    if hasattr(cli_console, "directory_manager"):
        from trae_agent.utils.working_directory_manager import WorkingDirectoryManager

        cli_console.directory_manager = WorkingDirectoryManager(working_dir)

    if not Path(working_dir).is_absolute():
        console.print(
            f"[red]Working directory must be absolute: {working_dir}. It should start with `/`[/red]"
        )
        sys.exit(1)

    try:
        if hasattr(cli_console, "directory_manager"):
            project_path = cli_console.directory_manager.get_agent_project_path()
        else:
            project_path = working_dir

        task_args = {
            "project_path": project_path,
            "issue": task,
            "must_patch": "true" if must_patch else "false",
            "patch_path": patch_path,
        }

        _ = asyncio.run(agent.run(task, task_args))

        console.print(f"\n[green]Trajectory saved to: {agent.trajectory_file}[/green]")

    except KeyboardInterrupt:
        console.print("\n[yellow]Task execution interrupted by user[/yellow]")
        console.print(f"[blue]Partial trajectory saved to: {agent.trajectory_file}[/blue]")
        sys.exit(1)
    except Exception as exc:  # pragma: no cover - surfaces agent failures
        console.print(f"\n[red]Unexpected error: {exc}[/red]")
        console.print(traceback.format_exc())
        console.print(f"[blue]Trajectory saved to: {agent.trajectory_file}[/blue]")
        sys.exit(1)


if __name__ == "__main__":
    cli()
