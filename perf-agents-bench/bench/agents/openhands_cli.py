"""
OpenHands CLI Integration Module

This module provides a clean, reliable interface to OpenHands CLI for LLM benchmarking.
Replaces the complex Docker orchestration approach with direct CLI integration.
"""

from pathlib import Path
from typing import Dict, Any, Optional, Union
import subprocess
import tempfile
import time
import os
import logging
import sys

logger = logging.getLogger(__name__)


class OpenHandsCLIError(Exception):
    """Base exception for OpenHands CLI operations."""
    pass


class OpenHandsTimeoutError(OpenHandsCLIError):
    """Raised when OpenHands operation times out."""
    pass


class OpenHandsExecutionError(OpenHandsCLIError):
    """Raised when OpenHands execution fails."""
    pass


class OpenHandsCLI:
    """
    Clean OpenHands CLI integration following their official patterns.

    This replaces the complex Docker orchestration with direct CLI calls,
    making it more reliable and easier to debug.
    """

    def __init__(self, config_path: Optional[Path] = None, llm_config_path: Optional[Path] = None):
        """
        Initialize OpenHands CLI wrapper.

        Args:
            config_path: Path to OpenHands main config file (optional)
            llm_config_path: Path to LLM configuration file (optional)
        """
        self.config_path = config_path
        self.llm_config_path = llm_config_path

        # Validate OpenHands installation
        self._validate_installation()

    def _validate_installation(self) -> None:
        """Validate that OpenHands is properly installed and accessible."""
        try:
            # Test basic import
            import openhands
            logger.info(f"OpenHands version: {getattr(openhands, '__version__', 'unknown')}")

            # Test CLI availability
            result = subprocess.run(
                [sys.executable, "-m", "openhands.core.main", "--help"],
                capture_output=True,
                text=True,
                timeout=10
            )

            if result.returncode != 0:
                raise OpenHandsCLIError(f"OpenHands CLI not accessible: {result.stderr}")

        except subprocess.TimeoutExpired:
            raise OpenHandsCLIError("OpenHands CLI validation timed out")
        except subprocess.SubprocessError as e:
            raise OpenHandsCLIError(f"OpenHands CLI validation failed: {e}")
        except ImportError:
            raise OpenHandsCLIError("OpenHands Python package not installed")

    def run_task(
        self,
        workspace_dir: Path,
        task_description: str,
        max_iterations: int = 30,
        timeout_minutes: int = 60,
        agent_class: str = "CodeActAgent",
        budget_limit: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        Run OpenHands on a task using their CLI interface.

        Args:
            workspace_dir: Directory containing the code to work on
            task_description: Description of the optimization task
            max_iterations: Maximum iterations for the agent
            timeout_minutes: Timeout in minutes
            agent_class: Agent class to use (default: CodeActAgent)
            budget_limit: Maximum budget per task in USD

        Returns:
            Dict containing execution results and metadata
        """
        if not workspace_dir.exists():
            raise ValueError(f"Workspace directory does not exist: {workspace_dir}")

        if not workspace_dir.is_dir():
            raise ValueError(f"Workspace path is not a directory: {workspace_dir}")

        # Create temporary task file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write(task_description)
            task_file = Path(f.name)

        try:
            # Set environment variables for workspace
            env = os.environ.copy()
            
            # Ensure critical environment variables are set
            if 'OPENAI_API_KEY' not in env:
                logger.warning("OPENAI_API_KEY not found in environment - OpenHands may fail")
            
            # Ensure tmux is available in PATH
            if '/usr/bin' not in env.get('PATH', ''):
                env['PATH'] = env.get('PATH', '') + ':/usr/bin'
            
            # Mount the target workspace directory directly to /workspace for the runtime
            workspace_abs = os.path.abspath(str(workspace_dir))
            env['SANDBOX_VOLUMES'] = f'{workspace_abs}:/workspace:rw'
            # Pass through current user id when supported
            try:
                env['SANDBOX_USER_ID'] = str(os.getuid())
            except Exception:
                pass
            
            # Configure browser paths - use system default if browsers exist there
            system_browser_path = os.path.expanduser('~/.cache/ms-playwright')
            local_browser_path = os.path.abspath('./playwright_browsers')
            
            if os.path.exists(system_browser_path):
                # Use system-installed browsers
                env['PLAYWRIGHT_BROWSERS_PATH'] = system_browser_path
            else:
                # Fall back to local path
                os.makedirs(local_browser_path, exist_ok=True)
                env['PLAYWRIGHT_BROWSERS_PATH'] = local_browser_path
            
            # Build OpenHands command
            cmd = self._build_command(
                workspace_dir=workspace_dir,
                task_file=task_file,
                max_iterations=max_iterations,
                agent_class=agent_class,
                budget_limit=budget_limit
            )

            logger.info(f"Running OpenHands command: {' '.join(str(x) for x in cmd)}")

            # Execute with timeout
            start_time = time.time()
            try:
                result = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    timeout=timeout_minutes * 60,
                    cwd=workspace_dir,
                    env=env
                )
            except subprocess.TimeoutExpired:
                execution_time = time.time() - start_time
                logger.warning(f"OpenHands execution timed out after {execution_time:.1f}s")
                raise OpenHandsTimeoutError(
                    f"OpenHands execution timed out after {timeout_minutes} minutes"
                )

            execution_time = time.time() - start_time

            # Parse and return results
            return self._parse_result(
                result=result,
                execution_time=execution_time,
                task_file=task_file,
                workspace_dir=workspace_dir
            )

        finally:
            # Clean up temporary task file
            try:
                task_file.unlink(missing_ok=True)
            except Exception as e:
                logger.warning(f"Failed to clean up task file {task_file}: {e}")

    def _build_command(
        self,
        workspace_dir: Path,
        task_file: Path,
        max_iterations: int,
        agent_class: str,
        budget_limit: Optional[float]
    ) -> list:
        """Build the OpenHands CLI command."""
        import sys
        # Use the same Python executable that's currently running (from virtual environment)
        python_executable = sys.executable
        cmd = [
            python_executable, "-m", "openhands.core.main",
            # Use the mounted path inside the runtime so the agent works in /workspace
            "-d", "/workspace",
            "-f", str(task_file),
            "-i", str(max_iterations),
            "-c", agent_class
        ]

        # Add optional parameters
        if budget_limit is not None:
            cmd.extend(["-b", str(budget_limit)])

        # Use only the main config file (which includes LLM config inline)
        if self.config_path:
            cmd.extend(["--config-file", str(self.config_path)])
        # Note: llm_config_path is ignored since LLM config is inline in main config

        return cmd

    def _parse_result(
        self,
        result: subprocess.CompletedProcess,
        execution_time: float,
        task_file: Path,
        workspace_dir: Path
    ) -> Dict[str, Any]:
        """Parse OpenHands execution result."""
        # Start with basic process success
        process_success = result.returncode == 0
        
        # But check the actual agent execution state for true success
        agent_success = process_success  # Default to process success
        error_reason = None
        
        # Check stderr for agent state information
        if result.stderr:
            stderr_content = result.stderr
            
            # Check for specific failure patterns
            if "Agent reached maximum iteration" in stderr_content:
                agent_success = False
                error_reason = "Agent reached maximum iterations without completing task"
            elif "AgentState.ERROR" in stderr_content:
                agent_success = False
                error_reason = "Agent entered error state"
            elif "BadRequestError" in stderr_content or "OpenAIException" in stderr_content:
                agent_success = False
                error_reason = "LLM API error occurred"
            elif "Exception" in stderr_content and "agent_controller.py" in stderr_content:
                agent_success = False
                error_reason = "Agent controller exception"
        
        # Final success determination
        success = process_success and agent_success

        # Extract basic metadata
        result_data = {
            "returncode": result.returncode,
            "success": success,
            "agent_success": agent_success,
            "process_success": process_success,
            "execution_time_seconds": execution_time,
            "stdout": result.stdout,
            "stderr": result.stderr,
            "task_file": str(task_file),
            "workspace_dir": str(workspace_dir)
        }

        # Try to extract more detailed information from output
        if result.stdout:
            # Look for common patterns in OpenHands output
            lines = result.stdout.split('\n')

            # Extract iteration count if available
            for line in lines:
                if "iteration" in line.lower():
                    # Try to extract iteration information
                    pass

        if result.stderr:
            logger.warning(f"OpenHands stderr: {result.stderr}")

        if not success:
            if error_reason:
                error_msg = f"OpenHands task failed: {error_reason}"
            else:
                error_msg = f"OpenHands execution failed with return code {result.returncode}"
            
            if result.stderr and not error_reason:
                error_msg += f": {result.stderr[:500]}"  # Truncate long error messages
            
            logger.error(error_msg)
            result_data["error"] = error_msg

        return result_data

    def validate_workspace(self, workspace_dir: Path) -> bool:
        """
        Validate that a workspace directory is suitable for OpenHands execution.

        Args:
            workspace_dir: Directory to validate

        Returns:
            True if workspace is valid, False otherwise
        """
        if not workspace_dir.exists():
            logger.error(f"Workspace directory does not exist: {workspace_dir}")
            return False

        if not workspace_dir.is_dir():
            logger.error(f"Workspace path is not a directory: {workspace_dir}")
            return False

        # Check if it's a git repository (recommended for OpenHands)
        if not (workspace_dir / ".git").exists():
            logger.warning(f"Workspace is not a git repository: {workspace_dir}")

        return True


def create_opensource_task(
    task_name: str,
    description: str,
    target_files: list,
    constraints: list = None,
    reference_commit: str = None,
    primary_metric: str = None
) -> str:
    """
    Create a standardized task description for OpenHands.

    This formats the task in a way that OpenHands can understand and execute effectively.
    """
    sections = [
        "# Performance Optimization Task",
        "",
        f"## Task: {task_name}",
        f"## Description: {description}",
        "",
        "## Objective",
        "Your goal is to optimize the performance of the specified files while maintaining functionality.",
    ]

    if reference_commit:
        sections.extend([
            "",
            f"## Reference Information",
            f"A human developer optimized from commit {reference_commit}.",
            "Your goal is to achieve similar or better optimizations independently."
        ])

    sections.extend([
        "",
        "## Target Files",
        "You should ONLY modify these files:",
    ])

    for file in target_files:
        sections.append(f"- `{file}`")

    if constraints:
        sections.extend([
            "",
            "## Constraints",
            "You must follow these constraints:"
        ])
        for constraint in constraints:
            sections.append(f"- {constraint}")

    if primary_metric:
        sections.extend([
            "",
            "## Success Criteria",
            f"- Primary metric: {primary_metric}",
            "- All existing tests must pass",
            "- No regression in functionality",
            "",
            "## Instructions",
            "1. Analyze the target files for performance bottlenecks",
            "2. Implement optimizations while respecting constraints",
            "3. Test your changes to ensure correctness",
            "4. Commit your optimizations",
            "",
            "## Task Completion",
            "When you have successfully completed the task:",
            "1. Save all changes to the target files",
            "2. Run `git add .` to stage changes",
            "3. Run `git commit -m 'Add descriptive comments to functions'` to commit",
            "4. Use the `finish` command to indicate task completion",
            "",
            "The task is complete when you have added the requested comments and committed the changes."
        ])

    return "\n".join(sections)
