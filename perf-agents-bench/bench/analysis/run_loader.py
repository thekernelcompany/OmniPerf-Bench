"""Run artifact loader for soft metrics analysis.

Loads all files from a benchmark run directory including:
- journal.json, run_summary.json, prompt.json
- Agent-specific stdout/stderr logs
- trajectory.json, model_patch.diff
- diff_targets.json, task.txt, prediction.jsonl
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Dict, Any, Optional, List

logger = logging.getLogger(__name__)


# Mapping of agent types to their log file names
AGENT_LOG_FILES = {
    "trae": {
        "stdout": "trae_stdout.txt",
        "stderr": "trae_stderr.txt",
    },
    "openhands": {
        "stdout": "openhands_stdout.txt",
        "stderr": "openhands_stderr.txt",
    },
    "codex": {
        "stdout": "codex_stdout.txt",
        "stderr": "codex_stderr.txt",
    },
    "claude_code": {
        "stdout": "claude_code_stdout.json",  # Stream JSON format
        "stderr": "claude_code_stderr.txt",
    },
}


class RunLoader:
    """Load all artifacts from a benchmark run directory.

    Handles different agent types and their specific file formats.
    """

    def __init__(self, item_dir: Path):
        """Initialize loader for a run directory.

        Args:
            item_dir: Path to the run item directory
                     (e.g., state/runs/vllm/trae/claude-sonnet-45/2025-11-27/item-0001)
        """
        self.item_dir = Path(item_dir)
        self._agent_type: Optional[str] = None
        self._data_cache: Dict[str, Any] = {}

    @property
    def agent_type(self) -> str:
        """Detect agent type from available files."""
        if self._agent_type:
            return self._agent_type

        # Try to detect from path
        parts = self.item_dir.parts
        try:
            runs_idx = [i for i, p in enumerate(parts) if p == "runs"][0]
            if runs_idx + 2 < len(parts):
                self._agent_type = parts[runs_idx + 2]
                return self._agent_type
        except (IndexError, ValueError):
            pass

        # Try to detect from files
        for agent, files in AGENT_LOG_FILES.items():
            stdout_file = self.item_dir / files["stdout"]
            if stdout_file.exists():
                self._agent_type = agent
                return self._agent_type

        # Check journal for agent info
        journal = self._load_json("journal.json")
        for agent in AGENT_LOG_FILES.keys():
            if agent in journal:
                self._agent_type = agent
                return self._agent_type

        self._agent_type = "unknown"
        return self._agent_type

    def extract_metadata(self) -> Dict[str, Any]:
        """Extract metadata from directory path and files."""
        parts = self.item_dir.parts

        # Try to extract from path structure
        try:
            runs_idx = [i for i, p in enumerate(parts) if p == "runs"][0]
            meta = {
                "repo": parts[runs_idx + 1] if runs_idx + 1 < len(parts) else "unknown",
                "agent": parts[runs_idx + 2] if runs_idx + 2 < len(parts) else "unknown",
                "model": parts[runs_idx + 3] if runs_idx + 3 < len(parts) else "unknown",
                "timestamp": parts[runs_idx + 4] if runs_idx + 4 < len(parts) else "unknown",
                "item_id": parts[runs_idx + 5] if runs_idx + 5 < len(parts) else self.item_dir.name,
            }
        except (IndexError, ValueError):
            meta = {
                "repo": "unknown",
                "agent": self.agent_type,
                "model": "unknown",
                "timestamp": "unknown",
                "item_id": self.item_dir.name,
            }

        # Enrich from run_summary.json
        run_summary = self._load_json("run_summary.json")
        if run_summary:
            summary_meta = run_summary.get("meta", {})
            meta.update({
                "model_full": summary_meta.get("model_full", meta["model"]),
                "task_id": summary_meta.get("task_id", ""),
            })
            meta["commits"] = run_summary.get("commits", {})

        # Fallback to journal
        if not meta.get("commits"):
            journal = self._load_json("journal.json")
            meta["commits"] = journal.get("commits", {})
            meta["task_id"] = journal.get("task_id", meta.get("task_id", ""))

        return meta

    def load_all(self) -> Dict[str, Any]:
        """Load all artifacts from the run directory.

        Returns:
            Dict with all loaded data:
                - metadata: Extracted metadata
                - journal: journal.json content
                - run_summary: run_summary.json content
                - prompt: prompt.json content
                - diff_targets: diff_targets.json content
                - task: task.txt content
                - stdout: Agent stdout content
                - stderr: Agent stderr content
                - patch: model_patch.diff content
                - trajectory: trajectory.json content
                - prediction: prediction.jsonl content (list of dicts)
                - source_files: Paths to all source files
        """
        data = {
            "metadata": self.extract_metadata(),
            "journal": self._load_json("journal.json"),
            "run_summary": self._load_json("run_summary.json"),
            "prompt": self._load_json("prompt.json"),
            "diff_targets": self._load_json("diff_targets.json"),
            "task": self._load_text("task.txt"),
            "stdout": self._load_stdout(),
            "stderr": self._load_stderr(),
            "patch": self._load_text("model_patch.diff"),
            "trajectory": self._load_json("trajectory.json"),
            "prediction": self._load_jsonl("prediction.jsonl"),
            "source_files": self._get_source_files(),
        }
        return data

    def _load_json(self, filename: str) -> Dict[str, Any]:
        """Load JSON file, returning empty dict if not found."""
        if filename in self._data_cache:
            return self._data_cache[filename]

        filepath = self.item_dir / filename
        if not filepath.exists():
            logger.debug(f"File not found: {filepath}")
            return {}

        try:
            data = json.loads(filepath.read_text())
            self._data_cache[filename] = data
            return data
        except json.JSONDecodeError as e:
            logger.warning(f"Failed to parse JSON {filepath}: {e}")
            return {}
        except Exception as e:
            logger.warning(f"Failed to load {filepath}: {e}")
            return {}

    def _load_text(self, filename: str) -> str:
        """Load text file, returning empty string if not found."""
        filepath = self.item_dir / filename
        if not filepath.exists():
            logger.debug(f"File not found: {filepath}")
            return ""

        try:
            return filepath.read_text()
        except Exception as e:
            logger.warning(f"Failed to load {filepath}: {e}")
            return ""

    def _load_jsonl(self, filename: str) -> List[Dict[str, Any]]:
        """Load JSONL file, returning empty list if not found."""
        filepath = self.item_dir / filename
        if not filepath.exists():
            logger.debug(f"File not found: {filepath}")
            return []

        try:
            lines = filepath.read_text().strip().split("\n")
            return [json.loads(line) for line in lines if line.strip()]
        except Exception as e:
            logger.warning(f"Failed to load {filepath}: {e}")
            return []

    def _load_stdout(self) -> str:
        """Load agent-specific stdout file."""
        agent = self.agent_type
        if agent in AGENT_LOG_FILES:
            filename = AGENT_LOG_FILES[agent]["stdout"]
            if agent == "claude_code":
                # Claude Code uses stream JSON format
                return self._load_text(filename)
            return self._load_text(filename)

        # Fallback: try all known stdout files
        for agent_files in AGENT_LOG_FILES.values():
            content = self._load_text(agent_files["stdout"])
            if content:
                return content

        return ""

    def _load_stderr(self) -> str:
        """Load agent-specific stderr file."""
        agent = self.agent_type
        if agent in AGENT_LOG_FILES:
            return self._load_text(AGENT_LOG_FILES[agent]["stderr"])

        # Fallback: try all known stderr files
        for agent_files in AGENT_LOG_FILES.values():
            content = self._load_text(agent_files["stderr"])
            if content:
                return content

        return ""

    def _get_source_files(self) -> Dict[str, str]:
        """Get paths to all source files."""
        files = {}
        file_map = {
            "journal": "journal.json",
            "run_summary": "run_summary.json",
            "prompt": "prompt.json",
            "diff_targets": "diff_targets.json",
            "task": "task.txt",
            "patch": "model_patch.diff",
            "trajectory": "trajectory.json",
            "prediction": "prediction.jsonl",
        }

        for key, filename in file_map.items():
            filepath = self.item_dir / filename
            if filepath.exists():
                files[key] = str(filepath)

        # Add agent-specific files
        agent = self.agent_type
        if agent in AGENT_LOG_FILES:
            stdout_file = self.item_dir / AGENT_LOG_FILES[agent]["stdout"]
            stderr_file = self.item_dir / AGENT_LOG_FILES[agent]["stderr"]
            if stdout_file.exists():
                files["stdout"] = str(stdout_file)
            if stderr_file.exists():
                files["stderr"] = str(stderr_file)

        return files


def extract_patch_metrics(patch_content: str) -> Dict[str, int]:
    """Extract metrics from a unified diff patch.

    Returns:
        Dict with:
            - lines_added: Number of lines added
            - lines_removed: Number of lines removed
            - hunks: Number of hunks (change blocks)
            - files_changed: Number of files changed
    """
    if not patch_content:
        return {
            "lines_added": 0,
            "lines_removed": 0,
            "hunks": 0,
            "files_changed": 0,
        }

    lines_added = 0
    lines_removed = 0
    hunks = 0
    files = set()

    for line in patch_content.split("\n"):
        if line.startswith("+++") or line.startswith("---"):
            if not line.startswith("+++ /dev/null") and not line.startswith("--- /dev/null"):
                # Extract filename
                parts = line.split("\t")[0].split(" ")
                if len(parts) > 1:
                    files.add(parts[1])
        elif line.startswith("@@"):
            hunks += 1
        elif line.startswith("+") and not line.startswith("+++"):
            lines_added += 1
        elif line.startswith("-") and not line.startswith("---"):
            lines_removed += 1

    return {
        "lines_added": lines_added,
        "lines_removed": lines_removed,
        "hunks": hunks,
        "files_changed": len(files),
    }


def extract_error_metrics(stderr_content: str) -> Dict[str, Any]:
    """Extract error metrics from stderr content.

    Returns:
        Dict with:
            - error_count: Number of errors found
            - error_count: Number of errors found
            - warning_count: Number of warnings found
            - exception_types: List of exception types found
    """
    if not stderr_content:
        return {
            "error_count": 0,
            "warning_count": 0,
            "exception_types": [],
        }

    error_count = 0
    warning_count = 0
    exception_types = set()
    lines = stderr_content.lower().split("\n")
    for line in lines:
        if "error" in line or "exception" in line:
            error_count += 1
        if "warning" in line:
            warning_count += 1

    # Extract exception types
    exception_pattern = r"(\w+(?:Error|Exception))"
    for match in re.finditer(exception_pattern, stderr_content):
        exception_types.add(match.group(1))

    return {
        "error_count": error_count,
        "warning_count": warning_count,
        "exception_types": list(exception_types),
    }


def count_tool_calls(stdout_content: str, agent_type: str) -> Dict[str, int]:
    """Count tool calls from stdout content.

    Different agents have different output formats.

    Returns:
        Dict mapping tool names to counts
    """
    tool_counts: Dict[str, int] = {}

    if not stdout_content:
        return tool_counts

    if agent_type == "trae":
        # TRAE uses specific tool markers
        patterns = [
            (r"Tool:\s*(\w+)", "general"),
            (r"bash\s*```", "bash"),
            (r"str_replace_based_edit_tool", "editor"),
            (r"Tool:\s*(\w+)", "general"),
            (r"bash\s*```", "bash"),
            (r"str_replace_based_edit_tool", "editor"),
            (r"view_file|cat ", "read"),
            (r"search_engine|web_search|google", "web_search"),
        ]
        for pattern, tool in patterns:
            matches = re.findall(pattern, stdout_content, re.IGNORECASE)
            if isinstance(matches, list) and matches:
                if tool == "general":
                    for match in matches:
                        tool_counts[match.lower()] = tool_counts.get(match.lower(), 0) + 1
                else:
                    tool_counts[tool] = tool_counts.get(tool, 0) + len(matches)

    elif agent_type == "openhands":
        # OpenHands uses action-based format
        patterns = [
            (r'"action":\s*"(\w+)"', None),
            (r"CmdRunAction", "bash"),
            (r"FileWriteAction|FileEditAction", "editor"),
            (r"CmdRunAction", "bash"),
            (r"FileWriteAction|FileEditAction", "editor"),
            (r"FileReadAction", "read"),
            (r"IPythonRunCellAction", "bash"),  # Often used like shell
            (r"BrowseInteractiveAction|BrowseURLAction", "web_search"),
        ]
        for pattern, tool in patterns:
            matches = re.findall(pattern, stdout_content)
            if matches:
                if tool is None:
                    for match in matches:
                        tool_counts[match.lower()] = tool_counts.get(match.lower(), 0) + 1
                else:
                    tool_counts[tool] = tool_counts.get(tool, 0) + len(matches)

    elif agent_type == "claude_code":
        # Claude Code uses stream JSON format
        patterns = [
            (r'"tool_use"', "tool_use"),
            (r'"name":\s*"(Bash|Read|Write|Edit|Grep|Glob)"', None),
            (r'"name":\s*"(WebSearch|BraveSearch)"', "web_search"),
        ]
        for pattern, tool in patterns:
            matches = re.findall(pattern, stdout_content, re.IGNORECASE)
            if matches:
                if tool is None:
                    for match in matches:
                        tool_counts[match.lower()] = tool_counts.get(match.lower(), 0) + 1
                else:
                    tool_counts[tool] = tool_counts.get(tool, 0) + len(matches)

    return tool_counts


# =============================================================================
# V3 Academic-Grade Trajectory Parser
# =============================================================================

class TrajectoryParser:
    """Parse trajectory.json to extract detailed per-interaction metrics.

    Extracts token counts, timing, and tool usage from LLM interactions.
    Supports multiple agent types: TRAE, OpenHands, Codex, Claude Code.
    """

    def __init__(self, trajectory_data: Dict[str, Any]):
        """Initialize with loaded trajectory.json data.

        Args:
            trajectory_data: Dict from trajectory.json
        """
        self.data = trajectory_data
        self._interactions: Optional[List[Dict[str, Any]]] = None

    @property
    def start_time(self) -> str:
        """Get run start time."""
        return self.data.get("start_time", "")

    @property
    def end_time(self) -> str:
        """Get run end time."""
        return self.data.get("end_time", "")

    @property
    def provider(self) -> str:
        """Get LLM provider."""
        return self.data.get("provider", "")

    @property
    def model(self) -> str:
        """Get LLM model name."""
        return self.data.get("model", "")

    def parse_all(self) -> Dict[str, Any]:
        """Parse trajectory and return full metrics dict.

        Returns:
            Dict matching TrajectoryMetrics schema fields
        """
        interactions = self._parse_interactions()
        timing = self._calculate_timing(interactions)
        tokens = self._aggregate_tokens(interactions)
        tools = self._aggregate_tool_usage(interactions)

        return {
            "start_time": self.start_time,
            "end_time": self.end_time,
            "total_duration_s": timing["total_duration_s"],
            "step_count": len(interactions),
            "step_durations": timing["step_durations"],
            "avg_step_duration_s": timing["avg_step_duration_s"],
            "max_step_duration_s": timing["max_step_duration_s"],
            "min_step_duration_s": timing["min_step_duration_s"],
            **tokens,
            **tools,
            "interactions": interactions,
        }

    def _parse_interactions(self) -> List[Dict[str, Any]]:
        """Parse LLM interactions from trajectory data.

        Handles different trajectory formats:
        - llm_interactions: Array of LLM call records (TRAE, etc.)
        - agent_steps: Array of step records

        Returns:
            List of interaction metrics dicts
        """
        if self._interactions is not None:
            return self._interactions

        interactions = []

        # Try llm_interactions format first (TRAE, some agents)
        llm_interactions = self.data.get("llm_interactions", [])
        if llm_interactions:
            for i, interaction in enumerate(llm_interactions):
                metrics = self._parse_llm_interaction(i, interaction)
                interactions.append(metrics)
        else:
            # Try agent_steps format (fallback)
            agent_steps = self.data.get("agent_steps", [])
            for i, step in enumerate(agent_steps):
                metrics = self._parse_agent_step(i, step)
                interactions.append(metrics)

        self._interactions = interactions
        return interactions

    def _parse_llm_interaction(self, index: int, interaction: Dict[str, Any]) -> Dict[str, Any]:
        """Parse a single LLM interaction record.

        Args:
            index: Step index
            interaction: Single interaction dict from llm_interactions array

        Returns:
            Dict matching InteractionMetrics schema
        """
        # Extract timestamp
        timestamp = interaction.get("timestamp", "")

        # Calculate elapsed time from start
        elapsed_s = 0.0
        if timestamp and self.start_time:
            elapsed_s = self._calculate_elapsed(self.start_time, timestamp)

        # Extract token usage from response.usage
        response = interaction.get("response", {})
        usage = response.get("usage", {})

        input_tokens = usage.get("input_tokens", 0)
        output_tokens = usage.get("output_tokens", 0)
        cache_read_tokens = usage.get("cache_read_input_tokens", 0)
        cache_creation_tokens = usage.get("cache_creation_input_tokens", 0)
        reasoning_tokens = usage.get("reasoning_tokens", 0)

        # Extract tool calls
        tool_calls = response.get("tool_calls", [])
        tool_names = [tc.get("name", "unknown") for tc in tool_calls if isinstance(tc, dict)]

        # Check tool success/failure from subsequent messages
        tool_success = len(tool_names)  # Default: assume success
        tool_failure = 0

        return {
            "step_index": index,
            "timestamp": timestamp,
            "elapsed_s": elapsed_s,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "cache_read_tokens": cache_read_tokens,
            "cache_creation_tokens": cache_creation_tokens,
            "reasoning_tokens": reasoning_tokens,
            "tool_calls": tool_names,
            "tool_success": tool_success,
            "tool_failure": tool_failure,
        }

    def _parse_agent_step(self, index: int, step: Dict[str, Any]) -> Dict[str, Any]:
        """Parse a single agent step record (fallback format).

        Args:
            index: Step index
            step: Single step dict from agent_steps array

        Returns:
            Dict matching InteractionMetrics schema
        """
        timestamp = step.get("timestamp", "")

        # Calculate elapsed time
        elapsed_s = 0.0
        if timestamp and self.start_time:
            elapsed_s = self._calculate_elapsed(self.start_time, timestamp)

        # Extract tool calls if available
        tool_calls_data = step.get("tool_calls", [])
        tool_names = []
        if tool_calls_data:
            for tc in tool_calls_data:
                if isinstance(tc, dict):
                    tool_names.append(tc.get("name", "unknown"))
                elif isinstance(tc, str):
                    tool_names.append(tc)

        # Check for error
        has_error = bool(step.get("error"))

        return {
            "step_index": index,
            "timestamp": timestamp,
            "elapsed_s": elapsed_s,
            "input_tokens": 0,  # Not available in this format
            "output_tokens": 0,
            "cache_read_tokens": 0,
            "cache_creation_tokens": 0,
            "reasoning_tokens": 0,
            "tool_calls": tool_names,
            "tool_success": len(tool_names) if not has_error else 0,
            "tool_failure": len(tool_names) if has_error else 0,
        }

    def _calculate_timing(self, interactions: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate timing statistics from interactions.

        Returns:
            Dict with timing metrics
        """
        # Calculate total duration from start/end times
        total_duration_s = 0.0
        if self.start_time and self.end_time:
            total_duration_s = self._calculate_elapsed(self.start_time, self.end_time)

        # Calculate per-step durations
        step_durations = []
        for i, interaction in enumerate(interactions):
            if i == 0:
                # First step: time from start to first interaction
                step_durations.append(interaction.get("elapsed_s", 0.0))
            else:
                # Subsequent steps: time since previous interaction
                prev_elapsed = interactions[i - 1].get("elapsed_s", 0.0)
                curr_elapsed = interaction.get("elapsed_s", 0.0)
                step_durations.append(max(0.0, curr_elapsed - prev_elapsed))

        # Calculate statistics
        avg_step_duration = 0.0
        max_step_duration = 0.0
        min_step_duration = 0.0

        if step_durations:
            avg_step_duration = sum(step_durations) / len(step_durations)
            max_step_duration = max(step_durations)
            min_step_duration = min(step_durations)

        return {
            "total_duration_s": total_duration_s,
            "step_durations": step_durations,
            "avg_step_duration_s": avg_step_duration,
            "max_step_duration_s": max_step_duration,
            "min_step_duration_s": min_step_duration,
        }

    def _aggregate_tokens(self, interactions: List[Dict[str, Any]]) -> Dict[str, int]:
        """Aggregate token counts across all interactions.

        Returns:
            Dict with total token counts
        """
        total_input = 0
        total_output = 0
        total_cache_read = 0
        total_cache_creation = 0
        total_reasoning = 0

        for interaction in interactions:
            total_input += interaction.get("input_tokens", 0)
            total_output += interaction.get("output_tokens", 0)
            total_cache_read += interaction.get("cache_read_tokens", 0)
            total_cache_creation += interaction.get("cache_creation_tokens", 0)
            total_reasoning += interaction.get("reasoning_tokens", 0)

        return {
            "total_input_tokens": total_input,
            "total_output_tokens": total_output,
            "total_cache_read_tokens": total_cache_read,
            "total_cache_creation_tokens": total_cache_creation,
            "total_reasoning_tokens": total_reasoning,
        }

    def _aggregate_tool_usage(self, interactions: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Aggregate tool usage across all interactions.

        Returns:
            Dict with tool usage statistics
        """
        total_calls = 0
        successful = 0
        failed = 0
        tool_counts: Dict[str, int] = {}

        for interaction in interactions:
            tools = interaction.get("tool_calls", [])
            total_calls += len(tools)
            successful += interaction.get("tool_success", 0)
            failed += interaction.get("tool_failure", 0)

            for tool in tools:
                tool_counts[tool] = tool_counts.get(tool, 0) + 1

        # Calculate success rate
        success_rate = 0.0
        if total_calls > 0:
            success_rate = successful / total_calls

        return {
            "total_tool_calls": total_calls,
            "successful_tool_calls": successful,
            "failed_tool_calls": failed,
            "tool_success_rate": success_rate,
            "tool_call_counts": tool_counts,
        }

    @staticmethod
    def _calculate_elapsed(start_time: str, end_time: str) -> float:
        """Calculate elapsed seconds between two ISO timestamps.

        Args:
            start_time: ISO format timestamp
            end_time: ISO format timestamp

        Returns:
            Elapsed seconds (float)
        """
        from datetime import datetime

        try:
            # Handle various ISO formats
            for fmt in [
                "%Y-%m-%dT%H:%M:%S.%f",
                "%Y-%m-%dT%H:%M:%S",
                "%Y-%m-%d %H:%M:%S.%f",
                "%Y-%m-%d %H:%M:%S",
            ]:
                try:
                    start = datetime.strptime(start_time[:26], fmt)
                    end = datetime.strptime(end_time[:26], fmt)
                    return (end - start).total_seconds()
                except ValueError:
                    continue

            return 0.0
        except Exception:
            return 0.0
