"""
Unified run summary schema and helpers.

This module provides a single-source-of-truth JSON schema for benchmark runs,
consolidating information from multiple source files (journal.json, trajectory.json,
prediction.jsonl, model_patch.diff) into one comprehensive summary.

Two-stage flow:
1. After agent run: Generate run_summary.json with meta + agent info in state/runs/
2. After evaluation: Copy + append evaluation results to eval_results/
"""

from __future__ import annotations

import json
import logging
import subprocess
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


# Canonical model name mapping
MODEL_CANONICAL_NAMES = {
    # OpenAI
    "gpt-5-2025-08-07": "gpt-5",
    "gpt-5": "gpt-5",
    "gpt-4o": "gpt-4o",
    "o4-mini": "o4-mini",
    # Anthropic/Bedrock
    "us.anthropic.claude-sonnet-4-5-20250929-v1:0": "claude-sonnet-45",
    "claude-sonnet-4-5-20250929": "claude-sonnet-45",
    "anthropic.claude-sonnet-4-5-20250929-v1:0": "claude-sonnet-45",
}


@dataclass
class RunMeta:
    """Run identification metadata."""
    repo: str  # vllm, sglang
    agent: str  # trae, codex, openhands
    model: str  # canonical name (gpt-5, claude-sonnet-45)
    model_full: str  # full model identifier
    timestamp: str  # run timestamp/id
    task_id: str  # task identifier
    item_id: str  # item identifier within run


@dataclass
class CommitInfo:
    """Git commit information."""
    human: str  # human-authored optimization commit
    pre: str  # commit before optimization (baseline)


@dataclass
class PatchStats:
    """Statistics about the generated patch."""
    lines_added: int = 0
    lines_removed: int = 0
    files_changed: int = 0


@dataclass
class AgentInfo:
    """Agent execution information."""
    status: str  # success, error, timeout, etc.
    patch_generated: bool = False
    patch_stats: Optional[PatchStats] = None
    duration_s: Optional[float] = None
    time_to_first_edit_s: Optional[float] = None


@dataclass
class EvaluationInfo:
    """Evaluation results."""
    status: str  # success, error, no_test, no_patch, patch_failed, timeout
    baseline_ms: Optional[float] = None
    patched_ms: Optional[float] = None
    speedup: Optional[float] = None
    improvement: bool = False
    error: Optional[str] = None


@dataclass
class SourceFiles:
    """Provenance: which files were used to generate this summary."""
    journal: Optional[str] = None
    trajectory: Optional[str] = None
    prediction: Optional[str] = None
    patch: Optional[str] = None


@dataclass
class RunSummary:
    """
    Unified run summary containing all information about a benchmark run.

    This is the single source of truth - no need to look at other files.
    """
    schema_version: str = "1.0"
    meta: Optional[RunMeta] = None
    commits: Optional[CommitInfo] = None
    agent: Optional[AgentInfo] = None
    evaluation: Optional[EvaluationInfo] = None
    _sources: Optional[SourceFiles] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        result = {"schema_version": self.schema_version}

        if self.meta:
            result["meta"] = asdict(self.meta)
        if self.commits:
            result["commits"] = asdict(self.commits)
        if self.agent:
            agent_dict = asdict(self.agent)
            # Remove None patch_stats
            if agent_dict.get("patch_stats") is None:
                del agent_dict["patch_stats"]
            result["agent"] = agent_dict
        if self.evaluation:
            result["evaluation"] = asdict(self.evaluation)
        if self._sources:
            result["_sources"] = asdict(self._sources)

        return result

    def to_json(self, indent: int = 2) -> str:
        """Serialize to JSON string."""
        return json.dumps(self.to_dict(), indent=indent)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "RunSummary":
        """Create RunSummary from dictionary."""
        summary = cls(schema_version=data.get("schema_version", "1.0"))

        if "meta" in data:
            summary.meta = RunMeta(**data["meta"])
        if "commits" in data:
            summary.commits = CommitInfo(**data["commits"])
        if "agent" in data:
            agent_data = data["agent"].copy()
            if "patch_stats" in agent_data and agent_data["patch_stats"]:
                agent_data["patch_stats"] = PatchStats(**agent_data["patch_stats"])
            summary.agent = AgentInfo(**agent_data)
        if "evaluation" in data:
            summary.evaluation = EvaluationInfo(**data["evaluation"])
        if "_sources" in data:
            summary._sources = SourceFiles(**data["_sources"])

        return summary

    @classmethod
    def from_json(cls, json_str: str) -> "RunSummary":
        """Create RunSummary from JSON string."""
        return cls.from_dict(json.loads(json_str))


def get_canonical_model_name(model_full: str) -> str:
    """Convert full model identifier to canonical short name."""
    return MODEL_CANONICAL_NAMES.get(model_full, model_full)


def compute_patch_stats(patch_path: Path) -> Optional[PatchStats]:
    """Compute statistics from a patch file."""
    if not patch_path.exists():
        return None

    try:
        content = patch_path.read_text()
        lines_added = content.count("\n+") - content.count("\n+++")
        lines_removed = content.count("\n-") - content.count("\n---")

        # Count files changed
        files_changed = content.count("diff --git")

        return PatchStats(
            lines_added=max(0, lines_added),
            lines_removed=max(0, lines_removed),
            files_changed=files_changed,
        )
    except Exception as e:
        logger.warning(f"Error computing patch stats: {e}")
        return None


def extract_model_from_trajectory(trajectory_path: Path) -> tuple[str, str]:
    """Extract model info from trajectory.json (trae/openhands agents)."""
    try:
        data = json.loads(trajectory_path.read_text())
        model_full = data.get("model", "")
        return model_full, get_canonical_model_name(model_full)
    except Exception as e:
        logger.warning(f"Error reading trajectory.json: {e}")
        return "", ""


def extract_model_from_prediction(prediction_path: Path) -> tuple[str, str]:
    """Extract model info from prediction.jsonl (codex agent)."""
    try:
        # prediction.jsonl has one JSON object per line, take the first
        first_line = prediction_path.read_text().strip().split("\n")[0]
        data = json.loads(first_line)
        model_full = data.get("model_name_or_path", "")
        return model_full, get_canonical_model_name(model_full)
    except Exception as e:
        logger.warning(f"Error reading prediction.jsonl: {e}")
        return "", ""


def generate_summary_from_state(
    item_dir: Path,
    repo: str,
    agent: str,
    model_hint: str,
    timestamp: str,
) -> Optional[RunSummary]:
    """
    Generate a RunSummary from state/runs item directory.

    This is Stage 1: called after agent run completes.

    Args:
        item_dir: Path to item directory (e.g., state/runs/vllm/trae/gpt-5/abc123/task-0000/)
        repo: Repository name (vllm, sglang)
        agent: Agent name (trae, codex, openhands)
        model_hint: Model name from directory structure (may need verification)
        timestamp: Run timestamp

    Returns:
        RunSummary with meta, commits, and agent info populated
    """
    journal_path = item_dir / "journal.json"
    if not journal_path.exists():
        logger.warning(f"No journal.json in {item_dir}")
        return None

    try:
        journal = json.loads(journal_path.read_text())
    except Exception as e:
        logger.error(f"Error reading journal.json: {e}")
        return None

    # Extract commits
    commits_data = journal.get("commits", {})
    human_commit = commits_data.get("human", "")
    pre_commit = commits_data.get("pre", "")

    if not human_commit or not pre_commit:
        logger.warning(f"Missing commits in {journal_path}")
        return None

    # Determine actual model from source files
    model_full = ""
    model_canonical = model_hint
    sources = SourceFiles(journal=str(journal_path))

    trajectory_path = item_dir / "trajectory.json"
    prediction_path = item_dir / "prediction.jsonl"

    if trajectory_path.exists():
        model_full, model_canonical = extract_model_from_trajectory(trajectory_path)
        sources.trajectory = str(trajectory_path)
    elif prediction_path.exists():
        model_full, model_canonical = extract_model_from_prediction(prediction_path)
        sources.prediction = str(prediction_path)

    # If we couldn't extract model, use the hint
    if not model_full:
        model_full = model_hint
    if not model_canonical:
        model_canonical = model_hint

    # Check for patch
    patch_path = item_dir / "model_patch.diff"
    patch_exists = patch_path.exists() and patch_path.stat().st_size > 0
    patch_stats = None

    if patch_exists:
        sources.patch = str(patch_path)
        patch_stats = compute_patch_stats(patch_path)

    # Extract agent metrics from journal
    metrics = journal.get("metrics", {})
    trae_info = journal.get("trae", {})

    duration_s = trae_info.get("duration_s") or metrics.get("duration_s")
    time_to_first_edit = metrics.get("time_to_first_edit_s")

    # Build summary
    summary = RunSummary(
        meta=RunMeta(
            repo=repo,
            agent=agent,
            model=model_canonical,
            model_full=model_full,
            timestamp=timestamp,
            task_id=journal.get("task_id", "unknown"),
            item_id=item_dir.name,
        ),
        commits=CommitInfo(
            human=human_commit,
            pre=pre_commit,
        ),
        agent=AgentInfo(
            status=journal.get("status", "unknown"),
            patch_generated=patch_exists,
            patch_stats=patch_stats,
            duration_s=duration_s,
            time_to_first_edit_s=time_to_first_edit,
        ),
        _sources=sources,
    )

    return summary


def load_summary(summary_path: Path) -> Optional[RunSummary]:
    """Load a RunSummary from a JSON file."""
    if not summary_path.exists():
        return None
    try:
        return RunSummary.from_json(summary_path.read_text())
    except Exception as e:
        logger.error(f"Error loading summary from {summary_path}: {e}")
        return None


def save_summary(summary: RunSummary, output_path: Path) -> None:
    """Save a RunSummary to a JSON file."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(summary.to_json())
    logger.info(f"Saved run summary to {output_path}")


def add_evaluation_to_summary(
    summary: RunSummary,
    status: str,
    baseline_ms: Optional[float] = None,
    patched_ms: Optional[float] = None,
    speedup: Optional[float] = None,
    improvement: bool = False,
    error: Optional[str] = None,
) -> RunSummary:
    """
    Add evaluation results to an existing summary.

    This is Stage 2: called after evaluation completes.
    """
    summary.evaluation = EvaluationInfo(
        status=status,
        baseline_ms=baseline_ms,
        patched_ms=patched_ms,
        speedup=speedup,
        improvement=improvement,
        error=error,
    )
    return summary


# Migration helpers

def migrate_item_dir(
    item_dir: Path,
    repo: str,
    agent: str,
    model: str,
    timestamp: str,
) -> Optional[Path]:
    """
    Generate run_summary.json for an existing item directory.

    Returns path to generated summary, or None if failed.
    """
    summary = generate_summary_from_state(item_dir, repo, agent, model, timestamp)
    if not summary:
        return None

    output_path = item_dir / "run_summary.json"
    save_summary(summary, output_path)
    return output_path


def migrate_state_runs(state_runs_dir: Path) -> Dict[str, int]:
    """
    Generate run_summary.json for all runs in state/runs directory.

    Expects hierarchical structure: {repo}/{agent}/{model}/{timestamp}/{item_id}/

    Returns dict with counts: {"success": N, "failed": M, "skipped": K}
    """
    stats = {"success": 0, "failed": 0, "skipped": 0}

    if not state_runs_dir.exists():
        logger.error(f"State runs directory not found: {state_runs_dir}")
        return stats

    for repo_dir in state_runs_dir.iterdir():
        if not repo_dir.is_dir():
            continue
        repo = repo_dir.name

        for agent_dir in repo_dir.iterdir():
            if not agent_dir.is_dir():
                continue
            agent = agent_dir.name

            for model_dir in agent_dir.iterdir():
                if not model_dir.is_dir():
                    continue
                model = model_dir.name

                for timestamp_dir in model_dir.iterdir():
                    if not timestamp_dir.is_dir():
                        continue
                    timestamp = timestamp_dir.name

                    for item_dir in timestamp_dir.iterdir():
                        if not item_dir.is_dir():
                            continue

                        # Skip if summary already exists
                        summary_path = item_dir / "run_summary.json"
                        if summary_path.exists():
                            stats["skipped"] += 1
                            continue

                        # Check if journal.json exists
                        if not (item_dir / "journal.json").exists():
                            stats["skipped"] += 1
                            continue

                        result = migrate_item_dir(item_dir, repo, agent, model, timestamp)
                        if result:
                            stats["success"] += 1
                        else:
                            stats["failed"] += 1

    return stats


if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.INFO)

    if len(sys.argv) < 2:
        print("Usage: python run_summary.py <state_runs_dir>")
        print("  Generates run_summary.json for all runs in the directory")
        sys.exit(1)

    state_runs_dir = Path(sys.argv[1])
    stats = migrate_state_runs(state_runs_dir)
    print(f"Migration complete: {stats}")
