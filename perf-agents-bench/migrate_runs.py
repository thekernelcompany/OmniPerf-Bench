#!/usr/bin/env python3
"""
Migration script to reorganize state/runs/ from flat structure to hierarchical tree.

Current structure:
    state/runs/{task_id}-{hash}/{item_id}/

New structure:
    state/runs/{repo}/{agent}/{model}/{timestamp}/{item_id}/

Usage:
    python migrate_runs.py [--dry-run] [--state-root ./state]
"""
from __future__ import annotations
import json
import os
import re
import shutil
import argparse
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, Tuple


def extract_repo_from_task_id(task_id: str) -> str:
    """Extract repository name from task_id."""
    # Patterns like: vllm_core, sglang_core, vllm_claude_sonnet45, etc.
    if task_id.startswith("vllm"):
        return "vllm"
    elif task_id.startswith("sglang"):
        return "sglang"
    elif task_id.startswith("chunked_local_attn"):
        return "vllm"
    elif task_id.startswith("moe_align"):
        return "vllm"
    elif task_id.startswith("prefix_caching"):
        return "vllm"
    else:
        # Default: try to extract first part before underscore
        parts = task_id.split("_")
        return parts[0] if parts else "unknown"


def extract_agent_from_journal(journal: Dict[str, Any], run_dir_name: str) -> str:
    """Extract agent name from journal data or run directory name."""
    # Check for explicit agent keys in journal
    if "trae" in journal:
        return "trae"
    elif "codex" in journal:
        return "codex"
    elif "openhands" in journal:
        return "openhands"

    # Infer from run directory name
    if "codex" in run_dir_name.lower():
        return "codex"
    elif "trae" in run_dir_name.lower():
        return "trae"
    elif "openhands" in run_dir_name.lower():
        return "openhands"

    # Default based on CLI present in journal
    for key in journal.keys():
        if key in ["trae", "codex", "openhands"]:
            return key

    return "unknown"


def extract_model_from_journal(journal: Dict[str, Any], run_dir_name: str) -> str:
    """Extract model name from journal data or run directory name."""
    # Check run directory name for model hints
    name_lower = run_dir_name.lower()

    if "sonnet45" in name_lower or "sonnet_45" in name_lower or "claude_sonnet45" in name_lower:
        return "claude-sonnet-45"
    elif "sonnet" in name_lower:
        return "claude-sonnet"
    elif "opus" in name_lower:
        return "claude-opus"
    elif "gpt4o" in name_lower or "gpt-4o" in name_lower:
        return "gpt-4o"
    elif "gpt4" in name_lower or "gpt-4" in name_lower:
        return "gpt-4"
    elif "bedrock" in name_lower:
        return "bedrock"

    # Try to get from journal prediction artifact
    try:
        model_name = journal.get("model_name_or_path")
        if model_name:
            return sanitize_model_name(model_name)
    except Exception:
        pass

    return "default"


def sanitize_model_name(model: str) -> str:
    """Sanitize model name for use in directory path."""
    # Remove special characters, convert to lowercase
    sanitized = re.sub(r'[^\w\-]', '-', model.lower())
    # Remove consecutive dashes
    sanitized = re.sub(r'-+', '-', sanitized)
    # Remove leading/trailing dashes
    return sanitized.strip('-')


def get_timestamp_from_journal(journal: Dict[str, Any]) -> str:
    """Extract timestamp from journal data."""
    try:
        written = journal.get("timestamps", {}).get("written")
        if written:
            dt = datetime.fromtimestamp(written)
            return dt.strftime("%Y-%m-%d_%H-%M-%S")
    except Exception:
        pass
    return datetime.now().strftime("%Y-%m-%d_%H-%M-%S")


def analyze_run_directory(run_dir: Path) -> Optional[Dict[str, Any]]:
    """Analyze a run directory and extract metadata for migration."""
    # Find first journal.json in subdirectories
    journal_data = None
    for item_dir in run_dir.iterdir():
        if item_dir.is_dir():
            journal_path = item_dir / "journal.json"
            if journal_path.exists():
                try:
                    journal_data = json.loads(journal_path.read_text())
                    break
                except Exception:
                    continue

    if not journal_data:
        return None

    task_id = journal_data.get("task_id", "unknown")
    run_dir_name = run_dir.name

    repo = extract_repo_from_task_id(task_id)
    agent = extract_agent_from_journal(journal_data, run_dir_name)
    model = extract_model_from_journal(journal_data, run_dir_name)
    timestamp = get_timestamp_from_journal(journal_data)

    return {
        "task_id": task_id,
        "repo": repo,
        "agent": agent,
        "model": model,
        "timestamp": timestamp,
        "original_path": run_dir,
        "journal_sample": journal_data,
    }


def build_new_path(state_root: Path, metadata: Dict[str, Any]) -> Path:
    """Build the new hierarchical path for a run."""
    return state_root / "runs" / metadata["repo"] / metadata["agent"] / metadata["model"] / metadata["timestamp"]


def migrate_runs(state_root: Path, dry_run: bool = True) -> Dict[str, Any]:
    """Migrate all runs from flat to hierarchical structure."""
    runs_dir = state_root / "runs"

    if not runs_dir.exists():
        print(f"Runs directory not found: {runs_dir}")
        return {"error": "runs directory not found"}

    results = {
        "analyzed": 0,
        "migrated": 0,
        "skipped": 0,
        "errors": [],
        "migrations": [],
    }

    # Analyze all existing run directories
    for run_dir in sorted(runs_dir.iterdir()):
        if not run_dir.is_dir():
            continue

        # Skip if already in new hierarchical structure (has depth > 1 of nested dirs)
        # New structure has: repo/agent/model/timestamp
        # Check if path contains one of the known repos as a subdir
        if run_dir.name in ["vllm", "sglang", "unknown"]:
            print(f"Skipping already-migrated structure: {run_dir.name}")
            results["skipped"] += 1
            continue

        results["analyzed"] += 1

        metadata = analyze_run_directory(run_dir)
        if not metadata:
            results["errors"].append({
                "path": str(run_dir),
                "error": "Could not extract metadata (no journal.json found)",
            })
            continue

        new_path = build_new_path(state_root, metadata)

        migration_info = {
            "from": str(run_dir),
            "to": str(new_path),
            "repo": metadata["repo"],
            "agent": metadata["agent"],
            "model": metadata["model"],
            "timestamp": metadata["timestamp"],
        }
        results["migrations"].append(migration_info)

        print(f"\n{'[DRY-RUN] ' if dry_run else ''}Migration:")
        print(f"  From: {run_dir}")
        print(f"  To:   {new_path}")
        print(f"  Repo: {metadata['repo']}, Agent: {metadata['agent']}, Model: {metadata['model']}")

        if not dry_run:
            try:
                # Create parent directories
                new_path.parent.mkdir(parents=True, exist_ok=True)

                # Handle timestamp collision by appending counter
                final_path = new_path
                counter = 1
                while final_path.exists():
                    final_path = new_path.parent / f"{new_path.name}_{counter:02d}"
                    counter += 1

                # Move the run directory contents
                shutil.move(str(run_dir), str(final_path))

                results["migrated"] += 1
                print(f"  Status: MIGRATED")

            except Exception as e:
                results["errors"].append({
                    "path": str(run_dir),
                    "error": str(e),
                })
                print(f"  Status: ERROR - {e}")
        else:
            print(f"  Status: DRY-RUN (no changes made)")

    return results


def print_summary(results: Dict[str, Any]):
    """Print migration summary."""
    print("\n" + "=" * 60)
    print("MIGRATION SUMMARY")
    print("=" * 60)
    print(f"Directories analyzed: {results['analyzed']}")
    print(f"Directories migrated: {results['migrated']}")
    print(f"Directories skipped:  {results['skipped']}")
    print(f"Errors:               {len(results['errors'])}")

    if results['errors']:
        print("\nErrors:")
        for err in results['errors']:
            print(f"  - {err['path']}: {err['error']}")

    # Group migrations by repo/agent/model
    if results['migrations']:
        print("\nMigration breakdown:")
        breakdown: Dict[str, int] = {}
        for m in results['migrations']:
            key = f"{m['repo']}/{m['agent']}/{m['model']}"
            breakdown[key] = breakdown.get(key, 0) + 1

        for key, count in sorted(breakdown.items()):
            print(f"  {key}: {count} runs")


def main():
    parser = argparse.ArgumentParser(description="Migrate runs to hierarchical structure")
    parser.add_argument("--dry-run", action="store_true", default=True,
                        help="Preview changes without actually moving files (default: True)")
    parser.add_argument("--execute", action="store_true",
                        help="Actually execute the migration (disables dry-run)")
    parser.add_argument("--state-root", type=str, default="./state",
                        help="Path to state directory (default: ./state)")

    args = parser.parse_args()

    dry_run = not args.execute
    state_root = Path(args.state_root).resolve()

    print(f"State root: {state_root}")
    print(f"Mode: {'DRY-RUN (preview only)' if dry_run else 'EXECUTE (will move files)'}")
    print()

    if not dry_run:
        response = input("This will reorganize all run directories. Continue? [y/N]: ")
        if response.lower() != 'y':
            print("Aborted.")
            return

    results = migrate_runs(state_root, dry_run=dry_run)
    print_summary(results)

    if dry_run:
        print("\nTo execute the migration, run with --execute flag:")
        print(f"  python migrate_runs.py --state-root {state_root} --execute")


if __name__ == "__main__":
    main()
