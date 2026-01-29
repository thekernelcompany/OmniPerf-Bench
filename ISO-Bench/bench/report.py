from __future__ import annotations
import json
from pathlib import Path
from typing import Dict, Any, List, Optional


def _find_item_dirs(run_dir: Path) -> List[Path]:
    """Find all item directories containing journal.json files.

    Handles both:
    - Old flat structure: run_dir/{item_id}/journal.json
    - New hierarchical structure: run_dir/{timestamp}/{item_id}/journal.json
    """
    item_dirs: List[Path] = []

    for path in run_dir.iterdir():
        if not path.is_dir():
            continue

        # Check if this is an item directory (has journal.json directly)
        journal = path / "journal.json"
        if journal.exists():
            item_dirs.append(path)
        else:
            # Check for nested item directories (hierarchical timestamp folder)
            for nested_path in path.iterdir():
                if nested_path.is_dir():
                    nested_journal = nested_path / "journal.json"
                    if nested_journal.exists():
                        item_dirs.append(nested_path)

    return sorted(item_dirs)


def _get_run_metadata(run_dir: Path) -> Dict[str, str]:
    """Extract run metadata from hierarchical path."""
    # Try to extract from path: state/runs/{repo}/{agent}/{model}/{timestamp}
    parts = run_dir.parts
    try:
        # Find 'runs' in path and get components after it
        runs_idx = parts.index("runs")
        remaining = parts[runs_idx + 1:]
        if len(remaining) >= 4:
            return {
                "repo": remaining[0],
                "agent": remaining[1],
                "model": remaining[2],
                "timestamp": remaining[3],
            }
    except (ValueError, IndexError):
        pass

    return {"repo": "unknown", "agent": "unknown", "model": "unknown", "timestamp": run_dir.name}


def summarize_stage_a(run_dir: Path) -> Dict[str, Any]:
    """Summarize Stage A journals into a compact report.

    Supports both flat and hierarchical run directory structures.
    """
    run_dir = Path(run_dir)
    items: List[Dict[str, Any]] = []

    for item_dir in _find_item_dirs(run_dir):
        journal = item_dir / "journal.json"
        if not journal.exists():
            continue
        try:
            data = json.loads(journal.read_text())
        except Exception:
            continue

        # Extract run_metadata if present
        run_metadata = data.get("run_metadata", {})

        items.append({
            "item_id": item_dir.name,
            "status": data.get("status"),
            "human": data.get("commits", {}).get("human"),
            "pre": data.get("commits", {}).get("pre"),
            "agent_branch": data.get("agent_branch"),
            "repo": run_metadata.get("repo"),
            "agent": run_metadata.get("agent"),
            "model": run_metadata.get("model"),
        })

    # Extract metadata from path for summary
    path_metadata = _get_run_metadata(run_dir)

    summary = {
        "run_id": str(run_dir.relative_to(run_dir.parent.parent.parent.parent)) if len(run_dir.parts) > 4 else run_dir.name,
        "run_path": str(run_dir),
        "metadata": path_metadata,
        "num_items": len(items),
        "num_success": sum(1 for x in items if x["status"] == "success"),
        "num_error": sum(1 for x in items if x["status"] == "error"),
        "items": items,
    }
    return summary


def summarize_all_runs(state_root: Path) -> Dict[str, Any]:
    """Summarize all runs in the state directory, organized by repo/agent/model.

    Handles hierarchical structure: state/runs/{repo}/{agent}/{model}/{timestamp}/
    """
    runs_dir = state_root / "runs"
    if not runs_dir.exists():
        return {"error": "runs directory not found"}

    all_runs: Dict[str, Dict[str, Dict[str, List[Dict[str, Any]]]]] = {}
    total_success = 0
    total_error = 0
    total_items = 0

    # Walk the hierarchical structure
    for repo_dir in sorted(runs_dir.iterdir()):
        if not repo_dir.is_dir():
            continue

        repo_name = repo_dir.name

        # Check if this is old flat structure (has journal.json in subdirs)
        is_flat = any((repo_dir / d / "journal.json").exists() for d in repo_dir.iterdir() if (repo_dir / d).is_dir())

        if is_flat:
            # Handle old flat structure
            summary = summarize_stage_a(repo_dir)
            if repo_name not in all_runs:
                all_runs[repo_name] = {"legacy": {"flat": []}}
            all_runs[repo_name]["legacy"]["flat"].append(summary)
            total_success += summary["num_success"]
            total_error += summary["num_error"]
            total_items += summary["num_items"]
            continue

        # Handle hierarchical structure
        for agent_dir in sorted(repo_dir.iterdir()):
            if not agent_dir.is_dir():
                continue

            agent_name = agent_dir.name

            for model_dir in sorted(agent_dir.iterdir()):
                if not model_dir.is_dir():
                    continue

                model_name = model_dir.name

                for timestamp_dir in sorted(model_dir.iterdir()):
                    if not timestamp_dir.is_dir():
                        continue

                    summary = summarize_stage_a(timestamp_dir)

                    if repo_name not in all_runs:
                        all_runs[repo_name] = {}
                    if agent_name not in all_runs[repo_name]:
                        all_runs[repo_name][agent_name] = {}
                    if model_name not in all_runs[repo_name][agent_name]:
                        all_runs[repo_name][agent_name][model_name] = []

                    all_runs[repo_name][agent_name][model_name].append(summary)
                    total_success += summary["num_success"]
                    total_error += summary["num_error"]
                    total_items += summary["num_items"]

    return {
        "total_items": total_items,
        "total_success": total_success,
        "total_error": total_error,
        "runs_by_repo": all_runs,
    }

