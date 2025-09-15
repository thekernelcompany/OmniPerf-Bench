from __future__ import annotations
import json
from pathlib import Path
from typing import Dict, Any, List


def summarize_stage_a(run_dir: Path) -> Dict[str, Any]:
    run_dir = Path(run_dir)
    items: List[Dict[str, Any]] = []
    for item_dir in sorted(p for p in run_dir.iterdir() if p.is_dir()):
        journal = item_dir / "journal.json"
        if not journal.exists():
            continue
        try:
            data = json.loads(journal.read_text())
        except Exception:
            continue
        items.append({
            "item_id": item_dir.name,
            "status": data.get("status"),
            "human": data.get("commits", {}).get("human"),
            "pre": data.get("commits", {}).get("pre"),
            "agent_branch": data.get("agent_branch"),
        })

    summary = {
        "run_id": run_dir.name,
        "num_items": len(items),
        "num_success": sum(1 for x in items if x["status"] == "success"),
        "num_error": sum(1 for x in items if x["status"] == "error"),
        "items": items,
    }
    return summary

