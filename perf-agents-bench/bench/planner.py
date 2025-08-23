from __future__ import annotations
import yaml
from pathlib import Path
from typing import Dict, Any, List, Optional


class MatrixPlanner:
    def build_plan(self, task_cfg: Dict[str, Any], commits_path: Optional[Path] = None) -> Dict[str, Any]:
        """Create a simple plan: list of {item_id, human_sha, pre_sha}.
        If commits_path is provided, parse it; otherwise expect task_cfg.repo.pairs.
        """
        repo = task_cfg["repo"]["url"]
        items: List[Dict[str, str]] = []

        if commits_path:
            text = commits_path.read_text().strip().splitlines()
            for i, line in enumerate(text):
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                parts = line.split()
                human = parts[0]
                pre = None
                parent_idx = None
                if len(parts) > 1:
                    token = parts[1]
                    if token.startswith("parent="):
                        parent_idx = int(token.split("=", 1)[1])
                    else:
                        pre = token
                item_id = f"{task_cfg['id']}-{i:04d}"
                items.append({"item_id": item_id, "human": human, "pre": pre or "", "pre_parent_index": parent_idx or 1})
        else:
            pairs = task_cfg["repo"].get("pairs", [])
            for i, p in enumerate(pairs):
                item_id = f"{task_cfg['id']}-{i:04d}"
                items.append({
                    "item_id": item_id,
                    "human": p["human"],
                    "pre": p.get("pre", ""),
                    "pre_parent_index": int(p.get("pre_parent_index", 1)),
                })

        return {
            "repo": repo,
            "task_id": task_cfg["id"],
            "items": items,
        }

