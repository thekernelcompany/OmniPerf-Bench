from __future__ import annotations
import json
import time
import subprocess
import concurrent.futures as futures
from pathlib import Path
from typing import Dict, Any

from .journal import JournalWriter
from .repo_manager import RepoManager
from .git_utils import get_changed_files, resolve_precommit
from .agents.openhands import OpenHandsAgent


class PrepareExecutor:
    def __init__(self, bench_cfg: Dict[str, Any], run_id: str):
        self.cfg = bench_cfg
        self.run_id = run_id
        self.state_root = Path(self.cfg["paths"]["state_root"]).resolve()
        self.work_root = Path(self.cfg["paths"]["work_root"]).resolve()
        (self.state_root / "runs" / self.run_id).mkdir(parents=True, exist_ok=True)

    def execute(self, task_cfg: Dict[str, Any], plan_path: Path, max_workers: int = 4, resume: bool = True):
        plan = json.loads(Path(plan_path).read_text()) if plan_path.suffix == ".json" else None
        if plan is None:
            # support YAML plans too
            import yaml
            plan = yaml.safe_load(Path(plan_path).read_text())

        items = plan["items"]
        repo_url = plan["repo"]
        repo_name = task_cfg["id"]

        rm = RepoManager(self.work_root, repo_url, repo_name)
        rm.ensure_base()

        def process(item: Dict[str, Any]):
            item_id = item["item_id"]
            run_dir = self.state_root / "runs" / self.run_id
            jw = JournalWriter(run_dir, item_id)
            if resume and jw.has_success():
                return f"skip:{item_id}"

            human = item["human"]
            pre = item.get("pre") or None
            if not pre:
                # resolve pre from parent index
                pre = resolve_precommit(rm.base_dir, human, None, item.get("pre_parent_index", 1))

            wt_dir = rm.create_worktree(pre, item_id)

            # Build OpenHands prompt
            prompt = {
                "task": task_cfg["name"],
                "description": task_cfg.get("description", ""),
                "constraints": task_cfg["optimization_contract"].get("constraints", []),
                "target_files": task_cfg["optimization_contract"]["target_files"],
                "success": {
                    "primary_metric": task_cfg["scoring"]["primary"],
                    "rules": [
                        "Do not modify tests or metrics harness",
                        "Preserve external behavior; optimize internals only",
                    ],
                },
                "commits": {"pre": pre, "human": human},
            }
            jw.write_prompt(prompt)

            # Run OpenHands locally
            agent_cfg = self.cfg["agents"]["openhands"]
            agent = OpenHandsAgent(cli=agent_cfg["cli"], time_budget_minutes=agent_cfg["time_budget_minutes"], container_image=None)
            branch = f"agent/{task_cfg['id']}/{human[:8]}"

            # Execute and capture logs
            cmd = [
                agent.cli, "run",
                "--repo", str(wt_dir),
                "--prompt-file", str((jw.dir / "prompt.json")),
                "--time", str(agent.time_budget_minutes),
                "--branch", branch,
            ]
            t0 = time.time()
            try:
                proc = subprocess.run(cmd, capture_output=True, text=True)
                dur = time.time() - t0
                jw.write_openhands_logs(proc.stdout or "", proc.stderr or "")
                status = "success" if proc.returncode == 0 else "error"
                # Enforce targets
                changed = get_changed_files(wt_dir, pre, "HEAD")
                targets = set(task_cfg["optimization_contract"]["target_files"])
                disallowed = [p for p in changed if p not in targets]
                ok = len(disallowed) == 0
                jw.write_diff_targets({"changed": changed, "allowed": list(targets), "disallowed": disallowed, "ok": ok})
                if task_cfg["optimization_contract"].get("strict_targets", False) and not ok:
                    status = "error"
                jw.write_journal({
                    "task_id": task_cfg["id"],
                    "commits": {"pre": pre, "human": human},
                    "agent_branch": branch,
                    "status": status,
                    "openhands": {
                        "cli": agent.cli,
                        "time_budget_minutes": agent.time_budget_minutes,
                        "returncode": proc.returncode,
                        "duration_s": dur,
                    },
                })
                return f"{status}:{item_id}"
            except Exception as e:
                jw.write_journal({
                    "task_id": task_cfg["id"],
                    "commits": {"pre": pre, "human": human},
                    "agent_branch": branch,
                    "status": "error",
                    "error": str(e),
                })
                return f"error:{item_id}"

        with futures.ThreadPoolExecutor(max_workers=max_workers) as ex:
            list(ex.map(process, items))

