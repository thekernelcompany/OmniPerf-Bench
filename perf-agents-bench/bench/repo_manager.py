from __future__ import annotations
import subprocess
from pathlib import Path
from typing import Optional
import logging

logger = logging.getLogger(__name__)


class RepoManager:
    def __init__(self, work_root: Path, repo_url: str, repo_name: str):
        self.work_root = work_root
        self.repo_url = repo_url
        self.repo_name = repo_name
        local_candidate = Path(repo_url)
        if local_candidate.exists():
            self.base_dir = local_candidate.resolve()
            self._using_local_repo = True
        else:
            self.base_dir = (self.work_root / "repos" / repo_name).resolve()
            self._using_local_repo = False

    def ensure_base(self):
        if self._using_local_repo:
            if not self.base_dir.exists():
                raise RuntimeError(f"Local repository path does not exist: {self.base_dir}")
            return

        self.base_dir.parent.mkdir(parents=True, exist_ok=True)
        if not self.base_dir.exists():
            subprocess.run(["git", "clone", self.repo_url, str(self.base_dir)], check=True)
        else:
            try:
                subprocess.run(["git", "fetch", "--all"], cwd=self.base_dir, check=True)
            except subprocess.CalledProcessError as e:
                logger.warning(f"Git fetch failed ({e}); proceeding with existing repo checkout")

    def create_worktree(self, ref: str, item_id: str) -> Path:
        wt_dir = self.work_root / "worktrees" / self.repo_name / item_id
        wt_dir.parent.mkdir(parents=True, exist_ok=True)
        if not wt_dir.exists():
            subprocess.run(["git", "worktree", "add", str(wt_dir), ref], cwd=self.base_dir, check=True)
        return wt_dir

    def remove_worktree(self, item_id: str):
        wt_dir = self.work_root / "worktrees" / self.repo_name / item_id
        if wt_dir.exists():
            subprocess.run(["git", "worktree", "remove", "--force", str(wt_dir)], cwd=self.base_dir, check=True)
