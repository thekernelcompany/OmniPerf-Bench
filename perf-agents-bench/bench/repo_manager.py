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

    def create_worktree(self, ref: str, item_id: str, detach_from_history: bool = False) -> Path:
        wt_dir = self.work_root / "worktrees" / self.repo_name / item_id
        wt_dir.parent.mkdir(parents=True, exist_ok=True)
        if not wt_dir.exists():
            subprocess.run(["git", "worktree", "add", str(wt_dir), ref], cwd=self.base_dir, check=True)

        if detach_from_history:
            # Remove link to main repo's git history and create fresh repo
            # This prevents agent from accessing commits via git log/show
            git_link = wt_dir / ".git"
            if git_link.exists():
                git_link.unlink()  # Remove the .git file (worktree link)
                # Initialize a fresh git repo with only current files
                subprocess.run(["git", "init"], cwd=wt_dir, check=True, capture_output=True)
                subprocess.run(["git", "config", "user.email", "bench@local"], cwd=wt_dir, check=True, capture_output=True)
                subprocess.run(["git", "config", "user.name", "Benchmark"], cwd=wt_dir, check=True, capture_output=True)
                subprocess.run(["git", "add", "-A"], cwd=wt_dir, check=True, capture_output=True)
                subprocess.run(["git", "commit", "-m", "Initial state"], cwd=wt_dir, check=True, capture_output=True)
                logger.info(f"Detached worktree {item_id} from git history")

        return wt_dir

    def remove_worktree(self, item_id: str):
        wt_dir = self.work_root / "worktrees" / self.repo_name / item_id
        if wt_dir.exists():
            # Check if it's a linked worktree or detached (has .git directory vs .git file)
            git_path = wt_dir / ".git"
            if git_path.is_file():
                # Linked worktree - use git worktree remove
                subprocess.run(["git", "worktree", "remove", "--force", str(wt_dir)], cwd=self.base_dir, check=True)
            else:
                # Detached worktree - just remove the directory
                import shutil
                shutil.rmtree(wt_dir)
