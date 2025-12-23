from __future__ import annotations
import json
import time
from pathlib import Path
from typing import Dict, Any


class JournalWriter:
    def __init__(self, run_dir: Path, item_id: str):
        self.dir = run_dir / item_id
        self.dir.mkdir(parents=True, exist_ok=True)

    def write_prompt(self, prompt: Dict[str, Any]):
        (self.dir / "prompt.json").write_text(json.dumps(prompt, indent=2))

    def write_diff_targets(self, info: Dict[str, Any]):
        (self.dir / "diff_targets.json").write_text(json.dumps(info, indent=2))

    def write_openhands_logs(self, stdout: str, stderr: str):
        (self.dir / "openhands_stdout.txt").write_text(stdout)
        (self.dir / "openhands_stderr.txt").write_text(stderr)

    def write_trae_logs(self, stdout: str, stderr: str):
        (self.dir / "trae_stdout.txt").write_text(stdout)
        (self.dir / "trae_stderr.txt").write_text(stderr)

    def write_codex_logs(self, stdout: str, stderr: str):
        (self.dir / "codex_stdout.txt").write_text(stdout)
        (self.dir / "codex_stderr.txt").write_text(stderr)

    def write_claude_code_logs(self, stdout: str, stderr: str):
        (self.dir / "claude_code_stdout.txt").write_text(stdout)
        (self.dir / "claude_code_stderr.txt").write_text(stderr)

    def write_journal(self, payload: Dict[str, Any]):
        payload.setdefault("timestamps", {})
        payload["timestamps"].setdefault("written", time.time())
        (self.dir / "journal.json").write_text(json.dumps(payload, indent=2))

    def has_success(self) -> bool:
        p = self.dir / "journal.json"
        if not p.exists():
            return False
        try:
            data = json.loads(p.read_text())
            return data.get("status") == "success"
        except Exception:
            return False
