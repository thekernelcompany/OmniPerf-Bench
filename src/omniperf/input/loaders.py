"""Input loaders for commit extractions (JSON files and HuggingFace)."""
from __future__ import annotations

import json
import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Iterator, List, Optional

from omniperf.data import CommitExtraction

logger = logging.getLogger(__name__)


class BaseLoader(ABC):
    """Abstract base for input loaders."""

    @abstractmethod
    def load(self) -> List[CommitExtraction]:
        """Load all commit extractions."""
        pass

    @abstractmethod
    def __iter__(self) -> Iterator[CommitExtraction]:
        """Iterate over commit extractions."""
        pass


class JSONDirectoryLoader(BaseLoader):
    """Load commit extractions from a directory of JSON files."""

    def __init__(self, directory: Path, pattern: str = "*.json"):
        self.directory = Path(directory)
        self.pattern = pattern
        self._skip_files = {"extraction_summary.json"}

    def load(self) -> List[CommitExtraction]:
        """Load all JSON files from directory."""
        return list(self)

    def __iter__(self) -> Iterator[CommitExtraction]:
        """Iterate over JSON files, yielding CommitExtraction objects."""
        if not self.directory.exists():
            raise FileNotFoundError(f"Directory not found: {self.directory}")

        json_files = sorted(self.directory.glob(self.pattern))

        for json_file in json_files:
            if json_file.name in self._skip_files:
                continue

            try:
                data = json.loads(json_file.read_text())
                yield CommitExtraction.from_json(data)
            except Exception as e:
                logger.warning(f"Failed to load {json_file}: {e}")
                continue


class JSONFileLoader(BaseLoader):
    """Load commit extractions from a single JSON file (or JSONL)."""

    def __init__(self, filepath: Path):
        self.filepath = Path(filepath)

    def load(self) -> List[CommitExtraction]:
        """Load from file."""
        return list(self)

    def __iter__(self) -> Iterator[CommitExtraction]:
        """Iterate over records in file."""
        if not self.filepath.exists():
            raise FileNotFoundError(f"File not found: {self.filepath}")

        content = self.filepath.read_text()

        # Try JSONL first (one JSON object per line)
        if self.filepath.suffix == ".jsonl":
            for line in content.strip().split("\n"):
                if line.strip():
                    data = json.loads(line)
                    yield CommitExtraction.from_json(data)
        else:
            # Try as single JSON (could be array or single object)
            data = json.loads(content)
            if isinstance(data, list):
                for item in data:
                    yield CommitExtraction.from_json(item)
            else:
                yield CommitExtraction.from_json(data)


class HuggingFaceLoader(BaseLoader):
    """Load commit extractions from HuggingFace dataset."""

    def __init__(self, repo_id: str, split: str = "train", config: Optional[str] = None):
        self.repo_id = repo_id
        self.split = split
        self.config = config
        self._dataset = None

    def _load_dataset(self):
        """Lazy load the HuggingFace dataset."""
        if self._dataset is None:
            try:
                from datasets import load_dataset
            except ImportError:
                raise ImportError("Install datasets: pip install datasets")

            if self.config:
                self._dataset = load_dataset(self.repo_id, self.config, split=self.split)
            else:
                self._dataset = load_dataset(self.repo_id, split=self.split)
        return self._dataset

    def load(self) -> List[CommitExtraction]:
        """Load all records from HuggingFace."""
        return list(self)

    def __iter__(self) -> Iterator[CommitExtraction]:
        """Iterate over HuggingFace dataset records."""
        dataset = self._load_dataset()
        for record in dataset:
            yield CommitExtraction.from_json(dict(record))


def create_loader(source: str) -> BaseLoader:
    """Factory function to create appropriate loader from source string.

    Args:
        source: Can be:
            - Directory path: "/path/to/jsons/"
            - File path: "/path/to/file.json" or ".jsonl"
            - HuggingFace: "hf://repo/name" or "hf://repo/name:split"

    Returns:
        Appropriate loader instance.
    """
    if source.startswith("hf://"):
        # Parse HuggingFace URL: hf://owner/repo or hf://owner/repo:split
        hf_path = source[5:]  # Remove "hf://"
        if ":" in hf_path:
            repo_id, split = hf_path.rsplit(":", 1)
        else:
            repo_id, split = hf_path, "train"
        return HuggingFaceLoader(repo_id, split)

    path = Path(source)
    if path.is_dir():
        return JSONDirectoryLoader(path)
    elif path.is_file():
        return JSONFileLoader(path)
    else:
        raise ValueError(f"Unknown source type: {source}")
