"""Dataset building - combine results into final dataset format."""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

from omniperf.data import (
    CommitExtraction,
    PerformanceTest,
    ExecutionResult,
    DatasetRecord,
)

logger = logging.getLogger(__name__)


@dataclass
class BuilderConfig:
    """Configuration for dataset building."""
    output_dir: Path = field(default_factory=lambda: Path("data"))
    dataset_name: str = "omniperf_dataset"
    min_speedup: float = 1.0  # Minimum speedup to include (1.0 = no filter)
    min_tests: int = 1  # Minimum tests required
    push_to_hf: bool = False
    hf_repo: Optional[str] = None

    def __post_init__(self):
        self.output_dir = Path(self.output_dir)


class DatasetBuilder:
    """Build dataset from execution results."""

    def __init__(self, config: BuilderConfig):
        self.config = config
        self.records: List[DatasetRecord] = []

    def add_result(
        self,
        extraction: CommitExtraction,
        tests: List[PerformanceTest],
        result: ExecutionResult,
    ) -> Optional[DatasetRecord]:
        """Add an execution result to the dataset.

        Returns the DatasetRecord if it passes filters, None otherwise.
        """
        # Calculate speedup
        speedup = self._calculate_speedup(result)

        if speedup is None:
            logger.warning(f"Could not calculate speedup for {extraction.commit_hash}")
            return None

        if speedup < self.config.min_speedup:
            logger.info(f"Skipping {extraction.commit_hash}: speedup {speedup:.2f} < {self.config.min_speedup}")
            return None

        if len(tests) < self.config.min_tests:
            logger.info(f"Skipping {extraction.commit_hash}: only {len(tests)} tests")
            return None

        # Build duration_changes
        duration_changes = self._build_duration_changes(result)

        # Create record
        record = DatasetRecord(
            instance_id=self._make_instance_id(extraction),
            repo=extraction.repo_name or "unknown",
            base_commit=extraction.parent_hash,
            head_commit=extraction.commit_hash,
            patch=extraction.diff_text,
            efficiency_test=[t.code for t in tests],
            duration_changes=duration_changes,
            human_performance=speedup,
            hints_text=extraction.message,
            api=extraction.affected_apis[0] if extraction.affected_apis else None,
        )

        self.records.append(record)
        return record

    def build(self) -> Path:
        """Write dataset to disk and optionally push to HuggingFace.

        Returns path to the output file.
        """
        self.config.output_dir.mkdir(parents=True, exist_ok=True)
        output_path = self.config.output_dir / f"{self.config.dataset_name}.jsonl"

        # Write JSONL
        with open(output_path, "w") as f:
            for record in self.records:
                f.write(json.dumps(record.to_dict()) + "\n")

        logger.info(f"Wrote {len(self.records)} records to {output_path}")

        # Push to HuggingFace if configured
        if self.config.push_to_hf and self.config.hf_repo:
            self._push_to_huggingface(output_path)

        return output_path

    def _calculate_speedup(self, result: ExecutionResult) -> Optional[float]:
        """Calculate speedup ratio (base_time / head_time)."""
        base_times = [
            r.mean_duration for r in result.base_results
            if r.mean_duration is not None
        ]
        head_times = [
            r.mean_duration for r in result.head_results
            if r.mean_duration is not None
        ]

        if not base_times or not head_times:
            return None

        avg_base = sum(base_times) / len(base_times)
        avg_head = sum(head_times) / len(head_times)

        if avg_head <= 0:
            return None

        return avg_base / avg_head

    def _build_duration_changes(self, result: ExecutionResult) -> List[Dict]:
        """Build duration_changes list from execution result."""
        changes = []

        for i, base_result in enumerate(result.base_results):
            change = {
                "base": base_result.durations,
                "head": result.head_results[i].durations if i < len(result.head_results) else [],
            }
            if i < len(result.main_results):
                change["main"] = result.main_results[i].durations
            changes.append(change)

        return changes

    def _make_instance_id(self, extraction: CommitExtraction) -> str:
        """Generate instance ID from extraction."""
        repo = extraction.repo_name or "unknown"
        repo_clean = repo.replace("/", "__")
        return f"{repo_clean}-{extraction.commit_hash[:8]}"

    def _push_to_huggingface(self, filepath: Path) -> None:
        """Push dataset to HuggingFace Hub."""
        try:
            from datasets import Dataset
            from huggingface_hub import HfApi

            # Load as Dataset
            records = [json.loads(line) for line in filepath.read_text().strip().split("\n")]
            dataset = Dataset.from_list(records)

            # Push
            dataset.push_to_hub(self.config.hf_repo)
            logger.info(f"Pushed to HuggingFace: {self.config.hf_repo}")

        except Exception as e:
            logger.error(f"Failed to push to HuggingFace: {e}")
